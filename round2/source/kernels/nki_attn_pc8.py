"""nki_attn_pc8.py: switch 9g (R2_ATTN_PRECAST, lane THR6): switch 9d's fp8 attention projections with the four
weights cast to e4m3 once an optimizer step instead of in every kernel call. Everything after this docstring up to the
lane THR6 block is switch 9d's module kernels/nki_attn_proj8.py (md5 275376c2) byte for byte (tools/build_pc.py), so
9d's four kernels are here unchanged and the probe compares the new kernels against them in one module.

Switch 9d's kernels stream each bf16 weight through every call: the forward loads every weight chunk by the hardware
DGE strip transpose (cwid / 16 instructions a chunk) and casts it on ScalarE, the backward loads natural rows and casts
them; at the training shape that is 3 x 4.7 MB (QKV) + 4.7 MB (c_proj) of bf16 a program a forward call and twice the
QKV weights a program a backward call, every micro-step, every layer. Switch 9g's kernels:

  _attn8_precast_kernel(wq, wk, wv, wp, sc, need, cwf, cwp, cwb, cwbp, forms, u8) -> w8 [8, d x d]
      one call a layer after each optimizer step's scale refresh (and in load_for_eval): the e4m3 bytes of every
      weight tile switch 9d's kernels read, written by switch 9d's own tile loaders (_load_wT_cols for the forward's
      transposed chunks, the three-level natural DMA for the backward's rows) and its own cast (_cast8) with the
      layer's scale tile, so every byte equals the in-kernel cast's. Rows of the pack (each d x d bytes; tile c of a
      region at c x the tile's bytes, partition-major: element [p, k, i] at p x nkb x cwid + k x cwid + i):
        0..2  the QKV forward's transposed chunks: tile c [P, d/128, cwf] = W_cat[c cwf + i, k 128 + p] (W_cat =
              [c_q; c_k; c_v], 3 d x d), the order _fwd_body walks them
        3     c_proj's forward chunks: tile c [P, d/128, cwp] = W_p[c cwp + i, k 128 + p]
        4..6  the QKV backward's joint tiles: tile c [P, 3 d/128, cwb] = W_cat[s 128 + p, c cwb + i], the tile
              _bwd_chunk_joint assembles from three DMAs
        7     c_proj's backward tiles: tile c [P, d/128, cwbp] = W_p[s 128 + p, c cwbp + i]
      need: 1 the forward's rows (0 to 3), 2 the backward's (4 to 7), 3 all. The two programs split each region's
      tiles.
  _proj8_fwd_pc_kernel(a2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8) -> out [M, nw x d]
  _proj8_bwd_pc_kernel(g2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8) -> dx [M, d]
      _fwd_body and _bwd_body with each weight tile read from the pack (one plain DMA of its e4m3 bytes, half the bf16
      bytes, no strip transpose, no cast); every other instruction as there, so the outputs are switch 9d's kernels'
      bit for bit (the backward's joint tile arrives whole, so forms 0 and 1 run the same one-tile PSUM chain, which
      switch 9d documents as bitwise its three-tile form). nw 3 / base 0 (forward) or 4 (backward): the QKV
      projection; nw 1 / base 3 or 7: c_proj.

The pack crosses to torch as uint8: under u8 1 (the device form, the only one train.py uses) the bytes are cast into
uint8 tiles viewed as e4m3 and read into uint8 tiles viewed as e4m3 (switch 9f's form, proven on silicon by F8P's
byte probe); under u8 0 the pack and the tiles are e4m3 (the form the host's simulator executes exactly; it writes
fp8 through a uint8 view numerically, lane THR5's probe_view.py). The weights are square (d x d: the attention width
is n_embd in this file). Memory: 8 d^2 bytes a layer (18.9 MB at 1,536; 340 MB a rank at 18 layers).

Written for the device-side NKI parser frontend under switch 9d's rules (below, its own docstring).
"""

import math
import os

import torch

_P = 128
_KS = 256                   # contraction per double-row matmul
_env = os.environ.get
_LNC = int(_env("NEURON_LOGICAL_NC_CONFIG", "1"))
E4M3_MAX = 240.0            # the chip's legacy e4m3: largest finite value (its cast writes inf above it)
E5M2_MAX = 57344.0
X_SCALE = 4.0               # the QKV input is the RMSNorm output: abs(x_i) <= sqrt(d) (1 + 2^-8); 4 x 39.35 = 157 < 240
Y_BOUND_FACTOR = 1.3        # c_proj's input bound: abs(y) <= 1.3 (sqrt(d) rho_v + |ve_lambda| amax(ve)) (see y_bound)
G_HEADROOM = 2.0            # a gradient's scale keeps amax x s <= 57,344 / 2 (switch 9c's DY_HEADROOM)
E_W = (-24, 40)             # weight and activation scales 2^e
E_G = (-40, 60)             # gradient scales 2^e
PLAN_KIB = 168.0            # SBUF a partition the plan may fill (of 192 KiB; switch 9c's kernels hold up to 186)
TB_CHOICES = (2048, 1024, 512, 256, 128)
CW_CHOICES = (512, 256, 128)
# The layer's scale row (one value per column; the caller keeps it as a [128, NSC] tile down every partition).
SC_S_X, SC_S_WQKV, SC_DQ_QKV, SC_S_Y, SC_S_WPROJ, SC_DQ_PROJ = 0, 1, 2, 3, 4, 5
SC_I_WQKV, SC_I_WPROJ, SC_RHO_V2, SC_AMAX_WQKV, SC_AMAX_WPROJ, SC_B_Y, SC_VB = 6, 7, 8, 9, 10, 11, 12
NSC = 16
# The backward's per-call tile [128, 3]: the gradient's scale, the weight's scale, the output descale.
BC_S_G, BC_S_W, BC_DQ = 0, 1, 2
NBC = 3

_HAS_NKI = False
_NKI_ERROR = None
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    _VEC = nisa.engine.vector
    _DR = nisa.matmul_perf_mode.double_row
    _F8E4 = nl.float8_e4m3
    _F8E5 = nl.float8_e5m2

    def _shard():
        if nl.program_ndim() > 0:
            return nl.program_id(0), nl.num_programs(0)
        return 0, 1

    def _span(n, pid, nprg):
        per = n // nprg
        rem = n - per * nprg
        i0 = pid * per
        if pid < rem:
            i0 = i0 + pid
        else:
            i0 = i0 + rem
        cnt = per
        if pid < rem:
            cnt = cnt + 1
        return i0, i0 + cnt

    def _ceil_div(a, b):
        return (a + b - 1) // b

    def _load_sc(sc, ns):
        """The call's scale tile [P, ns] fp32 by one 2-D DMA (each column one value down every partition)."""
        s = nl.ndarray((_P, ns), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s, src=sc)
        return s

    def _load_T_tiles(a2, K, NKB, row0, nt, dt):
        """The transposed token tiles of a2 [M, K] rows [row0, row0 + nt x 128): at [P, nt x NKB, P] with
        at[p, t x NKB + s, i] = a2[row0 + t x 128 + i, s x 128 + p], by the hardware-DGE strip transpose (16 tokens over
        all NKB slabs per instruction: switch 9c's _load_T_block, with each token tile's slabs contiguous so a tile's
        double-row pair is a [P, 2, P] view with a full last axis)."""
        at = nl.ndarray((_P, nt * NKB, _P), dtype=dt, buffer=nl.sbuf)
        for t in range(nt):
            for s in range(_P // 16):
                nisa.dma_transpose(dst=at.ap([[nt * NKB * _P, _P], [_P, NKB], [1, 1], [1, 16]],
                                             offset=t * NKB * _P + s * 16),
                                   src=a2.ap([[K, 16], [_P, NKB], [1, 1], [1, _P]],
                                             offset=(row0 + t * _P + s * 16) * K),
                                   dge_mode=nisa.dge_mode.hwdge)
        return at

    def _load_wT_cols(w, K, NKB, n0, cwid, dt):
        """The transposed weight columns n0 to n0 + cwid: [P (k within slab), NKB, cwid] from the natural w [Nw, K]
        (switch 9c's _load_wT_chunk with cwid columns in place of 128): cwid / 16 strip transposes."""
        wt = nl.ndarray((_P, NKB, cwid), dtype=dt, buffer=nl.sbuf)
        for s in range(cwid // 16):
            nisa.dma_transpose(dst=wt.ap([[NKB * cwid, _P], [cwid, NKB], [1, 1], [1, 16]], offset=s * 16),
                               src=w.ap([[K, 16], [_P, NKB], [1, 1], [1, _P]], offset=(n0 + s * 16) * K),
                               dge_mode=nisa.dge_mode.hwdge)
        return wt

    def _cast8(dst8, src, sc_sb, col, forms):
        """dst8 = fp8(src x sc[:, col]) for a whole tile, the fp32 product rounded once into dst8's format (switch 9c's
        _cast8, verbatim). forms 0: one ScalarE activation with the per-partition vector scale. forms 1: the exact
        product into src in place on VectorE, then a ScalarE copy into fp8 at the constant scale 1.0."""
        q0 = col
        q1 = q0 + 1
        if forms == 1:
            nisa.tensor_scalar(dst=src, data=src, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)
            nisa.activation(dst=dst8, op=nl.copy, data=src, scale=1.0)
        else:
            nisa.activation(dst=dst8, op=nl.copy, data=src, scale=sc_sb[:, q0:q1])

    def _tile_gemm8(out, a8, w8, sc_sb, cd, t, NKB, NKD, row0, o0, cwid, dt):
        """One 128-token tile against one weight chunk: PSUM [P, cwid] = sum over NKD double-row pairs of
        a8[:, t NKB + 2kd : + 2, :]^T @ w8[:, 2kd : 2kd + 2, :]; descaled into bf16 on VectorE (switch 9c's
        _evac_desc); one DMA to out[row0 + t 128 : + 128, o0 : o0 + cwid]."""
        ps = nl.ndarray((_P, cwid), dtype=nl.float32, buffer=nl.psum)
        for kd in range(NKD):
            j0 = t * NKB + kd * 2
            j1 = j0 + 2
            k0 = kd * 2
            k1 = k0 + 2
            nisa.nc_matmul(dst=ps, stationary=a8[:, j0:j1, :], moving=w8[:, k0:k1, :], accumulate=(kd > 0),
                           perf_mode=_DR)
        q0 = cd
        q1 = q0 + 1
        o_sb = nl.ndarray((_P, cwid), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o_sb, data=ps, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)
        r0 = row0 + t * _P
        r1 = r0 + _P
        o1 = o0 + cwid
        nisa.dma_copy(dst=out[r0:r1, o0:o1], src=o_sb)

    def _tile_gemm8_3(out, g8, w8a, w8b, w8c, sc_sb, cd, t, NN, NNw, row0, o0, cwid, dt):
        """_tile_gemm8 with the moving weight split over three whole tiles (the QKV backward's forms 1): the chain's
        double-row pair kd reads tile kd // (NNw / 2) at its local slabs; one PSUM chain, the same order of sums."""
        ps = nl.ndarray((_P, cwid), dtype=nl.float32, buffer=nl.psum)
        NND = NN // 2
        NWD = NNw // 2
        for kd in range(NND):
            j0 = t * NN + kd * 2
            j1 = j0 + 2
            wi = kd // NWD
            k0 = (kd - wi * NWD) * 2
            k1 = k0 + 2
            if wi == 0:
                nisa.nc_matmul(dst=ps, stationary=g8[:, j0:j1, :], moving=w8a[:, k0:k1, :], accumulate=(kd > 0),
                               perf_mode=_DR)
            elif wi == 1:
                nisa.nc_matmul(dst=ps, stationary=g8[:, j0:j1, :], moving=w8b[:, k0:k1, :], accumulate=(kd > 0),
                               perf_mode=_DR)
            else:
                nisa.nc_matmul(dst=ps, stationary=g8[:, j0:j1, :], moving=w8c[:, k0:k1, :], accumulate=(kd > 0),
                               perf_mode=_DR)
        q0 = cd
        q1 = q0 + 1
        o_sb = nl.ndarray((_P, cwid), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o_sb, data=ps, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)
        r0 = row0 + t * _P
        r1 = r0 + _P
        o1 = o0 + cwid
        nisa.dma_copy(dst=out[r0:r1, o0:o1], src=o_sb)

    def _fwd_body(out, a2, w0, w1, w2, sc, nw, ca, cw, cd, tb, cwid, forms):
        """The forward into out [M, nw x Nw] (allocated by the kernel that calls this; see the module docstring)."""
        M = a2.shape[0]
        K = a2.shape[1]
        Nw = w0.shape[0]
        dt = a2.dtype
        NK = K // _P
        NKD = K // _KS
        NCw = Nw // cwid
        NC = NCw * nw
        sc_sb = _load_sc(sc, sc.shape[1])
        pid, nprg = _shard()
        t0, t1 = _span(M // _P, pid, nprg)
        ntb = tb // _P
        nblk = _ceil_div(t1 - t0, ntb)
        for b in range(nblk):
            bt0 = t0 + b * ntb
            bt1 = bt0 + ntb
            if bt1 > t1:
                bt1 = t1
            nt = bt1 - bt0
            row0 = bt0 * _P
            at = _load_T_tiles(a2, K, NK, row0, nt, dt)
            a8 = nl.ndarray((_P, nt * NK, _P), dtype=_F8E4, buffer=nl.sbuf)
            _cast8(a8, at, sc_sb, ca, forms)
            for c in range(NC):
                wi = c // NCw
                n0 = (c - wi * NCw) * cwid
                if wi == 0:
                    wt = _load_wT_cols(w0, K, NK, n0, cwid, dt)
                elif wi == 1:
                    wt = _load_wT_cols(w1, K, NK, n0, cwid, dt)
                else:
                    wt = _load_wT_cols(w2, K, NK, n0, cwid, dt)
                w8 = nl.ndarray((_P, NK, cwid), dtype=_F8E4, buffer=nl.sbuf)
                _cast8(w8, wt, sc_sb, cw, forms)
                o0 = c * cwid
                for t in range(nt):
                    _tile_gemm8(out, a8, w8, sc_sb, cd, t, NK, NKD, row0, o0, cwid, dt)

    def _bwd_chunk_joint(dx, g8, w0, w1, w2, sc_sb, nw, cw, cd, nt, NN, NNw, NND, K, row0, k0, cwid, dt, forms):
        """One dx column chunk with the weight rows in ONE tile [P (n within slab), NN, cwid] = W[s 128 + p, k0 + i],
        W = [w0; w1; w2][:nw]: one whole-tile DMA for c_proj (switch 9c's _load_w_chunk form); for the QKV projection
        three DMAs into the tile's slab ranges (an SBUF destination AP with an offset; forms 1 avoids it)."""
        wn = nl.ndarray((_P, NN, cwid), dtype=dt, buffer=nl.sbuf)
        if nw == 1:
            nisa.dma_copy(dst=wn, src=w0.ap([[K, _P], [_P * K, NNw], [1, cwid]], offset=k0))
        else:
            nisa.dma_copy(dst=wn.ap([[NN * cwid, _P], [cwid, NNw], [1, cwid]], offset=0),
                          src=w0.ap([[K, _P], [_P * K, NNw], [1, cwid]], offset=k0))
            nisa.dma_copy(dst=wn.ap([[NN * cwid, _P], [cwid, NNw], [1, cwid]], offset=NNw * cwid),
                          src=w1.ap([[K, _P], [_P * K, NNw], [1, cwid]], offset=k0))
            nisa.dma_copy(dst=wn.ap([[NN * cwid, _P], [cwid, NNw], [1, cwid]], offset=2 * NNw * cwid),
                          src=w2.ap([[K, _P], [_P * K, NNw], [1, cwid]], offset=k0))
        w8 = nl.ndarray((_P, NN, cwid), dtype=_F8E4, buffer=nl.sbuf)
        _cast8(w8, wn, sc_sb, cw, forms)
        for t in range(nt):
            _tile_gemm8(dx, g8, w8, sc_sb, cd, t, NN, NND, row0, k0, cwid, dt)

    def _bwd_body(dx, g2, w0, w1, w2, sc, nw, ca, cw, cd, tb, cwid, forms):
        """The input gradient into dx [M, K] (allocated by the kernel that calls this)."""
        M = g2.shape[0]
        N = g2.shape[1]
        Nw = w0.shape[0]
        K = w0.shape[1]
        dt = g2.dtype
        NN = N // _P
        NNw = Nw // _P
        NND = N // _KS
        NC = K // cwid
        sc_sb = _load_sc(sc, sc.shape[1])
        pid, nprg = _shard()
        t0, t1 = _span(M // _P, pid, nprg)
        ntb = tb // _P
        nblk = _ceil_div(t1 - t0, ntb)
        for b in range(nblk):
            bt0 = t0 + b * ntb
            bt1 = bt0 + ntb
            if bt1 > t1:
                bt1 = t1
            nt = bt1 - bt0
            row0 = bt0 * _P
            gt = _load_T_tiles(g2, N, NN, row0, nt, dt)
            g8 = nl.ndarray((_P, nt * NN, _P), dtype=_F8E5, buffer=nl.sbuf)
            _cast8(g8, gt, sc_sb, ca, forms)
            for c in range(NC):
                k0 = c * cwid
                if nw == 3 and forms == 1:
                    # forms 1: each weight's natural rows in a whole tile of its own (switch 9c's _load_w_chunk form, a
                    # DMA into a whole tile), cast, and one PSUM chain across the three tiles (_tile_gemm8_3).
                    wa = nl.ndarray((_P, NNw, cwid), dtype=dt, buffer=nl.sbuf)
                    nisa.dma_copy(dst=wa, src=w0.ap([[K, _P], [_P * K, NNw], [1, cwid]], offset=k0))
                    wb = nl.ndarray((_P, NNw, cwid), dtype=dt, buffer=nl.sbuf)
                    nisa.dma_copy(dst=wb, src=w1.ap([[K, _P], [_P * K, NNw], [1, cwid]], offset=k0))
                    wc = nl.ndarray((_P, NNw, cwid), dtype=dt, buffer=nl.sbuf)
                    nisa.dma_copy(dst=wc, src=w2.ap([[K, _P], [_P * K, NNw], [1, cwid]], offset=k0))
                    w8a = nl.ndarray((_P, NNw, cwid), dtype=_F8E4, buffer=nl.sbuf)
                    _cast8(w8a, wa, sc_sb, cw, forms)
                    w8b = nl.ndarray((_P, NNw, cwid), dtype=_F8E4, buffer=nl.sbuf)
                    _cast8(w8b, wb, sc_sb, cw, forms)
                    w8c = nl.ndarray((_P, NNw, cwid), dtype=_F8E4, buffer=nl.sbuf)
                    _cast8(w8c, wc, sc_sb, cw, forms)
                    for t in range(nt):
                        _tile_gemm8_3(dx, g8, w8a, w8b, w8c, sc_sb, cd, t, NN, NNw, row0, k0, cwid, dt)
                else:
                    _bwd_chunk_joint(dx, g8, w0, w1, w2, sc_sb, nw, cw, cd, nt, NN, NNw, NND, K, row0, k0, cwid, dt,
                                     forms)

    # The four launches: the QKV projection with its three weights as three inputs, c_proj with its one (each input
    # tensor appears once in a launch's arguments; the bodies take the one weight three times only at trace level).
    @nki.jit
    def _proj8_fwd_kernel(a2, w0, w1, w2, sc, ca, cw, cd, tb, cwid, forms):
        out = nl.ndarray((a2.shape[0], 3 * w0.shape[0]), dtype=a2.dtype, buffer=nl.shared_hbm)
        _fwd_body(out, a2, w0, w1, w2, sc, 3, ca, cw, cd, tb, cwid, forms)
        return out

    @nki.jit
    def _proj8_fwd1_kernel(a2, w0, sc, ca, cw, cd, tb, cwid, forms):
        out = nl.ndarray((a2.shape[0], w0.shape[0]), dtype=a2.dtype, buffer=nl.shared_hbm)
        _fwd_body(out, a2, w0, w0, w0, sc, 1, ca, cw, cd, tb, cwid, forms)
        return out

    @nki.jit
    def _proj8_bwd_kernel(g2, w0, w1, w2, sc, ca, cw, cd, tb, cwid, forms):
        dx = nl.ndarray((g2.shape[0], w0.shape[1]), dtype=g2.dtype, buffer=nl.shared_hbm)
        _bwd_body(dx, g2, w0, w1, w2, sc, 3, ca, cw, cd, tb, cwid, forms)
        return dx

    @nki.jit
    def _proj8_bwd1_kernel(g2, w0, sc, ca, cw, cd, tb, cwid, forms):
        dx = nl.ndarray((g2.shape[0], w0.shape[1]), dtype=g2.dtype, buffer=nl.shared_hbm)
        _bwd_body(dx, g2, w0, w0, w0, sc, 1, ca, cw, cd, tb, cwid, forms)
        return dx

    # ---- lane THR6, switch 9g (R2_ATTN_PRECAST): everything from here to `_HAS_NKI = True` is new; every function
    # above it is nki_attn_proj8.py 275376c2's, byte for byte (tools/build_pc.py copies them)

    def _load_w8_cols(w8, off, nkb, cwid, u8):
        """Switch 9g: one weight tile [P, nkb, cwid] e4m3 of the pack w8 [8, d x d] at element offset off (the tile's
        bytes contiguous, partition-major), by one DMA. u8 1 (the device form): a uint8 tile viewed as e4m3 (switch
        9f's byte form); u8 0 (the simulator's form): an e4m3 tile."""
        if u8 == 1:
            wu = nl.ndarray((_P, nkb, cwid), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.dma_copy(dst=wu, src=w8.ap([[nkb * cwid, _P], [cwid, nkb], [1, cwid]], offset=off))
            return wu.view(_F8E4)
        wq = nl.ndarray((_P, nkb, cwid), dtype=_F8E4, buffer=nl.sbuf)
        nisa.dma_copy(dst=wq, src=w8.ap([[nkb * cwid, _P], [cwid, nkb], [1, cwid]], offset=off))
        return wq

    def _cast_store_cols(w8, off, rstride, nkb, cwid, wt, sc_sb, col, forms, u8):
        """Switch 9g: e4m3(wt x sc[:, col]) of the bf16 tile wt [P, nkb, cwid] by _cast8 (switch 9d's cast), then one
        DMA of its bytes into the pack: element [p, k, i] at off + p x rstride + k x cwid + i."""
        if u8 == 1:
            wu = nl.ndarray((_P, nkb, cwid), dtype=nl.uint8, buffer=nl.sbuf)
            _cast8(wu.view(_F8E4), wt, sc_sb, col, forms)
            nisa.dma_copy(dst=w8.ap([[rstride, _P], [cwid, nkb], [1, cwid]], offset=off), src=wu)
        else:
            wq = nl.ndarray((_P, nkb, cwid), dtype=_F8E4, buffer=nl.sbuf)
            _cast8(wq, wt, sc_sb, col, forms)
            nisa.dma_copy(dst=w8.ap([[rstride, _P], [cwid, nkb], [1, cwid]], offset=off), src=wq)

    @nki.jit
    def _attn8_precast_kernel(wq, wk, wv, wp, sc, need, cwf, cwp, cwb, cwbp, forms, u8):
        """Switch 9g: one layer's e4m3 weight tiles, once an optimizer step (the pack's rows: the module docstring)."""
        D = wq.shape[1]
        DD = D * D
        dt = wq.dtype
        NK = D // _P
        NN = 3 * NK
        if u8 == 1:
            w8 = nl.ndarray((8, DD), dtype=nl.uint8, buffer=nl.shared_hbm)
        else:
            w8 = nl.ndarray((8, DD), dtype=_F8E4, buffer=nl.shared_hbm)
        sc_sb = _load_sc(sc, sc.shape[1])
        pid, nprg = _shard()
        if need % 2 == 1:
            NCf = D // cwf
            a0, a1 = _span(3 * NCf, pid, nprg)
            for i in range(a1 - a0):
                c = a0 + i
                wi = c // NCf
                n0 = (c - wi * NCf) * cwf
                if wi == 0:
                    wt = _load_wT_cols(wq, D, NK, n0, cwf, dt)
                elif wi == 1:
                    wt = _load_wT_cols(wk, D, NK, n0, cwf, dt)
                else:
                    wt = _load_wT_cols(wv, D, NK, n0, cwf, dt)
                _cast_store_cols(w8, c * D * cwf, NK * cwf, NK, cwf, wt, sc_sb, SC_S_WQKV, forms, u8)
            NCp = D // cwp
            b0, b1 = _span(NCp, pid, nprg)
            for i in range(b1 - b0):
                c = b0 + i
                wt = _load_wT_cols(wp, D, NK, c * cwp, cwp, dt)
                _cast_store_cols(w8, 3 * DD + c * D * cwp, NK * cwp, NK, cwp, wt, sc_sb, SC_S_WPROJ, forms, u8)
        if need >= 2:
            NCb = D // cwb
            e0, e1 = _span(3 * NCb, pid, nprg)
            for i in range(e1 - e0):
                u = e0 + i
                c = u // 3
                wi = u - c * 3
                k0 = c * cwb
                wn = nl.ndarray((_P, NK, cwb), dtype=dt, buffer=nl.sbuf)
                if wi == 0:
                    nisa.dma_copy(dst=wn, src=wq.ap([[D, _P], [_P * D, NK], [1, cwb]], offset=k0))
                elif wi == 1:
                    nisa.dma_copy(dst=wn, src=wk.ap([[D, _P], [_P * D, NK], [1, cwb]], offset=k0))
                else:
                    nisa.dma_copy(dst=wn, src=wv.ap([[D, _P], [_P * D, NK], [1, cwb]], offset=k0))
                off = 4 * DD + c * _P * NN * cwb + wi * NK * cwb
                _cast_store_cols(w8, off, NN * cwb, NK, cwb, wn, sc_sb, SC_S_WQKV, forms, u8)
            NCq = D // cwbp
            f0, f1 = _span(NCq, pid, nprg)
            for i in range(f1 - f0):
                c = f0 + i
                k0 = c * cwbp
                wn = nl.ndarray((_P, NK, cwbp), dtype=dt, buffer=nl.sbuf)
                nisa.dma_copy(dst=wn, src=wp.ap([[D, _P], [_P * D, NK], [1, cwbp]], offset=k0))
                _cast_store_cols(w8, 7 * DD + c * D * cwbp, NK * cwbp, NK, cwbp, wn, sc_sb, SC_S_WPROJ, forms, u8)
        return w8

    def _fwd_pc_body(out, a2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8):
        """Switch 9g: _fwd_body with each weight chunk read from the pack's rows base.. (no strip transpose, no
        cast)."""
        M = a2.shape[0]
        K = a2.shape[1]
        DD = w8.shape[1]
        Nw = DD // K
        dt = a2.dtype
        NK = K // _P
        NKD = K // _KS
        NC = (Nw // cwid) * nw
        sc_sb = _load_sc(sc, sc.shape[1])
        pid, nprg = _shard()
        t0, t1 = _span(M // _P, pid, nprg)
        ntb = tb // _P
        nblk = _ceil_div(t1 - t0, ntb)
        for b in range(nblk):
            bt0 = t0 + b * ntb
            bt1 = bt0 + ntb
            if bt1 > t1:
                bt1 = t1
            nt = bt1 - bt0
            row0 = bt0 * _P
            at = _load_T_tiles(a2, K, NK, row0, nt, dt)
            a8 = nl.ndarray((_P, nt * NK, _P), dtype=_F8E4, buffer=nl.sbuf)
            _cast8(a8, at, sc_sb, ca, forms)
            for c in range(NC):
                w8t = _load_w8_cols(w8, base * DD + c * K * cwid, NK, cwid, u8)
                o0 = c * cwid
                for t in range(nt):
                    _tile_gemm8(out, a8, w8t, sc_sb, cd, t, NK, NKD, row0, o0, cwid, dt)

    def _bwd_pc_body(dx, g2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8):
        """Switch 9g: _bwd_body with each weight tile read whole from the pack's rows base.. (the joint tile for the
        QKV projection; no DMA into slab ranges, no cast)."""
        M = g2.shape[0]
        N = g2.shape[1]
        Nw = N // nw
        DD = w8.shape[1]
        K = DD // Nw
        dt = g2.dtype
        NN = N // _P
        NND = N // _KS
        NC = K // cwid
        sc_sb = _load_sc(sc, sc.shape[1])
        pid, nprg = _shard()
        t0, t1 = _span(M // _P, pid, nprg)
        ntb = tb // _P
        nblk = _ceil_div(t1 - t0, ntb)
        for b in range(nblk):
            bt0 = t0 + b * ntb
            bt1 = bt0 + ntb
            if bt1 > t1:
                bt1 = t1
            nt = bt1 - bt0
            row0 = bt0 * _P
            gt = _load_T_tiles(g2, N, NN, row0, nt, dt)
            g8 = nl.ndarray((_P, nt * NN, _P), dtype=_F8E5, buffer=nl.sbuf)
            _cast8(g8, gt, sc_sb, ca, forms)
            for c in range(NC):
                k0 = c * cwid
                w8t = _load_w8_cols(w8, base * DD + c * _P * NN * cwid, NN, cwid, u8)
                for t in range(nt):
                    _tile_gemm8(dx, g8, w8t, sc_sb, cd, t, NN, NND, row0, k0, cwid, dt)

    @nki.jit
    def _proj8_fwd_pc_kernel(a2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8):
        nout = nw * (w8.shape[1] // a2.shape[1])
        out = nl.ndarray((a2.shape[0], nout), dtype=a2.dtype, buffer=nl.shared_hbm)
        _fwd_pc_body(out, a2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8)
        return out

    @nki.jit
    def _proj8_bwd_pc_kernel(g2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8):
        kout = w8.shape[1] // (g2.shape[1] // nw)
        dx = nl.ndarray((g2.shape[0], kout), dtype=g2.dtype, buffer=nl.shared_hbm)
        _bwd_pc_body(dx, g2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8)
        return dx

    _HAS_NKI = True
except Exception as exc:  # nki unavailable (CPU dev box): the emulation below still works
    _NKI_ERROR = f"{type(exc).__name__}: {exc}"


def _launch(kernel):
    """kernel[2] under an LNC=2 runtime (both physical cores run half the token tiles), plain call otherwise."""
    return kernel[_LNC] if _LNC == 2 else kernel


def plan_kib(nk: int, n_out: int, tb: int, cwid: int, forms: int = 0) -> float:
    """SBUF a partition (KiB) one kernel holds: the activation's transposed tiles (bf16 and fp8) of a block of tb
    tokens over nk contraction slabs, the streamed weight chunk (bf16 and fp8, double-buffered), the evacuation tiles
    (double-buffered) and the scale tile. nk = K / 128 (the forward) or N / 128 (the backward); n_out is unused (the
    chunk width bounds the output side)."""
    nt = tb // _P
    act = nt * nk * _P * 3
    wts = 2 * nk * cwid * 3
    evac = 2 * cwid * 2
    return (act + wts + evac + NSC * 4) / 1024.0


def plan(nk: int, n_out: int, forms: int = 0, tb_max: int = 2048) -> tuple:
    """(tb, cwid): the largest token block, then the widest chunk dividing n_out, whose plan fits PLAN_KIB."""
    for tb in TB_CHOICES:
        if tb > tb_max:
            continue
        for cwid in CW_CHOICES:
            if n_out % cwid == 0 and plan_kib(nk, n_out, tb, cwid, forms) <= PLAN_KIB:
                return tb, cwid
    raise ValueError(f"switch 9d: no (tb, cwid) fits {PLAN_KIB} KiB at {nk} slabs, {n_out} columns")


def shapes_apply(m: int, k: int, nw: int) -> bool:
    """The shapes the kernels take: K and each weight's rows multiples of 256 and of 128-column chunks; M any (the
    caller pads to a multiple of 128)."""
    return m > 0 and k % _KS == 0 and nw % _KS == 0


# ----------------------------------------------------------------------------- scales (torch, any device)

def _pow2_le(v: torch.Tensor, limit: float, e_lo: int, e_hi: int):
    """(s, 1/s): s = 2^e for the largest integer e in [e_lo, e_hi] with v x s <= limit (v >= 0, fp32, any shape; v = 0
    gives 2^e_hi). Switch 9c's greedy search, verbatim: comparisons, selects and multiplications by powers of two only,
    so s and 1/s are exact powers of two and s x (1/s) == 1."""
    v = v.float()
    s = torch.full_like(v, 2.0 ** e_lo)
    inv = torch.full_like(v, 2.0 ** -e_lo)
    span = e_hi - e_lo
    for b in (64, 32, 16, 8, 4, 2, 1):
        if b > span:
            continue
        cand = s * (2.0 ** b)
        ok = (v * cand <= limit) & (cand <= 2.0 ** e_hi)
        s = torch.where(ok, cand, s)
        inv = torch.where(ok, inv * (2.0 ** -b), inv)
    return s, inv


def y_bound(rho_v2: torch.Tensor, d: int, vb: torch.Tensor) -> torch.Tensor:
    """The bound on c_proj's input y (the attention output, or its head-gated form): each output row is a softmax-
    weighted average of v rows (weights >= 0 summing to 1), and a sigmoid gate is at most 1, so abs(y_j) <= max over
    positions of abs(v_j); v_j = x . W_v[j] (+ ve_lambda ve_j on a value-embedding layer) with x the RMSNorm output
    (norm sqrt(d) (1 + 2^-8)), so abs(v_j) <= sqrt(d) rho_v + vb (rho_v: W_v's largest row norm; vb = |ve_lambda| x
    the table's amax). Y_BOUND_FACTOR 1.3 covers every rounding on the way: x and W_v in e4m3 when the QKV
    projection is fp8 (1.0625^2), the bf16 roundings of v, of the value-embedding term, of the attention's
    probabilities and of y (each 1 + 2^-8), the softmax weights' sum (1 + 2^-7): 1.153 in all. A pure function of
    the weights, never of the batch: the scored forward stays causal and the same for any batch it is in."""
    return Y_BOUND_FACTOR * (math.sqrt(d) * torch.sqrt(rho_v2) + vb)


def forward_scales(wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wp: torch.Tensor,
                   vb: torch.Tensor) -> torch.Tensor:
    """One layer's scale row [NSC] fp32 from its four weights as the kernels read them (bf16) and vb (y_bound's
    value-embedding term, 0 on a layer without one): x's constant 4; the QKV weights' common e4m3 multiplier from
    their joint exact amax; y's from y_bound; c_proj's weight's from its exact amax; the two descales. Training calls
    it after every optimizer step and load_for_eval once after loading, so the scored forward is the trained one;
    nothing is saved."""
    f = [w.detach().to(torch.bfloat16).float() for w in (wq, wk, wv, wp)]
    amax_qkv = torch.stack([f[0].abs().amax(), f[1].abs().amax(), f[2].abs().amax()]).amax()
    amax_p = f[3].abs().amax()
    s_wqkv, i_wqkv = _pow2_le(amax_qkv, E4M3_MAX, E_W[0], E_W[1])
    s_wp, i_wp = _pow2_le(amax_p, E4M3_MAX, E_W[0], E_W[1])
    rho_v2 = (f[2] * f[2]).sum(dim=1).amax()
    vbf = vb.detach().float().reshape(())
    b_y = y_bound(rho_v2, wq.shape[1], vbf)
    s_y, i_y = _pow2_le(b_y, E4M3_MAX, E_W[0], E_W[1])
    zero = torch.zeros_like(amax_qkv)
    s_x = torch.full_like(amax_qkv, X_SCALE)
    return torch.stack([s_x, s_wqkv, i_wqkv * (1.0 / X_SCALE), s_y, s_wp, i_y * i_wp, i_wqkv, i_wp, rho_v2, amax_qkv,
                        amax_p, b_y, vbf, zero, zero, zero])


def sc_tile(row: torch.Tensor) -> torch.Tensor:
    """[128, len(row)] fp32: the row down every partition (the kernels' scale input, read by one 2-D DMA)."""
    n = row.numel()
    return row.float().reshape(1, n).expand(_P, n).contiguous()


def backward_tile(g2: torch.Tensor, s_w: torch.Tensor, i_w: torch.Tensor) -> torch.Tensor:
    """The backward call's tile [128, NBC]: s_g from the gradient's exact amax (amax x s_g <= 57,344 / 2), the
    weight's multiplier, and the descale 1 / (s_g s_w). Training only: the scored forward never reads it."""
    amax_g = g2.detach().abs().amax().float()
    s_g, i_g = _pow2_le(amax_g, E5M2_MAX / G_HEADROOM, E_G[0], E_G[1])
    return sc_tile(torch.stack([s_g, s_w.float().reshape(()), (i_g * i_w).float().reshape(())]))


# ----------------------------------------------------------------------------- torch emulation (CPU and proofs)

def q8(t: torch.Tensor, fmt: int) -> torch.Tensor:
    """The fp8 rounding (round to nearest even) of t's fp32 values, back in fp32: e4m3 (fmt 0: torch's float8_e4m3fn,
    whose values and rounding inside +-240 are the chip's legacy e4m3's, the only range the kernels cast) or e5m2
    (fmt 1). Switch 9c's q8, verbatim."""
    return t.float().to(torch.float8_e4m3fn if fmt == 0 else torch.float8_e5m2).float()


def emulate_fwd(a2: torch.Tensor, w: torch.Tensor, s_a, s_w, dq) -> torch.Tensor:
    """_proj8_fwd_kernel's arithmetic in torch: out [M, N] = bf16((e4m3(a2 s_a) @ e4m3(w s_w)^T) dq), w the
    (concatenated) [N, K] weight; fp32 products and accumulation (torch's summation order)."""
    a8 = q8(a2.float() * s_a, 0)
    w8 = q8(w.float() * s_w, 0)
    return ((a8 @ w8.t()) * dq).to(a2.dtype)


def emulate_bwd(g2: torch.Tensor, w: torch.Tensor, s_g, s_w, dq) -> torch.Tensor:
    """_proj8_bwd_kernel's arithmetic in torch: dx [M, K] = bf16((e5m2(g2 s_g) @ e4m3(w s_w)) dq)."""
    g8 = q8(g2.float() * s_g, 1)
    w8 = q8(w.float() * s_w, 0)
    return ((g8 @ w8) * dq).to(g2.dtype)


def reference_fwd(a2: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The bf16 projection the switch replaces (F.linear in the activations' dtype)."""
    return torch.nn.functional.linear(a2, w)


# ----------------------------------------------------------------------------- lane THR6, switch 9g: host side

PACK_ROWS = 8
PK_FQKV, PK_FPROJ, PK_BQKV, PK_BPROJ = 0, 3, 4, 7    # the pack's first row of each region


def shapes_apply_pc(m: int, d: int) -> bool:
    """Switch 9g: the shapes the pre-cast kernels take: switch 9d's (M a multiple of 128, d a multiple of 256) with
    square weights."""
    return m > 0 and m % _P == 0 and d % _KS == 0


def _bytes_e4m3(t: torch.Tensor) -> torch.Tensor:
    """The e4m3 bytes of t's fp32 values (round to nearest even; torch's float8_e4m3fn, whose bytes inside +-240 are
    the chip's legacy e4m3's: the only range the kernels cast) as uint8."""
    return t.float().to(torch.float8_e4m3fn).view(torch.uint8)


def _region_fwd(w: torch.Tensor, cw: int) -> torch.Tensor:
    """[R, d] (R a multiple of cw) -> the forward's tiles, flat: tile c [p, k, i] = w[c cw + i, k 128 + p]."""
    r, d = w.shape
    return w.reshape(r // cw, cw, d // _P, _P).permute(0, 3, 2, 1).contiguous().reshape(-1)


def _region_bwd(w: torch.Tensor, cw: int) -> torch.Tensor:
    """[R, d] -> the backward's tiles, flat: tile c [p, s, i] = w[s 128 + p, c cw + i]."""
    r, d = w.shape
    return w.reshape(r // _P, _P, d // cw, cw).permute(2, 1, 0, 3).contiguous().reshape(-1)


def precast_ref(wq, wk, wv, wp, row, cwf: int, cwp: int, cwb: int, cwbp: int, need: int = 3) -> torch.Tensor:
    """Switch 9g: the bytes _attn8_precast_kernel writes, [8, d x d] uint8 (rows in the module docstring; a region not
    in need is zero here and unwritten there), from the weights (rounded to bf16 as the kernels read them) and the
    layer's scale row (forward_scales'): e4m3(W x s_wqkv) for c_q, c_k, c_v and e4m3(W_p x s_wproj)."""
    d = wq.shape[1]
    sq, sp = row[SC_S_WQKV].float(), row[SC_S_WPROJ].float()
    wcat = torch.cat([w.detach().to(torch.bfloat16).float() for w in (wq, wk, wv)]) * sq
    wpf = wp.detach().to(torch.bfloat16).float() * sp
    out = torch.zeros(PACK_ROWS, d * d, dtype=torch.uint8)
    if need % 2 == 1:
        out[PK_FQKV:PK_FQKV + 3] = _bytes_e4m3(_region_fwd(wcat, cwf)).reshape(3, d * d)
        out[PK_FPROJ] = _bytes_e4m3(_region_fwd(wpf, cwp))
    if need >= 2:
        out[PK_BQKV:PK_BQKV + 3] = _bytes_e4m3(_region_bwd(wcat, cwb)).reshape(3, d * d)
        out[PK_BPROJ] = _bytes_e4m3(_region_bwd(wpf, cwbp))
    return out


def _decode(b: torch.Tensor) -> torch.Tensor:
    """uint8 e4m3 bytes -> their fp32 values."""
    return b.contiguous().view(torch.float8_e4m3fn).float()


def unpack_fwd(pack: torch.Tensor, kind: int, d: int, cw: int) -> torch.Tensor:
    """Switch 9g: the forward's e4m3 weight values from the pack, [nw d, d] fp32 contiguous (kind 0: W_cat, from rows
    0 to 2; kind 1: W_p, from row 3): the inverse of _region_fwd."""
    nw = 3 if kind == 0 else 1
    r0 = PK_FQKV if kind == 0 else PK_FPROJ
    v = _decode(pack[r0:r0 + nw].reshape(-1))
    return v.reshape(nw * d // cw, _P, d // _P, cw).permute(0, 3, 2, 1).contiguous().reshape(nw * d, d)


def unpack_bwd(pack: torch.Tensor, kind: int, d: int, cw: int) -> torch.Tensor:
    """Switch 9g: the backward's e4m3 weight values from the pack, [nw d, d] fp32 contiguous (rows 4 to 6 or 7): the
    inverse of _region_bwd."""
    nw = 3 if kind == 0 else 1
    r0 = PK_BQKV if kind == 0 else PK_BPROJ
    v = _decode(pack[r0:r0 + nw].reshape(-1))
    return v.reshape(d // cw, _P, nw * d // _P, cw).permute(2, 1, 0, 3).contiguous().reshape(nw * d, d)


def emulate_fwd_pc(a2: torch.Tensor, pack: torch.Tensor, kind: int, s_a, dq, cw: int) -> torch.Tensor:
    """_proj8_fwd_pc_kernel's arithmetic in torch: emulate_fwd's expression with the weight's e4m3 values read from
    the pack (equal to q8(W s_w), so the result is emulate_fwd's bit for bit)."""
    w8 = unpack_fwd(pack, kind, a2.shape[1], cw)
    a8 = q8(a2.float() * s_a, 0)
    return ((a8 @ w8.t()) * dq).to(a2.dtype)


def emulate_bwd_pc(g2: torch.Tensor, pack: torch.Tensor, kind: int, s_g, dq, cw: int) -> torch.Tensor:
    """_proj8_bwd_pc_kernel's arithmetic in torch: emulate_bwd's expression with the pack's weight values."""
    nw = 3 if kind == 0 else 1
    w8 = unpack_bwd(pack, kind, g2.shape[1] // nw, cw)
    g8 = q8(g2.float() * s_g, 1)
    return ((g8 @ w8) * dq).to(g2.dtype)
