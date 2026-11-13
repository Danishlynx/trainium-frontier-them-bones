"""nki_attn_nf8.py: switch 9m (R2_NORM_FUSE, lane THR11): the attention-side RMSNorm folded into switch 9g's fp8 QKV
forward kernel. Everything after this docstring is switch 9g's module kernels/nki_attn_pc8.py (md5 91c0a087) byte for
byte, plus three insertions (tools/build_nf.py: the norm's two constants, the lane THR11 block in the nki block, the
host block at the end), so switch 9d's and 9g's kernels are here unchanged and the probe compares the new kernel
against 9g's in one module.

Today the block's mixed stream x goes through the kit's parameter-free F.rms_norm as a compiled graph op (an fp32
upcast, the squares, the mean, rsqrt, the product, one bf16 rounding; about 224 us a call on F's capture, with spills),
whose bf16 output xn is written to HBM; switch 9g's QKV forward kernel then reads xn back by hardware-DGE strip
transposes (8 a 128-token tile) and casts the block to e4m3 at the constant scale X_SCALE = 4 (xn is bounded by
sqrt(d) because it is the norm's output). Switch 9m's kernel:

  _proj8_fwd_nf_kernel(x2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8) -> out [M, nw x d], xn [M, d], rs [M, 1]
      x2 [M, d] bf16 the PRE-norm rows. Per 128-token tile (_nf_norm_tile): one natural DMA of the rows, the fp32 sum
      of their squares (ScalarE activation_reduce), rs = rsqrt(ss x (1/d) + 2^-23) (VectorE), xn = bf16(x rs) (one
      rounding, the composite's own), xn and rs stored (the saved tensors of the backward: the bf16 weight gradient
      g^T xn and the norm's backward read them); then (_nf_load_a8) xn's d / 128 bf16 slabs transposed on TensorE
      (bit-accurate) through a bf16 PSUM tile and each cast to e4m3 out of PSUM by switch 9d's own cast instruction
      and scale column. Every later instruction (the weight tiles from 9g's pack, the double-row matmuls, the
      descale, the stores) is 9g's _fwd_pc_body's, so given the kernel's own xn the output is switch 9g's (and 9d's)
      bit for bit. The block's norm thus costs no compiled graph op and no new custom-op crossing; the strip
      transposes of the activation are gone.

Rounding against the kit's expression: rs carries the engine's fp32 summation order and the product by 1/d (the CPU
kit divides by d), so it can differ from the compiled norm's rs in its last bits (stated bound: |rs / rs64 - 1| <=
2^-20, nf_rs_bound), and xn then by one bf16 rounding flip where x rs sits on a rounding boundary; nothing else
differs. Off a Neuron device train.py runs F.rms_norm's composite and switch 9g's emulation (bitwise the base).

SBUF a partition at the QKV forward's plan (2,048 tokens x 1,536): the e4m3 block (24 KiB) and one tile's natural
rows, squares and xn (3 + 6 + 3 KiB) in place of 9g's bf16 and e4m3 blocks (72 KiB). PSUM: one 1 KiB transpose tile
beside 9g's accumulators. HBM: x2 read once (as xn was), xn and rs written (as the compiled norm wrote xn).

Written for the device-side NKI parser frontend under switch 9d's rules (below, its own docstring); every new form
is one a kernel of ours compiled for the device: _nf_norm_tile is lane L1's chip-run prologue (activation_reduce with
square, tensor_scalar rsqrt, the bf16 product), the transposes and the cast out of PSUM are lane THR8's devcc-proven
_load_T8_tiles forms.
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

# Switch 9m (R2_NORM_FUSE, lane THR11): the RMSNorm's constants, at module level as X_SCALE is (the device parser
# resolves them as globals of the kernels' module).
NF_EPS = 2.0 ** -23     # F.rms_norm's eps for bf16 and fp32 inputs (the opmath type's epsilon)
NF_GROUP = 4            # slabs one transpose PSUM tile holds: 4 x 128 bf16 = 1 KiB a partition (switch 9c's group)

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

    # ---- lane THR11, switch 9m (R2_NORM_FUSE): everything from here to `_HAS_NKI = True` is new; every function
    # above it is nki_attn_pc8.py 91c0a087's after its docstring, byte for byte (tools/build_nf.py copies them).
    # Derived from the base's text by replacing the activation path only: _fwd_pc_body, _proj8_fwd_pc_kernel.

    def _nf_norm_tile(x2, xn, rs, r0, D, inv_d, dt):
        """Switch 9m: the RMSNorm of rows r0 .. r0 + 128 of the pre-norm x2 [M, D], tokens on the partitions: one
        natural DMA of the rows; ss, the sum of their squares in fp32 (one ScalarE activation_reduce: the exact fp32
        squares of the bf16 values, summed in the engine's order); rq = rsqrt(ss x (1/D) + 2^-23) by two VectorE
        tensor_scalar instructions (lane L1's chip-run form: the mean as a product by 1/D, no nl.divide); xq = bf16(x
        rq), the fp32 product rounded once, where F.rms_norm's composite rounds once; rq stored into rs [M, 1] fp32 and
        xq into xn [M, D] (the tensors the backward reads). Returns the xn tile [P, D] (SBUF)."""
        f32 = nl.float32
        r1 = r0 + _P
        xs = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=xs, src=x2[r0:r1, :])
        sq = nl.ndarray((_P, D), dtype=f32, buffer=nl.sbuf)
        ss = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=sq, op=nl.square, data=xs, reduce_op=nl.add, reduce_res=ss)
        nisa.tensor_scalar(dst=ss, data=ss, op0=nl.multiply, operand0=inv_d, op1=nl.add, operand1=NF_EPS)
        rq = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rq, data=ss, op0=nl.rsqrt, operand0=0.0)
        nisa.dma_copy(dst=rs[r0:r1, :], src=rq)
        xq = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=xq, data=xs, op0=nl.multiply, operand0=rq, engine=_VEC)
        nisa.dma_copy(dst=xn[r0:r1, :], src=xq)
        return xq

    def _nf_cast8_ps(dst8, src_ps, sc_sb, col, forms, dt):
        """Switch 9m: switch 9d's _cast8 of one transposed slab read from PSUM. forms 0: _cast8 itself (one ScalarE
        activation with the scale column, reading PSUM). forms 1: the exact product into a bf16 SBUF tile on VectorE
        (VectorE may not write a 2-byte PSUM tile), then the ScalarE copy into fp8 at scale 1.0: _cast8 forms 1's two
        steps and values."""
        if forms == 1:
            q0 = col
            q1 = q0 + 1
            tq = nl.ndarray((_P, _P), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=tq, data=src_ps, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)
            nisa.activation(dst=dst8, op=nl.copy, data=tq, scale=1.0)
        else:
            _cast8(dst8, src_ps, sc_sb, col, 0)

    def _nf_load_a8(a8, xn, rs, x2, K, NKB, row0, nt, dt, sc_sb, col, forms, inv_d):
        """Switch 9m: a8 [P, nt x NKB, P] e4m3 = the tiles switch 9g's _load_T_tiles and _cast8 make from the
        normalised rows, a8[p, t NKB + s, i] = e4m3(xn[row0 + t 128 + i, s 128 + p] x sc[:, col]), from the pre-norm
        rows x2: each token tile normalised by _nf_norm_tile (which stores its xn and rs rows), its NKB bf16 slabs
        transposed on TensorE (bit-accurate) through a bf16 PSUM tile of up to NF_GROUP slabs, each slab cast out of
        PSUM by _nf_cast8_ps."""
        for t in range(nt):
            r0 = row0 + t * _P
            xq = _nf_norm_tile(x2, xn, rs, r0, K, inv_d, dt)
            for g0 in range(0, NKB, NF_GROUP):
                ng = NKB - g0
                if ng > NF_GROUP:
                    ng = NF_GROUP
                pst = nl.ndarray((_P, ng * _P), dtype=dt, buffer=nl.psum)
                for gg in range(ng):
                    q0 = gg * _P
                    q1 = q0 + _P
                    s0 = (g0 + gg) * _P
                    s1 = s0 + _P
                    nisa.nc_transpose(dst=pst[:, q0:q1], data=xq[:, s0:s1])
                for gg in range(ng):
                    q0 = gg * _P
                    q1 = q0 + _P
                    j = t * NKB + g0 + gg
                    _nf_cast8_ps(a8[:, j, :], pst[:, q0:q1], sc_sb, col, forms, dt)

    def _fwd_nf_body(out, xn, rs, x2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8):
        """Switch 9m: switch 9g's _fwd_pc_body on the pre-norm rows x2, the activation's e4m3 tiles made by
        _nf_load_a8 (which also writes the normalised rows xn and their rs); every later instruction is
        _fwd_pc_body's."""
        M = x2.shape[0]
        K = x2.shape[1]
        DD = w8.shape[1]
        Nw = DD // K
        dt = x2.dtype
        NK = K // _P
        NKD = K // _KS
        NC = (Nw // cwid) * nw
        inv_d = 1.0 / K
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
            a8 = nl.ndarray((_P, nt * NK, _P), dtype=_F8E4, buffer=nl.sbuf)
            _nf_load_a8(a8, xn, rs, x2, K, NK, row0, nt, dt, sc_sb, ca, forms, inv_d)
            for c in range(NC):
                w8t = _load_w8_cols(w8, base * DD + c * K * cwid, NK, cwid, u8)
                o0 = c * cwid
                for t in range(nt):
                    _tile_gemm8(out, a8, w8t, sc_sb, cd, t, NK, NKD, row0, o0, cwid, dt)

    @nki.jit
    def _proj8_fwd_nf_kernel(x2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8):
        nout = nw * (w8.shape[1] // x2.shape[1])
        out = nl.ndarray((x2.shape[0], nout), dtype=x2.dtype, buffer=nl.shared_hbm)
        xn = nl.ndarray((x2.shape[0], x2.shape[1]), dtype=x2.dtype, buffer=nl.shared_hbm)
        rs = nl.ndarray((x2.shape[0], 1), dtype=nl.float32, buffer=nl.shared_hbm)
        _fwd_nf_body(out, xn, rs, x2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8)
        return out, xn, rs

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


# ----------------------------------------------------------------------------- lane THR11, switch 9m: host side

def shapes_apply_nf(m: int, d: int) -> bool:
    """Switch 9m: the shapes the norm-fused QKV forward takes: switch 9g's (rows a multiple of 128, d of 256)."""
    return shapes_apply_pc(m, d)


def norm_rows_ref(x2: torch.Tensor):
    """F.rms_norm's composite on [M, D] rows, op for op (bitwise F.rms_norm(x2, (D,)) on CPU): (xn in x2's dtype, rs
    [M, 1] fp32)."""
    up = x2.float()
    rs = torch.rsqrt(up.pow(2).mean(-1, keepdim=True) + NF_EPS)
    return (up * rs).to(x2.dtype), rs


def norm_rows_nf(x2: torch.Tensor, rs: torch.Tensor | None = None):
    """_nf_norm_tile's arithmetic in torch: rs = rsqrt(sum(x^2) x (1/D) + eps) in fp32 (torch's summation order: the
    engine's order can differ in the last bits, within nf_rs_bound) and xn = bf16(x rs), the fp32 product rounded once.
    With rs given (the kernel's own), xn is the kernel's xn bit for bit."""
    up = x2.float()
    if rs is None:
        rs = torch.rsqrt(up.pow(2).sum(-1, keepdim=True) * (1.0 / x2.shape[1]) + NF_EPS)
    return (up * rs).to(x2.dtype), rs


def nf_rs_bound(rs: torch.Tensor, x2: torch.Tensor) -> dict:
    """The stated bound on the kernel's rs: |rs / rs64 - 1| <= 2^-20 (8 fp32 ulps) against the float64 value of
    rsqrt(mean(x^2) + eps) of the same bf16 rows; returns the measured maximum, its ulps and the verdict."""
    x64 = x2.double()
    r64 = torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + NF_EPS)
    rel = float(((rs.double() / r64) - 1.0).abs().max())
    return {"max_rel": rel, "ulps": rel / 2.0 ** -24, "ok": rel <= 2.0 ** -20}


def emulate_fwd_nf(x2: torch.Tensor, pack: torch.Tensor, kind: int, s_a, dq, cw: int,
                   rs: torch.Tensor | None = None):
    """_proj8_fwd_nf_kernel's arithmetic: (out, xn, rs) with xn and rs from norm_rows_nf and out = emulate_fwd_pc on xn
    (switch 9g's emulation, so switch 9d's arithmetic)."""
    xn, rs = norm_rows_nf(x2, rs)
    return emulate_fwd_pc(xn, pack, kind, s_a, dq, cw), xn, rs
