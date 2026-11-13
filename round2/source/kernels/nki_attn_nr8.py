"""nki_attn_nr8.py: switch 9t (R2_NR, lane THR12, work/p2/thr12/NOTES.md), unit attn: the attention side's
U-net skip add, residual mix and RMSNorm folded into switch 9q's QKV forward, and the attention residual add folded into
its projection forward, so no compiled add, mix or norm runs between the MLP's kernel and the QKV kernel or between
the projection's kernel and the MLP's. PF7's captures (lane THR12's NOTES P.3): the MLP -> QKV window is 618 us a
layer-micro-step with 167 us of glue alone on the engines and TensorE idle; the projection -> MLP window holds the add.

  The QKV forward, four entry points over one body (_nm_body), so that no kernel argument is ever bound twice to one
  tensor (an aliased launch has never run on the device):
    _proj8_fwd_nm_kernel(xa, x0, lam, w8, sc, base, ca, cd, cwid, u8)         mode 0: no skip
    _proj8_fwd_nm_sk_kernel(xa, x0, sk, lam, w8, sc, base, ca, cd, cwid, u8)  mode 1: the U-net skip sk
    _proj8_fwd_nm_b0_kernel(x0, lam, w8, sc, base, ca, cd, cwid, u8)          mode 2: block 0 (its input is x0)
    _proj8_fwd_nm_ss_kernel(xa, x0, lam, w8, sc, base, ca, cd, cwid, u8)      mode 3: the skip is the input xa
  each -> out [M, 3 d], xm, xn [M, d], rs [M, 1] fp32. lam [P, 4] fp32: the gains as bf16 values down every
  partition (resid_lambda, x0_lambda, skip_lambda, unused). Per token tile, in eager torch's order and roundings:
  s = bf16(sl sk), x_s = bf16(xa + s) (modes 1 and 3), t1 = bf16(a x_s), t2 = bf16(b x0), xm = bf16(t1 + t2)
  (stored: the attention residual's left operand and the norm backward's input); then the norm (switch 9m's
  instructions; xn and rs stored), switch 9q's transposes of xn's slabs into [P, ng, P] PSUM groups and one cast a
  group at X_SCALE, and switch 9j's matmul chains and descales against every chunk of the three weights, the tile's
  outputs stored once. The weights stay in SBUF for the call (one tile each), so the loop is tile-major and
  software-pipelined: tile i + 1's mix, norm and casts are issued between tile i's chains, tile i + 2's rows at tile
  i's start, every per-tile buffer a parity pair (PSUM included). Each (tile, chunk) chain is switch 9j's
  _tile_gemm8 instruction for instruction, so out is switch 9q's QKV forward fed xn, bit for bit.
  _proj8_fwd_ra_kernel(a2, res, w8, sc, base, ca, cd, cwid, u8) -> out [M, d]
      switch 9q's c_proj forward (forms 0) with the residual added in the evacuation: out = bf16(res + bf16(acc dq)),
      the eager add after the projection; tile-major with the weight resident and pipelined as the QKV body; every
      other instruction is 9q's.
  Shapes: LNC 1 or 2 (a program a physical core), M a multiple of 128, d a multiple of 256 and of the chunk width;
  forms 0 only. SBUF a partition at d 1,536, cwid 512: the QKV kernels about 111 KiB (plan_kib_nm; the nine weight
  chunks 54), the projection about 39 (plan_kib_ra), of 168. PSUM: two accumulator banks and a pair of 1 KiB
  transpose groups.

Everything after this paragraph is nki_attn_tt8g.py's text (md5 2f0d0c02) but for the lane THR12 block before
`_HAS_NKI = True` and the lane THR12 torch block at the end (tools/build_nr.py).

nki_attn_tt8g.py: switch 9q (R2_ATTN_TTG, lane THR10, work/p2/thr10/NOTES.md): switch 9j's module with ONE
ScalarE activation a PSUM group in _load_T8_tiles. Under 9j each transposed 128 x 128 slab is cast out of PSUM by its
own activation, 128 elements a partition at about 343 ns an instruction, and in every tt kernel call that cast phase
holds ScalarE at 100 percent while TensorE, in order, waits to start the block's matmuls (PF7's captures: 30.7 ms of
casts a step a core). Under 9q (forms 0) the group's slabs, up to TT_GROUP = 4, are transposed into a [P, ng, P] bf16
PSUM tile and cast by one activation into a8[:, t NKB + g0 : t NKB + g0 + ng, :], where they sit contiguously: the
same instruction (copy with the scale column, one rounding into fp8) on the same values, element for element, so every
output is 9j's bit for bit; 4 x fewer ScalarE instructions in the phase. forms 1 is 9j's per-slab path verbatim.
Everything after this paragraph is nki_attn_tt8.py's text (md5 32046b16) but for _load_T8_tiles (tools/build_tt8g.py).

nki_attn_tt8.py: switch 9j (R2_ATTN_TT, lane THR8): switch 9g's pre-cast fp8 attention projections with each
activation operand made into its fp8 tiles by TensorE transposes instead of DMA strip transposes. Everything after
this docstring up to the lane THR8 block is switch 9g's module kernels/nki_attn_pc8.py (md5 91c0a087) byte for byte
(tools/build_tt.py), so switch 9d's four kernels and switch 9g's pre-cast and pc kernels are here unchanged and the
probe compares the new kernels against them in one module.

Switch 9d's kernels (and 9g's pc forms of them) load each block's activation operand (x into the QKV projection, y
into c_proj, dqkv and d(out) in the two input gradients) by the hardware-DGE strip transpose, 16 tokens over all
contraction slabs an instruction (8 instructions a 128-token tile), into bf16 tiles at[p, t NK + s, i] = a2[row0 +
t 128 + i, s 128 + p], and cast the whole block to fp8 on ScalarE. Switch 9j's kernels:

  _proj8_fwd_tt_kernel(a2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8) -> out [M, nw x d]
  _proj8_bwd_tt_kernel(g2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8) -> dx [M, d]
      _fwd_pc_body and _bwd_pc_body with the activation's tiles made by _load_T8_tiles: each 128-token tile's rows by
      one natural DMA [128 tokens, K] (contiguous rows), its K / 128 bf16 slabs transposed on TensorE (nc_transpose
      into a bf16 PSUM tile of up to TT_GROUP slabs, switch 9c's _transpose_group form; a TensorE transpose is
      bit-accurate on NeuronCore-v3, so PSUM holds the same bf16 values the strip transpose writes), and each
      transposed slab cast to fp8 straight out of PSUM into a8[:, t NK + s, :] by switch 9d's _cast8 (the same
      instruction, the same scale column and the same per-element product and rounding as 9d's cast of the block: the
      cast doubles as the PSUM evacuation). Every later instruction (the weight tiles from the pack, the double-row
      matmuls, the descale, the stores) is 9g's, so the outputs are switch 9g's kernels' (and so switch 9d's) bit for
      bit. (An fp8 transpose would need its PSUM output at an element step of 2, the device compiler's verifier rule;
      the bf16 transpose needs nothing new.)

SBUF a partition: the fp8 block (tb / 128 x K bytes) and one token tile's bf16 rows (2 K bytes) in place of the bf16
and fp8 block (3 tb / 128 x K bytes): 24 + 3 KiB against 72 KiB at the QKV forward's 2,048 x 1,536, 36 + 9 against
108 at the QKV backward's 1,024 x 4,608. HBM: nothing new. PSUM: one 1 KiB bank share a group.

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

    # ---- lane THR8, switch 9j (R2_ATTN_TT): everything from here to `_HAS_NKI = True` is new; every function above
    # it is nki_attn_pc8.py 91c0a087's after its docstring, byte for byte (tools/build_tt.py copies them)

    TT_GROUP = 4        # slabs a transpose PSUM tile holds: 4 x 128 bf16 = 1 KiB a partition (switch 9c's group)

    def _load_T8_tiles(a8, a2, K, NKB, row0, nt, dt, sc_sb, col, forms):
        """Switch 9j: a8 [P, nt x NKB, P] fp8 = _cast8 of _load_T_tiles' tiles, a8[p, t NKB + s, i] = fp8(a2[row0 +
        t 128 + i, s 128 + p] x sc[:, col]): each token tile's 128 rows by one natural DMA, its NKB slabs transposed on
        TensorE through a dt PSUM tile of up to TT_GROUP slabs (switch 9c's _transpose_group form), each slab cast out
        of PSUM into a8[:, t NKB + s, :] by _cast8. Switch 9q (forms 0): the group's PSUM tile is [P, ng, P] and ONE
        activation (_cast8's: copy with the scale column) casts the whole group into a8[:, t NKB + g0 : + ng, :], the
        same product and rounding element for element; forms 1 is 9j's per-slab path."""
        for t in range(nt):
            r0 = row0 + t * _P
            r1 = r0 + _P
            an = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=an, src=a2[r0:r1, 0:K])
            for g0 in range(0, NKB, TT_GROUP):
                ng = NKB - g0
                if ng > TT_GROUP:
                    ng = TT_GROUP
                if forms == 1:
                    pst = nl.ndarray((_P, ng * _P), dtype=dt, buffer=nl.psum)
                    for gg in range(ng):
                        q0 = gg * _P
                        q1 = q0 + _P
                        s0 = (g0 + gg) * _P
                        s1 = s0 + _P
                        nisa.nc_transpose(dst=pst[:, q0:q1], data=an[:, s0:s1])
                    for gg in range(ng):
                        q0 = gg * _P
                        q1 = q0 + _P
                        j = t * NKB + g0 + gg
                        _cast8_ps(a8[:, j, :], pst[:, q0:q1], sc_sb, col, forms, dt)
                else:
                    psg = nl.ndarray((_P, ng, _P), dtype=dt, buffer=nl.psum)
                    for gg in range(ng):
                        s0 = (g0 + gg) * _P
                        s1 = s0 + _P
                        nisa.nc_transpose(dst=psg[:, gg, :], data=an[:, s0:s1])
                    j0 = t * NKB + g0
                    j1 = j0 + ng
                    _cast8(a8[:, j0:j1, :], psg, sc_sb, col, 0)

    def _cast8_ps(dst8, src_ps, sc_sb, col, forms, dt):
        """Switch 9j: _cast8 of one transposed slab read from PSUM. forms 0: _cast8 itself (one ScalarE activation
        with the scale column, reading PSUM). forms 1: the exact product into a bf16 SBUF tile on VectorE (VectorE may
        not write a 2-byte PSUM tile), then the ScalarE copy into fp8 at scale 1.0: _cast8 forms 1's two roundings."""
        if forms == 1:
            q0 = col
            q1 = q0 + 1
            tq = nl.ndarray((_P, _P), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=tq, data=src_ps, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)
            nisa.activation(dst=dst8, op=nl.copy, data=tq, scale=1.0)
        else:
            _cast8(dst8, src_ps, sc_sb, col, 0)

    def _fwd_tt_body(out, a2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8):
        """Switch 9j: _fwd_pc_body with the activation's fp8 tiles from _load_T8_tiles."""
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
            a8 = nl.ndarray((_P, nt * NK, _P), dtype=_F8E4, buffer=nl.sbuf)
            _load_T8_tiles(a8, a2, K, NK, row0, nt, dt, sc_sb, ca, forms)
            for c in range(NC):
                w8t = _load_w8_cols(w8, base * DD + c * K * cwid, NK, cwid, u8)
                o0 = c * cwid
                for t in range(nt):
                    _tile_gemm8(out, a8, w8t, sc_sb, cd, t, NK, NKD, row0, o0, cwid, dt)

    def _bwd_tt_body(dx, g2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8):
        """Switch 9j: _bwd_pc_body with the gradient's e5m2 tiles from _load_T8_tiles."""
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
            g8 = nl.ndarray((_P, nt * NN, _P), dtype=_F8E5, buffer=nl.sbuf)
            _load_T8_tiles(g8, g2, N, NN, row0, nt, dt, sc_sb, ca, forms)
            for c in range(NC):
                k0 = c * cwid
                w8t = _load_w8_cols(w8, base * DD + c * _P * NN * cwid, NN, cwid, u8)
                for t in range(nt):
                    _tile_gemm8(dx, g8, w8t, sc_sb, cd, t, NN, NND, row0, k0, cwid, dt)

    @nki.jit
    def _proj8_fwd_tt_kernel(a2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8):
        nout = nw * (w8.shape[1] // a2.shape[1])
        out = nl.ndarray((a2.shape[0], nout), dtype=a2.dtype, buffer=nl.shared_hbm)
        _fwd_tt_body(out, a2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8)
        return out

    @nki.jit
    def _proj8_bwd_tt_kernel(g2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8):
        kout = w8.shape[1] // (g2.shape[1] // nw)
        dx = nl.ndarray((g2.shape[0], kout), dtype=g2.dtype, buffer=nl.shared_hbm)
        _bwd_tt_body(dx, g2, w8, sc, nw, base, ca, cd, tb, cwid, forms, u8)
        return dx

    # ---- lane THR12, switch 9t (R2_NR), unit attn: everything from here to `_HAS_NKI = True` is new;
    # every function above it is nki_attn_tt8g.py 2f0d0c02's, byte for byte (tools/build_nr.py copies them)

    NR_EPS = 2.0 ** -23         # F.rms_norm's eps for bf16 inputs (the opmath type's epsilon; switch 9m's NF_EPS)

    def _nr_pair(shape, dtype):
        """Switch 9t: two SBUF tiles of one shape, a parity pair: consecutive steps never share a buffer, so a step's
        loads and work can run under the previous step's (lane OS1A2's _lh_pair form; a tile allocated inside a loop
        can take the previous step's address, and the WAR then serialises the steps: work/p2/l2p/NOTES.md 9.3)."""
        return (nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf), nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf))

    def _nr_ppair(shape, dtype):
        """Switch 9t: a parity pair of PSUM tiles (each at most one bank), _nr_pair's form."""
        return (nl.ndarray(shape, dtype=dtype, buffer=nl.psum), nl.ndarray(shape, dtype=dtype, buffer=nl.psum))

    def _nr_ptriple(shape, dtype):
        """Switch 9t: three PSUM tiles of one shape, a rotation for the transpose groups: with three groups a token
        tile (d 1,536), a group's transposes never wait for a cast of the same tile."""
        return (nl.ndarray(shape, dtype=dtype, buffer=nl.psum), nl.ndarray(shape, dtype=dtype, buffer=nl.psum),
                nl.ndarray(shape, dtype=dtype, buffer=nl.psum))

    def _nr_pquad(shape, dtype):
        """Switch 9t: four PSUM tiles of one shape (each at most one bank), a rotation: a chain's accumulator is
        written again only four chains later, so one long VectorE item cannot stall the next two chains."""
        return (nl.ndarray(shape, dtype=dtype, buffer=nl.psum), nl.ndarray(shape, dtype=dtype, buffer=nl.psum),
                nl.ndarray(shape, dtype=dtype, buffer=nl.psum), nl.ndarray(shape, dtype=dtype, buffer=nl.psum))

    def _nr_imin(a, b):
        """Switch 9t: the smaller of two trace-time integers."""
        if a < b:
            return a
        return b

    def _nm_wsel(c, w0, w1, w2, w3, w4, w5, w6, w7, w8_):
        """Switch 9t: the resident tile of weight chunk c (c < 9; each chunk its own tile, loaded by switch 9g's
        _load_w8_cols, so a chain waits only for its own chunk's DMA)."""
        if c == 0:
            return w0
        if c == 1:
            return w1
        if c == 2:
            return w2
        if c == 3:
            return w3
        if c == 4:
            return w4
        if c == 5:
            return w5
        if c == 6:
            return w6
        if c == 7:
            return w7
        return w8_

    def _nr_norm1(xs_t, sq, ss):
        """Switch 9t: the norm's first instruction on one 128-token tile xs_t [P, D] bf16 (tokens on the partitions):
        ss = the sum of the squares in fp32 (one ScalarE activation_reduce: the exact fp32 squares of the bf16
        values, summed in the engine's order; sq is its elementwise output, never read)."""
        nisa.activation_reduce(dst=sq, op=nl.square, data=xs_t, reduce_op=nl.add, reduce_res=ss)

    def _nr_norm2(ss, rq, rs, r0, inv_d):
        """Switch 9t: rq = rsqrt(ss x (1/D) + 2^-23) (two tensor_scalar instructions, switch 9m's chip-run form: the
        mean as a product by 1/D) and its store into rs[r0 : r0 + 128] (fp32, the backward reads it)."""
        r1 = r0 + _P
        nisa.tensor_scalar(dst=ss, data=ss, op0=nl.multiply, operand0=inv_d, op1=nl.add, operand1=NR_EPS)
        nisa.tensor_scalar(dst=rq, data=ss, op0=nl.rsqrt, operand0=0.0)
        nisa.dma_copy(dst=rs[r0:r1, :], src=rq)

    def _nr_norm3(xs_t, xq, rq, xn, r0):
        """Switch 9t: xq = bf16(x rq) (VectorE: the fp32 product rounded once, where F.rms_norm's composite rounds
        once) and its store into xn[r0 : r0 + 128]. _nr_norm1, _nr_norm2, _nr_norm3 in turn are switch 9m's
        _nf_norm_tile instruction for instruction; split so a pipeline can issue them in separate slots."""
        r1 = r0 + _P
        nisa.tensor_scalar(dst=xq, data=xs_t, op0=nl.multiply, operand0=rq, engine=_VEC)
        nisa.dma_copy(dst=xn[r0:r1, :], src=xq)

    def _nm_cast_tile(a8, src, NKB, dt, sc_sb, col, psgs, gb):
        """Switch 9t: a8 [P, NKB, P] e4m3, a8[p, s, i] = fp8(src[i, s 128 + p] x sc[:, col]) for one token tile src
        [P, K] in SBUF: its NKB bf16 slabs transposed on TensorE into [P, ng, P] PSUM groups of up to TT_GROUP slabs,
        each group cast by ONE activation (_cast8, forms 0): switch 9q's _load_T8_tiles after its row DMA,
        instruction for instruction. The groups rotate through the three PSUM tiles psgs by the call's group count
        gb + g, so a group's transposes never wait for a cast of the same tile."""
        for g0 in range(0, NKB, TT_GROUP):
            ng = NKB - g0
            if ng > TT_GROUP:
                ng = TT_GROUP
            gq = (gb + g0 // TT_GROUP) % 3
            psg = psgs[gq]
            for gg in range(ng):
                s0 = (g0 + gg) * _P
                s1 = s0 + _P
                nisa.nc_transpose(dst=psg[:, gg, :], data=src[:, s0:s1])
            g1 = g0 + ng
            _cast8(a8[:, g0:g1, :], psg[:, 0:ng, :], sc_sb, col, 0)

    def _nm_chain(ob, a8, wr, wk0, sc_sb, cd, NKD, o0, cwid, ps):
        """Switch 9t: switch 9j's _tile_gemm8 for one token tile a8 [P, NK, P] against one weight chunk held in the
        resident tile wr at slab offset wk0: the same matmul chain (stationary the tile's double-row slab pairs,
        moving the chunk's) into the PSUM tile ps (the caller's pair), the same descale on VectorE into the tile's
        output stage ob[:, o0 : o0 + cwid] (stored once a tile), operand for operand."""
        for kd in range(NKD):
            j0 = kd * 2
            j1 = j0 + 2
            k0 = wk0 + kd * 2
            k1 = k0 + 2
            nisa.nc_matmul(dst=ps, stationary=a8[:, j0:j1, :], moving=wr[:, k0:k1, :], accumulate=(kd > 0),
                           perf_mode=_DR)
        q0 = cd
        q1 = q0 + 1
        o1 = o0 + cwid
        nisa.tensor_scalar(dst=ob[:, o0:o1], data=ps, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)

    def _nm_load(mode, xr, x0r, skr, xa, x0, sk, r0, K):
        """Switch 9t: one token tile's input rows by natural DMAs: xa's (not in mode 2, where xa is x0), x0's, and the
        skip's in mode 1 (in mode 3 the skip is xa, already loaded)."""
        r1 = r0 + _P
        if mode != 2:
            nisa.dma_copy(dst=xr, src=xa[r0:r1, 0:K])
        nisa.dma_copy(dst=x0r, src=x0[r0:r1, 0:K])
        if mode == 1:
            nisa.dma_copy(dst=skr, src=sk[r0:r1, 0:K])

    def _nm_mix_part(k, mode, xr, x0r, skr, xmt, lam_sb, xm, r0):
        """Switch 9t: part k of the block input's eager ops on one token tile, in eager torch's order and roundings
        (lam_sb's columns: resid_lambda a, x0_lambda b, the skip's lambda sl as bf16 values). Skip modes (1 and 3)
        have five parts: s = bf16(sl sk) (mode 3: sk is xa's tile), x_s = bf16(xa + s), t1 = bf16(a x_s), t2 = bf16(b
        x0), xm = bf16(t1 + t2) and its store; modes 0 and 2 the last three (mode 2: t1 from x0's tile, xa being x0).
        In place on the row tiles, as the first stage's kernel did."""
        hs = mode % 2
        kk = k - 2 * hs
        if kk == -2:
            if mode == 3:
                s_src = xr
            else:
                s_src = skr
            nisa.tensor_scalar(dst=skr, data=s_src, op0=nl.multiply, operand0=lam_sb[:, 2:3], engine=_VEC)
        if kk == -1:
            nisa.tensor_tensor(dst=xr, data1=xr, data2=skr, op=nl.add, engine=_VEC)
        if kk == 0:
            if mode == 2:
                a_src = x0r
            else:
                a_src = xr
            nisa.tensor_scalar(dst=xr, data=a_src, op0=nl.multiply, operand0=lam_sb[:, 0:1], engine=_VEC)
        if kk == 1:
            nisa.tensor_scalar(dst=x0r, data=x0r, op0=nl.multiply, operand0=lam_sb[:, 1:2], engine=_VEC)
        if kk == 2:
            nisa.tensor_tensor(dst=xmt, data1=xr, data2=x0r, op=nl.add, engine=_VEC)
            r1 = r0 + _P
            nisa.dma_copy(dst=xm[r0:r1, :], src=xmt)

    def _nm_body(out, xm, xn, rs, xa, x0, sk, lam, w8, sc, base, ca, cd, cwid, mode, u8):
        """Switch 9t, unit attn: the QKV forward with the block input's prologue (the module docstring), tile-major
        with the weights resident (each chunk its own tile, at most nine), software-pipelined. Order a program: tile
        0's rows, chunk 0, tile 1's rows, the other chunks; tile 0's mix, norm and casts; then for tile i: tile i +
        2's rows at its start, and its chains (the accumulators a rotation of four) with tile i + 1's mix parts, norm
        and casts issued between them in slots (the mix one part a slot from slot 0, the norm's three steps after
        it, the casts before the last chain), then the tile's one output store. Every instruction is the first stage
        kernel's, on the same operands: only their order and the buffers change."""
        M = x0.shape[0]
        K = x0.shape[1]
        DD = w8.shape[1]
        Nw = DD // K
        dt = x0.dtype
        f32 = nl.float32
        NK = K // _P
        NKD = K // _KS
        NCw = Nw // cwid
        NC = NCw * 3
        NG = _ceil_div(NK, TT_GROUP)
        NO = Nw * 3
        inv_d = 1.0 / K
        hs = mode % 2
        npart = 3 + 2 * hs
        s_n1 = _nr_imin(npart - 1, NC - 1)
        s_n2 = _nr_imin(npart, NC - 1)
        s_n3 = _nr_imin(npart + 1, NC - 1)
        s_c = NC - 2
        if s_c < s_n3:
            s_c = s_n3
        sc_sb = _load_sc(sc, sc.shape[1])
        lam_sb = _load_sc(lam, lam.shape[1])
        pid, nprg = _shard()
        t0, t1 = _span(M // _P, pid, nprg)
        n = t1 - t0
        xrs = _nr_pair((_P, K), dt)
        x0s = _nr_pair((_P, K), dt)
        if hs == 1:
            sks = _nr_pair((_P, K), dt)
        else:
            sks = xrs
        xms = _nr_pair((_P, K), dt)
        sq = nl.ndarray((_P, K), dtype=f32, buffer=nl.sbuf)
        sss = _nr_pair((_P, 1), f32)
        rqs = _nr_pair((_P, 1), f32)
        xqs = _nr_pair((_P, K), dt)
        a8s = _nr_pair((_P, NK, _P), _F8E4)
        obs = _nr_pair((_P, NO), dt)
        pss = _nr_pquad((_P, cwid), f32)
        psgs = _nr_ptriple((_P, TT_GROUP, _P), dt)
        r00 = t0 * _P
        wb = base * DD
        _nm_load(mode, xrs[0], x0s[0], sks[0], xa, x0, sk, r00, K)
        wc0 = _load_w8_cols(w8, wb, NK, cwid, u8)
        if n > 1:
            _nm_load(mode, xrs[1], x0s[1], sks[1], xa, x0, sk, r00 + _P, K)
        wc1 = wc0
        wc2 = wc0
        wc3 = wc0
        wc4 = wc0
        wc5 = wc0
        wc6 = wc0
        wc7 = wc0
        wc8 = wc0
        if NC > 1:
            wc1 = _load_w8_cols(w8, wb + 1 * K * cwid, NK, cwid, u8)
        if NC > 2:
            wc2 = _load_w8_cols(w8, wb + 2 * K * cwid, NK, cwid, u8)
        if NC > 3:
            wc3 = _load_w8_cols(w8, wb + 3 * K * cwid, NK, cwid, u8)
        if NC > 4:
            wc4 = _load_w8_cols(w8, wb + 4 * K * cwid, NK, cwid, u8)
        if NC > 5:
            wc5 = _load_w8_cols(w8, wb + 5 * K * cwid, NK, cwid, u8)
        if NC > 6:
            wc6 = _load_w8_cols(w8, wb + 6 * K * cwid, NK, cwid, u8)
        if NC > 7:
            wc7 = _load_w8_cols(w8, wb + 7 * K * cwid, NK, cwid, u8)
        if NC > 8:
            wc8 = _load_w8_cols(w8, wb + 8 * K * cwid, NK, cwid, u8)
        for k in range(npart):
            _nm_mix_part(k, mode, xrs[0], x0s[0], sks[0], xms[0], lam_sb, xm, r00)
        _nr_norm1(xms[0], sq, sss[0])
        _nr_norm2(sss[0], rqs[0], rs, r00, inv_d)
        _nr_norm3(xms[0], xqs[0], rqs[0], xn, r00)
        _nm_cast_tile(a8s[0], xqs[0], NK, dt, sc_sb, ca, psgs, 0)
        for i in range(n):
            q = i % 2
            p = 1 - q
            r0 = r00 + i * _P
            r1 = r0 + _P
            r2 = r1 + _P
            if i + 2 < n:
                _nm_load(mode, xrs[q], x0s[q], sks[q], xa, x0, sk, r2, K)
            ob = obs[q]
            a8 = a8s[q]
            for c in range(NC):
                wr = _nm_wsel(c, wc0, wc1, wc2, wc3, wc4, wc5, wc6, wc7, wc8)
                gq = (i * NC + c) % 4
                _nm_chain(ob, a8, wr, 0, sc_sb, cd, NKD, c * cwid, cwid, pss[gq])
                if i + 1 < n:
                    for k in range(npart):
                        slot = _nr_imin(k, NC - 1)
                        if slot == c:
                            _nm_mix_part(k, mode, xrs[p], x0s[p], sks[p], xms[p], lam_sb, xm, r1)
                    if s_n1 == c:
                        _nr_norm1(xms[p], sq, sss[p])
                    if s_n2 == c:
                        _nr_norm2(sss[p], rqs[p], rs, r1, inv_d)
                    if s_n3 == c:
                        _nr_norm3(xms[p], xqs[p], rqs[p], xn, r1)
                    if s_c == c:
                        _nm_cast_tile(a8s[p], xqs[p], NK, dt, sc_sb, ca, psgs, (i + 1) * NG)
            nisa.dma_copy(dst=out[r0:r1, 0:NO], src=ob)

    @nki.jit
    def _proj8_fwd_nm_kernel(xa, x0, lam, w8, sc, base, ca, cd, cwid, u8):
        """Switch 9t, unit attn, mode 0 (no skip; xa and x0 distinct tensors): the mix, the norm and switch 9q's QKV
        forward (_nm_body). Returns out [M, 3 Nw], xm, xn [M, K], rs [M, 1] fp32."""
        M = x0.shape[0]
        K = x0.shape[1]
        dt = x0.dtype
        out = nl.ndarray((M, 3 * (w8.shape[1] // K)), dtype=dt, buffer=nl.shared_hbm)
        xm = nl.ndarray((M, K), dtype=dt, buffer=nl.shared_hbm)
        xn = nl.ndarray((M, K), dtype=dt, buffer=nl.shared_hbm)
        rs = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        _nm_body(out, xm, xn, rs, xa, x0, x0, lam, w8, sc, base, ca, cd, cwid, 0, u8)
        return out, xm, xn, rs

    @nki.jit
    def _proj8_fwd_nm_sk_kernel(xa, x0, sk, lam, w8, sc, base, ca, cd, cwid, u8):
        """Switch 9t, unit attn, mode 1 (a decoder block; xa, x0 and the skip sk three distinct tensors): the skip
        add, the mix, the norm and switch 9q's QKV forward (_nm_body). Returns out, xm, xn, rs."""
        M = x0.shape[0]
        K = x0.shape[1]
        dt = x0.dtype
        out = nl.ndarray((M, 3 * (w8.shape[1] // K)), dtype=dt, buffer=nl.shared_hbm)
        xm = nl.ndarray((M, K), dtype=dt, buffer=nl.shared_hbm)
        xn = nl.ndarray((M, K), dtype=dt, buffer=nl.shared_hbm)
        rs = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        _nm_body(out, xm, xn, rs, xa, x0, sk, lam, w8, sc, base, ca, cd, cwid, 1, u8)
        return out, xm, xn, rs

    @nki.jit
    def _proj8_fwd_nm_b0_kernel(x0, lam, w8, sc, base, ca, cd, cwid, u8):
        """Switch 9t, unit attn, mode 2 (block 0, whose input is x0 itself: one tensor, read once): the mix a x0 + b
        x0, the norm and switch 9q's QKV forward (_nm_body). Returns out, xm, xn, rs."""
        M = x0.shape[0]
        K = x0.shape[1]
        dt = x0.dtype
        out = nl.ndarray((M, 3 * (w8.shape[1] // K)), dtype=dt, buffer=nl.shared_hbm)
        xm = nl.ndarray((M, K), dtype=dt, buffer=nl.shared_hbm)
        xn = nl.ndarray((M, K), dtype=dt, buffer=nl.shared_hbm)
        rs = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        _nm_body(out, xm, xn, rs, x0, x0, x0, lam, w8, sc, base, ca, cd, cwid, 2, u8)
        return out, xm, xn, rs

    @nki.jit
    def _proj8_fwd_nm_ss_kernel(xa, x0, lam, w8, sc, base, ca, cd, cwid, u8):
        """Switch 9t, unit attn, mode 3 (the first decoder block, whose U-net skip is its input xa itself: read once):
        the skip add bf16(xa + bf16(sl xa)), the mix, the norm and switch 9q's QKV forward (_nm_body). Returns out,
        xm, xn, rs."""
        M = x0.shape[0]
        K = x0.shape[1]
        dt = x0.dtype
        out = nl.ndarray((M, 3 * (w8.shape[1] // K)), dtype=dt, buffer=nl.shared_hbm)
        xm = nl.ndarray((M, K), dtype=dt, buffer=nl.shared_hbm)
        xn = nl.ndarray((M, K), dtype=dt, buffer=nl.shared_hbm)
        rs = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        _nm_body(out, xm, xn, rs, xa, x0, xa, lam, w8, sc, base, ca, cd, cwid, 3, u8)
        return out, xm, xn, rs

    def _ra_chain(ob, a8, wr, wk0, rr, sc_sb, cd, NKD, o0, cwid, ps):
        """Switch 9t, unit attn: _nm_chain with the residual added in the evacuation: the same matmul chain and
        descale into ob[:, o0 : o0 + cwid], then ob = bf16(rr + ob) there on VectorE (the eager add after the
        projection; rr the tile's residual rows)."""
        for kd in range(NKD):
            j0 = kd * 2
            j1 = j0 + 2
            k0 = wk0 + kd * 2
            k1 = k0 + 2
            nisa.nc_matmul(dst=ps, stationary=a8[:, j0:j1, :], moving=wr[:, k0:k1, :], accumulate=(kd > 0),
                           perf_mode=_DR)
        q0 = cd
        q1 = q0 + 1
        o1 = o0 + cwid
        nisa.tensor_scalar(dst=ob[:, o0:o1], data=ps, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)
        nisa.tensor_tensor(dst=ob[:, o0:o1], data1=rr[:, o0:o1], data2=ob[:, o0:o1], op=nl.add, engine=_VEC)

    @nki.jit
    def _proj8_fwd_ra_kernel(a2, res, w8, sc, base, ca, cd, cwid, u8):
        """Switch 9t, unit attn: switch 9q's c_proj forward (forms 0) with the residual res [M, Nw] added in each
        evacuation, out = bf16(res + bf16(acc dq)) (the eager add after the projection). Tile-major with the weight's
        chunks resident (each its own tile, at most three), software-pipelined: tile 0's rows, chunk 0, tile 0's
        residual, the other chunks, tile 1's rows and residual; then for tile i: tile i + 2's rows at its start, its
        chains with tile i + 1's transposes and casts before the last chain, the tile's one output store, tile i + 2's
        residual rows at its end."""
        M = a2.shape[0]
        K = a2.shape[1]
        DD = w8.shape[1]
        Nw = DD // K
        dt = a2.dtype
        f32 = nl.float32
        out = nl.ndarray((M, Nw), dtype=dt, buffer=nl.shared_hbm)
        NK = K // _P
        NKD = K // _KS
        NC = Nw // cwid
        NG = _ceil_div(NK, TT_GROUP)
        s_c = NC - 2
        if s_c < 0:
            s_c = 0
        sc_sb = _load_sc(sc, sc.shape[1])
        pid, nprg = _shard()
        t0, t1 = _span(M // _P, pid, nprg)
        n = t1 - t0
        ans = _nr_pair((_P, K), dt)
        rrs = _nr_pair((_P, Nw), dt)
        a8s = _nr_pair((_P, NK, _P), _F8E4)
        obs = _nr_pair((_P, Nw), dt)
        pss = _nr_ppair((_P, cwid), f32)
        psgs = _nr_ptriple((_P, TT_GROUP, _P), dt)
        r00 = t0 * _P
        r01 = r00 + _P
        r02 = r01 + _P
        wb = base * DD
        nisa.dma_copy(dst=ans[0], src=a2[r00:r01, 0:K])
        wc0 = _load_w8_cols(w8, wb, NK, cwid, u8)
        nisa.dma_copy(dst=rrs[0], src=res[r00:r01, 0:Nw])
        wc1 = wc0
        wc2 = wc0
        if NC > 1:
            wc1 = _load_w8_cols(w8, wb + 1 * K * cwid, NK, cwid, u8)
        if NC > 2:
            wc2 = _load_w8_cols(w8, wb + 2 * K * cwid, NK, cwid, u8)
        if n > 1:
            nisa.dma_copy(dst=ans[1], src=a2[r01:r02, 0:K])
            nisa.dma_copy(dst=rrs[1], src=res[r01:r02, 0:Nw])
        _nm_cast_tile(a8s[0], ans[0], NK, dt, sc_sb, ca, psgs, 0)
        for i in range(n):
            q = i % 2
            p = 1 - q
            r0 = r00 + i * _P
            r1 = r0 + _P
            ra = r1 + _P
            rb = ra + _P
            if i + 2 < n:
                nisa.dma_copy(dst=ans[q], src=a2[ra:rb, 0:K])
            ob = obs[q]
            a8 = a8s[q]
            rr = rrs[q]
            for c in range(NC):
                wr = _nm_wsel(c, wc0, wc1, wc2, wc0, wc0, wc0, wc0, wc0, wc0)
                gq = (i * NC + c) % 2
                _ra_chain(ob, a8, wr, 0, rr, sc_sb, cd, NKD, c * cwid, cwid, pss[gq])
                if i + 1 < n:
                    if s_c == c:
                        _nm_cast_tile(a8s[p], ans[p], NK, dt, sc_sb, ca, psgs, (i + 1) * NG)
            nisa.dma_copy(dst=out[r0:r1, 0:Nw], src=ob)
            if i + 2 < n:
                nisa.dma_copy(dst=rrs[q], src=res[ra:rb, 0:Nw])
        return out

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


# ----------------------------------------------------------------------------- lane THR8, switch 9j: host side

def sbuf_kib_tt(nk: int, tb: int, cwid: int) -> float:
    """Switch 9j: SBUF a partition (KiB) a tt kernel holds, as plan_kib counts it: the block's fp8 tiles, one token
    tile's bf16 rows, the pack's fp8 weight tile (double-buffered) and the evacuation tiles."""
    nt = tb // _P
    return (nt * nk * _P + 2 * nk * _P + 2 * nk * cwid + 2 * cwid * 2 + NSC * 4) / 1024.0


# ----------------------------------------------------------------------------- lane THR12, switch 9t: unit attn

NR_EPS_HOST = 2.0 ** -23    # NR_EPS outside the nki block (the same value)


def plan_kib_nm(d: int, cwid: int, nw: int = 3) -> float:
    """Switch 9t, unit attn: SBUF a partition (KiB) of the QKV kernels (_nm_body): the resident weight chunks (nw d /
    cwid of [P, d / 128, cwid] e4m3), the parity pairs of the row tiles (x, x0, skip, xm, xq bf16), the norm's
    scratch (fp32), the pair of one token tile's e4m3 slabs, the pair of a tile's output stages (nw d bf16) and the
    scale tiles."""
    nk = d // _P
    nc = (d // cwid) * nw
    wres = nc * nk * cwid
    rows = 2 * 5 * d * 2 + d * 4
    a8 = 2 * nk * _P
    return (wres + rows + a8 + 2 * nw * d * 2 + (NSC + 4) * 4) / 1024.0


def plan_kib_ra(d: int, cwid: int) -> float:
    """Switch 9t, unit attn: SBUF a partition (KiB) of _proj8_fwd_ra_kernel: the resident weight chunks, the pairs of
    a tile's input rows, residual rows, e4m3 slabs and output stage, and the scale tile."""
    nk = d // _P
    nc = d // cwid
    return (nc * nk * cwid + 2 * d * 2 + 2 * d * 2 + 2 * nk * _P + 2 * d * 2 + NSC * 4) / 1024.0


def shapes_apply_nm(m: int, d: int, cwid: int, lnc: int, cwid_p: int = 0) -> bool:
    """Switch 9t, unit attn: LNC 1 or 2, M a multiple of 128, d a multiple of 256 and of the QKV's chunk width cwid
    and the projection's cwid_p (cwid when 0), at most three chunks a weight (the kernels hold each chunk in its own
    tile: nine for the QKV, three for the projection), both SBUF plans within PLAN_KIB."""
    cp = cwid_p or cwid
    if lnc not in (1, 2) or m <= 0 or m % _P or d % _KS or d % cwid or d % cp or d // cwid > 3 or d // cp > 3:
        return False
    return plan_kib_nm(d, cwid) <= PLAN_KIB and plan_kib_ra(d, cp) <= PLAN_KIB


def lam_tile(a: torch.Tensor, b: torch.Tensor, sl) -> torch.Tensor:
    """Switch 9t: the kernel's gain tile [128, 4] fp32: resid_lambda, x0_lambda and the skip's lambda as their bf16
    values (the dtype the eager mix casts them to), down every partition; sl None is 0 (no skip)."""
    vals = [a.reshape(-1)[0:1].to(torch.bfloat16).float(), b.reshape(-1)[0:1].to(torch.bfloat16).float()]
    vals.append(torch.zeros_like(vals[0]) if sl is None else sl.reshape(-1)[0:1].to(torch.bfloat16).float())
    vals.append(torch.zeros_like(vals[0]))
    return torch.cat(vals).reshape(1, 4).expand(_P, 4).contiguous()


def nm_mix_ref(xa: torch.Tensor, x0: torch.Tensor, sk, a: torch.Tensor, b: torch.Tensor, sl) -> torch.Tensor:
    """Switch 9t (torch): the block's input as train.py's eager ops make it: x_s = xa + sl.to(bf16) sk (decoder
    blocks; sk None otherwise), xm = a.to(bf16) x_s + b.to(bf16) x0 (mix_residual's expression)."""
    dt = xa.dtype
    xs = xa if sk is None else xa + sl.to(dtype=dt) * sk
    return a.to(dtype=dt) * xs + b.to(dtype=dt) * x0


def nr_norm_rows(x2: torch.Tensor):
    """Switch 9t (torch): F.rms_norm's composite on [M, d] rows, op for op (switch 9m's nf_norm_rows: bitwise
    F.rms_norm(x2, (d,)) on CPU): (xn in x2's dtype, rs [M, 1] fp32)."""
    up = x2.float()
    rs = torch.rsqrt(up.pow(2).mean(-1, keepdim=True) + NR_EPS_HOST)
    return (up * rs).to(x2.dtype), rs
