"""nki_attn_proj8.py: switch 9d (R2_ATTN_FP8): the attention block's two projections in fp8 at the TensorE double-row
rate, with switch 9c's scheme (work/p2/sched1/kernels/nki_mlp_fused8.py 18369d38): e4m3 weights and activations at
per-tensor power-of-two scales (the chip's legacy e4m3: largest finite 240), e5m2 gradients from their exact amax per
call, fp32 PSUM accumulation, the weight gradients left to the caller in bf16.

  _proj8_fwd_kernel(a2, w0, w1, w2, sc, ca, cw, cd, tb, cwid, forms) -> out [M, 3 Nw]     (the QKV projection)
  _proj8_fwd1_kernel(a2, w0, sc, ca, cw, cd, tb, cwid, forms) -> out [M, Nw]                (c_proj)
      out = bf16(((e4m3(a2 x sc[ca]) @ e4m3(W x sc[cw])^T) x sc[cd]), W = [w0; w1; w2] or w0 (each [Nw, K], the
      nn.Linear layout: the QKV projection takes c_q, c_k, c_v as three inputs, so no concatenated weight is ever
      written).
  _proj8_bwd_kernel(g2, w0, w1, w2, sc, ca, cw, cd, tb, cwid, forms) -> dx [M, K]
  _proj8_bwd1_kernel(g2, w0, sc, ca, cw, cd, tb, cwid, forms) -> dx [M, K]
      dx = bf16((e5m2(g2 x sc[ca]) @ e4m3(W x sc[cw])) x sc[cd]), the activation gradient of the same projection.
  The four launches share two body helpers (_fwd_body, _bwd_body, with nw 3 or 1); every input tensor appears once in
  a launch's arguments.

Layout (token-stationary, no output transposes): per block of tb tokens the activation's transposed tiles
at[p, t x NK + s, i] = a2[row0 + t x 128 + i, s x 128 + p] are loaded by the hardware-DGE strip transpose (switch 9c's
_load_T_block instruction with a token-tile-major destination) and cast once to fp8; the weight streams in chunks of
cwid output columns (transposed for the forward by the same strip transpose, natural rows for the backward by one
three-level DMA), cast to e4m3 per chunk; then for each 128-token tile one PSUM bank [128 tokens, cwid] accumulates
K / 256 double-row matmuls with the activation tile STATIONARY ([128, 2, 128]: two 128-row slabs of one tile, the
fused8 kernels' stationary form) and the weight chunk MOVING ([128, 2, cwid]); the evacuation multiplies by the
descale on VectorE into bf16 (fused8's _evac_desc) and one DMA writes the token-major rows. Token-major PSUM is what
the next consumer reads (the attention kernel reads qkv [b, t, 3c]; the residual add reads c_proj's output), so the
TensorE transposes of fused8's weight-stationary form are not needed: at K 1,536 they would cost about a sixth of the
GEMM's TensorE time.

Every scale is a power of two, so every cast multiply and every descale is exact, and the torch emulation below has
the kernels' rounding points (fp8 casts, fp32 products and accumulation, one bf16 rounding of acc x descale); the
NKI simulator proves the kernels equal it up to summation order (proofs/sim_fp8q.py). The scales arrive as one fp32
tile sc [128, NS] per call (each column one value down every partition), read by one DMA; ca, cw, cd name the
columns. forms 0 is the design's instruction set; forms 1 uses only forms our device has compiled: switch 9c's
two-instruction cast in place of the activation with a vector scale into fp8, and in the QKV backward each weight's
rows in a whole tile of its own (9c's _load_w_chunk DMA) with one PSUM chain across the three tiles, in place of three
DMAs into one tile's slab ranges (bitwise the same results).

Shapes: M a multiple of 128 (the caller pads with zero rows: a zero row gives a zero output row and changes no other
row), K and Nw multiples of 256 (the double-row pair) and of cwid; tb a multiple of 128 up to 2048. plan() picks tb
and cwid so the SBUF a partition holds stays at most PLAN_KIB.

Written for the device-side NKI parser frontend under switch 9c's rules (its module docstring): every loop is
`for name in range(...)` with one plain target; `if` only on trace-time integers; tuples only returned and unpacked in
assignments; no list, comprehension, `while`, lambda or nested def in a traced body; every slice bound a plain name or
a literal; every PSUM tile 2-D and one bank; a tile subscripted once.
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
