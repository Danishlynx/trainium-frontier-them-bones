"""nki_mlp_fused.py: the fused ReLU-squared MLP of the Round 2 model as two NKI kernels (switch R2_MLP=fused).

The kit's MLP is c_proj(relu(c_fc(x))^2) with c_fc [F, d] and c_proj [d, F] (nn.Linear layouts, F = 4d), run as
two compiled matmuls with the activation materialised in HBM between them. Here:

  forward   y, rT = mlp_fwd(x2, w_fc, w_proj)
            x2 [M, d] bf16 token-major; w_fc [F, d], w_proj [d, F] bf16 (the Linear layouts, no transposed copies).
            Each program (kernel[2] under LNC 2: the two physical cores take contiguous halves of the M/128 token
            tiles) works in blocks of tb tokens. Phase 1: pre^T[f] = sum_k Wfc^T[k, f] x^T[k, :] accumulates over the
            d/128 slabs in fp32 PSUM (one one-bank tile per 512-column chunk of the block), the weight tiles
            stationary ([128 d, 128 F], from w_fc by the hardware-DGE 16-row strip transpose, the proven xt=hwt
            loader of candidate 33), the x^T block moving (the same strip transpose of x2, once per block). On
            evacuation r = relu(pre) (VectorE, bf16) is written to rT [F, M] (the one tensor saved for the backward:
            what torch's relu saves) and h^T = r^2 (ScalarE, bf16) stays in SBUF for the whole block: h never
            touches HBM. Phase 2: y^T[j] = sum_f Wproj^T[f, j] h^T[f, :] accumulates over the F/128 slabs, the
            weight tiles stationary (strips of w_proj), h^T moving; y^T is transposed back to token-major by TensorE
            (128 x 128 tiles through a bf16 PSUM tile) and stored as y [M, d].
  backward  dx, h, dpre = mlp_bwd(dy2, rT, w_fc, w_proj)
            Phase A: dh^T[f] = sum_k Wproj[k, f] dy^T[k, :] (natural [128 d, 128 F] slices of w_proj stationary,
            the dy^T strip-transposed block moving), and the activation gradient is the evacuation: dpre^T =
            (dh * 2) * r in one VectorE instruction reading the PSUM accumulator and the rT tile; dpre^T stays in
            SBUF for the block. h^T = r^2 is rebuilt from the same rT tile (one ScalarE square). Phase B:
            dx^T[j] = sum_f Wfc[f, j] dpre^T[f, :] (natural slices of w_fc stationary, dpre^T moving), transposed to
            dx [M, d] as in the forward. The two weight gradients are not in the kernel: dw_mode 0 ("token") writes
            h and dpre token-major [M, F] through TensorE transposes, so the caller's dW GEMMs are the kit's own
            expressions, dy2.t().mm(h) and dpre.t().mm(x2); dw_mode 1 ("fmajor") writes h^T and dpre^T [F, M] as
            they lie in SBUF (contiguous 2 KiB rows, no transposes) and the caller computes hT.mm(dy2).t() and
            dpreT.mm(x2). Which is faster is a chip measurement (queue_mlp2.txt runs both).

Rounding points (the torch emulation below has the same ones, so the simulator proof can be tight): r = bf16 of the
fp32 accumulator after relu (relu commutes with the rounding, so r is the kit's r up to fp32 summation order);
h = bf16(r * r) as the kit's square of the bf16 r (ScalarE's square is the exact fp32 product of the bf16 input,
rounded once); y and dx are bf16 of fp32 accumulators; dpre = bf16((dh_f32 * 2) * r) is rounded once where the kit
rounds dh to bf16 first (bf16(bf16(dh) * 2r)), a difference of at most one bf16 ulp of dpre. Nothing else differs
from the kit's arithmetic.

Shapes: M, d and F multiples of 128; tb a multiple of 128 up to 1024 (a block's tokens; the last block of a program
may be shorter). Memory per partition at tb 1024, d 1,536, F 6,144: forward about 182 KiB (x^T 24, h^T 96, the
streamed weight tiles 6 + 24 double-buffered, stages 32), backward about 186 KiB; PSUM at most five of the eight
2 KiB banks (two one-bank accumulators per output chunk, double-buffered, and one 1 KiB transpose tile). tb 512
halves the resident blocks (about 108 KiB) at the cost of streaming the weights four times per program instead of
twice and 80% instead of 89% TensorE streaming efficiency.

Knobs (read at import; the kernels also take them as arguments): R1_NKI_MLP_TB (1024), R1_NKI_MLP_DW (token |
fmajor), NEURON_LOGICAL_NC_CONFIG (the launch width: kernel[2] under LNC 2).

Written for the device-side NKI parser frontend, which interprets the kernel's source and accepts a narrower
Python than the CPU simulator (which executes the function): every construct here is one that a kernel of ours
has already compiled on the chip (candidate 33's GEMM and attention kernels, the S2b loss-head, fp8 and embedding
kernels). Every loop is `for name in range(...)` with one plain target; block bounds are integer arithmetic and
`if` on trace-time integers (program_id is a per-program constant: the fp8 kernel's identical `_span` compiled);
tuples are returned and unpacked only in assignments (`pid, nprg = _shard()`, the if-chain tuple of PSUM tiles);
no list, comprehension, `while`, `list(range(...))` or tuple-element loop target appears in a traced body; every
slice bound is a plain name or a literal (products and sums are computed into a variable first, the loss head's
`i0:i1` form); every PSUM tile is 2-D and at most one bank; a tile or view is subscripted once, never a view of a
view. An earlier form
(md5 4a329b55) iterated `for c0, c1 in chunks` over a list of pairs and was rejected on the chip with "expecting
simple variable"; the simulator had accepted it.
"""

# ---------------------------------------------------------------------------------------------------------------------
# Switch 9c (R2_MLP=fused8): nki_mlp_fused8.py is this module (c3209e4b, every line above kept) plus the same two
# kernels with their four GEMMs in fp8 at the TensorE double-row rate, added by exact insertions (work/p2/s2c_fp8/
# proofs/build_fused8.py; its strip gives c3209e4b byte for byte). The bf16 kernels below are untouched and still serve
# R2_MLP_FP8=fwd (bf16 backward) and =bwd (bf16 forward).
#
#   _mlp8_fwd_kernel(x2, w_fc, w_proj, sc, tb, forms) -> y [M, d], rT [F, M]
#       x^T cast to e4m3 at the constant scale 4 (x is the parameter-free RMSNorm output: abs(x_i) <= sqrt(d), so
#       4 abs(x_i) <= 157.4 at d 1,536); each streamed W_fc^T / W_proj^T tile cast to e4m3 at its tensor's power-of-two
#       scale; pre^T = sum over d/256 double-row matmuls (fp32 PSUM); r = bf16(max(acc x dq_fc, 0)) is the saved rT
#       exactly as the bf16 kernel saves it; rc = bf16(min(r x c, 15.375)) with c = sqrt(s_h) a power of two; h8 =
#       e4m3(rc^2) stays in SBUF (15.375^2 = 236.4 < 240: the cast cannot overflow even if the weight-derived bound
#       were wrong); y^T = sum over F/256 double-row matmuls with W_proj^T stationary, times dq_proj, transposed back.
#   _mlp8_bwd_kernel(dy2, rT, w_fc, w_proj, sc, tb, dw_mode, forms) -> dx, h, dpre (as the bf16 backward)
#       dy^T cast to e5m2 at its per-call scale; dh^T = double-row matmuls with the natural W_proj tiles (e4m3)
#       stationary; dpre = bf16((acc x dq_dh2) x r) on evacuation (dq_dh2 carries the activation gradient's 2), written
#       for the bf16 weight gradients exactly as the bf16 kernel writes it, then cast to e5m2 into a resident block;
#       dx^T = double-row matmuls with the natural W_fc tiles (e4m3) stationary, times dq_dx, transposed back.
#   Every scale is a power of two (h's a power of four), so every multiply and descale is exact and the torch
#   emulation below reproduces the kernels' fp8 bytes; the scales arrive as one fp32 tensor sc [128, NSC] per call (the
#   same value down each column), read by one DMA, used only as per-partition operands. forms 0 is the design's
#   instruction set; forms 1 replaces each instruction form our device has not compiled before with two device-proven
#   ones (bitwise the same results: the stages hold exact values).
# ---------------------------------------------------------------------------------------------------------------------

import os

import torch

_P = 128
_CH = 512
_env = os.environ.get
TB = int(_env("R1_NKI_MLP_TB", "1024"))
_DW_NAMES = {"token": 0, "fmajor": 1}
DW_MODE = _DW_NAMES[_env("R1_NKI_MLP_DW", "token")]
_LNC = int(_env("NEURON_LOGICAL_NC_CONFIG", "1"))
# Switch 9c (fp8): constants. The scale tensor's columns (sc [128, NSC] fp32 per call, one value per column).
_KS = 256                   # contraction per double-row matmul
TB8 = int(_env("R1_NKI_MLP8_TB", "1024"))
FORMS8 = int(_env("R1_NKI_MLP8_FORMS", "0"))
E4M3_MAX = 240.0            # the chip's legacy e4m3: largest finite value (the cast writes inf above it)
E5M2_MAX = 57344.0
X_SCALE = 4.0               # x's scale: abs(x_i) <= sqrt(d) (1 + 2^-8) for the RMSNorm output; x 4 <= 182 to d 2,048
H_CLAMP = 15.375            # r x c is clamped here before the square: exact in bf16, 15.375^2 = 236.39 < 240
H_BOUND_FACTOR = 1.3        # h <= 1.3 d rho_fc^2 (Cauchy-Schwarz, with the e4m3 rounding of x and W_fc and r's bf16)
DY_HEADROOM = 2.0           # dy's scale keeps amax(dy) x s_dy <= 57,344 / 2
DP_BOUND_FACTOR = 36.16     # dpre's bound B = 16 x 2 x 1.13 x d x rho_fc x kappa_proj x amax(dy)
E_W = (-24, 40)             # weight scales 2^e
E_H = (-16, 8)              # s_h = 4^e (c = sqrt(s_h) = 2^e)
E_G = (-40, 60)             # dy and dpre scales 2^e
SC_S_WFC, SC_DQ_FC, SC_C_H, SC_S_WPROJ, SC_DQ_PROJ = 0, 1, 2, 3, 4
SC_S_DY, SC_DQ_DH2, SC_S_DP, SC_DQ_DX = 5, 6, 7, 8
SC_RHO2, SC_KAPPA2, SC_I_WFC, SC_I_WPROJ, SC_S_H, SC_AMAX_WFC, SC_AMAX_WPROJ = 9, 10, 11, 12, 13, 14, 15
NSC = 16

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

    def _load_T_block(a2, K, NKB, row0, ntok, dt):
        """a^T block [P, NKB, ntok] of the token-major a2 [M, K] rows [row0, row0 + ntok) by the hardware-DGE strip
        transpose: one dma_transpose per 16 tokens over all NKB slabs (candidate 33's xt=hwt loader, verbatim)."""
        at = nl.ndarray((_P, NKB, ntok), dtype=dt, buffer=nl.sbuf)
        for s in range(ntok // 16):
            nisa.dma_transpose(dst=at.ap([[NKB * ntok, _P], [ntok, NKB], [1, 1], [1, 16]], offset=s * 16),
                               src=a2.ap([[K, 16], [_P, NKB], [1, 1], [1, _P]], offset=(row0 + s * 16) * K),
                               dge_mode=nisa.dge_mode.hwdge)
        return at

    def _load_wT_chunk(w, K, NKB, c, dt):
        """The transposed weight tiles of output chunk c: [P (k within slab), NKB, P (c's 128 rows of w)] from the
        natural w [R, K] (rows R = the chunk axis, columns K = the contraction): eight 16-row strip transposes."""
        wt = nl.ndarray((_P, NKB, _P), dtype=dt, buffer=nl.sbuf)
        for s in range(_P // 16):
            nisa.dma_transpose(dst=wt.ap([[NKB * _P, _P], [_P, NKB], [1, 1], [1, 16]], offset=s * 16),
                               src=w.ap([[K, 16], [_P, NKB], [1, 1], [1, _P]], offset=(c * _P + s * 16) * K),
                               dge_mode=nisa.dge_mode.hwdge)
        return wt

    def _load_w_chunk(w, N, NKB, c, dt):
        """The natural weight tiles of output chunk c: [P (k within slab), NKB, P] = w[k * 128 + p, c * 128 + i] for
        w [K, N] (rows K = the contraction, columns N = the chunk axis): one three-level DMA."""
        wt = nl.ndarray((_P, NKB, _P), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=w.ap([[N, _P], [_P * N, NKB], [1, _P]], offset=c * _P))
        return wt

    def _pchunks(ntok, f32):
        """The fp32 PSUM accumulators of one output chunk over a block of ntok tokens: one tile per 512-column chunk
        of the moving operand (one PSUM bank each: the GEMM kernel's `tiles` form), as a tuple of one or two."""
        w0 = ntok
        if w0 > _CH:
            w0 = _CH
        p0 = nl.ndarray((_P, w0), dtype=f32, buffer=nl.psum)
        if ntok <= _CH:
            return (p0,)
        p1 = nl.ndarray((_P, ntok - _CH), dtype=f32, buffer=nl.psum)
        return (p0, p1)

    def _accumulate(pss, wt, at, NKB, ntok):
        """pss[ci] = sum over the NKB slabs of wt[:, k, :]^T @ at[:, k, chunk ci] (fp32 PSUM), the k-slab loop
        outer so each stationary tile serves the block's chunks in turn."""
        w0 = ntok
        if w0 > _CH:
            w0 = _CH
        for k in range(NKB):
            nisa.nc_matmul(dst=pss[0], stationary=wt[:, k, :], moving=at[:, k, 0:w0], accumulate=(k > 0))
            if ntok > _CH:
                nisa.nc_matmul(dst=pss[1], stationary=wt[:, k, :], moving=at[:, k, _CH:ntok], accumulate=(k > 0))

    def _evac_relu(r_sb, hT, f, pss, ntok):
        """Forward phase 1 evacuation, per chunk: r = relu(acc) in bf16 on VectorE (the kit's own relu form,
        maximum with 0.0, read straight from PSUM) and h^T[f] = r^2 on ScalarE (the exact square, one rounding)."""
        nch = _ceil_div(ntok, _CH)
        for ci in range(nch):
            c0 = ci * _CH
            c1 = c0 + _CH
            if c1 > ntok:
                c1 = ntok
            nisa.tensor_scalar(dst=r_sb[:, c0:c1], data=pss[ci], op0=nl.maximum, operand0=0.0, engine=_VEC)
            nisa.activation(dst=hT[:, f, c0:c1], op=nl.square, data=r_sb[:, c0:c1])

    def _evac_copy(dst2, pss, ntok):
        """bf16 of the accumulator chunks into dst2 [P, ntok] on ScalarE (the GEMM kernel's evacuation)."""
        nch = _ceil_div(ntok, _CH)
        for ci in range(nch):
            c0 = ci * _CH
            c1 = c0 + _CH
            if c1 > ntok:
                c1 = ntok
            nisa.activation(dst=dst2[:, c0:c1], op=nl.copy, data=pss[ci])

    def _evac_fold(dpT, f, r_sb, pss, ntok):
        """Backward phase A evacuation, per chunk: dpre^T[f] = (dh * 2) * r in one VectorE instruction reading the
        PSUM accumulator and the r tile (bf16 out, one rounding)."""
        nch = _ceil_div(ntok, _CH)
        for ci in range(nch):
            c0 = ci * _CH
            c1 = c0 + _CH
            if c1 > ntok:
                c1 = ntok
            nisa.scalar_tensor_tensor(dst=dpT[:, f, c0:c1], data=pss[ci], op0=nl.multiply, operand0=2.0,
                                      op1=nl.multiply, operand1=r_sb[:, c0:c1])

    def _transpose_group(stage, j, src2, nt, dt):
        """stage[:, tt, j * 128 to (j + 1) * 128] = (src2[:, tt * 128 to (tt + 1) * 128])^T for the nt token tiles of
        the block, src2 a [P, ntok] tile: TensorE transposes in groups of up to four through a bf16 PSUM tile (a PSUM
        destination selects the TensorE), one VectorE copy out per tile (the attention kernel's form)."""
        j0 = j * _P
        j1 = j0 + _P
        for g0 in range(0, nt, 4):
            ng = nt - g0
            if ng > 4:
                ng = 4
            pst = nl.ndarray((_P, ng * _P), dtype=dt, buffer=nl.psum)
            for gg in range(ng):
                tt = g0 + gg
                q0 = gg * _P
                q1 = q0 + _P
                s0 = tt * _P
                s1 = s0 + _P
                nisa.nc_transpose(dst=pst[:, q0:q1], data=src2[:, s0:s1])
            for gg in range(ng):
                tt = g0 + gg
                q0 = gg * _P
                q1 = q0 + _P
                nisa.tensor_copy(dst=stage[:, tt, j0:j1], src=pst[:, q0:q1], engine=_VEC)

    def _transpose_group3(stage, j, src3, f, nt, dt):
        """_transpose_group for slab f of a [P, NF, ntok] tile (one subscript on the tile, no view of a view)."""
        j0 = j * _P
        j1 = j0 + _P
        for g0 in range(0, nt, 4):
            ng = nt - g0
            if ng > 4:
                ng = 4
            pst = nl.ndarray((_P, ng * _P), dtype=dt, buffer=nl.psum)
            for gg in range(ng):
                tt = g0 + gg
                q0 = gg * _P
                q1 = q0 + _P
                s0 = tt * _P
                s1 = s0 + _P
                nisa.nc_transpose(dst=pst[:, q0:q1], data=src3[:, f, s0:s1])
            for gg in range(ng):
                tt = g0 + gg
                q0 = gg * _P
                q1 = q0 + _P
                nisa.tensor_copy(dst=stage[:, tt, j0:j1], src=pst[:, q0:q1], engine=_VEC)

    def _store_rows(out, N, row0, nt, stage):
        """out[row0 + tt * 128 + p, :] = stage[p, tt, :] for the block's nt token tiles (one DMA)."""
        nisa.dma_copy(dst=out.ap([[N, _P], [_P * N, nt], [1, N]], offset=row0 * N), src=stage)

    def _store_cols(out, N, row0, nt, c, stage):
        """out[row0 + tt * 128 + p, c * 128:(c + 1) * 128] = stage[p, tt, :] (token-major columns of chunk c)."""
        nisa.dma_copy(dst=out.ap([[N, _P], [_P * N, nt], [1, _P]], offset=row0 * N + c * _P), src=stage)

    @nki.jit
    def _mlp_fwd_kernel(x2, w_fc, w_proj, tb):
        M = x2.shape[0]
        D = x2.shape[1]
        F = w_fc.shape[0]
        dt = x2.dtype
        f32 = nl.float32
        y = nl.ndarray((M, D), dtype=dt, buffer=nl.shared_hbm)
        rT = nl.ndarray((F, M), dtype=dt, buffer=nl.shared_hbm)
        ND = D // _P
        NF = F // _P
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
            ntok = nt * _P
            row0 = bt0 * _P
            row1 = row0 + ntok
            xt = _load_T_block(x2, D, ND, row0, ntok, dt)
            hT = nl.ndarray((_P, NF, ntok), dtype=dt, buffer=nl.sbuf)
            # Phase 1: pre^T chunks, r out, h^T resident.
            for f in range(NF):
                f0 = f * _P
                f1 = f0 + _P
                wt = _load_wT_chunk(w_fc, D, ND, f, dt)
                pss = _pchunks(ntok, f32)
                _accumulate(pss, wt, xt, ND, ntok)
                r_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                _evac_relu(r_sb, hT, f, pss, ntok)
                nisa.dma_copy(dst=rT[f0:f1, row0:row1], src=r_sb)
            # Phase 2: y^T chunks, transposed to token-major, one store per block.
            ystage = nl.ndarray((_P, nt, D), dtype=dt, buffer=nl.sbuf)
            for j in range(ND):
                wpt = _load_wT_chunk(w_proj, F, NF, j, dt)
                pss = _pchunks(ntok, f32)
                _accumulate(pss, wpt, hT, NF, ntok)
                yT_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                _evac_copy(yT_sb, pss, ntok)
                _transpose_group(ystage, j, yT_sb, nt, dt)
            _store_rows(y, D, row0, nt, ystage)
        return y, rT

    @nki.jit
    def _mlp_bwd_kernel(dy2, rT, w_fc, w_proj, tb, dw_mode):
        M = dy2.shape[0]
        D = dy2.shape[1]
        F = rT.shape[0]
        dt = dy2.dtype
        f32 = nl.float32
        dx = nl.ndarray((M, D), dtype=dt, buffer=nl.shared_hbm)
        if dw_mode == 1:
            h_out = nl.ndarray((F, M), dtype=dt, buffer=nl.shared_hbm)
            dp_out = nl.ndarray((F, M), dtype=dt, buffer=nl.shared_hbm)
        else:
            h_out = nl.ndarray((M, F), dtype=dt, buffer=nl.shared_hbm)
            dp_out = nl.ndarray((M, F), dtype=dt, buffer=nl.shared_hbm)
        ND = D // _P
        NF = F // _P
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
            ntok = nt * _P
            row0 = bt0 * _P
            row1 = row0 + ntok
            dyt = _load_T_block(dy2, D, ND, row0, ntok, dt)
            dpT = nl.ndarray((_P, NF, ntok), dtype=dt, buffer=nl.sbuf)
            # Phase A: dh^T chunks with the activation gradient on evacuation; h^T rebuilt from r.
            for f in range(NF):
                f0 = f * _P
                f1 = f0 + _P
                wp = _load_w_chunk(w_proj, F, ND, f, dt)
                pss = _pchunks(ntok, f32)
                _accumulate(pss, wp, dyt, ND, ntok)
                r_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                nisa.dma_copy(dst=r_sb, src=rT[f0:f1, row0:row1])
                _evac_fold(dpT, f, r_sb, pss, ntok)
                h_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                nisa.activation(dst=h_sb, op=nl.square, data=r_sb)
                if dw_mode == 1:
                    nisa.dma_copy(dst=h_out[f0:f1, row0:row1], src=h_sb)
                    nisa.dma_copy(dst=dp_out[f0:f1, row0:row1], src=dpT[:, f, :])
                else:
                    hst = nl.ndarray((_P, nt, _P), dtype=dt, buffer=nl.sbuf)
                    _transpose_group(hst, 0, h_sb, nt, dt)
                    _store_cols(h_out, F, row0, nt, f, hst)
                    dst_ = nl.ndarray((_P, nt, _P), dtype=dt, buffer=nl.sbuf)
                    _transpose_group3(dst_, 0, dpT, f, nt, dt)
                    _store_cols(dp_out, F, row0, nt, f, dst_)
            # Phase B: dx^T chunks, transposed to token-major, one store per block.
            dxstage = nl.ndarray((_P, nt, D), dtype=dt, buffer=nl.sbuf)
            for j in range(ND):
                wf = _load_w_chunk(w_fc, D, NF, j, dt)
                pss = _pchunks(ntok, f32)
                _accumulate(pss, wf, dpT, NF, ntok)
                dxT_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                _evac_copy(dxT_sb, pss, ntok)
                _transpose_group(dxstage, j, dxT_sb, nt, dt)
            _store_rows(dx, D, row0, nt, dxstage)
        return dx, h_out, dp_out

    # ---- switch 9c: the fp8 kernels (every construct below is a form our device has compiled or the documented ISA
    # form the forms probe compiles first; loops have one plain name over range(); slice bounds are plain names)

    def _load_sc(sc):
        """The call's scale tile [P, NSC] fp32 by one 2-D DMA (each column one value down every partition)."""
        s = nl.ndarray((_P, NSC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s, src=sc)
        return s

    def _cast8(dst8, src, sc_sb, col, forms):
        """dst8 = fp8(src x sc[:, col]) for a whole tile, the fp32 product rounded once into dst8's format. forms 0:
        one ScalarE activation with the per-partition vector scale. forms 1: the exact product into src in place on
        VectorE (the vector operand into bf16, as the attention kernel's normalise), then a ScalarE copy into fp8 at
        the constant scale 1.0 (S2b's cast). The product is a power-of-two multiple of a bf16 value: exact in bf16."""
        q0 = col
        q1 = q0 + 1
        if forms == 1:
            nisa.tensor_scalar(dst=src, data=src, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)
            nisa.activation(dst=dst8, op=nl.copy, data=src, scale=1.0)
        else:
            nisa.activation(dst=dst8, op=nl.copy, data=src, scale=sc_sb[:, q0:q1])

    def _accumulate8(pss, wt8, at8, NKD, ntok):
        """pss[ci] = sum over the NKD double-row pairs of wt8[:, 2kd:2kd+2, :]^T @ at8[:, 2kd:2kd+2, chunk ci] (fp32
        PSUM): _accumulate with two 128-row slabs per instruction (stationary [P, 2, 128], moving [P, 2, <= 512])."""
        w0 = ntok
        if w0 > _CH:
            w0 = _CH
        for kd in range(NKD):
            k0 = kd * 2
            k1 = k0 + 2
            nisa.nc_matmul(dst=pss[0], stationary=wt8[:, k0:k1, :], moving=at8[:, k0:k1, 0:w0], accumulate=(kd > 0),
                           perf_mode=_DR)
            if ntok > _CH:
                nisa.nc_matmul(dst=pss[1], stationary=wt8[:, k0:k1, :], moving=at8[:, k0:k1, _CH:ntok],
                               accumulate=(kd > 0), perf_mode=_DR)

    def _evac_relu8(r_sb, rc, hs, h8, f, pss, ntok, sc_sb, forms):
        """Forward phase 1 evacuation, per chunk: r = bf16(max(acc x dq_fc, 0)) (VectorE; r is the saved rT), rc =
        bf16(min(r x c, 15.375)) (VectorE; exact), h8[f] = e4m3(rc^2) (ScalarE square, the exact fp32 square rounded
        once). forms 1: each two-operator instruction as two one-operator ones, and the square through an fp32 stage
        hs, then a copy into e4m3 (the same values)."""
        a0 = SC_DQ_FC
        a1 = a0 + 1
        b0 = SC_C_H
        b1 = b0 + 1
        nch = _ceil_div(ntok, _CH)
        for ci in range(nch):
            c0 = ci * _CH
            c1 = c0 + _CH
            if c1 > ntok:
                c1 = ntok
            if forms == 1:
                nisa.tensor_scalar(dst=r_sb[:, c0:c1], data=pss[ci], op0=nl.multiply, operand0=sc_sb[:, a0:a1],
                                   engine=_VEC)
                nisa.tensor_scalar(dst=r_sb[:, c0:c1], data=r_sb[:, c0:c1], op0=nl.maximum, operand0=0.0, engine=_VEC)
                nisa.tensor_scalar(dst=rc[:, c0:c1], data=r_sb[:, c0:c1], op0=nl.multiply, operand0=sc_sb[:, b0:b1],
                                   engine=_VEC)
                nisa.tensor_scalar(dst=rc[:, c0:c1], data=rc[:, c0:c1], op0=nl.minimum, operand0=H_CLAMP,
                                   engine=_VEC)
                nisa.activation(dst=hs[:, c0:c1], op=nl.square, data=rc[:, c0:c1])
                nisa.activation(dst=h8[:, f, c0:c1], op=nl.copy, data=hs[:, c0:c1], scale=1.0)
            else:
                nisa.tensor_scalar(dst=r_sb[:, c0:c1], data=pss[ci], op0=nl.multiply, operand0=sc_sb[:, a0:a1],
                                   op1=nl.maximum, operand1=0.0, engine=_VEC)
                nisa.tensor_scalar(dst=rc[:, c0:c1], data=r_sb[:, c0:c1], op0=nl.multiply, operand0=sc_sb[:, b0:b1],
                                   op1=nl.minimum, operand1=H_CLAMP, engine=_VEC)
                nisa.activation(dst=h8[:, f, c0:c1], op=nl.square, data=rc[:, c0:c1])

    def _evac_desc(dst2, pss, ntok, sc_sb, col):
        """bf16 of the accumulator chunks times the per-partition descale sc[:, col] into dst2 [P, ntok] (VectorE: the
        attention kernel's normalise, a vector operand0 reading PSUM into bf16)."""
        q0 = col
        q1 = q0 + 1
        nch = _ceil_div(ntok, _CH)
        for ci in range(nch):
            c0 = ci * _CH
            c1 = c0 + _CH
            if c1 > ntok:
                c1 = ntok
            nisa.tensor_scalar(dst=dst2[:, c0:c1], data=pss[ci], op0=nl.multiply, operand0=sc_sb[:, q0:q1],
                               engine=_VEC)

    def _evac_fold8(dp_sb, r_sb, pss, ntok, sc_sb):
        """Backward phase A evacuation, per chunk: dpre = bf16((acc x dq_dh2) x r), one VectorE instruction (the
        attention's dS form: PSUM data, a vector operand0, a tile operand1; dq_dh2 carries the 2)."""
        q0 = SC_DQ_DH2
        q1 = q0 + 1
        nch = _ceil_div(ntok, _CH)
        for ci in range(nch):
            c0 = ci * _CH
            c1 = c0 + _CH
            if c1 > ntok:
                c1 = ntok
            nisa.scalar_tensor_tensor(dst=dp_sb[:, c0:c1], data=pss[ci], op0=nl.multiply, operand0=sc_sb[:, q0:q1],
                                      op1=nl.multiply, operand1=r_sb[:, c0:c1])

    def _cast_dp8(dp8, f, dp_sb, dps, sc_sb, forms):
        """dp8[f] = e5m2(dpre x s_dp). forms 0: one VectorE instruction (a vector operand0 into fp8: the SDK quantiser's
        form). forms 1: the exact product into the bf16 stage dps, then a ScalarE copy into e5m2 at scale 1.0."""
        q0 = SC_S_DP
        q1 = q0 + 1
        if forms == 1:
            nisa.tensor_scalar(dst=dps, data=dp_sb, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)
            nisa.activation(dst=dp8[:, f, :], op=nl.copy, data=dps, scale=1.0)
        else:
            nisa.tensor_scalar(dst=dp8[:, f, :], data=dp_sb, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)

    @nki.jit
    def _mlp8_fwd_kernel(x2, w_fc, w_proj, sc, tb, forms):
        M = x2.shape[0]
        D = x2.shape[1]
        F = w_fc.shape[0]
        dt = x2.dtype
        f32 = nl.float32
        y = nl.ndarray((M, D), dtype=dt, buffer=nl.shared_hbm)
        rT = nl.ndarray((F, M), dtype=dt, buffer=nl.shared_hbm)
        ND = D // _P
        NF = F // _P
        NDD = D // _KS
        NFD = F // _KS
        sc_sb = _load_sc(sc)
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
            ntok = nt * _P
            row0 = bt0 * _P
            row1 = row0 + ntok
            xt = _load_T_block(x2, D, ND, row0, ntok, dt)
            x8 = nl.ndarray((_P, ND, ntok), dtype=_F8E4, buffer=nl.sbuf)
            nisa.activation(dst=x8, op=nl.copy, data=xt, scale=X_SCALE)
            h8 = nl.ndarray((_P, NF, ntok), dtype=_F8E4, buffer=nl.sbuf)
            # Phase 1: pre^T chunks in fp8, r out (bf16), h^T resident in e4m3.
            for f in range(NF):
                f0 = f * _P
                f1 = f0 + _P
                wt = _load_wT_chunk(w_fc, D, ND, f, dt)
                w8 = nl.ndarray((_P, ND, _P), dtype=_F8E4, buffer=nl.sbuf)
                _cast8(w8, wt, sc_sb, SC_S_WFC, forms)
                pss = _pchunks(ntok, f32)
                _accumulate8(pss, w8, x8, NDD, ntok)
                r_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                rc = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                if forms == 1:
                    hs = nl.ndarray((_P, ntok), dtype=f32, buffer=nl.sbuf)
                else:
                    hs = rc
                _evac_relu8(r_sb, rc, hs, h8, f, pss, ntok, sc_sb, forms)
                nisa.dma_copy(dst=rT[f0:f1, row0:row1], src=r_sb)
            # Phase 2: y^T chunks in fp8, descaled, transposed to token-major, one store per block.
            ystage = nl.ndarray((_P, nt, D), dtype=dt, buffer=nl.sbuf)
            for j in range(ND):
                wpt = _load_wT_chunk(w_proj, F, NF, j, dt)
                wp8 = nl.ndarray((_P, NF, _P), dtype=_F8E4, buffer=nl.sbuf)
                _cast8(wp8, wpt, sc_sb, SC_S_WPROJ, forms)
                pss = _pchunks(ntok, f32)
                _accumulate8(pss, wp8, h8, NFD, ntok)
                yT_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                _evac_desc(yT_sb, pss, ntok, sc_sb, SC_DQ_PROJ)
                _transpose_group(ystage, j, yT_sb, nt, dt)
            _store_rows(y, D, row0, nt, ystage)
        return y, rT

    @nki.jit
    def _mlp8_bwd_kernel(dy2, rT, w_fc, w_proj, sc, tb, dw_mode, forms):
        M = dy2.shape[0]
        D = dy2.shape[1]
        F = rT.shape[0]
        dt = dy2.dtype
        f32 = nl.float32
        dx = nl.ndarray((M, D), dtype=dt, buffer=nl.shared_hbm)
        if dw_mode == 1:
            h_out = nl.ndarray((F, M), dtype=dt, buffer=nl.shared_hbm)
            dp_out = nl.ndarray((F, M), dtype=dt, buffer=nl.shared_hbm)
        else:
            h_out = nl.ndarray((M, F), dtype=dt, buffer=nl.shared_hbm)
            dp_out = nl.ndarray((M, F), dtype=dt, buffer=nl.shared_hbm)
        ND = D // _P
        NF = F // _P
        NDD = D // _KS
        NFD = F // _KS
        sc_sb = _load_sc(sc)
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
            ntok = nt * _P
            row0 = bt0 * _P
            row1 = row0 + ntok
            dyt = _load_T_block(dy2, D, ND, row0, ntok, dt)
            dy8 = nl.ndarray((_P, ND, ntok), dtype=_F8E5, buffer=nl.sbuf)
            _cast8(dy8, dyt, sc_sb, SC_S_DY, forms)
            dp8 = nl.ndarray((_P, NF, ntok), dtype=_F8E5, buffer=nl.sbuf)
            # Phase A: dh^T chunks in fp8; dpre on evacuation (bf16, written for the weight gradients), then e5m2.
            for f in range(NF):
                f0 = f * _P
                f1 = f0 + _P
                wp = _load_w_chunk(w_proj, F, ND, f, dt)
                wp8 = nl.ndarray((_P, ND, _P), dtype=_F8E4, buffer=nl.sbuf)
                _cast8(wp8, wp, sc_sb, SC_S_WPROJ, forms)
                pss = _pchunks(ntok, f32)
                _accumulate8(pss, wp8, dy8, NDD, ntok)
                r_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                nisa.dma_copy(dst=r_sb, src=rT[f0:f1, row0:row1])
                dp_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                _evac_fold8(dp_sb, r_sb, pss, ntok, sc_sb)
                h_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                nisa.activation(dst=h_sb, op=nl.square, data=r_sb)
                if dw_mode == 1:
                    nisa.dma_copy(dst=h_out[f0:f1, row0:row1], src=h_sb)
                    nisa.dma_copy(dst=dp_out[f0:f1, row0:row1], src=dp_sb)
                else:
                    hst = nl.ndarray((_P, nt, _P), dtype=dt, buffer=nl.sbuf)
                    _transpose_group(hst, 0, h_sb, nt, dt)
                    _store_cols(h_out, F, row0, nt, f, hst)
                    dst_ = nl.ndarray((_P, nt, _P), dtype=dt, buffer=nl.sbuf)
                    _transpose_group(dst_, 0, dp_sb, nt, dt)
                    _store_cols(dp_out, F, row0, nt, f, dst_)
                if forms == 1:
                    dps = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                else:
                    dps = dp_sb
                _cast_dp8(dp8, f, dp_sb, dps, sc_sb, forms)
            # Phase B: dx^T chunks in fp8, descaled, transposed to token-major, one store per block.
            dxstage = nl.ndarray((_P, nt, D), dtype=dt, buffer=nl.sbuf)
            for j in range(ND):
                wf = _load_w_chunk(w_fc, D, NF, j, dt)
                wf8 = nl.ndarray((_P, NF, _P), dtype=_F8E4, buffer=nl.sbuf)
                _cast8(wf8, wf, sc_sb, SC_S_WFC, forms)
                pss = _pchunks(ntok, f32)
                _accumulate8(pss, wf8, dp8, NFD, ntok)
                dxT_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                _evac_desc(dxT_sb, pss, ntok, sc_sb, SC_DQ_DX)
                _transpose_group(dxstage, j, dxT_sb, nt, dt)
            _store_rows(dx, D, row0, nt, dxstage)
        return dx, h_out, dp_out

    _HAS_NKI = True
except Exception as exc:  # nki unavailable (CPU dev box): the emulation below still works
    _NKI_ERROR = f"{type(exc).__name__}: {exc}"


def _launch(kernel):
    """kernel[2] under an LNC=2 runtime (both physical cores run half the token tiles), plain call otherwise."""
    return kernel[_LNC] if _LNC == 2 else kernel


def shapes_apply(m: int, d: int, f: int, tb: int = TB) -> bool:
    """The shapes the kernels take: multiples of 128 everywhere, tb a multiple of 128 up to 1024, at least one
    token tile per program under LNC 2 is not required (a program with no tiles does nothing)."""
    return (m % _P == 0 and d % _P == 0 and f % _P == 0 and m > 0 and tb % _P == 0 and 0 < tb <= 1024)


def shapes_apply8(m: int, d: int, f: int, tb: int = TB8) -> bool:
    """Switch 9c: the shapes the fp8 kernels take: M a multiple of 128, d and F multiples of 256 (the double-row pair),
    tb a multiple of 128 up to 1024."""
    return (m % _P == 0 and d % _KS == 0 and f % _KS == 0 and m > 0 and tb % _P == 0 and 0 < tb <= 1024)


def plan_kib8(d: int, f: int, tb: int = TB8, forms: int = FORMS8) -> tuple:
    """Switch 9c: the SBUF per partition (KiB) the fp8 forward and backward hold at a block of tb tokens, every tile
    the kernel allocates in a block counted, the streamed weight and stage tiles twice (double-buffered): (fwd, bwd)."""
    nd, nf, nt = d // _P, f // _P, tb // _P
    wts = 2 * (nd * _P * 3 + nf * _P * 3)                     # a bf16 tile and its fp8 copy, both weights
    fwd = (nd * tb * 3 + nf * tb + wts + 2 * tb * (4 + (4 if forms == 1 else 0)) + 2 * tb + nt * d * 2)
    bwd = (nd * tb * 3 + nf * tb + wts + 2 * tb * (6 + (2 if forms == 1 else 0)) + 4 * nt * _P + 2 * tb + nt * d * 2)
    return fwd / 1024.0, bwd / 1024.0


# ----------------------------------------------------------------------------- torch emulation (CPU and proofs)

def emulate_forward(x2: torch.Tensor, w_fc: torch.Tensor, w_proj: torch.Tensor):
    """The kernels' arithmetic in torch: fp32 accumulation (torch's summation order), the same rounding points.
    Returns y [M, d] bf16 and rT [F, M] bf16."""
    pre = x2.float() @ w_fc.float().t()
    r = torch.relu(pre).to(x2.dtype)
    h = (r.float() * r.float()).to(x2.dtype)
    y = (h.float() @ w_proj.float().t()).to(x2.dtype)
    return y, r.t().contiguous()


def emulate_backward(dy2: torch.Tensor, rT: torch.Tensor, w_fc: torch.Tensor, w_proj: torch.Tensor, dw_mode: int):
    """dx [M, d], and h, dpre token-major [M, F] (dw_mode 0) or h^T, dpre^T [F, M] (dw_mode 1), bf16."""
    r = rT.t()
    dh = dy2.float() @ w_proj.float()
    dpre = ((dh * 2.0) * r.float()).to(dy2.dtype)
    h = (r.float() * r.float()).to(dy2.dtype)
    dx = (dpre.float() @ w_fc.float()).to(dy2.dtype)
    if dw_mode == 1:
        return dx, h.t().contiguous(), dpre.t().contiguous()
    return dx, h.contiguous(), dpre.contiguous()


def weight_grads(dy2: torch.Tensor, x2: torch.Tensor, h_out: torch.Tensor, dp_out: torch.Tensor, dw_mode: int):
    """The two weight gradients from the backward kernel's outputs: the kit's expressions on token-major h and dpre
    (dw_mode 0), or their F-major forms (dw_mode 1). dw_fc [F, d], dw_proj [d, F], in the activations' dtype."""
    if dw_mode == 1:
        return dp_out.mm(x2), h_out.mm(dy2).t()
    return dp_out.t().mm(x2), dy2.t().mm(h_out)


def reference_forward(x2: torch.Tensor, w_fc: torch.Tensor, w_proj: torch.Tensor) -> torch.Tensor:
    """The kit's expression, c_proj(relu(c_fc(x))^2), in the activations' dtype (bf16 matmuls on CPU round as the
    chip's compiled matmuls do: fp32 accumulation, bf16 out)."""
    return torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(x2, w_fc)).square(), w_proj)


class EmulatedFusedMlp(torch.autograd.Function):
    """The emulation as an autograd function: the forward saves rT only; the backward is the kernel's plus the two
    weight gradients. Used where the kernels cannot run (CPU proxies and proofs)."""

    @staticmethod
    def forward(ctx, x2, w_fc, w_proj, dw_mode):
        y, rT = emulate_forward(x2, w_fc, w_proj)
        ctx.save_for_backward(x2, rT, w_fc, w_proj)
        ctx.dw_mode = dw_mode
        return y

    @staticmethod
    def backward(ctx, dy2):
        x2, rT, w_fc, w_proj = ctx.saved_tensors
        dy2 = dy2.contiguous()
        dx, h_out, dp_out = emulate_backward(dy2, rT, w_fc, w_proj, ctx.dw_mode)
        dw_fc, dw_proj = weight_grads(dy2, x2, h_out, dp_out, ctx.dw_mode)
        return dx, dw_fc, dw_proj, None


# ----------------------------------------------------------------------------- switch 9c: the scales and the emulation

def _pow2_le(v: torch.Tensor, limit: float, e_lo: int, e_hi: int):
    """(s, 1/s): s = 2^e for the largest integer e in [e_lo, e_hi] with v x s <= limit (v >= 0, fp32, any shape; v = 0
    gives 2^e_hi). A greedy search over the bits of e - e_lo with comparisons, selects and multiplications by powers of
    two only (S2b's fp8_proj_prescale form: no reciprocal, no transcendental op for a device to approximate), so s and
    1/s are exact powers of two and s x (1/s) == 1."""
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


def _pow4_le(v: torch.Tensor, limit: float, e_lo: int, e_hi: int):
    """(s, 1/s, c, 1/c): s = 4^e for the largest integer e in [e_lo, e_hi] with v x s <= limit, c = 2^e = sqrt(s); the
    same greedy search as _pow2_le, every value an exact power of two."""
    v = v.float()
    s = torch.full_like(v, 4.0 ** e_lo)
    inv = torch.full_like(v, 4.0 ** -e_lo)
    c = torch.full_like(v, 2.0 ** e_lo)
    ic = torch.full_like(v, 2.0 ** -e_lo)
    span = e_hi - e_lo
    for b in (32, 16, 8, 4, 2, 1):
        if b > span:
            continue
        cand = s * (4.0 ** b)
        ok = (v * cand <= limit) & (cand <= 4.0 ** e_hi)
        s = torch.where(ok, cand, s)
        inv = torch.where(ok, inv * (4.0 ** -b), inv)
        c = torch.where(ok, c * (2.0 ** b), c)
        ic = torch.where(ok, ic * (2.0 ** -b), ic)
    return s, inv, c, ic


def forward_scales(w_fc: torch.Tensor, w_proj: torch.Tensor) -> torch.Tensor:
    """One layer's scale row [NSC] fp32 from its two weights as the kernels read them (bf16): the e4m3 multipliers of
    W_fc and W_proj from their exact amax (amax x s_w <= 240), s_h = 4^e from W_fc's largest row norm (1.3 d rho_fc^2
    x s_h <= 240, the Cauchy-Schwarz bound on h with x at most sqrt(d) in norm), the forward's descales, and the
    statistics the backward's dpre bound reads. A pure function of the weights: training calls it after every
    optimizer step and load_for_eval once after loading, so the scored forward is the trained one; nothing is saved.
    The backward's four columns are zero here (backward_scales fills them per call)."""
    wf = w_fc.detach().to(torch.bfloat16).float()
    wp = w_proj.detach().to(torch.bfloat16).float()
    amax_fc = wf.abs().amax()
    amax_pj = wp.abs().amax()
    s_wfc, i_wfc = _pow2_le(amax_fc, E4M3_MAX, E_W[0], E_W[1])
    s_wpj, i_wpj = _pow2_le(amax_pj, E4M3_MAX, E_W[0], E_W[1])
    rho2 = (wf * wf).sum(dim=1).amax()
    kappa2 = (wp * wp).sum(dim=0).amax()
    s_h, i_h, c_h, _ic = _pow4_le(rho2 * (H_BOUND_FACTOR * w_fc.shape[1]), E4M3_MAX, E_H[0], E_H[1])
    zero = torch.zeros_like(amax_fc)
    return torch.stack([s_wfc, i_wfc * (1.0 / X_SCALE), c_h, s_wpj, i_h * i_wpj, zero, zero, zero, zero, rho2, kappa2,
                        i_wfc, i_wpj, s_h, amax_fc, amax_pj])


def backward_scales(dy2: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
    """The backward call's row [NSC]: the forward row with S_DY, DQ_DH2, S_DP, DQ_DX filled from this call's dy: s_dy
    from its exact amax (amax x s_dy <= 57,344 / 2), s_dp from the bound on dpre = 2 r dh (r <= 1.13 sqrt(d) rho_fc,
    abs(dh) <= sqrt(d) amax(dy) kappa_proj, x 16 for the fp8 roundings and headroom), so neither e5m2 cast can reach
    57,344. Training only: the scored forward never reads these columns."""
    amax_dy = dy2.detach().abs().amax().float()
    s_dy, i_dy = _pow2_le(amax_dy, E5M2_MAX / DY_HEADROOM, E_G[0], E_G[1])
    bound = torch.sqrt(row[SC_RHO2] * row[SC_KAPPA2]) * amax_dy * (DP_BOUND_FACTOR * dy2.shape[1])
    s_dp, i_dp = _pow2_le(bound, E5M2_MAX, E_G[0], E_G[1])
    mid = torch.stack([s_dy, i_dy * row[SC_I_WPROJ] * 2.0, s_dp, i_dp * row[SC_I_WFC]])
    return torch.cat([row[:SC_S_DY], mid, row[SC_RHO2:]])


def sc_tile(row: torch.Tensor) -> torch.Tensor:
    """[128, NSC] fp32: the row down every partition (the kernels' scale input, read by one 2-D DMA)."""
    return row.float().reshape(1, NSC).expand(_P, NSC).contiguous()


def q8(t: torch.Tensor, fmt: int) -> torch.Tensor:
    """The fp8 rounding (round to nearest even) of t's fp32 values, back in fp32: e4m3 (fmt 0: torch's float8_e4m3fn,
    whose values and rounding inside +-240 are the chip's legacy e4m3's, the only range the kernels cast) or e5m2
    (fmt 1)."""
    return t.float().to(torch.float8_e4m3fn if fmt == 0 else torch.float8_e5m2).float()


def emulate8_forward(x2: torch.Tensor, w_fc: torch.Tensor, w_proj: torch.Tensor, row: torch.Tensor):
    """_mlp8_fwd_kernel's arithmetic in torch: the same casts and rounding points, fp32 products and accumulation (in
    torch's summation order). row [NSC] fp32 (forward_scales'). Returns y [M, d] and rT [F, M] in x2's dtype."""
    dt = x2.dtype
    x8 = q8(x2.float() * X_SCALE, 0)
    w8 = q8(w_fc.float() * row[SC_S_WFC], 0)
    r = torch.clamp_min((x8 @ w8.t()) * row[SC_DQ_FC], 0.0).to(dt)
    rc = torch.clamp_max(r.float() * row[SC_C_H], H_CLAMP).to(dt)
    h8 = q8(rc.float() * rc.float(), 0)
    wp8 = q8(w_proj.float() * row[SC_S_WPROJ], 0)
    y = ((h8 @ wp8.t()) * row[SC_DQ_PROJ]).to(dt)
    return y, r.t().contiguous()


def emulate8_backward(dy2: torch.Tensor, rT: torch.Tensor, w_fc: torch.Tensor, w_proj: torch.Tensor, row: torch.Tensor,
                      dw_mode: int):
    """_mlp8_bwd_kernel's arithmetic in torch. row [NSC] fp32 (backward_scales'). Returns dx [M, d], and h, dpre
    token-major [M, F] (dw_mode 0) or F-major [F, M] (dw_mode 1), in dy2's dtype: h and dpre are the bf16 values the
    weight gradients read, as in the bf16 backward."""
    dt = dy2.dtype
    dy8 = q8(dy2.float() * row[SC_S_DY], 1)
    wp8 = q8(w_proj.float() * row[SC_S_WPROJ], 0)
    r = rT.t().float()
    dpre = (((dy8 @ wp8) * row[SC_DQ_DH2]) * r).to(dt)
    h = (r * r).to(dt)
    dp8 = q8(dpre.float() * row[SC_S_DP], 1)
    wf8 = q8(w_fc.float() * row[SC_S_WFC], 0)
    dx = ((dp8 @ wf8) * row[SC_DQ_DX]).to(dt)
    if dw_mode == 1:
        return dx, h.t().contiguous(), dpre.t().contiguous()
    return dx, h.contiguous(), dpre.contiguous()


class EmulatedFusedMlp8(torch.autograd.Function):
    """Switch 9c where the kernels cannot run (CPU proxies and proofs): forward(x2, w_fc, w_proj, sc, mode, dw_mode),
    sc the layer's [128, NSC] scale tile (forward_scales' row), mode 1 (fp8 forward, bf16 backward), 2 (bf16 forward,
    fp8 backward) or 3 (both): the kernels' arithmetic, the backward's scales computed from dy as the device's backward
    computes them; the weight gradients are the kit's expressions on the backward's bf16 h and dpre."""

    @staticmethod
    def forward(ctx, x2, w_fc, w_proj, sc, mode, dw_mode):
        row = sc[0]
        if mode & 1:
            y, rT = emulate8_forward(x2, w_fc, w_proj, row)
        else:
            y, rT = emulate_forward(x2, w_fc, w_proj)
        ctx.save_for_backward(x2, rT, w_fc, w_proj, sc)
        ctx.mode = mode
        ctx.dw_mode = dw_mode
        return y

    @staticmethod
    def backward(ctx, dy2):
        x2, rT, w_fc, w_proj, sc = ctx.saved_tensors
        dy2 = dy2.contiguous()
        if ctx.mode & 2:
            dx, h_out, dp_out = emulate8_backward(dy2, rT, w_fc, w_proj, backward_scales(dy2, sc[0]), ctx.dw_mode)
        else:
            dx, h_out, dp_out = emulate_backward(dy2, rT, w_fc, w_proj, ctx.dw_mode)
        dw_fc, dw_proj = weight_grads(dy2, x2, h_out, dp_out, ctx.dw_mode)
        return dx, dw_fc, dw_proj, None, None, None


# ---------------------------------------------------------------------------------------------------------------------
# Switch 48 on the fp8 line (lane STEM8, work/p2/stem8/NOTES.md section 2): nki_mlp_stem8.py is nki_mlp_fused8.py
# (18369d38, every line above kept byte for byte; tools/build_kernel8.py appends this section) plus switch 9c's two fp8
# kernels with a token-indexed gate g = U[t] on the c_proj input: y = c_proj(relu(c_fc x)^2 * g).
#
# The scales. switch 9c casts h = r^2 to e4m3 at s_h = 4^e, from the weight bound h <= 1.3 d rho_fc^2, and clamps r x c
# (c = sqrt(s_h)) at 15.375 before the square, so the cast can never reach 240. The gated c_proj input hg = h x g is
# bounded by |hg| <= 1.3 d rho_fc^2 x G with G = max |U| over the layer's table (as the forward reads it, bf16). Let
# gamma = 2^e be the smallest power of two with G <= gamma (s_g = 1 / gamma, both exact: _pow2_le(G, 1.0, ...)). The
# kernel receives the gate pre-scaled, g' = g x s_g (exact; |g'| <= 1), and casts e4m3(rc^2 x g'): rc^2 <= 236.39 and
# |g'| <= 1, so the cast still cannot reach 240, whatever U holds. The effective scale of hg in e4m3 is s_h x s_g, so
# the c_proj descale is dq_proj x gamma. In the backward the gradient of the c_proj input is dhg; dpre = 2 r dh =
# 2 r dhg g = (acc x dq_dh2 x gamma) x bf16(r x g'), its bound grows by G <= gamma, so the e5m2 scale of dpre is
# s_dp x s_g and the dx descale dq_dx x gamma; the gate's gradient (with respect to g', the kernel's input) is
# dg' = (acc x dq_dh2 x gamma / 2) x h; hg (for dW_proj, bf16) = bf16(h x g') x gamma = bf16(h x g). gamma and s_g are
# a pure function of the table: training writes them after every optimizer step (with switch 9c's scale refresh, so
# the refresh must see U) and load_for_eval once after loading, so the scored forward is the trained one; nothing is
# saved. With U at ones (the table's initialisation) G = 1, gamma = s_g = 1 and both kernels are switch 9c's bit for
# bit (the gate multiplies an exact fp32 square by 1 before the same single e4m3 rounding; r x 1 = r; h x 1 = h).
#
#   _mlp8g_fwd_kernel(x2, w_fc, w_proj, sc, gT, tb, forms) -> y [M, d], rT [F, M]
#       switch 9c's forward; per F slab the gate slab g'^T [128, ntok] (bf16, one DMA as the backward's rT slab), and
#       the evacuation r, rc as 9c's, then hs = rc^2 into an fp32 stage (forms 1's square), hs = hs x g' (VectorE
#       tensor_tensor in place: rc^2 has at most 16 significant bits and g' 8, so the fp32 product is exact), and
#       h8[f] = e4m3(hs) (forms 1's copy at scale 1.0: one rounding, the value 9c's square-into-e4m3 rounds).
#   _mlp8g_bwd_kernel(dy2, rT, gT, w_fc, w_proj, sc, tb, forms) -> dx [M, d], hg^T, dpre^T, dg'^T [F, M] (F-major)
#       switch 9c's backward (dw_mode 1 only); per F slab, h = r^2 (bf16) first, then per chunk dg' = bf16((acc x
#       dq_dg) x h) and dpre = bf16((acc x dq_dh2') x rg') with rg' = bf16(r x g') in place (9c's fold form, twice), then
#       hg = bf16(h x g') x gamma in place (tensor_tensor, then tensor_scalar by the tile's gamma column: exact).
#   The scale tile sc [128, NSCG] fp32 is 9c's [128, NSC] with SC_DQ_PROJ x gamma (forward), SC_DQ_DH2 x gamma,
#   SC_S_DP x s_g, SC_DQ_DX x gamma (backward), and two more columns: SC_DQ_DG (SC_DQ_DH2 x gamma / 2) and SC_GAMMA.
# ---------------------------------------------------------------------------------------------------------------------

SC_DQ_DG, SC_GAMMA = 16, 17
NSCG = 18
E_GATE = (-24, 24)          # gate scales 2^e

if _HAS_NKI:
    def _load_scg(sc):
        """The call's gated scale tile [P, NSCG] fp32 by one 2-D DMA (9c's _load_sc at NSCG columns)."""
        s = nl.ndarray((_P, NSCG), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s, src=sc)
        return s

    def _evac_relu8g(r_sb, rc, hs, g_sb, h8, f, pss, ntok, sc_sb, forms):
        """Gated forward phase 1 evacuation, per chunk: 9c's r = bf16(max(acc x dq_fc, 0)) and rc = bf16(min(r x c,
        15.375)) (forms 0: two two-operator VectorE instructions; forms 1: four one-operator ones), then hs = rc^2 into
        the fp32 stage (ScalarE, exact), hs = hs x g' (VectorE, exact), h8[f] = e4m3(hs) (ScalarE copy at scale 1.0,
        one rounding)."""
        a0 = SC_DQ_FC
        a1 = a0 + 1
        b0 = SC_C_H
        b1 = b0 + 1
        nch = _ceil_div(ntok, _CH)
        for ci in range(nch):
            c0 = ci * _CH
            c1 = c0 + _CH
            if c1 > ntok:
                c1 = ntok
            if forms == 1:
                nisa.tensor_scalar(dst=r_sb[:, c0:c1], data=pss[ci], op0=nl.multiply, operand0=sc_sb[:, a0:a1],
                                   engine=_VEC)
                nisa.tensor_scalar(dst=r_sb[:, c0:c1], data=r_sb[:, c0:c1], op0=nl.maximum, operand0=0.0, engine=_VEC)
                nisa.tensor_scalar(dst=rc[:, c0:c1], data=r_sb[:, c0:c1], op0=nl.multiply, operand0=sc_sb[:, b0:b1],
                                   engine=_VEC)
                nisa.tensor_scalar(dst=rc[:, c0:c1], data=rc[:, c0:c1], op0=nl.minimum, operand0=H_CLAMP,
                                   engine=_VEC)
            else:
                nisa.tensor_scalar(dst=r_sb[:, c0:c1], data=pss[ci], op0=nl.multiply, operand0=sc_sb[:, a0:a1],
                                   op1=nl.maximum, operand1=0.0, engine=_VEC)
                nisa.tensor_scalar(dst=rc[:, c0:c1], data=r_sb[:, c0:c1], op0=nl.multiply, operand0=sc_sb[:, b0:b1],
                                   op1=nl.minimum, operand1=H_CLAMP, engine=_VEC)
            nisa.activation(dst=hs[:, c0:c1], op=nl.square, data=rc[:, c0:c1])
            nisa.tensor_tensor(dst=hs[:, c0:c1], data1=hs[:, c0:c1], data2=g_sb[:, c0:c1], op=nl.multiply)
            nisa.activation(dst=h8[:, f, c0:c1], op=nl.copy, data=hs[:, c0:c1], scale=1.0)

    def _evac_fold8g(dp_sb, dg_sb, r_sb, h_sb, g_sb, pss, ntok, sc_sb):
        """Gated backward phase A evacuation, per chunk (h_sb holds h = r^2): dg' = bf16((acc x dq_dg) x h) (9c's fold
        form), r_sb = bf16(r x g') in place (VectorE), dpre = bf16((acc x dq_dh2') x rg') (9c's fold form)."""
        a0 = SC_DQ_DG
        a1 = a0 + 1
        b0 = SC_DQ_DH2
        b1 = b0 + 1
        nch = _ceil_div(ntok, _CH)
        for ci in range(nch):
            c0 = ci * _CH
            c1 = c0 + _CH
            if c1 > ntok:
                c1 = ntok
            nisa.scalar_tensor_tensor(dst=dg_sb[:, c0:c1], data=pss[ci], op0=nl.multiply, operand0=sc_sb[:, a0:a1],
                                      op1=nl.multiply, operand1=h_sb[:, c0:c1])
            nisa.tensor_tensor(dst=r_sb[:, c0:c1], data1=r_sb[:, c0:c1], data2=g_sb[:, c0:c1], op=nl.multiply)
            nisa.scalar_tensor_tensor(dst=dp_sb[:, c0:c1], data=pss[ci], op0=nl.multiply, operand0=sc_sb[:, b0:b1],
                                      op1=nl.multiply, operand1=r_sb[:, c0:c1])

    @nki.jit
    def _mlp8g_fwd_kernel(x2, w_fc, w_proj, sc, gT, tb, forms):
        M = x2.shape[0]
        D = x2.shape[1]
        F = w_fc.shape[0]
        dt = x2.dtype
        f32 = nl.float32
        y = nl.ndarray((M, D), dtype=dt, buffer=nl.shared_hbm)
        rT = nl.ndarray((F, M), dtype=dt, buffer=nl.shared_hbm)
        ND = D // _P
        NF = F // _P
        NDD = D // _KS
        NFD = F // _KS
        sc_sb = _load_scg(sc)
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
            ntok = nt * _P
            row0 = bt0 * _P
            row1 = row0 + ntok
            xt = _load_T_block(x2, D, ND, row0, ntok, dt)
            x8 = nl.ndarray((_P, ND, ntok), dtype=_F8E4, buffer=nl.sbuf)
            nisa.activation(dst=x8, op=nl.copy, data=xt, scale=X_SCALE)
            h8 = nl.ndarray((_P, NF, ntok), dtype=_F8E4, buffer=nl.sbuf)
            # Phase 1: pre^T chunks in fp8, r out (bf16), the gated h^T resident in e4m3.
            for f in range(NF):
                f0 = f * _P
                f1 = f0 + _P
                wt = _load_wT_chunk(w_fc, D, ND, f, dt)
                w8 = nl.ndarray((_P, ND, _P), dtype=_F8E4, buffer=nl.sbuf)
                _cast8(w8, wt, sc_sb, SC_S_WFC, forms)
                g_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                nisa.dma_copy(dst=g_sb, src=gT[f0:f1, row0:row1])
                pss = _pchunks(ntok, f32)
                _accumulate8(pss, w8, x8, NDD, ntok)
                r_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                rc = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                hs = nl.ndarray((_P, ntok), dtype=f32, buffer=nl.sbuf)
                _evac_relu8g(r_sb, rc, hs, g_sb, h8, f, pss, ntok, sc_sb, forms)
                nisa.dma_copy(dst=rT[f0:f1, row0:row1], src=r_sb)
            # Phase 2: y^T chunks in fp8, descaled (dq_proj x gamma), transposed to token-major, one store per block.
            ystage = nl.ndarray((_P, nt, D), dtype=dt, buffer=nl.sbuf)
            for j in range(ND):
                wpt = _load_wT_chunk(w_proj, F, NF, j, dt)
                wp8 = nl.ndarray((_P, NF, _P), dtype=_F8E4, buffer=nl.sbuf)
                _cast8(wp8, wpt, sc_sb, SC_S_WPROJ, forms)
                pss = _pchunks(ntok, f32)
                _accumulate8(pss, wp8, h8, NFD, ntok)
                yT_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                _evac_desc(yT_sb, pss, ntok, sc_sb, SC_DQ_PROJ)
                _transpose_group(ystage, j, yT_sb, nt, dt)
            _store_rows(y, D, row0, nt, ystage)
        return y, rT

    @nki.jit
    def _mlp8g_bwd_kernel(dy2, rT, gT, w_fc, w_proj, sc, tb, forms):
        M = dy2.shape[0]
        D = dy2.shape[1]
        F = rT.shape[0]
        dt = dy2.dtype
        f32 = nl.float32
        dx = nl.ndarray((M, D), dtype=dt, buffer=nl.shared_hbm)
        h_out = nl.ndarray((F, M), dtype=dt, buffer=nl.shared_hbm)
        dp_out = nl.ndarray((F, M), dtype=dt, buffer=nl.shared_hbm)
        dg_out = nl.ndarray((F, M), dtype=dt, buffer=nl.shared_hbm)
        ND = D // _P
        NF = F // _P
        NDD = D // _KS
        NFD = F // _KS
        sc_sb = _load_scg(sc)
        g0 = SC_GAMMA
        g1 = g0 + 1
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
            ntok = nt * _P
            row0 = bt0 * _P
            row1 = row0 + ntok
            dyt = _load_T_block(dy2, D, ND, row0, ntok, dt)
            dy8 = nl.ndarray((_P, ND, ntok), dtype=_F8E5, buffer=nl.sbuf)
            _cast8(dy8, dyt, sc_sb, SC_S_DY, forms)
            dp8 = nl.ndarray((_P, NF, ntok), dtype=_F8E5, buffer=nl.sbuf)
            # Phase A: dhg^T chunks in fp8; dg', dpre on evacuation (bf16), hg for the weight gradient, dpre to e5m2.
            for f in range(NF):
                f0 = f * _P
                f1 = f0 + _P
                wp = _load_w_chunk(w_proj, F, ND, f, dt)
                wp8 = nl.ndarray((_P, ND, _P), dtype=_F8E4, buffer=nl.sbuf)
                _cast8(wp8, wp, sc_sb, SC_S_WPROJ, forms)
                pss = _pchunks(ntok, f32)
                _accumulate8(pss, wp8, dy8, NDD, ntok)
                r_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                nisa.dma_copy(dst=r_sb, src=rT[f0:f1, row0:row1])
                g_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                nisa.dma_copy(dst=g_sb, src=gT[f0:f1, row0:row1])
                h_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                nisa.activation(dst=h_sb, op=nl.square, data=r_sb)
                dp_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                dg_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                _evac_fold8g(dp_sb, dg_sb, r_sb, h_sb, g_sb, pss, ntok, sc_sb)
                nisa.tensor_tensor(dst=h_sb, data1=h_sb, data2=g_sb, op=nl.multiply)
                nisa.tensor_scalar(dst=h_sb, data=h_sb, op0=nl.multiply, operand0=sc_sb[:, g0:g1], engine=_VEC)
                nisa.dma_copy(dst=h_out[f0:f1, row0:row1], src=h_sb)
                nisa.dma_copy(dst=dp_out[f0:f1, row0:row1], src=dp_sb)
                nisa.dma_copy(dst=dg_out[f0:f1, row0:row1], src=dg_sb)
                if forms == 1:
                    dps = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                else:
                    dps = dp_sb
                _cast_dp8(dp8, f, dp_sb, dps, sc_sb, forms)
            # Phase B: dx^T chunks in fp8, descaled (dq_dx x gamma), transposed to token-major, one store per block.
            dxstage = nl.ndarray((_P, nt, D), dtype=dt, buffer=nl.sbuf)
            for j in range(ND):
                wf = _load_w_chunk(w_fc, D, NF, j, dt)
                wf8 = nl.ndarray((_P, NF, _P), dtype=_F8E4, buffer=nl.sbuf)
                _cast8(wf8, wf, sc_sb, SC_S_WFC, forms)
                pss = _pchunks(ntok, f32)
                _accumulate8(pss, wf8, dp8, NFD, ntok)
                dxT_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                _evac_desc(dxT_sb, pss, ntok, sc_sb, SC_DQ_DX)
                _transpose_group(dxstage, j, dxT_sb, nt, dt)
            _store_rows(dx, D, row0, nt, dxstage)
        return dx, h_out, dp_out, dg_out


def gate_scales(table: torch.Tensor) -> torch.Tensor:
    """Switch 48 on the fp8 line: [s_g, gamma] fp32 from the gate table as the forward reads it (bf16): G = max |U|,
    s_g = 2^e the largest power of two with G x s_g <= 1 (e in E_GATE), gamma = 1 / s_g; both exact. G = 1 (the table
    at ones) gives [1, 1]."""
    amax = table.detach().to(torch.bfloat16).float().abs().amax()
    s_g, gamma = _pow2_le(amax, 1.0, E_GATE[0], E_GATE[1])
    return torch.stack([s_g, gamma])


def gated_row(row: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    """The gated forward's row [NSCG] from 9c's forward row [NSC] (forward_scales') and [s_g, gamma]: SC_DQ_PROJ x
    gamma, and SC_DQ_DG = 0 (the forward does not read it), SC_GAMMA = gamma."""
    gamma = gs[1]
    out = torch.cat([row[:SC_DQ_PROJ], (row[SC_DQ_PROJ] * gamma).reshape(1), row[SC_DQ_PROJ + 1:]])
    return torch.cat([out, torch.zeros_like(gamma).reshape(1), gamma.reshape(1)])


def gated_bwd_row(row_b: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    """The gated backward's row [NSCG] from 9c's backward row [NSC] (backward_scales' on 9c's forward row) and
    [s_g, gamma]: SC_DQ_PROJ, SC_DQ_DH2 and SC_DQ_DX x gamma, SC_S_DP x s_g, SC_DQ_DG = SC_DQ_DH2 x gamma / 2,
    SC_GAMMA = gamma (every factor a power of two: exact)."""
    s_g, gamma = gs[0], gs[1]
    mult = torch.ones(NSC, dtype=torch.float32, device=row_b.device)
    mult = torch.where(torch.arange(NSC, device=row_b.device) == SC_DQ_PROJ, gamma, mult)
    mult = torch.where(torch.arange(NSC, device=row_b.device) == SC_DQ_DH2, gamma, mult)
    mult = torch.where(torch.arange(NSC, device=row_b.device) == SC_DQ_DX, gamma, mult)
    mult = torch.where(torch.arange(NSC, device=row_b.device) == SC_S_DP, s_g, mult)
    out = row_b.float() * mult
    return torch.cat([out, (out[SC_DQ_DH2] * 0.5).reshape(1), gamma.reshape(1)])


def sc_tile_g(row: torch.Tensor) -> torch.Tensor:
    """[128, NSCG] fp32: the gated row down every partition (the gated kernels' scale input)."""
    return row.float().reshape(1, NSCG).expand(_P, NSCG).contiguous()


def emulate8g_forward(x2: torch.Tensor, w_fc: torch.Tensor, w_proj: torch.Tensor, row: torch.Tensor, gT: torch.Tensor):
    """_mlp8g_fwd_kernel's arithmetic in torch (the same casts and rounding points, fp32 products and accumulation in
    torch's order). row [NSCG] (gated_row's), gT [F, M] the pre-scaled gate g' (bf16). Returns y [M, d], rT [F, M]."""
    dt = x2.dtype
    x8 = q8(x2.float() * X_SCALE, 0)
    w8 = q8(w_fc.float() * row[SC_S_WFC], 0)
    r = torch.clamp_min((x8 @ w8.t()) * row[SC_DQ_FC], 0.0).to(dt)
    rc = torch.clamp_max(r.float() * row[SC_C_H], H_CLAMP).to(dt)
    hg8 = q8((rc.float() * rc.float()) * gT.t().float(), 0)
    wp8 = q8(w_proj.float() * row[SC_S_WPROJ], 0)
    y = ((hg8 @ wp8.t()) * row[SC_DQ_PROJ]).to(dt)
    return y, r.t().contiguous()


def emulate8g_backward(dy2: torch.Tensor, rT: torch.Tensor, gT: torch.Tensor, w_fc: torch.Tensor,
                       w_proj: torch.Tensor, row: torch.Tensor):
    """_mlp8g_bwd_kernel's arithmetic in torch. row [NSCG] (gated_bwd_row's). Returns dx [M, d], and hg^T, dpre^T,
    dg'^T [F, M] in dy2's dtype (hg and dpre are the bf16 values the weight gradients read)."""
    dt = dy2.dtype
    dy8 = q8(dy2.float() * row[SC_S_DY], 1)
    wp8 = q8(w_proj.float() * row[SC_S_WPROJ], 0)
    acc = dy8 @ wp8
    r = rT.t().float()
    g = gT.t().float()
    h = (r * r).to(dt)
    dg = ((acc * row[SC_DQ_DG]) * h.float()).to(dt)
    rg = (r * g).to(dt)
    dpre = ((acc * row[SC_DQ_DH2]) * rg.float()).to(dt)
    hg = ((h.float() * g).to(dt).float() * row[SC_GAMMA]).to(dt)
    dp8 = q8(dpre.float() * row[SC_S_DP], 1)
    wf8 = q8(w_fc.float() * row[SC_S_WFC], 0)
    dx = ((dp8 @ wf8) * row[SC_DQ_DX]).to(dt)
    return dx, hg.t().contiguous(), dpre.t().contiguous(), dg.t().contiguous()


def weight_grads_g(dy2: torch.Tensor, x2: torch.Tensor, hg_out: torch.Tensor, dp_out: torch.Tensor):
    """The two weight gradients from the gated backward's F-major outputs (9c's dw_mode 1 expressions, hg in h's
    place): dw_fc [F, d] = dpre^T x2, dw_proj [d, F] = (hg^T dy2)^T."""
    return dp_out.mm(x2), hg_out.mm(dy2).t()


class EmulatedGatedMlp8(torch.autograd.Function):
    """The gated fp8 MLP where the kernels cannot run (CPU proxies and proofs): forward(x2, w_fc, w_proj, sc, gs, gT),
    sc the layer's 9c tile [128, NSC] (forward_scales'), gs [s_g, gamma] (gate_scales'), gT [F, M] the pre-scaled gate
    g'; the kernels' arithmetic, the backward's scales from dy as the device's backward forms them; d(gT) = dg'^T."""

    @staticmethod
    def forward(ctx, x2, w_fc, w_proj, sc, gs, gT):
        y, rT = emulate8g_forward(x2, w_fc, w_proj, gated_row(sc[0], gs), gT)
        ctx.save_for_backward(x2, rT, gT, w_fc, w_proj, sc, gs)
        return y

    @staticmethod
    def backward(ctx, dy2):
        x2, rT, gT, w_fc, w_proj, sc, gs = ctx.saved_tensors
        dy2 = dy2.contiguous()
        row = gated_bwd_row(backward_scales(dy2, sc[0]), gs)
        dx, hg_out, dp_out, dg_out = emulate8g_backward(dy2, rT, gT, w_fc, w_proj, row)
        dw_fc, dw_proj = weight_grads_g(dy2, x2, hg_out, dp_out)
        return dx, dw_fc, dw_proj, None, None, dg_out
