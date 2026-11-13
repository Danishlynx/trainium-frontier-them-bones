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

import os

import torch

_P = 128
_CH = 512
_env = os.environ.get
TB = int(_env("R1_NKI_MLP_TB", "1024"))
_DW_NAMES = {"token": 0, "fmajor": 1}
DW_MODE = _DW_NAMES[_env("R1_NKI_MLP_DW", "token")]
_LNC = int(_env("NEURON_LOGICAL_NC_CONFIG", "1"))

_HAS_NKI = False
_NKI_ERROR = None
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    _VEC = nisa.engine.vector

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
