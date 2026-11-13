"""nki_mlp_stem.py: the fused ReLU-squared MLP with a per-token gate on its hidden activation, as two NKI kernels
(switch 48, R2_STEM with R2_STEM_KERNEL=1; lane STEM, work/p2/stem/NOTES.md).

The STEM form of the Round 2 MLP: y = c_proj(relu(c_fc(x))^2 * g) with g = U[t], one row of a per-layer token table U
[vocab, F] for each token (F = 4 d). switch 9b's kernels (nki_mlp_fused.py, c3209e4b) keep h^T = r^2 in SBUF between
the two matmuls, so the gate cannot be applied outside them; these two kernels are switch 9b's with the gate inside.
The helpers this file uses (_shard, _span, _ceil_div, _load_T_block, _load_wT_chunk, _load_w_chunk, _pchunks,
_accumulate, _evac_copy, _transpose_group, _store_rows) are nki_mlp_fused.py's, byte for byte (proofs/
static check); the two gated evacuations, the two kernels and the emulation are new. Only the F-major
weight-gradient layout (switch 9b's dw_mode 1, R2_MLP_DW=fmajor) is built.

  forward   y, rT = mlp_fwd_g(x2, w_fc, w_proj, gT, tb)
            x2 [M, d] bf16; w_fc [F, d], w_proj [d, F] bf16; gT [F, M] bf16, the gate rows F-major (the caller
            transposes U[t]: the compiler's transpose, so the kernel loads a slab of g exactly as the backward loads
            a slab of rT, one dma_copy of a [128, ntok] slice). Phase 1 is switch 9b's (r = relu(acc) on VectorE, r
            stored to rT, h^T[f] = r^2 on ScalarE into the resident h^T tile) plus one VectorE tensor_tensor that
            multiplies the slab in place by the gate: h^T[f] = h^T[f] * g^T[f] (bf16 out). Phase 2 is switch 9b's.
  backward  dx, hgT, dpT, dgT = mlp_bwd_g(dy2, rT, gT, w_fc, w_proj, tb)
            Phase A: the accumulator acc = dhg^T[f] = sum_k Wproj[k, f] dy^T[k, :] (switch 9b's GEMM; dhg is the
            gradient of the gated activation hg = h * g). Per 512-column chunk, with h = r^2 (ScalarE, as 9b):
              dg^T   = acc * h            (scalar_tensor_tensor reading PSUM: 9b's own evacuation form, operand 1.0)
              hg     = h * g              (tensor_tensor, in place on the h tile)
              rg     = r * g              (tensor_tensor, in place on the r tile)
              dpre^T = (acc * 2) * rg     (9b's evacuation, rg in r's place)
            dpre^T stays in SBUF for phase B (9b's, unchanged); hg^T, dpre^T and dg^T are stored F-major [F, M] as
            they lie (9b's dw_mode 1 stores, plus one for dg^T). The caller computes the weight gradients as 9b's
            weight_grads(dw_mode 1) with hg in h's place (dWproj = dy^T hg is the gradient through the gated h), and
            d(gT) = dg^T goes back through the transpose to the gate rows (the table's scatter is the caller's).

With g = 1 (the table's initialisation) both kernels are switch 9b's bit for bit: h * 1 = h, r * 1 = r, and dpre is 9b's
single evacuation of (acc * 2) * r; dg^T is the extra output.

Rounding points (the emulation below has the same ones): r = bf16(relu(acc)); h = bf16(r^2); hg = bf16(h * g) (the
kit chain's two roundings of relu(.)^2 * g); y, dx bf16 of fp32 accumulators; dg = bf16(acc_f32 * h); rg = bf16(r * g);
dpre = bf16((acc_f32 * 2) * rg). The kit chain under autograd rounds dhg to bf16 first and forms dh = bf16(dhg * g)
then dpre = bf16(2 r dh): a few bf16 ulps apart, the same class of difference as switch 9b's against the kit.

Instruction forms (device record): every instruction is one switch 9b's kernels or candidate 33's attention kernel
(c26 nki_attn.py) has compiled on the chip: nc_matmul, dma_transpose (hwdge strips), dma_copy of [128, ntok] slices of
an [F, M] tensor (9b's rT load), tensor_scalar maximum, activation square, scalar_tensor_tensor reading PSUM with an
SBUF bf16 operand1 (9b's fold), nc_transpose through bf16 PSUM, tensor_copy, and tensor_tensor multiply of two SBUF
tiles (c26 nki_attn.py lines 346 and 399, the latter in place). The new combination is a tensor_tensor whose dst and
data1 are one subscript of a 3-D tile (hT[:, f, c0:c1]); 9b's activation writes the same subscript.

SBUF per partition (bf16, ntok = tb): forward 9b's about 182 KiB at tb 1024 plus one g tile (2 KiB, about 4 with the
allocator's double buffer); backward 9b's about 186 KiB plus the g and dg tiles (4 KiB, about 8 double-buffered), so
about 194 KiB at tb 1024 against 224 KiB a partition. tb 512 (the switch's default, R2_STEM_TB) halves the resident
blocks (about 108 + 8 KiB). PSUM as 9b's (each accumulator is read twice in the backward, by two VectorE
instructions, before it is released).

Written for the device-side NKI parser frontend under 9b's rules (nki_mlp_fused.py's docstring): every loop is `for
name in range(...)`; block bounds are integer arithmetic; tuples only in assignments; no list, comprehension, `while`
or tuple-element loop target in a traced body; every slice bound a plain name or a literal; every PSUM tile 2-D and at
most one bank; a tile or view is subscripted once.

Knobs: NEURON_LOGICAL_NC_CONFIG (the launch width: kernel[2] under LNC 2); tb is a kernel argument (the caller's).
"""

import os

import torch

_P = 128
_CH = 512
_env = os.environ.get
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

    def _evac_copy(dst2, pss, ntok):
        """bf16 of the accumulator chunks into dst2 [P, ntok] on ScalarE (the GEMM kernel's evacuation)."""
        nch = _ceil_div(ntok, _CH)
        for ci in range(nch):
            c0 = ci * _CH
            c1 = c0 + _CH
            if c1 > ntok:
                c1 = ntok
            nisa.activation(dst=dst2[:, c0:c1], op=nl.copy, data=pss[ci])

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

    def _store_rows(out, N, row0, nt, stage):
        """out[row0 + tt * 128 + p, :] = stage[p, tt, :] for the block's nt token tiles (one DMA)."""
        nisa.dma_copy(dst=out.ap([[N, _P], [_P * N, nt], [1, N]], offset=row0 * N), src=stage)

    # ------------------------------------------------------------------------------------- the gated evacuations

    def _evac_relu_gate(r_sb, hT, f, g_sb, pss, ntok):
        """Forward phase 1 evacuation, per chunk: switch 9b's r = relu(acc) (VectorE, bf16) and h^T[f] = r^2 (ScalarE,
        into the resident tile), then the gate in place: h^T[f] = h^T[f] * g (VectorE tensor_tensor, bf16 out)."""
        nch = _ceil_div(ntok, _CH)
        for ci in range(nch):
            c0 = ci * _CH
            c1 = c0 + _CH
            if c1 > ntok:
                c1 = ntok
            nisa.tensor_scalar(dst=r_sb[:, c0:c1], data=pss[ci], op0=nl.maximum, operand0=0.0, engine=_VEC)
            nisa.activation(dst=hT[:, f, c0:c1], op=nl.square, data=r_sb[:, c0:c1])
            nisa.tensor_tensor(dst=hT[:, f, c0:c1], data1=hT[:, f, c0:c1], data2=g_sb[:, c0:c1], op=nl.multiply)

    def _evac_fold_gate(dpT, f, dg_sb, h_sb, r_sb, g_sb, pss, ntok):
        """Backward phase A evacuation, per chunk (h_sb holds h = r^2 on entry): dg^T = acc * h (switch 9b's fold form,
        operand 1.0); then h_sb = h * g (hg, in place) and r_sb = r * g (rg, in place) on VectorE; then dpre^T[f] =
        (acc * 2) * rg (switch 9b's fold, rg in r's place). Every output bf16, one rounding each."""
        nch = _ceil_div(ntok, _CH)
        for ci in range(nch):
            c0 = ci * _CH
            c1 = c0 + _CH
            if c1 > ntok:
                c1 = ntok
            nisa.scalar_tensor_tensor(dst=dg_sb[:, c0:c1], data=pss[ci], op0=nl.multiply, operand0=1.0,
                                      op1=nl.multiply, operand1=h_sb[:, c0:c1])
            nisa.tensor_tensor(dst=h_sb[:, c0:c1], data1=h_sb[:, c0:c1], data2=g_sb[:, c0:c1], op=nl.multiply)
            nisa.tensor_tensor(dst=r_sb[:, c0:c1], data1=r_sb[:, c0:c1], data2=g_sb[:, c0:c1], op=nl.multiply)
            nisa.scalar_tensor_tensor(dst=dpT[:, f, c0:c1], data=pss[ci], op0=nl.multiply, operand0=2.0,
                                      op1=nl.multiply, operand1=r_sb[:, c0:c1])

    @nki.jit
    def _mlp_fwd_g_kernel(x2, w_fc, w_proj, gT, tb):
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
            # Phase 1: pre^T chunks, r out, h^T = r^2 * g resident.
            for f in range(NF):
                f0 = f * _P
                f1 = f0 + _P
                wt = _load_wT_chunk(w_fc, D, ND, f, dt)
                g_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                nisa.dma_copy(dst=g_sb, src=gT[f0:f1, row0:row1])
                pss = _pchunks(ntok, f32)
                _accumulate(pss, wt, xt, ND, ntok)
                r_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                _evac_relu_gate(r_sb, hT, f, g_sb, pss, ntok)
                nisa.dma_copy(dst=rT[f0:f1, row0:row1], src=r_sb)
            # Phase 2: y^T chunks, transposed to token-major, one store per block (switch 9b's).
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
    def _mlp_bwd_g_kernel(dy2, rT, gT, w_fc, w_proj, tb):
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
            # Phase A: dhg^T chunks; dg, hg, dpre on evacuation (h rebuilt from r as in switch 9b).
            for f in range(NF):
                f0 = f * _P
                f1 = f0 + _P
                wp = _load_w_chunk(w_proj, F, ND, f, dt)
                pss = _pchunks(ntok, f32)
                _accumulate(pss, wp, dyt, ND, ntok)
                r_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                nisa.dma_copy(dst=r_sb, src=rT[f0:f1, row0:row1])
                g_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                nisa.dma_copy(dst=g_sb, src=gT[f0:f1, row0:row1])
                h_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                nisa.activation(dst=h_sb, op=nl.square, data=r_sb)
                dg_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                _evac_fold_gate(dpT, f, dg_sb, h_sb, r_sb, g_sb, pss, ntok)
                nisa.dma_copy(dst=h_out[f0:f1, row0:row1], src=h_sb)
                nisa.dma_copy(dst=dp_out[f0:f1, row0:row1], src=dpT[:, f, :])
                nisa.dma_copy(dst=dg_out[f0:f1, row0:row1], src=dg_sb)
            # Phase B: dx^T chunks, transposed to token-major, one store per block (switch 9b's).
            dxstage = nl.ndarray((_P, nt, D), dtype=dt, buffer=nl.sbuf)
            for j in range(ND):
                wf = _load_w_chunk(w_fc, D, NF, j, dt)
                pss = _pchunks(ntok, f32)
                _accumulate(pss, wf, dpT, NF, ntok)
                dxT_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                _evac_copy(dxT_sb, pss, ntok)
                _transpose_group(dxstage, j, dxT_sb, nt, dt)
            _store_rows(dx, D, row0, nt, dxstage)
        return dx, h_out, dp_out, dg_out

    _HAS_NKI = True
except Exception as exc:  # nki unavailable (CPU dev box): the emulation below still works
    _NKI_ERROR = f"{type(exc).__name__}: {exc}"


def _launch(kernel):
    """kernel[2] under an LNC=2 runtime (both physical cores run half the token tiles), plain call otherwise."""
    return kernel[_LNC] if _LNC == 2 else kernel


def shapes_apply(m: int, d: int, f: int, tb: int) -> bool:
    """The shapes the kernels take: multiples of 128 everywhere, tb a multiple of 128 up to 1024."""
    return (m % _P == 0 and d % _P == 0 and f % _P == 0 and m > 0 and tb % _P == 0 and 0 < tb <= 1024)


# ----------------------------------------------------------------------------- torch emulation (CPU and proofs)

def emulate_forward(x2: torch.Tensor, w_fc: torch.Tensor, w_proj: torch.Tensor, gT: torch.Tensor):
    """The forward kernel's arithmetic in torch (fp32 accumulation in torch's order, the kernel's rounding points).
    Returns y [M, d] bf16 and rT [F, M] bf16."""
    dt = x2.dtype
    pre = x2.float() @ w_fc.float().t()
    r = torch.relu(pre).to(dt)
    h = (r.float() * r.float()).to(dt)
    hg = (h.float() * gT.t().float()).to(dt)
    y = (hg.float() @ w_proj.float().t()).to(dt)
    return y, r.t().contiguous()


def emulate_backward(dy2: torch.Tensor, rT: torch.Tensor, gT: torch.Tensor, w_fc: torch.Tensor,
                     w_proj: torch.Tensor):
    """The backward kernel's arithmetic: dx [M, d], and hg^T, dpre^T, dg^T [F, M], bf16."""
    dt = dy2.dtype
    r = rT.t()
    g = gT.t()
    acc = dy2.float() @ w_proj.float()                      # dhg, fp32 (the PSUM accumulator)
    h = (r.float() * r.float()).to(dt)
    dg = (acc * h.float()).to(dt)
    hg = (h.float() * g.float()).to(dt)
    rg = (r.float() * g.float()).to(dt)
    dpre = ((acc * 2.0) * rg.float()).to(dt)
    dx = (dpre.float() @ w_fc.float()).to(dt)
    return dx, hg.t().contiguous(), dpre.t().contiguous(), dg.t().contiguous()


def weight_grads(dy2: torch.Tensor, x2: torch.Tensor, hg_out: torch.Tensor, dp_out: torch.Tensor):
    """The two weight gradients from the backward kernel's F-major outputs (switch 9b's dw_mode 1 expressions, with
    the gated hg in h's place): dw_fc [F, d] = dpre^T x2, dw_proj [d, F] = (hg^T dy2)^T, in the activations' dtype."""
    return dp_out.mm(x2), hg_out.mm(dy2).t()


def reference_forward(x2: torch.Tensor, w_fc: torch.Tensor, w_proj: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """The kit chain with the gate, c_proj(relu(c_fc(x))^2 * g), in the activations' dtype (g token-major [M, F])."""
    return torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(x2, w_fc)).square() * g, w_proj)


class EmulatedGatedMlp(torch.autograd.Function):
    """The emulation as an autograd function over (x2, w_fc, w_proj, gT): the forward saves x2, rT, gT and the
    weights; the backward is the kernel's plus the two weight gradients, and d(gT) = dg^T. Used where the kernels
    cannot run (CPU proxies and proofs)."""

    @staticmethod
    def forward(ctx, x2, w_fc, w_proj, gT):
        y, rT = emulate_forward(x2, w_fc, w_proj, gT)
        ctx.save_for_backward(x2, rT, gT, w_fc, w_proj)
        return y

    @staticmethod
    def backward(ctx, dy2):
        x2, rT, gT, w_fc, w_proj = ctx.saved_tensors
        dy2 = dy2.contiguous()
        dx, hg_out, dp_out, dg_out = emulate_backward(dy2, rT, gT, w_fc, w_proj)
        dw_fc, dw_proj = weight_grads(dy2, x2, hg_out, dp_out)
        return dx, dw_fc, dw_proj, dg_out
