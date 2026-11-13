"""nki_resmix.py: switch 9i (R2_RESMIX_FUSE, lane THR8): the residual mix's backward in one pass a block.

The block's mix is out = a x + b x0 (a = resid_lambda, b = x0_lambda, both cast to the stream's bf16), so its backward
is dx = g a, dx0 = g b and the two lambda gradients sum(g x) and sum(g x0), full reductions over every token and
channel. The compiled graph runs them as separate passes (two elementwise products, two full reductions on VectorE,
two scaled copies: 18.5 ms a step at 18 x 1,536, design 06 row 7). One kernel call a block:

  _resmix_bwd_kernel(g2, x2, x02, ab) -> (dx2 [M, D], dx02 [M, D], part [nprg x 128, 2] fp32)
      per 128-token tile: g, x and x0 rows by three natural DMAs; dx = bf16(g x a) and dx0 = bf16(g x b) on VectorE
      (tensor_scalar: the fp32 product, one rounding, as torch's bf16 multiply; no bias add, so a zero keeps its
      sign); the products bf16(g x) and bf16(g x0) on VectorE (tensor_tensor: torch's bf16 product, as the autograd
      backward forms it before its reduction) and their row sums in fp32 on ScalarE (activation_reduce, one column of
      an fp32 accumulator a tile); at the end each program reduces its columns (tensor_reduce) and writes its 128
      partition partials, row pid x 128 + p of part: sum over the program's tiles t of sum_j bf16(g x)[t 128 + p, j]
      (column 0) and of bf16(g x0) (column 1). The caller sums part's rows in fp32 and rounds to the lambdas' bf16 as
      the autograd backward does.

  ab [128, 2] fp32: a and b (the bf16 values, exact in fp32) down every partition (ab_tile).

What is the same as the compiled backward and what is not: dx and dx0 are torch's bf16 products element for element;
each product of the reductions is torch's bf16 product; only the order of the fp32 sums differs (within a tile's row
the instruction's order, then the program's tiles, then the caller's sum over 128 x nprg partials), so each lambda
gradient is within gamma_h sum |bf16(g x)| of the exact sum of those products, h = (D - 1) + (tiles a program - 1) +
(128 nprg - 1) the depth of the summation, gamma_h = h u / (1 - h u), u = 2^-24 (sum_bound), before the bf16 rounding
both paths share. The order of the compiled graph's own reduction is the compiler's and is not known here.

Shapes: M a multiple of 128 with at least one tile a program, D any multiple of 1 (one row a DMA; 1,536 in training).
Written for the device-side NKI parser frontend under switch 9d's rules (kernels/nki_attn_proj8.py's docstring):
plain loop targets, if only on trace-time integers, no list, comprehension, while, lambda or nested def in a traced
body, every slice bound a plain name or a literal, a tile subscripted once.
"""

import os

import torch

_P = 128
_env = os.environ.get
_LNC = int(_env("NEURON_LOGICAL_NC_CONFIG", "1"))
NAB = 2

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

    @nki.jit
    def _resmix_bwd_kernel(g2, x2, x02, ab):
        M = g2.shape[0]
        D = g2.shape[1]
        dt = g2.dtype
        f32 = nl.float32
        pid, nprg = _shard()
        dx2 = nl.ndarray((M, D), dtype=dt, buffer=nl.shared_hbm)
        dx02 = nl.ndarray((M, D), dtype=dt, buffer=nl.shared_hbm)
        part = nl.ndarray((nprg * _P, NAB), dtype=f32, buffer=nl.shared_hbm)
        ab_sb = nl.ndarray((_P, NAB), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ab_sb, src=ab)
        t0, t1 = _span(M // _P, pid, nprg)
        n = t1 - t0
        acc = nl.ndarray((_P, NAB, n), dtype=f32, buffer=nl.sbuf)
        for i in range(n):
            r0 = (t0 + i) * _P
            r1 = r0 + _P
            c0 = i
            c1 = c0 + 1
            gb = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=gb, src=g2[r0:r1, 0:D])
            xb = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x2[r0:r1, 0:D])
            x0b = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=x0b, src=x02[r0:r1, 0:D])
            db = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=db, data=gb, op0=nl.multiply, operand0=ab_sb[:, 0:1], engine=_VEC)
            nisa.dma_copy(dst=dx2[r0:r1, 0:D], src=db)
            d0b = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=d0b, data=gb, op0=nl.multiply, operand0=ab_sb[:, 1:2], engine=_VEC)
            nisa.dma_copy(dst=dx02[r0:r1, 0:D], src=d0b)
            pb = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pb, data1=gb, data2=xb, op=nl.multiply, engine=_VEC)
            jb = nl.ndarray((_P, D), dtype=f32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=jb, op=nl.copy, data=pb, reduce_op=nl.add, reduce_res=acc[:, 0, c0:c1])
            qb = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=qb, data1=gb, data2=x0b, op=nl.multiply, engine=_VEC)
            kb = nl.ndarray((_P, D), dtype=f32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=kb, op=nl.copy, data=qb, reduce_op=nl.add, reduce_res=acc[:, 1, c0:c1])
        red = nl.ndarray((_P, NAB), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=red, op=nl.add, data=acc, axis=2)
        p0 = pid * _P
        p1 = p0 + _P
        nisa.dma_copy(dst=part[p0:p1, 0:NAB], src=red)
        return dx2, dx02, part

    _HAS_NKI = True
except Exception as exc:  # nki unavailable (CPU dev box): the emulation below still works
    _NKI_ERROR = f"{type(exc).__name__}: {exc}"


def _launch(kernel):
    """kernel[2] under an LNC=2 runtime (both physical cores run half the token tiles), plain call otherwise."""
    return kernel[_LNC] if _LNC == 2 else kernel


def programs() -> int:
    """The number of programs a launch runs (the rows of part are programs() x 128)."""
    return 2 if _LNC == 2 else 1


def shapes_apply(m: int, d: int) -> bool:
    """The shapes the kernel takes: M a multiple of 128 with at least one 128-token tile a program."""
    return m > 0 and d > 0 and m % _P == 0 and m // _P >= programs()


def ab_tile(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[128, 2] fp32: a and b (the bf16 lambdas, exact in fp32) down every partition (the kernel's scale input)."""
    return torch.stack([a.float().reshape(()), b.float().reshape(())]).reshape(1, NAB).expand(_P, NAB).contiguous()


def lambda_grads(part: torch.Tensor, dtype: torch.dtype) -> tuple:
    """The two lambda gradients from the kernel's partials: the fp32 sum of part's rows, rounded to the lambdas'
    compute dtype (the autograd backward's bf16 [1] gradients), as [1] tensors."""
    lam = part.sum(dim=0)
    return lam[0:1].to(dtype), lam[1:2].to(dtype)


def emulate(g2: torch.Tensor, x2: torch.Tensor, x02: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
            nprg: int = 1) -> tuple:
    """The kernel's arithmetic in torch (proofs): dx2, dx02 = g a, g b in the stream's dtype; part [nprg x 128, 2]
    the per-partition fp32 sums of the bf16 products over each program's tiles (torch's summation order)."""
    m, d = g2.shape
    nt = m // _P
    per, rem = nt // nprg, nt % nprg
    px = (g2 * x2).float().reshape(nt, _P, d).sum(dim=2)
    p0 = (g2 * x02).float().reshape(nt, _P, d).sum(dim=2)
    rows = []
    t = 0
    for p in range(nprg):
        cnt = per + (1 if p < rem else 0)
        rows.append(torch.stack([px[t:t + cnt].sum(dim=0), p0[t:t + cnt].sum(dim=0)], dim=1))
        t += cnt
    return g2 * a, g2 * b, torch.cat(rows, dim=0)


def sum_bound(m: int, d: int, nprg: int) -> float:
    """gamma_h for the kernel's summation tree (module docstring): h = (D - 1) + (tiles a program - 1) + (128 nprg - 1),
    u = 2^-24; a lambda gradient's fp32 sum is within gamma_h x sum |bf16 product| of the exact sum."""
    tiles = -(-(m // _P) // nprg)
    h = (d - 1) + (tiles - 1) + (_P * nprg - 1)
    u = 2.0 ** -24
    return h * u / (1.0 - h * u)
