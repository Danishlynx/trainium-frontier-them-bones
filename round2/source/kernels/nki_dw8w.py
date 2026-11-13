"""nki_dw8.py: switch 9k (R2_DW8): the weight gradients of the fp8 projections in fp8 at the TensorE double-row rate
(lane THR9, work/p2/thr9/NOTES.md), with switch 9c's and 9d's scale discipline (work/p2/sched1/kernels/nki_mlp_fused8.py
18369d38, work/p2/fp8q/kernels/nki_attn_proj8.py 275376c2): one kernel form for the four weight-gradient GEMMs of a
block (the MLP's c_fc and c_proj under switch 9c, the attention block's QKV projection and c_proj under switch 9d), each
operand cast in the kernel with the format and the power-of-two scale that the forward or the input-gradient kernel
already uses for that tensor, fp32 PSUM accumulation over the rank's tokens, one exact descale, one bf16 rounding.

  _dw8_kernel(g2, a2, sc, fg, fa, forms) -> out [O, D]
      out = bf16((fp8_fg(g2 x sc[:, 0])^T @ fp8_fa(a2 x sc[:, 1])) x sc[:, 2])
      g2 [M, O] and a2 [M, D] bf16, both token-major (a projection's output gradient and its input, or the MLP's h and
      dy); sc [128, NDW] fp32: the g-side scale s_g, the a-side scale s_a and the descale dq = 1 / (s_g s_a), each one
      value down every partition. fg, fa: FMT_E4M3 (0), FMT_E5M2 (1) or FMT_E4M3C (2: e4m3 of min(v x s, 240), the
      MLP's h, whose forward cast is clamped too).

Layout (both operands natural rows: no strip transposes and no weight path): the two programs of an LNC 2 rank split
the O / 128 output-row blocks between them (no partial sums: each output element is one fp32 PSUM chain over all M
tokens of the rank). Each program first loads every token's a-side row (D columns) by three-level row DMAs, AB token
tiles at a time (switch 9e's btok form), and casts it once into the resident tile a8 [P, MT, D] (48 KiB a partition at
M 4,096 and D 1,536); then for each of its output blocks it loads the g-side column block [M, 128] (switch 9c's
_load_w_chunk form: 128-element row segments), casts it into g8 [P, MT, 128], and accumulates M / 256 double-row matmuls
per 512-column chunk of D with g8[:, 2kd : 2kd + 2, :] STATIONARY (the fused8 kernels' [P, 2, 128] form) and
a8[:, 2kd : 2kd + 2, chunk] MOVING (switch 9c's _accumulate8 moving form), one stationary serving the D / 512 chunks in
turn (switch 9e's _dw_acc order); the evacuation multiplies by dq on VectorE into bf16 (switch 9c's _evac_desc) and one
DMA writes the block's [128, D] rows (switch 9e's dw_fc row store).

Rounding points (the torch emulation below has the same ones, so the simulator proof can be tight): each operand
rounded once into its fp8 format from the exact product of a bf16 value and a power of two (the clamp of format 2 is a
comparison: exact); fp8 x fp8 products exact in fp32; fp32 accumulation (the simulator in its order, the device's
double-row instruction with its 2^-14 relative inner accumulation); one exact descale (a power of two); one bf16
rounding.

Shapes: M a multiple of 256 (a double-row pair of token tiles; the caller pads with zero rows, whose products are exact
zeros), O a multiple of 128, D a multiple of 128 and at most 2,048 (four PSUM banks); plan_kib bounds the SBUF a
partition holds.

nki_dw8w.py (lane L2, work/p2/l2/NOTES.md D1; built from nki_dw8.py 0420bca1 by l2/tools/build_dw8w.py, pure
insertions): _dw8w_kernel(g2, a2, sc, fg, fa, forms, gw) is _dw8_kernel for an a-side wider than four PSUM banks (the
MLP's c_proj weight gradient dy^T h, out [d, F] at F 6,144): the D columns in NG = D / gw groups of gw (a multiple of
128, at most 2,048; group_width picks the largest with NG even), the two programs of an LNC 2 rank splitting the
groups, each group's a-side cast once into a8 [P, MT, gw] and every output block run as _dw8_body runs it, the rows
stored to out[o0:o1, g0:g1]. Each output element is still one fp32 PSUM chain over all M tokens in _dw8_body's order,
one exact descale, one bf16 rounding; emulate_dw is its emulation. The plan at gw is plan_kib(m, gw) (126 KiB at M
4,096 and gw 1,536, the c_fc call's). Every instruction form is _dw8_body's; the group's row load is its three-level
DMA with a column offset (_load_w_chunk's form) and the block store a 2-D slice of the HBM output (the fused8
kernels' rT[f0:f1, row0:row1] store).

Written for the device-side NKI parser frontend under switch 9c's rules (its module docstring): every loop is
`for name in range(...)` with one plain target; `if` only on trace-time integers; tuples only returned and unpacked in
assignments; no list, comprehension, `while`, lambda or nested def in a traced body; every slice bound a plain name or
a literal; every PSUM tile 2-D and one bank; a tile subscripted once. Every instruction form below is one that a kernel
of ours has run on the chip (switches 9c, 9d and 9e): the module copies those helpers verbatim, as the device parser
reads each kernel's own module.
"""

import os

import torch

_P = 128
_CH = 512
_KS = 256                   # contraction per double-row matmul (two 128-token slabs)
_env = os.environ.get
_LNC = int(_env("NEURON_LOGICAL_NC_CONFIG", "1"))
E4M3_MAX = 240.0            # the chip's legacy e4m3: largest finite value (its cast writes inf at 248 and above)
E5M2_MAX = 57344.0
AB = 8                      # token tiles of the a-side a row load carries (1,024 tokens)
PLAN_KIB = 168.0            # SBUF a partition the plan may fill (switch 9d's limit)
D_MAX = 2048                # four PSUM banks of 512 fp32 columns
# The call's scale tile [128, NDW]: the g-side scale, the a-side scale, the descale 1 / (s_g s_a).
DW_S_G, DW_S_A, DW_DQ = 0, 1, 2
NDW = 3
# The operand formats (trace-time integers).
FMT_E4M3, FMT_E5M2, FMT_E4M3C = 0, 1, 2
E_INV = (-60, 60)           # the exponents pow2_inv takes (every scale of switches 9c and 9d lies in 2^-40 .. 2^60)

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

    def _load_w_chunk(w, N, NKB, c, dt):
        """Natural [128 k, 128 n] slices of w [K, N] at column chunk c: [P, NKB, P] (switch 9c's loader, verbatim);
        here w is the g-side [M, O], so the tile holds the 128 columns of output block c for every token tile."""
        wt = nl.ndarray((_P, NKB, _P), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=w.ap([[N, _P], [_P * N, NKB], [1, _P]], offset=c * _P))
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

    def _cast8c(dst8, src, sc_sb, col, forms):
        """dst8 = e4m3(min(src x sc[:, col], 240)): the product and the clamp into src in place on VectorE (exact: a
        power-of-two multiple of a bf16 value, then a comparison; switch 9c's multiply-then-minimum instruction of its
        forward evacuation, forms 1 as two one-operator ones), then a ScalarE copy into e4m3 at the constant scale 1.0
        (the forms-1 cast of _cast8)."""
        q0 = col
        q1 = q0 + 1
        if forms == 1:
            nisa.tensor_scalar(dst=src, data=src, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)
            nisa.tensor_scalar(dst=src, data=src, op0=nl.minimum, operand0=E4M3_MAX, engine=_VEC)
        else:
            nisa.tensor_scalar(dst=src, data=src, op0=nl.multiply, operand0=sc_sb[:, q0:q1], op1=nl.minimum,
                               operand1=E4M3_MAX, engine=_VEC)
        nisa.activation(dst=dst8, op=nl.copy, data=src, scale=1.0)

    def _cast_fmt(dst8, src, sc_sb, col, fmt, forms):
        """The operand's cast by its format: _cast8c for FMT_E4M3C, else _cast8 (dst8's dtype is the format)."""
        if fmt == 2:
            _cast8c(dst8, src, sc_sb, col, forms)
        else:
            _cast8(dst8, src, sc_sb, col, forms)

    def _ptiles(ncol, f32):
        """The fp32 PSUM tiles of one [P, ncol] accumulator, ncol up to 2,048, one bank (512 columns) each, as a tuple
        of one to four (switch 9e's _ptiles, verbatim)."""
        w0 = ncol
        if w0 > _CH:
            w0 = _CH
        p0 = nl.ndarray((_P, w0), dtype=f32, buffer=nl.psum)
        if ncol <= _CH:
            return (p0,)
        w1 = ncol - _CH
        if w1 > _CH:
            w1 = _CH
        p1 = nl.ndarray((_P, w1), dtype=f32, buffer=nl.psum)
        if ncol <= 2 * _CH:
            return (p0, p1)
        w2 = ncol - 2 * _CH
        if w2 > _CH:
            w2 = _CH
        p2 = nl.ndarray((_P, w2), dtype=f32, buffer=nl.psum)
        if ncol <= 3 * _CH:
            return (p0, p1, p2)
        w3 = ncol - 3 * _CH
        p3 = nl.ndarray((_P, w3), dtype=f32, buffer=nl.psum)
        return (p0, p1, p2, p3)

    def _evac_desc(dst2, pss, ntok, sc_sb, col):
        """bf16 of the accumulator chunks times the per-partition descale sc[:, col] into dst2 [P, ntok] (VectorE: the
        attention kernel's normalise, a vector operand0 reading PSUM into bf16; switch 9c's _evac_desc, verbatim)."""
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

    def _dw8_body(out, g2, a2, sc, fg, fa, forms):
        """out [O, D] (allocated by the kernel) = the module docstring's product, this program's output blocks."""
        M = g2.shape[0]
        O = g2.shape[1]
        D = a2.shape[1]
        dt = g2.dtype
        f32 = nl.float32
        MT = M // _P
        MTD = M // _KS
        NOB = O // _P
        nci = _ceil_div(D, _CH)
        sc_sb = _load_sc(sc, NDW)
        pid, nprg = _shard()
        b0, b1 = _span(NOB, pid, nprg)
        # the a-side: every token of the rank, cast once into a8, AB token tiles a row load
        if fa == 1:
            a8 = nl.ndarray((_P, MT, D), dtype=_F8E5, buffer=nl.sbuf)
        else:
            a8 = nl.ndarray((_P, MT, D), dtype=_F8E4, buffer=nl.sbuf)
        nab = _ceil_div(MT, AB)
        for ab in range(nab):
            t0 = ab * AB
            t1 = t0 + AB
            if t1 > MT:
                t1 = MT
            nt = t1 - t0
            at = nl.ndarray((_P, nt, D), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=at, src=a2.ap([[D, _P], [_P * D, nt], [1, D]], offset=t0 * _P * D))
            _cast_fmt(a8[:, t0:t1, :], at, sc_sb, DW_S_A, fa, forms)
        # this program's output blocks: the g-side block cast, M / 256 double-row matmuls a chunk, the descaled rows
        for ob in range(b0, b1):
            o0 = ob * _P
            o1 = o0 + _P
            gt = _load_w_chunk(g2, O, MT, ob, dt)
            if fg == 1:
                g8 = nl.ndarray((_P, MT, _P), dtype=_F8E5, buffer=nl.sbuf)
            else:
                g8 = nl.ndarray((_P, MT, _P), dtype=_F8E4, buffer=nl.sbuf)
            _cast_fmt(g8, gt, sc_sb, DW_S_G, fg, forms)
            psw = _ptiles(D, f32)
            for kd in range(MTD):
                j0 = kd * 2
                j1 = j0 + 2
                for ci in range(nci):
                    c0 = ci * _CH
                    c1 = c0 + _CH
                    if c1 > D:
                        c1 = D
                    nisa.nc_matmul(dst=psw[ci], stationary=g8[:, j0:j1, :], moving=a8[:, j0:j1, c0:c1],
                                   accumulate=(kd > 0), perf_mode=_DR)
            o_sb = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
            _evac_desc(o_sb, psw, D, sc_sb, DW_DQ)
            nisa.dma_copy(dst=out[o0:o1, :], src=o_sb)

    @nki.jit
    def _dw8_kernel(g2, a2, sc, fg, fa, forms):
        out = nl.ndarray((g2.shape[1], a2.shape[1]), dtype=g2.dtype, buffer=nl.shared_hbm)
        _dw8_body(out, g2, a2, sc, fg, fa, forms)
        return out

    def _dw8w_body(out, g2, a2, sc, fg, fa, forms, gw):
        """Lane L2: out [O, D] = the module docstring's product for D up to any multiple of gw: this program's column
        groups (gw wide), each one a-side cast and every output block run as _dw8_body runs it."""
        M = g2.shape[0]
        O = g2.shape[1]
        D = a2.shape[1]
        dt = g2.dtype
        f32 = nl.float32
        MT = M // _P
        MTD = M // _KS
        NOB = O // _P
        NG = D // gw
        nci = _ceil_div(gw, _CH)
        nab = _ceil_div(MT, AB)
        sc_sb = _load_sc(sc, NDW)
        pid, nprg = _shard()
        q0, q1 = _span(NG, pid, nprg)
        for gi in range(q0, q1):
            gc0 = gi * gw
            gc1 = gc0 + gw
            # the a-side group: every token of the rank, columns gc0 to gc1, cast once into a8
            if fa == 1:
                a8 = nl.ndarray((_P, MT, gw), dtype=_F8E5, buffer=nl.sbuf)
            else:
                a8 = nl.ndarray((_P, MT, gw), dtype=_F8E4, buffer=nl.sbuf)
            for ab in range(nab):
                t0 = ab * AB
                t1 = t0 + AB
                if t1 > MT:
                    t1 = MT
                nt = t1 - t0
                at = nl.ndarray((_P, nt, gw), dtype=dt, buffer=nl.sbuf)
                aoff = t0 * _P * D + gc0
                nisa.dma_copy(dst=at, src=a2.ap([[D, _P], [_P * D, nt], [1, gw]], offset=aoff))
                _cast_fmt(a8[:, t0:t1, :], at, sc_sb, DW_S_A, fa, forms)
            # every output block: _dw8_body's g-side block cast, matmuls and evacuation, the rows at columns gc0:gc1
            for ob in range(NOB):
                o0 = ob * _P
                o1 = o0 + _P
                gt = _load_w_chunk(g2, O, MT, ob, dt)
                if fg == 1:
                    g8 = nl.ndarray((_P, MT, _P), dtype=_F8E5, buffer=nl.sbuf)
                else:
                    g8 = nl.ndarray((_P, MT, _P), dtype=_F8E4, buffer=nl.sbuf)
                _cast_fmt(g8, gt, sc_sb, DW_S_G, fg, forms)
                psw = _ptiles(gw, f32)
                for kd in range(MTD):
                    j0 = kd * 2
                    j1 = j0 + 2
                    for ci in range(nci):
                        c0 = ci * _CH
                        c1 = c0 + _CH
                        if c1 > gw:
                            c1 = gw
                        nisa.nc_matmul(dst=psw[ci], stationary=g8[:, j0:j1, :], moving=a8[:, j0:j1, c0:c1],
                                       accumulate=(kd > 0), perf_mode=_DR)
                o_sb = nl.ndarray((_P, gw), dtype=dt, buffer=nl.sbuf)
                _evac_desc(o_sb, psw, gw, sc_sb, DW_DQ)
                nisa.dma_copy(dst=out[o0:o1, gc0:gc1], src=o_sb)

    @nki.jit
    def _dw8w_kernel(g2, a2, sc, fg, fa, forms, gw):
        out = nl.ndarray((g2.shape[1], a2.shape[1]), dtype=g2.dtype, buffer=nl.shared_hbm)
        _dw8w_body(out, g2, a2, sc, fg, fa, forms, gw)
        return out

    _HAS_NKI = True
except Exception as exc:  # nki unavailable (CPU dev box): the emulation below still works
    _NKI_ERROR = f"{type(exc).__name__}: {exc}"


def _launch(kernel):
    """kernel[2] under an LNC=2 runtime (each of the two programs takes half the output blocks), plain call
    otherwise."""
    return kernel[_LNC] if _LNC == 2 else kernel


def plan_kib(m: int, d: int) -> float:
    """SBUF a partition (KiB) the kernel holds at M m and width d: the resident a8, the a-side row block (bf16,
    double-buffered), the g-side block (bf16 and fp8, double-buffered), the output rows (double-buffered), the
    scales."""
    mt = m // _P
    a8 = mt * d
    at = 2 * AB * d * 2
    gt = 2 * mt * _P * 3
    o = 2 * d * 2
    return (a8 + at + gt + o + NDW * 4) / 1024.0


def shapes_apply(m: int, o: int, d: int) -> bool:
    """The shapes the kernel takes: M a multiple of 256 (the caller pads), O and D multiples of 128, D at most 2,048,
    and a plan within PLAN_KIB."""
    return (m > 0 and m % _KS == 0 and o > 0 and o % _P == 0 and 0 < d <= D_MAX and d % _P == 0
            and plan_kib(m, d) <= PLAN_KIB)


def group_width(d: int) -> int:
    """Lane L2: the wide form's a-side group width for D = d: the largest multiple of 128 up to D_MAX that divides d
    into an even number of groups (the two programs of an LNC 2 rank take half each), else the largest that divides d
    at all; 0 when d is not a positive multiple of 128. 1,536 at d 6,144 (four groups)."""
    if d <= 0 or d % _P != 0:
        return 0
    first = 0
    for gw in range(D_MAX, 0, -_P):
        if d % gw == 0:
            if (d // gw) % 2 == 0:
                return gw
            if first == 0:
                first = gw
    return first


def shapes_apply_w(m: int, o: int, d: int) -> bool:
    """Lane L2: the shapes _dw8w_kernel takes: M a multiple of 256 (the caller pads), O a multiple of 128, D a multiple
    of 128 with a group width, and the plan at that width within PLAN_KIB."""
    gw = group_width(d)
    return (m > 0 and m % _KS == 0 and o > 0 and o % _P == 0 and gw > 0 and d % gw == 0
            and plan_kib(m, gw) <= PLAN_KIB)


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


def pow2_inv(s: torch.Tensor) -> torch.Tensor:
    """1 / s exactly for a power of two s = 2^k, k in E_INV, with no reciprocal instruction: _pow2_le's search for the
    largest 2^e with s x 2^e <= 1 (every product a power of two inside fp32's normal range)."""
    return _pow2_le(s.float(), 1.0, -E_INV[1], -E_INV[0])[0]


def dw_row(s_g: torch.Tensor, i_g: torch.Tensor, s_a: torch.Tensor, i_a: torch.Tensor) -> torch.Tensor:
    """The call's scale row [NDW] fp32: s_g, s_a and the descale i_g x i_a (exact: powers of two)."""
    return torch.stack([s_g.float().reshape(()), s_a.float().reshape(()), (i_g.float() * i_a.float()).reshape(())])


def sc_tile(row: torch.Tensor) -> torch.Tensor:
    """[128, len(row)] fp32: the row down every partition (the kernel's scale input, read by one 2-D DMA)."""
    n = row.numel()
    return row.float().reshape(1, n).expand(_P, n).contiguous()


# ----------------------------------------------------------------------------- torch emulation (CPU and proofs)

def q8(t: torch.Tensor, fmt: int) -> torch.Tensor:
    """The fp8 rounding (round to nearest even) of t's fp32 values, back in fp32: e4m3 (fmt 0 or 2: torch's
    float8_e4m3fn, whose values and rounding inside +-240 are the chip's legacy e4m3's, the only range the kernel casts)
    or e5m2 (fmt 1). Format 2 clamps at 240 first (the kernel's minimum). Switch 9c's q8 with the clamp."""
    t = t.float()
    if fmt == FMT_E4M3C:
        t = torch.clamp_max(t, E4M3_MAX)
    return t.to(torch.float8_e5m2 if fmt == FMT_E5M2 else torch.float8_e4m3fn).float()


def emulate_dw(g2: torch.Tensor, a2: torch.Tensor, s_g, s_a, dq, fg: int, fa: int) -> torch.Tensor:
    """_dw8_kernel's arithmetic in torch: out [O, D] = bf16((fp8_fg(g2 s_g)^T @ fp8_fa(a2 s_a)) dq); fp32 products and
    accumulation (torch's summation order)."""
    g8 = q8(g2.float() * s_g, fg)
    a8 = q8(a2.float() * s_a, fa)
    return ((g8.t() @ a8) * dq).to(g2.dtype)


def reference_dw(g2: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    """The bf16 weight gradient the switch replaces (the kit's expression, g^T a in the activations' dtype)."""
    return g2.t().mm(a2)
