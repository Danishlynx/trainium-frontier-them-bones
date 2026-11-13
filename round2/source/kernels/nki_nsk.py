"""nki_nsk.py: switch 63 (R2_NSK, lane NSK): the experts' Polar Express (Newton-Schulz) iteration of switch 55's local
Muon step as one NKI kernel a step, both expert stacks of a rank in one call (work/p2/nsk/NOTES.md).

  _nsk_kernel(ga, gb, sc) -> ya, yb, va, vb
      ga [K, M, N] fp32: the tall stack (the experts' w_fc, [5, 3072, 1536] a rank), after muon_math_body's two lerps
      gb [K2, N, M] fp32: the wide stack (w_proj, [5, 1536, 3072]), the same
      sc [128, K + K2] fp32: each matrix's 1 / d, d = bf16(bf16(||bf16(g)||) * 1.01 + 1e-6) (torch's own
         denominator and its fp32 reciprocal, computed outside the kernel), one value down every partition, the tall
         stack's first (the device parser has no divide operator: the kernel multiplies by 1 / d)
      ya [K, M, N], yb [K2, N, M] bf16: the five iterations' output X5 (muon_math_body's x after its loop)
      va [K, M], vb [K2, M] fp32: the NorMuon row statistic's sum over the N-long side of x^2, one value for each
         index of the M-long side (muon_math_body's v_mean times N, red_dim -1 of the tall, -2 of the wide; the
         caller divides by N, as torch's mean does)

Per matrix, in SBUF (one physical core, no HBM between the iterations): X0 = bf16(bf16(g) x (1 / d)), then for each of
the five (a, b, c) of the kit's POLAR_EXPRESS_COEFFS, with X held tall [M, N] (M >= N; the wide stack is turned tall
by TensorE transposes on the way in and back on the way out):

  G  = X^T X                 [N, N]: 128 x 128 stationary slices of X, 512-column moving chunks, fp32 PSUM over the
                             M / 128 row tiles; only the 512-column blocks on or right of the diagonal are multiplied
                             (G is symmetric) and the blocks left of it are TensorE transposes of their mirror images
  B  = c G^2 + b G           G^2 = G^T G (= G G), the same upper-block scheme over the N / 128 row tiles of G; the
                             evacuation is one VectorE (PSUM x c) + (b x G) per chunk, b x G by ScalarE in fp32
  X' = a X + X B             X^T slices by TensorE transposes of each 128-row tile (one pass of X an iteration), B as
                             the moving operand; the evacuation (PSUM x 1) + (a x X) on VectorE, a x X by ScalarE

Layout and plan: X [128, M / 128, N] bf16 (72 KiB a partition at the real shape), G and B [128, N / 128, N] bf16
(36 KiB each), stages and work tiles 32 KiB: plan_kib's 176.25 KiB (180,480 B) of the 212,984 B a partition the
gen3 backend allocates (nki/language/_tile_size.py; neuronx-cc allocated the real shape, devcompile 10:11 IST Oct 23).
PSUM: one to three one-bank [128, 512] fp32 accumulators a row block (N / 512 at most 3), double-buffered, and one
1 KiB bf16 transpose tile. Under LNC 2 (kernel[2]) program 0 takes the tall stack and program 1 the wide one (equal
TensorE work but the wide stack's 2 x 288 transposes a matrix); without a launch grid one program takes both.

Rounding points (emulate_nsk below has the same ones): bf16(g), then the product with fp32(1 / d) rounded once to
bf16 (torch's x / d rounds the quotient: the two differ by one bf16 step on the rare elements whose fp32 value lies
within an fp32 step or two of a bf16 midpoint, about 6e-5 of them); G rounded once to bf16 from its fp32
accumulator; B = bf16(fp32(fp32(G^2) x c) + fp32(b x G)), one bf16 rounding where the kit's expression rounds G^2,
c G^2, b G and their sum (four); X' = bf16(fp32(X B) + fp32(a x X)), one bf16 rounding where the kit rounds a X, X B
and their sum (three); the row sums of squares in fp32 from the bf16 X5; G^2 and X B read the bf16 G and B
(as the kit's matmuls read its bf16 operands); fp32 accumulation everywhere, in TensorE's order (128 rows a tile into
PSUM, tile after tile). The kernel's iteration is therefore the kit's within the bf16 band, not bitwise; the proof's
tolerance is the record's for the dense Muon (NOTES section 3).

Shapes: M and N multiples of 512, N at most 1,536 (three PSUM banks a row block), M >= N, K >= 0 (a stack of 0
matrices is skipped); sc has K + K2 columns. shapes_apply checks them and the SBUF plan.

Written for the device-side NKI parser frontend under switch 9c's rules (kernels/nki_mlp_fused8.py's docstring):
every loop is `for name in range(...)` with one plain target; `if` only on trace-time integers (program_id
included); tuples only returned and unpacked in assignments (the PSUM-tuple of _ps_chunks indexed by a loop
variable, as switch 9k's psw[ci]); no list, comprehension, conditional expression, `while`, `break`, `continue`,
dict, lambda or nested def in a traced body; every slice bound a plain name or a literal; every PSUM tile 2-D and one
bank at most; a tile or view subscripted once. Instruction forms: nc_matmul bf16 into fp32 PSUM with accumulate,
nc_transpose of a [128, 128] bf16 slice into a bf16 PSUM tile, tensor_copy on VectorE out of PSUM, activation copy
(with and without a float scale) out of PSUM or SBUF, scalar_tensor_tensor with PSUM data and an SBUF operand1,
dma_copy with .ap() patterns, activation_reduce with nl.square (lane L1's chip-run prologue, nki_attn_nf8.py's
_nf_norm_tile): each has run on the chip in switches 8, 9c, 9d, 9e, 9j, 9k, 9m or lane OS1A2's pair. New in this
combination are the operands (integer-indexed slices of 3-D SBUF tiles into nc_matmul, nc_transpose and
activation_reduce, a column slice of reduce_res, the stt writing the bf16 tile its fp32 operand1 was derived from,
the statistic's 4-byte strided store, 3,072 elements a matrix), which the devcompile gate compiles and the chain's
NSK_dev line runs on a chip against the emulation before any training line. (The first form divided by d on
VectorE: the device parser refused `op0=nl.divide`, devcompile 10:03 IST Oct 23; the reciprocal is now torch's and
the kernel multiplies. Review finding 7's contiguous [128, M / 128] statistic, permuted in torch, was tried and
withdrawn: the stride audit found the permute's 3-level copy in the compiled pair graph, NCC_IBCG901's class,
delta review D1.)
"""

import os

import torch

_P = 128
_CH = 512
_GRP = 4                    # 128-column blocks a 512-column chunk
_NS = 5                     # the kit's ns_steps
_env = os.environ.get
_LNC = int(_env("NEURON_LOGICAL_NC_CONFIG", "1"))
PLAN_KIB = 200.0            # SBUF a partition the plan may fill: the gen3 backend's 212,984 B (207.99 KiB) less a
#                             margin; the real shape's plan is 176.25 KiB, which neuronx-cc allocated (devcompile).
#                             (The first form copied switch 9d's 168, a co-residency limit of another kernel, and so
#                             refused the real shape: review finding 1, REVIEW.md.)
N_MAX = 1536                # three PSUM banks of 512 fp32 columns a row block
# The kit's POLAR_EXPRESS_COEFFS (train.py), the five (a, b, c): _coef returns them by iteration, train.py checks them.
NSK_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)

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

    def _coef(it):
        """The kit's (a, b, c) of iteration it (POLAR_EXPRESS_COEFFS[it]), by an if-chain on the trace-time index."""
        a = 2.3465413258596377
        b = -1.7097828382687081
        c = 0.42323551169305323
        if it == 0:
            a = 8.156554524902461
            b = -22.48329292557795
            c = 15.878769915207462
        if it == 1:
            a = 4.042929935166739
            b = -2.808917465908714
            c = 0.5000178451051316
        if it == 2:
            a = 3.8916678022926607
            b = -2.772484153217685
            c = 0.5060648178503393
        if it == 3:
            a = 3.285753657755655
            b = -2.3681294933425376
            c = 0.46449024233003106
        return a, b, c

    def _ps_chunks(nch, f32):
        """One to three one-bank fp32 PSUM accumulators [128, 512] (a row block's 512-column chunks), as a tuple."""
        p0 = nl.ndarray((_P, _CH), dtype=f32, buffer=nl.psum)
        if nch <= 1:
            return (p0,)
        p1 = nl.ndarray((_P, _CH), dtype=f32, buffer=nl.psum)
        if nch <= 2:
            return (p0, p1)
        p2 = nl.ndarray((_P, _CH), dtype=f32, buffer=nl.psum)
        return (p0, p1, p2)

    def _sym_fill(T, NT, dt):
        """T [128, NT, N] holds a symmetric [N, N] matrix whose 512-column blocks on and right of the diagonal are set:
        every 128-block left of the diagonal chunk, T[rows ib, cols jb], is written as the TensorE transpose of
        T[rows jb, cols ib] (four a bf16 PSUM tile, one VectorE copy out)."""
        for ib in range(NT):
            gi = ib // _GRP
            nlow = gi * _GRP
            s0 = ib * _P
            s1 = s0 + _P
            for g0 in range(0, nlow, _GRP):
                pst = nl.ndarray((_P, _GRP * _P), dtype=dt, buffer=nl.psum)
                for gg in range(_GRP):
                    jb = g0 + gg
                    q0 = gg * _P
                    q1 = q0 + _P
                    nisa.nc_transpose(dst=pst[:, q0:q1], data=T[:, jb, s0:s1])
                d0 = g0 * _P
                d1 = d0 + _GRP * _P
                nisa.tensor_copy(dst=T[:, ib, d0:d1], src=pst, engine=_VEC)

    def _gram(G, X, MT, NT, NG, f32):
        """G = X^T X on and right of the diagonal chunk: row block ib (128 columns of X) against the 512-column chunks
        from its own onwards, accumulated over the MT row tiles of X, each stationary slice serving its chunks in turn;
        evacuated to bf16 by ScalarE."""
        for ib in range(NT):
            gi = ib // _GRP
            nch = NG - gi
            i0 = ib * _P
            i1 = i0 + _P
            pss = _ps_chunks(nch, f32)
            for t in range(MT):
                for ci in range(nch):
                    c0 = (gi + ci) * _CH
                    c1 = c0 + _CH
                    nisa.nc_matmul(dst=pss[ci], stationary=X[:, t, i0:i1], moving=X[:, t, c0:c1], accumulate=(t > 0))
            for ci in range(nch):
                c0 = (gi + ci) * _CH
                c1 = c0 + _CH
                nisa.activation(dst=G[:, ib, c0:c1], op=nl.copy, data=pss[ci])

    def _square(B, G, b, c, NT, NG, f32):
        """B = c G^2 + b G on and right of the diagonal chunk (G^2 = G^T G, G symmetric), the last row blocks first
        (they read no transposed block); per chunk ScalarE writes b x G in fp32 and VectorE (PSUM x c) + that into
        bf16."""
        for r in range(NT):
            ib = NT - 1 - r
            gi = ib // _GRP
            nch = NG - gi
            i0 = ib * _P
            i1 = i0 + _P
            pss = _ps_chunks(nch, f32)
            for kt in range(NT):
                for ci in range(nch):
                    c0 = (gi + ci) * _CH
                    c1 = c0 + _CH
                    nisa.nc_matmul(dst=pss[ci], stationary=G[:, kt, i0:i1], moving=G[:, kt, c0:c1],
                                   accumulate=(kt > 0))
            for ci in range(nch):
                c0 = (gi + ci) * _CH
                c1 = c0 + _CH
                bg = nl.ndarray((_P, _CH), dtype=f32, buffer=nl.sbuf)
                nisa.activation(dst=bg, op=nl.copy, data=G[:, ib, c0:c1], scale=b)
                nisa.scalar_tensor_tensor(dst=B[:, ib, c0:c1], data=pss[ci], op0=nl.multiply, operand0=c,
                                          op1=nl.add, operand1=bg)

    def _update(X, B, a, MT, NT, NG, f32, dt):
        """X' = a X + X B, row tile by row tile in place: the tile's NT slices transposed on TensorE (four a bf16 PSUM
        tile, ScalarE copies them out), then the products with B moving; per chunk ScalarE writes a x X in fp32 and
        VectorE (PSUM x 1) + that into the tile."""
        for t in range(MT):
            xt = nl.ndarray((_P, NT * _P), dtype=dt, buffer=nl.sbuf)
            for g0 in range(0, NT, _GRP):
                pst = nl.ndarray((_P, _GRP * _P), dtype=dt, buffer=nl.psum)
                for gg in range(_GRP):
                    kt = g0 + gg
                    s0 = kt * _P
                    s1 = s0 + _P
                    q0 = gg * _P
                    q1 = q0 + _P
                    nisa.nc_transpose(dst=pst[:, q0:q1], data=X[:, t, s0:s1])
                d0 = g0 * _P
                d1 = d0 + _GRP * _P
                nisa.activation(dst=xt[:, d0:d1], op=nl.copy, data=pst)
            pss = _ps_chunks(NG, f32)
            for kt in range(NT):
                s0 = kt * _P
                s1 = s0 + _P
                for ci in range(NG):
                    c0 = ci * _CH
                    c1 = c0 + _CH
                    nisa.nc_matmul(dst=pss[ci], stationary=xt[:, s0:s1], moving=B[:, kt, c0:c1],
                                   accumulate=(kt > 0))
            for ci in range(NG):
                c0 = ci * _CH
                c1 = c0 + _CH
                ax = nl.ndarray((_P, _CH), dtype=f32, buffer=nl.sbuf)
                nisa.activation(dst=ax, op=nl.copy, data=X[:, t, c0:c1], scale=a)
                nisa.scalar_tensor_tensor(dst=X[:, t, c0:c1], data=pss[ci], op0=nl.multiply, operand0=1.0,
                                          op1=nl.add, operand1=ax)

    def _ns5(X, G, B, MT, NT, NG, f32, dt):
        """The kit's five Polar Express iterations on the resident tall X."""
        for it in range(_NS):
            a, b, c = _coef(it)
            _gram(G, X, MT, NT, NG, f32)
            _sym_fill(G, NT, dt)
            _square(B, G, b, c, NT, NG, f32)
            _sym_fill(B, NT, dt)
            _update(X, B, a, MT, NT, NG, f32, dt)

    def _load_tall(X, ga, k, M, N, MT, sc_sb, col, f32, dt):
        """X0 of tall matrix k: each 128-row tile of ga[k] by one DMA (fp32), rounded to bf16 by ScalarE, times the
        matrix's 1 / d (column col of sc) on VectorE into X's tile."""
        base = k * M * N
        q0 = col
        q1 = q0 + 1
        for t in range(MT):
            gt = nl.ndarray((_P, N), dtype=f32, buffer=nl.sbuf)
            off = base + t * _P * N
            nisa.dma_copy(dst=gt, src=ga.ap([[N, _P], [1, N]], offset=off))
            xb = nl.ndarray((_P, N), dtype=dt, buffer=nl.sbuf)
            nisa.activation(dst=xb, op=nl.copy, data=gt)
            nisa.tensor_scalar(dst=X[:, t, :], data=xb, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)

    def _load_wide(X, gb, k, M, N, NT, sc_sb, col, f32, dt):
        """X0 of wide matrix k turned tall: each [128 rows, 512 columns] piece of gb[k] by one DMA, rounded and scaled
        as in _load_tall, its four [128, 128] blocks transposed on TensorE into X[:, m tile, the rows' 128 columns]."""
        base = k * N * M
        nc = M // _CH
        q0 = col
        q1 = q0 + 1
        for nt in range(NT):
            n0 = nt * _P
            n1 = n0 + _P
            for cc in range(nc):
                gw = nl.ndarray((_P, _CH), dtype=f32, buffer=nl.sbuf)
                off = base + nt * _P * M + cc * _CH
                nisa.dma_copy(dst=gw, src=gb.ap([[M, _P], [1, _CH]], offset=off))
                xb = nl.ndarray((_P, _CH), dtype=dt, buffer=nl.sbuf)
                nisa.activation(dst=xb, op=nl.copy, data=gw)
                xd = nl.ndarray((_P, _CH), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=xd, data=xb, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)
                pst = nl.ndarray((_P, _GRP * _P), dtype=dt, buffer=nl.psum)
                for gg in range(_GRP):
                    s0 = gg * _P
                    s1 = s0 + _P
                    nisa.nc_transpose(dst=pst[:, s0:s1], data=xd[:, s0:s1])
                for gg in range(_GRP):
                    mt = cc * _GRP + gg
                    s0 = gg * _P
                    s1 = s0 + _P
                    nisa.tensor_copy(dst=X[:, mt, n0:n1], src=pst[:, s0:s1], engine=_VEC)

    def _row_stats(vm, X, MT, N, f32):
        """vm[p, t] = the sum over the N columns of X[t 128 + p, :]^2 (fp32 squares summed by ScalarE; the caller
        divides by N)."""
        for t in range(MT):
            sq = nl.ndarray((_P, N), dtype=f32, buffer=nl.sbuf)
            t1 = t + 1
            nisa.activation_reduce(dst=sq, op=nl.square, data=X[:, t, :], reduce_op=nl.add, reduce_res=vm[:, t:t1])

    def _store_tall(ya, va, X, vm, k, M, N, MT):
        """ya[k] = X (one three-level DMA of the resident rows) and va[k, t 128 + p] = vm[p, t] (one two-level DMA
        of 4-byte elements)."""
        nisa.dma_copy(dst=ya.ap([[N, _P], [_P * N, MT], [1, N]], offset=k * M * N), src=X)
        nisa.dma_copy(dst=va.ap([[1, _P], [_P, MT]], offset=k * M), src=vm)

    def _store_wide(yb, vb, X, vm, k, M, N, MT, NT, dt):
        """yb[k] = X^T: for each 128-row block of the wide matrix, its MT [128, 128] blocks transposed back on TensorE
        (four a bf16 PSUM tile, one VectorE copy out each) into one [128, M] row stage, then one DMA; vb[k] as
        _store_tall stores va[k]."""
        base = k * N * M
        for nt in range(NT):
            n0 = nt * _P
            n1 = n0 + _P
            ys = nl.ndarray((_P, M), dtype=dt, buffer=nl.sbuf)
            for g0 in range(0, MT, _GRP):
                pst = nl.ndarray((_P, _GRP * _P), dtype=dt, buffer=nl.psum)
                for gg in range(_GRP):
                    mt = g0 + gg
                    q0 = gg * _P
                    q1 = q0 + _P
                    nisa.nc_transpose(dst=pst[:, q0:q1], data=X[:, mt, n0:n1])
                d0 = g0 * _P
                d1 = d0 + _GRP * _P
                nisa.tensor_copy(dst=ys[:, d0:d1], src=pst, engine=_VEC)
            off = base + nt * _P * M
            nisa.dma_copy(dst=yb.ap([[M, _P], [1, M]], offset=off), src=ys)
        nisa.dma_copy(dst=vb.ap([[1, _P], [_P, MT]], offset=k * M), src=vm)

    def _nsk_body(ya, yb, va, vb, ga, gb, sc):
        """The module docstring's computation: program 0 the tall stack, program 1 the wide one (one program: both)."""
        K = ga.shape[0]
        M = ga.shape[1]
        N = ga.shape[2]
        K2 = gb.shape[0]
        MT = M // _P
        NT = N // _P
        NG = N // _CH
        dt = nl.bfloat16
        f32 = nl.float32
        nsc = sc.shape[1]
        sc_sb = nl.ndarray((_P, nsc), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sc_sb, src=sc)
        pid, nprg = _shard()
        do_a = 1
        do_b = 1
        if nprg > 1:
            if pid == 0:
                do_b = 0
            else:
                do_a = 0
        X = nl.ndarray((_P, MT, N), dtype=dt, buffer=nl.sbuf)
        G = nl.ndarray((_P, NT, N), dtype=dt, buffer=nl.sbuf)
        B = nl.ndarray((_P, NT, N), dtype=dt, buffer=nl.sbuf)
        vm = nl.ndarray((_P, MT), dtype=f32, buffer=nl.sbuf)
        if do_a == 1:
            for k in range(K):
                _load_tall(X, ga, k, M, N, MT, sc_sb, k, f32, dt)
                _ns5(X, G, B, MT, NT, NG, f32, dt)
                _row_stats(vm, X, MT, N, f32)
                _store_tall(ya, va, X, vm, k, M, N, MT)
        if do_b == 1:
            for k in range(K2):
                col = K + k
                _load_wide(X, gb, k, M, N, NT, sc_sb, col, f32, dt)
                _ns5(X, G, B, MT, NT, NG, f32, dt)
                _row_stats(vm, X, MT, N, f32)
                _store_wide(yb, vb, X, vm, k, M, N, MT, NT, dt)

    @nki.jit
    def _nsk_kernel(ga, gb, sc):
        ya = nl.ndarray((ga.shape[0], ga.shape[1], ga.shape[2]), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        yb = nl.ndarray((gb.shape[0], gb.shape[1], gb.shape[2]), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        va = nl.ndarray((ga.shape[0], ga.shape[1]), dtype=nl.float32, buffer=nl.shared_hbm)
        vb = nl.ndarray((gb.shape[0], gb.shape[2]), dtype=nl.float32, buffer=nl.shared_hbm)
        _nsk_body(ya, yb, va, vb, ga, gb, sc)
        return ya, yb, va, vb

    _HAS_NKI = True
except Exception as exc:  # nki unavailable (CPU dev box): the emulation below still works
    _NKI_ERROR = f"{type(exc).__name__}: {exc}"


def _launch(kernel):
    """kernel[2] under an LNC=2 runtime (program 0 the tall stack, program 1 the wide one), plain call otherwise."""
    return kernel[_LNC] if _LNC == 2 else kernel


def plan_kib(m: int, n: int) -> float:
    """SBUF a partition (KiB) the kernel holds at a tall [m, n]: X, G and B resident; the largest phase's stages,
    double-buffered (the tall load's fp32 and bf16 rows, or the row statistic's fp32 squares, or the wide store's
    row stage), the transposed tile, the two fp32 evacuation operands, the statistic and the scales."""
    mt, nt = m // _P, n // _P
    resident = mt * n * 2 + 2 * nt * n * 2
    stage = 2 * max(n * 4 + n * 2, n * 4, m * 2)
    work = 2 * (nt * _P * 2) + 2 * 2 * (_CH * 4)
    small = 2 * mt * 4 + 64
    return (resident + stage + work + small) / 1024.0


def shapes_apply(m: int, n: int) -> bool:
    """The shapes the kernel takes: M and N multiples of 512, N at most 1,536, M >= N, and a plan within PLAN_KIB."""
    return (m > 0 and n > 0 and m % _CH == 0 and n % _CH == 0 and n <= N_MAX and m >= n
            and plan_kib(m, n) <= PLAN_KIB)


def sc_tile(d_a: torch.Tensor, d_b: torch.Tensor) -> torch.Tensor:
    """[128, K + K2] fp32: 1 / d of the tall then the wide stack (the denominators [K, 1, 1] and [K2, 1, 1], any
    float dtype, exact in fp32; the reciprocal in fp32), each down every partition (the kernel's sc, one 2-D DMA)."""
    row = 1.0 / torch.cat([d_a.reshape(-1).float(), d_b.reshape(-1).float()])
    return row.reshape(1, -1).expand(_P, row.numel()).contiguous()


# ----------------------------------------------------------------------------- torch emulation (CPU and proofs)

def _sym_fill_t(t: torch.Tensor) -> torch.Tensor:
    """The kernel's fill on a square [N, N] matrix: every 128-block left of the diagonal 512-column chunk replaced by
    the transpose of its mirror image (the blocks on and right of the diagonal chunk kept)."""
    n = t.shape[-1]
    out = t.clone()
    for gi in range(n // _CH):
        r0, r1 = gi * _CH, (gi + 1) * _CH
        out[r0:r1, 0:r0] = t[0:r0, r0:r1].transpose(0, 1)
    return out


def _ns_tall(x0: torch.Tensor, coeffs) -> torch.Tensor:
    """The kernel's five iterations on one tall bf16 matrix [M, N], in its rounding points, fp32 accumulation by row
    tiles of 128 in TensorE's order (each tile's product, then the running sum)."""
    x = x0
    m = x.shape[0]
    for a, b, c in coeffs:
        xf = x.float()
        g32 = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=x.device)
        for t0 in range(0, m, _P):
            xt = xf[t0:t0 + _P]
            g32 = g32 + xt.transpose(0, 1) @ xt
        g = _sym_fill_t(g32.to(torch.bfloat16))
        gf = g.float()
        g2 = torch.zeros_like(gf)
        for k0 in range(0, gf.shape[0], _P):
            gk = gf[k0:k0 + _P]
            g2 = g2 + gk.transpose(0, 1) @ gk
        bm = _sym_fill_t(((g2 * c) + (gf * b)).to(torch.bfloat16))
        bf = bm.float()
        xb = torch.zeros(m, x.shape[1], dtype=torch.float32, device=x.device)
        for k0 in range(0, bf.shape[0], _P):
            xb = xb + xf[:, k0:k0 + _P] @ bf[k0:k0 + _P]
        x = ((xb * 1.0) + (xf * a)).to(torch.bfloat16)
    return x


def emulate_nsk(ga: torch.Tensor, gb: torch.Tensor, d_a: torch.Tensor, d_b: torch.Tensor, coeffs=NSK_COEFFS):
    """_nsk_kernel's arithmetic in torch: (ya, yb, va, vb) from the two fp32 stacks and their denominators (any float
    dtype, [K, 1, 1] or [K]): X0 = bf16(bf16(g) x fp32(1 / d)), the iterations, X5 and its row sums of squares."""
    ra = 1.0 / d_a.reshape(-1).float()
    rb = 1.0 / d_b.reshape(-1).float()
    ya, va, yb, vb = [], [], [], []
    for j in range(ga.shape[0]):
        x0 = (ga[j].to(torch.bfloat16).float() * ra[j]).to(torch.bfloat16)
        x5 = _ns_tall(x0, coeffs)
        ya.append(x5)
        va.append(x5.float().square().sum(dim=-1))
    for j in range(gb.shape[0]):
        x0 = (gb[j].to(torch.bfloat16).float() * rb[j]).to(torch.bfloat16).transpose(0, 1).contiguous()
        x5 = _ns_tall(x0, coeffs)
        yb.append(x5.transpose(0, 1))
        vb.append(x5.float().square().sum(dim=-1))
    return torch.stack(ya), torch.stack(yb), torch.stack(va), torch.stack(vb)


def reference_ns(x: torch.Tensor, coeffs=NSK_COEFFS) -> torch.Tensor:
    """muon_math_body's loop on a bf16 stack [K, r, c] (the current iteration, the kit's expression)."""
    if x.shape[-2] > x.shape[-1]:
        for a, b, c in coeffs:
            xtx = x.mT @ x
            x = a * x + x @ (b * xtx + c * (xtx @ xtx))
    else:
        for a, b, c in coeffs:
            xxt = x @ x.mT
            x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
    return x
