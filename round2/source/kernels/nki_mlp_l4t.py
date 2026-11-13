"""nki_mlp_l4t.py: switch 9r (R2_MLP_TT, lane THR10, work/p2/thr10/NOTES.md): switch 9f's module with the pc kernels'
activation operands (x in the forward, dy in the backward) made into their fp8 tiles by natural row DMAs and TensorE
transposes instead of DMA strip transposes. PF7's captures show TensorE idle while each call's first block of x^T or
dy^T arrives through the strip transposes (93 to 94 us at every forward call's start, 77 to 123 us at the backward's,
often again for the second block): 1.5 ms an execution for 18 MLPs, about 12.8 ms a step. Under 9r (forms 0)
_load_T8_rows loads each 128-token tile by one natural DMA, transposes its slabs on TensorE into [P, ng, P] bf16 PSUM
groups of up to four (bit-accurate, switch 9j's form) and casts each group with ONE ScalarE activation (switch 9q's
form) with the scale the block cast uses (X_SCALE for x, the column SC_S_DY for dy): the same values, the same
instruction, element for element, so every output is 9f's bit for bit. forms 1 keeps 9f's lines. Everything after this
paragraph is nki_mlp_l4.py's text (md5 6f712d96) but for _load_T8_rows and the two pc kernels' loads
(tools/build_l4t.py).

nki_mlp_l4.py: lane THR5's kernels for switch 9c's fp8 fused MLP (kernels/nki_mlp_fused8.py 18369d38): the MLP
weight gradients inside the fp8 backward kernel (switch 9e, R2_MLP_L4=1) and the e4m3 weights cast once an optimizer
step instead of once a 1,024-token block (switch 9f, R2_MLP_PRECAST=1). Every helper this module shares with
18369d38 is that module's text byte for byte (tools/build_l4.py copies them; the battery's static group checks it);
the kernels and helpers marked "Switch 9e" or "Switch 9f" are new.

Switch 9e, L4: _mlp8_bwd_l4_kernel(dy2, rT, x2, w_fc, w_proj, w8, sc, tb, gq, pc, forms, u8) -> dx, dw_fc, dw_proj,
dp8
    The base backward (_mlp8_bwd_kernel, dw_mode fmajor) writes h^T and dpre^T [F, M] bf16 to HBM (2 x 50.3 MB a
    layer-micro-step at M 4,096, F 6,144) for two compiled bf16 GEMMs, dw_fc = dpre^T x and dw_proj = (h^T dy)^T,
    that read them back. Here the two GEMMs are in the kernel and h and dpre never leave SBUF. A weight gradient
    needs every token of the rank (a sum over M) while each of the two programs (the two physical cores of an LNC 2
    rank) holds half of the tokens, so the programs split the WEIGHTS instead of the tokens for this part: program 0
    computes dw_fc over all M tokens, program 1 dw_proj, and for each hidden chunk each sends the other, SBUF to SBUF
    (nisa.sendrecv), the token-major tile of its own tokens that the other needs (program 0 sends h, program 1 sends
    dpre). No partial sums: every weight-gradient element is one fp32 PSUM chain over the M tokens in token order,
    rounded once to bf16.
    Per program, in order:
      prologue  dy^T of the program's tokens in e5m2, resident (_load_T_block and _cast8 per block, as the base);
                every token of the rank token-major in bf16 (x2 for program 0, dy2 for program 1): btok [P, MT, d].
      hidden chunks f = 0 .. F/128 - 1:
                phase A of f over the program's tokens (_phase_a_chunk: the base's phase A instructions, block by
                block, on the same operands: dh^T in fp32 PSUM, dpre = bf16((acc x dq_dh2) x r), h = bf16(r^2), the
                e5m2 dpre to the scratch dp8 [F, M] e5m2 in HBM); the chunk's two token-major tiles by TensorE
                transposes (bit-accurate on NeuronCore-v3): own (program 0 dpre, program 1 h) and snd (the other);
                sendrecv(snd -> the other program's rcv); then chunk f - 1's weight-gradient rows (software
                pipelining by one chunk, so the exchange runs on the DMA queues behind a phase A):
                psw [P, d] fp32 (three banks at d 1,536) = sum over the rank's token tiles in order of
                tile^T [P tokens, P hidden] @ btok[:, tile, :] (stationary the token-major dpre or h tile, moving x
                or dy); bf16 on evacuation; program 0 writes dw_fc's rows, program 1 transposes its [P hidden, d] rows
                into a [d, gq x 128] staging tile and writes gq chunks of dw_proj's columns at once (1 KiB rows).
      phase B   after core_barrier on dp8 (the writes complete before the read-back), the program's e5m2 dpre read
                back [P, F/128, ntok] and dx computed as the base's phase B (the same instructions, the W_fc tile
                loaded once for all the program's tokens instead of once a block).
    Rounding: dx, dpre, h and the e5m2 dpre are the base kernel's values bit for bit (the same instructions on the
    same operands; only the tiling of the loops around them changes). dw_fc and dw_proj are bf16 of an fp32
    accumulation of the exact bf16 x bf16 products of the same bf16 dpre, h, x and dy that the base hands to the
    compiled GEMMs, so they equal the base path's weight gradients up to fp32 summation order: within one bf16 ulp
    plus 2 M 2^-24 sum|products| of each other (stated and checked in the simulator proof).
    Shapes: LNC 2 only (two programs), M a multiple of 256 (equal token spans), d and F multiples of 256, F a multiple
    of gq x 128, tb a multiple of 128 up to 1,024; SBUF a partition at M 4,096, d 1,536, F 6,144, tb 1,024, gq 4
    (plan_kib_l4): about 185 KiB in the main loop, 184 in phase B, of 208 usable.
    pc 1 reads the backward's two weight tiles pre-cast (switch 9f's layouts 2 and 3) instead of bf16 + _cast8.

Switch 9f, the pre-cast: _precast8_kernel(w_fc, w_proj, sc, need, forms, u8) -> w8 [4, F x d]
    The fused8 kernels stream both bf16 weights through every 1,024-token block and cast each tile in-kernel
    (THR3 2(8): 302 MB a layer-micro-step, 21.7 GB a rank-step, about 6.1 J a chip-step, plus the casts). Here one
    kernel call a layer, after each optimizer step's scale refresh (and in load_for_eval), writes the e4m3 bytes of
    every tile the kernels read, in the kernels' own tile order (layout 0 the forward's W_fc^T tiles, 1 its W_proj^T
    tiles, 2 the backward's natural W_proj tiles, 3 its natural W_fc tiles), by the kernels' own loaders and the
    kernels' own cast instruction (_cast8) with the same scale tile, so every byte equals the in-kernel cast's.
    _mlp8_fwd_pc_kernel and _mlp8_bwd_pc_kernel are _mlp8_fwd_kernel and _mlp8_bwd_kernel with each weight tile read
    as its pre-cast bytes (one plain DMA of half the bytes, no cast): their outputs are those kernels' outputs bit for
    bit. The pack crosses to torch as a uint8 buffer: under u8 1 (the device form, the only one train.py uses) the
    bytes are cast into uint8 tiles viewed as e4m3 and read into uint8 tiles viewed as e4m3 (switch 9's form, compiled
    and run on the device since candidate 33); under u8 0 the pack and the tiles are e4m3 (the form the host's
    simulator executes exactly: it writes fp8 through a uint8 view numerically, lane THR5's probe_view.py, while it
    reads through one exactly). The instructions are the same in both forms. Memory: 4 x F x d bytes a layer (37.7 MB
    at 1,536 x 6,144; 0.68 GB a rank at 18 layers).

Written for the device-side NKI parser frontend, under 18369d38's rules: every loop is `for name in range(...)` with
one plain target; `if` on trace-time integers only (program_id included); no list, comprehension, `while`,
`list(range(...))`, conditional expression, chained assignment or tuple-element loop target in a traced body; every
slice bound a plain name or a literal; every PSUM tile 2-D and at most one bank; a tile is subscripted once (a dtype
view and then one subscript is switch 9's proven form). The two instructions between programs (sendrecv, core_barrier)
passed the host's NKI 0.5.0 simulator and its device compiler (lane THR5's probe_xchg.py) and have not run on silicon.
"""

# Built by work/p2/thr5/tools/build_l4.py from work/p2/sched1/kernels/nki_mlp_fused8.py
# (md5 18369d3831905542d1e0362789613321): its imports, constants, nki import block, _launch and the 18 helpers
# listed in build_l4.py's SHARED, byte for byte.

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

    TT_GROUP = 4        # switch 9r: slabs a transpose PSUM group holds (4 x 128 bf16 = 1 KiB a partition)

    def _load_T8_rows(a8, a2, K, NKB, row0, nt, dt, scale):
        """Switch 9r: a8[p, s, t 128 + i] = fp8(a2[row0 + t 128 + i, s 128 + p] x scale) for the block's nt token
        tiles (a8 [P, NKB, nt 128], the layout _load_T_block's tiles have once cast): each tile's 128 rows by one
        natural DMA, its NKB slabs transposed on TensorE into a [P, ng, P] dt PSUM tile (groups of up to TT_GROUP
        slabs; a TensorE transpose is bit-accurate), each group cast by ONE activation (copy with the scale: the
        block cast's instruction) into a8[:, g0 : g0 + ng, t 128 : (t + 1) 128]. scale is a float or a [P, 1] column."""
        for t in range(nt):
            r0 = row0 + t * _P
            r1 = r0 + _P
            an = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=an, src=a2[r0:r1, 0:K])
            c0 = t * _P
            c1 = c0 + _P
            for g0 in range(0, NKB, TT_GROUP):
                ng = NKB - g0
                if ng > TT_GROUP:
                    ng = TT_GROUP
                psg = nl.ndarray((_P, ng, _P), dtype=dt, buffer=nl.psum)
                for gg in range(ng):
                    s0 = (g0 + gg) * _P
                    s1 = s0 + _P
                    nisa.nc_transpose(dst=psg[:, gg, :], data=an[:, s0:s1])
                g1 = g0 + ng
                nisa.activation(dst=a8[:, g0:g1, c0:c1], op=nl.copy, data=psg, scale=scale)

    def _store_rows(out, N, row0, nt, stage):
        """out[row0 + tt * 128 + p, :] = stage[p, tt, :] for the block's nt token tiles (one DMA)."""
        nisa.dma_copy(dst=out.ap([[N, _P], [_P * N, nt], [1, N]], offset=row0 * N), src=stage)

    def _store_cols(out, N, row0, nt, c, stage):
        """out[row0 + tt * 128 + p, c * 128:(c + 1) * 128] = stage[p, tt, :] (token-major columns of chunk c)."""
        nisa.dma_copy(dst=out.ap([[N, _P], [_P * N, nt], [1, _P]], offset=row0 * N + c * _P), src=stage)

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

    # ---- lane THR5, switches 9e (R2_MLP_L4) and 9f (R2_MLP_PRECAST): everything from here to the end of this block
    # is new; every function above it is nki_mlp_fused8.py 18369d38's, byte for byte (tools/build_l4.py copies them)

    def _load_w8_tile(w8, FD, t, c, nkb, u8):
        """Switch 9f: tile c of layout t of the pre-cast pack w8 [4, F x d] (_precast8_kernel's), [P, nkb, P] e4m3, by
        one DMA of its bytes into SBUF. u8 1 (the device form): w8 holds uint8, the tile is a uint8 tile viewed as
        e4m3 (switch 9's byte form, _f8_load_w_chunks'). u8 0 (the simulator's form): w8 holds e4m3, the tile too."""
        off = t * FD + c * _P * nkb * _P
        if u8 == 1:
            wu = nl.ndarray((_P, nkb, _P), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.dma_copy(dst=wu, src=w8.ap([[nkb * _P, _P], [_P, nkb], [1, _P]], offset=off))
            return wu.view(_F8E4)
        wq = nl.ndarray((_P, nkb, _P), dtype=_F8E4, buffer=nl.sbuf)
        nisa.dma_copy(dst=wq, src=w8.ap([[nkb * _P, _P], [_P, nkb], [1, _P]], offset=off))
        return wq

    def _cast_store8(w8, FD, t, c, nkb, wt, sc_sb, col, forms, u8):
        """Switch 9f: tile c of layout t = e4m3(wt x sc[:, col]) by _cast8, the kernels' own cast instruction on the
        tile the kernels' own loader gives (u8 1: into a uint8 tile viewed as e4m3, switch 9's form; u8 0: into an
        e4m3 tile); then one DMA of its bytes into w8."""
        off = t * FD + c * _P * nkb * _P
        if u8 == 1:
            wu = nl.ndarray((_P, nkb, _P), dtype=nl.uint8, buffer=nl.sbuf)
            _cast8(wu.view(_F8E4), wt, sc_sb, col, forms)
            nisa.dma_copy(dst=w8.ap([[nkb * _P, _P], [_P, nkb], [1, _P]], offset=off), src=wu)
        else:
            wq = nl.ndarray((_P, nkb, _P), dtype=_F8E4, buffer=nl.sbuf)
            _cast8(wq, wt, sc_sb, col, forms)
            nisa.dma_copy(dst=w8.ap([[nkb * _P, _P], [_P, nkb], [1, _P]], offset=off), src=wq)

    @nki.jit
    def _precast8_kernel(w_fc, w_proj, sc, need, forms, u8):
        """Switch 9f: the e4m3 weight tiles of one layer, once an optimizer step. Layouts (each F x d bytes, tile c at
        c x P x nkb x P, element [p, k, i] at p x nkb x P + k x P + i): 0 the forward's W_fc^T tiles (_load_wT_chunk of
        w_fc, NF chunks), 1 the forward's W_proj^T tiles (ND chunks), 2 the backward's natural W_proj tiles
        (_load_w_chunk of w_proj, NF chunks), 3 the backward's natural W_fc tiles (ND chunks). need: 1 the forward's
        two, 2 the backward's two, 3 all four (a layout not needed is not written). The two programs split each
        layout's chunks. u8 1: the pack is uint8 (the device form); u8 0: e4m3 (the simulator's form)."""
        F = w_fc.shape[0]
        D = w_fc.shape[1]
        dt = w_fc.dtype
        FD = F * D
        ND = D // _P
        NF = F // _P
        if u8 == 1:
            w8 = nl.ndarray((4, FD), dtype=nl.uint8, buffer=nl.shared_hbm)
        else:
            w8 = nl.ndarray((4, FD), dtype=_F8E4, buffer=nl.shared_hbm)
        sc_sb = _load_sc(sc)
        pid, nprg = _shard()
        fa, fb = _span(NF, pid, nprg)
        ja, jb = _span(ND, pid, nprg)
        if need % 2 == 1:
            for i in range(fb - fa):
                c = fa + i
                wt = _load_wT_chunk(w_fc, D, ND, c, dt)
                _cast_store8(w8, FD, 0, c, ND, wt, sc_sb, SC_S_WFC, forms, u8)
            for i in range(jb - ja):
                c = ja + i
                wt = _load_wT_chunk(w_proj, F, NF, c, dt)
                _cast_store8(w8, FD, 1, c, NF, wt, sc_sb, SC_S_WPROJ, forms, u8)
        if need >= 2:
            for i in range(fb - fa):
                c = fa + i
                wn = _load_w_chunk(w_proj, F, ND, c, dt)
                _cast_store8(w8, FD, 2, c, ND, wn, sc_sb, SC_S_WPROJ, forms, u8)
            for i in range(jb - ja):
                c = ja + i
                wn = _load_w_chunk(w_fc, D, NF, c, dt)
                _cast_store8(w8, FD, 3, c, NF, wn, sc_sb, SC_S_WFC, forms, u8)
        return w8

    @nki.jit
    def _mlp8_fwd_pc_kernel(x2, w8, sc, tb, forms, u8):
        """Switch 9f: _mlp8_fwd_kernel with its two weight tiles read pre-cast (layouts 0 and 1 of w8) instead of
        streamed in bf16 and cast; every other instruction as there, so the outputs are its outputs bit for bit."""
        M = x2.shape[0]
        D = x2.shape[1]
        FD = w8.shape[1]
        F = FD // D
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
            x8 = nl.ndarray((_P, ND, ntok), dtype=_F8E4, buffer=nl.sbuf)
            if forms == 1:
                xt = _load_T_block(x2, D, ND, row0, ntok, dt)
                nisa.activation(dst=x8, op=nl.copy, data=xt, scale=X_SCALE)
            else:
                _load_T8_rows(x8, x2, D, ND, row0, nt, dt, X_SCALE)      # switch 9r
            h8 = nl.ndarray((_P, NF, ntok), dtype=_F8E4, buffer=nl.sbuf)
            for f in range(NF):
                f0 = f * _P
                f1 = f0 + _P
                w8t = _load_w8_tile(w8, FD, 0, f, ND, u8)
                pss = _pchunks(ntok, f32)
                _accumulate8(pss, w8t, x8, NDD, ntok)
                r_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                rc = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                if forms == 1:
                    hs = nl.ndarray((_P, ntok), dtype=f32, buffer=nl.sbuf)
                else:
                    hs = rc
                _evac_relu8(r_sb, rc, hs, h8, f, pss, ntok, sc_sb, forms)
                nisa.dma_copy(dst=rT[f0:f1, row0:row1], src=r_sb)
            ystage = nl.ndarray((_P, nt, D), dtype=dt, buffer=nl.sbuf)
            for j in range(ND):
                wp8 = _load_w8_tile(w8, FD, 1, j, NF, u8)
                pss = _pchunks(ntok, f32)
                _accumulate8(pss, wp8, h8, NFD, ntok)
                yT_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                _evac_desc(yT_sb, pss, ntok, sc_sb, SC_DQ_PROJ)
                _transpose_group(ystage, j, yT_sb, nt, dt)
            _store_rows(y, D, row0, nt, ystage)
        return y, rT

    @nki.jit
    def _mlp8_bwd_pc_kernel(dy2, rT, w8, sc, tb, dw_mode, forms, u8):
        """Switch 9f without 9e: _mlp8_bwd_kernel with its two weight tiles read pre-cast (layouts 2 and 3 of w8);
        every other instruction as there, so dx, h and dpre are its outputs bit for bit."""
        M = dy2.shape[0]
        D = dy2.shape[1]
        F = rT.shape[0]
        FD = F * D
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
            dy8 = nl.ndarray((_P, ND, ntok), dtype=_F8E5, buffer=nl.sbuf)
            if forms == 1:
                dyt = _load_T_block(dy2, D, ND, row0, ntok, dt)
                _cast8(dy8, dyt, sc_sb, SC_S_DY, forms)
            else:
                q0 = SC_S_DY
                q1 = q0 + 1
                _load_T8_rows(dy8, dy2, D, ND, row0, nt, dt, sc_sb[:, q0:q1])      # switch 9r
            dp8 = nl.ndarray((_P, NF, ntok), dtype=_F8E5, buffer=nl.sbuf)
            for f in range(NF):
                f0 = f * _P
                f1 = f0 + _P
                wp8 = _load_w8_tile(w8, FD, 2, f, ND, u8)
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
            dxstage = nl.ndarray((_P, nt, D), dtype=dt, buffer=nl.sbuf)
            for j in range(ND):
                wf8 = _load_w8_tile(w8, FD, 3, j, NF, u8)
                pss = _pchunks(ntok, f32)
                _accumulate8(pss, wf8, dp8, NFD, ntok)
                dxT_sb = nl.ndarray((_P, ntok), dtype=dt, buffer=nl.sbuf)
                _evac_desc(dxT_sb, pss, ntok, sc_sb, SC_DQ_DX)
                _transpose_group(dxstage, j, dxT_sb, nt, dt)
            _store_rows(dx, D, row0, nt, dxstage)
        return dx, h_out, dp_out

    def _accumulate8_at(pss, wt8, at8, NKD, c0, ntok):
        """Switch 9e: _accumulate8 over the columns c0 to c0 + ntok of the resident at8 (a block of the program's
        tokens): the same instructions, operand for operand, as _accumulate8 on a tile holding just those columns."""
        w0 = ntok
        if w0 > _CH:
            w0 = _CH
        a0 = c0
        a1 = c0 + w0
        b1 = c0 + ntok
        for kd in range(NKD):
            k0 = kd * 2
            k1 = k0 + 2
            nisa.nc_matmul(dst=pss[0], stationary=wt8[:, k0:k1, :], moving=at8[:, k0:k1, a0:a1], accumulate=(kd > 0),
                           perf_mode=_DR)
            if ntok > _CH:
                nisa.nc_matmul(dst=pss[1], stationary=wt8[:, k0:k1, :], moving=at8[:, k0:k1, a1:b1],
                               accumulate=(kd > 0), perf_mode=_DR)

    def _cast_dp8_2d(dst8, dp_sb, dps, sc_sb, forms):
        """Switch 9e: dst8 = e5m2(dpre x s_dp) for a whole [P, n] tile: _cast_dp8's two forms, the destination a 2-D
        tile instead of a slot of the resident 3-D one."""
        q0 = SC_S_DP
        q1 = q0 + 1
        if forms == 1:
            nisa.tensor_scalar(dst=dps, data=dp_sb, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)
            nisa.activation(dst=dst8, op=nl.copy, data=dps, scale=1.0)
        else:
            nisa.tensor_scalar(dst=dst8, data=dp_sb, op0=nl.multiply, operand0=sc_sb[:, q0:q1], engine=_VEC)

    def _transpose_group_at(stage, j, src2, nt, tt0, dt):
        """Switch 9e: _transpose_group with the nt token tiles placed from stage's token tile tt0 on (a block inside
        the program's tokens): stage[:, tt0 + tt, j * 128 to (j + 1) * 128] = (src2[:, tt * 128 to (tt + 1) * 128])
        transposed."""
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
                ti = tt0 + tt
                nisa.tensor_copy(dst=stage[:, ti, j0:j1], src=pst[:, q0:q1], engine=_VEC)

    def _transpose_cols(stage, q, src2, nj, dt):
        """Switch 9e: stage[:, j, q * 128 to (q + 1) * 128] = (src2[:, j * 128 to (j + 1) * 128])^T for j < nj: rows of
        dW_proj^T [hidden, d] turned into dW_proj's [d, hidden] tiles (_transpose_group's form)."""
        q0 = q * _P
        q1 = q0 + _P
        for g0 in range(0, nj, 4):
            ng = nj - g0
            if ng > 4:
                ng = 4
            pst = nl.ndarray((_P, ng * _P), dtype=dt, buffer=nl.psum)
            for gg in range(ng):
                j = g0 + gg
                p0 = gg * _P
                p1 = p0 + _P
                s0 = j * _P
                s1 = s0 + _P
                nisa.nc_transpose(dst=pst[:, p0:p1], data=src2[:, s0:s1])
            for gg in range(ng):
                j = g0 + gg
                p0 = gg * _P
                p1 = p0 + _P
                nisa.tensor_copy(dst=stage[:, j, q0:q1], src=pst[:, p0:p1], engine=_VEC)

    def _ptiles(ncol, f32):
        """Switch 9e: the fp32 PSUM tiles of one [P, ncol] accumulator, ncol up to 2,048, one bank (512 columns) each,
        as a tuple of one to four (_pchunks' if-chain form, extended)."""
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

    def _dw_acc(psw, st3, btok, tb0, nt, D, first):
        """Switch 9e: psw[ci] (+)= st3[:, i, :]^T @ btok[:, tb0 + i, ci-th 512 columns] over st3's nt token tiles (its
        tile i is the rank's token tile tb0 + i): a [P tokens, P hidden] stationary serves the d chunks in turn; with
        first = 1 the chain's first instruction (i = 0) starts the accumulators."""
        nci = _ceil_div(D, _CH)
        for i in range(nt):
            tt = tb0 + i
            acc = 1
            if first == 1:
                if i == 0:
                    acc = 0
            for ci in range(nci):
                c0 = ci * _CH
                c1 = c0 + _CH
                if c1 > D:
                    c1 = D
                nisa.nc_matmul(dst=psw[ci], stationary=st3[:, i, :], moving=btok[:, tt, c0:c1],
                               accumulate=(acc == 1))

    def _phase_a_chunk(f, own, snd, wp8, dy8, rT, dp8h, sc_sb, row0, ntok, nhb, tb, pid, NDD, forms, dt, f32):
        """Switch 9e: phase A of hidden chunk f over the program's tokens, block by block, as _mlp8_bwd_kernel's
        phase A computes it (the same instructions on the same operands): dh^T, dpre = bf16((acc x dq_dh2) x r),
        h = bf16(r^2), the e5m2 dpre bytes to dp8h; then the chunk's token-major tiles: own (program 0 dpre, program
        1 h) and snd (the other one, for the other program)."""
        f0 = f * _P
        f1 = f0 + _P
        for hb in range(nhb):
            c0 = hb * tb
            c1 = c0 + tb
            if c1 > ntok:
                c1 = ntok
            w = c1 - c0
            r0 = row0 + c0
            r1 = row0 + c1
            pss = _pchunks(w, f32)
            _accumulate8_at(pss, wp8, dy8, NDD, c0, w)
            r_sb = nl.ndarray((_P, w), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=r_sb, src=rT[f0:f1, r0:r1])
            dp_sb = nl.ndarray((_P, w), dtype=dt, buffer=nl.sbuf)
            _evac_fold8(dp_sb, r_sb, pss, w, sc_sb)
            h_sb = nl.ndarray((_P, w), dtype=dt, buffer=nl.sbuf)
            nisa.activation(dst=h_sb, op=nl.square, data=r_sb)
            if forms == 1:
                dps = nl.ndarray((_P, w), dtype=dt, buffer=nl.sbuf)
            else:
                dps = dp_sb
            dpq = nl.ndarray((_P, w), dtype=_F8E5, buffer=nl.sbuf)
            _cast_dp8_2d(dpq, dp_sb, dps, sc_sb, forms)
            nisa.dma_copy(dst=dp8h[f0:f1, r0:r1], src=dpq)
            nth = w // _P
            tt0 = c0 // _P
            if pid == 0:
                _transpose_group_at(own, 0, dp_sb, nth, tt0, dt)
                _transpose_group_at(snd, 0, h_sb, nth, tt0, dt)
            else:
                _transpose_group_at(own, 0, h_sb, nth, tt0, dt)
                _transpose_group_at(snd, 0, dp_sb, nth, tt0, dt)

    @nki.jit
    def _mlp8_bwd_l4_kernel(dy2, rT, x2, w_fc, w_proj, w8, sc, tb, gq, pc, forms, u8):
        """Switch 9e: the fp8 backward with the two weight gradients inside (module docstring, section L4). Under LNC 2
        program p trains its token tiles exactly as _mlp8_bwd_kernel does (dpre, h, the e5m2 dpre and dx are its
        values bit for bit) and computes ONE weight gradient over all of the rank's tokens: program 0 dW_fc = dpre^T x,
        program 1 dW_proj = dy^T h; each sends the other, chunk by chunk, the token-major tile of its own tokens that
        the other needs (sendrecv, SBUF to SBUF). pc 1 reads the weights pre-cast (switch 9f's layouts 2 and 3).
        Returns dx [M, d], dw_fc [F, d], dw_proj [d, F] (bf16 of fp32 sums over the M tokens) and the scratch dp8
        [F, M] e5m2 (the dpre bytes phase B reads back; the caller drops it). u8 is the pre-cast pack's form (pc 1)."""
        M = dy2.shape[0]
        D = dy2.shape[1]
        F = rT.shape[0]
        dt = dy2.dtype
        f32 = nl.float32
        FD = F * D
        dx = nl.ndarray((M, D), dtype=dt, buffer=nl.shared_hbm)
        dw_fc = nl.ndarray((F, D), dtype=dt, buffer=nl.shared_hbm)
        dw_proj = nl.ndarray((D, F), dtype=dt, buffer=nl.shared_hbm)
        dp8h = nl.ndarray((F, M), dtype=_F8E5, buffer=nl.shared_hbm)
        ND = D // _P
        NF = F // _P
        NDD = D // _KS
        NFD = F // _KS
        MT = M // _P
        sc_sb = _load_sc(sc)
        pid, nprg = _shard()
        other = 1 - pid
        t0, t1 = _span(MT, pid, nprg)
        o0, o1 = _span(MT, other, nprg)
        nt = t1 - t0
        ntok = nt * _P
        row0 = t0 * _P
        nhb = _ceil_div(ntok, tb)
        lo0 = t0
        hi0 = o0
        if o0 < t0:
            lo0 = o0
            hi0 = t0
        # dy^T of the program's tokens in e5m2, resident for phase A: the base's load and cast, block by block
        dy8 = nl.ndarray((_P, ND, ntok), dtype=_F8E5, buffer=nl.sbuf)
        for hb in range(nhb):
            c0 = hb * tb
            c1 = c0 + tb
            if c1 > ntok:
                c1 = ntok
            r0 = row0 + c0
            w = c1 - c0
            dyt = _load_T_block(dy2, D, ND, r0, w, dt)
            _cast8(dy8[:, :, c0:c1], dyt, sc_sb, SC_S_DY, forms)
        # every token of the rank, token-major: x (program 0, dW_fc's moving operand) or dy (program 1, dW_proj's)
        btok = nl.ndarray((_P, MT, D), dtype=dt, buffer=nl.sbuf)
        if pid == 0:
            nisa.dma_copy(dst=btok, src=x2.ap([[D, _P], [_P * D, MT], [1, D]], offset=0))
        else:
            nisa.dma_copy(dst=btok, src=dy2.ap([[D, _P], [_P * D, MT], [1, D]], offset=0))
        # The hidden chunks in order; chunk f's weight-gradient rows are computed after chunk f + 1's phase A, so the
        # exchange of chunk f runs on the DMA queues while the tensor engine does chunk f + 1's phase A (software
        # pipelining by one chunk; every loop is unrolled at trace time, so own_p and rcv_p name chunk f - 1's tiles).
        own_p = nl.ndarray((_P, 1, _P), dtype=dt, buffer=nl.sbuf)
        rcv_p = own_p
        if pid == 1:
            stg = nl.ndarray((_P, ND, gq * _P), dtype=dt, buffer=nl.sbuf)
        else:
            stg = own_p
        for f in range(NF + 1):
            if f < NF:
                if pc == 1:
                    wp8 = _load_w8_tile(w8, FD, 2, f, ND, u8)
                else:
                    wp = _load_w_chunk(w_proj, F, ND, f, dt)
                    wp8 = nl.ndarray((_P, ND, _P), dtype=_F8E4, buffer=nl.sbuf)
                    _cast8(wp8, wp, sc_sb, SC_S_WPROJ, forms)
                own_c = nl.ndarray((_P, nt, _P), dtype=dt, buffer=nl.sbuf)
                snd = nl.ndarray((_P, nt, _P), dtype=dt, buffer=nl.sbuf)
                _phase_a_chunk(f, own_c, snd, wp8, dy8, rT, dp8h, sc_sb, row0, ntok, nhb, tb, pid, NDD, forms, dt,
                               f32)
                rcv_c = nl.ndarray((_P, nt, _P), dtype=dt, buffer=nl.sbuf)
                pipe = f % 2
                nisa.sendrecv(src=snd, dst=rcv_c, send_to_rank=other, recv_from_rank=other, pipe_id=pipe)
            if f > 0:
                fp = f - 1
                fp0 = fp * _P
                fp1 = fp0 + _P
                if pid == 0:
                    lo = own_p
                    hi = rcv_p
                else:
                    lo = rcv_p
                    hi = own_p
                psw = _ptiles(D, f32)
                _dw_acc(psw, lo, btok, lo0, nt, D, 1)
                _dw_acc(psw, hi, btok, hi0, nt, D, 0)
                dwo = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
                _evac_copy(dwo, psw, D)
                if pid == 0:
                    nisa.dma_copy(dst=dw_fc[fp0:fp1, :], src=dwo)
                else:
                    qp = fp % gq
                    _transpose_cols(stg, qp, dwo, ND, dt)
                    if qp == gq - 1:
                        gc0 = (fp - qp) * _P
                        nisa.dma_copy(dst=dw_proj.ap([[F, _P], [_P * F, ND], [1, gq * _P]], offset=gc0), src=stg)
                        if fp < NF - 1:
                            stg = nl.ndarray((_P, ND, gq * _P), dtype=dt, buffer=nl.sbuf)
            if f < NF:
                own_p = own_c
                rcv_p = rcv_c
        # phase B: dx of the program's tokens from its e5m2 dpre, read back (the barrier orders the writes first)
        nisa.core_barrier(data=dp8h, cores=(0, 1))
        dp8 = nl.ndarray((_P, NF, ntok), dtype=_F8E5, buffer=nl.sbuf)
        nisa.dma_copy(dst=dp8, src=dp8h.ap([[M, _P], [_P * M, NF], [1, ntok]], offset=row0))
        dxstage = nl.ndarray((_P, nt, D), dtype=dt, buffer=nl.sbuf)
        for j in range(ND):
            if pc == 1:
                wf8 = _load_w8_tile(w8, FD, 3, j, NF, u8)
            else:
                wf = _load_w_chunk(w_fc, D, NF, j, dt)
                wf8 = nl.ndarray((_P, NF, _P), dtype=_F8E4, buffer=nl.sbuf)
                _cast8(wf8, wf, sc_sb, SC_S_WFC, forms)
            for hb in range(nhb):
                c0 = hb * tb
                c1 = c0 + tb
                if c1 > ntok:
                    c1 = ntok
                w = c1 - c0
                pss = _pchunks(w, f32)
                _accumulate8_at(pss, wf8, dp8, NFD, c0, w)
                dxT_sb = nl.ndarray((_P, w), dtype=dt, buffer=nl.sbuf)
                _evac_desc(dxT_sb, pss, w, sc_sb, SC_DQ_DX)
                nth = w // _P
                tt0 = c0 // _P
                _transpose_group_at(dxstage, j, dxT_sb, nth, tt0, dt)
        _store_rows(dx, D, row0, nt, dxstage)
        return dx, dw_fc, dw_proj, dp8h

    _HAS_NKI = True
except Exception as exc:  # nki unavailable (CPU dev box): the emulation below still works
    _NKI_ERROR = f"{type(exc).__name__}: {exc}"


def _launch(kernel):
    """kernel[2] under an LNC=2 runtime (both physical cores run half the token tiles), plain call otherwise."""
    return kernel[_LNC] if _LNC == 2 else kernel


# ----------------------------------------------------------------------------- lane THR5: shapes, plans, references

GQ = 4                      # switch 9e: hidden chunks of 128 per exchange group (512 hidden units)
SBUF_KIB = 212984 / 1024.0  # trn2 (gen3): usable SBUF a partition (nki.language.tile_size.sbuf_fmax_bytes)


def precast_need(mode: int) -> int:
    """Switch 9f: the layouts _precast8_kernel writes for switch 9c's mode (1 fp8 forward, 2 fp8 backward, 3 both):
    1 the forward's two, 2 the backward's two, 3 all four."""
    return int(mode)


def shapes_apply_pc(m: int, d: int, f: int, tb: int) -> bool:
    """Switch 9f: the shapes the pre-cast kernels take: switch 9c's (M a multiple of 128, d and F multiples of 256, tb
    a multiple of 128 up to 1024)."""
    return (m % _P == 0 and d % _KS == 0 and f % _KS == 0 and m > 0 and tb % _P == 0 and 0 < tb <= 1024)


def plan_kib_l4(m: int, d: int, f: int, tb: int = 1024, gq: int = GQ, pc: int = 0, forms: int = 0,
                nprg: int = 2) -> tuple:
    """Switch 9e: SBUF a partition (KiB) at the L4 backward's two peaks, every tile the kernel holds then counted once
    and each streamed or per-chunk tile twice (double-buffered): (main loop, phase B). Main loop: dy8 (the program's
    tokens, e5m2), btok (every token, bf16), two chunks' own, received and sent token-major tiles (the current
    chunk's and the previous one's, whose weight-gradient rows are computed after the current phase A), the weight
    tile (bf16 + e4m3, or the e4m3 bytes under pc), r, dpre, h and the dpre bytes of one block (and the forms-1
    stage), the dW row staging, program 1's [d, gq x 128] dW_proj tile. Phase B: the e5m2 dpre of the program's
    tokens, dx's stage, the W_fc tile, one block's dx^T."""
    nt = (m // _P) // nprg
    ntok = nt * _P
    mt = m // _P
    nd, nf = d // _P, f // _P
    w = min(tb, ntok)
    wt_main = (nd * _P) * (1 if pc else 3)
    blk = w * (2 + 2 + 2 + 1 + (2 if forms == 1 else 0))
    main = (nd * ntok + mt * d * 2 + 2 * 3 * nt * _P * 2 + 2 * wt_main + 2 * blk + 2 * d * 2 + nd * gq * _P * 2)
    wt_b = (nf * _P) * (1 if pc else 3)
    phb = nf * ntok + nt * d * 2 + 2 * wt_b + 2 * w * 2
    return main / 1024.0, phb / 1024.0


def shapes_apply_l4(m: int, d: int, f: int, tb: int, lnc: int, gq: int = GQ, pc: int = 0, forms: int = 0) -> bool:
    """Switch 9e: where the L4 backward runs: LNC 2 (two programs, one weight gradient each, exchanging tiles), M a
    multiple of 256 (equal token spans), d and F multiples of 256, F a multiple of gq x 128, tb a multiple of 128 up to
    1024, and both SBUF peaks within the usable SBUF."""
    if lnc != 2 or m <= 0 or m % (2 * _P) or d % _KS or f % _KS or f % (gq * _P) or tb % _P or not 0 < tb <= 1024:
        return False
    a, b = plan_kib_l4(m, d, f, tb, gq, pc, forms)
    return a <= SBUF_KIB and b <= SBUF_KIB


def l4_weight_grads(dy2: torch.Tensor, x2: torch.Tensor, hT: torch.Tensor, dpT: torch.Tensor):
    """Switch 9e: the L4 kernel's weight gradients in torch, from the backward's bf16 h^T and dpre^T [F, M] (the values
    the base kernel writes for the compiled GEMMs): exact products, fp32 sums (torch's order), one rounding to the
    activations' dtype. dw_fc [F, d] = dpre^T x, dw_proj [d, F] = dy^T h."""
    dt = dy2.dtype
    dw_fc = (dpT.float() @ x2.float()).to(dt)
    dw_proj = (dy2.float().t() @ hT.float().t()).to(dt)
    return dw_fc, dw_proj


def _bytes_e4m3(t: torch.Tensor) -> torch.Tensor:
    """The e4m3 bytes of t's fp32 values (round to nearest even; torch's float8_e4m3fn, whose bytes inside +-240 are the
    chip's legacy e4m3's: the only range the kernels cast) as uint8."""
    return t.float().to(torch.float8_e4m3fn).view(torch.uint8)


def precast_ref(w_fc: torch.Tensor, w_proj: torch.Tensor, row: torch.Tensor, need: int = 3) -> torch.Tensor:
    """Switch 9f: the bytes _precast8_kernel writes, [4, F x d] uint8 (layouts in its docstring; a layout not in need is
    zero here and unwritten there), from the bf16 weights and the layer's scale row (forward_scales'). Tile c, element
    [p, k, i] of layout 0 = e4m3(W_fc[c P + i, k P + p] s_wfc); 1: e4m3(W_proj[c P + i, k P + p] s_wproj); 2:
    e4m3(W_proj[k P + p, c P + i] s_wproj); 3: e4m3(W_fc[k P + p, c P + i] s_wfc)."""
    F, D = w_fc.shape
    nd, nf = D // _P, F // _P
    wf = w_fc.to(torch.bfloat16).float() * row[SC_S_WFC]
    wp = w_proj.to(torch.bfloat16).float() * row[SC_S_WPROJ]
    out = torch.zeros(4, F * D, dtype=torch.uint8)
    if need % 2 == 1:
        # 0: chunk c = rows c of W_fc; [c, i, k, p] -> [c, p, k, i]
        t0 = wf.reshape(nf, _P, nd, _P).permute(0, 3, 2, 1)
        out[0] = _bytes_e4m3(t0.contiguous()).reshape(-1)
        t1 = wp.reshape(nd, _P, nf, _P).permute(0, 3, 2, 1)
        out[1] = _bytes_e4m3(t1.contiguous()).reshape(-1)
    if need >= 2:
        # 2: chunk c = columns c of W_proj [d, F]: W_proj[k P + p, c P + i] -> [c, p, k, i]
        t2 = wp.reshape(nd, _P, nf, _P).permute(2, 1, 0, 3)
        out[2] = _bytes_e4m3(t2.contiguous()).reshape(-1)
        t3 = wf.reshape(nf, _P, nd, _P).permute(2, 1, 0, 3)
        out[3] = _bytes_e4m3(t3.contiguous()).reshape(-1)
    return out
