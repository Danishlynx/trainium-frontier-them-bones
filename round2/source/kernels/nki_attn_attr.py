"""NKI causal attention (forward + backward) for train.py's CausalSelfAttention, one launch each way; v8 adds the
fused RoPE + QK-norm + attention entry point (plan v11 lane B, default off).

Replaces, in CausalSelfAttention.forward (q, k, v are [B, H, T, D] after rope_norm / the eager RoPE+norm path):

    y = F.scaled_dot_product_attention(q, k, v, is_causal=True[, scale=QK_SCALE])
    y = y.transpose(1, 2).contiguous().view(b, t, c)

with

    y = nki_causal_attention(q, k, v, scale=QK_SCALE or None).view(b, t, c)     # [B, T, H, D] straight from the kernel

Public entry point
------------------
    nki_causal_attention(q, k, v, scale=None) -> y [B, T, H, D]
        q, k, v   [B, H, T, D], same dtype (bf16 in training; fp32 only for tests with R1_NKI_ATTN_FP32=1)
        scale     softmax scale (None = 1/sqrt(D), SDPA's default)
        y         == F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale).transpose(1, 2)
                  written by the kernel directly into [B, T, H, D] memory, so the block's `.view(b, t, c)` is free
                  (today's path pays a transpose + contiguous copy of 32 MiB per block).  The eager fallback
                  returns the same layout.

Math (per (b, h) pair; S = scale * Q K^T, causal, D = head dim, T tokens)
--------------------------------------------------------------------------
Forward, "transposed" formulation (keys on the partition axis, queries on the free axis):
    S^T[k, q] = K[k, :] . Q[q, :]                        nc_matmul(stationary = K^T[d, k], moving = Q^T[d, q]) -> PSUM
    E^T[k, q] = exp(scale * S^T - c)  (bf16, 0 above the diagonal)
    [O | s][q, :] = sum_k E^T[k, q] * [V | 1][k, :]     nc_matmul(stationary = E^T[k, q], moving = V_aug[k, d+1]) -> PSUM
    O[q, :] = O[q, :] / s[q];   lse[q] = c + log s[q]
  The fixed shift c replaces the row max: after the model's parameter-free QK-norm every element of q and k has
  rms 1, so |q . k| <= D and |scale * S| <= scale * D =: c.  exp(scale*S - c) is then in [e^-2c, 1], every term a
  normal fp32/bf16 number (c = 11.3 at D = 128, scale = 1/sqrt(128); 16.3 at QK_SCALE 0.127), and the log-sum-exp is
  exact to fp32 rounding.  (This is the loss-head kernel's bounded-logits trick; it is what makes the transposed
  forward possible: a per-query max would be a partition-axis reduction here.)  The wrapper sets c = scale * D
  (+ R1_NKI_ATTN_SHIFT_MARGIN); scores may exceed c by up to ~80 before fp32 exp overflows, so the bound only has
  to hold approximately.  The ones column of V_aug makes the PV matmul produce the row sums for free: no P
  transposes, no max pass, no separate sum pass.  Row sums are sums of the SAME bf16 E values the PV matmul uses.
Backward, row formulation (queries on the partition axis, nkilib's layout), per query tile g, key tiles j <= g:
    P = exp(scale * S - lse[q])  (bf16, masked on the diagonal block)                 ScalarE, per-partition bias
    dP = dO V^T (PSUM),  D[q] = sum_d dO[q, d] O[q, d]  (VectorE),  dS = P * (dP - D[q])  (one VectorE pass, bf16)
    dV[k, :] += P^T dO        nc_matmul(stationary = P[q, k],    moving = dO[q, d])      PSUM accumulator over g
    dK[k, :] += dS^T Q        nc_matmul(stationary = dS[q, k],   moving = Q[q, d])       PSUM accumulator over g
    dQ[q, :] += dS K          nc_matmul(stationary = dS^T[k, q], moving = K[k, d])       PSUM accumulator over j
    dK, dQ scaled by `scale` on the way out.  dS^T comes from TensorE nc_transpose (bit-accurate on Trn2).
  Causal work at 128-token granularity: 36 of 64 (q-tile, k-tile) blocks at T = 1024 (nkilib: 48, its K tiles are
  512 wide); only the 8 diagonal blocks are masked (GpSimd affine_select, in place, on the bf16 probabilities).

Layouts / DMAs: q, k, v are read per (b, h) as [128 tokens, D] tiles (one DMA per head tensor with an access
pattern, 256 B per partition per tile); Q^T/K^T/V^T/dO^T come from TensorE nc_transpose of those tiles
(R1_NKI_ATTN_LOAD_T=pe, default) or from one DMA transpose per head (=dma, nkilib's bf16 form).  O and dO live in
[B, T, H, D] memory (the block's layout), dV is written there too and handed back as a permuted view (what
`v.transpose(1, 2)`'s backward wants, no copy); dQ, dK come back as [B, H, T, D] (rope_norm's layout).

Guards: R1_NKI_ATTN=1 turns the kernel on (default OFF).  Kernel only on Neuron, bf16 (fp32 with R1_NKI_ATTN_FP32=1),
D <= 128, T % 128 == 0, T <= 4096, same shape/dtype for q, k, v.  A kernel failure prints once, disables the kernel
for the process and answers that call with the eager math (forward: SDPA; backward: SDPA's autograd).

Knobs (env, read once at import; module-level constants because kernel bodies only see module globals):
  R1_NKI_ATTN_LNC          2 -> kernel[2] under NEURON_LOGICAL_NC_CONFIG=2 (both physical cores; default from the env)
  R1_NKI_ATTN_SHARD        1 (default) | 0: no program_id sharding (debug)
  R1_NKI_ATTN_LOAD_T       dma (default) | pe: Q^T/K^T/V^T tiles by one DMA transpose per head, or TensorE transposes
  R1_NKI_ATTN_BWD_ACC      sbuf (default) | psum: dK/dV accumulate in SBUF with one VectorE add per block, or in
                           PSUM across query tiles (psum was CLOBBERED on chip in gap 2: dk/dv wrong at T=1024)
  R1_NKI_ATTN_KCHUNK       key tiles per PSUM chunk (default 4 = 512 columns, the PSUM bank / matmul moving max)
  R1_NKI_ATTN_BWD_GROUP    1 (default = the shipped v5 loop) | 2..8: query tiles per group in the backward (nkilib's
                           q_tile_group_size form): P and dS of the group are stashed in SBUF and, per key tile, the
                           group's <= GROUP dV / dK matmuls accumulate in one fresh PSUM tile (accumulate inside the
                           group only, never across loops) folded ONCE into the SBUF accumulators; the softmax scale
                           is folded into the dK fold.  24 folds per pair at T=1024 with GROUP=4 instead of 72.
  R1_NKI_ATTN_ABLATE       "" (default) | adds | matmuls -- BENCH ONLY (gradients wrong by design): skip the dV/dK
                           SBUF folds, or the folds and their matmuls, to split the fold cost from the matmul cost
  R1_NKI_ATTN_COPY_ENGINE  auto (default) | scalar | vector: engine of the PSUM -> SBUF copies after transposes
  R1_NKI_ATTN_SHIFT_MARGIN added to the fixed shift c (default 0.0)
  R1_NKI_ATTN_BATCH_SPLIT  n > 1: n launches over batch slices per call (smaller unrolled kernels; outputs cat'd)
  R1_NKI_ATTN_FP32         1 allows fp32 inputs through the kernel (tests); default 0

Fused RoPE + QK-norm + attention (plan v11 lane B; default off)
--------------------------------------------------------------
    nki_rope_attention(qkv, cos, sin, n_head, scale=None) -> y [B, T, H, D] or None
        qkv       [B, T, 3C] bf16, the fused QKV projection output (q = columns [0, C), k = [C, 2C), v = [2C, 3C),
                  head h at columns h*D .. (h+1)*D of each third), contiguous
        cos, sin  train.py's RoPE tables for the T positions ([1, T, 1, D/2]; anything that reshapes to [T, D/2])
        y         == nki_causal_attention(rope_norm(q), rope_norm(k), v.transpose(1, 2)) to the BIT: the forward
                  kernel reads q, k, v straight from qkv, applies the RoPE+QK-norm kernel's instruction sequence on
                  the natural [128 tokens, G, q|k, D] tile in SBUF, forms Q^T/K^T on TensorE and runs the v7 loop;
                  the backward kernel applies the norm Jacobian and the inverse rotation to the v7 kernel's bf16
                  dQ/dK tiles and writes dq | dk | dv into ONE [B, T, 3C] gradient (no split-cat, no v copy).
                  None when the fused path does not apply (switch off, CPU, D != 128, T % 128, ...): the caller
                  keeps today's rope_norm -> nki_causal_attention chain.  A kernel failure disables the fused path
                  for the process and answers that call with the eager math (RoPE, rms_norm, SDPA and its autograd).
  R1_NKI_ATTN_ROPE         1 turns the fused path on (default OFF); needs R1_NKI_ATTN=1 and R1_NKI_ATTN_BWD_GROUP >= 2
  R1_NKI_ATTN_ROPE_SAVE    r (default): the forward saves RoPE(q|k) (bf16) and 1/rms (fp32) for the backward
                           | x: the backward recomputes both from qkv (less memory, more backward work)
  R1_NKI_ATTN_ROPE_VT      dma (default) | pe: the backward's V^T by dma_transpose from the strided qkv rows, or by
                           a natural load + TensorE transposes
  R1_NKI_ATTN_ROPE_DMA     split (default) | ap4 | tile: the q|k tile load, the r save/load and the dq|dk store as one
                           3-level access pattern per q/k (2 KiB runs), one 4-level pattern for both (its stores lower
                           to a slow descriptor form: +3.9 ms per block on chip), or one DMA per 128-token block
  R1_NKI_ATTN_ROPE_VLOAD   tile (default) | one: the forward's V tiles as one DMA per key tile (the v7 form) or one
                           3-level DMA into the strided V_aug view
  R1_NKI_ATTN_ROPE_BCAST   1 (default): stride-0 broadcast views of the cos/sin tiles over the q|k axis | 0: copies
  R1_NKI_ATTN_V81          1: the v8.1 engine placement of the fused kernels (bit-identical to v8; attn_audit gap 2,
                           -0.22 ms per block): forward normalise and lse add on VectorE, the transposes' PSUM copies
                           on VectorE, the 8 row sums gathered into ONE log; backward without the accumulator memsets
                           (the first fold of each key tile is a ScalarE copy / scale), D[q] as one product + one
                           reduce, the dK / dV casts as two wide ScalarE copies and one dV DMA per pair.  Default 0.
  R1_NKI_ATTN_QKN_DT       fp32 (default) | bf16: dtype of the fused backward's two wide QK-norm Jacobian scratch
                           tiles, `sq` (the g*r product that feeds the row-sum) and `u` (g - coef*r).  bf16 keeps every
                           term -- a precision change, not an approximation: one extra rounding of values that are
                           rounded to bf16 before they leave the block in any case (0.81 bf16 ulp rms on the stored
                           dq|dk).  The two allocations are the only instructions that change.  Transplanted for
                           candidate 22 from the chips' nki_attn.py (eb84f4be, the attn_deletions lane), kernel and
                           emulation lines verbatim; with the default this file is v8.1 exactly.

Lane ATTR (work/p2/attr/NOTES.md; tools/build_attr_kernel.py): nki_attn_v8.py b4944199 plus two forward riders
------------------------------------------------------------------------------------------------------------
  R1_NKI_ATTN_PRO          1: the fused forward's DMA forms, in both forward kernels: a pair's 16 V tile loads as
                           ONE 3-level load (the ROPE_VLOAD=one pattern) into one of two V_aug tiles allocated once a
                           program and filled with ones once (the load writes only the V columns, so no per-pair
                           memset gates it), and its 16 output stores as ONE 3-level store from one of two
                           [128, G, 128] tiles that the normalisations write; both DMAs on the Act queue's hardware
                           DGE (on this compiler the 16 stores were issued by ScalarE inside the loop it paces).
                           The same values reach the same instructions: o3, lse3, r5, inv3 bit for bit. Default 0.
  _attn_rope_fwd_kernel_pl lane A9M's software-pipelined forward (kernels/nki_attn_a9m.py 76ded7fa, its forward block
                           verbatim but for the R1_NKI_ATTN_PRO branches): the next pair's prologue is issued inside
                           this pair's query-tile loop; bit for bit the base forward. The backward is v8's either way.
"""

import math
import os

import torch
import torch.nn.functional as F

_P = 128


def _env_flag(name, default):
    return os.environ.get(name, default) == "1"


def _env_str(name, default):
    return os.environ.get(name, "") or default


def _env_int(name, default):
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


USE_NKI_ATTN = _env_flag("R1_NKI_ATTN", "0")
_LNC = _env_int("R1_NKI_ATTN_LNC", _env_int("NEURON_LOGICAL_NC_CONFIG", 1))
_SHARD = _env_int("R1_NKI_ATTN_SHARD", 1)
_LOAD_T = _env_str("R1_NKI_ATTN_LOAD_T", "dma")     # gap 2: dma passed every check and is faster (fwd 1.43 vs 1.53 ms)
_BWD_ACC = _env_str("R1_NKI_ATTN_BWD_ACC", "sbuf")  # gap 2: the PSUM-resident dK/dV accumulators were clobbered on chip
_KCHUNK = max(1, min(4, _env_int("R1_NKI_ATTN_KCHUNK", 4)))
_BWD_GROUP = max(1, min(8, _env_int("R1_NKI_ATTN_BWD_GROUP", 1)))   # query tiles per backward group (1 = shipped loop)
_ABLATE = _env_str("R1_NKI_ATTN_ABLATE", "")   # bench only: "adds" skips the dV/dK folds, "matmuls" their matmuls too
_COPY_ENGINE = _env_str("R1_NKI_ATTN_COPY_ENGINE", "auto")
_SHIFT_MARGIN = float(os.environ.get("R1_NKI_ATTN_SHIFT_MARGIN", "0") or 0.0)
_BATCH_SPLIT = max(1, _env_int("R1_NKI_ATTN_BATCH_SPLIT", 1))   # launches per call over batch slices (compile-time escape hatch)
ALLOW_FP32 = _env_flag("R1_NKI_ATTN_FP32", "0")
EMULATE = False   # test-only: route NkiAttnFn through the torch emulation of the kernels (any device)

# [rope_attn] RoPE + QK-norm folded into the attention kernel (plan v11 lane B): q, k, v read straight from the fused
# QKV projection output [B, T, 3C], dq | dk | dv written into ONE [B, T, 3C] gradient.  Default off; with the switch
# off this module behaves exactly as v7 (the v7 kernels and wrapper are untouched, see build_module.py).
USE_NKI_ATTN_ROPE = _env_flag("R1_NKI_ATTN_ROPE", "0")
_ROPE_SAVE = _env_str("R1_NKI_ATTN_ROPE_SAVE", "r")      # r: forward saves RoPE(q|k) + 1/rms | x: backward recomputes them from qkv
_ROPE_VT = _env_str("R1_NKI_ATTN_ROPE_VT", "dma")        # backward V^T: dma_transpose of the strided qkv rows | pe: natural load + TensorE
_ROPE_DMA = _env_str("R1_NKI_ATTN_ROPE_DMA", "split")    # q|k load, r save/load, dq|dk store: split | ap4 | tile (gap 1: the 4-level store pattern is 3.9 ms slower per block)
_ROPE_VLOAD = _env_str("R1_NKI_ATTN_ROPE_VLOAD", "tile") # forward V tiles: tile (one DMA per key tile, v7's form) | one
_ROPE_BCAST = _env_int("R1_NKI_ATTN_ROPE_BCAST", 1)      # 1: stride-0 cos/sin broadcast views over the q|k axis | 0: materialised
_V81 = _env_flag("R1_NKI_ATTN_V81", "0")                 # 1: the v8.1 engine placement (attn_audit gap 2, g3 + g4; bit-identical)
_QKN_DT = _env_str("R1_NKI_ATTN_QKN_DT", "fp32")         # fp32 (v8.1) | bf16: the backward Jacobian's sq and u tiles
_PRO = _env_flag("R1_NKI_ATTN_PRO", "0")                 # [pro] lane ATTR: one V load and one output store a pair
# [R1 attn_audit gap 3] BENCH-ONLY ABLATION, not a correctness switch. 0 removes ONLY the backward
# QK-norm Jacobian and inverse rotation (the "epilogue"), keeping the prologue and the dq|dk store, so a
# paired timing read isolates the epilogue EXPOSED wall cost. The audit measured the whole v9 ablation
# (forward RoPE + prologue recompute + epilogue together) at -0.888 ms/block, an UPPER bound on the
# epilogue alone. Its notes call the epilogue "fully exposed": it runs on VectorE while TensorE, ScalarE
# and the DMA queues idle, and it has no dependence on the next pair. With 0 the gradient is WRONG by
# construction -- never set it outside a bench gap, and never in a candidate.
_AUD_JAC = _env_flag("R1_AUD_JAC", "1")
_ROPE_EPS = 2.0 ** -23                                    # the QK-norm's rms eps (nki_rope_norm_opt.RMS_EPS default == F.rms_norm's)

_KERNEL_DISABLED = False
_LAST_KERNEL_ERROR = None
_MAX_T = 4096


# ----------------------------------------------------------------------------- eager math

def eager_causal_attention(q, k, v, scale=None):
    """SDPA(q, k, v, causal) delivered as [B, T, H, D] (the layout the kernel writes)."""
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    return y.transpose(1, 2).contiguous()


def _eager_backward(q, k, v, gy_bthd, scale):
    with torch.enable_grad():
        qr = q.detach().requires_grad_(True)
        kr = k.detach().requires_grad_(True)
        vr = v.detach().requires_grad_(True)
        y = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True, scale=scale)
        return torch.autograd.grad(y, (qr, kr, vr), gy_bthd.transpose(1, 2))


# ----------------------------------------------------------------------------- the kernels

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    def _shard():
        """(program_id, num_programs); (0, 1) without a launch grid or with sharding off."""
        if _SHARD and nl.program_ndim() > 0:
            return nl.program_id(0), nl.num_programs(0)
        return 0, 1

    def _copy_engine():
        if _COPY_ENGINE == "scalar":
            return nisa.engine.scalar
        if _COPY_ENGINE == "vector":
            return nisa.engine.vector
        return nisa.engine.unknown

    def _load_nat(x3, base, T, D, G):
        """[P, G*D] SBUF tile of one head: partition p, block g = token g*128 + p (one DMA, 256 B runs)."""
        nat = nl.ndarray((_P, G * D), dtype=x3.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=nat, src=x3.ap(pattern=[[D, _P], [_P * D, G], [1, D]], offset=base))
        return nat

    def _load_nat_bthd(x3, b, h, T, C, D, G):
        """Same tile layout from a [B, T, C] tensor (head h's D columns of each token row)."""
        nat = nl.ndarray((_P, G * D), dtype=x3.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=nat, src=x3.ap(pattern=[[C, _P], [_P * C, G], [1, D]], offset=b * T * C + h * D))
        return nat

    def _transpose_nat(nat, T, D, G):
        """[D, T] SBUF tile (head dim on partitions) from the natural [P, G*D] tile: G TensorE transposes."""
        xt = nl.ndarray((D, T), dtype=nat.dtype, buffer=nl.sbuf)
        for g0 in range(0, G, 4):
            ng = G - g0
            if ng > 4:
                ng = 4
            ps = nl.ndarray((D, ng * _P), dtype=nat.dtype, buffer=nl.psum)
            for gg in range(ng):
                g = g0 + gg
                nisa.nc_transpose(dst=ps[:, gg * _P:(gg + 1) * _P], data=nat[:, g * D:(g + 1) * D])
            nisa.tensor_copy(dst=xt[:, g0 * _P:(g0 + ng) * _P], src=ps, engine=_copy_engine())
        return xt

    def _load_T(x3, base, T, D, G):
        """[D, T] SBUF tile of one head, head dim on partitions (nkilib's bf16 form: one DMA transpose)."""
        if _LOAD_T == "dma":
            xt = nl.ndarray((D, T), dtype=x3.dtype, buffer=nl.sbuf)
            nisa.dma_transpose(dst=xt.ap([[T, D], [1, 1], [1, 1], [1, T]]),
                               src=x3.ap([[D, T], [1, 1], [1, 1], [1, D]], offset=base))
            return xt
        return _transpose_nat(_load_nat(x3, base, T, D, G), T, D, G)

    @nki.jit
    def _attn_fwd_kernel(q4, k4, v4, scale, shift):
        """o3[B, T, H*D] (q4.dtype) and lse3[B*H, 128, G] (fp32; [i, p, g] = query g*128 + p of pair i = b*H + h)
        from q4, k4, v4 [B, H, T, D]."""
        B, H, T, D = q4.shape
        bs = B * H
        G = T // _P
        C = H * D
        W = D + 1
        f32 = nl.float32
        dt = q4.dtype
        o3 = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
        lse3 = nl.ndarray((bs, _P, G), dtype=f32, buffer=nl.shared_hbm)
        pid, nprg = _shard()
        nbias = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)   # the fixed shift as an exp bias tile
        nisa.memset(nbias, value=-shift)
        for i in range(pid, bs, nprg):
            b = i // H
            h = i % H
            base = i * T * D
            qT = _load_T(q4, base, T, D, G)                       # [D, T]
            kT = _load_T(k4, base, T, D, G)                       # [D, T]
            vaug = nl.ndarray((_P, G * W), dtype=dt, buffer=nl.sbuf)   # block j = [V tile j | 1]
            nisa.memset(vaug, value=1.0)                          # column D of every block = 1: the row sums
            for j in range(G):
                nisa.dma_copy(dst=vaug[:, j * W:j * W + D], src=v4.ap(pattern=[[D, _P], [1, D]], offset=base + j * _P * D))
            lse_blk = nl.ndarray((_P, G), dtype=f32, buffer=nl.sbuf)
            for g in range(G):
                oacc = nl.ndarray((_P, W), dtype=f32, buffer=nl.psum)
                for j0 in range(0, g + 1, _KCHUNK):
                    nj = g + 1 - j0
                    if nj > _KCHUNK:
                        nj = _KCHUNK
                    st = nl.ndarray((_P, nj * _P), dtype=f32, buffer=nl.psum)
                    for jj in range(nj):
                        j = j0 + jj
                        nisa.nc_matmul(dst=st[:, jj * _P:(jj + 1) * _P], stationary=kT[:, j * _P:(j + 1) * _P],
                                       moving=qT[:, g * _P:(g + 1) * _P], accumulate=False)
                    et = nl.ndarray((_P, nj * _P), dtype=dt, buffer=nl.sbuf)
                    nisa.activation(dst=et, op=nl.exp, data=st, bias=nbias, scale=scale)
                    if j0 + nj - 1 == g:                          # diagonal block: keep key p <= query f
                        jj = nj - 1
                        nisa.affine_select(dst=et[:, jj * _P:(jj + 1) * _P], pattern=[[1, _P]], channel_multiplier=-1,
                                           offset=0, cmp_op=nl.greater_equal,
                                           on_true_tile=et[:, jj * _P:(jj + 1) * _P], on_false_value=0.0)
                    for jj in range(nj):
                        j = j0 + jj
                        nisa.nc_matmul(dst=oacc, stationary=et[:, jj * _P:(jj + 1) * _P],
                                       moving=vaug[:, j * W:(j + 1) * W], accumulate=(j > 0))
                rs = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rs, data=oacc[:, D:D + 1])
                ot = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=ot, data=oacc[:, 0:D], op0=nl.multiply, operand0=rs)
                nisa.dma_copy(dst=o3.ap(pattern=[[C, _P], [1, D]], offset=(b * T + g * _P) * C + h * D), src=ot)
                nisa.activation(dst=lse_blk[:, g:g + 1], op=nl.log, data=oacc[:, D:D + 1])
            nisa.tensor_scalar(dst=lse_blk, data=lse_blk, op0=nl.add, operand0=shift)
            nisa.dma_copy(dst=lse3[i, :, :], src=lse_blk)
        return o3, lse3

    @nki.jit
    def _attn_bwd_kernel(q4, k4, v4, o3, do3, lse3, scale):
        """dq4, dk4 [B, H, T, D] and dv3 [B, T, H*D] (all q4.dtype) from q4, k4, v4 [B, H, T, D], o3, do3 [B, T, H*D],
        lse3 [B*H, 128, G] fp32."""
        B, H, T, D = q4.shape
        bs = B * H
        G = T // _P
        C = H * D
        f32 = nl.float32
        dt = q4.dtype
        dq4 = nl.ndarray((B, H, T, D), dtype=dt, buffer=nl.shared_hbm)
        dk4 = nl.ndarray((B, H, T, D), dtype=dt, buffer=nl.shared_hbm)
        dv3 = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
        pid, nprg = _shard()
        for i in range(pid, bs, nprg):
            b = i // H
            h = i % H
            base = i * T * D
            qn = _load_nat(q4, base, T, D, G)                     # [P, G*D]: partition = query, block = q tile
            kn = _load_nat(k4, base, T, D, G)
            on = _load_nat_bthd(o3, b, h, T, C, D, G)
            don = _load_nat_bthd(do3, b, h, T, C, D, G)
            qT = _transpose_nat(qn, T, D, G)                      # [D, T]
            kT = _transpose_nat(kn, T, D, G)
            doT = _transpose_nat(don, T, D, G)
            if _LOAD_T == "dma":
                vT = _load_T(v4, base, T, D, G)
            else:
                vT = _transpose_nat(_load_nat(v4, base, T, D, G), T, D, G)
            nb = nl.ndarray((_P, G), dtype=f32, buffer=nl.sbuf)  # -lse, the exp bias per query
            nisa.dma_copy(dst=nb, src=lse3[i, :, :])
            nisa.tensor_scalar(dst=nb, data=nb, op0=nl.multiply, operand0=-1.0)
            dsum = nl.ndarray((_P, G), dtype=f32, buffer=nl.sbuf)  # D[q] = sum_d dO * O
            for g in range(G):
                tmp = nl.ndarray((_P, D), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=tmp, data1=don[:, g * D:(g + 1) * D], data2=on[:, g * D:(g + 1) * D],
                                   op=nl.multiply)
                nisa.tensor_reduce(dst=dsum[:, g:g + 1], op=nl.add, data=tmp, axis=1)
            if _BWD_ACC == "psum" and _BWD_GROUP == 1:
                dkacc = nl.ndarray((_P, G * D), dtype=f32, buffer=nl.psum)
                dvacc = nl.ndarray((_P, G * D), dtype=f32, buffer=nl.psum)
            else:
                dkacc = nl.ndarray((_P, G * D), dtype=f32, buffer=nl.sbuf)
                dvacc = nl.ndarray((_P, G * D), dtype=f32, buffer=nl.sbuf)
                nisa.memset(dkacc, value=0.0)
                nisa.memset(dvacc, value=0.0)
            if _BWD_GROUP == 1:
              # the shipped v5 loop (gap 3: 24/24 PASS, 4.213 ms fwd+bwd per block), one fold per (g, j) block
              for g in range(G):
                dqacc = nl.ndarray((_P, D), dtype=f32, buffer=nl.psum)
                for j0 in range(0, g + 1, _KCHUNK):
                    nj = g + 1 - j0
                    if nj > _KCHUNK:
                        nj = _KCHUNK
                    w = nj * _P
                    s = nl.ndarray((_P, w), dtype=f32, buffer=nl.psum)
                    nisa.nc_matmul(dst=s, stationary=qT[:, g * _P:(g + 1) * _P], moving=kT[:, j0 * _P:j0 * _P + w],
                                   accumulate=False)
                    p = nl.ndarray((_P, w), dtype=dt, buffer=nl.sbuf)
                    nisa.activation(dst=p, op=nl.exp, data=s, bias=nb[:, g:g + 1], scale=scale)
                    if j0 + nj - 1 == g:                          # diagonal block: keep key f <= query p
                        jj = nj - 1
                        nisa.affine_select(dst=p[:, jj * _P:(jj + 1) * _P], pattern=[[-1, _P]], channel_multiplier=1,
                                           offset=0, cmp_op=nl.greater_equal,
                                           on_true_tile=p[:, jj * _P:(jj + 1) * _P], on_false_value=0.0)
                    dp = nl.ndarray((_P, w), dtype=f32, buffer=nl.psum)
                    nisa.nc_matmul(dst=dp, stationary=doT[:, g * _P:(g + 1) * _P], moving=vT[:, j0 * _P:j0 * _P + w],
                                   accumulate=False)
                    ds = nl.ndarray((_P, w), dtype=dt, buffer=nl.sbuf)
                    nisa.scalar_tensor_tensor(dst=ds, data=dp, op0=nl.subtract, operand0=dsum[:, g:g + 1],
                                              op1=nl.multiply, operand1=p)
                    dst_ps = nl.ndarray((_P, w), dtype=dt, buffer=nl.psum)
                    for jj in range(nj):
                        nisa.nc_transpose(dst=dst_ps[:, jj * _P:(jj + 1) * _P], data=ds[:, jj * _P:(jj + 1) * _P])
                    dsT = nl.ndarray((_P, w), dtype=dt, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dsT, src=dst_ps, engine=_copy_engine())
                    for jj in range(nj):
                        j = j0 + jj
                        if _BWD_ACC == "psum":
                            nisa.nc_matmul(dst=dvacc[:, j * D:(j + 1) * D], stationary=p[:, jj * _P:(jj + 1) * _P],
                                           moving=don[:, g * D:(g + 1) * D], accumulate=(g > j))
                            nisa.nc_matmul(dst=dkacc[:, j * D:(j + 1) * D], stationary=ds[:, jj * _P:(jj + 1) * _P],
                                           moving=qn[:, g * D:(g + 1) * D], accumulate=(g > j))
                        elif _ABLATE != "matmuls":
                            dv_ps = nl.ndarray((_P, D), dtype=f32, buffer=nl.psum)
                            nisa.nc_matmul(dst=dv_ps, stationary=p[:, jj * _P:(jj + 1) * _P],
                                           moving=don[:, g * D:(g + 1) * D], accumulate=False)
                            if _ABLATE == "":
                                nisa.tensor_tensor(dst=dvacc[:, j * D:(j + 1) * D], data1=dvacc[:, j * D:(j + 1) * D],
                                                   data2=dv_ps, op=nl.add)
                            dk_ps = nl.ndarray((_P, D), dtype=f32, buffer=nl.psum)
                            nisa.nc_matmul(dst=dk_ps, stationary=ds[:, jj * _P:(jj + 1) * _P],
                                           moving=qn[:, g * D:(g + 1) * D], accumulate=False)
                            if _ABLATE == "":
                                nisa.tensor_tensor(dst=dkacc[:, j * D:(j + 1) * D], data1=dkacc[:, j * D:(j + 1) * D],
                                                   data2=dk_ps, op=nl.add)
                        nisa.nc_matmul(dst=dqacc, stationary=dsT[:, jj * _P:(jj + 1) * _P],
                                       moving=kn[:, j * D:(j + 1) * D], accumulate=(j > 0))
                dqt = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=dqt, data=dqacc, op0=nl.multiply, operand0=scale)
                nisa.dma_copy(dst=dq4[b, h, g * _P:(g + 1) * _P, :], src=dqt)
            else:
              # grouped form (nkilib attention_bwd.py:1858-1912, q_tile_group_size): query tiles in groups of
              # _BWD_GROUP; P and dS of the group live in SBUF (row gg = query tile g0+gg at columns [gg*wg, ...),
              # key tile j at gg*wg + j*_P); after the group's S/P/dP/dS/dQ pass, per key tile j the group's
              # <= _BWD_GROUP dV (dK) matmuls accumulate in ONE fresh [P, D] fp32 PSUM tile and fold once.
              for g0 in range(0, G, _BWD_GROUP):
                ng = G - g0
                if ng > _BWD_GROUP:
                    ng = _BWD_GROUP
                wg = (g0 + ng) * _P                               # key range of the group's last query tile
                pg = nl.ndarray((_P, ng * wg), dtype=dt, buffer=nl.sbuf)
                dsg = nl.ndarray((_P, ng * wg), dtype=dt, buffer=nl.sbuf)
                for gg in range(ng):
                    g = g0 + gg
                    dqacc = nl.ndarray((_P, D), dtype=f32, buffer=nl.psum)
                    for j0 in range(0, g + 1, _KCHUNK):
                        nj = g + 1 - j0
                        if nj > _KCHUNK:
                            nj = _KCHUNK
                        w = nj * _P
                        cb = gg * wg + j0 * _P                    # this chunk's columns in the group stash
                        s = nl.ndarray((_P, w), dtype=f32, buffer=nl.psum)
                        nisa.nc_matmul(dst=s, stationary=qT[:, g * _P:(g + 1) * _P],
                                       moving=kT[:, j0 * _P:j0 * _P + w], accumulate=False)
                        nisa.activation(dst=pg[:, cb:cb + w], op=nl.exp, data=s, bias=nb[:, g:g + 1], scale=scale)
                        if j0 + nj - 1 == g:                      # diagonal block: keep key f <= query p
                            db = cb + (nj - 1) * _P
                            nisa.affine_select(dst=pg[:, db:db + _P], pattern=[[-1, _P]], channel_multiplier=1,
                                               offset=0, cmp_op=nl.greater_equal,
                                               on_true_tile=pg[:, db:db + _P], on_false_value=0.0)
                        dp = nl.ndarray((_P, w), dtype=f32, buffer=nl.psum)
                        nisa.nc_matmul(dst=dp, stationary=doT[:, g * _P:(g + 1) * _P],
                                       moving=vT[:, j0 * _P:j0 * _P + w], accumulate=False)
                        nisa.scalar_tensor_tensor(dst=dsg[:, cb:cb + w], data=dp, op0=nl.subtract,
                                                  operand0=dsum[:, g:g + 1], op1=nl.multiply, operand1=pg[:, cb:cb + w])
                        dst_ps = nl.ndarray((_P, w), dtype=dt, buffer=nl.psum)
                        for jj in range(nj):
                            nisa.nc_transpose(dst=dst_ps[:, jj * _P:(jj + 1) * _P],
                                              data=dsg[:, cb + jj * _P:cb + (jj + 1) * _P])
                        dsT = nl.ndarray((_P, w), dtype=dt, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=dsT, src=dst_ps, engine=_copy_engine())
                        for jj in range(nj):
                            j = j0 + jj
                            nisa.nc_matmul(dst=dqacc, stationary=dsT[:, jj * _P:(jj + 1) * _P],
                                           moving=kn[:, j * D:(j + 1) * D], accumulate=(j > 0))
                    dqt = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=dqt, data=dqacc, op0=nl.multiply, operand0=scale)
                    nisa.dma_copy(dst=dq4[b, h, g * _P:(g + 1) * _P, :], src=dqt)
                for j in range(g0 + ng):
                    fg = j - g0                                   # first query tile of the group that sees key tile j
                    if fg < 0:
                        fg = 0
                    if _ABLATE != "matmuls":
                        dv_ps = nl.ndarray((_P, D), dtype=f32, buffer=nl.psum)
                        for gg in range(fg, ng):
                            g = g0 + gg
                            nisa.nc_matmul(dst=dv_ps, stationary=pg[:, gg * wg + j * _P:gg * wg + (j + 1) * _P],
                                           moving=don[:, g * D:(g + 1) * D], accumulate=(gg > fg))
                        if _ABLATE == "":
                            nisa.tensor_tensor(dst=dvacc[:, j * D:(j + 1) * D], data1=dvacc[:, j * D:(j + 1) * D],
                                               data2=dv_ps, op=nl.add)
                        dk_ps = nl.ndarray((_P, D), dtype=f32, buffer=nl.psum)
                        for gg in range(fg, ng):
                            g = g0 + gg
                            nisa.nc_matmul(dst=dk_ps, stationary=dsg[:, gg * wg + j * _P:gg * wg + (j + 1) * _P],
                                           moving=qn[:, g * D:(g + 1) * D], accumulate=(gg > fg))
                        if _ABLATE == "":                         # scale folded in: (dk_ps * scale) + dkacc
                            nisa.scalar_tensor_tensor(dst=dkacc[:, j * D:(j + 1) * D], data=dk_ps, op0=nl.multiply,
                                                      operand0=scale, op1=nl.add, operand1=dkacc[:, j * D:(j + 1) * D])
            for j in range(G):
                dkt = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
                if _BWD_GROUP == 1:
                    nisa.tensor_scalar(dst=dkt, data=dkacc[:, j * D:(j + 1) * D], op0=nl.multiply, operand0=scale)
                else:
                    nisa.tensor_copy(dst=dkt, src=dkacc[:, j * D:(j + 1) * D])
                nisa.dma_copy(dst=dk4[b, h, j * _P:(j + 1) * _P, :], src=dkt)
                dvt = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dvt, src=dvacc[:, j * D:(j + 1) * D])
                nisa.dma_copy(dst=dv3.ap(pattern=[[C, _P], [1, D]], offset=(b * T + j * _P) * C + h * D), src=dvt)
        return dq4, dk4, dv3

    # ------------------------------------------------------------ [rope_attn] RoPE + QK-norm folded into attention
    #
    # Per (b, h) pair the un-rotated q and k rows of the projection output come in as ONE natural SBUF tile
    # xqk[P, G, 2, D] (row (p, g, s) = token g*128 + p of q (s = 0) / k (s = 1)); rotation, row rms and the
    # normalisation are the RoPE+QK-norm kernel's instructions (nki_rope_norm_opt v19 with its default knobs: rotation
    # temporaries in the input dtype, ScalarE square, VectorE row reduce, tensor_scalar rsqrt, ONE broadcast scale =
    # the single rounding) issued on this tile shape instead of its [P, K, H, D]; Q^T / K^T come from TensorE
    # transposes of the normed tile (the v6 LOAD_T=pe path) and the attention loop is the v7 loop verbatim.  The
    # backward keeps the bf16 dQ / dK tiles the v7 kernel stores in an SBUF stash and applies the norm Jacobian + the
    # inverse rotation (the RoPE+QK-norm backward's instructions, verbatim) once per pair, then stores dq | dk | dv
    # into dqkv[B, T, 3C] at the projection's column offsets.

    def _rope_tables(cos2, sin2, G, Dh):
        """cos / sin for this program as [P, G, 1, Dh] SBUF tiles ([p, g] = token g*128 + p), returned as [P, G, 2, Dh]
        views: stride-0 broadcasts over the q|k axis (the rope kernel's `broadcast` idiom) or materialised copies."""
        cosb = nl.ndarray((_P, G, 1, Dh), dtype=cos2.dtype, buffer=nl.sbuf)
        sinb = nl.ndarray((_P, G, 1, Dh), dtype=sin2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=cosb, src=cos2.ap(pattern=[[Dh, _P], [_P * Dh, G], [1, Dh]], offset=0))
        nisa.dma_copy(dst=sinb, src=sin2.ap(pattern=[[Dh, _P], [_P * Dh, G], [1, Dh]], offset=0))
        if _ROPE_BCAST:
            return cosb.broadcast(2, 2), sinb.broadcast(2, 2)
        cos2t = nl.ndarray((_P, G, 2, Dh), dtype=cos2.dtype, buffer=nl.sbuf)
        sin2t = nl.ndarray((_P, G, 2, Dh), dtype=sin2.dtype, buffer=nl.sbuf)
        for s in range(2):
            nisa.tensor_copy(dst=cos2t[:, :, s, :], src=cosb[:, :, 0, :])
            nisa.tensor_copy(dst=sin2t[:, :, s, :], src=sinb[:, :, 0, :])
        return cos2t, sin2t

    def _rope_rotate(dst, src, cb, sb, tmp_a, tmp_b, Dh, transpose):
        """dst = R(src) (transpose=False) or R^T(src) on [P, G, 2, 2*Dh] tiles; cb / sb are [P, G, 2, Dh] views.
        The six VectorE instructions of nki_rope_norm_opt._rotate, engines and dtypes as its defaults."""
        D = 2 * Dh
        s1, s2 = src[:, :, :, 0:Dh], src[:, :, :, Dh:D]
        d1, d2 = dst[:, :, :, 0:Dh], dst[:, :, :, Dh:D]
        nisa.tensor_tensor(dst=tmp_a, data1=s1, data2=cb, op=nl.multiply)
        nisa.tensor_tensor(dst=tmp_b, data1=s2, data2=sb, op=nl.multiply)
        nisa.tensor_tensor(dst=d1, data1=tmp_a, data2=tmp_b, op=nl.subtract if transpose else nl.add,
                           engine=nisa.engine.unknown)
        nisa.tensor_tensor(dst=tmp_a, data1=s2, data2=cb, op=nl.multiply)
        nisa.tensor_tensor(dst=tmp_b, data1=s1, data2=sb, op=nl.multiply)
        nisa.tensor_tensor(dst=d2, data1=tmp_a, data2=tmp_b, op=nl.add if transpose else nl.subtract,
                           engine=nisa.engine.unknown)

    def _rope_inv_rms(r, sq, G, D):
        """inv[P, G, 2, 1] = rsqrt(mean_d r^2 + eps) in fp32: nki_rope_norm_opt._inv_rms with SQ=act, RSQRT=gpsimd
        (ScalarE square into the fp32 scratch sq, VectorE row reduce, the scale/eps tensor_scalar, the rsqrt)."""
        ss = nl.ndarray((_P, G, 2, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=r)
        nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=3, keepdims=True)
        inv = nl.ndarray((_P, G, 2, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ss, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=_ROPE_EPS)
        nisa.tensor_scalar(dst=inv, data=ss, op0=nl.rsqrt, operand0=0.0)
        return inv

    def _rope_qk_dma(tile, qkv3, b, h, T, C, G, D, store):
        """tile[P, G, 2, D] <- the q|k rows of head h of batch b (store=False), or the dq|dk columns of dqkv <- tile
        (store=True): row (p, g, s) <-> token g*128 + p, columns s*C + h*D .. of the [B, T, 3C] tensor."""
        C3 = 3 * C
        base = b * T * C3 + h * D
        if _ROPE_DMA == "ap4":
            ap = qkv3.ap(pattern=[[C3, _P], [_P * C3, G], [C, 2], [1, D]], offset=base)
            if store:
                nisa.dma_copy(dst=ap, src=tile)
            else:
                nisa.dma_copy(dst=tile, src=ap)
        elif _ROPE_DMA == "split":
            for s in range(2):
                ap = qkv3.ap(pattern=[[C3, _P], [_P * C3, G], [1, D]], offset=base + s * C)
                if store:
                    nisa.dma_copy(dst=ap, src=tile[:, :, s, :])
                else:
                    nisa.dma_copy(dst=tile[:, :, s, :], src=ap)
        else:
            for s in range(2):
                for g in range(G):
                    ap = qkv3.ap(pattern=[[C3, _P], [1, D]], offset=base + g * _P * C3 + s * C)
                    if store:
                        nisa.dma_copy(dst=ap, src=tile[:, g, s, :])
                    else:
                        nisa.dma_copy(dst=tile[:, g, s, :], src=ap)

    def _rope_r_dma(tile, r5, b, h, B, H, T, G, D, store):
        """r5[s, b, h, g*128 + p, :] <-> tile[p, g, s, :] (the saved RoPE(q|k); 32 KiB contiguous per (s, g))."""
        base = (b * H + h) * T * D
        ss = B * H * T * D
        if _ROPE_DMA == "ap4":
            ap = r5.ap(pattern=[[D, _P], [_P * D, G], [ss, 2], [1, D]], offset=base)
            if store:
                nisa.dma_copy(dst=ap, src=tile)
            else:
                nisa.dma_copy(dst=tile, src=ap)
        elif _ROPE_DMA == "split":
            for s in range(2):
                ap = r5.ap(pattern=[[D, _P], [_P * D, G], [1, D]], offset=base + s * ss)
                if store:
                    nisa.dma_copy(dst=ap, src=tile[:, :, s, :])
                else:
                    nisa.dma_copy(dst=tile[:, :, s, :], src=ap)
        else:
            for s in range(2):
                for g in range(G):
                    if store:
                        nisa.dma_copy(dst=r5[s, b, h, g * _P:(g + 1) * _P, :], src=tile[:, g, s, :])
                    else:
                        nisa.dma_copy(dst=tile[:, g, s, :], src=r5[s, b, h, g * _P:(g + 1) * _P, :])

    def _rope_transpose(yn, s, T, D, G):
        """[D, T] SBUF tile of q (s = 0) or k (s = 1) from the normed natural tile yn[P, G, 2, D]: G TensorE
        transposes through PSUM in groups of 4 (the _transpose_nat form on the 4-D tile)."""
        xt = nl.ndarray((D, T), dtype=yn.dtype, buffer=nl.sbuf)
        for g0 in range(0, G, 4):
            ng = G - g0
            if ng > 4:
                ng = 4
            ps = nl.ndarray((D, ng * _P), dtype=yn.dtype, buffer=nl.psum)
            for gg in range(ng):
                g = g0 + gg
                nisa.nc_transpose(dst=ps[:, gg * _P:(gg + 1) * _P], data=yn[:, g, s, :])
            nisa.tensor_copy(dst=xt[:, g0 * _P:(g0 + ng) * _P], src=ps, engine=_copy_engine())
        return xt

    def _rope_transpose_v81(yn, s, T, D, G):
        """[v81] _rope_transpose with the PSUM -> SBUF copies pinned to VectorE (the forward's ScalarE is the paced
        engine; gap 2 of the attention audit)."""
        xt = nl.ndarray((D, T), dtype=yn.dtype, buffer=nl.sbuf)
        for g0 in range(0, G, 4):
            ng = G - g0
            if ng > 4:
                ng = 4
            ps = nl.ndarray((D, ng * _P), dtype=yn.dtype, buffer=nl.psum)
            for gg in range(ng):
                g = g0 + gg
                nisa.nc_transpose(dst=ps[:, gg * _P:(gg + 1) * _P], data=yn[:, g, s, :])
            nisa.tensor_copy(dst=xt[:, g0 * _P:(g0 + ng) * _P], src=ps, engine=nisa.engine.vector)
        return xt

    def _rope_vaug(qkv3, b, h, T, C, G, D):
        """[P, G*(D+1)] tile of the V key tiles of head h with a ones column per block (the row sums), read from the
        [B, T, 3C] projection output (v = columns 2C + h*D ..)."""
        C3 = 3 * C
        W = D + 1
        vbase = b * T * C3 + 2 * C + h * D
        vaug = nl.ndarray((_P, G * W), dtype=qkv3.dtype, buffer=nl.sbuf)
        nisa.memset(vaug, value=1.0)
        if _ROPE_VLOAD == "one":
            nisa.dma_copy(dst=vaug.reshape_dim(1, (G, W))[:, :, 0:D],
                          src=qkv3.ap(pattern=[[C3, _P], [_P * C3, G], [1, D]], offset=vbase))
        else:
            for j in range(G):
                nisa.dma_copy(dst=vaug[:, j * W:j * W + D], src=qkv3.ap(pattern=[[C3, _P], [1, D]], offset=vbase + j * _P * C3))
        return vaug

    def _pro_vload(vaug, qkv3, b, h, T, C, G, D):
        """[pro] the V tiles of head h of batch b into vaug[P, G*(D+1)] (its ones columns set once a program): ONE
        3-level DMA (the ROPE_VLOAD=one pattern) on the Act queue's hardware DGE. Returns vaug."""
        C3 = 3 * C
        W = D + 1
        nisa.dma_copy(dst=vaug.reshape_dim(1, (G, W))[:, :, 0:D],
                      src=qkv3.ap(pattern=[[C3, _P], [_P * C3, G], [1, D]], offset=b * T * C3 + 2 * C + h * D),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
        return vaug

    def _pro_ostore(obuf, o3, b, h, T, C, G, D):
        """[pro] o3[b, g*128 + p, h*D + d] <- obuf[p, g, d] for the whole pair: ONE 3-level DMA with the per-tile
        stores' address map, on the Act queue's hardware DGE."""
        nisa.dma_copy(dst=o3.ap(pattern=[[C, _P], [_P * C, G], [1, D]], offset=b * T * C + h * D), src=obuf,
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)

    def _pro_bufs(dt, G, D):
        """[pro] a program's two V_aug tiles (all ones: the loads write only the V columns, the ones columns stay)
        and two output tiles [P, G, D]; pair k of the program uses tile k % 2 of each."""
        W = D + 1
        vb0 = nl.ndarray((_P, G * W), dtype=dt, buffer=nl.sbuf)
        vb1 = nl.ndarray((_P, G * W), dtype=dt, buffer=nl.sbuf)
        nisa.memset(vb0, value=1.0)
        nisa.memset(vb1, value=1.0)
        ob0 = nl.ndarray((_P, G, D), dtype=dt, buffer=nl.sbuf)
        ob1 = nl.ndarray((_P, G, D), dtype=dt, buffer=nl.sbuf)
        return vb0, vb1, ob0, ob1

    @nki.jit
    def _attn_rope_fwd_kernel(qkv3, cos2, sin2, scale, shift):
        """o3[B, T, C] and lse3[B*H, 128, G] (fp32) as the v7 forward, plus r5[2, B, H, T, D] = RoPE(q | k) (bf16,
        written only with SAVE=r) and inv3[B*H, 128, G, 2, 1] = 1/rms of every q | k row (fp32), from the projection
        output qkv3[B, T, 3C] and the tables cos2 / sin2 [T, D/2].  D = 128 (one head per tile row)."""
        B, T, C3 = qkv3.shape
        C = C3 // 3
        D = _P
        H = C // D
        Dh = D // 2
        bs = B * H
        G = T // _P
        W = D + 1
        f32 = nl.float32
        dt = qkv3.dtype
        o3 = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
        lse3 = nl.ndarray((bs, _P, G), dtype=f32, buffer=nl.shared_hbm)
        inv3 = nl.ndarray((bs, _P, G, 2, 1), dtype=f32, buffer=nl.shared_hbm)
        if _ROPE_SAVE == "r":
            r5 = nl.ndarray((2, B, H, T, D), dtype=dt, buffer=nl.shared_hbm)
        else:
            r5 = nl.ndarray((2, 1, 1, 1, 1), dtype=dt, buffer=nl.shared_hbm)   # placeholder, never read
        pid, nprg = _shard()
        cb, sb = _rope_tables(cos2, sin2, G, Dh)                 # once per program
        nbias = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)   # the fixed shift as an exp bias tile
        nisa.memset(nbias, value=-shift)
        if _PRO:                                                  # [pro] the program's two V_aug and output tiles
            vb0, vb1, ob0, ob1 = _pro_bufs(dt, G, D)
        for i in range(pid, bs, nprg):
            b = i // H
            h = i % H
            # --- RoPE + QK-norm on the natural q|k tile (the rope kernel's forward, verbatim)
            xqk = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            _rope_qk_dma(xqk, qkv3, b, h, T, C, G, D, False)
            tmp_a = nl.ndarray((_P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
            tmp_b = nl.ndarray((_P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
            r = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            _rope_rotate(r, xqk, cb, sb, tmp_a, tmp_b, Dh, False)
            sq = nl.ndarray((_P, G, 2, D), dtype=f32, buffer=nl.sbuf)
            inv = _rope_inv_rms(r, sq, G, D)
            yn = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yn, data1=r, data2=inv.broadcast(3, D), op=nl.multiply)   # the one rounding
            if _ROPE_SAVE == "r":
                _rope_r_dma(r, r5, b, h, B, H, T, G, D, True)
            nisa.dma_copy(dst=inv3[i, :, :, :, :], src=inv)
            # --- the v7 forward on Q^T / K^T from TensorE transposes of the normed tile
            if _V81:                                              # [v81] the transposes' PSUM copies on VectorE
                qT = _rope_transpose_v81(yn, 0, T, D, G)
                kT = _rope_transpose_v81(yn, 1, T, D, G)
            else:
                qT = _rope_transpose(yn, 0, T, D, G)                  # [D, T]
                kT = _rope_transpose(yn, 1, T, D, G)                  # [D, T]
            if _PRO:                                              # [pro] pair k = i // nprg of the program: tiles k % 2
                vaug = _pro_vload(vb0 if (i // nprg) % 2 == 0 else vb1, qkv3, b, h, T, C, G, D)
                obuf = ob0 if (i // nprg) % 2 == 0 else ob1
            else:
                vaug = _rope_vaug(qkv3, b, h, T, C, G, D)             # block j = [V tile j | 1]
            lse_blk = nl.ndarray((_P, G), dtype=f32, buffer=nl.sbuf)
            if _V81:
                sums = nl.ndarray((_P, G), dtype=f32, buffer=nl.sbuf)   # [v81] the 8 row sums, gathered for ONE log
            for g in range(G):
                oacc = nl.ndarray((_P, W), dtype=f32, buffer=nl.psum)
                for j0 in range(0, g + 1, _KCHUNK):
                    nj = g + 1 - j0
                    if nj > _KCHUNK:
                        nj = _KCHUNK
                    st = nl.ndarray((_P, nj * _P), dtype=f32, buffer=nl.psum)
                    for jj in range(nj):
                        j = j0 + jj
                        nisa.nc_matmul(dst=st[:, jj * _P:(jj + 1) * _P], stationary=kT[:, j * _P:(j + 1) * _P],
                                       moving=qT[:, g * _P:(g + 1) * _P], accumulate=False)
                    et = nl.ndarray((_P, nj * _P), dtype=dt, buffer=nl.sbuf)
                    nisa.activation(dst=et, op=nl.exp, data=st, bias=nbias, scale=scale)
                    if j0 + nj - 1 == g:                          # diagonal block: keep key p <= query f
                        jj = nj - 1
                        nisa.affine_select(dst=et[:, jj * _P:(jj + 1) * _P], pattern=[[1, _P]], channel_multiplier=-1,
                                           offset=0, cmp_op=nl.greater_equal,
                                           on_true_tile=et[:, jj * _P:(jj + 1) * _P], on_false_value=0.0)
                    for jj in range(nj):
                        j = j0 + jj
                        nisa.nc_matmul(dst=oacc, stationary=et[:, jj * _P:(jj + 1) * _P],
                                       moving=vaug[:, j * W:(j + 1) * W], accumulate=(j > 0))
                rs = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rs, data=oacc[:, D:D + 1])
                if _PRO:                                          # [pro] the normalised tile lands in the pair's tile
                    ot = obuf[:, g, :]
                else:
                    ot = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
                if _V81:                                          # [v81] normalise on VectorE, the row sum gathered
                    nisa.tensor_scalar(dst=ot, data=oacc[:, 0:D], op0=nl.multiply, operand0=rs, engine=nisa.engine.vector)
                    nisa.tensor_copy(dst=sums[:, g:g + 1], src=oacc[:, D:D + 1], engine=nisa.engine.vector)
                else:
                    nisa.tensor_scalar(dst=ot, data=oacc[:, 0:D], op0=nl.multiply, operand0=rs)
                if not _PRO:                                      # [pro] one store a pair, after the loop
                    nisa.dma_copy(dst=o3.ap(pattern=[[C, _P], [1, D]], offset=(b * T + g * _P) * C + h * D), src=ot)
                if not _V81:
                    nisa.activation(dst=lse_blk[:, g:g + 1], op=nl.log, data=oacc[:, D:D + 1])
            if _PRO:                                              # [pro] the pair's output, one DMA
                _pro_ostore(obuf, o3, b, h, T, C, G, D)
            if _V81:                                              # [v81] ONE log over the gathered sums
                nisa.activation(dst=lse_blk, op=nl.log, data=sums)
                nisa.tensor_scalar(dst=lse_blk, data=lse_blk, op0=nl.add, operand0=shift, engine=nisa.engine.vector)
            else:
                nisa.tensor_scalar(dst=lse_blk, data=lse_blk, op0=nl.add, operand0=shift)
            nisa.dma_copy(dst=lse3[i, :, :], src=lse_blk)
        return o3, lse3, r5, inv3

    @nki.jit
    def _attn_rope_bwd_kernel(qkv3, r5, inv3, o3, do3, lse3, cos2, sin2, scale):
        """dqkv3[B, T, 3C] = dq | dk | dv (qkv3.dtype) from the projection output qkv3, the forward's r5 / inv3 (or,
        with SAVE=x, a recompute from qkv3), o3 / do3 [B, T, C], lse3 [B*H, 128, G] and the tables.  The v7 grouped
        backward (BWD_GROUP >= 2, SBUF accumulators) with the norm Jacobian + inverse rotation on its bf16 dQ / dK."""
        B, T, C3 = qkv3.shape
        C = C3 // 3
        D = _P
        H = C // D
        Dh = D // 2
        bs = B * H
        G = T // _P
        f32 = nl.float32
        dt = qkv3.dtype
        dqkv3 = nl.ndarray((B, T, C3), dtype=dt, buffer=nl.shared_hbm)
        pid, nprg = _shard()
        cb, sb = _rope_tables(cos2, sin2, G, Dh)
        for i in range(pid, bs, nprg):
            b = i // H
            h = i % H
            # --- r = RoPE(q|k) and inv = 1/rms: saved by the forward (SAVE=r) or recomputed exactly as it did (SAVE=x)
            rr = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            if _ROPE_SAVE == "r":
                _rope_r_dma(rr, r5, b, h, B, H, T, G, D, False)
                inv = nl.ndarray((_P, G, 2, 1), dtype=f32, buffer=nl.sbuf)
                nisa.dma_copy(dst=inv, src=inv3[i, :, :, :, :])
            else:
                xqk = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
                _rope_qk_dma(xqk, qkv3, b, h, T, C, G, D, False)
                tmp_a0 = nl.ndarray((_P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
                tmp_b0 = nl.ndarray((_P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
                _rope_rotate(rr, xqk, cb, sb, tmp_a0, tmp_b0, Dh, False)
                sq0 = nl.ndarray((_P, G, 2, D), dtype=f32, buffer=nl.sbuf)
                inv = _rope_inv_rms(rr, sq0, G, D)
            yn = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)  # the normed q|k rows, the same bits as the forward's
            nisa.tensor_tensor(dst=yn, data1=rr, data2=inv.broadcast(3, D), op=nl.multiply)
            # --- the v7 backward prologue: natural O / dO tiles, the transposed operands, -lse, D[q]
            on = _load_nat_bthd(o3, b, h, T, C, D, G)
            don = _load_nat_bthd(do3, b, h, T, C, D, G)
            qT = _rope_transpose(yn, 0, T, D, G)                  # [D, T]
            kT = _rope_transpose(yn, 1, T, D, G)
            doT = _transpose_nat(don, T, D, G)
            vbase = b * T * C3 + 2 * C + h * D
            if _ROPE_VT == "dma":
                vT = nl.ndarray((D, T), dtype=dt, buffer=nl.sbuf)
                nisa.dma_transpose(dst=vT.ap([[T, D], [1, 1], [1, 1], [1, T]]),
                                   src=qkv3.ap([[C3, T], [1, 1], [1, 1], [1, D]], offset=vbase))
            else:
                vn = nl.ndarray((_P, G * D), dtype=dt, buffer=nl.sbuf)
                nisa.dma_copy(dst=vn, src=qkv3.ap(pattern=[[C3, _P], [_P * C3, G], [1, D]], offset=vbase))
                vT = _transpose_nat(vn, T, D, G)
            nb = nl.ndarray((_P, G), dtype=f32, buffer=nl.sbuf)  # -lse, the exp bias per query
            nisa.dma_copy(dst=nb, src=lse3[i, :, :])
            nisa.tensor_scalar(dst=nb, data=nb, op0=nl.multiply, operand0=-1.0)
            dsum = nl.ndarray((_P, G), dtype=f32, buffer=nl.sbuf)  # D[q] = sum_d dO * O
            if _V81:                                              # [v81] D[q] as ONE product + ONE reduce
                tmp_all = nl.ndarray((_P, G * D), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=tmp_all, data1=don, data2=on, op=nl.multiply)
                nisa.tensor_reduce(dst=dsum, op=nl.add, data=tmp_all.reshape_dim(1, (G, D)), axis=2)
            else:
                for g in range(G):
                    tmp = nl.ndarray((_P, D), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=tmp, data1=don[:, g * D:(g + 1) * D], data2=on[:, g * D:(g + 1) * D],
                                       op=nl.multiply)
                    nisa.tensor_reduce(dst=dsum[:, g:g + 1], op=nl.add, data=tmp, axis=1)
            dkacc = nl.ndarray((_P, G * D), dtype=f32, buffer=nl.sbuf)
            dvacc = nl.ndarray((_P, G * D), dtype=f32, buffer=nl.sbuf)
            if not _V81:                                          # [v81] the first fold of each key tile is a copy
                nisa.memset(dkacc, value=0.0)
                nisa.memset(dvacc, value=0.0)
            gqk = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)   # the bf16 dQ (s = 0) / dK (s = 1) tiles v7 stores
            # --- the v7 grouped loop (query tiles in groups of _BWD_GROUP; P and dS stashed; per key tile one fresh
            #     PSUM accumulator over the group folded once into the SBUF accumulators, scale folded into the dK fold)
            for g0 in range(0, G, _BWD_GROUP):
                ng = G - g0
                if ng > _BWD_GROUP:
                    ng = _BWD_GROUP
                wg = (g0 + ng) * _P                               # key range of the group's last query tile
                pg = nl.ndarray((_P, ng * wg), dtype=dt, buffer=nl.sbuf)
                dsg = nl.ndarray((_P, ng * wg), dtype=dt, buffer=nl.sbuf)
                for gg in range(ng):
                    g = g0 + gg
                    dqacc = nl.ndarray((_P, D), dtype=f32, buffer=nl.psum)
                    for j0 in range(0, g + 1, _KCHUNK):
                        nj = g + 1 - j0
                        if nj > _KCHUNK:
                            nj = _KCHUNK
                        w = nj * _P
                        cbk = gg * wg + j0 * _P                   # this chunk's columns in the group stash
                        s = nl.ndarray((_P, w), dtype=f32, buffer=nl.psum)
                        nisa.nc_matmul(dst=s, stationary=qT[:, g * _P:(g + 1) * _P],
                                       moving=kT[:, j0 * _P:j0 * _P + w], accumulate=False)
                        nisa.activation(dst=pg[:, cbk:cbk + w], op=nl.exp, data=s, bias=nb[:, g:g + 1], scale=scale)
                        if j0 + nj - 1 == g:                      # diagonal block: keep key f <= query p
                            db = cbk + (nj - 1) * _P
                            nisa.affine_select(dst=pg[:, db:db + _P], pattern=[[-1, _P]], channel_multiplier=1,
                                               offset=0, cmp_op=nl.greater_equal,
                                               on_true_tile=pg[:, db:db + _P], on_false_value=0.0)
                        dp = nl.ndarray((_P, w), dtype=f32, buffer=nl.psum)
                        nisa.nc_matmul(dst=dp, stationary=doT[:, g * _P:(g + 1) * _P],
                                       moving=vT[:, j0 * _P:j0 * _P + w], accumulate=False)
                        nisa.scalar_tensor_tensor(dst=dsg[:, cbk:cbk + w], data=dp, op0=nl.subtract,
                                                  operand0=dsum[:, g:g + 1], op1=nl.multiply, operand1=pg[:, cbk:cbk + w])
                        dst_ps = nl.ndarray((_P, w), dtype=dt, buffer=nl.psum)
                        for jj in range(nj):
                            nisa.nc_transpose(dst=dst_ps[:, jj * _P:(jj + 1) * _P],
                                              data=dsg[:, cbk + jj * _P:cbk + (jj + 1) * _P])
                        dsT = nl.ndarray((_P, w), dtype=dt, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=dsT, src=dst_ps, engine=_copy_engine())
                        for jj in range(nj):
                            j = j0 + jj
                            nisa.nc_matmul(dst=dqacc, stationary=dsT[:, jj * _P:(jj + 1) * _P],
                                           moving=yn[:, j, 1, :], accumulate=(j > 0))
                    nisa.tensor_scalar(dst=gqk[:, g, 0, :], data=dqacc, op0=nl.multiply, operand0=scale)   # bf16 dQ_g
                for j in range(g0 + ng):
                    fg = j - g0                                   # first query tile of the group that sees key tile j
                    if fg < 0:
                        fg = 0
                    dv_ps = nl.ndarray((_P, D), dtype=f32, buffer=nl.psum)
                    for gg in range(fg, ng):
                        g = g0 + gg
                        nisa.nc_matmul(dst=dv_ps, stationary=pg[:, gg * wg + j * _P:gg * wg + (j + 1) * _P],
                                       moving=don[:, g * D:(g + 1) * D], accumulate=(gg > fg))
                    if _V81 and j >= g0:                          # [v81] first touch of key tile j: a copy on ScalarE
                        nisa.tensor_copy(dst=dvacc[:, j * D:(j + 1) * D], src=dv_ps, engine=nisa.engine.scalar)
                    else:
                        nisa.tensor_tensor(dst=dvacc[:, j * D:(j + 1) * D], data1=dvacc[:, j * D:(j + 1) * D],
                                           data2=dv_ps, op=nl.add)
                    dk_ps = nl.ndarray((_P, D), dtype=f32, buffer=nl.psum)
                    for gg in range(fg, ng):
                        g = g0 + gg
                        nisa.nc_matmul(dst=dk_ps, stationary=dsg[:, gg * wg + j * _P:gg * wg + (j + 1) * _P],
                                       moving=yn[:, g, 0, :], accumulate=(gg > fg))
                    if _V81 and j >= g0:                          # [v81] first touch: the scale on ScalarE
                        nisa.tensor_scalar(dst=dkacc[:, j * D:(j + 1) * D], data=dk_ps, op0=nl.multiply,
                                           operand0=scale, engine=nisa.engine.scalar)
                    else:
                        nisa.scalar_tensor_tensor(dst=dkacc[:, j * D:(j + 1) * D], data=dk_ps, op0=nl.multiply,
                                                  operand0=scale, op1=nl.add, operand1=dkacc[:, j * D:(j + 1) * D])
            # --- outputs: dV straight into dqkv (v7's dv3 store at column 2C), bf16 dK into the stash
            if _V81:                                              # [v81] two wide casts on ScalarE, ONE dV DMA
                nisa.tensor_copy(dst=gqk[:, :, 1, :], src=dkacc.reshape_dim(1, (G, D)), engine=nisa.engine.scalar)
                dvt_all = nl.ndarray((_P, G, D), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dvt_all, src=dvacc.reshape_dim(1, (G, D)), engine=nisa.engine.scalar)
                nisa.dma_copy(dst=dqkv3.ap(pattern=[[C3, _P], [_P * C3, G], [1, D]], offset=b * T * C3 + 2 * C + h * D), src=dvt_all)
            else:
                for j in range(G):
                    nisa.tensor_copy(dst=gqk[:, j, 1, :], src=dkacc[:, j * D:(j + 1) * D])                     # bf16 dK_j
                    dvt = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dvt, src=dvacc[:, j * D:(j + 1) * D])
                    nisa.dma_copy(dst=dqkv3.ap(pattern=[[C3, _P], [1, D]], offset=(b * T + j * _P) * C3 + 2 * C + h * D), src=dvt)
            # --- the QK-norm Jacobian and the inverse rotation on the whole q|k tile (the rope backward, verbatim):
            if not _AUD_JAC:
                # ABLATION PATH: skip the Jacobian and the inverse rotation; store the raw q|k gradient.
                gx_ab = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_copy(dst=gx_ab, src=gqk)
                _rope_qk_dma(gx_ab, dqkv3, b, h, T, C, G, D, True)
                continue
            #     dot = sum_d g*r ; coef = dot * inv^2 / D ; u = g - coef * r (fp32) ; gr = inv * u (bf16) ; gx = R^T gr
            # [qkn_dt] the two wide scratch tiles.  Their only consumers are the fp32 row-sum (sq) and the
            #          bf16 `gr` (u), so bf16 here keeps every term and costs one extra rounding of a value
            #          that is rounded to bf16 before it leaves this block in any case.
            jdt = f32 if _QKN_DT == "fp32" else dt
            sq = nl.ndarray((_P, G, 2, D), dtype=jdt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=gqk, data2=rr, op=nl.multiply)
            dot = nl.ndarray((_P, G, 2, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=dot, op=nl.add, data=sq, axis=3, keepdims=True)
            coef = nl.ndarray((_P, G, 2, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=coef, data1=inv, data2=inv, op=nl.multiply)
            nisa.tensor_tensor(dst=coef, data1=coef, data2=dot, op=nl.multiply)
            nisa.tensor_scalar(dst=coef, data=coef, op0=nl.multiply, operand0=1.0 / D)
            u = nl.ndarray((_P, G, 2, D), dtype=jdt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=u, data1=rr, data2=coef.broadcast(3, D), op=nl.multiply)
            nisa.tensor_tensor(dst=u, data1=gqk, data2=u, op=nl.subtract, engine=nisa.engine.unknown)
            gr = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=gr, data1=u, data2=inv.broadcast(3, D), op=nl.multiply)
            gx = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            tmp_a = nl.ndarray((_P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
            tmp_b = nl.ndarray((_P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
            _rope_rotate(gx, gr, cb, sb, tmp_a, tmp_b, Dh, True)
            _rope_qk_dma(gx, dqkv3, b, h, T, C, G, D, True)
        return dqkv3


    # ---- [v8] the backward with transposed, 512-wide-moving dQ / dV / dK accumulation (build_attn_v8.py: derived from _attn_rope_bwd_kernel by exact edits)
    @nki.jit
    def _attn_rope_bwd_kernel_v8(qkv3, r5, inv3, o3, do3, lse3, cos2, sin2, scale):
        """[v8] _attn_rope_bwd_kernel with dQ, dV and dK accumulated transposed (head dim on partitions) by
        512-wide moving operands from the group stashes (the NKI library's form): the same products in the same
        accumulation order, the outputs transposed back to the base's layouts. Needs BWD_GROUP % KCHUNK == 0 and V81.
        dqkv3[B, T, 3C] = dq | dk | dv (qkv3.dtype) from the projection output qkv3, the forward's r5 / inv3 (or,
        with SAVE=x, a recompute from qkv3), o3 / do3 [B, T, C], lse3 [B*H, 128, G] and the tables.  The v7 grouped
        backward (BWD_GROUP >= 2, SBUF accumulators) with the norm Jacobian + inverse rotation on its bf16 dQ / dK."""
        B, T, C3 = qkv3.shape
        C = C3 // 3
        D = _P
        H = C // D
        Dh = D // 2
        bs = B * H
        G = T // _P
        f32 = nl.float32
        dt = qkv3.dtype
        dqkv3 = nl.ndarray((B, T, C3), dtype=dt, buffer=nl.shared_hbm)
        pid, nprg = _shard()
        cb, sb = _rope_tables(cos2, sin2, G, Dh)
        for i in range(pid, bs, nprg):
            b = i // H
            h = i % H
            # --- r = RoPE(q|k) and inv = 1/rms: saved by the forward (SAVE=r) or recomputed exactly as it did (SAVE=x)
            rr = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            if _ROPE_SAVE == "r":
                _rope_r_dma(rr, r5, b, h, B, H, T, G, D, False)
                inv = nl.ndarray((_P, G, 2, 1), dtype=f32, buffer=nl.sbuf)
                nisa.dma_copy(dst=inv, src=inv3[i, :, :, :, :])
            else:
                xqk = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
                _rope_qk_dma(xqk, qkv3, b, h, T, C, G, D, False)
                tmp_a0 = nl.ndarray((_P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
                tmp_b0 = nl.ndarray((_P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
                _rope_rotate(rr, xqk, cb, sb, tmp_a0, tmp_b0, Dh, False)
                sq0 = nl.ndarray((_P, G, 2, D), dtype=f32, buffer=nl.sbuf)
                inv = _rope_inv_rms(rr, sq0, G, D)
            yn = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)  # the normed q|k rows, the same bits as the forward's
            nisa.tensor_tensor(dst=yn, data1=rr, data2=inv.broadcast(3, D), op=nl.multiply)
            # --- the v7 backward prologue: natural O / dO tiles, the transposed operands, -lse, D[q]
            on = _load_nat_bthd(o3, b, h, T, C, D, G)
            don = _load_nat_bthd(do3, b, h, T, C, D, G)
            qT = _rope_transpose(yn, 0, T, D, G)                  # [D, T]
            kT = _rope_transpose(yn, 1, T, D, G)
            doT = _transpose_nat(don, T, D, G)
            vbase = b * T * C3 + 2 * C + h * D
            if _ROPE_VT == "dma":
                vT = nl.ndarray((D, T), dtype=dt, buffer=nl.sbuf)
                nisa.dma_transpose(dst=vT.ap([[T, D], [1, 1], [1, 1], [1, T]]),
                                   src=qkv3.ap([[C3, T], [1, 1], [1, 1], [1, D]], offset=vbase))
            else:
                vn = nl.ndarray((_P, G * D), dtype=dt, buffer=nl.sbuf)
                nisa.dma_copy(dst=vn, src=qkv3.ap(pattern=[[C3, _P], [_P * C3, G], [1, D]], offset=vbase))
                vT = _transpose_nat(vn, T, D, G)
            nb = nl.ndarray((_P, G), dtype=f32, buffer=nl.sbuf)  # -lse, the exp bias per query
            nisa.dma_copy(dst=nb, src=lse3[i, :, :])
            nisa.tensor_scalar(dst=nb, data=nb, op0=nl.multiply, operand0=-1.0)
            dsum = nl.ndarray((_P, G), dtype=f32, buffer=nl.sbuf)  # D[q] = sum_d dO * O
            if _V81:                                              # [v81] D[q] as ONE product + ONE reduce
                tmp_all = nl.ndarray((_P, G * D), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=tmp_all, data1=don, data2=on, op=nl.multiply)
                nisa.tensor_reduce(dst=dsum, op=nl.add, data=tmp_all.reshape_dim(1, (G, D)), axis=2)
            else:
                for g in range(G):
                    tmp = nl.ndarray((_P, D), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=tmp, data1=don[:, g * D:(g + 1) * D], data2=on[:, g * D:(g + 1) * D],
                                       op=nl.multiply)
                    nisa.tensor_reduce(dst=dsum[:, g:g + 1], op=nl.add, data=tmp, axis=1)
            dkaccT = nl.ndarray((D, T), dtype=f32, buffer=nl.sbuf)   # [v8] dK^T, dV^T: head dim on partitions
            dvaccT = nl.ndarray((D, T), dtype=f32, buffer=nl.sbuf)
            zst = nl.ndarray((_P, _P), dtype=dt, buffer=nl.sbuf)     # [v8] a zero stationary: opens a PSUM sum
            nisa.memset(zst, value=0.0)
            gqk = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)   # the bf16 dQ (s = 0) / dK (s = 1) tiles v7 stores
            # --- the v7 grouped loop (query tiles in groups of _BWD_GROUP; P and dS stashed; per key tile one fresh
            #     PSUM accumulator over the group folded once into the SBUF accumulators, scale folded into the dK fold)
            for g0 in range(0, G, _BWD_GROUP):
                ng = G - g0
                if ng > _BWD_GROUP:
                    ng = _BWD_GROUP
                wg = (g0 + ng) * _P                               # key range of the group's last query tile
                pg = nl.ndarray((_P, ng * wg), dtype=dt, buffer=nl.sbuf)
                dsg = nl.ndarray((_P, ng * wg), dtype=dt, buffer=nl.sbuf)
                J = g0 + ng                                       # [v8] key tiles the group sees
                dsTg = nl.ndarray((_P, J, ng * _P), dtype=dt, buffer=nl.sbuf)   # [v8] dS^T blocks [k, key tile, query tiles]
                for gg in range(ng):
                    g = g0 + gg
                    for j0 in range(0, g + 1, _KCHUNK):
                        nj = g + 1 - j0
                        if nj > _KCHUNK:
                            nj = _KCHUNK
                        w = nj * _P
                        cbk = gg * wg + j0 * _P                   # this chunk's columns in the group stash
                        s = nl.ndarray((_P, w), dtype=f32, buffer=nl.psum)
                        nisa.nc_matmul(dst=s, stationary=qT[:, g * _P:(g + 1) * _P],
                                       moving=kT[:, j0 * _P:j0 * _P + w], accumulate=False)
                        nisa.activation(dst=pg[:, cbk:cbk + w], op=nl.exp, data=s, bias=nb[:, g:g + 1], scale=scale)
                        if j0 + nj - 1 == g:                      # diagonal block: keep key f <= query p
                            db = cbk + (nj - 1) * _P
                            nisa.affine_select(dst=pg[:, db:db + _P], pattern=[[-1, _P]], channel_multiplier=1,
                                               offset=0, cmp_op=nl.greater_equal,
                                               on_true_tile=pg[:, db:db + _P], on_false_value=0.0)
                        dp = nl.ndarray((_P, w), dtype=f32, buffer=nl.psum)
                        nisa.nc_matmul(dst=dp, stationary=doT[:, g * _P:(g + 1) * _P],
                                       moving=vT[:, j0 * _P:j0 * _P + w], accumulate=False)
                        nisa.scalar_tensor_tensor(dst=dsg[:, cbk:cbk + w], data=dp, op0=nl.subtract,
                                                  operand0=dsum[:, g:g + 1], op1=nl.multiply, operand1=pg[:, cbk:cbk + w])
                        dst_ps = nl.ndarray((_P, nj, _P), dtype=dt, buffer=nl.psum)
                        for jj in range(nj):
                            nisa.nc_transpose(dst=dst_ps[:, jj, :],
                                              data=dsg[:, cbk + jj * _P:cbk + (jj + 1) * _P])
                        # [v8] the chunk's dS^T blocks (key tiles j0.., query tile gg) into the stash: one copy
                        nisa.tensor_copy(dst=dsTg[:, j0:j0 + nj, gg * _P:(gg + 1) * _P], src=dst_ps, engine=_copy_engine())
                # [v8] dQ^T of the group [D, ng * 128]: one PSUM sum over the key tiles (ascending, the base's order),
                #      stationary the normed K rows [k, D], moving the stash row of key tile j (512 columns per load)
                dqT_ps = nl.ndarray((D, ng * _P), dtype=f32, buffer=nl.psum)
                for j in range(J):
                    fg = j - g0                                   # first query tile of the group that sees key tile j
                    if fg < 0:
                        fg = 0
                    nisa.nc_matmul(dst=dqT_ps[:, fg * _P:ng * _P], stationary=yn[:, j, 1, :],
                                   moving=dsTg[:, j, fg * _P:ng * _P], accumulate=(j > 0))
                dqT_bf = nl.ndarray((D, ng * _P), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=dqT_bf, data=dqT_ps, op0=nl.multiply, operand0=scale)   # bf16 dQ^T, scaled
                dq_ps = nl.ndarray((_P, ng, D), dtype=dt, buffer=nl.psum)
                for gg in range(ng):
                    nisa.nc_transpose(dst=dq_ps[:, gg, :], data=dqT_bf[:, gg * _P:(gg + 1) * _P])
                nisa.tensor_copy(dst=gqk[:, g0:g0 + ng, 0, :], src=dq_ps, engine=_copy_engine())   # the stash's dQ rows
                # [v8] dV^T and dK^T [D, chunk] per key chunk of the group: stationary dO_g / Q_g [q, D], moving the
                #      stash rows of P / dS (up to 512 columns), the group's query tiles ascending, one fold per chunk
                for c0 in range(0, J, _KCHUNK):
                    nc = J - c0
                    if nc > _KCHUNK:
                        nc = _KCHUNK
                    wc = nc * _P
                    kc = c0 * _P                                  # the chunk's first key
                    diag = (c0 + nc > g0)                         # the chunk holds keys of the group's own query tiles
                    dvT_ps = nl.ndarray((D, wc), dtype=f32, buffer=nl.psum)
                    dkT_ps = nl.ndarray((D, wc), dtype=f32, buffer=nl.psum)
                    if diag:                                      # open the full width with zeros (0 + x = x)
                        nisa.nc_matmul(dst=dvT_ps, stationary=zst, moving=kT[:, 0:wc], accumulate=False)
                        nisa.nc_matmul(dst=dkT_ps, stationary=zst, moving=kT[:, 0:wc], accumulate=False)
                    for gg in range(ng):
                        g = g0 + gg
                        w = (g + 1) * _P - kc                     # keys of this chunk that query tile g sees
                        if w > wc:
                            w = wc
                        nisa.nc_matmul(dst=dvT_ps[:, 0:w], stationary=don[:, g * D:(g + 1) * D],
                                       moving=pg[:, gg * wg + kc:gg * wg + kc + w], accumulate=(diag or gg > 0))
                        nisa.nc_matmul(dst=dkT_ps[:, 0:w], stationary=yn[:, g, 0, :],
                                       moving=dsg[:, gg * wg + kc:gg * wg + kc + w], accumulate=(diag or gg > 0))
                    if c0 >= g0:                                  # first touch of these key tiles: the base's ScalarE copy / scale
                        nisa.tensor_copy(dst=dvaccT[:, kc:kc + wc], src=dvT_ps, engine=nisa.engine.scalar)
                        nisa.tensor_scalar(dst=dkaccT[:, kc:kc + wc], data=dkT_ps, op0=nl.multiply,
                                           operand0=scale, engine=nisa.engine.scalar)
                    else:
                        nisa.tensor_tensor(dst=dvaccT[:, kc:kc + wc], data1=dvaccT[:, kc:kc + wc], data2=dvT_ps, op=nl.add)
                        nisa.scalar_tensor_tensor(dst=dkaccT[:, kc:kc + wc], data=dkT_ps, op0=nl.multiply,
                                                  operand0=scale, op1=nl.add, operand1=dkaccT[:, kc:kc + wc])
            # --- outputs [v8]: dK^T, dV^T cast to bf16 on ScalarE (the base's rounding), transposed back to the natural
            #     rows through PSUM (TensorE, bit-accurate), then the base's single dV DMA; bf16 dK into the stash
            dkT_bf = nl.ndarray((D, T), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dkT_bf, src=dkaccT, engine=nisa.engine.scalar)
            dvT_bf = nl.ndarray((D, T), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dvT_bf, src=dvaccT, engine=nisa.engine.scalar)
            dvt_all = nl.ndarray((_P, G, D), dtype=dt, buffer=nl.sbuf)
            for g0 in range(0, G, 4):
                ng = G - g0
                if ng > 4:
                    ng = 4
                kps = nl.ndarray((_P, ng, D), dtype=dt, buffer=nl.psum)
                vps = nl.ndarray((_P, ng, D), dtype=dt, buffer=nl.psum)
                for gg in range(ng):
                    g = g0 + gg
                    nisa.nc_transpose(dst=kps[:, gg, :], data=dkT_bf[:, g * _P:(g + 1) * _P])
                    nisa.nc_transpose(dst=vps[:, gg, :], data=dvT_bf[:, g * _P:(g + 1) * _P])
                nisa.tensor_copy(dst=gqk[:, g0:g0 + ng, 1, :], src=kps, engine=_copy_engine())
                nisa.tensor_copy(dst=dvt_all[:, g0:g0 + ng, :], src=vps, engine=_copy_engine())
            nisa.dma_copy(dst=dqkv3.ap(pattern=[[C3, _P], [_P * C3, G], [1, D]], offset=b * T * C3 + 2 * C + h * D), src=dvt_all)
            # --- the QK-norm Jacobian and the inverse rotation on the whole q|k tile (the rope backward, verbatim):
            if not _AUD_JAC:
                # ABLATION PATH: skip the Jacobian and the inverse rotation; store the raw q|k gradient.
                gx_ab = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_copy(dst=gx_ab, src=gqk)
                _rope_qk_dma(gx_ab, dqkv3, b, h, T, C, G, D, True)
                continue
            #     dot = sum_d g*r ; coef = dot * inv^2 / D ; u = g - coef * r (fp32) ; gr = inv * u (bf16) ; gx = R^T gr
            # [qkn_dt] the two wide scratch tiles.  Their only consumers are the fp32 row-sum (sq) and the
            #          bf16 `gr` (u), so bf16 here keeps every term and costs one extra rounding of a value
            #          that is rounded to bf16 before it leaves this block in any case.
            jdt = f32 if _QKN_DT == "fp32" else dt
            sq = nl.ndarray((_P, G, 2, D), dtype=jdt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=gqk, data2=rr, op=nl.multiply)
            dot = nl.ndarray((_P, G, 2, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=dot, op=nl.add, data=sq, axis=3, keepdims=True)
            coef = nl.ndarray((_P, G, 2, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=coef, data1=inv, data2=inv, op=nl.multiply)
            nisa.tensor_tensor(dst=coef, data1=coef, data2=dot, op=nl.multiply)
            nisa.tensor_scalar(dst=coef, data=coef, op0=nl.multiply, operand0=1.0 / D)
            u = nl.ndarray((_P, G, 2, D), dtype=jdt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=u, data1=rr, data2=coef.broadcast(3, D), op=nl.multiply)
            nisa.tensor_tensor(dst=u, data1=gqk, data2=u, op=nl.subtract, engine=nisa.engine.unknown)
            gr = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=gr, data1=u, data2=inv.broadcast(3, D), op=nl.multiply)
            gx = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            tmp_a = nl.ndarray((_P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
            tmp_b = nl.ndarray((_P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
            _rope_rotate(gx, gr, cb, sb, tmp_a, tmp_b, Dh, True)
            _rope_qk_dma(gx, dqkv3, b, h, T, C, G, D, True)
        return dqkv3

    # ---- [pl] the fused kernels software-pipelined across (b, h) pairs (build_attn_pl.py; the same instructions on
    #      the same data as the kernels above, only the program order across consecutive pairs of a program changes)
    # ---- [a9m] lane A9M (work/p2/a9m/tools/build_a9m.py): each program's pairs are i = pid + nprg * k over the
    #      fixed trip count k < bs // nprg (B x H a multiple of the program count: the caller's guard), in place of
    #      the materialised pair list the device's parser frontend refused (A9's line 1027); nothing else changes
    # ---- [attr] lane ATTR (work/p2/attr/tools/build_attr_kernel.py): this forward block of nki_attn_a9m.py 76ded7fa
    #      beside v8's backward, with the R1_NKI_ATTN_PRO branches (one V load and one output store a pair)

    def _pl_rotate(dst, src, cb, sb, tmp_a, tmp_b, Dh, transpose, part):
        """_rope_rotate's six VectorE instructions in two halves (part 0: the first output half, part 1: the second):
        the same instructions in the same order."""
        D = 2 * Dh
        s1, s2 = src[:, :, :, 0:Dh], src[:, :, :, Dh:D]
        d1, d2 = dst[:, :, :, 0:Dh], dst[:, :, :, Dh:D]
        if part == 0:
            nisa.tensor_tensor(dst=tmp_a, data1=s1, data2=cb, op=nl.multiply)
            nisa.tensor_tensor(dst=tmp_b, data1=s2, data2=sb, op=nl.multiply)
            nisa.tensor_tensor(dst=d1, data1=tmp_a, data2=tmp_b, op=nl.subtract if transpose else nl.add,
                               engine=nisa.engine.unknown)
        else:
            nisa.tensor_tensor(dst=tmp_a, data1=s2, data2=cb, op=nl.multiply)
            nisa.tensor_tensor(dst=tmp_b, data1=s1, data2=sb, op=nl.multiply)
            nisa.tensor_tensor(dst=d2, data1=tmp_a, data2=tmp_b, op=nl.add if transpose else nl.subtract,
                               engine=nisa.engine.unknown)

    def _pl_transpose_group(xt, src4, s, nat, T, D, G, g0, engine):
        """One group of up to four TensorE transposes through PSUM into xt[D, T] (the inner step of _rope_transpose /
        _rope_transpose_v81 on src4[P, G, 2, D] with s = q | k, or of _transpose_nat on the natural tile nat[P, G*D])."""
        ng = G - g0
        if ng > 4:
            ng = 4
        ps = nl.ndarray((D, ng * _P), dtype=xt.dtype, buffer=nl.psum)
        for gg in range(ng):
            g = g0 + gg
            if nat is None:
                nisa.nc_transpose(dst=ps[:, gg * _P:(gg + 1) * _P], data=src4[:, g, s, :])
            else:
                nisa.nc_transpose(dst=ps[:, gg * _P:(gg + 1) * _P], data=nat[:, g * D:(g + 1) * D])
        nisa.tensor_copy(dst=xt[:, g0 * _P:(g0 + ng) * _P], src=ps, engine=engine)

    # forward prologue tiles: XQK 0, TA 1, TB 2, R 3, SQ 4, SS 5, INV 6, YN 7, QT 8, KT 9, VAUG 10
    def _pl_fwd_pro(s, st, i, qkv3, r5, inv3, cb, sb, B, H, T, C, G, D, Dh, dt, f32, vsel):
        """Stage s (0 to 8) of pair i's forward prologue into the tile list st: the base prologue's instructions in
        the base's order, cut into nine pieces. [attr] vsel: None, or (R1_NKI_ATTN_PRO) the pair's pre-filled V_aug
        tile, which stage 6 loads with one DMA."""
        b = i // H
        h = i % H
        if s == 0:
            st[0] = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            _rope_qk_dma(st[0], qkv3, b, h, T, C, G, D, False)
        elif s == 1:
            st[1] = nl.ndarray((_P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
            st[2] = nl.ndarray((_P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
            st[3] = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            _pl_rotate(st[3], st[0], cb, sb, st[1], st[2], Dh, False, 0)
        elif s == 2:
            _pl_rotate(st[3], st[0], cb, sb, st[1], st[2], Dh, False, 1)
        elif s == 3:
            st[4] = nl.ndarray((_P, G, 2, D), dtype=f32, buffer=nl.sbuf)
            st[5] = nl.ndarray((_P, G, 2, 1), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=st[4], op=nl.square, data=st[3])
        elif s == 4:
            nisa.tensor_reduce(dst=st[5], op=nl.add, data=st[4], axis=3, keepdims=True)
            st[6] = nl.ndarray((_P, G, 2, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=st[5], data=st[5], op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=_ROPE_EPS)
            nisa.tensor_scalar(dst=st[6], data=st[5], op0=nl.rsqrt, operand0=0.0)
        elif s == 5:
            st[7] = nl.ndarray((_P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=st[7], data1=st[3], data2=st[6].broadcast(3, D), op=nl.multiply)   # the one rounding
            if _ROPE_SAVE == "r":
                _rope_r_dma(st[3], r5, b, h, B, H, T, G, D, True)
            nisa.dma_copy(dst=inv3[i, :, :, :, :], src=st[6])
        elif s == 6:
            if vsel is None:
                st[10] = _rope_vaug(qkv3, b, h, T, C, G, D)
            else:                                                 # [pro] the pre-filled V_aug tile, one DMA
                st[10] = _pro_vload(vsel, qkv3, b, h, T, C, G, D)
        elif s == 7:
            st[8] = nl.ndarray((D, T), dtype=dt, buffer=nl.sbuf)
            for g0 in range(0, G, 4):
                _pl_transpose_group(st[8], st[7], 0, None, T, D, G, g0, nisa.engine.vector)
        elif s == 8:
            st[9] = nl.ndarray((D, T), dtype=dt, buffer=nl.sbuf)
            for g0 in range(0, G, 4):
                _pl_transpose_group(st[9], st[7], 1, None, T, D, G, g0, nisa.engine.vector)

    @nki.jit
    def _attn_rope_fwd_kernel_pl(qkv3, cos2, sin2, scale, shift):
        """[pl] _attn_rope_fwd_kernel software-pipelined across pairs: the next pair's prologue (stages of _pl_fwd_pro)
        is issued inside this pair's query-tile loop, stage s at query tile (s * G) // 9, into its own fresh tiles. The
        same instructions on the same data: o3, lse3, r5, inv3 bit for bit the base's. Needs V81 and ROPE_SAVE r."""
        B, T, C3 = qkv3.shape
        C = C3 // 3
        D = _P
        H = C // D
        Dh = D // 2
        bs = B * H
        G = T // _P
        W = D + 1
        f32 = nl.float32
        dt = qkv3.dtype
        o3 = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
        lse3 = nl.ndarray((bs, _P, G), dtype=f32, buffer=nl.shared_hbm)
        inv3 = nl.ndarray((bs, _P, G, 2, 1), dtype=f32, buffer=nl.shared_hbm)
        if _ROPE_SAVE == "r":
            r5 = nl.ndarray((2, B, H, T, D), dtype=dt, buffer=nl.shared_hbm)
        else:
            r5 = nl.ndarray((2, 1, 1, 1, 1), dtype=dt, buffer=nl.shared_hbm)   # placeholder, never read
        pid, nprg = _shard()
        cb, sb = _rope_tables(cos2, sin2, G, Dh)                 # once per program
        nbias = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)   # the fixed shift as an exp bias tile
        nisa.memset(nbias, value=-shift)
        vb0 = None
        vb1 = None
        if _PRO:                                                  # [pro] the program's two V_aug and output tiles
            vb0, vb1, ob0, ob1 = _pro_bufs(dt, G, D)
        npair = bs // nprg                                        # [a9m] this program's pairs: pid + nprg * k
        NS = 9
        st = [None] * 11
        if npair > 0:
            for s in range(NS):                                   # the first pair's prologue, whole
                _pl_fwd_pro(s, st, pid, qkv3, r5, inv3, cb, sb, B, H, T, C, G, D, Dh, dt, f32, vb0)
        for k in range(npair):
            i = pid + nprg * k
            b = i // H
            h = i % H
            has_next = k + 1 < npair
            stn = [None] * 11
            j = i + nprg if has_next else i
            vn = None
            obuf = None
            if _PRO:                                              # [pro] pair k's output tile, pair k + 1's V_aug tile
                vn = vb0 if (k + 1) % 2 == 0 else vb1
                obuf = ob0 if k % 2 == 0 else ob1
            qT = st[8]
            kT = st[9]
            vaug = st[10]
            lse_blk = nl.ndarray((_P, G), dtype=f32, buffer=nl.sbuf)
            sums = nl.ndarray((_P, G), dtype=f32, buffer=nl.sbuf)   # [v81] the row sums, gathered for ONE log
            for g in range(G):
                if has_next:                                      # [pl] the next pair's prologue, stage by stage
                    for s in range(NS):
                        if (s * G) // NS == g:
                            _pl_fwd_pro(s, stn, j, qkv3, r5, inv3, cb, sb, B, H, T, C, G, D, Dh, dt, f32, vn)
                oacc = nl.ndarray((_P, W), dtype=f32, buffer=nl.psum)
                for j0 in range(0, g + 1, _KCHUNK):
                    nj = g + 1 - j0
                    if nj > _KCHUNK:
                        nj = _KCHUNK
                    st_ps = nl.ndarray((_P, nj * _P), dtype=f32, buffer=nl.psum)
                    for jj in range(nj):
                        jt = j0 + jj
                        nisa.nc_matmul(dst=st_ps[:, jj * _P:(jj + 1) * _P], stationary=kT[:, jt * _P:(jt + 1) * _P],
                                       moving=qT[:, g * _P:(g + 1) * _P], accumulate=False)
                    et = nl.ndarray((_P, nj * _P), dtype=dt, buffer=nl.sbuf)
                    nisa.activation(dst=et, op=nl.exp, data=st_ps, bias=nbias, scale=scale)
                    if j0 + nj - 1 == g:                          # diagonal block: keep key p <= query f
                        jj = nj - 1
                        nisa.affine_select(dst=et[:, jj * _P:(jj + 1) * _P], pattern=[[1, _P]], channel_multiplier=-1,
                                           offset=0, cmp_op=nl.greater_equal,
                                           on_true_tile=et[:, jj * _P:(jj + 1) * _P], on_false_value=0.0)
                    for jj in range(nj):
                        jt = j0 + jj
                        nisa.nc_matmul(dst=oacc, stationary=et[:, jj * _P:(jj + 1) * _P],
                                       moving=vaug[:, jt * W:(jt + 1) * W], accumulate=(jt > 0))
                rs = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rs, data=oacc[:, D:D + 1])
                if _PRO:                                          # [pro] the normalised tile lands in the pair's tile
                    ot = obuf[:, g, :]
                else:
                    ot = nl.ndarray((_P, D), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=ot, data=oacc[:, 0:D], op0=nl.multiply, operand0=rs, engine=nisa.engine.vector)
                nisa.tensor_copy(dst=sums[:, g:g + 1], src=oacc[:, D:D + 1], engine=nisa.engine.vector)
                if not _PRO:                                      # [pro] one store a pair, after the loop
                    nisa.dma_copy(dst=o3.ap(pattern=[[C, _P], [1, D]], offset=(b * T + g * _P) * C + h * D), src=ot)
            if _PRO:                                              # [pro] the pair's output, one DMA
                _pro_ostore(obuf, o3, b, h, T, C, G, D)
            nisa.activation(dst=lse_blk, op=nl.log, data=sums)    # [v81] ONE log over the gathered sums
            nisa.tensor_scalar(dst=lse_blk, data=lse_blk, op0=nl.add, operand0=shift, engine=nisa.engine.vector)
            nisa.dma_copy(dst=lse3[i, :, :], src=lse_blk)
            if has_next:
                st = stn
        return o3, lse3, r5, inv3

    _HAS_NKI = True
except Exception:
    _HAS_NKI = False


# ----------------------------------------------------------------------------- torch emulation of the kernels

def _causal_mask(T, device):
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def emulate_forward(q4, k4, v4, scale, shift):
    """The forward kernel's arithmetic in torch (bf16 roundings at the kernel's points): o3 [B, T, H*D], lse3."""
    B, H, T, D = q4.shape
    bs = B * H
    G = T // _P
    dt = q4.dtype
    q3, k3, v3 = q4.reshape(bs, T, D), k4.reshape(bs, T, D), v4.reshape(bs, T, D)
    qf, kf, vf = q3.float(), k3.float(), v3.float()
    s = torch.matmul(kf, qf.transpose(1, 2))                              # [bs, k, q] fp32 (PSUM)
    e = torch.exp(scale * s - shift).to(dt)                               # bf16 E^T
    e = e * _causal_mask(T, q3.device).transpose(0, 1).to(dt)             # keep k <= q
    vaug = torch.cat((vf, torch.ones(bs, T, 1, dtype=torch.float32, device=q3.device)), dim=-1)
    oacc = torch.matmul(e.float().transpose(1, 2), vaug)                  # [bs, q, D+1] fp32
    rs = 1.0 / oacc[..., D:]
    o = (oacc[..., :D] * rs).to(dt)
    lse = torch.log(oacc[..., D]) + shift                                 # [bs, T]
    o3 = o.view(B, H, T, D).transpose(1, 2).contiguous().view(B, T, H * D)
    lse3 = lse.view(bs, G, _P).transpose(1, 2).contiguous()               # [bs, p, g]
    return o3, lse3


def emulate_backward(q4, k4, v4, o3, do3, lse3, scale):
    B, H, T, D = q4.shape
    bs = B * H
    G = T // _P
    dt = q4.dtype
    q3, k3, v3 = q4.reshape(bs, T, D), k4.reshape(bs, T, D), v4.reshape(bs, T, D)
    qf, kf, vf = q3.float(), k3.float(), v3.float()
    of = o3.view(B, T, H, D).permute(0, 2, 1, 3).reshape(bs, T, D).float()
    dof = do3.view(B, T, H, D).permute(0, 2, 1, 3).reshape(bs, T, D).float()
    lse = lse3.transpose(1, 2).reshape(bs, T)                             # [bs, q]
    s = torch.matmul(qf, kf.transpose(1, 2))                              # [bs, q, k]
    p = torch.exp(scale * s - lse[..., None]).to(dt)
    p = p * _causal_mask(T, q3.device).to(dt)
    dp = torch.matmul(dof, vf.transpose(1, 2))                            # fp32
    dsum = (dof * of).sum(-1)                                             # [bs, q]
    ds = ((dp - dsum[..., None]) * p.float()).to(dt)
    dv = torch.matmul(p.float().transpose(1, 2), dof).to(dt)              # [bs, k, D]
    dk = (scale * torch.matmul(ds.float().transpose(1, 2), qf)).to(dt)
    dq = (scale * torch.matmul(ds.float(), kf)).to(dt)
    dv3 = dv.view(B, H, T, D).transpose(1, 2).contiguous().view(B, T, H * D)
    return dq.contiguous().view(B, H, T, D), dk.contiguous().view(B, H, T, D), dv3


# ----------------------------------------------------------------------------- [rope_attn] torch emulation of the fused kernels

def _emu_rope_fwd(x, cos2, sin2):
    """The RoPE+QK-norm kernel's forward arithmetic (nki_rope_norm_opt.emulate_forward with its defaults) on
    x [B, T, H, D]: r = R(x) with the products and sums rounded in the input dtype (returned as fp32 values),
    inv = rsqrt(mean r^2 + eps) in fp32 [B, T, H, 1], y = (r * inv) rounded once, as [B, H, T, D]."""
    B, T, H, D = x.shape
    Dh = D // 2
    c = cos2.reshape(1, T, 1, Dh).to(x.dtype)
    s = sin2.reshape(1, T, 1, Dh).to(x.dtype)
    x1, x2 = x[..., :Dh], x[..., Dh:]
    r = torch.cat((x1 * c + x2 * s, x2 * c - x1 * s), dim=-1).float()
    ss = r.square().sum(dim=-1, keepdim=True)
    inv = torch.rsqrt(ss * (1.0 / D) + _ROPE_EPS)
    y = (r * inv).to(x.dtype).transpose(1, 2).contiguous()
    return r, inv, y


def _emu_rope_bwd(r, inv, gy, cos2, sin2, dtype):
    """The RoPE+QK-norm kernel's backward arithmetic given its recomputed r [B, T, H, D] (fp32 values of the
    input-dtype rotation) and inv [B, T, H, 1]; gy is the normed-side gradient [B, H, T, D]; returns [B, T, H, D]."""
    B, T, H, D = r.shape
    Dh = D // 2
    c = cos2.reshape(1, T, 1, Dh).to(dtype)
    s = sin2.reshape(1, T, 1, Dh).to(dtype)
    g = gy.transpose(1, 2).float()
    sq = g * r                         # [qkn_dt] the kernel's `sq` tile: fp32 today, the input dtype under bf16
    if _QKN_DT == "bf16":
        sq = sq.to(dtype).float()
    dot = sq.sum(dim=-1, keepdim=True)
    coef = dot * (inv * inv) * (1.0 / D)
    u = coef * r                       # [qkn_dt] the kernel's `u` tile is written twice, so it rounds twice
    if _QKN_DT == "bf16":
        u = u.to(dtype).float()
    u = (g - u).float()
    if _QKN_DT == "bf16":
        u = u.to(dtype).float()
    gr = (u * inv).to(dtype)
    g1, g2 = gr[..., :Dh], gr[..., Dh:]
    gx = torch.cat((g1 * c - g2 * s, g2 * c + g1 * s), dim=-1)
    return gx.to(dtype)


def _emu_inv3(inv_q, inv_k, B, T, H):
    """inv3[i, p, g, s, 0] = inv_s[b, g*128 + p, h] for pair i = b*H + h (the kernel's layout)."""
    G = T // _P
    parts = []
    for inv in (inv_q, inv_k):
        parts.append(inv.reshape(B, G, _P, H).permute(0, 3, 2, 1).reshape(B * H, _P, G))
    return torch.stack(parts, dim=-1).unsqueeze(-1).contiguous()


def _emu_inv_from3(inv3, s, B, T, H):
    G = T // _P
    return inv3[..., s, 0].reshape(B, H, _P, G).permute(0, 3, 2, 1).reshape(B, T, H, 1)


def emulate_rope_forward(qkv3, cos2, sin2, scale, shift):
    """The fused forward's arithmetic in torch: (o3, lse3, r5, inv3) from qkv3 [B, T, 3C]."""
    B, T, C3 = qkv3.shape
    C = C3 // 3
    D = _P
    H = C // D
    q, k, v = qkv3.split(C, dim=-1)
    rq, invq, yq = _emu_rope_fwd(q.reshape(B, T, H, D), cos2, sin2)
    rk, invk, yk = _emu_rope_fwd(k.reshape(B, T, H, D), cos2, sin2)
    v4 = v.reshape(B, T, H, D).transpose(1, 2).contiguous()
    o3, lse3 = emulate_forward(yq, yk, v4, scale, shift)
    r5 = torch.stack((rq.transpose(1, 2), rk.transpose(1, 2)), dim=0).to(qkv3.dtype).contiguous()
    return o3, lse3, r5, _emu_inv3(invq, invk, B, T, H)


def emulate_rope_backward(qkv3, r5, inv3, o3, do3, lse3, cos2, sin2, scale):
    """The fused backward's arithmetic in torch: dqkv3 [B, T, 3C]; r and inv from the forward's r5 / inv3 (SAVE=r)
    or recomputed from qkv3 (SAVE=x), exactly as the kernel does."""
    B, T, C3 = qkv3.shape
    C = C3 // 3
    D = _P
    H = C // D
    dt = qkv3.dtype
    q, k, v = qkv3.split(C, dim=-1)
    if _ROPE_SAVE == "r":
        rq = r5[0].float().transpose(1, 2)
        rk = r5[1].float().transpose(1, 2)
        invq = _emu_inv_from3(inv3, 0, B, T, H)
        invk = _emu_inv_from3(inv3, 1, B, T, H)
    else:
        rq, invq, _ = _emu_rope_fwd(q.reshape(B, T, H, D), cos2, sin2)
        rk, invk, _ = _emu_rope_fwd(k.reshape(B, T, H, D), cos2, sin2)
    yq = (rq * invq).to(dt).transpose(1, 2).contiguous()
    yk = (rk * invk).to(dt).transpose(1, 2).contiguous()
    v4 = v.reshape(B, T, H, D).transpose(1, 2).contiguous()
    dq, dk, dv3 = emulate_backward(yq, yk, v4, o3, do3, lse3, scale)
    gxq = _emu_rope_bwd(rq, invq, dq, cos2, sin2, dt)
    gxk = _emu_rope_bwd(rk, invk, dk, cos2, sin2, dt)
    return torch.cat((gxq.reshape(B, T, C), gxk.reshape(B, T, C), dv3), dim=-1).contiguous()


# ----------------------------------------------------------------------------- torch-side wrapper

def kernel_available(device):
    return (EMULATE or (_HAS_NKI and device.type == "neuron")) and not _KERNEL_DISABLED


def _disable_kernel(where, err):
    global _KERNEL_DISABLED, _LAST_KERNEL_ERROR
    if not _KERNEL_DISABLED:
        _KERNEL_DISABLED = True
        _LAST_KERNEL_ERROR = f"{where}: {type(err).__name__}: {err}"
        print(f"[nki_attn] kernel disabled after failure in {where}: {type(err).__name__}: {err}", flush=True)


def _launch(kernel):
    return kernel[_LNC] if _LNC == 2 else kernel


def _fwd_kernel():
    return emulate_forward if EMULATE else _launch(_attn_fwd_kernel)


def _bwd_kernel():
    return emulate_backward if EMULATE else _launch(_attn_bwd_kernel)


def shape_ok(B, H, T, D):
    return B > 0 and H > 0 and 0 < D <= _P and T > 0 and T % _P == 0 and T <= _MAX_T


def fixed_shift(scale, D):
    """c = scale * D: the bound on |scale * q . k| after the parameter-free QK-norm (|q|, |k| <= sqrt(D))."""
    return float(scale) * D + _SHIFT_MARGIN


class NkiAttnFn(torch.autograd.Function):
    """y[B, T, H, D] = causal SDPA(q, k, v) via the NKI kernels; eager math if a kernel raises."""

    @staticmethod
    def forward(ctx, q, k, v, scale, shift):
        B, H, T, D = q.shape
        ctx.scale, ctx.shift = scale, shift
        ctx.fallback = False
        try:
            if _BATCH_SPLIT > 1 and B % _BATCH_SPLIT == 0:          # compile-time escape hatch: smaller unrolled loops
                n = B // _BATCH_SPLIT
                parts = [_fwd_kernel()(q[s:s + n], k[s:s + n], v[s:s + n], float(scale), float(shift))
                         for s in range(0, B, n)]
                o3 = torch.cat([p[0] for p in parts], dim=0)
                lse3 = torch.cat([p[1] for p in parts], dim=0)
            else:
                o3, lse3 = _fwd_kernel()(q, k, v, float(scale), float(shift))
            o = o3.view(B, T, H, D)
        except Exception as err:
            _disable_kernel("forward", err)
            ctx.fallback = True
            ctx.save_for_backward(q, k, v)
            with torch.no_grad():
                return eager_causal_attention(q, k, v, scale)
        ctx.save_for_backward(q, k, v, o, lse3)
        return o

    @staticmethod
    def backward(ctx, gy):
        if ctx.fallback:
            q, k, v = ctx.saved_tensors
            gq, gk, gv = _eager_backward(q, k, v, gy, ctx.scale)
            return gq, gk, gv, None, None
        q, k, v, o, lse3 = ctx.saved_tensors
        B, H, T, D = q.shape
        C = H * D
        if not _KERNEL_DISABLED:
            try:
                go = gy.contiguous().view(B, T, C)                        # [B, T, H, D] (the block's own layout)
                o3 = o.view(B, T, C)
                if _BATCH_SPLIT > 1 and B % _BATCH_SPLIT == 0:
                    n = B // _BATCH_SPLIT
                    parts = [_bwd_kernel()(q[s:s + n], k[s:s + n], v[s:s + n], o3[s:s + n], go[s:s + n],
                                           lse3[s * H:(s + n) * H], float(ctx.scale)) for s in range(0, B, n)]
                    dq = torch.cat([p[0] for p in parts], dim=0)
                    dk = torch.cat([p[1] for p in parts], dim=0)
                    dv3 = torch.cat([p[2] for p in parts], dim=0)
                else:
                    dq, dk, dv3 = _bwd_kernel()(q, k, v, o3, go, lse3, float(ctx.scale))
                dv = dv3.view(B, T, H, D).permute(0, 2, 1, 3)             # what v.transpose(1, 2)'s backward wants
                return dq, dk, dv, None, None
            except Exception as err:
                _disable_kernel("backward", err)
        gq, gk, gv = _eager_backward(q, k, v, gy, ctx.scale)
        return gq, gk, gv, None, None


def _kernel_ok(q, k, v):
    if not USE_NKI_ATTN or not kernel_available(q.device):
        return False
    if q.dim() != 4 or q.shape != k.shape or q.shape != v.shape:
        return False
    if not (q.dtype == k.dtype == v.dtype) or q.dtype not in (torch.bfloat16, torch.float32):
        return False
    if q.dtype == torch.float32 and not ALLOW_FP32:
        return False
    B, H, T, D = q.shape
    return shape_ok(B, H, T, D)


def nki_causal_attention(q, k, v, scale=None):
    """Causal SDPA(q, k, v) for [B, H, T, D] inputs, returned as [B, T, H, D]; NKI kernels when applicable."""
    sc = float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    if not _kernel_ok(q, k, v):
        return eager_causal_attention(q, k, v, sc if scale is not None else None)
    q = q if q.is_contiguous() else q.contiguous()
    k = k if k.is_contiguous() else k.contiguous()
    v = v if v.is_contiguous() else v.contiguous()
    return NkiAttnFn.apply(q, k, v, sc, fixed_shift(sc, q.shape[-1]))


# ----------------------------------------------------------------------------- [rope_attn] torch-side wrapper of the fused kernels

def _rope_fwd_kernel():
    return emulate_rope_forward if EMULATE else _launch(_attn_rope_fwd_kernel)


def _rope_bwd_kernel():
    return emulate_rope_backward if EMULATE else _launch(_attn_rope_bwd_kernel)


def _tables_2d(cos, sin, T, Dh, dtype):
    """train.py's [1, T, 1, D/2] tables as contiguous [T, D/2] in the kernel dtype (nki_rope_norm_opt._tables_2d)."""
    c2 = cos.reshape(T, Dh)
    s2 = sin.reshape(T, Dh)
    if c2.dtype != dtype:
        c2 = c2.to(dtype)
    if s2.dtype != dtype:
        s2 = s2.to(dtype)
    if not c2.is_contiguous():
        c2 = c2.contiguous()
    if not s2.is_contiguous():
        s2 = s2.contiguous()
    return c2, s2


def _eager_rope_norm(x, cos2, sin2):
    """train.py's norm(apply_rotary_emb(x, cos, sin)).transpose(1, 2) on x [B, T, H, D] (the eager fallback)."""
    B, T, H, D = x.shape
    Dh = D // 2
    c = cos2.reshape(1, T, 1, Dh)
    s = sin2.reshape(1, T, 1, Dh)
    x1, x2 = x[..., :Dh], x[..., Dh:]
    r = torch.cat((x1 * c + x2 * s, x1 * (-s) + x2 * c), dim=-1)
    return F.rms_norm(r, (D,)).transpose(1, 2)


def _eager_rope_attention(qkv, cos2, sin2, H, scale):
    """The eager chain on the projection output: RoPE + QK-norm on q and k, causal SDPA, [B, T, H, D] out."""
    B, T, C3 = qkv.shape
    C = C3 // 3
    D = C // H
    q, k, v = qkv.split(C, dim=-1)
    q = _eager_rope_norm(q.reshape(B, T, H, D), cos2, sin2)
    k = _eager_rope_norm(k.reshape(B, T, H, D), cos2, sin2)
    v = v.reshape(B, T, H, D).transpose(1, 2)
    return eager_causal_attention(q, k, v, scale)


def _eager_rope_backward(qkv, cos2, sin2, H, scale, gy):
    with torch.enable_grad():
        qr = qkv.detach().requires_grad_(True)
        y = _eager_rope_attention(qr, cos2, sin2, H, scale)
        (gqkv,) = torch.autograd.grad(y, qr, gy)
    return gqkv


def _rope_fwd_launch(qkv, cos2, sin2, H, scale, shift):
    B = qkv.shape[0]
    if _BATCH_SPLIT > 1 and B % _BATCH_SPLIT == 0:          # compile-time escape hatch: smaller unrolled loops
        n = B // _BATCH_SPLIT
        parts = [_rope_fwd_kernel()(qkv[s:s + n], cos2, sin2, float(scale), float(shift)) for s in range(0, B, n)]
        o3 = torch.cat([p[0] for p in parts], dim=0)
        lse3 = torch.cat([p[1] for p in parts], dim=0)
        r5 = torch.cat([p[2] for p in parts], dim=1) if _ROPE_SAVE == "r" else parts[0][2]
        inv3 = torch.cat([p[3] for p in parts], dim=0)
        return o3, lse3, r5, inv3
    return _rope_fwd_kernel()(qkv, cos2, sin2, float(scale), float(shift))


def _rope_bwd_launch(qkv, r5, inv3, o3, go, lse3, cos2, sin2, H, scale):
    B = qkv.shape[0]
    if _BATCH_SPLIT > 1 and B % _BATCH_SPLIT == 0:
        n = B // _BATCH_SPLIT
        parts = []
        for s in range(0, B, n):
            r5s = r5[:, s:s + n] if _ROPE_SAVE == "r" else r5
            parts.append(_rope_bwd_kernel()(qkv[s:s + n], r5s, inv3[s * H:(s + n) * H], o3[s:s + n], go[s:s + n],
                                            lse3[s * H:(s + n) * H], cos2, sin2, float(scale)))
        return torch.cat(parts, dim=0)
    return _rope_bwd_kernel()(qkv, r5, inv3, o3, go, lse3, cos2, sin2, float(scale))


class NkiRopeAttnFn(torch.autograd.Function):
    """y[B, T, H, D] = causal SDPA(QKnorm(RoPE(q)), QKnorm(RoPE(k)), v) from qkv [B, T, 3C] via the fused NKI
    kernels; dqkv [B, T, 3C] from the fused backward; the eager math if a kernel raises."""

    @staticmethod
    def forward(ctx, qkv, cos2, sin2, H, scale, shift):
        B, T, C3 = qkv.shape
        C = C3 // 3
        D = C // H
        ctx.scale, ctx.shift, ctx.H = scale, shift, H
        ctx.fallback = False
        try:
            o3, lse3, r5, inv3 = _rope_fwd_launch(qkv, cos2, sin2, H, scale, shift)
            o = o3.view(B, T, H, D)
        except Exception as err:
            _disable_kernel("rope forward", err)
            ctx.fallback = True
            ctx.save_for_backward(qkv, cos2, sin2)
            with torch.no_grad():
                return _eager_rope_attention(qkv, cos2, sin2, H, scale)
        if _ROPE_SAVE == "r":
            ctx.save_for_backward(qkv, cos2, sin2, r5, inv3, o, lse3)
        else:
            ctx.save_for_backward(qkv, cos2, sin2, inv3, o, lse3)
        return o

    @staticmethod
    def backward(ctx, gy):
        if ctx.fallback:
            qkv, cos2, sin2 = ctx.saved_tensors
            return _eager_rope_backward(qkv, cos2, sin2, ctx.H, ctx.scale, gy), None, None, None, None, None
        if _ROPE_SAVE == "r":
            qkv, cos2, sin2, r5, inv3, o, lse3 = ctx.saved_tensors
        else:
            qkv, cos2, sin2, inv3, o, lse3 = ctx.saved_tensors
            r5 = inv3                                             # placeholder argument, never read by the kernel
        B, T, C3 = qkv.shape
        C = C3 // 3
        if not _KERNEL_DISABLED:
            try:
                go = gy.contiguous().view(B, T, C)                # [B, T, H, D] (the block's own layout)
                o3 = o.view(B, T, C)
                dqkv = _rope_bwd_launch(qkv, r5, inv3, o3, go, lse3, cos2, sin2, ctx.H, ctx.scale)
                return dqkv.view(B, T, C3), None, None, None, None, None
            except Exception as err:
                _disable_kernel("rope backward", err)
        return _eager_rope_backward(qkv, cos2, sin2, ctx.H, ctx.scale, gy), None, None, None, None, None


def rope_attention_ok(qkv, cos, sin, n_head):
    """True when the fused path applies: switch on, the attention kernel usable, a contiguous [B, T, 3C] bf16 (fp32
    with R1_NKI_ATTN_FP32=1) projection output with D == 128, T a multiple of 128 up to 4096, tables for T tokens,
    and the grouped backward (the fused backward is the v7 grouped loop)."""
    if not USE_NKI_ATTN_ROPE or not USE_NKI_ATTN or not kernel_available(qkv.device):
        return False
    if _BWD_GROUP < 2 or _BWD_ACC != "sbuf":
        return False
    if qkv.dim() != 3 or qkv.dtype not in (torch.bfloat16, torch.float32):
        return False
    if qkv.dtype == torch.float32 and not ALLOW_FP32:
        return False
    B, T, C3 = qkv.shape
    if C3 % 3 or n_head <= 0:
        return False
    C = C3 // 3
    if C % n_head or C // n_head != _P:
        return False
    if not shape_ok(B, n_head, T, _P) or not qkv.is_contiguous():
        return False
    Dh = _P // 2
    return cos.numel() == T * Dh and sin.numel() == T * Dh


def nki_rope_attention(qkv, cos, sin, n_head, scale=None):
    """y [B, T, H, D] = causal SDPA(QKnorm(RoPE(q)), QKnorm(RoPE(k)), v) straight from the fused projection output
    qkv [B, T, 3C] (one NKI launch each way, dqkv written as one [B, T, 3C] tensor); None when the fused path does
    not apply, so the caller keeps today's rope_norm -> nki_causal_attention chain."""
    if not rope_attention_ok(qkv, cos, sin, n_head):
        return None
    D = _P
    sc = float(scale) if scale is not None else 1.0 / math.sqrt(D)
    cos2, sin2 = _tables_2d(cos, sin, qkv.shape[1], D // 2, qkv.dtype)
    return NkiRopeAttnFn.apply(qkv, cos2, sin2, n_head, sc, fixed_shift(sc, D))


__all__ = ["nki_causal_attention", "NkiAttnFn", "eager_causal_attention", "emulate_forward", "emulate_backward",
           "kernel_available", "shape_ok", "fixed_shift", "USE_NKI_ATTN",
           "nki_rope_attention", "NkiRopeAttnFn", "rope_attention_ok", "emulate_rope_forward", "emulate_rope_backward",
           "USE_NKI_ATTN_ROPE"]
