"""
Trainium Frontier, Phase 1 submission (team Them Bones). Single-file train.py built from our working copy with
every experiment switch resolved to the submitted configuration and the measurement-only code removed.

Reviewer notes
  * Model: 12-layer, 1024-wide decoder (RMSNorm, RoPE + QK-norm, ReLU^2 MLP with hidden 6x, logit softcap 15,
    untied embedding / head), U-net style residual skips between encoder and decoder blocks, bf16 activations,
    bf16 Linear weights with hardware stochastic rounding (fp32 elsewhere).
  * Attention sublayers sit in 9 of the 12 blocks (0,2,3,4,5,6,7,9,10); each of the other blocks adds a learned
    causal one-token-shift mixer in place of its attention output add (x = x + mix_lambda * shift1(x),
    where shift1 moves every row to the next position, so position t reads only t-1 -- causal by
    construction; mix_lambda is a per-block scalar, zero-initialised, in the AdamW scalar group).
  * Optimizer: Muon (Polar Express Newton-Schulz orthogonalisation, rank-sharded: reduce-scatter, orthogonalise a
    quarter of each stack per rank, all-gather) for the matrices, AdamW for embeddings / head / scalar gains;
    global batch 262,144 tokens with a half-batch ramp for the first 30% of the clock, linear warmdown over the
    last 60% to 2% of peak, weight decay 0.05.
  * Objective: next-token cross-entropy; during the first half of the clock, auxiliary losses on the tokens
    two and three ahead are added, scored on the same logits together with the next-token loss by one
    multi-target loss-head kernel pair (no extra parameters; its first call is checked against the
    single-target kernels before the clock starts) and annealed to zero by half the run (the t+3 weight
    over the first quarter of the clock, the t+2 weight over the second), so the second half trains, and
    the checkpoint is, a plain next-token model.  Evaluation is unchanged.
  * Data: only the provided ClimbMix shards through the shipped loader/tokenizer; documents are best-fit packed
    into 8,192-token rows and sliced into 1,024-token training windows, the same windowing evaluate_bpb() uses.
  * Timer: prepare.record_step_boundary() is called once per optimizer step; the run's own budget clock starts
    after the fifth optimizer step (the excluded startup / compilation allowance, per the organizers' ruling on
    the forum) and training stops before 1,800 s minus a checkpoint reserve. Nothing here shortens the eval.
  * NKI kernels (each a drop-in for the eager PyTorch expression it replaces, with an exact eager fallback and a
    correctness test against eager on the chip): fused RoPE + QK-norm, ReLU^2, fused softcap + log-softmax +
    cross-entropy loss head, fused residual-gain + RMSNorm block head (with the U-net skip add and the previous
    block's MLP-tail residual add folded in, and the x0 / skip gradient accumulations folded into its backward),
    the second RMSNorm with its residual add, the last residual add + final RMSNorm, causal attention (fixed-shift
    softmax, flash-style backward; it reads q, k, v from the QKV projection output and applies the rotary embedding
    and the parameter-free QK RMS-norm in SBUF before the causal softmax, and its backward returns one [B, T, 3C]
    gradient - the separate RoPE kernel is the fallback path), the embedding backward as a one-hot contraction,
    and the one-token-shift mixer fused with its residual add and the second RMSNorm (one launch each way;
    the plain ops are its fallback, and a self-check against them runs on the first training step, before
    the clock starts).  They are inlined below with per-module name prefixes.
  * load_for_eval() rebuilds the identical model from the checkpoint's model_config and runs the identical
    forward (same kernels, same fallbacks). No behavior depends on __name__ beyond the usual main guard, and
    nothing inspects whether it is being evaluated.
  * Environment: the only environment values read are NEURON_LOGICAL_NC_CONFIG (launch geometry, from the launch
    command), NEURON_RT_STOCHASTIC_ROUNDING_EN (set by main() for the bf16 master weights) and torchrun's
    RANK, which the fused mixer reads only to print its one-time diagnostics from rank 0.
"""
from __future__ import annotations
import os
os.environ.setdefault('NEURON_CC_FLAGS', '--enable-saturate-infinity --enable-fast-loading-neuron-binaries')
os.environ.setdefault('NEURON_RT_STOCHASTIC_ROUNDING_EN', '0')
import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
_PROCESS_STARTED = time.monotonic()
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from prepare import EVAL_TIMEOUT_SECONDS, TOKENIZER_VOCAB_SIZE, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS, TRAIN_STARTUP_STEPS_EXCLUDED, TRAIN_TIME_BUDGET_SECONDS, Tokenizer, causality_check, ensure_tokenizer, evaluate_bpb, get_dist_info, make_dataloader, print0, record_step_boundary
SEQ_LEN = 1024
ROPE_TABLE_LEN = 4096
DEPTH = 12
ASPECT_RATIO = 64
HEAD_DIM = 128
N_HEAD = 6
_NARROW = {'fused': 0}
TOTAL_BATCH_SIZE = 262144
DEVICE_BATCH_SIZE = 32
PACK_WINDOWS = 8
LONG_DOC_CHUNKS = False
REPLAY = False
REPLAY_ALPHA = 1.0
REPLAY_FLOOR = 0.25
NUM_STEPS = 10 ** 9
MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS
CHECKPOINT_RESERVE_SECONDS = 10.0
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0
LOGIT_SOFTCAP = 15.0
CE_PATH = 'static'
BF16_MASTER = True
WKN = True
HEAD_DW_NSPLIT = 2
VALUE_EMB = 'off'
UNET_SKIP = True
ATTN_GATE = 0
ATTN_SCALE = 'fixed'
ATTN_BLOCKS = '0,2,3,4,5,6,7,9,10'
MIXER = 'shift1'
MIXER_FUSED = True
ROPE_BASE = 100000
TIE_EMBEDDINGS = False
TIE_LR = 0.0424
TIE_INIT_STD = 0.001
_QL_FIRED: dict = {}
MLP_RATIO = 6
BATCH_RAMP = True
BATCH_RAMP_FRAC = 0.3
QKV_STACK_VIEW = True
BIGRAM_BUCKETS = 0
BIGRAM_MODE = 'input'
BIGRAM_INIT = 'normal'
SCHED_ENDPOINT = 'budget'
MUON_SHARD = True
MUON_SHARD_RS = 'auto'
MUON_SHARD_AG = 'auto'
MUON_SHARD_GATHER = 'params'
MUON_SHARD_SLAB = 0
ADAMW_STACK = '1'
MUON_MOMENTUM_SCHED = 'nanochat'
RAMP_LR_COUPLING = 0
ADAM_BETAS_BUNDLE = False
QK_SCALE = 0.11
INIT_RESID = 1.0
INIT_X0 = 0.05
INIT_SKIP = 1.0
ADAMW_WD_EMB = 0.001
ADAMW_WD_HEAD = 0.01
MUON_BETA2 = 0.9
MUON_NS_STEPS = 5
GRADNORM_STACK = '0'
NS_TENSOR_COEFS = '1'
MUON_PERSISTENT_VIEWS = 2
ZERO_GRAD_NONE = True
GRAD_ACCUM_INPLACE = 'all'
GAIN_BATCH = True
CAST_CACHE = '1'
EMB_SHADOW = True
_ACCUM_TAGS = frozenset((t.strip() for t in GRAD_ACCUM_INPLACE.split(',') if t.strip()))
MUON_ELEMENTWISE_FUSE = True
FUSE_DENY = frozenset((s for s in ''.split(',') if s))
EMA_DECAY = 0.0
EMA_START_FRAC = 0.0
EMA_SAVE = 'ema'
EMBEDDING_LR = 0.3
UNEMBEDDING_LR = 0.006
MATRIX_LR = 0.06508
SCALAR_LR = 0.05
WEIGHT_DECAY = 0.05
RESID_DROPOUT = 0.0
WARMUP_STEPS = 20
WARMDOWN_RATIO = 0.6
FINAL_LR_FRAC = 0.02
GRAD_CLIP = 0.0
LOG_EVERY = 10
_OPT_TAIL_TAGS = frozenset(('mask', 'vnorm', 'hostgate', 'stage'))
OPT_TAIL_MASK = 'mask' in _OPT_TAIL_TAGS
OPT_TAIL_VNORM = 'vnorm' in _OPT_TAIL_TAGS
OPT_TAIL_HOSTGATE = 'hostgate' in _OPT_TAIL_TAGS
OPT_TAIL_STAGE = 'stage' in _OPT_TAIL_TAGS
MUON_NS_AFOLD = True
MUON_NS_ALPHA = False
_OPTTAIL_FIRED = {'mask': 0, 'vnorm': 0, 'hostgate': 0, 'stage': 0, 'afold': 0, 'alpha': 0}
_FP32M = {'n': 0, 'bytes': 0}

def neuron_available() -> bool:
    try:
        torch.device('neuron')
        return True
    except Exception:
        return False

def detect_device_type(requested: str='') -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if neuron_available():
        return 'neuron'
    return 'cpu'

def init_runtime(device_type: str):
    assert device_type in {'cpu', 'cuda', 'neuron'}
    torch.manual_seed(42)
    ddp, rank, local_rank, world_size = get_dist_info()
    if ddp and device_type == 'cuda':
        device = torch.device('cuda', local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group('nccl', device_id=device)
    elif ddp and device_type == 'neuron':
        dist.init_process_group('neuron')
        device = torch.device('neuron')
    elif ddp:
        dist.init_process_group('gloo')
        device = torch.device('cpu')
    else:
        device = torch.device(device_type)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return (ddp, rank, local_rank, world_size, device)

def cleanup_runtime() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'neuron' and hasattr(torch, 'neuron'):
        torch.neuron.synchronize()

def compute_dtype_for(device: torch.device) -> torch.dtype:
    if '' in ('fp32', 'float32'):
        return torch.float32
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == 'neuron':
        return torch.bfloat16
    return torch.float32

@dataclass
class GPTConfig:
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    mlp_ratio: int = 6
    value_emb: str = 'off'
    unet_skip: bool = False
    attn_scale: str = 'fixed'
    bigram_buckets: int = 0
    bigram_mode: str = 'input'
    wkn: bool = False
    attn_gate: int = 0
    attn_blocks: str = '0,2,3,4,5,6,7,9,10'
    rope_base: int = 100000
    tie_embeddings: bool = False
    mixer: str = 'shift1'
    qk_scale: float = 0.0
    head_dim: int = 0

def attn_block_ids(config: GPTConfig) -> tuple:
    """[R1 arch A9] The block indices that own an attention sublayer, parsed from the config field (never from the
    environment: the scorer's eval process has neither env vars nor CLI args, and rebuilding a DIFFERENT set of
    blocks than was trained is exactly what hard rule 3 forbids).  "all" is the default and the value a checkpoint
    written before this field existed loads with, so an old checkpoint rebuilds as the full-attention model it was.
    Raises on anything else it cannot parse: a mask that is silently wrong is the failure this whole design exists
    to prevent."""
    spec = str(config.attn_blocks).strip()
    if spec == 'all':
        return tuple(range(config.n_layer))
    if spec == 'none':
        return ()
    ids: list = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok.isdigit():
            raise ValueError(f"attn_blocks={config.attn_blocks!r}: {tok!r} is not a block index (expected 'all', 'none' or e.g. '0,2,4,6,8')")
        i = int(tok)
        if not 0 <= i < config.n_layer:
            raise ValueError(f'attn_blocks={config.attn_blocks!r}: block {i} out of range for n_layer={config.n_layer}')
        if i in ids:
            raise ValueError(f'attn_blocks={config.attn_blocks!r}: block {i} listed twice')
        ids.append(i)
    return tuple(sorted(ids))

def attn_dims(config: GPTConfig) -> tuple:
    """[R1 narrow] (head_dim, width) of the attention, from the CONFIG (never the environment: load_for_eval rebuilds
    exactly the trained geometry): head_dim 0 = n_embd // n_head; width = n_head x head_dim.  An earlier checkpoint's
    config gives (n_embd // n_head, n_embd): today's."""
    d = config.head_dim or config.n_embd // config.n_head
    return (d, config.n_head * d)

def attn_geometry(config: GPTConfig) -> tuple:
    """[R1 narrow] attn_dims, validated where an attention sublayer is built (today's assert head_dim x n_head == n_embd is
    its unset-field case).  Raises on what this lane does not build: an unset head_dim with n_embd not a multiple of n_head
    (the model_config of a narrow run stripped of its head_dim field: rule 3 must fail loudly, never rebuild a different
    model), a width above n_embd, and the value embedding or the [C, C] output gate on a narrow geometry (both are n_embd
    wide by construction)."""
    if not config.head_dim and config.n_embd % config.n_head:
        raise ValueError(f'n_embd={config.n_embd} is not a multiple of n_head={config.n_head} and head_dim is unset: the model_config of a narrow-attention run without its head_dim field cannot rebuild the trained model')
    d, w = attn_dims(config)
    if not 0 < w <= config.n_embd:
        raise ValueError(f'attention geometry n_head={config.n_head} head_dim={d}: the width {w} must be in (0, n_embd={config.n_embd}]')
    if w != config.n_embd and (config.value_emb != 'off' or config.attn_gate):
        raise ValueError(f'value_emb={config.value_emb!r} / attn_gate={config.attn_gate} with a narrow attention (width {w}): refused (both are n_embd wide)')
    return (d, w)

def narrow_desc(model) -> str:
    """[R1 narrow] the attention geometry of the CONSTRUCTED model (never the env): the first attention block's projection
    shapes and the c_proj class (LinearT = stored [in, out])."""
    a = next((b.attn for b in model.transformer.h if b.attn is not None), None)
    d, w = model.attn_dims
    if a is None:
        return f'n_head={model.config.n_head} head_dim={d} width={w} (no attention block)'
    return f'n_head={model.config.n_head} head_dim={d} width={w} c_q {list(a.c_q.weight.shape)} c_k {list(a.c_k.weight.shape)} c_v {list(a.c_v.weight.shape)} c_proj {('LinearT' if isinstance(a.c_proj, LinearT) else 'Linear')} {list(a.c_proj.weight.shape)}'

def mixer_mode(config: GPTConfig) -> str:
    """[R1 arch A12] Validate the mixer field.  Same contract as attn_block_ids: parsed from the CONFIG, never
    the environment (the scorer's eval process has neither env vars nor CLI args), and anything unparseable
    raises at model construction rather than training a silently different architecture than the label claims."""
    m = str(config.mixer).strip()
    if m not in ('off', 'shift1'):
        raise ValueError(f"mixer={config.mixer!r}: expected 'off' or 'shift1'")
    return m

def shift1(x: torch.Tensor) -> torch.Tensor:
    """[R1 arch A12] Causal one-token shift over the sequence dim of [B, T, C]: output row t is input row t-1,
    row 0 is zeros.  Causal BY CONSTRUCTION -- position t reads only position t-1 < t -- so the causality gate
    cannot be perturbed by it.  Zero matmul FLOPs: one zeros row, one slice, one concat."""
    return torch.cat((x.new_zeros(x.shape[0], 1, x.shape[2]), x[:, :-1, :]), dim=1)
COMPUTE_DTYPE: torch.dtype = torch.float32 if 'bf16' in ('fp32', 'float32') else torch.bfloat16

def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))
_NKI_PARTITION = 128
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def _relu_sq_fwd(x):
        """Fused y = relu(x)**2 for a 2-D [M, N] tensor, tiled over rows."""
        M, N = x.shape
        out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
        for t in nl.affine_range((M + _NKI_PARTITION - 1) // _NKI_PARTITION):
            rows = nl.ds(t * _NKI_PARTITION, _NKI_PARTITION)
            xb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[rows])
            rb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rb, data=xb, op0=nl.maximum, operand0=0.0)
            yb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yb, data1=rb, data2=rb, op=nl.multiply)
            nisa.dma_copy(dst=out[rows], src=yb)
        return out

    @nki.jit
    def _relu_sq_bwd(x, grad_out):
        """Backward of relu(x)**2: grad_in = grad_out * 2 * relu(x)."""
        M, N = x.shape
        out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
        for t in nl.affine_range((M + _NKI_PARTITION - 1) // _NKI_PARTITION):
            rows = nl.ds(t * _NKI_PARTITION, _NKI_PARTITION)
            xb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=x[rows])
            gb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=gb, src=grad_out[rows])
            rb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rb, data=xb, op0=nl.maximum, operand0=0.0)
            two_r = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=two_r, data=rb, op0=nl.multiply, operand0=2.0)
            yb = nl.ndarray((_NKI_PARTITION, N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yb, data1=two_r, data2=gb, op=nl.multiply)
            nisa.dma_copy(dst=out[rows], src=yb)
        return out

    class _ReluSquaredNKI(torch.autograd.Function):
        """Makes the NKI relu**2 kernel differentiable so it can train."""

        @staticmethod
        def forward(ctx, x):
            y = _relu_sq_fwd(x.reshape(-1, x.shape[-1])).reshape(x.shape)
            ctx.save_for_backward(x)
            return y

        @staticmethod
        def backward(ctx, grad_out):
            x, = ctx.saved_tensors
            flat_x = x.reshape(-1, x.shape[-1])
            flat_g = grad_out.reshape(-1, x.shape[-1]).contiguous()
            return _relu_sq_bwd(flat_x, flat_g).reshape(x.shape)
    _HAS_NKI = True
except Exception:
    _HAS_NKI = False
USE_NKI_RELU2 = False
'Fused ReLU-squared, y = relu(x)**2, as ONE NKI launch forward and ONE launch backward, sharded over the two\nphysical cores of an LNC=2 logical core.\n\nOptimised replacement for the kit-style example kernel in work/train.py (`_relu_sq_fwd` / `_relu_sq_bwd` /\n`_ReluSquaredNKI`), which runs on ONE physical core and costs 0.73 ms + 1.11 ms per launch at [16384, 4096] bf16\n(14.7 ms per micro-step over 8 blocks; device profile of 2026-09-07, docs/experiments.md "Device-side profile").\n\nPublic entry point (identical to train.py\'s)\n--------------------------------------------\n    y = relu_squared(x)          # any shape; the kernel sees the 2-D view [M = numel // N, N]\n\nand N % 128 == 0; anything else, and any kernel failure (printed once, after which the kernel is disabled for the\nprocess), takes the eager chain F.relu(x).square() and its autograd.\n\nDesign (measurements and reasoning in NOTES.md)\n-----------------------------------------------\n* The kernel is DMA-bound: 256 MiB (forward) / 384 MiB (backward) of HBM traffic per launch at the training shape\n  against ~0.1-0.2 ms of engine time, so the design is the DMA pattern:\n  - rows are viewed as [T, 128, K*N]: partition p of tile t holds the K consecutive rows t*128*K + p*K + k, ONE\n    contiguous K*N*itemsize-byte run per partition (32 KiB at K=4, N=4096, bf16: the ">= 4 KiB per partition,\n    >= 32 KiB per transfer" regime of the NKI DMA bandwidth guide), 4 MiB per DMA;\n  - `kernel[2]` launch with the tile loop split on nl.program_id(0), so the 16 DMA engines of BOTH physical cores\n    move data (a plain call under LNC=2 is ONE physical core: work/agents/kernel_opt/NOTES.md section 2.2);\n  - the tile loop is unrolled at trace time and the compiler\'s allocator keeps several tiles in flight (the\n* Forward: ONE VectorE instruction per slab, scalar_tensor_tensor (x max 0) * x == relu(x)**2, in place over the\n  input slab (fp32 math inside the engine, one rounding on the store: bit-identical to eager bf16).\n* Backward: ScalarE activation relu(2*x) (exact: == 2*relu(x)) in place over x, then ONE VectorE multiply by dy in\n  place over dy; the two engines pipeline across tiles.  Same single rounding as eager\'s autograd\n  (grad * (2 * relu(x)), threshold-masked), so bf16 is bit-identical there too.\n* relu(x) is recomputed from the saved x; y is not saved (same memory as the kit kernel and as eager).\n\nKnobs (env, read once at import; all default to the design above)\n'
import os
import torch
import torch.nn.functional as F
relu2k__P = 128
relu2k__env = os.environ.get
relu2k_USE_NKI_RELU2_OPT = True
relu2k__MAX_K = 4
relu2k__MAX_BYTES = int(str(32 * 1024))
relu2k__LNC = int(relu2k__env('NEURON_LOGICAL_NC_CONFIG', '1'))
relu2k__NBUF = 3
relu2k__H_R32 = False
relu2k__KERNEL_DISABLED = False
relu2k__LAST_KERNEL_ERROR = None
relu2k__H_KERNEL_DISABLED = False
relu2k__H_LAST_KERNEL_ERROR = None

def relu2k_eager_relu_squared(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()

def relu2k__plan(M: int, N: int, itemsize: int):
    """(T, P, K) or None when [M, N] is not tileable (M % 128 or N % 128).  K is the largest divisor of M/128
    within the slab cap that leaves an even number of tiles >= 2 (both programs busy), else the largest leaving
    >= 2 tiles, else 1."""
    if M <= 0 or N <= 0 or M % relu2k__P or N % relu2k__P:
        return None
    n = M // relu2k__P
    kmax = max(1, min(relu2k__MAX_K, relu2k__MAX_BYTES // (N * itemsize), n))
    ks = [k for k in range(1, kmax + 1) if n % k == 0]
    even = [k for k in ks if n // k >= 2 and n // k % 2 == 0]
    two = [k for k in ks if n // k >= 2]
    k = max(even) if even else max(two) if two else 1
    return (n // k, relu2k__P, k)

def relu2k__tile_view(x2: torch.Tensor, plan) -> torch.Tensor:
    T, P, K = plan
    return x2.view(T, P, K * x2.shape[-1])
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    def relu2k__shard():
        """(program_id, num_programs); (0, 1) without a launch grid or with sharding off."""
        if nl.program_ndim() > 0:
            return (nl.program_id(0), nl.num_programs(0))
        return (0, 1)

    def relu2k__my_tiles(T):
        """Tile indices of this program (a Python range, fixed at trace time)."""
        pid, nprg = relu2k__shard()
        return range(pid, T, nprg)

    def relu2k__slab(P, F, dtype, slot, which, nslabs):
        """One [P, F] SBUF slab; with ADDR=1 at an explicit rotating address (slot < NBUF, which < nslabs)."""
        return nl.ndarray((P, F), dtype=dtype, buffer=nl.sbuf)

    def relu2k__step(F):
        """Elements per instruction: the whole slab unless CHUNK divides it."""
        return F

    def relu2k__fwd_compute(xb, yb, rb, F):
        """yb = relu(xb)**2 (yb may alias xb; rb is scratch for the two-instruction modes)."""
        step = relu2k__step(F)
        for a in range(0, F, step):
            x_ = xb[:, a:a + step]
            y_ = yb[:, a:a + step]
            nisa.scalar_tensor_tensor(dst=y_, data=x_, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=x_)

    def relu2k__bwd_compute(xb, gb, ob, rb, F):
        """ob = 2 * relu(xb) * gb (ob may alias gb; rb may alias xb)."""
        step = relu2k__step(F)
        for a in range(0, F, step):
            x_ = xb[:, a:a + step]
            g_ = gb[:, a:a + step]
            o_ = ob[:, a:a + step]
            r_ = rb[:, a:a + step]
            nisa.activation(dst=r_, op=nl.relu, data=x_, scale=2.0)
            nisa.tensor_tensor(dst=o_, data1=r_, data2=g_, op=nl.multiply)

    def relu2k__bwd_h_compute(hb, gb, ob, rb, F):
        """[R1 relu2fold a] ob = 2 * sqrt(hb) * gb -- the ReLU^2 backward rebuilt from the forward
        OUTPUT h = relu(pre)^2 instead of from the saved pre-activation.

        Instruction for instruction this is `act` mode with relu replaced by sqrt: `activation`'s own
        pre-scale carries the *2, because sqrt(4h) == 2*sqrt(h) EXACTLY (4 is a power of two, so 4h is
        exact until h is within 2 bits of the fp32 range; measured h amax on this model is 189-327).
        Same ScalarE + VectorE pair, same engine split, same single rounding on the product.

        sqrt, never rsqrt.  The forward writes h = (pre max 0) * pre, so h is +0.0 or -0.0 across the
        whole non-positive half of pre; IEEE sqrt(-0.0) = -0.0 and the product is a signed zero, which
        is the right gradient.  A reciprocal-sqrt form would give -inf there and 0 * -inf = NaN."""
        step = relu2k__step(F)
        for a in range(0, F, step):
            h_ = hb[:, a:a + step]
            g_ = gb[:, a:a + step]
            o_ = ob[:, a:a + step]
            r_ = rb[:, a:a + step]
            nisa.activation(dst=r_, op=nl.sqrt, data=h_, scale=4.0)
            nisa.tensor_tensor(dst=o_, data1=r_, data2=g_, op=nl.multiply)

    def relu2k__bwd_h_tile(h3, g3, gx3, t, slot, P, F):
        """[R1 relu2fold a] one tile of the sqrt form.  Same slab structure as _bwd_tile: h in slab 0,
        g in slab 1, the output written over g when INPLACE.  The r slab aliases h (the `act` mode
        pattern: activation reads and writes the same slice) unless H_R32 asks for an fp32 r, which
        needs its own allocation and therefore the compiler's allocator (ADDR=1's manual addresses
        assume one itemsize across slabs, so H_R32 is ignored under ADDR=1)."""
        dt = h3.dtype
        nslabs = 2
        hb = relu2k__slab(P, F, dt, slot, 0, nslabs)
        gb = relu2k__slab(P, F, g3.dtype, slot, 1, nslabs)
        nisa.dma_copy(dst=hb, src=h3[t, :, :])
        nisa.dma_copy(dst=gb, src=g3[t, :, :])
        rb = hb
        ob = gb
        relu2k__bwd_h_compute(hb, gb, ob, rb, F)
        nisa.dma_copy(dst=gx3[t, :, :], src=ob)

    def relu2k__fwd_tile(x3, y3, t, slot, P, F):
        dt = x3.dtype
        needs_r = False
        nslabs = 1
        nslabs = 1
        if needs_r:
            nslabs = nslabs + 1
        xb = relu2k__slab(P, F, dt, slot, 0, nslabs)
        nisa.dma_copy(dst=xb, src=x3[t, :, :])
        yb = xb
        if needs_r:
            rb = relu2k__slab(P, F, dt, slot, nslabs - 1, nslabs)
        else:
            rb = xb
        relu2k__fwd_compute(xb, yb, rb, F)
        nisa.dma_copy(dst=y3[t, :, :], src=yb)

    def relu2k__bwd_tile(x3, g3, gx3, t, slot, P, F):
        dt = x3.dtype
        needs_r = True
        nslabs = 2
        xb = relu2k__slab(P, F, dt, slot, 0, nslabs)
        gb = relu2k__slab(P, F, g3.dtype, slot, 1, nslabs)
        nisa.dma_copy(dst=xb, src=x3[t, :, :])
        nisa.dma_copy(dst=gb, src=g3[t, :, :])
        ob = gb
        rb = xb
        relu2k__bwd_compute(xb, gb, ob, rb, F)
        nisa.dma_copy(dst=gx3[t, :, :], src=ob)

    @nki.jit
    def relu2k__relu2_fwd_kernel(x3):
        """y3[T, P, F] = relu(x3)**2, tiles split over the programs of the launch grid."""
        T, P, F = x3.shape
        y3 = nl.ndarray((T, P, F), dtype=x3.dtype, buffer=nl.shared_hbm)
        for t in relu2k__my_tiles(T):
            relu2k__fwd_tile(x3, y3, t, t % relu2k__NBUF, P, F)
        return y3

    @nki.jit
    def relu2k__relu2_bwd_kernel(x3, g3):
        """gx3[T, P, F] = 2 * relu(x3) * g3."""
        T, P, F = x3.shape
        gx3 = nl.ndarray((T, P, F), dtype=x3.dtype, buffer=nl.shared_hbm)
        for t in relu2k__my_tiles(T):
            relu2k__bwd_tile(x3, g3, gx3, t, t % relu2k__NBUF, P, F)
        return gx3

    @nki.jit
    def relu2k__relu2_bwd_h_kernel(h3, g3):
        """[R1 relu2fold a] gx3[T, P, F] = 2 * sqrt(h3) * g3 -- the ReLU^2 backward from the forward
        OUTPUT.  Same tile loop, same launch grid and same HBM traffic as _relu2_bwd_kernel; the only
        difference is which [M, N] tensor the first operand is, which is what lets the caller stop
        writing the pre-activation in the forward."""
        T, P, F = h3.shape
        gx3 = nl.ndarray((T, P, F), dtype=h3.dtype, buffer=nl.shared_hbm)
        for t in relu2k__my_tiles(T):
            relu2k__bwd_h_tile(h3, g3, gx3, t, t % relu2k__NBUF, P, F)
        return gx3
    relu2k__HAS_NKI = True
except Exception:
    relu2k__HAS_NKI = False

def relu2k__disable_kernel(where: str, err: BaseException) -> None:
    global relu2k__KERNEL_DISABLED, relu2k__LAST_KERNEL_ERROR
    if not relu2k__KERNEL_DISABLED:
        relu2k__KERNEL_DISABLED = True
        relu2k__LAST_KERNEL_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[nki_relu2_opt] kernel disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)

def relu2k__launch(kernel):
    """kernel[2] under an LNC=2 runtime (both physical cores run the sharded tile loop), plain call otherwise."""
    return kernel[relu2k__LNC] if relu2k__LNC == 2 else kernel

class relu2k_ReLU2OptFn(torch.autograd.Function):
    """y = relu(x)**2 via the NKI kernels; the eager math if a kernel raises (the kernel is then disabled)."""

    @staticmethod
    def forward(ctx, x, plan):
        ctx.save_for_backward(x)
        ctx.plan = plan
        try:
            y3 = relu2k__launch(relu2k__relu2_fwd_kernel)(relu2k__tile_view(x.reshape(-1, x.shape[-1]), plan))
            return y3.view(x.shape)
        except Exception as err:
            relu2k__disable_kernel('forward', err)
            with torch.no_grad():
                return relu2k_eager_relu_squared(x)

    @staticmethod
    def backward(ctx, gy):
        x, = ctx.saved_tensors
        N = x.shape[-1]
        if not relu2k__KERNEL_DISABLED:
            try:
                g2 = gy.reshape(-1, N)
                if g2.dtype != x.dtype:
                    g2 = g2.to(x.dtype)
                if not g2.is_contiguous():
                    g2 = g2.contiguous()
                gx3 = relu2k__launch(relu2k__relu2_bwd_kernel)(relu2k__tile_view(x.reshape(-1, N), ctx.plan), relu2k__tile_view(g2, ctx.plan))
                return (gx3.view(x.shape), None)
            except Exception as err:
                relu2k__disable_kernel('backward', err)
        with torch.enable_grad():
            xr = x.detach().requires_grad_(True)
            gx, = torch.autograd.grad(relu2k_eager_relu_squared(xr), xr, gy)
        return (gx, None)

def relu2k_relu2_backward(x: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """[R1 gemm] dx = 2 relu(x) gy for a SAVED pre-activation x, outside autograd (the fused c_fc + ReLU^2 Function
    calls this from its own backward): the backward kernel when it applies, else the eager formula (autograd's own:
    the *2 is exact, one rounding on the product, so both are bit-identical to ReLU2OptFn's backward)."""
    if relu2k__HAS_NKI and (not relu2k__KERNEL_DISABLED) and (x.device.type == 'neuron') and (x.dtype in (torch.bfloat16, torch.float32)) and (x.dim() >= 2):
        N = x.shape[-1]
        plan = relu2k__plan(x.numel() // N, N, x.element_size())
        if plan is not None:
            try:
                x2 = x.contiguous().reshape(-1, N)
                g2 = gy.reshape(-1, N)
                if g2.dtype != x.dtype:
                    g2 = g2.to(x.dtype)
                if not g2.is_contiguous():
                    g2 = g2.contiguous()
                return relu2k__launch(relu2k__relu2_bwd_kernel)(relu2k__tile_view(x2, plan), relu2k__tile_view(g2, plan)).view(x.shape)
            except Exception as err:
                relu2k__disable_kernel('backward', err)
    return gy * (2.0 * F.relu(x))

def relu2k__disable_h_kernel(where: str, err: BaseException) -> None:
    """Disables ONLY the sqrt-form kernel (see _H_KERNEL_DISABLED)."""
    global relu2k__H_KERNEL_DISABLED, relu2k__H_LAST_KERNEL_ERROR
    if not relu2k__H_KERNEL_DISABLED:
        relu2k__H_KERNEL_DISABLED = True
        relu2k__H_LAST_KERNEL_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[nki_relu2_opt] sqrt-form kernel disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)

def relu2k_eager_relu2_backward_from_h(h: torch.Tensor, dh: torch.Tensor) -> torch.Tensor:
    """The sqrt-form kernel's arithmetic in torch: 2*sqrt(h) in fp32 (sqrt(4h), the *4 exact), rounded
    to the r slab's dtype unless H_R32, then one rounding on the product -- so this is the reference
    the CPU smoke and every off-chip path run, and it models the kernel rather than idealising it."""
    r = torch.sqrt(h.to(torch.float32) * 4.0)
    r = r.to(h.dtype).to(torch.float32)
    return (r * dh.to(torch.float32)).to(dh.dtype)

def relu2k_relu2_backward_from_h(h: torch.Tensor, dh: torch.Tensor) -> torch.Tensor:
    """[R1 relu2fold a] dpre = 2 relu(pre) dh computed as 2 sqrt(h) dh, from the forward OUTPUT
    h = relu(pre)^2 rather than the saved pre-activation.

    WHY: pre and h are the same [M, N] shape and h has to be saved anyway (the MLP c_proj's dW is
    h^T dy), so saving pre as well is a redundant [M, N] tensor -- 335.5 MB per layer per micro-step
    at device batch 32 / hidden 5120 / bf16.  Dropping it deletes that WRITE in the forward
    (nki_gemm.fused_fc_relu2_forward(..., save_pre=0)) and 335.5 MB of live HBM per layer.  It does
    NOT delete a read: this kernel reads h where the other read pre, byte for byte.

    EXACTNESS: relu(pre) and sqrt(h) agree exactly wherever pre <= 0 (both are a signed zero) and to
    the rounding of h wherever pre > 0.  h = bf16(relu(pre)^2) = relu(pre)^2 (1+d), |d| <= 2^-8, so
    sqrt(h) carries |d|/2 <= 2^-9 = 0.195 %, plus one more bf16 rounding of 2*sqrt(h) unless H_R32.
    Bound and measurement: work/agents/fp8/RELU2FOLD_HANDOVER.md."""
    if relu2k__HAS_NKI and (not relu2k__KERNEL_DISABLED) and (not relu2k__H_KERNEL_DISABLED) and (h.device.type == 'neuron') and (h.dtype in (torch.bfloat16, torch.float32)) and (h.dim() >= 2):
        N = h.shape[-1]
        plan = relu2k__plan(h.numel() // N, N, h.element_size())
        if plan is not None:
            try:
                h2 = h.contiguous().reshape(-1, N)
                g2 = dh.reshape(-1, N)
                if g2.dtype != h.dtype:
                    g2 = g2.to(h.dtype)
                if not g2.is_contiguous():
                    g2 = g2.contiguous()
                return relu2k__launch(relu2k__relu2_bwd_h_kernel)(relu2k__tile_view(h2, plan), relu2k__tile_view(g2, plan)).view(h.shape)
            except Exception as err:
                relu2k__disable_h_kernel('backward_from_h', err)
    return relu2k_eager_relu2_backward_from_h(h, dh)

def relu2k_relu_squared(x: torch.Tensor) -> torch.Tensor:
    """relu(x)**2 -- one NKI launch forward / one backward when the kernel applies, else the eager chain."""
    if relu2k__HAS_NKI and (not relu2k__KERNEL_DISABLED) and (x.device.type == 'neuron') and (x.dtype in (torch.bfloat16, torch.float32)) and (x.dim() >= 2):
        N = x.shape[-1]
        plan = relu2k__plan(x.numel() // N, N, x.element_size())
        if plan is not None:
            if not x.is_contiguous():
                x = x.contiguous()
            return relu2k_ReLU2OptFn.apply(x, plan)
    return relu2k_eager_relu_squared(x)
_relu_squared_opt = relu2k_relu_squared
USE_NKI_RELU2_OPT = relu2k_USE_NKI_RELU2_OPT
"Hand-written NKI bf16 GEMM primitive (rung 1 of the utilization roadmap) and its first consumer, fused c_fc + ReLU^2.\n\nPublic entry points\n-------------------\n    out = nki_matmul(a, b, trans_a=False, trans_b=False)\n        trans_a=False, trans_b=False   out[M, N] = a[M, K] @ b[K, N]         (class A/B; weight as R1_WKN stores it, [K, N])\n        trans_a=False, trans_b=True    out[M, N] = a[M, K] @ b[N, K].T       (class B with the weight as nn.Linear stores it)\n        trans_a=True,  trans_b=False   out[K, N] = a[M, K].T @ b[M, N]       (class C, the dW form: contraction over tokens)\n        bf16 in, fp32 PSUM accumulation, bf16 out.  Any shape the kernel does not take (see `plan_*`), a non-Neuron device,\n        or a kernel failure (printed once, after which the kernel is disabled for the process) gives torch.matmul.\n\n    h, h_pre = fused_fc_relu2_forward(x2, w1)\n        one launch: h_pre = x2 @ w1 (bf16, rounded once from the fp32 PSUM = today's `c_fc` output), h = relu(h_pre)^2\n        (the shipped ReLU^2 kernel's own VectorE instruction on the bf16 h_pre, so h is bit-identical to today's chain\n        given the same h_pre).  fused_fc_relu2.py wraps it into an autograd Function.\n\nDesign (NOTES.md section 1 has the roofline; the numbers below are its predictions)\n--------------------------------------------------------------------------------\n* nc_matmul contracts over the partition axis of BOTH operands, so the operand whose contraction runs along its free\n  axis is transposed on the way in: a[M, K] -> a^T slabs [128 k, tokens] by `dma_transpose` from HBM (arch guide: up to\n  90 % of DMA throughput), or by batched TensorE transposes (XT=pe), or given pre-transposed (XT=pre, the roofline\n  reference).  The weight is resident in SBUF as [128 k, NK, N] (80 KiB per partition at the MLP shapes); a [N, K]\n  weight is transposed by the same DMA on load.\n* Stationary reuse: inside a pass of G output chunks the loop is `for k-slab: stationary; for chunk: nc_matmul`, so G\n  consecutive matmuls share one stationary (the LoadStationary cost L is unhidden on this chip as far as our data goes:\n  efficiency G*512/(G*512+L)).  Fresh PSUM tiles per pass let the allocator alternate banks so the evacuation of pass i\n  (ScalarE activation(copy) / VectorE tensor_copy, alternating) overlaps the matmuls of pass i+1.\n* Class C (dW, contraction over tokens): the output is 80 fp32 tiles against 8 PSUM banks, so each program keeps its\n  40 output tiles as fp32 SBUF accumulators and folds a PSUM group into them once per token chunk (VectorE add).\n* kernel[2] under LNC=2: the two programs take contiguous halves of the token tiles (A/B) or of the N-chunks (C).\n\nKnobs (env, read once at import; every kernel also takes them as int arguments so a bench can vary them in one process)\n"
import os
import torch
gemmk__P = 128
gemmk__CH = 512
gemmk__env = os.environ.get
gemmk_USE_NKI_GEMM = True
gemmk__XT_NAMES = {'dma': 0, 'pe': 1, 'pre': 2, 'hwt': 3}
gemmk__EVAC_NAMES = {'alt': 0, 'scalar': 1, 'vector': 2, 'auto': 3}
gemmk__OUT_NAMES = {'row': 0, 'chunk': 1}
gemmk__PSUM_NAMES = {'tiles': 0, 'contig': 1}
gemmk__FOLD_NAMES = {'chunk': 0, 'group': 1}
gemmk_XT = gemmk__XT_NAMES['dma']
gemmk_GROUP = 4
gemmk_TB = 1024
gemmk_TB_B = 512
gemmk_EVAC = gemmk__EVAC_NAMES['alt']
gemmk_OUT = gemmk__OUT_NAMES['row']
gemmk_PSUM = gemmk__PSUM_NAMES['tiles']
gemmk_TC = 1024
gemmk_GROUP_C = 4
gemmk_FOLD = gemmk__FOLD_NAMES['chunk']
gemmk__LOOP_NAMES = {'unroll': 0, 'affine': 1}
gemmk_LOOP = gemmk__LOOP_NAMES['unroll']
gemmk_USE_NKI_FC_RELU2 = True
gemmk__CLASS_DEFAULTS = {'A': {'xt': 3, 'group': 2, 'tb': 1024, 'evac': 2, 'out': 0, 'psum': 0, 'loop': 1}, 'B': {'xt': 3, 'group': 2, 'tb': 512, 'evac': 0, 'out': 0, 'psum': 0, 'loop': 0}, 'C': {'tc': 1024, 'group': 4, 'fold': 0, 'evac': 0, 'loop': 1}, 'F': {'xt': 3, 'group': 2, 'tb': 1024, 'evac': 0, 'psum': 0, 'loop': 1}}

def gemmk_class_knobs(cls: str) -> dict:
    """The locked defaults of a shape class, with the R1_NKI_GEMM_KNOBS_<cls> overrides applied."""
    k = dict(gemmk__CLASS_DEFAULTS[cls])
    return k
gemmk__LNC = int(gemmk__env('NEURON_LOGICAL_NC_CONFIG', '1'))
gemmk_SBUF_BUDGET = 200 * 1024
gemmk__KERNEL_DISABLED = False
gemmk__LAST_KERNEL_ERROR = None

def gemmk_eager_matmul(a: torch.Tensor, b: torch.Tensor, trans_a: bool=False, trans_b: bool=False) -> torch.Tensor:
    aa = a.t() if trans_a else a
    bb = b.t() if trans_b else b
    return torch.matmul(aa, bb)

def gemmk_plan_nn(M: int, K: int, N: int, xt: int=gemmk_XT, group: int=gemmk_GROUP, tb: int=gemmk_TB, nprg: int=2):
    """SBUF bytes per partition of the class A/B plan, or None if the shape is not tileable."""
    if M % gemmk__P or K % gemmk__P or N % gemmk__CH or (M < 2 * gemmk__P) or (K < gemmk__P) or (N < gemmk__CH):
        return None
    if group < 1 or group > 8:
        return None
    if tb % gemmk__P or tb < gemmk__P:
        return None
    nk = K // gemmk__P
    w_bytes = nk * N * 2
    if xt == 1:
        at_bytes = 2 * (nk * gemmk__P * 2 + K * 2)
    else:
        at_bytes = 2 * nk * tb * 2
    ost_bytes = 2 * N * 2
    return {'nk': nk, 'nc': N // gemmk__CH, 'sbuf': w_bytes + at_bytes + ost_bytes}

def gemmk_plan_tn(M: int, Kp: int, N: int, tc: int=gemmk_TC, group: int=gemmk_GROUP_C, nprg: int=2):
    """SBUF bytes per partition of the class C plan, or None."""
    if M % gemmk__P or Kp % gemmk__P or N % gemmk__CH or (M < gemmk__P) or (Kp < gemmk__P) or (N < gemmk__CH * nprg):
        return None
    if tc % gemmk__P or tc < gemmk__P or group < 1 or (group > 8):
        return None
    nc = N // gemmk__CH
    ncp = (nc + nprg - 1) // nprg
    nkp = Kp // gemmk__P
    tct = tc // gemmk__P
    acc_bytes = nkp * ncp * gemmk__CH * 4
    stage_bytes = 2 * tct * (Kp + ncp * gemmk__CH) * 2
    out_bytes = ncp * gemmk__CH * 2
    return {'nkp': nkp, 'nc': nc, 'ncp': ncp, 'sbuf': acc_bytes + stage_bytes + out_bytes}
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    def gemmk__dma(dst, src):
        nisa.dma_copy(dst=dst, src=src)

    def gemmk__shard():
        """(program_id, num_programs); (0, 1) without a launch grid or with sharding off."""
        if nl.program_ndim() > 0:
            return (nl.program_id(0), nl.num_programs(0))
        return (0, 1)

    def gemmk__span(n, pid, nprg):
        """This program's contiguous range [i0, i1) of n items (concrete ints at trace time)."""
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
        return (i0, i0 + cnt)

    def gemmk__ceil_div(a, b):
        return (a + b - 1) // b

    def gemmk__ptiles(n, shape, dtype, buf):
        """A tuple of n (<= 8) fresh tiles; built with an if-chain (no comprehensions, no raise in kernel helpers)."""
        t0 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 1:
            return (t0,)
        t1 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 2:
            return (t0, t1)
        t2 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 3:
            return (t0, t1, t2)
        t3 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 4:
            return (t0, t1, t2, t3)
        t4 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 5:
            return (t0, t1, t2, t3, t4)
        t5 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 6:
            return (t0, t1, t2, t3, t4, t5)
        t6 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 7:
            return (t0, t1, t2, t3, t4, t5, t6)
        t7 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        return (t0, t1, t2, t3, t4, t5, t6, t7)

    def gemmk__evac(dst, src, k, evac):
        """PSUM [P, n] fp32 -> SBUF dst (dst.dtype): ScalarE / VectorE alternating by k, or pinned, or compiler-chosen."""
        if evac == 0:
            if k % 2 == 0:
                nisa.activation(dst=dst, op=nl.copy, data=src)
            else:
                nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.vector)
        elif evac == 1:
            nisa.activation(dst=dst, op=nl.copy, data=src)
        elif evac == 2:
            nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.vector)
        else:
            nisa.tensor_copy(dst=dst, src=src)

    def gemmk__load_w(w2, wt_mode, K, N, NK, dt):
        """Resident weight [P, NK, N]: slab kk holds rows kk*128.. of the [K, N] weight (transposed on load from [N, K])."""
        w_sb = nl.ndarray((gemmk__P, NK, N), dtype=dt, buffer=nl.sbuf)
        for kk in range(NK):
            if wt_mode == 1:
                nisa.dma_transpose(dst=w_sb.ap([[NK * N, gemmk__P], [1, 1], [1, 1], [1, N]], offset=kk * N), src=w2.ap([[K, N], [1, 1], [1, 1], [1, gemmk__P]], offset=kk * gemmk__P))
            else:
                gemmk__dma(dst=w_sb[:, kk, :], src=w2[kk * gemmk__P:(kk + 1) * gemmk__P, :])
        return w_sb

    def gemmk__load_at_block(a2, xt_mode, K, NK, row0, ntok, dt):
        """a^T block [P, NK, ntok] for tokens row0..row0+ntok: DMA transpose (xt 0) or a plain load of a given a^T (xt 2)."""
        at = nl.ndarray((gemmk__P, NK, ntok), dtype=dt, buffer=nl.sbuf)
        if xt_mode == 3:
            for s in range(ntok // 16):
                nisa.dma_transpose(dst=at.ap([[NK * ntok, gemmk__P], [ntok, NK], [1, 1], [1, 16]], offset=s * 16), src=a2.ap([[K, 16], [gemmk__P, NK], [1, 1], [1, gemmk__P]], offset=(row0 + s * 16) * K), dge_mode=nisa.dge_mode.hwdge)
            return at
        for kk in range(NK):
            if xt_mode == 2:
                gemmk__dma(dst=at[:, kk, :], src=a2[kk * gemmk__P:(kk + 1) * gemmk__P, row0:row0 + ntok])
            else:
                nisa.dma_transpose(dst=at.ap([[NK * ntok, gemmk__P], [1, 1], [1, 1], [1, ntok]], offset=kk * ntok), src=a2.ap([[K, ntok], [1, 1], [1, 1], [1, gemmk__P]], offset=row0 * K + kk * gemmk__P))
        return at

    def gemmk__load_at_pe(a2, K, NK, row0, dt, evac):
        """a^T tile [P, NK, 128] for one token tile by a natural load + NK TensorE transposes (4 per PSUM tile)."""
        nat = nl.ndarray((gemmk__P, K), dtype=dt, buffer=nl.sbuf)
        gemmk__dma(dst=nat, src=a2[row0:row0 + gemmk__P, :])
        at = nl.ndarray((gemmk__P, NK, gemmk__P), dtype=dt, buffer=nl.sbuf)
        for g0 in range(0, NK, 4):
            ng = NK - g0
            if ng > 4:
                ng = 4
            ps = nl.ndarray((gemmk__P, ng * gemmk__P), dtype=dt, buffer=nl.psum)
            for j in range(ng):
                nisa.nc_transpose(dst=ps[:, j * gemmk__P:(j + 1) * gemmk__P], data=nat[:, (g0 + j) * gemmk__P:(g0 + j + 1) * gemmk__P])
            gemmk__evac(at[:, g0:g0 + ng, :], ps, g0 // 4, evac)
        return at

    def gemmk__psum_for_pass(g, psum_mode, f32):
        if psum_mode == 1:
            return (None, nl.ndarray((gemmk__P, g * gemmk__CH), dtype=f32, buffer=nl.psum))
        return (gemmk__ptiles(g, (gemmk__P, gemmk__CH), f32, nl.psum), None)

    def gemmk__ps_j(pss, pcontig, psum_mode, j):
        if psum_mode == 1:
            return pcontig[:, j * gemmk__CH:(j + 1) * gemmk__CH]
        return pss[j]

    def gemmk__matmul_pass(at, kk_tok, w_sb, NK, c0, g, pss, pcontig, psum_mode):
        """acc[j] = sum_kk at[:, kk, tok].T @ w[:, kk, chunk c0+j] for j < g (k outer, chunk inner: stationary reuse g)."""
        for kk in range(NK):
            st = at[:, kk, kk_tok:kk_tok + gemmk__P]
            for j in range(g):
                c = c0 + j
                nisa.nc_matmul(dst=gemmk__ps_j(pss, pcontig, psum_mode, j), stationary=st, moving=w_sb[:, kk, c * gemmk__CH:(c + 1) * gemmk__CH], accumulate=kk > 0)

    def gemmk__nn_tile(at, w_sb, out, ti, bt0, N, NK, NC, npass, group, evac, out_mode, psum_mode, dt, f32):
        """One 128-token tile of the class A/B loop: passes of `group` chunks, evacuation, store."""
        row0 = (bt0 + ti) * gemmk__P
        if out_mode == 0:
            ost = nl.ndarray((gemmk__P, N), dtype=dt, buffer=nl.sbuf)
        for p in range(npass):
            c0 = p * group
            g = NC - c0
            if g > group:
                g = group
            pss, pcontig = gemmk__psum_for_pass(g, psum_mode, f32)
            gemmk__matmul_pass(at, ti * gemmk__P, w_sb, NK, c0, g, pss, pcontig, psum_mode)
            for j in range(g):
                c = c0 + j
                if out_mode == 0:
                    gemmk__evac(ost[:, c * gemmk__CH:(c + 1) * gemmk__CH], gemmk__ps_j(pss, pcontig, psum_mode, j), c, evac)
                else:
                    ob = nl.ndarray((gemmk__P, gemmk__CH), dtype=dt, buffer=nl.sbuf)
                    gemmk__evac(ob, gemmk__ps_j(pss, pcontig, psum_mode, j), c, evac)
                    gemmk__dma(dst=out[row0:row0 + gemmk__P, c * gemmk__CH:(c + 1) * gemmk__CH], src=ob)
        if out_mode == 0:
            gemmk__dma(dst=out[row0:row0 + gemmk__P, :], src=ost)

    def gemmk__fc_tile(at, w_sb, h, pre, ti, bt0, N, NK, NC, npass, group, psum_mode, save_pre, dt, f32, evac):
        """One 128-token tile of the fused loop: pre = bf16(PSUM) on ScalarE (evac 0 / 1; 2 = VectorE tensor_copy,
        3 = compiler-chosen), h = (pre max 0) * pre on VectorE."""
        row0 = (bt0 + ti) * gemmk__P
        pst = nl.ndarray((gemmk__P, N), dtype=dt, buffer=nl.sbuf)
        hst = nl.ndarray((gemmk__P, N), dtype=dt, buffer=nl.sbuf)
        for p in range(npass):
            c0 = p * group
            g = NC - c0
            if g > group:
                g = group
            pss, pcontig = gemmk__psum_for_pass(g, psum_mode, f32)
            gemmk__matmul_pass(at, ti * gemmk__P, w_sb, NK, c0, g, pss, pcontig, psum_mode)
            for j in range(g):
                c = c0 + j
                pj = pst[:, c * gemmk__CH:(c + 1) * gemmk__CH]
                if evac == 2:
                    nisa.tensor_copy(dst=pj, src=gemmk__ps_j(pss, pcontig, psum_mode, j), engine=nisa.engine.vector)
                elif evac == 3:
                    nisa.tensor_copy(dst=pj, src=gemmk__ps_j(pss, pcontig, psum_mode, j))
                else:
                    nisa.activation(dst=pj, op=nl.copy, data=gemmk__ps_j(pss, pcontig, psum_mode, j))
                nisa.scalar_tensor_tensor(dst=hst[:, c * gemmk__CH:(c + 1) * gemmk__CH], data=pj, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=pj)
        gemmk__dma(dst=h[row0:row0 + gemmk__P, :], src=hst)
        if save_pre == 1:
            gemmk__dma(dst=pre[row0:row0 + gemmk__P, :], src=pst)

    @nki.jit
    def gemmk__mm_nn_kernel(a2, w2, xt_mode, wt_mode, group, tb, evac, out_mode, psum_mode, loop):
        """out[M, N] = a @ w.  a2 is [M, K] (xt_mode 0: DMA transpose, 1: PE transposes) or a^T [K, M] (xt_mode 2);
        w2 is [K, N] (wt_mode 0) or [N, K] (wt_mode 1, transposed on load); loop 0 unrolls the tile loop, 1 = affine_range."""
        if xt_mode == 2:
            K = a2.shape[0]
            M = a2.shape[1]
        else:
            M = a2.shape[0]
            K = a2.shape[1]
        if wt_mode == 1:
            N = w2.shape[0]
        else:
            N = w2.shape[1]
        dt = a2.dtype
        f32 = nl.float32
        out = nl.ndarray((M, N), dtype=dt, buffer=nl.shared_hbm)
        NK = K // gemmk__P
        NC = N // gemmk__CH
        npass = gemmk__ceil_div(NC, group)
        pid, nprg = gemmk__shard()
        t0, t1 = gemmk__span(M // gemmk__P, pid, nprg)
        w_sb = gemmk__load_w(w2, wt_mode, K, N, NK, dt)
        ntb = tb // gemmk__P
        if xt_mode == 1:
            ntb = 1
        nblk = gemmk__ceil_div(t1 - t0, ntb)
        for b in range(nblk):
            bt0 = t0 + b * ntb
            bt1 = bt0 + ntb
            if bt1 > t1:
                bt1 = t1
            nt = bt1 - bt0
            if xt_mode == 1:
                at = gemmk__load_at_pe(a2, K, NK, bt0 * gemmk__P, dt, evac)
            else:
                at = gemmk__load_at_block(a2, xt_mode, K, NK, bt0 * gemmk__P, nt * gemmk__P, dt)
            if loop == 1:
                for ti in nl.affine_range(nt):
                    gemmk__nn_tile(at, w_sb, out, ti, bt0, N, NK, NC, npass, group, evac, out_mode, psum_mode, dt, f32)
            else:
                for ti in range(nt):
                    gemmk__nn_tile(at, w_sb, out, ti, bt0, N, NK, NC, npass, group, evac, out_mode, psum_mode, dt, f32)
        return out

    @nki.jit
    def gemmk__fc_relu2_kernel(a2, w2, xt_mode, group, tb, evac, psum_mode, save_pre, loop):
        """h[M, N] = relu(a @ w)^2 and pre[M, N] = bf16(a @ w) (pre is written only when save_pre == 1)."""
        if xt_mode == 2:
            K = a2.shape[0]
            M = a2.shape[1]
        else:
            M = a2.shape[0]
            K = a2.shape[1]
        N = w2.shape[1]
        dt = a2.dtype
        f32 = nl.float32
        h = nl.ndarray((M, N), dtype=dt, buffer=nl.shared_hbm)
        pre = nl.ndarray((M, N), dtype=dt, buffer=nl.shared_hbm)
        NK = K // gemmk__P
        NC = N // gemmk__CH
        npass = gemmk__ceil_div(NC, group)
        pid, nprg = gemmk__shard()
        t0, t1 = gemmk__span(M // gemmk__P, pid, nprg)
        w_sb = gemmk__load_w(w2, 0, K, N, NK, dt)
        ntb = tb // gemmk__P
        if xt_mode == 1:
            ntb = 1
        nblk = gemmk__ceil_div(t1 - t0, ntb)
        for b in range(nblk):
            bt0 = t0 + b * ntb
            bt1 = bt0 + ntb
            if bt1 > t1:
                bt1 = t1
            nt = bt1 - bt0
            if xt_mode == 1:
                at = gemmk__load_at_pe(a2, K, NK, bt0 * gemmk__P, dt, evac)
            else:
                at = gemmk__load_at_block(a2, xt_mode, K, NK, bt0 * gemmk__P, nt * gemmk__P, dt)
            if loop == 1:
                for ti in nl.affine_range(nt):
                    gemmk__fc_tile(at, w_sb, h, pre, ti, bt0, N, NK, NC, npass, group, psum_mode, save_pre, dt, f32, evac)
            else:
                for ti in range(nt):
                    gemmk__fc_tile(at, w_sb, h, pre, ti, bt0, N, NK, NC, npass, group, psum_mode, save_pre, dt, f32, evac)
        return (h, pre)

    def gemmk__tn_chunk(a2, b2, acc, ct0, n, first, Kp, NKp, NCp, Np, c0, c1, npass, group, fold_mode, evac, dt, f32):
        """One tc-token chunk of the class C loop: stage a and b tiles, then per k'-tile and chunk group chain the n
        token tiles in PSUM and fold into the SBUF fp32 accumulators (a copy on the first chunk, an add afterwards)."""
        a_sb = nl.ndarray((gemmk__P, n, Kp), dtype=dt, buffer=nl.sbuf)
        b_sb = nl.ndarray((gemmk__P, n, Np), dtype=dt, buffer=nl.sbuf)
        for t in range(n):
            r0 = (ct0 + t) * gemmk__P
            gemmk__dma(dst=a_sb[:, t, :], src=a2[r0:r0 + gemmk__P, :])
            gemmk__dma(dst=b_sb[:, t, :], src=b2[r0:r0 + gemmk__P, c0 * gemmk__CH:c1 * gemmk__CH])
        for kt in range(NKp):
            for p in range(npass):
                g0 = p * group
                g = NCp - g0
                if g > group:
                    g = group
                if fold_mode == 1:
                    pcontig = nl.ndarray((gemmk__P, g * gemmk__CH), dtype=f32, buffer=nl.psum)
                    pss = None
                else:
                    pss = gemmk__ptiles(g, (gemmk__P, gemmk__CH), f32, nl.psum)
                    pcontig = None
                for t in range(n):
                    st = a_sb[:, t, kt * gemmk__P:(kt + 1) * gemmk__P]
                    for j in range(g):
                        nisa.nc_matmul(dst=gemmk__ps_j(pss, pcontig, fold_mode, j), stationary=st, moving=b_sb[:, t, (g0 + j) * gemmk__CH:(g0 + j + 1) * gemmk__CH], accumulate=t > 0)
                if fold_mode == 1:
                    accs = acc[:, kt, g0 * gemmk__CH:(g0 + g) * gemmk__CH]
                    if first == 1:
                        gemmk__evac(accs, pcontig, kt + p, evac)
                    else:
                        nisa.tensor_tensor(dst=accs, data1=pcontig, data2=accs, op=nl.add)
                else:
                    for j in range(g):
                        accs = acc[:, kt, (g0 + j) * gemmk__CH:(g0 + j + 1) * gemmk__CH]
                        if first == 1:
                            gemmk__evac(accs, pss[j], kt + j, evac)
                        else:
                            nisa.tensor_tensor(dst=accs, data1=pss[j], data2=accs, op=nl.add)

    @nki.jit
    def gemmk__mm_tn_kernel(a2, b2, tc, group, fold_mode, evac, loop):
        """out[Kp, N] = a2[M, Kp].T @ b2[M, N]: contraction over the token axis, fp32 SBUF accumulators per program
        (its N-chunk range), PSUM groups of `group` chunks folded once per tc-token chunk; loop 1 = affine_range over
        the chunks after the first (which is peeled: its fold is a copy)."""
        M = a2.shape[0]
        Kp = a2.shape[1]
        N = b2.shape[1]
        dt = a2.dtype
        f32 = nl.float32
        out = nl.ndarray((Kp, N), dtype=dt, buffer=nl.shared_hbm)
        NT = M // gemmk__P
        NKp = Kp // gemmk__P
        NC = N // gemmk__CH
        pid, nprg = gemmk__shard()
        c0, c1 = gemmk__span(NC, pid, nprg)
        NCp = c1 - c0
        Np = NCp * gemmk__CH
        npass = gemmk__ceil_div(NCp, group)
        acc = nl.ndarray((gemmk__P, NKp, Np), dtype=f32, buffer=nl.sbuf)
        tct = tc // gemmk__P
        if tct > NT:
            tct = NT
        nfull = NT // tct
        ntail = NT - nfull * tct
        if loop == 1:
            gemmk__tn_chunk(a2, b2, acc, 0, tct, 1, Kp, NKp, NCp, Np, c0, c1, npass, group, fold_mode, evac, dt, f32)
            for i in nl.affine_range(nfull - 1):
                gemmk__tn_chunk(a2, b2, acc, (i + 1) * tct, tct, 0, Kp, NKp, NCp, Np, c0, c1, npass, group, fold_mode, evac, dt, f32)
        else:
            for ch in range(nfull):
                first = 0
                if ch == 0:
                    first = 1
                gemmk__tn_chunk(a2, b2, acc, ch * tct, tct, first, Kp, NKp, NCp, Np, c0, c1, npass, group, fold_mode, evac, dt, f32)
        if ntail > 0:
            gemmk__tn_chunk(a2, b2, acc, nfull * tct, ntail, 0, Kp, NKp, NCp, Np, c0, c1, npass, group, fold_mode, evac, dt, f32)
        for kt in range(NKp):
            ob = nl.ndarray((gemmk__P, Np), dtype=dt, buffer=nl.sbuf)
            if kt % 2 == 0:
                nisa.activation(dst=ob, op=nl.copy, data=acc[:, kt, :])
            else:
                nisa.tensor_copy(dst=ob, src=acc[:, kt, :], engine=nisa.engine.vector)
            gemmk__dma(dst=out[kt * gemmk__P:(kt + 1) * gemmk__P, c0 * gemmk__CH:c1 * gemmk__CH], src=ob)
        return out
    gemmk__HAS_NKI = True
except Exception:
    gemmk__HAS_NKI = False

def gemmk__disable_kernel(where: str, err: BaseException) -> None:
    global gemmk__KERNEL_DISABLED, gemmk__LAST_KERNEL_ERROR
    if not gemmk__KERNEL_DISABLED:
        gemmk__KERNEL_DISABLED = True
        gemmk__LAST_KERNEL_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[nki_gemm] kernel disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)

def gemmk__launch(kernel):
    """kernel[2] under an LNC=2 runtime (both physical cores run half the work), plain call otherwise."""
    return kernel[gemmk__LNC] if gemmk__LNC == 2 else kernel

def gemmk__nprg() -> int:
    return 2 if gemmk__LNC == 2 else 1

def gemmk_run_nn(a: torch.Tensor, w: torch.Tensor, trans_b: bool=False, xt: int=gemmk_XT, group: int=gemmk_GROUP, tb: int=None, evac: int=gemmk_EVAC, out_mode: int=gemmk_OUT, psum_mode: int=gemmk_PSUM, loop: int=gemmk_LOOP) -> torch.Tensor:
    """Direct launch of the class A/B kernel (no fallback).  xt == 2 means `a` is the pre-transposed a^T [K, M]."""
    K = a.shape[0] if xt == 2 else a.shape[1]
    if tb is None:
        tb = gemmk_TB if K <= 2048 else gemmk_TB_B
    return gemmk__launch(gemmk__mm_nn_kernel)(a, w, int(xt), 1 if trans_b else 0, int(group), int(tb), int(evac), int(out_mode), int(psum_mode), int(loop))

def gemmk_run_tn(a: torch.Tensor, b: torch.Tensor, tc: int=gemmk_TC, group: int=gemmk_GROUP_C, fold: int=gemmk_FOLD, evac: int=gemmk_EVAC, loop: int=gemmk_LOOP) -> torch.Tensor:
    """Direct launch of the class C kernel: out[Kp, N] = a[M, Kp].T @ b[M, N]."""
    return gemmk__launch(gemmk__mm_tn_kernel)(a, b, int(tc), int(group), int(fold), int(evac), int(loop))

def gemmk_run_fc_relu2(a: torch.Tensor, w: torch.Tensor, xt: int=gemmk_XT, group: int=gemmk_GROUP, tb: int=gemmk_TB, evac: int=gemmk_EVAC, psum_mode: int=gemmk_PSUM, save_pre: int=1, loop: int=gemmk_LOOP):
    """Direct launch of the fused kernel: (h, pre)."""
    return gemmk__launch(gemmk__fc_relu2_kernel)(a, w, int(xt), int(group), int(tb), int(evac), int(psum_mode), int(save_pre), int(loop))

def gemmk__usable(*ts) -> bool:
    if not gemmk__HAS_NKI or gemmk__KERNEL_DISABLED:
        return False
    for t in ts:
        if t.device.type != 'neuron' or t.dtype != torch.bfloat16 or t.dim() != 2 or (not t.is_contiguous()):
            return False
    return True

def gemmk_nki_matmul(a: torch.Tensor, b: torch.Tensor, trans_a: bool=False, trans_b: bool=False) -> torch.Tensor:
    """out = op(a) @ op(b) on the NKI kernels when the shape fits the plans (class A: K <= 2048 NN/NT; B: K > 2048;
    C: trans_a), torch.matmul otherwise (or on a kernel failure, which disables the kernels for the process)."""
    if not gemmk__usable(a, b) or (trans_a and trans_b):
        return gemmk_eager_matmul(a, b, trans_a, trans_b)
    try:
        if trans_a:
            M, Kp = a.shape
            M2, N = b.shape
            if M2 != M:
                return gemmk_eager_matmul(a, b, trans_a, trans_b)
            k = gemmk_class_knobs('C')
            tc = k['tc']
            pl = gemmk_plan_tn(M, Kp, N, tc=tc, group=k['group'], nprg=gemmk__nprg())
            if pl is not None and pl['sbuf'] > gemmk_SBUF_BUDGET and (tc > gemmk__P):
                tc = tc // 2
                pl = gemmk_plan_tn(M, Kp, N, tc=tc, group=k['group'], nprg=gemmk__nprg())
            if pl is None or pl['sbuf'] > gemmk_SBUF_BUDGET:
                return gemmk_eager_matmul(a, b, trans_a, trans_b)
            return gemmk_run_tn(a, b, tc=tc, group=k['group'], fold=k['fold'], evac=k['evac'], loop=k['loop'])
        M, K = a.shape
        if trans_b:
            N, K2 = b.shape
        else:
            K2, N = b.shape
        k = gemmk_class_knobs('A' if K <= 2048 else 'B')
        xt = k['xt'] if k['xt'] != 2 else 0
        pl = gemmk_plan_nn(M, K, N, xt=xt, group=k['group'], tb=k['tb'], nprg=gemmk__nprg()) if K2 == K else None
        if pl is None or pl['sbuf'] > gemmk_SBUF_BUDGET:
            return gemmk_eager_matmul(a, b, trans_a, trans_b)
        return gemmk_run_nn(a, b, trans_b=trans_b, xt=xt, group=k['group'], tb=k['tb'], evac=k['evac'], out_mode=k['out'], psum_mode=k['psum'], loop=k['loop'])
    except Exception as err:
        gemmk__disable_kernel('nki_matmul', err)
        return gemmk_eager_matmul(a, b, trans_a, trans_b)

def gemmk_fused_fc_relu2_forward(x2: torch.Tensor, w1: torch.Tensor, save_pre: int=1):
    """(h, h_pre) = (relu(x2 @ w1)^2, x2 @ w1) in one launch; the eager pair when the kernel is not applicable.

    [R1 relu2fold a] save_pre=0 skips the kernel's [M, N] `pre` STORE -- 335.5 MB per call at device
    batch 32 / hidden 5120 / bf16 -- for a consumer that rebuilds 2*relu(pre) as 2*sqrt(h) instead
    return arity without changing the kernel's signature), so the second element of the tuple is then
    uninitialised device memory and the caller MUST drop it unread.  Default 1 = every existing caller
    unchanged, same NEFF, same bytes."""
    if gemmk__usable(x2, w1):
        M, K = x2.shape
        K2, N = w1.shape
        k = gemmk_class_knobs('F')
        xt = k['xt'] if k['xt'] != 2 else 0
        pl = gemmk_plan_nn(M, K, N, xt=xt, group=k['group'], tb=k['tb'], nprg=gemmk__nprg()) if K2 == K else None
        if pl is not None and pl['sbuf'] + 2 * N * 2 <= gemmk_SBUF_BUDGET:
            try:
                h, pre = gemmk_run_fc_relu2(x2, w1, xt=xt, group=k['group'], tb=k['tb'], evac=k['evac'], psum_mode=k['psum'], save_pre=int(save_pre), loop=k['loop'])
                return (h, pre)
            except Exception as err:
                gemmk__disable_kernel('fused_fc_relu2', err)
    pre = torch.matmul(x2, w1)
    return (torch.nn.functional.relu(pre).square(), pre)
try:
    gemmk__relu2_backward = relu2k_relu2_backward
except Exception:
    gemmk__relu2_backward = None

def gemmk_eager_relu2_backward(pre: torch.Tensor, dh: torch.Tensor) -> torch.Tensor:
    """autograd's own formula for relu(x).square(): grad * 2 * relu(x) (the *2 is exact, one rounding on the product)."""
    return dh * (2.0 * torch.nn.functional.relu(pre))

class gemmk_FcRelu2Fn(torch.autograd.Function):
    """h = relu(x2d @ W)^2 for the LinearT c_fc weight W [in, out]: forward = one fused launch (h and the saved bf16
    pre = today's h_pre, both bitwise today's chain; eager pair off-chip), backward = today's ReLU^2 backward kernel on
    the saved pre (dpre = 2 relu(pre) dh), dx = dpre @ W^T on the class-B primitive (bitwise torch) and dW either
    Parameter) or returned as x2d^T dpre from the class-C primitive (1-ulp fold order vs torch; eager off-chip)."""

    @staticmethod
    def forward(ctx, x2d, w, inplace):
        h, pre = gemmk_fused_fc_relu2_forward(x2d, w)
        ctx.save_for_backward(x2d, w, pre)
        ctx.w = w
        ctx.inplace = inplace
        return h

    @staticmethod
    def backward(ctx, dh):
        x2d, w, pre = ctx.saved_tensors
        dh = dh.contiguous()
        if gemmk__relu2_backward is not None:
            dpre = gemmk__relu2_backward(pre, dh)
        else:
            dpre = gemmk_eager_relu2_backward(pre, dh)
        dx = gemmk_nki_matmul(dpre, w, trans_b=True)
        if ctx.inplace:
            ctx.w.grad.addmm_(x2d.t(), dpre)
            return (dx, None, None)
        return (dx, gemmk_nki_matmul(x2d, dpre, trans_a=True), None)

def gemmk_fused_fc_relu2(x: torch.Tensor, w: torch.Tensor, inplace: bool=False) -> torch.Tensor:
    """relu(x @ w)^2 for any leading shape (w [in, out]); with inplace=True the c_fc dW goes straight into w.grad."""
    K = x.shape[-1]
    x2d = x.reshape(-1, K)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    h = gemmk_FcRelu2Fn.apply(x2d, w, inplace)
    return h.view(*x.shape[:-1], w.shape[1])
nki_matmul = gemmk_nki_matmul
fused_fc_relu2 = gemmk_fused_fc_relu2
USE_NKI_GEMM = gemmk_USE_NKI_GEMM
USE_NKI_FC_RELU2 = gemmk_USE_NKI_FC_RELU2
"Lane C phase 1 (plan-v12 lever 4): two NKI kernels for the per-step optimizer lane, plus the wrappers train.py calls.\n\nKernel 1 -- `adamw3`: ONE launch for the three AdamW groups (wte table, lm_head table, the flat stack of the 1-element\n    gains), replacing 2 x 11-13 + 13 (+13 for skip_lambda) eager launches and 6 host->device bias copies per step.\n    fp32 in, fp32 math, fp32 out, written IN PLACE into the parameter / exp_avg / exp_avg_sq tensors (INPLACE=1) or\n    returned as fresh tensors (INPLACE=0, the fallback form).  The arithmetic is the eager `_adamw_update` of\n    work/train.py instruction by instruction:\n        p  <- p * (1 - lr*wd)                       (skipped when the group's weight decay is 0, as eager skips it)\n        m  <- m + (1-b1) * (g - m)                   lerp_, weight < 0.5 form\n        v  <- v + (1-b2) * (g*g - v)\n        u  <- m / bias1 ;  den <- sqrt(v / bias2) + eps ;  p <- p - (u / den) * lr\n    Per-group scalars (1 - lr*wd, 1-b1, 1-b2, bias1, bias2, eps, lr, ...) arrive as ONE [1, 48] fp32 tensor staged on\n    the host in the same fp32 arithmetic the device used (the S8e precedent), broadcast to [128, 48] by one DMA.\n    Whether the result is BITWISE the device's eager chain depends on how torch-neuron lowers lerp_/div/sqrt; the knobs\n    LERP / DIV / SQRT / ENG select the instruction forms and gap 1 picks the bitwise one (or reports the ulp class).\n\nKernel 2 -- `gradnorm`: the four big gradient-norm terms of GradReducer.reduce (the two bf16 Muon grad shards, the two\n    fp32 table grads) plus the reduced 1-element stack, in one launch: ScalarE activation(square) with the free-axis\n    reduce, per-partition fp32 partials, output [128, 4] (program q writes columns 2q = shard terms, 2q+1 = table\n    terms); the wrapper's `out.view(128, 2, 2).sum((0, 1))` gives (shards, tables) in a fixed order.  Replaces ~14\n    launches (the .float() of the [5,1024,5120] shard alone writes 105 MB).  fp32 reassociation of `gnorm` only; the\n    finiteness gate is invariant (NaN/Inf poison every summation order; no overflow is possible).\n\nWrappers (called from work/train.py under the switches; every one falls back to the exact eager ops):\n    adamw_fused_step(opt)          -> bool   the fused AdamW step for opt's three groups (False = not applicable: eager)\n    reduce_fused(reducer, clip)    -> finite the whole GradReducer.reduce with the norm kernel (collectives as eager)\n    muon_finish_view(...)                    the slab-0 finish: no per-matrix copy_, torch.sub(w_view, upd, out=shard)\n\nSwitches (env, read once at import; ALL default OFF; the off path of train.py is textually unchanged):\nKnobs: R1_NKI_ADAMW_T (2048) tile columns; R1_NKI_ADAMW_LERP 0|1; R1_NKI_ADAMW_DIV div|recip|tt; R1_NKI_ADAMW_SQRT\n    act|ts|rsqrt; R1_NKI_ADAMW_ENG 0|1 (1: the two scalar multiplies and the square on ScalarE); R1_NKI_ADAMW_LOOP\n    affine|unroll; R1_NKI_ADAMW_INPLACE 1|0; R1_NKI_GRADNORM_T (4096 bf16 / 2048 fp32 columns); R1_NKI_GRADNORM_LOOP.\n    Defaults = the gap-1 winner (Melbourne 2, 2026-09-15 14:39Z, NOTES.md section 3): LERP 0, DIV recip, SQRT act, ENG 1\n    is BITWISE the device's eager `_adamw_update` (50,331,723/50,331,723 elements of p/m/v over three step counts) at\n    1.13 ms per launch against the eager chain's 7.7 ms; `div`/`ts`/`tt` do not lower (no divide/sqrt ALU op), `rsqrt`\n    is the 1-ulp class (the device's own sqrt is the ScalarE activation).\nTest hook: R1_NKI_OPTIM_EMULATE=kernel|eager runs the torch emulation in place of the kernel on any device.\n"
import os
import torch
optk__env = os.environ.get
optk__P = 128
optk__NSC = 48
optk__SLOT = {'cwd': 0, 'w1': 1, 'w2': 2, 'b1': 3, 'b2': 4, 'eps': 5, 'lr': 6, 'rb1': 7, 'rb2': 8, 'beta1': 9, 'beta2': 10}
optk_USE_NKI_ADAMW = True
optk_USE_NKI_GRADNORM = True
optk_MUON_SHARD_VIEW = 'sub'
optk_ADAMW_STACK_SMALL = 8
optk_EMULATE = '0'
optk__DIV_NAMES = {'div': 0, 'recip': 1, 'tt': 2}
optk__SQRT_NAMES = {'act': 0, 'ts': 1, 'rsqrt': 2}
optk__LOOP_NAMES = {'unroll': 0, 'affine': 1}
optk_ADAMW_T = 2048
optk_ADAMW_LERP = 0
optk_ADAMW_DIV = optk__DIV_NAMES['recip']
optk_ADAMW_SQRT = optk__SQRT_NAMES['act']
optk_ADAMW_ENG = 1
optk_ADAMW_LOOP = optk__LOOP_NAMES['affine']
optk_ADAMW_INPLACE = 1
optk_GRADNORM_T = 4096
optk_GRADNORM_LOOP = optk__LOOP_NAMES['unroll']
optk__LNC = int(optk__env('NEURON_LOGICAL_NC_CONFIG', '1'))
optk__ADAMW_DISABLED = False
optk__GRADNORM_DISABLED = False
optk__LAST_ERROR = None

def optk_config_string() -> str:
    return f'adamw={int(optk_USE_NKI_ADAMW)} gradnorm={int(optk_USE_NKI_GRADNORM)} shard_view={optk_MUON_SHARD_VIEW} stack_small={optk_ADAMW_STACK_SMALL} T={optk_ADAMW_T} lerp={optk_ADAMW_LERP} div={optk_ADAMW_DIV} sqrt={optk_ADAMW_SQRT} eng={optk_ADAMW_ENG} loop={optk_ADAMW_LOOP} inplace={optk_ADAMW_INPLACE} normT={optk_GRADNORM_T} normloop={optk_GRADNORM_LOOP} lnc={optk__LNC} emulate={optk_EMULATE}'

def optk_group_scalars(lr_cpu: torch.Tensor, beta1: float, beta2: float, eps: float, weight_decay: float, step_count: int) -> torch.Tensor:
    """The 16 fp32 scalars of one group for step `step_count` (1-based), computed on the host in the arithmetic the
    device uses: `1 - lr*wd` is fp32 (1,) x fp32(wd) then 1 - x (the S8e rule: bitwise the device's own products);
    1-b1 / 1-b2 are the Python doubles torch's lerp_ rounds to fp32; bias1/2 exactly as _adamw_step fills them."""
    s = torch.zeros(16, dtype=torch.float32)
    lr = lr_cpu.reshape(1).to(torch.float32)
    cwd = 1.0 - lr * weight_decay if weight_decay > 0 else torch.ones(1)
    s[optk__SLOT['cwd']] = cwd[0]
    s[optk__SLOT['w1']] = torch.tensor(1 - beta1, dtype=torch.float32)
    s[optk__SLOT['w2']] = torch.tensor(1 - beta2, dtype=torch.float32)
    b1 = torch.full((1,), 1 - beta1 ** step_count, dtype=torch.float32)
    b2 = torch.full((1,), 1 - beta2 ** step_count, dtype=torch.float32)
    s[optk__SLOT['b1']] = b1[0]
    s[optk__SLOT['b2']] = b2[0]
    s[optk__SLOT['eps']] = torch.tensor(eps, dtype=torch.float32)
    s[optk__SLOT['lr']] = lr[0]
    s[optk__SLOT['rb1']] = (1.0 / b1)[0]
    s[optk__SLOT['rb2']] = (1.0 / b2)[0]
    s[optk__SLOT['beta1']] = torch.tensor(beta1, dtype=torch.float32)
    s[optk__SLOT['beta2']] = torch.tensor(beta2, dtype=torch.float32)
    return s
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    def optk__dma(dst, src):
        nisa.dma_copy(dst=dst, src=src)

    def optk__shard():
        """(program_id, num_programs); (0, 1) without a launch grid or with sharding off."""
        if nl.program_ndim() > 0:
            return (nl.program_id(0), nl.num_programs(0))
        return (0, 1)

    def optk__sb(shape, dtype):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    def optk__bcast_tile(sct, col, Pn, Tn):
        """A [Pn, Tn] fp32 SBUF tile filled with the scalar sct[0:Pn, col] (the divisor tile of the DIV=2 form)."""
        t = optk__sb((Pn, Tn), nl.float32)
        nisa.memset(t, 0.0)
        nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=sct[0:Pn, col:col + 1], engine=nisa.engine.vector)
        return t

    def optk__adamw_tile(dp, dm, dv, sp, sg, sm, sv, sct, row, Pn, Tn, LERP, DIV, SQRT, ENG, WD, bt1, bt2):
        """One [Pn, Tn] fp32 tile: reads sp/sg/sm/sv (HBM views), writes dp/dm/dv (HBM views), scalars from sct[0:Pn, row*16 + slot].
        DIV: 0 tensor_scalar divide (does not lower), 1 multiply by the host reciprocal / VectorE reciprocal, 2 tensor_tensor
        divide against the broadcast divisor tiles bt1/bt2.  SQRT: 0 ScalarE activation sqrt, 1 tensor_scalar sqrt (does not
        lower), 2 rsqrt (compiler-chosen engine) then reciprocal."""
        f32 = nl.float32
        vec = nisa.engine.vector
        r0 = row * 16
        tp = optk__sb((Pn, Tn), f32)
        tg = optk__sb((Pn, Tn), f32)
        tm = optk__sb((Pn, Tn), f32)
        tv = optk__sb((Pn, Tn), f32)
        optk__dma(dst=tp, src=sp)
        optk__dma(dst=tg, src=sg)
        optk__dma(dst=tm, src=sm)
        optk__dma(dst=tv, src=sv)
        t1 = optk__sb((Pn, Tn), f32)
        t2 = optk__sb((Pn, Tn), f32)
        t3 = optk__sb((Pn, Tn), f32)
        t4 = optk__sb((Pn, Tn), f32)
        if WD == 1:
            if ENG == 1:
                nisa.tensor_scalar(dst=tp, data=tp, op0=nl.multiply, operand0=sct[0:Pn, r0 + 0:r0 + 1], engine=nisa.engine.scalar)
            else:
                nisa.tensor_scalar(dst=tp, data=tp, op0=nl.multiply, operand0=sct[0:Pn, r0 + 0:r0 + 1], engine=vec)
        if LERP == 0:
            nisa.tensor_tensor(dst=t1, data1=tg, data2=tm, op=nl.subtract, engine=vec)
            nisa.scalar_tensor_tensor(dst=t1, data=t1, op0=nl.multiply, operand0=sct[0:Pn, r0 + 1:r0 + 2], op1=nl.add, operand1=tm)
        else:
            nisa.tensor_scalar(dst=t1, data=tg, op0=nl.multiply, operand0=sct[0:Pn, r0 + 1:r0 + 2], engine=vec)
            nisa.scalar_tensor_tensor(dst=t1, data=tm, op0=nl.multiply, operand0=sct[0:Pn, r0 + 9:r0 + 10], op1=nl.add, operand1=t1)
        if ENG == 1:
            nisa.activation(dst=t2, op=nl.square, data=tg)
        else:
            nisa.tensor_tensor(dst=t2, data1=tg, data2=tg, op=nl.multiply, engine=vec)
        nisa.tensor_tensor(dst=t2, data1=t2, data2=tv, op=nl.subtract, engine=vec)
        nisa.scalar_tensor_tensor(dst=t2, data=t2, op0=nl.multiply, operand0=sct[0:Pn, r0 + 2:r0 + 3], op1=nl.add, operand1=tv)
        optk__dma(dst=dm, src=t1)
        optk__dma(dst=dv, src=t2)
        if DIV == 0:
            nisa.tensor_scalar(dst=t3, data=t1, op0=nl.divide, operand0=sct[0:Pn, r0 + 3:r0 + 4], engine=vec)
            nisa.tensor_scalar(dst=t4, data=t2, op0=nl.divide, operand0=sct[0:Pn, r0 + 4:r0 + 5], engine=vec)
        elif DIV == 2:
            nisa.tensor_tensor(dst=t3, data1=t1, data2=bt1, op=nl.divide, engine=vec)
            nisa.tensor_tensor(dst=t4, data1=t2, data2=bt2, op=nl.divide, engine=vec)
        else:
            nisa.tensor_scalar(dst=t3, data=t1, op0=nl.multiply, operand0=sct[0:Pn, r0 + 7:r0 + 8], engine=vec)
            nisa.tensor_scalar(dst=t4, data=t2, op0=nl.multiply, operand0=sct[0:Pn, r0 + 8:r0 + 9], engine=vec)
        if SQRT == 0:
            nisa.activation(dst=t4, op=nl.sqrt, data=t4)
            nisa.tensor_scalar(dst=t4, data=t4, op0=nl.add, operand0=sct[0:Pn, r0 + 5:r0 + 6], engine=vec)
        elif SQRT == 1:
            nisa.tensor_scalar(dst=t4, data=t4, op0=nl.sqrt, operand0=0.0, op1=nl.add, operand1=sct[0:Pn, r0 + 5:r0 + 6], engine=vec)
        else:
            nisa.tensor_scalar(dst=t4, data=t4, op0=nl.rsqrt, operand0=0.0)
            nisa.reciprocal(dst=t4, data=t4)
            nisa.tensor_scalar(dst=t4, data=t4, op0=nl.add, operand0=sct[0:Pn, r0 + 5:r0 + 6], engine=vec)
        if DIV == 0 or DIV == 2:
            nisa.tensor_tensor(dst=t3, data1=t3, data2=t4, op=nl.divide, engine=vec)
        else:
            nisa.reciprocal(dst=t4, data=t4)
            nisa.tensor_tensor(dst=t3, data1=t3, data2=t4, op=nl.multiply, engine=vec)
        if ENG == 1:
            nisa.tensor_scalar(dst=t3, data=t3, op0=nl.multiply, operand0=sct[0:Pn, r0 + 6:r0 + 7], engine=nisa.engine.scalar)
        else:
            nisa.tensor_scalar(dst=t3, data=t3, op0=nl.multiply, operand0=sct[0:Pn, r0 + 6:r0 + 7], engine=vec)
        nisa.tensor_tensor(dst=tp, data1=tp, data2=t3, op=nl.subtract, engine=vec)
        optk__dma(dst=dp, src=tp)

    def optk__adamw_table(po, mo, vo, pi, gi, mi, vi, sct, row, T, LERP, DIV, SQRT, ENG, LOOP, WD, pid, nprg):
        """All tiles of one [128, F] fp32 table that belong to program pid (contiguous column half)."""
        P, F = pi.shape
        Fp = F // nprg
        NT = Fp // T
        c_base = pid * Fp
        bt1 = None
        bt2 = None
        if DIV == 2:
            bt1 = optk__bcast_tile(sct, row * 16 + 3, P, T)
            bt2 = optk__bcast_tile(sct, row * 16 + 4, P, T)
        if LOOP == 1:
            for i in nl.affine_range(NT):
                c0 = c_base + i * T
                optk__adamw_tile(po[:, c0:c0 + T], mo[:, c0:c0 + T], vo[:, c0:c0 + T], pi[:, c0:c0 + T], gi[:, c0:c0 + T], mi[:, c0:c0 + T], vi[:, c0:c0 + T], sct, row, P, T, LERP, DIV, SQRT, ENG, WD, bt1, bt2)
        else:
            for i in range(NT):
                c0 = c_base + i * T
                optk__adamw_tile(po[:, c0:c0 + T], mo[:, c0:c0 + T], vo[:, c0:c0 + T], pi[:, c0:c0 + T], gi[:, c0:c0 + T], mi[:, c0:c0 + T], vi[:, c0:c0 + T], sct, row, P, T, LERP, DIV, SQRT, ENG, WD, bt1, bt2)

    @nki.jit
    def optk__adamw3_kernel(p0, g0, m0, v0, p1, g1, m1, v1, ps, gs, ms, vs, sc, T, LERP, DIV, SQRT, ENG, LOOP, INPLACE, WD0, WD1, WD2):
        """Three AdamW updates in one launch: two [128, F] fp32 tables (p, grad, exp_avg, exp_avg_sq each) and the
        [1, M] fp32 stack; sc [1, 48] holds 16 host-staged scalars per group.  INPLACE=1 writes p/m/v of every group
        in place and returns a (1, 1) status; INPLACE=0 returns nine fresh tensors."""
        f32 = nl.float32
        P = optk__P
        pid, nprg = optk__shard()
        sct = optk__sb((P, optk__NSC), f32)
        optk__dma(dst=sct, src=sc.broadcast(0, P))
        M = ps.shape[1]
        if INPLACE == 1:
            optk__adamw_table(p0, m0, v0, p0, g0, m0, v0, sct, 0, T, LERP, DIV, SQRT, ENG, LOOP, WD0, pid, nprg)
            optk__adamw_table(p1, m1, v1, p1, g1, m1, v1, sct, 1, T, LERP, DIV, SQRT, ENG, LOOP, WD1, pid, nprg)
            if pid == 0:
                sb1 = None
                sb2 = None
                if DIV == 2:
                    sb1 = optk__bcast_tile(sct, 2 * 16 + 3, 1, M)
                    sb2 = optk__bcast_tile(sct, 2 * 16 + 4, 1, M)
                optk__adamw_tile(ps[0:1, 0:M], ms[0:1, 0:M], vs[0:1, 0:M], ps[0:1, 0:M], gs[0:1, 0:M], ms[0:1, 0:M], vs[0:1, 0:M], sct, 2, 1, M, LERP, DIV, SQRT, ENG, WD2, sb1, sb2)
            status = nl.ndarray((1, 1), dtype=f32, buffer=nl.shared_hbm)
            st = optk__sb((1, 1), f32)
            nisa.memset(st, 0.0)
            optk__dma(dst=status, src=st)
            return status
        po0 = nl.ndarray(p0.shape, dtype=f32, buffer=nl.shared_hbm)
        mo0 = nl.ndarray(p0.shape, dtype=f32, buffer=nl.shared_hbm)
        vo0 = nl.ndarray(p0.shape, dtype=f32, buffer=nl.shared_hbm)
        po1 = nl.ndarray(p1.shape, dtype=f32, buffer=nl.shared_hbm)
        mo1 = nl.ndarray(p1.shape, dtype=f32, buffer=nl.shared_hbm)
        vo1 = nl.ndarray(p1.shape, dtype=f32, buffer=nl.shared_hbm)
        pos = nl.ndarray(ps.shape, dtype=f32, buffer=nl.shared_hbm)
        mos = nl.ndarray(ps.shape, dtype=f32, buffer=nl.shared_hbm)
        vos = nl.ndarray(ps.shape, dtype=f32, buffer=nl.shared_hbm)
        optk__adamw_table(po0, mo0, vo0, p0, g0, m0, v0, sct, 0, T, LERP, DIV, SQRT, ENG, LOOP, WD0, pid, nprg)
        optk__adamw_table(po1, mo1, vo1, p1, g1, m1, v1, sct, 1, T, LERP, DIV, SQRT, ENG, LOOP, WD1, pid, nprg)
        if pid == 0:
            sb1 = None
            sb2 = None
            if DIV == 2:
                sb1 = optk__bcast_tile(sct, 2 * 16 + 3, 1, M)
                sb2 = optk__bcast_tile(sct, 2 * 16 + 4, 1, M)
            optk__adamw_tile(pos[0:1, 0:M], mos[0:1, 0:M], vos[0:1, 0:M], ps[0:1, 0:M], gs[0:1, 0:M], ms[0:1, 0:M], vs[0:1, 0:M], sct, 2, 1, M, LERP, DIV, SQRT, ENG, WD2, sb1, sb2)
        return (po0, mo0, vo0, po1, mo1, vo1, pos, mos, vos)

    def optk__sumsq_term(src, T, LOOP, pid, nprg, dst_col):
        """dst_col[128, 1] fp32 <- per-partition sum of squares of this program's column half of src [128, F]."""
        f32 = nl.float32
        P, F = src.shape
        Fp = F // nprg
        NT = Fp // T
        c_base = pid * Fp
        ss = optk__sb((P, NT), f32)
        if LOOP == 1:
            for i in nl.affine_range(NT):
                c0 = c_base + i * T
                tb = optk__sb((P, T), src.dtype)
                optk__dma(dst=tb, src=src[:, c0:c0 + T])
                sq = optk__sb((P, T), f32)
                nisa.activation_reduce(dst=sq, op=nl.square, data=tb, reduce_op=nl.add, reduce_res=ss[:, i:i + 1])
        else:
            for i in range(NT):
                c0 = c_base + i * T
                tb = optk__sb((P, T), src.dtype)
                optk__dma(dst=tb, src=src[:, c0:c0 + T])
                sq = optk__sb((P, T), f32)
                nisa.activation_reduce(dst=sq, op=nl.square, data=tb, reduce_op=nl.add, reduce_res=ss[:, i:i + 1])
        nisa.tensor_reduce(dst=dst_col, op=nl.add, data=ss, axis=1)

    @nki.jit
    def optk__gradnorm_kernel(gA, gB, wt, lm, sm, TA, TW, LOOP, WITH_SMALL):
        """out[128, 4] fp32: column 2*pid = per-partition sum of squares of gA + gB (bf16 [128, F] views of the two Muon
        grad shards), column 2*pid + 1 = of wt + lm (fp32 [128, F] views of the table grads), plus sm [1, M] fp32 (the
        reduced 1-element stack) on partition 0 when WITH_SMALL=1."""
        f32 = nl.float32
        P = optk__P
        pid, nprg = optk__shard()
        out = nl.ndarray((P, 4), dtype=f32, buffer=nl.shared_hbm)
        cA = optk__sb((P, 1), f32)
        cB = optk__sb((P, 1), f32)
        cW = optk__sb((P, 1), f32)
        cL = optk__sb((P, 1), f32)
        optk__sumsq_term(gA, TA, LOOP, pid, nprg, cA)
        optk__sumsq_term(gB, TA, LOOP, pid, nprg, cB)
        optk__sumsq_term(wt, TW, LOOP, pid, nprg, cW)
        optk__sumsq_term(lm, TW, LOOP, pid, nprg, cL)
        res = optk__sb((P, 2), f32)
        nisa.tensor_tensor(dst=res[:, 0:1], data1=cA, data2=cB, op=nl.add, engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=res[:, 1:2], data1=cW, data2=cL, op=nl.add, engine=nisa.engine.vector)
        if WITH_SMALL == 1:
            if pid == 0:
                M = sm.shape[1]
                tsm = optk__sb((1, M), f32)
                optk__dma(dst=tsm, src=sm[0:1, 0:M])
                sq1 = optk__sb((1, M), f32)
                s1 = optk__sb((1, 1), f32)
                nisa.activation_reduce(dst=sq1, op=nl.square, data=tsm, reduce_op=nl.add, reduce_res=s1)
                nisa.tensor_tensor(dst=res[0:1, 1:2], data1=res[0:1, 1:2], data2=s1, op=nl.add, engine=nisa.engine.vector)
        optk__dma(dst=out[:, 2 * pid:2 * pid + 2], src=res)
        return out
    optk__HAS_NKI = True
except Exception:
    optk__HAS_NKI = False

def optk__launch(kernel):
    """kernel[2] under an LNC=2 runtime (both physical cores take half the columns), plain call otherwise."""
    return kernel[optk__LNC] if optk__LNC == 2 else kernel

def optk__disable(which: str, where: str, err: BaseException) -> None:
    global optk__ADAMW_DISABLED, optk__GRADNORM_DISABLED, optk__LAST_ERROR
    optk__LAST_ERROR = f'{where}: {type(err).__name__}: {err}'
    if which == 'adamw' and (not optk__ADAMW_DISABLED):
        optk__ADAMW_DISABLED = True
        print(f'[nki_optim] AdamW kernel disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)
    if which == 'gradnorm' and (not optk__GRADNORM_DISABLED):
        optk__GRADNORM_DISABLED = True
        print(f'[nki_optim] grad-norm kernel disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)

def optk_adamw_kernel_available(device: torch.device) -> bool:
    return optk__HAS_NKI and (not optk__ADAMW_DISABLED) and (device.type == 'neuron')

def optk_gradnorm_kernel_available(device: torch.device) -> bool:
    return optk__HAS_NKI and (not optk__GRADNORM_DISABLED) and (device.type == 'neuron')

def optk_run_adamw3(p0, g0, m0, v0, p1, g1, m1, v1, ps, gs, ms, vs, sc, wd_flags, T: int=optk_ADAMW_T, lerp: int=optk_ADAMW_LERP, div: int=optk_ADAMW_DIV, sqrt: int=optk_ADAMW_SQRT, eng: int=optk_ADAMW_ENG, loop: int=optk_ADAMW_LOOP, inplace: int=optk_ADAMW_INPLACE):
    """Direct launch (no fallback) on [128, F] fp32 views, [1, M] stack views and sc [1, 48].  Returns the status
    tensor (inplace) or the nine outputs."""
    return optk__launch(optk__adamw3_kernel)(p0, g0, m0, v0, p1, g1, m1, v1, ps, gs, ms, vs, sc, int(T), int(lerp), int(div), int(sqrt), int(eng), int(loop), int(inplace), int(wd_flags[0]), int(wd_flags[1]), int(wd_flags[2]))

def optk_run_gradnorm(gA, gB, wt, lm, sm, T: int=optk_GRADNORM_T, loop: int=optk_GRADNORM_LOOP):
    """Direct launch: out [128, 4] fp32 for [128, F] views (gA/gB bf16, wt/lm fp32) and sm [1, M] fp32 or None."""
    with_small = 1 if sm is not None else 0
    if sm is None:
        sm = torch.zeros(1, 1, dtype=torch.float32, device=gA.device)
    return optk__launch(optk__gradnorm_kernel)(gA, gB, wt, lm, sm, int(T), int(T // 2), int(loop), int(with_small))

def optk__adamw_plan(opt):
    """Validate the optimizer's AdamW groups against the kernel's fixed layout: group 0 and 1 hold one fp32 2-D parameter
    each (numel % 128 == 0), group 2 is the stacked 1-element group with nothing left on the per-parameter path.
    Returns the plan dict or None (-> eager)."""
    groups = [g for g in opt.param_groups if g['kind'] == 'adamw']
    if len(groups) != 3:
        return None
    tables = []
    for g in groups[:2]:
        ps = [p for p in g['params']]
        if len(ps) != 1 or g.get('stack_params'):
            return None
        p = ps[0]
        if p.dtype != torch.float32 or p.dim() != 2 or p.numel() % (optk__P * optk_ADAMW_T * 2) != 0 or (not p.is_contiguous()):
            return None
        tables.append(p)
    gs = groups[2]
    stack = gs.get('stack_params')
    if not stack or gs.get('stack_master') is None:
        return None
    if any((id(p) not in gs['stack_ids'] for p in gs['params'])):
        return None
    if gs['stack_master'].dtype != torch.float32:
        return None
    return {'groups': groups, 'tables': tables, 'stack_group': gs}

@torch.no_grad()
def optk_adamw_fused_step(opt) -> bool:
    """The fused AdamW step for the optimizer's three AdamW groups.  Returns True when the groups were updated (the
    caller must then skip its per-group _adamw_step), False when the layout is not the kernel's (eager).  A kernel
    failure disables the kernel for the process and falls back to the exact eager update for this step."""
    plan = opt.__dict__.get('_nki_adamw_plan')
    if plan is None:
        plan = optk__adamw_plan(opt)
        opt.__dict__['_nki_adamw_plan'] = plan if plan is not None else False
    if not plan:
        return False
    tables, gs, groups = (plan['tables'], plan['stack_group'], plan['groups'])
    dev = tables[0].device
    if not optk_adamw_kernel_available(dev):
        return False
    for p in tables:
        if p.grad is None or p.grad.dtype != torch.float32 or (not p.grad.is_contiguous()) or (p.grad.shape != p.shape):
            return False
    if not gs.get('stack_staged'):
        opt.stage_scalar_grads(gs, distributed=False)
    st = []
    for p in tables:
        s = opt.state[p]
        if not s:
            s['exp_avg'] = torch.zeros_like(p)
            s['exp_avg_sq'] = torch.zeros_like(p)
        st.append(s)
    sc_cpu = opt.__dict__.get('_nki_adamw_sc_cpu')
    if sc_cpu is None:
        sc_cpu = torch.zeros(1, optk__NSC, dtype=torch.float32)
        opt.__dict__['_nki_adamw_sc_cpu'] = sc_cpu
        opt.__dict__['_nki_adamw_sc_dev'] = torch.zeros(1, optk__NSC, dtype=torch.float32, device=dev)
    sc_dev = opt.__dict__['_nki_adamw_sc_dev']
    wd_flags = []
    for row, g in enumerate(groups):
        b1, b2 = g['betas']
        n = g.get('step_count', 0) + 1
        sc_cpu[0, 16 * row:16 * row + 16] = optk_group_scalars(g['lr_cpu'], b1, b2, g['eps'], g['weight_decay'], n)
        wd_flags.append(1 if g['weight_decay'] > 0 else 0)
    sc_dev.copy_(sc_cpu)
    p0, p1 = tables
    args = (p0.view(optk__P, -1), p0.grad.view(optk__P, -1), st[0]['exp_avg'].view(optk__P, -1), st[0]['exp_avg_sq'].view(optk__P, -1), p1.view(optk__P, -1), p1.grad.view(optk__P, -1), st[1]['exp_avg'].view(optk__P, -1), st[1]['exp_avg_sq'].view(optk__P, -1), gs['stack_master'].view(1, -1), gs['stack_grad'].view(1, -1), gs['stack_exp_avg'].view(1, -1), gs['stack_exp_avg_sq'].view(1, -1), sc_dev, wd_flags)
    try:
        optk_run_adamw3(*args)
    except Exception as err:
        optk__disable('adamw', 'adamw3 launch', err)
        gs['stack_staged'] = True
        for g in groups:
            opt._adamw_step(g)
        return True
    for p in tables:
        torch.autograd.graph.increment_version(p)
    torch.autograd.graph.increment_version(gs['stack_master'])
    for g in groups:
        g['step_count'] = g.get('step_count', 0) + 1
        b1, b2 = g['betas']
        g['bias1_cpu'].fill_(1 - b1 ** g['step_count'])
        g['bias2_cpu'].fill_(1 - b2 ** g['step_count'])
    gs['stack_staged'] = False
    return True

@torch.no_grad()
def optk_reduce_fused(reducer, clip: float) -> torch.Tensor:
    """GradReducer.reduce with the norm kernel: the collectives exactly as the eager body issues them (per-parameter
    all_reduce for the non-stacked params, the stack's cat + all_reduce, the Muon reduce-scatters), then ONE kernel
    launch for the four big norm terms + the stack, then the shard_sq all_reduce and the gate.  If the kernel is not
    applicable or fails, the norm terms are computed with the eager ops from the same (already reduced) tensors, so the
    fallback never re-issues a collective."""
    import torch.distributed as dist
    distributed = dist.is_available() and dist.is_initialized()
    opt = reducer.optimizer
    big = []
    for p in reducer.params:
        if p.grad is None or id(p) in reducer.skip_ids:
            continue
        if distributed:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        big.append(p)
    stack_flats = []
    for group in reducer.stack_groups:
        stack_flats.append(opt.stage_scalar_grads(group, distributed))
    shards = []
    for group in reducer.muon_groups:
        opt.stage_muon_grads(group, distributed, norm=False)
        shards.append(opt.state[group['params'][0]]['grad_shard'])
    tabs = [p for p in big if p.grad.dim() == 2 and p.grad.dtype == torch.float32 and p.grad.is_contiguous() and (p.grad.numel() % (optk__P * optk_GRADNORM_T) == 0)]
    extra = [p for p in big if not any((p is q for q in tabs))]
    sq_tables = None
    sq_shards = None
    kern_ok = optk_gradnorm_kernel_available(reducer.last_gnorm.device) and len(tabs) == 2 and (len(shards) == 2) and (len(stack_flats) <= 1) and all((s.dtype == torch.bfloat16 and s.is_contiguous() and (s.numel() % (optk__P * optk_GRADNORM_T * 2) == 0) for s in shards))
    if kern_ok:
        sm = stack_flats[0].view(1, -1) if stack_flats else None
        try:
            out = optk_run_gradnorm(shards[0].view(optk__P, -1), shards[1].view(optk__P, -1), tabs[0].grad.view(optk__P, -1), tabs[1].grad.view(optk__P, -1), sm)
            t = out.view(optk__P, 2, 2).sum(dim=(0, 1))
            sq_shards, sq_tables = (t[0], t[1])
            for p in extra:
                sq_tables = sq_tables + p.grad.float().square().sum()
        except Exception as err:
            optk__disable('gradnorm', 'gradnorm launch', err)
            sq_tables = sq_shards = None
    if sq_shards is None:
        sq = None
        for p in big:
            s = p.grad.float().square().sum()
            sq = s if sq is None else sq + s
        for flat in stack_flats:
            s = flat.float().square().sum()
            sq = s if sq is None else sq + s
        sh = None
        for shard in shards:
            s = shard.float().square().sum()
            sh = s if sh is None else sh + s
        sq_tables, sq_shards = (sq, sh)
    if sq_shards is not None:
        reducer.shard_sq.copy_(sq_shards)
        if distributed:
            dist.all_reduce(reducer.shard_sq, op=dist.ReduceOp.SUM)
        s = reducer.shard_sq[0]
        sq_total = s if sq_tables is None else sq_tables + s
    else:
        sq_total = sq_tables
    gnorm = sq_total.sqrt()
    host_gate = getattr(reducer, 'host_gate', False)
    finite = None if host_gate else torch.isfinite(gnorm)
    if clip > 0:
        scale = torch.clamp(clip / (gnorm + 1e-06), max=1.0)
        for p in big:
            p.grad.mul_(scale)
        for group in reducer.stack_groups:
            group['stack_grad'].mul_(scale)
        for group in reducer.muon_groups:
            opt.state[group['params'][0]]['grad_shard'].mul_(scale)
    if host_gate:
        return gnorm
    reducer.last_gnorm.copy_(torch.where(finite, gnorm, torch.zeros_like(gnorm)))
    reducer.bad_steps += (~finite).to(torch.float32)
    return finite
optk_REDUCE_HOST_GATE = True

@torch.no_grad()
def optk_muon_finish_view(opt, state, g, w, k, s, world, rank, upd_fn) -> bool:
    """The slab-0 finish of _muon_step_sharded (payload "params", persistent views): rank R's matrices are the
    contiguous stack rows [R*k, (R+1)*k), so the k per-matrix copy_ into param_shard go.  "sub": the update is computed
    from the stack view and written into param_shard by one torch.sub(out=) (the same two SR roundings of the same
    values as today's upd.to(bf16) + w.sub_); "copy": one contiguous copy_ then today's ops.  Returns False when the
    preconditions do not hold (caller runs today's path)."""
    if s != k or not state.get('param_views'):
        return False
    w_view = state['stacked_params'][rank * k:(rank + 1) * k]
    upd = upd_fn(g, w_view)
    torch.sub(w_view, upd if upd.dtype == w.dtype else upd.to(w.dtype), out=w)
    opt._all_gather(state['stacked_params'][0:world * k], w)
    return True

class optk__ApiNamespace:
    pass
optk_API = optk__ApiNamespace()
optk_API.USE_NKI_ADAMW = optk_USE_NKI_ADAMW
optk_API.USE_NKI_GRADNORM = optk_USE_NKI_GRADNORM
optk_API.MUON_SHARD_VIEW = optk_MUON_SHARD_VIEW
optk_API.ADAMW_STACK_SMALL = optk_ADAMW_STACK_SMALL
optk_API.adamw_fused_step = optk_adamw_fused_step
optk_API.reduce_fused = optk_reduce_fused
optk_API.REDUCE_HOST_GATE = optk_REDUCE_HOST_GATE
optk_API.muon_finish_view = optk_muon_finish_view
optk_API.config_string = optk_config_string
nki_optim = optk_API

class _FreezeNS:

    def __init__(self, **kw):
        self.__dict__.update(kw)
_fp8_bf16_ns = _FreezeNS(FcRelu2Fn=gemmk_FcRelu2Fn, nki_matmul=gemmk_nki_matmul, fused_fc_relu2_forward=gemmk_fused_fc_relu2_forward, eager_relu2_backward=gemmk_eager_relu2_backward, _relu2_backward=gemmk__relu2_backward)
"FP8 e4m3 `double_row` c_fc forward for the MLP (plan-v13 T2, slice 1).  DEFAULT OFF.\n\nWhat this is\n------------\nOne switch, `R1_NKI_GEMM_FP8`, that replaces the *forward* of the MLP's c_fc GEMM -- and only that\none GEMM of the six the MLP issues per block -- with an fp8 e4m3 `double_row` NKI kernel.  Nothing\nelse in the model changes: the ReLU^2, the backward (dX, dW), c_proj, attention, the loss head and\nthe optimizer all stay exactly as they are today.  The point of the slice is NOT its own net gain\n(+2.3 % steps ~ -0.0012 bpb, which the precision cost may well swallow) but to MEASURE the bpb\nprice of fp8 at one MLP GEMM, so the full lane -- all six, ~ +8.5 % steps ~ -0.0042 bpb -- can be\nfunded or killed on a number instead of the untested +0.0018 guess in docs/road-to-0930/fp8.md.\n\n\nWhy `qdq` exists, and why it is the arm that decides the lane\n------------------------------------------------------------\nEvery finite e4m3 value is EXACTLY representable in bf16 (3 mantissa bits of 8; exponent range\n2^-9 .. 2^8 inside bf16's 2^-126 .. 2^127 -- verified, 254/254 values round-trip bitwise).  So a\nbf16 matmul over e4m3-rounded operands with fp32 accumulation forms exactly the same products as\nthe fp8 kernel, and differs only in the fp32 accumulation order.  `qdq` therefore measures the fp8\nquality cost faithfully while being SLOWER than the baseline, which is the point: the arm loses\nsteps instead of gaining them, so the training-loss curve at equal step count isolates the\nper-token cost with no throughput confound.  `1` then measures the speed.  Run `qdq` first.\n\nThe arithmetic\n--------------\n    out = (s_a * A)^T (s_w * W) / (s_a * s_w),  both operands rounded to e4m3 after scaling,\n    fp32 PSUM accumulation, one bf16 rounding on the way out (today's `pre`).\n\nScales are FIXED powers of two, not amax.  Three reasons:\n  * a power of two is an exponent shift, so the scaling itself is exact -- no extra rounding before\n    the fp8 round -- and e4m3 being a floating format, the scale does not change relative precision\n    at all.  All a scale decides is how much of the top tail clips at 448 and how much of the bottom\n    tail falls into the subnormals below 2^-6;\n  * the activation side needs no measurement: c_fc's input is a parameter-free RMSNorm output, so\n    its per-row RMS is 1 BY CONSTRUCTION for the whole of training.  s_a = 16 puts a GAUSSIAN amax\n    at ~82 (2.9x under the legacy max 240; the original text said 5.4x under 448) and 0.08 % of\n    elements subnormal.  [2026-09-21] But a unit-RMS row's single element is bounded only by\n    sqrt(n_embd) = 32, i.e. 512 at s_a = 16 -- and outlier channels reach it: F1_qdq diverged at\n    step ~75 for exactly this reason (D1-D3, FIRST_CHIP_RUN_FAILED.md).  s_a = 4 is safe by\n    construction (128 of 240) at identical precision; the clamp (CLIP) is the safety net;\n  * the weight side was calibrated off the real checkpoints (agents/fp8_integration/calibrate_scales.py):\n    c_fc |W| rms runs 0.0125 at init to 0.063-0.084 at the end of a 30-minute run, a 5.0-6.7x drift,\n    and end-of-run amax 0.39-0.73.  s_w = 256 gives amax*s_w = 5.5 at init and 100-187 at the end,\n    i.e. only 1.28x of clipping headroom against 240 at the worst point measured (the original text\n    said 2.4x against 448), at the cost of 0.07 % of (negligible) elements going subnormal.  Weights\n    grow without a bound, so the weight cast needs the clamp at every scale (D3_overflow).  The\n    asymmetry is deliberate -- clipping truncates the LARGEST weights and biases the product,\n    underflow zeroes the smallest and does almost nothing -- so the scale is biased low.\nAn amax scale is possible but not free: `nisa.activation`'s reduce_op only takes nl.add, so a max\nneeds a separate `nisa.tensor_reduce` pass, and the resulting scale is a DEVICE scalar -- feeding it\nto the kernel as a Python float costs a host sync per step (fatal), and feeding it as an SBUF tile\ndepends on `nisa.activation(scale=<tile>)` being accepted, which is unverified.  See PATCH.md S7.\n\nWhere the weight quantisation lives\n-----------------------------------\nIn `quantised_weight()`, keyed on (step epoch, data_ptr, _version) and held in `_WCACHE`.  The epoch\nis bumped by `begin_step()`, which train.py calls once per optimizer step, so the fp8 copy of each\nc_fc weight is produced ONCE per step and reused by all 2 (device batch 32) or 4 (device batch 16)\nmicro-steps.  `_version` and `data_ptr` are belt and braces: either one changing on its own also\ninvalidates, so a missed `begin_step()` cannot serve a stale weight for more than the step in which\nthe parameter was actually written.  `R1_NKI_GEMM_FP8_WCACHE=call` turns the cache off entirely\n(requantise every micro-step) -- the control arm that proves the cache is not the source of a delta.\n\nFallback chain (exact, and disable-once)\n----------------------------------------\n    fp8 kernel -> nki_gemm.fused_fc_relu2_forward (today's bf16 fused kernel) -> eager matmul.\nAnything that does not fit -- no NKI, not a neuron device, not bf16, a shape the plan refuses, an\nSBUF plan over budget, or a kernel that raises (printed once, kernels then disabled for the process)\n-- takes the next link.  On CPU the chain always reaches the last link, so a CPU smoke with the\n\nKnobs (env, read once at import)\n\n  [slice 2, 2026-09-21] the plain fp8 matmul `nki_matmul_fp8` / `run_mm_fp8` (class B = c_proj forward first):\n\n  The fp8 format on this chip is the LEGACY e4m3 (max finite 240, then +-inf and NaN), not the OCP e4m3fn (max\n  448) that torch has and that the 448 in `emulate_fp8_round` assumes -- work/agents/fp8/NOTES.md section 2.\n  F1_qdq (2026-09-21, FIRST_CHIP_RUN_FAILED.md) diverged at step ~75; a scaled activation above 240 is the\n  prime suspect, which is why the slice-2 kernel clamps before it casts by default and why every number in the\n  slice-2 code is 240.\n"
import os
import torch
fp8k__P = 128
fp8k__CH = 512
fp8k__KS = 256
fp8k__J = fp8k__KS // fp8k__P
fp8k__env = os.environ.get
fp8k_MODE = '1'
fp8k_USE_NKI_GEMM_FP8 = fp8k_MODE in ('1', 'qdq')
fp8k_QDQ_ONLY = fp8k_MODE == 'qdq'
fp8k_EVAL_MODE = 'train'
fp8k_A_SCALE = 4
fp8k_W_SCALE = 64
fp8k_WCACHE = 'step'
fp8k_TB = 1024
fp8k_GROUP = 2
fp8k__LNC = int(fp8k__env('NEURON_LOGICAL_NC_CONFIG', '1'))
fp8k__SHARD = True
fp8k_SBUF_BUDGET = 200 * 1024
fp8k__E4M3_MAX_LEGACY = 240.0
fp8k_TB_B = 512
fp8k_GROUP_B = 2
fp8k__EVAC_NAMES = {'alt': 0, 'scalar': 1, 'vector': 2}
fp8k_EVAC_B = fp8k__EVAC_NAMES['alt']
fp8k__CLIP_NAMES = {'none': 0, 'vec': 1, 'fused': 2}
fp8k_CLIP = fp8k__CLIP_NAMES['vec']
fp8k_LOOP_B = 0
fp8k__CONV_NAMES = {'slab': 0, 'flat': 1}
fp8k_CONV_B = fp8k__CONV_NAMES['slab']
fp8k_A_SCALE_B = 4
fp8k_W_SCALE_B = 64
fp8k_INV_SCALE = 1.0 / (fp8k_A_SCALE * fp8k_W_SCALE)

def fp8k_clip_arg(mode: int=None) -> int:
    """The `clip` argument for the kernels whose operand is SIGNED (_quant_w_kernel, _qdq_kernel, _fc_relu2_fp8_kernel):
    they implement only the vec clamp, so `fused` collapses to it and only `none` disables it.  (A review caught the
    earlier `int(CLIP == 1)`, which silently turned the clamp OFF under CLIP=fused.)"""
    m = fp8k_CLIP if mode is None else mode
    return 0 if m == 0 else 1

def fp8k_clip_bound(scale: float) -> float:
    """The clamp bound in the UNSCALED bf16 domain: the largest bf16 value b with b * scale <= 240.  The clamped value
    lives in a bf16 tile, so for a non-power-of-two scale 240/scale would round UP in bf16 (2.4 -> 2.40625 at scale 100,
    i.e. 240.6 after scaling, past the legacy max); power-of-two scales are exact."""
    b = torch.tensor(fp8k__E4M3_MAX_LEGACY / scale, dtype=torch.float32).to(torch.bfloat16)
    for _ in range(4):
        if float(b) * scale <= fp8k__E4M3_MAX_LEGACY:
            break
        b = torch.nextafter(b, torch.tensor(0.0, dtype=torch.bfloat16))
    return float(b)
fp8k__KERNEL_DISABLED = False
fp8k__LAST_KERNEL_ERROR = None
fp8k__STEP_EPOCH = 0
fp8k__WCACHE = {}

def fp8k_plan_fp8(M: int, K: int, N: int, tb: int=fp8k_TB, group: int=fp8k_GROUP, nprg: int=2):
    """SBUF bytes per partition of the fused fp8 plan, or None if the shape is not tileable.

    weight (fp8, resident)        NK_d * J * N                 = K/256 * 2 * N
    a^T block (bf16, from hwt)    NK_b * tb * 2                = K/128 * tb * 2
    a^T block (fp8, converted)    NK_d * J * tb
    pre + h row staging           2 * N * 2
    """
    if M % fp8k__P or K % fp8k__KS or N % fp8k__CH or (M < 2 * fp8k__P) or (K < fp8k__KS) or (N < fp8k__CH):
        return None
    if tb % fp8k__P or tb < fp8k__P or tb % 16:
        return None
    if group < 1 or group > 8:
        return None
    nkd = K // fp8k__KS
    nkb = K // fp8k__P
    w_bytes = nkd * fp8k__J * N
    at_bytes = nkb * tb * 2 + nkd * fp8k__J * tb
    st_bytes = 2 * N * 2
    return {'nkd': nkd, 'nkb': nkb, 'nc': N // fp8k__CH, 'sbuf': w_bytes + 2 * at_bytes + st_bytes}

def fp8k_plan_quant(K: int, N: int):
    """SBUF bytes per partition of the weight quantiser / qdq kernel, or None."""
    if K % fp8k__P or N < 1:
        return None
    return {'sbuf': N * 2 + N + N * 2}
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    fp8k__PM = nisa.matmul_perf_mode.double_row

    def fp8k__shard():
        if nl.program_ndim() > 0:
            return (nl.program_id(0), nl.num_programs(0))
        return (0, 1)

    def fp8k__span(n, pid, nprg):
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
        return (i0, i0 + cnt)

    def fp8k__ceil_div(a, b):
        return (a + b - 1) // b

    def fp8k__ptiles(n, shape, dtype, buf):
        """A tuple of n (<= 8) fresh tiles; an if-chain, as nki_gemm does (no comprehensions)."""
        t0 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 1:
            return (t0,)
        t1 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 2:
            return (t0, t1)
        t2 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 3:
            return (t0, t1, t2)
        t3 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 4:
            return (t0, t1, t2, t3)
        t4 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 5:
            return (t0, t1, t2, t3, t4)
        t5 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 6:
            return (t0, t1, t2, t3, t4, t5)
        t6 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 7:
            return (t0, t1, t2, t3, t4, t5, t6)
        t7 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        return (t0, t1, t2, t3, t4, t5, t6, t7)

    def fp8k__load_at_hwt(a2, K, NKB, row0, ntok, dt):
        """a^T block [P, NKB, ntok] in bf16 by the hardware-DGE strip transpose.

        VERBATIM the proven `xt=hwt` branch of nki_gemm._load_at_block (plan-v12 lever 2): one
        `dma_transpose(dge_mode=hwdge)` per 16-token strip covering all NKB k-slabs.  hwdge needs
        src.shape[0] == 16 and a 2-byte dtype, which is why the transpose happens in bf16 BEFORE the
        fp8 conversion and not after.  The per-slab form lowers to runtime-generated descriptors
        cached per input address and cost 3.3-4.4 ms per call in situ against 1.2 ms on a reused
        buffer -- the rung-1 collapse.  Do not change this loop.
        """
        at = nl.ndarray((fp8k__P, NKB, ntok), dtype=dt, buffer=nl.sbuf)
        for s in range(ntok // 16):
            nisa.dma_transpose(dst=at.ap([[NKB * ntok, fp8k__P], [ntok, NKB], [1, 1], [1, 16]], offset=s * 16), src=a2.ap([[K, 16], [fp8k__P, NKB], [1, 1], [1, fp8k__P]], offset=(row0 + s * 16) * K), dge_mode=nisa.dge_mode.hwdge)
        return at

    def fp8k__quantise_block(at, NKD, ntok, a_scale):
        """bf16 a^T block [P, NKB, ntok] -> fp8 [P, NKD, J, ntok], scale folded into the SAME
        ScalarE instruction as the conversion (`nisa.activation` applies scale BEFORE the op).

        Index map: the bf16 slab kk holds contraction row kk*128 + p; the fp8 element (p, kd, j, t)
        must hold contraction row kd*256 + j*128 + p, so kk = kd*J + j.  Both slices are [P, ntok]
        with the same free stride, so this is a pure reindex -- no gather, no transpose.
        """
        a_f8 = nl.ndarray((fp8k__P, NKD, fp8k__J, ntok), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for kd in range(NKD):
            for j in range(fp8k__J):
                nisa.activation(dst=a_f8[:, kd, j, :], op=nl.copy, data=at[:, kd * fp8k__J + j, :], scale=a_scale)
        return a_f8

    def fp8k__load_w_fp8(w_u8, K, N, NKD):
        """Resident fp8 weight [P, NKD, J, N] from the uint8 [K, N] tensor (bitcast in SBUF, as
        mm_fp8 does: NRT has no fp8 dtype, so fp8 bytes travel inside uint8 and nothing is cast)."""
        w_sb = nl.ndarray((fp8k__P, NKD, fp8k__J, N), dtype=nl.uint8, buffer=nl.sbuf)
        for kd in range(NKD):
            for j in range(fp8k__J):
                r0 = kd * fp8k__KS + j * fp8k__P
                nisa.dma_copy(dst=w_sb[:, kd, j, :], src=w_u8[r0:r0 + fp8k__P, :])
        return w_sb.view(nl.float8_e4m3)

    def fp8k__fc_tile_fp8(a_f8, w_f8, h, pre, ti, bt0, N, NKD, NC, npass, group, inv_scale, dt, f32, save_pre):
        """One 128-token tile: passes of `group` output chunks sharing one stationary per k-slab,
        `pre` = bf16(PSUM * inv_scale) on ScalarE (ONE rounding, and the descale is free because the
        evacuation already runs an activation), `h` = (pre max 0) * pre on VectorE -- exactly the
        instruction pair of nki_gemm._fc_tile, so `h` is today's function of today's `pre`."""
        row0 = (bt0 + ti) * fp8k__P
        pst = nl.ndarray((fp8k__P, N), dtype=dt, buffer=nl.sbuf)
        hst = nl.ndarray((fp8k__P, N), dtype=dt, buffer=nl.sbuf)
        for p in range(npass):
            c0 = p * group
            g = NC - c0
            if g > group:
                g = group
            pss = fp8k__ptiles(g, (fp8k__P, fp8k__CH), f32, nl.psum)
            for kd in range(NKD):
                st = a_f8[:, kd, :, ti * fp8k__P:(ti + 1) * fp8k__P]
                for j in range(g):
                    c = c0 + j
                    nisa.nc_matmul(dst=pss[j], stationary=st, moving=w_f8[:, kd, :, c * fp8k__CH:(c + 1) * fp8k__CH], accumulate=kd > 0, perf_mode=fp8k__PM)
            for j in range(g):
                c = c0 + j
                pj = pst[:, c * fp8k__CH:(c + 1) * fp8k__CH]
                nisa.activation(dst=pj, op=nl.copy, data=pss[j], scale=inv_scale)
                nisa.scalar_tensor_tensor(dst=hst[:, c * fp8k__CH:(c + 1) * fp8k__CH], data=pj, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=pj)
        nisa.dma_copy(dst=h[row0:row0 + fp8k__P, :], src=hst)
        if save_pre == 1:
            nisa.dma_copy(dst=pre[row0:row0 + fp8k__P, :], src=pst)

    @nki.jit
    def fp8k__fc_relu2_fp8_kernel(a2, w_u8, tb, group, a_scale, inv_scale, clip, a_bound, save_pre):
        """h[M, N] = relu(pre)^2 and pre[M, N] = bf16((a2 @ w) / (s_a s_w)) with both matmul operands
        in fp8 e4m3 and the contraction double-pumped (256 per nc_matmul).

        a2 is the bf16 activation in its natural [M, K] training layout; w_u8 is the fp8 weight,
        already scaled and quantised, carried as uint8 [K, N].  clip == 1 clamps every bf16 block to
        +-a_bound (= 240 / s_a) on VectorE before the ScalarE cast (_clip_block, slice 2): legacy e4m3
        has no finite value above 240, and the RMSNorm output this kernel eats can reach sqrt(n_embd) = 32
        in one element, i.e. 512 at s_a = 16."""
        M = a2.shape[0]
        K = a2.shape[1]
        N = w_u8.shape[1]
        dt = a2.dtype
        f32 = nl.float32
        h = nl.ndarray((M, N), dtype=dt, buffer=nl.shared_hbm)
        pre = nl.ndarray((M, N), dtype=dt, buffer=nl.shared_hbm)
        NKD = K // fp8k__KS
        NKB = K // fp8k__P
        NC = N // fp8k__CH
        npass = fp8k__ceil_div(NC, group)
        pid, nprg = fp8k__shard()
        t0, t1 = fp8k__span(M // fp8k__P, pid, nprg)
        w_f8 = fp8k__load_w_fp8(w_u8, K, N, NKD)
        ntb = tb // fp8k__P
        nblk = fp8k__ceil_div(t1 - t0, ntb)
        for b in range(nblk):
            bt0 = t0 + b * ntb
            bt1 = bt0 + ntb
            if bt1 > t1:
                bt1 = t1
            nt = bt1 - bt0
            at = fp8k__load_at_hwt(a2, K, NKB, bt0 * fp8k__P, nt * fp8k__P, dt)
            if clip == 1:
                fp8k__clip_block(at, NKB, a_bound)
            a_f8 = fp8k__quantise_block(at, NKD, nt * fp8k__P, a_scale)
            for ti in nl.affine_range(nt):
                fp8k__fc_tile_fp8(a_f8, w_f8, h, pre, ti, bt0, N, NKD, NC, npass, group, inv_scale, dt, f32, save_pre)
        return (h, pre)

    def fp8k__clip_block(at, NKB, bound):
        """Clamp the bf16 a^T block in place to [-bound, bound] on VectorE, one tensor_scalar per k-slab, BEFORE the
        ScalarE scale+cast; bound = 240 / s_a keeps every scaled value inside legacy e4m3's finite range.  AWS's own
        quantiser does the same clip (vendor/nkilib/core/quantization/fp8_quantize.py, static_quantization: a
        tensor_scalar(min, max) into the fp8 tile); here the clip stays in bf16 so the cast remains on ScalarE."""
        for kk in range(NKB):
            nisa.tensor_scalar(dst=at[:, kk, :], data=at[:, kk, :], op0=nl.minimum, operand0=bound, op1=nl.maximum, operand1=-bound, engine=nisa.engine.vector)

    def fp8k__clip_tile(stage, bound):
        """Clamp one bf16 [P, C] staging tile in place to [-bound, bound] on VectorE (bound = 240 / scale), so the
        ScalarE cast that follows never sees a scaled value outside legacy e4m3's finite range."""
        nisa.tensor_scalar(dst=stage, data=stage, op0=nl.minimum, operand0=bound, op1=nl.maximum, operand1=-bound, engine=nisa.engine.vector)

    @nki.jit
    def fp8k__quant_w_kernel(w, scale, clip, bound):
        """bf16 [K, N] -> fp8 e4m3 bytes in a uint8 [K, N] tensor, scaled by `scale` in the same
        ScalarE instruction.  One 128-row slab at a time; ~15 MB of traffic per c_fc weight.
        clip == 1: clamp the slab to +-bound (= 240 / scale) first, one VectorE instruction per slab
        (once per optimizer step, so free): s_w = 256 leaves the trained weights only 1.3x under 240."""
        K = w.shape[0]
        N = w.shape[1]
        out = nl.ndarray((K, N), dtype=nl.uint8, buffer=nl.shared_hbm)
        pid, nprg = fp8k__shard()
        r0, r1 = fp8k__span(K // fp8k__P, pid, nprg)
        for i in range(r1 - r0):
            k0 = (r0 + i) * fp8k__P
            stage = nl.ndarray((fp8k__P, N), dtype=w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=stage, src=w[k0:k0 + fp8k__P, :])
            if clip == 1:
                fp8k__clip_tile(stage, bound)
            q_u8 = nl.ndarray((fp8k__P, N), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.activation(dst=q_u8.view(nl.float8_e4m3), op=nl.copy, data=stage, scale=scale)
            nisa.dma_copy(dst=out[k0:k0 + fp8k__P, :], src=q_u8)
        return out
    fp8k__HAS_NKI = True
except Exception:
    fp8k__HAS_NKI = False
try:
    fp8k__bf16 = _fp8_bf16_ns
except Exception:
    fp8k__bf16 = None

def fp8k__disable_kernel(where: str, err: BaseException) -> None:
    global fp8k__KERNEL_DISABLED, fp8k__LAST_KERNEL_ERROR
    if not fp8k__KERNEL_DISABLED:
        fp8k__KERNEL_DISABLED = True
        fp8k__LAST_KERNEL_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[nki_gemm_fp8] kernel disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)

def fp8k__launch(kernel):
    return kernel[fp8k__LNC] if fp8k__LNC == 2 else kernel

def fp8k__nprg() -> int:
    return 2 if fp8k__LNC == 2 else 1

def fp8k_active(training: bool) -> bool:
    """True when MLP.forward should take the fp8 branch at all.  The default keeps the forward in
    bf16 at eval (`R1_NKI_GEMM_FP8_EVAL=train`): the weights are the trained ones either way, and
    evaluating them at higher precision is the dropout convention, not a different model.  `always`
    makes the evaluated function bit-for-bit the trained one; it is the conservative reading of the
    consistency rule and it is Danish's call which one a submission ships."""
    return bool(training)

def fp8k__usable(*ts) -> bool:
    if not fp8k__HAS_NKI or fp8k__KERNEL_DISABLED:
        return False
    for t in ts:
        if t.device.type != 'neuron' or t.dtype != torch.bfloat16 or t.dim() != 2 or (not t.is_contiguous()):
            return False
    return True

def fp8k_begin_step() -> None:
    """Called by train.py once per optimizer step: the cached fp8 weights die with the step that
    produced them.  Cheap (one int), and safe to call when the switch is off."""
    global fp8k__STEP_EPOCH
    fp8k__STEP_EPOCH += 1

def fp8k__wcache_get(w: torch.Tensor, tag: str):
    ent = fp8k__WCACHE.get((id(w), tag))
    if ent is None:
        return None
    key, val = ent
    if key == (fp8k__STEP_EPOCH, w.data_ptr(), w._version):
        return val
    return None

def fp8k__wcache_put(w: torch.Tensor, tag: str, val) -> None:
    fp8k__WCACHE[id(w), tag] = ((fp8k__STEP_EPOCH, w.data_ptr(), w._version), val)

def fp8k_quantised_weight(w: torch.Tensor, scale: float=fp8k_W_SCALE, trans: bool=False):
    """The fp8 (uint8-carried) [K, N] copy of a weight, produced at most once per optimizer step.

    The key is (step epoch, data_ptr, _version) of the PARAMETER `w`.  The epoch alone would be enough if begin_step()
    is never missed; _version alone would be enough if every optimizer write bumps it (which it does for a Parameter and
    for a view of the Muon stack, views sharing their base's counter).  Requiring both means either mechanism failing
    alone still invalidates the entry.  `trans=True` takes an [N, K] parameter (a plain nn.Linear, e.g. the MLP c_proj)
    and quantises its transpose, so the kernel's [K, N] operand costs one 10 MB device transpose per step per layer (a
    transpose-on-load quantiser is the follow-up if that ever shows).  The cache tag carries the scale and the layout.
    Returns None when the quantiser cannot run, which sends the caller down the bf16 path."""
    tag = f'f8@{scale:g}{('T' if trans else '')}'
    cached = fp8k__wcache_get(w, tag)
    if cached is not None:
        return cached
    if not fp8k__usable(w):
        return None
    src = w.detach().t().contiguous() if trans else w.detach()
    K, N = src.shape
    if K % fp8k__P or fp8k_plan_quant(K, N) is None:
        return None
    try:
        wq = fp8k__launch(fp8k__quant_w_kernel)(src, float(scale), fp8k_clip_arg(), fp8k_clip_bound(scale))
    except Exception as err:
        fp8k__disable_kernel('quant_w', err)
        return None
    fp8k__wcache_put(w, tag, wq)
    return wq

def fp8k_fused_fc_relu2_fp8_forward(x2d: torch.Tensor, w: torch.Tensor, save_pre: int=1):
    """(h, pre) for the MLP's c_fc + ReLU^2 with the GEMM in fp8, or the bf16 pair when anything
    does not apply.  Never raises: a kernel failure disables the kernels and falls through.

    [R1 relu2fold a] save_pre=0 skips the [M, N] `pre` store, for a consumer that rebuilds
    2*relu(pre) as 2*sqrt(h).  It has to exist HERE and not only on the bf16 forward: every candidate
    since 16 launches with R1_NKI_GEMM_FP8=1, so THIS is the function that owns the MLP forward on the
    live recipe, and a save_pre that only reached nki_gemm.fused_fc_relu2_forward would have deleted
    exactly nothing while passing every CPU check (off-chip both fall through to the eager pair).
    Default 1 = every existing caller unchanged."""
    if fp8k__usable(x2d, w):
        M, K = x2d.shape
        K2, N = w.shape
        pl = fp8k_plan_fp8(M, K, N, tb=fp8k_TB, group=fp8k_GROUP, nprg=fp8k__nprg()) if K2 == K else None
        if pl is not None and pl['sbuf'] <= fp8k_SBUF_BUDGET:
            wq = fp8k_quantised_weight(w)
            if wq is not None:
                try:
                    return fp8k__launch(fp8k__fc_relu2_fp8_kernel)(x2d, wq, int(fp8k_TB), int(fp8k_GROUP), float(fp8k_A_SCALE), float(fp8k_INV_SCALE), fp8k_clip_arg(), fp8k_clip_bound(fp8k_A_SCALE), int(save_pre))
                except Exception as err:
                    fp8k__disable_kernel('fc_relu2_fp8', err)
    return fp8k__bf16_forward(x2d, w, save_pre)

def fp8k__bf16_forward(x2d: torch.Tensor, w: torch.Tensor, save_pre: int=1):
    """Today's path, unchanged: the bf16 fused kernel when R1_NKI_FC_RELU2 allows it, eager otherwise."""
    if fp8k__bf16 is not None:
        return fp8k__bf16.fused_fc_relu2_forward(x2d, w, save_pre=save_pre)
    pre = torch.matmul(x2d, w)
    return (torch.nn.functional.relu(pre).square(), pre)

def fp8k_eager_backward(ctx, dh):
    """The bf16 backward, verbatim nki_gemm.FcRelu2Fn.backward when the module is present (so the
    gradient path is literally today's code, not a copy of it), and its own eager formula when it
    is not.  Straight-through in the fp8 sense: dX and dW are formed against the UNQUANTISED bf16
    weight and activation, i.e. the gradient is of the bf16 function, not of the fp8 one.  That is
    the standard QAT estimator and it is a real deviation -- see PATCH.md S6."""
    x2d, w, pre = ctx.saved_tensors
    dh = dh.contiguous()
    dpre = dh * (2.0 * torch.nn.functional.relu(pre))
    dx = torch.matmul(dpre, w.t())
    if ctx.inplace:
        ctx.w.grad.addmm_(x2d.t(), dpre)
        return (dx, None, None)
    return (dx, torch.matmul(x2d.t(), dpre), None)
fp8k__BACKWARD = fp8k_eager_backward
if fp8k__bf16 is not None:
    fp8k__BACKWARD = fp8k__bf16.FcRelu2Fn.backward

class fp8k_FcRelu2Fp8Fn(torch.autograd.Function):
    """h = relu(x2d @ W)^2 with the forward GEMM in fp8 (or on e4m3-rounded operands in qdq mode)
    and the backward BIT-FOR-BIT today's: `backward` is the very function object
    nki_gemm.FcRelu2Fn uses, so the saved tuple and the ctx fields must match its contract
    exactly -- (x2d, w, pre) saved, ctx.w the Parameter, ctx.inplace the accumulate flag, and a
    3-tuple returned for the 3 forward arguments."""

    @staticmethod
    def forward(ctx, x2d, w, inplace):
        h, pre = fp8k_fused_fc_relu2_fp8_forward(x2d, w)
        ctx.save_for_backward(x2d, w, pre)
        ctx.w = w
        ctx.inplace = inplace
        return h
    backward = staticmethod(fp8k__BACKWARD)

def fp8k_fused_fc_relu2_fp8(x: torch.Tensor, w: torch.Tensor, inplace: bool=False) -> torch.Tensor:
    """relu(x @ w)^2 for any leading shape (w [in, out]); with inplace=True the c_fc dW goes
    straight into w.grad, exactly as the bf16 path does."""
    K = x.shape[-1]
    x2d = x.reshape(-1, K)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    h = fp8k_FcRelu2Fp8Fn.apply(x2d, w, inplace)
    return h.view(*x.shape[:-1], w.shape[1])

def fp8k_config_string() -> str:
    return f'mode={fp8k_MODE!r} eval={fp8k_EVAL_MODE} a_scale={fp8k_A_SCALE:g} w_scale={fp8k_W_SCALE:g} inv={fp8k_INV_SCALE:.3e} wcache={fp8k_WCACHE} tb={fp8k_TB} group={fp8k_GROUP} aalloc={'direct'} lnc={fp8k__LNC} shard={fp8k__SHARD} tb_b={fp8k_TB_B} group_b={fp8k_GROUP_B} evac_b={fp8k_EVAC_B} clip={fp8k_CLIP} loop_b={fp8k_LOOP_B} conv_b={fp8k_CONV_B} a_scale_b={fp8k_A_SCALE_B:g} w_scale_b={fp8k_W_SCALE_B:g} nki={fp8k__HAS_NKI} disabled={fp8k__KERNEL_DISABLED}' + (f' err={fp8k__LAST_KERNEL_ERROR}' if fp8k__LAST_KERNEL_ERROR else '')

class fp8k__ApiNamespace:
    """Namespace object for the freeze.  train.py uses this module as a NAMESPACE, and
    work/freeze_candidate.py inlines a module's names under a prefix -- which would leave the module
    name itself undefined.  The freeze binds it to this object, as candidate 13 bound
    nki_optim = optk_API.  Attributes are assigned with `API.x = x` and NOT declared in the class
    body: the freeze's token renamer rewrites bare names, so a class-body `x = staticmethod(x)`
    becomes `tag_x = staticmethod(tag_x)` and the attribute silently changes name.  (Found 2026-09-22,
    when the first candidate-16 freeze produced an API object with no `proj_active`.)"""
fp8k_API = fp8k__ApiNamespace()
fp8k_API.begin_step = fp8k_begin_step
fp8k_API.active = fp8k_active
fp8k_API.config_string = fp8k_config_string
fp8k_API.fused_fc_relu2_fp8 = fp8k_fused_fc_relu2_fp8
fp8k_API.USE_NKI_GEMM_FP8 = fp8k_USE_NKI_GEMM_FP8
fp8k_API.MODE = fp8k_MODE
nki_gemm_fp8 = fp8k_API
USE_NKI_GEMM_FP8 = fp8k_USE_NKI_GEMM_FP8

class _FreezeLiveNS:
    """Like _FreezeNS, but it does not snapshot: it forwards to the inlined module's prefixed
    global every time, so a value the module still changes while the run proceeds (a kernel
    marking itself disabled after a failure) is read live rather than as it was at import."""

    def __init__(self, prefix):
        self.__dict__['_freeze_prefix'] = prefix

    def __getattr__(self, name):
        try:
            return globals()[self.__dict__['_freeze_prefix'] + name]
        except KeyError:
            raise AttributeError(name)
_fp8_slice1_ns = _FreezeLiveNS('fp8k_')
'FP8 rows 2-4 of the MLP GEMM lane (c_proj forward, c_fc dX, c_proj dX).  DEFAULT OFF.\n\nSlice 1 (`work/nki_gemm_fp8.py`, `R1_NKI_GEMM_FP8`) put ONE of the MLP\'s six GEMMs in fp8: the c_fc\nforward (row 1).  This module adds the three that need no new matmul structure, because the class-A\nand class-B `nn` shapes are already covered by `nki_gemm_fp8._mm_fp8_kernel` and class B is benched:\n\n    row 2  c_proj forward   h[M,5120]    @ W2^T[5120,1024]    class B    measured 0.878 ms = 1.51x\n    row 3  c_fc    dX       dpre[M,5120] @ W1^T[5120,1024]    class B    same shape as row 2\n    row 4  c_proj  dX       dy[M,1024]   @ W2 [1024,5120]     class A    shape never fp8-benched\n\nRows 5-6 (the two dW GEMMs) contract over TOKENS, which needs a different kernel entirely.  That\nkernel is `work/nki_gemm_fp8_tn.py` (copied BYTE-IDENTICAL out of `work/agents/fp8_classc/`, md5\nc7d9516c72ea5dcd5a0ab51e9973edfa, so its 90 host checks still describe the file that is deployed);\nthis module owns the WIRING and nothing else.\n\n    row 5  c_fc   dW  x[M,1024]^T @ dpre[M,5120]  benched 0.860 ms = 1.43x torch   ON in this build\n    row 6  c_proj dW  dy[M,1024]^T @ h[M,5120]    ON in this build, h at scale 1/128\n\nWHY ROW 5 SHIPPED FIRST, AND WHY ROW 6 IS NO LONGER BARRED [R1 fp8 row 6, 2026-09-25].  `dW = x^T\ndpre` quantises `x`, an RMSNorm output at unit RMS -- row 1\'s operand class, measured +0.0003 -- and\n`dpre`, an activation gradient -- rows 3/4\'s class, measured residual +0.000092 over two seeds\n(K6 seed 91, K11 seed 98, each against its own same-seed base), i.e. free.  Row 6 quantises `h =\nrelu(pre)^2`, row 2\'s operand, and row 2\'s ORIGINAL form measured +0.014334 residual -- which is why\nan earlier vintage of this docstring said row 6 "must never ship".  THAT VERDICT WAS A CLIPPING\nARTEFACT, NOT FP8 PHYSICS (work/agents/fp8_rowscale/, 2026-09-24, then confirmed on-chip): at the\nold H_SCALE=0.125 default the clamp sits at 1920 while the REAL h amax reaches 3,568/4,704/9,536 on\n3 of 10 layers (the recorded "amax 189-326" calibration undersampled 29x), for a product relF up to\n0.443.  One global R1_NKI_GEMM_FP8_HSCALE=0.0078125 (clamp 30720, 3.2x headroom over the worst\nlayer, the module\'s own >= 3x rule) fixes it to ~0.03, and with it the h-quantise measured\nFAVOURABLE quality on silicon: `W92_row2fix` residual -0.001146 vs the clipped arm\'s +0.005583\n(a ~0.0067 swing, exactly as the laptop study predicted), then `G91_row2`/`G92_row2` residuals\n-0.002016/-0.000508 on the live c19k7m recipe.  Row 2 still does not ship -- but for THROUGHPUT\n(its forward quantise work inverted to -1.9% steps on today\'s recipe), a cost row 6 does not share:\nrow 6 rides `nki_gemm_fp8_tn`, the kernel that serves row 5 in production at the IDENTICAL\n[M,1024]^T @ [M,5120] shape and measured +2.34% charged steps (n=2, dw_fired 43,400/0 fallbacks).\n\nRow 6\'s call site is `_ProjFn.backward` -- `dw_matmul(dy2d, x2d, 6, key=ctx.w)`, gated on ROW6,\nwith the incumbent `ctx.w.grad.addmm_(dy2d.t(), x2d)` kept verbatim as the None-fallback, exactly\nthe row-5 pattern in `_fc_backward`.  Note the old orientation objection is RESOLVED by this\nwiring, not waved away: the class-C bench once measured `h^T dy` ([5120,16384] x [16384,1024]) and\nthat number described nothing -- but the shipped call is `dy^T h` = a[M,1024]^T @ b[M,5120], the\nbyte-identical plan class row 5 runs in production (folds=32, sbuf=169KiB at M=32768), so the\nthroughput reference is row 5\'s MEASURED arm, not the stale bench.  Row 6 still needs THREE env\ngates (see the knob table) and its per-arm quality charge on the dW path is UNMEASURED until the\nR6 arms return; the pre-registered kill rule lives in work/queue/queue_row6.txt.\n\nTHE ONE REAL INTEGRATION DIFFERENCE FROM ROWS 1-4, AND IT IS A COST, NOT A DETAIL.  The incumbent at\nrow 5\'s site is NOT a matmul: under R1_GRAD_ACCUM_INPLACE the live path does\n`ctx.w.grad.addmm_(x2d.t(), dpre)`, an ACCUMULATE.  The fp8 kernel returns dW, so the caller must add\nit back (`w.grad.add_(dw)`): one extra ~10 MB read + write per layer per micro-step, ~36 us at the\n585 GB/s this project has MEASURED, ~0.7-1.4 ms/step over 10 blocks.  That is subtracted from the\n+2.0-2.1 % charged-step estimate; the gross bench saving is NOT the number to quote.  Folding the\naccumulate into the kernel\'s output pass would remove most of it and is the obvious follow-up.\nAlso: NO WEIGHT CACHE APPLIES.  Both operands are fresh every micro-step, so there is no\n`begin_step()` analogue here and nothing to amortise -- the third way class C is not symmetric.\n\nWhy this is a separate module and not an edit to `work/nki_gemm_fp8.py`\n----------------------------------------------------------------------\n`work/nki_gemm_fp8.py` is deployed, its md5 is recorded in the class-B gap\'s `identity.log`, and\n`F2_qdqsafe`\'s measured quality number is attached to that exact file.  Editing it would invalidate\nboth.  `work/agents/fp8_classc/nki_gemm_fp8_tn.py` is standalone for the same reason.  This file is\nthe third of the set and follows the same rule: it imports `nki_gemm` and `nki_gemm_fp8` for their\nPUBLIC helpers and for the bf16 fallbacks, and owns its own kernels, its own weight cache and its own\nstep epoch, so the three modules cannot interfere.  The duplicated trace-time helpers\n(`_load_at_hwt`, `_load_w_fp8_chunks`, `_mm_tile_fp8`, ...) are verbatim copies of the proven ones;\nthey are duplicated rather than imported because another module\'s trace-time privates are not an API\nand because the freeze inlines whole modules.\n\nThe formats and scales, and why they differ per row\n---------------------------------------------------\n    row 2   a = h = relu(pre)^2   e4m3 @ s=0.125   w = W2^T   e4m3 @ s=16\n    row 3   a = dpre  (GRADIENT)  e5m2 @ s=auto    w = W1^T   e4m3 @ s=128\n    row 4   a = dy    (GRADIENT)  e5m2 @ s=auto    w = W2     e4m3 @ s=16\n\nThe three e4m3 scales are the largest power of two leaving >= 3x clipping headroom against 240 at\nthe worst layer of the R371M checkpoint, measured on the real tensors (the calibration section of\n`test_rows234_cpu.py`), with a unit-RMS Gaussian standing in for the rmsnorm output that the c_fc\ninput is by construction:\n\n    W1 amax 0.5469-0.6133 over 10 layers -> 0.6133 * 128 =  78.5 of 240   (3.06x)\n    W2 amax 0.5430-3.9844 over 10 layers -> 3.9844 *  16 =  63.8 of 240   (3.76x)\n    h  amax   189.2-326.5 over 10 layers ->  326.5 * .125 = 40.8 of 240   (5.88x)\n\nTwo of those differ from the class-B path\'s shipped `A_SCALE_B = 4` / `W_SCALE_B = 64`, and the\ndifference is not cosmetic.  At s_a = 4 the real h clips 0.148 % of its elements and the c_proj\nforward\'s relative error is 6.10 %; at any s_a <= 1 nothing clips and the error is 2.43 % -- flat\nacross s_a 0.125..1 and s_w 8..64, exactly as e4m3\'s scale-invariance inside the normal range\npredicts.  And W2 has one outlier layer at amax 3.98, so s_w = 64 puts it at 255, past 240.\nUnderflow costs nothing at these scales: h\'s subnormal floor at s = 0.125 is 0.0156 = 0.2 % of h\'s\nRMS, and 2 % of elements land there, all of them next to zero.\n\nThe GRADIENT scale cannot be calibrated from a checkpoint, because no checkpoint contains a\ngradient.  An order-of-magnitude estimate from the logged `gnorm` ~ 0.3 puts the c_proj activation\ngradient near 1e-8 RMS, three decades below e5m2\'s subnormal floor of 1.53e-5 -- i.e. a fixed\nscale of 1 would flush the entire gradient to zero.  The estimate carries two decades of\nuncertainty, which is exactly the situation in which a guessed constant is how a lane dies, so the\ndefault is `auto`: on the FIRST call at each site the module reads one amax off the device (one\n`.item()` per site per process -- train.py already does one per step in its gradient gate), picks\nthe power of two that puts the amax at 2048 of 57344, prints it, and freezes it for the run.  A\nsubmission pins the printed values as literals.  e5m2 rather than a correctly-scaled e4m3 because\ne5m2 tolerates about five decades of drift WITHIN the run against e4m3\'s one and a half, at the\nprice of 5.9 % element error against 2.4 %.\n\ne4m3 on this chip is the LEGACY format: max finite 240, and the cast emits +-inf above it rather\nthan saturating (confirmed at ISA level with `--enable-saturate-infinity` set, class-B gap probe:\nclip=0 scale=16 -> 25 of 25 over-range inputs became inf).  So EVERY cast site in this module clamps\nbefore it casts; `CLIP=none` exists only to reproduce the unclamped behaviour for a probe and is\nnever a shippable setting.\n\ne5m2 (max finite 57344, subnormal floor 2^-16 = 1.53e-5) is the gradient format.  Measured at scale\n1: on a 1e-3-RMS gradient e4m3 flushes 67 % of elements to zero for a 56 % error while e5m2 costs\n5.3 %, because e5m2\'s normals span 2^29 against e4m3\'s 2^14.  At a CORRECTLY chosen scale e4m3 is\nthe sharper format (2.4 % against 5.9 %), but its usable window for a Gaussian is about 1.5 decades\nagainst e5m2\'s five, and a scale frozen at step 1 has to survive 2,500 steps of drift.  `nc_matmul`\ntakes a MIXED e4m3 x e5m2 `double_row` pair in both operand orders (frontend compile at the real\nshapes, `work/agents/fp8_classc/host_check_classc.py` Q1 PASS); `R1_NKI_GEMM_FP8_GFMT=e4m3` collapses\nthe gradient side to e4m3 in one env var if a chip disagrees.\n\nWhat the clamp costs: one VectorE instruction for row 2, two for the gradient rows\n-----------------------------------------------------------------------------------\n`nisa.tensor_scalar` has two op slots and converts on write to the destination dtype, so:\n\n    non-negative operand, any scale   (multiply s, minimum +max)          1 instruction   `aq=pos`\n    signed operand,       scale == 1  (minimum +max, maximum -max)        1 instruction   `aq=sign1`\n    signed operand,       scale != 1  clamp on VectorE, then ScalarE cast 2 instructions  `aq=vec`\n\nRow 2\'s operand is relu(pre)^2 >= 0, so it gets `pos` and one instruction.  Rows 3 and 4 are signed\nand their `auto` scale is never exactly 1, so they take `vec` and pay a second pass -- the +0.057 ms\nper class-B call that the benched `clip=vec` row shows, ~2.7 ms/step in total, priced in PATCH.md S5.\n`sign1` is there because pinning R1_NKI_GEMM_FP8_GSCALE=1 buys that back, and a probe may show the\ngradient magnitude allows it.\n\nKnobs (env, read once at import)\n\nTHE GRADIENT SCALE IS NOT OPTIONAL FOR ROW 5, AND THE BENCHED CONFIGURATION CANNOT BE THE SHIPPED\nONE.  The class-C bench\'s winning cell (`qeng=vec,shard=n`, 0.860 ms) ran at its module defaults\n`aq=bq=unit`, i.e. BOTH operands at scale 1 -- and `unit` is the one-instruction signed clamped cast\nprecisely because scale 1 leaves tensor_scalar\'s second op slot free.  At scale 1 the real `dpre`\nis annihilated: its amax is ~1.9e-6 (the number candidate 16\'s 2^30 pin is derived from) and e5m2\'s\nsmallest subnormal is 2^-16 = 1.53e-5, so every element would flush to zero, silently, with the loss\nstill descending.  (The bench did not see this because NOTES.md §4 item 7 says its gradient operand\nwas a synthetic `--g-rms 1e-3`.)  So row 5 ships with a SCALED signed operand, which is the class-C\nkernel\'s `qmode 2 "split"`: a VectorE two-op clamp of the staged bf16 tile, then a ScalarE\n`nisa.activation(scale)` into the fp8 tile.  Two instructions -- but the VectorE pass count over each\noperand is UNCHANGED from `unit` (one either way), and what is added is a ScalarE pass, on the engine\nNOTES.md §2b prices at 17 us of a 544 us TensorE floor.  So the expected cost is near zero and the\n0.860 ms should carry over, but THIS IS AN ARGUMENT, NOT A MEASUREMENT: `qmode 2` is not the cell that\nwas benched, and only an arm can say whether the ScalarE pass hides.  (What the gap DID kill is\n`qeng=split`, the attempt to put the one-instruction two-op `tensor_scalar` itself on the Activation\nengine -- NCC_IBIR444, a hardware constraint.  `qmode 2`\'s ScalarE instruction is `nisa.activation`,\na single-op form, which is the shipped `clip=vec` order this module\'s own rows 3/4 already use.)\n\nFallback chain (exact, and disable-once)\n    fp8 kernel -> the op train.py issues today (nki_gemm.nki_matmul for rows 2/3, torch.matmul for\n    row 4) -> torch.  Anything that does not fit -- no NKI, not a neuron device, not bf16, a shape\n    the plan refuses, an SBUF plan over budget, or a kernel that raises (printed once, kernels then\n    disabled for the process) -- takes the next link.  On CPU the chain always reaches the last link,\n    so a CPU smoke with the switch ON must reproduce the switch-OFF value exactly.\n\n    ROW 5\'S FALLBACK IS NOT IN THAT CHAIN AND MUST NOT BE PUT IN IT.  Its incumbent is neither\n    `torch.matmul` nor the bf16 NKI kernel unconditionally -- it is whichever of THREE ops the site\n    would have issued, and which one depends on `ctx.inplace`:\n\n        inplace   (the live candidate path)   ctx.w.grad.addmm_(x2d.t(), dpre)\n        otherwise, with nki_gemm present      _bf16.nki_matmul(x2d, dpre, trans_a=True)\n        otherwise                             torch.matmul(x2d.t(), dpre)\n\n    `_eager(a2d, w, trans_b, row)` cannot express any of those -- it computes `a2d @ w`, and for row 5\n    `w` is not a weight -- so `_eager` REFUSES rows 5/6 loudly instead of guessing, and `dw_matmul`\n    returns None rather than a fallback product.  None means "I did nothing": the call site then runs\n    its own unchanged incumbent line.  That is why the three lines above are still present verbatim in\n    `_fc_backward`, why `_ProjFn.backward` keeps its `addmm_` / `torch.matmul` dW lines verbatim as\n    row 6\'s None-fallback, and why the off path is bitwise.\n'
import math
import os
import torch
fp8nnk__P = 128
fp8nnk__CH = 512
fp8nnk__KS = 256
fp8nnk__J = fp8nnk__KS // fp8nnk__P
fp8nnk__E4M3_MAX = 240.0
fp8nnk__E5M2_MAX = 57344.0
fp8nnk__env = os.environ.get
fp8nnk_MODE = '1'
fp8nnk__ROWS_RAW = '2,3,4'
fp8nnk_ROWS = tuple(sorted({int(t) for t in fp8nnk__ROWS_RAW.replace(' ', '').split(',') if t in ('2', '3', '4')}))
fp8nnk_USE_NKI_GEMM_FP8_NN = len(fp8nnk_ROWS) > 0
fp8nnk_QDQ_ONLY = fp8nnk_MODE == 'qdq'
fp8nnk_EVAL_MODE = 'train'
fp8nnk_DW_MODE = '1'
fp8nnk__DW_ROWS_RAW = '5,6'
fp8nnk_DW_ROWS = tuple(sorted({int(t) for t in fp8nnk__DW_ROWS_RAW.replace(' ', '').split(',') if t in ('5', '6')}))
fp8nnk_H_SCALE = 0.0078125
fp8nnk_USE_NKI_GEMM_FP8_DW = len(fp8nnk_DW_ROWS) > 0
fp8nnk_DW_EVAL_MODE = 'train'
fp8nnk__GS_RAW = '1073741824'
fp8nnk_G_SCALE = float(fp8nnk__GS_RAW)
fp8nnk_G_TARGET = 2048
fp8nnk__GMIN = 1e-06
fp8nnk_W_SCALE_FC = 128
fp8nnk_W_SCALE_PROJ = 16
fp8nnk_A_SCALE_DW = 16
fp8nnk__FMT_NAMES = {'e4m3': 0, 'e5m2': 1}
fp8nnk_GFMT = fp8nnk__FMT_NAMES.get('e5m2', 1)
fp8nnk_WCACHE = 'step'
fp8nnk_WQT = 'copy'
fp8nnk_TB_A = 1024
fp8nnk_TB_B = 512
fp8nnk_GROUP = 2
fp8nnk__EVAC_NAMES = {'alt': 0, 'scalar': 1, 'vector': 2}
fp8nnk_EVAC = fp8nnk__EVAC_NAMES.get('vector', 1)
fp8nnk_LOOP = 0
fp8nnk__LNC = int(fp8nnk__env('NEURON_LOGICAL_NC_CONFIG', '1'))
fp8nnk__SHARD = True
fp8nnk_SBUF_BUDGET = 200 * 1024
fp8nnk__RELU2_WHY = []
fp8nnk_RELU2_FROM_H = not fp8nnk__RELU2_WHY
fp8nnk_RELU2_FOLD_DX = not fp8nnk__RELU2_WHY
if fp8nnk__RELU2_WHY:
    print('[nki_gemm_fp8_nn] IGNORING R1_RELU2_FROM_H/R1_RELU2_FOLD_DX -- it would be a silent no-op: ' + '; '.join(fp8nnk__RELU2_WHY), flush=True)
fp8nnk_AQ_VEC, fp8nnk_AQ_POS, fp8nnk_AQ_SIGN1 = (0, 1, 2)
fp8nnk__KERNEL_DISABLED = False
fp8nnk__LAST_KERNEL_ERROR = None
fp8nnk__STEP_EPOCH = 0
fp8nnk__WCACHE = {}
fp8nnk__GCACHE = {}

def fp8nnk_fmt_max(fmt: int) -> float:
    return fp8nnk__E5M2_MAX if fmt == 1 else fp8nnk__E4M3_MAX

def fp8nnk_clip_bound(scale: float, fmt: int=0) -> float:
    """The clamp bound in the UNSCALED bf16 domain: the largest bf16 value b with b * scale <= the
    format's max finite value.  The clamped value lives in a bf16 tile, so for a non-power-of-two
    scale max/scale would round UP in bf16 and let an over-range value through; power-of-two scales
    are exact.  (nki_gemm_fp8.clip_bound, generalised over the format.)"""
    hi = fp8nnk_fmt_max(fmt)
    b = torch.tensor(hi / scale, dtype=torch.float32).to(torch.bfloat16)
    for _ in range(4):
        if float(b) * scale <= hi:
            break
        b = torch.nextafter(b, torch.tensor(0.0, dtype=torch.bfloat16))
    return float(b)

def fp8nnk_auto_scale(amax: float, fmt: int) -> float:
    """The power of two that puts `amax` at G_TARGET of the format's maximum.  Deterministic, and
    biased LOW (floor), so the headroom is at least the nominal 28x: a gradient grows early and
    shrinks late, and e5m2 has about five decades of slack underneath."""
    if not amax > 0.0 or not math.isfinite(amax):
        return 1.0
    s = 2.0 ** math.floor(math.log2(min(fp8nnk_G_TARGET, fp8nnk_fmt_max(fmt)) / amax))
    return min(max(s, 2.0 ** (-16)), 2.0 ** 40)

def fp8nnk_conv_form(signed: bool, scale: float) -> int:
    """Which of the three conversion forms a (signedness, scale) pair gets.  One instruction unless
    the operand is signed AND the scale is not exactly 1."""
    if not signed:
        return fp8nnk_AQ_POS
    if scale == 1.0:
        return fp8nnk_AQ_SIGN1
    return fp8nnk_AQ_VEC
fp8nnk_ROW_SPEC = {2: {'afmt': 0, 'ascale': fp8nnk_H_SCALE, 'signed': False, 'wscale': fp8nnk_W_SCALE_PROJ, 'trans_b': True, 'what': 'c_proj fwd  h @ W2^T'}, 3: {'afmt': fp8nnk_GFMT, 'ascale': fp8nnk_G_SCALE, 'signed': True, 'wscale': fp8nnk_W_SCALE_FC, 'trans_b': True, 'what': 'c_fc   dX   dpre @ W1^T'}, 4: {'afmt': fp8nnk_GFMT, 'ascale': fp8nnk_G_SCALE, 'signed': True, 'wscale': fp8nnk_W_SCALE_PROJ, 'trans_b': False, 'what': 'c_proj dX   dy @ W2'}, 5: {'classc': True, 'gslot': 'b', 'afmt': 0, 'ascale': fp8nnk_A_SCALE_DW, 'asigned': True, 'bfmt': fp8nnk_GFMT, 'bscale': fp8nnk_G_SCALE, 'bsigned': True, 'what': 'c_fc   dW   x^T @ dpre'}, 6: {'classc': True, 'gslot': 'a', 'afmt': fp8nnk_GFMT, 'ascale': fp8nnk_G_SCALE, 'asigned': True, 'bfmt': 0, 'bscale': fp8nnk_H_SCALE, 'bsigned': False, 'what': 'c_proj dW   dy^T @ h'}}
fp8nnk_ROW2, fp8nnk_ROW3, fp8nnk_ROW4 = (2 in fp8nnk_ROWS, 3 in fp8nnk_ROWS, 4 in fp8nnk_ROWS)
fp8nnk__ROW2_CAP = fp8nnk_clip_bound(fp8nnk_ROW_SPEC[2]['ascale'], fp8nnk_ROW_SPEC[2]['afmt']) if fp8nnk_ROW2 else None
fp8nnk__ROW2_CAP_N = 0

def fp8nnk_row2_cap(h: torch.Tensor) -> torch.Tensor:
    """h capped at row 2's training-time saturation bound when R1_ROW2_EVAL_CAP applies, else h itself."""
    global fp8nnk__ROW2_CAP_N
    if fp8nnk__ROW2_CAP is None:
        return h
    fp8nnk__ROW2_CAP_N += 1
    return h.clamp(max=fp8nnk__ROW2_CAP)
fp8nnk_ROW5 = 5 in fp8nnk_DW_ROWS

def fp8nnk_pick_tb(K: int) -> int:
    return fp8nnk_TB_A if K <= 2048 else fp8nnk_TB_B

def fp8nnk_plan_nn_fp8(M: int, K: int, N: int, tb: int=None, group: int=fp8nnk_GROUP, nprg: int=2, fold: int=0):
    """SBUF bytes per partition, or None if the shape is not tileable.  Same accounting as
    nki_gemm_fp8.plan_mm_fp8 -- the layout is the same:

        weight (fp8, resident)                NK_d * J * N       = K/256 * 2 * N
        a^T block bf16 (hwt), double-buffered 2 * NK_b * tb * 2  = 2 * K/128 * tb * 2
        a^T block fp8,        double-buffered 2 * NK_d * J * tb  = 2 * K/256 * 2 * tb
        out row staging,      double-buffered 2 * N * 2

    class B  [.,5120]x[5120,1024]  tb=512 : 40 + 80 + 40 + 4  = 164 KiB
    class A  [.,1024]x[1024,5120]  tb=1024: 40 + 32 + 16 + 20 = 108 KiB

    [R1 relu2fold b] fold=1 budgets _mm_fp8_nn_relu2_kernel's extra `hst` row staging (the h tile of
    the fused epilogue), double-buffered like `ost`: + 2 * N * 2.  Row 4's live class A grows
    108 -> 128 KiB, still 72 KiB inside SBUF_BUDGET's 200.  fold=0 is byte-for-byte the old plan.
    """
    if tb is None:
        tb = fp8nnk_pick_tb(K)
    if M % fp8nnk__P or K % fp8nnk__KS or N % fp8nnk__CH or (M < 2 * fp8nnk__P) or (K < fp8nnk__KS) or (N < fp8nnk__CH):
        return None
    if tb % fp8nnk__P or tb < fp8nnk__P or tb % 16:
        return None
    if group < 1 or group > 8:
        return None
    nkd = K // fp8nnk__KS
    nkb = K // fp8nnk__P
    w_bytes = nkd * fp8nnk__J * N
    at_bytes = nkb * tb * 2 + nkd * fp8nnk__J * tb
    ost_bytes = 2 * N * 2
    hst_bytes = 2 * N * 2 if fold else 0
    return {'nkd': nkd, 'nkb': nkb, 'nc': N // fp8nnk__CH, 'tb': tb, 'sbuf': w_bytes + 2 * at_bytes + ost_bytes + hst_bytes}

def fp8nnk_plan_quant(K: int, N: int):
    """SBUF bytes per partition of the weight quantiser / qdq kernel, or None."""
    if K % fp8nnk__P or N < 1:
        return None
    return {'sbuf': N * 2 + N + N * 2}
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    fp8nnk__PM = nisa.matmul_perf_mode.double_row

    def fp8nnk__f8(fmt):
        """Trace-time dtype selector.  `fmt` is a Python int baked in at trace time, exactly as
        `xt_mode` / `clip` / `conv` are in nki_gemm_fp8."""
        if fmt == 1:
            return nl.float8_e5m2
        return nl.float8_e4m3

    def fp8nnk__shard():
        if nl.program_ndim() > 0:
            return (nl.program_id(0), nl.num_programs(0))
        return (0, 1)

    def fp8nnk__span(n, pid, nprg):
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
        return (i0, i0 + cnt)

    def fp8nnk__ceil_div(a, b):
        return (a + b - 1) // b

    def fp8nnk__ptiles(n, shape, dtype, buf):
        """A tuple of n (<= 8) fresh tiles; an if-chain, as nki_gemm does (no comprehensions)."""
        t0 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 1:
            return (t0,)
        t1 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 2:
            return (t0, t1)
        t2 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 3:
            return (t0, t1, t2)
        t3 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 4:
            return (t0, t1, t2, t3)
        t4 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 5:
            return (t0, t1, t2, t3, t4)
        t5 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 6:
            return (t0, t1, t2, t3, t4, t5)
        t6 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 7:
            return (t0, t1, t2, t3, t4, t5, t6)
        t7 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        return (t0, t1, t2, t3, t4, t5, t6, t7)

    def fp8nnk__load_at_hwt(a2, K, NKB, row0, ntok, dt):
        """a^T block [P, NKB, ntok] in bf16 by the hardware-DGE strip transpose: one
        dma_transpose(dge_mode=hwdge) per 16-token strip covering all NKB k-slabs.  hwdge requires
        src.shape[0] == 16 and a 2-byte dtype, which is why the transpose happens in bf16 BEFORE the
        fp8 conversion.  The per-slab form lowers to runtime-generated descriptors cached per input
        address and cost 3.3-4.4 ms in situ against 1.2 ms on a reused buffer (the rung-1 collapse)."""
        at = nl.ndarray((fp8nnk__P, NKB, ntok), dtype=dt, buffer=nl.sbuf)
        for s in range(ntok // 16):
            nisa.dma_transpose(dst=at.ap([[NKB * ntok, fp8nnk__P], [ntok, NKB], [1, 1], [1, 16]], offset=s * 16), src=a2.ap([[K, 16], [fp8nnk__P, NKB], [1, 1], [1, fp8nnk__P]], offset=(row0 + s * 16) * K), dge_mode=nisa.dge_mode.hwdge)
        return at

    def fp8nnk__load_at_pre(a2t, K, NKB, row0, ntok, dt):
        """a^T block by plain DMA from an ALREADY TRANSPOSED a^T [K, M].  Bench diagnostic only: it
        isolates the cost of the hwt strip transpose."""
        at = nl.ndarray((fp8nnk__P, NKB, ntok), dtype=dt, buffer=nl.sbuf)
        for kk in range(NKB):
            nisa.dma_copy(dst=at[:, kk, :], src=a2t[kk * fp8nnk__P:(kk + 1) * fp8nnk__P, row0:row0 + ntok])
        return at

    def fp8nnk__quantise_block(at, NKD, ntok, scale, aq, bound, hi, fmt):
        """bf16 a^T block [P, NKB, ntok] -> fp8 [P, NKD, J, ntok].

        Index map: the bf16 slab kk holds contraction row kk*128 + p; the fp8 element (p, kd, j, t)
        must hold contraction row kd*256 + j*128 + p, so kk = kd*J + j.  Both slices are [P, ntok]
        with the same free stride, so this is a pure reindex -- no gather, no transpose.

        aq 1 (non-negative operand): one VectorE tensor_scalar does scale, clip at +hi and the cast.
        aq 2 (signed operand at scale 1): one VectorE tensor_scalar does clip at +-hi and the cast.
        aq 0: a VectorE clamp of the bf16 slab to +-bound, then the ScalarE scale+cast (two
              instructions, the only form that supports a signed operand at a scale != 1).  With
              CLIP_ON false the clamp is skipped, which reproduces the unclamped cast for a probe."""
        a_f8 = nl.ndarray((fp8nnk__P, NKD, fp8nnk__J, ntok), dtype=fp8nnk__f8(fmt), buffer=nl.sbuf)
        for kd in range(NKD):
            for j in range(fp8nnk__J):
                kk = kd * fp8nnk__J + j
                if aq == 1:
                    nisa.tensor_scalar(dst=a_f8[:, kd, j, :], data=at[:, kk, :], op0=nl.multiply, operand0=scale, op1=nl.minimum, operand1=hi, engine=nisa.engine.vector)
                elif aq == 2:
                    nisa.tensor_scalar(dst=a_f8[:, kd, j, :], data=at[:, kk, :], op0=nl.minimum, operand0=hi, op1=nl.maximum, operand1=-hi, engine=nisa.engine.vector)
                else:
                    nisa.tensor_scalar(dst=at[:, kk, :], data=at[:, kk, :], op0=nl.minimum, operand0=bound, op1=nl.maximum, operand1=-bound, engine=nisa.engine.vector)
                    nisa.activation(dst=a_f8[:, kd, j, :], op=nl.copy, data=at[:, kk, :], scale=scale)
        return a_f8

    def fp8nnk__load_w_fp8_chunks(w_u8, K, N, NKD, NC):
        """Resident fp8 weight laid out per (k double-slab, output chunk): [P, NKD*NC, J, 512] with
        the pair index kd*NC + c, so every moving tile w[:, kd*NC + c] is a CONTIGUOUS [P, 2, 512].
        Element (p, kd*NC + c, j, x) holds w[kd*256 + j*128 + p, c*512 + x].  uint8 carriage, bitcast
        to fp8 in SBUF: NRT has no fp8 dtype, so fp8 bytes travel inside uint8 and nothing is cast.
        The weight is ALWAYS e4m3 in this lane -- only the activation side varies."""
        w_sb = nl.ndarray((fp8nnk__P, NKD * NC, fp8nnk__J, fp8nnk__CH), dtype=nl.uint8, buffer=nl.sbuf)
        for kd in range(NKD):
            for c in range(NC):
                for j in range(fp8nnk__J):
                    r0 = kd * fp8nnk__KS + j * fp8nnk__P
                    nisa.dma_copy(dst=w_sb[:, kd * NC + c, j, :], src=w_u8[r0:r0 + fp8nnk__P, c * fp8nnk__CH:(c + 1) * fp8nnk__CH])
        return w_sb.view(nl.float8_e4m3)

    def fp8nnk__evac_scaled(dst, src, k, evac, inv_scale):
        """PSUM fp32 -> bf16 dst with the 1/(s_a s_w) descale folded into the evacuation."""
        if evac == 1:
            nisa.activation(dst=dst, op=nl.copy, data=src, scale=inv_scale)
        elif evac == 2:
            nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=inv_scale, engine=nisa.engine.vector)
        elif k % 2 == 0:
            nisa.activation(dst=dst, op=nl.copy, data=src, scale=inv_scale)
        else:
            nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=inv_scale, engine=nisa.engine.vector)

    def fp8nnk__mm_tile(a_f8, w_f8, out, ti, bt0, N, NKD, NC, npass, group, evac, inv_scale, dt, f32):
        """One 128-token tile: passes of `group` output chunks sharing one stationary per k
        double-slab; the NKD double-pumped matmuls of a chunk accumulate in ONE PSUM bank
        (accumulate=(kd > 0)); evacuate with the descale; one [128, N] row store.  Verbatim
        nki_gemm_fp8._mm_tile_fp8 -- the stationary is the ACTIVATION (possibly e5m2) and the moving
        is the WEIGHT (always e4m3), the mixed pair the class-C host check compiled in both orders."""
        row0 = (bt0 + ti) * fp8nnk__P
        ost = nl.ndarray((fp8nnk__P, N), dtype=dt, buffer=nl.sbuf)
        for p in range(npass):
            c0 = p * group
            g = NC - c0
            if g > group:
                g = group
            pss = fp8nnk__ptiles(g, (fp8nnk__P, fp8nnk__CH), f32, nl.psum)
            for kd in range(NKD):
                st = a_f8[:, kd, :, ti * fp8nnk__P:(ti + 1) * fp8nnk__P]
                for j in range(g):
                    nisa.nc_matmul(dst=pss[j], stationary=st, moving=w_f8[:, kd * NC + c0 + j, :, :], accumulate=kd > 0, perf_mode=fp8nnk__PM)
            for j in range(g):
                c = c0 + j
                fp8nnk__evac_scaled(ost[:, c * fp8nnk__CH:(c + 1) * fp8nnk__CH], pss[j], c, evac, inv_scale)
        nisa.dma_copy(dst=out[row0:row0 + fp8nnk__P, :], src=ost)

    @nki.jit
    def fp8nnk__mm_fp8_nn_kernel(a2, w_u8, xt_mode, tb, group, evac, aq, loop, afmt, a_scale, a_bound, a_hi, inv_scale):
        """out[M, N] = bf16((a2 @ w) / (s_a s_w)); activation fp8 `afmt`, weight fp8 e4m3,
        contraction double-pumped (256 per nc_matmul).

        a2       bf16 activation, token-major [M, K] (xt_mode 3: the hardware-DGE strip transpose)
                 or an already transposed a^T [K, M] (xt_mode 2, bench diagnostic only).
        w_u8     the fp8 e4m3 weight [K, N] as uint8, quantised once per optimizer step.
        aq       0 clamp-then-cast (2 instructions) | 1 fused non-negative | 2 fused signed at s=1.
        a_bound  max_fmt / s_a, the clamp bound in the UNSCALED bf16 domain (aq 0).
        a_hi     the clamp bound in the SCALED domain: s_a * a_bound for aq 1, max_fmt for aq 2."""
        if xt_mode == 2:
            K = a2.shape[0]
            M = a2.shape[1]
        else:
            M = a2.shape[0]
            K = a2.shape[1]
        N = w_u8.shape[1]
        dt = a2.dtype
        f32 = nl.float32
        out = nl.ndarray((M, N), dtype=dt, buffer=nl.shared_hbm)
        NKD = K // fp8nnk__KS
        NKB = K // fp8nnk__P
        NC = N // fp8nnk__CH
        npass = fp8nnk__ceil_div(NC, group)
        pid, nprg = fp8nnk__shard()
        t0, t1 = fp8nnk__span(M // fp8nnk__P, pid, nprg)
        w_f8 = fp8nnk__load_w_fp8_chunks(w_u8, K, N, NKD, NC)
        ntb = tb // fp8nnk__P
        nblk = fp8nnk__ceil_div(t1 - t0, ntb)
        for b in range(nblk):
            bt0 = t0 + b * ntb
            bt1 = bt0 + ntb
            if bt1 > t1:
                bt1 = t1
            nt = bt1 - bt0
            if xt_mode == 2:
                at = fp8nnk__load_at_pre(a2, K, NKB, bt0 * fp8nnk__P, nt * fp8nnk__P, dt)
            else:
                at = fp8nnk__load_at_hwt(a2, K, NKB, bt0 * fp8nnk__P, nt * fp8nnk__P, dt)
            a_f8 = fp8nnk__quantise_block(at, NKD, nt * fp8nnk__P, a_scale, aq, a_bound, a_hi, afmt)
            if loop == 1:
                for ti in nl.affine_range(nt):
                    fp8nnk__mm_tile(a_f8, w_f8, out, ti, bt0, N, NKD, NC, npass, group, evac, inv_scale, dt, f32)
            else:
                for ti in range(nt):
                    fp8nnk__mm_tile(a_f8, w_f8, out, ti, bt0, N, NKD, NC, npass, group, evac, inv_scale, dt, f32)
        return out

    def fp8nnk__mm_tile_relu2(a_f8, w_f8, h2, out, ti, bt0, N, NKD, NC, npass, group, evac, inv_scale, dt, f32):
        """[R1 relu2fold b] _mm_tile with the ReLU^2 backward fused between the PSUM evacuation and
        the row store.  After the pass loop `ost` holds this tile's dh = (dy @ W2) / (s_a s_w) in
        bf16 -- the very values the standalone kernel would have round-tripped through HBM -- and the
        epilogue turns them into dpre = 2 sqrt(h) dh IN SBUF.  What stops existing: the [128, N] dh
        store here, and the standalone kernel's dh read, h/pre read and dpre write.  What is added:
        one dma_copy of the h tile (1.31 MB at N = 5120, into the DMA slack the dh store vacated).

        Instruction for instruction the epilogue is nki_relu2_opt._bwd_h_compute applied per output
        chunk: ScalarE `activation(sqrt, scale=4.0)` -- sqrt(4h) == 2 sqrt(h) EXACTLY, 4 being a
        power of two -- in place over the bf16 h slab (the act-mode in-place precedent), then ONE
        VectorE multiply with one rounding into the bf16 `ost` chunk.  Same bf16 dh values, same
        bf16 r, same single rounding on the product => bitwise the standalone path (the exhaustive
        proof is RELU2FOLD_HANDOVER.md section 6).  The multiply's dst aliases data2, which is the
        alias order act mode runs on every candidate (`tensor_tensor(dst=g, data1=r, data2=g)`);
        the handover snippet wrote dst==data1, and multiply commutes, so the value is identical --
        this form is used because it is the one already proven on silicon.

        sqrt and NEVER rsqrt: the forward writes h = (pre max 0) * pre, so h is a signed zero across
        the whole non-positive half of pre; IEEE sqrt(-0.0) = -0.0 keeps the gradient a signed zero
        where a reciprocal form would produce 0 * -inf = NaN."""
        row0 = (bt0 + ti) * fp8nnk__P
        ost = nl.ndarray((fp8nnk__P, N), dtype=dt, buffer=nl.sbuf)
        for p in range(npass):
            c0 = p * group
            g = NC - c0
            if g > group:
                g = group
            pss = fp8nnk__ptiles(g, (fp8nnk__P, fp8nnk__CH), f32, nl.psum)
            for kd in range(NKD):
                st = a_f8[:, kd, :, ti * fp8nnk__P:(ti + 1) * fp8nnk__P]
                for j in range(g):
                    nisa.nc_matmul(dst=pss[j], stationary=st, moving=w_f8[:, kd * NC + c0 + j, :, :], accumulate=kd > 0, perf_mode=fp8nnk__PM)
            for j in range(g):
                c = c0 + j
                fp8nnk__evac_scaled(ost[:, c * fp8nnk__CH:(c + 1) * fp8nnk__CH], pss[j], c, evac, inv_scale)
        hst = nl.ndarray((fp8nnk__P, N), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=hst, src=h2[row0:row0 + fp8nnk__P, :])
        for c in range(NC):
            hs = hst[:, c * fp8nnk__CH:(c + 1) * fp8nnk__CH]
            o_ = ost[:, c * fp8nnk__CH:(c + 1) * fp8nnk__CH]
            nisa.activation(dst=hs, op=nl.sqrt, data=hs, scale=4.0)
            nisa.tensor_tensor(dst=o_, data1=hs, data2=o_, op=nl.multiply)
        nisa.dma_copy(dst=out[row0:row0 + fp8nnk__P, :], src=ost)

    @nki.jit
    def fp8nnk__mm_fp8_nn_relu2_kernel(a2, w_u8, h2, xt_mode, tb, group, evac, aq, loop, afmt, a_scale, a_bound, a_hi, inv_scale):
        """[R1 relu2fold b] _mm_fp8_nn_kernel with ONE extra HBM input `h2` [M, N] and the ReLU^2
        backward fused as the output epilogue: out = 2 sqrt(h2) * ((a2 @ w) / (s_a s_w)).  In
        practice row 4 only (a2 = dy [M, 1024], w_u8 = W2 [1024, 5120] e4m3, h2 = h [M, 5120]).
        Everything outside _mm_tile_relu2's epilogue is _mm_fp8_nn_kernel's text verbatim; under
        LNC=2 sharding each program touches only its own [t0, t1) token tiles, and the epilogue's
        h2 rows are indexed by the same row0 the store uses, so the shard split is unchanged."""
        if xt_mode == 2:
            K = a2.shape[0]
            M = a2.shape[1]
        else:
            M = a2.shape[0]
            K = a2.shape[1]
        N = w_u8.shape[1]
        dt = a2.dtype
        f32 = nl.float32
        out = nl.ndarray((M, N), dtype=dt, buffer=nl.shared_hbm)
        NKD = K // fp8nnk__KS
        NKB = K // fp8nnk__P
        NC = N // fp8nnk__CH
        npass = fp8nnk__ceil_div(NC, group)
        pid, nprg = fp8nnk__shard()
        t0, t1 = fp8nnk__span(M // fp8nnk__P, pid, nprg)
        w_f8 = fp8nnk__load_w_fp8_chunks(w_u8, K, N, NKD, NC)
        ntb = tb // fp8nnk__P
        nblk = fp8nnk__ceil_div(t1 - t0, ntb)
        for b in range(nblk):
            bt0 = t0 + b * ntb
            bt1 = bt0 + ntb
            if bt1 > t1:
                bt1 = t1
            nt = bt1 - bt0
            if xt_mode == 2:
                at = fp8nnk__load_at_pre(a2, K, NKB, bt0 * fp8nnk__P, nt * fp8nnk__P, dt)
            else:
                at = fp8nnk__load_at_hwt(a2, K, NKB, bt0 * fp8nnk__P, nt * fp8nnk__P, dt)
            a_f8 = fp8nnk__quantise_block(at, NKD, nt * fp8nnk__P, a_scale, aq, a_bound, a_hi, afmt)
            if loop == 1:
                for ti in nl.affine_range(nt):
                    fp8nnk__mm_tile_relu2(a_f8, w_f8, h2, out, ti, bt0, N, NKD, NC, npass, group, evac, inv_scale, dt, f32)
            else:
                for ti in range(nt):
                    fp8nnk__mm_tile_relu2(a_f8, w_f8, h2, out, ti, bt0, N, NKD, NC, npass, group, evac, inv_scale, dt, f32)
        return out

    @nki.jit
    def fp8nnk__quant_w_nn_kernel(w, scale, clip, bound):
        """bf16 [K, N] -> fp8 e4m3 bytes in a uint8 [K, N] tensor, scaled in the same ScalarE
        instruction.  One 128-row slab at a time.  clip == 1 clamps the slab to +-bound first, one
        VectorE instruction per slab -- it runs once per optimizer step, so it is free.  A copy of
        nki_gemm_fp8._quant_w_kernel; the CPU test checks its emulation against that module's
        `quantise_bits_host`, byte for byte."""
        K = w.shape[0]
        N = w.shape[1]
        out = nl.ndarray((K, N), dtype=nl.uint8, buffer=nl.shared_hbm)
        pid, nprg = fp8nnk__shard()
        r0, r1 = fp8nnk__span(K // fp8nnk__P, pid, nprg)
        for i in range(r1 - r0):
            k0 = (r0 + i) * fp8nnk__P
            stage = nl.ndarray((fp8nnk__P, N), dtype=w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=stage, src=w[k0:k0 + fp8nnk__P, :])
            if clip == 1:
                nisa.tensor_scalar(dst=stage, data=stage, op0=nl.minimum, operand0=bound, op1=nl.maximum, operand1=-bound, engine=nisa.engine.vector)
            q_u8 = nl.ndarray((fp8nnk__P, N), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.activation(dst=q_u8.view(nl.float8_e4m3), op=nl.copy, data=stage, scale=scale)
            nisa.dma_copy(dst=out[k0:k0 + fp8nnk__P, :], src=q_u8)
        return out

    @nki.jit
    def fp8nnk__quant_w_t_kernel(w, scale, clip, bound):
        """bf16 [R, C] -> the fp8 e4m3 bytes of its TRANSPOSE, uint8 [C, R], in ONE launch.

        The two transposed weight copies (W1^T for row 3, W2^T for row 2) otherwise cost a
        `w.t().contiguous()` in bf16 first: 10.5 MB read and 10.5 MB written per weight per step
        that this kernel does not spend, plus a second launch, plus the risk that a torch transpose
        of a 10 MB tensor lowers badly on this backend (the rung-1 class of defect).

        Nothing here is new: the load is `_load_at_hwt` verbatim, with the weight's leading axis
        playing the part of the token axis, and the store is one dma_copy per k-slab.  Default OFF
        (`R1_NKI_GEMM_FP8_NN_WQT=copy`) until it is benched -- `copy` is the proven path.

        SBUF at [1024, 5120] sharded two ways: bf16 40 * 512 * 2 = 40 KiB + fp8 40 * 512 = 20 KiB."""
        R = w.shape[0]
        C = w.shape[1]
        out = nl.ndarray((C, R), dtype=nl.uint8, buffer=nl.shared_hbm)
        NKB = C // fp8nnk__P
        pid, nprg = fp8nnk__shard()
        r0, r1 = fp8nnk__span(R // 16, pid, nprg)
        row0 = r0 * 16
        ntok = (r1 - r0) * 16
        at = fp8nnk__load_at_hwt(w, C, NKB, row0, ntok, w.dtype)
        q = nl.ndarray((fp8nnk__P, NKB, ntok), dtype=nl.uint8, buffer=nl.sbuf)
        qf = q.view(nl.float8_e4m3)
        for kk in range(NKB):
            if clip == 1:
                nisa.tensor_scalar(dst=at[:, kk, :], data=at[:, kk, :], op0=nl.minimum, operand0=bound, op1=nl.maximum, operand1=-bound, engine=nisa.engine.vector)
            nisa.activation(dst=qf[:, kk, :], op=nl.copy, data=at[:, kk, :], scale=scale)
            nisa.dma_copy(dst=out[kk * fp8nnk__P:(kk + 1) * fp8nnk__P, row0:row0 + ntok], src=q[:, kk, :])
        return out

    @nki.jit
    def fp8nnk__quant_w3_kernel(w2, n, K, dr, scale, clip, bound):
        """[R1 fp8 wprep] n weights in ONE launch: member m is rows [m*dr, m*dr + K) of the bf16 [R, N] window
        `w2`; out is their fp8 e4m3 bytes, uint8 [n*K, N] (contiguous, member m at rows [m*K, (m+1)*K)).

        Per 128-row slab the instruction sequence is `_quant_w_nn_kernel`'s (and nki_gemm_fp8._quant_w_kernel's)
        verbatim -- dma_copy the slab, the optional VectorE clamp to +-bound, ONE ScalarE activation(scale) into the
        fp8 view of a uint8 tile, dma_copy out -- so every output byte is the byte the per-weight kernel produces for
        the same bf16 value (each is elementwise; nothing crosses a slab).  The slab loop runs over the flattened
        (weight, slab) index and is split across the launch grid exactly as the per-weight kernel splits its slabs.

        THE INPUT IS A CONTIGUOUS 2-D WINDOW, NEVER A STRIDED VIEW.  The n layer weights are rows 0,2,..,22 (or
        1,3,..,23) of the Muon parameter stack, i.e. evenly spaced inside one contiguous run of memory; the caller
        passes that run as one row-contiguous [R, N] tensor (dr = rows between members, a trace-time constant) and
        the kernel picks member rows by tensor-slice indexing `w2[row:row+P, :]` -- the access form, and the input
        class (a contiguous 2-D tensor at a storage offset), of every per-weight quantise that runs today.  No
        `.ap()` (offset semantics on a view unverified, plan_v11 refute C1) and no dim-0 stride reaches NKI.  The
        first batch of every class is still checked on device against the per-weight kernel before any batched
        byte is used."""
        N = w2.shape[1]
        out = nl.ndarray((n * K, N), dtype=nl.uint8, buffer=nl.shared_hbm)
        nsl = K // fp8nnk__P
        pid, nprg = fp8nnk__shard()
        r0, r1 = fp8nnk__span(n * nsl, pid, nprg)
        for i in range(r1 - r0):
            s = r0 + i
            m = s // nsl
            k0 = (s - m * nsl) * fp8nnk__P
            src0 = m * dr + k0
            dst0 = m * K + k0
            stage = nl.ndarray((fp8nnk__P, N), dtype=w2.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=stage, src=w2[src0:src0 + fp8nnk__P, :])
            if clip == 1:
                nisa.tensor_scalar(dst=stage, data=stage, op0=nl.minimum, operand0=bound, op1=nl.maximum, operand1=-bound, engine=nisa.engine.vector)
            q_u8 = nl.ndarray((fp8nnk__P, N), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.activation(dst=q_u8.view(nl.float8_e4m3), op=nl.copy, data=stage, scale=scale)
            nisa.dma_copy(dst=out[dst0:dst0 + fp8nnk__P, :], src=q_u8)
        return out

    @nki.jit
    def fp8nnk__qdq_nn_kernel(x, scale, inv, clip, bound, fmt):
        """bf16 [R, C] -> fp8_fmt(scale * x) / scale, back in bf16.  The quality probe: every finite
        e4m3 and e5m2 value is exactly representable in bf16, so the products a bf16 matmul then
        forms are bitwise the products the fp8 kernel forms and only the fp32 accumulation order
        differs.  clip == 1 clamps to +-bound before the cast."""
        R = x.shape[0]
        C = x.shape[1]
        dt = x.dtype
        out = nl.ndarray((R, C), dtype=dt, buffer=nl.shared_hbm)
        pid, nprg = fp8nnk__shard()
        r0, r1 = fp8nnk__span(R // fp8nnk__P, pid, nprg)
        for i in range(r1 - r0):
            k0 = (r0 + i) * fp8nnk__P
            stage = nl.ndarray((fp8nnk__P, C), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=stage, src=x[k0:k0 + fp8nnk__P, :])
            if clip == 1:
                nisa.tensor_scalar(dst=stage, data=stage, op0=nl.minimum, operand0=bound, op1=nl.maximum, operand1=-bound, engine=nisa.engine.vector)
            q_u8 = nl.ndarray((fp8nnk__P, C), dtype=nl.uint8, buffer=nl.sbuf)
            q_f8 = q_u8.view(fp8nnk__f8(fmt))
            nisa.activation(dst=q_f8, op=nl.copy, data=stage, scale=scale)
            back = nl.ndarray((fp8nnk__P, C), dtype=dt, buffer=nl.sbuf)
            nisa.activation(dst=back, op=nl.copy, data=q_f8, scale=inv)
            nisa.dma_copy(dst=out[k0:k0 + fp8nnk__P, :], src=back)
        return out
    fp8nnk__HAS_NKI = True
except Exception:
    fp8nnk__HAS_NKI = False
try:
    fp8nnk__bf16 = _fp8_bf16_ns
except Exception:
    fp8nnk__bf16 = None
_fp8_tn_ns = _FreezeLiveNS('fp8tnk_')
_tn = _fp8_tn_ns
'FP8 class-C (`tn`, weight-gradient) matmul: out[Kp, N] = a[M, Kp].T @ b[M, N], contraction over TOKENS.\n\n    [1024, 5120] = [16384, 1024].T @ [16384, 5120]      c_fc   dW  (x^T dh)\n    [5120, 1024] = [16384, 5120].T @ [16384, 1024]      c_proj dW  (h^T dy)\n\nThis is the third orientation of the MLP GEMM lane and the one with no kernel at all (brief 06 §1 rows\n5-6; `work/agents/ceiling_research/NOTES.md` A2: "the largest protective item on the board").  Slice 2\nported `nki_gemm`\'s bf16 class-B structure into fp8 as `nki_gemm_fp8._mm_fp8_kernel`; this module does the\nsame for `nki_gemm`\'s proven bf16 class-C path, `_mm_tn_kernel` / `nki_matmul(a, b, trans_a=True)`\n(1.187 ms = 92.0 % of the bf16 peak, the best class in the workspace).\n\n**Nothing here is wired into train.py and nothing is default-on.**  `work/nki_gemm_fp8.py` is owned by\nanother party; this file is standalone so it can be reviewed and merged there (see PATCH.md).\n\nWhy class C is NOT a relabelling of class B\n-------------------------------------------\nIn A/B the contraction runs over the feature axis, so the token-major activation must be TRANSPOSED into\n[k, token] slabs before it can be a matmul operand (the `xt=hwt` hardware-DGE strip transpose, 13 % of the\nclass-B kernel).  In C the contraction runs over the token axis, which is ALREADY the partition axis of\nboth token-major operands, so **class C needs no transpose at all** -- and under `double_row` the pair\naxis simply selects two adjacent 128-token tiles, i.e. one nc_matmul consumes a 256-token group.  What C\npays instead is an fp32 SBUF accumulator (the output is [Kp, N], far larger than PSUM) that must be folded\nonce per staged token chunk, and TWO per-call activation operands to convert instead of one activation\nplus a once-per-step weight.  Those two costs, not the transpose, are what decides the bar; NOTES.md §2\ncounts them.\n\nStructure (mirrors `nki_gemm._mm_tn_kernel`, one level of blocking added for the fp8 operand copies)\n----------------------------------------------------------------------------------------------------\n* The M = 16384 contraction is blocked into chunks of `tc` tokens.  Inside a chunk the `tc/256` double\n  tiles chain in ONE PSUM bank per output chunk (`accumulate=(d > 0)`); at the end of the chunk the bank\n  is folded into the fp32 SBUF accumulator (a copy on the first chunk, a VectorE add afterwards).\n* Operands are staged token-major by plain DMA -- `a_bf[p, j, k] = a[t0 + j*128 + p, k]`, two DMAs per\n  double tile -- then converted to fp8 in SBUF.  The fp8 `a` block is [P, nd, J, Kp_p] so the stationary\n  `a_f8[:, d, :, kt*128:]` is [P, J, 128] with J stride Kp_p (mm_fp8_qa\'s proven stationary form); the fp8\n  `b` block is [P, nd*NCp, J, 512] so every moving tile is a CONTIGUOUS [P, J, 512] (the proven moving\n  form, `bmv=chunk`), or [P, nd, J, Np] with a strided moving tile and one conversion per block\n  (`bmv=flat`).\n* `shard=auto` puts the program split on the axis of the LARGER operand -- N-chunks when N > Kp, Kp-tiles\n  when Kp > N -- because the operand whose axis is NOT split is converted in full by BOTH programs.  The\n  bf16 kernel only shards on N; at c_proj dW (Kp 5120, N 1024) that duplicates the 84 MB operand and costs\n  92.3 M element-conversions per core against 58.8 M for the Kp shard, and it does not fit SBUF either\n  (213 KiB vs 166).  This is the one structural change from the bf16 reference.\n\nOperand formats -- e4m3 on the activation, e5m2 on the gradient\n---------------------------------------------------------------\n`nc_matmul`\'s `double_row` tiles may be `float8_e4m3` or `float8_e5m2` and (per AWS\'s nc_matmul page) the\ntwo may differ between the operands; the installed package\'s `validate_matmul_dtypes` lists e5m2\nunconditionally and gates only OCP `float8_e4m3fn` on gen4.  Brief 02 §2\'s "e5m2 is not available on this\nTensorE" is false.  **The mixed pair is not verified on chip by this agent** -- `afmt`/`bfmt` are\nindependent kernel arguments precisely so the fallback (both e4m3, or both e5m2) is one integer away, and\nthe bench measures all four combinations.  NOTES.md §4 lists what a chip would have to confirm.\n\nMaxima, verified rather than assumed (NOTES.md §3):\n    nl.float8_e4m3  (LEGACY, this chip)  max finite  240.0     subnormal floor 2^-9  = 1.95e-3\n    nl.float8_e5m2                       max finite  57344.0   subnormal floor 2^-16 = 1.53e-5\n    nl.float8_e4m3fn (OCP, gen4 only)    max finite  448.0     -- NOT this chip, and 448 is the constant\n                                                                  that caused the F1_qdq divergence\nSources: `work/vendor/nkilib/core/utils/kernel_helpers.py:59-63` (`_max_pos_range_map`), `torch.finfo`\non the chip host (`work/agents/fp8/NOTES.md` §1a), and a host round trip in `test_classc_cpu.py`.\n\nCLAMP BEFORE EVERY CAST.  The chip\'s bf16->fp8 cast emits inf, not saturation, measured at ISA level with\n`--enable-saturate-infinity` set (D1-D3, `work/agents/fp8/CLASSB_HANDOVER.md` §2).  e5m2 is IEEE-like and\nhas its own inf encoding, and torch\'s `.to(float8_e5m2)` reproduces it (1e5 -> inf), so the rule applies to\ne5m2 exactly as to e4m3 -- only the bound changes.  Every conversion form in this module clamps; `qmode`\nchooses HOW, never WHETHER:\n\n    qmode 0  "unit"   signed operand, scale == 1: ONE `tensor_scalar(min(+FMAX), max(-FMAX))` writing the\n                      fp8 tile.  This is AWS\'s own `static_quantization` final instruction, verbatim\n                      (`work/vendor/nkilib/core/quantization/fp8_quantize.py:89-96`).  Available only at\n                      scale 1 because tensor_scalar has exactly two op slots and a scaled signed clamp\n                      needs three ops -- which is the real reason e5m2 matters here: its 32 binades make\n                      scale 1 a correct choice for a gradient, and scale 1 halves the conversion bill.\n    qmode 1  "pos"    non-negative operand: ONE `tensor_scalar(multiply(s), minimum(FMAX))`.  h = relu^2.\n    qmode 2  "split"  signed operand with a scale: VectorE `tensor_scalar(min, max)` clamp in the unscaled\n                      bf16 domain, then ScalarE `activation(scale)` into fp8 -- two instructions on two\n                      engines, the shipped `clip=vec` form.\n\nKnobs (env, read once at import; every one is also a kernel argument so a bench varies them in one process)\n'
import os
import torch
fp8tnk__P = 128
fp8tnk__CH = 512
fp8tnk__KS = 256
fp8tnk__J = fp8tnk__KS // fp8tnk__P
fp8tnk_FMAX = {0: 240.0, 1: 57344.0}
fp8tnk__env = os.environ.get
fp8tnk_TC = 1024
fp8tnk_GROUP = 4
fp8tnk__FOLD_NAMES = {'chunk': 0, 'group': 1}
fp8tnk_FOLD = fp8tnk__FOLD_NAMES['chunk']
fp8tnk__EVAC_NAMES = {'alt': 0, 'scalar': 1, 'vector': 2}
fp8tnk_EVAC = fp8tnk__EVAC_NAMES['alt']
fp8tnk_LOOP = 1
fp8tnk__SHARD_AXIS_NAMES = {'auto': 2, 'n': 0, 'k': 1}
fp8tnk_SHARD_AXIS = fp8tnk__SHARD_AXIS_NAMES['auto']
fp8tnk__BMV_NAMES = {'chunk': 0, 'flat': 1}
fp8tnk_BMV = fp8tnk__BMV_NAMES['chunk']
fp8tnk__QENG_NAMES = {'vec': 0, 'scalar': 1, 'split': 2, 'alt': 3}
fp8tnk_QENG = fp8tnk__QENG_NAMES['vec']
fp8tnk__FMT_NAMES = {'e4m3': 0, 'e5m2': 1}
fp8tnk_AFMT = fp8tnk__FMT_NAMES['e4m3']
fp8tnk_BFMT = fp8tnk__FMT_NAMES['e5m2']
fp8tnk__QMODE_NAMES = {'unit': 0, 'pos': 1, 'split': 2}
fp8tnk_AQ = fp8tnk__QMODE_NAMES['unit']
fp8tnk_BQ = fp8tnk__QMODE_NAMES['unit']
fp8tnk_A_SCALE = 1
fp8tnk_B_SCALE = 1
fp8tnk__LNC = int(fp8tnk__env('NEURON_LOGICAL_NC_CONFIG', '1'))
fp8tnk_SBUF_BUDGET = 200 * 1024
fp8tnk__KERNEL_DISABLED = False
fp8tnk__LAST_KERNEL_ERROR = None

def fp8tnk_effective_scale(qmode: int, scale: float) -> float:
    """qmode `unit` IS scale 1 -- the whole point of it is that the scale slot is free for the clamp."""
    return 1.0 if qmode == 0 else float(scale)

def fp8tnk_clip_bound(fmt: int, scale: float) -> float:
    """The clamp bound in the UNSCALED bf16 domain: the largest bf16 b with b * scale <= FMAX.  The clamped
    value lives in a bf16 tile, so for a non-power-of-two scale FMAX/scale would round UP in bf16 and let a
    value past the maximum through; power-of-two scales are exact.  (nki_gemm_fp8.clip_bound, generalised
    over the format.)"""
    fmax = fp8tnk_FMAX[fmt]
    b = torch.tensor(fmax / scale, dtype=torch.float32).to(torch.bfloat16)
    for _ in range(4):
        if float(b) * scale <= fmax:
            break
        b = torch.nextafter(b, torch.tensor(0.0, dtype=torch.bfloat16))
    return float(b)

def fp8tnk_shard_axis_for(Kp: int, N: int, axis: int=None) -> int:
    """0 = split the N-chunks (replicates `a`), 1 = split the Kp-tiles (replicates `b`).

    Each program converts the operand whose axis is NOT split IN FULL, so the per-core conversion is
    M*(Kp + N/2) for the N split and M*(Kp/2 + N) for the Kp split: the N split wins exactly when Kp < N.
    At the two real shapes this is the difference between 58.8 M and 92.3 M elements per core."""
    a = fp8tnk_SHARD_AXIS if axis is None else axis
    if a != 2:
        return a
    return 0 if N > Kp else 1

def fp8tnk_plan_tn_fp8(M: int, Kp: int, N: int, tc: int=None, group: int=fp8tnk_GROUP, nprg: int=2, axis: int=None, qbuf: int=2):
    """SBUF bytes per partition of the class-C fp8 plan, or None if the shape is not tileable.

        acc      (fp32, resident)        NKp_p * Np * 4          80 KiB at BOTH real shapes -- 40 % of the
                                                                 budget, identical in bf16, and the reason
                                                                 tc cannot grow far
        a fp8    (double-buffered)       2 * tct * Kp_p
        b fp8    (double-buffered)       2 * tct * Np
        bf16 staging, ONE double tile    2 * J * (Kp_p + Np) * 2   (the fp8 copy is what persists, not this)
        out staging                      Np * 2

    c_fc   dW [16384,1024]^T x [16384,5120], N split,  tc=1024 -> 169 KiB (80 + 16 + 40 + 28 + 5)
    c_proj dW [16384,5120]^T x [16384,1024], Kp split, tc=1024 -> 166 KiB (80 + 40 + 16 + 28 + 2)
    The same c_proj shape on the bf16 kernel's N split is 213 KiB and does not fit, which is the second
    reason `shard=auto` exists.  tc=2048 is 211/208 KiB: inside the 224 KiB hardware limit, outside the
    200 KiB default budget, and worth a bench row because it halves the fold count."""
    if tc is None:
        tc = fp8tnk_TC
    if M % fp8tnk__KS or Kp % fp8tnk__P or N % fp8tnk__CH:
        return None
    if M < fp8tnk__KS or Kp < fp8tnk__P or N < fp8tnk__CH:
        return None
    if tc % fp8tnk__KS or tc < fp8tnk__KS or tc > M:
        return None
    if group < 1 or group > 8:
        return None
    sh = fp8tnk_shard_axis_for(Kp, N, axis)
    NKp = Kp // fp8tnk__P
    NC = N // fp8tnk__CH
    if sh == 1:
        NKp_p = (NKp + nprg - 1) // nprg
        NCp = NC
    else:
        NKp_p = NKp
        NCp = (NC + nprg - 1) // nprg
    if NKp_p < 1 or NCp < 1:
        return None
    Kp_p = NKp_p * fp8tnk__P
    Np = NCp * fp8tnk__CH
    tct = tc // fp8tnk__P
    acc_bytes = NKp_p * Np * 4
    af8_bytes = qbuf * tct * Kp_p
    bf8_bytes = qbuf * tct * Np
    stage_bytes = qbuf * fp8tnk__J * (Kp_p + Np) * 2
    out_bytes = Np * 2
    return {'shard': sh, 'NKp': NKp, 'NC': NC, 'NKp_p': NKp_p, 'NCp': NCp, 'Kp_p': Kp_p, 'Np': Np, 'tc': tc, 'tct': tct, 'nd': tct // fp8tnk__J, 'folds': M // tc, 'acc': acc_bytes, 'af8': af8_bytes, 'bf8': bf8_bytes, 'stage': stage_bytes, 'out': out_bytes, 'sbuf': acc_bytes + af8_bytes + bf8_bytes + stage_bytes + out_bytes}

def fp8tnk_pick_tc(M: int, Kp: int, N: int, group: int=fp8tnk_GROUP, nprg: int=2, axis: int=None):
    """The largest legal tc whose plan fits the budget (fold volume is M/tc * Kp*N/nprg elements, so a
    bigger tc is strictly better for VectorE and strictly worse for SBUF).  None when nothing fits."""
    tc = fp8tnk_TC
    while tc >= fp8tnk__KS:
        pl = fp8tnk_plan_tn_fp8(M, Kp, N, tc=tc, group=group, nprg=nprg, axis=axis)
        if pl is not None and pl['sbuf'] <= fp8tnk_SBUF_BUDGET and (M % tc == 0):
            return tc
        tc = tc // 2
    return None
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    fp8tnk__PM = nisa.matmul_perf_mode.double_row

    def fp8tnk__f8(fmt):
        if fmt == 1:
            return nl.float8_e5m2
        return nl.float8_e4m3

    def fp8tnk__eng(which):
        if which == 1:
            return nisa.engine.scalar
        return nisa.engine.vector

    def fp8tnk__shard():
        if nl.program_ndim() > 0:
            return (nl.program_id(0), nl.num_programs(0))
        return (0, 1)

    def fp8tnk__span(n, pid, nprg):
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
        return (i0, i0 + cnt)

    def fp8tnk__ceil_div(a, b):
        return (a + b - 1) // b

    def fp8tnk__ptiles(n, shape, dtype, buf):
        """A tuple of n (<= 8) fresh tiles; an if-chain, as nki_gemm does (no comprehensions in a kernel)."""
        t0 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 1:
            return (t0,)
        t1 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 2:
            return (t0, t1)
        t2 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 3:
            return (t0, t1, t2)
        t3 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 4:
            return (t0, t1, t2, t3)
        t4 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 5:
            return (t0, t1, t2, t3, t4)
        t5 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 6:
            return (t0, t1, t2, t3, t4, t5)
        t6 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        if n == 7:
            return (t0, t1, t2, t3, t4, t5, t6)
        t7 = nl.ndarray(shape, dtype=dtype, buffer=buf)
        return (t0, t1, t2, t3, t4, t5, t6, t7)

    def fp8tnk__ps_j(pss, pcontig, fold_mode, j):
        if fold_mode == 1:
            return pcontig[:, j * fp8tnk__CH:(j + 1) * fp8tnk__CH]
        return pss[j]

    def fp8tnk__evac(dst, src, k, evac):
        """fp32 PSUM -> fp32 SBUF accumulator on the FIRST chunk (a copy, not an add)."""
        if evac == 1:
            nisa.activation(dst=dst, op=nl.copy, data=src)
        elif evac == 2:
            nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.vector)
        elif k % 2 == 0:
            nisa.activation(dst=dst, op=nl.copy, data=src)
        else:
            nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.vector)

    def fp8tnk__convert(dst, src, qmode, scale, fmax, bound, eng):
        """bf16 `src` -> fp8 `dst` (dst carries the format).  Every branch clamps.

        qmode 0 `unit`  : ONE tensor_scalar min(+FMAX)/max(-FMAX) straight into the fp8 tile -- AWS's
                          `static_quantization` final instruction verbatim.  Correct for a SIGNED operand
                          and needs the scale to be 1, because tensor_scalar has two op slots and a scaled
                          signed clamp wants three.
        qmode 1 `pos`   : ONE tensor_scalar multiply(scale)/minimum(FMAX) -- NON-NEGATIVE operands only
                          (h = relu(pre)^2).  A negative input is NOT clamped by this form.
        qmode 2 `split` : VectorE clamp of the bf16 tile to +-bound (= FMAX/scale) IN PLACE, then a ScalarE
                          activation(scale) into the fp8 tile: two instructions on two engines, the shipped
                          `clip=vec` order.  `src` is a staging tile, so the in-place clamp is safe."""
        if qmode == 0:
            nisa.tensor_scalar(dst=dst, data=src, op0=nl.minimum, operand0=fmax, op1=nl.maximum, operand1=-fmax, engine=eng)
        elif qmode == 1:
            nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=scale, op1=nl.minimum, operand1=fmax, engine=eng)
        else:
            nisa.tensor_scalar(dst=src, data=src, op0=nl.minimum, operand0=bound, op1=nl.maximum, operand1=-bound, engine=nisa.engine.vector)
            nisa.activation(dst=dst, op=nl.copy, data=src, scale=scale)

    def fp8tnk__stage_double_tile(a2, b2, a_f8, b_f8, d, row0, k_off, n_off, Kp_p, Np, NCp, dt, aq, bq, a_scale, b_scale, a_fmax, b_fmax, a_bound, b_bound, aeng, beng, bmv):
        """Load the 256 tokens [row0, row0+256) of both operands token-major and convert them to fp8.

        NO TRANSPOSE: the contraction axis is the token axis, which is already the partition axis, so this
        is two plain DMAs per operand (one per pair index j).  Each DMA is [128 partitions, Kp_p or Np
        contiguous elements] = 2-5 KiB per partition, above the contiguity guideline."""
        a_bf = nl.ndarray((fp8tnk__P, fp8tnk__J, Kp_p), dtype=dt, buffer=nl.sbuf)
        for j in range(fp8tnk__J):
            r0 = row0 + j * fp8tnk__P
            nisa.dma_copy(dst=a_bf[:, j, :], src=a2[r0:r0 + fp8tnk__P, k_off:k_off + Kp_p])
        fp8tnk__convert(a_f8[:, d, :, :], a_bf, aq, a_scale, a_fmax, a_bound, aeng)
        b_bf = nl.ndarray((fp8tnk__P, fp8tnk__J, Np), dtype=dt, buffer=nl.sbuf)
        for j in range(fp8tnk__J):
            r0 = row0 + j * fp8tnk__P
            nisa.dma_copy(dst=b_bf[:, j, :], src=b2[r0:r0 + fp8tnk__P, n_off:n_off + Np])
        if bmv == 0:
            for c in range(NCp):
                for j in range(fp8tnk__J):
                    fp8tnk__convert(b_f8[:, d * NCp + c, j, :], b_bf[:, j, c * fp8tnk__CH:(c + 1) * fp8tnk__CH], bq, b_scale, b_fmax, b_bound, beng)
        else:
            fp8tnk__convert(b_f8[:, d, :, :], b_bf, bq, b_scale, b_fmax, b_bound, beng)

    def fp8tnk__tn_chunk_fp8(a_f8, b_f8, acc, nd, first, NKp_p, NCp, npass, group, fold_mode, evac, bmv, f32):
        """The matmuls of one staged token chunk: per k'-tile and per group of output chunks, chain the
        chunk's `nd` double tiles in PSUM (accumulate over d), then fold into the fp32 accumulator."""
        for kt in range(NKp_p):
            for p in range(npass):
                g0 = p * group
                g = NCp - g0
                if g > group:
                    g = group
                if fold_mode == 1:
                    pcontig = nl.ndarray((fp8tnk__P, g * fp8tnk__CH), dtype=f32, buffer=nl.psum)
                    pss = None
                else:
                    pss = fp8tnk__ptiles(g, (fp8tnk__P, fp8tnk__CH), f32, nl.psum)
                    pcontig = None
                for d in range(nd):
                    st = a_f8[:, d, :, kt * fp8tnk__P:(kt + 1) * fp8tnk__P]
                    for j in range(g):
                        if bmv == 0:
                            mv = b_f8[:, d * NCp + g0 + j, :, :]
                        else:
                            mv = b_f8[:, d, :, (g0 + j) * fp8tnk__CH:(g0 + j + 1) * fp8tnk__CH]
                        nisa.nc_matmul(dst=fp8tnk__ps_j(pss, pcontig, fold_mode, j), stationary=st, moving=mv, accumulate=d > 0, perf_mode=fp8tnk__PM)
                if fold_mode == 1:
                    accs = acc[:, kt, g0 * fp8tnk__CH:(g0 + g) * fp8tnk__CH]
                    if first == 1:
                        fp8tnk__evac(accs, pcontig, kt + p, evac)
                    else:
                        nisa.tensor_tensor(dst=accs, data1=pcontig, data2=accs, op=nl.add)
                else:
                    for j in range(g):
                        accs = acc[:, kt, (g0 + j) * fp8tnk__CH:(g0 + j + 1) * fp8tnk__CH]
                        if first == 1:
                            fp8tnk__evac(accs, pss[j], kt + j, evac)
                        else:
                            nisa.tensor_tensor(dst=accs, data1=pss[j], data2=accs, op=nl.add)

    def fp8tnk__chunk(a2, b2, acc, ct0, nd, first, k_off, n_off, Kp_p, Np, NKp_p, NCp, npass, group, fold_mode, evac, bmv, dt, f32, aq, bq, a_scale, b_scale, a_fmax, b_fmax, a_bound, b_bound, a_f8_dt, b_f8_dt, qeng):
        """Stage `nd` double tiles (= nd*256 tokens) of both operands in fp8, then run their matmuls."""
        a_f8 = nl.ndarray((fp8tnk__P, nd, fp8tnk__J, Kp_p), dtype=a_f8_dt, buffer=nl.sbuf)
        if bmv == 0:
            b_f8 = nl.ndarray((fp8tnk__P, nd * NCp, fp8tnk__J, fp8tnk__CH), dtype=b_f8_dt, buffer=nl.sbuf)
        else:
            b_f8 = nl.ndarray((fp8tnk__P, nd, fp8tnk__J, Np), dtype=b_f8_dt, buffer=nl.sbuf)
        big_a = 1
        if Np > Kp_p:
            big_a = 0
        for d in range(nd):
            if qeng == 1:
                aeng, beng = (fp8tnk__eng(1), fp8tnk__eng(1))
            elif qeng == 2:
                aeng, beng = (fp8tnk__eng(big_a), fp8tnk__eng(1 - big_a))
            elif qeng == 3:
                aeng, beng = (fp8tnk__eng(d % 2), fp8tnk__eng(d % 2))
            else:
                aeng, beng = (fp8tnk__eng(0), fp8tnk__eng(0))
            fp8tnk__stage_double_tile(a2, b2, a_f8, b_f8, d, ct0 + d * fp8tnk__KS, k_off, n_off, Kp_p, Np, NCp, dt, aq, bq, a_scale, b_scale, a_fmax, b_fmax, a_bound, b_bound, aeng, beng, bmv)
        fp8tnk__tn_chunk_fp8(a_f8, b_f8, acc, nd, first, NKp_p, NCp, npass, group, fold_mode, evac, bmv, f32)

    @nki.jit
    def fp8tnk__mm_tn_fp8_kernel(a2, b2, tc, group, fold_mode, evac, loop, shard, bmv, qeng, afmt, bfmt, aq, bq, a_scale, b_scale, a_bound, b_bound, inv_scale):
        """out[Kp, N] = bf16((a2[M, Kp].T @ b2[M, N]) / (s_a s_b)), both operands converted to fp8 in SBUF
        and the contraction double-pumped over tokens (256 tokens per nc_matmul).

        a2, b2     bf16, token-major, exactly as the training tensors are laid out.  No transpose.
        shard      0: split the N-chunks across the two programs; 1: split the Kp-tiles.
        aq/bq      0 unit (signed, scale 1), 1 pos (non-negative), 2 split (signed, scaled) -- all clamp.
        a_bound    FMAX_a / s_a in the unscaled bf16 domain (used by qmode 2 only).
        inv_scale  1 / (s_a s_b), folded into the one pass that writes the output."""
        M = a2.shape[0]
        Kp = a2.shape[1]
        N = b2.shape[1]
        dt = a2.dtype
        f32 = nl.float32
        out = nl.ndarray((Kp, N), dtype=dt, buffer=nl.shared_hbm)
        NKp = Kp // fp8tnk__P
        NC = N // fp8tnk__CH
        pid, nprg = fp8tnk__shard()
        if shard == 1:
            k0, k1 = fp8tnk__span(NKp, pid, nprg)
            c0 = 0
            c1 = NC
        else:
            k0 = 0
            k1 = NKp
            c0, c1 = fp8tnk__span(NC, pid, nprg)
        NKp_p = k1 - k0
        NCp = c1 - c0
        Kp_p = NKp_p * fp8tnk__P
        Np = NCp * fp8tnk__CH
        k_off = k0 * fp8tnk__P
        n_off = c0 * fp8tnk__CH
        npass = fp8tnk__ceil_div(NCp, group)
        a_f8_dt = fp8tnk__f8(afmt)
        b_f8_dt = fp8tnk__f8(bfmt)
        a_fmax = 240.0
        if afmt == 1:
            a_fmax = 57344.0
        b_fmax = 240.0
        if bfmt == 1:
            b_fmax = 57344.0
        acc = nl.ndarray((fp8tnk__P, NKp_p, Np), dtype=f32, buffer=nl.sbuf)
        ndc = tc // fp8tnk__KS
        NDT = M // fp8tnk__KS
        if ndc > NDT:
            ndc = NDT
        nfull = NDT // ndc
        ntail = NDT - nfull * ndc
        if loop == 1:
            fp8tnk__chunk(a2, b2, acc, 0, ndc, 1, k_off, n_off, Kp_p, Np, NKp_p, NCp, npass, group, fold_mode, evac, bmv, dt, f32, aq, bq, a_scale, b_scale, a_fmax, b_fmax, a_bound, b_bound, a_f8_dt, b_f8_dt, qeng)
            for i in nl.affine_range(nfull - 1):
                fp8tnk__chunk(a2, b2, acc, (i + 1) * ndc * fp8tnk__KS, ndc, 0, k_off, n_off, Kp_p, Np, NKp_p, NCp, npass, group, fold_mode, evac, bmv, dt, f32, aq, bq, a_scale, b_scale, a_fmax, b_fmax, a_bound, b_bound, a_f8_dt, b_f8_dt, qeng)
        else:
            for ch in range(nfull):
                first = 0
                if ch == 0:
                    first = 1
                fp8tnk__chunk(a2, b2, acc, ch * ndc * fp8tnk__KS, ndc, first, k_off, n_off, Kp_p, Np, NKp_p, NCp, npass, group, fold_mode, evac, bmv, dt, f32, aq, bq, a_scale, b_scale, a_fmax, b_fmax, a_bound, b_bound, a_f8_dt, b_f8_dt, qeng)
        if ntail > 0:
            fp8tnk__chunk(a2, b2, acc, nfull * ndc * fp8tnk__KS, ntail, 0, k_off, n_off, Kp_p, Np, NKp_p, NCp, npass, group, fold_mode, evac, bmv, dt, f32, aq, bq, a_scale, b_scale, a_fmax, b_fmax, a_bound, b_bound, a_f8_dt, b_f8_dt, qeng)
        for i in range(NKp_p):
            ob = nl.ndarray((fp8tnk__P, Np), dtype=dt, buffer=nl.sbuf)
            if i % 2 == 0:
                nisa.activation(dst=ob, op=nl.copy, data=acc[:, i, :], scale=inv_scale)
            else:
                nisa.tensor_scalar(dst=ob, data=acc[:, i, :], op0=nl.multiply, operand0=inv_scale, engine=nisa.engine.vector)
            nisa.dma_copy(dst=out[(k0 + i) * fp8tnk__P:(k0 + i + 1) * fp8tnk__P, n_off:n_off + Np], src=ob)
        return out
    fp8tnk__HAS_NKI = True
except Exception:
    fp8tnk__HAS_NKI = False

def fp8tnk__disable_kernel(where: str, err: BaseException) -> None:
    global fp8tnk__KERNEL_DISABLED, fp8tnk__LAST_KERNEL_ERROR
    if not fp8tnk__KERNEL_DISABLED:
        fp8tnk__KERNEL_DISABLED = True
        fp8tnk__LAST_KERNEL_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[fp8_tn] kernel disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)

def fp8tnk__launch(kernel):
    return kernel[fp8tnk__LNC] if fp8tnk__LNC == 2 else kernel

def fp8tnk__nprg() -> int:
    return 2 if fp8tnk__LNC == 2 else 1

def fp8tnk_kernel_available(device: torch.device) -> bool:
    return fp8tnk__HAS_NKI and (not fp8tnk__KERNEL_DISABLED) and (device.type == 'neuron')

def fp8tnk_tn_kernel_args(M: int, Kp: int, N: int, tc: int=None, group: int=fp8tnk_GROUP, fold: int=fp8tnk_FOLD, evac: int=fp8tnk_EVAC, loop: int=fp8tnk_LOOP, axis: int=None, bmv: int=fp8tnk_BMV, qeng: int=fp8tnk_QENG, afmt: int=fp8tnk_AFMT, bfmt: int=fp8tnk_BFMT, aq: int=fp8tnk_AQ, bq: int=fp8tnk_BQ, a_scale: float=fp8tnk_A_SCALE, b_scale: float=fp8tnk_B_SCALE, inv_scale: float=None, nprg: int=None) -> tuple:
    """The kernel's SCALAR argument tuple, in order, after every default has been resolved.

    Single source of truth: `run_tn_fp8` and `host_check_classc.py` both go through here, so a harness
    cannot silently compile a different argument order from the one production launches.  `qmode` unit
    forces its scale to 1 -- that is what buys the one-instruction clamped cast, so it must not be
    overridable by a stray `a_scale`."""
    nprg = fp8tnk__nprg() if nprg is None else nprg
    sh = fp8tnk_shard_axis_for(Kp, N, axis)
    if tc is None:
        tc = fp8tnk_pick_tc(M, Kp, N, group=group, nprg=nprg, axis=sh) or fp8tnk_TC
    sa = fp8tnk_effective_scale(aq, a_scale)
    sb = fp8tnk_effective_scale(bq, b_scale)
    if inv_scale is None:
        inv_scale = 1.0 / (sa * sb)
    return (int(tc), int(group), int(fold), int(evac), int(loop), int(sh), int(bmv), int(qeng), int(afmt), int(bfmt), int(aq), int(bq), float(sa), float(sb), fp8tnk_clip_bound(afmt, sa), fp8tnk_clip_bound(bfmt, sb), float(inv_scale))

def fp8tnk_run_tn_fp8(a: torch.Tensor, b: torch.Tensor, tc: int=None, group: int=fp8tnk_GROUP, fold: int=fp8tnk_FOLD, evac: int=fp8tnk_EVAC, loop: int=fp8tnk_LOOP, axis: int=None, bmv: int=fp8tnk_BMV, qeng: int=fp8tnk_QENG, afmt: int=fp8tnk_AFMT, bfmt: int=fp8tnk_BFMT, aq: int=fp8tnk_AQ, bq: int=fp8tnk_BQ, a_scale: float=fp8tnk_A_SCALE, b_scale: float=fp8tnk_B_SCALE, inv_scale: float=None) -> torch.Tensor:
    """Direct launch of the class-C fp8 kernel (no fallback): out[Kp, N] = (a[M,Kp].T @ b[M,N]) / (s_a s_b).

    Both operands are bf16 token-major training tensors; the kernel converts them in SBUF."""
    M, Kp = a.shape
    N = b.shape[1]
    ka = fp8tnk_tn_kernel_args(M, Kp, N, tc=tc, group=group, fold=fold, evac=evac, loop=loop, axis=axis, bmv=bmv, qeng=qeng, afmt=afmt, bfmt=bfmt, aq=aq, bq=bq, a_scale=a_scale, b_scale=b_scale, inv_scale=inv_scale)
    return fp8tnk__launch(fp8tnk__mm_tn_fp8_kernel)(a, b, *ka)
try:
    fp8nnk__relu2 = _FreezeNS(relu2_backward_from_h=relu2k_relu2_backward_from_h, _H_R32=relu2k__H_R32)
except Exception:
    fp8nnk__relu2 = None
fp8nnk__H_R32_LIVE = bool(getattr(fp8nnk__relu2, '_H_R32', False)) if fp8nnk__relu2 is not None else False

def fp8nnk__disable_kernel(where: str, err: BaseException) -> None:
    global fp8nnk__KERNEL_DISABLED, fp8nnk__LAST_KERNEL_ERROR
    if not fp8nnk__KERNEL_DISABLED:
        fp8nnk__KERNEL_DISABLED = True
        fp8nnk__LAST_KERNEL_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[nki_gemm_fp8_nn] kernel disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)

def fp8nnk__launch(kernel):
    return kernel[fp8nnk__LNC] if fp8nnk__LNC == 2 else kernel

def fp8nnk_active(training: bool) -> bool:
    """True when a patched site should take this module's branch at all.  The default keeps the
    forward in bf16 at eval (`R1_NKI_GEMM_FP8_NN_EVAL=train`): the weights are the trained ones
    either way.  `always` makes the evaluated function bit-for-bit the trained one."""
    return bool(training)

def fp8nnk_proj_active(training: bool) -> bool:
    """The MLP c_proj Linear takes this module's branch when row 2 or row 4 is selected (row 2 is
    its forward, row 4 its dX -- one autograd Function owns both)."""
    return (fp8nnk_ROW2 or fp8nnk_ROW4) and fp8nnk_active(training)

def fp8nnk_dw_active(training: bool) -> bool:
    """True when a dW row may fire.  Its own master switch, like the head's.  Rows 5 and 6 are both
    BACKWARD rows, so `training` is what decides in practice and DW_EVAL_MODE is cosmetic."""
    return bool(training)

def fp8nnk_fc_active(training: bool) -> bool:
    """The MLP c_fc site takes this module's branch for row 3 (its dX) or row 5 (its dW).  Row 1 stays
    governed by R1_NKI_GEMM_FP8 and this module never turns it on or off.

    THE ROW-5 TERM IS LOAD-BEARING AND WAS THE FIRST OF TWO PLACES A dW-ONLY ARM DIED.  train.py:1034
    tests this function before routing the c_fc through `fused_fc_relu2_nn`, so with `ROW3 and
    enters this module's autograd Function at all and row 5 cannot run -- with no error, and with
    config_string() still printing `live=dw:5`.  Note the two terms carry DIFFERENT gates: `active()`
    reads USE_NKI_GEMM_FP8_NN and `dw_active()` reads USE_NKI_GEMM_FP8_DW, so neither switch can turn
    the other's row on."""
    return fp8nnk_ROW3 and fp8nnk_active(training) or fp8nnk_dw_active(training)

def fp8nnk__qdq_for(row: int) -> bool:
    """`qdq` mode: today's bf16 arithmetic on fp8-ROUNDED operands, the quality read with no
    throughput confound.  R1_NKI_GEMM_FP8_NN=qdq selects it for every row this module owns."""
    return fp8nnk_QDQ_ONLY

def fp8nnk__usable(*ts) -> bool:
    if not fp8nnk__HAS_NKI or fp8nnk__KERNEL_DISABLED:
        return False
    for t in ts:
        if t.device.type != 'neuron' or t.dtype != torch.bfloat16 or t.dim() != 2 or (not t.is_contiguous()):
            return False
    return True

def fp8nnk_begin_step() -> None:
    """Called by train.py once per optimizer step: the cached fp8 weights die with the step that
    produced them.  Independent of nki_gemm_fp8's epoch, so the two modules cannot interfere."""
    global fp8nnk__STEP_EPOCH
    fp8nnk__STEP_EPOCH += 1

def fp8nnk__cache_get(w: torch.Tensor, tag: str):
    ent = fp8nnk__WCACHE.get((id(w), tag))
    if ent is None:
        return None
    key, val = ent
    if key == (fp8nnk__STEP_EPOCH, w.data_ptr(), w._version):
        return val
    return None

def fp8nnk__cache_put(w: torch.Tensor, tag: str, val) -> None:
    fp8nnk__WCACHE[id(w), tag] = ((fp8nnk__STEP_EPOCH, w.data_ptr(), w._version), val)

def fp8nnk_quantised_weight(w: torch.Tensor, scale: float, trans: bool, key: torch.Tensor=None):
    """The fp8 e4m3 (uint8-carried) [K, N] copy of a weight, produced at most once per optimizer
    step.  Key: (step epoch, data_ptr, _version) of the PARAMETER -- the epoch alone would be enough
    if begin_step() is never missed, _version alone would be enough if every optimizer write bumps
    it; requiring both means either mechanism failing alone still invalidates the entry.

    `trans=True` quantises the TRANSPOSE of the stored parameter, which the c_proj forward (row 2)
    and the c_fc dX (row 3) both need: rows 1/4 consume the natural [in, out] layout and rows 2/3 the
    [out, in] one, and fp8 cannot be DMA-transposed (hwdge needs a 2-byte dtype), so the two
    orientations are two distinct byte arrays.  The transpose is a bf16 `.t().contiguous()` today --
    10.5 MB read + 10.5 MB written per weight per step, the price of not having a transposing
    quantiser kernel (PATCH.md S6, the named follow-up).  Returns None when the quantiser cannot
    run, which sends the caller down the bf16 path.

    `key` is the object the cache is keyed on, and a BACKWARD caller must pass `ctx.w`.
    `ctx.saved_tensors` returns the ORIGINAL object only when the saved tensor is a leaf that
    requires grad; anything torch reconstructs on unpack -- the `_weight_for` cast on an fp32-master
    module, for instance -- comes back as a NEW Tensor with a new `id()`, and the cache would then
    miss on every micro-step, requantising four weights per block and, with an auto scale, syncing
    the host per block.  `ctx.w` is the very object the forward was handed, so it is stable always."""
    tag = f'f8@{scale:g}{('T' if trans else '')}'
    k = w if key is None else key
    cached = fp8nnk__cache_get(k, tag)
    if cached is not None:
        return cached
    if fp8nnk_WPREP and tag in fp8nnk__WPREP_NN_TAGS and fp8nnk__usable(w) and fp8nnk__same_data(k, w):
        wq = fp8nnk__wprep_miss('nn', tag, k, scale, trans, 1, fp8nnk_clip_bound(scale, 0))
        if wq is not None:
            return wq
    if not fp8nnk__usable(w):
        return None
    try:
        if trans and fp8nnk_WQT == 'hwt':
            wq = fp8nnk__launch(fp8nnk__quant_w_t_kernel)(w.detach(), float(scale), 1, fp8nnk_clip_bound(scale, 0))
        else:
            src = w.detach().t().contiguous() if trans else w.detach()
            K, N = src.shape
            if K % fp8nnk__P or fp8nnk_plan_quant(K, N) is None:
                return None
            wq = fp8nnk__launch(fp8nnk__quant_w_nn_kernel)(src, float(scale), 1, fp8nnk_clip_bound(scale, 0))
    except Exception as err:
        fp8nnk__disable_kernel('quant_w_nn', err)
        return None
    fp8nnk__cache_put(k, tag, wq)
    return wq

def fp8nnk_rounded_weight(w: torch.Tensor, scale: float, key: torch.Tensor=None):
    """The e4m3-rounded bf16 copy of a weight for `qdq` mode.  NOTE: rounding is elementwise, so it
    COMMUTES with the transpose -- one cached rounded tensor in the parameter's natural layout serves
    both orientations, and qdq therefore needs half the cached weights the fp8 path does.
    `key`: as quantised_weight's."""
    tag = f'qdq@{scale:g}'
    k = w if key is None else key
    cached = fp8nnk__cache_get(k, tag)
    if cached is not None:
        return cached
    if not fp8nnk__usable(w):
        return None
    K, N = w.shape
    if K % fp8nnk__P or fp8nnk_plan_quant(K, N) is None:
        return None
    try:
        wr = fp8nnk__launch(fp8nnk__qdq_nn_kernel)(w.detach(), float(scale), float(1.0 / scale), 1, fp8nnk_clip_bound(scale, 0), 0)
    except Exception as err:
        fp8nnk__disable_kernel('qdq_w_nn', err)
        return None
    fp8nnk__cache_put(k, tag, wr)
    return wr

def fp8nnk_round_activation(a2d: torch.Tensor, scale: float, fmt: int):
    """The fp8-rounded bf16 copy of an activation (qdq mode only; never cached -- it is fresh every
    micro-step by definition)."""
    if not fp8nnk__usable(a2d):
        return None
    R, C = a2d.shape
    if R % fp8nnk__P or fp8nnk_plan_quant(R, C) is None:
        return None
    try:
        return fp8nnk__launch(fp8nnk__qdq_nn_kernel)(a2d, float(scale), float(1.0 / scale), 1, fp8nnk_clip_bound(scale, fmt), int(fmt))
    except Exception as err:
        fp8nnk__disable_kernel('qdq_a_nn', err)
        return None

def fp8nnk__eager(a2d: torch.Tensor, w: torch.Tensor, trans_b: bool, row: int) -> torch.Tensor:
    """The op train.py issues TODAY at this row, so the fallback is exact rather than merely equal:
      row 2  nki_matmul(h,    W2, trans_b=True)   (train.py:760, _LinearGemm.forward)
      row 3  nki_matmul(dpre, W1, trans_b=True)   (nki_gemm.FcRelu2Fn.backward)
      row 4  torch.matmul(dy, W2)                 (train.py:777, _LinearGemm.backward: the class-A shape)
    nki_matmul itself returns torch.matmul whenever its own kernel does not apply.

    ROWS 5 AND 6 ARE REFUSED, NOT ADDED TO EITHER BRANCH, AND THIS IS THE SAME CLASS OF BUG SEEN FROM
    THE OTHER SIDE.  A dW row is not `a2d @ w`: its second operand is an activation, not a weight,
    there is no `trans_b`, and its incumbent depends on `ctx.inplace` -- it is `w.grad.addmm_(x^T,
    dpre)` on the live path, `_bf16.nki_matmul(x2d, dpre, trans_a=True)` when the caller wants a
    returned dW, and `torch.matmul` only when nki_gemm is absent.  Putting row 5 in the torch branch
    would silently replace an ACCUMULATE with a PRODUCT (dropping the previous micro-step's gradient)
    and leaving it to fall through to `_bf16.nki_matmul` would pass a [M,5120] activation where a
    weight is expected.  So this function raises, `dw_matmul` returns None instead of falling back, and
    the call site keeps its own three incumbent lines.  Nothing calls `_eager` with 5 or 6; the guard
    is here so that a future edit that does fails at once and loudly rather than measuring nonsense."""
    if row in (5, 6):
        raise AssertionError(f"_eager() called with dW row {row}: a dW row's incumbent depends on ctx.inplace and cannot be expressed as a2d @ w.  Its fallback is the call site's own unchanged line; dw_matmul() returns None to select it.  See this function's docstring.")
    if row == 2:
        a2d = fp8nnk_row2_cap(a2d)
    if row == 4 or fp8nnk__bf16 is None:
        return torch.matmul(a2d, w.t() if trans_b else w)
    return fp8nnk__bf16.nki_matmul(a2d, w, trans_b=trans_b)

def fp8nnk_resolve_ascale(a2d: torch.Tensor, w: torch.Tensor, row: int, key: torch.Tensor=None, slot: str='a') -> float:
    """The activation scale for one site.  Fixed for row 2 and for a pinned gradient scale; for `auto`
    it is read from one device amax and frozen -- but ONLY once that amax is a real gradient.

    `slot` names which of a class-C row's two operands is being scaled ("a" or "b"); every nn row has
    only the one and passes the default.  It is deliberately ADDITIVE: at slot "a" the spec keys read
    ("ascale", "afmt") and the `_GCACHE` key is `(row, id(...))`, both exactly as before, so rows
    2/3/4/9/10 are bit-for-bit unchanged by its existence.

    MEASURED 2026-09-22 (K2_fp8_14_150, Melbourne 1): freezing on the FIRST call reads the scale
    before the gradients exist.  40 row-3 sites reported `|a|max 0.0000e+00 -> s_g 2^0` and 40
    row-4 sites `|a|max ~1e-7 -> s_g 2^33..2^36`.  Both are garbage for the rest of the run: at
    2^36 a later gradient of 1e-3 lands at 7e7 against e5m2's max of 57344 and clips to nothing,
    and at 2^0 it underflows instead.  Neither shows up as an error -- `bad_steps` stays 0 and the
    loss descends -- so the arm looks healthy and simply learns less.  That is the silent failure
    the rows-2-4 handover flagged as the dominant risk, and it fires on every run by default.

    The guard: a scale is only CACHED when the amax clears `_GMIN`.  Below it the value is still
    returned (so the op is well defined) but not frozen, and the next call re-reads.  Cost is one
    extra `.item()` host sync per site per step until gradients become real, which is a handful of
    steps; train.py already does one every step in its gradient gate."""
    spec = fp8nnk_ROW_SPEC[row]
    if spec[slot + 'scale'] is not None:
        return spec[slot + 'scale']
    key = (row, id(w if key is None else key)) if slot == 'a' else (row, slot, id(w if key is None else key))
    s = fp8nnk__GCACHE.get(key)
    if s is not None:
        return s
    try:
        amax = float(a2d.detach().abs().max().item())
    except Exception:
        amax = 0.0
    s = fp8nnk_auto_scale(amax, spec[slot + 'fmt'])
    if amax < fp8nnk__GMIN:
        return s
    fp8nnk__GCACHE[key] = s
    print(f'[nki_gemm_fp8_nn] row {row} ({spec['what']}){('' if slot == 'a' else ' slot ' + slot)} site {key[-1]:#x}: |a|max {amax:.4e} -> s_g 2^{int(round(math.log2(s)))} = {s:g}', flush=True)
    return s

def fp8nnk_row_matmul(a2d: torch.Tensor, w: torch.Tensor, row: int, key: torch.Tensor=None) -> torch.Tensor:
    """out = a2d @ (w.T or w) for one of rows 2/3/4, with both operands in fp8 (`MODE=1`), with both
    operands fp8-ROUNDED but the arithmetic today's bf16 (`MODE=qdq`), or exactly today's op.

    A BACKWARD caller passes `key=ctx.w`: the weight and gradient-scale caches key on `id()`, and
    `ctx.saved_tensors` hands back a fresh Tensor object every access."""
    spec = fp8nnk_ROW_SPEC[row]
    trans_b = spec['trans_b']
    if not fp8nnk__usable(a2d, w):
        return fp8nnk__eager(a2d, w, trans_b, row)
    s_a = fp8nnk_resolve_ascale(a2d, w, row, key)
    if fp8nnk__qdq_for(row):
        ar = fp8nnk_round_activation(a2d, s_a, spec['afmt'])
        wr = fp8nnk_rounded_weight(w, spec['wscale'], key)
        if ar is not None and wr is not None:
            return fp8nnk__eager(ar, wr, trans_b, row)
        return fp8nnk__eager(a2d, w, trans_b, row)
    M, K = a2d.shape
    if trans_b:
        N, K2 = w.shape
    else:
        K2, N = w.shape
    pl = fp8nnk_plan_nn_fp8(M, K, N, tb=None) if K2 == K else None
    if pl is None or pl['sbuf'] > fp8nnk_SBUF_BUDGET:
        return fp8nnk__eager(a2d, w, trans_b, row)
    wq = fp8nnk_quantised_weight(w, spec['wscale'], trans_b, key)
    if wq is None:
        return fp8nnk__eager(a2d, w, trans_b, row)
    fmt = spec['afmt']
    aq = fp8nnk_conv_form(spec['signed'], s_a)
    bound = fp8nnk_clip_bound(s_a, fmt)
    hi = fp8nnk_fmt_max(fmt) if aq == 2 else bound * s_a
    try:
        return fp8nnk__launch(fp8nnk__mm_fp8_nn_kernel)(a2d, wq, 3, int(pl['tb']), int(fp8nnk_GROUP), int(fp8nnk_EVAC), int(aq), int(fp8nnk_LOOP), int(fmt), float(s_a), float(bound), float(hi), float(1.0 / (s_a * spec['wscale'])))
    except Exception as err:
        fp8nnk__disable_kernel(f'mm_fp8_nn row{row}', err)
        return fp8nnk__eager(a2d, w, trans_b, row)
fp8nnk__DW_CALLS = {}
fp8nnk__DW_SKIPS = {}

def fp8nnk_dw_counters() -> dict:
    """How many times each dW row PRODUCED a result, and how many times it was asked and could not.
    A test proves liveness from `calls`; an arm reads `skips` to catch a row that was requested, logged
    live, and then fell back on every single call -- which looks exactly like a row that is not there."""
    return {'calls': dict(fp8nnk__DW_CALLS), 'skips': dict(fp8nnk__DW_SKIPS)}

def fp8nnk_dw_qmode(signed: bool, scale: float) -> int:
    """The class-C kernel's qmode for one operand: 0 `unit`, 1 `pos`, 2 `split`.  Every one clamps --
    only the instruction count and the engine differ (see that module's `_convert`).

    This is the class-C analogue of `conv_form`, and it is a SEPARATE function with a DIFFERENT
    numbering because the two kernels' enums disagree: here 0 is the unit-scale signed form and 1 is
    the non-negative form, while `AQ_VEC, AQ_POS, AQ_SIGN1 = 0, 1, 2` above puts the two-instruction
    form at 0.  Passing one kernel's constant to the other would select a legal-looking wrong form --
    `AQ_POS` (1) on a signed operand is the `pos` form, which does NOT clamp negatives and casts a
    large negative straight to -inf.  Hence no shared constant and no shared helper."""
    if not signed:
        return 1
    if scale == 1.0:
        return 0
    return 2

def fp8nnk__dw_usable(*ts) -> bool:
    """Rows 5/6's own usability test.  Not `_usable`: that one is shared with the nn rows and would
    tie a dW arm to the nn kernel's disable-once flag, and the class-C kernel has its own."""
    if _tn is None:
        return False
    if not _tn.kernel_available(ts[0].device):
        return False
    for t in ts:
        if t.device.type != 'neuron' or t.dtype != torch.bfloat16 or t.dim() != 2 or (not t.is_contiguous()):
            return False
    return True

def fp8nnk_dw_matmul(a2d: torch.Tensor, b2d: torch.Tensor, row: int, key: torch.Tensor=None):
    """dW = a2d[M, Ka]^T @ b2d[M, N] for a class-C row, or **None**.

    None is the whole contract: it means "nothing was done, run today's line".  A dW row's incumbent
    is an ACCUMULATE on the live path (`w.grad.addmm_(x^T, dpre)`) and a returned product otherwise,
    so there is no single fallback op this function could return instead -- see `_eager`'s guard.  The
    caller keeps its own three incumbent lines and picks one; that is what makes the off path bitwise
    and what keeps the fallback exact instead of merely equal.

    `key` must be the Parameter (`ctx.w`), and is REQUIRED when the gradient scale is `auto`: the scale
    cache keys on `id()`, and `ctx.saved_tensors` hands back a fresh Tensor object every access, so an
    unstable key would re-read an amax per call and, worse, freeze against a recycled object address.

    MODE `qdq` returns today's op on fp8-ROUNDED operands: the same products, none of the speed.  Note
    what it does and does not isolate.  Against MODE 1 it is exact -- same operands, same rounding,
    same add-back -- so it separates the FORMAT from the KERNEL.  Against the switch OFF it also
    carries the one extra bf16 rounding that returning dW instead of accumulating it costs, because
    both modes return.  Read it as the first, not the second.  `work/agents/fp8_classc/NOTES.md`'s
    closing section is explicit that this measurement, not the throughput, is what now gates class C:
    fp8 vs the bf16 product on dW is 5.9 % mean relative error, and no wgrad qdq arm has ever run."""
    spec = fp8nnk_ROW_SPEC[row]
    if not spec.get('classc'):
        raise AssertionError(f'dw_matmul() called with row {row}, which is not a class-C row')
    gslot = spec['gslot']
    if spec[gslot + 'scale'] is None and key is None:
        raise AssertionError(f"dw_matmul() row {row}: the {gslot} operand's scale is `auto` and key is None.  An auto scale freezes against id(key), and a saved tensor's id is not stable across accesses -- pass key=ctx.w, or pin R1_NKI_GEMM_FP8_GSCALE.")
    if not fp8nnk__dw_usable(a2d, b2d):
        fp8nnk__DW_SKIPS[row] = fp8nnk__DW_SKIPS.get(row, 0) + 1
        return None
    M, Ka = a2d.shape
    M2, N = b2d.shape
    if M2 != M:
        fp8nnk__DW_SKIPS[row] = fp8nnk__DW_SKIPS.get(row, 0) + 1
        return None
    s_a = spec['ascale']
    s_b = spec['bscale']
    if gslot == 'a':
        s_a = fp8nnk_resolve_ascale(a2d, key, row, key=key, slot='a')
    else:
        s_b = fp8nnk_resolve_ascale(b2d, key, row, key=key, slot='b')
    aq = fp8nnk_dw_qmode(spec['asigned'], s_a)
    bq = fp8nnk_dw_qmode(spec['bsigned'], s_b)
    nprg = _tn._nprg()
    axis = _tn.shard_axis_for(Ka, N)
    tc = _tn.pick_tc(M, Ka, N, nprg=nprg, axis=axis)
    pl = _tn.plan_tn_fp8(M, Ka, N, tc=tc, nprg=nprg, axis=axis) if tc else None
    if pl is None or pl['sbuf'] > _tn.SBUF_BUDGET:
        fp8nnk__DW_SKIPS[row] = fp8nnk__DW_SKIPS.get(row, 0) + 1
        return None
    try:
        dw = _tn.run_tn_fp8(a2d, b2d, tc=tc, axis=axis, afmt=spec['afmt'], bfmt=spec['bfmt'], aq=aq, bq=bq, a_scale=s_a, b_scale=s_b)
    except Exception as err:
        _tn._disable_kernel(f'mm_tn_fp8 row{row}', err)
        fp8nnk__DW_SKIPS[row] = fp8nnk__DW_SKIPS.get(row, 0) + 1
        return None
    fp8nnk__dw_fired(row, spec, M, Ka, N, s_a, s_b, aq, bq, pl, 'fp8')
    return dw

def fp8nnk__dw_fired(row, spec, M, Ka, N, s_a, s_b, aq, bq, pl, what) -> None:
    """The POSITIVE liveness signal, printed once per row from INSIDE the branch, after the kernel has
    actually returned.  Every other signal this project has relied on -- `live=` in config_string, the
    env var in the launch string, the plan printing fine -- is satisfied by a branch that never
    executes, and one reviewer produced a run where all four of them held and the code ran zero times.
    This line cannot appear unless a result was produced, and the counter behind it is what the CPU
    reachability test asserts on."""
    n = fp8nnk__DW_CALLS.get(row, 0) + 1
    fp8nnk__DW_CALLS[row] = n
    if n == 1:
        qn = {0: 'unit', 1: 'pos', 2: 'split'}
        print(f'[nki_gemm_fp8_nn] row {row} ({spec['what']}) FIRED [{what}]: [{M},{Ka}]^T x [{M},{N}] a={('e5m2' if spec['afmt'] else 'e4m3')}@{s_a:g}/{qn[aq]} b={('e5m2' if spec['bfmt'] else 'e4m3')}@{s_b:g}/{qn[bq]}' + (f' tc={pl['tc']} shard={pl['shard']} folds={pl['folds']} sbuf={pl['sbuf'] / 1024:.0f}KiB' if pl else '') + ' -- dW is RETURNED, the caller adds it back', flush=True)

class fp8nnk__RELU2_LINK:
    """[R1 relu2fold b] one per c_fc/c_proj PAIR per forward call, so 10 layers x 2 micro-steps get 10
    x 2 independent ones and nothing is keyed on a global.

    WHY A LINK AT ALL.  Deletion (b) moves the ReLU^2 backward from its own kernel into the producer of
    its input: `dh` IS the c_proj dX, so _ProjFn.backward can apply 2*sqrt(h) and return `dpre` where
    it used to return `dh`.  But autograd then hands that tensor to FcRelu2NNFn.backward as `dh`,
    which would apply 2*relu(pre) a SECOND time.  That is a wrong gradient that raises nothing: the
    loss still descends, the model simply learns the wrong function.  The two Functions therefore have
    to agree, per call, on who applied the factor.

    HOW THE LINK TRAVELS.  fused_fc_relu2_nn attaches it to the `h` tensor it returns, and
    MLP.forward's very next statement is `self.c_proj(h)`, so _ProjFn.forward reads it straight off its
    own input.  No change to train.py, no reliance on grad_fn identity, no global registry.  Ordering
    is a data dependency, not an assumption: _ProjFn.backward cannot run after FcRelu2NNFn.backward
    for the same layer, because its output IS that function's input.

    `folded` is written by exactly one place (_ProjFn.backward) and read by exactly one place
    (FcRelu2NNFn.backward).  If the link never arrives, or arrives and stays False, the ReLU^2 backward
    runs where it always did -- the failure mode is "no speedup", never "wrong gradient"."""
    __slots__ = ('folded',)

    def __init__(self):
        self.folded = False
fp8nnk__FOLD_CALLS = 0
fp8nnk__FROM_H_CALLS = 0
fp8nnk__FOLD_LINK_MISSES = 0
fp8nnk__FOLD_KERNEL_CALLS = 0
fp8nnk__FOLD_KERNEL_DISABLED = False
fp8nnk__FOLD_LAST_ERROR = None

def fp8nnk_fold_counters() -> dict:
    """The reachability evidence, by execution.  A test (work/agents/fp8/test_relu2fold_cpu.py) reads
    these after a real forward+backward: nonzero with the switch on, zero with it off.  A line number
    is not proof that a branch runs -- that is how three patches in this project shipped a no-op."""
    return {'from_h': fp8nnk__FROM_H_CALLS, 'fold_dx': fp8nnk__FOLD_CALLS, 'link_misses': fp8nnk__FOLD_LINK_MISSES, 'fold_kernel': fp8nnk__FOLD_KERNEL_CALLS}

def fp8nnk__relu2_from_h(h: torch.Tensor, dh: torch.Tensor) -> torch.Tensor:
    """dpre = 2 sqrt(h) dh through nki_relu2_opt's sqrt-form kernel (eager formula off-chip)."""
    global fp8nnk__FROM_H_CALLS
    fp8nnk__FROM_H_CALLS += 1
    if fp8nnk__relu2 is not None:
        return fp8nnk__relu2.relu2_backward_from_h(h, dh)
    r = torch.sqrt(h.to(torch.float32) * 4.0).to(h.dtype).to(torch.float32)
    return (r * dh.to(torch.float32)).to(dh.dtype)

def fp8nnk__disable_fold_kernel(where: str, err: BaseException) -> None:
    """[R1 relu2fold b] disable-once for the FUSED kernel alone.  Loud twice over: the print names
    the consequence (the arm degrades to the separate-pass form and its throughput number is void),
    and config_string() carries `(DISABLED)` plus fold_err so the gate grep sees it after the fact."""
    global fp8nnk__FOLD_KERNEL_DISABLED, fp8nnk__FOLD_LAST_ERROR
    if not fp8nnk__FOLD_KERNEL_DISABLED:
        fp8nnk__FOLD_KERNEL_DISABLED = True
        fp8nnk__FOLD_LAST_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[nki_gemm_fp8_nn] relu2-fold FUSED kernel disabled after failure in {where}: {type(err).__name__}: {err} -- every further fold takes the SEPARATE-PASS form (gradient unchanged, traffic NOT deleted); do not read a throughput number off this run', flush=True)

def fp8nnk__fold_kernel_fired(M: int, K: int, N: int, s_a: float, aq: int, pl: dict) -> None:
    """[R1 relu2fold b] the positive liveness signal, printed once from INSIDE the branch after the
    kernel returned -- the _dw_fired pattern, because every other signal is satisfied by a branch
    that never executes."""
    global fp8nnk__FOLD_KERNEL_CALLS
    fp8nnk__FOLD_KERNEL_CALLS += 1
    if fp8nnk__FOLD_KERNEL_CALLS == 1:
        qn = {fp8nnk_AQ_VEC: 'vec', fp8nnk_AQ_POS: 'pos', fp8nnk_AQ_SIGN1: 'sign1'}
        print(f'[nki_gemm_fp8_nn] relu2-fold FUSED epilogue FIRED: row 4 [{M},{K}]x[{K},{N}] + 2*sqrt(h)*dh in-kernel, a={('e5m2' if fp8nnk_ROW_SPEC[4]['afmt'] else 'e4m3')}@{s_a:g}/{qn[aq]} tb={pl['tb']} sbuf={pl['sbuf'] / 1024:.0f}KiB -- the standalone ReLU^2 backward kernel is DELETED at this site', flush=True)

def fp8nnk__row4_matmul_fold(dy2d: torch.Tensor, w: torch.Tensor, h2d: torch.Tensor, key: torch.Tensor):
    """[R1 relu2fold b] dpre = 2 sqrt(h2d) * ((dy2d @ w) / (s_a s_w)) through the fused kernel, or
    **None** -- dw_matmul's contract: None means "nothing was done, run today's two-step path".
    Mirrors row_matmul's row-4 body gate for gate (same plan, same weight cache, same scale
    resolution against the same _GCACHE key, so a run that starts on the fallback and later fuses
    carries ONE frozen gradient scale), with three extra refusals of its own:

      * qdq mode -- the fused kernel computes real fp8; qdq's contract is today's arithmetic on
        rounded operands, which the two-step path already provides;
      * _H_R32_LIVE -- see the flag's comment; the fused epilogue has no fp32-r form;
      * its own disable-once (_FOLD_KERNEL_DISABLED), so a compile failure here cannot silence the
        proven kernels, and vice versa.

    The plan is checked with fold=1 (the epilogue's hst staging): class A at the live shape plans
    128 KiB of the 200 KiB budget."""
    if fp8nnk__FOLD_KERNEL_DISABLED or fp8nnk__qdq_for(4) or fp8nnk__H_R32_LIVE:
        return None
    if not fp8nnk__usable(dy2d, w, h2d):
        return None
    spec = fp8nnk_ROW_SPEC[4]
    M, K = dy2d.shape
    K2, N = w.shape
    if K2 != K or h2d.shape[0] != M or h2d.shape[1] != N:
        return None
    pl = fp8nnk_plan_nn_fp8(M, K, N, tb=spec.get('tb'), fold=1)
    if pl is None or pl['sbuf'] > fp8nnk_SBUF_BUDGET:
        return None
    s_a = fp8nnk_resolve_ascale(dy2d, w, 4, key)
    wq = fp8nnk_quantised_weight(w, spec['wscale'], False, key)
    if wq is None:
        return None
    fmt = spec['afmt']
    aq = fp8nnk_conv_form(spec['signed'], s_a)
    bound = fp8nnk_clip_bound(s_a, fmt)
    hi = fp8nnk_fmt_max(fmt) if aq == 2 else bound * s_a
    try:
        out = fp8nnk__launch(fp8nnk__mm_fp8_nn_relu2_kernel)(dy2d, wq, h2d, 3, int(pl['tb']), int(fp8nnk_GROUP), int(fp8nnk_EVAC), int(aq), int(fp8nnk_LOOP), int(fmt), float(s_a), float(bound), float(hi), float(1.0 / (s_a * spec['wscale'])))
    except Exception as err:
        fp8nnk__disable_fold_kernel('mm_fp8_nn_relu2 row4', err)
        return None
    fp8nnk__fold_kernel_fired(M, K, N, s_a, aq, pl)
    return out

class fp8nnk__ProjFn(torch.autograd.Function):
    """y = x @ W^T for the MLP c_proj (W [out, in], the plain nn.Linear layout), with the forward in
    fp8 when row 2 is selected, the dX in fp8 when row 4 is, and [R1 fp8 row 6, 2026-09-25] the dW
    in fp8 when row 6 is.  Everything else is train.py's `_LinearGemm` for `transposed=False`, line
    for line: the same saved tuple, the same `ctx.w.grad.addmm_(dy2d.t(), x2d)` in-place accumulate
    (now the None-fallback of row 6's `dw_matmul`, kept verbatim), the same `None` for the weight
    when the accumulate is active, the same `.view(*ctx.lead, ...)` on the way out.  With no row
    selected this class is never reached (proj_active is False), so it adds nothing to any other
    path -- and row 6's guard block refuses any configuration that selects row 6 without a row that
    routes the c_proj here."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, inplace: bool) -> torch.Tensor:
        x2d = x.reshape(-1, x.shape[-1])
        ctx.save_for_backward(x2d, w)
        ctx.w = w
        ctx.inplace = inplace
        ctx.lead = x.shape[:-1]
        ctx.link = getattr(x, '_r1_relu2_link', None) if fp8nnk_RELU2_FOLD_DX else None
        if fp8nnk_ROW2:
            y = fp8nnk_row_matmul(x2d, w, 2)
        else:
            y = fp8nnk__eager(x2d, w, True, 2)
        return y.view(*x.shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        global fp8nnk__FOLD_CALLS, fp8nnk__FOLD_LINK_MISSES
        x2d, w = ctx.saved_tensors
        dy2d = dy.reshape(-1, dy.shape[-1])
        if not dy2d.is_contiguous():
            dy2d = dy2d.contiguous()
        dx = None
        if ctx.link is not None and fp8nnk_ROW4:
            dx = fp8nnk__row4_matmul_fold(dy2d, w, x2d, ctx.w)
            if dx is not None:
                fp8nnk__FOLD_CALLS += 1
                ctx.link.folded = True
        if dx is None:
            if fp8nnk_ROW4:
                dx = fp8nnk_row_matmul(dy2d, w, 4, key=ctx.w)
            else:
                dx = torch.matmul(dy2d, w)
            if ctx.link is not None:
                fp8nnk__FOLD_CALLS += 1
                dx = fp8nnk__relu2_from_h(x2d, dx)
                ctx.link.folded = True
            elif fp8nnk_RELU2_FOLD_DX:
                fp8nnk__FOLD_LINK_MISSES += 1
                if fp8nnk__FOLD_LINK_MISSES == 1:
                    print('[nki_gemm_fp8_nn] R1_RELU2_FOLD_DX: _ProjFn.backward found no c_fc link on its saved input; the ReLU^2 backward is NOT folded at this site', flush=True)
        dw_q = fp8nnk_dw_matmul(dy2d, x2d, 6, key=ctx.w)
        if ctx.inplace:
            if dw_q is not None:
                ctx.w.grad.add_(dw_q)
            else:
                ctx.w.grad.addmm_(dy2d.t(), x2d)
            dw = None
        else:
            dw = dw_q if dw_q is not None else torch.matmul(dy2d.t(), x2d)
        return (dx.view(*ctx.lead, dx.shape[-1]), dw, None)

def fp8nnk_c_proj(x: torch.Tensor, w: torch.Tensor, inplace: bool) -> torch.Tensor:
    """The MLP c_proj forward for any leading shape (w [out, in]); with inplace=True the dW goes
    straight into w.grad, exactly as train.py's `_LinearGemm` does."""
    return fp8nnk__ProjFn.apply(x, w, inplace)
try:
    fp8nnk__fp8_slice1 = _fp8_slice1_ns
except Exception:
    fp8nnk__fp8_slice1 = None
fp8nnk__SP_OK = {}
import inspect as _inspect
for fp8nnk__key, _mod, _fn in (('bf16', 'nki_gemm', getattr(fp8nnk__bf16, 'fused_fc_relu2_forward', None)), ('fp8', 'nki_gemm_fp8', getattr(fp8nnk__fp8_slice1, 'fused_fc_relu2_fp8_forward', None))):
    fp8nnk__SP_OK[fp8nnk__key] = _fn is not None and 'save_pre' in _inspect.signature(_fn).parameters
    if _fn is not None and (not fp8nnk__SP_OK[fp8nnk__key]):
        print(f"[nki_gemm_fp8_nn] NOTE: work/{_mod}.py predates this file -- {_fn.__name__} does not take save_pre. Calls will omit it (so `pre` IS written, i.e. today's behaviour). Sync that module and re-check its md5 if you want the relu2 fold.", flush=True)
del _inspect
if fp8nnk_RELU2_FROM_H:
    for fp8nnk__key, _mod in (('bf16', 'nki_gemm'), ('fp8', 'nki_gemm_fp8')):
        if not fp8nnk__SP_OK[fp8nnk__key]:
            fp8nnk_RELU2_FROM_H = False
            fp8nnk__RELU2_WHY.append(f'work/{_mod}.py is older than this file -- its fused forward does not take save_pre, so `pre` would still be written and (a) would delete nothing; sync it and re-check its md5')
            print('[nki_gemm_fp8_nn] IGNORING R1_RELU2_FROM_H: ' + fp8nnk__RELU2_WHY[-1], flush=True)
if (fp8nnk_RELU2_FROM_H or fp8nnk_RELU2_FOLD_DX) and fp8nnk__relu2 is not None and (not hasattr(fp8nnk__relu2, 'relu2_backward_from_h')):
    fp8nnk__RELU2_WHY.append('work/nki_relu2_opt.py is older than this file -- it has no relu2_backward_from_h, so the first folded backward would raise AttributeError mid-run; sync it (md5 762b23e9...) and re-check')
    fp8nnk_RELU2_FROM_H = False
    fp8nnk_RELU2_FOLD_DX = False
    print('[nki_gemm_fp8_nn] IGNORING R1_RELU2_FROM_H/R1_RELU2_FOLD_DX: ' + fp8nnk__RELU2_WHY[-1], flush=True)
fp8nnk__WPREP_CLASSES_RAW = 'r1,r3,r4'
fp8nnk_WPREP_CLASSES_ASKED = tuple(sorted({t for t in fp8nnk__WPREP_CLASSES_RAW.replace(' ', '').split(',') if t in ('r1', 'r3', 'r4')}))
fp8nnk__R2_WHY = []
fp8nnk_WPREP_CLASSES = ()
fp8nnk__cls = []
for _c in fp8nnk_WPREP_CLASSES_ASKED:
    fp8nnk__why = None
    if _c == 'r1':
        if fp8nnk__fp8_slice1 is None or not getattr(fp8nnk__fp8_slice1, 'USE_NKI_GEMM_FP8', False):
            fp8nnk__why = 'row 1 reads as off from here: R1_NKI_GEMM_FP8 unset, or a frozen slice-1 namespace that lacks USE_NKI_GEMM_FP8'
        elif fp8nnk__fp8_slice1.WCACHE == 'call' or fp8nnk__fp8_slice1.QDQ_ONLY:
            fp8nnk__why = 'slice 1 runs WCACHE=call or qdq, so it keeps no fp8 weight cache'
    elif _c == 'r3':
        if not fp8nnk_ROW3:
            fp8nnk__why = 'row 3 is not live: check R1_NKI_GEMM_FP8_NN and its ROWS, or qdq mode'
    elif _c == 'r4':
        if not fp8nnk_ROW4:
            fp8nnk__why = 'row 4 is not live: check R1_NKI_GEMM_FP8_NN and its ROWS, or qdq mode'
    if fp8nnk__why is None:
        fp8nnk__cls.append(_c)
    else:
        fp8nnk__R2_WHY.append(f'R1_NKI_GEMM_FP8_WPREP class {_c}: {fp8nnk__why}')
        print('[nki_gemm_fp8_nn] IGNORING ' + fp8nnk__R2_WHY[-1], flush=True)
fp8nnk_WPREP_CLASSES = tuple(fp8nnk__cls)
fp8nnk_WPREP = bool(fp8nnk_WPREP_CLASSES)
fp8nnk__WPREP_TAG = {}
if 'r1' in fp8nnk_WPREP_CLASSES:
    fp8nnk__WPREP_TAG['r1'] = ('slice1', f'f8@{fp8nnk__fp8_slice1.W_SCALE:g}')
if 'r3' in fp8nnk_WPREP_CLASSES:
    fp8nnk__WPREP_TAG['r3'] = ('nn', f'f8@{fp8nnk_W_SCALE_FC:g}T')
if 'r4' in fp8nnk_WPREP_CLASSES:
    fp8nnk__WPREP_TAG['r4'] = ('nn', f'f8@{fp8nnk_W_SCALE_PROJ:g}')
fp8nnk__WPREP_NN_TAGS = frozenset((t for o, t in fp8nnk__WPREP_TAG.values() if o == 'nn'))
fp8nnk__WPREP_CLASS_OF = {(o, t): c for c, (o, t) in fp8nnk__WPREP_TAG.items()}
fp8nnk__WPREP_REG = {}
fp8nnk__WPREP_STATS = {}
fp8nnk__WPREP_OFF = {}
fp8nnk__WPREP_SAID = set()

def fp8nnk__same_data(k: torch.Tensor, w: torch.Tensor) -> bool:
    """The cache key object and the tensor to quantise are the same bytes (always true on the live path, where
    both are the Parameter; checked because the batch reads the KEY's storage)."""
    return k is w or (k.data_ptr() == w.data_ptr() and tuple(k.shape) == tuple(w.shape) and (tuple(k.stride()) == tuple(w.stride())) and (k.dtype == w.dtype))

def fp8nnk__wprep_epoch(owner: str) -> int:
    return fp8nnk__fp8_slice1._STEP_EPOCH if owner == 'slice1' else fp8nnk__STEP_EPOCH

def fp8nnk__wprep_say(key, msg: str) -> None:
    if key not in fp8nnk__WPREP_SAID:
        fp8nnk__WPREP_SAID.add(key)
        print(msg, flush=True)

def fp8nnk__wprep_plan(members: dict, trans: bool):
    """The batch plan, or a reason string.  Pure host arithmetic on shapes, strides and storage offsets."""
    ts = list(members.values())
    if len(ts) < 2:
        return 'fewer than 2 weights in the class'
    t0 = ts[0]
    if t0.dim() != 2:
        return 'not 2-D'
    K, N = t0.shape
    sp = t0.untyped_storage().data_ptr()
    for t in ts:
        if t.dtype != torch.bfloat16 or t.dim() != 2 or tuple(t.shape) != (K, N) or (tuple(t.stride()) != (N, 1)) or (t.device != t0.device) or (t.untyped_storage().data_ptr() != sp):
            return 'members are not same-shape row-contiguous bf16 rows of one storage'
    if (N if trans else K) % fp8nnk__P:
        return f'quantised row count {(N if trans else K)} is not a multiple of {fp8nnk__P}'
    offs = sorted(((t.storage_offset(), i) for i, t in members.items()))
    d = offs[1][0] - offs[0][0]
    if d < K * N or d % N or any((offs[q + 1][0] - offs[q][0] != d for q in range(len(offs) - 1))):
        return 'storage offsets are not a uniform, whole-row, non-overlapping stride'
    return {'order': [i for _, i in offs], 'n': len(offs), 'K': K, 'N': N, 'd': d, 'off0': offs[0][0]}

def fp8nnk__wprep_view(plan: dict, members: dict):
    """(the CONTIGUOUS [(n-1)*dr + K, N] window over the members' rows, the members in window order, dr), or
    (None, members, dr) if a member is not where the window says.  One as_strided with the storage's own row
    stride: a plain row-contiguous tensor at the first member's storage offset -- the input class of every
    per-weight quantise today (each weight is itself such a row view of the stack)."""
    ts = [members[i] for i in plan['order']]
    K, N, dr = (plan['K'], plan['N'], plan['d'] // plan['N'])
    win = torch.as_strided(ts[0].detach(), ((plan['n'] - 1) * dr + K, N), (N, 1), plan['off0'])
    for j, t in enumerate(ts):
        if win[j * dr:j * dr + K].data_ptr() != t.data_ptr():
            return (None, ts, dr)
    return (win, ts, dr)

def fp8nnk__wprep_batch(plan: dict, members: dict, scale: float, trans: bool, clip: int, bound: float):
    """The launch itself: (members in order, their [K, N] -- or [N, K] under `trans` -- fp8 byte tensors), or None.

    natural: ONE _quant_w3_kernel over the window, member rows every dr rows.
    trans:   ONE batched bf16 transpose (torch; bitwise the per-weight `.t().contiguous()`s, stacked) into a fresh
             contiguous [n*N, K] tensor, then ONE _quant_w3_kernel over it with dr = N (members adjacent)."""
    win, ts, dr = fp8nnk__wprep_view(plan, members)
    if win is None:
        return None
    n, K, N = (plan['n'], plan['K'], plan['N'])
    if trans:
        v3 = torch.as_strided(ts[0].detach(), (n, K, N), (plan['d'], N, 1), plan['off0'])
        src = v3.transpose(1, 2).contiguous().view(n * N, K)
        rows, dr_ = (N, N)
    else:
        src, rows, dr_ = (win, K, dr)
    out = fp8nnk__launch(fp8nnk__quant_w3_kernel)(src, n, rows, dr_, float(scale), int(clip), float(bound))
    return (ts, [out[j * rows:(j + 1) * rows] for j in range(n)])

def fp8nnk__wprep_ref(owner: str, t: torch.Tensor, scale: float, trans: bool, clip: int, bound: float):
    """The owner's OWN per-weight product, exactly today's call (the self-check's reference)."""
    src = t.detach().t().contiguous() if trans else t.detach()
    if owner == 'slice1':
        return fp8nnk__fp8_slice1._launch(fp8nnk__fp8_slice1._quant_w_kernel)(src, float(scale), clip, bound)
    return fp8nnk__launch(fp8nnk__quant_w_nn_kernel)(src, float(scale), clip, bound)

def fp8nnk__fp8_ulp_gap(a_u8: torch.Tensor, b_u8: torch.Tensor) -> float:
    """Largest |a - b| in e4m3 ulps of max(|a|, |b|), on the host.  The legacy e4m3 and torch's e4m3fn agree on
    every byte the clamped quantiser can emit (<= 0x77 in magnitude), so the host decode is exact."""
    fa = a_u8.view(torch.float8_e4m3fn).float()
    fb = b_u8.view(torch.float8_e4m3fn).float()
    if torch.isnan(fa).any() or torch.isnan(fb).any():
        return float('inf')
    m = torch.maximum(fa.abs(), fb.abs())
    ulp = torch.where(m >= 2.0 ** (-6), torch.exp2(torch.floor(torch.log2(m.clamp_min(2.0 ** (-6)))) - 3.0), torch.full_like(m, 2.0 ** (-9)))
    return float(((fa - fb).abs() / ulp).max())

def fp8nnk__wprep_check(tag: str, owner: str, ts, outs, scale, trans, clip, bound):
    """First batch of a class, on device: every member's batched bytes against the owner's per-weight kernel."""
    refs = [fp8nnk__wprep_ref(owner, t, scale, trans, clip, bound) for t in ts]
    exact = total = 0
    worst = 0.0
    for o, r in zip(outs, refs):
        a = o.detach().cpu()
        b = r.detach().cpu()
        if tuple(a.shape) != tuple(b.shape):
            return (False, f'shape {tuple(a.shape)} vs per-weight {tuple(b.shape)} -> FAIL', refs)
        eq = a == b
        exact += int(eq.sum())
        total += a.numel()
        if not bool(eq.all()):
            worst = max(worst, fp8nnk__fp8_ulp_gap(a, b))
    ok = exact == total or worst <= 1.0
    line = f'{len(ts)} weights, bytes equal {exact}/{total}' + ('' if exact == total else f', worst gap {worst:.3g} ulp (<= 1 allowed: stochastic rounding)') + f' -> {('PASS' if ok else 'FAIL')}'
    return (ok, line, refs)

def fp8nnk__wprep_put(owner: str, t: torch.Tensor, tag: str, val) -> None:
    if owner == 'slice1':
        fp8nnk__fp8_slice1._wcache_put(t, tag, val)
    else:
        fp8nnk__cache_put(t, tag, val)

def fp8nnk__wprep_miss(owner: str, tag: str, k: torch.Tensor, scale: float, trans: bool, clip: int, bound: float):
    """A cache MISS on `k` for a batched class.  Returns k's fp8 bytes after filing EVERY member's, or None =
    "not batched now" (the caller runs today's per-weight path, which also files k's entry as today)."""
    rk = (owner, tag)
    if rk in fp8nnk__WPREP_OFF:
        return None
    ep = fp8nnk__wprep_epoch(owner)
    reg = fp8nnk__WPREP_REG.get(rk)
    if reg is None:
        reg = fp8nnk__WPREP_REG[rk] = {'members': {}, 'since': ep, 'plan': None, 'done': None}
    if reg['members'].get(id(k)) is not k:
        reg['members'][id(k)] = k
        reg['since'] = ep
        reg['plan'] = None
        return None
    if ep <= reg['since'] or reg['done'] == ep:
        return None
    if reg['plan'] is None:
        p = fp8nnk__wprep_plan(reg['members'], trans)
        reg['plan'] = p if isinstance(p, dict) else False
        if not isinstance(p, dict):
            fp8nnk__wprep_say(('plan',) + rk, f"[nki_gemm_fp8_nn] wprep {fp8nnk__WPREP_CLASS_OF.get(rk, tag)} ({tag}): NOT batched ({p}) -- today's per-weight path for this class")
    if reg['plan'] is False:
        return None
    reg['done'] = ep
    try:
        r = fp8nnk__wprep_batch(reg['plan'], reg['members'], scale, trans, clip, bound)
        if r is None:
            reg['plan'] = None
            return None
        ts, outs = r
    except Exception as err:
        fp8nnk__WPREP_OFF[rk] = f'batch launch: {type(err).__name__}: {err}'
        print(f"[nki_gemm_fp8_nn] wprep {fp8nnk__WPREP_CLASS_OF.get(rk, tag)} ({tag}) DISABLED after failure: {type(err).__name__}: {err} -- per-weight quantisation (today's path) for the rest of the run; this run is NOT a wprep measurement", flush=True)
        return None
    st = fp8nnk__WPREP_STATS.setdefault(rk, {'batches': 0, 'weights': 0, 'check': 'not run'})
    if st['check'] == 'not run':
        try:
            ok, line, refs = fp8nnk__wprep_check(tag, owner, ts, outs, scale, trans, clip, bound)
        except Exception as err:
            ok, line, refs = (False, f'ERROR {type(err).__name__}: {err} -> FAIL', None)
        st['check'] = line
        cls = fp8nnk__WPREP_CLASS_OF.get(rk, tag)
        if not ok:
            fp8nnk__WPREP_OFF[rk] = 'self-check ' + line
            print(f"[nki_gemm_fp8_nn] wprep {cls} ({tag}) self-check {line} -- class DISABLED, per-weight quantisation (today's path) for the rest of the run; this run is NOT a wprep measurement", flush=True)
            if refs is None:
                return None
            outs = refs
        else:
            print(f'[nki_gemm_fp8_nn] wprep {cls} ({tag}) FIRED: {len(ts)} weights {('[%d,%d] (W^T)' % (reg['plan']['N'], reg['plan']['K']) if trans else '[%d,%d]' % (reg['plan']['K'], reg['plan']['N']))} in ONE quantise launch{(' + ONE batched transpose' if trans else '')} (replaces {len(ts) * (2 if trans else 1)} launches); self-check {line}', flush=True)
    for t, o in zip(ts, outs):
        fp8nnk__wprep_put(owner, t, tag, o)
    if rk not in fp8nnk__WPREP_OFF:
        st['batches'] += 1
        st['weights'] += len(ts)
    return outs[[id(t) for t in ts].index(id(k))]

def fp8nnk__wprep_slice1(x2d: torch.Tensor, w: torch.Tensor) -> None:
    """r1: fill slice 1's cache for the whole class on its first miss of the step, before slice 1's forward asks.
    Mirrors slice 1's own gates (usable, plan) so a batch is never produced for a forward that would not use it."""
    s1 = fp8nnk__fp8_slice1
    tag = fp8nnk__WPREP_TAG['r1'][1]
    if s1._wcache_get(w, tag) is not None or not s1._usable(x2d, w):
        return
    M, K = x2d.shape
    K2, N = w.shape
    pl = s1.plan_fp8(M, K, N, tb=s1.TB, group=s1.GROUP, nprg=s1._nprg()) if K2 == K else None
    if pl is None or pl['sbuf'] > s1.SBUF_BUDGET:
        return
    fp8nnk__wprep_miss('slice1', tag, w, s1.W_SCALE, False, s1.clip_arg(), s1.clip_bound(s1.W_SCALE))

def fp8nnk__wprep_field() -> str:
    """config_string's `wprep=` value: `False`, `False(asked)`, or per class `r3:b<batches>/w<weights>/<verdict>`."""
    if not fp8nnk_WPREP:
        return 'False(asked)'
    out = []
    for c, (o, t) in sorted(fp8nnk__WPREP_TAG.items()):
        st = fp8nnk__WPREP_STATS.get((o, t), {'batches': 0, 'weights': 0, 'check': 'not run'})
        v = st['check'].split(' -> ')[-1] if ' -> ' in st['check'] else st['check'].replace(' ', '_')
        out.append(f'{c}:b{st['batches']}/w{st['weights']}/{('OFF' if (o, t) in fp8nnk__WPREP_OFF else v)}')
    miss = [c for c in fp8nnk_WPREP_CLASSES_ASKED if c not in fp8nnk__WPREP_TAG]
    return ','.join(out) + (f'(asked:{','.join(miss)})' if miss else '')

def fp8nnk_wprep_counters() -> dict:
    return {c: dict(fp8nnk__WPREP_STATS.get((o, t), {'batches': 0, 'weights': 0, 'check': 'not run'}), off=fp8nnk__WPREP_OFF.get((o, t))) for c, (o, t) in fp8nnk__WPREP_TAG.items()}

def fp8nnk__fc_forward(x2d: torch.Tensor, w: torch.Tensor, save_pre: int=1):
    """(h, pre) for c_fc + ReLU^2.  Row 1 is NOT this module's business: if slice 1 has it on we call
    slice 1's forward, so row 1 behaves exactly as it does under R1_NKI_GEMM_FP8 alone; otherwise we
    call today's bf16 fused forward.  Either way the tuple contract is nki_gemm.FcRelu2Fn's.

    [R1 relu2fold a] save_pre=0 asks the fused kernel to skip the [M, N] `pre` store; the second
    element of the returned tuple is then meaningless and the caller must drop it.

    BOTH branches take it, and that is the whole point.  Every candidate since 16 launches with
    kernel.  A save_pre plumbed only into nki_gemm.fused_fc_relu2_forward would have deleted exactly
    zero bytes on the live recipe while passing every check available off-chip -- CPU takes neither
    kernel -- which is the silent-no-op shape this project has shipped three times."""
    if fp8nnk__fp8_slice1 is not None and getattr(fp8nnk__fp8_slice1, 'USE_NKI_GEMM_FP8', False):
        if fp8nnk_WPREP and 'r1' in fp8nnk__WPREP_TAG:
            fp8nnk__wprep_slice1(x2d, w)
        if fp8nnk__SP_OK['fp8']:
            return fp8nnk__fp8_slice1.fused_fc_relu2_fp8_forward(x2d, w, save_pre=save_pre)
        assert save_pre, 'save_pre=0 requested but work/nki_gemm_fp8.py does not support it'
        return fp8nnk__fp8_slice1.fused_fc_relu2_fp8_forward(x2d, w)
    if fp8nnk__bf16 is not None:
        if fp8nnk__SP_OK['bf16']:
            return fp8nnk__bf16.fused_fc_relu2_forward(x2d, w, save_pre=save_pre)
        assert save_pre, 'save_pre=0 requested but work/nki_gemm.py does not support it'
        return fp8nnk__bf16.fused_fc_relu2_forward(x2d, w)
    pre = torch.matmul(x2d, w)
    return (torch.nn.functional.relu(pre).square(), pre)
fp8nnk__BF16_BACKWARD = None
if fp8nnk__bf16 is not None:
    fp8nnk__BF16_BACKWARD = fp8nnk__bf16.FcRelu2Fn.backward

def fp8nnk__fc_backward(ctx, dh, dpre_ready: bool=False):
    """Today's c_fc backward with ONE line changed: dX through the fp8 kernel.

    [R1 relu2fold] the saved third tensor is `pre` by default and `h` under R1_RELU2_FROM_H, and the
    three ways dpre can be obtained are ordered so that the default text is untouched:
      dpre_ready  -- _ProjFn.backward already applied 2*sqrt(h) (deletion b); dh IS dpre
      RELU2_FROM_H -- 2*sqrt(h) here, from the forward output (deletion a)
      otherwise    -- today's ReLU^2 backward kernel on the saved bf16 pre.

    When row 3 is off this function is never entered -- `FcRelu2NNFn.backward` delegates to
    `nki_gemm.FcRelu2Fn.backward`, the same function object slice 1 uses, so the "the backward IS
    today's code, not a copy of it" property survives every configuration that does not enable
    row 3.  When row 3 IS on, only `dx` changes: `dpre` is still today's ReLU^2 backward kernel on
    the saved bf16 `pre`, and `dW` is still the same `addmm_` (or the same class-C primitive)."""
    x2d, w, pre = ctx.saved_tensors
    dh = dh.contiguous()
    if dpre_ready:
        dpre = dh
    elif fp8nnk_RELU2_FROM_H:
        dpre = fp8nnk__relu2_from_h(pre, dh)
    elif fp8nnk__bf16 is not None and fp8nnk__bf16._relu2_backward is not None:
        dpre = fp8nnk__bf16._relu2_backward(pre, dh)
    elif fp8nnk__bf16 is not None:
        dpre = fp8nnk__bf16.eager_relu2_backward(pre, dh)
    else:
        dpre = dh * (2.0 * torch.nn.functional.relu(pre))
    if fp8nnk_ROW3:
        dx = fp8nnk_row_matmul(dpre, w, 3, key=ctx.w)
    else:
        dx = fp8nnk__eager(dpre, w, True, 3)
    dw = fp8nnk_dw_matmul(x2d, dpre, 5, key=ctx.w)
    if ctx.inplace:
        if dw is not None:
            ctx.w.grad.add_(dw)
        else:
            ctx.w.grad.addmm_(x2d.t(), dpre)
        return (dx, None, None)
    if dw is not None:
        return (dx, dw, None)
    if fp8nnk__bf16 is not None:
        return (dx, fp8nnk__bf16.nki_matmul(x2d, dpre, trans_a=True), None)
    return (dx, torch.matmul(x2d.t(), dpre), None)

class fp8nnk_FcRelu2NNFn(torch.autograd.Function):
    """h = relu(x2d @ W)^2 for the LinearT c_fc weight W [in, out].  Forward: slice 1's fp8 forward
    when R1_NKI_GEMM_FP8 has row 1 on, today's bf16 fused kernel otherwise.  Backward: today's
    function object when row 3 is off, `_fc_backward` (today's code with the dX line in fp8) when it
    is on.  The saved tuple and ctx fields are nki_gemm.FcRelu2Fn's contract exactly -- (x2d, w, pre)
    saved, ctx.w the Parameter, ctx.inplace the accumulate flag, a 3-tuple returned."""

    @staticmethod
    def forward(ctx, x2d, w, inplace, link=None):
        if fp8nnk_RELU2_FROM_H:
            out = fp8nnk__fc_forward(x2d, w, save_pre=0)
            h = out[0] if isinstance(out, (tuple, list)) else out
            ctx.save_for_backward(x2d, w, h)
        else:
            h, pre = fp8nnk__fc_forward(x2d, w)
            ctx.save_for_backward(x2d, w, pre)
        ctx.w = w
        ctx.inplace = inplace
        ctx.link = link
        return h

    @staticmethod
    def backward(ctx, dh):
        return fp8nnk_FcRelu2NNFn._grads(ctx, dh) + (None,)

    @staticmethod
    def _grads(ctx, dh):
        link = ctx.link
        if link is not None and link.folded:
            return fp8nnk__fc_backward(ctx, dh, dpre_ready=True)
        if not fp8nnk_ROW3 and (not fp8nnk_ROW5):
            if fp8nnk__BF16_BACKWARD is not None:
                return fp8nnk__BF16_BACKWARD(ctx, dh)
        return fp8nnk__fc_backward(ctx, dh)

def fp8nnk_fused_fc_relu2_nn(x: torch.Tensor, w: torch.Tensor, inplace: bool=False) -> torch.Tensor:
    """relu(x @ w)^2 for any leading shape (w [in, out]); with inplace=True the c_fc dW goes straight
    into w.grad, exactly as the bf16 and slice-1 paths do."""
    K = x.shape[-1]
    x2d = x.reshape(-1, K)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    link = fp8nnk__RELU2_LINK() if fp8nnk_RELU2_FOLD_DX else None
    h = fp8nnk_FcRelu2NNFn.apply(x2d, w, inplace, link)
    hv = h.view(*x.shape[:-1], w.shape[1])
    if link is not None:
        hv._r1_relu2_link = link
    return hv

def fp8nnk_config_string() -> str:
    """train.py:3137 prints this once at startup and it is the only in-log evidence of what this lane
    is doing, so it leads with `live=`: the rows that will actually reach a kernel, after the master switch."""
    rows = ','.join((str(r) for r in fp8nnk_ROWS))
    dw_rows = ','.join((str(r) for r in fp8nnk_DW_ROWS))
    live = ['mlp:' + rows] + ['dw:' + dw_rows]
    gs = f'{fp8nnk_G_SCALE:g}'
    if fp8nnk__GCACHE:
        _gi = sorted(fp8nnk__GCACHE.items(), key=lambda kv: (kv[0][0], str(kv[0][1:])))
        gs += '[' + ','.join((f'{k[0]}:{s:g}' for k, s in _gi[:3])) + '..]'
    return f'live={(' '.join(live) if live else 'none')} mode={fp8nnk_MODE!r} rows={rows} eval={fp8nnk_EVAL_MODE} h_scale={fp8nnk_H_SCALE:g} g_scale={gs} gfmt={'e5m2'} w_fc={fp8nnk_W_SCALE_FC:g} w_proj={fp8nnk_W_SCALE_PROJ:g} clip={'on'} r2cap={fp8nnk__ROW2_CAP}:{fp8nnk__ROW2_CAP_N} dw={fp8nnk_DW_MODE!r} dw_rows={dw_rows} dw_eval={fp8nnk_DW_EVAL_MODE} a_dw={fp8nnk_A_SCALE_DW:g} dw_fired={fp8nnk_dw_counters()['calls'] or '{}'} dw_skips={fp8nnk_dw_counters()['skips'] or '{}'} dw_tn={('-' if _tn is None else 'nki=%s disabled=%s' % (_tn._HAS_NKI, _tn._KERNEL_DISABLED))} wcache={fp8nnk_WCACHE} wqt={fp8nnk_WQT} tb_a={fp8nnk_TB_A} tb_b={fp8nnk_TB_B} group={fp8nnk_GROUP} evac={fp8nnk_EVAC} loop={fp8nnk_LOOP} relu2_from_h={fp8nnk_RELU2_FROM_H}{('(asked)' if not fp8nnk_RELU2_FROM_H else '')} relu2_fold_dx={fp8nnk_RELU2_FOLD_DX}{('(asked)' if not fp8nnk_RELU2_FOLD_DX else '')} relu2_fold_kernel={fp8nnk__FOLD_KERNEL_CALLS}{('(DISABLED)' if fp8nnk__FOLD_KERNEL_DISABLED else '')} wprep={fp8nnk__wprep_field()} lnc={fp8nnk__LNC} shard={fp8nnk__SHARD} nki={fp8nnk__HAS_NKI} disabled={fp8nnk__KERNEL_DISABLED}' + (f' err={fp8nnk__LAST_KERNEL_ERROR}' if fp8nnk__LAST_KERNEL_ERROR else '') + (f' fold_err={fp8nnk__FOLD_LAST_ERROR}' if fp8nnk__FOLD_LAST_ERROR else '')

class fp8nnk__ApiNamespace:
    """Namespace object for the freeze.  train.py uses this module as a NAMESPACE, and
    work/freeze_candidate.py inlines a module's names under a prefix -- which would leave the module
    name itself undefined.  The freeze binds it to this object, as candidate 13 bound
    nki_optim = optk_API.  Attributes are assigned with `API.x = x` and NOT declared in the class
    body: the freeze's token renamer rewrites bare names, so a class-body `x = staticmethod(x)`
    becomes `tag_x = staticmethod(tag_x)` and the attribute silently changes name.  (Found 2026-09-22,
    when the first candidate-16 freeze produced an API object with no `proj_active`.)"""
fp8nnk_API = fp8nnk__ApiNamespace()
fp8nnk_API.begin_step = fp8nnk_begin_step
fp8nnk_API.active = fp8nnk_active
fp8nnk_API.proj_active = fp8nnk_proj_active
fp8nnk_API.fc_active = fp8nnk_fc_active
fp8nnk_API.c_proj = fp8nnk_c_proj
fp8nnk_API.fused_fc_relu2_nn = fp8nnk_fused_fc_relu2_nn
fp8nnk_API.config_string = fp8nnk_config_string
fp8nnk_API.row2_cap = fp8nnk_row2_cap
fp8nnk_API.USE_NKI_GEMM_FP8_NN = fp8nnk_USE_NKI_GEMM_FP8_NN
fp8nnk_API.MODE = fp8nnk_MODE
fp8nnk_API.ROWS = fp8nnk_ROWS
fp8nnk_API.fold_counters = fp8nnk_fold_counters
fp8nnk_API.RELU2_FROM_H = fp8nnk_RELU2_FROM_H
fp8nnk_API.RELU2_FOLD_DX = fp8nnk_RELU2_FOLD_DX
fp8nnk_API.dw_active = fp8nnk_dw_active
fp8nnk_API.dw_matmul = fp8nnk_dw_matmul
fp8nnk_API.dw_counters = fp8nnk_dw_counters
fp8nnk_API.USE_NKI_GEMM_FP8_DW = fp8nnk_USE_NKI_GEMM_FP8_DW
fp8nnk_API.DW_MODE = fp8nnk_DW_MODE
fp8nnk_API.DW_ROWS = fp8nnk_DW_ROWS
fp8nnk_API.WPREP = fp8nnk_WPREP
fp8nnk_API.WPREP_CLASSES = fp8nnk_WPREP_CLASSES
fp8nnk_API.wprep_counters = fp8nnk_wprep_counters
nki_gemm_fp8_nn = fp8nnk_API
USE_NKI_GEMM_FP8_NN = fp8nnk_USE_NKI_GEMM_FP8_NN
DW_FP8_ENV = '1' != '0'
USE_NKI_GEMM_FP8_DW = DW_FP8_ENV and getattr(nki_gemm_fp8_nn, 'USE_NKI_GEMM_FP8_DW', False)
USE_NKI_GEMM_FP8_ANY = USE_NKI_GEMM_FP8_NN or USE_NKI_GEMM_FP8_DW

def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """relu(x)**2, via the optimised NKI kernel (R1_NKI_RELU2_OPT=1), else the example kernel (--nki-relu2), else eager."""
    return _relu_squared_opt(x)
    if USE_NKI_RELU2 and _HAS_NKI and (x.device.type == 'neuron') and (x.numel() // x.shape[-1] % _NKI_PARTITION == 0):
        return _ReluSquaredNKI.apply(x)
    return F.relu(x).square()
"Fused RoPE + RMSNorm (the attention block's QK-norm) as ONE NKI launch forward and ONE backward.\n\nOptimised variant of work/nki_rope_norm.py (same public API, same guards, same eager fallback, same emulation).\n\nPublic entry point\n------------------\n    y = rope_norm(x, cos, sin)\n\n    x        [B, T, H, D] with D == 128 (a view of the q / k projection output), bf16 or fp32\n    cos, sin train.py's RoPE tables, [1, T, 1, D // 2] (anything that reshapes to [T, D // 2] works)\n    y        [B, H, T, D]  ==  norm(apply_rotary_emb(x, cos, sin)).transpose(1, 2)\n\nWhat changed vs the original kernel (see NOTES.md in this folder for the measurements)\n-------------------------------------------------------------------------------------\n* Tile = [P=128 partitions, K, H, D]: partition p holds K consecutive tokens with ALL H heads, so the input\n  DMA is contiguous (K*H*D*2 bytes per partition) and every arithmetic instruction works on K*H*D elements\n  per partition (fewer, larger instructions).  cos/sin are [P, K, 1, D/2] tiles applied through a stride-0\n  `broadcast` view over H; the per-row 1/rms is a [P, K, H, 1] tile applied through a broadcast view over D,\n  so the scale is ONE instruction instead of K (or K*H) `tensor_scalar`s.\n* The (b, c) tile loop is sharded over the two physical cores of an LNC=2 logical core: the kernel is\n  launched as `kernel[2](...)` and each program takes the tiles `i % num_programs == program_id`.\n* No bf16 -> fp32 tensor_copy: the engines upcast bf16 operands to fp32 internally (documented), so\n* The square of the row norm can run on ScalarE (`R1_NKI_ROPE_NORM_SQ=act`, default) so the VectorE\n  chain is shorter; `vec` uses tensor_tensor as before.\n\nbf16/fp32, T <= 128 or T % 128 == 0; a kernel failure prints once, disables the kernel for the process and\nanswers with the eager math.  Numerics: fp32 math in SBUF, one rounding on the store; RMS eps = 2**-23.\n"
import os
import torch
import torch.nn.functional as F
ropek__NKI_PARTITION = 128
ropek_HEAD_DIM = 128
ropek_RMS_EPS = float(2.0 ** (-23))
ropek__MAX_K = 8
ropek__MAX_ROWS = 32
ropek__ROT_DTYPE = 'input'
ropek__BWD_ROT_DTYPE = 'input'
ropek__GR_OUT_DTYPE = 'input'
ropek__LNC = int(os.environ.get('NEURON_LOGICAL_NC_CONFIG', '1'))
ropek__SCALE_MODE = 'bcast'
ropek_USE_NKI_ROPE_NORM = True
ropek__KERNEL_DISABLED = False
ropek__LAST_KERNEL_ERROR = None

def ropek_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))

def ropek_apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = (x[..., :d], x[..., d:])
    y1 = x1 * cos + x2 * sin
    y2 = x1 * -sin + x2 * cos
    return torch.cat((y1, y2), dim=-1)

def ropek_eager_rope_norm(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return ropek_norm(ropek_apply_rotary_emb(x, cos, sin)).transpose(1, 2)

def ropek__plan(T: int, H: int=8):
    """(C, P, K) with T == C*P*K, P <= 128, K <= min(_MAX_K, _MAX_ROWS // H), or None when T is not tileable."""
    if T <= 0:
        return None
    if T <= ropek__NKI_PARTITION:
        return (1, T, 1)
    if T % ropek__NKI_PARTITION:
        return None
    n = T // ropek__NKI_PARTITION
    kmax = max(1, min(ropek__MAX_K, ropek__MAX_ROWS // max(H, 1), n))
    k = max((d for d in range(1, kmax + 1) if n % d == 0))
    return (n // k, ropek__NKI_PARTITION, k)
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    def ropek__pick(mode, in_dtype):
        """Temporaries dtype for a knob: 'input' -> the kernel input dtype, else the named dtype."""
        if mode == 'input':
            return in_dtype
        return nl.bfloat16 if mode == 'bf16' else nl.float32
    ropek__GR_DT = nl.float32
    ropek__ADD_ENG = nisa.engine.unknown

    def ropek__dma(dst, src):
        """dma_copy with the descriptor-generation mode knob (all addresses here are static)."""
        nisa.dma_copy(dst=dst, src=src)

    def ropek__add(dst, a, b, op):
        nisa.tensor_tensor(dst=dst, data1=a, data2=b, op=op, engine=ropek__ADD_ENG)

    def ropek__shard():
        """(program_id, num_programs); (0, 1) without a launch grid or with sharding off."""
        if nl.program_ndim() > 0:
            return (nl.program_id(0), nl.num_programs(0))
        return (0, 1)

    def ropek__ranges(B, C):
        """Tile ranges of this program: shard the T chunks c (each program loads its own tables and keeps every
        batch element, so B == 1 still uses both cores); shard the batch only when there are fewer chunks than
        programs."""
        pid, nprg = ropek__shard()
        if C >= nprg:
            return (range(pid, C, nprg), range(B))
        return (range(C), range(pid, B, nprg))

    def ropek__load_tables(cos5, sin5, c, P, K, Dh, H):
        """cos/sin for chunk c as [P, K, 1, Dh] SBUF tiles, returned as broadcast views [P, K, H, Dh]."""
        cosb = nl.ndarray((P, K, 1, Dh), dtype=cos5.dtype, buffer=nl.sbuf)
        sinb = nl.ndarray((P, K, 1, Dh), dtype=sin5.dtype, buffer=nl.sbuf)
        ropek__dma(dst=cosb, src=cos5[c, :, :, :, :])
        ropek__dma(dst=sinb, src=sin5[c, :, :, :, :])
        return (cosb.broadcast(2, H), sinb.broadcast(2, H))
        cosH = nl.ndarray((P, K, H, Dh), dtype=cosb.dtype, buffer=nl.sbuf)
        sinH = nl.ndarray((P, K, H, Dh), dtype=sinb.dtype, buffer=nl.sbuf)
        for h in range(H):
            nisa.tensor_copy(dst=cosH[:, :, h, :], src=cosb[:, :, 0, :])
            nisa.tensor_copy(dst=sinH[:, :, h, :], src=sinb[:, :, 0, :])
        return (cosH, sinH)

    def ropek__scale_rows(dst, src, vec, K, H, op=None):
        """dst[p,k,h,:] = src[p,k,h,:] * vec[p,k,h,0]: one broadcast VectorE tensor_tensor (bcast), K*H ScalarE
        activation(copy, scale=vec row) instructions (act), or K*H VectorE/GpSimd tensor_scalars (ts)."""
        mode = ropek__SCALE_MODE
        if mode == 'bcast':
            nisa.tensor_tensor(dst=dst, data1=src, data2=vec.broadcast(3, dst.shape[3]), op=nl.multiply)
        elif mode == 'act':
            for k in range(K):
                for h in range(H):
                    nisa.activation(dst=dst[:, k, h, :], op=nl.copy, data=src[:, k, h, :], scale=vec[:, k, h, :])
        else:
            for k in range(K):
                for h in range(H):
                    nisa.tensor_scalar(dst=dst[:, k, h, :], data=src[:, k, h, :], op0=nl.multiply, operand0=vec[:, k, h, :])

    def ropek__row_sums(dst, sq, scratch, K, H):
        """dst[P,K,H,1] = sum_d sq: one VectorE tensor_reduce (vec) or K*H ScalarE activation_reduce (act)."""
        nisa.tensor_reduce(dst=dst, op=nl.add, data=sq, axis=3, keepdims=True)

    def ropek__maybe_cast(tile, shape):
        return tile

    def ropek__mul(dst, a, b, which):
        """Table multiply of the rotation; which in {0,1}: with MUL_ENGINE=gpsimd the odd ones go to GpSimd."""
        nisa.tensor_tensor(dst=dst, data1=a, data2=b, op=nl.multiply)

    def ropek__rotate(dst, src, cb, sb, tmp_a, tmp_b, Dh, transpose):
        """dst = R(src) or R^T(src) on [P, K, H, 2*Dh] tiles; cb/sb are [P, K, H, Dh] (broadcast) views."""
        D = 2 * Dh
        s1, s2 = (src[:, :, :, 0:Dh], src[:, :, :, Dh:D])
        d1, d2 = (dst[:, :, :, 0:Dh], dst[:, :, :, Dh:D])
        ropek__mul(tmp_a, s1, cb, 0)
        ropek__mul(tmp_b, s2, sb, 1)
        ropek__add(d1, tmp_a, tmp_b, nl.subtract if transpose else nl.add)
        ropek__mul(tmp_a, s2, cb, 0)
        ropek__mul(tmp_b, s1, sb, 1)
        ropek__add(d2, tmp_a, tmp_b, nl.add if transpose else nl.subtract)

    def ropek__inv_rms(r, sq, P, K, H, D):
        """inv[P, K, H, 1] = rsqrt(mean_d r^2 + eps) in fp32; sq is fp32 [P, K, H, D] scratch."""
        ss = nl.ndarray((P, K, H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=r)
        nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=3, keepdims=True)
        inv = nl.ndarray((P, K, H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ss, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=ropek_RMS_EPS)
        nisa.tensor_scalar(dst=inv, data=ss, op0=nl.rsqrt, operand0=0.0)
        return inv

    @nki.jit
    def ropek__rope_norm_fwd_kernel(x6, cos5, sin5):
        """y6[B, H, C, P, K, D] = RMSNorm(RoPE(x6[B, C, P, K, H, D])); tables cos5 / sin5 [C, P, K, 1, D/2]."""
        B, C, P, K, H, D = x6.shape
        Dh = D // 2
        f32 = nl.float32
        y6 = nl.ndarray((B, H, C, P, K, D), dtype=x6.dtype, buffer=nl.shared_hbm)
        c_range, b_range = ropek__ranges(B, C)
        for c in c_range:
            cb, sb = ropek__load_tables(cos5, sin5, c, P, K, Dh, H)
            for b in b_range:
                ropek__fwd_tile(x6, y6, cb, sb, b, c, P, K, H, D, Dh)
        return y6

    def ropek__fwd_tile(x6, y6, cb, sb, b, c, P, K, H, D, Dh):
        f32 = nl.float32
        xb = nl.ndarray((P, K, H, D), dtype=x6.dtype, buffer=nl.sbuf)
        ropek__dma(dst=xb, src=x6[b, c, :, :, :, :])
        xb = ropek__maybe_cast(xb, (P, K, H, D))
        rdt = ropek__pick(ropek__ROT_DTYPE, x6.dtype)
        tmp_a = nl.ndarray((P, K, H, Dh), dtype=rdt, buffer=nl.sbuf)
        tmp_b = nl.ndarray((P, K, H, Dh), dtype=rdt, buffer=nl.sbuf)
        r = nl.ndarray((P, K, H, D), dtype=rdt, buffer=nl.sbuf)
        ropek__rotate(r, xb, cb, sb, tmp_a, tmp_b, Dh, transpose=False)
        sq = nl.ndarray((P, K, H, D), dtype=f32, buffer=nl.sbuf)
        inv = ropek__inv_rms(r, sq, P, K, H, D)
        yb = nl.ndarray((P, K, H, D), dtype=x6.dtype, buffer=nl.sbuf)
        ropek__scale_rows(yb, r, inv, K, H)
        for h in range(H):
            ropek__dma(dst=y6[b, h, c, :, :, :], src=yb[:, :, h, :])

    @nki.jit
    def ropek__rope_norm_bwd_kernel(x6, cos5, sin5, g6):
        """gx6[B, C, P, K, H, D] = R^T( J_rmsnorm^T(g) ), r = R(x) recomputed; g6 is [B, H, C, P, K, D]."""
        B, C, P, K, H, D = x6.shape
        Dh = D // 2
        f32 = nl.float32
        gx6 = nl.ndarray((B, C, P, K, H, D), dtype=x6.dtype, buffer=nl.shared_hbm)
        c_range, b_range = ropek__ranges(B, C)
        for c in c_range:
            cb, sb = ropek__load_tables(cos5, sin5, c, P, K, Dh, H)
            for b in b_range:
                ropek__bwd_tile(x6, g6, gx6, cb, sb, b, c, P, K, H, D, Dh)
        return gx6

    def ropek__bwd_tile(x6, g6, gx6, cb, sb, b, c, P, K, H, D, Dh):
        f32 = nl.float32
        xb = nl.ndarray((P, K, H, D), dtype=x6.dtype, buffer=nl.sbuf)
        ropek__dma(dst=xb, src=x6[b, c, :, :, :, :])
        xb = ropek__maybe_cast(xb, (P, K, H, D))
        gf = nl.ndarray((P, K, H, D), dtype=g6.dtype, buffer=nl.sbuf)
        for h in range(H):
            ropek__dma(dst=gf[:, :, h, :], src=g6[b, h, c, :, :, :])
        gf = ropek__maybe_cast(gf, (P, K, H, D))
        rdt = ropek__pick(ropek__BWD_ROT_DTYPE, x6.dtype)
        tmp_a = nl.ndarray((P, K, H, Dh), dtype=rdt, buffer=nl.sbuf)
        tmp_b = nl.ndarray((P, K, H, Dh), dtype=rdt, buffer=nl.sbuf)
        r = nl.ndarray((P, K, H, D), dtype=rdt, buffer=nl.sbuf)
        ropek__rotate(r, xb, cb, sb, tmp_a, tmp_b, Dh, transpose=False)
        sq = nl.ndarray((P, K, H, D), dtype=f32, buffer=nl.sbuf)
        inv = ropek__inv_rms(r, sq, P, K, H, D)
        nisa.tensor_tensor(dst=sq, data1=gf, data2=r, op=nl.multiply)
        dot = nl.ndarray((P, K, H, 1), dtype=f32, buffer=nl.sbuf)
        u = nl.ndarray((P, K, H, D), dtype=ropek__GR_DT, buffer=nl.sbuf)
        ropek__row_sums(dot, sq, sq, K, H)
        coef = nl.ndarray((P, K, H, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=coef, data1=inv, data2=inv, op=nl.multiply)
        nisa.tensor_tensor(dst=coef, data1=coef, data2=dot, op=nl.multiply)
        nisa.tensor_scalar(dst=coef, data=coef, op0=nl.multiply, operand0=1.0 / D)
        ropek__scale_rows(u, r, coef, K, H)
        ropek__add(u, gf, u, nl.subtract)
        gr = nl.ndarray((P, K, H, D), dtype=ropek__pick(ropek__GR_OUT_DTYPE, x6.dtype), buffer=nl.sbuf)
        ropek__scale_rows(gr, u, inv, K, H)
        gxb = nl.ndarray((P, K, H, D), dtype=x6.dtype, buffer=nl.sbuf)
        ropek__rotate(gxb, gr, cb, sb, tmp_a, tmp_b, Dh, transpose=True)
        ropek__dma(dst=gx6[b, c, :, :, :, :], src=gxb)
    ropek__HAS_NKI = True
except Exception:
    ropek__HAS_NKI = False

def ropek__disable_kernel(where: str, err: BaseException) -> None:
    global ropek__KERNEL_DISABLED, ropek__LAST_KERNEL_ERROR
    if not ropek__KERNEL_DISABLED:
        ropek__KERNEL_DISABLED = True
        ropek__LAST_KERNEL_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[nki_rope_norm_opt] kernel disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)

def ropek__launch(kernel):
    """kernel[2] under an LNC=2 runtime (both physical cores run the sharded tile loop), plain call otherwise."""
    return kernel[ropek__LNC] if ropek__LNC == 2 else kernel

def ropek__tables_2d(cos: torch.Tensor, sin: torch.Tensor, T: int, Dh: int, dtype: torch.dtype):
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
    return (c2, s2)

class ropek_RopeNormFn(torch.autograd.Function):
    """y = RMSNorm(RoPE(x)) in [B, H, T, D] layout via the NKI kernels; eager math if a kernel raises."""

    @staticmethod
    def forward(ctx, x, cos2, sin2, plan):
        B, T, H, D = x.shape
        C, P, K = plan
        ctx.save_for_backward(x, cos2, sin2)
        ctx.plan = plan
        try:
            x6 = x.view(B, C, P, K, H, D)
            y6 = ropek__launch(ropek__rope_norm_fwd_kernel)(x6, cos2.view(C, P, K, 1, D // 2), sin2.view(C, P, K, 1, D // 2))
            return y6.view(B, H, T, D)
        except Exception as err:
            ropek__disable_kernel('forward', err)
            with torch.no_grad():
                return ropek_eager_rope_norm(x, cos2.view(1, T, 1, D // 2), sin2.view(1, T, 1, D // 2)).contiguous()

    @staticmethod
    def backward(ctx, gy):
        x, cos2, sin2 = ctx.saved_tensors
        B, T, H, D = x.shape
        C, P, K = ctx.plan
        if not ropek__KERNEL_DISABLED:
            try:
                g6 = gy.contiguous().view(B, H, C, P, K, D)
                gx6 = ropek__launch(ropek__rope_norm_bwd_kernel)(x.view(B, C, P, K, H, D), cos2.view(C, P, K, 1, D // 2), sin2.view(C, P, K, 1, D // 2), g6)
                return (gx6.view(B, T, H, D), None, None, None)
            except Exception as err:
                ropek__disable_kernel('backward', err)
        with torch.enable_grad():
            xr = x.detach().requires_grad_(True)
            y = ropek_eager_rope_norm(xr, cos2.view(1, T, 1, D // 2), sin2.view(1, T, 1, D // 2))
            gx, = torch.autograd.grad(y, xr, gy)
        return (gx, None, None, None)

def ropek_rope_norm(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """norm(apply_rotary_emb(x, cos, sin)).transpose(1, 2) -- one NKI launch when possible, eager otherwise."""
    if x.dim() != 4:
        return ropek_eager_rope_norm(x, cos, sin)
    B, T, H, D = x.shape
    use_kernel = ropek__HAS_NKI and (not ropek__KERNEL_DISABLED) and (x.device.type == 'neuron') and (D == ropek_HEAD_DIM) and (x.dtype in (torch.bfloat16, torch.float32)) and (cos.numel() == T * (D // 2)) and (sin.numel() == T * (D // 2))
    plan = ropek__plan(T, H) if use_kernel else None
    if plan is None:
        return ropek_eager_rope_norm(x, cos, sin)
    if not (x.stride(3) == 1 and x.stride(2) == D and (x.stride(1) >= H * D)):
        x = x.contiguous()
    cos2, sin2 = ropek__tables_2d(cos, sin, T, D // 2, x.dtype)
    return ropek_RopeNormFn.apply(x, cos2, sin2, plan)
rope_norm = ropek_rope_norm
USE_NKI_ROPE_NORM = ropek_USE_NKI_ROPE_NORM
try:
    from nki_gain_norm import gain_residual_norm, USE_NKI_GAIN_NORM
except Exception:
    USE_NKI_GAIN_NORM = False
'Fused loss head: softcap + log-softmax + one-hot cross-entropy as ONE NKI launch forward and ONE backward.\n\nReplaces, in train.py\'s GPT.forward (CE_PATH == "static"):\n\n    logits = self.lm_head(x)[..., :V].float()                 # bf16 [B, T, V] -> fp32\n    logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)\n    logp   = F.log_softmax(logits.reshape(-1, V), dim=-1)\n    onehot = (targets.reshape(-1, 1) == torch.arange(V)).to(logp.dtype)\n    nll    = -(logp * onehot).sum(dim=-1)                      # [N]\n    loss   = nll.mean() | nll.sum() | nll.view_as(targets)\n\nwith\n\n    loss = fused_loss(self.lm_head(x), targets, reduction=loss_reduction, softcap=LOGIT_SOFTCAP)\n\nPublic entry point\n------------------\n    fused_loss(logits, targets, reduction="mean", softcap=15.0)\n        logits   [..., V] bf16 (or fp32), the RAW lm_head output (the softcap is fused in); V == 8192 (the padded\n                 vocab equals the tokenizer vocab here, so train.py\'s [..., :V] slice is the identity; asserted)\n        targets  [...] int64 or int32 token ids in [0, V); N = targets.numel() must be a multiple of 128\n        returns  0-dim fp32 for "mean" / "sum", fp32 [targets.shape] for "none" -- the same objects the eager chain\n                 returns.  Gradient w.r.t. `logits` in logits.dtype (bf16), as the eager chain\'s `.float()` gives.\n\nMath (per position i, c = softcap, fp32 inside the kernel)\n----------------------------------------------------------\n    t_v = tanh(x_iv / c);  s_v = c t_v  (|s_v| <= c)\n    z   = sum_v exp(s_v - c)                      fixed shift c instead of the row max: every term is a normal fp32\n                                                  number in [e^-2c, 1], so log-sum-exp keeps full fp32 relative\n                                                  precision and no max pass is needed\n    lz  = log z;   lse = c + lz\n    loss_i = lse - s_{t_i} = c + lz - c tanh(x_{i,t_i} / c)          (the kernel returns scale * loss_i)\n    d x_iv = grad_i * scale * (softmax_v - [v == t_i]) * (1 - t_v^2),   softmax_v = exp(c t_v - c - lz)\n\nKernel design (see NOTES.md for the measurements)\n-------------------------------------------------\n* Rows n = p * NT + i live on partition p, tile i: x is viewed as [128, NT, V] so tile i is the HBM slice\n  x3[:, i, :] -- one contiguous 2*V-byte (16 KiB) run per partition, 2 MiB per DMA -- and every per-row vector\n  (targets, log z, loss, grad rows) is a [128, NT] block loaded/stored ONCE per program, not per tile.\n* Forward per tile: ScalarE tanh (bf16 in, fp32 out), ScalarE exp with the row sum in the reduce registers,\n  GpSimd nc_n_gather for the target logit (one element per partition; no iota, no one-hot, no VectorE pass).\n  Epilogue on the [128, n] blocks: log, tanh of the target logits, one affine -> loss.  Saves ONLY log z per row.\n* Backward per tile: recompute t (ScalarE), p = exp(c t - c - lz) (ScalarE, per-row bias), q = onehot - p in one\n  VectorE scalar_tensor_tensor(equal, subtract) on an iota row generated once, g t^2 on ScalarE (square with a\n  per-row sqrt(g) scale), d = (g t^2 - g) * q in one VectorE pass straight into the bf16 output buffer.\n  Recompute beats saving the fp32 softcapped logits: that would add 2 x 512 MiB of HBM traffic per micro-step to\n  save one ~7 us ScalarE pass per tile.\n* kernel[2] launch under LNC=2: program p takes a contiguous half of the tiles (nl.program_id(0)); the two programs\n  write disjoint HBM regions.  A plain call (LNC=1) runs the whole loop on one core.\n\nR1_NKI_LOSS_HEAD_ANY_V=1), N % 128 == 0, bf16/fp32 logits; a kernel failure prints once, disables the kernel for\nthe process and answers with the eager chain (which is also what the wrapper returns whenever the kernel is not\napplicable, so the eval path and odd shapes are always the eager math).\n\nKnobs (env, read once at import; module-level constants because kernel bodies only see module globals):\nMeasured 2026-09-07 (NOTES.md section 7): fwd 1.21 ms / fwd+bwd 3.53 ms vs eager 34.1 / 57.0 ms at [16384, 8192].\n'
import os
import torch
import torch.nn.functional as F
lossk__P = 128
lossk_VOCAB = 8192
lossk_LOGIT_SOFTCAP = 15.0
lossk_USE_NKI_LOSS_HEAD = True
lossk__PREFETCH = min(1, 1)
lossk__NBUF = 2
lossk__LNC = int(os.environ.get('NEURON_LOGICAL_NC_CONFIG', '1'))
lossk__ANY_V = False
lossk__KERNEL_DISABLED = False
lossk__LAST_KERNEL_ERROR = None

def lossk_eager_loss_rows(x2: torch.Tensor, targets: torch.Tensor, softcap: float=lossk_LOGIT_SOFTCAP) -> torch.Tensor:
    """train.py's chain on [N, V] logits and [N] targets: per-position nll, fp32 [N]."""
    logits = x2.float()
    logits = softcap * torch.tanh(logits / softcap)
    vocab = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    onehot = (targets.reshape(-1, 1) == torch.arange(vocab, device=logits.device).view(1, -1)).to(logp.dtype)
    return -(logp * onehot).sum(dim=-1)

def lossk_eager_loss(logits: torch.Tensor, targets: torch.Tensor, reduction: str='mean', softcap: float=lossk_LOGIT_SOFTCAP) -> torch.Tensor:
    """The eager chain with train.py's reductions ("none" returns [targets.shape])."""
    nll = lossk_eager_loss_rows(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), softcap)
    if reduction == 'none':
        return nll.view_as(targets)
    return nll.sum() if reduction == 'sum' else nll.mean()
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    lossk__EXP_DT = nl.float32

    def lossk__dma(dst, src):
        nisa.dma_copy(dst=dst, src=src)

    def lossk__shard():
        """(program_id, num_programs); (0, 1) without a launch grid or with sharding off."""
        if nl.program_ndim() > 0:
            return (nl.program_id(0), nl.num_programs(0))
        return (0, 1)

    def lossk__tile_range(NT):
        """This program's contiguous tile range [i0, i1) of the NT tiles (concrete ints at trace time)."""
        pid, nprg = lossk__shard()
        per, rem = (NT // nprg, NT % nprg)
        i0 = pid * per + min(pid, rem)
        n = per + (1 if pid < rem else 0)
        return (i0, i0 + n)

    def lossk__load_block(dst_dtype, src, P, n):
        """[P, n] SBUF block from an HBM slice, converted to dst_dtype (by the DMA, or by a tensor_copy)."""
        blk = nl.ndarray((P, n), dtype=dst_dtype, buffer=nl.sbuf)
        lossk__dma(dst=blk, src=src)
        return blk

    def lossk__targets_slice(tg2, i0, i1):
        """HBM view of this program's targets: [P, n] int32, or the low words of int64 pairs ([P, NT, 2] view)."""
        if len(tg2.shape) == 3:
            return tg2[:, i0:i1, 0]
        return tg2[:, i0:i1]

    def lossk__pair(shape, dtype):
        """Two SBUF tiles for explicit double buffering.  Built here, not with a comprehension: the NKI frontend
        rejects comprehensions inside kernel bodies ('unsupported expression': they are nested scopes)."""
        return (nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf), nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf))

    def lossk__iota_row(P, V):
        """[P, V] fp32 tile with row 0, 1, ..., V-1 on every partition (GpSimd, once per program)."""
        io = nl.ndarray((P, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=io, pattern=[[1, V]], offset=0, channel_multiplier=0)
        return io

    @nki.jit
    def lossk__loss_fwd_kernel(x3, tg2, softcap, scale):
        """loss2[P, NT], lz2[P, NT] (fp32) from x3[P, NT, V] logits and tg2[P, NT] int32 targets.
        loss2[p, i] = scale * (softcap + log sum_v exp(softcap*tanh(x/softcap) - softcap) - softcap*tanh(x[t]/softcap))."""
        P, NT, V = x3.shape
        f32 = nl.float32
        loss2 = nl.ndarray((P, NT), dtype=f32, buffer=nl.shared_hbm)
        lz2 = nl.ndarray((P, NT), dtype=f32, buffer=nl.shared_hbm)
        i0, i1 = lossk__tile_range(NT)
        n = i1 - i0
        if n > 0:
            tsrc = lossk__targets_slice(tg2, i0, i1)
            zb = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            tgt = lossk__load_block(nl.uint32, tsrc, P, n)
            xtb = nl.ndarray((P, n), dtype=x3.dtype, buffer=nl.sbuf)
            xbs = lossk__pair((P, V), x3.dtype)
            ts = lossk__pair((P, V), f32)
            es = lossk__pair((P, V), lossk__EXP_DT)
            for j in range(min(lossk__PREFETCH, n)):
                lossk__dma(dst=xbs[j % lossk__NBUF], src=x3[:, i0 + j, :])
            for j in range(n):
                i = i0 + j
                xb, t, e = (xbs[j % lossk__NBUF], ts[j % lossk__NBUF], es[j % lossk__NBUF])
                jn = j + lossk__PREFETCH
                if jn < n:
                    lossk__dma(dst=xbs[jn % lossk__NBUF], src=x3[:, i0 + jn, :])
                nisa.activation(dst=t, op=nl.tanh, data=xb, scale=1.0 / softcap)
                nisa.activation_reduce(dst=e, op=nl.exp, data=t, reduce_op=nl.add, reduce_res=zb[:, j:j + 1], scale=softcap, bias=-softcap)
                nisa.nc_n_gather(dst=xtb[:, j:j + 1], data=xb, indices=tgt[:, j:j + 1])
            lzb = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=lzb, op=nl.log, data=zb)
            lossk__dma(dst=lz2[:, i0:i1], src=lzb)
            tb = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=tb, op=nl.tanh, data=xtb, scale=1.0 / softcap)
            lzs = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=lzs, data=lzb, op0=nl.multiply, operand0=scale, op1=nl.add, operand1=softcap * scale)
            lossb = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=lossb, data=tb, op0=nl.multiply, operand0=-softcap * scale, op1=nl.add, operand1=lzs)
            lossk__dma(dst=loss2[:, i0:i1], src=lossb)
        return (loss2, lz2)

    @nki.jit
    def lossk__loss_bwd_kernel(x3, tg2, lz2, g2, softcap, scale):
        """d3[P, NT, V] (x3.dtype) = g2[p, i] * scale * (softmax - onehot) * (1 - tanh^2(x / softcap))."""
        P, NT, V = x3.shape
        f32 = nl.float32
        d3 = nl.ndarray((P, NT, V), dtype=x3.dtype, buffer=nl.shared_hbm)
        i0, i1 = lossk__tile_range(NT)
        n = i1 - i0
        if n > 0:
            tgt = lossk__load_block(f32, lossk__targets_slice(tg2, i0, i1), P, n)
            lzb = lossk__load_block(f32, lz2[:, i0:i1], P, n)
            gb = lossk__load_block(f32, g2[:, i0:i1], P, n)
            nb = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=nb, data=lzb, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=-softcap)
            gs = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=gs, data=gb, op0=nl.multiply, operand0=scale)
            sq = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=sq, op=nl.sqrt, data=gs)
            io = lossk__iota_row(P, V)
            xbs = lossk__pair((P, V), x3.dtype)
            ts = lossk__pair((P, V), f32)
            ps = lossk__pair((P, V), f32)
            qs = ps
            ds = xbs
            for j in range(min(lossk__PREFETCH, n)):
                lossk__dma(dst=xbs[j % lossk__NBUF], src=x3[:, i0 + j, :])
            for j in range(n):
                i = i0 + j
                xb, t, p, q, d = (xbs[j % lossk__NBUF], ts[j % lossk__NBUF], ps[j % lossk__NBUF], qs[j % lossk__NBUF], ds[j % lossk__NBUF])
                jn = j + lossk__PREFETCH
                if jn < n:
                    lossk__dma(dst=xbs[jn % lossk__NBUF], src=x3[:, i0 + jn, :])
                nisa.activation(dst=t, op=nl.tanh, data=xb, scale=1.0 / softcap)
                nisa.activation(dst=p, op=nl.exp, data=t, scale=softcap, bias=nb[:, j:j + 1])
                nisa.scalar_tensor_tensor(dst=q, data=io, op0=nl.equal, operand0=tgt[:, j:j + 1], op1=nl.subtract, operand1=p)
                d = xb
                nisa.activation(dst=t, op=nl.square, data=t, scale=sq[:, j:j + 1])
                nisa.scalar_tensor_tensor(dst=d, data=t, op0=nl.subtract, operand0=gs[:, j:j + 1], op1=nl.multiply, operand1=q)
                lossk__dma(dst=d3[:, i, :], src=d)
        return d3
    lossk__HAS_NKI = True
except Exception:
    lossk__HAS_NKI = False

def lossk__disable_kernel(where: str, err: BaseException) -> None:
    global lossk__KERNEL_DISABLED, lossk__LAST_KERNEL_ERROR
    if not lossk__KERNEL_DISABLED:
        lossk__KERNEL_DISABLED = True
        lossk__LAST_KERNEL_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[nki_loss_head] kernel disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)

def lossk__launch(kernel):
    """kernel[2] under an LNC=2 runtime (both physical cores run half the tiles), plain call otherwise."""
    return kernel[lossk__LNC] if lossk__LNC == 2 else kernel

def lossk__kernel_targets(tg: torch.Tensor):
    """Targets as the kernel wants them: [P, NT] int32, or (TGT=view) the int64 bit pattern as [P, NT, 2] int32."""
    N = tg.numel()
    NT = N // lossk__P
    if tg.dtype == torch.int32:
        return tg.contiguous().view(lossk__P, NT)
    if tg.dtype == torch.int64:
        try:
            return tg.contiguous().view(torch.int32).view(lossk__P, NT, 2)
        except Exception:
            pass
    return tg.to(torch.int32).view(lossk__P, NT)

def lossk__tile_views(x2: torch.Tensor):
    N, V = x2.shape
    return x2.view(lossk__P, N // lossk__P, V)

class lossk_LossHeadFn(torch.autograd.Function):
    """rows[N] = scale * nll_i via the NKI kernels; eager math if a kernel raises."""

    @staticmethod
    def forward(ctx, x2, tgt_k, targets, softcap, scale):
        N, V = x2.shape
        ctx.softcap, ctx.scale = (softcap, scale)
        lz2 = None
        try:
            loss2, lz2 = lossk__launch(lossk__loss_fwd_kernel)(lossk__tile_views(x2), tgt_k, float(softcap), float(scale))
            rows = loss2.view(N)
        except Exception as err:
            lossk__disable_kernel('forward', err)
            with torch.no_grad():
                rows = lossk_eager_loss_rows(x2, targets, softcap) * scale
        ctx.save_for_backward(x2, tgt_k, targets, lz2)
        return rows

    @staticmethod
    def backward(ctx, grad_rows):
        x2, tgt_k, targets, lz2 = ctx.saved_tensors
        N, V = x2.shape
        softcap, scale = (ctx.softcap, ctx.scale)
        if lz2 is not None and (not lossk__KERNEL_DISABLED):
            try:
                g2 = grad_rows.contiguous().view(lossk__P, N // lossk__P)
                d3 = lossk__launch(lossk__loss_bwd_kernel)(lossk__tile_views(x2), tgt_k, lz2, g2, float(softcap), float(scale))
                return (d3.view(N, V), None, None, None, None)
            except Exception as err:
                lossk__disable_kernel('backward', err)
        with torch.enable_grad():
            xr = x2.detach().requires_grad_(True)
            rows = lossk_eager_loss_rows(xr, targets, softcap) * scale
            gx, = torch.autograd.grad(rows, xr, grad_rows)
        return (gx, None, None, None, None)

def lossk_fused_loss(logits: torch.Tensor, targets: torch.Tensor, reduction: str='mean', softcap: float=lossk_LOGIT_SOFTCAP) -> torch.Tensor:
    """train.py's static-CE chain on the raw lm_head output -- one NKI launch when possible, eager otherwise."""
    V = logits.shape[-1]
    N = targets.numel()
    use_kernel = lossk__HAS_NKI and (not lossk__KERNEL_DISABLED) and (logits.device.type == 'neuron') and (logits.dtype in (torch.bfloat16, torch.float32)) and (V == lossk_VOCAB or lossk__ANY_V) and (N % lossk__P == 0) and (N > 0) and (logits.numel() == N * V) and (reduction in ('mean', 'sum', 'none')) and (targets.dtype in (torch.int64, torch.int32))
    if not use_kernel:
        return lossk_eager_loss(logits, targets, reduction, softcap)
    x2 = logits.reshape(N, V)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    tg = targets.reshape(N)
    tgt_k = lossk__kernel_targets(tg)
    scale = 1.0 / N if reduction == 'mean' else 1.0
    rows = lossk_LossHeadFn.apply(x2, tgt_k, tg, softcap, scale)
    if reduction == 'none':
        return rows.view_as(targets)
    return rows.sum()
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    lossk__VEC = nisa.engine.vector

    def lossk__multi_iota(P, V):
        """The compare row 0..V-1 of the backward: fp32 (today's _iota_row, silicon-proven; the default) or int16 (i16:
        half the bytes per read; ids < 2^15 are exact and every compare still runs in fp32).  i16 is a PROBE."""
        return lossk__iota_row(P, V)

    def lossk__multi_onehot(dst, io, t):
        """dst = [iota == t] (exact 0 / 1): nki_embedding's silicon-proven one-hot instruction.  auto (default) is that
        exact call (no engine argument: the compiler places it); vector / gpsimd pin the engine (host_compile probes)."""
        nisa.tensor_scalar(dst=dst, data=io, op0=nl.equal, operand0=t)

    @nki.jit
    def lossk__loss_multi_fwd_kernel(x3, tg3, rw3, softcap):
        """[R1 mtp ship] loss2[P, NT] = sum_k rw3[k] * (c + lz - c tanh(x[t_k] / c)) and lz2[P, NT] (fp32), from x3[P, NT, V]
        logits, tg3[K, P, NT] int32 targets and rw3[K, P, NT] fp32 row weights (row n = p * NT + i, as in fused_loss)."""
        P, NT, V = x3.shape
        K = tg3.shape[0]
        f32 = nl.float32
        loss2 = nl.ndarray((P, NT), dtype=f32, buffer=nl.shared_hbm)
        lz2 = nl.ndarray((P, NT), dtype=f32, buffer=nl.shared_hbm)
        i0, i1 = lossk__tile_range(NT)
        n = i1 - i0
        if n > 0:
            zb = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            tgt = nl.ndarray((P, K, n), dtype=nl.uint32, buffer=nl.sbuf)
            xtb = nl.ndarray((P, K, n), dtype=x3.dtype, buffer=nl.sbuf)
            for k in range(K):
                lossk__dma(dst=tgt[:, k, :], src=tg3[k, :, i0:i1])
            xbs = lossk__pair((P, V), x3.dtype)
            ts = lossk__pair((P, V), f32)
            es = lossk__pair((P, V), lossk__EXP_DT)
            for j in range(min(lossk__PREFETCH, n)):
                lossk__dma(dst=xbs[j % lossk__NBUF], src=x3[:, i0 + j, :])
            for j in range(n):
                i = i0 + j
                xb, t, e = (xbs[j % lossk__NBUF], ts[j % lossk__NBUF], es[j % lossk__NBUF])
                jn = j + lossk__PREFETCH
                if jn < n:
                    lossk__dma(dst=xbs[jn % lossk__NBUF], src=x3[:, i0 + jn, :])
                nisa.activation(dst=t, op=nl.tanh, data=xb, scale=1.0 / softcap)
                nisa.activation_reduce(dst=e, op=nl.exp, data=t, reduce_op=nl.add, reduce_res=zb[:, j:j + 1], scale=softcap, bias=-softcap)
                for k in range(K):
                    nisa.nc_n_gather(dst=xtb[:, k, j:j + 1], data=xb, indices=tgt[:, k, j:j + 1])
            lzb = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=lzb, op=nl.log, data=zb)
            lossk__dma(dst=lz2[:, i0:i1], src=lzb)
            lzc = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=lzc, data=lzb, op0=nl.add, operand0=softcap)
            rwb = nl.ndarray((P, K, n), dtype=f32, buffer=nl.sbuf)
            tb = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            nll = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            acc = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            for k in range(K):
                lossk__dma(dst=rwb[:, k, :], src=rw3[k, :, i0:i1])
                nisa.activation(dst=tb, op=nl.tanh, data=xtb[:, k, :], scale=1.0 / softcap)
                nisa.scalar_tensor_tensor(dst=nll, data=tb, op0=nl.multiply, operand0=-softcap, op1=nl.add, operand1=lzc)
                if k == 0:
                    nisa.tensor_tensor(dst=acc, data1=nll, data2=rwb[:, 0, :], op=nl.multiply, engine=lossk__VEC)
                else:
                    nisa.tensor_tensor(dst=nll, data1=nll, data2=rwb[:, k, :], op=nl.multiply, engine=lossk__VEC)
                    nisa.tensor_tensor(dst=acc, data1=acc, data2=nll, op=nl.add, engine=lossk__VEC)
            lossk__dma(dst=loss2[:, i0:i1], src=acc)
        return (loss2, lz2)

    @nki.jit
    def lossk__loss_multi_bwd_kernel(x3, tg3, rw3, lz2, g2, softcap):
        """[R1 mtp ship] d3[P, NT, V] (x3.dtype) = g (1 - t^2) (W softmax - sum_k rw_k [v == t_k]), t = tanh(x / c),
        W = sum_k rw_k: the gradient of sum_k rw_k nll_k (fast form; see the block comment above)."""
        P, NT, V = x3.shape
        K = tg3.shape[0]
        f32 = nl.float32
        d3 = nl.ndarray((P, NT, V), dtype=x3.dtype, buffer=nl.shared_hbm)
        i0, i1 = lossk__tile_range(NT)
        n = i1 - i0
        if n > 0:
            tgt = nl.ndarray((P, K, n), dtype=f32, buffer=nl.sbuf)
            rwb = nl.ndarray((P, K, n), dtype=f32, buffer=nl.sbuf)
            for k in range(K):
                lossk__dma(dst=tgt[:, k, :], src=tg3[k, :, i0:i1])
                lossk__dma(dst=rwb[:, k, :], src=rw3[k, :, i0:i1])
            nb = lossk__load_block(f32, lz2[:, i0:i1], P, n)
            ab = lossk__load_block(f32, g2[:, i0:i1], P, n)
            nisa.tensor_tensor(dst=ab, data1=ab, data2=rwb[:, 0, :], op=nl.multiply, engine=lossk__VEC)
            sq = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=sq, op=nl.sqrt, data=ab)
            nisa.tensor_scalar(dst=nb, data=nb, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=-softcap)
            if K > 1:
                nisa.reciprocal(dst=rwb[:, 0, :], data=rwb[:, 0, :])
                for k in range(1, K):
                    nisa.tensor_tensor(dst=rwb[:, k, :], data1=rwb[:, k, :], data2=rwb[:, 0, :], op=nl.multiply, engine=lossk__VEC)
                wb = nl.ndarray((P, n), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=wb, data=rwb[:, 1, :], op0=nl.add, operand0=1.0)
                for k in range(2, K):
                    nisa.tensor_tensor(dst=wb, data1=wb, data2=rwb[:, k, :], op=nl.add, engine=lossk__VEC)
                nisa.activation(dst=wb, op=nl.log, data=wb)
                nisa.tensor_tensor(dst=nb, data1=nb, data2=wb, op=nl.add, engine=lossk__VEC)
            io = lossk__multi_iota(P, V)
            xbs = lossk__pair((P, V), x3.dtype)
            ts = lossk__pair((P, V), f32)
            qs = lossk__pair((P, V), f32)
            for j in range(min(lossk__PREFETCH, n)):
                lossk__dma(dst=xbs[j % lossk__NBUF], src=x3[:, i0 + j, :])
            for j in range(n):
                i = i0 + j
                xb, t, q = (xbs[j % lossk__NBUF], ts[j % lossk__NBUF], qs[j % lossk__NBUF])
                jn = j + lossk__PREFETCH
                if jn < n:
                    lossk__dma(dst=xbs[jn % lossk__NBUF], src=x3[:, i0 + jn, :])
                nisa.activation(dst=t, op=nl.tanh, data=xb, scale=1.0 / softcap)
                nisa.activation(dst=q, op=nl.exp, data=t, scale=softcap, bias=nb[:, j:j + 1])
                nisa.scalar_tensor_tensor(dst=q, data=io, op0=nl.equal, operand0=tgt[:, 0, j:j + 1], op1=nl.subtract, operand1=q)
                for k in range(1, K):
                    lossk__multi_onehot(xb, io, tgt[:, k, j:j + 1])
                    nisa.scalar_tensor_tensor(dst=q, data=xb, op0=nl.multiply, operand0=rwb[:, k, j:j + 1], op1=nl.add, operand1=q)
                nisa.activation(dst=t, op=nl.square, data=t, scale=sq[:, j:j + 1])
                nisa.scalar_tensor_tensor(dst=xb, data=t, op0=nl.subtract, operand0=ab[:, j:j + 1], op1=nl.multiply, operand1=q)
                lossk__dma(dst=d3[:, i, :], src=xb)
        return d3
    lossk__HAS_NKI_MULTI = True
    lossk__MULTI_IMPORT_ERROR = None
except Exception as _err:
    lossk__HAS_NKI_MULTI = False
    lossk__MULTI_IMPORT_ERROR = f'{type(_err).__name__}: {_err}'
lossk__MULTI_TOL_ROWS = 1e-05
lossk__MULTI_TOL_D = 2.0 ** (-7)
lossk__MULTI_TOL_BIAS = 0.001
lossk__MULTI_TOL_FRO = 2.0 ** (-6)
lossk__MULTI_TOL_LZ_U = 2.0 ** (-24)
lossk__MULTI_CHECKED = False
lossk__MULTI_DISABLED = False
lossk__MULTI_ERROR = None
lossk__MULTI_CHECK = 'not run'
lossk_MULTI_FIRED = [0, 0]

def lossk__multi_device_ok(device) -> bool:
    return device.type == 'neuron'

def lossk_multi_available(device) -> bool:
    """[R1 mtp ship] fused_loss_multi can run its kernels on `device` (the caller's cheap pre-check, no device work)."""
    return lossk__HAS_NKI_MULTI and (not lossk__MULTI_DISABLED) and (not lossk__KERNEL_DISABLED) and lossk__multi_device_ok(device)

def lossk__multi_disable(where: str, err: BaseException) -> None:
    global lossk__MULTI_DISABLED, lossk__MULTI_ERROR
    if not lossk__MULTI_DISABLED:
        lossk__MULTI_DISABLED = True
        lossk__MULTI_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[nki_loss_head] mtp ship DISABLED after failure in {where}: {type(err).__name__}: {err} -- the run continues on the TEST FORM (one fused_loss pass per target)', flush=True)

def lossk__multi_rows_test_form(x2, tg3, rw3, softcap):
    """rows[N] = sum_k rw_k nll_k the test form's way, one fused_loss(..., "none") per target (the fallback)."""
    N = x2.shape[0]
    rows = None
    for k in range(tg3.shape[0]):
        r = lossk_fused_loss(x2, tg3[k].reshape(N), reduction='none', softcap=softcap) * rw3[k].reshape(N)
        rows = r if rows is None else rows + r
    return rows

class lossk_LossMultiFn(torch.autograd.Function):
    """[R1 mtp ship] rows[N] = sum_k rw[k] * nll(x, tg[k]) via the two multi-target kernels.  A kernel failure answers
    this call with the test form's math and disables the ship path for the process."""

    @staticmethod
    def forward(ctx, x2, tg3, rw3, softcap):
        N, V = x2.shape
        ctx.softcap = softcap
        lz2 = None
        try:
            loss2, lz2 = lossk__launch(lossk__loss_multi_fwd_kernel)(lossk__tile_views(x2), tg3, rw3, float(softcap))
            rows = loss2.view(N)
            lossk_MULTI_FIRED[0] += 1
        except Exception as err:
            lossk__multi_disable('forward', err)
            lz2 = None
            rows = lossk__multi_rows_test_form(x2, tg3, rw3, softcap)
        ctx.save_for_backward(x2, tg3, rw3, lz2)
        return rows

    @staticmethod
    def backward(ctx, grad_rows):
        x2, tg3, rw3, lz2 = ctx.saved_tensors
        N, V = x2.shape
        if lz2 is not None and (not lossk__MULTI_DISABLED):
            try:
                g2 = grad_rows.contiguous().view(lossk__P, N // lossk__P)
                d3 = lossk__launch(lossk__loss_multi_bwd_kernel)(lossk__tile_views(x2), tg3, rw3, lz2, g2, float(ctx.softcap))
                lossk_MULTI_FIRED[1] += 1
                return (d3.view(N, V), None, None, None)
            except Exception as err:
                lossk__multi_disable('backward', err)
        with torch.enable_grad():
            xr = x2.detach().requires_grad_(True)
            rows = lossk__multi_rows_test_form(xr, tg3, rw3, ctx.softcap)
            gx, = torch.autograd.grad(rows, xr, grad_rows)
        return (gx, None, None, None)

def lossk_fused_loss_multi(logits: torch.Tensor, tg: torch.Tensor, rw: torch.Tensor, softcap: float=lossk_LOGIT_SOFTCAP, ref=None):
    """[R1 mtp ship] 0-dim fp32 sum_i sum_k rw[k, i] nll(logits_i, tg[k, i]) through ONE multi-target kernel pair, or
    None when the kernels cannot run here (the caller then runs the test form).  See the block comment above."""
    V = logits.shape[-1]
    K, N = tg.shape
    if not (lossk_multi_available(logits.device) and logits.dtype in (torch.bfloat16, torch.float32) and (V == lossk_VOCAB) and (N % lossk__P == 0) and (N > 0) and (logits.numel() == N * V) and (tg.dtype == torch.int32) and (rw.dtype == torch.float32) and (tuple(rw.shape) == (K, N))):
        return None
    x2 = logits.reshape(N, V)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    tg3 = tg.contiguous().view(K, lossk__P, N // lossk__P)
    rw3 = rw.contiguous().view(K, lossk__P, N // lossk__P)
    if not lossk__MULTI_CHECKED:
        lossk__multi_selfcheck(x2, tg3, rw3, softcap, ref)
        if lossk__MULTI_DISABLED:
            return None
    return lossk_LossMultiFn.apply(x2, tg3, rw3, softcap).sum()

def lossk__multi_blocks(N: int):
    """The self-check's sampled dlogits rows: contiguous row blocks [a, b) (whole rows: each is ONE device->host copy;
    row n = p * NT + i lives on partition p, tile i).  At the live NT = 256 they cover EVERY tile index 0..255 and both
    LNC programs: program 0's first tiles and the program boundary (partition 0), tiles NT/4..3NT/8 (partition 2) and
    5NT/8..3NT/4 (partition 5), both programs across the partition boundary at N/2, program 1's last tiles (partition
    127), and the end of the first 1024-token window (the masked t+2 / t+3 rows)."""
    NT = N // lossk__P
    h = 64
    blocks = []
    for a, b in sorted(((0, h), (NT // 2 - h // 2, NT // 2 + h // 2), (2 * NT + NT // 4, 2 * NT + 3 * NT // 8), (5 * NT + 5 * NT // 8, 5 * NT + 3 * NT // 4), (N // 2 - h, N // 2 + h), (N - h, N), (1024 - h, 1024 + h))):
        a, b = (max(0, a), min(N, b))
        if b <= a:
            continue
        if blocks and a <= blocks[-1][1]:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], b))
        else:
            blocks.append((a, b))
    return blocks

def lossk__pull_rows(t2, blocks):
    return torch.cat([t2[a:b].cpu().double() for a, b in blocks])

def lossk__multi_selfcheck(x2, tg3, rw3, softcap, ref) -> None:
    """ONCE per process, on the first multi-target call (step 0: an excluded startup step, outside the charged clock).
    Runs the ship kernels AND the test form -- one single-target kernel pair per target, in the test form's own target
    layout, scale and upstream gradient (target 0 is exactly fused_loss(..., "mean"), so the phase-3 NEFFs trace here,
    before the clock) -- on the SAME live logits, targets and weights, and compares on the host in fp64:
      loss rows (all N)       |ship - test| <= 1e-5 * sum_k |rw_k nll_k|, and log z to 2 gamma_{V-1} + 1e-6 |lz|
                              (two fp32 row sums of V positive terms in different orders; 9.77e-4 at V = 8192)
      dlogits (sampled rows)  |ship - sum_k d_k| <= 1.1 * 2^-7 (|ship| + sum_k |d_k|) per element: ONE bf16 ulp per
                              rounded value.  Stochastic rounding (on in training) moves a bf16 value by < 1 ulp <=
                              2^-7 |value| (RNE: <= 1/2), so for ANY draw the ratio stays below 1.0079 / 1.1 = 0.92 plus
                              the fp32 / activation-table differences (~1e-5 relative, 0.1 % of the bound): the SR
                              false-FAIL of fp8r2_build/NOTES.md sec 11-12 cannot recur.  A real defect (a lost,
                              mis-weighted or shifted target, a wrong W') is an O(1) error at the affected elements.
                              Plus two statistics SR cannot move: signed bias <= 1e-3, relative Frobenius <= 2^-6.
    A non-finite value or an exception is a FAIL too.  A FAIL disables the ship path LOUDLY (every rank prints); train.py
    then runs the test form for the rest of the run.
    What it certifies at step 0 (init logits, std ~0.03): the target positions and gathers, the weights as data, the
    1 / rw_0 normalisation, the ln W' fold, the per-target sum and log z.  What it CANNOT see there: the t-dependent
    softcap chain ((1 - t^2), the tanh of the gathered logit), since t ~ 2e-3 makes (1 - t^2) = 1 - 5e-6.  That chain
    is today's silicon-proven fast-backward pair with gs -> a = g rw_0; host_compile_mtpship.py (nki.simulate + BIRSim)
    and bench_mtpship.py's on-device self-check (REQUIRED before an arm) certify it at randn * 4 logits."""
    global lossk__MULTI_CHECKED, lossk__MULTI_CHECK
    lossk__MULTI_CHECKED = True
    try:
        N, V = x2.shape
        K = tg3.shape[0]
        NT = N // lossk__P
        if ref is None:
            ref = [(tg3[k].reshape(N), rw3[k].reshape(N)) for k in range(K)]
        assert len(ref) == K, f'ref has {len(ref)} targets, the ship tensors {K}'
        x3 = lossk__tile_views(x2)
        blocks = lossk__multi_blocks(N)
        with torch.no_grad():
            ones2 = torch.ones(lossk__P, NT, dtype=torch.float32, device=x2.device)
            loss2, lz2 = lossk__launch(lossk__loss_multi_fwd_kernel)(x3, tg3, rw3, float(softcap))
            s_rows, s_lz = (loss2.reshape(N).cpu().double(), lz2.reshape(N).cpu().double())
            d3 = lossk__launch(lossk__loss_multi_bwd_kernel)(x3, tg3, rw3, lz2, ones2, float(softcap))
            s_d = lossk__pull_rows(d3.view(N, V), blocks)
            del d3
            t_rows = torch.zeros(N, dtype=torch.float64)
            t_mag = torch.zeros(N, dtype=torch.float64)
            t_d = torch.zeros_like(s_d)
            t_mag_d = torch.zeros_like(s_d)
            t_lz = None
            for tk, wk in ref:
                tgt_k = lossk__kernel_targets(tk.reshape(N))
                scale = 1.0 / N if wk is None else 1.0
                g2 = ones2 if wk is None else wk.reshape(lossk__P, NT).contiguous()
                rows_k, lz_k = lossk__launch(lossk__loss_fwd_kernel)(x3, tgt_k, float(softcap), float(scale))
                r = rows_k.reshape(N).cpu().double()
                if wk is not None:
                    r = r * wk.reshape(N).cpu().double()
                t_rows += r
                t_mag += r.abs()
                if t_lz is None:
                    t_lz = lz_k.reshape(N).cpu().double()
                dk3 = lossk__launch(lossk__loss_bwd_kernel)(x3, tgt_k, lz_k, g2, float(softcap), float(scale))
                dk = lossk__pull_rows(dk3.view(N, V), blocks)
                del dk3
                t_d += dk
                t_mag_d += dk.abs()
        finite = bool(torch.isfinite(s_rows).all() and torch.isfinite(s_lz).all() and torch.isfinite(s_d).all())
        e_rows = float(((s_rows - t_rows).abs() / (lossk__MULTI_TOL_ROWS * t_mag + 1e-30)).max())
        g_v = (V - 1) * lossk__MULTI_TOL_LZ_U / (1.0 - (V - 1) * lossk__MULTI_TOL_LZ_U)
        e_lz = float(((s_lz - t_lz).abs() / (2.0 * g_v + 1e-06 * t_lz.abs())).max())
        e_tot = float((s_rows.sum() - t_rows.sum()).abs() / t_mag.sum().clamp_min(1e-30))
        err = s_d - t_d
        ratio = err.abs() / (lossk__MULTI_TOL_D * 1.1 * (s_d.abs() + t_mag_d) + 1e-35)
        e_d = float(ratio.max())
        bias = float(err.sum()) / max(float((s_d.abs() + t_mag_d).sum()), 1e-30)
        fro = float(err.norm()) / max(float(t_d.norm()), 1e-30)
        ndiff = float((s_d != t_d).double().mean())
        ok = finite and e_rows <= 1.0 and (e_lz <= 1.0) and (e_tot <= lossk__MULTI_TOL_ROWS) and (e_d <= 1.0) and (abs(bias) <= lossk__MULTI_TOL_BIAS) and (fro <= lossk__MULTI_TOL_FRO)
        line = f'K={K} rows={N} | loss rows {e_rows:.2e} of tol, log z {e_lz:.2e} of tol, total {e_tot:.1e} | dlogits {s_d.shape[0]} rows x {V}: worst {e_d:.3f} of the SR-safe bound (1.1 x 2^-7 per component), bias {bias:.1e}, rel-fro {fro:.1e}, {100 * ndiff:.2f}% differ'
        if not ok:
            w = int(ratio.argmax())
            line += f' | worst at sampled row {w // V} col {w % V}: ship {float(s_d.reshape(-1)[w]):.6e} test form {float(t_d.reshape(-1)[w]):.6e} finite={finite}'
        lossk__MULTI_CHECK = line + (' -> PASS' if ok else ' -> FAIL')
        print(f'[nki_loss_head] mtp ship self-check: {lossk__MULTI_CHECK}', flush=True)
        if not ok:
            lossk__multi_disable('self-check', RuntimeError(lossk__MULTI_CHECK))
    except Exception as err:
        lossk__MULTI_CHECK = f'ERROR {type(err).__name__}: {err} -> FAIL'
        print(f'[nki_loss_head] mtp ship self-check: {lossk__MULTI_CHECK}', flush=True)
        lossk__multi_disable('self-check', err)

def lossk_multi_status() -> str:
    """[R1 mtp ship] the liveness fields of train.py's end-of-run `mtp ship:` line."""
    return f'fired_fwd={lossk_MULTI_FIRED[0]} fired_bwd={lossk_MULTI_FIRED[1]} selfcheck={lossk__MULTI_CHECK!r} disabled={lossk__MULTI_ERROR!r} kernels={('yes' if lossk__HAS_NKI_MULTI else 'NO (' + str(lossk__MULTI_IMPORT_ERROR) + ')')}'
fused_loss = lossk_fused_loss
USE_NKI_LOSS_HEAD = lossk_USE_NKI_LOSS_HEAD
MTP = True
_MTP = {'on': (False,) * 2, 'ph': 0, 'n': [[0] * 3 for _ in range(3)], 'm': {}, 'cov': ''}

def mtp_set(p: float) -> None:
    """[R1 mtp] Once per optimizer step at clock fraction p: host flags, per-phase step count, one h2d copy.
    [c27] The window ends at clock 1/2: phase 1 = [0, 1/4) (t+2 .5, t+3 .25 -> 0), phase 2 = [1/4, 1/2) (t+2 .5 -> 0),
    phase 3 = [1/2, 1] next-token only."""
    u = 1.0 - 4.0 * p
    w = (0.5, 0.25 * u) if p < 0.25 else (0.5 * (1 + u), 0.0) if p < 0.5 else (0.0, 0.0)
    _MTP['on'], _MTP['ph'] = (tuple((v > 0 for v in w)), min(int(p * 4), 2))
    _MTP['n'][_MTP['ph']][0] += 1
    if any(_MTP['on']):
        _MTP['w'].copy_(torch.tensor(w, dtype=torch.float32))

def mtp_terms(y: torch.Tensor, live) -> list:
    """[R1 mtp] the (target, row weight) pairs of the live extra targets (t+2, t+3): shifts of y in y's dtype with a
    VALID dummy id and weight 0 past the window.  The test form (mtp_aux) and the ship form (mtp_ship) both build
    their targets here."""
    B, T = y.shape
    key = (B, T, y.device)
    if key not in _MTP['m']:
        c = torch.arange(T, device=y.device).view(1, T)
        _MTP['m'][key] = [((c < T - j).float() / (B * T)).expand(B, T).contiguous() for j in (1, 2)]
    m, w = (_MTP['m'][key], _MTP['w'])
    return [(torch.cat((y[:, j + 1:], y[:, T - j - 1:]), 1), w[j:j + 1] * m[j]) for j in (0, 1) if live[j]]
MTP_SHIP = True
fused_loss_multi = lossk_fused_loss_multi
multi_available = lossk_multi_available
multi_status = lossk_multi_status
_MTP.update(s={}, scov='', ns=[0, 0, 0])

def mtp_ship(x: torch.Tensor, y: torch.Tensor):
    """[R1 mtp ship] next-token loss + both aux terms as ONE multi-target kernel pair: 0-dim fp32, or None when the
    kernel cannot run.  The same objective as fused_loss(x, y) + mtp_aux(x, y)."""
    B, T = y.shape
    key = (B, T, y.device)
    if key not in _MTP['s']:
        _MTP['s'][key] = torch.full((B, T), 1.0 / (B * T), device=y.device)
    live = (MTP, MTP)
    y32 = y.to(torch.int32)
    terms = mtp_terms(y32, live)
    K = 1 + len(terms)
    tg = torch.stack([y32] + [yk for yk, _ in terms]).view(K, B * T)
    rw = torch.stack([_MTP['s'][key]] + [mk for _, mk in terms]).view(K, B * T)
    ref = None if _MTP['scov'] else [(y, None)] + mtp_terms(y, live)
    loss = fused_loss_multi(x, tg, rw, softcap=LOGIT_SOFTCAP, ref=ref)
    if not _MTP['scov']:
        _MTP['scov'] = f'path={('ship' if loss is not None else 'NEXT-TOKEN ONLY (ship kernel unavailable)')} K={K} rows={B * T} targets=next,t+2,t+3'
        print(f'mtp: ship first micro-step {_MTP['scov']}', flush=True)
        bosx_arm(x, y)
    if loss is not None:
        n = _MTP['n'][_MTP['ph']]
        for j in range(2):
            n[j + 1] += _MTP['on'][j]
        _MTP['ns'][_MTP['ph']] += 1
    return loss
BOS_EXCL = 'post'
_BOSX = {'on': False, 'armed': False, 'id': None, 's': {}, 'calls': 0, 'declined': 0, 'hand': None, 'zero': None, 'chk': None}

def bosx_rw(y: torch.Tensor):
    """[R1 bosx] (int32 targets, fp32 row weights), both [B, T]: the weight is 1/N, and 0 where the target is BOS (data)."""
    B, T = y.shape
    key = (B, T, y.device)
    if key not in _BOSX['s']:
        _BOSX['s'][key] = torch.full((B, T), 1.0 / (B * T), device=y.device)
    y32 = y.to(torch.int32)
    return (y32, _BOSX['s'][key] * (y32 != _BOSX['id']))

def bosx_arm(x: torch.Tensor, y: torch.Tensor) -> None:
    """[R1 bosx] ONCE, from the first ship micro-step (step 0, excluded, after its K = 3 self-check): the module's own self-check
    re-armed for K = 1 (as R1_TST does) on these logits with a BOS-zeroed weight row, then one K = 1 forward + backward on a
    detached view of them (the gradient is discarded) and the hand-over line's count: every graph the post-END path launches
    exists before the clock.  A FAIL, or no kernel here, RAISES (the arm would train the base loss under its name: VOID)."""
    global lossk__MULTI_CHECKED, lossk__MULTI_CHECK
    N = y.numel()
    y32, rw = bosx_rw(y)
    k3, lossk__MULTI_CHECKED = (lossk__MULTI_CHECK, False)
    with torch.enable_grad():
        xd = x.detach().requires_grad_(True)
        l1 = fused_loss_multi(xd, y32.view(1, N), rw.view(1, N), softcap=LOGIT_SOFTCAP, ref=[(y, rw)])
        chk = lossk__MULTI_CHECK
        if l1 is None or not chk.endswith('-> PASS'):
            raise RuntimeError(f'bosx: the K=1 multi-target pair is unavailable or failed its self-check ({multi_status()}); refusing to run without it')
        l1.backward()
    z = int((rw == 0).sum().item())
    del xd, l1
    _BOSX.update(armed=True, chk=chk)
    lossk__MULTI_CHECK = k3
    print(f'bosx: K=1 pair traced + self-checked at step 0 (rows {N}, {z} BOS targets at weight 0, bos_id {_BOSX['id']}): {chk}', flush=True)

def bosx_loss(x: torch.Tensor, y: torch.Tensor):
    """[R1 bosx] AFTER THE MTP END: sum_i rw_i nll(x_i, y_i) via fused_loss_multi at K = 1 (0-dim fp32), or None when the kernel
    declines (the caller then runs the base loss; counted and printed: such an arm is VOID)."""
    if not _BOSX['armed']:
        raise RuntimeError('bosx: the K=1 pair was never traced before the clock (no ship micro-step at step 0)')
    N = y.numel()
    y32, rw = bosx_rw(y)
    loss = fused_loss_multi(x, y32.view(1, N), rw.view(1, N), softcap=LOGIT_SOFTCAP)
    if loss is None:
        _BOSX['declined'] += 1
        if _BOSX['declined'] == 1:
            print(f'bosx: K=1 path DECLINED ({multi_status()}): the base loss runs from here', flush=True)
        return None
    _BOSX['calls'] += 1
    if _BOSX['zero'] is None:
        _BOSX['zero'] = int((rw == 0).sum().item())
        print(f'bosx: K=1 path engaged at step {_BOSX['hand'][0]} progress {_BOSX['hand'][1]:.4f}: {_BOSX['zero']} of {N} rows at weight 0 (BOS targets) on its first micro-step', flush=True)
    return loss
"Fused transformer-block head -- gains + residual mix + the first RMSNorm -- as ONE NKI launch forward and ONE\nbackward, sharded over the two physical cores of an LNC=2 logical core.  Second attempt at T4 (work/agents/kernel/\nnki_gain_norm.py was 1.08x / 0.85x on hardware); built on the transfer rules of the loss-head and RoPE+norm kernels.\n\nReplaces, in train.py's Block.forward:\n\n    x  = self.resid_lambda.to(x.dtype) * x + self.x0_lambda.to(x.dtype) * x0      # 2 casts, 2 muls, 1 add\n    xn = norm(x)                                                                    # F.rms_norm, parameter-free\n    (and their backward: rms_norm backward, the residual-fork accumulate, 2 gain muls, 2 (g*x).sum(), casts)\n\nwith\n\n    x, xn = block_head(x, x0, self.resid_lambda, self.x0_lambda)\n\n`rmsnorm_fast(x)`, a plain two-core RMSNorm forward / backward from the same code.\n\nPublic entry points\n-------------------\n    x_new, x_norm = block_head(x, x0, l1, l2)\n        x, x0    [B, T, C] (or any [..., C]) bf16 or fp32, same dtype and shape; N = rows must be a multiple of 128\n        l1, l2   the Block's (1,) fp32 gains (read INSIDE the kernel as tensor operands: never Python floats,\n                 nothing recompiles when the optimizer changes them)\n        x_new    l1 * x + l2 * x0, rounded once into x.dtype (the residual stream continues from it)\n        x_norm   RMSNorm(x_new) = x_new * rsqrt(mean(x_new^2) + 2**-23), rounded once; the bf16-ROUNDED x_new is\n                 what gets normalised, exactly as F.rms_norm(x_new) does on the eager path\n        grads    grad_x = l1 * G, grad_x0 = l2 * G (x.dtype), grad_l1 = sum(G * x), grad_l2 = sum(G * x0) ((1,) fp32)\n                 with G = g_new + J_rmsnorm^T(g_norm) = g_new + inv * (g_norm - x_new * (g_norm . x_new) * inv^2 / C)\n\nKernel design (measurements in NOTES.md)\n----------------------------------------\n* Rows n = p * NT + i live on partition p, index i (x viewed as [128, NT, C]); a tile is K consecutive i, i.e. ONE\n  contiguous K*C*itemsize run per partition (8 KiB bf16 at K=4, 1 MiB per DMA).  kernel[2]: program q takes the\n  contiguous i-range [q*NT/2, (q+1)*NT/2) -- disjoint HBM writes, both cores' 16 DMA engines busy.  Every per-row\n  vector (the saved 1/rms `inv`, the gain-gradient partials r1 / r2) is a [128, n] block moved ONCE per program.\n* Forward per tile: ScalarE t = l2 * x0 (fp32, per-partition `scale` tile), VectorE x_new = (x * l1) + t in ONE\n  scalar_tensor_tensor rounded into the stream dtype, K ScalarE activation_reduce(square) row sums, an exact\n  tensor_scalar rsqrt on the [128, K] block (the ScalarE rsqrt activation is a 1.8e-4 polynomial), K VectorE\n  tensor_scalar row scales into the bf16 x_norm tile.  Only `inv` ([N] fp32, 64 KiB) is saved for the backward.\n* Backward per tile: recompute x_new (same two instructions, bit-identical), VectorE prod = g_norm * x_new (fp32),\n  K ScalarE copy-reduces -> dot, nc = -dot * inv^2 / C on the [128, K] block, two per-row VectorE\n  scalar_tensor_tensors u = x_new * nc + g_norm ; G = u * inv + g_new (fp32, G in place over u), two whole-tile\n  VectorE tensor_scalars grad_x = G * l1 / grad_x0 = G * l2 (bf16, in place over the dead x_new / g_new buffers),\n  and for the scalar gains two VectorE products G * x / G * x0 with K ScalarE copy-reduces each into the per-row\n  partial blocks r1 / r2 -- NO reduction over the partition axis inside the kernel; the wrapper finishes with one\n  torch .sum() over the [2, 128, NT] partials (one tiny launch for both gains).  Knob GAIN_GRAD=matmul instead\n  writes G (bf16) and lets the wrapper do two [1, N*C] x [N*C, 1] matmul-reduces.\n* Explicit 2-deep prefetch (`_pair`): tile j+1's loads are issued before tile j's compute.  No comprehensions\n  anywhere in a kernel body (frontend rule).\n* Numerics: eager rounds the gains to bf16 and rounds after each of its three bf16 ops; the kernel keeps the fp32\n  gains and t in fp32 and rounds x_new once (correctly rounded), then normalises that bf16 value like eager.  The\n  test's rule: within 1 bf16 ulp of eager, or the kernel closer to an fp64 reference than eager is.\n\nrmsnorm_fast's kernel on (default OFF).  Kernel only on Neuron, bf16 / fp32, N % 128 == 0; a kernel failure prints\nonce, disables every kernel of this module for the process and answers with the eager chain -- which is also what\nthe wrappers return whenever a kernel does not apply.\n\nKnobs (env, read once at import; module-level constants because kernel bodies only see module globals):\n"
import os
import torch
import torch.nn.functional as F
bheadk__P = 128
bheadk__env = os.environ.get
bheadk_RMS_EPS = float(2.0 ** (-23))
bheadk_USE_NKI_BLOCK_HEAD = True
bheadk_USE_NKI_NORM2 = True
bheadk__MAX_K = 4
bheadk__GAIN_GRAD = 'rows'
bheadk__LNC = int(bheadk__env('NEURON_LOGICAL_NC_CONFIG', '1'))
bheadk__KERNEL_DISABLED = False
bheadk__LAST_KERNEL_ERROR = None

def bheadk_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))

def bheadk_eager_block_head(x: torch.Tensor, x0: torch.Tensor, l1: torch.Tensor, l2: torch.Tensor):
    """train.py's Block head: (x_new, norm(x_new)) with the (1,) gains cast to the stream dtype."""
    x_new = l1.to(x.dtype) * x + l2.to(x.dtype) * x0
    return (x_new, bheadk_norm(x_new))
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    def bheadk__dma(dst, src):
        nisa.dma_copy(dst=dst, src=src)

    def bheadk__shard():
        """(program_id, num_programs); (0, 1) without a launch grid or with sharding off."""
        if nl.program_ndim() > 0:
            return (nl.program_id(0), nl.num_programs(0))
        return (0, 1)

    def bheadk__row_range(NT):
        """This program's contiguous i-range [i0, i1) of the NT row indices per partition (concrete ints)."""
        pid, nprg = bheadk__shard()
        per, rem = (NT // nprg, NT % nprg)
        i0 = pid * per + min(pid, rem)
        n = per + (1 if pid < rem else 0)
        return (i0, i0 + n)

    def bheadk__rows_per_tile(n):
        """K = the largest divisor of n not above _MAX_K (no comprehension: kernel-body rule)."""
        k = 1
        for d in range(1, bheadk__MAX_K + 1):
            if n % d == 0:
                k = d
        return k

    def bheadk__pair(shape, dtype):
        """Two SBUF tiles for the explicit double buffer (built here, never with a comprehension)."""
        return (nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf), nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf))

    def bheadk__one(shape, dtype):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    def bheadk__load_gain(lg, P):
        """(1, 1) fp32 HBM gain -> fp32 [P, 1] SBUF tile, the same value on every partition (stride-0 DMA source
        via NkiTensor.broadcast; verified on chip by T4).  A [P, 1] HBM tensor (expand mode) is copied as is."""
        t = bheadk__one((P, 1), nl.float32)
        if lg.shape[0] == 1:
            bheadk__dma(dst=t, src=lg.broadcast(0, P))
        else:
            bheadk__dma(dst=t, src=lg)
        return t

    def bheadk__mix_dtype(dt):
        return nl.float32

    def bheadk__prod_dtype(dt):
        return nl.float32

    def bheadk__mix(xb, x0b, tb, xnb, l1t, l2t):
        """t = l2 * x0 (ScalarE or VectorE, dtype of tb); x_new = (x * l1) + t in ONE VectorE instruction, rounded
        into xnb's dtype (the stream dtype).  fp32 math inside both engines."""
        nisa.activation(dst=tb, op=nl.copy, data=x0b, scale=l2t)
        nisa.scalar_tensor_tensor(dst=xnb, data=xb, op0=nl.multiply, operand0=l1t, op1=nl.add, operand1=tb)

    def bheadk__sum_sq(xb, sq, ss, K):
        """ss[P, K] = sum_c xb[:, k, :]^2 in fp32 (K ScalarE square-reduces; sq is scratch of xb's dtype)."""
        for k in range(K):
            nisa.activation_reduce(dst=sq, op=nl.square, data=xb[:, k, :], reduce_op=nl.add, reduce_res=ss[:, k:k + 1])

    def bheadk__row_rsqrt(ss, inv_dst, C):
        """inv_dst[P, K] = rsqrt(ss / C + eps) (in place over ss for the intermediate)."""
        nisa.tensor_scalar(dst=ss, data=ss, op0=nl.multiply, operand0=1.0 / C, op1=nl.add, operand1=bheadk_RMS_EPS)
        nisa.tensor_scalar(dst=inv_dst, data=ss, op0=nl.rsqrt, operand0=0.0)

    def bheadk__scale_rows(dst, src, inv, K):
        """dst[:, k, :] = src[:, k, :] * inv[:, k] (K VectorE tensor_scalars with a [P, 1] operand; one rounding)."""
        for k in range(K):
            nisa.tensor_scalar(dst=dst[:, k, :], data=src[:, k, :], op0=nl.multiply, operand0=inv[:, k:k + 1])

    def bheadk__reduce_rows(pb, dst, K):
        """dst[P, K] = sum_c pb[:, k, :] in fp32: K ScalarE copy-reduces in place (act) or one VectorE reduce."""
        for k in range(K):
            nisa.activation_reduce(dst=pb[:, k, :], op=nl.copy, data=pb[:, k, :], reduce_op=nl.add, reduce_res=dst[:, k:k + 1])

    def bheadk__sq_shape(P, K, C):
        return (P, C)

    def bheadk__sq_dtype(dt):
        """SQ=vec reduces the STORED squares on VectorE, so they must be fp32; SQ=act reduces the fp32 activation
        output before the store, so the (discarded) stored squares can stay in the stream dtype (smaller scratch)."""
        return dt

    @nki.jit
    def bheadk__bh_fwd_kernel(x3, x03, l1g, l2g):
        """xn3 = l1*x3 + l2*x03, y3 = RMSNorm(xn3) (x3.dtype, [P, NT, C]); inv2[P, NT] fp32 = the per-row 1/rms."""
        P, NT, C = x3.shape
        dt = x3.dtype
        f32 = nl.float32
        xn3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        y3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        inv2 = nl.ndarray((P, NT), dtype=f32, buffer=nl.shared_hbm)
        i0, i1 = bheadk__row_range(NT)
        n = i1 - i0
        if n > 0:
            l1t = bheadk__load_gain(l1g, P)
            l2t = bheadk__load_gain(l2g, P)
            K = bheadk__rows_per_tile(n)
            nt = n // K
            mdt = bheadk__mix_dtype(dt)
            invb = bheadk__one((P, n), f32)
            xbs = bheadk__pair((P, K, C), dt)
            x0bs = bheadk__pair((P, K, C), dt)
            tbs = bheadk__pair((P, K, C), mdt)
            xnbs = bheadk__pair((P, K, C), dt)
            ybs = bheadk__pair((P, K, C), dt)
            sqs = bheadk__pair(bheadk__sq_shape(P, K, C), bheadk__sq_dtype(dt))
            bheadk__dma(dst=xbs[0], src=x3[:, i0:i0 + K, :])
            bheadk__dma(dst=x0bs[0], src=x03[:, i0:i0 + K, :])
            for j in range(nt):
                a = i0 + j * K
                s = j % 2
                xb, x0b, tb, xnb, yb, sq = (xbs[s], x0bs[s], tbs[s], xnbs[s], ybs[s], sqs[s])
                if j + 1 < nt:
                    s1 = (j + 1) % 2
                    bheadk__dma(dst=xbs[s1], src=x3[:, a + K:a + 2 * K, :])
                    bheadk__dma(dst=x0bs[s1], src=x03[:, a + K:a + 2 * K, :])
                bheadk__mix(xb, x0b, tb, xnb, l1t, l2t)
                ss = bheadk__one((P, K), f32)
                bheadk__sum_sq(xnb, sq, ss, K)
                bheadk__row_rsqrt(ss, invb[:, j * K:(j + 1) * K], C)
                bheadk__scale_rows(yb, xnb, invb[:, j * K:(j + 1) * K], K)
                bheadk__dma(dst=xn3[:, a:a + K, :], src=xnb)
                bheadk__dma(dst=y3[:, a:a + K, :], src=yb)
            bheadk__dma(dst=inv2[:, i0:i1], src=invb)
        return (xn3, y3, inv2)

    @nki.jit
    def bheadk__bh_bwd_kernel(x3, x03, gn3, gy3, inv2, l1g, l2g):
        """gx3 = l1 * G, gx03 = l2 * G (x3.dtype) with G = gn + inv * (gy - xn * (gy . xn) * inv^2 / C), xn recomputed.
        GAIN_GRAD=rows: also r3[2, P, NT] fp32 = per-row sum_c G*x / G*x0 (no partition reduction in the kernel);
        GAIN_GRAD=matmul: also g3 = G rounded to x3.dtype."""
        P, NT, C = x3.shape
        dt = x3.dtype
        f32 = nl.float32
        pdt = bheadk__prod_dtype(dt)
        mdt = bheadk__mix_dtype(dt)
        gx3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        gx03 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        r3 = nl.ndarray((2, P, NT), dtype=f32, buffer=nl.shared_hbm)
        i0, i1 = bheadk__row_range(NT)
        n = i1 - i0
        if n > 0:
            l1t = bheadk__load_gain(l1g, P)
            l2t = bheadk__load_gain(l2g, P)
            K = bheadk__rows_per_tile(n)
            nt = n // K
            invb = bheadk__one((P, n), f32)
            bheadk__dma(dst=invb, src=inv2[:, i0:i1])
            r1b = bheadk__one((P, n), f32)
            r2b = bheadk__one((P, n), f32)
            xbs = bheadk__pair((P, K, C), dt)
            x0bs = bheadk__pair((P, K, C), dt)
            gnbs = bheadk__pair((P, K, C), dt)
            gybs = bheadk__pair((P, K, C), dt)
            tbs = bheadk__pair((P, K, C), mdt)
            xnbs = bheadk__pair((P, K, C), dt)
            ubs = bheadk__pair((P, K, C), f32)
            pbs = bheadk__pair((P, K, C), pdt)
            if mdt != pdt and bheadk__GAIN_GRAD != 'matmul':
                p2bs = bheadk__pair((P, K, C), pdt)
            else:
                p2bs = tbs
            gxbs, gx0bs = (xnbs, gnbs)
            bheadk__dma(dst=xbs[0], src=x3[:, i0:i0 + K, :])
            bheadk__dma(dst=x0bs[0], src=x03[:, i0:i0 + K, :])
            bheadk__dma(dst=gnbs[0], src=gn3[:, i0:i0 + K, :])
            bheadk__dma(dst=gybs[0], src=gy3[:, i0:i0 + K, :])
            for j in range(nt):
                a = i0 + j * K
                s = j % 2
                xb, x0b, gnb, gyb = (xbs[s], x0bs[s], gnbs[s], gybs[s])
                tb, xnb, ub, pb, p2b = (tbs[s], xnbs[s], ubs[s], pbs[s], p2bs[s])
                gxb, gx0b = (gxbs[s], gx0bs[s])
                if j + 1 < nt:
                    s1 = (j + 1) % 2
                    bheadk__dma(dst=xbs[s1], src=x3[:, a + K:a + 2 * K, :])
                    bheadk__dma(dst=x0bs[s1], src=x03[:, a + K:a + 2 * K, :])
                    bheadk__dma(dst=gnbs[s1], src=gn3[:, a + K:a + 2 * K, :])
                    bheadk__dma(dst=gybs[s1], src=gy3[:, a + K:a + 2 * K, :])
                invs = invb[:, j * K:(j + 1) * K]
                bheadk__mix(xb, x0b, tb, xnb, l1t, l2t)
                nisa.tensor_tensor(dst=pb, data1=gyb, data2=xnb, op=nl.multiply)
                dot = bheadk__one((P, K), f32)
                bheadk__reduce_rows(pb, dot, K)
                nc = bheadk__one((P, K), f32)
                nisa.tensor_tensor(dst=nc, data1=invs, data2=invs, op=nl.multiply)
                nisa.tensor_tensor(dst=nc, data1=nc, data2=dot, op=nl.multiply)
                nisa.tensor_scalar(dst=nc, data=nc, op0=nl.multiply, operand0=-1.0 / C)
                for k in range(K):
                    nisa.scalar_tensor_tensor(dst=ub[:, k, :], data=xnb[:, k, :], op0=nl.multiply, operand0=nc[:, k:k + 1], op1=nl.add, operand1=gyb[:, k, :])
                    nisa.scalar_tensor_tensor(dst=ub[:, k, :], data=ub[:, k, :], op0=nl.multiply, operand0=invs[:, k:k + 1], op1=nl.add, operand1=gnb[:, k, :])
                nisa.tensor_scalar(dst=gxb, data=ub, op0=nl.multiply, operand0=l1t)
                nisa.tensor_scalar(dst=gx0b, data=ub, op0=nl.multiply, operand0=l2t)
                bheadk__dma(dst=gx3[:, a:a + K, :], src=gxb)
                bheadk__dma(dst=gx03[:, a:a + K, :], src=gx0b)
                nisa.tensor_tensor(dst=pb, data1=ub, data2=xb, op=nl.multiply)
                nisa.tensor_tensor(dst=p2b, data1=ub, data2=x0b, op=nl.multiply)
                bheadk__reduce_rows(pb, r1b[:, j * K:(j + 1) * K], K)
                bheadk__reduce_rows(p2b, r2b[:, j * K:(j + 1) * K], K)
            bheadk__dma(dst=r3[0, :, i0:i1], src=r1b)
            bheadk__dma(dst=r3[1, :, i0:i1], src=r2b)
        return (gx3, gx03, r3)

    @nki.jit
    def bheadk__rms_fwd_kernel(x3):
        """y3 = RMSNorm(x3) (x3.dtype, [P, NT, C]); inv2[P, NT] fp32."""
        P, NT, C = x3.shape
        dt = x3.dtype
        f32 = nl.float32
        y3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        inv2 = nl.ndarray((P, NT), dtype=f32, buffer=nl.shared_hbm)
        i0, i1 = bheadk__row_range(NT)
        n = i1 - i0
        if n > 0:
            K = bheadk__rows_per_tile(n)
            nt = n // K
            invb = bheadk__one((P, n), f32)
            xbs = bheadk__pair((P, K, C), dt)
            ybs = bheadk__pair((P, K, C), dt)
            sqs = bheadk__pair(bheadk__sq_shape(P, K, C), bheadk__sq_dtype(dt))
            bheadk__dma(dst=xbs[0], src=x3[:, i0:i0 + K, :])
            for j in range(nt):
                a = i0 + j * K
                s = j % 2
                xb, yb, sq = (xbs[s], ybs[s], sqs[s])
                if j + 1 < nt:
                    bheadk__dma(dst=xbs[(j + 1) % 2], src=x3[:, a + K:a + 2 * K, :])
                ss = bheadk__one((P, K), f32)
                bheadk__sum_sq(xb, sq, ss, K)
                bheadk__row_rsqrt(ss, invb[:, j * K:(j + 1) * K], C)
                bheadk__scale_rows(yb, xb, invb[:, j * K:(j + 1) * K], K)
                bheadk__dma(dst=y3[:, a:a + K, :], src=yb)
            bheadk__dma(dst=inv2[:, i0:i1], src=invb)
        return (y3, inv2)

    @nki.jit
    def bheadk__rms_bwd_kernel(x3, gy3, inv2):
        """gx3 = inv * (gy - x * (gy . x) * inv^2 / C) = x * nc + gy * inv with nc = -dot * inv^3 / C."""
        P, NT, C = x3.shape
        dt = x3.dtype
        f32 = nl.float32
        pdt = bheadk__prod_dtype(dt)
        gx3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        i0, i1 = bheadk__row_range(NT)
        n = i1 - i0
        if n > 0:
            K = bheadk__rows_per_tile(n)
            nt = n // K
            invb = bheadk__one((P, n), f32)
            bheadk__dma(dst=invb, src=inv2[:, i0:i1])
            xbs = bheadk__pair((P, K, C), dt)
            gybs = bheadk__pair((P, K, C), dt)
            pbs = bheadk__pair((P, K, C), pdt)
            gibs = bheadk__pair((P, K, C), f32)
            gxbs = gybs
            bheadk__dma(dst=xbs[0], src=x3[:, i0:i0 + K, :])
            bheadk__dma(dst=gybs[0], src=gy3[:, i0:i0 + K, :])
            for j in range(nt):
                a = i0 + j * K
                s = j % 2
                xb, gyb, pb, gib, gxb = (xbs[s], gybs[s], pbs[s], gibs[s], gxbs[s])
                if j + 1 < nt:
                    s1 = (j + 1) % 2
                    bheadk__dma(dst=xbs[s1], src=x3[:, a + K:a + 2 * K, :])
                    bheadk__dma(dst=gybs[s1], src=gy3[:, a + K:a + 2 * K, :])
                invs = invb[:, j * K:(j + 1) * K]
                nisa.tensor_tensor(dst=pb, data1=gyb, data2=xb, op=nl.multiply)
                dot = bheadk__one((P, K), f32)
                bheadk__reduce_rows(pb, dot, K)
                nc = bheadk__one((P, K), f32)
                nisa.tensor_tensor(dst=nc, data1=invs, data2=invs, op=nl.multiply)
                nisa.tensor_tensor(dst=nc, data1=nc, data2=invs, op=nl.multiply)
                nisa.tensor_tensor(dst=nc, data1=nc, data2=dot, op=nl.multiply)
                nisa.tensor_scalar(dst=nc, data=nc, op0=nl.multiply, operand0=-1.0 / C)
                for k in range(K):
                    nisa.activation(dst=gib[:, k, :], op=nl.copy, data=gyb[:, k, :], scale=invs[:, k:k + 1])
                for k in range(K):
                    nisa.scalar_tensor_tensor(dst=gxb[:, k, :], data=xb[:, k, :], op0=nl.multiply, operand0=nc[:, k:k + 1], op1=nl.add, operand1=gib[:, k, :])
                bheadk__dma(dst=gx3[:, a:a + K, :], src=gxb)
        return gx3
    bheadk__HAS_NKI = True
except Exception:
    bheadk__HAS_NKI = False

def bheadk__disable_kernel(where: str, err: BaseException) -> None:
    global bheadk__KERNEL_DISABLED, bheadk__LAST_KERNEL_ERROR
    if not bheadk__KERNEL_DISABLED:
        bheadk__KERNEL_DISABLED = True
        bheadk__LAST_KERNEL_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[nki_block_head] kernel disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)

def bheadk__launch(kernel):
    """kernel[2] under an LNC=2 runtime (both physical cores run half the rows), plain call otherwise."""
    return kernel[bheadk__LNC] if bheadk__LNC == 2 else kernel

def bheadk__v3(x2: torch.Tensor) -> torch.Tensor:
    """[N, C] contiguous -> the kernel's [P, NT, C] view (row n = p * NT + i)."""
    N, C = x2.shape
    return x2.view(bheadk__P, N // bheadk__P, C)

def bheadk__prep_gain(l: torch.Tensor) -> torch.Tensor:
    """(1,) fp32 gain -> the kernel's HBM operand: a (1, 1) view (bcast: the kernel's DMA broadcasts it across the
    partitions; no launch) or a stride-0 [P, 1] view (expand)."""
    if l.dtype != torch.float32:
        l = l.float()
    v = l.reshape(1, 1)
    return v

def bheadk__gain_scalar(lg: torch.Tensor) -> torch.Tensor:
    """Inverse of _prep_gain for the eager fallback: any layout -> a (1,) fp32 tensor."""
    return lg.reshape(-1)[0:1]

class bheadk_BlockHeadFn(torch.autograd.Function):
    """(x_new, x_norm) on 2-D [N, C] contiguous inputs via the NKI kernels; eager math if a kernel raises."""

    @staticmethod
    def forward(ctx, x2, x02, l1, l2):
        N, C = x2.shape
        l1g = bheadk__prep_gain(l1)
        l2g = bheadk__prep_gain(l2)
        inv2 = None
        try:
            xn3, y3, inv2 = bheadk__launch(bheadk__bh_fwd_kernel)(bheadk__v3(x2), bheadk__v3(x02), l1g, l2g)
            x_new, x_norm = (xn3.view(N, C), y3.view(N, C))
        except Exception as err:
            bheadk__disable_kernel('forward', err)
            with torch.no_grad():
                x_new, x_norm = bheadk_eager_block_head(x2, x02, bheadk__gain_scalar(l1g), bheadk__gain_scalar(l2g))
        ctx.save_for_backward(x2, x02, l1g, l2g, inv2)
        return (x_new, x_norm)

    @staticmethod
    def backward(ctx, gn, gy):
        x2, x02, l1g, l2g, inv2 = ctx.saved_tensors
        N, C = x2.shape
        if inv2 is not None and (not bheadk__KERNEL_DISABLED):
            try:
                gn2 = gn.reshape(N, C)
                gy2 = gy.reshape(N, C)
                if gn2.dtype != x2.dtype:
                    gn2 = gn2.to(x2.dtype)
                if gy2.dtype != x2.dtype:
                    gy2 = gy2.to(x2.dtype)
                if not gn2.is_contiguous():
                    gn2 = gn2.contiguous()
                if not gy2.is_contiguous():
                    gy2 = gy2.contiguous()
                out = bheadk__launch(bheadk__bh_bwd_kernel)(bheadk__v3(x2), bheadk__v3(x02), bheadk__v3(gn2), bheadk__v3(gy2), inv2, l1g, l2g)
                gx3, gx03, extra = out
                gl = extra.view(2, -1).sum(dim=1)
                gl1, gl2 = (gl[0:1], gl[1:2])
                return (gx3.view(N, C), gx03.view(N, C), gl1, gl2)
            except Exception as err:
                bheadk__disable_kernel('backward', err)
        with torch.enable_grad():
            xr = x2.detach().requires_grad_(True)
            x0r = x02.detach().requires_grad_(True)
            l1r = bheadk__gain_scalar(l1g).detach().requires_grad_(True)
            l2r = bheadk__gain_scalar(l2g).detach().requires_grad_(True)
            x_new, x_norm = bheadk_eager_block_head(xr, x0r, l1r, l2r)
            gx, gx0, gl1, gl2 = torch.autograd.grad((x_new, x_norm), (xr, x0r, l1r, l2r), (gn, gy))
        return (gx, gx0, gl1, gl2)

class bheadk_RmsNormFn(torch.autograd.Function):
    """y = RMSNorm(x) on a 2-D [N, C] contiguous input via the NKI kernels; eager math if a kernel raises."""

    @staticmethod
    def forward(ctx, x2):
        N, C = x2.shape
        inv2 = None
        try:
            y3, inv2 = bheadk__launch(bheadk__rms_fwd_kernel)(bheadk__v3(x2))
            y = y3.view(N, C)
        except Exception as err:
            bheadk__disable_kernel('rms forward', err)
            with torch.no_grad():
                y = bheadk_norm(x2)
        ctx.save_for_backward(x2, inv2)
        return y

    @staticmethod
    def backward(ctx, gy):
        x2, inv2 = ctx.saved_tensors
        N, C = x2.shape
        if inv2 is not None and (not bheadk__KERNEL_DISABLED):
            try:
                gy2 = gy.reshape(N, C)
                if gy2.dtype != x2.dtype:
                    gy2 = gy2.to(x2.dtype)
                if not gy2.is_contiguous():
                    gy2 = gy2.contiguous()
                gx3 = bheadk__launch(bheadk__rms_bwd_kernel)(bheadk__v3(x2), bheadk__v3(gy2), inv2)
                return gx3.view(N, C)
            except Exception as err:
                bheadk__disable_kernel('rms backward', err)
        with torch.enable_grad():
            xr = x2.detach().requires_grad_(True)
            gx, = torch.autograd.grad(bheadk_norm(xr), xr, gy)
        return gx

def bheadk__rows_ok(x: torch.Tensor) -> bool:
    N = x.numel() // x.shape[-1]
    return N > 0 and N % bheadk__P == 0

def bheadk_block_head(x: torch.Tensor, x0: torch.Tensor, l1: torch.Tensor, l2: torch.Tensor):
    """(l1*x + l2*x0, norm(l1*x + l2*x0)) -- one NKI launch when possible, train.py's eager chain otherwise."""
    use_kernel = bheadk__HAS_NKI and (not bheadk__KERNEL_DISABLED) and (x.device.type == 'neuron') and (x.dim() >= 2) and (x.dtype in (torch.bfloat16, torch.float32)) and (x0.dtype == x.dtype) and (x0.shape == x.shape) and (l1.numel() == 1) and (l2.numel() == 1) and bheadk__rows_ok(x)
    if not use_kernel:
        return bheadk_eager_block_head(x, x0, l1, l2)
    C = x.shape[-1]
    N = x.numel() // C
    x2 = x.reshape(N, C)
    x02 = x0.reshape(N, C)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if not x02.is_contiguous():
        x02 = x02.contiguous()
    x_new, x_norm = bheadk_BlockHeadFn.apply(x2, x02, l1, l2)
    return (x_new.view(x.shape), x_norm.view(x.shape))

def bheadk_rmsnorm_fast(x: torch.Tensor) -> torch.Tensor:
    """F.rms_norm(x, (C,)) -- the two-core NKI RMSNorm when R1_NKI_BLOCK_HEAD_NORM2=1 and it applies, else eager."""
    use_kernel = bheadk__HAS_NKI and (not bheadk__KERNEL_DISABLED) and (x.device.type == 'neuron') and (x.dim() >= 2) and (x.dtype in (torch.bfloat16, torch.float32)) and bheadk__rows_ok(x)
    if not use_kernel:
        return bheadk_norm(x)
    C = x.shape[-1]
    x2 = x.reshape(-1, C)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    return bheadk_RmsNormFn.apply(x2).view(x.shape)
block_head = bheadk_block_head
USE_NKI_BLOCK_HEAD = bheadk_USE_NKI_BLOCK_HEAD
USE_NKI_NORM2 = bheadk_USE_NKI_NORM2
'Megablock v1 (plan-v10 L4 / brief C) + v2 (phase 2, work/agents/megablock2: the MLP-tail residual add folded into\nthe next block\'s head, x0 / encoder-output gradient accumulates folded into the head backward; see the phase-2\n\nv1: two more fused stream kernels next to the block head (nki_block_head), both\ntwo-core NKI launches forward and backward, folding the U-net skip add into the head and the attention residual add +\nthe block\'s second RMSNorm (+ its backward fork accumulate) into one tail kernel.\n\nReplaces, for a DECODER block (train.py GPT.forward 794-804 + Block.forward 557-580 on the R193 stack):\n\n    x  = x + skip_lambda[j:j+1].to(dt) * skip           # GPT.forward: cast + mul + add        (3 launches fwd, 4 bwd)\n    x, xn = block_head(x, x0, l1, l2)                   # the T4 v2 kernel (kept for encoder / middle blocks)\n    x  = x + attn(xn)                                   # eager add                             (1 fwd, 1 accumulate bwd)\n    xn2 = rmsnorm_fast(x)                               # the T4 v2 plain-norm kernel\n    x  = x + mlp(xn2)                                   # stays eager (phase 2 / T15\'s epilogue)\n\nwith\n\n    x, xn  = head_skip(x, x0, skip, l1, l2, l3)         # s\' = x + l3*skip ; x_new = l1*s\' + l2*x0 ; xn = rmsnorm(x_new)\n    x, xn2 = tail_norm2(x, attn(xn))                    # s1 = x_new + attn_out ; xn2 = rmsnorm(s1)\n    x  = x + mlp(xn2)\n\nPublic entry points\n-------------------\n    x_new, xn = head_skip(x, x0, skip, l1, l2, l3)\n        skip None (encoder / middle blocks)  -> nki_block_head.block_head(x, x0, l1, l2), untouched\n        kernel not applicable / switch off   -> block_head(x + l3.to(dt) * skip, x0, l1, l2): today\'s op sequence exactly\n        kernel: forward s\' = x + l3*skip kept in fp32 (eager rounds l3 to bf16 and rounds s\' to bf16),\n                x_new = bf16(l1*s\' + l2*x0) (ONE rounding), xn = rmsnorm(x_new) of the ROUNDED x_new (eager semantics);\n                backward with G = gn + J_rmsnorm^T(gy):  grad_x = l1*G, grad_x0 = l2*G, grad_skip = (l1*l3)*G,\n                grad_l1 = sum(G*s\'), grad_l2 = sum(G*x0), grad_l3 = l1 * sum(G*skip); the three gain gradients leave\n                the kernel as [3, 128, NT] fp32 per-row partials finished by ONE torch .sum (no partition reduction).\n    s1, xn2 = tail_norm2(x_new, attn_out)\n        switch off / not applicable -> (x_new + attn_out, rmsnorm_fast(x_new + attn_out)): today\'s path exactly\n        kernel: forward s1 = bf16(x_new + attn_out) (== eager\'s add, bit for bit), xn2 = rmsnorm(s1), inv saved;\n                backward folds the residual fork: g = ge + inv*(gy - s1*(gy.s1)*inv^2/C) in fp32, rounded once, and\n                the SAME tensor is returned as the gradient of both inputs (autograd accumulates each on its own).\n\nKernel design: the block-head module\'s proven layout and instruction forms only (rows viewed [128, NT, C] with\n8 KiB-per-partition contiguous slab DMAs at MAX_K=4, kernel[2] on contiguous row halves, gains broadcast inside a\n512-byte DMA per program, ScalarE activation_reduce row sums, exact tensor_scalar rsqrt, VectorE\nscalar_tensor_tensor for every "a*s + b" pass, fp32 per-row partials, explicit 2-deep prefetch, no comprehensions).\nNew instruction uses, each an already-compiled form: the skip add as the STT `s\' = skip*l3 + x` with an fp32 dst\n(the block head\'s `u` form: bf16 data, [P,1] fp32 operand0, bf16 operand1), l1*l3 as a [P,1] tensor_tensor, the\nper-row gain partials finished on the [P, n] blocks once per program (grad_l1 rows = rows(G*x) + l3*rows(G*skip),\ngrad_l3 rows = l1*rows(G*skip)), the tail\'s fork add as the STT `gi = gy*inv + ge` (fp32 dst) that feeds the\nrms-backward STT.  In the backward s\' is formed in the (still dead) fp32 G buffer, so no extra tile: SBUF at K=4 =\n96 KiB per buffer set, two sets = 192 KiB (blockhead: 176); forward 80 KiB per set; MAX_K=2 halves both (knob\nR1_NKI_MEGABLOCK_MAX_K).\n\ndtypes; a kernel failure prints once, disables both kernels of this module for the process and answers with the\neager math.  The block-head module\'s own switch and knobs are untouched.\n\nKnobs (env, read once at import; module-level constants because kernel bodies only see module globals):\n  shared with nki_block_head (same env names): R1_NKI_BLOCK_HEAD_{MIX_TMP, PROD_DTYPE, REDUCE, SQ, RSQRT, INPLACE,\n  PREFETCH, GAIN, LNC, SHARD, DGE, EPS}\n'
import os
import torch
import torch.nn.functional as F
try:
    mblk_block_head = bheadk_block_head
    mblk_rmsnorm_fast = bheadk_rmsnorm_fast
except Exception:
    mblk_block_head = None
    mblk_rmsnorm_fast = None
mblk__P = 128
mblk__env = os.environ.get
mblk_RMS_EPS = float(2.0 ** (-23))
mblk_USE_NKI_MEGABLOCK = True
mblk__MAX_K = 4
mblk__PROD_DTYPE = 'fp32'
mblk__REDUCE = 'act'
mblk__SQ = 'act'
mblk__RSQRT = 'ts'
mblk__INPLACE = True
mblk__LNC = int(mblk__env('NEURON_LOGICAL_NC_CONFIG', '1'))
mblk__SHARD = True
mblk__DGE = 'auto'
mblk__TAIL_GI = 'vec'
mblk__ADD_ENGINE = 'unknown'
mblk__GAIN_STASH: dict = {}
mblk__GAIN_KEY = None
mblk__KERNEL_DISABLED = False
mblk__LAST_KERNEL_ERROR = None

def mblk_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))

def mblk_eager_block_head(x, x0, l1, l2):
    x_new = l1.to(x.dtype) * x + l2.to(x.dtype) * x0
    return (x_new, mblk_norm(x_new))

def mblk_eager_head_skip(x, x0, skip, l1, l2, l3):
    """GPT.forward's skip add followed by the Block head, all eager (the (1,) gains cast to the stream dtype)."""
    s = x + l3.to(x.dtype) * skip
    return mblk_eager_block_head(s, x0, l1, l2)

def mblk_eager_tail(x_new, attn_out):
    s1 = x_new + attn_out
    return (s1, mblk_norm(s1))

def mblk__today_head(x, x0, l1, l2):
    """The block head as train.py runs it today (the T4 v2 kernel when its switch is on, eager otherwise)."""
    if mblk_block_head is not None:
        return mblk_block_head(x, x0, l1, l2)
    return mblk_eager_block_head(x, x0, l1, l2)

def mblk__today_norm2(x):
    if mblk_rmsnorm_fast is not None:
        return mblk_rmsnorm_fast(x)
    return mblk_norm(x)
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    mblk__ADD_ENG = nisa.engine.unknown

    def mblk__dma(dst, src):
        nisa.dma_copy(dst=dst, src=src)

    def mblk__shard():
        if nl.program_ndim() > 0:
            return (nl.program_id(0), nl.num_programs(0))
        return (0, 1)

    def mblk__row_range(NT):
        pid, nprg = mblk__shard()
        per, rem = (NT // nprg, NT % nprg)
        i0 = pid * per + min(pid, rem)
        n = per + (1 if pid < rem else 0)
        return (i0, i0 + n)

    def mblk__rows_per_tile(n):
        k = 1
        for d in range(1, mblk__MAX_K + 1):
            if n % d == 0:
                k = d
        return k

    def mblk__pair(shape, dtype):
        return (nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf), nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf))

    def mblk__one(shape, dtype):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    def mblk__load_gain(lg, P):
        t = mblk__one((P, 1), nl.float32)
        if lg.shape[0] == 1:
            mblk__dma(dst=t, src=lg.broadcast(0, P))
        else:
            mblk__dma(dst=t, src=lg)
        return t

    def mblk__mix_dtype(dt):
        return nl.float32

    def mblk__prod_dtype(dt):
        return nl.float32

    def mblk__mix(xb, x0b, tb, xnb, l1t, l2t):
        """t = l2 * x0 ; x_new = (x * l1) + t in ONE VectorE instruction, rounded into xnb's dtype."""
        nisa.activation(dst=tb, op=nl.copy, data=x0b, scale=l2t)
        nisa.scalar_tensor_tensor(dst=xnb, data=xb, op0=nl.multiply, operand0=l1t, op1=nl.add, operand1=tb)

    def mblk__skip_add(dst, skb, xb, l3t):
        """dst = skb * l3 + xb (the act16-style STT: bf16 data, [P,1] fp32 operand0, stream-dtype operand1)."""
        nisa.scalar_tensor_tensor(dst=dst, data=skb, op0=nl.multiply, operand0=l3t, op1=nl.add, operand1=xb)

    def mblk__sum_sq(xb, sq, ss, K):
        for k in range(K):
            nisa.activation_reduce(dst=sq, op=nl.square, data=xb[:, k, :], reduce_op=nl.add, reduce_res=ss[:, k:k + 1])

    def mblk__row_rsqrt(ss, inv_dst, C):
        nisa.tensor_scalar(dst=ss, data=ss, op0=nl.multiply, operand0=1.0 / C, op1=nl.add, operand1=mblk_RMS_EPS)
        nisa.tensor_scalar(dst=inv_dst, data=ss, op0=nl.rsqrt, operand0=0.0)

    def mblk__scale_rows(dst, src, inv, K):
        for k in range(K):
            nisa.tensor_scalar(dst=dst[:, k, :], data=src[:, k, :], op0=nl.multiply, operand0=inv[:, k:k + 1])

    def mblk__reduce_rows(pb, dst, K):
        for k in range(K):
            nisa.activation_reduce(dst=pb[:, k, :], op=nl.copy, data=pb[:, k, :], reduce_op=nl.add, reduce_res=dst[:, k:k + 1])

    def mblk__sq_shape(P, K, C):
        return (P, C)

    def mblk__sq_dtype(dt):
        return dt

    @nki.jit
    def mblk__hs_fwd_kernel(x3, x03, sk3, l1g, l2g, l3g):
        """xn3 = l1*(x3 + l3*sk3) + l2*x03, y3 = RMSNorm(xn3) (x3.dtype, [P, NT, C]); inv2[P, NT] fp32."""
        P, NT, C = x3.shape
        dt = x3.dtype
        f32 = nl.float32
        xn3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        y3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        inv2 = nl.ndarray((P, NT), dtype=f32, buffer=nl.shared_hbm)
        i0, i1 = mblk__row_range(NT)
        n = i1 - i0
        if n > 0:
            l1t = mblk__load_gain(l1g, P)
            l2t = mblk__load_gain(l2g, P)
            l3t = mblk__load_gain(l3g, P)
            K = mblk__rows_per_tile(n)
            nt = n // K
            mdt = mblk__mix_dtype(dt)
            invb = mblk__one((P, n), f32)
            xbs = mblk__pair((P, K, C), dt)
            x0bs = mblk__pair((P, K, C), dt)
            skbs = mblk__pair((P, K, C), dt)
            sfbs = mblk__pair((P, K, C), f32)
            tbs = mblk__pair((P, K, C), mdt)
            xnbs = mblk__pair((P, K, C), dt)
            ybs = mblk__pair((P, K, C), dt)
            sqs = mblk__pair(mblk__sq_shape(P, K, C), mblk__sq_dtype(dt))
            mblk__dma(dst=xbs[0], src=x3[:, i0:i0 + K, :])
            mblk__dma(dst=x0bs[0], src=x03[:, i0:i0 + K, :])
            mblk__dma(dst=skbs[0], src=sk3[:, i0:i0 + K, :])
            for j in range(nt):
                a = i0 + j * K
                s = j % 2
                xb, x0b, skb, sfb, tb, xnb, yb, sq = (xbs[s], x0bs[s], skbs[s], sfbs[s], tbs[s], xnbs[s], ybs[s], sqs[s])
                if j + 1 < nt:
                    s1 = (j + 1) % 2
                    mblk__dma(dst=xbs[s1], src=x3[:, a + K:a + 2 * K, :])
                    mblk__dma(dst=x0bs[s1], src=x03[:, a + K:a + 2 * K, :])
                    mblk__dma(dst=skbs[s1], src=sk3[:, a + K:a + 2 * K, :])
                mblk__skip_add(sfb, skb, xb, l3t)
                mblk__mix(sfb, x0b, tb, xnb, l1t, l2t)
                ss = mblk__one((P, K), f32)
                mblk__sum_sq(xnb, sq, ss, K)
                mblk__row_rsqrt(ss, invb[:, j * K:(j + 1) * K], C)
                mblk__scale_rows(yb, xnb, invb[:, j * K:(j + 1) * K], K)
                mblk__dma(dst=xn3[:, a:a + K, :], src=xnb)
                mblk__dma(dst=y3[:, a:a + K, :], src=yb)
            mblk__dma(dst=inv2[:, i0:i1], src=invb)
        return (xn3, y3, inv2)

    @nki.jit
    def mblk__hs_bwd_kernel(x3, x03, sk3, gn3, gy3, inv2, l1g, l2g, l3g):
        """gx3 = l1*G, gx03 = l2*G, gsk3 = (l1*l3)*G (x3.dtype); r3[3, P, NT] fp32 = per-row sums of G*s', G*x0 and
        l1 * G*skip (the three gain gradients, finished by one torch .sum in the wrapper)."""
        P, NT, C = x3.shape
        dt = x3.dtype
        f32 = nl.float32
        pdt = mblk__prod_dtype(dt)
        mdt = mblk__mix_dtype(dt)
        gx3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        gx03 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        gsk3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        r3 = nl.ndarray((3, P, NT), dtype=f32, buffer=nl.shared_hbm)
        i0, i1 = mblk__row_range(NT)
        n = i1 - i0
        if n > 0:
            l1t = mblk__load_gain(l1g, P)
            l2t = mblk__load_gain(l2g, P)
            l3t = mblk__load_gain(l3g, P)
            l13t = mblk__one((P, 1), f32)
            nisa.tensor_tensor(dst=l13t, data1=l1t, data2=l3t, op=nl.multiply)
            K = mblk__rows_per_tile(n)
            nt = n // K
            invb = mblk__one((P, n), f32)
            mblk__dma(dst=invb, src=inv2[:, i0:i1])
            rxb = mblk__one((P, n), f32)
            rskb = mblk__one((P, n), f32)
            r1b = mblk__one((P, n), f32)
            r2b = mblk__one((P, n), f32)
            r3b = mblk__one((P, n), f32)
            xbs = mblk__pair((P, K, C), dt)
            x0bs = mblk__pair((P, K, C), dt)
            skbs = mblk__pair((P, K, C), dt)
            gnbs = mblk__pair((P, K, C), dt)
            gybs = mblk__pair((P, K, C), dt)
            tbs = mblk__pair((P, K, C), mdt)
            xnbs = mblk__pair((P, K, C), dt)
            ubs = mblk__pair((P, K, C), f32)
            pbs = mblk__pair((P, K, C), pdt)
            if mdt != pdt:
                p2bs = mblk__pair((P, K, C), pdt)
            else:
                p2bs = tbs
            gxbs, gx0bs, gskbs = (xnbs, gnbs, x0bs)
            mblk__dma(dst=xbs[0], src=x3[:, i0:i0 + K, :])
            mblk__dma(dst=x0bs[0], src=x03[:, i0:i0 + K, :])
            mblk__dma(dst=skbs[0], src=sk3[:, i0:i0 + K, :])
            mblk__dma(dst=gnbs[0], src=gn3[:, i0:i0 + K, :])
            mblk__dma(dst=gybs[0], src=gy3[:, i0:i0 + K, :])
            for j in range(nt):
                a = i0 + j * K
                s = j % 2
                xb, x0b, skb, gnb, gyb = (xbs[s], x0bs[s], skbs[s], gnbs[s], gybs[s])
                tb, xnb, ub, pb, p2b = (tbs[s], xnbs[s], ubs[s], pbs[s], p2bs[s])
                gxb, gx0b, gskb = (gxbs[s], gx0bs[s], gskbs[s])
                if j + 1 < nt:
                    s1 = (j + 1) % 2
                    mblk__dma(dst=xbs[s1], src=x3[:, a + K:a + 2 * K, :])
                    mblk__dma(dst=x0bs[s1], src=x03[:, a + K:a + 2 * K, :])
                    mblk__dma(dst=skbs[s1], src=sk3[:, a + K:a + 2 * K, :])
                    mblk__dma(dst=gnbs[s1], src=gn3[:, a + K:a + 2 * K, :])
                    mblk__dma(dst=gybs[s1], src=gy3[:, a + K:a + 2 * K, :])
                invs = invb[:, j * K:(j + 1) * K]
                mblk__skip_add(ub, skb, xb, l3t)
                mblk__mix(ub, x0b, tb, xnb, l1t, l2t)
                nisa.tensor_tensor(dst=pb, data1=gyb, data2=xnb, op=nl.multiply)
                dot = mblk__one((P, K), f32)
                mblk__reduce_rows(pb, dot, K)
                nc = mblk__one((P, K), f32)
                nisa.tensor_tensor(dst=nc, data1=invs, data2=invs, op=nl.multiply)
                nisa.tensor_tensor(dst=nc, data1=nc, data2=dot, op=nl.multiply)
                nisa.tensor_scalar(dst=nc, data=nc, op0=nl.multiply, operand0=-1.0 / C)
                for k in range(K):
                    nisa.scalar_tensor_tensor(dst=ub[:, k, :], data=xnb[:, k, :], op0=nl.multiply, operand0=nc[:, k:k + 1], op1=nl.add, operand1=gyb[:, k, :])
                    nisa.scalar_tensor_tensor(dst=ub[:, k, :], data=ub[:, k, :], op0=nl.multiply, operand0=invs[:, k:k + 1], op1=nl.add, operand1=gnb[:, k, :])
                nisa.tensor_tensor(dst=pb, data1=ub, data2=xb, op=nl.multiply)
                nisa.tensor_tensor(dst=p2b, data1=ub, data2=x0b, op=nl.multiply)
                mblk__reduce_rows(pb, rxb[:, j * K:(j + 1) * K], K)
                mblk__reduce_rows(p2b, r2b[:, j * K:(j + 1) * K], K)
                nisa.tensor_tensor(dst=pb, data1=ub, data2=skb, op=nl.multiply)
                mblk__reduce_rows(pb, rskb[:, j * K:(j + 1) * K], K)
                nisa.tensor_scalar(dst=gxb, data=ub, op0=nl.multiply, operand0=l1t)
                nisa.tensor_scalar(dst=gx0b, data=ub, op0=nl.multiply, operand0=l2t)
                nisa.tensor_scalar(dst=gskb, data=ub, op0=nl.multiply, operand0=l13t)
                mblk__dma(dst=gx3[:, a:a + K, :], src=gxb)
                mblk__dma(dst=gx03[:, a:a + K, :], src=gx0b)
                mblk__dma(dst=gsk3[:, a:a + K, :], src=gskb)
            nisa.scalar_tensor_tensor(dst=r1b, data=rskb, op0=nl.multiply, operand0=l3t, op1=nl.add, operand1=rxb)
            nisa.tensor_scalar(dst=r3b, data=rskb, op0=nl.multiply, operand0=l1t)
            mblk__dma(dst=r3[0, :, i0:i1], src=r1b)
            mblk__dma(dst=r3[1, :, i0:i1], src=r2b)
            mblk__dma(dst=r3[2, :, i0:i1], src=r3b)
        return (gx3, gx03, gsk3, r3)

    @nki.jit
    def mblk__tail_fwd_kernel(xn3, ao3):
        """s3 = xn3 + ao3 (one rounding == eager), y3 = RMSNorm(s3); inv2[P, NT] fp32."""
        P, NT, C = xn3.shape
        dt = xn3.dtype
        f32 = nl.float32
        s3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        y3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        inv2 = nl.ndarray((P, NT), dtype=f32, buffer=nl.shared_hbm)
        i0, i1 = mblk__row_range(NT)
        n = i1 - i0
        if n > 0:
            K = mblk__rows_per_tile(n)
            nt = n // K
            invb = mblk__one((P, n), f32)
            xbs = mblk__pair((P, K, C), dt)
            abs_ = mblk__pair((P, K, C), dt)
            sbs = mblk__pair((P, K, C), dt)
            ybs = mblk__pair((P, K, C), dt)
            sqs = mblk__pair(mblk__sq_shape(P, K, C), mblk__sq_dtype(dt))
            mblk__dma(dst=xbs[0], src=xn3[:, i0:i0 + K, :])
            mblk__dma(dst=abs_[0], src=ao3[:, i0:i0 + K, :])
            for j in range(nt):
                a = i0 + j * K
                s = j % 2
                xb, ab, sb, yb, sq = (xbs[s], abs_[s], sbs[s], ybs[s], sqs[s])
                if j + 1 < nt:
                    s1 = (j + 1) % 2
                    mblk__dma(dst=xbs[s1], src=xn3[:, a + K:a + 2 * K, :])
                    mblk__dma(dst=abs_[s1], src=ao3[:, a + K:a + 2 * K, :])
                nisa.tensor_tensor(dst=sb, data1=xb, data2=ab, op=nl.add, engine=mblk__ADD_ENG)
                ss = mblk__one((P, K), f32)
                mblk__sum_sq(sb, sq, ss, K)
                mblk__row_rsqrt(ss, invb[:, j * K:(j + 1) * K], C)
                mblk__scale_rows(yb, sb, invb[:, j * K:(j + 1) * K], K)
                mblk__dma(dst=s3[:, a:a + K, :], src=sb)
                mblk__dma(dst=y3[:, a:a + K, :], src=yb)
            mblk__dma(dst=inv2[:, i0:i1], src=invb)
        return (s3, y3, inv2)

    @nki.jit
    def mblk__tail_bwd_kernel(s3, gy3, ge3, inv2):
        """g3 = ge + inv*(gy - s*(gy . s)*inv^2/C) = s*nc + (gy*inv + ge), nc = -dot*inv^3/C; one rounding."""
        P, NT, C = s3.shape
        dt = s3.dtype
        f32 = nl.float32
        pdt = mblk__prod_dtype(dt)
        g3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        i0, i1 = mblk__row_range(NT)
        n = i1 - i0
        if n > 0:
            K = mblk__rows_per_tile(n)
            nt = n // K
            invb = mblk__one((P, n), f32)
            mblk__dma(dst=invb, src=inv2[:, i0:i1])
            sbs = mblk__pair((P, K, C), dt)
            gybs = mblk__pair((P, K, C), dt)
            gebs = mblk__pair((P, K, C), dt)
            pbs = mblk__pair((P, K, C), pdt)
            gibs = mblk__pair((P, K, C), f32)
            gbs = gybs
            mblk__dma(dst=sbs[0], src=s3[:, i0:i0 + K, :])
            mblk__dma(dst=gybs[0], src=gy3[:, i0:i0 + K, :])
            mblk__dma(dst=gebs[0], src=ge3[:, i0:i0 + K, :])
            for j in range(nt):
                a = i0 + j * K
                s = j % 2
                sb, gyb, geb, pb, gib, gb = (sbs[s], gybs[s], gebs[s], pbs[s], gibs[s], gbs[s])
                if j + 1 < nt:
                    s1 = (j + 1) % 2
                    mblk__dma(dst=sbs[s1], src=s3[:, a + K:a + 2 * K, :])
                    mblk__dma(dst=gybs[s1], src=gy3[:, a + K:a + 2 * K, :])
                    mblk__dma(dst=gebs[s1], src=ge3[:, a + K:a + 2 * K, :])
                invs = invb[:, j * K:(j + 1) * K]
                nisa.tensor_tensor(dst=pb, data1=gyb, data2=sb, op=nl.multiply)
                dot = mblk__one((P, K), f32)
                mblk__reduce_rows(pb, dot, K)
                nc = mblk__one((P, K), f32)
                nisa.tensor_tensor(dst=nc, data1=invs, data2=invs, op=nl.multiply)
                nisa.tensor_tensor(dst=nc, data1=nc, data2=invs, op=nl.multiply)
                nisa.tensor_tensor(dst=nc, data1=nc, data2=dot, op=nl.multiply)
                nisa.tensor_scalar(dst=nc, data=nc, op0=nl.multiply, operand0=-1.0 / C)
                for k in range(K):
                    nisa.scalar_tensor_tensor(dst=gib[:, k, :], data=gyb[:, k, :], op0=nl.multiply, operand0=invs[:, k:k + 1], op1=nl.add, operand1=geb[:, k, :])
                for k in range(K):
                    nisa.scalar_tensor_tensor(dst=gb[:, k, :], data=sb[:, k, :], op0=nl.multiply, operand0=nc[:, k:k + 1], op1=nl.add, operand1=gib[:, k, :])
                mblk__dma(dst=g3[:, a:a + K, :], src=gb)
        return g3
    mblk__HAS_NKI = True
except Exception:
    mblk__HAS_NKI = False

def mblk__disable_kernel(where: str, err: BaseException) -> None:
    global mblk__KERNEL_DISABLED, mblk__LAST_KERNEL_ERROR
    if not mblk__KERNEL_DISABLED:
        mblk__KERNEL_DISABLED = True
        mblk__LAST_KERNEL_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[nki_megablock] kernel disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)

def mblk__launch(kernel):
    return kernel[mblk__LNC] if mblk__LNC == 2 else kernel

def mblk__v3(x2: torch.Tensor) -> torch.Tensor:
    N, C = x2.shape
    return x2.view(mblk__P, N // mblk__P, C)

def mblk__prep_gain(l: torch.Tensor) -> torch.Tensor:
    if l.dtype != torch.float32:
        l = l.float()
    v = l.reshape(1, 1)
    return v

def mblk__gain_scalar(lg: torch.Tensor) -> torch.Tensor:
    return lg.reshape(-1)[0:1]

def mblk__as2d(g, like):
    """A gradient (or None -> zeros) as a contiguous [N, C] tensor of the input's dtype."""
    N, C = like.shape
    if g is None:
        return torch.zeros_like(like)
    g2 = g.reshape(N, C)
    if g2.dtype != like.dtype:
        g2 = g2.to(like.dtype)
    if not g2.is_contiguous():
        g2 = g2.contiguous()
    return g2

class mblk_HeadSkipFn(torch.autograd.Function):
    """(x_new, x_norm) = head(x + l3*skip, x0) on 2-D [N, C] contiguous inputs via the NKI kernels; eager if a kernel raises."""

    @staticmethod
    def forward(ctx, x2, x02, sk2, l1, l2, l3):
        N, C = x2.shape
        l1g, l2g, l3g = (mblk__prep_gain(l1), mblk__prep_gain(l2), mblk__prep_gain(l3))
        inv2 = None
        try:
            xn3, y3, inv2 = mblk__launch(mblk__hs_fwd_kernel)(mblk__v3(x2), mblk__v3(x02), mblk__v3(sk2), l1g, l2g, l3g)
            x_new, x_norm = (xn3.view(N, C), y3.view(N, C))
        except Exception as err:
            mblk__disable_kernel('head_skip forward', err)
            with torch.no_grad():
                x_new, x_norm = mblk_eager_head_skip(x2, x02, sk2, mblk__gain_scalar(l1g), mblk__gain_scalar(l2g), mblk__gain_scalar(l3g))
        ctx.save_for_backward(x2, x02, sk2, l1g, l2g, l3g, inv2)
        return (x_new, x_norm)

    @staticmethod
    def backward(ctx, gn, gy):
        x2, x02, sk2, l1g, l2g, l3g, inv2 = ctx.saved_tensors
        N, C = x2.shape
        if inv2 is not None and (not mblk__KERNEL_DISABLED):
            try:
                gn2, gy2 = (mblk__as2d(gn, x2), mblk__as2d(gy, x2))
                gx3, gx03, gsk3, r3 = mblk__launch(mblk__hs_bwd_kernel)(mblk__v3(x2), mblk__v3(x02), mblk__v3(sk2), mblk__v3(gn2), mblk__v3(gy2), inv2, l1g, l2g, l3g)
                gl = r3.view(3, -1).sum(dim=1)
                return (gx3.view(N, C), gx03.view(N, C), gsk3.view(N, C), gl[0:1], gl[1:2], gl[2:3])
            except Exception as err:
                mblk__disable_kernel('head_skip backward', err)
        with torch.enable_grad():
            xr = x2.detach().requires_grad_(True)
            x0r = x02.detach().requires_grad_(True)
            skr = sk2.detach().requires_grad_(True)
            l1r = mblk__gain_scalar(l1g).detach().requires_grad_(True)
            l2r = mblk__gain_scalar(l2g).detach().requires_grad_(True)
            l3r = mblk__gain_scalar(l3g).detach().requires_grad_(True)
            x_new, x_norm = mblk_eager_head_skip(xr, x0r, skr, l1r, l2r, l3r)
            gn_ = torch.zeros_like(x_new) if gn is None else gn
            gy_ = torch.zeros_like(x_norm) if gy is None else gy
            grads = torch.autograd.grad((x_new, x_norm), (xr, x0r, skr, l1r, l2r, l3r), (gn_, gy_))
        return grads

class mblk_TailNorm2Fn(torch.autograd.Function):
    """(s1, xn2) = (x_new + attn_out, RMSNorm(s1)) on 2-D [N, C] contiguous inputs; the backward returns ONE tensor
    as the gradient of both inputs (g_s1 = g_ext + norm backward)."""

    @staticmethod
    def forward(ctx, xn2, ao2):
        N, C = xn2.shape
        inv2 = None
        try:
            s3, y3, inv2 = mblk__launch(mblk__tail_fwd_kernel)(mblk__v3(xn2), mblk__v3(ao2))
            s1, y = (s3.view(N, C), y3.view(N, C))
        except Exception as err:
            mblk__disable_kernel('tail forward', err)
            with torch.no_grad():
                s1, y = mblk_eager_tail(xn2, ao2)
        ctx.save_for_backward(s1, inv2)
        return (s1, y)

    @staticmethod
    def backward(ctx, ge, gy):
        s1, inv2 = ctx.saved_tensors
        N, C = s1.shape
        if inv2 is not None and (not mblk__KERNEL_DISABLED):
            try:
                ge2, gy2 = (mblk__as2d(ge, s1), mblk__as2d(gy, s1))
                g3 = mblk__launch(mblk__tail_bwd_kernel)(mblk__v3(s1), mblk__v3(gy2), mblk__v3(ge2), inv2)
                g = g3.view(N, C)
                return (g, g)
            except Exception as err:
                mblk__disable_kernel('tail backward', err)
        with torch.enable_grad():
            sr = s1.detach().requires_grad_(True)
            gy_ = torch.zeros_like(s1) if gy is None else gy
            gs, = torch.autograd.grad(mblk_norm(sr), sr, gy_)
        g = gs if ge is None else gs + ge
        return (g, g)

def mblk__rows_ok(x: torch.Tensor) -> bool:
    N = x.numel() // x.shape[-1]
    return N > 0 and N % mblk__P == 0

def mblk__ok_dtype(x: torch.Tensor) -> bool:
    return x.dtype in (torch.bfloat16, torch.float32)

def mblk__c2d(x: torch.Tensor, C: int) -> torch.Tensor:
    x2 = x.reshape(-1, C)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    return x2

def mblk_head_skip(x, x0, skip, l1, l2, l3):
    """Block head with the U-net skip folded in. skip None -> the plain block head (encoder / middle blocks)."""
    if skip is None:
        return mblk__today_head(x, x0, l1, l2)
    use_kernel = mblk__HAS_NKI and (not mblk__KERNEL_DISABLED) and (x.device.type == 'neuron') and (x.dim() >= 2) and mblk__ok_dtype(x) and (x0.dtype == x.dtype) and (skip.dtype == x.dtype) and (x0.shape == x.shape) and (skip.shape == x.shape) and (l1.numel() == 1) and (l2.numel() == 1) and (l3.numel() == 1) and mblk__rows_ok(x)
    if not use_kernel:
        return mblk__today_head(x + l3.to(x.dtype) * skip, x0, l1, l2)
    C = x.shape[-1]
    x_new, x_norm = mblk_HeadSkipFn.apply(mblk__c2d(x, C), mblk__c2d(x0, C), mblk__c2d(skip, C), l1, l2, l3)
    return (x_new.view(x.shape), x_norm.view(x.shape))

def mblk_tail_norm2(x_new, attn_out):
    """(x_new + attn_out, norm(x_new + attn_out)) -- one launch forward / one backward when the kernel applies."""
    use_kernel = mblk__HAS_NKI and (not mblk__KERNEL_DISABLED) and (x_new.device.type == 'neuron') and (x_new.dim() >= 2) and mblk__ok_dtype(x_new) and (attn_out.dtype == x_new.dtype) and (attn_out.shape == x_new.shape) and mblk__rows_ok(x_new)
    if not use_kernel:
        s1 = x_new + attn_out
        return (s1, mblk__today_norm2(s1))
    C = x_new.shape[-1]
    s1, y = mblk_TailNorm2Fn.apply(mblk__c2d(x_new, C), mblk__c2d(attn_out, C))
    return (s1.view(x_new.shape), y.view(x_new.shape))
mblk_USE_NKI_MEGABLOCK_TAIL = True
mblk__MAX_K2 = int(str(mblk__MAX_K))
mblk__SBUF_LIMIT2 = int(str(200 * 1024))
mblk_M_X0, mblk_M_NONE, mblk_M_SELF, mblk_M_EXT = (0, 1, 2, 3)

class mblk__SelfSkip:
    """Sentinel: the block's skip operand is its own (not yet materialised) input."""

    def __repr__(self):
        return 'SELF_SKIP'
mblk_SELF_SKIP = mblk__SelfSkip()
mblk__KERNEL2_DISABLED = False
mblk__LAST_KERNEL2_ERROR = None

def mblk__disable_kernel2(where: str, err: BaseException) -> None:
    global mblk__KERNEL2_DISABLED, mblk__LAST_KERNEL2_ERROR
    if not mblk__KERNEL2_DISABLED:
        mblk__KERNEL2_DISABLED = True
        mblk__LAST_KERNEL2_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[nki_megablock] phase-2 kernels disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)

def mblk_eager_head2(mode, s1, mo, x0, sk, l1, l2, l3):
    """(x_new, x_norm, x_in) with the eager ops: x_in = s1 + mo (the block's input), then the phase-1 eager head."""
    if mode == mblk_M_X0:
        x_new, x_norm = mblk_eager_block_head(x0, x0, l1, l2)
        return (x_new, x_norm, x0)
    x_in = s1 + mo
    if mode == mblk_M_NONE:
        x_new, x_norm = mblk_eager_block_head(x_in, x0, l1, l2)
    elif mode == mblk_M_SELF:
        x_new, x_norm = mblk_eager_head_skip(x_in, x0, x_in, l1, l2, l3)
    else:
        x_new, x_norm = mblk_eager_head_skip(x_in, x0, sk, l1, l2, l3)
    return (x_new, x_norm, x_in)
mblk__HAS_NKI2 = False
try:
    if not mblk__HAS_NKI:
        raise ImportError('nki not importable')
    try:
        mblk__bh_rms_bwd_kernel = bheadk__rms_bwd_kernel
    except Exception:
        mblk__bh_rms_bwd_kernel = None

    def mblk__isz(dt):
        if dt == nl.float32:
            return 4
        return 2

    def mblk__k_budget(n, C, dt, n_in, n_ext, fwd, mode):
        """Largest divisor K of n (<= _MAX_K2) whose double-buffered SBUF estimate fits _SBUF_LIMIT2 bytes per partition."""
        isz = mblk__isz(dt)
        k = 1
        for d in range(1, mblk__MAX_K2 + 1):
            if n % d == 0:
                if fwd:
                    per_set = d * C * (isz * n_in + 4 + 4 + isz + isz) + C * isz
                else:
                    per_set = d * C * (isz * (n_in + 2) + 4 + isz + 4)
                    per_set = per_set + d * C * 4
                tot = 2 * per_set + n_ext * d * C * isz + 4096
                if tot <= mblk__SBUF_LIMIT2:
                    k = d
        return k

    def mblk__rows_of(ub, other, pb, dst_block, K):
        """dst_block[P, K] = per-row sums of G * other (VectorE product into pb, K ScalarE copy-reduces)."""
        nisa.tensor_tensor(dst=pb, data1=ub, data2=other, op=nl.multiply)
        mblk__reduce_rows(pb, dst_block, K)

    def mblk__ext_mul(dst, ub, lt, extb):
        """dst = lt * G (+ ext), rounded once into dst's dtype (whole-tile VectorE pass)."""
        if extb is None:
            nisa.tensor_scalar(dst=dst, data=ub, op0=nl.multiply, operand0=lt)
        else:
            nisa.scalar_tensor_tensor(dst=dst, data=ub, op0=nl.multiply, operand0=lt, op1=nl.add, operand1=extb)

    def mblk__n_in(mode):
        if mode == mblk_M_X0:
            return 1
        if mode == mblk_M_EXT:
            return 4
        return 3

    def mblk__h2f_body(mode, emit, s13, mo3, x03, sk3, l1g, l2g, l3g, xn3, y3, inv2, xin3):
        P, NT, C = x03.shape
        dt = x03.dtype
        f32 = nl.float32
        i0, i1 = mblk__row_range(NT)
        n = i1 - i0
        if n > 0:
            l1t = mblk__load_gain(l1g, P)
            l2t = mblk__load_gain(l2g, P)
            if mode >= mblk_M_SELF:
                l3t = mblk__load_gain(l3g, P)
            K = mblk__k_budget(n, C, dt, mblk__n_in(mode), 0, True, mode)
            nt = n // K
            mdt = mblk__mix_dtype(dt)
            invb = mblk__one((P, n), f32)
            x0bs = mblk__pair((P, K, C), dt)
            if mode >= mblk_M_NONE:
                s1bs = mblk__pair((P, K, C), dt)
                mobs = mblk__pair((P, K, C), dt)
            if mode == mblk_M_EXT:
                skbs = mblk__pair((P, K, C), dt)
            if mode >= mblk_M_SELF:
                sfbs = mblk__pair((P, K, C), f32)
            tbs = mblk__pair((P, K, C), mdt)
            xnbs = mblk__pair((P, K, C), dt)
            ybs = mblk__pair((P, K, C), dt)
            sqs = mblk__pair(mblk__sq_shape(P, K, C), mblk__sq_dtype(dt))
            mblk__dma(dst=x0bs[0], src=x03[:, i0:i0 + K, :])
            if mode >= mblk_M_NONE:
                mblk__dma(dst=s1bs[0], src=s13[:, i0:i0 + K, :])
                mblk__dma(dst=mobs[0], src=mo3[:, i0:i0 + K, :])
            if mode == mblk_M_EXT:
                mblk__dma(dst=skbs[0], src=sk3[:, i0:i0 + K, :])
            for j in range(nt):
                a = i0 + j * K
                s = j % 2
                x0b, tb, xnb, yb, sq = (x0bs[s], tbs[s], xnbs[s], ybs[s], sqs[s])
                if j + 1 < nt:
                    s1_ = (j + 1) % 2
                    mblk__dma(dst=x0bs[s1_], src=x03[:, a + K:a + 2 * K, :])
                    if mode >= mblk_M_NONE:
                        mblk__dma(dst=s1bs[s1_], src=s13[:, a + K:a + 2 * K, :])
                        mblk__dma(dst=mobs[s1_], src=mo3[:, a + K:a + 2 * K, :])
                    if mode == mblk_M_EXT:
                        mblk__dma(dst=skbs[s1_], src=sk3[:, a + K:a + 2 * K, :])
                if mode == mblk_M_X0:
                    xb = x0b
                else:
                    xb = s1bs[s]
                    nisa.tensor_tensor(dst=xb, data1=xb, data2=mobs[s], op=nl.add, engine=mblk__ADD_ENG)
                    if emit:
                        mblk__dma(dst=xin3[:, a:a + K, :], src=xb)
                if mode == mblk_M_SELF:
                    mblk__skip_add(sfbs[s], xb, xb, l3t)
                    mblk__mix(sfbs[s], x0b, tb, xnb, l1t, l2t)
                elif mode == mblk_M_EXT:
                    mblk__skip_add(sfbs[s], skbs[s], xb, l3t)
                    mblk__mix(sfbs[s], x0b, tb, xnb, l1t, l2t)
                else:
                    mblk__mix(xb, x0b, tb, xnb, l1t, l2t)
                ss = mblk__one((P, K), f32)
                mblk__sum_sq(xnb, sq, ss, K)
                mblk__row_rsqrt(ss, invb[:, j * K:(j + 1) * K], C)
                mblk__scale_rows(yb, xnb, invb[:, j * K:(j + 1) * K], K)
                mblk__dma(dst=xn3[:, a:a + K, :], src=xnb)
                mblk__dma(dst=y3[:, a:a + K, :], src=yb)
            mblk__dma(dst=inv2[:, i0:i1], src=invb)

    def mblk__h2b_body(mode, s13, mo3, x03, sk3, gn3, gy3, inv2, x0e3, xine3, l1g, l2g, l3g, gxin3, gx03, gsk3, r3):
        P, NT, C = x03.shape
        dt = x03.dtype
        f32 = nl.float32
        mdt = mblk__mix_dtype(dt)
        has_x0e = x0e3 is not None
        has_xine = xine3 is not None
        i0, i1 = mblk__row_range(NT)
        n = i1 - i0
        if n > 0:
            l1t = mblk__load_gain(l1g, P)
            l2t = mblk__load_gain(l2g, P)
            if mode == mblk_M_X0:
                l12t = mblk__one((P, 1), f32)
                nisa.tensor_tensor(dst=l12t, data1=l1t, data2=l2t, op=nl.add)
            if mode >= mblk_M_SELF:
                l3t = mblk__load_gain(l3g, P)
                l13t = mblk__one((P, 1), f32)
                nisa.tensor_tensor(dst=l13t, data1=l1t, data2=l3t, op=nl.multiply)
            if mode == mblk_M_SELF:
                lst = mblk__one((P, 1), f32)
                nisa.tensor_tensor(dst=lst, data1=l13t, data2=l1t, op=nl.add)
                l3p = mblk__one((P, 1), f32)
                nisa.tensor_scalar(dst=l3p, data=l3t, op0=nl.add, operand0=1.0)
            n_ext = 0
            if has_x0e:
                n_ext = n_ext + 1
            if has_xine:
                n_ext = n_ext + 1
            K = mblk__k_budget(n, C, dt, mblk__n_in(mode), n_ext, False, mode)
            nt = n // K
            invb = mblk__one((P, n), f32)
            mblk__dma(dst=invb, src=inv2[:, i0:i1])
            rab = mblk__one((P, n), f32)
            if mode >= mblk_M_NONE:
                r2b = mblk__one((P, n), f32)
            if mode == mblk_M_EXT:
                rskb = mblk__one((P, n), f32)
            if mode >= mblk_M_SELF:
                r1b = mblk__one((P, n), f32)
                r3b = mblk__one((P, n), f32)
            x0bs = mblk__pair((P, K, C), dt)
            gnbs = mblk__pair((P, K, C), dt)
            gybs = mblk__pair((P, K, C), dt)
            if mode >= mblk_M_NONE:
                s1bs = mblk__pair((P, K, C), dt)
                mobs = mblk__pair((P, K, C), dt)
            if mode == mblk_M_EXT:
                skbs = mblk__pair((P, K, C), dt)
            tbs = mblk__pair((P, K, C), mdt)
            xnbs = mblk__pair((P, K, C), dt)
            ubs = mblk__pair((P, K, C), f32)
            pbs = mblk__pair((P, K, C), f32)
            if has_x0e:
                x0eb = mblk__one((P, K, C), dt)
            else:
                x0eb = None
            if has_xine:
                xineb = mblk__one((P, K, C), dt)
            else:
                xineb = None
            mblk__dma(dst=x0bs[0], src=x03[:, i0:i0 + K, :])
            mblk__dma(dst=gnbs[0], src=gn3[:, i0:i0 + K, :])
            mblk__dma(dst=gybs[0], src=gy3[:, i0:i0 + K, :])
            if mode >= mblk_M_NONE:
                mblk__dma(dst=s1bs[0], src=s13[:, i0:i0 + K, :])
                mblk__dma(dst=mobs[0], src=mo3[:, i0:i0 + K, :])
            if mode == mblk_M_EXT:
                mblk__dma(dst=skbs[0], src=sk3[:, i0:i0 + K, :])
            for j in range(nt):
                a = i0 + j * K
                s = j % 2
                x0b, gnb, gyb, tb, xnb, ub = (x0bs[s], gnbs[s], gybs[s], tbs[s], xnbs[s], ubs[s])
                if has_x0e:
                    mblk__dma(dst=x0eb, src=x0e3[:, a:a + K, :])
                if has_xine:
                    mblk__dma(dst=xineb, src=xine3[:, a:a + K, :])
                if j + 1 < nt:
                    s1_ = (j + 1) % 2
                    mblk__dma(dst=x0bs[s1_], src=x03[:, a + K:a + 2 * K, :])
                    mblk__dma(dst=gnbs[s1_], src=gn3[:, a + K:a + 2 * K, :])
                    mblk__dma(dst=gybs[s1_], src=gy3[:, a + K:a + 2 * K, :])
                    if mode >= mblk_M_NONE:
                        mblk__dma(dst=s1bs[s1_], src=s13[:, a + K:a + 2 * K, :])
                        mblk__dma(dst=mobs[s1_], src=mo3[:, a + K:a + 2 * K, :])
                    if mode == mblk_M_EXT:
                        mblk__dma(dst=skbs[s1_], src=sk3[:, a + K:a + 2 * K, :])
                invs = invb[:, j * K:(j + 1) * K]
                if mode == mblk_M_X0:
                    xb = x0b
                else:
                    xb = s1bs[s]
                    nisa.tensor_tensor(dst=xb, data1=xb, data2=mobs[s], op=nl.add, engine=mblk__ADD_ENG)
                if mode == mblk_M_SELF:
                    mblk__skip_add(ub, xb, xb, l3t)
                    mblk__mix(ub, x0b, tb, xnb, l1t, l2t)
                elif mode == mblk_M_EXT:
                    mblk__skip_add(ub, skbs[s], xb, l3t)
                    mblk__mix(ub, x0b, tb, xnb, l1t, l2t)
                else:
                    mblk__mix(xb, x0b, tb, xnb, l1t, l2t)
                pd = pbs[s]
                pa = pbs[s]
                if mdt == f32:
                    pb2 = tb
                else:
                    pb2 = pbs[s]
                nisa.tensor_tensor(dst=pd, data1=gyb, data2=xnb, op=nl.multiply)
                dot = mblk__one((P, K), f32)
                mblk__reduce_rows(pd, dot, K)
                nc = mblk__one((P, K), f32)
                nisa.tensor_tensor(dst=nc, data1=invs, data2=invs, op=nl.multiply)
                nisa.tensor_tensor(dst=nc, data1=nc, data2=dot, op=nl.multiply)
                nisa.tensor_scalar(dst=nc, data=nc, op0=nl.multiply, operand0=-1.0 / C)
                for k in range(K):
                    nisa.scalar_tensor_tensor(dst=ub[:, k, :], data=xnb[:, k, :], op0=nl.multiply, operand0=nc[:, k:k + 1], op1=nl.add, operand1=gyb[:, k, :])
                    nisa.scalar_tensor_tensor(dst=ub[:, k, :], data=ub[:, k, :], op0=nl.multiply, operand0=invs[:, k:k + 1], op1=nl.add, operand1=gnb[:, k, :])
                mblk__rows_of(ub, xb, pa, rab[:, j * K:(j + 1) * K], K)
                if mode >= mblk_M_NONE:
                    mblk__rows_of(ub, x0b, pb2, r2b[:, j * K:(j + 1) * K], K)
                if mode == mblk_M_EXT:
                    mblk__rows_of(ub, skbs[s], pa, rskb[:, j * K:(j + 1) * K], K)
                if mode == mblk_M_X0:
                    mblk__ext_mul(gnb, ub, l12t, x0eb)
                    mblk__dma(dst=gx03[:, a:a + K, :], src=gnb)
                else:
                    if mode == mblk_M_NONE:
                        mblk__ext_mul(xnb, ub, l1t, xineb)
                    elif mode == mblk_M_SELF:
                        mblk__ext_mul(xnb, ub, lst, None)
                    else:
                        mblk__ext_mul(xnb, ub, l1t, None)
                    mblk__dma(dst=gxin3[:, a:a + K, :], src=xnb)
                    mblk__ext_mul(gnb, ub, l2t, x0eb)
                    mblk__dma(dst=gx03[:, a:a + K, :], src=gnb)
                    if mode == mblk_M_EXT:
                        nisa.tensor_scalar(dst=x0b, data=ub, op0=nl.multiply, operand0=l13t)
                        mblk__dma(dst=gsk3[:, a:a + K, :], src=x0b)
            if mode == mblk_M_X0:
                mblk__dma(dst=r3[0, :, i0:i1], src=rab)
                mblk__dma(dst=r3[1, :, i0:i1], src=rab)
            elif mode == mblk_M_NONE:
                mblk__dma(dst=r3[0, :, i0:i1], src=rab)
                mblk__dma(dst=r3[1, :, i0:i1], src=r2b)
            elif mode == mblk_M_SELF:
                nisa.tensor_scalar(dst=r1b, data=rab, op0=nl.multiply, operand0=l3p)
                nisa.tensor_scalar(dst=r3b, data=rab, op0=nl.multiply, operand0=l1t)
                mblk__dma(dst=r3[0, :, i0:i1], src=r1b)
                mblk__dma(dst=r3[1, :, i0:i1], src=r2b)
                mblk__dma(dst=r3[2, :, i0:i1], src=r3b)
            else:
                nisa.scalar_tensor_tensor(dst=r1b, data=rskb, op0=nl.multiply, operand0=l3t, op1=nl.add, operand1=rab)
                nisa.tensor_scalar(dst=r3b, data=rskb, op0=nl.multiply, operand0=l1t)
                mblk__dma(dst=r3[0, :, i0:i1], src=r1b)
                mblk__dma(dst=r3[1, :, i0:i1], src=r2b)
                mblk__dma(dst=r3[2, :, i0:i1], src=r3b)

    def mblk__outs_f(x03, emit):
        P, NT, C = x03.shape
        xn3 = nl.ndarray((P, NT, C), dtype=x03.dtype, buffer=nl.shared_hbm)
        y3 = nl.ndarray((P, NT, C), dtype=x03.dtype, buffer=nl.shared_hbm)
        inv2 = nl.ndarray((P, NT), dtype=nl.float32, buffer=nl.shared_hbm)
        if emit:
            xin3 = nl.ndarray((P, NT, C), dtype=x03.dtype, buffer=nl.shared_hbm)
        else:
            xin3 = None
        return (xn3, y3, inv2, xin3)

    @nki.jit
    def mblk__h2f_x0(x03, l1g, l2g):
        xn3, y3, inv2, _ = mblk__outs_f(x03, False)
        mblk__h2f_body(mblk_M_X0, False, None, None, x03, None, l1g, l2g, None, xn3, y3, inv2, None)
        return (xn3, y3, inv2)

    @nki.jit
    def mblk__h2f_none(s13, mo3, x03, l1g, l2g):
        xn3, y3, inv2, _ = mblk__outs_f(x03, False)
        mblk__h2f_body(mblk_M_NONE, False, s13, mo3, x03, None, l1g, l2g, None, xn3, y3, inv2, None)
        return (xn3, y3, inv2)

    @nki.jit
    def mblk__h2f_none_emit(s13, mo3, x03, l1g, l2g):
        xn3, y3, inv2, xin3 = mblk__outs_f(x03, True)
        mblk__h2f_body(mblk_M_NONE, True, s13, mo3, x03, None, l1g, l2g, None, xn3, y3, inv2, xin3)
        return (xn3, y3, inv2, xin3)

    @nki.jit
    def mblk__h2f_self(s13, mo3, x03, l1g, l2g, l3g):
        xn3, y3, inv2, _ = mblk__outs_f(x03, False)
        mblk__h2f_body(mblk_M_SELF, False, s13, mo3, x03, None, l1g, l2g, l3g, xn3, y3, inv2, None)
        return (xn3, y3, inv2)

    @nki.jit
    def mblk__h2f_ext(s13, mo3, x03, sk3, l1g, l2g, l3g):
        xn3, y3, inv2, _ = mblk__outs_f(x03, False)
        mblk__h2f_body(mblk_M_EXT, False, s13, mo3, x03, sk3, l1g, l2g, l3g, xn3, y3, inv2, None)
        return (xn3, y3, inv2)

    def mblk__outs_b(x03, mode):
        P, NT, C = x03.shape
        dt = x03.dtype
        gx03 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        if mode == mblk_M_X0:
            gxin3 = None
        else:
            gxin3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        if mode == mblk_M_EXT:
            gsk3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        else:
            gsk3 = None
        if mode >= mblk_M_SELF:
            r3 = nl.ndarray((3, P, NT), dtype=nl.float32, buffer=nl.shared_hbm)
        else:
            r3 = nl.ndarray((2, P, NT), dtype=nl.float32, buffer=nl.shared_hbm)
        return (gxin3, gx03, gsk3, r3)

    @nki.jit
    def mblk__h2b_x0_1(x03, gn3, gy3, inv2, x0e3, l1g, l2g):
        _, gx03, _, r3 = mblk__outs_b(x03, mblk_M_X0)
        mblk__h2b_body(mblk_M_X0, None, None, x03, None, gn3, gy3, inv2, x0e3, None, l1g, l2g, None, None, gx03, None, r3)
        return (gx03, r3)

    @nki.jit
    def mblk__h2b_x0_0(x03, gn3, gy3, inv2, l1g, l2g):
        _, gx03, _, r3 = mblk__outs_b(x03, mblk_M_X0)
        mblk__h2b_body(mblk_M_X0, None, None, x03, None, gn3, gy3, inv2, None, None, l1g, l2g, None, None, gx03, None, r3)
        return (gx03, r3)

    @nki.jit
    def mblk__h2b_none_11(s13, mo3, x03, gn3, gy3, inv2, x0e3, xine3, l1g, l2g):
        gxin3, gx03, _, r3 = mblk__outs_b(x03, mblk_M_NONE)
        mblk__h2b_body(mblk_M_NONE, s13, mo3, x03, None, gn3, gy3, inv2, x0e3, xine3, l1g, l2g, None, gxin3, gx03, None, r3)
        return (gxin3, gx03, r3)

    @nki.jit
    def mblk__h2b_none_10(s13, mo3, x03, gn3, gy3, inv2, x0e3, l1g, l2g):
        gxin3, gx03, _, r3 = mblk__outs_b(x03, mblk_M_NONE)
        mblk__h2b_body(mblk_M_NONE, s13, mo3, x03, None, gn3, gy3, inv2, x0e3, None, l1g, l2g, None, gxin3, gx03, None, r3)
        return (gxin3, gx03, r3)

    @nki.jit
    def mblk__h2b_none_01(s13, mo3, x03, gn3, gy3, inv2, xine3, l1g, l2g):
        gxin3, gx03, _, r3 = mblk__outs_b(x03, mblk_M_NONE)
        mblk__h2b_body(mblk_M_NONE, s13, mo3, x03, None, gn3, gy3, inv2, None, xine3, l1g, l2g, None, gxin3, gx03, None, r3)
        return (gxin3, gx03, r3)

    @nki.jit
    def mblk__h2b_none_00(s13, mo3, x03, gn3, gy3, inv2, l1g, l2g):
        gxin3, gx03, _, r3 = mblk__outs_b(x03, mblk_M_NONE)
        mblk__h2b_body(mblk_M_NONE, s13, mo3, x03, None, gn3, gy3, inv2, None, None, l1g, l2g, None, gxin3, gx03, None, r3)
        return (gxin3, gx03, r3)

    @nki.jit
    def mblk__h2b_self_1(s13, mo3, x03, gn3, gy3, inv2, x0e3, l1g, l2g, l3g):
        gxin3, gx03, _, r3 = mblk__outs_b(x03, mblk_M_SELF)
        mblk__h2b_body(mblk_M_SELF, s13, mo3, x03, None, gn3, gy3, inv2, x0e3, None, l1g, l2g, l3g, gxin3, gx03, None, r3)
        return (gxin3, gx03, r3)

    @nki.jit
    def mblk__h2b_self_0(s13, mo3, x03, gn3, gy3, inv2, l1g, l2g, l3g):
        gxin3, gx03, _, r3 = mblk__outs_b(x03, mblk_M_SELF)
        mblk__h2b_body(mblk_M_SELF, s13, mo3, x03, None, gn3, gy3, inv2, None, None, l1g, l2g, l3g, gxin3, gx03, None, r3)
        return (gxin3, gx03, r3)

    @nki.jit
    def mblk__h2b_ext_1(s13, mo3, x03, sk3, gn3, gy3, inv2, x0e3, l1g, l2g, l3g):
        gxin3, gx03, gsk3, r3 = mblk__outs_b(x03, mblk_M_EXT)
        mblk__h2b_body(mblk_M_EXT, s13, mo3, x03, sk3, gn3, gy3, inv2, x0e3, None, l1g, l2g, l3g, gxin3, gx03, gsk3, r3)
        return (gxin3, gx03, gsk3, r3)

    @nki.jit
    def mblk__h2b_ext_0(s13, mo3, x03, sk3, gn3, gy3, inv2, l1g, l2g, l3g):
        gxin3, gx03, gsk3, r3 = mblk__outs_b(x03, mblk_M_EXT)
        mblk__h2b_body(mblk_M_EXT, s13, mo3, x03, sk3, gn3, gy3, inv2, None, None, l1g, l2g, l3g, gxin3, gx03, gsk3, r3)
        return (gxin3, gx03, gsk3, r3)
    mblk__FWD2 = {(mblk_M_X0, False): mblk__h2f_x0, (mblk_M_NONE, False): mblk__h2f_none, (mblk_M_NONE, True): mblk__h2f_none_emit, (mblk_M_SELF, False): mblk__h2f_self, (mblk_M_EXT, False): mblk__h2f_ext}
    mblk__BWD2 = {(mblk_M_X0, True, False): mblk__h2b_x0_1, (mblk_M_X0, False, False): mblk__h2b_x0_0, (mblk_M_NONE, True, True): mblk__h2b_none_11, (mblk_M_NONE, True, False): mblk__h2b_none_10, (mblk_M_NONE, False, True): mblk__h2b_none_01, (mblk_M_NONE, False, False): mblk__h2b_none_00, (mblk_M_SELF, True, False): mblk__h2b_self_1, (mblk_M_SELF, False, False): mblk__h2b_self_0, (mblk_M_EXT, True, False): mblk__h2b_ext_1, (mblk_M_EXT, False, False): mblk__h2b_ext_0}
    mblk__HAS_NKI2 = True
except Exception:
    mblk__HAS_NKI2 = False
    mblk__bh_rms_bwd_kernel = None
    mblk__FWD2 = {}
    mblk__BWD2 = {}

def mblk__launch2(key, table):
    return mblk__launch(table[key])

def mblk__as2d_opt(g, like):
    if g is None:
        return None
    return mblk__as2d(g, like)

def mblk__split_ts(mode, ts):
    """The Function's positional tensors by mode -> (s1, mo, x0, sk, l1, l2, l3) with None for the absent ones."""
    if mode == mblk_M_X0:
        x0, l1, l2 = ts
        return (None, None, x0, None, l1, l2, None)
    if mode == mblk_M_NONE:
        s1, mo, x0, l1, l2 = ts
        return (s1, mo, x0, None, l1, l2, None)
    if mode == mblk_M_SELF:
        s1, mo, x0, l1, l2, l3 = ts
        return (s1, mo, x0, None, l1, l2, l3)
    s1, mo, x0, sk, l1, l2, l3 = ts
    return (s1, mo, x0, sk, l1, l2, l3)

def mblk__gains_out(ctx, r3, k):
    """[glue d] The k gain gradients of one head backward: (1,) slices of one .sum launch (today), or k x None when
    the batched path stashes the fp32 partials r3 [k, P, NT] under the block's key for train.py's flush_gain_grads
    (one cat + one sum for every block of the micro-step)."""
    if ctx.key is not None:
        mblk__GAIN_STASH[ctx.key] = r3
        return (None,) * k
    gl = r3.view(k, -1).sum(dim=1)
    return tuple((gl[i:i + 1] for i in range(k)))

def mblk_gain_partials_take() -> dict:
    """[glue d] The stashed gain-gradient partials of the last backward, keyed by block; clears the stash."""
    out = dict(mblk__GAIN_STASH)
    mblk__GAIN_STASH.clear()
    return out

class mblk_Head2Fn(torch.autograd.Function):
    """The phase-2 head on 2-D [N, C] contiguous inputs.  apply(mode, emit, *tensors) with tensors by mode:
    X0 (x0, l1, l2) | NONE (s1, mo, x0, l1, l2) | SELF (s1, mo, x0, l1, l2, l3) | EXT (s1, mo, x0, sk, l1, l2, l3).
    Outputs (x_new, x_norm[, x_in when emit], x0_pass); x0_pass is the x0 input returned as an output, so the
    gradient of everything downstream that consumes it arrives here and is folded into grad_x0.  The gradient of
    x_in (the decoder's grad_skip) is folded into grad_x_in, which is returned once for both s1 and mo."""

    @staticmethod
    def forward(ctx, mode, emit, *ts):
        ctx.set_materialize_grads(False)
        s1, mo, x0, sk, l1, l2, l3 = mblk__split_ts(mode, ts)
        N, C = x0.shape
        l1g, l2g = (mblk__prep_gain(l1), mblk__prep_gain(l2))
        l3g = mblk__prep_gain(l3) if l3 is not None else None
        inv2 = None
        x_in = None
        try:
            if mode == mblk_M_X0:
                args = (mblk__v3(x0), l1g, l2g)
            elif mode == mblk_M_NONE:
                args = (mblk__v3(s1), mblk__v3(mo), mblk__v3(x0), l1g, l2g)
            elif mode == mblk_M_SELF:
                args = (mblk__v3(s1), mblk__v3(mo), mblk__v3(x0), l1g, l2g, l3g)
            else:
                args = (mblk__v3(s1), mblk__v3(mo), mblk__v3(x0), mblk__v3(sk), l1g, l2g, l3g)
            out = mblk__launch2((mode, bool(emit)), mblk__FWD2)(*args)
            if emit:
                xn3, y3, inv2, xin3 = out
                x_in = xin3.view(N, C)
            else:
                xn3, y3, inv2 = out
            x_new, x_norm = (xn3.view(N, C), y3.view(N, C))
        except Exception as err:
            mblk__disable_kernel2('head2 forward', err)
            inv2 = None
            with torch.no_grad():
                x_new, x_norm, x_in_e = mblk_eager_head2(mode, s1, mo, x0, sk, mblk__gain_scalar(l1g), mblk__gain_scalar(l2g), mblk__gain_scalar(l3g) if l3g is not None else None)
                x_in = x_in_e if emit else None
        ctx.mode, ctx.emit = (mode, bool(emit))
        ctx.key = mblk__GAIN_KEY
        ctx.save_for_backward(s1, mo, x0, sk, l1g, l2g, l3g, inv2)
        if emit:
            return (x_new, x_norm, x_in, x0)
        return (x_new, x_norm, x0)

    @staticmethod
    def backward(ctx, *grads):
        s1, mo, x0, sk, l1g, l2g, l3g, inv2 = ctx.saved_tensors
        mode, emit = (ctx.mode, ctx.emit)
        if emit:
            gn, gy, gxine, gx0e = grads
        else:
            gn, gy, gx0e = grads
            gxine = None
        N, C = x0.shape
        if inv2 is not None and (not mblk__KERNEL2_DISABLED):
            try:
                gn2, gy2 = (mblk__as2d(gn, x0), mblk__as2d(gy, x0))
                x0e2, xine2 = (mblk__as2d_opt(gx0e, x0), mblk__as2d_opt(gxine, x0))
                key = (mode, x0e2 is not None, xine2 is not None)
                if mode == mblk_M_X0:
                    args = (mblk__v3(x0), mblk__v3(gn2), mblk__v3(gy2), inv2) + ((mblk__v3(x0e2),) if x0e2 is not None else ()) + (l1g, l2g)
                    gx03, r3 = mblk__launch2(key, mblk__BWD2)(*args)
                    g1, g2 = mblk__gains_out(ctx, r3, 2)
                    return (None, None, gx03.view(N, C), g1, g2)
                if mode == mblk_M_NONE:
                    args = (mblk__v3(s1), mblk__v3(mo), mblk__v3(x0), mblk__v3(gn2), mblk__v3(gy2), inv2)
                    if x0e2 is not None:
                        args = args + (mblk__v3(x0e2),)
                    if xine2 is not None:
                        args = args + (mblk__v3(xine2),)
                    args = args + (l1g, l2g)
                    gxin3, gx03, r3 = mblk__launch2(key, mblk__BWD2)(*args)
                    g1, g2 = mblk__gains_out(ctx, r3, 2)
                    gxin = gxin3.view(N, C)
                    return (None, None, gxin, gxin, gx03.view(N, C), g1, g2)
                if mode == mblk_M_SELF:
                    args = (mblk__v3(s1), mblk__v3(mo), mblk__v3(x0), mblk__v3(gn2), mblk__v3(gy2), inv2) + ((mblk__v3(x0e2),) if x0e2 is not None else ()) + (l1g, l2g, l3g)
                    gxin3, gx03, r3 = mblk__launch2(key, mblk__BWD2)(*args)
                    g1, g2, g3 = mblk__gains_out(ctx, r3, 3)
                    gxin = gxin3.view(N, C)
                    return (None, None, gxin, gxin, gx03.view(N, C), g1, g2, g3)
                args = (mblk__v3(s1), mblk__v3(mo), mblk__v3(x0), mblk__v3(sk), mblk__v3(gn2), mblk__v3(gy2), inv2) + ((mblk__v3(x0e2),) if x0e2 is not None else ()) + (l1g, l2g, l3g)
                gxin3, gx03, gsk3, r3 = mblk__launch2(key, mblk__BWD2)(*args)
                g1, g2, g3 = mblk__gains_out(ctx, r3, 3)
                gxin = gxin3.view(N, C)
                return (None, None, gxin, gxin, gx03.view(N, C), gsk3.view(N, C), g1, g2, g3)
            except Exception as err:
                mblk__disable_kernel2('head2 backward', err)
        with torch.enable_grad():
            x0r = x0.detach().requires_grad_(True)
            l1r = mblk__gain_scalar(l1g).detach().requires_grad_(True)
            l2r = mblk__gain_scalar(l2g).detach().requires_grad_(True)
            leaves = [x0r, l1r, l2r]
            s1r = mor = skr = l3r = None
            if mode != mblk_M_X0:
                s1r = s1.detach().requires_grad_(True)
                mor = mo.detach().requires_grad_(True)
                leaves = [s1r, mor] + leaves
            if mode == mblk_M_EXT:
                skr = sk.detach().requires_grad_(True)
                leaves.append(skr)
            if mode >= mblk_M_SELF:
                l3r = mblk__gain_scalar(l3g).detach().requires_grad_(True)
                leaves.append(l3r)
            x_new, x_norm, x_in = mblk_eager_head2(mode, s1r, mor, x0r, skr, l1r, l2r, l3r)
            outs = [x_new, x_norm]
            gouts = [torch.zeros_like(x_new) if gn is None else gn, torch.zeros_like(x_norm) if gy is None else gy]
            if gxine is not None and mode != mblk_M_X0:
                outs.append(x_in)
                gouts.append(gxine)
            g = list(torch.autograd.grad(outs, leaves, gouts, allow_unused=True))
        g = [torch.zeros_like(t) if gg is None else gg for gg, t in zip(g, leaves)]
        if mode == mblk_M_X0:
            gx0, gl1, gl2 = g
            if gx0e is not None:
                gx0 = gx0 + gx0e
            return (None, None, gx0, gl1, gl2)
        if mode == mblk_M_NONE:
            gs1, gmo, gx0, gl1, gl2 = g
            if gx0e is not None:
                gx0 = gx0 + gx0e
            return (None, None, gs1, gmo, gx0, gl1, gl2)
        if mode == mblk_M_SELF:
            gs1, gmo, gx0, gl1, gl2, gl3 = g
            if gx0e is not None:
                gx0 = gx0 + gx0e
            return (None, None, gs1, gmo, gx0, gl1, gl2, gl3)
        gs1, gmo, gx0, gl1, gl2, gsk, gl3 = g
        if gx0e is not None:
            gx0 = gx0 + gx0e
        return (None, None, gs1, gmo, gx0, gsk, gl1, gl2, gl3)

class mblk_FinalNormFn(torch.autograd.Function):
    """y = norm(s1 + mo) for the last block: the phase-1 tail forward kernel (its s1 output is the saved sum) and the
    block-head RMSNorm backward kernel; the gradient is returned once for both inputs."""

    @staticmethod
    def forward(ctx, s12, mo2):
        N, C = s12.shape
        inv2 = None
        s3 = None
        try:
            s3, y3, inv2 = mblk__launch(mblk__tail_fwd_kernel)(mblk__v3(s12), mblk__v3(mo2))
            y = y3.view(N, C)
            s = s3.view(N, C)
        except Exception as err:
            mblk__disable_kernel2('final_norm forward', err)
            inv2 = None
            with torch.no_grad():
                s = s12 + mo2
                y = mblk_norm(s)
        ctx.save_for_backward(s, inv2)
        return y

    @staticmethod
    def backward(ctx, gy):
        s, inv2 = ctx.saved_tensors
        N, C = s.shape
        if inv2 is not None and (not mblk__KERNEL2_DISABLED) and (mblk__bh_rms_bwd_kernel is not None):
            try:
                gy2 = mblk__as2d(gy, s)
                g3 = mblk__launch(mblk__bh_rms_bwd_kernel)(mblk__v3(s), mblk__v3(gy2), inv2)
                g = g3.view(N, C)
                return (g, g)
            except Exception as err:
                mblk__disable_kernel2('final_norm backward', err)
        with torch.enable_grad():
            sr = s.detach().requires_grad_(True)
            gy_ = torch.zeros_like(s) if gy is None else gy
            g, = torch.autograd.grad(mblk_norm(sr), sr, gy_)
        return (g, g)

def mblk__on_neuron(t) -> bool:
    return t.device.type == 'neuron'

def mblk__same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape

def mblk_head2(x, x0, skip, l1, l2, l3, emit=False, key=None):
    """Block head with the previous block's MLP-tail add folded in (phase 2).  x: the previous block's (s1, mlp_out)
    pair, or a tensor (block 0 passes x0 itself).  skip: None | tensor | SELF_SKIP.  Returns
    (x_new, x_norm, x_in, x0_pass): x_in is the materialised block input when emit (else None); x0_pass is the x0
    to hand to the next block.  Off / not applicable: the eager add, then today's head_skip (its own switch decides)."""
    pair = isinstance(x, (tuple, list))
    is_self = skip is mblk_SELF_SKIP
    if pair:
        s1, mo = x
        base = s1
        if is_self:
            mode = mblk_M_SELF
        elif skip is None:
            mode = mblk_M_NONE
        else:
            mode = mblk_M_EXT
    else:
        s1 = mo = None
        base = x
        mode = mblk_M_X0 if x is x0 and skip is None else -1
    use_kernel = mblk__HAS_NKI2 and (not mblk__KERNEL2_DISABLED) and (mode >= 0) and mblk__on_neuron(base) and (base.dim() >= 2) and mblk__ok_dtype(base) and mblk__rows_ok(base) and mblk__same(x0, base) and (not pair or mblk__same(mo, base)) and (mode != mblk_M_EXT or mblk__same(skip, base)) and (l1.numel() == 1) and (l2.numel() == 1) and (mode < mblk_M_SELF or (l3 is not None and l3.numel() == 1))
    if not use_kernel:
        x_t = s1 + mo if pair else x
        sk_t = x_t if is_self else skip
        x_new, x_norm = mblk_head_skip(x_t, x0, sk_t, l1, l2, l3)
        return (x_new, x_norm, x_t if emit else None, x0)
    C = base.shape[-1]
    if mode == mblk_M_X0:
        x02 = mblk__c2d(x0, C)
        ts = (x02, l1, l2)
    elif mode == mblk_M_NONE:
        ts = (mblk__c2d(s1, C), mblk__c2d(mo, C), mblk__c2d(x0, C), l1, l2)
    elif mode == mblk_M_SELF:
        ts = (mblk__c2d(s1, C), mblk__c2d(mo, C), mblk__c2d(x0, C), l1, l2, l3)
    else:
        ts = (mblk__c2d(s1, C), mblk__c2d(mo, C), mblk__c2d(x0, C), mblk__c2d(skip, C), l1, l2, l3)
    global mblk__GAIN_KEY
    mblk__GAIN_KEY = key
    outs = mblk_Head2Fn.apply(mode, bool(emit), *ts)
    mblk__GAIN_KEY = None
    if emit:
        x_new, x_norm, x_in, x0p = outs
        x_in = x_in.view(base.shape)
    else:
        x_new, x_norm, x0p = outs
        x_in = None
    return (x_new.view(base.shape), x_norm.view(base.shape), x_in, x0p.view(x0.shape))

def mblk_final_norm(s1, mo):
    """norm(s1 + mo) -- the last block's residual add and the final RMSNorm in one launch forward / one backward."""
    use_kernel = mblk__HAS_NKI and mblk__HAS_NKI2 and (not mblk__KERNEL2_DISABLED) and (mblk__bh_rms_bwd_kernel is not None) and mblk__on_neuron(s1) and (s1.dim() >= 2) and mblk__ok_dtype(s1) and mblk__same(mo, s1) and mblk__rows_ok(s1)
    if not use_kernel:
        return mblk_norm(s1 + mo)
    C = s1.shape[-1]
    return mblk_FinalNormFn.apply(mblk__c2d(s1, C), mblk__c2d(mo, C)).view(s1.shape)
head_skip = mblk_head_skip
tail_norm2 = mblk_tail_norm2
USE_NKI_MEGABLOCK = mblk_USE_NKI_MEGABLOCK
head2 = mblk_head2
final_norm = mblk_final_norm
SELF_SKIP = mblk_SELF_SKIP
USE_NKI_MEGABLOCK_TAIL = mblk_USE_NKI_MEGABLOCK_TAIL
try:
    gain_partials_take = mblk_gain_partials_take
except Exception:
    gain_partials_take = None
'[R1 mixfuse] The A12 shift1 mixer fused into the megablock tail kernel: one forward launch, one backward launch.\n\nReplaces, in each attention-less block on the megablock-tail path (train.py Block.forward, `mix_lambda` branch), per\nmixer block per micro-step (launch counts measured on the laptop by dispatcher census, test_mixfuse_model.py M5; they\nreproduce the device profile\'s list op for op):\n\n    today   fwd: gain cast, new_zeros, cat, mul, TailNorm2Fn                                      5 launches  8 passes\n            bwd: TailNorm2FnBackward, mul, mul, sum, slice_backward, add, gain-grad cast,\n                 + Head2Fn\'s zeros_like for the head\'s x_norm a mixer block never reads           8 launches 17 passes\n    fused   fwd: gain cast (unchanged), cat of the [128,1,C] halo, MixTail fwd kernel               3 launches  3 passes\n            bwd: MixTail bwd kernel, sum of the [128,NT] partials, cast, gain-grad cast (unchanged) 4 launches  5 passes\n                 + a CACHED zero tensor handed to Head2Fn as the x_norm gradient                  0 launches\n\n(passes = [N, C] stream passes through HBM; the head\'s read of the zeros is in both columns and not counted.)\ndevprof_c19k7m measured today\'s 13 launches at 3,378 us per mixer block (10.13 ms/micro over blocks 1, 7, 9): its\n"mixer_shift1" class (6.97 ms, 33 launches) plus the three TailNorm2 pairs the mixer re-adds (3.16 ms, 6 launches).\n\nTHE SAME FUNCTION, ROUNDING FOR ROUNDING.  s1 = bf16(x + bf16(lam * shift1(x))) and xn2 = RMSNorm(s1) with the tail\nkernel\'s own instructions (one tensor_tensor add, the ScalarE square-reduce, the tensor_scalar rsqrt, the per-row\nscale); the gain product bf16(lam * shifted) is the eager mul\'s (the product of two bf16 values is exact in fp32, so\nboth round once, to the same value).  Backward: g is the tail kernel\'s chain instruction for instruction; the shift\'s\nadjoint is gx[t] = bf16(g[t] + bf16(lam * g[t+1])) (autograd\'s mul-backward rounding, then its accumulate); the gain\'s\ngradient is the fp32 sum of the bf16 products bf16(g * shift1(x)) -- the eager mul-backward\'s own products -- rounded\nonce to bf16 as autograd\'s sum_to_size does.  Only the summation order differs, and on the laptop the result is the\nCORRECTLY ROUNDED exact sum in every draw (<= 0.5 ulp), where torch-CPU\'s own bf16 sum_to_size was up to ~26 ulp off;\nthe device\'s unfused reduction order is the compiler\'s.\n\nLAYOUT.  The [N, C] stream is viewed [P=128, NT, C] partition-major (row r = p*NT + a), as every megablock kernel does.\nshift1 reads row r-1: row a-1 of the SAME partition for a > 0; for a == 0 the last row of partition p-1, which the\nwrapper gathers into a [P, 1, C] halo with ONE torch cat of two strided slices (the forward needs SBUF partition p to\nreceive partition p-1\'s data, and every compute and DMA destination here starts at partition 0).  A row is a sequence\nstart iff r % T == 0; with NT <= T and T % NT == 0 (the guard) that happens ONLY at a == 0, at partitions p with\np*NT % T == 0, so one per-partition mask hm2[P, 1] zeroes exactly those halo rows.  The backward\'s adjoint reads row\nr+1: within the tile, across tiles through a one-row carry, and at each program\'s last row through a one-row\nrecompute of g (row i1 of the same partition at the LNC=2 program boundary; row 0 of partition p+1 at the end, DMA\'d\nfrom HBM rows 1..P-1 into SBUF partitions 0..P-2 -- partition start 0 -- and masked by hmn2[p] = hm2[p+1]).\n\nGuards: the kernels run only on Neuron, bf16/fp32, N % 128 == 0, NT <= T, T % NT == 0.  Anything else -> train.py\'s\nunfused branch (today\'s ops), counted.  A kernel failure disables THIS module for the process (never nki_megablock\'s\nkernels) and answers that call with the unfused ops, loudly.  The first fused TRAINING call of a process (a startup\nstep, outside the charged clock) runs a SELF-CHECK on the real stream with a NONZERO test gain (mix_lambda is 0 at\ninit, where every shift bug is invisible): both fused kernels vs the unfused path, forward and backward; a mismatch\nor an exception (e.g. a compile failure) disables the fused path, loudly.  Eval-mode calls before that run today\'s ops.\n\nKnobs: none of its own.  Rows-per-tile cap, reductions, products, engines and the launch grid are read\nfrom the inlined nki_megablock (through `_mb`), so the fused tail is the tail kernel\'s arithmetic under\nevery setting.\n'
import os
import torch
mixk__mb = _FreezeLiveNS('mblk_')
mixk__P = 128
mixk_RMS_EPS = mixk__mb.RMS_EPS
mixk__MAX_K = mixk__mb._MAX_K
mixk__LNC = mixk__mb._LNC
mixk__SQ = mixk__mb._SQ
mixk__RSQRT = mixk__mb._RSQRT
mixk__REDUCE = mixk__mb._REDUCE
mixk__PROD_DTYPE = mixk__mb._PROD_DTYPE
mixk__INPLACE = mixk__mb._INPLACE
mixk__SHARD = mixk__mb._SHARD
mixk__DGE = mixk__mb._DGE
mixk__TAIL_GI = mixk__mb._TAIL_GI
mixk__ADD_ENGINE = mixk__mb._ADD_ENGINE
mixk__SBUF_LIMIT = 200 * 1024
mixk__SELFCHECK_LAM = 0.5
mixk__SELFCHECK_TOL = 2.0 ** (-4)
mixk__KERNEL_DISABLED = False
mixk__LAST_ERROR = None
mixk_FIRED_FWD = 0
mixk_FIRED_BWD = 0
mixk_DECLINED = 0
mixk__SAID: set = set()
mixk__SELFCHECK_DONE = False
mixk__SELFCHECK_LINE = 'not run'
mixk__ZEROS: dict = {}
mixk__MASKS: dict = {}

def mixk__rank0() -> bool:
    return os.environ.get('RANK', '0') == '0'

def mixk__say(key, msg, everywhere=False):
    """Print a diagnostic ONCE per process per key (rank 0 only unless everywhere)."""
    if key in mixk__SAID:
        return
    mixk__SAID.add(key)
    if everywhere or mixk__rank0():
        print(msg, flush=True)

def mixk__disable(where: str, err: BaseException) -> None:
    global mixk__KERNEL_DISABLED, mixk__LAST_ERROR
    if not mixk__KERNEL_DISABLED:
        mixk__KERNEL_DISABLED = True
        mixk__LAST_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[mixfuse] fused mixer DISABLED after failure in {where}: {type(err).__name__}: {err} -- the unfused mixer runs from here on (nki_megablock untouched)', flush=True)

def mixk__isz(dt) -> int:
    return 4 if dt == torch.float32 else 2

def mixk__k_pick(n: int, C: int, isz: int, fwd: bool) -> int:
    """Rows per tile per partition: the largest divisor of n (<= the megablock cap) whose double-buffered SBUF plan
    fits _SBUF_LIMIT.  Plain integer arithmetic: the kernels call it at trace time, the wrapper for its print."""
    psz = 4 if mixk__PROD_DTYPE == 'fp32' else isz
    k = 1
    for d in range(1, mixk__MAX_K + 1):
        if n % d == 0:
            if fwd:
                sq = d * C * 4 if mixk__SQ == 'vec' else C * isz
                per_set = d * C * isz * 4 + sq
                fixed = C * isz + 8 * n + 2048
            else:
                per_set = d * C * (isz * 4 + psz + 4)
                if not mixk__INPLACE:
                    per_set = per_set + d * C * isz
                fixed = C * isz + C * (isz * 4 + psz + 4) + 8 * n + 2048
            if 2 * per_set + fixed <= mixk__SBUF_LIMIT:
                k = d
    return k

def mixk__row_range_py(NT: int, pid: int, nprg: int):
    per, rem = (NT // nprg, NT % nprg)
    i0 = pid * per + min(pid, rem)
    return (i0, i0 + per + (1 if pid < rem else 0))
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    mixk__DGE_MODE = None if mixk__DGE == 'auto' else getattr(nisa.dge_mode, mixk__DGE)
    if mixk__ADD_ENGINE == 'gpsimd':
        mixk__ADD_ENG = nisa.engine.gpsimd
    elif mixk__ADD_ENGINE == 'vector':
        mixk__ADD_ENG = nisa.engine.vector
    else:
        mixk__ADD_ENG = nisa.engine.unknown

    def mixk__dma(dst, src):
        if mixk__DGE_MODE is None:
            nisa.dma_copy(dst=dst, src=src)
        else:
            nisa.dma_copy(dst=dst, src=src, dge_mode=mixk__DGE_MODE)

    def mixk__shard():
        if mixk__SHARD and nl.program_ndim() > 0:
            return (nl.program_id(0), nl.num_programs(0))
        return (0, 1)

    def mixk__row_range(NT):
        pid, nprg = mixk__shard()
        per, rem = (NT // nprg, NT % nprg)
        i0 = pid * per + min(pid, rem)
        n = per + (1 if pid < rem else 0)
        return (i0, i0 + n)

    def mixk__pair(shape, dtype):
        return (nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf), nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf))

    def mixk__one(shape, dtype):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    def mixk__sum_sq(xb, sq, ss, K):
        if mixk__SQ == 'vec':
            nisa.activation(dst=sq, op=nl.square, data=xb)
            nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=2)
        else:
            for k in range(K):
                nisa.activation_reduce(dst=sq, op=nl.square, data=xb[:, k, :], reduce_op=nl.add, reduce_res=ss[:, k:k + 1])

    def mixk__row_rsqrt(ss, inv_dst, C):
        if mixk__RSQRT == 'act':
            nisa.activation(dst=inv_dst, op=nl.rsqrt, data=ss, scale=1.0 / C, bias=mixk_RMS_EPS)
        else:
            nisa.tensor_scalar(dst=ss, data=ss, op0=nl.multiply, operand0=1.0 / C, op1=nl.add, operand1=mixk_RMS_EPS)
            nisa.tensor_scalar(dst=inv_dst, data=ss, op0=nl.rsqrt, operand0=0.0)

    def mixk__scale_rows(dst, src, inv, K):
        for k in range(K):
            nisa.tensor_scalar(dst=dst[:, k, :], data=src[:, k, :], op0=nl.multiply, operand0=inv[:, k:k + 1])

    def mixk__reduce_rows(pb, dst, K):
        if mixk__REDUCE == 'vec':
            nisa.tensor_reduce(dst=dst, op=nl.add, data=pb, axis=2)
        else:
            for k in range(K):
                nisa.activation_reduce(dst=pb[:, k, :], op=nl.copy, data=pb[:, k, :], reduce_op=nl.add, reduce_res=dst[:, k:k + 1])

    def mixk__sq_shape(P, K, C):
        if mixk__SQ == 'vec':
            return (P, K, C)
        return (P, C)

    def mixk__sq_dtype(dt):
        if mixk__SQ == 'vec':
            return nl.float32
        return dt

    def mixk__prod_dtype(dt):
        return nl.float32 if mixk__PROD_DTYPE == 'fp32' else dt

    def mixk__nki_isz(dt):
        if dt == nl.float32:
            return 4
        return 2

    def mixk__load_lam(lam2, P, dt):
        """[P, 1] fp32 = float(lam) broadcast to every partition.  lam2 is [1, 1] in the STREAM dtype (the bf16 value
        the unfused mul multiplies by); a bf16 value widens to fp32 exactly (ScalarE copy)."""
        f32 = nl.float32
        t = mixk__one((P, 1), dt)
        mixk__dma(dst=t, src=lam2.broadcast(0, P))
        if dt == f32:
            return t
        t32 = mixk__one((P, 1), f32)
        nisa.activation(dst=t32, op=nl.copy, data=t)
        return t32

    def mixk__load_hm(hm2, P):
        t = mixk__one((P, 1), nl.float32)
        mixk__dma(dst=t, src=hm2)
        return t

    def mixk__prev_row(cr, x3, h3, hm2, i0, P):
        """cr[P, 1, C] = x at the row BEFORE row i0 of every partition: row i0-1 of the same partition when i0 > 0
        (an LNC program boundary -- the row is an INPUT, readable by any program); for i0 == 0 the last row of
        partition p-1 = h3[p], zeroed where row p*NT starts a sequence (hm2[p] == 0; always at p == 0)."""
        if i0 > 0:
            mixk__dma(dst=cr, src=x3[:, i0 - 1:i0, :])
        else:
            mixk__dma(dst=cr, src=h3)
            hmt = mixk__load_hm(hm2, P)
            nisa.tensor_scalar(dst=cr[:, 0, :], data=cr[:, 0, :], op0=nl.multiply, operand0=hmt)

    def mixk__load_xs(dst, x3, h3, hm2, a, K, P):
        """dst[P, K, C] = shift1(x) at rows a..a+K-1 of every partition = x rows a-1..a+K-2, the a == 0 row from the
        halo (masked as in _prev_row)."""
        if a > 0:
            mixk__dma(dst=dst, src=x3[:, a - 1:a - 1 + K, :])
        else:
            mixk__dma(dst=dst[:, 0:1, :], src=h3)
            if K > 1:
                mixk__dma(dst=dst[:, 1:K, :], src=x3[:, 0:K - 1, :])
            hmt = mixk__load_hm(hm2, P)
            nisa.tensor_scalar(dst=dst[:, 0, :], data=dst[:, 0, :], op0=nl.multiply, operand0=hmt)

    def mixk__g_rows(sb, gyb, geb, pb, gib, gb, invs, K, C, P):
        """gb = ge + inv*(gy - s*(gy.s)*inv^2/C) rounded once -- _tail_bwd_kernel's instructions in its order."""
        f32 = nl.float32
        nisa.tensor_tensor(dst=pb, data1=gyb, data2=sb, op=nl.multiply)
        dot = mixk__one((P, K), f32)
        mixk__reduce_rows(pb, dot, K)
        nc = mixk__one((P, K), f32)
        nisa.tensor_tensor(dst=nc, data1=invs, data2=invs, op=nl.multiply)
        nisa.tensor_tensor(dst=nc, data1=nc, data2=invs, op=nl.multiply)
        nisa.tensor_tensor(dst=nc, data1=nc, data2=dot, op=nl.multiply)
        nisa.tensor_scalar(dst=nc, data=nc, op0=nl.multiply, operand0=-1.0 / C)
        if mixk__TAIL_GI == 'act':
            for k in range(K):
                nisa.activation(dst=gib[:, k, :], op=nl.copy, data=gyb[:, k, :], scale=invs[:, k:k + 1])
            nisa.tensor_tensor(dst=gib, data1=gib, data2=geb, op=nl.add, engine=mixk__ADD_ENG)
        else:
            for k in range(K):
                nisa.scalar_tensor_tensor(dst=gib[:, k, :], data=gyb[:, k, :], op0=nl.multiply, operand0=invs[:, k:k + 1], op1=nl.add, operand1=geb[:, k, :])
        for k in range(K):
            nisa.scalar_tensor_tensor(dst=gb[:, k, :], data=sb[:, k, :], op0=nl.multiply, operand0=nc[:, k:k + 1], op1=nl.add, operand1=gib[:, k, :])

    def mixk__next_c(s3, gy3, ge3, inv2, hmn2, lamt, i1, NT, P, C, dt, pdt):
        """[P, 1, C] = bf16(lam * g) at the row AFTER this program's last row, g recomputed with the tail chain (the
        chain is per row, so K = 1 gives the row the same bits as the program that owns it).  i1 < NT: row i1 of the
        same partition (an interior LNC boundary; never a sequence start).  i1 == NT: row 0 of partition p+1, DMA'd
        from HBM rows 1..P-1 into SBUF partitions 0..P-2 (destination partition start 0; partition P-1 keeps its
        memset zeros, a finite row), then masked by hmn2[p] = [p < P-1 and row (p+1)*NT does not start a sequence]."""
        f32 = nl.float32
        sl = mixk__one((P, 1, C), dt)
        gyl = mixk__one((P, 1, C), dt)
        gel = mixk__one((P, 1, C), dt)
        invl = mixk__one((P, 1), f32)
        if i1 < NT:
            mixk__dma(dst=sl, src=s3[:, i1:i1 + 1, :])
            mixk__dma(dst=gyl, src=gy3[:, i1:i1 + 1, :])
            mixk__dma(dst=gel, src=ge3[:, i1:i1 + 1, :])
            mixk__dma(dst=invl, src=inv2[:, i1:i1 + 1])
        else:
            nisa.memset(dst=sl, value=0.0)
            nisa.memset(dst=gyl, value=0.0)
            nisa.memset(dst=gel, value=0.0)
            nisa.memset(dst=invl, value=0.0)
            mixk__dma(dst=sl[0:P - 1, :, :], src=s3[1:P, 0:1, :])
            mixk__dma(dst=gyl[0:P - 1, :, :], src=gy3[1:P, 0:1, :])
            mixk__dma(dst=gel[0:P - 1, :, :], src=ge3[1:P, 0:1, :])
            mixk__dma(dst=invl[0:P - 1, :], src=inv2[1:P, 0:1])
        pl = mixk__one((P, 1, C), pdt)
        gil = mixk__one((P, 1, C), f32)
        gl = gyl if mixk__INPLACE else mixk__one((P, 1, C), dt)
        mixk__g_rows(sl, gyl, gel, pl, gil, gl, invl, 1, C, P)
        nisa.activation(dst=sl, op=nl.copy, data=gl, scale=lamt)
        if i1 == NT:
            hmt = mixk__load_hm(hmn2, P)
            nisa.tensor_scalar(dst=sl[:, 0, :], data=sl[:, 0, :], op0=nl.multiply, operand0=hmt)
        return sl

    @nki.jit
    def mixk__mix_fwd_kernel(x3, h3, hm2, lam2):
        """s3 = bf16(x3 + bf16(lam * shift1(x3))), y3 = RMSNorm(s3), inv2[P, NT] fp32.  x3 is read ONCE: shift1's rows
        1..K-1 of a tile are the tile's own rows 0..K-2, row 0 is a one-row carry of the previous tile's last row (the
        program's first tile: the row before it, _prev_row).  3 HBM passes (x in, s and y out) against the unfused
        path's 8 (cat 2, mul 2, TailNorm2Fn 4)."""
        P, NT, C = x3.shape
        dt = x3.dtype
        f32 = nl.float32
        s3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        y3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        inv2 = nl.ndarray((P, NT), dtype=f32, buffer=nl.shared_hbm)
        i0, i1 = mixk__row_range(NT)
        n = i1 - i0
        if n > 0:
            K = mixk__k_pick(n, C, mixk__nki_isz(dt), True)
            nt = n // K
            lamt = mixk__load_lam(lam2, P, dt)
            invb = mixk__one((P, n), f32)
            xbs = mixk__pair((P, K, C), dt)
            mbs = mixk__pair((P, K, C), dt)
            sbs = mixk__pair((P, K, C), dt)
            ybs = mixk__pair((P, K, C), dt)
            sqs = mixk__pair(mixk__sq_shape(P, K, C), mixk__sq_dtype(dt))
            cr = mixk__one((P, 1, C), dt)
            mixk__prev_row(cr, x3, h3, hm2, i0, P)
            mixk__dma(dst=xbs[0], src=x3[:, i0:i0 + K, :])
            for j in range(nt):
                a = i0 + j * K
                s = j % 2
                xb, mb, sb, yb, sq = (xbs[s], mbs[s], sbs[s], ybs[s], sqs[s])
                if j + 1 < nt:
                    mixk__dma(dst=xbs[(j + 1) % 2], src=x3[:, a + K:a + 2 * K, :])
                nisa.activation(dst=mb[:, 0:1, :], op=nl.copy, data=cr, scale=lamt)
                if K > 1:
                    nisa.activation(dst=mb[:, 1:K, :], op=nl.copy, data=xb[:, 0:K - 1, :], scale=lamt)
                if j + 1 < nt:
                    nisa.activation(dst=cr, op=nl.copy, data=xb[:, K - 1:K, :])
                nisa.tensor_tensor(dst=sb, data1=xb, data2=mb, op=nl.add, engine=mixk__ADD_ENG)
                ss = mixk__one((P, K), f32)
                mixk__sum_sq(sb, sq, ss, K)
                mixk__row_rsqrt(ss, invb[:, j * K:(j + 1) * K], C)
                mixk__scale_rows(yb, sb, invb[:, j * K:(j + 1) * K], K)
                mixk__dma(dst=s3[:, a:a + K, :], src=sb)
                mixk__dma(dst=y3[:, a:a + K, :], src=yb)
            mixk__dma(dst=inv2[:, i0:i1], src=invb)
        return (s3, y3, inv2)

    @nki.jit
    def mixk__mix_bwd_kernel(s3, gy3, ge3, x3, h3, inv2, hm2, hmn2, lam2):
        """gx3 = g + shift1^T(bf16(lam * g)), g = the tail kernel's g (ge + RMSNorm'(s)^T gy, one rounding); r2[P, NT]
        fp32 = per-row sums of bf16(g * shift1(x)), the gain's gradient finished by ONE torch .sum.  shift1^T reads
        row r+1: inside the tile the tile's own rows 1..K-1; across tiles a one-row carry (the tile's last row is
        stored one iteration late); at the program's last row _next_c.  5 HBM passes (s, gy, ge, x in; gx out)
        against the unfused path's 16 (tail 4, mul 2, mul 3, sum 1, slice_backward 3, add 3) + the zeros pass."""
        P, NT, C = s3.shape
        dt = s3.dtype
        f32 = nl.float32
        pdt = mixk__prod_dtype(dt)
        gx3 = nl.ndarray((P, NT, C), dtype=dt, buffer=nl.shared_hbm)
        r2 = nl.ndarray((P, NT), dtype=f32, buffer=nl.shared_hbm)
        i0, i1 = mixk__row_range(NT)
        n = i1 - i0
        if n > 0:
            K = mixk__k_pick(n, C, mixk__nki_isz(dt), False)
            nt = n // K
            lamt = mixk__load_lam(lam2, P, dt)
            invb = mixk__one((P, n), f32)
            mixk__dma(dst=invb, src=inv2[:, i0:i1])
            rrb = mixk__one((P, n), f32)
            sbs = mixk__pair((P, K, C), dt)
            gybs = mixk__pair((P, K, C), dt)
            gebs = mixk__pair((P, K, C), dt)
            xsbs = mixk__pair((P, K, C), dt)
            pbs = mixk__pair((P, K, C), pdt)
            gibs = mixk__pair((P, K, C), f32)
            gbs = gybs if mixk__INPLACE else mixk__pair((P, K, C), dt)
            cg = mixk__one((P, 1, C), dt)
            mixk__dma(dst=sbs[0], src=s3[:, i0:i0 + K, :])
            mixk__dma(dst=gybs[0], src=gy3[:, i0:i0 + K, :])
            mixk__dma(dst=gebs[0], src=ge3[:, i0:i0 + K, :])
            mixk__load_xs(xsbs[0], x3, h3, hm2, i0, K, P)
            for j in range(nt):
                a = i0 + j * K
                s = j % 2
                sb, gyb, geb, xsb, pb, gib, gb = (sbs[s], gybs[s], gebs[s], xsbs[s], pbs[s], gibs[s], gbs[s])
                if j + 1 < nt:
                    s1_ = (j + 1) % 2
                    mixk__dma(dst=sbs[s1_], src=s3[:, a + K:a + 2 * K, :])
                    mixk__dma(dst=gybs[s1_], src=gy3[:, a + K:a + 2 * K, :])
                    mixk__dma(dst=gebs[s1_], src=ge3[:, a + K:a + 2 * K, :])
                    mixk__load_xs(xsbs[s1_], x3, h3, hm2, a + K, K, P)
                invs = invb[:, j * K:(j + 1) * K]
                mixk__g_rows(sb, gyb, geb, pb, gib, gb, invs, K, C, P)
                nisa.tensor_tensor(dst=geb, data1=gb, data2=xsb, op=nl.multiply)
                mixk__reduce_rows(geb, rrb[:, j * K:(j + 1) * K], K)
                nisa.activation(dst=sb, op=nl.copy, data=gb, scale=lamt)
                if j > 0:
                    nisa.tensor_tensor(dst=cg, data1=cg, data2=sb[:, 0:1, :], op=nl.add, engine=mixk__ADD_ENG)
                    mixk__dma(dst=gx3[:, a - 1:a, :], src=cg)
                if j + 1 < nt:
                    nisa.activation(dst=cg, op=nl.copy, data=gb[:, K - 1:K, :])
                if K > 1:
                    nisa.tensor_tensor(dst=gb[:, 0:K - 1, :], data1=gb[:, 0:K - 1, :], data2=sb[:, 1:K, :], op=nl.add, engine=mixk__ADD_ENG)
                if j + 1 < nt:
                    if K > 1:
                        mixk__dma(dst=gx3[:, a:a + K - 1, :], src=gb[:, 0:K - 1, :])
                else:
                    cl = mixk__next_c(s3, gy3, ge3, inv2, hmn2, lamt, i1, NT, P, C, dt, pdt)
                    nisa.tensor_tensor(dst=gb[:, K - 1:K, :], data1=gb[:, K - 1:K, :], data2=cl, op=nl.add, engine=mixk__ADD_ENG)
                    mixk__dma(dst=gx3[:, a:a + K, :], src=gb)
            mixk__dma(dst=r2[:, i0:i1], src=rrb)
        return (gx3, r2)
    mixk__HAS_NKI = True
except Exception:
    mixk__HAS_NKI = False

def mixk__mix_kernels():
    """(fwd, bwd) NKI kernels of this module, or (None, None) without nki."""
    if mixk__HAS_NKI:
        return (mixk__mix_fwd_kernel, mixk__mix_bwd_kernel)
    return (None, None)
mixk__KF, mixk__KB = mixk__mix_kernels()

def mixk__on_neuron(t) -> bool:
    return t.device.type == 'neuron'

def mixk__launch(kernel):
    """kernel[2] under LNC=2 (the megablock convention); a plain callable (a test emulation) is called as-is."""
    if mixk__LNC == 2 and hasattr(kernel, '__getitem__'):
        return kernel[mixk__LNC]
    return kernel

def mixk__masks(NT: int, T: int, device):
    """(hm2, hmn2), [P, 1] fp32 on device, built once per shape on the host: hm2[p] = 0 iff row p*NT starts a
    sequence (p*NT % T == 0), else 1; hmn2[p] = hm2[p+1] for p < P-1, 0 for p = P-1 (no next row)."""
    key = (NT, T, str(device))
    m = mixk__MASKS.get(key)
    if m is None:
        hm = [0.0 if p * NT % T == 0 else 1.0 for p in range(mixk__P)]
        hmn = hm[1:] + [0.0]
        m = (torch.tensor(hm, dtype=torch.float32).view(mixk__P, 1).to(device), torch.tensor(hmn, dtype=torch.float32).view(mixk__P, 1).to(device))
        mixk__MASKS[key] = m
    return m

def mixk__halo(x3):
    """[P, 1, C]: row p holds x at row p*NT - 1 (the last row of partition p-1); row 0 is a finite dummy (x's own
    last row of partition 0) that the kernels mask to zero -- ONE cat of two strided slices, the op the unfused
    mixer already runs on this backend (gather/index ops are rejected by the runtime, train.py P017 note)."""
    NT = x3.shape[1]
    return torch.cat((x3[0:1, NT - 1:NT, :], x3[0:mixk__P - 1, NT - 1:NT, :]), dim=0)

def mixk__zeros_for(meta):
    """The cached all-zero gradient for the head's unused x_norm (allocated once, on a startup step; nothing ever
    writes it: Head2Fn's backward only DMAs it in).  Handing Head2Fn a real tensor instead of None removes its
    per-call torch.zeros_like (one launch + one 67 MB write per mixer block per micro-step, devprof_c19k7m)."""
    if meta is None:
        return None
    z = mixk__ZEROS.get(meta)
    if z is None:
        shape, dtype, device = meta
        z = torch.zeros(shape, dtype=dtype, device=device)
        mixk__ZEROS[meta] = z
    return z

def mixk__as3(g, like3):
    if g is None:
        return torch.zeros_like(like3)
    g3 = g.reshape(like3.shape)
    if g3.dtype != like3.dtype:
        g3 = g3.to(like3.dtype)
    if not g3.is_contiguous():
        g3 = g3.contiguous()
    return g3

def mixk_shift1(x: torch.Tensor) -> torch.Tensor:
    """train.py's shift1, byte for byte (the unfused fallback must be today's op)."""
    return torch.cat((x.new_zeros(x.shape[0], 1, x.shape[2]), x[:, :-1, :]), dim=1)

def mixk__unfused(x, lam):
    """Today's mixer ops exactly (train.py's `else:` branch with no gain / dropout): the fallback."""
    mix_out = lam * mixk_shift1(x)
    return mixk__mb.tail_norm2(x, mix_out)

def mixk_applies(x) -> bool:
    """True when the fused kernels take this call.  False -> train.py runs its unfused branch (today's ops); every
    distinct reason prints once and DECLINED counts the calls, so an arm whose log shows fired_fwd=0 is VOID."""
    global mixk_DECLINED
    reason = None
    if mixk__KF is None or mixk__KB is None:
        reason = 'nki not importable'
    elif mixk__KERNEL_DISABLED:
        reason = f'disabled ({mixk__LAST_ERROR})'
    elif not mixk__on_neuron(x):
        reason = f'device {x.device.type}'
    elif x.dim() != 3 or x.dtype not in (torch.bfloat16, torch.float32) or (not x.is_contiguous()):
        reason = f'tensor {tuple(x.shape)} {x.dtype} contiguous={x.is_contiguous()}'
    else:
        B, T, C = x.shape
        N = B * T
        if N % mixk__P:
            reason = f'rows {N} % {mixk__P}'
        else:
            NT = N // mixk__P
            if NT > T or T % NT:
                reason = f'layout NT={NT} T={T} (needs NT <= T and T % NT == 0)'
    if reason is None:
        return True
    mixk_DECLINED += 1
    mixk__say(('declined', reason), f'[mixfuse] fused mixer not applicable: {reason} -- unfused mixer for such calls')
    return False

def mixk_mix_tail(x, xn_dead, lam):
    """(s1, xn2) = (x + lam * shift1(x), RMSNorm(s1)) for a [B, T, C] stream: train.py's mixer branch in one fused
    launch each way.  xn_dead is the head's x_norm output the mixer block never reads (only its gradient slot is
    used); lam is mix_lambda.to(stream dtype), the SAME tensor the unfused mul consumes."""
    B, T, C = x.shape
    if not mixk__SELFCHECK_DONE:
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            return mixk__unfused(x, lam)
        mixk__selfcheck(x, T)
        if mixk__KERNEL_DISABLED:
            return mixk__unfused(x, lam)
    s1, y = mixk_MixTailFn.apply(x.reshape(B * T, C), xn_dead, lam, T)
    return (s1.view(B, T, C), y.view(B, T, C))

class mixk_MixTailFn(torch.autograd.Function):
    """(s1, xn2) on the [N, C] stream via _KF / _KB.  Inputs (x2, xn_dead, lam, T); gradients (gx, zeros, dlam, None)."""

    @staticmethod
    def forward(ctx, x2, xn_dead, lam, T):
        global mixk_FIRED_FWD
        ctx.set_materialize_grads(False)
        N, C = x2.shape
        NT = N // mixk__P
        ctx.T = T
        ctx.xn_meta = (tuple(xn_dead.shape), xn_dead.dtype, xn_dead.device) if xn_dead is not None else None
        x3 = x2.view(mixk__P, NT, C)
        hm2, _ = mixk__masks(NT, T, x2.device)
        lam2 = lam.reshape(1, 1)
        try:
            h3 = mixk__halo(x3)
            s3, y3, inv2 = mixk__launch(mixk__KF)(x3, h3, hm2, lam2)
        except Exception as err:
            mixk__disable('forward', err)
            ctx.fused = False
            ctx.save_for_backward(x2, lam)
            with torch.no_grad():
                s1, y = mixk__unfused(x2.view(N // T, T, C), lam)
            return (s1.reshape(N, C), y.reshape(N, C))
        mixk_FIRED_FWD += 1
        if mixk_FIRED_FWD == 1:
            nprg = mixk__LNC if mixk__LNC == 2 else 1
            i0, i1 = mixk__row_range_py(NT, 0, nprg)
            mixk__say('fired_fwd', f'mixfuse:           FIRED fwd x=[{N}, {C}] T={T} NT={NT} nprg={nprg} K={mixk__k_pick(i1 - i0, C, mixk__isz(x2.dtype), True)}/{mixk__k_pick(i1 - i0, C, mixk__isz(x2.dtype), False)} halo=[{mixk__P}, 1, {C}] seq_start_partitions={sum((1 for p in range(mixk__P) if p * NT % T == 0))}')
        ctx.fused = True
        ctx.save_for_backward(s3, x3, h3, inv2, lam2)
        return (s3.view(N, C), y3.view(N, C))

    @staticmethod
    def backward(ctx, ge, gy):
        global mixk_FIRED_BWD
        zeros = mixk__zeros_for(ctx.xn_meta)
        T = ctx.T
        if ctx.fused:
            s3, x3, h3, inv2, lam2 = ctx.saved_tensors
            P, NT, C = s3.shape
            if not mixk__KERNEL_DISABLED:
                try:
                    hm2, hmn2 = mixk__masks(NT, T, s3.device)
                    gx3, r2 = mixk__launch(mixk__KB)(s3, mixk__as3(gy, s3), mixk__as3(ge, s3), x3, h3, inv2, hm2, hmn2, lam2)
                    mixk_FIRED_BWD += 1
                    return (gx3.view(P * NT, C), zeros, r2.sum().to(lam2.dtype).reshape(1), None)
                except Exception as err:
                    mixk__disable('backward', err)
            x2, lam = (x3.reshape(P * NT, C), lam2.reshape(1))
        else:
            x2, lam = ctx.saved_tensors
        N, C = x2.shape
        with torch.enable_grad():
            xr = x2.detach().requires_grad_(True)
            lr = lam.detach().requires_grad_(True)
            s1r, yr = mixk__unfused(xr.view(N // T, T, C), lr)
            outs, gouts = ([], [])
            if ge is not None:
                outs.append(s1r)
                gouts.append(ge.reshape(s1r.shape))
            if gy is not None:
                outs.append(yr)
                gouts.append(gy.reshape(yr.shape))
            if not outs:
                return (None, zeros, None, None)
            gx, gl = torch.autograd.grad(outs, (xr, lr), gouts, allow_unused=True)
        return (gx, zeros, gl, None)

def mixk__nerr(a, b) -> float:
    d = (a.float() - b.float()).abs().max()
    return float(d / b.float().abs().max().clamp_min(1e-30))

def mixk__selfcheck(x, T, lam_test: float=mixk__SELFCHECK_LAM):
    """Once per process, on the first fused call (a startup step, outside the charged clock): the fused kernels and
    the unfused path on the SAME real stream with a nonzero test gain and synthetic upstream gradients (gy = x,
    ge = -x/4), forward and backward.  Normalised max errors: s1, xn2, gx by max|ref|, the gain gradient by
    sum|g * shift1(x)|.  Any error above _SELFCHECK_TOL (or any exception) disables the fused path, loudly: the
    run then measures the unfused mixer and says so -- never a silently-wrong fused one."""
    global mixk__SELFCHECK_DONE, mixk__SELFCHECK_LINE
    mixk__SELFCHECK_DONE = True
    try:
        B, _, C = x.shape
        N = B * T
        NT = N // mixk__P
        with torch.no_grad():
            x2 = x.detach().reshape(N, C)
            lam = torch.full((1,), lam_test, dtype=x.dtype, device=x.device)
            gy2 = x2.clone()
            ge2 = x2 * -0.25
            x3 = x2.view(mixk__P, NT, C)
            hm2, hmn2 = mixk__masks(NT, T, x.device)
            h3 = mixk__halo(x3)
            lam2 = lam.reshape(1, 1)
            s3, y3, inv2 = mixk__launch(mixk__KF)(x3, h3, hm2, lam2)
            gx3, r2 = mixk__launch(mixk__KB)(s3, gy2.view(mixk__P, NT, C), ge2.view(mixk__P, NT, C), x3, h3, inv2, hm2, hmn2, lam2)
            dl_k = r2.sum()
        with torch.enable_grad():
            xr = x2.detach().clone().requires_grad_(True)
            lr = lam.detach().clone().requires_grad_(True)
            xv = xr.view(B, T, C)
            sh = mixk_shift1(xv)
            m = lr * sh
            s_r, y_r = mixk__mb.tail_norm2(xv, m)
            gx_r, dl_r, gm = torch.autograd.grad((s_r, y_r), (xr, lr, m), (ge2.view(B, T, C), gy2.view(B, T, C)))
        with torch.no_grad():
            scale = (gm.float() * sh.float()).abs().sum().clamp_min(1e-30)
            es = mixk__nerr(s3.view(N, C), s_r.reshape(N, C))
            ey = mixk__nerr(y3.view(N, C), y_r.reshape(N, C))
            eg = mixk__nerr(gx3.view(N, C), gx_r)
            ed = float((dl_k.float() - dl_r.float().reshape(())).abs() / scale)
        ok = max(es, ey, eg, ed) <= mixk__SELFCHECK_TOL
        mixk__SELFCHECK_LINE = f'lam={lam_test} s1 {es:.2e} y {ey:.2e} gx {eg:.2e} dlam {ed:.2e} (tol {mixk__SELFCHECK_TOL:.0e}) -> {('PASS' if ok else 'FAIL')}'
        mixk__say('selfcheck', f'mixfuse:           selfcheck {mixk__SELFCHECK_LINE}', everywhere=not ok)
        if not ok:
            mixk__disable('selfcheck', RuntimeError(mixk__SELFCHECK_LINE))
    except Exception as err:
        mixk__SELFCHECK_LINE = f'ERROR {type(err).__name__}: {err}'
        mixk__disable('selfcheck', err)

def mixk_status() -> str:
    """The liveness fields of train.py's `mixfuse:` line."""
    kind = 'none' if mixk__KF is None else 'nki' if mixk__KF is mixk__mix_kernels()[0] else 'substituted'
    return f'kernel={kind} fired_fwd={mixk_FIRED_FWD} fired_bwd={mixk_FIRED_BWD} declined={mixk_DECLINED} disabled={mixk__LAST_ERROR!r} selfcheck={mixk__SELFCHECK_LINE!r}'
nki_mixtail = _FreezeLiveNS('mixk_')
try:
    from nkilib_attention import nki_causal_attention, USE_NKILIB_ATTN
except Exception:
    USE_NKILIB_ATTN = False
'NKI causal attention (forward + backward) for train.py\'s CausalSelfAttention, one launch each way; v8 adds the\nfused RoPE + QK-norm + attention entry point (plan v11 lane B, default off).\n\nReplaces, in CausalSelfAttention.forward (q, k, v are [B, H, T, D] after rope_norm / the eager RoPE+norm path):\n\n    y = F.scaled_dot_product_attention(q, k, v, is_causal=True[, scale=QK_SCALE])\n    y = y.transpose(1, 2).contiguous().view(b, t, c)\n\nwith\n\n    y = nki_causal_attention(q, k, v, scale=QK_SCALE or None).view(b, t, c)     # [B, T, H, D] straight from the kernel\n\nPublic entry point\n------------------\n    nki_causal_attention(q, k, v, scale=None) -> y [B, T, H, D]\n        q, k, v   [B, H, T, D], same dtype (bf16 in training; fp32 only for tests with R1_NKI_ATTN_FP32=1)\n        scale     softmax scale (None = 1/sqrt(D), SDPA\'s default)\n        y         == F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale).transpose(1, 2)\n                  written by the kernel directly into [B, T, H, D] memory, so the block\'s `.view(b, t, c)` is free\n                  (today\'s path pays a transpose + contiguous copy of 32 MiB per block).  The eager fallback\n                  returns the same layout.\n\nMath (per (b, h) pair; S = scale * Q K^T, causal, D = head dim, T tokens)\n--------------------------------------------------------------------------\nForward, "transposed" formulation (keys on the partition axis, queries on the free axis):\n    S^T[k, q] = K[k, :] . Q[q, :]                        nc_matmul(stationary = K^T[d, k], moving = Q^T[d, q]) -> PSUM\n    E^T[k, q] = exp(scale * S^T - c)  (bf16, 0 above the diagonal)\n    [O | s][q, :] = sum_k E^T[k, q] * [V | 1][k, :]     nc_matmul(stationary = E^T[k, q], moving = V_aug[k, d+1]) -> PSUM\n    O[q, :] = O[q, :] / s[q];   lse[q] = c + log s[q]\n  The fixed shift c replaces the row max: after the model\'s parameter-free QK-norm every element of q and k has\n  rms 1, so |q . k| <= D and |scale * S| <= scale * D =: c.  exp(scale*S - c) is then in [e^-2c, 1], every term a\n  normal fp32/bf16 number (c = 11.3 at D = 128, scale = 1/sqrt(128); 16.3 at QK_SCALE 0.127), and the log-sum-exp is\n  exact to fp32 rounding.  (This is the loss-head kernel\'s bounded-logits trick; it is what makes the transposed\n  forward possible: a per-query max would be a partition-axis reduction here.)  The wrapper sets c = scale * D\n  (+ R1_NKI_ATTN_SHIFT_MARGIN); scores may exceed c by up to ~80 before fp32 exp overflows, so the bound only has\n  to hold approximately.  The ones column of V_aug makes the PV matmul produce the row sums for free: no P\n  transposes, no max pass, no separate sum pass.  Row sums are sums of the SAME bf16 E values the PV matmul uses.\nBackward, row formulation (queries on the partition axis, nkilib\'s layout), per query tile g, key tiles j <= g:\n    P = exp(scale * S - lse[q])  (bf16, masked on the diagonal block)                 ScalarE, per-partition bias\n    dP = dO V^T (PSUM),  D[q] = sum_d dO[q, d] O[q, d]  (VectorE),  dS = P * (dP - D[q])  (one VectorE pass, bf16)\n    dV[k, :] += P^T dO        nc_matmul(stationary = P[q, k],    moving = dO[q, d])      PSUM accumulator over g\n    dK[k, :] += dS^T Q        nc_matmul(stationary = dS[q, k],   moving = Q[q, d])       PSUM accumulator over g\n    dQ[q, :] += dS K          nc_matmul(stationary = dS^T[k, q], moving = K[k, d])       PSUM accumulator over j\n    dK, dQ scaled by `scale` on the way out.  dS^T comes from TensorE nc_transpose (bit-accurate on Trn2).\n  Causal work at 128-token granularity: 36 of 64 (q-tile, k-tile) blocks at T = 1024 (nkilib: 48, its K tiles are\n  512 wide); only the 8 diagonal blocks are masked (GpSimd affine_select, in place, on the bf16 probabilities).\n\nLayouts / DMAs: q, k, v are read per (b, h) as [128 tokens, D] tiles (one DMA per head tensor with an access\npattern, 256 B per partition per tile); Q^T/K^T/V^T/dO^T come from TensorE nc_transpose of those tiles\n(R1_NKI_ATTN_LOAD_T=pe, default) or from one DMA transpose per head (=dma, nkilib\'s bf16 form).  O and dO live in\n[B, T, H, D] memory (the block\'s layout), dV is written there too and handed back as a permuted view (what\n`v.transpose(1, 2)`\'s backward wants, no copy); dQ, dK come back as [B, H, T, D] (rope_norm\'s layout).\n\nD <= 128, T % 128 == 0, T <= 4096, same shape/dtype for q, k, v.  A kernel failure prints once, disables the kernel\nfor the process and answers that call with the eager math (forward: SDPA; backward: SDPA\'s autograd).\n\nKnobs (env, read once at import; module-level constants because kernel bodies only see module globals):\n\nFused RoPE + QK-norm + attention (plan v11 lane B; default off)\n--------------------------------------------------------------\n    nki_rope_attention(qkv, cos, sin, n_head, scale=None) -> y [B, T, H, D] or None\n        qkv       [B, T, 3C] bf16, the fused QKV projection output (q = columns [0, C), k = [C, 2C), v = [2C, 3C),\n                  head h at columns h*D .. (h+1)*D of each third), contiguous\n        cos, sin  train.py\'s RoPE tables for the T positions ([1, T, 1, D/2]; anything that reshapes to [T, D/2])\n        y         == nki_causal_attention(rope_norm(q), rope_norm(k), v.transpose(1, 2)) to the BIT: the forward\n                  kernel reads q, k, v straight from qkv, applies the RoPE+QK-norm kernel\'s instruction sequence on\n                  the natural [128 tokens, G, q|k, D] tile in SBUF, forms Q^T/K^T on TensorE and runs the v7 loop;\n                  the backward kernel applies the norm Jacobian and the inverse rotation to the v7 kernel\'s bf16\n                  dQ/dK tiles and writes dq | dk | dv into ONE [B, T, 3C] gradient (no split-cat, no v copy).\n                  None when the fused path does not apply (switch off, CPU, D != 128, T % 128, ...): the caller\n                  keeps today\'s rope_norm -> nki_causal_attention chain.  A kernel failure disables the fused path\n                  for the process and answers that call with the eager math (RoPE, rms_norm, SDPA and its autograd).\n'
import math
import os
import torch
import torch.nn.functional as F
attnk__P = 128

def attnk__env_int(name, default):
    try:
        return int(os.environ.get(name, '') or default)
    except ValueError:
        return default
attnk_USE_NKI_ATTN = '1'
attnk__LNC = attnk__env_int('NEURON_LOGICAL_NC_CONFIG', 1)
attnk__KCHUNK = max(1, min(4, 4))
attnk__BWD_GROUP = max(1, min(8, 4))
attnk__SHIFT_MARGIN = float('0')
attnk_ALLOW_FP32 = '0'
attnk_USE_NKI_ATTN_ROPE = '1'
attnk__V81 = '1'
attnk__ROPE_EPS = 2.0 ** (-23)
attnk__KERNEL_DISABLED = False
attnk__LAST_KERNEL_ERROR = None
attnk__MAX_T = 4096

def attnk_eager_causal_attention(q, k, v, scale=None):
    """SDPA(q, k, v, causal) delivered as [B, T, H, D] (the layout the kernel writes)."""
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    return y.transpose(1, 2).contiguous()

def attnk__eager_backward(q, k, v, gy_bthd, scale):
    with torch.enable_grad():
        qr = q.detach().requires_grad_(True)
        kr = k.detach().requires_grad_(True)
        vr = v.detach().requires_grad_(True)
        y = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True, scale=scale)
        return torch.autograd.grad(y, (qr, kr, vr), gy_bthd.transpose(1, 2))
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    def attnk__shard():
        """(program_id, num_programs); (0, 1) without a launch grid or with sharding off."""
        if nl.program_ndim() > 0:
            return (nl.program_id(0), nl.num_programs(0))
        return (0, 1)

    def attnk__copy_engine():
        return nisa.engine.unknown

    def attnk__load_nat(x3, base, T, D, G):
        """[P, G*D] SBUF tile of one head: partition p, block g = token g*128 + p (one DMA, 256 B runs)."""
        nat = nl.ndarray((attnk__P, G * D), dtype=x3.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=nat, src=x3.ap(pattern=[[D, attnk__P], [attnk__P * D, G], [1, D]], offset=base))
        return nat

    def attnk__load_nat_bthd(x3, b, h, T, C, D, G):
        """Same tile layout from a [B, T, C] tensor (head h's D columns of each token row)."""
        nat = nl.ndarray((attnk__P, G * D), dtype=x3.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=nat, src=x3.ap(pattern=[[C, attnk__P], [attnk__P * C, G], [1, D]], offset=b * T * C + h * D))
        return nat

    def attnk__transpose_nat(nat, T, D, G):
        """[D, T] SBUF tile (head dim on partitions) from the natural [P, G*D] tile: G TensorE transposes."""
        xt = nl.ndarray((D, T), dtype=nat.dtype, buffer=nl.sbuf)
        for g0 in range(0, G, 4):
            ng = G - g0
            if ng > 4:
                ng = 4
            ps = nl.ndarray((D, ng * attnk__P), dtype=nat.dtype, buffer=nl.psum)
            for gg in range(ng):
                g = g0 + gg
                nisa.nc_transpose(dst=ps[:, gg * attnk__P:(gg + 1) * attnk__P], data=nat[:, g * D:(g + 1) * D])
            nisa.tensor_copy(dst=xt[:, g0 * attnk__P:(g0 + ng) * attnk__P], src=ps, engine=attnk__copy_engine())
        return xt

    def attnk__load_T(x3, base, T, D, G):
        """[D, T] SBUF tile of one head, head dim on partitions (nkilib's bf16 form: one DMA transpose)."""
        xt = nl.ndarray((D, T), dtype=x3.dtype, buffer=nl.sbuf)
        nisa.dma_transpose(dst=xt.ap([[T, D], [1, 1], [1, 1], [1, T]]), src=x3.ap([[D, T], [1, 1], [1, 1], [1, D]], offset=base))
        return xt
        return attnk__transpose_nat(attnk__load_nat(x3, base, T, D, G), T, D, G)

    @nki.jit
    def attnk__attn_fwd_kernel(q4, k4, v4, scale, shift):
        """o3[B, T, H*D] (q4.dtype) and lse3[B*H, 128, G] (fp32; [i, p, g] = query g*128 + p of pair i = b*H + h)
        from q4, k4, v4 [B, H, T, D]."""
        B, H, T, D = q4.shape
        bs = B * H
        G = T // attnk__P
        C = H * D
        W = D + 1
        f32 = nl.float32
        dt = q4.dtype
        o3 = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
        lse3 = nl.ndarray((bs, attnk__P, G), dtype=f32, buffer=nl.shared_hbm)
        pid, nprg = attnk__shard()
        nbias = nl.ndarray((attnk__P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(nbias, value=-shift)
        for i in range(pid, bs, nprg):
            b = i // H
            h = i % H
            base = i * T * D
            qT = attnk__load_T(q4, base, T, D, G)
            kT = attnk__load_T(k4, base, T, D, G)
            vaug = nl.ndarray((attnk__P, G * W), dtype=dt, buffer=nl.sbuf)
            nisa.memset(vaug, value=1.0)
            for j in range(G):
                nisa.dma_copy(dst=vaug[:, j * W:j * W + D], src=v4.ap(pattern=[[D, attnk__P], [1, D]], offset=base + j * attnk__P * D))
            lse_blk = nl.ndarray((attnk__P, G), dtype=f32, buffer=nl.sbuf)
            for g in range(G):
                oacc = nl.ndarray((attnk__P, W), dtype=f32, buffer=nl.psum)
                for j0 in range(0, g + 1, attnk__KCHUNK):
                    nj = g + 1 - j0
                    if nj > attnk__KCHUNK:
                        nj = attnk__KCHUNK
                    st = nl.ndarray((attnk__P, nj * attnk__P), dtype=f32, buffer=nl.psum)
                    for jj in range(nj):
                        j = j0 + jj
                        nisa.nc_matmul(dst=st[:, jj * attnk__P:(jj + 1) * attnk__P], stationary=kT[:, j * attnk__P:(j + 1) * attnk__P], moving=qT[:, g * attnk__P:(g + 1) * attnk__P], accumulate=False)
                    et = nl.ndarray((attnk__P, nj * attnk__P), dtype=dt, buffer=nl.sbuf)
                    nisa.activation(dst=et, op=nl.exp, data=st, bias=nbias, scale=scale)
                    if j0 + nj - 1 == g:
                        jj = nj - 1
                        nisa.affine_select(dst=et[:, jj * attnk__P:(jj + 1) * attnk__P], pattern=[[1, attnk__P]], channel_multiplier=-1, offset=0, cmp_op=nl.greater_equal, on_true_tile=et[:, jj * attnk__P:(jj + 1) * attnk__P], on_false_value=0.0)
                    for jj in range(nj):
                        j = j0 + jj
                        nisa.nc_matmul(dst=oacc, stationary=et[:, jj * attnk__P:(jj + 1) * attnk__P], moving=vaug[:, j * W:(j + 1) * W], accumulate=j > 0)
                rs = nl.ndarray((attnk__P, 1), dtype=f32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rs, data=oacc[:, D:D + 1])
                ot = nl.ndarray((attnk__P, D), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=ot, data=oacc[:, 0:D], op0=nl.multiply, operand0=rs)
                nisa.dma_copy(dst=o3.ap(pattern=[[C, attnk__P], [1, D]], offset=(b * T + g * attnk__P) * C + h * D), src=ot)
                nisa.activation(dst=lse_blk[:, g:g + 1], op=nl.log, data=oacc[:, D:D + 1])
            nisa.tensor_scalar(dst=lse_blk, data=lse_blk, op0=nl.add, operand0=shift)
            nisa.dma_copy(dst=lse3[i, :, :], src=lse_blk)
        return (o3, lse3)

    @nki.jit
    def attnk__attn_bwd_kernel(q4, k4, v4, o3, do3, lse3, scale):
        """dq4, dk4 [B, H, T, D] and dv3 [B, T, H*D] (all q4.dtype) from q4, k4, v4 [B, H, T, D], o3, do3 [B, T, H*D],
        lse3 [B*H, 128, G] fp32."""
        B, H, T, D = q4.shape
        bs = B * H
        G = T // attnk__P
        C = H * D
        f32 = nl.float32
        dt = q4.dtype
        dq4 = nl.ndarray((B, H, T, D), dtype=dt, buffer=nl.shared_hbm)
        dk4 = nl.ndarray((B, H, T, D), dtype=dt, buffer=nl.shared_hbm)
        dv3 = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
        pid, nprg = attnk__shard()
        for i in range(pid, bs, nprg):
            b = i // H
            h = i % H
            base = i * T * D
            qn = attnk__load_nat(q4, base, T, D, G)
            kn = attnk__load_nat(k4, base, T, D, G)
            on = attnk__load_nat_bthd(o3, b, h, T, C, D, G)
            don = attnk__load_nat_bthd(do3, b, h, T, C, D, G)
            qT = attnk__transpose_nat(qn, T, D, G)
            kT = attnk__transpose_nat(kn, T, D, G)
            doT = attnk__transpose_nat(don, T, D, G)
            vT = attnk__load_T(v4, base, T, D, G)
            nb = nl.ndarray((attnk__P, G), dtype=f32, buffer=nl.sbuf)
            nisa.dma_copy(dst=nb, src=lse3[i, :, :])
            nisa.tensor_scalar(dst=nb, data=nb, op0=nl.multiply, operand0=-1.0)
            dsum = nl.ndarray((attnk__P, G), dtype=f32, buffer=nl.sbuf)
            for g in range(G):
                tmp = nl.ndarray((attnk__P, D), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=tmp, data1=don[:, g * D:(g + 1) * D], data2=on[:, g * D:(g + 1) * D], op=nl.multiply)
                nisa.tensor_reduce(dst=dsum[:, g:g + 1], op=nl.add, data=tmp, axis=1)
            dkacc = nl.ndarray((attnk__P, G * D), dtype=f32, buffer=nl.sbuf)
            dvacc = nl.ndarray((attnk__P, G * D), dtype=f32, buffer=nl.sbuf)
            nisa.memset(dkacc, value=0.0)
            nisa.memset(dvacc, value=0.0)
            for g0 in range(0, G, attnk__BWD_GROUP):
                ng = G - g0
                if ng > attnk__BWD_GROUP:
                    ng = attnk__BWD_GROUP
                wg = (g0 + ng) * attnk__P
                pg = nl.ndarray((attnk__P, ng * wg), dtype=dt, buffer=nl.sbuf)
                dsg = nl.ndarray((attnk__P, ng * wg), dtype=dt, buffer=nl.sbuf)
                for gg in range(ng):
                    g = g0 + gg
                    dqacc = nl.ndarray((attnk__P, D), dtype=f32, buffer=nl.psum)
                    for j0 in range(0, g + 1, attnk__KCHUNK):
                        nj = g + 1 - j0
                        if nj > attnk__KCHUNK:
                            nj = attnk__KCHUNK
                        w = nj * attnk__P
                        cb = gg * wg + j0 * attnk__P
                        s = nl.ndarray((attnk__P, w), dtype=f32, buffer=nl.psum)
                        nisa.nc_matmul(dst=s, stationary=qT[:, g * attnk__P:(g + 1) * attnk__P], moving=kT[:, j0 * attnk__P:j0 * attnk__P + w], accumulate=False)
                        nisa.activation(dst=pg[:, cb:cb + w], op=nl.exp, data=s, bias=nb[:, g:g + 1], scale=scale)
                        if j0 + nj - 1 == g:
                            db = cb + (nj - 1) * attnk__P
                            nisa.affine_select(dst=pg[:, db:db + attnk__P], pattern=[[-1, attnk__P]], channel_multiplier=1, offset=0, cmp_op=nl.greater_equal, on_true_tile=pg[:, db:db + attnk__P], on_false_value=0.0)
                        dp = nl.ndarray((attnk__P, w), dtype=f32, buffer=nl.psum)
                        nisa.nc_matmul(dst=dp, stationary=doT[:, g * attnk__P:(g + 1) * attnk__P], moving=vT[:, j0 * attnk__P:j0 * attnk__P + w], accumulate=False)
                        nisa.scalar_tensor_tensor(dst=dsg[:, cb:cb + w], data=dp, op0=nl.subtract, operand0=dsum[:, g:g + 1], op1=nl.multiply, operand1=pg[:, cb:cb + w])
                        dst_ps = nl.ndarray((attnk__P, w), dtype=dt, buffer=nl.psum)
                        for jj in range(nj):
                            nisa.nc_transpose(dst=dst_ps[:, jj * attnk__P:(jj + 1) * attnk__P], data=dsg[:, cb + jj * attnk__P:cb + (jj + 1) * attnk__P])
                        dsT = nl.ndarray((attnk__P, w), dtype=dt, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=dsT, src=dst_ps, engine=attnk__copy_engine())
                        for jj in range(nj):
                            j = j0 + jj
                            nisa.nc_matmul(dst=dqacc, stationary=dsT[:, jj * attnk__P:(jj + 1) * attnk__P], moving=kn[:, j * D:(j + 1) * D], accumulate=j > 0)
                    dqt = nl.ndarray((attnk__P, D), dtype=dt, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=dqt, data=dqacc, op0=nl.multiply, operand0=scale)
                    nisa.dma_copy(dst=dq4[b, h, g * attnk__P:(g + 1) * attnk__P, :], src=dqt)
                for j in range(g0 + ng):
                    fg = j - g0
                    if fg < 0:
                        fg = 0
                    dv_ps = nl.ndarray((attnk__P, D), dtype=f32, buffer=nl.psum)
                    for gg in range(fg, ng):
                        g = g0 + gg
                        nisa.nc_matmul(dst=dv_ps, stationary=pg[:, gg * wg + j * attnk__P:gg * wg + (j + 1) * attnk__P], moving=don[:, g * D:(g + 1) * D], accumulate=gg > fg)
                    nisa.tensor_tensor(dst=dvacc[:, j * D:(j + 1) * D], data1=dvacc[:, j * D:(j + 1) * D], data2=dv_ps, op=nl.add)
                    dk_ps = nl.ndarray((attnk__P, D), dtype=f32, buffer=nl.psum)
                    for gg in range(fg, ng):
                        g = g0 + gg
                        nisa.nc_matmul(dst=dk_ps, stationary=dsg[:, gg * wg + j * attnk__P:gg * wg + (j + 1) * attnk__P], moving=qn[:, g * D:(g + 1) * D], accumulate=gg > fg)
                    nisa.scalar_tensor_tensor(dst=dkacc[:, j * D:(j + 1) * D], data=dk_ps, op0=nl.multiply, operand0=scale, op1=nl.add, operand1=dkacc[:, j * D:(j + 1) * D])
            for j in range(G):
                dkt = nl.ndarray((attnk__P, D), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dkt, src=dkacc[:, j * D:(j + 1) * D])
                nisa.dma_copy(dst=dk4[b, h, j * attnk__P:(j + 1) * attnk__P, :], src=dkt)
                dvt = nl.ndarray((attnk__P, D), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dvt, src=dvacc[:, j * D:(j + 1) * D])
                nisa.dma_copy(dst=dv3.ap(pattern=[[C, attnk__P], [1, D]], offset=(b * T + j * attnk__P) * C + h * D), src=dvt)
        return (dq4, dk4, dv3)

    def attnk__rope_tables(cos2, sin2, G, Dh):
        """cos / sin for this program as [P, G, 1, Dh] SBUF tiles ([p, g] = token g*128 + p), returned as [P, G, 2, Dh]
        views: stride-0 broadcasts over the q|k axis (the rope kernel's `broadcast` idiom) or materialised copies."""
        cosb = nl.ndarray((attnk__P, G, 1, Dh), dtype=cos2.dtype, buffer=nl.sbuf)
        sinb = nl.ndarray((attnk__P, G, 1, Dh), dtype=sin2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=cosb, src=cos2.ap(pattern=[[Dh, attnk__P], [attnk__P * Dh, G], [1, Dh]], offset=0))
        nisa.dma_copy(dst=sinb, src=sin2.ap(pattern=[[Dh, attnk__P], [attnk__P * Dh, G], [1, Dh]], offset=0))
        return (cosb.broadcast(2, 2), sinb.broadcast(2, 2))
        cos2t = nl.ndarray((attnk__P, G, 2, Dh), dtype=cos2.dtype, buffer=nl.sbuf)
        sin2t = nl.ndarray((attnk__P, G, 2, Dh), dtype=sin2.dtype, buffer=nl.sbuf)
        for s in range(2):
            nisa.tensor_copy(dst=cos2t[:, :, s, :], src=cosb[:, :, 0, :])
            nisa.tensor_copy(dst=sin2t[:, :, s, :], src=sinb[:, :, 0, :])
        return (cos2t, sin2t)

    def attnk__rope_rotate(dst, src, cb, sb, tmp_a, tmp_b, Dh, transpose):
        """dst = R(src) (transpose=False) or R^T(src) on [P, G, 2, 2*Dh] tiles; cb / sb are [P, G, 2, Dh] views.
        The six VectorE instructions of nki_rope_norm_opt._rotate, engines and dtypes as its defaults."""
        D = 2 * Dh
        s1, s2 = (src[:, :, :, 0:Dh], src[:, :, :, Dh:D])
        d1, d2 = (dst[:, :, :, 0:Dh], dst[:, :, :, Dh:D])
        nisa.tensor_tensor(dst=tmp_a, data1=s1, data2=cb, op=nl.multiply)
        nisa.tensor_tensor(dst=tmp_b, data1=s2, data2=sb, op=nl.multiply)
        nisa.tensor_tensor(dst=d1, data1=tmp_a, data2=tmp_b, op=nl.subtract if transpose else nl.add, engine=nisa.engine.unknown)
        nisa.tensor_tensor(dst=tmp_a, data1=s2, data2=cb, op=nl.multiply)
        nisa.tensor_tensor(dst=tmp_b, data1=s1, data2=sb, op=nl.multiply)
        nisa.tensor_tensor(dst=d2, data1=tmp_a, data2=tmp_b, op=nl.add if transpose else nl.subtract, engine=nisa.engine.unknown)

    def attnk__rope_inv_rms(r, sq, G, D):
        """inv[P, G, 2, 1] = rsqrt(mean_d r^2 + eps) in fp32: nki_rope_norm_opt._inv_rms with SQ=act, RSQRT=gpsimd
        (ScalarE square into the fp32 scratch sq, VectorE row reduce, the scale/eps tensor_scalar, the rsqrt)."""
        ss = nl.ndarray((attnk__P, G, 2, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=r)
        nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=3, keepdims=True)
        inv = nl.ndarray((attnk__P, G, 2, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ss, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=attnk__ROPE_EPS)
        nisa.tensor_scalar(dst=inv, data=ss, op0=nl.rsqrt, operand0=0.0)
        return inv

    def attnk__rope_qk_dma(tile, qkv3, b, h, T, C, G, D, store):
        """tile[P, G, 2, D] <- the q|k rows of head h of batch b (store=False), or the dq|dk columns of dqkv <- tile
        (store=True): row (p, g, s) <-> token g*128 + p, columns s*C + h*D .. of the [B, T, 3C] tensor."""
        C3 = 3 * C
        base = b * T * C3 + h * D
        for s in range(2):
            ap = qkv3.ap(pattern=[[C3, attnk__P], [attnk__P * C3, G], [1, D]], offset=base + s * C)
            if store:
                nisa.dma_copy(dst=ap, src=tile[:, :, s, :])
            else:
                nisa.dma_copy(dst=tile[:, :, s, :], src=ap)

    def attnk__rope_r_dma(tile, r5, b, h, B, H, T, G, D, store):
        """r5[s, b, h, g*128 + p, :] <-> tile[p, g, s, :] (the saved RoPE(q|k); 32 KiB contiguous per (s, g))."""
        base = (b * H + h) * T * D
        ss = B * H * T * D
        for s in range(2):
            ap = r5.ap(pattern=[[D, attnk__P], [attnk__P * D, G], [1, D]], offset=base + s * ss)
            if store:
                nisa.dma_copy(dst=ap, src=tile[:, :, s, :])
            else:
                nisa.dma_copy(dst=tile[:, :, s, :], src=ap)

    def attnk__rope_transpose(yn, s, T, D, G):
        """[D, T] SBUF tile of q (s = 0) or k (s = 1) from the normed natural tile yn[P, G, 2, D]: G TensorE
        transposes through PSUM in groups of 4 (the _transpose_nat form on the 4-D tile)."""
        xt = nl.ndarray((D, T), dtype=yn.dtype, buffer=nl.sbuf)
        for g0 in range(0, G, 4):
            ng = G - g0
            if ng > 4:
                ng = 4
            ps = nl.ndarray((D, ng * attnk__P), dtype=yn.dtype, buffer=nl.psum)
            for gg in range(ng):
                g = g0 + gg
                nisa.nc_transpose(dst=ps[:, gg * attnk__P:(gg + 1) * attnk__P], data=yn[:, g, s, :])
            nisa.tensor_copy(dst=xt[:, g0 * attnk__P:(g0 + ng) * attnk__P], src=ps, engine=attnk__copy_engine())
        return xt

    def attnk__rope_transpose_v81(yn, s, T, D, G):
        """[v81] _rope_transpose with the PSUM -> SBUF copies pinned to VectorE (the forward's ScalarE is the paced
        engine; gap 2 of the attention audit)."""
        xt = nl.ndarray((D, T), dtype=yn.dtype, buffer=nl.sbuf)
        for g0 in range(0, G, 4):
            ng = G - g0
            if ng > 4:
                ng = 4
            ps = nl.ndarray((D, ng * attnk__P), dtype=yn.dtype, buffer=nl.psum)
            for gg in range(ng):
                g = g0 + gg
                nisa.nc_transpose(dst=ps[:, gg * attnk__P:(gg + 1) * attnk__P], data=yn[:, g, s, :])
            nisa.tensor_copy(dst=xt[:, g0 * attnk__P:(g0 + ng) * attnk__P], src=ps, engine=nisa.engine.vector)
        return xt

    def attnk__rope_vaug(qkv3, b, h, T, C, G, D):
        """[P, G*(D+1)] tile of the V key tiles of head h with a ones column per block (the row sums), read from the
        [B, T, 3C] projection output (v = columns 2C + h*D ..)."""
        C3 = 3 * C
        W = D + 1
        vbase = b * T * C3 + 2 * C + h * D
        vaug = nl.ndarray((attnk__P, G * W), dtype=qkv3.dtype, buffer=nl.sbuf)
        nisa.memset(vaug, value=1.0)
        for j in range(G):
            nisa.dma_copy(dst=vaug[:, j * W:j * W + D], src=qkv3.ap(pattern=[[C3, attnk__P], [1, D]], offset=vbase + j * attnk__P * C3))
        return vaug

    @nki.jit
    def attnk__attn_rope_fwd_kernel(qkv3, cos2, sin2, scale, shift):
        """o3[B, T, C] and lse3[B*H, 128, G] (fp32) as the v7 forward, plus r5[2, B, H, T, D] = RoPE(q | k) (bf16,
        written only with SAVE=r) and inv3[B*H, 128, G, 2, 1] = 1/rms of every q | k row (fp32), from the projection
        output qkv3[B, T, 3C] and the tables cos2 / sin2 [T, D/2].  D = 128 (one head per tile row)."""
        B, T, C3 = qkv3.shape
        C = C3 // 3
        D = attnk__P
        H = C // D
        Dh = D // 2
        bs = B * H
        G = T // attnk__P
        W = D + 1
        f32 = nl.float32
        dt = qkv3.dtype
        o3 = nl.ndarray((B, T, C), dtype=dt, buffer=nl.shared_hbm)
        lse3 = nl.ndarray((bs, attnk__P, G), dtype=f32, buffer=nl.shared_hbm)
        inv3 = nl.ndarray((bs, attnk__P, G, 2, 1), dtype=f32, buffer=nl.shared_hbm)
        r5 = nl.ndarray((2, B, H, T, D), dtype=dt, buffer=nl.shared_hbm)
        pid, nprg = attnk__shard()
        cb, sb = attnk__rope_tables(cos2, sin2, G, Dh)
        nbias = nl.ndarray((attnk__P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(nbias, value=-shift)
        for i in range(pid, bs, nprg):
            b = i // H
            h = i % H
            xqk = nl.ndarray((attnk__P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            attnk__rope_qk_dma(xqk, qkv3, b, h, T, C, G, D, False)
            tmp_a = nl.ndarray((attnk__P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
            tmp_b = nl.ndarray((attnk__P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
            r = nl.ndarray((attnk__P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            attnk__rope_rotate(r, xqk, cb, sb, tmp_a, tmp_b, Dh, False)
            sq = nl.ndarray((attnk__P, G, 2, D), dtype=f32, buffer=nl.sbuf)
            inv = attnk__rope_inv_rms(r, sq, G, D)
            yn = nl.ndarray((attnk__P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yn, data1=r, data2=inv.broadcast(3, D), op=nl.multiply)
            attnk__rope_r_dma(r, r5, b, h, B, H, T, G, D, True)
            nisa.dma_copy(dst=inv3[i, :, :, :, :], src=inv)
            qT = attnk__rope_transpose_v81(yn, 0, T, D, G)
            kT = attnk__rope_transpose_v81(yn, 1, T, D, G)
            vaug = attnk__rope_vaug(qkv3, b, h, T, C, G, D)
            lse_blk = nl.ndarray((attnk__P, G), dtype=f32, buffer=nl.sbuf)
            sums = nl.ndarray((attnk__P, G), dtype=f32, buffer=nl.sbuf)
            for g in range(G):
                oacc = nl.ndarray((attnk__P, W), dtype=f32, buffer=nl.psum)
                for j0 in range(0, g + 1, attnk__KCHUNK):
                    nj = g + 1 - j0
                    if nj > attnk__KCHUNK:
                        nj = attnk__KCHUNK
                    st = nl.ndarray((attnk__P, nj * attnk__P), dtype=f32, buffer=nl.psum)
                    for jj in range(nj):
                        j = j0 + jj
                        nisa.nc_matmul(dst=st[:, jj * attnk__P:(jj + 1) * attnk__P], stationary=kT[:, j * attnk__P:(j + 1) * attnk__P], moving=qT[:, g * attnk__P:(g + 1) * attnk__P], accumulate=False)
                    et = nl.ndarray((attnk__P, nj * attnk__P), dtype=dt, buffer=nl.sbuf)
                    nisa.activation(dst=et, op=nl.exp, data=st, bias=nbias, scale=scale)
                    if j0 + nj - 1 == g:
                        jj = nj - 1
                        nisa.affine_select(dst=et[:, jj * attnk__P:(jj + 1) * attnk__P], pattern=[[1, attnk__P]], channel_multiplier=-1, offset=0, cmp_op=nl.greater_equal, on_true_tile=et[:, jj * attnk__P:(jj + 1) * attnk__P], on_false_value=0.0)
                    for jj in range(nj):
                        j = j0 + jj
                        nisa.nc_matmul(dst=oacc, stationary=et[:, jj * attnk__P:(jj + 1) * attnk__P], moving=vaug[:, j * W:(j + 1) * W], accumulate=j > 0)
                rs = nl.ndarray((attnk__P, 1), dtype=f32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rs, data=oacc[:, D:D + 1])
                ot = nl.ndarray((attnk__P, D), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=ot, data=oacc[:, 0:D], op0=nl.multiply, operand0=rs, engine=nisa.engine.vector)
                nisa.tensor_copy(dst=sums[:, g:g + 1], src=oacc[:, D:D + 1], engine=nisa.engine.vector)
                nisa.dma_copy(dst=o3.ap(pattern=[[C, attnk__P], [1, D]], offset=(b * T + g * attnk__P) * C + h * D), src=ot)
            nisa.activation(dst=lse_blk, op=nl.log, data=sums)
            nisa.tensor_scalar(dst=lse_blk, data=lse_blk, op0=nl.add, operand0=shift, engine=nisa.engine.vector)
            nisa.dma_copy(dst=lse3[i, :, :], src=lse_blk)
        return (o3, lse3, r5, inv3)

    @nki.jit
    def attnk__attn_rope_bwd_kernel(qkv3, r5, inv3, o3, do3, lse3, cos2, sin2, scale):
        """dqkv3[B, T, 3C] = dq | dk | dv (qkv3.dtype) from the projection output qkv3, the forward's r5 / inv3 (or,
        with SAVE=x, a recompute from qkv3), o3 / do3 [B, T, C], lse3 [B*H, 128, G] and the tables.  The v7 grouped
        backward (BWD_GROUP >= 2, SBUF accumulators) with the norm Jacobian + inverse rotation on its bf16 dQ / dK."""
        B, T, C3 = qkv3.shape
        C = C3 // 3
        D = attnk__P
        H = C // D
        Dh = D // 2
        bs = B * H
        G = T // attnk__P
        f32 = nl.float32
        dt = qkv3.dtype
        dqkv3 = nl.ndarray((B, T, C3), dtype=dt, buffer=nl.shared_hbm)
        pid, nprg = attnk__shard()
        cb, sb = attnk__rope_tables(cos2, sin2, G, Dh)
        for i in range(pid, bs, nprg):
            b = i // H
            h = i % H
            rr = nl.ndarray((attnk__P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            attnk__rope_r_dma(rr, r5, b, h, B, H, T, G, D, False)
            inv = nl.ndarray((attnk__P, G, 2, 1), dtype=f32, buffer=nl.sbuf)
            nisa.dma_copy(dst=inv, src=inv3[i, :, :, :, :])
            yn = nl.ndarray((attnk__P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=yn, data1=rr, data2=inv.broadcast(3, D), op=nl.multiply)
            on = attnk__load_nat_bthd(o3, b, h, T, C, D, G)
            don = attnk__load_nat_bthd(do3, b, h, T, C, D, G)
            qT = attnk__rope_transpose(yn, 0, T, D, G)
            kT = attnk__rope_transpose(yn, 1, T, D, G)
            doT = attnk__transpose_nat(don, T, D, G)
            vbase = b * T * C3 + 2 * C + h * D
            vn = nl.ndarray((attnk__P, G * D), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=vn, src=qkv3.ap(pattern=[[C3, attnk__P], [attnk__P * C3, G], [1, D]], offset=vbase))
            vT = attnk__transpose_nat(vn, T, D, G)
            nb = nl.ndarray((attnk__P, G), dtype=f32, buffer=nl.sbuf)
            nisa.dma_copy(dst=nb, src=lse3[i, :, :])
            nisa.tensor_scalar(dst=nb, data=nb, op0=nl.multiply, operand0=-1.0)
            dsum = nl.ndarray((attnk__P, G), dtype=f32, buffer=nl.sbuf)
            tmp_all = nl.ndarray((attnk__P, G * D), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=tmp_all, data1=don, data2=on, op=nl.multiply)
            nisa.tensor_reduce(dst=dsum, op=nl.add, data=tmp_all.reshape_dim(1, (G, D)), axis=2)
            dkacc = nl.ndarray((attnk__P, G * D), dtype=f32, buffer=nl.sbuf)
            dvacc = nl.ndarray((attnk__P, G * D), dtype=f32, buffer=nl.sbuf)
            gqk = nl.ndarray((attnk__P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            for g0 in range(0, G, attnk__BWD_GROUP):
                ng = G - g0
                if ng > attnk__BWD_GROUP:
                    ng = attnk__BWD_GROUP
                wg = (g0 + ng) * attnk__P
                pg = nl.ndarray((attnk__P, ng * wg), dtype=dt, buffer=nl.sbuf)
                dsg = nl.ndarray((attnk__P, ng * wg), dtype=dt, buffer=nl.sbuf)
                for gg in range(ng):
                    g = g0 + gg
                    dqacc = nl.ndarray((attnk__P, D), dtype=f32, buffer=nl.psum)
                    for j0 in range(0, g + 1, attnk__KCHUNK):
                        nj = g + 1 - j0
                        if nj > attnk__KCHUNK:
                            nj = attnk__KCHUNK
                        w = nj * attnk__P
                        cbk = gg * wg + j0 * attnk__P
                        s = nl.ndarray((attnk__P, w), dtype=f32, buffer=nl.psum)
                        nisa.nc_matmul(dst=s, stationary=qT[:, g * attnk__P:(g + 1) * attnk__P], moving=kT[:, j0 * attnk__P:j0 * attnk__P + w], accumulate=False)
                        nisa.activation(dst=pg[:, cbk:cbk + w], op=nl.exp, data=s, bias=nb[:, g:g + 1], scale=scale)
                        if j0 + nj - 1 == g:
                            db = cbk + (nj - 1) * attnk__P
                            nisa.affine_select(dst=pg[:, db:db + attnk__P], pattern=[[-1, attnk__P]], channel_multiplier=1, offset=0, cmp_op=nl.greater_equal, on_true_tile=pg[:, db:db + attnk__P], on_false_value=0.0)
                        dp = nl.ndarray((attnk__P, w), dtype=f32, buffer=nl.psum)
                        nisa.nc_matmul(dst=dp, stationary=doT[:, g * attnk__P:(g + 1) * attnk__P], moving=vT[:, j0 * attnk__P:j0 * attnk__P + w], accumulate=False)
                        nisa.scalar_tensor_tensor(dst=dsg[:, cbk:cbk + w], data=dp, op0=nl.subtract, operand0=dsum[:, g:g + 1], op1=nl.multiply, operand1=pg[:, cbk:cbk + w])
                        dst_ps = nl.ndarray((attnk__P, w), dtype=dt, buffer=nl.psum)
                        for jj in range(nj):
                            nisa.nc_transpose(dst=dst_ps[:, jj * attnk__P:(jj + 1) * attnk__P], data=dsg[:, cbk + jj * attnk__P:cbk + (jj + 1) * attnk__P])
                        dsT = nl.ndarray((attnk__P, w), dtype=dt, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=dsT, src=dst_ps, engine=attnk__copy_engine())
                        for jj in range(nj):
                            j = j0 + jj
                            nisa.nc_matmul(dst=dqacc, stationary=dsT[:, jj * attnk__P:(jj + 1) * attnk__P], moving=yn[:, j, 1, :], accumulate=j > 0)
                    nisa.tensor_scalar(dst=gqk[:, g, 0, :], data=dqacc, op0=nl.multiply, operand0=scale)
                for j in range(g0 + ng):
                    fg = j - g0
                    if fg < 0:
                        fg = 0
                    dv_ps = nl.ndarray((attnk__P, D), dtype=f32, buffer=nl.psum)
                    for gg in range(fg, ng):
                        g = g0 + gg
                        nisa.nc_matmul(dst=dv_ps, stationary=pg[:, gg * wg + j * attnk__P:gg * wg + (j + 1) * attnk__P], moving=don[:, g * D:(g + 1) * D], accumulate=gg > fg)
                    if attnk__V81 and j >= g0:
                        nisa.tensor_copy(dst=dvacc[:, j * D:(j + 1) * D], src=dv_ps, engine=nisa.engine.scalar)
                    else:
                        nisa.tensor_tensor(dst=dvacc[:, j * D:(j + 1) * D], data1=dvacc[:, j * D:(j + 1) * D], data2=dv_ps, op=nl.add)
                    dk_ps = nl.ndarray((attnk__P, D), dtype=f32, buffer=nl.psum)
                    for gg in range(fg, ng):
                        g = g0 + gg
                        nisa.nc_matmul(dst=dk_ps, stationary=dsg[:, gg * wg + j * attnk__P:gg * wg + (j + 1) * attnk__P], moving=yn[:, g, 0, :], accumulate=gg > fg)
                    if attnk__V81 and j >= g0:
                        nisa.tensor_scalar(dst=dkacc[:, j * D:(j + 1) * D], data=dk_ps, op0=nl.multiply, operand0=scale, engine=nisa.engine.scalar)
                    else:
                        nisa.scalar_tensor_tensor(dst=dkacc[:, j * D:(j + 1) * D], data=dk_ps, op0=nl.multiply, operand0=scale, op1=nl.add, operand1=dkacc[:, j * D:(j + 1) * D])
            nisa.tensor_copy(dst=gqk[:, :, 1, :], src=dkacc.reshape_dim(1, (G, D)), engine=nisa.engine.scalar)
            dvt_all = nl.ndarray((attnk__P, G, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dvt_all, src=dvacc.reshape_dim(1, (G, D)), engine=nisa.engine.scalar)
            nisa.dma_copy(dst=dqkv3.ap(pattern=[[C3, attnk__P], [attnk__P * C3, G], [1, D]], offset=b * T * C3 + 2 * C + h * D), src=dvt_all)
            jdt = dt
            sq = nl.ndarray((attnk__P, G, 2, D), dtype=jdt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=gqk, data2=rr, op=nl.multiply)
            dot = nl.ndarray((attnk__P, G, 2, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=dot, op=nl.add, data=sq, axis=3, keepdims=True)
            coef = nl.ndarray((attnk__P, G, 2, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=coef, data1=inv, data2=inv, op=nl.multiply)
            nisa.tensor_tensor(dst=coef, data1=coef, data2=dot, op=nl.multiply)
            nisa.tensor_scalar(dst=coef, data=coef, op0=nl.multiply, operand0=1.0 / D)
            u = nl.ndarray((attnk__P, G, 2, D), dtype=jdt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=u, data1=rr, data2=coef.broadcast(3, D), op=nl.multiply)
            nisa.tensor_tensor(dst=u, data1=gqk, data2=u, op=nl.subtract, engine=nisa.engine.unknown)
            gr = nl.ndarray((attnk__P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=gr, data1=u, data2=inv.broadcast(3, D), op=nl.multiply)
            gx = nl.ndarray((attnk__P, G, 2, D), dtype=dt, buffer=nl.sbuf)
            tmp_a = nl.ndarray((attnk__P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
            tmp_b = nl.ndarray((attnk__P, G, 2, Dh), dtype=dt, buffer=nl.sbuf)
            attnk__rope_rotate(gx, gr, cb, sb, tmp_a, tmp_b, Dh, True)
            attnk__rope_qk_dma(gx, dqkv3, b, h, T, C, G, D, True)
        return dqkv3
    attnk__HAS_NKI = True
except Exception:
    attnk__HAS_NKI = False

def attnk_kernel_available(device):
    return (attnk__HAS_NKI and device.type == 'neuron') and (not attnk__KERNEL_DISABLED)

def attnk__disable_kernel(where, err):
    global attnk__KERNEL_DISABLED, attnk__LAST_KERNEL_ERROR
    if not attnk__KERNEL_DISABLED:
        attnk__KERNEL_DISABLED = True
        attnk__LAST_KERNEL_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[nki_attn] kernel disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)

def attnk__launch(kernel):
    return kernel[attnk__LNC] if attnk__LNC == 2 else kernel

def attnk__fwd_kernel():
    return attnk__launch(attnk__attn_fwd_kernel)

def attnk__bwd_kernel():
    return attnk__launch(attnk__attn_bwd_kernel)

def attnk_shape_ok(B, H, T, D):
    return B > 0 and H > 0 and (0 < D <= attnk__P) and (T > 0) and (T % attnk__P == 0) and (T <= attnk__MAX_T)

def attnk_fixed_shift(scale, D):
    """c = scale * D: the bound on |scale * q . k| after the parameter-free QK-norm (|q|, |k| <= sqrt(D))."""
    return float(scale) * D + attnk__SHIFT_MARGIN

class attnk_NkiAttnFn(torch.autograd.Function):
    """y[B, T, H, D] = causal SDPA(q, k, v) via the NKI kernels; eager math if a kernel raises."""

    @staticmethod
    def forward(ctx, q, k, v, scale, shift):
        B, H, T, D = q.shape
        ctx.scale, ctx.shift = (scale, shift)
        ctx.fallback = False
        try:
            o3, lse3 = attnk__fwd_kernel()(q, k, v, float(scale), float(shift))
            o = o3.view(B, T, H, D)
        except Exception as err:
            attnk__disable_kernel('forward', err)
            ctx.fallback = True
            ctx.save_for_backward(q, k, v)
            with torch.no_grad():
                return attnk_eager_causal_attention(q, k, v, scale)
        ctx.save_for_backward(q, k, v, o, lse3)
        return o

    @staticmethod
    def backward(ctx, gy):
        if ctx.fallback:
            q, k, v = ctx.saved_tensors
            gq, gk, gv = attnk__eager_backward(q, k, v, gy, ctx.scale)
            return (gq, gk, gv, None, None)
        q, k, v, o, lse3 = ctx.saved_tensors
        B, H, T, D = q.shape
        C = H * D
        if not attnk__KERNEL_DISABLED:
            try:
                go = gy.contiguous().view(B, T, C)
                o3 = o.view(B, T, C)
                dq, dk, dv3 = attnk__bwd_kernel()(q, k, v, o3, go, lse3, float(ctx.scale))
                dv = dv3.view(B, T, H, D).permute(0, 2, 1, 3)
                return (dq, dk, dv, None, None)
            except Exception as err:
                attnk__disable_kernel('backward', err)
        gq, gk, gv = attnk__eager_backward(q, k, v, gy, ctx.scale)
        return (gq, gk, gv, None, None)

def attnk__kernel_ok(q, k, v):
    if not attnk_kernel_available(q.device):
        return False
    if q.dim() != 4 or q.shape != k.shape or q.shape != v.shape:
        return False
    if not q.dtype == k.dtype == v.dtype or q.dtype not in (torch.bfloat16, torch.float32):
        return False
    if q.dtype == torch.float32 and (not attnk_ALLOW_FP32):
        return False
    B, H, T, D = q.shape
    return attnk_shape_ok(B, H, T, D)

def attnk_nki_causal_attention(q, k, v, scale=None):
    """Causal SDPA(q, k, v) for [B, H, T, D] inputs, returned as [B, T, H, D]; NKI kernels when applicable."""
    sc = float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    if not attnk__kernel_ok(q, k, v):
        return attnk_eager_causal_attention(q, k, v, sc if scale is not None else None)
    q = q if q.is_contiguous() else q.contiguous()
    k = k if k.is_contiguous() else k.contiguous()
    v = v if v.is_contiguous() else v.contiguous()
    return attnk_NkiAttnFn.apply(q, k, v, sc, attnk_fixed_shift(sc, q.shape[-1]))

def attnk__rope_fwd_kernel():
    return attnk__launch(attnk__attn_rope_fwd_kernel)

def attnk__rope_bwd_kernel():
    return attnk__launch(attnk__attn_rope_bwd_kernel)

def attnk__tables_2d(cos, sin, T, Dh, dtype):
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
    return (c2, s2)

def attnk__eager_rope_norm(x, cos2, sin2):
    """train.py's norm(apply_rotary_emb(x, cos, sin)).transpose(1, 2) on x [B, T, H, D] (the eager fallback)."""
    B, T, H, D = x.shape
    Dh = D // 2
    c = cos2.reshape(1, T, 1, Dh)
    s = sin2.reshape(1, T, 1, Dh)
    x1, x2 = (x[..., :Dh], x[..., Dh:])
    r = torch.cat((x1 * c + x2 * s, x1 * -s + x2 * c), dim=-1)
    return F.rms_norm(r, (D,)).transpose(1, 2)

def attnk__eager_rope_attention(qkv, cos2, sin2, H, scale):
    """The eager chain on the projection output: RoPE + QK-norm on q and k, causal SDPA, [B, T, H, D] out."""
    B, T, C3 = qkv.shape
    C = C3 // 3
    D = C // H
    q, k, v = qkv.split(C, dim=-1)
    q = attnk__eager_rope_norm(q.reshape(B, T, H, D), cos2, sin2)
    k = attnk__eager_rope_norm(k.reshape(B, T, H, D), cos2, sin2)
    v = v.reshape(B, T, H, D).transpose(1, 2)
    return attnk_eager_causal_attention(q, k, v, scale)

def attnk__eager_rope_backward(qkv, cos2, sin2, H, scale, gy):
    with torch.enable_grad():
        qr = qkv.detach().requires_grad_(True)
        y = attnk__eager_rope_attention(qr, cos2, sin2, H, scale)
        gqkv, = torch.autograd.grad(y, qr, gy)
    return gqkv

def attnk__rope_fwd_launch(qkv, cos2, sin2, H, scale, shift):
    B = qkv.shape[0]
    return attnk__rope_fwd_kernel()(qkv, cos2, sin2, float(scale), float(shift))

def attnk__rope_bwd_launch(qkv, r5, inv3, o3, go, lse3, cos2, sin2, H, scale):
    B = qkv.shape[0]
    return attnk__rope_bwd_kernel()(qkv, r5, inv3, o3, go, lse3, cos2, sin2, float(scale))

class attnk_NkiRopeAttnFn(torch.autograd.Function):
    """y[B, T, H, D] = causal SDPA(QKnorm(RoPE(q)), QKnorm(RoPE(k)), v) from qkv [B, T, 3C] via the fused NKI
    kernels; dqkv [B, T, 3C] from the fused backward; the eager math if a kernel raises."""

    @staticmethod
    def forward(ctx, qkv, cos2, sin2, H, scale, shift):
        B, T, C3 = qkv.shape
        C = C3 // 3
        D = C // H
        ctx.scale, ctx.shift, ctx.H = (scale, shift, H)
        ctx.fallback = False
        try:
            o3, lse3, r5, inv3 = attnk__rope_fwd_launch(qkv, cos2, sin2, H, scale, shift)
            o = o3.view(B, T, H, D)
        except Exception as err:
            attnk__disable_kernel('rope forward', err)
            ctx.fallback = True
            ctx.save_for_backward(qkv, cos2, sin2)
            with torch.no_grad():
                return attnk__eager_rope_attention(qkv, cos2, sin2, H, scale)
        ctx.save_for_backward(qkv, cos2, sin2, r5, inv3, o, lse3)
        return o

    @staticmethod
    def backward(ctx, gy):
        if ctx.fallback:
            qkv, cos2, sin2 = ctx.saved_tensors
            return (attnk__eager_rope_backward(qkv, cos2, sin2, ctx.H, ctx.scale, gy), None, None, None, None, None)
        qkv, cos2, sin2, r5, inv3, o, lse3 = ctx.saved_tensors
        B, T, C3 = qkv.shape
        C = C3 // 3
        if not attnk__KERNEL_DISABLED:
            try:
                go = gy.contiguous().view(B, T, C)
                o3 = o.view(B, T, C)
                dqkv = attnk__rope_bwd_launch(qkv, r5, inv3, o3, go, lse3, cos2, sin2, ctx.H, ctx.scale)
                return (dqkv.view(B, T, C3), None, None, None, None, None)
            except Exception as err:
                attnk__disable_kernel('rope backward', err)
        return (attnk__eager_rope_backward(qkv, cos2, sin2, ctx.H, ctx.scale, gy), None, None, None, None, None)

def attnk_rope_attention_ok(qkv, cos, sin, n_head):
    """True when the fused path applies: switch on, the attention kernel usable, a contiguous [B, T, 3C] bf16 (fp32
    with R1_NKI_ATTN_FP32=1) projection output with D == 128, T a multiple of 128 up to 4096, tables for T tokens,
    and the grouped backward (the fused backward is the v7 grouped loop)."""
    if not attnk_kernel_available(qkv.device):
        return False
    if qkv.dim() != 3 or qkv.dtype not in (torch.bfloat16, torch.float32):
        return False
    if qkv.dtype == torch.float32 and (not attnk_ALLOW_FP32):
        return False
    B, T, C3 = qkv.shape
    if C3 % 3 or n_head <= 0:
        return False
    C = C3 // 3
    if C % n_head or C // n_head != attnk__P:
        return False
    if not attnk_shape_ok(B, n_head, T, attnk__P) or not qkv.is_contiguous():
        return False
    Dh = attnk__P // 2
    return cos.numel() == T * Dh and sin.numel() == T * Dh

def attnk_nki_rope_attention(qkv, cos, sin, n_head, scale=None):
    """y [B, T, H, D] = causal SDPA(QKnorm(RoPE(q)), QKnorm(RoPE(k)), v) straight from the fused projection output
    qkv [B, T, 3C] (one NKI launch each way, dqkv written as one [B, T, 3C] tensor); None when the fused path does
    not apply, so the caller keeps today's rope_norm -> nki_causal_attention chain."""
    if not attnk_rope_attention_ok(qkv, cos, sin, n_head):
        return None
    D = attnk__P
    sc = float(scale) if scale is not None else 1.0 / math.sqrt(D)
    cos2, sin2 = attnk__tables_2d(cos, sin, qkv.shape[1], D // 2, qkv.dtype)
    return attnk_NkiRopeAttnFn.apply(qkv, cos2, sin2, n_head, sc, attnk_fixed_shift(sc, D))
nki_attn_fn = attnk_nki_causal_attention
USE_NKI_ATTN = attnk_USE_NKI_ATTN
nki_rope_attention = attnk_nki_rope_attention
USE_NKI_ATTN_ROPE = attnk_USE_NKI_ATTN_ROPE
"NKI embedding backward: the token-embedding gradient as a one-hot contraction on TensorE (plan-v10 lane L3).\n\nReplaces, in train.py's GPT.forward,\n\n    x = self.transformer.wte(idx).to(COMPUTE_DTYPE)            # F.embedding (fp32 rows) -> bf16 cast\n\nwith\n\n    x = embedding_fwd(idx, self.transformer.wte.weight, COMPUTE_DTYPE)\n\nThe forward is unchanged (the same two eager ops).  The backward replaces autograd's chain\n(bf16 -> fp32 cast of dY [N, C], 0.18 ms; `embedding_dense_backward`, a 16k-row scatter-add at ~9 GB/s, 10.6 ms)\nwith ONE NKI launch that reads dY once per vocab pass and writes dW [V, C] fp32 once.\n\nMath\n----\n    dW[v, :] = sum_{n : idx[n] == v} dY[n, :]     ==     dW = OneHot^T @ dY,   OneHot[n, v] = [idx[n] == v]\n\nKernel design (kernel[2]: program p owns vocab rows [p V/2, (p+1) V/2))\n------------------------------------------------------------------------\n* Tokens n = p * NT + t live on partition p, tile t: dY is viewed as [128, NT, C] so tile t is the HBM slice\n  dy3[:, t, :] -- one contiguous 2 KiB run per partition (C = 1024 bf16) -- and the ids are ONE [128, NT] block\n  loaded once per program (int32 -> fp32 by the DMA).  The token order is irrelevant to the sum.\n* Pass structure: VPP (default 4) vocab tiles of 128 per pass = W = 512 one-hot columns, the width of one iota row;\n  per pass VPP x (C / 512) = 8 PSUM accumulators [128 vocab, 512 C] fp32 = the whole PSUM (8 banks x 2 KiB).\n  Per token tile: load dY tile, build onehot[128 tok, W] bf16 = (iota == idx - v0) in one VectorE tensor_scalar,\n  then 8 nc_matmul(stationary = onehot[:, v-tile] [128 tok, 128 vocab], moving = dY[:, c-half] [128 tok, 512 C])\n  accumulated in fp32 PSUM over the NT token tiles.  After the token loop each accumulator is copied to SBUF and\n  DMAed to its 128 dW rows (4 KiB contiguous per partition).  NPASS = V / (2 * 512) = 8 passes per program, so\n  dY is re-read 8 times per core (8 x 32 MiB, overlapped with the matmuls).\n* Numerics: products are exact (1.0 x bf16), every sum is fp32 (TensorE internal + PSUM) -- the same fp32 sum\n  of the same bf16 rows as `embedding_dense_backward`, reassociated.  Duplicate ids are correct by construction.\n* Cost model: 8 passes x NT x 8 matmuls = 8,192 matmuls per core at NT = 128 (B = 16), 275 GFLOP per chip;\n  2.2-2.8 ms on TensorE plus ~0.3 ms of evacuation; target <= 5 ms vs 10.8 ms eager.\n\nN % 128 == 0, C % 512 == 0, V % (128 * VPP * LNC) == 0; a kernel failure prints once, disables the kernel for\nthe process and answers with the eager backward (`embedding_dense_backward`), which is also what the wrapper\nuses whenever the kernel does not apply.  The eval path never calls the backward.\n\nKnobs (env, read once at import; module-level constants because kernel bodies only see module globals):\n"
import os
import torch
import torch.nn.functional as F
embk__P = 128
embk_USE_NKI_EMB_BWD = True
embk__VPP = 4
embk__NBUF = max(1, min(4, 4))
embk__ONEZERO = False
embk__MOVING = 512
embk__LNC = int(os.environ.get('NEURON_LOGICAL_NC_CONFIG', '1'))
embk__KERNEL_DISABLED = False
embk__LAST_KERNEL_ERROR = None
embk__W = embk__VPP * embk__P

def embk_eager_embedding(idx: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """train.py's two ops: F.embedding on the fp32 table, cast to the compute dtype."""
    return F.embedding(idx, weight).to(out_dtype)

def embk_eager_embedding_backward(dy: torch.Tensor, idx: torch.Tensor, V: int, C: int, wdtype: torch.dtype) -> torch.Tensor:
    """What autograd runs today for `F.embedding(idx, w).to(bf16)`: the cast's backward (dY -> w.dtype), then
    aten::embedding_dense_backward (padding_idx -1, no frequency scaling).  index_add_ if the op is unavailable."""
    g = dy.reshape(-1, C).to(wdtype)
    ids = idx.reshape(-1)
    try:
        return torch.ops.aten.embedding_dense_backward(g, ids, V, -1, False)
    except Exception:
        dw = torch.zeros(V, C, dtype=wdtype, device=dy.device)
        dw.index_add_(0, ids.long(), g)
        return dw
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    def embk__dma(dst, src):
        nisa.dma_copy(dst=dst, src=src)

    def embk__shard():
        """(program_id, num_programs); (0, 1) without a launch grid or with sharding off."""
        if nl.program_ndim() > 0:
            return (nl.program_id(0), nl.num_programs(0))
        return (0, 1)

    def embk__iota_row(P, W):
        """[P, W] fp32 tile with 0, 1, ..., W-1 on every partition (GpSimd, once per program)."""
        io = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=io, pattern=[[1, W]], offset=0, channel_multiplier=0)
        return io

    def embk__idx_src(idx3):
        """HBM view of the ids: [P, NT] int32, or the low words of int64 pairs ([P, NT, 2] view)."""
        if len(idx3.shape) == 3:
            return idx3[:, :, 0]
        return idx3

    def embk__tiles(n, shape, dtype, buf):
        """n tiles as a tuple (n in 1, 2, 3, 4, 8).  Explicit, not a comprehension, and no `raise`: the NKI frontend
        walks every branch of a helper and rejects comprehensions and raise statements even in untaken ones."""
        if n == 1:
            return (nl.ndarray(shape, dtype=dtype, buffer=buf),)
        if n == 2:
            return (nl.ndarray(shape, dtype=dtype, buffer=buf), nl.ndarray(shape, dtype=dtype, buffer=buf))
        if n == 3:
            return (nl.ndarray(shape, dtype=dtype, buffer=buf), nl.ndarray(shape, dtype=dtype, buffer=buf), nl.ndarray(shape, dtype=dtype, buffer=buf))
        if n == 4:
            return (nl.ndarray(shape, dtype=dtype, buffer=buf), nl.ndarray(shape, dtype=dtype, buffer=buf), nl.ndarray(shape, dtype=dtype, buffer=buf), nl.ndarray(shape, dtype=dtype, buffer=buf))
        return (nl.ndarray(shape, dtype=dtype, buffer=buf), nl.ndarray(shape, dtype=dtype, buffer=buf), nl.ndarray(shape, dtype=dtype, buffer=buf), nl.ndarray(shape, dtype=dtype, buffer=buf), nl.ndarray(shape, dtype=dtype, buffer=buf), nl.ndarray(shape, dtype=dtype, buffer=buf), nl.ndarray(shape, dtype=dtype, buffer=buf), nl.ndarray(shape, dtype=dtype, buffer=buf))

    def embk__onehot(oh, io, idxb, ids, t, v0):
        """oh[P, W] bf16 = [iota == idx - v0] for token tile t (one VectorE instruction)."""
        nisa.tensor_scalar(dst=oh, data=io, op0=nl.equal, operand0=ids[:, t:t + 1])

    def embk__matmuls(accs, oh, dyb, CC, acc_flag):
        """The VPP x CC matmuls of one token tile: accs[v*CC + c] (+)= onehot[:, v-tile]^T @ dY[:, c-chunk]."""
        for v in range(embk__VPP):
            for c in range(CC):
                nisa.nc_matmul(dst=accs[v * CC + c], stationary=oh[:, v * embk__P:(v + 1) * embk__P], moving=dyb[:, c * embk__MOVING:(c + 1) * embk__MOVING], accumulate=acc_flag, is_stationary_onezero=embk__ONEZERO)

    def embk__acc_flag(first):
        """accumulate= for a tile: ACC=expr -> False on the first tile, True after; none -> compiler-inferred."""
        return first

    def embk__store_psum(dw, accs, CC, C, v0):
        """PSUM accumulators -> SBUF [P, C] fp32 per vocab tile -> dW rows (4 KiB contiguous per partition)."""
        for v in range(embk__VPP):
            ob = nl.ndarray((embk__P, C), dtype=nl.float32, buffer=nl.sbuf)
            for c in range(CC):
                nisa.tensor_copy(dst=ob[:, c * embk__MOVING:(c + 1) * embk__MOVING], src=accs[v * CC + c])
            embk__dma(dst=dw[v0 + v * embk__P:v0 + (v + 1) * embk__P, :], src=ob)

    @nki.jit
    def embk__emb_bwd_kernel(idx3, dy3, V):
        """dw[V, C] fp32 = OneHot(idx)^T @ dy3 for dy3[P, NT, C] bf16 and idx3[P, NT] int32 (token n = p*NT + t)."""
        P, NT, C = dy3.shape
        f32, bf16 = (nl.float32, nl.bfloat16)
        dw = nl.ndarray((V, C), dtype=f32, buffer=nl.shared_hbm)
        pid, nprg = embk__shard()
        VTp = V // P // nprg
        NPASS = VTp // embk__VPP
        CC = C // embk__MOVING
        NACC = embk__VPP * CC
        io = embk__iota_row(P, embk__W)
        idxb = nl.ndarray((P, NT), dtype=f32, buffer=nl.sbuf)
        embk__dma(dst=idxb, src=embk__idx_src(idx3))
        for ps in range(NPASS):
            v0 = (pid * VTp + ps * embk__VPP) * P
            ids = nl.ndarray((P, NT), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ids, data=idxb, op0=nl.subtract, operand0=v0 * 1.0)
            accs = embk__tiles(NACC, (P, embk__MOVING), f32, nl.psum)
            dybs = embk__tiles(embk__NBUF, (P, C), dy3.dtype, nl.sbuf)
            ohs = embk__tiles(2, (P, embk__W), bf16, nl.sbuf)
            for t in range(min(embk__NBUF - 1, NT)):
                embk__dma(dst=dybs[t % embk__NBUF], src=dy3[:, t, :])
            for t in range(NT):
                tn = t + embk__NBUF - 1
                if tn < NT and embk__NBUF > 1:
                    embk__dma(dst=dybs[tn % embk__NBUF], src=dy3[:, tn, :])
                oh = ohs[t % 2]
                embk__onehot(oh, io, idxb, ids, t, v0)
                embk__matmuls(accs, oh, dybs[t % embk__NBUF], CC, embk__acc_flag(t > 0))
            embk__store_psum(dw, accs, CC, C, v0)
        return dw
    embk__HAS_NKI = True
except Exception:
    embk__HAS_NKI = False

def embk__disable_kernel(where: str, err: BaseException) -> None:
    global embk__KERNEL_DISABLED, embk__LAST_KERNEL_ERROR
    if not embk__KERNEL_DISABLED:
        embk__KERNEL_DISABLED = True
        embk__LAST_KERNEL_ERROR = f'{where}: {type(err).__name__}: {err}'
        print(f'[nki_embedding] kernel disabled after failure in {where}: {type(err).__name__}: {err}', flush=True)

def embk__launch(kernel):
    """kernel[2] under an LNC=2 runtime (each physical core owns half the vocab rows), plain call otherwise."""
    return kernel[embk__LNC] if embk__LNC == 2 else kernel

def embk_kernel_idx(idx: torch.Tensor) -> torch.Tensor:
    """The ids as the kernel wants them: [P, NT] int32, or (IDX=view) int64 bit patterns as [P, NT, 2] int32."""
    N = idx.numel()
    NT = N // embk__P
    flat = idx.reshape(N)
    if flat.dtype == torch.int32:
        return flat.contiguous().view(embk__P, NT)
    if flat.dtype == torch.int64:
        try:
            return flat.contiguous().view(torch.int32).view(embk__P, NT, 2)
        except Exception:
            pass
    return flat.to(torch.int32).view(embk__P, NT)

def embk_kernel_dy(dy: torch.Tensor, C: int) -> torch.Tensor:
    N = dy.numel() // C
    return dy.reshape(N, C).contiguous().view(embk__P, N // embk__P, C)

def embk__run_kernel(idx_k: torch.Tensor, dy3: torch.Tensor, V: int) -> torch.Tensor:
    """One launch; tests monkeypatch this to run the emulation through the same plumbing on CPU."""
    return embk__launch(embk__emb_bwd_kernel)(idx_k, dy3, int(V))

def embk__kernel_usable() -> bool:
    return embk__HAS_NKI and (not embk__KERNEL_DISABLED)

def embk__device_ok(t: torch.Tensor) -> bool:
    """The kernel runs on the Neuron device only (tests monkeypatch this to exercise the plumbing on CPU)."""
    return t.device.type == 'neuron'

def embk_shapes_ok(idx: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype) -> bool:
    V, C = weight.shape
    N = idx.numel()
    nprg = 2 if embk__LNC == 2 else 1
    return out_dtype == torch.bfloat16 and weight.dtype == torch.float32 and (idx.dtype in (torch.int32, torch.int64)) and (N > 0) and (N % embk__P == 0) and (C % embk__MOVING == 0) and (C // embk__MOVING * embk__VPP in (1, 2, 4, 8)) and (V % (embk__P * embk__VPP * nprg) == 0) and (embk__VPP in (1, 2, 4))

class embk_EmbeddingFn(torch.autograd.Function):
    """out = F.embedding(idx, weight).to(out_dtype); backward: dW via the NKI kernel (eager if it raises)."""

    @staticmethod
    def forward(ctx, idx, weight, out_dtype, shadow=None):
        out = F.embedding(idx, shadow) if shadow is not None else embk_eager_embedding(idx, weight, out_dtype)
        ctx.save_for_backward(idx)
        ctx.wshape = tuple(weight.shape)
        ctx.wdtype = weight.dtype
        return out

    @staticmethod
    def backward(ctx, dy):
        idx, = ctx.saved_tensors
        V, C = ctx.wshape
        if embk__kernel_usable() and dy.dtype == torch.bfloat16 and embk__device_ok(dy):
            try:
                dw = embk__run_kernel(embk_kernel_idx(idx), embk_kernel_dy(dy, C), V)
                return (None, dw, None, None)
            except Exception as err:
                embk__disable_kernel('backward', err)
        return (None, embk_eager_embedding_backward(dy, idx, V, C, ctx.wdtype), None, None)

def embk_embedding_fwd(idx: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype, shadow=None) -> torch.Tensor:
    """train.py's `wte(idx).to(dtype)` with the NKI backward when it applies; the eager ops otherwise.  [glue f1]
    shadow (optional): the table in out_dtype; used by the kernel path's forward only (the eager fallback keeps the
    cast so its autograd still reaches `weight`)."""
    if embk__kernel_usable() and embk__device_ok(weight) and embk_shapes_ok(idx, weight, out_dtype):
        if shadow is not None:
            return embk_EmbeddingFn.apply(idx, weight, out_dtype, shadow)
        return embk_EmbeddingFn.apply(idx, weight, out_dtype)
    return embk_eager_embedding(idx, weight, out_dtype)
embedding_fwd = embk_embedding_fwd
USE_NKI_EMB_BWD = embk_USE_NKI_EMB_BWD

class Linear(nn.Linear):
    """Linear without bias that casts weight to input dtype (fp32 master weights, bf16 compute)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inplace = _accum_inplace(self, x, 'sq' if self.in_features == self.out_features else 'wide')
        if USE_NKI_GEMM_FP8_NN and USE_NKI_GEMM and nki_gemm_fp8_nn.proj_active(self.training) and (getattr(self, 'gemm_class', None) == 'proj'):
            return nki_gemm_fp8_nn.c_proj(x, self.weight if inplace else _weight_for(self, x), inplace)
        if USE_NKI_GEMM and getattr(self, 'gemm_class', None) == 'proj':
            if USE_NKI_GEMM_FP8_NN:
                x = nki_gemm_fp8_nn.row2_cap(x)
            return _LinearGemm.apply(x, self.weight if inplace else _weight_for(self, x), False, inplace)
        if inplace:
            return _LinearAccum.apply(x, self.weight, False)
        return F.linear(x, _weight_for(self, x), None)

class LinearT(Linear):
    """[R1 L2] Linear whose weight is stored TRANSPOSED, [in_features, out_features], so the forward is one plain
    torch.matmul(x2d, weight) with the contraction dim leading in memory: the [K,N] operand layout that runs the
    deep-K / narrow-N GEMMs (MLP c_proj K=5120, lm_head backward) at the isolated matmul rate instead of the
    in-situ F.linear rate (plan-v10 L2; numbers in work/agents/gemm_layout/NOTES.md). Same parameter count, same
    state_dict key ("weight"); the saved shape is [in, out] and GPTConfig.wkn makes load_for_eval rebuild this
    class. Muon is orientation-symmetric (the Newton-Schulz branch, the lr ratio and NorMuon's red_dim all follow
    p.shape), so the update equals the transpose of the [out, in] update up to rounding; the bf16-master cast in
    main() selects on isinstance(m, Linear), which this class passes. torch autograd only (no custom Function):
    dX = dy @ W.t() and dW = x2d.t() @ dy are the two backward matmuls. Two switches replace that autograd backward
    with a custom Function issuing the same two matmuls: R1_NKI_GEMM (gemm_class "fc", the MLP c_fc) and
    HEAD_DW_NSPLIT > 1 (gemm_class "head", the lm_head: its dW as column blocks); with both unset the two lines at the end of forward and
    autograd's own mm backward are the whole story."""

    def __init__(self, in_features: int, out_features: int, bias: bool=False) -> None:
        super().__init__(in_features, out_features, bias=False)
        self.weight = nn.Parameter(torch.empty(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inplace = _accum_inplace(self, x, 't')
        if USE_NKI_GEMM and getattr(self, 'gemm_class', None) == 'fc':
            return _LinearGemm.apply(x, self.weight if inplace else _weight_for(self, x), True, inplace)
        if getattr(self, 'gemm_class', None) == 'head' and HEAD_DW_NSPLIT > 1 and self.training and torch.is_grad_enabled() and self.weight.requires_grad:
            return _LmHeadDwSplit.apply(x, _weight_for(self, x))
        if inplace:
            return _LinearAccum.apply(x, self.weight, True)
        w = _weight_for(self, x)
        return torch.matmul(x.reshape(-1, x.shape[-1]), w).view(*x.shape[:-1], w.shape[1])

def _accum_inplace(m: nn.Module, x: torch.Tensor, tag: str) -> bool:
    """[R1 glue c] True when this Linear's weight gradient can be accumulated in place: the switch names its class,
    a gradient tensor already exists in the weight's dtype and shape (the persistent grad rows of
    is in the compute dtype (bf16 masters) and autograd is recording.  Eval, the fp32-master lm_head and every
    switch-off case take the plain path below, unchanged."""
    if not _ACCUM_TAGS or not ('all' in _ACCUM_TAGS or tag in _ACCUM_TAGS):
        return False
    w = m.weight
    g = w.grad
    return g is not None and torch.is_grad_enabled() and w.requires_grad and (w.dtype == x.dtype) and (g.dtype == w.dtype) and (g.shape == w.shape) and g.is_contiguous()

def _weight_for(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The weight in the compute dtype: the parameter itself when it already is (bf16 masters), today's .to() cast
    otherwise -- or, with R1_CAST_CACHE, one cast per weight version shared by the micro-steps of a step (f2)."""
    w = m.weight
    if w.dtype == x.dtype:
        return w
    if CAST_CACHE != '0' and m.training and torch.is_grad_enabled() and w.requires_grad:
        return _MasterCast.apply(w, m, x.dtype)
    return w.to(dtype=x.dtype)

class _LinearAccum(torch.autograd.Function):
    """[R1 glue c] y = x @ W^T (weight [out, in]) or y = x2d @ W (transposed=True, the LinearT [in, out] layout) with
    exactly the forward op Linear / LinearT issue today; the backward computes dX with one matmul and accumulates dW
    straight into the parameter's existing .grad with one addmm_ (grad += mat1 @ mat2) instead of producing a fresh
    dW that autograd's AccumulateGrad then adds with a second launch.  Same gradient sum; the accumulate happens in the
    matmul's fp32 output stage, so the sum is rounded to the grad dtype once per micro-step instead of twice.  The
    weight gets None back so autograd leaves .grad alone.  Used only when _accum_inplace says the grad exists."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, transposed: bool) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.w = w
        ctx.transposed = transposed
        if transposed:
            return torch.matmul(x.reshape(-1, x.shape[-1]), w).view(*x.shape[:-1], w.shape[1])
        return F.linear(x, w, None)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, = ctx.saved_tensors
        w = ctx.w
        x2d = x.reshape(-1, x.shape[-1])
        dy2d = dy.reshape(-1, dy.shape[-1])
        if ctx.transposed:
            dx = torch.matmul(dy2d, w.t())
            w.grad.addmm_(x2d.t(), dy2d)
        else:
            dx = torch.matmul(dy2d, w)
            w.grad.addmm_(dy2d.t(), x2d)
        return (dx.view(*x.shape[:-1], dx.shape[-1]), None, None)

class _LinearGemm(torch.autograd.Function):
    """[R1 gemm] y = x2d @ W (transposed=True: the LinearT [in, out] c_fc) or x @ W^T (the [out, in] c_proj) with the
    two K=5120 GEMMs of the MLP on the NKI primitive -- c_proj's forward and c_fc's dX, both bitwise the compiler's --
    and everything else exactly today's op: c_fc's forward is torch.matmul (the class-A shape runs at parity), c_proj's
    dX is torch.matmul, dW is one addmm_ into the existing .grad when the in-place accumulate is active (inplace=True,
    W is the Parameter, None returned for it as _LinearAccum does) and otherwise the returned dW (c_fc: the class-C
    primitive, 1 ulp on ~0.02 % of the elements vs torch; c_proj: torch).  nki_matmul itself returns torch.matmul
    whenever the kernel does not apply, so the switch is exact off-chip."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, transposed: bool, inplace: bool) -> torch.Tensor:
        x2d = x.reshape(-1, x.shape[-1])
        ctx.save_for_backward(x2d, w)
        ctx.w = w
        ctx.transposed = transposed
        ctx.inplace = inplace
        ctx.lead = x.shape[:-1]
        if transposed:
            y = torch.matmul(x2d, w)
        else:
            y = nki_matmul(x2d, w, trans_b=True)
        return y.view(*x.shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x2d, w = ctx.saved_tensors
        dy2d = dy.reshape(-1, dy.shape[-1])
        if not dy2d.is_contiguous():
            dy2d = dy2d.contiguous()
        if ctx.transposed:
            dx = nki_matmul(dy2d, w, trans_b=True)
            if ctx.inplace:
                ctx.w.grad.addmm_(x2d.t(), dy2d)
                dw = None
            else:
                dw = nki_matmul(x2d, dy2d, trans_a=True)
        else:
            dx = torch.matmul(dy2d, w)
            if ctx.inplace:
                ctx.w.grad.addmm_(dy2d.t(), x2d)
                dw = None
            else:
                dw = torch.matmul(dy2d.t(), x2d)
        return (dx.view(*ctx.lead, dx.shape[-1]), dw, None, None)

class _MasterCast(torch.autograd.Function):
    """[R1 glue f2/f3] The fp32-master weight's compute-dtype copy, cast once per weight version (p._version moves
    only when something writes the parameter in place: the optimizer step, a checkpoint load) and reused by every
    forward until then -- today's `weight.to(dtype)` recasts on every micro-step.  The backward is the cast's backward
    (compute-dtype dW -> the master dtype); when a gradient already exists it is accumulated in place with one
    ("cast"), and None is returned so autograd does not add a second time."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, m: nn.Module, dtype: torch.dtype) -> torch.Tensor:
        cache = getattr(m, '_cast_cache', None)
        if cache is None or m._cast_ver != w._version or cache.dtype != dtype:
            cache = w.detach().to(dtype=dtype)
            m._cast_cache, m._cast_ver = (cache, w._version)
        ctx.w = w
        return cache

    @staticmethod
    def backward(ctx, dw: torch.Tensor):
        w = ctx.w
        g = w.grad
        if g is not None and g.shape == dw.shape and (g.dtype == w.dtype) and g.is_contiguous():
            g.add_(dw)
            return (None, None, None)
        return (dw.to(dtype=w.dtype), None, None)

class _LinearSplitBwd(torch.autograd.Function):
    """[R1 L6] y = x @ W.t() with the backward issued as two plain matmuls, dX = dy @ W and dW = dy.t() @ x2d, instead of
    the fused `linear_backward` NEFF (66-71 TFLOP/s at the lm_head shape vs 105-125 for the two mm launches; the pair
    measured 4.98 ms vs 7.78 ms, work/agents/gemm_layout/NOTES.md section 5). The weight is passed already cast to the
    compute dtype, so the fp32-master cast and its backward stay in torch autograd as before."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x2d = x.reshape(-1, x.shape[-1])
        ctx.save_for_backward(x2d, w)
        ctx.lead = x.shape[:-1]
        return F.linear(x2d, w).view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x2d, w = ctx.saved_tensors
        dy2d = dy.reshape(-1, dy.shape[-1])
        dx = torch.matmul(dy2d, w).view(*ctx.lead, w.shape[1])
        dw = torch.matmul(dy2d.t(), x2d)
        return (dx, dw)

class LinearSplitBwd(Linear):
    """[R1 L6] Linear (weight [out, in], same checkpoint layout) whose backward is the explicit dX / dW pair. Not a model
    shape change, so no GPTConfig field: the scorer's load_for_eval builds the plain Linear with the same weights."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _LinearSplitBwd.apply(x, self.weight.to(dtype=x.dtype))
_HEAD_DW_FIRED = 0
_HEAD_DW_SAID = False

def _headdw_line(model, pad: bool=False) -> str:
    """[R1 F7 headdw] the config / liveness line, printed by BOTH halves of a pair so the diff of the two
    startup blocks shows the one token that moved (the 2026-09-23 env-inheritance confound is invisible in
    queue.log).  `head_class` is the REACHABILITY read, taken off the constructed model and not off the env:
    'head' means GPT.__init__ tagged the lm_head and the LinearT branch can fire; None means the brief-11
    silent-no-op trap is live and the arm is VOID.  `fired` is the liveness read: 0 on an arm with nsplit>1
    means the branch never ran (the A92_vload lesson), so the arm measured its own base.
    Every lookup is defensive -- a config line must never be able to end a run."""
    head = getattr(model, 'lm_head', None)
    w = getattr(head, 'weight', None)
    n_out = int(w.shape[-1]) if w is not None else 0
    return f'headdw:{('            ' if pad else '')} nsplit={HEAD_DW_NSPLIT} fired={_HEAD_DW_FIRED} head_class={getattr(head, 'gemm_class', None)!r} head_out={n_out} divides={HEAD_DW_NSPLIT <= 1 or (n_out > 0 and n_out % HEAD_DW_NSPLIT == 0)}'

def _mixfuse_line(model, pad: bool=False) -> str:
    """[R1 mixfuse] the config / liveness line, printed by BOTH halves of a pair (the base prints fused=0
    module=not-imported), so the diff of the two startup blocks shows the one token that moved -- the 2026-09-23
    env-inheritance check.  The liveness fields come from the module: fired_fwd/fired_bwd count the fused launches
    actually issued (3 each per micro-step at k7m: blocks 1, 7, 9); fired_fwd=0 on the arm = the branch never ran =
    VOID, not a null (the A92_vload lesson).  declined counts calls sent back to the unfused branch, disabled names a
    kernel failure, selfcheck is the first-call on-device comparison against the unfused path."""
    blocks = list(getattr(model, 'mixer_layers', ()))
    mod = 'loaded' if nki_mixtail is not None else 'MISSING'
    return f'mixfuse:{('           ' if pad else '')} fused={int(MIXER_FUSED)} module={mod} mixer_blocks={blocks}' + (f' {nki_mixtail.status()}' if nki_mixtail is not None else '')

def _head_dw(x2d: torch.Tensor, dy2d: torch.Tensor) -> torch.Tensor:
    """[R1 F7 headdw] dW = x2d^T @ dy2d for the lm_head, as ONE torch.matmul (HEAD_DW_NSPLIT <= 1, today's op,
    byte-for-byte the line it replaces) or as HEAD_DW_NSPLIT column blocks concatenated back.

    The split is over the OUTPUT free axis only.  dw[:, j] depends on dy2d[:, j] alone, so block j of the
    split computes exactly the terms block j of the single call computes, over the identical 32,768-term
    contraction; nothing is dropped, added or reordered WITHIN a dot product.  The compiler may pick a
    different contraction blocking for the narrower N, so the result is equal up to fp reassociation,
    not guaranteed bitwise -- the same class as R1_NS_TENSOR_COEFS and R1_NKI_GRADNORM, and it lands in
    an fp32 master accumulate either way.

    N % k != 0 falls back to the single call rather than splitting unevenly (padded_vocab 8192 divides by
    2, 4, 8 and 16; a future vocab that does not simply does not engage, loudly, via the print below)."""
    global _HEAD_DW_FIRED, _HEAD_DW_SAID
    k = HEAD_DW_NSPLIT
    n_out = dy2d.shape[1]
    if k <= 1 or n_out % k:
        if k > 1 and (not _HEAD_DW_SAID):
            _HEAD_DW_SAID = True
            print(f'NOTE: R1_HEAD_DW_NSPLIT={k} does not divide the lm_head output {n_out} -- single GEMM, switch INERT')
        return torch.matmul(x2d.t(), dy2d)
    blk = n_out // k
    _HEAD_DW_FIRED += 1
    if not _HEAD_DW_SAID:
        _HEAD_DW_SAID = True
        acc_kib = x2d.shape[1] // 128 * ((blk + 1023) // 1024) * 512 * 4 // 1024
        print(f'headdw:            nsplit={k} FIRED  x2d={list(x2d.shape)} dy2d={list(dy2d.shape)} -> {k} blocks of [{x2d.shape[1]},{blk}]  acc {acc_kib} KiB/partition (was {x2d.shape[1] // 128 * ((n_out + 1023) // 1024) * 512 * 4 // 1024})')
    return torch.cat([torch.matmul(x2d.t(), dy2d[:, i * blk:(i + 1) * blk]) for i in range(k)], dim=1)

class _LmHeadDwSplit(torch.autograd.Function):
    """[R1 F7 headdw] y = x2d @ W for the lm_head LinearT (W [in, out] = [n_embd, padded_vocab]): LinearT.forward's
    own last two lines and autograd's mm backward, op for op (the same reshape, the same .view out, dX = dy @ W^T),
    with only the dW GEMM issued through _head_dw as HEAD_DW_NSPLIT column blocks.  dW is RETURNED (not accumulated
    in place) so _MasterCast's backward routes it to the fp32 master exactly as the plain tail does.  This is the
    runtime's fp8 lm_head Function with both of its fp8 rows (9, the forward, and 10, the dX) out of the subset."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x2d = x.reshape(-1, x.shape[-1])
        ctx.save_for_backward(x2d, w)
        ctx.lead = x.shape[:-1]
        y = torch.matmul(x2d, w)
        return y.view(*x.shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x2d, w = ctx.saved_tensors
        dy2d = dy.reshape(-1, dy.shape[-1])
        if not dy2d.is_contiguous():
            dy2d = dy2d.contiguous()
        dx = torch.matmul(dy2d, w.t())
        dw = _head_dw(x2d, dy2d)
        return (dx.view(*ctx.lead, dx.shape[-1]), dw)

class _FusedQKV(torch.autograd.Function):
    """[R1 v11 B] F.linear(x, W3) where W3 [3C, C] is the persistent Muon stack's contiguous view of (c_q, c_k, c_v);
    the backward is the explicit dX = dy @ W3, dW3 = dy^T x pair, its dW3 handed back as the three row blocks (the
    narrow views CatBackward returns today, accumulated by the same in-place adds). On the chip the pair is bitwise
    equal to autograd's fused linear_backward at this shape and 1.44x faster (work/agents/rope_attn/NOTES.md 6.2)."""

    @staticmethod
    def forward(ctx, x, wq, wk, wv, w3):
        ctx.save_for_backward(x, w3)
        return F.linear(x, w3, None)

    @staticmethod
    def backward(ctx, gy):
        x, w3 = ctx.saved_tensors
        n = w3.shape[0] // 3
        g2 = gy.reshape(-1, gy.shape[-1])
        gx = torch.matmul(g2, w3).view_as(x)
        gw3 = torch.matmul(g2.t(), x.reshape(-1, x.shape[-1]))
        return (gx, gw3[0:n], gw3[n:2 * n], gw3[2 * n:], None)

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim, self.width = attn_geometry(config)
        self.c_q = Linear(config.n_embd, self.width, bias=False)
        self.c_k = Linear(config.n_embd, self.width, bias=False)
        self.c_v = Linear(config.n_embd, self.width, bias=False)
        self.c_proj = (LinearT if self.width < config.n_embd else Linear)(self.width, config.n_embd, bias=False)
        self.attn_gate = config.attn_gate
        self.qk_scale = config.qk_scale if config.qk_scale else QK_SCALE
        if self.attn_gate:
            self.c_gate = Linear(config.n_embd, config.n_embd, bias=False)
            torch.nn.init.zeros_(self.c_gate.weight)

    def qkv_weight(self, dtype):
        """[R1 v11 B] the [3C, C] projection weight: the Muon stack's contiguous rows (set by setup_optimizer when
        R1_QKV_STACK_VIEW=1) through _FusedQKV, else today's torch.cat (eval, CPU, MUON_PERSISTENT_VIEWS < 1)."""
        v = getattr(self, 'qkv_view', None)
        if v is not None and v.dtype == dtype:
            return v
        return torch.cat((self.c_q.weight, self.c_k.weight, self.c_v.weight), dim=0).to(dtype=dtype)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ve: torch.Tensor | None=None, ve_lambda: torch.Tensor | None=None, attn_scale: torch.Tensor | None=None) -> torch.Tensor:
        b, t, c = x.shape
        c = self.width
        w = self.qkv_weight(x.dtype)
        if w is getattr(self, 'qkv_view', None):
            qkv = _FusedQKV.apply(x, self.c_q.weight, self.c_k.weight, self.c_v.weight, w)
        else:
            qkv = F.linear(x, w, None)
        if USE_NKI_ATTN_ROPE and USE_NKI_ATTN and (ve is None) and (attn_scale is None):
            y = nki_rope_attention(qkv, cos, sin, self.n_head, scale=self.qk_scale if self.qk_scale > 0.0 else None)
            if y is not None:
                _NARROW['fused'] += 1
                y = y.view(b, t, c)
                if self.attn_gate:
                    g = torch.sigmoid(self.c_gate(x))
                    y = y * (g if self.attn_gate == 1 else 2.0 * g)
                return self.c_proj(y)
        q, k, v = qkv.split(self.width, dim=-1)
        q = q.reshape(b, t, self.n_head, self.head_dim)
        k = k.reshape(b, t, self.n_head, self.head_dim)
        v = v.reshape(b, t, self.n_head, self.head_dim)
        if ve is not None:
            v = v + ve_lambda.to(v.dtype) * ve.view(b, t, self.n_head, self.head_dim)
        q = rope_norm(q, cos, sin)
        k = rope_norm(k, cos, sin)
        if attn_scale is not None:
            q = q * attn_scale.view(1, self.n_head, 1, 1).to(q.dtype)
        v = v.transpose(1, 2)
        if USE_NKI_ATTN and attn_scale is None:
            y = nki_attn_fn(q, k, v, scale=self.qk_scale if self.qk_scale > 0.0 else None).view(b, t, c)
        elif USE_NKILIB_ATTN:
            if self.qk_scale > 0.0:
                q = q * (self.qk_scale * math.sqrt(self.head_dim))
            y = nki_causal_attention(q, k, v, 'bhdt')
            y = y.view(b, c, t).transpose(1, 2).contiguous()
        elif self.qk_scale > 0.0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.qk_scale)
            y = y.transpose(1, 2).contiguous().view(b, t, c)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(b, t, c)
        if self.attn_gate:
            g = torch.sigmoid(self.c_gate(x))
            y = y * (g if self.attn_gate == 1 else 2.0 * g)
        return self.c_proj(y)

class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.c_fc = (LinearT if config.wkn else Linear)(config.n_embd, hidden, bias=False)
        self.c_proj = Linear(hidden, config.n_embd, bias=False)
        self.c_proj.gemm_class = 'proj'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (USE_NKI_GEMM_FP8_NN or USE_NKI_GEMM_FP8_DW) and USE_NKI_FC_RELU2 and nki_gemm_fp8_nn.fc_active(self.training) and isinstance(self.c_fc, LinearT):
            inplace = _accum_inplace(self.c_fc, x, 't')
            h = nki_gemm_fp8_nn.fused_fc_relu2_nn(x, self.c_fc.weight if inplace else _weight_for(self.c_fc, x), inplace)
            return self.c_proj(h)
        if USE_NKI_GEMM_FP8 and nki_gemm_fp8.active(self.training) and isinstance(self.c_fc, LinearT):
            inplace = _accum_inplace(self.c_fc, x, 't')
            h = nki_gemm_fp8.fused_fc_relu2_fp8(x, self.c_fc.weight if inplace else _weight_for(self.c_fc, x), inplace)
            return self.c_proj(h)
        if USE_NKI_FC_RELU2 and isinstance(self.c_fc, LinearT):
            inplace = _accum_inplace(self.c_fc, x, 't')
            h = fused_fc_relu2(x, self.c_fc.weight if inplace else _weight_for(self.c_fc, x), inplace)
            return self.c_proj(h)
        return self.c_proj(relu_squared(self.c_fc(x)))

class Block(nn.Module):

    def __init__(self, config: GPTConfig, layer_idx: int=0):
        super().__init__()
        self.has_attn = layer_idx in attn_block_ids(config)
        self.attn = CausalSelfAttention(config) if self.has_attn else None
        self.mlp = MLP(config)
        self.layer_idx = layer_idx
        self.resid_lambda = nn.Parameter(torch.ones(1))
        self.x0_lambda = nn.Parameter(torch.zeros(1))
        use_ve = self.has_attn and (config.value_emb == 'all' or (config.value_emb == 'alt' and layer_idx % 2 == 0))
        self.ve_lambda = nn.Parameter(torch.full((1,), 0.5)) if use_ve else None
        self.attn_scale = nn.Parameter(torch.ones(config.n_head)) if self.has_attn and config.attn_scale == 'learn' else None
        use_bigram = config.bigram_buckets > 0 and config.bigram_mode == 'layers'
        self.bigram_lambda = nn.Parameter(torch.full((1,), 0.1)) if use_bigram else None
        use_mixer = not self.has_attn and mixer_mode(config) == 'shift1'
        self.mix_lambda = nn.Parameter(torch.zeros(1)) if use_mixer else None

    def forward(self, x, x0: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ve: torch.Tensor | None=None, xb: torch.Tensor | None=None, skip=None, skip_lambda: torch.Tensor | None=None, emit: bool=False):
        if USE_NKI_MEGABLOCK_TAIL and xb is None:
            x_new, xn, x_in, x0 = head2(x, x0, skip, self.resid_lambda, self.x0_lambda, skip_lambda, emit, key=self.layer_idx)
            if self.attn is None:
                if self.mix_lambda is None:
                    xn2 = xn
                elif MIXER_FUSED and nki_mixtail is not None and (not (RESID_DROPOUT > 0.0 and self.training)) and nki_mixtail.applies(x_new):
                    x_new, xn2 = nki_mixtail.mix_tail(x_new, xn, self.mix_lambda.to(x_new.dtype))
                else:
                    mix_out = self.mix_lambda.to(x_new.dtype) * shift1(x_new)
                    x_new, xn2 = tail_norm2(x_new, mix_out)
            else:
                attn_out = self.attn(xn, cos, sin, ve if self.ve_lambda is not None else None, self.ve_lambda, self.attn_scale)
                x_new, xn2 = tail_norm2(x_new, attn_out)
            mlp_out = self.mlp(xn2)
            return ((x_new, mlp_out), x_in, x0)
        fold_skip = USE_NKI_MEGABLOCK and xb is None
        if skip is not None and (not fold_skip):
            x = x + skip_lambda.to(x.dtype) * skip
        if fold_skip:
            x, xn = head_skip(x, x0, skip, self.resid_lambda, self.x0_lambda, skip_lambda)
        elif USE_NKI_BLOCK_HEAD and xb is None:
            x, xn = block_head(x, x0, self.resid_lambda, self.x0_lambda)
        elif USE_NKI_GAIN_NORM and xb is None:
            x, xn = gain_residual_norm(x, x0, self.resid_lambda, self.x0_lambda)
        else:
            x = self.resid_lambda.to(x.dtype) * x + self.x0_lambda.to(x.dtype) * x0
            if xb is not None:
                x = x + self.bigram_lambda.to(x.dtype) * xb
            xn = norm(x)
        if self.attn is None:
            if self.mix_lambda is None:
                xn2 = xn
            else:
                mix_out = self.mix_lambda.to(x.dtype) * shift1(x)
                x, xn2 = tail_norm2(x, mix_out)
        else:
            attn_out = self.attn(xn, cos, sin, ve if self.ve_lambda is not None else None, self.ve_lambda, self.attn_scale)
            x, xn2 = tail_norm2(x, attn_out)
        mlp_out = self.mlp(xn2)
        x = x + mlp_out
        return x

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.attn_layers = attn_block_ids(config)
        self.attn_dims = attn_dims(config)
        self.narrow = self.attn_dims[1] != config.n_embd
        self.mixer_layers = tuple((i for i in range(config.n_layer) if i not in self.attn_layers)) if mixer_mode(config) == 'shift1' else ()
        padded_vocab_size = (config.vocab_size + 63) // 64 * 64
        self.transformer = nn.ModuleDict({'wte': nn.Embedding(padded_vocab_size, config.n_embd), 'h': nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)])})
        self.tied_head = bool(config.tie_embeddings)
        head_cls = LinearSplitBwd if self.tied_head else LinearT if config.wkn else Linear
        self.lm_head = head_cls(config.n_embd, padded_vocab_size, bias=False)
        if self.tied_head:
            wte_w = self.transformer.wte.weight
            assert self.lm_head.weight.shape == wte_w.shape, f'tie_embeddings: head {tuple(self.lm_head.weight.shape)} != wte {tuple(wte_w.shape)}'
            self.lm_head.weight = wte_w
            _QL_FIRED['tie_share'] = _QL_FIRED.get('tie_share', 0) + 1
        self.lm_head.gemm_class = 'head'
        self.value_emb_on = config.value_emb != 'off'
        if self.value_emb_on:
            self.transformer['value_emb'] = nn.Embedding(padded_vocab_size, config.n_embd)
        self.n_skip = config.n_layer // 2 if config.unet_skip else 0
        self.skip_lambda = nn.Parameter(torch.ones(self.n_skip)) if self.n_skip > 0 else None
        self.bigram_on = config.bigram_buckets > 0
        if self.bigram_on:
            assert config.bigram_buckets & config.bigram_buckets - 1 == 0, 'bigram_buckets must be a power of two'
            self.transformer['bigram'] = nn.Embedding(config.bigram_buckets + 1, config.n_embd)
        self.bigram_lambda = nn.Parameter(torch.ones(1)) if self.bigram_on and config.bigram_mode == 'input' else None
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, self.attn_dims[0])
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        if self.tied_head:
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=TIE_INIT_STD)
            _QL_FIRED['tie_init'] = _QL_FIRED.get('tie_init', 0) + 1
        else:
            torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
            if isinstance(self.lm_head, LinearT):
                w = self.lm_head.weight
                w.copy_(torch.empty(w.shape[1], w.shape[0], dtype=w.dtype, device=w.device).normal_(mean=0.0, std=0.001).t())
            else:
                torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        if self.value_emb_on:
            torch.nn.init.normal_(self.transformer.value_emb.weight, mean=0.0, std=0.8)
        if self.bigram_on:
            torch.nn.init.normal_(self.transformer.bigram.weight, mean=0.0, std=0.8)
        s = math.sqrt(3.0) * self.config.n_embd ** (-0.5)
        n_layer = self.config.n_layer
        for i, block in enumerate(self.transformer.h):
            if block.attn is not None:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            if isinstance(block.mlp.c_fc, LinearT):
                w = block.mlp.c_fc.weight
                w.copy_(torch.empty(w.shape[1], w.shape[0], dtype=w.dtype, device=w.device).uniform_(-0.4 * s, 0.4 * s).t())
            else:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * s, 0.4 * s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            block.resid_lambda.fill_(INIT_RESID)
            block.x0_lambda.fill_(INIT_X0)
            if block.ve_lambda is not None:
                block.ve_lambda.fill_(0.5)
            if block.attn_scale is not None:
                block.attn_scale.fill_(1.0)
            if block.bigram_lambda is not None:
                block.bigram_lambda.fill_(0.1)
            if block.mix_lambda is not None:
                block.mix_lambda.fill_(0.0)
        if self.skip_lambda is not None:
            self.skip_lambda.fill_(INIT_SKIP)
        if self.bigram_lambda is not None:
            self.bigram_lambda.fill_(1.0)
        cos, sin = self._precompute_rotary_embeddings(self.config.sequence_len, self.attn_dims[0])
        self.cos.copy_(cos)
        self.sin.copy_(sin)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int | None=None):
        if base is None:
            base = self.config.rope_base
        self._rope_base_used = base
        device = self.transformer.wte.weight.device
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / base ** (freqs / head_dim)
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        table = torch.outer(t, inv_freq)
        cos = table.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = table.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        return (cos, sin)

    def estimate_flops(self, seq_len: int) -> int:
        """Estimate FLOPs per token (forward + backward) at a given training window."""
        nparams = sum((p.numel() for p in self.parameters()))
        non_matmul = (0 if self.tied_head else self.transformer.wte.weight.numel()) + 2 * self.config.n_layer
        if self.value_emb_on:
            non_matmul += self.transformer.value_emb.weight.numel()
        if self.bigram_on:
            non_matmul += self.transformer.bigram.weight.numel()
        non_matmul += sum((p.numel() for p in self._arch_gain_params()))
        h = self.config.n_head
        d = self.attn_dims[0]
        attn_flops = len(self.attn_layers) * 12 * h * d * (seq_len // 2)
        return 6 * (nparams - non_matmul) + attn_flops

    def _arch_gain_params(self) -> list[nn.Parameter]:
        """[R1 arch] The 1-D gains the switches add (ve_lambda, attn_scale, skip_lambda, A12's mix_lambda, and
        P01's bigram_lambda per block or on the model): scalar AdamW group. Empty when every switch is off."""
        gains: list[nn.Parameter] = []
        for block in self.transformer.h:
            if block.ve_lambda is not None:
                gains.append(block.ve_lambda)
            if block.attn_scale is not None:
                gains.append(block.attn_scale)
            if block.bigram_lambda is not None:
                gains.append(block.bigram_lambda)
            if block.mix_lambda is not None:
                gains.append(block.mix_lambda)
        if self.skip_lambda is not None:
            gains.append(self.skip_lambda)
        if self.bigram_lambda is not None:
            gains.append(self.bigram_lambda)
        return gains

    def setup_optimizer(self):
        scalar_params = []
        x0_params = []
        matrix_params = []
        for block in self.transformer.h:
            scalar_params.extend([block.resid_lambda, block.x0_lambda])
            if block.attn is not None:
                matrix_params.extend(block.attn.parameters())
            matrix_params.extend(block.mlp.parameters())
        scalar_params.extend(self._arch_gain_params())
        embedding_params = list(self.transformer.wte.parameters())
        if self.value_emb_on:
            embedding_params.extend(self.transformer.value_emb.parameters())
        if self.bigram_on:
            embedding_params.extend(self.transformer.bigram.parameters())
        lm_head_params = list(self.lm_head.parameters())
        tied_params: list = []
        if self.tied_head:
            shared = self.lm_head.weight
            embedding_params = [p for p in embedding_params if p is not shared]
            lm_head_params = [p for p in lm_head_params if p is not shared]
            lm_head_params = [shared] + lm_head_params
            n_groups = sum((1 for lst in (embedding_params, lm_head_params, tied_params, scalar_params, x0_params, matrix_params) for p in lst if p is shared))
            assert n_groups == 1, f'tie_embeddings: the shared tensor is in {n_groups} optimizer groups, must be exactly 1'
            _QL_FIRED['tie_group'] = _QL_FIRED.get('tie_group', 0) + 1
        dmodel_lr_scale = (self.config.n_embd / 768) ** (-0.5)
        emb_betas, head_betas = ((0.8, 0.995), (0.8, 0.96))
        param_groups = [dict(kind='adamw', params=embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale, betas=emb_betas, eps=1e-10, weight_decay=ADAMW_WD_EMB), dict(kind='adamw', params=lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=head_betas, eps=1e-10, weight_decay=ADAMW_WD_HEAD), dict(kind='adamw', params=scalar_params, lr=SCALAR_LR, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0)]
        if x0_params:
            param_groups.append(dict(kind='adamw', params=x0_params, lr=SCALAR_LR, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0))
        if tied_params:
            param_groups.append(dict(kind='adamw', params=tied_params, lr=TIE_LR * dmodel_lr_scale, betas=head_betas, eps=1e-10, weight_decay=0.01))
        param_groups = [g for g in param_groups if g['params']]
        for shape in sorted({p.shape for p in matrix_params}):
            params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=params, lr=MATRIX_LR, momentum=0.95, beta2=MUON_BETA2, ns_steps=MUON_NS_STEPS, weight_decay=WEIGHT_DECAY))
        opt = MuonAdamW(param_groups)
        device = self.transformer.wte.weight.device
        for group in opt.param_groups:
            group['initial_lr'] = group['lr']
            group['lr_t'] = torch.full((1,), float(group['lr']), dtype=torch.float32, device=device)
            group['lr_cpu'] = torch.full((1,), float(group['lr']), dtype=torch.float32)
            if group['kind'] == 'muon':
                group['momentum_t'] = torch.full((1,), float(group['momentum']), dtype=torch.float32, device=device)
                group['momentum_cpu'] = torch.full((1,), float(group['momentum']), dtype=torch.float32)
            else:
                group['bias1_t'] = torch.ones(1, dtype=torch.float32, device=device)
                group['bias2_t'] = torch.ones(1, dtype=torch.float32, device=device)
                group['bias1_cpu'] = torch.ones(1, dtype=torch.float32)
                group['bias2_cpu'] = torch.ones(1, dtype=torch.float32)
        if OPT_TAIL_STAGE:
            opt.build_scalar_stage(device)
        for group in opt.param_groups:
            if group['kind'] == 'adamw':
                opt.stack_scalars(group)
        for group in opt.param_groups:
            if group['kind'] == 'muon':
                opt.persist_muon_views(group)
        for group in opt.param_groups:
            if group['kind'] != 'muon':
                continue
            params = group['params']
            stack = opt.state[params[0]].get('stacked_params')
            if stack is None:
                continue
            for block in self.transformer.h:
                a = block.attn
                if a is None:
                    continue
                try:
                    i = next((j for j, p in enumerate(params) if p is a.c_q.weight))
                except StopIteration:
                    continue
                same_order = params[i + 1] is a.c_k.weight and params[i + 2] is a.c_v.weight and (stack[i].data_ptr() == a.c_q.weight.data_ptr()) and (stack[i + 2].data_ptr() == a.c_v.weight.data_ptr())
                if same_order:
                    a.qkv_view = stack[i:i + 3].view(3 * stack.shape[1], stack.shape[2])
        return opt

    def _wte_shadow(self, dtype: torch.dtype):
        """[R1 glue f1] The token table in the compute dtype, re-cast only when the parameter's version changes (an
        optimizer write, a checkpoint load); None when the switch is off, in eval, or when no cast is needed."""
        w = self.transformer.wte.weight
        if not EMB_SHADOW or w.dtype == dtype or (not self.training):
            return None
        sh = getattr(self, '_shadow_cache', None)
        if sh is None or self._shadow_ver != w._version or sh.dtype != dtype:
            with torch.no_grad():
                sh = w.detach().to(dtype=dtype)
            self._shadow_cache, self._shadow_ver = (sh, w._version)
        return sh

    @torch.no_grad()
    def flush_gain_grads(self) -> None:
        """[R1 glue d] Finish this micro-step's gain gradients of the megablock heads in one batched op.  With
        skip_lambda[j]]) instead of issuing one .sum per block plus autograd's per-parameter slice/add launches; here
        they are concatenated (one launch), summed per row (one launch) and put into the parameters' gradients: on the
        first micro-step of a step the (1,) gains' .grad become views of the summed vector (no launch) and
        skip_lambda.grad one clone; on later micro-steps one add_ each.  Same fp32 partials, one reduction per row
        instead of one per block: fp32-reassociation class.  No-op when nothing was stashed (kernel path not taken);
        a partial stash (kernel disabled mid-backward) takes the generic per-parameter path, correct but not batched."""
        if gain_partials_take is None:
            return
        parts = gain_partials_take()
        if not parts:
            return
        n_layer, n_skip = (self.config.n_layer, self.n_skip)
        ones, rows, skip_rows, skip_j = ([], [], [], [])
        for i, block in enumerate(self.transformer.h):
            r3 = parts.pop(i, None)
            if r3 is None:
                continue
            rows.append(r3[0].reshape(1, -1))
            ones.append(block.resid_lambda)
            rows.append(r3[1].reshape(1, -1))
            ones.append(block.x0_lambda)
            if r3.shape[0] > 2:
                skip_rows.append(r3[2].reshape(1, -1))
                skip_j.append(i - (n_layer - n_skip))
        if parts:
            raise RuntimeError(f'gain partials stashed for unknown blocks {sorted(parts)}')
        sums = torch.cat(rows + skip_rows).sum(dim=1)
        n1 = len(ones)
        skip = self.skip_lambda
        if n1 == 2 * n_layer and len(skip_rows) == n_skip:
            views = getattr(self, '_gain_views', None)
            same = views is not None and all((p.grad is not None and p.grad.data_ptr() == v.data_ptr() for p, v in zip(ones, views)))
            if same and (n_skip == 0 or (skip.grad is not None and skip.grad.data_ptr() == self._skip_grad.data_ptr())):
                self._gain_buf.add_(sums)
                if n_skip:
                    self._skip_grad.add_(sums[n1:])
                return
            if all((p.grad is None for p in ones)) and (n_skip == 0 or skip.grad is None):
                self._gain_buf = sums
                self._gain_views = [sums[k:k + 1] for k in range(n1)]
                for p, v in zip(ones, self._gain_views):
                    p.grad = v
                if n_skip:
                    self._skip_grad = sums[n1:].clone()
                    skip.grad = self._skip_grad
                return
        for k, p in enumerate(ones):
            g = sums[k:k + 1]
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.add_(g)
        if skip_rows:
            if skip.grad is None:
                skip.grad = torch.zeros_like(skip)
            for j, k in zip(skip_j, range(n1, n1 + len(skip_rows))):
                skip.grad[j:j + 1].add_(sums[k:k + 1])

    def bigram_code(self, idx: torch.Tensor) -> torch.Tensor:
        """[R1 P01] Hash of (previous token, current token) into [0, buckets), int32 throughout (the loader yields
        int32); column 0 (no previous token) is the reserved row `buckets`. Elementwise ops on static shapes only:
        no gather/scatter/nonzero, no data-dependent shape, no per-step Python scalar. For a fixed current token
        prev -> code is a bijection modulo `buckets` (odd multiplier, xor with a constant), so power-of-two buckets
        keep the pairs sharing a current token in distinct rows. 8191 x 36313 < 2^29: no overflow (the reserved
        prev in column 0 could wrap for buckets >= 2^17, but that column is replaced below)."""
        b = idx.shape[0]
        buckets = self.config.bigram_buckets
        h = idx.to(torch.int32)
        prev = torch.cat((h.new_full((b, 1), buckets), h[:, :-1]), dim=1)
        code = torch.bitwise_and(torch.bitwise_xor(h * 36313, prev * 27191), buckets - 1)
        return torch.cat((code.new_full((b, 1), buckets), code[:, 1:]), dim=1)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None=None, loss_reduction: str='mean'):
        _b, t = idx.shape
        assert t <= self.config.sequence_len
        cos, sin = (self.cos[:, :t], self.sin[:, :t])
        shadow = self._wte_shadow(COMPUTE_DTYPE)
        x = embedding_fwd(idx, self.transformer.wte.weight, COMPUTE_DTYPE) if shadow is None else embedding_fwd(idx, self.transformer.wte.weight, COMPUTE_DTYPE, shadow)
        xb = self.transformer.bigram(self.bigram_code(idx)).to(COMPUTE_DTYPE) if self.bigram_on else None
        if self.bigram_lambda is not None:
            x = x + self.bigram_lambda.to(COMPUTE_DTYPE) * xb
            xb = None
        x = norm(x)
        x0 = x
        ve = self.transformer.value_emb(idx).to(COMPUTE_DTYPE) if self.value_emb_on else None
        n_layer, n_skip = (self.config.n_layer, self.n_skip)
        skips: list = []
        if USE_NKI_MEGABLOCK_TAIL and xb is None:
            cur = x
            for i, block in enumerate(self.transformer.h):
                skip, skip_lambda = (None, None)
                if n_skip and i >= n_layer - n_skip:
                    j = i - (n_layer - n_skip)
                    skip, skip_lambda = (skips.pop(), self.skip_lambda[j:j + 1])
                emit = bool(skips) and skips[-1] is SELF_SKIP
                cur, x_in, x0 = block(cur, x0, cos, sin, ve, xb, skip, skip_lambda, emit)
                if emit:
                    skips[-1] = x_in
                if n_skip and i < n_skip:
                    skips.append(SELF_SKIP)
            x = final_norm(cur[0], cur[1])
        else:
            for i, block in enumerate(self.transformer.h):
                skip, skip_lambda = (None, None)
                if n_skip and i >= n_layer - n_skip:
                    j = i - (n_layer - n_skip)
                    skip, skip_lambda = (skips.pop(), self.skip_lambda[j:j + 1])
                x = block(x, x0, cos, sin, ve, xb, skip, skip_lambda)
                if n_skip and i < n_skip:
                    skips.append(x)
            x = norm(x)
        logits_raw = self.lm_head(x)
        if USE_NKI_LOSS_HEAD and targets is not None and (CE_PATH == 'static') and (logits_raw.shape[-1] == self.config.vocab_size):
            if MTP_SHIP and loss_reduction == 'mean' and any(_MTP['on']) and torch.is_grad_enabled() and multi_available(logits_raw.device):
                loss = mtp_ship(logits_raw, targets)
                if loss is not None:
                    return loss
            if BOS_EXCL == 'post' and _BOSX['on'] and (loss_reduction == 'mean') and torch.is_grad_enabled():
                loss = bosx_loss(logits_raw, targets)
                if loss is not None:
                    return loss
            loss = fused_loss(logits_raw, targets, reduction=loss_reduction, softcap=LOGIT_SOFTCAP)
            return loss
        logits = logits_raw[..., :self.config.vocab_size].float()
        logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
        if targets is None:
            return logits
        vocab = logits.size(-1)
        logp = F.log_softmax(logits.reshape(-1, vocab), dim=-1)
        onehot = (targets.reshape(-1, 1) == torch.arange(vocab, device=logits.device).view(1, -1)).to(logp.dtype)
        nll = -(logp * onehot).sum(dim=-1)
        if loss_reduction == 'none':
            return nll.view_as(targets)
        return nll.sum() if loss_reduction == 'sum' else nll.mean()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction=loss_reduction)
        if loss_reduction == 'none':
            return loss.view_as(targets)
        return loss

def build_config(depth: int=DEPTH, n_embd: int=0) -> GPTConfig:
    if not n_embd:
        base_dim = depth * ASPECT_RATIO
        n_embd = (base_dim + HEAD_DIM - 1) // HEAD_DIM * HEAD_DIM
    assert n_embd % HEAD_DIM == 0, f'n_embd={n_embd} must be a multiple of HEAD_DIM={HEAD_DIM}'
    n_head = N_HEAD
    return GPTConfig(sequence_len=ROPE_TABLE_LEN, vocab_size=TOKENIZER_VOCAB_SIZE, n_layer=depth, n_head=n_head, n_embd=n_embd, mlp_ratio=MLP_RATIO, value_emb=VALUE_EMB, unet_skip=UNET_SKIP, attn_gate=ATTN_GATE, attn_blocks=ATTN_BLOCKS, mixer=MIXER, qk_scale=QK_SCALE, rope_base=ROPE_BASE, tie_embeddings=TIE_EMBEDDINGS, attn_scale=ATTN_SCALE, bigram_buckets=BIGRAM_BUCKETS, bigram_mode=BIGRAM_MODE, wkn=WKN, head_dim=HEAD_DIM if n_head * HEAD_DIM != n_embd else 0)
POLAR_EXPRESS_COEFFS = [(8.156554524902461, -22.48329292557795, 15.878769915207462), (4.042929935166739, -2.808917465908714, 0.5000178451051316), (3.8916678022926607, -2.772484153217685, 0.5060648178503393), (3.285753657755655, -2.3681294933425376, 0.46449024233003106), (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)]

class _ScalarStage:
    """[R1 opttail e] The optimizer's per-step (1,) scalars as views of ONE host tensor and ONE device tensor per dtype.

    slots: (owner dict, device key, host key, dtype, initial value).  The owner's two keys are re-pointed at row views
    of the pair, the initial values are written and sent, and from then on a step's values travel with one copy per
    dtype.  ensure_sent() copies only when the host rows differ from what was last sent; the device rows equal the
    last-sent host rows by construction (nothing but these copies writes them -- every consumer only reads a scalar).
    them to device ops in every live run); still, a one-time check at construction reads every device view back through
    the two op forms the optimizer uses them in (a (1,) operand of mul, the weight of lerp_), and on any mismatch -- or
    per scalar (the base file's layout, same values)."""

    def __init__(self, device: torch.device, slots: list):
        self.device = device
        self.slots = slots
        self.host, self.dev, self.sent = ({}, {}, {})
        self.views = True
        self.copies = 0
        for dt in dict.fromkeys((s[3] for s in slots)):
            rows = [s for s in slots if s[3] == dt]
            self.host[dt] = torch.zeros(len(rows), dtype=dt)
            self.dev[dt] = torch.zeros(len(rows), dtype=dt, device=device)
            for i, (owner, kd, kh, _, _) in enumerate(rows):
                owner[kh] = self.host[dt][i:i + 1]
                owner[kd] = self.dev[dt][i:i + 1]
        self._check()
        for owner, _, kh, _, init in slots:
            owner[kh].fill_(init)
        self.send()

    @torch.no_grad()
    def _check(self) -> None:
        err = 'R1_FUSE_DENY=stage_views' if 'stage_views' in FUSE_DENY else None
        if err is None:
            try:
                for dt, host in self.host.items():
                    n = host.numel()
                    host.copy_(torch.arange(1, n + 1, dtype=torch.float32).mul_(0.125).to(dt))
                    self.dev[dt].copy_(host)
                    ones = torch.ones(2, dtype=dt, device=self.device)
                    outs = []
                    for i in range(n):
                        v = self.dev[dt][i:i + 1]
                        outs.append(ones * v)
                        outs.append(torch.zeros(2, dtype=dt, device=self.device).lerp_(ones, v))
                    got = torch.cat(outs).float().cpu()
                    want = host.float().repeat_interleave(4)
                    if not torch.equal(got, want):
                        err = f'{dt} views read back {got.tolist()} != {want.tolist()}'
                        break
            except Exception as e:
                err = f'{type(e).__name__}: {e}'
        if err is not None:
            print0(f'opttail: stage views FAILED their check ({err}); standalone (1,) tensors, one copy per scalar (the base form)')
            self.views = False
            for owner, kd, _, dt, _ in self.slots:
                owner[kd] = torch.zeros(1, dtype=dt, device=self.device)

    @torch.no_grad()
    def send(self) -> None:
        if self.views:
            for dt, host in self.host.items():
                self.dev[dt].copy_(host)
                self.copies += 1
        else:
            for owner, kd, kh, _, _ in self.slots:
                owner[kd].copy_(owner[kh])
                self.copies += 1
        self.sent = {dt: h.clone() for dt, h in self.host.items()}
        _OPTTAIL_FIRED['stage'] += 1

    def ensure_sent(self) -> None:
        if any((not torch.equal(h, self.sent[dt]) for dt, h in self.host.items())):
            self.send()

    def desc(self) -> str:
        rows = ','.join((f'{str(dt).replace('torch.', '')}:{h.numel()}' for dt, h in self.host.items()))
        return f'views={self.views} rows={rows} copies_per_send={(len(self.host) if self.views else len(self.slots))}'

class MuonAdamW(torch.optim.Optimizer):

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self.use_reduce_scatter = MUON_SHARD_RS != '0'
        self.use_all_gather_into_tensor = MUON_SHARD_AG != '0'
        self._op_ok: dict = {}
        self._ns_coefs: dict = {}
        self._grad_views = False
        self._stage = None

    def op_supported(self, name: str, device: torch.device) -> bool:
        """[R1 optim2] Whether a merged op (addcmul, baddbmm, maximum, foreach_norm) runs on this device with the expected
        numbers. Probed ONCE per (op, device type) on tiny tensors inside try/except -- at construction (foreach_norm, from
        GradReducer) or during the first optimizer step (the excluded startup steps), never inside the clock -- so a backend
        that lacks the kernel, or gets the broadcast/dtype case wrong, falls back to the eager form of that one op only.
        R1_FUSE_DENY=name,... forces the fallback (tests exercise both forms)."""
        key = (name, device.type)
        if key not in self._op_ok:
            ok = name not in FUSE_DENY
            if ok:
                try:
                    ok = self._probe_op(name, device)
                    if not ok:
                        print0(f'optim2: {name} gives wrong values on {device.type}; using the eager form')
                except Exception as err:
                    print0(f'optim2: {name} unavailable on {device.type} ({type(err).__name__}: {err}); using the eager form')
                    ok = False
            self._op_ok[key] = ok
        return self._op_ok[key]

    @staticmethod
    @torch.no_grad()
    def _probe_op(name: str, device: torch.device) -> bool:
        a = (torch.arange(1.0, 9.0, device=device).view(2, 2, 2) / 8.0).contiguous()
        k = torch.full((1,), 0.25, dtype=torch.float32, device=device)
        if name == 'addcmul':
            got = torch.cat((torch.addcmul(a, a, k).flatten(), torch.addcmul(a, a, a >= k).flatten()))
            want = torch.cat(((a + a * k).flatten(), (a + a * (a >= k)).flatten()))
        elif name == 'baddbmm':
            got, want = (torch.baddbmm(a, a, a), a + a @ a)
        elif name == 'maximum':
            got, want = (torch.maximum(a, k), a.clamp_min(0.25))
        elif name == 'foreach_norm':
            got = torch.stack(torch._foreach_norm([a, a.to(torch.bfloat16), k], dtype=torch.float32))
            want = torch.stack([a.norm(), a.to(torch.bfloat16).float().norm(), k.norm()])
        elif name == 'vnorm_dtype':
            ab = a.to(torch.bfloat16)
            got = torch.linalg.vector_norm(ab, 2, dim=-2, keepdim=True, dtype=torch.float32).square()
            want = ab.float().square().sum(dim=-2, keepdim=True)
        elif name == 'ge_lowp':
            ab = (a - 0.5).to(torch.bfloat16)
            got = (ab >= torch.zeros(1, dtype=torch.bfloat16, device=device)).float()
            want = (ab.float() >= 0).float()
        elif name == 'afold':
            got = (a + (torch.eye(2) * 0.5).to(device)) @ a
            want = 0.5 * a + a @ a
        elif name == 'add_alpha':
            got, want = (torch.add(a, a, alpha=0.25), a + 0.25 * a)
        else:
            raise ValueError(f'unknown probe {name}')
        return bool(torch.allclose(got.float().cpu(), want.float().cpu(), rtol=0.01, atol=1e-06))

    def _ns_coefs_for(self, ns_steps: int, device: torch.device, dtype: torch.dtype) -> list:
        """[R1 optim2 S8b] The Polar Express coefficients as (1,) device tensors in the Newton-Schulz dtype, created once per
        (device, dtype). fp32 compute: bitwise the Python floats. bf16 compute: the coefficients are rounded to bf16 (a of
        iteration 1: 8.1566 -> 8.1875, +0.38%; b, c and the later iterations within 0.3%), a fixed perturbation of the
        polynomial well inside the tolerance of a bf16 iteration (NOTES.md, S8b)."""
        key = (device.type, dtype, ns_steps)
        coefs = self._ns_coefs.get(key)
        if coefs is None:
            coefs = [tuple((torch.full((1,), float(v), dtype=dtype, device=device) for v in abc)) for abc in POLAR_EXPRESS_COEFFS[:ns_steps]]
            self._ns_coefs[key] = coefs
        return coefs

    def _ns_fold_consts(self, ns_steps: int, n: int, device: torch.device, dtype: torch.dtype, want_eye: bool) -> tuple:
        """[R1 opttail a/rider] Per Newton-Schulz iteration: aI = a * I_n in the NS dtype (built on the host, copied once;
        want_eye) and c as a Python float, both from the (1,) coefficient tensors' OWN values (one tiny device->host read
        each, once per (device, dtype, ns_steps, n), inside the first -- excluded -- optimizer step), so the fold and the
        memory (10 MiB at n = 1024 in bf16), shared by every group with the same n."""
        key = ('fold', device.type, dtype, ns_steps, n if want_eye else 0)
        got = self._ns_coefs.get(key)
        if got is None:
            vals = [tuple((float(t.detach().cpu().float()[0]) for t in abc)) for abc in self._ns_coefs_for(ns_steps, device, dtype)]
            eye = torch.eye(n, dtype=torch.float32)
            a_eyes = [(eye * a).to(dtype).to(device) for a, _, _ in vals] if want_eye else None
            got = (a_eyes, [c for _, _, c in vals])
            self._ns_coefs[key] = got
        return got

    def _newton_schulz(self, group, x: torch.Tensor) -> torch.Tensor:
        """[R1 optim2 S8b] The baseline's polynomial x <- a x + x (b xtx + c xtx xtx) (tall; the wide mirror likewise) with
        (1,) tensor coefficients instead of Python floats, so no coefficient is a host->device scalar copy or a graph
        constant. R1_NS_TENSOR_COEFS=1 keeps the op structure (3 matmuls + 5 elementwise per iteration); =2 merges
        b xtx + c xx into one addcmul and a x + x inner into one baddbmm (3 matmuls + 3 elementwise), bitwise the same
        arithmetic in fp32 (c xx == xx c; baddbmm == bmm + add on CPU)."""
        tall = x.shape[-2] > x.shape[-1]
        coefs = self._ns_coefs_for(group['ns_steps'], x.device, x.dtype)
        merged = NS_TENSOR_COEFS == '2'
        use_addcmul = merged and self.op_supported('addcmul', x.device)
        use_baddbmm = merged and self.op_supported('baddbmm', x.device)
        return self._newton_schulz_fold(group, x, tall)
        for a_t, b_t, c_t in coefs:
            if tall:
                xtx = x.mT @ x
                xx = xtx @ xtx
                inner = torch.addcmul(b_t * xtx, xx, c_t) if use_addcmul else b_t * xtx + c_t * xx
                x = torch.baddbmm(a_t * x, x, inner) if use_baddbmm else a_t * x + x @ inner
            else:
                xxt = x @ x.mT
                xx = xxt @ xxt
                inner = torch.addcmul(b_t * xxt, xx, c_t) if use_addcmul else b_t * xxt + c_t * xx
                x = torch.baddbmm(a_t * x, inner, x) if use_baddbmm else a_t * x + inner @ x
        return x

    def _newton_schulz_fold(self, group, x: torch.Tensor, tall: bool) -> torch.Tensor:
        """[R1 opttail a/rider] R1_NS_TENSOR_COEFS=1's iteration with the Polar Express `a` folded into the operator
        final add and adds one [k, n, n] add (8 -> 7 launches); the rider deletes the `c * xx` mul (-> 6 with both).
        Each form is probed once and falls back loudly (op_supported prints) to mode 1's own expression, so a denied
        form is BITWISE mode 1.  The fold is not bitwise: bf16(a + inner_ii) is rounded once (NOTES.md sec 5.4)."""
        dev = x.device
        afold = MUON_NS_AFOLD and self.op_supported('afold', dev)
        alpha = MUON_NS_ALPHA and self.op_supported('add_alpha', dev)
        coefs = self._ns_coefs_for(group['ns_steps'], dev, x.dtype)
        n = x.shape[-1] if tall else x.shape[-2]
        a_eyes, cvals = self._ns_fold_consts(group['ns_steps'], n, dev, x.dtype, afold) if afold or alpha else (None, None)
        for i, (a_t, b_t, c_t) in enumerate(coefs):
            if tall:
                xtx = x.mT @ x
                xx = xtx @ xtx
                inner = torch.add(b_t * xtx, xx, alpha=cvals[i]) if alpha else b_t * xtx + c_t * xx
                x = x @ (inner + a_eyes[i]) if afold else a_t * x + x @ inner
            else:
                xxt = x @ x.mT
                xx = xxt @ xxt
                inner = torch.add(b_t * xxt, xx, alpha=cvals[i]) if alpha else b_t * xxt + c_t * xx
                x = (inner + a_eyes[i]) @ x if afold else a_t * x + inner @ x
        _OPTTAIL_FIRED['afold'] += len(coefs) if afold else 0
        _OPTTAIL_FIRED['alpha'] += len(coefs) if alpha else 0
        return x

    def _fused_consts(self, group, dtype: torch.dtype, device: torch.device) -> dict:
        """[R1 optim2 S8e] The fixed scalars of the Muon tail as (1,) device tensors, created once per group: 1.01 and 1e-6
        of the pre-NS normalisation in the NS dtype, 1e-10 / (1 - beta2) / 0 of the fp32 NorMuon statistics."""
        key = f'fused_consts_{dtype}'
        consts = group.get(key)
        if consts is None:
            consts = {'k101': torch.full((1,), 1.01, dtype=dtype, device=device), 'eps6': torch.full((1,), 1e-06, dtype=dtype, device=device), 'eps10': torch.full((1,), 1e-10, dtype=torch.float32, device=device), 'beta2_1m': torch.full((1,), 1 - group['beta2'], dtype=torch.float32, device=device), 'zero': torch.zeros(1, dtype=dtype if OPT_TAIL_MASK and dtype != torch.float32 and self.op_supported('ge_lowp', device) else torch.float32, device=device)}
            group[key] = consts
        return consts

    @torch.no_grad()
    def _refresh_fused_scalars(self, group, dtype: torch.dtype) -> None:
        """[R1 optim2 S8e] The per-step (1,) scalars of the fused Muon tail, derived on the host from the values the loop
        already staged (lr_cpu, momentum_cpu, weight_decay) in the same fp32 (or bf16-master) arithmetic the device used for
        `lr_t * ratio`, `lr * wd`, `1 - momentum_t` and `momentum_t.to(dtype)`, then copied to the device as data: four
        host->device copies per group replace three tiny device ops plus their three scalar copies, and nothing is a graph
        constant. `dtype` is the momentum buffer's (the parameter's) dtype."""
        if self._stage is not None:
            self._refresh_fused_host(group, dtype)
            self._stage.ensure_sent()
            return
        if 'lr_eff_t' not in group:
            dev = group['lr_t'].device
            shape = group['params'][0].shape
            group['ratio_cpu'] = torch.full((1,), max(1.0, shape[-2] / shape[-1]) ** 0.5, dtype=torch.float32)
            group['wd_cpu'] = torch.zeros(1, dtype=torch.float32)
            group['lr_eff_cpu'] = torch.zeros(1, dtype=torch.float32)
            group['lr_wd_cpu'] = torch.zeros(1, dtype=torch.float32)
            group['lr_eff_t'] = torch.zeros(1, dtype=torch.float32, device=dev)
            group['lr_wd_t'] = torch.zeros(1, dtype=torch.float32, device=dev)
            group['momentum1m_cpu'] = torch.zeros(1, dtype=dtype)
            group['momentum1m_t'] = torch.zeros(1, dtype=dtype, device=dev)
            if dtype == torch.float32:
                group['momentum_lo_t'] = group['momentum_t']
            else:
                group['momentum_lo_cpu'] = torch.zeros(1, dtype=dtype)
                group['momentum_lo_t'] = torch.zeros(1, dtype=dtype, device=dev)
        group['wd_cpu'].fill_(float(group['weight_decay']))
        torch.mul(group['lr_cpu'], group['ratio_cpu'], out=group['lr_eff_cpu'])
        torch.mul(group['lr_eff_cpu'], group['wd_cpu'], out=group['lr_wd_cpu'])
        group['lr_eff_t'].copy_(group['lr_eff_cpu'])
        group['lr_wd_t'].copy_(group['lr_wd_cpu'])
        m_lo = group['momentum_cpu'].to(dtype)
        group['momentum1m_cpu'].copy_(1.0 - m_lo)
        group['momentum1m_t'].copy_(group['momentum1m_cpu'])
        if dtype != torch.float32:
            group['momentum_lo_cpu'].copy_(m_lo)
            group['momentum_lo_t'].copy_(group['momentum_lo_cpu'])

    @torch.no_grad()
    def _refresh_fused_host(self, group, dtype: torch.dtype) -> None:
        """[R1 opttail e] _refresh_fused_scalars' HOST arithmetic, line for line (lr * ratio, lr * wd, 1 - momentum and
        momentum.to(dtype) in the same fp32 / bf16 order), written into the stage's host views; the device views get
        them with the stage's one copy per dtype."""
        assert group.get('_stage_dtype') == dtype, f'opttail stage: group staged for {group.get('_stage_dtype')}, refreshed as {dtype}'
        group['wd_cpu'].fill_(float(group['weight_decay']))
        torch.mul(group['lr_cpu'], group['ratio_cpu'], out=group['lr_eff_cpu'])
        torch.mul(group['lr_eff_cpu'], group['wd_cpu'], out=group['lr_wd_cpu'])
        m_lo = group['momentum_cpu'].to(dtype)
        group['momentum1m_cpu'].copy_(1.0 - m_lo)
        if dtype != torch.float32:
            group['momentum_lo_cpu'].copy_(m_lo)

    @torch.no_grad()
    def build_scalar_stage(self, device: torch.device) -> None:
        """[R1 opttail e] Called by setup_optimizer right after the (1,) lr / momentum tensors exist.  Re-points every
        group's lr_t / lr_cpu (and each Muon group's momentum_t / momentum_cpu) and pre-creates each Muon group's
        _refresh_fused_scalars tensors (the base file creates them lazily on the first step) as (1,) VIEWS of one host
        and one device tensor per dtype, so a step's scalars travel in one host->device copy per dtype instead of
        5 + 2 + 4 x (Muon groups).  Same values, dtypes and shapes: bitwise."""
        muon = [g for g in self.param_groups if g['kind'] == 'muon' and g['params']]
        slots = []
        for g in self.param_groups:
            slots.append((g, 'lr_t', 'lr_cpu', torch.float32, float(g['lr'])))
            if g['kind'] == 'muon':
                slots.append((g, 'momentum_t', 'momentum_cpu', torch.float32, float(g['momentum'])))
        for g in muon:
            p0 = g['params'][0]
            dt = p0.dtype
            g['_stage_dtype'] = dt
            g['ratio_cpu'] = torch.full((1,), max(1.0, p0.shape[-2] / p0.shape[-1]) ** 0.5, dtype=torch.float32)
            g['wd_cpu'] = torch.zeros(1, dtype=torch.float32)
            slots += [(g, 'lr_eff_t', 'lr_eff_cpu', torch.float32, 0.0), (g, 'lr_wd_t', 'lr_wd_cpu', torch.float32, 0.0), (g, 'momentum1m_t', 'momentum1m_cpu', dt, 0.0)]
            if dt != torch.float32:
                slots.append((g, 'momentum_lo_t', 'momentum_lo_cpu', dt, 0.0))
        self._stage = _ScalarStage(device, slots)
        for g in muon:
            if g['_stage_dtype'] == torch.float32:
                g['momentum_lo_t'] = g['momentum_t']

    @torch.no_grad()
    def stage_step_scalars(self) -> None:
        """[R1 opttail e] The loop's half: after its lr / momentum / weight-decay host fills, every Muon group's tail
        scalars on the host, then the stage's copy (one per dtype).  Inside the step _refresh_fused_scalars recomputes
        the same host values, finds them already sent, and issues nothing."""
        for g in self.param_groups:
            if g['kind'] == 'muon' and g['params']:
                self._refresh_fused_host(g, g['_stage_dtype'])
        self._stage.ensure_sent()

    def _fused_normalize(self, group, x: torch.Tensor) -> torch.Tensor:
        """[R1 optim2 S8e] x / (||x||_F * 1.01 + 1e-6) per matrix with tensor scalars: norm, one addcmul (or mul + add), div."""
        c = self._fused_consts(group, x.dtype, x.device)
        nrm = x.norm(dim=(-2, -1), keepdim=True)
        if self.op_supported('addcmul', x.device):
            denom = torch.addcmul(c['eps6'], nrm, c['k101'])
        else:
            denom = nrm * c['k101'] + c['eps6']
        return x / denom

    def _fused_tail(self, group, g: torch.Tensor, w: torch.Tensor, second: torch.Tensor, red_dim: int) -> torch.Tensor:
        """[R1 optim2 S8e] NorMuon scaling + the cautious weight-decay update with tensor scalars only (no scalar copies) and
        the products merged: v_mean * R is computed once (sum(v_mean * R) == sum(v_mean) * R exactly when R is a power of
        two, as every width in the sweep is), the 1e-10 floors are torch.maximum against a (1,) tensor, `lr * ratio` and
        `lr * wd` arrive from the host (see _refresh_fused_scalars), and lr g + lr wd w mask is one addcmul. Same fp32
        arithmetic as the baseline: bitwise on CPU fp32."""
        dev = g.device
        c = self._fused_consts(group, g.dtype, dev)
        use_max = self.op_supported('maximum', dev)
        use_addcmul = self.op_supported('addcmul', dev)
        if 'red_size_t' not in group:
            group['red_size_t'] = torch.full((1,), float(g.size(red_dim)), dtype=torch.float32, device=dev)
        if OPT_TAIL_VNORM and self.op_supported('vnorm_dtype', dev):
            v_sum = torch.linalg.vector_norm(g, 2, dim=red_dim, keepdim=True, dtype=torch.float32).square()
            v_mean = v_sum / group['red_size_t']
            _OPTTAIL_FIRED['vnorm'] += 1
        else:
            v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
            v_sum = v_mean * group['red_size_t']
        v_norm = v_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
        second.lerp_(v_mean, c['beta2_1m'])
        floored = torch.maximum(second, c['eps10']) if use_max else second.clamp_min(1e-10)
        step_size = floored.rsqrt()
        scaled = v_sum * step_size.square()
        v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
        v_norm_new = torch.maximum(v_norm_new, c['eps10']) if use_max else v_norm_new.clamp_min(1e-10)
        g = g * (step_size * (v_norm / v_norm_new)).to(g.dtype)
        mask = g * w >= c['zero']
        if OPT_TAIL_MASK and mask is not None and (c['zero'].dtype != torch.float32):
            _OPTTAIL_FIRED['mask'] += 1
        if mask is None:
            return group['lr_eff_t'] * g + group['lr_wd_t'] * w
        t1 = group['lr_eff_t'] * g
        t2 = group['lr_wd_t'] * w
        return torch.addcmul(t1, t2, mask) if use_addcmul else t1 + t2 * mask

    @torch.no_grad()
    def persist_muon_views(self, group) -> None:
        """[R1 optim2 S8c] Re-point the group's parameters at rows of one persistent [n, r, c] stack (n_pad rows under
        parameters disappear: the stack IS the parameters, `w.sub_(upd)` updates them in place, state_dict() returns the
        row views (same keys and shapes; the checkpoint reloads through load_for_eval unchanged). Level 2 also makes each
        .grad a row view of a zero grad stack: autograd accumulates into it in place (backward runs with grad mode off, so
        AccumulateGrad does `grad += new` for a defined .grad), the reducer and the Muon step read the stack directly, and
        zero_grad zeroes the stack with one launch per group. Called by setup_optimizer, after the bf16-master cast."""
        params = group['params']
        if not params:
            return
        p0 = params[0]
        state = self.state[p0]
        n, shape = (len(params), p0.shape)
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        n_rows = (n + world - 1) // world * world
        w = torch.zeros(n_rows, *shape, dtype=p0.dtype, device=p0.device)
        for i, p in enumerate(params):
            w[i].copy_(p)
            p.data = w[i]
        state['stacked_params'] = w
        state['param_views'] = True
        g = torch.zeros(n_rows, *shape, dtype=p0.dtype, device=p0.device)
        for i, p in enumerate(params):
            p.grad = g[i]
        state['stacked_grads'] = g
        state['grad_views'] = True
        group['grad_stack'] = g
        self._grad_views = True

    @torch.no_grad()
    def stage_muon_stack(self, group) -> torch.Tensor:
        """[R1 optim2 S8a] Copy the group's gradients into its [n, r, c] stack now (the copies _muon_step would issue anyway;
        none when the gradients are views of the stack, S8c=2) and mark them staged, so GradReducer takes the group's squared
        norm from one tensor and _muon_step skips its copy-in. Unsharded Muon only (S1 stages through stage_muon_grads)."""
        params = group['params']
        p0 = params[0]
        state = self.state[p0]
        if 'stacked_grads' not in state:
            state['stacked_grads'] = torch.empty(len(params), *p0.shape, dtype=p0.dtype, device=p0.device)
        g = state['stacked_grads']
        if not state.get('grad_views'):
            for i, p in enumerate(params):
                if p.grad is None:
                    g[i].zero_()
                else:
                    g[i].copy_(p.grad)
        state['grads_staged'] = True
        return g

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool=True) -> None:
        """[R1 optim2 S8c/S8d] Off: torch.optim.Optimizer.zero_grad, i.e. the baseline's per-parameter zero_(). With
        is overridden on purpose -- so the next backward assigns its first contribution (no zero_, no first add) and the
        group; their row views stay attached to the parameters."""
        for group in self.param_groups:
            stack = group.get('grad_stack')
            if stack is not None:
                stack.zero_()
                continue
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad = None

    @staticmethod
    def _adamw_update(p, grad, exp_avg, exp_avg_sq, bias1, bias2, lr_t, beta1, beta2, eps, weight_decay):
        """One AdamW update (the baseline's math, unchanged) on a parameter or on a stack of them."""
        if weight_decay > 0:
            p.mul_(1 - lr_t * weight_decay)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2)
        update = exp_avg / bias1 / ((exp_avg_sq / bias2).sqrt() + eps)
        p.sub_(update * lr_t)

    @torch.no_grad()
    def _adamw_step(self, group):
        beta1, beta2 = group['betas']
        group['step_count'] = group.get('step_count', 0) + 1
        group['bias1_cpu'].fill_(1 - beta1 ** group['step_count'])
        group['bias2_cpu'].fill_(1 - beta2 ** group['step_count'])
        group['bias1_t'].copy_(group['bias1_cpu'])
        group['bias2_t'].copy_(group['bias2_cpu'])
        bias1 = group['bias1_t']
        bias2 = group['bias2_t']
        lr_t = group['lr_t']
        stacked = group.get('stack_params')
        if stacked is not None:
            if not group.get('stack_staged'):
                self.stage_scalar_grads(group, distributed=False)
            group['stack_staged'] = False
            self._adamw_update(group['stack_master'], group['stack_grad'], group['stack_exp_avg'], group['stack_exp_avg_sq'], bias1, bias2, lr_t, beta1, beta2, group['eps'], group['weight_decay'])
        skip = group.get('stack_ids', ())
        for p in group['params']:
            if p.grad is None or id(p) in skip:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            self._adamw_update(p, grad, state['exp_avg'], state['exp_avg_sq'], bias1, bias2, lr_t, beta1, beta2, group['eps'], group['weight_decay'])

    @torch.no_grad()
    def stack_scalars(self, group) -> None:
        """[R1 optim S2] One flat master tensor for the group's 1-element parameters (resid_lambda, x0_lambda, ve_lambda,
        skip/bigram gains ...), so the AdamW state and update run once on the [m] stack instead of once per parameter, and
        GradReducer averages their gradients with one all_reduce. R1_ADAMW_STACK=1: the parameters become (1,) views of the
        master, so the in-place update lands in them with no write-back op; "copy": the master is separate and written back
        with one foreach copy. Parameters with more than one element (attn_scale, skip_lambda) keep the per-parameter path."""
        small = nki_optim.ADAMW_STACK_SMALL if nki_optim is not None else 0
        ones = [p for p in group['params'] if p.numel() == 1 or 1 < p.numel() <= small]
        if len(ones) < 2:
            return
        master = torch.cat([p.detach().reshape(-1) for p in ones]).contiguous()
        group['stack_params'] = ones
        group['stack_ids'] = frozenset((id(p) for p in ones))
        group['stack_master'] = master
        group['stack_grad'] = torch.zeros_like(master)
        group['stack_exp_avg'] = torch.zeros_like(master)
        group['stack_exp_avg_sq'] = torch.zeros_like(master)
        group['stack_staged'] = False
        off = 0
        for p in ones:
            p.data = master[off:off + p.numel()].view(p.shape)
            off += p.numel()

    @torch.no_grad()
    def stage_scalar_grads(self, group, distributed: bool) -> torch.Tensor:
        """[R1 optim S2] Concatenate the 1-element gradients of the group into one [m] tensor (one op) and average it across
        ranks with one all_reduce instead of m; the stacked AdamW update reads it. Returns the reduced stack."""
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in group['stack_params']])
        if distributed:
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
        group['stack_grad'] = flat
        group['stack_staged'] = True
        return flat

    def configure_sharding(self, device: torch.device) -> str:
        """[R1 optim S1] Resolve the collectives of the sharded Muon step once, before the clock starts. "auto" probes
        reduce_scatter_tensor / all_gather_into_tensor on a tiny tensor and checks the result against the expected values
        (a backend that raises OR silently no-ops falls back: all_reduce + local slice / list-form all_gather).
        Returns a one-line description for the log."""
        distributed = dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)
        if not (MUON_SHARD and distributed):
            self.use_reduce_scatter = False
            self.use_all_gather_into_tensor = False
            return 'single process: the shard is the whole stack, no collectives'
        world, rank = (dist.get_world_size(), dist.get_rank())
        base = torch.arange(4 * world, dtype=torch.float32, device=device)
        try:
            out = torch.zeros(4, dtype=torch.float32, device=device)
            dist.reduce_scatter_tensor(out, (base + float(rank)).contiguous(), op=dist.ReduceOp.AVG)
            want = torch.arange(4 * rank, 4 * rank + 4, dtype=torch.float32) + (world - 1) / 2
            self.use_reduce_scatter = bool(torch.allclose(out.cpu(), want))
            if not self.use_reduce_scatter:
                print0(f'muon shard: reduce_scatter_tensor returned {out.cpu().tolist()} (wanted {want.tolist()}); using all_reduce + slice')
        except Exception as err:
            print0(f'muon shard: reduce_scatter_tensor unavailable ({type(err).__name__}: {err}); using all_reduce + slice')
            self.use_reduce_scatter = False
        try:
            full = torch.zeros(4 * world, dtype=torch.float32, device=device)
            dist.all_gather_into_tensor(full, (base[:4] + float(rank)).contiguous())
            want = torch.cat([torch.arange(4, dtype=torch.float32) + r for r in range(world)])
            self.use_all_gather_into_tensor = bool(torch.allclose(full.cpu(), want))
            if not self.use_all_gather_into_tensor:
                print0(f'muon shard: all_gather_into_tensor returned {full.cpu().tolist()} (wanted {want.tolist()}); using list-form all_gather')
        except Exception as err:
            print0(f'muon shard: all_gather_into_tensor unavailable ({type(err).__name__}: {err}); using list-form all_gather')
            self.use_all_gather_into_tensor = False
        return f'world {world}: reduce={('reduce_scatter_tensor' if self.use_reduce_scatter else 'all_reduce + local slice')} gather={('all_gather_into_tensor' if self.use_all_gather_into_tensor else 'all_gather (list)')} payload={MUON_SHARD_GATHER} slab={MUON_SHARD_SLAB}'

    def _all_gather(self, out_full: torch.Tensor, shard: torch.Tensor) -> None:
        """[R1 optim S1] out_full[r*k:(r+1)*k] <- rank r's shard, for every r (a copy when there is no process group)."""
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            out_full.copy_(shard)
            return
        if self.use_all_gather_into_tensor:
            dist.all_gather_into_tensor(out_full, shard.contiguous())
        else:
            dist.all_gather(list(out_full.chunk(dist.get_world_size(), dim=0)), shard.contiguous())

    @staticmethod
    def _owned_index(j: int, s: int, world: int, rank: int) -> int:
        """[R1 optim S1] Stack index of shard row j: slab G = j // s covers stack rows [G*world*s, (G+1)*world*s), of
        which rank R owns rows [G*world*s + R*s, G*world*s + (R+1)*s) -- exactly what reduce_scatter_tensor over that
        contiguous block hands to rank R, and where all_gather_into_tensor puts R's piece back."""
        G, t = divmod(j, s)
        return G * world * s + rank * s + t

    @torch.no_grad()
    def stage_muon_grads(self, group, distributed: bool, norm: bool=True) -> torch.Tensor:
        """[R1 optim S1] Stack this rank's local Muon gradients of the group ([n_pad, r, c], n_pad = k * world, zero pad rows)
        and reduce them ACROSS ranks straight into this rank's shard ([k, r, c], k = ceil(n / world) matrices): one
        reduce_scatter_tensor(AVG) per slab of world*s matrices (s = R1_MUON_SHARD_SLAB, so no collective exceeds
        world*s*r*c elements; the backend stalls on very large flat reductions), or all_reduce(AVG) + a local copy of the
        owned rows when the backend lacks reduce_scatter. Called by GradReducer (before the finiteness gate) in place of n
        per-parameter all_reduces. Returns the shard's sum of squared gradients (0-dim); the reducer sums it over ranks so
        the global norm, and the gate, are identical on every rank."""
        params = group['params']
        p0 = params[0]
        state = self.state[p0]
        shape = p0.shape
        world = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        n = len(params)
        k = (n + world - 1) // world
        n_pad = k * world
        if 'grad_shard' not in state:
            s = k
            state['shard_k'], state['shard_s'], state['shard_world'], state['shard_rank'] = (k, s, world, rank)
            if 'stacked_grads' not in state:
                state['stacked_grads'] = torch.zeros(n_pad, *shape, dtype=p0.dtype, device=p0.device)
            state['grad_shard'] = torch.zeros(k, *shape, dtype=p0.dtype, device=p0.device)
            state['param_shard'] = torch.zeros(k, *shape, dtype=p0.dtype, device=p0.device)
            state['momentum_buffer'] = torch.zeros(k, *shape, dtype=p0.dtype, device=p0.device)
            state_shape = (k, shape[-2], 1) if shape[-2] >= shape[-1] else (k, 1, shape[-1])
            state['second_momentum_buffer'] = torch.zeros(state_shape, dtype=torch.float32, device=p0.device)
            if 'stacked_params' not in state:
                state['stacked_params'] = torch.zeros(n_pad, *shape, dtype=p0.dtype, device=p0.device)
        g = state['stacked_grads']
        if not state.get('grad_views'):
            for i, p in enumerate(params):
                if p.grad is None:
                    g[i].zero_()
                else:
                    g[i].copy_(p.grad)
        shard = state['grad_shard']
        s = state['shard_s']
        if world == 1:
            shard.copy_(g)
        elif self.use_reduce_scatter:
            for G in range(k // s):
                dist.reduce_scatter_tensor(shard[G * s:(G + 1) * s], g[G * world * s:(G + 1) * world * s], op=dist.ReduceOp.AVG)
        else:
            for G in range(k // s):
                dist.all_reduce(g[G * world * s:(G + 1) * world * s], op=dist.ReduceOp.AVG)
            for j in range(k):
                shard[j].copy_(g[self._owned_index(j, s, world, rank)])
        state['shard_staged'] = True
        return shard.float().square().sum() if norm else None

    @torch.no_grad()
    def _muon_update(self, group, g, w, momentum_buffer, second, shape):
        """Momentum, Polar Express Newton-Schulz, NorMuon scaling and the cautious weight decay on a stack of matrices
        (the baseline's math, unchanged). g and momentum_buffer are updated in place; returns the update to subtract."""
        momentum_t = group['momentum_t']
        momentum_buffer.lerp_(g, group['momentum1m_t'])
        g = g.lerp_(momentum_buffer, group['momentum_lo_t'])
        x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
        x = self._fused_normalize(group, x)
        x = self._newton_schulz(group, x)
        g = x
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        return self._fused_tail(group, g, w, second, red_dim)
        v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = g.size(red_dim)
        v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
        second.lerp_(v_mean, 1 - group['beta2'])
        step_size = second.clamp_min(1e-10).rsqrt()
        scaled = v_mean * red_dim_size * step_size.square()
        v_norm_new = scaled.sum(dim=(-2, -1), keepdim=True).sqrt()
        g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)
        lr = group['lr_t'] * max(1.0, shape[-2] / shape[-1]) ** 0.5
        wd = group['weight_decay']
        mask = g * w >= 0
        return lr * g + lr * wd * w * mask

    @torch.no_grad()
    def _muon_step_sharded(self, group, state, shape):
        """[R1 optim S1] This rank runs momentum + Newton-Schulz + NorMuon + the update on its k = ceil(n / world) matrices
        only (their momentum and second-moment state live here and nowhere else), then every rank receives every rank's
        result with one all_gather and copies the n matrices back into the parameters. Payload "params": the updated fp32
        (or bf16-master) shard itself, so all ranks hold bit-identical weights by construction; "update_bf16": the update
        rounded to bf16 (half the bytes), applied by every rank to its own full stack (identical weights too, since every
        rank applies the same bytes to the same weights)."""
        params = group['params']
        n = len(params)
        if not state.get('shard_staged'):
            self.stage_muon_grads(group, distributed=False)
        state['shard_staged'] = False
        k, s, world, rank = (state['shard_k'], state['shard_s'], state['shard_world'], state['shard_rank'])
        g = state['grad_shard']
        w = state['param_shard']
        if nki_optim is not None and nki_optim.MUON_SHARD_VIEW != '0' and (MUON_SHARD_GATHER == 'params'):
            self._refresh_fused_scalars(group, params[0].dtype)
            self._muon_finish_fp32(group, state, g, w, k, s, world, rank, shape)
            return
        raise RuntimeError('fp32master: the sharded Muon step missed the slab-0 view finish: refusing the per-matrix finish, which has no fp32 master')

    @torch.no_grad()
    def _muon_finish_fp32(self, group, state, g, w, k, s, world, rank, shape) -> None:
        """[R1 fp32m] R1_MUON_FP32_MASTER=1: nki_optim.muon_finish_view's slab-0 "sub" finish with an fp32 master of this
        rank's owned rows [rank*k, (rank+1)*k).  The update is today's: the same _muon_update on the same bf16 stack view,
        so the cautious mask and the decay term read today's weights and `upd` is today's fp32 tensor.  It is subtracted
        from the fp32 master (one fp32 rounding) instead of being rounded into bf16 twice (upd.to(bf16), then the bf16
        sub), and the bf16 rows this rank all-gathers are the master's ONE cast: the runtime's fp32 -> bf16 rounding
        (stochastic on the chip under R1_BF16_MASTER=1's NEURON_RT_STOCHASTIC_ROUNDING_EN=1, round-to-nearest on a CPU),
        either way a fresh error below one ulp each step that is never accumulated.  The master starts as the exact fp32
        value of the rows at the first finish.  Raises unless the view finish's preconditions hold: never a silent
        fall-through to a path without the master."""
        if s != k or not state.get('param_views'):
            raise RuntimeError(f'fp32master: needs the slab-0 persistent-view finish: s={s} k={k} param_views={state.get('param_views')}')
        stack = state['stacked_params']
        w_view = stack[rank * k:(rank + 1) * k]
        master = state.get('master32')
        if master is None:
            master = state['master32'] = w_view.to(torch.float32, copy=True)
            _FP32M['bytes'] += master.numel() * 4
            print(f'fp32master[rank {rank}]: group{tuple(shape)} rows [{rank * k}, {(rank + 1) * k}) of {world * k}: fp32 master <- the {w_view.dtype} rows ({master.numel() * 4:,} B)', flush=True)
        upd = self._muon_update(group, g, w_view, state['momentum_buffer'], state['second_momentum_buffer'], shape)
        assert upd.dtype == torch.float32 and upd.shape == master.shape, f'fp32master: the update is {upd.dtype} {tuple(upd.shape)}'
        master.sub_(upd)
        w.copy_(master)
        self._all_gather(stack[0:world * k], w)
        _FP32M['n'] += 1

    @torch.no_grad()
    def _muon_step(self, group):
        params = group['params']
        if not params:
            return
        p0 = params[0]
        state = self.state[p0]
        shape = p0.shape
        self._muon_step_sharded(group, state, shape)
        return
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if 'second_momentum_buffer' not in state:
            state_shape = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            state['second_momentum_buffer'] = torch.zeros(state_shape, dtype=torch.float32, device=p0.device)
        if 'stacked_grads' not in state:
            state['stacked_grads'] = torch.empty(len(params), *shape, dtype=p0.dtype, device=p0.device)
        if 'stacked_params' not in state:
            state['stacked_params'] = torch.empty(len(params), *shape, dtype=p0.dtype, device=p0.device)
        g = state['stacked_grads']
        w = state['stacked_params']
        copy_g = not (state.get('grad_views') or state.pop('grads_staged', False))
        copy_w = not state.get('param_views')
        for i, p in enumerate(params):
            if copy_g:
                if p.grad is None:
                    g[i].zero_()
                else:
                    g[i].copy_(p.grad)
            if copy_w:
                w[i].copy_(p)
        self._refresh_fused_scalars(group, p0.dtype)
        upd = self._muon_update(group, g, w, state['momentum_buffer'], state['second_momentum_buffer'], shape)
        w.sub_(upd if upd.dtype == w.dtype else upd.to(w.dtype))
        if copy_w:
            torch._foreach_copy_(params, list(w.unbind(0)))

    @torch.no_grad()
    def step(self):
        fused_adamw = nki_optim is not None and nki_optim.adamw_fused_step(self)
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                if fused_adamw:
                    continue
                self._adamw_step(group)
            elif group['kind'] == 'muon':
                self._muon_step(group)
            else:
                raise ValueError(f'unknown optimizer kind: {group['kind']}')

class GradReducer:
    """Averages gradients across ranks (one all_reduce per parameter, exactly as the baseline
    does: a single flat 59.6M-element buffer compiled pathologically on this backend), then
    computes the global gradient norm from parameter-shaped tensors and reports whether it is
    finite. The caller skips the optimizer update on a non-finite step instead of crashing."""

    def __init__(self, params, device: torch.device, optimizer: 'MuonAdamW | None'=None):
        self.params = list(params)
        self.bad_steps = torch.zeros((), dtype=torch.float32, device=device)
        self.last_gnorm = torch.zeros((), dtype=torch.float32, device=device)
        self.host_gate = OPT_TAIL_HOSTGATE
        self.gnorm_host = 0.0
        self.bad_host = 0
        self.optimizer = optimizer
        self.muon_groups = [g for g in optimizer.param_groups if g['kind'] == 'muon'] if optimizer is not None and MUON_SHARD else []
        self.stack_groups = [g for g in optimizer.param_groups if g.get('stack_params')] if optimizer is not None else []
        self.skip_ids = frozenset([id(p) for g in self.muon_groups for p in g['params']] + [id(p) for g in self.stack_groups for p in g['stack_params']])
        self.shard_sq = torch.zeros(1, dtype=torch.float32, device=device)
        self.norm_muon_groups: list = []
        self.norm_muon_ids: frozenset = frozenset()
        self.norm_scalar_params: list = []
        self.norm_ids: frozenset = frozenset()
        self.use_foreach_norm = False
        if self.host_gate and nki_optim is not None and nki_optim.USE_NKI_GRADNORM and self.muon_groups and (GRADNORM_STACK == '0') and (not getattr(nki_optim, 'REDUCE_HOST_GATE', False)):
            print0('opttail: hostgate REFUSED -- this nki_optim.reduce_fused predates it (no REDUCE_HOST_GATE); the device gate stays')
            self.host_gate = False

    @torch.no_grad()
    def reduce(self, clip: float) -> torch.Tensor:
        if nki_optim is not None and nki_optim.USE_NKI_GRADNORM and self.muon_groups and (GRADNORM_STACK == '0'):
            return nki_optim.reduce_fused(self, clip)
        distributed = dist.is_available() and dist.is_initialized()
        sq = None
        foreach_grads = [] if self.use_foreach_norm else None
        for p in self.params:
            if p.grad is None or id(p) in self.skip_ids:
                continue
            if distributed:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            if foreach_grads is not None:
                foreach_grads.append(p.grad)
                continue
            if id(p) in self.norm_ids:
                continue
            s = p.grad.float().square().sum()
            sq = s if sq is None else sq + s
        if foreach_grads:
            s = torch.stack(torch._foreach_norm(foreach_grads, dtype=torch.float32)).square().sum()
            sq = s if sq is None else sq + s
        for group in self.norm_muon_groups:
            g = self.optimizer.stage_muon_stack(group)
            s = g.float().square().sum()
            sq = s if sq is None else sq + s
        if self.norm_scalar_params:
            flat = torch.cat([p.grad if p.grad is not None else torch.zeros_like(p) for p in self.norm_scalar_params])
            s = flat.float().square().sum()
            sq = s if sq is None else sq + s
        for group in self.stack_groups:
            flat = self.optimizer.stage_scalar_grads(group, distributed)
            s = flat.float().square().sum()
            sq = s if sq is None else sq + s
        if self.muon_groups:
            shard_sq = None
            for group in self.muon_groups:
                s = self.optimizer.stage_muon_grads(group, distributed)
                shard_sq = s if shard_sq is None else shard_sq + s
            self.shard_sq.copy_(shard_sq)
            if distributed:
                dist.all_reduce(self.shard_sq, op=dist.ReduceOp.SUM)
            s = self.shard_sq.sum()
            sq = s if sq is None else sq + s
        gnorm = sq.sqrt()
        finite = None if self.host_gate else torch.isfinite(gnorm)
        if clip > 0:
            scale = torch.clamp(clip / (gnorm + 1e-06), max=1.0)
            for p in self.params:
                if p.grad is not None and id(p) not in self.skip_ids and (id(p) not in self.norm_muon_ids):
                    p.grad.mul_(scale)
            for group in self.norm_muon_groups:
                self.optimizer.state[group['params'][0]]['stacked_grads'].mul_(scale)
            for group in self.stack_groups:
                group['stack_grad'].mul_(scale)
            for group in self.muon_groups:
                self.optimizer.state[group['params'][0]]['grad_shard'].mul_(scale)
        if self.host_gate:
            return gnorm
        self.last_gnorm.copy_(torch.where(finite, gnorm, torch.zeros_like(gnorm)))
        self.bad_steps += (~finite).to(torch.float32)
        return finite

    def gate_host(self, gnorm: torch.Tensor) -> bool:
        """[R1 opttail d] The finiteness gate from the one host read the loop pays anyway (gnorm.item() instead of
        finite.item()): the same decision as isfinite on the fp32 value, last_gnorm = gnorm or 0 (the fp32 value itself,
        exactly), bad_steps + 1 on a non-finite step.  gnorm is bit-identical on every rank (the Muon term is one
        all_reduce of the shards' sums, the rest come from averaged grads), so every rank takes the same branch."""
        assert gnorm.dtype == torch.float32 and gnorm.numel() == 1, f'opttail hostgate: reduce returned {gnorm.dtype}{tuple(gnorm.shape)}, not the fp32 gnorm (a reduce_fused that ignores host_gate returns the bool finite)'
        g = float(gnorm.item())
        ok = math.isfinite(g)
        self.gnorm_host = g if ok else 0.0
        self.bad_host += 0 if ok else 1
        _OPTTAIL_FIRED['hostgate'] += 1
        return ok

    def gnorm_value(self) -> float:
        """[R1 opttail d] The logged gnorm: the host value under hostgate, else the device last_gnorm (the base read)."""
        return self.gnorm_host if self.host_gate else float(self.last_gnorm.item())

    def bad_steps_value(self) -> int:
        """[R1 opttail d] The logged bad_steps: the host count under hostgate, else the device counter (the base read)."""
        return self.bad_host if self.host_gate else int(self.bad_steps.item())

def _opttail_desc(optimizer, reducer) -> str:
    """[R1 opttail] On-device reachability, off the CONSTRUCTED objects (the A92_vload lesson): the items asked for, the
    NS switches, whether the host gate is live (a refused gate reads False) and the stage's layout (off / views)."""
    st = getattr(optimizer, '_stage', None)
    return f'items={','.join(sorted(_OPT_TAIL_TAGS)) or '-'} afold={MUON_NS_AFOLD} alpha={MUON_NS_ALPHA} hostgate_live={bool(getattr(reducer, 'host_gate', False))} stage={(st.desc() if st is not None else 'off')}'
TAIL_BLEND_G = -0.075
TAIL_BLEND_REF = 0.8
TAIL_BLEND_SAVE = 'final'

def tail_blend_apply(model: nn.Module, ref: dict, g: float) -> int:
    """[R1 tailblend] theta_T + g (theta_T - theta_ref) written into `model` in place; returns the number of tensors moved.

    tail_readout.py's `extrap` line for line: every state-dict tensor is read to the host in its stored dtype (a bf16
    master from its bf16 value, exactly as the offline tool reads final_raw.pt); a floating tensor that differs from
    its reference is replaced by (t.float() + g * (t.float() - r.float())).to(t.dtype) -- three separate fp32 ops, no
    fused multiply-add, one rounding back -- and every other tensor (an integer or a persistent buffer; a frozen or an
    unmoved parameter: bitwise equal in both operands) is left as it is.  Every read precedes the first write, so an
    aliased pair of keys would be written one value twice, never blended twice.  Host-side on purpose: nothing new is
    compiled at the end of the run (the write-back is the host->device copy `model.to(device)` already makes at
    startup), and the bytes are the offline tool's, so a confirm run's final.pt is reproducible by hand."""
    sd = model.state_dict()
    live = {k: v.detach().to('cpu', copy=True) for k, v in sd.items()}
    n = 0
    with torch.no_grad():
        for k, v in sd.items():
            t, r = (live[k], ref[k])
            if t.is_floating_point() and (not torch.equal(t, r)):
                tf = t.float()
                v.copy_((tf + g * (tf - r.float())).to(t.dtype))
                n += 1
    return n

def save_checkpoint(path: Path, model: GPT, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'model_config': asdict(model.config), 'meta': meta}
    torch.save(payload, path)

def torch_load_weights(path: str | Path, map_location):
    return torch.load(path, map_location=map_location, weights_only=True)

def load_for_eval(checkpoint_path: str, device: torch.device) -> nn.Module:
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    payload = torch_load_weights(checkpoint_path, map_location='cpu')
    config = GPTConfig(**payload['model_config'])
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(payload['model_state'], strict=True)
    model.to(device)
    if COMPUTE_DTYPE != torch.float32:
        model.to(COMPUTE_DTYPE)
    model.eval()
    return model

def lr_multiplier(step: int, progress: float, warmdown_ratio: float, final_lr_frac: float) -> float:
    """[R1] Step-based warmup, then flat, then a linear warmdown over the last `warmdown_ratio`
    of the clock down to `final_lr_frac`. `progress` is elapsed/max_train_seconds on the same
    clock the loop exits on, so the last step lands at (or within one step of) the floor."""
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    warmdown_start = 1.0 - warmdown_ratio
    if warmdown_ratio <= 0.0 or progress <= warmdown_start:
        return 1.0
    frac = max(0.0, (1.0 - progress) / warmdown_ratio)
    return final_lr_frac + (1.0 - final_lr_frac) * frac

def grad_accum_for_step(progress: float, grad_accum_steps: int) -> int:
    """[R1 arch A7] Batch ramp: half the micro-steps (half the global batch) while `progress` -- the loop's own
    elapsed/max_train_seconds, 0 during the excluded startup steps -- is below BATCH_RAMP_FRAC, the full count
    after. Decided once at the top of each optimizer step, so the switch lands on a step boundary. A plain int
    on purpose: it changes once, so `loss / accum` recompiles once. The schedule and the clock never see it."""
    if BATCH_RAMP and progress < BATCH_RAMP_FRAC:
        return grad_accum_steps // 2
    return grad_accum_steps

def muon_momentum(step: int, sched_progress: float=0.0, warmdown_ratio: float=0.0, steps_left: float | None=None) -> float:
    """Muon momentum 0.85 -> 0.95 over the first 300 steps (the baseline). [R1 optim S3] R1_MUON_MOMENTUM_SCHED=nanochat
    (6ed7d1d / a825e63): 0.85 -> 0.97 over 300 steps, flat, then 0.97 -> 0.90 linearly across the LR warmdown window
    (`sched_progress`, `warmdown_ratio` are the loop's own, the same ones lr_multiplier sees); cooldown50 (modded-nanogpt
    #40): the baseline ramp plus 0.95 -> 0.85 over the last 50 steps, `steps_left` from estimated_steps_left()."""
    warmdown_start = 1.0 - warmdown_ratio
    if warmdown_ratio > 0.0 and sched_progress > warmdown_start:
        return 0.97 - 0.07 * min(1.0, (sched_progress - warmdown_start) / warmdown_ratio)
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.97
    frac = min(step / 300, 1.0)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

def ramp_lr_factor(accum: int, grad_accum_steps: int) -> float:
    """[R1 optim S4] LR factor while the batch ramp (A7) runs the reduced batch: (accum / grad_accum_steps) ** exponent,
    i.e. (1/2)**0.5 = 0.71 or (1/2)**0.6 = 0.66 in the half-batch phase (nanochat: LR ~ sqrt(B); speedrun #46: (B/B0)^0.6;
    both LOWER the LR at the smaller batch). 1.0 when the coupling or the ramp is off, or the batch is full."""
    return 1.0

def main() -> None:
    parser = argparse.ArgumentParser(description='Neuron Competition R1 -- team working copy')
    env = os.environ.get
    parser.add_argument('--device-type', type=str, default='')
    parser.add_argument('--num-steps', type=int, default=int(env('NEURON_COMPETITION_R1_NUM_STEPS', NUM_STEPS)))
    parser.add_argument('--max-train-seconds', type=int, default=int(env('NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS', MAX_TRAIN_SECONDS)))
    parser.add_argument('--out-dir', type=str, default=env('NEURON_COMPETITION_R1_OUT_DIR', 'out'))
    parser.add_argument('--eval-public', action=argparse.BooleanOptionalAction, default=env('NEURON_COMPETITION_R1_EVAL_PUBLIC', '1') != '0')
    parser.add_argument('--eval-tokens', type=int, default=int(env('NEURON_COMPETITION_R1_EVAL_TOKENS', str(4 * 524288))), help='tokens for the in-run public eval; <= 0 = the whole public val shard')
    parser.add_argument('--compile', action='store_true', default=env('NEURON_COMPETITION_R1_COMPILE', '0') != '0', help='enable torch.compile on the full model (compiles during the excluded startup steps)')
    parser.add_argument('--nki-relu2', action='store_true', default=env('NEURON_COMPETITION_R1_NKI_RELU2', '0') != '0', help='use the example fused NKI relu(x)**2 kernel in the MLP (Neuron only; illustration)')
    parser.add_argument('--depth', type=int, default=int(DEPTH))
    parser.add_argument('--n-embd', type=int, default=1024, help='0 = derive from depth (baseline rule)')
    parser.add_argument('--device-batch-size', type=int, default=int(DEVICE_BATCH_SIZE), help='0 = auto')
    parser.add_argument('--total-batch-size', type=int, default=int(TOTAL_BATCH_SIZE))
    parser.add_argument('--pack-windows', type=int, default=int(PACK_WINDOWS), help='1 = shipped packer')
    parser.add_argument('--warmdown-ratio', type=float, default=float(WARMDOWN_RATIO))
    parser.add_argument('--final-lr-frac', type=float, default=float(FINAL_LR_FRAC))
    parser.add_argument('--grad-clip', type=float, default=float(GRAD_CLIP))
    parser.add_argument('--seed', type=int, default=91)
    args = parser.parse_args()
    global USE_NKI_RELU2
    USE_NKI_RELU2 = args.nki_relu2
    device_type = detect_device_type(args.device_type)
    if True:
        os.environ['NEURON_RT_STOCHASTIC_ROUNDING_EN'] = '1'
    ddp, rank, _local_rank, world_size, device = init_runtime(device_type)
    torch.manual_seed(args.seed)
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = compute_dtype_for(device)
    print0(f'device={device} dtype={COMPUTE_DTYPE} world_size={world_size} lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG', '')}')
    print0(f'NEURON_CC_FLAGS={os.environ.get('NEURON_CC_FLAGS', '')!r} stochastic_rounding={os.environ.get('NEURON_RT_STOCHASTIC_ROUNDING_EN', '')}')
    print0(f'budget: official={TRAIN_TIME_BUDGET_SECONDS}s max_train_seconds={args.max_train_seconds} excluded_startup_steps={TRAIN_STARTUP_STEPS_EXCLUDED} (cap {TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS}s) checkpoint_reserve={CHECKPOINT_RESERVE_SECONDS}s step_cap={args.num_steps} sched_endpoint={SCHED_ENDPOINT}')
    tokenizer = ensure_tokenizer(build_if_missing=False)
    config = build_config(args.depth, args.n_embd)
    print0(f'model_config={json.dumps(asdict(config), indent=2)}  (training window {SEQ_LEN}, RoPE table {ROPE_TABLE_LEN})')
    model = GPT(config)
    model.init_weights()
    model.to(device)
    num_params = sum((p.numel() for p in model.parameters()))
    flops_per_token = model.estimate_flops(SEQ_LEN)
    print0(f'num_params={num_params:,} flops_per_token={flops_per_token:e}')
    if BF16_MASTER and COMPUTE_DTYPE == torch.bfloat16:
        n_cast = 0
        head = model.lm_head if hasattr(model, 'lm_head') else None
        for m in model.modules():
            if isinstance(m, Linear) and m is not head:
                m.weight.data = m.weight.data.to(torch.bfloat16)
                n_cast += 1
        print0(f'bf16 master weights enabled for {n_cast} Linear layers (NEURON_RT_STOCHASTIC_ROUNDING_EN=' + os.environ.get('NEURON_RT_STOCHASTIC_ROUNDING_EN', '') + ')')
    orig_model = model
    if args.compile:
        compile_backend = 'neuron' if device_type == 'neuron' else 'inductor'
        model = torch.compile(model, backend=compile_backend, dynamic=False)
        print0(f'torch.compile enabled (backend={compile_backend})')
    total_batch = args.total_batch_size
    device_batch = args.device_batch_size
    if device_batch <= 0:
        device_batch = max(1, min(16, total_batch // (SEQ_LEN * world_size)))
    world_tokens_per_micro = device_batch * SEQ_LEN * world_size
    if total_batch % world_tokens_per_micro != 0:
        raise ValueError(f'TOTAL_BATCH_SIZE={total_batch} must be a multiple of micro tokens={world_tokens_per_micro} (device_batch={device_batch} x seq={SEQ_LEN} x world={world_size})')
    grad_accum_steps = total_batch // world_tokens_per_micro
    if BATCH_RAMP and grad_accum_steps % 2 != 0:
        raise ValueError(f'R1_BATCH_RAMP=1 needs an even grad_accum_steps, got {grad_accum_steps} (device_batch={device_batch}, total_batch={total_batch}, world={world_size})')
    pack_windows = max(1, args.pack_windows)
    if device_batch % pack_windows != 0:
        raise ValueError(f'device_batch={device_batch} must be a multiple of pack_windows={pack_windows}')
    loader_rows = device_batch // pack_windows
    print0(f'device_batch={device_batch} seq={SEQ_LEN} world={world_size} micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps} total_batch={total_batch:,} pack_windows={pack_windows} loader_rows={loader_rows}')
    print0(f'arch: attn_blocks={ATTN_BLOCKS!r} -> {len(orig_model.attn_layers)}/{config.n_layer} blocks {list(orig_model.attn_layers)} mixer={MIXER!r} -> {len(orig_model.mixer_layers)} blocks {list(orig_model.mixer_layers)} value_emb={VALUE_EMB} unet_skip={UNET_SKIP} attn_scale={ATTN_SCALE} mlp_ratio={MLP_RATIO} head_dim={HEAD_DIM} bigram_buckets={BIGRAM_BUCKETS} bigram_mode={BIGRAM_MODE} bigram_init={BIGRAM_INIT} init_resid={INIT_RESID} init_x0={INIT_X0} init_skip={INIT_SKIP} batch_ramp={BATCH_RAMP}' + f' (grad_accum {grad_accum_steps // 2} -> {grad_accum_steps} at progress {BATCH_RAMP_FRAC})' + (f' attention: {narrow_desc(orig_model)} num_params={num_params:,}' if orig_model.narrow else ''))
    print0(f'arch2: rope_base={ROPE_BASE} (tables built from {getattr(orig_model, '_rope_base_used', None)}) tie_embeddings={TIE_EMBEDDINGS}' + '' + f' fired={_QL_FIRED}')
    print0(f'optim: muon_shard={MUON_SHARD} (rs={MUON_SHARD_RS} ag={MUON_SHARD_AG} payload={MUON_SHARD_GATHER}) adamw_stack={ADAMW_STACK} momentum_sched={MUON_MOMENTUM_SCHED} ramp_lr_coupling={RAMP_LR_COUPLING} adam_betas_bundle={ADAM_BETAS_BUNDLE} qk_scale={QK_SCALE} (cfg {config.qk_scale} attn {next((b.attn.qk_scale for b in orig_model.transformer.h if b.attn is not None), None)}) wd_emb={ADAMW_WD_EMB} wd_head={ADAMW_WD_HEAD}')
    print0(f'optim2: gradnorm_stack={GRADNORM_STACK} ns_tensor_coefs={NS_TENSOR_COEFS} persistent_views={MUON_PERSISTENT_VIEWS} zero_grad_none={ZERO_GRAD_NONE} elementwise_fuse={MUON_ELEMENTWISE_FUSE}' + (f' fuse_deny={sorted(FUSE_DENY)}' if FUSE_DENY else ''))
    print0(_headdw_line(orig_model))
    print0(_mixfuse_line(orig_model))

    class _RowChunkedTokenizer:

        def __init__(self, inner, chunk_len: int):
            self._inner = inner
            self._chunk_len = chunk_len

        def get_bos_token_id(self):
            return self._inner.get_bos_token_id()

        def encode(self, text, prepend=None, num_threads: int=8):
            rows = self._inner.encode(text, prepend=prepend, num_threads=num_threads)
            if isinstance(text, str):
                return rows
            n = self._chunk_len
            return [ids[i:i + n] for ids in rows for i in range(0, len(ids), n)]
    loader_tokenizer = tokenizer
    train_loader = make_dataloader(loader_tokenizer, loader_rows, SEQ_LEN * pack_windows, 'train', device)
    replay = None
    if replay is not None:
        print0(f'replay: {replay.describe()}')

    def next_batch():
        if replay is not None:
            return replay.next()
        x, y, state = next(train_loader)
        if pack_windows > 1:
            x = x.view(-1, SEQ_LEN)
            y = y.view(-1, SEQ_LEN)
        return (x, y, state)
    x, y, loader_state = next_batch()
    optimizer = orig_model.setup_optimizer()
    reducer = GradReducer(orig_model.parameters(), device, optimizer)
    print0(f'muon shard: {optimizer.configure_sharding(device)}')
    print0(f'opttail: {_opttail_desc(optimizer, reducer)}')
    _fm = [(tuple(pg['params'][0].shape), len(pg['params'])) for pg in optimizer.param_groups if pg['kind'] == 'muon' and pg['params']]
    print0(f"fp32master: ARMED: the Muon update lands on an fp32 master of each rank's owned rows and the bf16 rows are its one cast, all-gathered as today; groups (shape, n, owned k) {[(s_, n_, -(-n_ // world_size)) for s_, n_ in _fm]}, {sum((-(-n_ // world_size) * math.prod(s_) * 4 for s_, n_ in _fm)):,} B of masters per rank")
    ema = None
    tb_ref = None
    print0(f'tailblend: ARMED ref={TAIL_BLEND_REF} g={TAIL_BLEND_G} save={TAIL_BLEND_SAVE}: final.pt = theta_T + g (theta_T - theta_ref)')
    _MTP['w'] = torch.zeros(2, device=device)
    print0('mtp: on w(t+2,t+3): clock [0,1/4) (.5,.25->0) [1/4,1/2) (.5->0,0) then next-token only')
    print0(f'mtp: ship ON: next-token + aux targets in ONE multi-target kernel pair; available={multi_available(device)} {multi_status()}')
    _BOSX['id'] = tokenizer.get_bos_token_id()
    if not multi_available(device):
        raise RuntimeError(f'bosx: no multi-target kernel pair here ({multi_status()})')
    print0(f'bosx: ARMED post: from clock 0.5 (the MTP end) the next-token loss is fused_loss_multi K=1 with row weight 1/N and 0 at BOS targets (id {_BOSX['id']}); the MTP window runs the base program')
    loop_started = time.monotonic()
    budget_started = loop_started
    smooth_loss = 0.0
    ema_updates = 0
    total_train_tokens = 0
    step = 0
    clock_start_step = 0
    startup_allowance_est = 0.0
    recent_step_times: list[float] = []
    last_lrm = 1.0
    last_loss = None
    consecutive_nan_logs = 0
    t_log = time.monotonic()
    gc.collect()
    gc.freeze()
    gc.disable()
    while step < args.num_steps:
        t0 = time.monotonic()
        progress = 0.0 if step < TRAIN_STARTUP_STEPS_EXCLUDED else min((t0 - budget_started) / args.max_train_seconds, 1.0)
        sched_progress = progress
        accum = grad_accum_for_step(progress, grad_accum_steps)
        mtp_set(progress)
        _BOSX['on'] = _MTP['ph'] == 2
        if _BOSX['on'] and _BOSX['hand'] is None:
            _BOSX['hand'] = (step, progress)
        step_tokens = accum * world_tokens_per_micro
        steps_left = None
        optimizer.zero_grad(set_to_none=False)
        for _ in range(accum):
            if replay is not None:
                tok_loss = model(x, y, loss_reduction='none')
                win_loss = tok_loss.sum(dim=1)
                replay.record(win_loss)
                loss = win_loss.sum() / tok_loss.numel()
            else:
                loss = model(x, y)
            last_loss = loss.detach()
            (loss / accum).backward()
            orig_model.flush_gain_grads()
            x, y, loader_state = next_batch()
        finite = reducer.reduce(args.grad_clip)
        last_lrm = lr_multiplier(step, sched_progress, args.warmdown_ratio, args.final_lr_frac) * ramp_lr_factor(accum, grad_accum_steps)
        _stage = optimizer._stage
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * last_lrm
            group['lr_cpu'].fill_(group['lr'])
            if _stage is None:
                group['lr_t'].copy_(group['lr_cpu'])
            if group['kind'] == 'muon':
                group['momentum'] = muon_momentum(step, sched_progress, args.warmdown_ratio, steps_left)
                group['momentum_cpu'].fill_(group['momentum'])
                if _stage is None:
                    group['momentum_t'].copy_(group['momentum_cpu'])
                group['weight_decay'] = WEIGHT_DECAY
        if _stage is not None:
            optimizer.stage_step_scalars()
        if reducer.host_gate:
            applied = reducer.gate_host(finite)
        else:
            applied = bool(finite.item())
        if applied:
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=False)
        if USE_NKI_GEMM_FP8:
            nki_gemm_fp8.begin_step()
        if USE_NKI_GEMM_FP8_ANY:
            nki_gemm_fp8_nn.begin_step()
        synchronize(device)
        if replay is not None:
            replay.end_step()
        record_step_boundary(step)
        dt = time.monotonic() - t0
        if step >= TRAIN_STARTUP_STEPS_EXCLUDED:
            recent_step_times.append(dt)
        if step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED:
            now = time.monotonic()
            observed_startup = now - _PROCESS_STARTED
            startup_allowance_est = min(observed_startup, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
            excess = max(0.0, observed_startup + STARTUP_LAUNCH_MARGIN_SECONDS - TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
            budget_started = now - excess
            clock_start_step = step + 1
            print0(f'startup allowance boundary: {step + 1} steps in {observed_startup:.1f}s since process start (excess charged {excess:.1f}s). Training clock starts now.')
        elapsed_after_step = time.monotonic() - budget_started
        reached_min_steps = step + 1 >= TRAIN_STARTUP_STEPS_EXCLUDED
        predicted_end = elapsed_after_step + (max(recent_step_times[-20:]) if recent_step_times else dt) + CHECKPOINT_RESERVE_SECONDS
        stop_after_step = step + 1 >= args.num_steps or (reached_min_steps and (elapsed_after_step >= args.max_train_seconds or predicted_end >= args.max_train_seconds))
        control_vals = [1 if stop_after_step else 0]
        if ema is not None:
            control_vals.append(1 if applied and progress >= EMA_START_FRAC else 0)
        control_vals.append(1 if tb_ref is None and progress >= TAIL_BLEND_REF else 0)
        control = torch.tensor(control_vals, device=device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(control, op=dist.ReduceOp.MAX)
        if ema is not None and int(control[1].item()):
            ema.update()
        if tb_ref is None and int(control[1].item()):
            _ts = time.monotonic()
            tb_ref = (step + 1, progress, {k: v.detach().to('cpu', copy=True) for k, v in orig_model.state_dict().items()})
            print0(f'tailblend: reference COPIED at step {step + 1} progress={progress:.4f} ({time.monotonic() - _ts:.2f}s)')
        if step >= TRAIN_STARTUP_STEPS_EXCLUDED:
            total_train_tokens += step_tokens
        step += 1
        log_step = step % LOG_EVERY == 0 or step <= TRAIN_STARTUP_STEPS_EXCLUDED + 3
        if log_step:
            train_loss = float(last_loss.item())
            if math.isnan(train_loss):
                consecutive_nan_logs += 1
                if consecutive_nan_logs >= 3:
                    raise RuntimeError('loss has been NaN for 3 consecutive log intervals')
            else:
                consecutive_nan_logs = 0
                smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
                ema_updates += 1
            debiased = smooth_loss / (1 - 0.9 ** max(ema_updates, 1))
            now = time.monotonic()
            interval_steps = LOG_EVERY if step % LOG_EVERY == 0 else 1
            tok_per_sec = interval_steps * step_tokens / max(now - t_log, 1e-06)
            t_log = now
            if rank == 0:
                print(f'step {step:05d} | loss {debiased:.4f} | lrm {last_lrm:.3f} | dt {dt:.2f}s | tok/s {tok_per_sec:,.0f} | gnorm {reducer.gnorm_value():.3f} | bad_steps {reducer.bad_steps_value()} | elapsed {now - budget_started:.0f}s | remaining {max(0.0, args.max_train_seconds - (now - budget_started)):.0f}s | epoch {loader_state['epoch']}', flush=True)
        if int(control[0].item()) != 0:
            break
    budget_elapsed_before_save = time.monotonic() - budget_started
    _MTP['on'] = (False,) * 2
    _BOSX['on'] = False
    checkpoint_path = Path(args.out_dir) / 'final.pt'
    raw_checkpoint_path = Path(args.out_dir) / 'final_raw.pt'
    meta = {'step': step, 'clocked_steps': step - clock_start_step, 'startup_allowance_est': startup_allowance_est, 'total_train_tokens': total_train_tokens, 'budget_elapsed_before_save': budget_elapsed_before_save, 'world_size': world_size, 'device_batch': device_batch, 'total_batch': total_batch, 'pack_windows': pack_windows, 'long_doc_chunks': LONG_DOC_CHUNKS, 'replay': REPLAY, 'batch_ramp': BATCH_RAMP, 'compiled': args.compile, 'seed': args.seed}
    ema_applied = False
    if ema is not None:
        meta['ema'] = {'decay': EMA_DECAY, 'start_frac': EMA_START_FRAC, 'warmup': ema.warmup, 'updates': ema.updates, 'save': EMA_SAVE, 'path': ema.path_exact + (f'/{ema.path_cast}' if ema.cast_p else ''), 'weights': 'raw'}
        if ema.updates == 0:
            print0(f'WARNING: R1_EMA_DECAY={EMA_DECAY} but the clock never reached R1_EMA_START_FRAC={EMA_START_FRAC} (no EMA update ran): the checkpoint holds the raw weights')
        else:
            ema.copy_to_model()
            meta['ema']['weights'] = 'ema'
            ema_applied = True
    tb = None
    meta['tail_blend'] = {'ref': TAIL_BLEND_REF, 'g': TAIL_BLEND_G, 'weights': 'raw', 'ref_step': tb_ref and tb_ref[0]}
    if tb_ref is None:
        print0(f'WARNING: tailblend: the clock never reached R1_TAIL_BLEND_REF={TAIL_BLEND_REF}: final.pt holds the raw weights')
    else:
        _t0 = time.monotonic()
        tb = tail_blend_apply(orig_model, tb_ref[2], TAIL_BLEND_G)
        meta['tail_blend'].update(weights='blend', blended=tb, seconds=round(time.monotonic() - _t0, 2))
        print0(f'tailblend: FIRED ref={TAIL_BLEND_REF} (step {tb_ref[0]}) g={TAIL_BLEND_G} blended {tb} of {len(tb_ref[2])} tensors ({meta['tail_blend']['seconds']}s): final.pt = theta_T + g (theta_T - theta_ref)')
    if rank == 0:
        save_checkpoint(checkpoint_path, orig_model, meta)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    budget_elapsed_after_save = time.monotonic() - budget_started
    public_eval = None
    causal = None
    if args.eval_public:
        eval_batch = int(min(device_batch, 16))
        if eval_batch != device_batch:
            print0(f'eval batch {eval_batch} (training device batch {device_batch})')
        causal = causality_check(orig_model, tokenizer, eval_batch, SEQ_LEN, device)
        public_eval = evaluate_bpb(orig_model, tokenizer, eval_batch, SEQ_LEN, device, split='public_val', max_tokens=args.eval_tokens if args.eval_tokens > 0 else None, timeout_seconds=EVAL_TIMEOUT_SECONDS)
    if rank == 0:
        print('---')
        print(f'checkpoint_path:    {checkpoint_path}')
        print(f'training_seconds:  {budget_elapsed_before_save:.1f}  (clock restarted after step {TRAIN_STARTUP_STEPS_EXCLUDED})')
        print(f'budget_seconds:    {TRAIN_TIME_BUDGET_SECONDS}')
        print(f'max_train_seconds: {args.max_train_seconds}')
        print(f'startup_allowance_est: {startup_allowance_est:.1f}')
        print(f'charged_seconds_est:   {budget_elapsed_after_save:.1f}')
        print(f'over_budget:       {budget_elapsed_after_save > TRAIN_TIME_BUDGET_SECONDS}')
        print(f'total_seconds:     {time.monotonic() - loop_started:.1f}')
        print(f'final_lrm:         {last_lrm:.4f}  (target floor {args.final_lr_frac})')
        print(f'sched_endpoint:    {SCHED_ENDPOINT}')
        print(f'bad_steps:         {reducer.bad_steps_value()}')
        print(f'opttail:           {_opttail_desc(optimizer, reducer)} fired={_OPTTAIL_FIRED}')
        print(f'fp32master:        finishes={_FP32M['n']} master_bytes={_FP32M['bytes']:,} (rank 0)')
        print('mtp:               fired ' + ' | '.join((f'phase{i + 1} steps={a} t+2={b} t+3={c}' for i, (a, b, c) in enumerate(_MTP['n']))))
        print(f'mtp ship:          calls phase1={_MTP['ns'][0]} phase2={_MTP['ns'][1]} phase3={_MTP['ns'][2]} {multi_status()}')
        print(f'bosx:              k1_calls={_BOSX['calls']} declined={_BOSX['declined']} hand_over={_BOSX['hand']} zero_rows_first={_BOSX['zero']} selfcheck_k1={_BOSX['chk']!r}')
        print(f'compiled:          {args.compile}')
        print(f'nki_relu2:         {args.nki_relu2}')
        print(f'nki_rope_norm:     {USE_NKI_ROPE_NORM}')
        print(f'nki_relu2_opt:     {USE_NKI_RELU2_OPT}')
        print(f'nki_gemm:          {USE_NKI_GEMM}  fused fc+relu2: {USE_NKI_FC_RELU2}')
        print(f'nki_gain_norm:     {USE_NKI_GAIN_NORM}')
        print(f'nki_block_head:    {USE_NKI_BLOCK_HEAD}  (norm2 kernel: {USE_NKI_NORM2})')
        print(f'nki_megablock:     {USE_NKI_MEGABLOCK}  tail(phase2): {USE_NKI_MEGABLOCK_TAIL}')
        print(f'nki_emb_bwd:       {USE_NKI_EMB_BWD}')
        print(f'nki_loss_head:     {USE_NKI_LOSS_HEAD}')
        print(f'nkilib_attn:       {USE_NKILIB_ATTN}')
        print(f'nki_attn:          {USE_NKI_ATTN}  fused rope+attn: {USE_NKI_ATTN_ROPE}  qkv stack view: {QKV_STACK_VIEW}')
        print(f'glue:              grad_accum_inplace={GRAD_ACCUM_INPLACE!r} gain_batch={GAIN_BATCH} cast_cache={CAST_CACHE} emb_shadow={EMB_SHADOW}')
        print(f'nki_optim:         {(nki_optim.config_string() if nki_optim is not None else 'unavailable')}')
        print(f'nki_gemm_fp8:      {(nki_gemm_fp8.config_string() if nki_gemm_fp8 is not None else 'unavailable')}')
        print(f'nki_gemm_fp8_nn:   {(nki_gemm_fp8_nn.config_string() if nki_gemm_fp8_nn is not None else 'unavailable')}')
        print(f'num_steps:         {step}  (charged {step - clock_start_step})')
        print(f'total_tokens_M:    {total_train_tokens / 1000000.0:.1f}  (charged steps only)')
        print(f'world_size:        {world_size}')
        print(f'lnc:               {os.environ.get('NEURON_LOGICAL_NC_CONFIG', '')}')
        print(f'device_batch:      {device_batch}  grad_accum {grad_accum_steps}  total_batch {total_batch}')
        print(f'pack_windows:      {pack_windows}')
        print(f'replay:            {REPLAY}' + (f'  alpha {REPLAY_ALPHA} floor {REPLAY_FLOOR} passes {replay.passes} windows {replay.n:,}' if replay is not None else ''))
        print(f'arch:              attn_blocks={ATTN_BLOCKS!r} -> {len(orig_model.attn_layers)}/{config.n_layer} blocks {list(orig_model.attn_layers)} value_emb={VALUE_EMB} unet_skip={UNET_SKIP} attn_scale={ATTN_SCALE} mlp_ratio={MLP_RATIO} head_dim={HEAD_DIM} bigram_buckets={BIGRAM_BUCKETS} bigram_mode={BIGRAM_MODE} bigram_init={BIGRAM_INIT} init_resid={INIT_RESID} init_x0={INIT_X0} init_skip={INIT_SKIP} batch_ramp={BATCH_RAMP}')
        print(f'arch3:             mixer={MIXER!r} -> {len(orig_model.mixer_layers)} blocks {list(orig_model.mixer_layers)}' + (f'  mix_lambda={[round(float(b.mix_lambda), 6) for b in orig_model.transformer.h if b.mix_lambda is not None]}' if orig_model.mixer_layers else ''))
        if orig_model.narrow:
            print(f'narrow:            {narrow_desc(orig_model)} fused={_NARROW['fused']} qkv_views={sum((1 for b in orig_model.transformer.h if b.attn is not None and getattr(b.attn, 'qkv_view', None) is not None))}/{len(orig_model.attn_layers)}')
        print(f'arch2:             rope_base={ROPE_BASE} (tables {getattr(orig_model, '_rope_base_used', None)}) tie_embeddings={TIE_EMBEDDINGS}' + '' + f' fired={_QL_FIRED}')
        print(f'optim:             muon_shard={MUON_SHARD} adamw_stack={ADAMW_STACK} momentum_sched={MUON_MOMENTUM_SCHED} ramp_lr_coupling={RAMP_LR_COUPLING} adam_betas_bundle={ADAM_BETAS_BUNDLE} qk_scale={QK_SCALE} (cfg {config.qk_scale} attn {next((b.attn.qk_scale for b in orig_model.transformer.h if b.attn is not None), None)}) wd_emb={ADAMW_WD_EMB} wd_head={ADAMW_WD_HEAD}')
        print(f'optim2:            gradnorm_stack={GRADNORM_STACK} ns_tensor_coefs={NS_TENSOR_COEFS} persistent_views={MUON_PERSISTENT_VIEWS} zero_grad_none={ZERO_GRAD_NONE} elementwise_fuse={MUON_ELEMENTWISE_FUSE}')
        print(_headdw_line(orig_model, pad=True))
        print(_mixfuse_line(orig_model, pad=True))
        if ema is not None:
            print(f'ema:               decay={EMA_DECAY} start_frac={EMA_START_FRAC} save={EMA_SAVE} updates={ema.updates} applied={ema_applied} path={ema.path_exact}' + (f'/{ema.path_cast}' if ema.cast_p else ''))
        print(f'tailblend:         ref={TAIL_BLEND_REF} g={TAIL_BLEND_G} save={TAIL_BLEND_SAVE} applied={tb is not None} blended={tb} ref_step={tb_ref and tb_ref[0]}')
        print(f'seq_len:           {SEQ_LEN}')
        print(f'num_params_M:      {num_params / 1000000.0:.1f}')
        if causal is not None:
            print(f'causality_passed:  {causal['passed']}')
            print(f'causality_max_abs: {causal['max_abs_diff']:.6f}')
        if public_eval is not None:
            print(f'public_val_bpb:    {public_eval['val_bpb']:.6f}')
            print(f'eval_seconds:      {public_eval['eval_seconds']:.1f}')
            print(f'eval_tokens:       {public_eval['num_tokens']}  (requested {(args.eval_tokens if args.eval_tokens > 0 else 'whole shard')})')
            print(f'eval_timeout:      {public_eval['eval_timeout']}')
            print(f'eval_over_timeout: {public_eval['eval_over_timeout']}')
    cleanup_runtime()
if __name__ == '__main__':
    main()