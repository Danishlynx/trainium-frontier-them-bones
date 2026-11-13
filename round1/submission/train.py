"""Trainium competition Phase 1 submission (team Them Bones).

Single-file training script: nanochat-derived GPT, depth 12, attention in 9 of 12
blocks and a causal one-token-shift mixer in the other 3, MLP ratio 6, Muon +
AdamW, bf16 with fp8 MLP GEMMs, hand-written NKI kernels inlined.
The c_proj forward saturates h at the fp8 range in evaluation as it does in training.
Training adds auxiliary losses on the tokens two and three ahead during the first
half of the run, computed with the next-token loss in one fused loss-head kernel
pair and annealed to zero by half the run; evaluation is unchanged.
The saved weights are theta_T + g (theta_T - theta_0.80), g = -0.075: a two-point average
of the final iterate with the iterate at 80% of the clock, written after training.
The lm_head weight gradient x^T dy is computed as 2 column blocks, concatenated.
After the auxiliary window the next-token loss gives weight 0 to positions whose target is the
BOS token that opens the next document, computed by the same fused loss-head kernel pair.
Each rank keeps an fp32 copy of the Muon matrices it updates; the update is applied to that copy
and the bf16 weights are its rounding.
The attention has 6 heads of 128: its width is 768 of the 1024-wide residual stream, and its output
projection (768 to 1024) is stored [in, out] like the MLP's c_fc.

How this file was derived from the documented source, to fit the 262,144-byte
submission limit that appeared on 2026-09-26:
 1. Code that the shipped configuration cannot execute was removed: branches
    decided by module-level constants bound exactly once to a literal (kernel
    layout, engine and dtype variants; CPU emulation and fp8 quality-probe
    switches), and the functions, classes and constants that only those branches
    read.  In the kernel modules this also covers constants computed from
    literals (e.g. max(1, min(4, 4))) and constant operands of conditions:
    `if SWITCH and x:` with SWITCH always True reads `if x:`.  Every fallback
    that guards a kernel launch is kept.
 2. Comments, docstrings and blank lines were removed.  Repeated literals (strings,
    None, True, False) and frequently used builtins (print, len, range, int and
    others) are referenced through short module-level aliases, e.g. C=None,
    I=True, D=print, so code reads `x is C` for `x is None`.
 3. Inside the inlined NKI kernel modules (the name prefixes listed below) the
    module-level names, the local variables, the arguments that no call passes
    by keyword and the NKI namespace aliases (nki, nki.language, nki.isa) were
    shortened, except the entry points listed at the end, which the model and
    optimizer code call.  Every name in the model, optimizer, data and
    training-loop code (module-level, argument and local) is as in the source.
The program is unchanged: a 3-step CPU training run gives a bitwise-identical
checkpoint to the documented source, available on request.  Launch:
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py

Inlined kernel modules, by name prefix:
 attnk_   fused RoPE + QK-norm + causal attention, forward and backward kernels,
          and the plain causal attention kernel of the fallback path
 ropek_   RoPE + QK-norm kernel of the fallback path
 mblk_    block head (residual gains + U-net skip + RMSNorm), tail norm, final norm
 bheadk_  block-head and RMSNorm kernels of the fallback path
 gemmk_   bf16 GEMM and fused c_fc + ReLU^2 kernels
 fp8k_, fp8nnk_, fp8tnk_   fp8 MLP GEMMs (forward rows 1 and 2, gradient rows
          3 to 6), batched weight quantisation, the row-2 eval cap
 relu2k_  ReLU^2 backward rebuilt from h
 lossk_   softcap + log-softmax + cross-entropy loss head, single and multi target
 mixk_    shift mixer fused with the residual add and RMSNorm
 optk_    AdamW, gradient-norm and Muon-view kernels
 embk_    embedding with an NKI backward
Kernel-module names kept readable (41 of them; every other name with one of the
prefixes above is shortened):
 attnk_USE_NKI_ATTN, attnk_USE_NKI_ATTN_ROPE, attnk_nki_causal_attention,
 attnk_nki_rope_attention, bheadk_USE_NKI_NORM2, bheadk_block_head,
 bheadk_rmsnorm_fast, embk_USE_NKI_EMB_BWD, embk_embedding_fwd, fp8k_API,
 fp8k_USE_NKI_GEMM_FP8, fp8nnk_ROW3, fp8nnk_ROW4, fp8nnk__fp8_slice1,
 gemmk_FcRelu2Fn, gemmk_USE_NKI_FC_RELU2, gemmk_USE_NKI_GEMM,
 gemmk__relu2_backward, gemmk_eager_relu2_backward, gemmk_fused_fc_relu2,
 gemmk_fused_fc_relu2_forward, gemmk_nki_matmul, lossk_USE_NKI_LOSS_HEAD,
 lossk__MULTI_CHECK, lossk__MULTI_CHECKED, lossk__multi_selfcheck,
 lossk_fused_loss, lossk_fused_loss_multi, lossk_multi_available,
 mblk_SELF_SKIP, mblk_final_norm, mblk_gain_partials_take, mblk_head2,
 mblk_head_skip, mblk_tail_norm2, mixk_applies, mixk_mix_tail, optk_API,
 relu2k_relu_squared, ropek_USE_NKI_ROPE_NORM, ropek_rope_norm
"""
from __future__ import annotations
Kc='model_config'
Kb='model_state'
Ka='param_shard'
KZ='shard_rank'
KY='shard_world'
KX='shard_k'
KW='grads_staged'
KV='grad_stack'
KU='beta2_1m'
KT='add_alpha'
KS='ge_lowp'
KR='vnorm_dtype'
KQ='maximum'
KP='baddbmm'
KO='initial_lr'
KN='lm_head'
KM='nki not importable'
KL='nki_gemm_fp8'
KK='nki_gemm'
KJ='host_gate'
KI='stack_group'
KH='tables'
KG='groups'
KF='gradnorm'
KE='float32'
KD='0,2,3,4,5,6,7,9,10'
KC='static'
KB='NEURON_CC_FLAGS'
Gm='shard_staged'
Gl='shard_s'
Gk='eps10'
Gj='eps6'
Gi='k101'
Gh='ns_steps'
Gg='bias2_t'
Gf='bias1_t'
Ge='qkv_view'
Gd='proj'
Gc='sum'
Gb='off0'
Ga='scale'
GZ='pos'
GY='split'
GX='bsigned'
GW='bscale'
GV='asigned'
GU='gslot'
GT='classc'
GS='step'
GR='qdq'
GQ='stack_exp_avg_sq'
GP='stack_exp_avg'
GO='betas'
GN='stack_ids'
GM='fold'
GL='affine'
GK='bytes'
GJ=round
GI=AssertionError
Eq='grad_views'
Ep='momentum1m_t'
Eo='ratio_cpu'
En='addcmul'
Em='ns'
El='r4'
Ek='r3'
Ej='fp8'
Ei='nn'
Eh='unit'
Eg='bfmt'
Ef='trans_b'
Ee='train'
Ed='bias2_cpu'
Ec='bias1_cpu'
Eb='beta2'
Ea='nc'
EZ='tc'
EY='unroll'
EX='afold'
EW='hostgate'
EV='vnorm'
EU='mask'
ET='ema'
ES='all'
ER='fused'
EQ='NEURON_RT_STOCHASTIC_ROUNDING_EN'
EP=hasattr
D8='second_momentum_buffer'
D7='momentum_buffer'
D6='_stage_dtype'
D5='momentum_lo_cpu'
D4='momentum1m_cpu'
D3='lr_wd_t'
D2='lr_wd_cpu'
D1='momentum'
D0='gemm_class'
C_='declined'
Cz='id'
Cy='batches'
Cx='slice1'
Cw='calls'
Cv='signed'
Cu='param_views'
Ct='exp_avg_sq'
Cs='exp_avg'
Cr='eps'
Cq='psum'
Cp='out'
Co='scalar'
Cn='bf16'
Cm='fp32'
Cl='stage'
Ck='shift1'
Cj=next
CI='momentum_lo_t'
CH='wd_cpu'
CG='lr_eff_t'
CF='momentum_cpu'
CE='momentum_t'
CD='lr_t'
CC='hand'
CB='ph'
CA='not run'
C9='r1'
C8='e4m3'
C7='stack_grad'
C6='stack_master'
C5='vector'
C4='cpu'
C3='cuda'
C2='off'
Bg='lr_eff_cpu'
Bf='s'
Be='mean'
Bd='what'
Bc='ascale'
Bb='grad_shard'
Ba='lr_cpu'
BZ='stack_staged'
BY='stack_params'
BX='act'
BW='loop'
BV='evac'
BU='xt'
BT='chunk'
BS='forward'
BR='auto'
BQ='input'
BP=sum
B6='zero'
B5='check'
B4='weights'
B3='wscale'
B2='e5m2'
B1='weight_decay'
B0='lr'
A_=frozenset
Ao='step_count'
An='alt'
Am='backward'
Al=RuntimeError
Ag='stacked_params'
Af='adamw'
AZ='stacked_grads'
AY='muon'
AX='vec'
AW='none'
AV='n'
AS='on'
AR='afmt'
AQ='tb'
AP=isinstance
AO=sorted
AH='group'
AG=','
AF=ValueError
AC='kind'
AB='NEURON_LOGICAL_NC_CONFIG'
AA=dict
A7=enumerate
y='sbuf'
A1='0'
x=id
A0=list
z=bool
v=str
q='1'
n=getattr
l='params'
g=min
i=max
a='neuron'
e=tuple
Y=''
W=type
V=len
N=staticmethod
S=.0
P=Exception
Q=1.
G=float
D=print
B=int
O=False
L=range
I=True
C=None
import os
os.environ.setdefault(KB,'--enable-saturate-infinity --enable-fast-loading-neuron-binaries')
os.environ.setdefault(EQ,A1)
import argparse,gc,json,math,time
from dataclasses import asdict,dataclass
from pathlib import Path
_PROCESS_STARTED=time.monotonic()
import torch,torch.distributed as dist,torch.nn as nn,torch.nn.functional as F
from prepare import EVAL_TIMEOUT_SECONDS,TOKENIZER_VOCAB_SIZE,TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS,TRAIN_STARTUP_STEPS_EXCLUDED,TRAIN_TIME_BUDGET_SECONDS,Tokenizer,causality_check,ensure_tokenizer,evaluate_bpb,get_dist_info,make_dataloader,print0,record_step_boundary
SEQ_LEN=1024
ROPE_TABLE_LEN=4096
DEPTH=12
ASPECT_RATIO=64
HEAD_DIM=128
N_HEAD=6
_NARROW={ER:0}
TOTAL_BATCH_SIZE=262144
DEVICE_BATCH_SIZE=32
PACK_WINDOWS=8
LONG_DOC_CHUNKS=O
REPLAY=O
REPLAY_ALPHA=Q
REPLAY_FLOOR=.25
NUM_STEPS=10**9
MAX_TRAIN_SECONDS=TRAIN_TIME_BUDGET_SECONDS
CHECKPOINT_RESERVE_SECONDS=1e1
STARTUP_LAUNCH_MARGIN_SECONDS=3e1
LOGIT_SOFTCAP=15.
CE_PATH=KC
BF16_MASTER=I
WKN=I
HEAD_DW_NSPLIT=2
VALUE_EMB=C2
UNET_SKIP=I
ATTN_GATE=0
ATTN_SCALE='fixed'
ATTN_BLOCKS=KD
MIXER=Ck
MIXER_FUSED=I
ROPE_BASE=100000
TIE_EMBEDDINGS=O
TIE_LR=.0424
TIE_INIT_STD=.001
_QL_FIRED={}
MLP_RATIO=6
BATCH_RAMP=I
BATCH_RAMP_FRAC=.3
QKV_STACK_VIEW=I
BIGRAM_BUCKETS=0
BIGRAM_MODE=BQ
BIGRAM_INIT='normal'
SCHED_ENDPOINT='budget'
MUON_SHARD=I
MUON_SHARD_RS=BR
MUON_SHARD_AG=BR
MUON_SHARD_GATHER=l
MUON_SHARD_SLAB=0
ADAMW_STACK=q
MUON_MOMENTUM_SCHED='nanochat'
RAMP_LR_COUPLING=0
ADAM_BETAS_BUNDLE=O
QK_SCALE=.11
INIT_RESID=Q
INIT_X0=.05
INIT_SKIP=Q
ADAMW_WD_EMB=.001
ADAMW_WD_HEAD=.01
MUON_BETA2=.9
MUON_NS_STEPS=5
GRADNORM_STACK=A1
NS_TENSOR_COEFS=q
MUON_PERSISTENT_VIEWS=2
ZERO_GRAD_NONE=I
GRAD_ACCUM_INPLACE=ES
GAIN_BATCH=I
CAST_CACHE=q
EMB_SHADOW=I
_ACCUM_TAGS=A_(t.strip()for t in GRAD_ACCUM_INPLACE.split(AG)if t.strip())
MUON_ELEMENTWISE_FUSE=I
FUSE_DENY=A_(s for s in Y.split(AG)if s)
EMA_DECAY=S
EMA_START_FRAC=S
EMA_SAVE=ET
EMBEDDING_LR=.3
UNEMBEDDING_LR=.006
MATRIX_LR=.06508
SCALAR_LR=.05
WEIGHT_DECAY=.05
RESID_DROPOUT=S
WARMUP_STEPS=20
WARMDOWN_RATIO=.6
FINAL_LR_FRAC=.02
GRAD_CLIP=S
LOG_EVERY=10
_OPT_TAIL_TAGS=A_((EU,EV,EW,Cl))
OPT_TAIL_MASK=EU in _OPT_TAIL_TAGS
OPT_TAIL_VNORM=EV in _OPT_TAIL_TAGS
OPT_TAIL_HOSTGATE=EW in _OPT_TAIL_TAGS
OPT_TAIL_STAGE=Cl in _OPT_TAIL_TAGS
MUON_NS_AFOLD=I
MUON_NS_ALPHA=O
_OPTTAIL_FIRED={EU:0,EV:0,EW:0,Cl:0,EX:0,'alpha':0}
_FP32M={AV:0,GK:0}
def neuron_available():
	try:torch.device(a);return I
	except P:return O
def detect_device_type(requested=Y):
	if requested:return requested
	if torch.cuda.is_available():return C3
	if neuron_available():return a
	return C4
def init_runtime(device_type):
	assert device_type in{C4,C3,a};torch.manual_seed(42);ddp,rank,local_rank,world_size=get_dist_info()
	if ddp and device_type==C3:device=torch.device(C3,local_rank);torch.cuda.set_device(device);dist.init_process_group('nccl',device_id=device)
	elif ddp and device_type==a:dist.init_process_group(a);device=torch.device(a)
	elif ddp:dist.init_process_group('gloo');device=torch.device(C4)
	else:device=torch.device(device_type)
	if dist.is_available()and dist.is_initialized():dist.barrier()
	return ddp,rank,local_rank,world_size,device
def cleanup_runtime():
	if dist.is_available()and dist.is_initialized():dist.destroy_process_group()
def synchronize(device):
	if device.type==C3:torch.cuda.synchronize()
	elif device.type==a and EP(torch,a):torch.neuron.synchronize()
def compute_dtype_for(device):
	if Y in(Cm,KE):return torch.float32
	if device.type==C3 and torch.cuda.is_bf16_supported():return torch.bfloat16
	if device.type==a:return torch.bfloat16
	return torch.float32
@dataclass
class GPTConfig:sequence_len:B;vocab_size:B;n_layer:B;n_head:B;n_embd:B;mlp_ratio:B=6;value_emb:v=C2;unet_skip:z=O;attn_scale:v='fixed';bigram_buckets:B=0;bigram_mode:v=BQ;wkn:z=O;attn_gate:B=0;attn_blocks:v=KD;rope_base:B=100000;tie_embeddings:z=O;mixer:v=Ck;qk_scale:G=S;head_dim:B=0
def attn_block_ids(config):
	spec=v(config.attn_blocks).strip()
	if spec==ES:return e(L(config.n_layer))
	if spec==AW:return()
	ids=[]
	for tok in spec.split(AG):
		tok=tok.strip()
		if not tok.isdigit():raise AF(f"attn_blocks={config.attn_blocks!r}: {tok!r} is not a block index (expected 'all', 'none' or e.g. '0,2,4,6,8')")
		i=B(tok)
		if not 0<=i<config.n_layer:raise AF(f"attn_blocks={config.attn_blocks!r}: block {i} out of range for n_layer={config.n_layer}")
		if i in ids:raise AF(f"attn_blocks={config.attn_blocks!r}: block {i} listed twice")
		ids.append(i)
	return e(AO(ids))
def attn_dims(config):d=config.head_dim or config.n_embd//config.n_head;return d,config.n_head*d
def attn_geometry(config):
	if not config.head_dim and config.n_embd%config.n_head:raise AF(f"n_embd={config.n_embd} is not a multiple of n_head={config.n_head} and head_dim is unset: the model_config of a narrow-attention run without its head_dim field cannot rebuild the trained model")
	d,w=attn_dims(config)
	if not 0<w<=config.n_embd:raise AF(f"attention geometry n_head={config.n_head} head_dim={d}: the width {w} must be in (0, n_embd={config.n_embd}]")
	if w!=config.n_embd and(config.value_emb!=C2 or config.attn_gate):raise AF(f"value_emb={config.value_emb!r} / attn_gate={config.attn_gate} with a narrow attention (width {w}): refused (both are n_embd wide)")
	return d,w
def narrow_desc(model):
	a=Cj((b.attn for b in model.transformer.h if b.attn is not C),C);d,w=model.attn_dims
	if a is C:return f"n_head={model.config.n_head} head_dim={d} width={w} (no attention block)"
	return f"n_head={model.config.n_head} head_dim={d} width={w} c_q {A0(a.c_q.weight.shape)} c_k {A0(a.c_k.weight.shape)} c_v {A0(a.c_v.weight.shape)} c_proj {"LinearT"if AP(a.c_proj,LinearT)else"Linear"} {A0(a.c_proj.weight.shape)}"
def mixer_mode(config):
	m=v(config.mixer).strip()
	if m not in(C2,Ck):raise AF(f"mixer={config.mixer!r}: expected 'off' or 'shift1'")
	return m
def shift1(x):return torch.cat((x.new_zeros(x.shape[0],1,x.shape[2]),x[:,:-1,:]),dim=1)
COMPUTE_DTYPE=torch.float32 if Cn in(Cm,KE)else torch.bfloat16
def norm(x):return F.rms_norm(x,(x.size(-1),))
_NKI_PARTITION=128
try:
	import nki as K,nki.language as A,nki.isa as E
	@K.jit
	def _relu_sq_fwd(x):
		M,N=x.shape;out=A.ndarray((M,N),dtype=x.dtype,buffer=A.shared_hbm)
		for t in A.affine_range((M+_NKI_PARTITION-1)//_NKI_PARTITION):rows=A.ds(t*_NKI_PARTITION,_NKI_PARTITION);xb=A.ndarray((_NKI_PARTITION,N),dtype=x.dtype,buffer=A.sbuf);E.dma_copy(dst=xb,src=x[rows]);rb=A.ndarray((_NKI_PARTITION,N),dtype=x.dtype,buffer=A.sbuf);E.tensor_scalar(dst=rb,data=xb,op0=A.maximum,operand0=S);yb=A.ndarray((_NKI_PARTITION,N),dtype=x.dtype,buffer=A.sbuf);E.tensor_tensor(dst=yb,data1=rb,data2=rb,op=A.multiply);E.dma_copy(dst=out[rows],src=yb)
		return out
	@K.jit
	def _relu_sq_bwd(x,grad_out):
		M,N=x.shape;out=A.ndarray((M,N),dtype=x.dtype,buffer=A.shared_hbm)
		for t in A.affine_range((M+_NKI_PARTITION-1)//_NKI_PARTITION):rows=A.ds(t*_NKI_PARTITION,_NKI_PARTITION);xb=A.ndarray((_NKI_PARTITION,N),dtype=x.dtype,buffer=A.sbuf);E.dma_copy(dst=xb,src=x[rows]);gb=A.ndarray((_NKI_PARTITION,N),dtype=x.dtype,buffer=A.sbuf);E.dma_copy(dst=gb,src=grad_out[rows]);rb=A.ndarray((_NKI_PARTITION,N),dtype=x.dtype,buffer=A.sbuf);E.tensor_scalar(dst=rb,data=xb,op0=A.maximum,operand0=S);two_r=A.ndarray((_NKI_PARTITION,N),dtype=x.dtype,buffer=A.sbuf);E.tensor_scalar(dst=two_r,data=rb,op0=A.multiply,operand0=2.);yb=A.ndarray((_NKI_PARTITION,N),dtype=x.dtype,buffer=A.sbuf);E.tensor_tensor(dst=yb,data1=two_r,data2=gb,op=A.multiply);E.dma_copy(dst=out[rows],src=yb)
		return out
	class _ReluSquaredNKI(torch.autograd.Function):
		@N
		def forward(ctx,x):y=_relu_sq_fwd(x.reshape(-1,x.shape[-1])).reshape(x.shape);ctx.save_for_backward(x);return y
		@N
		def backward(ctx,grad_out):x,=ctx.saved_tensors;flat_x=x.reshape(-1,x.shape[-1]);flat_g=grad_out.reshape(-1,x.shape[-1]).contiguous();return _relu_sq_bwd(flat_x,flat_g).reshape(x.shape)
	_HAS_NKI=I
except P:_HAS_NKI=O
USE_NKI_RELU2=O
import os,torch,torch.nn.functional as F
D9=128
Kd=os.environ.get
Ke=I
Kf=4
Kg=B(v(32*1024))
Gn=B(Kd(AB,q))
Er=3
Kh=O
B7=O
Go=C
DA=O
Gp=C
def Es(x):return F.relu(x).square()
def Et(M,N,itemsize):
	if M<=0 or N<=0 or M%D9 or N%D9:return C
	n=M//D9;E=i(1,g(Kf,Kg//(N*itemsize),n));A=[k for k in L(1,E+1)if n%k==0];B=[k for k in A if n//k>=2 and n//k%2==0];D=[k for k in A if n//k>=2];k=i(B)if B else i(D)if D else 1;return n//k,D9,k
def B8(x2,plan):T,A,K=plan;return x2.view(T,A,K*x2.shape[-1])
try:
	import nki as K,nki.language as A,nki.isa as E
	def Ki():
		if A.program_ndim()>0:return A.program_id(0),A.num_programs(0)
		return 0,1
	def Eu(T):A,B=Ki();return L(A,T,B)
	def Bh(P,F,dtype,slot,which,nslabs):return A.ndarray((P,F),dtype=dtype,buffer=A.sbuf)
	def Ev(F):return F
	def Kj(xb,yb,rb,F):
		step=Ev(F)
		for a in L(0,F,step):B=xb[:,a:a+step];C=yb[:,a:a+step];E.scalar_tensor_tensor(dst=C,data=B,op0=A.maximum,operand0=S,op1=A.multiply,operand1=B)
	def Kk(xb,gb,ob,rb,F):
		step=Ev(F)
		for a in L(0,F,step):C=xb[:,a:a+step];D=gb[:,a:a+step];G=ob[:,a:a+step];B=rb[:,a:a+step];E.activation(dst=B,op=A.relu,data=C,scale=2.);E.tensor_tensor(dst=G,data1=B,data2=D,op=A.multiply)
	def Kl(hb,gb,ob,rb,F):
		step=Ev(F)
		for a in L(0,F,step):C=hb[:,a:a+step];D=gb[:,a:a+step];G=ob[:,a:a+step];B=rb[:,a:a+step];E.activation(dst=B,op=A.sqrt,data=C,scale=4.);E.tensor_tensor(dst=G,data1=B,data2=D,op=A.multiply)
	def Km(h3,g3,gx3,t,slot,P,F):dt=h3.dtype;B=2;A=Bh(P,F,dt,slot,0,B);gb=Bh(P,F,g3.dtype,slot,1,B);E.dma_copy(dst=A,src=h3[t,:,:]);E.dma_copy(dst=gb,src=g3[t,:,:]);rb=A;C=gb;Kl(A,gb,C,rb,F);E.dma_copy(dst=gx3[t,:,:],src=C)
	def Kn(x3,y3,t,slot,P,F):
		dt=x3.dtype;B=O;A=1;A=1
		if B:A=A+1
		xb=Bh(P,F,dt,slot,0,A);E.dma_copy(dst=xb,src=x3[t,:,:]);yb=xb
		if B:rb=Bh(P,F,dt,slot,A-1,A)
		else:rb=xb
		Kj(xb,yb,rb,F);E.dma_copy(dst=y3[t,:,:],src=yb)
	def Ko(x3,g3,gx3,t,slot,P,F):dt=x3.dtype;C=I;A=2;xb=Bh(P,F,dt,slot,0,A);gb=Bh(P,F,g3.dtype,slot,1,A);E.dma_copy(dst=xb,src=x3[t,:,:]);E.dma_copy(dst=gb,src=g3[t,:,:]);B=gb;rb=xb;Kk(xb,gb,B,rb,F);E.dma_copy(dst=gx3[t,:,:],src=B)
	@K.jit
	def Kp(x3):
		T,B,C=x3.shape;D=A.ndarray((T,B,C),dtype=x3.dtype,buffer=A.shared_hbm)
		for t in Eu(T):Kn(x3,D,t,t%Er,B,C)
		return D
	@K.jit
	def Gq(x3,g3):
		T,B,C=x3.shape;D=A.ndarray((T,B,C),dtype=x3.dtype,buffer=A.shared_hbm)
		for t in Eu(T):Ko(x3,g3,D,t,t%Er,B,C)
		return D
	@K.jit
	def Kq(h3,g3):
		T,B,C=h3.shape;D=A.ndarray((T,B,C),dtype=h3.dtype,buffer=A.shared_hbm)
		for t in Eu(T):Km(h3,g3,D,t,t%Er,B,C)
		return D
	DB=I
except P:DB=O
def Ew(where,err):
	A=where;global B7,Go
	if not B7:B7=I;Go=f"{A}: {W(err).__name__}: {err}";D(f"[nki_relu2_opt] kernel disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
def DC(kernel):A=kernel;return A[Gn]if Gn==2 else A
class Kr(torch.autograd.Function):
	@N
	def forward(ctx,x,plan):
		ctx.save_for_backward(x);ctx.plan=plan
		try:A=DC(Kp)(B8(x.reshape(-1,x.shape[-1]),plan));return A.view(x.shape)
		except P as err:
			Ew(BS,err)
			with torch.no_grad():return Es(x)
	@N
	def backward(ctx,gy):
		x,=ctx.saved_tensors;N=x.shape[-1]
		if not B7:
			try:
				g2=gy.reshape(-1,N)
				if g2.dtype!=x.dtype:g2=g2.to(x.dtype)
				if not g2.is_contiguous():g2=g2.contiguous()
				B=DC(Gq)(B8(x.reshape(-1,N),ctx.plan),B8(g2,ctx.plan));return B.view(x.shape),C
			except P as err:Ew(Am,err)
		with torch.enable_grad():A=x.detach().requires_grad_(I);gx,=torch.autograd.grad(Es(A),A,gy)
		return gx,C
def Ks(x,gy):
	if DB and not B7 and x.device.type==a and x.dtype in(torch.bfloat16,torch.float32)and x.dim()>=2:
		N=x.shape[-1];A=Et(x.numel()//N,N,x.element_size())
		if A is not C:
			try:
				B=x.contiguous().reshape(-1,N);g2=gy.reshape(-1,N)
				if g2.dtype!=x.dtype:g2=g2.to(x.dtype)
				if not g2.is_contiguous():g2=g2.contiguous()
				return DC(Gq)(B8(B,A),B8(g2,A)).view(x.shape)
			except P as err:Ew(Am,err)
	return gy*(2.*F.relu(x))
def Kt(where,err):
	A=where;global DA,Gp
	if not DA:DA=I;Gp=f"{A}: {W(err).__name__}: {err}";D(f"[nki_relu2_opt] sqrt-form kernel disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
def Ku(h,dh):r=torch.sqrt(h.to(torch.float32)*4.);r=r.to(h.dtype).to(torch.float32);return(r*dh.to(torch.float32)).to(dh.dtype)
def Kv(h,dh):
	if DB and not B7 and not DA and h.device.type==a and h.dtype in(torch.bfloat16,torch.float32)and h.dim()>=2:
		N=h.shape[-1];A=Et(h.numel()//N,N,h.element_size())
		if A is not C:
			try:
				B=h.contiguous().reshape(-1,N);g2=dh.reshape(-1,N)
				if g2.dtype!=h.dtype:g2=g2.to(h.dtype)
				if not g2.is_contiguous():g2=g2.contiguous()
				return DC(Kq)(B8(B,A),B8(g2,A)).view(h.shape)
			except P as err:Kt('backward_from_h',err)
	return Ku(h,dh)
def relu2k_relu_squared(x):
	if DB and not B7 and x.device.type==a and x.dtype in(torch.bfloat16,torch.float32)and x.dim()>=2:
		N=x.shape[-1];A=Et(x.numel()//N,N,x.element_size())
		if A is not C:
			if not x.is_contiguous():x=x.contiguous()
			return Kr.apply(x,A)
	return Es(x)
_relu_squared_opt=relu2k_relu_squared
USE_NKI_RELU2_OPT=Ke
import os,torch
H=128
X=512
Kw=os.environ.get
gemmk_USE_NKI_GEMM=I
Kx={'dma':0,'pe':1,'pre':2,'hwt':3}
Ky={An:0,Co:1,C5:2,BR:3}
Kz={'row':0,BT:1}
K_={'tiles':0,'contig':1}
L0={BT:0,AH:1}
Ex=Kx['dma']
Ey=4
Ez=1024
L1=512
E_=Ky[An]
L2=Kz['row']
Gr=K_['tiles']
Gs=1024
Gt=4
L3=L0[BT]
L4={EY:0,GL:1}
F0=L4[EY]
gemmk_USE_NKI_FC_RELU2=I
L5={'A':{BU:3,AH:2,AQ:1024,BV:2,Cp:0,Cq:0,BW:1},'B':{BU:3,AH:2,AQ:512,BV:0,Cp:0,Cq:0,BW:0},'C':{EZ:1024,AH:4,GM:0,BV:0,BW:1},'F':{BU:3,AH:2,AQ:1024,BV:0,Cq:0,BW:1}}
def F1(cls):k=AA(L5[cls]);return k
F2=B(Kw(AB,q))
DD=200*1024
DE=O
Gu=C
def CJ(a,b,trans_a=O,trans_b=O):A=a.t()if trans_a else a;B=b.t()if trans_b else b;return torch.matmul(A,B)
def Gv(M,K,N,xt=Ex,group=Ey,tb=Ez,nprg=2):
	if M%H or K%H or N%X or M<2*H or K<H or N<X:return C
	if group<1 or group>8:return C
	if tb%H or tb<H:return C
	A=K//H;D=A*N*2
	if xt==1:B=2*(A*H*2+K*2)
	else:B=2*A*tb*2
	E=2*N*2;return{'nk':A,Ea:N//X,y:D+B+E}
def Gw(M,Kp,N,tc=Gs,group=Gt,nprg=2):
	B=nprg
	if M%H or Kp%H or N%X or M<H or Kp<H or N<X*B:return C
	if tc%H or tc<H or group<1 or group>8:return C
	D=N//X;A=(D+B-1)//B;E=Kp//H;F=tc//H;G=E*A*X*4;I=2*F*(Kp+A*X)*2;J=A*X*2;return{'nkp':E,Ea:D,'ncp':A,y:G+I+J}
try:
	import nki as K,nki.language as A,nki.isa as E
	def Aa(dst,src):E.dma_copy(dst=dst,src=src)
	def F3():
		if A.program_ndim()>0:return A.program_id(0),A.num_programs(0)
		return 0,1
	def F4(n,pid,nprg):
		B=pid;C=n//nprg;D=n-C*nprg;A=B*C
		if B<D:A=A+B
		else:A=A+D
		E=C
		if B<D:E=E+1
		return A,A+E
	def CK(a,b):return(a+b-1)//b
	def Gx(n,shape,dtype,buf):
		B=buf;t0=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==1:return t0,
		t1=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==2:return t0,t1
		t2=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==3:return t0,t1,t2
		C=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==4:return t0,t1,t2,C
		D=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==5:return t0,t1,t2,C,D
		E=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==6:return t0,t1,t2,C,D,E
		F=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==7:return t0,t1,t2,C,D,E,F
		G=A.ndarray(shape,dtype=dtype,buffer=B);return t0,t1,t2,C,D,E,F,G
	def CL(dst,src,k,evac):
		D=evac;C=src;B=dst
		if D==0:
			if k%2==0:E.activation(dst=B,op=A.copy,data=C)
			else:E.tensor_copy(dst=B,src=C,engine=E.engine.vector)
		elif D==1:E.activation(dst=B,op=A.copy,data=C)
		elif D==2:E.tensor_copy(dst=B,src=C,engine=E.engine.vector)
		else:E.tensor_copy(dst=B,src=C)
	def Gy(w2,wt_mode,K,N,NK,dt):
		C=A.ndarray((H,NK,N),dtype=dt,buffer=A.sbuf)
		for B in L(NK):
			if wt_mode==1:E.dma_transpose(dst=C.ap([[NK*N,H],[1,1],[1,1],[1,N]],offset=B*N),src=w2.ap([[K,N],[1,1],[1,1],[1,H]],offset=B*H))
			else:Aa(dst=C[:,B,:],src=w2[B*H:(B+1)*H,:])
		return C
	def Gz(a2,xt_mode,K,NK,row0,ntok,dt):
		I=xt_mode;G=row0;C=NK;B=ntok;D=A.ndarray((H,C,B),dtype=dt,buffer=A.sbuf)
		if I==3:
			for s in L(B//16):E.dma_transpose(dst=D.ap([[C*B,H],[B,C],[1,1],[1,16]],offset=s*16),src=a2.ap([[K,16],[H,C],[1,1],[1,H]],offset=(G+s*16)*K),dge_mode=E.dge_mode.hwdge)
			return D
		for F in L(C):
			if I==2:Aa(dst=D[:,F,:],src=a2[F*H:(F+1)*H,G:G+B])
			else:E.dma_transpose(dst=D.ap([[C*B,H],[1,1],[1,1],[1,B]],offset=F*B),src=a2.ap([[K,B],[1,1],[1,1],[1,H]],offset=G*K+F*H))
		return D
	def G_(a2,K,NK,row0,dt,evac):
		D=A.ndarray((H,K),dtype=dt,buffer=A.sbuf);Aa(dst=D,src=a2[row0:row0+H,:]);F=A.ndarray((H,NK,H),dtype=dt,buffer=A.sbuf)
		for B in L(0,NK,4):
			C=NK-B
			if C>4:C=4
			G=A.ndarray((H,C*H),dtype=dt,buffer=A.psum)
			for j in L(C):E.nc_transpose(dst=G[:,j*H:(j+1)*H],data=D[:,(B+j)*H:(B+j+1)*H])
			CL(F[:,B:B+C,:],G,B//4,evac)
		return F
	def H0(g,psum_mode,f32):
		if psum_mode==1:return C,A.ndarray((H,g*X),dtype=f32,buffer=A.psum)
		return Gx(g,(H,X),f32,A.psum),C
	def B9(pss,pcontig,psum_mode,j):
		if psum_mode==1:return pcontig[:,j*X:(j+1)*X]
		return pss[j]
	def H1(at,kk_tok,w_sb,NK,c0,g,pss,pcontig,psum_mode):
		B=kk_tok
		for A in L(NK):
			st=at[:,A,B:B+H]
			for j in L(g):c=c0+j;E.nc_matmul(dst=B9(pss,pcontig,psum_mode,j),stationary=st,moving=w_sb[:,A,c*X:(c+1)*X],accumulate=A>0)
	def H2(at,w_sb,out,ti,bt0,N,NK,NC,npass,group,evac,out_mode,psum_mode,dt,f32):
		D=out_mode;B=psum_mode;C=(bt0+ti)*H
		if D==0:I=A.ndarray((H,N),dtype=dt,buffer=A.sbuf)
		for p in L(npass):
			E=p*group;g=NC-E
			if g>group:g=group
			F,G=H0(g,B,f32);H1(at,ti*H,w_sb,NK,E,g,F,G,B)
			for j in L(g):
				c=E+j
				if D==0:CL(I[:,c*X:(c+1)*X],B9(F,G,B,j),c,evac)
				else:J=A.ndarray((H,X),dtype=dt,buffer=A.sbuf);CL(J,B9(F,G,B,j),c,evac);Aa(dst=out[C:C+H,c*X:(c+1)*X],src=J)
		if D==0:Aa(dst=out[C:C+H,:],src=I)
	def H3(at,w_sb,h,pre,ti,bt0,N,NK,NC,npass,group,psum_mode,save_pre,dt,f32,evac):
		B=psum_mode;D=(bt0+ti)*H;J=A.ndarray((H,N),dtype=dt,buffer=A.sbuf);K=A.ndarray((H,N),dtype=dt,buffer=A.sbuf)
		for p in L(npass):
			I=p*group;g=NC-I
			if g>group:g=group
			F,G=H0(g,B,f32);H1(at,ti*H,w_sb,NK,I,g,F,G,B)
			for j in L(g):
				c=I+j;C=J[:,c*X:(c+1)*X]
				if evac==2:E.tensor_copy(dst=C,src=B9(F,G,B,j),engine=E.engine.vector)
				elif evac==3:E.tensor_copy(dst=C,src=B9(F,G,B,j))
				else:E.activation(dst=C,op=A.copy,data=B9(F,G,B,j))
				E.scalar_tensor_tensor(dst=K[:,c*X:(c+1)*X],data=C,op0=A.maximum,operand0=S,op1=A.multiply,operand1=C)
		Aa(dst=h[D:D+H,:],src=K)
		if save_pre==1:Aa(dst=pre[D:D+H,:],src=J)
	@K.jit
	def L6(a2,w2,xt_mode,wt_mode,group,tb,evac,out_mode,psum_mode,loop):
		T=psum_mode;S=out_mode;R=wt_mode;G=evac;E=xt_mode;B=a2
		if E==2:K=B.shape[0];M=B.shape[1]
		else:M=B.shape[0];K=B.shape[1]
		if R==1:N=w2.shape[0]
		else:N=w2.shape[1]
		dt=B.dtype;U=A.float32;out=A.ndarray((M,N),dtype=dt,buffer=A.shared_hbm);D=K//H;I=N//X;V=CK(I,group);Y,Z=F3();t0,t1=F4(M//H,Y,Z);W=Gy(w2,R,K,N,D,dt);F=tb//H
		if E==1:F=1
		a=CK(t1-t0,F)
		for b in L(a):
			C=t0+b*F;J=C+F
			if J>t1:J=t1
			O=J-C
			if E==1:P=G_(B,K,D,C*H,dt,G)
			else:P=Gz(B,E,K,D,C*H,O*H,dt)
			if loop==1:
				for Q in A.affine_range(O):H2(P,W,out,Q,C,N,D,I,V,group,G,S,T,dt,U)
			else:
				for Q in L(O):H2(P,W,out,Q,C,N,D,I,V,group,G,S,T,dt,U)
		return out
	@K.jit
	def L7(a2,w2,xt_mode,group,tb,evac,psum_mode,save_pre,loop):
		T=save_pre;S=psum_mode;G=evac;E=xt_mode;B=a2
		if E==2:K=B.shape[0];M=B.shape[1]
		else:M=B.shape[0];K=B.shape[1]
		N=w2.shape[1];dt=B.dtype;U=A.float32;h=A.ndarray((M,N),dtype=dt,buffer=A.shared_hbm);I=A.ndarray((M,N),dtype=dt,buffer=A.shared_hbm);D=K//H;J=N//X;V=CK(J,group);Y,Z=F3();t0,t1=F4(M//H,Y,Z);W=Gy(w2,0,K,N,D,dt);F=tb//H
		if E==1:F=1
		a=CK(t1-t0,F)
		for b in L(a):
			C=t0+b*F;O=C+F
			if O>t1:O=t1
			P=O-C
			if E==1:Q=G_(B,K,D,C*H,dt,G)
			else:Q=Gz(B,E,K,D,C*H,P*H,dt)
			if loop==1:
				for R in A.affine_range(P):H3(Q,W,h,I,R,C,N,D,J,V,group,S,T,dt,U,G)
			else:
				for R in L(P):H3(Q,W,h,I,R,C,N,D,J,V,group,S,T,dt,U,G)
		return h,I
	def DF(a2,b2,acc,ct0,n,first,Kp,NKp,NCp,Np,c0,c1,npass,group,fold_mode,evac,dt,f32):
		M=first;K=fold_mode;N=A.ndarray((H,n,Kp),dtype=dt,buffer=A.sbuf);O=A.ndarray((H,n,Np),dtype=dt,buffer=A.sbuf)
		for t in L(n):G=(ct0+t)*H;Aa(dst=N[:,t,:],src=a2[G:G+H,:]);Aa(dst=O[:,t,:],src=b2[G:G+H,c0*X:c1*X])
		for F in L(NKp):
			for p in L(npass):
				B=p*group;g=NCp-B
				if g>group:g=group
				if K==1:I=A.ndarray((H,g*X),dtype=f32,buffer=A.psum);J=C
				else:J=Gx(g,(H,X),f32,A.psum);I=C
				for t in L(n):
					st=N[:,t,F*H:(F+1)*H]
					for j in L(g):E.nc_matmul(dst=B9(J,I,K,j),stationary=st,moving=O[:,t,(B+j)*X:(B+j+1)*X],accumulate=t>0)
				if K==1:
					D=acc[:,F,B*X:(B+g)*X]
					if M==1:CL(D,I,F+p,evac)
					else:E.tensor_tensor(dst=D,data1=I,data2=D,op=A.add)
				else:
					for j in L(g):
						D=acc[:,F,(B+j)*X:(B+j+1)*X]
						if M==1:CL(D,J[j],F+j,evac)
						else:E.tensor_tensor(dst=D,data1=J[j],data2=D,op=A.add)
	@K.jit
	def L8(a2,b2,tc,group,fold_mode,evac,loop):
		T=evac;S=fold_mode;P=b2;C=a2;M=C.shape[0];D=C.shape[1];N=P.shape[1];dt=C.dtype;Q=A.float32;out=A.ndarray((D,N),dtype=dt,buffer=A.shared_hbm);U=M//H;F=D//H;c=N//X;d,e=F3();G,I=F4(c,d,e);J=I-G;K=J*X;V=CK(J,group);O=A.ndarray((H,F,K),dtype=Q,buffer=A.sbuf);B=tc//H
		if B>U:B=U
		W=U//B;Z=U-W*B
		if loop==1:
			DF(C,P,O,0,B,1,D,F,J,K,G,I,V,group,S,T,dt,Q)
			for i in A.affine_range(W-1):DF(C,P,O,(i+1)*B,B,0,D,F,J,K,G,I,V,group,S,T,dt,Q)
		else:
			for a in L(W):
				b=0
				if a==0:b=1
				DF(C,P,O,a*B,B,b,D,F,J,K,G,I,V,group,S,T,dt,Q)
		if Z>0:DF(C,P,O,W*B,Z,0,D,F,J,K,G,I,V,group,S,T,dt,Q)
		for R in L(F):
			Y=A.ndarray((H,K),dtype=dt,buffer=A.sbuf)
			if R%2==0:E.activation(dst=Y,op=A.copy,data=O[:,R,:])
			else:E.tensor_copy(dst=Y,src=O[:,R,:],engine=E.engine.vector)
			Aa(dst=out[R*H:(R+1)*H,G*X:I*X],src=Y)
		return out
	H4=I
except P:H4=O
def H5(where,err):
	A=where;global DE,Gu
	if not DE:DE=I;Gu=f"{A}: {W(err).__name__}: {err}";D(f"[nki_gemm] kernel disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
def F5(kernel):A=kernel;return A[F2]if F2==2 else A
def DG():return 2 if F2==2 else 1
def L9(a,w,trans_b=O,xt=Ex,group=Ey,tb=C,evac=E_,out_mode=L2,psum_mode=Gr,loop=F0):
	K=a.shape[0]if xt==2 else a.shape[1]
	if tb is C:tb=Ez if K<=2048 else L1
	return F5(L6)(a,w,B(xt),1 if trans_b else 0,B(group),B(tb),B(evac),B(out_mode),B(psum_mode),B(loop))
def LA(a,b,tc=Gs,group=Gt,fold=L3,evac=E_,loop=F0):return F5(L8)(a,b,B(tc),B(group),B(fold),B(evac),B(loop))
def LB(a,w,xt=Ex,group=Ey,tb=Ez,evac=E_,psum_mode=Gr,save_pre=1,loop=F0):return F5(L7)(a,w,B(xt),B(group),B(tb),B(evac),B(psum_mode),B(save_pre),B(loop))
def H6(*A):
	if not H4 or DE:return O
	for t in A:
		if t.device.type!=a or t.dtype!=torch.bfloat16 or t.dim()!=2 or not t.is_contiguous():return O
	return I
def gemmk_nki_matmul(a,b,trans_a=O,trans_b=O):
	D=trans_a;A=trans_b
	if not H6(a,b)or D and A:return CJ(a,b,D,A)
	try:
		if D:
			M,F=a.shape;J,N=b.shape
			if J!=M:return CJ(a,b,D,A)
			k=F1('C');E=k[EZ];B=Gw(M,F,N,tc=E,group=k[AH],nprg=DG())
			if B is not C and B[y]>DD and E>H:E=E//2;B=Gw(M,F,N,tc=E,group=k[AH],nprg=DG())
			if B is C or B[y]>DD:return CJ(a,b,D,A)
			return LA(a,b,tc=E,group=k[AH],fold=k[GM],evac=k[BV],loop=k[BW])
		M,K=a.shape
		if A:N,G=b.shape
		else:G,N=b.shape
		k=F1('A'if K<=2048 else'B');I=k[BU]if k[BU]!=2 else 0;B=Gv(M,K,N,xt=I,group=k[AH],tb=k[AQ],nprg=DG())if G==K else C
		if B is C or B[y]>DD:return CJ(a,b,D,A)
		return L9(a,b,trans_b=A,xt=I,group=k[AH],tb=k[AQ],evac=k[BV],out_mode=k[Cp],psum_mode=k[Cq],loop=k[BW])
	except P as err:H5('nki_matmul',err);return CJ(a,b,D,A)
def gemmk_fused_fc_relu2_forward(x2,w1,save_pre=1):
	if H6(x2,w1):
		M,K=x2.shape;F,N=w1.shape;k=F1('F');D=k[BU]if k[BU]!=2 else 0;E=Gv(M,K,N,xt=D,group=k[AH],tb=k[AQ],nprg=DG())if F==K else C
		if E is not C and E[y]+2*N*2<=DD:
			try:h,A=LB(x2,w1,xt=D,group=k[AH],tb=k[AQ],evac=k[BV],psum_mode=k[Cq],save_pre=B(save_pre),loop=k[BW]);return h,A
			except P as err:H5('fused_fc_relu2',err)
	A=torch.matmul(x2,w1);return torch.nn.functional.relu(A).square(),A
try:gemmk__relu2_backward=Ks
except P:gemmk__relu2_backward=C
def gemmk_eager_relu2_backward(pre,dh):return dh*(2.*torch.nn.functional.relu(pre))
class gemmk_FcRelu2Fn(torch.autograd.Function):
	@N
	def forward(ctx,x2d,w,inplace):h,A=gemmk_fused_fc_relu2_forward(x2d,w);ctx.save_for_backward(x2d,w,A);ctx.w=w;ctx.inplace=inplace;return h
	@N
	def backward(ctx,dh):
		x2d,w,B=ctx.saved_tensors;dh=dh.contiguous()
		if gemmk__relu2_backward is not C:A=gemmk__relu2_backward(B,dh)
		else:A=gemmk_eager_relu2_backward(B,dh)
		dx=gemmk_nki_matmul(A,w,trans_b=I)
		if ctx.inplace:ctx.w.grad.addmm_(x2d.t(),A);return dx,C,C
		return dx,gemmk_nki_matmul(x2d,A,trans_a=I),C
def gemmk_fused_fc_relu2(x,w,inplace=O):
	K=x.shape[-1];x2d=x.reshape(-1,K)
	if not x2d.is_contiguous():x2d=x2d.contiguous()
	h=gemmk_FcRelu2Fn.apply(x2d,w,inplace);return h.view(*x.shape[:-1],w.shape[1])
nki_matmul=gemmk_nki_matmul
fused_fc_relu2=gemmk_fused_fc_relu2
USE_NKI_GEMM=gemmk_USE_NKI_GEMM
USE_NKI_FC_RELU2=gemmk_USE_NKI_FC_RELU2
import os,torch
LC=os.environ.get
A2=128
F6=48
AT={'cwd':0,'w1':1,'w2':2,'b1':3,'b2':4,Cr:5,B0:6,'rb1':7,'rb2':8,'beta1':9,Eb:10}
H7=I
H8=I
H9='sub'
HA=8
LD=A1
LE={'div':0,'recip':1,'tt':2}
LF={BX:0,'ts':1,'rsqrt':2}
HB={EY:0,GL:1}
F7=2048
HC=0
HD=LE['recip']
HE=LF[BX]
HF=1
HG=HB[GL]
HH=1
DH=4096
HI=HB[EY]
F8=B(LC(AB,q))
DI=O
DJ=O
HJ=C
def LG():return f"adamw={B(H7)} gradnorm={B(H8)} shard_view={H9} stack_small={HA} T={F7} lerp={HC} div={HD} sqrt={HE} eng={HF} loop={HG} inplace={HH} normT={DH} normloop={HI} lnc={F8} emulate={LD}"
def LH(lr_cpu,beta1,beta2,eps,weight_decay,step_count):A=step_count;s=torch.zeros(16,dtype=torch.float32);lr=lr_cpu.reshape(1).to(torch.float32);D=Q-lr*weight_decay if weight_decay>0 else torch.ones(1);s[AT['cwd']]=D[0];s[AT['w1']]=torch.tensor(1-beta1,dtype=torch.float32);s[AT['w2']]=torch.tensor(1-beta2,dtype=torch.float32);B=torch.full((1,),1-beta1**A,dtype=torch.float32);C=torch.full((1,),1-beta2**A,dtype=torch.float32);s[AT['b1']]=B[0];s[AT['b2']]=C[0];s[AT[Cr]]=torch.tensor(eps,dtype=torch.float32);s[AT[B0]]=lr[0];s[AT['rb1']]=(Q/B)[0];s[AT['rb2']]=(Q/C)[0];s[AT['beta1']]=torch.tensor(beta1,dtype=torch.float32);s[AT[Eb]]=torch.tensor(beta2,dtype=torch.float32);return s
try:
	import nki as K,nki.language as A,nki.isa as E
	def AD(dst,src):E.dma_copy(dst=dst,src=src)
	def HK():
		if A.program_ndim()>0:return A.program_id(0),A.num_programs(0)
		return 0,1
	def j(shape,dtype):return A.ndarray(shape,dtype=dtype,buffer=A.sbuf)
	def Bi(sct,col,Pn,Tn):t=j((Pn,Tn),A.float32);E.memset(t,S);E.tensor_scalar(dst=t,data=t,op0=A.add,operand0=sct[0:Pn,col:col+1],engine=E.engine.vector);return t
	def DK(dp,dm,dv,sp,sg,sm,sv,sct,row,Pn,Tn,LERP,DIV,SQRT,ENG,WD,bt1,bt2):
		N=ENG;L=DIV;I=Tn;G=sct;C=Pn;J=A.float32;F=E.engine.vector;B=row*16;K=j((C,I),J);tg=j((C,I),J);M=j((C,I),J);O=j((C,I),J);AD(dst=K,src=sp);AD(dst=tg,src=sg);AD(dst=M,src=sm);AD(dst=O,src=sv);t1=j((C,I),J);t2=j((C,I),J);H=j((C,I),J);D=j((C,I),J)
		if WD==1:
			if N==1:E.tensor_scalar(dst=K,data=K,op0=A.multiply,operand0=G[0:C,B+0:B+1],engine=E.engine.scalar)
			else:E.tensor_scalar(dst=K,data=K,op0=A.multiply,operand0=G[0:C,B+0:B+1],engine=F)
		if LERP==0:E.tensor_tensor(dst=t1,data1=tg,data2=M,op=A.subtract,engine=F);E.scalar_tensor_tensor(dst=t1,data=t1,op0=A.multiply,operand0=G[0:C,B+1:B+2],op1=A.add,operand1=M)
		else:E.tensor_scalar(dst=t1,data=tg,op0=A.multiply,operand0=G[0:C,B+1:B+2],engine=F);E.scalar_tensor_tensor(dst=t1,data=M,op0=A.multiply,operand0=G[0:C,B+9:B+10],op1=A.add,operand1=t1)
		if N==1:E.activation(dst=t2,op=A.square,data=tg)
		else:E.tensor_tensor(dst=t2,data1=tg,data2=tg,op=A.multiply,engine=F)
		E.tensor_tensor(dst=t2,data1=t2,data2=O,op=A.subtract,engine=F);E.scalar_tensor_tensor(dst=t2,data=t2,op0=A.multiply,operand0=G[0:C,B+2:B+3],op1=A.add,operand1=O);AD(dst=dm,src=t1);AD(dst=dv,src=t2)
		if L==0:E.tensor_scalar(dst=H,data=t1,op0=A.divide,operand0=G[0:C,B+3:B+4],engine=F);E.tensor_scalar(dst=D,data=t2,op0=A.divide,operand0=G[0:C,B+4:B+5],engine=F)
		elif L==2:E.tensor_tensor(dst=H,data1=t1,data2=bt1,op=A.divide,engine=F);E.tensor_tensor(dst=D,data1=t2,data2=bt2,op=A.divide,engine=F)
		else:E.tensor_scalar(dst=H,data=t1,op0=A.multiply,operand0=G[0:C,B+7:B+8],engine=F);E.tensor_scalar(dst=D,data=t2,op0=A.multiply,operand0=G[0:C,B+8:B+9],engine=F)
		if SQRT==0:E.activation(dst=D,op=A.sqrt,data=D);E.tensor_scalar(dst=D,data=D,op0=A.add,operand0=G[0:C,B+5:B+6],engine=F)
		elif SQRT==1:E.tensor_scalar(dst=D,data=D,op0=A.sqrt,operand0=S,op1=A.add,operand1=G[0:C,B+5:B+6],engine=F)
		else:E.tensor_scalar(dst=D,data=D,op0=A.rsqrt,operand0=S);E.reciprocal(dst=D,data=D);E.tensor_scalar(dst=D,data=D,op0=A.add,operand0=G[0:C,B+5:B+6],engine=F)
		if L==0 or L==2:E.tensor_tensor(dst=H,data1=H,data2=D,op=A.divide,engine=F)
		else:E.reciprocal(dst=D,data=D);E.tensor_tensor(dst=H,data1=H,data2=D,op=A.multiply,engine=F)
		if N==1:E.tensor_scalar(dst=H,data=H,op0=A.multiply,operand0=G[0:C,B+6:B+7],engine=E.engine.scalar)
		else:E.tensor_scalar(dst=H,data=H,op0=A.multiply,operand0=G[0:C,B+6:B+7],engine=F)
		E.tensor_tensor(dst=K,data1=K,data2=H,op=A.subtract,engine=F);AD(dst=dp,src=K)
	def DL(po,mo,vo,pi,gi,mi,vi,sct,row,T,LERP,DIV,SQRT,ENG,LOOP,WD,pid,nprg):
		G=DIV;E=row;D=sct;F,N=pi.shape;J=N//nprg;K=J//T;M=pid*J;H=C;I=C
		if G==2:H=Bi(D,E*16+3,F,T);I=Bi(D,E*16+4,F,T)
		if LOOP==1:
			for i in A.affine_range(K):B=M+i*T;DK(po[:,B:B+T],mo[:,B:B+T],vo[:,B:B+T],pi[:,B:B+T],gi[:,B:B+T],mi[:,B:B+T],vi[:,B:B+T],D,E,F,T,LERP,G,SQRT,ENG,WD,H,I)
		else:
			for i in L(K):B=M+i*T;DK(po[:,B:B+T],mo[:,B:B+T],vo[:,B:B+T],pi[:,B:B+T],gi[:,B:B+T],mi[:,B:B+T],vi[:,B:B+T],D,E,F,T,LERP,G,SQRT,ENG,WD,H,I)
	@K.jit
	def LI(p0,g0,m0,v0,p1,g1,m1,v1,ps,gs,ms,vs,sc,T,LERP,DIV,SQRT,ENG,LOOP,INPLACE,WD0,WD1,WD2):
		P=LOOP;K=ENG;J=SQRT;I=LERP;H=p1;G=ps;F=DIV;B=A.float32;R=A2;L,Q=HK();D=j((R,F6),B);AD(dst=D,src=sc.broadcast(0,R));M=G.shape[1]
		if INPLACE==1:
			DL(p0,m0,v0,p0,g0,m0,v0,D,0,T,I,F,J,K,P,WD0,L,Q);DL(H,m1,v1,H,g1,m1,v1,D,1,T,I,F,J,K,P,WD1,L,Q)
			if L==0:
				N=C;O=C
				if F==2:N=Bi(D,2*16+3,1,M);O=Bi(D,2*16+4,1,M)
				DK(G[0:1,0:M],ms[0:1,0:M],vs[0:1,0:M],G[0:1,0:M],gs[0:1,0:M],ms[0:1,0:M],vs[0:1,0:M],D,2,1,M,I,F,J,K,WD2,N,O)
			U=A.ndarray((1,1),dtype=B,buffer=A.shared_hbm);st=j((1,1),B);E.memset(st,S);AD(dst=U,src=st);return U
		V=A.ndarray(p0.shape,dtype=B,buffer=A.shared_hbm);W=A.ndarray(p0.shape,dtype=B,buffer=A.shared_hbm);X=A.ndarray(p0.shape,dtype=B,buffer=A.shared_hbm);Y=A.ndarray(H.shape,dtype=B,buffer=A.shared_hbm);Z=A.ndarray(H.shape,dtype=B,buffer=A.shared_hbm);a=A.ndarray(H.shape,dtype=B,buffer=A.shared_hbm);b=A.ndarray(G.shape,dtype=B,buffer=A.shared_hbm);c=A.ndarray(G.shape,dtype=B,buffer=A.shared_hbm);d=A.ndarray(G.shape,dtype=B,buffer=A.shared_hbm);DL(V,W,X,p0,g0,m0,v0,D,0,T,I,F,J,K,P,WD0,L,Q);DL(Y,Z,a,H,g1,m1,v1,D,1,T,I,F,J,K,P,WD1,L,Q)
		if L==0:
			N=C;O=C
			if F==2:N=Bi(D,2*16+3,1,M);O=Bi(D,2*16+4,1,M)
			DK(b[0:1,0:M],c[0:1,0:M],d[0:1,0:M],G[0:1,0:M],gs[0:1,0:M],ms[0:1,0:M],vs[0:1,0:M],D,2,1,M,I,F,J,K,WD2,N,O)
		return V,W,X,Y,Z,a,b,c,d
	def DM(src,T,LOOP,pid,nprg,dst_col):
		B=src;F=A.float32;C,K=B.shape;I=K//nprg;G=I//T;J=pid*I;H=j((C,G),F)
		if LOOP==1:
			for i in A.affine_range(G):D=J+i*T;tb=j((C,T),B.dtype);AD(dst=tb,src=B[:,D:D+T]);sq=j((C,T),F);E.activation_reduce(dst=sq,op=A.square,data=tb,reduce_op=A.add,reduce_res=H[:,i:i+1])
		else:
			for i in L(G):D=J+i*T;tb=j((C,T),B.dtype);AD(dst=tb,src=B[:,D:D+T]);sq=j((C,T),F);E.activation_reduce(dst=sq,op=A.square,data=tb,reduce_op=A.add,reduce_res=H[:,i:i+1])
		E.tensor_reduce(dst=dst_col,op=A.add,data=H,axis=1)
	@K.jit
	def LJ(gA,gB,wt,lm,sm,TA,TW,LOOP,WITH_SMALL):
		G=LOOP;B=A.float32;D=A2;C,H=HK();out=A.ndarray((D,4),dtype=B,buffer=A.shared_hbm);I=j((D,1),B);J=j((D,1),B);K=j((D,1),B);L=j((D,1),B);DM(gA,TA,G,C,H,I);DM(gB,TA,G,C,H,J);DM(wt,TW,G,C,H,K);DM(lm,TW,G,C,H,L);F=j((D,2),B);E.tensor_tensor(dst=F[:,0:1],data1=I,data2=J,op=A.add,engine=E.engine.vector);E.tensor_tensor(dst=F[:,1:2],data1=K,data2=L,op=A.add,engine=E.engine.vector)
		if WITH_SMALL==1:
			if C==0:M=sm.shape[1];N=j((1,M),B);AD(dst=N,src=sm[0:1,0:M]);P=j((1,M),B);O=j((1,1),B);E.activation_reduce(dst=P,op=A.square,data=N,reduce_op=A.add,reduce_res=O);E.tensor_tensor(dst=F[0:1,1:2],data1=F[0:1,1:2],data2=O,op=A.add,engine=E.engine.vector)
		AD(dst=out[:,2*C:2*C+2],src=F);return out
	F9=I
except P:F9=O
def HL(kernel):A=kernel;return A[F8]if F8==2 else A
def HM(which,where,err):
	B=which;A=where;global DI,DJ,HJ;HJ=f"{A}: {W(err).__name__}: {err}"
	if B==Af and not DI:DI=I;D(f"[nki_optim] AdamW kernel disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
	if B==KF and not DJ:DJ=I;D(f"[nki_optim] grad-norm kernel disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
def LK(device):return F9 and not DI and device.type==a
def LL(device):return F9 and not DJ and device.type==a
def LM(p0,g0,m0,v0,p1,g1,m1,v1,ps,gs,ms,vs,sc,wd_flags,T=F7,lerp=HC,div=HD,sqrt=HE,eng=HF,loop=HG,inplace=HH):A=wd_flags;return HL(LI)(p0,g0,m0,v0,p1,g1,m1,v1,ps,gs,ms,vs,sc,B(T),B(lerp),B(div),B(sqrt),B(eng),B(loop),B(inplace),B(A[0]),B(A[1]),B(A[2]))
def LN(gA,gB,wt,lm,sm,T=DH,loop=HI):
	A=1 if sm is not C else 0
	if sm is C:sm=torch.zeros(1,1,dtype=torch.float32,device=gA.device)
	return HL(LJ)(gA,gB,wt,lm,sm,B(T),B(T//2),B(loop),B(A))
def LO(opt):
	B=[g for g in opt.param_groups if g[AC]==Af]
	if V(B)!=3:return C
	D=[]
	for g in B[:2]:
		E=[p for p in g[l]]
		if V(E)!=1 or g.get(BY):return C
		p=E[0]
		if p.dtype!=torch.float32 or p.dim()!=2 or p.numel()%(A2*F7*2)!=0 or not p.is_contiguous():return C
		D.append(p)
	A=B[2];stack=A.get(BY)
	if not stack or A.get(C6)is C:return C
	if any(x(p)not in A[GN]for p in A[l]):return C
	if A[C6].dtype!=torch.float32:return C
	return{KG:B,KH:D,KI:A}
@torch.no_grad()
def LP(opt):
	R='_nki_adamw_sc_dev';Q='_nki_adamw_sc_cpu';N='_nki_adamw_plan';B=opt.__dict__.get(N)
	if B is C:B=LO(opt);opt.__dict__[N]=B if B is not C else O
	if not B:return O
	D,A,F=B[KH],B[KI],B[KG];dev=D[0].device
	if not LK(dev):return O
	for p in D:
		if p.grad is C or p.grad.dtype!=torch.float32 or not p.grad.is_contiguous()or p.grad.shape!=p.shape:return O
	if not A.get(BZ):opt.stage_scalar_grads(A,distributed=O)
	st=[]
	for p in D:
		s=opt.state[p]
		if not s:s[Cs]=torch.zeros_like(p);s[Ct]=torch.zeros_like(p)
		st.append(s)
	E=opt.__dict__.get(Q)
	if E is C:E=torch.zeros(1,F6,dtype=torch.float32);opt.__dict__[Q]=E;opt.__dict__[R]=torch.zeros(1,F6,dtype=torch.float32,device=dev)
	J=opt.__dict__[R];K=[]
	for(L,g)in A7(F):G,H=g[GO];n=g.get(Ao,0)+1;E[0,16*L:16*L+16]=LH(g[Ba],G,H,g[Cr],g[B1],n);K.append(1 if g[B1]>0 else 0)
	J.copy_(E);p0,M=D;args=p0.view(A2,-1),p0.grad.view(A2,-1),st[0][Cs].view(A2,-1),st[0][Ct].view(A2,-1),M.view(A2,-1),M.grad.view(A2,-1),st[1][Cs].view(A2,-1),st[1][Ct].view(A2,-1),A[C6].view(1,-1),A[C7].view(1,-1),A[GP].view(1,-1),A[GQ].view(1,-1),J,K
	try:LM(*args)
	except P as err:
		HM(Af,'adamw3 launch',err);A[BZ]=I
		for g in F:opt._adamw_step(g)
		return I
	for p in D:torch.autograd.graph.increment_version(p)
	torch.autograd.graph.increment_version(A[C6])
	for g in F:g[Ao]=g.get(Ao,0)+1;G,H=g[GO];g[Ec].fill_(1-G**g[Ao]);g[Ed].fill_(1-H**g[Ao])
	A[BZ]=O;return I
@torch.no_grad()
def LQ(reducer,clip):
	import torch.distributed as B;distributed=B.is_available()and B.is_initialized();opt=reducer.optimizer;F=[]
	for p in reducer.params:
		if p.grad is C or x(p)in reducer.skip_ids:continue
		if distributed:B.all_reduce(p.grad,op=B.ReduceOp.AVG)
		F.append(p)
	G=[]
	for group in reducer.stack_groups:G.append(opt.stage_scalar_grads(group,distributed))
	D=[]
	for group in reducer.muon_groups:opt.stage_muon_grads(group,distributed,norm=O);D.append(opt.state[group[l][0]][Bb])
	H=[p for p in F if p.grad.dim()==2 and p.grad.dtype==torch.float32 and p.grad.is_contiguous()and p.grad.numel()%(A2*DH)==0];K=[p for p in F if not any(p is q for q in H)];A=C;E=C;L=LL(reducer.last_gnorm.device)and V(H)==2 and V(D)==2 and V(G)<=1 and all(s.dtype==torch.bfloat16 and s.is_contiguous()and s.numel()%(A2*DH*2)==0 for s in D)
	if L:
		M=G[0].view(1,-1)if G else C
		try:
			out=LN(D[0].view(A2,-1),D[1].view(A2,-1),H[0].grad.view(A2,-1),H[1].grad.view(A2,-1),M);t=out.view(A2,2,2).sum(dim=(0,1));E,A=t[0],t[1]
			for p in K:A=A+p.grad.float().square().sum()
		except P as err:HM(KF,'gradnorm launch',err);A=E=C
	if E is C:
		sq=C
		for p in F:s=p.grad.float().square().sum();sq=s if sq is C else sq+s
		for flat in G:s=flat.float().square().sum();sq=s if sq is C else sq+s
		sh=C
		for shard in D:s=shard.float().square().sum();sh=s if sh is C else sh+s
		A,E=sq,sh
	if E is not C:
		reducer.shard_sq.copy_(E)
		if distributed:B.all_reduce(reducer.shard_sq,op=B.ReduceOp.SUM)
		s=reducer.shard_sq[0];I=s if A is C else A+s
	else:I=A
	gnorm=I.sqrt();J=n(reducer,KJ,O);finite=C if J else torch.isfinite(gnorm)
	if clip>0:
		scale=torch.clamp(clip/(gnorm+1e-06),max=Q)
		for p in F:p.grad.mul_(scale)
		for group in reducer.stack_groups:group[C7].mul_(scale)
		for group in reducer.muon_groups:opt.state[group[l][0]][Bb].mul_(scale)
	if J:return gnorm
	reducer.last_gnorm.copy_(torch.where(finite,gnorm,torch.zeros_like(gnorm)));reducer.bad_steps+=(~finite).to(torch.float32);return finite
LR=I
@torch.no_grad()
def LS(opt,state,g,w,k,s,world,rank,upd_fn):
	if s!=k or not state.get(Cu):return O
	w_view=state[Ag][rank*k:(rank+1)*k];upd=upd_fn(g,w_view);torch.sub(w_view,upd if upd.dtype==w.dtype else upd.to(w.dtype),out=w);opt._all_gather(state[Ag][0:world*k],w);return I
class LT:pass
optk_API=LT()
optk_API.USE_NKI_ADAMW=H7
optk_API.USE_NKI_GRADNORM=H8
optk_API.MUON_SHARD_VIEW=H9
optk_API.ADAMW_STACK_SMALL=HA
optk_API.adamw_fused_step=LP
optk_API.reduce_fused=LQ
optk_API.REDUCE_HOST_GATE=LR
optk_API.muon_finish_view=LS
optk_API.config_string=LG
nki_optim=optk_API
class _FreezeNS:
	def __init__(self,**kw):self.__dict__.update(kw)
_fp8_bf16_ns=_FreezeNS(FcRelu2Fn=gemmk_FcRelu2Fn,nki_matmul=gemmk_nki_matmul,fused_fc_relu2_forward=gemmk_fused_fc_relu2_forward,eager_relu2_backward=gemmk_eager_relu2_backward,_relu2_backward=gemmk__relu2_backward)
import os,torch
Z=128
AU=512
Bj=256
BA=Bj//Z
LU=os.environ.get
DN=q
fp8k_USE_NKI_GEMM_FP8=DN in(q,GR)
LV=DN==GR
LW=Ee
DO=4
Bk=64
HN=GS
CM=1024
CN=2
DP=B(LU(AB,q))
LX=I
HO=200*1024
HP=24e1
LY=512
LZ=2
La={An:0,Co:1,C5:2}
Lb=La[An]
Lc={AW:0,AX:1,ER:2}
HQ=Lc[AX]
Ld=0
Le={'slab':0,'flat':1}
Lf=Le['slab']
Lg=4
Lh=64
HR=Q/(DO*Bk)
def FA(mode=C):m=HQ if mode is C else mode;return 0 if m==0 else 1
def FB(scale):
	b=torch.tensor(HP/scale,dtype=torch.float32).to(torch.bfloat16)
	for _ in L(4):
		if G(b)*scale<=HP:break
		b=torch.nextafter(b,torch.tensor(S,dtype=torch.bfloat16))
	return G(b)
CO=O
DQ=C
CP=0
HS={}
def HT(M,K,N,tb=CM,group=CN,nprg=2):
	if M%Z or K%Bj or N%AU or M<2*Z or K<Bj or N<AU:return C
	if tb%Z or tb<Z or tb%16:return C
	if group<1 or group>8:return C
	A=K//Bj;B=K//Z;D=A*BA*N;E=B*tb*2+A*BA*tb;F=2*N*2;return{'nkd':A,'nkb':B,Ea:N//AU,y:D+2*E+F}
def Li(K,N):
	if K%Z or N<1:return C
	return{y:N*2+N+N*2}
try:
	import nki as K,nki.language as A,nki.isa as E;Lj=E.matmul_perf_mode.double_row
	def HU():
		if A.program_ndim()>0:return A.program_id(0),A.num_programs(0)
		return 0,1
	def HV(n,pid,nprg):
		B=pid;C=n//nprg;D=n-C*nprg;A=B*C
		if B<D:A=A+B
		else:A=A+D
		E=C
		if B<D:E=E+1
		return A,A+E
	def HW(a,b):return(a+b-1)//b
	def Lk(n,shape,dtype,buf):
		B=buf;t0=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==1:return t0,
		t1=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==2:return t0,t1
		t2=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==3:return t0,t1,t2
		C=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==4:return t0,t1,t2,C
		D=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==5:return t0,t1,t2,C,D
		E=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==6:return t0,t1,t2,C,D,E
		F=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==7:return t0,t1,t2,C,D,E,F
		G=A.ndarray(shape,dtype=dtype,buffer=B);return t0,t1,t2,C,D,E,F,G
	def Ll(a2,K,NKB,row0,ntok,dt):
		C=ntok;B=NKB;D=A.ndarray((Z,B,C),dtype=dt,buffer=A.sbuf)
		for s in L(C//16):E.dma_transpose(dst=D.ap([[B*C,Z],[C,B],[1,1],[1,16]],offset=s*16),src=a2.ap([[K,16],[Z,B],[1,1],[1,Z]],offset=(row0+s*16)*K),dge_mode=E.dge_mode.hwdge)
		return D
	def Lm(at,NKD,ntok,a_scale):
		B=A.ndarray((Z,NKD,BA,ntok),dtype=A.float8_e4m3,buffer=A.sbuf)
		for kd in L(NKD):
			for j in L(BA):E.activation(dst=B[:,kd,j,:],op=A.copy,data=at[:,kd*BA+j,:],scale=a_scale)
		return B
	def Ln(w_u8,K,N,NKD):
		B=A.ndarray((Z,NKD,BA,N),dtype=A.uint8,buffer=A.sbuf)
		for kd in L(NKD):
			for j in L(BA):C=kd*Bj+j*Z;E.dma_copy(dst=B[:,kd,j,:],src=w_u8[C:C+Z,:])
		return B.view(A.float8_e4m3)
	def Lo(a_f8,w_f8,h,pre,ti,bt0,N,NKD,NC,npass,group,inv_scale,dt,f32,save_pre):
		B=(bt0+ti)*Z;F=A.ndarray((Z,N),dtype=dt,buffer=A.sbuf);G=A.ndarray((Z,N),dtype=dt,buffer=A.sbuf)
		for p in L(npass):
			C=p*group;g=NC-C
			if g>group:g=group
			H=Lk(g,(Z,AU),f32,A.psum)
			for kd in L(NKD):
				st=a_f8[:,kd,:,ti*Z:(ti+1)*Z]
				for j in L(g):c=C+j;E.nc_matmul(dst=H[j],stationary=st,moving=w_f8[:,kd,:,c*AU:(c+1)*AU],accumulate=kd>0,perf_mode=Lj)
			for j in L(g):c=C+j;D=F[:,c*AU:(c+1)*AU];E.activation(dst=D,op=A.copy,data=H[j],scale=inv_scale);E.scalar_tensor_tensor(dst=G[:,c*AU:(c+1)*AU],data=D,op0=A.maximum,operand0=S,op1=A.multiply,operand1=D)
		E.dma_copy(dst=h[B:B+Z,:],src=G)
		if save_pre==1:E.dma_copy(dst=pre[B:B+Z,:],src=F)
	@K.jit
	def Lp(a2,w_u8,tb,group,a_scale,inv_scale,clip,a_bound,save_pre):
		M=a2.shape[0];K=a2.shape[1];N=w_u8.shape[1];dt=a2.dtype;O=A.float32;h=A.ndarray((M,N),dtype=dt,buffer=A.shared_hbm);G=A.ndarray((M,N),dtype=dt,buffer=A.shared_hbm);C=K//Bj;H=K//Z;I=N//AU;P=HW(I,group);Q,R=HU();t0,t1=HV(M//Z,Q,R);S=Ln(w_u8,K,N,C);D=tb//Z;T=HW(t1-t0,D)
		for b in L(T):
			B=t0+b*D;E=B+D
			if E>t1:E=t1
			F=E-B;J=Ll(a2,K,H,B*Z,F*Z,dt)
			if clip==1:Lq(J,H,a_bound)
			U=Lm(J,C,F*Z,a_scale)
			for V in A.affine_range(F):Lo(U,S,h,G,V,B,N,C,I,P,group,inv_scale,dt,O,save_pre)
		return h,G
	def Lq(at,NKB,bound):
		B=bound
		for C in L(NKB):E.tensor_scalar(dst=at[:,C,:],data=at[:,C,:],op0=A.minimum,operand0=B,op1=A.maximum,operand1=-B,engine=E.engine.vector)
	def Lr(stage,bound):C=bound;B=stage;E.tensor_scalar(dst=B,data=B,op0=A.minimum,operand0=C,op1=A.maximum,operand1=-C,engine=E.engine.vector)
	@K.jit
	def HX(w,scale,clip,bound):
		K=w.shape[0];N=w.shape[1];out=A.ndarray((K,N),dtype=A.uint8,buffer=A.shared_hbm);G,H=HU();D,I=HV(K//Z,G,H)
		for i in L(I-D):
			B=(D+i)*Z;C=A.ndarray((Z,N),dtype=w.dtype,buffer=A.sbuf);E.dma_copy(dst=C,src=w[B:B+Z,:])
			if clip==1:Lr(C,bound)
			F=A.ndarray((Z,N),dtype=A.uint8,buffer=A.sbuf);E.activation(dst=F.view(A.float8_e4m3),op=A.copy,data=C,scale=scale);E.dma_copy(dst=out[B:B+Z,:],src=F)
		return out
	FC=I
except P:FC=O
try:CQ=_fp8_bf16_ns
except P:CQ=C
def HY(where,err):
	A=where;global CO,DQ
	if not CO:CO=I;DQ=f"{A}: {W(err).__name__}: {err}";D(f"[nki_gemm_fp8] kernel disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
def FD(kernel):A=kernel;return A[DP]if DP==2 else A
def HZ():return 2 if DP==2 else 1
def Ls(training):return z(training)
def FE(*A):
	if not FC or CO:return O
	for t in A:
		if t.device.type!=a or t.dtype!=torch.bfloat16 or t.dim()!=2 or not t.is_contiguous():return O
	return I
def Lt():global CP;CP+=1
def Ha(w,tag):
	A=HS.get((x(w),tag))
	if A is C:return C
	key,B=A
	if key==(CP,w.data_ptr(),w._version):return B
	return C
def Hb(w,tag,val):HS[x(w),tag]=(CP,w.data_ptr(),w._version),val
def Lu(w,scale=Bk,trans=O):
	A=trans;tag=f"f8@{scale:g}{"T"if A else Y}";B=Ha(w,tag)
	if B is not C:return B
	if not FE(w):return C
	D=w.detach().t().contiguous()if A else w.detach();K,N=D.shape
	if K%Z or Li(K,N)is C:return C
	try:wq=FD(HX)(D,G(scale),FA(),FB(scale))
	except P as err:HY('quant_w',err);return C
	Hb(w,tag,wq);return wq
def DR(x2d,w,save_pre=1):
	A=save_pre
	if FE(x2d,w):
		M,K=x2d.shape;E,N=w.shape;D=HT(M,K,N,tb=CM,group=CN,nprg=HZ())if E==K else C
		if D is not C and D[y]<=HO:
			wq=Lu(w)
			if wq is not C:
				try:return FD(Lp)(x2d,wq,B(CM),B(CN),G(DO),G(HR),FA(),FB(DO),B(A))
				except P as err:HY('fc_relu2_fp8',err)
	return Lv(x2d,w,A)
def Lv(x2d,w,save_pre=1):
	if CQ is not C:return CQ.fused_fc_relu2_forward(x2d,w,save_pre=save_pre)
	A=torch.matmul(x2d,w);return torch.nn.functional.relu(A).square(),A
def Lw(ctx,dh):
	x2d,w,B=ctx.saved_tensors;dh=dh.contiguous();A=dh*(2.*torch.nn.functional.relu(B));dx=torch.matmul(A,w.t())
	if ctx.inplace:ctx.w.grad.addmm_(x2d.t(),A);return dx,C,C
	return dx,torch.matmul(x2d.t(),A),C
Hc=Lw
if CQ is not C:Hc=CQ.FcRelu2Fn.backward
class Lx(torch.autograd.Function):
	@N
	def forward(ctx,x2d,w,inplace):h,A=DR(x2d,w);ctx.save_for_backward(x2d,w,A);ctx.w=w;ctx.inplace=inplace;return h
	backward=N(Hc)
def Ly(x,w,inplace=O):
	K=x.shape[-1];x2d=x.reshape(-1,K)
	if not x2d.is_contiguous():x2d=x2d.contiguous()
	h=Lx.apply(x2d,w,inplace);return h.view(*x.shape[:-1],w.shape[1])
def Lz():return f"mode={DN!r} eval={LW} a_scale={DO:g} w_scale={Bk:g} inv={HR:.3e} wcache={HN} tb={CM} group={CN} aalloc={"direct"} lnc={DP} shard={LX} tb_b={LY} group_b={LZ} evac_b={Lb} clip={HQ} loop_b={Ld} conv_b={Lf} a_scale_b={Lg:g} w_scale_b={Lh:g} nki={FC} disabled={CO}"+(f" err={DQ}"if DQ else Y)
class L_:0
fp8k_API=L_()
fp8k_API.begin_step=Lt
fp8k_API.active=Ls
fp8k_API.config_string=Lz
fp8k_API.fused_fc_relu2_fp8=Ly
fp8k_API.USE_NKI_GEMM_FP8=fp8k_USE_NKI_GEMM_FP8
fp8k_API.MODE=DN
nki_gemm_fp8=fp8k_API
USE_NKI_GEMM_FP8=fp8k_USE_NKI_GEMM_FP8
_fp8_slice1_ns=object()
import math,os,torch
R=128
A3=512
BB=256
BC=BB//R
M0=24e1
M1=57344.
M2=os.environ.get
FF=q
M3='2,3,4'
Bl=e(AO({B(t)for t in M3.replace(' ',Y).split(AG)if t in('2','3','4')}))
Hd=V(Bl)>0
M4=FF==GR
M5=Ee
He=q
M6='5,6'
DS=e(AO({B(t)for t in M6.replace(' ',Y).split(AG)if t in('5','6')}))
FG=.0078125
M7=V(DS)>0
M8=Ee
M9='1073741824'
CR=G(M9)
MA=2048
MB=1e-06
FH=128
DT=16
Hf=16
MC={C8:0,B2:1}
DU=MC.get(B2,1)
MD=GS
Hg='copy'
Hh=1024
Hi=512
DV=2
ME={An:0,Co:1,C5:2}
FI=ME.get(C5,1)
FJ=0
FK=B(M2(AB,q))
MF=I
Hj=200*1024
Ap=[]
Ah=not Ap
Aq=not Ap
if Ap:D('[nki_gemm_fp8_nn] IGNORING R1_RELU2_FROM_H/R1_RELU2_FOLD_DX -- it would be a silent no-op: '+'; '.join(Ap),flush=I)
Hk,Hl,Hm=0,1,2
CS=O
DW=C
CT=0
Hn={}
DX={}
def DY(fmt):return M1 if fmt==1 else M0
def Ar(scale,fmt=0):
	A=DY(fmt);b=torch.tensor(A/scale,dtype=torch.float32).to(torch.bfloat16)
	for _ in L(4):
		if G(b)*scale<=A:break
		b=torch.nextafter(b,torch.tensor(S,dtype=torch.bfloat16))
	return G(b)
def MG(amax,fmt):
	A=amax
	if not A>S or not math.isfinite(A):return Q
	s=2.**math.floor(math.log2(g(MA,DY(fmt))/A));return g(i(s,2.**-16),2.**40)
def Ho(signed,scale):
	if not signed:return Hl
	if scale==Q:return Hm
	return Hk
BD={2:{AR:0,Bc:FG,Cv:O,B3:DT,Ef:I,Bd:'c_proj fwd  h @ W2^T'},3:{AR:DU,Bc:CR,Cv:I,B3:FH,Ef:I,Bd:'c_fc   dX   dpre @ W1^T'},4:{AR:DU,Bc:CR,Cv:I,B3:DT,Ef:O,Bd:'c_proj dX   dy @ W2'},5:{GT:I,GU:'b',AR:0,Bc:Hf,GV:I,Eg:DU,GW:CR,GX:I,Bd:'c_fc   dW   x^T @ dpre'},6:{GT:I,GU:'a',AR:DU,Bc:CR,GV:I,Eg:0,GW:FG,GX:O,Bd:'c_proj dW   dy^T @ h'}}
FL,fp8nnk_ROW3,fp8nnk_ROW4=2 in Bl,3 in Bl,4 in Bl
FM=Ar(BD[2][Bc],BD[2][AR])if FL else C
FN=0
def Hp(h):
	global FN
	if FM is C:return h
	FN+=1;return h.clamp(max=FM)
MH=5 in DS
def MI(K):return Hh if K<=2048 else Hi
def Hq(M,K,N,tb=C,group=DV,nprg=2,fold=0):
	if tb is C:tb=MI(K)
	if M%R or K%BB or N%A3 or M<2*R or K<BB or N<A3:return C
	if tb%R or tb<R or tb%16:return C
	if group<1 or group>8:return C
	A=K//BB;B=K//R;D=A*BC*N;E=B*tb*2+A*BC*tb;F=2*N*2;G=2*N*2 if fold else 0;return{'nkd':A,'nkb':B,Ea:N//A3,AQ:tb,y:D+2*E+F+G}
def FO(K,N):
	if K%R or N<1:return C
	return{y:N*2+N+N*2}
try:
	import nki as K,nki.language as A,nki.isa as E;Hr=E.matmul_perf_mode.double_row
	def Hs(fmt):
		if fmt==1:return A.float8_e5m2
		return A.float8_e4m3
	def Bm():
		if A.program_ndim()>0:return A.program_id(0),A.num_programs(0)
		return 0,1
	def Bn(n,pid,nprg):
		B=pid;C=n//nprg;D=n-C*nprg;A=B*C
		if B<D:A=A+B
		else:A=A+D
		E=C
		if B<D:E=E+1
		return A,A+E
	def DZ(a,b):return(a+b-1)//b
	def Ht(n,shape,dtype,buf):
		B=buf;t0=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==1:return t0,
		t1=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==2:return t0,t1
		t2=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==3:return t0,t1,t2
		C=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==4:return t0,t1,t2,C
		D=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==5:return t0,t1,t2,C,D
		E=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==6:return t0,t1,t2,C,D,E
		F=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==7:return t0,t1,t2,C,D,E,F
		G=A.ndarray(shape,dtype=dtype,buffer=B);return t0,t1,t2,C,D,E,F,G
	def FP(a2,K,NKB,row0,ntok,dt):
		C=ntok;B=NKB;D=A.ndarray((R,B,C),dtype=dt,buffer=A.sbuf)
		for s in L(C//16):E.dma_transpose(dst=D.ap([[B*C,R],[C,B],[1,1],[1,16]],offset=s*16),src=a2.ap([[K,16],[R,B],[1,1],[1,R]],offset=(row0+s*16)*K),dge_mode=E.dge_mode.hwdge)
		return D
	def Hu(a2t,K,NKB,row0,ntok,dt):
		C=A.ndarray((R,NKB,ntok),dtype=dt,buffer=A.sbuf)
		for B in L(NKB):E.dma_copy(dst=C[:,B,:],src=a2t[B*R:(B+1)*R,row0:row0+ntok])
		return C
	def Hv(at,NKD,ntok,scale,aq,bound,hi,fmt):
		F=bound;B=at;D=A.ndarray((R,NKD,BC,ntok),dtype=Hs(fmt),buffer=A.sbuf)
		for kd in L(NKD):
			for j in L(BC):
				C=kd*BC+j
				if aq==1:E.tensor_scalar(dst=D[:,kd,j,:],data=B[:,C,:],op0=A.multiply,operand0=scale,op1=A.minimum,operand1=hi,engine=E.engine.vector)
				elif aq==2:E.tensor_scalar(dst=D[:,kd,j,:],data=B[:,C,:],op0=A.minimum,operand0=hi,op1=A.maximum,operand1=-hi,engine=E.engine.vector)
				else:E.tensor_scalar(dst=B[:,C,:],data=B[:,C,:],op0=A.minimum,operand0=F,op1=A.maximum,operand1=-F,engine=E.engine.vector);E.activation(dst=D[:,kd,j,:],op=A.copy,data=B[:,C,:],scale=scale)
		return D
	def Hw(w_u8,K,N,NKD,NC):
		B=A.ndarray((R,NKD*NC,BC,A3),dtype=A.uint8,buffer=A.sbuf)
		for kd in L(NKD):
			for c in L(NC):
				for j in L(BC):C=kd*BB+j*R;E.dma_copy(dst=B[:,kd*NC+c,j,:],src=w_u8[C:C+R,c*A3:(c+1)*A3])
		return B.view(A.float8_e4m3)
	def Hx(dst,src,k,evac,inv_scale):
		D=inv_scale;C=src;B=dst
		if evac==1:E.activation(dst=B,op=A.copy,data=C,scale=D)
		elif evac==2:E.tensor_scalar(dst=B,data=C,op0=A.multiply,operand0=D,engine=E.engine.vector)
		elif k%2==0:E.activation(dst=B,op=A.copy,data=C,scale=D)
		else:E.tensor_scalar(dst=B,data=C,op0=A.multiply,operand0=D,engine=E.engine.vector)
	def Hy(a_f8,w_f8,out,ti,bt0,N,NKD,NC,npass,group,evac,inv_scale,dt,f32):
		C=(bt0+ti)*R;D=A.ndarray((R,N),dtype=dt,buffer=A.sbuf)
		for p in L(npass):
			B=p*group;g=NC-B
			if g>group:g=group
			F=Ht(g,(R,A3),f32,A.psum)
			for kd in L(NKD):
				st=a_f8[:,kd,:,ti*R:(ti+1)*R]
				for j in L(g):E.nc_matmul(dst=F[j],stationary=st,moving=w_f8[:,kd*NC+B+j,:,:],accumulate=kd>0,perf_mode=Hr)
			for j in L(g):c=B+j;Hx(D[:,c*A3:(c+1)*A3],F[j],c,evac,inv_scale)
		E.dma_copy(dst=out[C:C+R,:],src=D)
	@K.jit
	def MJ(a2,w_u8,xt_mode,tb,group,evac,aq,loop,afmt,a_scale,a_bound,a_hi,inv_scale):
		O=inv_scale;J=xt_mode;B=a2
		if J==2:K=B.shape[0];M=B.shape[1]
		else:M=B.shape[0];K=B.shape[1]
		N=w_u8.shape[1];dt=B.dtype;P=A.float32;out=A.ndarray((M,N),dtype=dt,buffer=A.shared_hbm);E=K//BB;Q=K//R;F=N//A3;S=DZ(F,group);W,X=Bm();t0,t1=Bn(M//R,W,X);T=Hw(w_u8,K,N,E,F);G=tb//R;Y=DZ(t1-t0,G)
		for b in L(Y):
			C=t0+b*G;H=C+G
			if H>t1:H=t1
			D=H-C
			if J==2:U=Hu(B,K,Q,C*R,D*R,dt)
			else:U=FP(B,K,Q,C*R,D*R,dt)
			V=Hv(U,E,D*R,a_scale,aq,a_bound,a_hi,afmt)
			if loop==1:
				for I in A.affine_range(D):Hy(V,T,out,I,C,N,E,F,S,group,evac,O,dt,P)
			else:
				for I in L(D):Hy(V,T,out,I,C,N,E,F,S,group,evac,O,dt,P)
		return out
	def Hz(a_f8,w_f8,h2,out,ti,bt0,N,NKD,NC,npass,group,evac,inv_scale,dt,f32):
		B=(bt0+ti)*R;C=A.ndarray((R,N),dtype=dt,buffer=A.sbuf)
		for p in L(npass):
			D=p*group;g=NC-D
			if g>group:g=group
			G=Ht(g,(R,A3),f32,A.psum)
			for kd in L(NKD):
				st=a_f8[:,kd,:,ti*R:(ti+1)*R]
				for j in L(g):E.nc_matmul(dst=G[j],stationary=st,moving=w_f8[:,kd*NC+D+j,:,:],accumulate=kd>0,perf_mode=Hr)
			for j in L(g):c=D+j;Hx(C[:,c*A3:(c+1)*A3],G[j],c,evac,inv_scale)
		H=A.ndarray((R,N),dtype=dt,buffer=A.sbuf);E.dma_copy(dst=H,src=h2[B:B+R,:])
		for c in L(NC):F=H[:,c*A3:(c+1)*A3];I=C[:,c*A3:(c+1)*A3];E.activation(dst=F,op=A.sqrt,data=F,scale=4.);E.tensor_tensor(dst=I,data1=F,data2=I,op=A.multiply)
		E.dma_copy(dst=out[B:B+R,:],src=C)
	@K.jit
	def MK(a2,w_u8,h2,xt_mode,tb,group,evac,aq,loop,afmt,a_scale,a_bound,a_hi,inv_scale):
		O=inv_scale;J=xt_mode;B=a2
		if J==2:K=B.shape[0];M=B.shape[1]
		else:M=B.shape[0];K=B.shape[1]
		N=w_u8.shape[1];dt=B.dtype;P=A.float32;out=A.ndarray((M,N),dtype=dt,buffer=A.shared_hbm);E=K//BB;Q=K//R;F=N//A3;S=DZ(F,group);W,X=Bm();t0,t1=Bn(M//R,W,X);T=Hw(w_u8,K,N,E,F);G=tb//R;Y=DZ(t1-t0,G)
		for b in L(Y):
			C=t0+b*G;H=C+G
			if H>t1:H=t1
			D=H-C
			if J==2:U=Hu(B,K,Q,C*R,D*R,dt)
			else:U=FP(B,K,Q,C*R,D*R,dt)
			V=Hv(U,E,D*R,a_scale,aq,a_bound,a_hi,afmt)
			if loop==1:
				for I in A.affine_range(D):Hz(V,T,h2,out,I,C,N,E,F,S,group,evac,O,dt,P)
			else:
				for I in L(D):Hz(V,T,h2,out,I,C,N,E,F,S,group,evac,O,dt,P)
		return out
	@K.jit
	def H_(w,scale,clip,bound):
		D=bound;K=w.shape[0];N=w.shape[1];out=A.ndarray((K,N),dtype=A.uint8,buffer=A.shared_hbm);H,I=Bm();F,J=Bn(K//R,H,I)
		for i in L(J-F):
			B=(F+i)*R;C=A.ndarray((R,N),dtype=w.dtype,buffer=A.sbuf);E.dma_copy(dst=C,src=w[B:B+R,:])
			if clip==1:E.tensor_scalar(dst=C,data=C,op0=A.minimum,operand0=D,op1=A.maximum,operand1=-D,engine=E.engine.vector)
			G=A.ndarray((R,N),dtype=A.uint8,buffer=A.sbuf);E.activation(dst=G.view(A.float8_e4m3),op=A.copy,data=C,scale=scale);E.dma_copy(dst=out[B:B+R,:],src=G)
		return out
	@K.jit
	def ML(w,scale,clip,bound):
		I=bound;J=w.shape[0];C=w.shape[1];out=A.ndarray((C,J),dtype=A.uint8,buffer=A.shared_hbm);D=C//R;M,N=Bm();K,O=Bn(J//16,M,N);F=K*16;G=(O-K)*16;H=FP(w,C,D,F,G,w.dtype);q=A.ndarray((R,D,G),dtype=A.uint8,buffer=A.sbuf);P=q.view(A.float8_e4m3)
		for B in L(D):
			if clip==1:E.tensor_scalar(dst=H[:,B,:],data=H[:,B,:],op0=A.minimum,operand0=I,op1=A.maximum,operand1=-I,engine=E.engine.vector)
			E.activation(dst=P[:,B,:],op=A.copy,data=H[:,B,:],scale=scale);E.dma_copy(dst=out[B*R:(B+1)*R,F:F+G],src=q[:,B,:])
		return out
	@K.jit
	def MM(w2,n,K,dr,scale,clip,bound):
		D=bound;N=w2.shape[1];out=A.ndarray((n*K,N),dtype=A.uint8,buffer=A.shared_hbm);C=K//R;M,O=Bm();F,P=Bn(n*C,M,O)
		for i in L(P-F):
			s=F+i;m=s//C;G=(s-m*C)*R;H=m*dr+G;I=m*K+G;B=A.ndarray((R,N),dtype=w2.dtype,buffer=A.sbuf);E.dma_copy(dst=B,src=w2[H:H+R,:])
			if clip==1:E.tensor_scalar(dst=B,data=B,op0=A.minimum,operand0=D,op1=A.maximum,operand1=-D,engine=E.engine.vector)
			J=A.ndarray((R,N),dtype=A.uint8,buffer=A.sbuf);E.activation(dst=J.view(A.float8_e4m3),op=A.copy,data=B,scale=scale);E.dma_copy(dst=out[I:I+R,:],src=J)
		return out
	@K.jit
	def I0(x,scale,inv,clip,bound,fmt):
		F=bound;G=x.shape[0];B=x.shape[1];dt=x.dtype;out=A.ndarray((G,B),dtype=dt,buffer=A.shared_hbm);K,M=Bm();H,N=Bn(G//R,K,M)
		for i in L(N-H):
			C=(H+i)*R;D=A.ndarray((R,B),dtype=dt,buffer=A.sbuf);E.dma_copy(dst=D,src=x[C:C+R,:])
			if clip==1:E.tensor_scalar(dst=D,data=D,op0=A.minimum,operand0=F,op1=A.maximum,operand1=-F,engine=E.engine.vector)
			O=A.ndarray((R,B),dtype=A.uint8,buffer=A.sbuf);I=O.view(Hs(fmt));E.activation(dst=I,op=A.copy,data=D,scale=scale);J=A.ndarray((R,B),dtype=dt,buffer=A.sbuf);E.activation(dst=J,op=A.copy,data=I,scale=inv);E.dma_copy(dst=out[C:C+R,:],src=J)
		return out
	FQ=I
except P:FQ=O
try:A6=_fp8_bf16_ns
except P:A6=C
_fp8_tn_ns=object()
_tn=_fp8_tn_ns
import os,torch
k=128
s=512
AI=256
Ab=AI//k
MN={0:24e1,1:57344.}
MO=os.environ.get
FR=1024
Da=4
MP={BT:0,AH:1}
I1=MP[BT]
MQ={An:0,Co:1,C5:2}
I2=MQ[An]
I3=1
MR={BR:2,AV:0,'k':1}
MS=MR[BR]
MT={BT:0,'flat':1}
I4=MT[BT]
MU={AX:0,Co:1,GY:2,An:3}
I5=MU[AX]
I6={C8:0,B2:1}
I7=I6[C8]
I8=I6[B2]
I9={Eh:0,GZ:1,GY:2}
IA=I9[Eh]
IB=I9[Eh]
IC=1
ID=1
FS=B(MO(AB,q))
IE=200*1024
CU=O
IF=C
def IG(qmode,scale):return Q if qmode==0 else G(scale)
def IH(fmt,scale):
	A=MN[fmt];b=torch.tensor(A/scale,dtype=torch.float32).to(torch.bfloat16)
	for _ in L(4):
		if G(b)*scale<=A:break
		b=torch.nextafter(b,torch.tensor(S,dtype=torch.bfloat16))
	return G(b)
def FT(Kp,N,axis=C):
	a=MS if axis is C else axis
	if a!=2:return a
	return 0 if N>Kp else 1
def II(M,Kp,N,tc=C,group=Da,nprg=2,axis=C,qbuf=2):
	H=qbuf;E=nprg;A=tc
	if A is C:A=FR
	if M%AI or Kp%k or N%s:return C
	if M<AI or Kp<k or N<s:return C
	if A%AI or A<AI or A>M:return C
	if group<1 or group>8:return C
	sh=FT(Kp,N,axis);I=Kp//k;J=N//s
	if sh==1:B=(I+E-1)//E;F=J
	else:B=I;F=(J+E-1)//E
	if B<1 or F<1:return C
	K=B*k;D=F*s;G=A//k;L=B*D*4;O=H*G*K;P=H*G*D;Q=H*Ab*(K+D)*2;R=D*2;return{'shard':sh,'NKp':I,'NC':J,'NKp_p':B,'NCp':F,'Kp_p':K,'Np':D,EZ:A,'tct':G,'nd':G//Ab,'folds':M//A,'acc':L,'af8':O,'bf8':P,Cl:Q,Cp:R,y:L+O+P+Q+R}
def IJ(M,Kp,N,group=Da,nprg=2,axis=C):
	A=FR
	while A>=AI:
		B=II(M,Kp,N,tc=A,group=group,nprg=nprg,axis=axis)
		if B is not C and B[y]<=IE and M%A==0:return A
		A=A//2
	return C
try:
	import nki as K,nki.language as A,nki.isa as E;MV=E.matmul_perf_mode.double_row
	def IK(fmt):
		if fmt==1:return A.float8_e5m2
		return A.float8_e4m3
	def As(which):
		if which==1:return E.engine.scalar
		return E.engine.vector
	def MW():
		if A.program_ndim()>0:return A.program_id(0),A.num_programs(0)
		return 0,1
	def IL(n,pid,nprg):
		B=pid;C=n//nprg;D=n-C*nprg;A=B*C
		if B<D:A=A+B
		else:A=A+D
		E=C
		if B<D:E=E+1
		return A,A+E
	def MX(a,b):return(a+b-1)//b
	def MY(n,shape,dtype,buf):
		B=buf;t0=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==1:return t0,
		t1=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==2:return t0,t1
		t2=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==3:return t0,t1,t2
		C=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==4:return t0,t1,t2,C
		D=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==5:return t0,t1,t2,C,D
		E=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==6:return t0,t1,t2,C,D,E
		F=A.ndarray(shape,dtype=dtype,buffer=B)
		if n==7:return t0,t1,t2,C,D,E,F
		G=A.ndarray(shape,dtype=dtype,buffer=B);return t0,t1,t2,C,D,E,F,G
	def MZ(pss,pcontig,fold_mode,j):
		if fold_mode==1:return pcontig[:,j*s:(j+1)*s]
		return pss[j]
	def IM(dst,src,k,evac):
		C=src;B=dst
		if evac==1:E.activation(dst=B,op=A.copy,data=C)
		elif evac==2:E.tensor_copy(dst=B,src=C,engine=E.engine.vector)
		elif k%2==0:E.activation(dst=B,op=A.copy,data=C)
		else:E.tensor_copy(dst=B,src=C,engine=E.engine.vector)
	def FU(dst,src,qmode,scale,fmax,bound,eng):
		G=bound;F=qmode;D=fmax;C=dst;B=src
		if F==0:E.tensor_scalar(dst=C,data=B,op0=A.minimum,operand0=D,op1=A.maximum,operand1=-D,engine=eng)
		elif F==1:E.tensor_scalar(dst=C,data=B,op0=A.multiply,operand0=scale,op1=A.minimum,operand1=D,engine=eng)
		else:E.tensor_scalar(dst=B,data=B,op0=A.minimum,operand0=G,op1=A.maximum,operand1=-G,engine=E.engine.vector);E.activation(dst=C,op=A.copy,data=B,scale=scale)
	def Ma(a2,b2,a_f8,b_f8,d,row0,k_off,n_off,Kp_p,Np,NCp,dt,aq,bq,a_scale,b_scale,a_fmax,b_fmax,a_bound,b_bound,aeng,beng,bmv):
		I=b_bound;H=b_fmax;G=b_scale;F=n_off;D=k_off;J=A.ndarray((k,Ab,Kp_p),dtype=dt,buffer=A.sbuf)
		for j in L(Ab):B=row0+j*k;E.dma_copy(dst=J[:,j,:],src=a2[B:B+k,D:D+Kp_p])
		FU(a_f8[:,d,:,:],J,aq,a_scale,a_fmax,a_bound,aeng);C=A.ndarray((k,Ab,Np),dtype=dt,buffer=A.sbuf)
		for j in L(Ab):B=row0+j*k;E.dma_copy(dst=C[:,j,:],src=b2[B:B+k,F:F+Np])
		if bmv==0:
			for c in L(NCp):
				for j in L(Ab):FU(b_f8[:,d*NCp+c,j,:],C[:,j,c*s:(c+1)*s],bq,G,H,I,beng)
		else:FU(b_f8[:,d,:,:],C,bq,G,H,I,beng)
	def Mb(a_f8,b_f8,acc,nd,first,NKp_p,NCp,npass,group,fold_mode,evac,bmv,f32):
		J=first;I=fold_mode
		for F in L(NKp_p):
			for p in L(npass):
				B=p*group;g=NCp-B
				if g>group:g=group
				if I==1:G=A.ndarray((k,g*s),dtype=f32,buffer=A.psum);H=C
				else:H=MY(g,(k,s),f32,A.psum);G=C
				for d in L(nd):
					st=a_f8[:,d,:,F*k:(F+1)*k]
					for j in L(g):
						if bmv==0:K=b_f8[:,d*NCp+B+j,:,:]
						else:K=b_f8[:,d,:,(B+j)*s:(B+j+1)*s]
						E.nc_matmul(dst=MZ(H,G,I,j),stationary=st,moving=K,accumulate=d>0,perf_mode=MV)
				if I==1:
					D=acc[:,F,B*s:(B+g)*s]
					if J==1:IM(D,G,F+p,evac)
					else:E.tensor_tensor(dst=D,data1=G,data2=D,op=A.add)
				else:
					for j in L(g):
						D=acc[:,F,(B+j)*s:(B+j+1)*s]
						if J==1:IM(D,H[j],F+j,evac)
						else:E.tensor_tensor(dst=D,data1=H[j],data2=D,op=A.add)
	def Db(a2,b2,acc,ct0,nd,first,k_off,n_off,Kp_p,Np,NKp_p,NCp,npass,group,fold_mode,evac,bmv,dt,f32,aq,bq,a_scale,b_scale,a_fmax,b_fmax,a_bound,b_bound,a_f8_dt,b_f8_dt,qeng):
		K=b_f8_dt;H=qeng;G=bmv;F=NCp;E=Kp_p;B=nd;M=A.ndarray((k,B,Ab,E),dtype=a_f8_dt,buffer=A.sbuf)
		if G==0:I=A.ndarray((k,B*F,Ab,s),dtype=K,buffer=A.sbuf)
		else:I=A.ndarray((k,B,Ab,Np),dtype=K,buffer=A.sbuf)
		J=1
		if Np>E:J=0
		for d in L(B):
			if H==1:C,D=As(1),As(1)
			elif H==2:C,D=As(J),As(1-J)
			elif H==3:C,D=As(d%2),As(d%2)
			else:C,D=As(0),As(0)
			Ma(a2,b2,M,I,d,ct0+d*AI,k_off,n_off,E,Np,F,dt,aq,bq,a_scale,b_scale,a_fmax,b_fmax,a_bound,b_bound,C,D,G)
		Mb(M,I,acc,B,first,NKp_p,F,npass,group,fold_mode,evac,G,f32)
	@K.jit
	def Mc(a2,b2,tc,group,fold_mode,evac,loop,shard,bmv,qeng,afmt,bfmt,aq,bq,a_scale,b_scale,a_bound,b_bound,inv_scale):
		j=inv_scale;Y=b_bound;X=a_bound;W=b_scale;V=a_scale;U=qeng;T=bmv;S=evac;R=fold_mode;J=b2;C=a2;M=C.shape[0];l=C.shape[1];N=J.shape[1];dt=C.dtype;K=A.float32;out=A.ndarray((l,N),dtype=dt,buffer=A.shared_hbm);m=l//k;n=N//s;o,p=MW()
		if shard==1:O,q=IL(m,o,p);g=0;r=n
		else:O=0;q=m;g,r=IL(n,o,p)
		D=q-O;G=r-g;Z=D*k;F=G*s;a=O*k;H=g*s;b=MX(G,group);c=IK(afmt);d=IK(bfmt);P=24e1
		if afmt==1:P=57344.
		Q=24e1
		if bfmt==1:Q=57344.
		I=A.ndarray((k,D,F),dtype=K,buffer=A.sbuf);B=tc//AI;e=M//AI
		if B>e:B=e
		f=e//B;t=e-f*B
		if loop==1:
			Db(C,J,I,0,B,1,a,H,Z,F,D,G,b,group,R,S,T,dt,K,aq,bq,V,W,P,Q,X,Y,c,d,U)
			for i in A.affine_range(f-1):Db(C,J,I,(i+1)*B*AI,B,0,a,H,Z,F,D,G,b,group,R,S,T,dt,K,aq,bq,V,W,P,Q,X,Y,c,d,U)
		else:
			for u in L(f):
				v=0
				if u==0:v=1
				Db(C,J,I,u*B*AI,B,v,a,H,Z,F,D,G,b,group,R,S,T,dt,K,aq,bq,V,W,P,Q,X,Y,c,d,U)
		if t>0:Db(C,J,I,f*B*AI,t,0,a,H,Z,F,D,G,b,group,R,S,T,dt,K,aq,bq,V,W,P,Q,X,Y,c,d,U)
		for i in L(D):
			h=A.ndarray((k,F),dtype=dt,buffer=A.sbuf)
			if i%2==0:E.activation(dst=h,op=A.copy,data=I[:,i,:],scale=j)
			else:E.tensor_scalar(dst=h,data=I[:,i,:],op0=A.multiply,operand0=j,engine=E.engine.vector)
			E.dma_copy(dst=out[(O+i)*k:(O+i+1)*k,H:H+F],src=h)
		return out
	FV=I
except P:FV=O
def Md(where,err):
	A=where;global CU,IF
	if not CU:CU=I;IF=f"{A}: {W(err).__name__}: {err}";D(f"[fp8_tn] kernel disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
def Me(kernel):A=kernel;return A[FS]if FS==2 else A
def IN():return 2 if FS==2 else 1
def Mf(device):return FV and not CU and device.type==a
def Mg(M,Kp,N,tc=C,group=Da,fold=I1,evac=I2,loop=I3,axis=C,bmv=I4,qeng=I5,afmt=I7,bfmt=I8,aq=IA,bq=IB,a_scale=IC,b_scale=ID,inv_scale=C,nprg=C):
	D=inv_scale;A=nprg;A=IN()if A is C else A;sh=FT(Kp,N,axis)
	if tc is C:tc=IJ(M,Kp,N,group=group,nprg=A,axis=sh)or FR
	E=IG(aq,a_scale);F=IG(bq,b_scale)
	if D is C:D=Q/(E*F)
	return B(tc),B(group),B(fold),B(evac),B(loop),B(sh),B(bmv),B(qeng),B(afmt),B(bfmt),B(aq),B(bq),G(E),G(F),IH(afmt,E),IH(bfmt,F),G(D)
def Mh(a,b,tc=C,group=Da,fold=I1,evac=I2,loop=I3,axis=C,bmv=I4,qeng=I5,afmt=I7,bfmt=I8,aq=IA,bq=IB,a_scale=IC,b_scale=ID,inv_scale=C):M,A=a.shape;N=b.shape[1];B=Mg(M,A,N,tc=tc,group=group,fold=fold,evac=evac,loop=loop,axis=axis,bmv=bmv,qeng=qeng,afmt=afmt,bfmt=bfmt,aq=aq,bq=bq,a_scale=a_scale,b_scale=b_scale,inv_scale=inv_scale);return Me(Mc)(a,b,*B)
try:BE=_FreezeNS(relu2_backward_from_h=Kv,_H_R32=Kh)
except P:BE=C
Mi=z(n(BE,'_H_R32',O))if BE is not C else O
def Dc(where,err):
	A=where;global CS,DW
	if not CS:CS=I;DW=f"{A}: {W(err).__name__}: {err}";D(f"[nki_gemm_fp8_nn] kernel disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
def At(kernel):A=kernel;return A[FK]if FK==2 else A
def FW(training):return z(training)
def Mj(training):return(FL or fp8nnk_ROW4)and FW(training)
def IO(training):return z(training)
def Mk(training):A=training;return fp8nnk_ROW3 and FW(A)or IO(A)
def IP(row):return M4
def Bo(*A):
	if not FQ or CS:return O
	for t in A:
		if t.device.type!=a or t.dtype!=torch.bfloat16 or t.dim()!=2 or not t.is_contiguous():return O
	return I
def Ml():global CT;CT+=1
def IQ(w,tag):
	A=Hn.get((x(w),tag))
	if A is C:return C
	key,B=A
	if key==(CT,w.data_ptr(),w._version):return B
	return C
def FX(w,tag,val):Hn[x(w),tag]=(CT,w.data_ptr(),w._version),val
def IR(w,scale,trans,key=C):
	A=trans;tag=f"f8@{scale:g}{"T"if A else Y}";k=w if key is C else key;B=IQ(k,tag)
	if B is not C:return B
	if Dj and tag in My and Bo(w)and Mz(k,w):
		wq=IZ(Ei,tag,k,scale,A,1,Ar(scale,0))
		if wq is not C:return wq
	if not Bo(w):return C
	try:
		if A and Hg=='hwt':wq=At(ML)(w.detach(),G(scale),1,Ar(scale,0))
		else:
			D=w.detach().t().contiguous()if A else w.detach();K,N=D.shape
			if K%R or FO(K,N)is C:return C
			wq=At(H_)(D,G(scale),1,Ar(scale,0))
	except P as err:Dc('quant_w_nn',err);return C
	FX(k,tag,wq);return wq
def Mm(w,scale,key=C):
	tag=f"qdq@{scale:g}";k=w if key is C else key;A=IQ(k,tag)
	if A is not C:return A
	if not Bo(w):return C
	K,N=w.shape
	if K%R or FO(K,N)is C:return C
	try:B=At(I0)(w.detach(),G(scale),G(Q/scale),1,Ar(scale,0),0)
	except P as err:Dc('qdq_w_nn',err);return C
	FX(k,tag,B);return B
def Mn(a2d,scale,fmt):
	A=a2d
	if not Bo(A):return C
	D,E=A.shape
	if D%R or FO(D,E)is C:return C
	try:return At(I0)(A,G(scale),G(Q/scale),1,Ar(scale,fmt),B(fmt))
	except P as err:Dc('qdq_a_nn',err);return C
def Au(a2d,w,trans_b,row):
	D=trans_b;B=row;A=a2d
	if B in(5,6):raise GI(f"_eager() called with dW row {B}: a dW row's incumbent depends on ctx.inplace and cannot be expressed as a2d @ w.  Its fallback is the call site's own unchanged line; dw_matmul() returns None to select it.  See this function's docstring.")
	if B==2:A=Hp(A)
	if B==4 or A6 is C:return torch.matmul(A,w.t()if D else w)
	return A6.nki_matmul(A,w,trans_b=D)
def Dd(a2d,w,row,key=C,slot='a'):
	E=row;A=slot;spec=BD[E]
	if spec[A+Ga]is not C:return spec[A+Ga]
	key=(E,x(w if key is C else key))if A=='a'else(E,A,x(w if key is C else key));s=DX.get(key)
	if s is not C:return s
	try:F=G(a2d.detach().abs().max().item())
	except P:F=S
	s=MG(F,spec[A+'fmt'])
	if F<MB:return s
	DX[key]=s;D(f"[nki_gemm_fp8_nn] row {E} ({spec[Bd]}){Y if A=="a"else" slot "+A} site {key[-1]:#x}: |a|max {F:.4e} -> s_g 2^{B(GJ(math.log2(s)))} = {s:g}",flush=I);return s
def FY(a2d,w,row,key=C):
	D=row;A=a2d;spec=BD[D];E=spec[Ef]
	if not Bo(A,w):return Au(A,w,E,D)
	F=Dd(A,w,D,key)
	if IP(D):
		J=Mn(A,F,spec[AR]);L=Mm(w,spec[B3],key)
		if J is not C and L is not C:return Au(J,L,E,D)
		return Au(A,w,E,D)
	M,K=A.shape
	if E:N,O=w.shape
	else:O,N=w.shape
	H=Hq(M,K,N,tb=C)if O==K else C
	if H is C or H[y]>Hj:return Au(A,w,E,D)
	wq=IR(w,spec[B3],E,key)
	if wq is C:return Au(A,w,E,D)
	I=spec[AR];R=Ho(spec[Cv],F);S=Ar(F,I);T=DY(I)if R==2 else S*F
	try:return At(MJ)(A,wq,3,B(H[AQ]),B(DV),B(FI),B(R),B(FJ),B(I),G(F),G(S),G(T),G(Q/(F*spec[B3])))
	except P as err:Dc(f"mm_fp8_nn row{D}",err);return Au(A,w,E,D)
FZ={}
Ai={}
def Fa():return{Cw:AA(FZ),'skips':AA(Ai)}
def IS(signed,scale):
	if not signed:return 1
	if scale==Q:return 0
	return 2
def Mo(*A):
	if _tn is C:return O
	if not Mf(A[0].device):return O
	for t in A:
		if t.device.type!=a or t.dtype!=torch.bfloat16 or t.dim()!=2 or not t.is_contiguous():return O
	return I
def Fb(a2d,b2d,row,key=C):
	D=b2d;B=a2d;A=row;spec=BD[A]
	if not spec.get(GT):raise GI(f"dw_matmul() called with row {A}, which is not a class-C row")
	H=spec[GU]
	if spec[H+Ga]is C and key is C:raise GI(f"dw_matmul() row {A}: the {H} operand's scale is `auto` and key is None.  An auto scale freezes against id(key), and a saved tensor's id is not stable across accesses -- pass key=ctx.w, or pin R1_NKI_GEMM_FP8_GSCALE.")
	if not Mo(B,D):Ai[A]=Ai.get(A,0)+1;return C
	M,E=B.shape;R,N=D.shape
	if R!=M:Ai[A]=Ai.get(A,0)+1;return C
	F=spec[Bc];G=spec[GW]
	if H=='a':F=Dd(B,key,A,key=key,slot='a')
	else:G=Dd(D,key,A,key=key,slot='b')
	L=IS(spec[GV],F);O=IS(spec[GX],G);Q=IN();I=FT(E,N);J=IJ(M,E,N,nprg=Q,axis=I);K=II(M,E,N,tc=J,nprg=Q,axis=I)if J else C
	if K is C or K[y]>IE:Ai[A]=Ai.get(A,0)+1;return C
	try:dw=Mh(B,D,tc=J,axis=I,afmt=spec[AR],bfmt=spec[Eg],aq=L,bq=O,a_scale=F,b_scale=G)
	except P as err:Md(f"mm_tn_fp8 row{A}",err);Ai[A]=Ai.get(A,0)+1;return C
	Mp(A,spec,M,E,N,F,G,L,O,K,Ej);return dw
def Mp(row,spec,M,Ka,N,s_a,s_b,aq,bq,pl,what):
	B=row;A=pl;n=FZ.get(B,0)+1;FZ[B]=n
	if n==1:C={0:Eh,1:GZ,2:GY};D(f"[nki_gemm_fp8_nn] row {B} ({spec[Bd]}) FIRED [{what}]: [{M},{Ka}]^T x [{M},{N}] a={B2 if spec[AR]else C8}@{s_a:g}/{C[aq]} b={B2 if spec[Eg]else C8}@{s_b:g}/{C[bq]}"+(f" tc={A[EZ]} shard={A["shard"]} folds={A["folds"]} sbuf={A[y]/1024:.0f}KiB"if A else Y)+' -- dW is RETURNED, the caller adds it back',flush=I)
class Mq:
	__slots__='folded',
	def __init__(self):self.folded=O
De=0
Fc=0
Df=0
CV=0
CW=O
Dg=C
def Mr():return{'from_h':Fc,'fold_dx':De,'link_misses':Df,'fold_kernel':CV}
def IT(h,dh):
	global Fc;Fc+=1
	if BE is not C:return BE.relu2_backward_from_h(h,dh)
	r=torch.sqrt(h.to(torch.float32)*4.).to(h.dtype).to(torch.float32);return(r*dh.to(torch.float32)).to(dh.dtype)
def Ms(where,err):
	A=where;global CW,Dg
	if not CW:CW=I;Dg=f"{A}: {W(err).__name__}: {err}";D(f"[nki_gemm_fp8_nn] relu2-fold FUSED kernel disabled after failure in {A}: {W(err).__name__}: {err} -- every further fold takes the SEPARATE-PASS form (gradient unchanged, traffic NOT deleted); do not read a throughput number off this run",flush=I)
def Mt(M,K,N,s_a,aq,pl):
	global CV;CV+=1
	if CV==1:A={Hk:AX,Hl:GZ,Hm:'sign1'};D(f"[nki_gemm_fp8_nn] relu2-fold FUSED epilogue FIRED: row 4 [{M},{K}]x[{K},{N}] + 2*sqrt(h)*dh in-kernel, a={B2 if BD[4][AR]else C8}@{s_a:g}/{A[aq]} tb={pl[AQ]} sbuf={pl[y]/1024:.0f}KiB -- the standalone ReLU^2 backward kernel is DELETED at this site",flush=I)
def Mu(dy2d,w,h2d,key):
	D=h2d
	if CW or IP(4)or Mi:return C
	if not Bo(dy2d,w,D):return C
	spec=BD[4];M,K=dy2d.shape;J,N=w.shape
	if J!=K or D.shape[0]!=M or D.shape[1]!=N:return C
	E=Hq(M,K,N,tb=spec.get(AQ),fold=1)
	if E is C or E[y]>Hj:return C
	A=Dd(dy2d,w,4,key);wq=IR(w,spec[B3],O,key)
	if wq is C:return C
	F=spec[AR];H=Ho(spec[Cv],A);I=Ar(A,F);L=DY(F)if H==2 else I*A
	try:out=At(MK)(dy2d,wq,D,3,B(E[AQ]),B(DV),B(FI),B(H),B(FJ),B(F),G(A),G(I),G(L),G(Q/(A*spec[B3])))
	except P as err:Ms('mm_fp8_nn_relu2 row4',err);return C
	Mt(M,K,N,A,H,E);return out
class Mv(torch.autograd.Function):
	@N
	def forward(ctx,x,w,inplace):
		x2d=x.reshape(-1,x.shape[-1]);ctx.save_for_backward(x2d,w);ctx.w=w;ctx.inplace=inplace;ctx.lead=x.shape[:-1];ctx.link=n(x,'_r1_relu2_link',C)if Aq else C
		if FL:y=FY(x2d,w,2)
		else:y=Au(x2d,w,I,2)
		return y.view(*x.shape[:-1],y.shape[-1])
	@N
	def backward(ctx,dy):
		global De,Df;x2d,w=ctx.saved_tensors;dy2d=dy.reshape(-1,dy.shape[-1])
		if not dy2d.is_contiguous():dy2d=dy2d.contiguous()
		dx=C
		if ctx.link is not C and fp8nnk_ROW4:
			dx=Mu(dy2d,w,x2d,ctx.w)
			if dx is not C:De+=1;ctx.link.folded=I
		if dx is C:
			if fp8nnk_ROW4:dx=FY(dy2d,w,4,key=ctx.w)
			else:dx=torch.matmul(dy2d,w)
			if ctx.link is not C:De+=1;dx=IT(x2d,dx);ctx.link.folded=I
			elif Aq:
				Df+=1
				if Df==1:D('[nki_gemm_fp8_nn] R1_RELU2_FOLD_DX: _ProjFn.backward found no c_fc link on its saved input; the ReLU^2 backward is NOT folded at this site',flush=I)
		A=Fb(dy2d,x2d,6,key=ctx.w)
		if ctx.inplace:
			if A is not C:ctx.w.grad.add_(A)
			else:ctx.w.grad.addmm_(dy2d.t(),x2d)
			dw=C
		else:dw=A if A is not C else torch.matmul(dy2d.t(),x2d)
		return dx.view(*ctx.lead,dx.shape[-1]),dw,C
def Mw(x,w,inplace):return Mv.apply(x,w,inplace)
try:fp8nnk__fp8_slice1=_fp8_slice1_ns
except P:fp8nnk__fp8_slice1=C
CX={}
import inspect as _inspect
for(Dh,Fd,Di)in((Cn,KK,n(A6,'fused_fc_relu2_forward',C)),(Ej,KL,DR)):
	CX[Dh]=Di is not C and'save_pre'in _inspect.signature(Di).parameters
	if Di is not C and not CX[Dh]:D(f"[nki_gemm_fp8_nn] NOTE: work/{Fd}.py predates this file -- {Di.__name__} does not take save_pre. Calls will omit it (so `pre` IS written, i.e. today's behaviour). Sync that module and re-check its md5 if you want the relu2 fold.",flush=I)
del _inspect
if Ah:
	for(Dh,Fd)in((Cn,KK),(Ej,KL)):
		if not CX[Dh]:Ah=O;Ap.append(f"work/{Fd}.py is older than this file -- its fused forward does not take save_pre, so `pre` would still be written and (a) would delete nothing; sync it and re-check its md5");D('[nki_gemm_fp8_nn] IGNORING R1_RELU2_FROM_H: '+Ap[-1],flush=I)
if(Ah or Aq)and BE is not C and not EP(BE,'relu2_backward_from_h'):Ap.append('work/nki_relu2_opt.py is older than this file -- it has no relu2_backward_from_h, so the first folded backward would raise AttributeError mid-run; sync it (md5 762b23e9...) and re-check');Ah=O;Aq=O;D('[nki_gemm_fp8_nn] IGNORING R1_RELU2_FROM_H/R1_RELU2_FOLD_DX: '+Ap[-1],flush=I)
Mx='r1,r3,r4'
IU=e(AO({t for t in Mx.replace(' ',Y).split(AG)if t in(C9,Ek,El)}))
IV=[]
Bp=()
IW=[]
for CY in IU:
	Bq=C
	if CY==C9:
		if fp8nnk__fp8_slice1 is C or not fp8k_USE_NKI_GEMM_FP8:Bq='row 1 reads as off from here: R1_NKI_GEMM_FP8 unset, or a frozen slice-1 namespace that lacks USE_NKI_GEMM_FP8'
		elif HN=='call'or LV:Bq='slice 1 runs WCACHE=call or qdq, so it keeps no fp8 weight cache'
	elif CY==Ek:
		if not fp8nnk_ROW3:Bq='row 3 is not live: check R1_NKI_GEMM_FP8_NN and its ROWS, or qdq mode'
	elif CY==El:
		if not fp8nnk_ROW4:Bq='row 4 is not live: check R1_NKI_GEMM_FP8_NN and its ROWS, or qdq mode'
	if Bq is C:IW.append(CY)
	else:IV.append(f"R1_NKI_GEMM_FP8_WPREP class {CY}: {Bq}");D('[nki_gemm_fp8_nn] IGNORING '+IV[-1],flush=I)
Bp=e(IW)
Dj=z(Bp)
Ac={}
if C9 in Bp:Ac[C9]=Cx,f"f8@{Bk:g}"
if Ek in Bp:Ac[Ek]=Ei,f"f8@{FH:g}T"
if El in Bp:Ac[El]=Ei,f"f8@{DT:g}"
My=A_(t for(A,t)in Ac.values()if A==Ei)
Fe={(A,t):c for(c,(A,t))in Ac.items()}
IX={}
Ff={}
Br={}
IY=set()
def Mz(k,w):return k is w or k.data_ptr()==w.data_ptr()and e(k.shape)==e(w.shape)and e(k.stride())==e(w.stride())and k.dtype==w.dtype
def M_(owner):return CP if owner==Cx else CT
def N0(key,msg):
	if key not in IY:IY.add(key);D(msg,flush=I)
def N1(members,trans):
	D=trans;C=members;B=A0(C.values())
	if V(B)<2:return'fewer than 2 weights in the class'
	t0=B[0]
	if t0.dim()!=2:return'not 2-D'
	K,N=t0.shape;E=t0.untyped_storage().data_ptr()
	for t in B:
		if t.dtype!=torch.bfloat16 or t.dim()!=2 or e(t.shape)!=(K,N)or e(t.stride())!=(N,1)or t.device!=t0.device or t.untyped_storage().data_ptr()!=E:return'members are not same-shape row-contiguous bf16 rows of one storage'
	if(N if D else K)%R:return f"quantised row count {N if D else K} is not a multiple of {R}"
	A=AO((t.storage_offset(),i)for(i,t)in C.items());d=A[1][0]-A[0][0]
	if d<K*N or d%N or any(A[q+1][0]-A[q][0]!=d for q in L(V(A)-1)):return'storage offsets are not a uniform, whole-row, non-overlapping stride'
	return{'order':[i for(_,i)in A],AV:V(A),'K':K,'N':N,'d':d,Gb:A[0][0]}
def N2(plan,members):
	A=plan;D=[members[i]for i in A['order']];K,N,B=A['K'],A['N'],A['d']//A['N'];E=torch.as_strided(D[0].detach(),((A[AV]-1)*B+K,N),(N,1),A[Gb])
	for(j,t)in A7(D):
		if E[j*B:j*B+K].data_ptr()!=t.data_ptr():return C,D,B
	return E,D,B
def N3(plan,members,scale,trans,clip,bound):
	A=plan;D,E,I=N2(A,members)
	if D is C:return C
	n,K,N=A[AV],A['K'],A['N']
	if trans:J=torch.as_strided(E[0].detach(),(n,K,N),(A['d'],N,1),A[Gb]);F=J.transpose(1,2).contiguous().view(n*N,K);rows,H=N,N
	else:F,rows,H=D,K,I
	out=At(MM)(F,n,rows,H,G(scale),B(clip),G(bound));return E,[out[j*rows:(j+1)*rows]for j in L(n)]
def N4(owner,t,scale,trans,clip,bound):
	A=bound;B=t.detach().t().contiguous()if trans else t.detach()
	if owner==Cx:return FD(HX)(B,G(scale),clip,A)
	return At(H_)(B,G(scale),clip,A)
def N5(a_u8,b_u8):
	A=a_u8.view(torch.float8_e4m3fn).float();B=b_u8.view(torch.float8_e4m3fn).float()
	if torch.isnan(A).any()or torch.isnan(B).any():return G('inf')
	m=torch.maximum(A.abs(),B.abs());C=torch.where(m>=2.**-6,torch.exp2(torch.floor(torch.log2(m.clamp_min(2.**-6)))-3.),torch.full_like(m,2.**-9));return G(((A-B).abs()/C).max())
def N6(tag,owner,ts,outs,scale,trans,clip,bound):
	E=[N4(owner,t,scale,trans,clip,bound)for t in ts];A=C=0;D=S
	for(G,r)in zip(outs,E):
		a=G.detach().cpu();b=r.detach().cpu()
		if e(a.shape)!=e(b.shape):return O,f"shape {e(a.shape)} vs per-weight {e(b.shape)} -> FAIL",E
		F=a==b;A+=B(F.sum());C+=a.numel()
		if not z(F.all()):D=i(D,N5(a,b))
	ok=A==C or D<=Q;H=f"{V(ts)} weights, bytes equal {A}/{C}"+(Y if A==C else f", worst gap {D:.3g} ulp (<= 1 allowed: stochastic rounding)")+f" -> {"PASS"if ok else"FAIL"}";return ok,H,E
def N7(owner,t,tag,val):
	if owner==Cx:Hb(t,tag,val)
	else:FX(t,tag,val)
def IZ(owner,tag,k,scale,trans,clip,bound):
	Q=bound;N='done';M='since';K='members';F=trans;B='plan';E=owner,tag
	if E in Br:return C
	H=M_(owner);A=IX.get(E)
	if A is C:A=IX[E]={K:{},M:H,B:C,N:C}
	if A[K].get(x(k))is not k:A[K][x(k)]=k;A[M]=H;A[B]=C;return C
	if H<=A[M]or A[N]==H:return C
	if A[B]is C:
		p=N1(A[K],F);A[B]=p if AP(p,AA)else O
		if not AP(p,AA):N0((B,)+E,f"[nki_gemm_fp8_nn] wprep {Fe.get(E,tag)} ({tag}): NOT batched ({p}) -- today's per-weight path for this class")
	if A[B]is O:return C
	A[N]=H
	try:
		r=N3(A[B],A[K],scale,F,clip,Q)
		if r is C:A[B]=C;return C
		G,outs=r
	except P as err:Br[E]=f"batch launch: {W(err).__name__}: {err}";D(f"[nki_gemm_fp8_nn] wprep {Fe.get(E,tag)} ({tag}) DISABLED after failure: {W(err).__name__}: {err} -- per-weight quantisation (today's path) for the rest of the run; this run is NOT a wprep measurement",flush=I);return C
	st=Ff.setdefault(E,{Cy:0,B4:0,B5:CA})
	if st[B5]==CA:
		try:ok,J,L=N6(tag,owner,G,outs,scale,F,clip,Q)
		except P as err:ok,J,L=O,f"ERROR {W(err).__name__}: {err} -> FAIL",C
		st[B5]=J;R=Fe.get(E,tag)
		if not ok:
			Br[E]='self-check '+J;D(f"[nki_gemm_fp8_nn] wprep {R} ({tag}) self-check {J} -- class DISABLED, per-weight quantisation (today's path) for the rest of the run; this run is NOT a wprep measurement",flush=I)
			if L is C:return C
			outs=L
		else:D(f"[nki_gemm_fp8_nn] wprep {R} ({tag}) FIRED: {V(G)} weights {"[%d,%d] (W^T)"%(A[B]["N"],A[B]["K"])if F else"[%d,%d]"%(A[B]["K"],A[B]["N"])} in ONE quantise launch{" + ONE batched transpose"if F else Y} (replaces {V(G)*(2 if F else 1)} launches); self-check {J}",flush=I)
	for(t,S)in zip(G,outs):N7(owner,t,tag,S)
	if E not in Br:st[Cy]+=1;st[B4]+=V(G)
	return outs[[x(t)for t in G].index(x(k))]
def N8(x2d,w):
	D=fp8nnk__fp8_slice1;tag=Ac[C9][1]
	if Ha(w,tag)is not C or not FE(x2d,w):return
	M,K=x2d.shape;B,N=w.shape;A=HT(M,K,N,tb=CM,group=CN,nprg=HZ())if B==K else C
	if A is C or A[y]>HO:return
	IZ(Cx,tag,w,Bk,O,FA(),FB(Bk))
def N9():
	C=' -> '
	if not Dj:return'False(asked)'
	out=[]
	for(c,(A,t))in AO(Ac.items()):st=Ff.get((A,t),{Cy:0,B4:0,B5:CA});v=st[B5].split(C)[-1]if C in st[B5]else st[B5].replace(' ','_');out.append(f"{c}:b{st[Cy]}/w{st[B4]}/{"OFF"if(A,t)in Br else v}")
	B=[c for c in IU if c not in Ac];return AG.join(out)+(f"(asked:{AG.join(B)})"if B else Y)
def NA():return{c:AA(Ff.get((A,t),{Cy:0,B4:0,B5:CA}),off=Br.get((A,t)))for(c,(A,t))in Ac.items()}
def Ia(x2d,w,save_pre=1):
	A=save_pre
	if fp8nnk__fp8_slice1 is not C and fp8k_USE_NKI_GEMM_FP8:
		if Dj and C9 in Ac:N8(x2d,w)
		if CX[Ej]:return DR(x2d,w,save_pre=A)
		assert A,'save_pre=0 requested but work/nki_gemm_fp8.py does not support it';return DR(x2d,w)
	if A6 is not C:
		if CX[Cn]:return A6.fused_fc_relu2_forward(x2d,w,save_pre=A)
		assert A,'save_pre=0 requested but work/nki_gemm.py does not support it';return A6.fused_fc_relu2_forward(x2d,w)
	B=torch.matmul(x2d,w);return torch.nn.functional.relu(B).square(),B
Fg=C
if A6 is not C:Fg=A6.FcRelu2Fn.backward
def Ib(ctx,dh,dpre_ready=O):
	B=dh;x2d,w,D=ctx.saved_tensors;B=B.contiguous()
	if dpre_ready:A=B
	elif Ah:A=IT(D,B)
	elif A6 is not C and A6._relu2_backward is not C:A=A6._relu2_backward(D,B)
	elif A6 is not C:A=A6.eager_relu2_backward(D,B)
	else:A=B*(2.*torch.nn.functional.relu(D))
	if fp8nnk_ROW3:dx=FY(A,w,3,key=ctx.w)
	else:dx=Au(A,w,I,3)
	dw=Fb(x2d,A,5,key=ctx.w)
	if ctx.inplace:
		if dw is not C:ctx.w.grad.add_(dw)
		else:ctx.w.grad.addmm_(x2d.t(),A)
		return dx,C,C
	if dw is not C:return dx,dw,C
	if A6 is not C:return dx,A6.nki_matmul(x2d,A,trans_a=I),C
	return dx,torch.matmul(x2d.t(),A),C
class Ic(torch.autograd.Function):
	@N
	def forward(ctx,x2d,w,inplace,link=C):
		if Ah:out=Ia(x2d,w,save_pre=0);h=out[0]if AP(out,(e,A0))else out;ctx.save_for_backward(x2d,w,h)
		else:h,A=Ia(x2d,w);ctx.save_for_backward(x2d,w,A)
		ctx.w=w;ctx.inplace=inplace;ctx.link=link;return h
	@N
	def backward(ctx,dh):return Ic._grads(ctx,dh)+(C,)
	@N
	def _grads(ctx,dh):
		A=ctx.link
		if A is not C and A.folded:return Ib(ctx,dh,dpre_ready=I)
		if not fp8nnk_ROW3 and not MH:
			if Fg is not C:return Fg(ctx,dh)
		return Ib(ctx,dh)
def NB(x,w,inplace=O):
	K=x.shape[-1];x2d=x.reshape(-1,K)
	if not x2d.is_contiguous():x2d=x2d.contiguous()
	A=Mq()if Aq else C;h=Ic.apply(x2d,w,inplace,A);B=h.view(*x.shape[:-1],w.shape[1])
	if A is not C:B._r1_relu2_link=A
	return B
def NC():
	D='(asked)';rows=AG.join(v(r)for r in Bl);A=AG.join(v(r)for r in DS);live=['mlp:'+rows]+['dw:'+A];B=f"{CR:g}"
	if DX:E=AO(DX.items(),key=lambda kv:(kv[0][0],v(kv[0][1:])));B+='['+AG.join(f"{k[0]}:{s:g}"for(k,s)in E[:3])+'..]'
	return f"live={" ".join(live)if live else AW} mode={FF!r} rows={rows} eval={M5} h_scale={FG:g} g_scale={B} gfmt={B2} w_fc={FH:g} w_proj={DT:g} clip={AS} r2cap={FM}:{FN} dw={He!r} dw_rows={A} dw_eval={M8} a_dw={Hf:g} dw_fired={Fa()[Cw]or"{}"} dw_skips={Fa()["skips"]or"{}"} dw_tn={"-"if _tn is C else"nki=%s disabled=%s"%(FV,CU)} wcache={MD} wqt={Hg} tb_a={Hh} tb_b={Hi} group={DV} evac={FI} loop={FJ} relu2_from_h={Ah}{D if not Ah else Y} relu2_fold_dx={Aq}{D if not Aq else Y} relu2_fold_kernel={CV}{"(DISABLED)"if CW else Y} wprep={N9()} lnc={FK} shard={MF} nki={FQ} disabled={CS}"+(f" err={DW}"if DW else Y)+(f" fold_err={Dg}"if Dg else Y)
class ND:0
m=ND()
m.begin_step=Ml
m.active=FW
m.proj_active=Mj
m.fc_active=Mk
m.c_proj=Mw
m.fused_fc_relu2_nn=NB
m.config_string=NC
m.row2_cap=Hp
m.USE_NKI_GEMM_FP8_NN=Hd
m.MODE=FF
m.ROWS=Bl
m.fold_counters=Mr
m.RELU2_FROM_H=Ah
m.RELU2_FOLD_DX=Aq
m.dw_active=IO
m.dw_matmul=Fb
m.dw_counters=Fa
m.USE_NKI_GEMM_FP8_DW=M7
m.DW_MODE=He
m.DW_ROWS=DS
m.WPREP=Dj
m.WPREP_CLASSES=Bp
m.wprep_counters=NA
nki_gemm_fp8_nn=m
USE_NKI_GEMM_FP8_NN=Hd
DW_FP8_ENV=q!=A1
USE_NKI_GEMM_FP8_DW=DW_FP8_ENV and n(nki_gemm_fp8_nn,'USE_NKI_GEMM_FP8_DW',O)
USE_NKI_GEMM_FP8_ANY=USE_NKI_GEMM_FP8_NN or USE_NKI_GEMM_FP8_DW
def relu_squared(x):
	return _relu_squared_opt(x)
	if USE_NKI_RELU2 and _HAS_NKI and x.device.type==a and x.numel()//x.shape[-1]%_NKI_PARTITION==0:return _ReluSquaredNKI.apply(x)
	return F.relu(x).square()
import os,torch,torch.nn.functional as F
Dk=128
NE=128
NF=G(2.**-23)
NG=8
NH=32
NI=BQ
NJ=BQ
NK=BQ
Id=B(os.environ.get(AB,q))
NL='bcast'
ropek_USE_NKI_ROPE_NORM=I
CZ=O
Ie=C
def NM(x):return F.rms_norm(x,(x.size(-1),))
def NN(x,cos,sin):d=x.shape[-1]//2;A,B=x[...,:d],x[...,d:];C=A*cos+B*sin;D=A*-sin+B*cos;return torch.cat((C,D),dim=-1)
def Dl(x,cos,sin):return NM(NN(x,cos,sin)).transpose(1,2)
def NO(T,H=8):
	if T<=0:return C
	if T<=Dk:return 1,T,1
	if T%Dk:return C
	n=T//Dk;A=i(1,g(NG,NH//i(H,1),n));k=i(d for d in L(1,A+1)if n%d==0);return n//k,Dk,k
try:
	import nki as K,nki.language as A,nki.isa as E
	def Fh(mode,in_dtype):
		if mode==BQ:return in_dtype
		return A.bfloat16 if mode==Cn else A.float32
	NP=A.float32;NQ=E.engine.unknown
	def BF(dst,src):E.dma_copy(dst=dst,src=src)
	def Fi(dst,a,b,op):E.tensor_tensor(dst=dst,data1=a,data2=b,op=op,engine=NQ)
	def NR():
		if A.program_ndim()>0:return A.program_id(0),A.num_programs(0)
		return 0,1
	def If(B,C):
		D,A=NR()
		if C>=A:return L(D,C,A),L(B)
		return L(C),L(D,B,A)
	def Ig(cos5,sin5,c,P,K,Dh,H):
		B=A.ndarray((P,K,1,Dh),dtype=cos5.dtype,buffer=A.sbuf);C=A.ndarray((P,K,1,Dh),dtype=sin5.dtype,buffer=A.sbuf);BF(dst=B,src=cos5[c,:,:,:,:]);BF(dst=C,src=sin5[c,:,:,:,:]);return B.broadcast(2,H),C.broadcast(2,H);D=A.ndarray((P,K,H,Dh),dtype=B.dtype,buffer=A.sbuf);F=A.ndarray((P,K,H,Dh),dtype=C.dtype,buffer=A.sbuf)
		for h in L(H):E.tensor_copy(dst=D[:,:,h,:],src=B[:,:,0,:]);E.tensor_copy(dst=F[:,:,h,:],src=C[:,:,0,:])
		return D,F
	def Fj(dst,src,vec,K,H,op=C):
		D=vec;C=src;B=dst;F=NL
		if F=='bcast':E.tensor_tensor(dst=B,data1=C,data2=D.broadcast(3,B.shape[3]),op=A.multiply)
		elif F==BX:
			for k in L(K):
				for h in L(H):E.activation(dst=B[:,k,h,:],op=A.copy,data=C[:,k,h,:],scale=D[:,k,h,:])
		else:
			for k in L(K):
				for h in L(H):E.tensor_scalar(dst=B[:,k,h,:],data=C[:,k,h,:],op0=A.multiply,operand0=D[:,k,h,:])
	def NS(dst,sq,scratch,K,H):E.tensor_reduce(dst=dst,op=A.add,data=sq,axis=3,keepdims=I)
	def Fk(tile,shape):return tile
	def Dm(dst,a,b,which):E.tensor_tensor(dst=dst,data1=a,data2=b,op=A.multiply)
	def Fl(dst,src,cb,sb,tmp_a,tmp_b,Dh,transpose):E=transpose;D=tmp_b;C=tmp_a;B=Dh;F=2*B;G,H=src[:,:,:,0:B],src[:,:,:,B:F];I,J=dst[:,:,:,0:B],dst[:,:,:,B:F];Dm(C,G,cb,0);Dm(D,H,sb,1);Fi(I,C,D,A.subtract if E else A.add);Dm(C,H,cb,0);Dm(D,G,sb,1);Fi(J,C,D,A.add if E else A.subtract)
	def Ih(r,sq,P,K,H,D):B=A.ndarray((P,K,H,1),dtype=A.float32,buffer=A.sbuf);E.activation(dst=sq,op=A.square,data=r);E.tensor_reduce(dst=B,op=A.add,data=sq,axis=3,keepdims=I);C=A.ndarray((P,K,H,1),dtype=A.float32,buffer=A.sbuf);E.tensor_scalar(dst=B,data=B,op0=A.multiply,operand0=Q/D,op1=A.add,operand1=NF);E.tensor_scalar(dst=C,data=B,op0=A.rsqrt,operand0=S);return C
	@K.jit
	def NT(x6,cos5,sin5):
		B,F,C,K,D,E=x6.shape;G=E//2;N=A.float32;H=A.ndarray((B,D,F,C,K,E),dtype=x6.dtype,buffer=A.shared_hbm);I,J=If(B,F)
		for c in I:
			L,M=Ig(cos5,sin5,c,C,K,G,D)
			for b in J:NU(x6,H,L,M,b,c,C,K,D,E,G)
		return H
	def NU(x6,y6,cb,sb,b,c,P,K,H,D,Dh):
		C=A.float32;xb=A.ndarray((P,K,H,D),dtype=x6.dtype,buffer=A.sbuf);BF(dst=xb,src=x6[b,c,:,:,:,:]);xb=Fk(xb,(P,K,H,D));B=Fh(NI,x6.dtype);E=A.ndarray((P,K,H,Dh),dtype=B,buffer=A.sbuf);F=A.ndarray((P,K,H,Dh),dtype=B,buffer=A.sbuf);r=A.ndarray((P,K,H,D),dtype=B,buffer=A.sbuf);Fl(r,xb,cb,sb,E,F,Dh,transpose=O);sq=A.ndarray((P,K,H,D),dtype=C,buffer=A.sbuf);G=Ih(r,sq,P,K,H,D);yb=A.ndarray((P,K,H,D),dtype=x6.dtype,buffer=A.sbuf);Fj(yb,r,G,K,H)
		for h in L(H):BF(dst=y6[b,h,c,:,:,:],src=yb[:,:,h,:])
	@K.jit
	def NV(x6,cos5,sin5,g6):
		B,F,C,K,D,E=x6.shape;G=E//2;N=A.float32;H=A.ndarray((B,F,C,K,D,E),dtype=x6.dtype,buffer=A.shared_hbm);I,J=If(B,F)
		for c in I:
			L,M=Ig(cos5,sin5,c,C,K,G,D)
			for b in J:NW(x6,g6,H,L,M,b,c,C,K,D,E,G)
		return H
	def NW(x6,g6,gx6,cb,sb,b,c,P,K,H,D,Dh):
		C=x6;G=A.float32;xb=A.ndarray((P,K,H,D),dtype=C.dtype,buffer=A.sbuf);BF(dst=xb,src=C[b,c,:,:,:,:]);xb=Fk(xb,(P,K,H,D));F=A.ndarray((P,K,H,D),dtype=g6.dtype,buffer=A.sbuf)
		for h in L(H):BF(dst=F[:,:,h,:],src=g6[b,h,c,:,:,:])
		F=Fk(F,(P,K,H,D));J=Fh(NJ,C.dtype);N=A.ndarray((P,K,H,Dh),dtype=J,buffer=A.sbuf);R=A.ndarray((P,K,H,Dh),dtype=J,buffer=A.sbuf);r=A.ndarray((P,K,H,D),dtype=J,buffer=A.sbuf);Fl(r,xb,cb,sb,N,R,Dh,transpose=O);sq=A.ndarray((P,K,H,D),dtype=G,buffer=A.sbuf);M=Ih(r,sq,P,K,H,D);E.tensor_tensor(dst=sq,data1=F,data2=r,op=A.multiply);S=A.ndarray((P,K,H,1),dtype=G,buffer=A.sbuf);u=A.ndarray((P,K,H,D),dtype=NP,buffer=A.sbuf);NS(S,sq,sq,K,H);B=A.ndarray((P,K,H,1),dtype=G,buffer=A.sbuf);E.tensor_tensor(dst=B,data1=M,data2=M,op=A.multiply);E.tensor_tensor(dst=B,data1=B,data2=S,op=A.multiply);E.tensor_scalar(dst=B,data=B,op0=A.multiply,operand0=Q/D);Fj(u,r,B,K,H);Fi(u,F,u,A.subtract);T=A.ndarray((P,K,H,D),dtype=Fh(NK,C.dtype),buffer=A.sbuf);Fj(T,u,M,K,H);U=A.ndarray((P,K,H,D),dtype=C.dtype,buffer=A.sbuf);Fl(U,T,cb,sb,N,R,Dh,transpose=I);BF(dst=gx6[b,c,:,:,:,:],src=U)
	Ii=I
except P:Ii=O
def Ij(where,err):
	A=where;global CZ,Ie
	if not CZ:CZ=I;Ie=f"{A}: {W(err).__name__}: {err}";D(f"[nki_rope_norm_opt] kernel disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
def Ik(kernel):A=kernel;return A[Id]if Id==2 else A
def NX(cos,sin,T,Dh,dtype):
	A=cos.reshape(T,Dh);B=sin.reshape(T,Dh)
	if A.dtype!=dtype:A=A.to(dtype)
	if B.dtype!=dtype:B=B.to(dtype)
	if not A.is_contiguous():A=A.contiguous()
	if not B.is_contiguous():B=B.contiguous()
	return A,B
class NY(torch.autograd.Function):
	@N
	def forward(ctx,x,cos2,sin2,plan):
		D=sin2;C=cos2;B,T,G,A=x.shape;E,F,K=plan;ctx.save_for_backward(x,C,D);ctx.plan=plan
		try:H=x.view(B,E,F,K,G,A);I=Ik(NT)(H,C.view(E,F,K,1,A//2),D.view(E,F,K,1,A//2));return I.view(B,G,T,A)
		except P as err:
			Ij(BS,err)
			with torch.no_grad():return Dl(x,C.view(1,T,1,A//2),D.view(1,T,1,A//2)).contiguous()
	@N
	def backward(ctx,gy):
		x,G,H=ctx.saved_tensors;B,T,F,A=x.shape;D,E,K=ctx.plan
		if not CZ:
			try:L=gy.contiguous().view(B,F,D,E,K,A);M=Ik(NV)(x.view(B,D,E,K,F,A),G.view(D,E,K,1,A//2),H.view(D,E,K,1,A//2),L);return M.view(B,T,F,A),C,C,C
			except P as err:Ij(Am,err)
		with torch.enable_grad():J=x.detach().requires_grad_(I);y=Dl(J,G.view(1,T,1,A//2),H.view(1,T,1,A//2));gx,=torch.autograd.grad(y,J,gy)
		return gx,C,C,C
def ropek_rope_norm(x,cos,sin):
	if x.dim()!=4:return Dl(x,cos,sin)
	B,T,D,A=x.shape;F=Ii and not CZ and x.device.type==a and A==NE and x.dtype in(torch.bfloat16,torch.float32)and cos.numel()==T*(A//2)and sin.numel()==T*(A//2);E=NO(T,D)if F else C
	if E is C:return Dl(x,cos,sin)
	if not(x.stride(3)==1 and x.stride(2)==A and x.stride(1)>=D*A):x=x.contiguous()
	G,H=NX(cos,sin,T,A//2,x.dtype);return NY.apply(x,G,H,E)
rope_norm=ropek_rope_norm
USE_NKI_ROPE_NORM=ropek_USE_NKI_ROPE_NORM
try:from nki_gain_norm import gain_residual_norm,USE_NKI_GAIN_NORM
except P:USE_NKI_GAIN_NORM=O
import os,torch,torch.nn.functional as F
u=128
Il=8192
Dn=15.
lossk_USE_NKI_LOSS_HEAD=I
Av=g(1,1)
r=2
Im=B(os.environ.get(AB,q))
NZ=O
Bs=O
In=C
def Fm(x2,targets,softcap=Dn):A=softcap;logits=x2.float();logits=A*torch.tanh(logits/A);vocab=logits.size(-1);logp=F.log_softmax(logits,dim=-1);onehot=(targets.reshape(-1,1)==torch.arange(vocab,device=logits.device).view(1,-1)).to(logp.dtype);return-(logp*onehot).sum(dim=-1)
def Na(logits,targets,reduction=Be,softcap=Dn):
	A=reduction;nll=Fm(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),softcap)
	if A==AW:return nll.view_as(targets)
	return nll.sum()if A==Gc else nll.mean()
try:
	import nki as K,nki.language as A,nki.isa as E;Io=A.float32
	def w(dst,src):E.dma_copy(dst=dst,src=src)
	def Nb():
		if A.program_ndim()>0:return A.program_id(0),A.num_programs(0)
		return 0,1
	def Do(NT):A,B=Nb();C,D=NT//B,NT%B;E=A*C+g(A,D);n=C+(1 if A<D else 0);return E,E+n
	def Bt(dst_dtype,src,P,n):blk=A.ndarray((P,n),dtype=dst_dtype,buffer=A.sbuf);w(dst=blk,src=src);return blk
	def Ip(tg2,i0,i1):
		A=tg2
		if V(A.shape)==3:return A[:,i0:i1,0]
		return A[:,i0:i1]
	def AJ(shape,dtype):return A.ndarray(shape,dtype=dtype,buffer=A.sbuf),A.ndarray(shape,dtype=dtype,buffer=A.sbuf)
	def Iq(P,V):B=A.ndarray((P,V),dtype=A.float32,buffer=A.sbuf);E.iota(dst=B,pattern=[[1,V]],offset=0,channel_multiplier=0);return B
	@K.jit
	def Ir(x3,tg2,softcap,scale):
		G=x3;F=softcap;B,I,J=G.shape;C=A.float32;O=A.ndarray((B,I),dtype=C,buffer=A.shared_hbm);P=A.ndarray((B,I),dtype=C,buffer=A.shared_hbm);D,H=Do(I);n=H-D
		if n>0:
			V=Ip(tg2,D,H);R=A.ndarray((B,n),dtype=C,buffer=A.sbuf);W=Bt(A.uint32,V,B,n);S=A.ndarray((B,n),dtype=G.dtype,buffer=A.sbuf);K=AJ((B,J),G.dtype);X=AJ((B,J),C);Y=AJ((B,J),Io)
			for j in L(g(Av,n)):w(dst=K[j%r],src=G[:,D+j,:])
			for j in L(n):
				i=D+j;xb,t,e=K[j%r],X[j%r],Y[j%r];M=j+Av
				if M<n:w(dst=K[M%r],src=G[:,D+M,:])
				E.activation(dst=t,op=A.tanh,data=xb,scale=Q/F);E.activation_reduce(dst=e,op=A.exp,data=t,reduce_op=A.add,reduce_res=R[:,j:j+1],scale=F,bias=-F);E.nc_n_gather(dst=S[:,j:j+1],data=xb,indices=W[:,j:j+1])
			N=A.ndarray((B,n),dtype=C,buffer=A.sbuf);E.activation(dst=N,op=A.log,data=R);w(dst=P[:,D:H],src=N);tb=A.ndarray((B,n),dtype=C,buffer=A.sbuf);E.activation(dst=tb,op=A.tanh,data=S,scale=Q/F);T=A.ndarray((B,n),dtype=C,buffer=A.sbuf);E.tensor_scalar(dst=T,data=N,op0=A.multiply,operand0=scale,op1=A.add,operand1=F*scale);U=A.ndarray((B,n),dtype=C,buffer=A.sbuf);E.scalar_tensor_tensor(dst=U,data=tb,op0=A.multiply,operand0=-F*scale,op1=A.add,operand1=T);w(dst=O[:,D:H],src=U)
		return O,P
	@K.jit
	def Is(x3,tg2,lz2,g2,softcap,scale):
		J=softcap;F=x3;B,N,G=F.shape;C=A.float32;O=A.ndarray((B,N,G),dtype=F.dtype,buffer=A.shared_hbm);D,H=Do(N);n=H-D
		if n>0:
			S=Bt(C,Ip(tg2,D,H),B,n);T=Bt(C,lz2[:,D:H],B,n);gb=Bt(C,g2[:,D:H],B,n);P=A.ndarray((B,n),dtype=C,buffer=A.sbuf);E.tensor_scalar(dst=P,data=T,op0=A.multiply,operand0=-Q,op1=A.add,operand1=-J);K=A.ndarray((B,n),dtype=C,buffer=A.sbuf);E.tensor_scalar(dst=K,data=gb,op0=A.multiply,operand0=scale);sq=A.ndarray((B,n),dtype=C,buffer=A.sbuf);E.activation(dst=sq,op=A.sqrt,data=K);U=Iq(B,G);I=AJ((B,G),F.dtype);V=AJ((B,G),C);R=AJ((B,G),C);W=R;X=I
			for j in L(g(Av,n)):w(dst=I[j%r],src=F[:,D+j,:])
			for j in L(n):
				i=D+j;xb,t,p,q,d=I[j%r],V[j%r],R[j%r],W[j%r],X[j%r];M=j+Av
				if M<n:w(dst=I[M%r],src=F[:,D+M,:])
				E.activation(dst=t,op=A.tanh,data=xb,scale=Q/J);E.activation(dst=p,op=A.exp,data=t,scale=J,bias=P[:,j:j+1]);E.scalar_tensor_tensor(dst=q,data=U,op0=A.equal,operand0=S[:,j:j+1],op1=A.subtract,operand1=p);d=xb;E.activation(dst=t,op=A.square,data=t,scale=sq[:,j:j+1]);E.scalar_tensor_tensor(dst=d,data=t,op0=A.subtract,operand0=K[:,j:j+1],op1=A.multiply,operand1=q);w(dst=O[:,i,:],src=d)
		return O
	It=I
except P:It=O
def Iu(where,err):
	A=where;global Bs,In
	if not Bs:Bs=I;In=f"{A}: {W(err).__name__}: {err}";D(f"[nki_loss_head] kernel disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
def Aw(kernel):A=kernel;return A[Im]if Im==2 else A
def Iv(tg):
	N=tg.numel();A=N//u
	if tg.dtype==torch.int32:return tg.contiguous().view(u,A)
	if tg.dtype==torch.int64:
		try:return tg.contiguous().view(torch.int32).view(u,A,2)
		except P:pass
	return tg.to(torch.int32).view(u,A)
def Ca(x2):N,A=x2.shape;return x2.view(u,N//u,A)
class Nc(torch.autograd.Function):
	@N
	def forward(ctx,x2,tgt_k,targets,softcap,scale):
		B=tgt_k;A=softcap;N,F=x2.shape;ctx.softcap,ctx.scale=A,scale;D=C
		try:E,D=Aw(Ir)(Ca(x2),B,G(A),G(scale));rows=E.view(N)
		except P as err:
			Iu(BS,err)
			with torch.no_grad():rows=Fm(x2,targets,A)*scale
		ctx.save_for_backward(x2,B,targets,D);return rows
	@N
	def backward(ctx,grad_rows):
		B=grad_rows;A,H,targets,D=ctx.saved_tensors;N,J=A.shape;E,scale=ctx.softcap,ctx.scale
		if D is not C and not Bs:
			try:g2=B.contiguous().view(u,N//u);K=Aw(Is)(Ca(A),H,D,g2,G(E),G(scale));return K.view(N,J),C,C,C,C
			except P as err:Iu(Am,err)
		with torch.enable_grad():F=A.detach().requires_grad_(I);rows=Fm(F,targets,E)*scale;gx,=torch.autograd.grad(rows,F,B)
		return gx,C,C,C,C
def lossk_fused_loss(logits,targets,reduction=Be,softcap=Dn):
	D=softcap;A=reduction;C=logits.shape[-1];N=targets.numel();E=It and not Bs and logits.device.type==a and logits.dtype in(torch.bfloat16,torch.float32)and(C==Il or NZ)and N%u==0 and N>0 and logits.numel()==N*C and A in(Be,Gc,AW)and targets.dtype in(torch.int64,torch.int32)
	if not E:return Na(logits,targets,A,D)
	B=logits.reshape(N,C)
	if not B.is_contiguous():B=B.contiguous()
	tg=targets.reshape(N);F=Iv(tg);scale=Q/N if A==Be else Q;rows=Nc.apply(B,F,tg,D,scale)
	if A==AW:return rows.view_as(targets)
	return rows.sum()
try:
	import nki as K,nki.language as A,nki.isa as E;BG=E.engine.vector
	def Nd(P,V):return Iq(P,V)
	def Ne(dst,io,t):E.tensor_scalar(dst=dst,data=io,op0=A.equal,operand0=t)
	@K.jit
	def Iw(x3,tg3,rw3,softcap):
		G=x3;F=softcap;B,J,M=G.shape;K=tg3.shape[0];C=A.float32;S=A.ndarray((B,J),dtype=C,buffer=A.shared_hbm);T=A.ndarray((B,J),dtype=C,buffer=A.shared_hbm);D,H=Do(J);n=H-D
		if n>0:
			U=A.ndarray((B,n),dtype=C,buffer=A.sbuf);V=A.ndarray((B,K,n),dtype=A.uint32,buffer=A.sbuf);W=A.ndarray((B,K,n),dtype=G.dtype,buffer=A.sbuf)
			for k in L(K):w(dst=V[:,k,:],src=tg3[k,:,D:H])
			N=AJ((B,M),G.dtype);Y=AJ((B,M),C);Z=AJ((B,M),Io)
			for j in L(g(Av,n)):w(dst=N[j%r],src=G[:,D+j,:])
			for j in L(n):
				i=D+j;xb,t,e=N[j%r],Y[j%r],Z[j%r];O=j+Av
				if O<n:w(dst=N[O%r],src=G[:,D+O,:])
				E.activation(dst=t,op=A.tanh,data=xb,scale=Q/F);E.activation_reduce(dst=e,op=A.exp,data=t,reduce_op=A.add,reduce_res=U[:,j:j+1],scale=F,bias=-F)
				for k in L(K):E.nc_n_gather(dst=W[:,k,j:j+1],data=xb,indices=V[:,k,j:j+1])
			P=A.ndarray((B,n),dtype=C,buffer=A.sbuf);E.activation(dst=P,op=A.log,data=U);w(dst=T[:,D:H],src=P);X=A.ndarray((B,n),dtype=C,buffer=A.sbuf);E.tensor_scalar(dst=X,data=P,op0=A.add,operand0=F);R=A.ndarray((B,K,n),dtype=C,buffer=A.sbuf);tb=A.ndarray((B,n),dtype=C,buffer=A.sbuf);nll=A.ndarray((B,n),dtype=C,buffer=A.sbuf);I=A.ndarray((B,n),dtype=C,buffer=A.sbuf)
			for k in L(K):
				w(dst=R[:,k,:],src=rw3[k,:,D:H]);E.activation(dst=tb,op=A.tanh,data=W[:,k,:],scale=Q/F);E.scalar_tensor_tensor(dst=nll,data=tb,op0=A.multiply,operand0=-F,op1=A.add,operand1=X)
				if k==0:E.tensor_tensor(dst=I,data1=nll,data2=R[:,0,:],op=A.multiply,engine=BG)
				else:E.tensor_tensor(dst=nll,data1=nll,data2=R[:,k,:],op=A.multiply,engine=BG);E.tensor_tensor(dst=I,data1=I,data2=nll,op=A.add,engine=BG)
			w(dst=S[:,D:H],src=I)
		return S,T
	@K.jit
	def Ix(x3,tg3,rw3,lz2,g2,softcap):
		N=softcap;H=x3;B,S,I=H.shape;K=tg3.shape[0];D=A.float32;T=A.ndarray((B,S,I),dtype=H.dtype,buffer=A.shared_hbm);F,J=Do(S);n=J-F
		if n>0:
			O=A.ndarray((B,K,n),dtype=D,buffer=A.sbuf);C=A.ndarray((B,K,n),dtype=D,buffer=A.sbuf)
			for k in L(K):w(dst=O[:,k,:],src=tg3[k,:,F:J]);w(dst=C[:,k,:],src=rw3[k,:,F:J])
			M=Bt(D,lz2[:,F:J],B,n);ab=Bt(D,g2[:,F:J],B,n);E.tensor_tensor(dst=ab,data1=ab,data2=C[:,0,:],op=A.multiply,engine=BG);sq=A.ndarray((B,n),dtype=D,buffer=A.sbuf);E.activation(dst=sq,op=A.sqrt,data=ab);E.tensor_scalar(dst=M,data=M,op0=A.multiply,operand0=-Q,op1=A.add,operand1=-N)
			if K>1:
				E.reciprocal(dst=C[:,0,:],data=C[:,0,:])
				for k in L(1,K):E.tensor_tensor(dst=C[:,k,:],data1=C[:,k,:],data2=C[:,0,:],op=A.multiply,engine=BG)
				G=A.ndarray((B,n),dtype=D,buffer=A.sbuf);E.tensor_scalar(dst=G,data=C[:,1,:],op0=A.add,operand0=Q)
				for k in L(2,K):E.tensor_tensor(dst=G,data1=G,data2=C[:,k,:],op=A.add,engine=BG)
				E.activation(dst=G,op=A.log,data=G);E.tensor_tensor(dst=M,data1=M,data2=G,op=A.add,engine=BG)
			U=Nd(B,I);P=AJ((B,I),H.dtype);V=AJ((B,I),D);W=AJ((B,I),D)
			for j in L(g(Av,n)):w(dst=P[j%r],src=H[:,F+j,:])
			for j in L(n):
				i=F+j;xb,t,q=P[j%r],V[j%r],W[j%r];R=j+Av
				if R<n:w(dst=P[R%r],src=H[:,F+R,:])
				E.activation(dst=t,op=A.tanh,data=xb,scale=Q/N);E.activation(dst=q,op=A.exp,data=t,scale=N,bias=M[:,j:j+1]);E.scalar_tensor_tensor(dst=q,data=U,op0=A.equal,operand0=O[:,0,j:j+1],op1=A.subtract,operand1=q)
				for k in L(1,K):Ne(xb,U,O[:,k,j:j+1]);E.scalar_tensor_tensor(dst=q,data=xb,op0=A.multiply,operand0=C[:,k,j:j+1],op1=A.add,operand1=q)
				E.activation(dst=t,op=A.square,data=t,scale=sq[:,j:j+1]);E.scalar_tensor_tensor(dst=xb,data=t,op0=A.subtract,operand0=ab[:,j:j+1],op1=A.multiply,operand1=q);w(dst=T[:,i,:],src=xb)
		return T
	Fn=I;Iy=C
except P as Iz:Fn=O;Iy=f"{W(Iz).__name__}: {Iz}"
I_=1e-05
Nf=2.**-7
Ng=.001
Nh=2.**-6
J0=2.**-24
lossk__MULTI_CHECKED=O
Bu=O
Fo=C
lossk__MULTI_CHECK=CA
Dp=[0,0]
def Ni(device):return device.type==a
def lossk_multi_available(device):return Fn and not Bu and not Bs and Ni(device)
def Dq(where,err):
	A=where;global Bu,Fo
	if not Bu:Bu=I;Fo=f"{A}: {W(err).__name__}: {err}";D(f"[nki_loss_head] mtp ship DISABLED after failure in {A}: {W(err).__name__}: {err} -- the run continues on the TEST FORM (one fused_loss pass per target)",flush=I)
def J1(x2,tg3,rw3,softcap):
	N=x2.shape[0];rows=C
	for k in L(tg3.shape[0]):r=lossk_fused_loss(x2,tg3[k].reshape(N),reduction=AW,softcap=softcap)*rw3[k].reshape(N);rows=r if rows is C else rows+r
	return rows
class Nj(torch.autograd.Function):
	@N
	def forward(ctx,x2,tg3,rw3,softcap):
		D=softcap;B=rw3;A=tg3;N,H=x2.shape;ctx.softcap=D;E=C
		try:F,E=Aw(Iw)(Ca(x2),A,B,G(D));rows=F.view(N);Dp[0]+=1
		except P as err:Dq(BS,err);E=C;rows=J1(x2,A,B,D)
		ctx.save_for_backward(x2,A,B,E);return rows
	@N
	def backward(ctx,grad_rows):
		B=grad_rows;A,D,E,F=ctx.saved_tensors;N,J=A.shape
		if F is not C and not Bu:
			try:g2=B.contiguous().view(u,N//u);K=Aw(Ix)(Ca(A),D,E,F,g2,G(ctx.softcap));Dp[1]+=1;return K.view(N,J),C,C,C
			except P as err:Dq(Am,err)
		with torch.enable_grad():H=A.detach().requires_grad_(I);rows=J1(H,D,E,ctx.softcap);gx,=torch.autograd.grad(rows,H,B)
		return gx,C,C,C
def lossk_fused_loss_multi(logits,tg,rw,softcap=Dn,ref=C):
	D=softcap;B=logits.shape[-1];K,N=tg.shape
	if not(lossk_multi_available(logits.device)and logits.dtype in(torch.bfloat16,torch.float32)and B==Il and N%u==0 and N>0 and logits.numel()==N*B and tg.dtype==torch.int32 and rw.dtype==torch.float32 and e(rw.shape)==(K,N)):return C
	A=logits.reshape(N,B)
	if not A.is_contiguous():A=A.contiguous()
	E=tg.contiguous().view(K,u,N//u);F=rw.contiguous().view(K,u,N//u)
	if not lossk__MULTI_CHECKED:
		lossk__multi_selfcheck(A,E,F,D,ref)
		if Bu:return C
	return Nj.apply(A,E,F,D).sum()
def Nk(N):
	A=N//u;h=64;blocks=[]
	for(a,b)in AO(((0,h),(A//2-h//2,A//2+h//2),(2*A+A//4,2*A+3*A//8),(5*A+5*A//8,5*A+3*A//4),(N//2-h,N//2+h),(N-h,N),(1024-h,1024+h))):
		a,b=i(0,a),g(N,b)
		if b<=a:continue
		if blocks and a<=blocks[-1][1]:blocks[-1]=blocks[-1][0],i(blocks[-1][1],b)
		else:blocks.append((a,b))
	return blocks
def J2(t2,blocks):return torch.cat([t2[a:b].cpu().double()for(a,b)in blocks])
def lossk__multi_selfcheck(x2,tg3,rw3,softcap,ref):
	p='self-check';R=rw3;J=softcap;H=tg3;global lossk__MULTI_CHECKED,lossk__MULTI_CHECK;lossk__MULTI_CHECKED=I
	try:
		N,E=x2.shape;K=H.shape[0];Y=N//u
		if ref is C:ref=[(H[k].reshape(N),R[k].reshape(N))for k in L(K)]
		assert V(ref)==K,f"ref has {V(ref)} targets, the ship tensors {K}";M=Ca(x2);blocks=Nk(N)
		with torch.no_grad():
			Z=torch.ones(u,Y,dtype=torch.float32,device=x2.device);q,a=Aw(Iw)(M,H,R,G(J));S,b=q.reshape(N).cpu().double(),a.reshape(N).cpu().double();c=Aw(Ix)(M,H,R,a,Z,G(J));A=J2(c.view(N,E),blocks);del c;T=torch.zeros(N,dtype=torch.float64);U=torch.zeros(N,dtype=torch.float64);F=torch.zeros_like(A);X=torch.zeros_like(A);O=C
			for(s,wk)in ref:
				d=Iv(s.reshape(N));scale=Q/N if wk is C else Q;g2=Z if wk is C else wk.reshape(u,Y).contiguous();t,e=Aw(Ir)(M,d,G(J),G(scale));r=t.reshape(N).cpu().double()
				if wk is not C:r=r*wk.reshape(N).cpu().double()
				T+=r;U+=r.abs()
				if O is C:O=e.reshape(N).cpu().double()
				f=Aw(Is)(M,d,e,g2,G(J),G(scale));g=J2(f.view(N,E),blocks);del f;F+=g;X+=g.abs()
		finite=z(torch.isfinite(S).all()and torch.isfinite(b).all()and torch.isfinite(A).all());h=G(((S-T).abs()/(I_*U+1e-30)).max());v=(E-1)*J0/(Q-(E-1)*J0);j=G(((b-O).abs()/(2.*v+1e-06*O.abs())).max());k=G((S.sum()-T.sum()).abs()/U.sum().clamp_min(1e-30));err=A-F;l=err.abs()/(Nf*1.1*(A.abs()+X)+1e-35);m=G(l.max());bias=G(err.sum())/i(G((A.abs()+X).sum()),1e-30);n=G(err.norm())/i(G(F.norm()),1e-30);x=G((A!=F).double().mean());ok=finite and h<=Q and j<=Q and k<=I_ and m<=Q and abs(bias)<=Ng and n<=Nh;o=f"K={K} rows={N} | loss rows {h:.2e} of tol, log z {j:.2e} of tol, total {k:.1e} | dlogits {A.shape[0]} rows x {E}: worst {m:.3f} of the SR-safe bound (1.1 x 2^-7 per component), bias {bias:.1e}, rel-fro {n:.1e}, {100*x:.2f}% differ"
		if not ok:w=B(l.argmax());o+=f" | worst at sampled row {w//E} col {w%E}: ship {G(A.reshape(-1)[w]):.6e} test form {G(F.reshape(-1)[w]):.6e} finite={finite}"
		lossk__MULTI_CHECK=o+(' -> PASS'if ok else' -> FAIL');D(f"[nki_loss_head] mtp ship self-check: {lossk__MULTI_CHECK}",flush=I)
		if not ok:Dq(p,Al(lossk__MULTI_CHECK))
	except P as err:lossk__MULTI_CHECK=f"ERROR {W(err).__name__}: {err} -> FAIL";D(f"[nki_loss_head] mtp ship self-check: {lossk__MULTI_CHECK}",flush=I);Dq(p,err)
def Nl():return f"fired_fwd={Dp[0]} fired_bwd={Dp[1]} selfcheck={lossk__MULTI_CHECK!r} disabled={Fo!r} kernels={"yes"if Fn else"NO ("+v(Iy)+")"}"
fused_loss=lossk_fused_loss
USE_NKI_LOSS_HEAD=lossk_USE_NKI_LOSS_HEAD
MTP=I
_MTP={AS:(O,)*2,CB:0,AV:[[0]*3 for _ in L(3)],'m':{},'cov':Y}
def mtp_set(p):
	u=Q-4.*p;w=(.5,.25*u)if p<.25 else(.5*(1+u),S)if p<.5 else(S,S);_MTP[AS],_MTP[CB]=e(v>0 for v in w),g(B(p*4),2);_MTP[AV][_MTP[CB]][0]+=1
	if any(_MTP[AS]):_MTP['w'].copy_(torch.tensor(w,dtype=torch.float32))
def mtp_terms(y,live):
	B,T=y.shape;key=B,T,y.device
	if key not in _MTP['m']:c=torch.arange(T,device=y.device).view(1,T);_MTP['m'][key]=[((c<T-j).float()/(B*T)).expand(B,T).contiguous()for j in(1,2)]
	m,w=_MTP['m'][key],_MTP['w'];return[(torch.cat((y[:,j+1:],y[:,T-j-1:]),1),w[j:j+1]*m[j])for j in(0,1)if live[j]]
MTP_SHIP=I
fused_loss_multi=lossk_fused_loss_multi
multi_available=lossk_multi_available
multi_status=Nl
_MTP.update(s={},scov=Y,ns=[0,0,0])
def mtp_ship(x,y):
	A='scov';B,T=y.shape;key=B,T,y.device
	if key not in _MTP[Bf]:_MTP[Bf][key]=torch.full((B,T),Q/(B*T),device=y.device)
	live=MTP,MTP;y32=y.to(torch.int32);terms=mtp_terms(y32,live);K=1+V(terms);tg=torch.stack([y32]+[yk for(yk,_)in terms]).view(K,B*T);rw=torch.stack([_MTP[Bf][key]]+[mk for(_,mk)in terms]).view(K,B*T);ref=C if _MTP[A]else[(y,C)]+mtp_terms(y,live);loss=fused_loss_multi(x,tg,rw,softcap=LOGIT_SOFTCAP,ref=ref)
	if not _MTP[A]:_MTP[A]=f"path={"ship"if loss is not C else"NEXT-TOKEN ONLY (ship kernel unavailable)"} K={K} rows={B*T} targets=next,t+2,t+3";D(f"mtp: ship first micro-step {_MTP[A]}",flush=I);bosx_arm(x,y)
	if loss is not C:
		n=_MTP[AV][_MTP[CB]]
		for j in L(2):n[j+1]+=_MTP[AS][j]
		_MTP[Em][_MTP[CB]]+=1
	return loss
BOS_EXCL='post'
_BOSX={AS:O,'armed':O,Cz:C,Bf:{},Cw:0,C_:0,CC:C,B6:C,'chk':C}
def bosx_rw(y):
	B,T=y.shape;key=B,T,y.device
	if key not in _BOSX[Bf]:_BOSX[Bf][key]=torch.full((B,T),Q/(B*T),device=y.device)
	y32=y.to(torch.int32);return y32,_BOSX[Bf][key]*(y32!=_BOSX[Cz])
def bosx_arm(x,y):
	global lossk__MULTI_CHECKED,lossk__MULTI_CHECK;N=y.numel();y32,rw=bosx_rw(y);k3,lossk__MULTI_CHECKED=lossk__MULTI_CHECK,O
	with torch.enable_grad():
		xd=x.detach().requires_grad_(I);l1=fused_loss_multi(xd,y32.view(1,N),rw.view(1,N),softcap=LOGIT_SOFTCAP,ref=[(y,rw)]);chk=lossk__MULTI_CHECK
		if l1 is C or not chk.endswith('-> PASS'):raise Al(f"bosx: the K=1 multi-target pair is unavailable or failed its self-check ({multi_status()}); refusing to run without it")
		l1.backward()
	z=B((rw==0).sum().item());del xd,l1;_BOSX.update(armed=I,chk=chk);lossk__MULTI_CHECK=k3;D(f"bosx: K=1 pair traced + self-checked at step 0 (rows {N}, {z} BOS targets at weight 0, bos_id {_BOSX[Cz]}): {chk}",flush=I)
def bosx_loss(x,y):
	if not _BOSX['armed']:raise Al('bosx: the K=1 pair was never traced before the clock (no ship micro-step at step 0)')
	N=y.numel();y32,rw=bosx_rw(y);loss=fused_loss_multi(x,y32.view(1,N),rw.view(1,N),softcap=LOGIT_SOFTCAP)
	if loss is C:
		_BOSX[C_]+=1
		if _BOSX[C_]==1:D(f"bosx: K=1 path DECLINED ({multi_status()}): the base loss runs from here",flush=I)
		return C
	_BOSX[Cw]+=1
	if _BOSX[B6]is C:_BOSX[B6]=B((rw==0).sum().item());D(f"bosx: K=1 path engaged at step {_BOSX[CC][0]} progress {_BOSX[CC][1]:.4f}: {_BOSX[B6]} of {N} rows at weight 0 (BOS targets) on its first micro-step",flush=I)
	return loss
import os,torch,torch.nn.functional as F
Fp=128
Nm=os.environ.get
Nn=G(2.**-23)
No=I
bheadk_USE_NKI_NORM2=I
Np=4
Nq='rows'
J3=B(Nm(AB,q))
BH=O
J4=C
def Dr(x):return F.rms_norm(x,(x.size(-1),))
def Fq(x,x0,l1,l2):x_new=l1.to(x.dtype)*x+l2.to(x.dtype)*x0;return x_new,Dr(x_new)
try:
	import nki as K,nki.language as A,nki.isa as E
	def d(dst,src):E.dma_copy(dst=dst,src=src)
	def Nr():
		if A.program_ndim()>0:return A.program_id(0),A.num_programs(0)
		return 0,1
	def Ds(NT):A,B=Nr();C,D=NT//B,NT%B;E=A*C+g(A,D);n=C+(1 if A<D else 0);return E,E+n
	def Dt(n):
		k=1
		for d in L(1,Np+1):
			if n%d==0:k=d
		return k
	def t(shape,dtype):return A.ndarray(shape,dtype=dtype,buffer=A.sbuf),A.ndarray(shape,dtype=dtype,buffer=A.sbuf)
	def AE(shape,dtype):return A.ndarray(shape,dtype=dtype,buffer=A.sbuf)
	def Du(lg,P):
		t=AE((P,1),A.float32)
		if lg.shape[0]==1:d(dst=t,src=lg.broadcast(0,P))
		else:d(dst=t,src=lg)
		return t
	def J5(dt):return A.float32
	def J6(dt):return A.float32
	def J7(xb,x0b,tb,xnb,l1t,l2t):E.activation(dst=tb,op=A.copy,data=x0b,scale=l2t);E.scalar_tensor_tensor(dst=xnb,data=xb,op0=A.multiply,operand0=l1t,op1=A.add,operand1=tb)
	def J8(xb,sq,ss,K):
		for k in L(K):E.activation_reduce(dst=sq,op=A.square,data=xb[:,k,:],reduce_op=A.add,reduce_res=ss[:,k:k+1])
	def J9(ss,inv_dst,C):E.tensor_scalar(dst=ss,data=ss,op0=A.multiply,operand0=Q/C,op1=A.add,operand1=Nn);E.tensor_scalar(dst=inv_dst,data=ss,op0=A.rsqrt,operand0=S)
	def JA(dst,src,inv,K):
		for k in L(K):E.tensor_scalar(dst=dst[:,k,:],data=src[:,k,:],op0=A.multiply,operand0=inv[:,k:k+1])
	def Dv(pb,dst,K):
		for k in L(K):E.activation_reduce(dst=pb[:,k,:],op=A.copy,data=pb[:,k,:],reduce_op=A.add,reduce_res=dst[:,k:k+1])
	def JB(P,K,C):return P,C
	def JC(dt):return dt
	@K.jit
	def Ns(x3,x03,l1g,l2g):
		B,E,C=x3.shape;dt=x3.dtype;G=A.float32;M=A.ndarray((B,E,C),dtype=dt,buffer=A.shared_hbm);N=A.ndarray((B,E,C),dtype=dt,buffer=A.shared_hbm);O=A.ndarray((B,E),dtype=G,buffer=A.shared_hbm);D,P=Ds(E);n=P-D
		if n>0:
			T=Du(l1g,B);U=Du(l2g,B);K=Dt(n);Q=n//K;V=J5(dt);H=AE((B,n),G);I=t((B,K,C),dt);J=t((B,K,C),dt);W=t((B,K,C),V);X=t((B,K,C),dt);Y=t((B,K,C),dt);Z=t(JB(B,K,C),JC(dt));d(dst=I[0],src=x3[:,D:D+K,:]);d(dst=J[0],src=x03[:,D:D+K,:])
			for j in L(Q):
				a=D+j*K;s=j%2;xb,b,tb,F,yb,sq=I[s],J[s],W[s],X[s],Y[s],Z[s]
				if j+1<Q:R=(j+1)%2;d(dst=I[R],src=x3[:,a+K:a+2*K,:]);d(dst=J[R],src=x03[:,a+K:a+2*K,:])
				J7(xb,b,tb,F,T,U);S=AE((B,K),G);J8(F,sq,S,K);J9(S,H[:,j*K:(j+1)*K],C);JA(yb,F,H[:,j*K:(j+1)*K],K);d(dst=M[:,a:a+K,:],src=F);d(dst=N[:,a:a+K,:],src=yb)
			d(dst=O[:,D:P],src=H)
		return M,N,O
	@K.jit
	def Nt(x3,x03,gn3,gy3,inv2,l1g,l2g):
		B,I,D=x3.shape;dt=x3.dtype;F=A.float32;P=J6(dt);W=J5(dt);X=A.ndarray((B,I,D),dtype=dt,buffer=A.shared_hbm);Y=A.ndarray((B,I,D),dtype=dt,buffer=A.shared_hbm);r3=A.ndarray((2,B,I),dtype=F,buffer=A.shared_hbm);C,J=Ds(I);n=J-C
		if n>0:
			Z=Du(l1g,B);b=Du(l2g,B);K=Dt(n);c=n//K;e=AE((B,n),F);d(dst=e,src=inv2[:,C:J]);f=AE((B,n),F);g=AE((B,n),F);R=t((B,K,D),dt);S=t((B,K,D),dt);M=t((B,K,D),dt);T=t((B,K,D),dt);h=t((B,K,D),W);i=t((B,K,D),dt);v=t((B,K,D),F);w=t((B,K,D),P)
			if W!=P and Nq!='matmul':l=t((B,K,D),P)
			else:l=h
			x,y=i,M;d(dst=R[0],src=x3[:,C:C+K,:]);d(dst=S[0],src=x03[:,C:C+K,:]);d(dst=M[0],src=gn3[:,C:C+K,:]);d(dst=T[0],src=gy3[:,C:C+K,:])
			for j in L(c):
				a=C+j*K;s=j%2;xb,m,z,o=R[s],S[s],M[s],T[s];tb,U,G,N,p=h[s],i[s],v[s],w[s],l[s];q,r=x[s],y[s]
				if j+1<c:O=(j+1)%2;d(dst=R[O],src=x3[:,a+K:a+2*K,:]);d(dst=S[O],src=x03[:,a+K:a+2*K,:]);d(dst=M[O],src=gn3[:,a+K:a+2*K,:]);d(dst=T[O],src=gy3[:,a+K:a+2*K,:])
				V=e[:,j*K:(j+1)*K];J7(xb,m,tb,U,Z,b);E.tensor_tensor(dst=N,data1=o,data2=U,op=A.multiply);u=AE((B,K),F);Dv(N,u,K);H=AE((B,K),F);E.tensor_tensor(dst=H,data1=V,data2=V,op=A.multiply);E.tensor_tensor(dst=H,data1=H,data2=u,op=A.multiply);E.tensor_scalar(dst=H,data=H,op0=A.multiply,operand0=-Q/D)
				for k in L(K):E.scalar_tensor_tensor(dst=G[:,k,:],data=U[:,k,:],op0=A.multiply,operand0=H[:,k:k+1],op1=A.add,operand1=o[:,k,:]);E.scalar_tensor_tensor(dst=G[:,k,:],data=G[:,k,:],op0=A.multiply,operand0=V[:,k:k+1],op1=A.add,operand1=z[:,k,:])
				E.tensor_scalar(dst=q,data=G,op0=A.multiply,operand0=Z);E.tensor_scalar(dst=r,data=G,op0=A.multiply,operand0=b);d(dst=X[:,a:a+K,:],src=q);d(dst=Y[:,a:a+K,:],src=r);E.tensor_tensor(dst=N,data1=G,data2=xb,op=A.multiply);E.tensor_tensor(dst=p,data1=G,data2=m,op=A.multiply);Dv(N,f[:,j*K:(j+1)*K],K);Dv(p,g[:,j*K:(j+1)*K],K)
			d(dst=r3[0,:,C:J],src=f);d(dst=r3[1,:,C:J],src=g)
		return X,Y,r3
	@K.jit
	def Nu(x3):
		B,E,C=x3.shape;dt=x3.dtype;F=A.float32;I=A.ndarray((B,E,C),dtype=dt,buffer=A.shared_hbm);J=A.ndarray((B,E),dtype=F,buffer=A.shared_hbm);D,M=Ds(E);n=M-D
		if n>0:
			K=Dt(n);N=n//K;G=AE((B,n),F);H=t((B,K,C),dt);P=t((B,K,C),dt);Q=t(JB(B,K,C),JC(dt));d(dst=H[0],src=x3[:,D:D+K,:])
			for j in L(N):
				a=D+j*K;s=j%2;xb,yb,sq=H[s],P[s],Q[s]
				if j+1<N:d(dst=H[(j+1)%2],src=x3[:,a+K:a+2*K,:])
				O=AE((B,K),F);J8(xb,sq,O,K);J9(O,G[:,j*K:(j+1)*K],C);JA(yb,xb,G[:,j*K:(j+1)*K],K);d(dst=I[:,a:a+K,:],src=yb)
			d(dst=J[:,D:M],src=G)
		return I,J
	@K.jit
	def JD(x3,gy3,inv2):
		B,M,F=x3.shape;dt=x3.dtype;G=A.float32;Y=J6(dt);N=A.ndarray((B,M,F),dtype=dt,buffer=A.shared_hbm);D,O=Ds(M);n=O-D
		if n>0:
			K=Dt(n);P=n//K;R=AE((B,n),G);d(dst=R,src=inv2[:,D:O]);J=t((B,K,F),dt);H=t((B,K,F),dt);Z=t((B,K,F),Y);b=t((B,K,F),G);c=H;d(dst=J[0],src=x3[:,D:D+K,:]);d(dst=H[0],src=gy3[:,D:D+K,:])
			for j in L(P):
				a=D+j*K;s=j%2;xb,S,T,U,V=J[s],H[s],Z[s],b[s],c[s]
				if j+1<P:W=(j+1)%2;d(dst=J[W],src=x3[:,a+K:a+2*K,:]);d(dst=H[W],src=gy3[:,a+K:a+2*K,:])
				I=R[:,j*K:(j+1)*K];E.tensor_tensor(dst=T,data1=S,data2=xb,op=A.multiply);X=AE((B,K),G);Dv(T,X,K);C=AE((B,K),G);E.tensor_tensor(dst=C,data1=I,data2=I,op=A.multiply);E.tensor_tensor(dst=C,data1=C,data2=I,op=A.multiply);E.tensor_tensor(dst=C,data1=C,data2=X,op=A.multiply);E.tensor_scalar(dst=C,data=C,op0=A.multiply,operand0=-Q/F)
				for k in L(K):E.activation(dst=U[:,k,:],op=A.copy,data=S[:,k,:],scale=I[:,k:k+1])
				for k in L(K):E.scalar_tensor_tensor(dst=V[:,k,:],data=xb[:,k,:],op0=A.multiply,operand0=C[:,k:k+1],op1=A.add,operand1=U[:,k,:])
				d(dst=N[:,a:a+K,:],src=V)
		return N
	Fr=I
except P:Fr=O
def Dw(where,err):
	A=where;global BH,J4
	if not BH:BH=I;J4=f"{A}: {W(err).__name__}: {err}";D(f"[nki_block_head] kernel disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
def Dx(kernel):A=kernel;return A[J3]if J3==2 else A
def Aj(x2):N,A=x2.shape;return x2.view(Fp,N//Fp,A)
def JE(l):
	if l.dtype!=torch.float32:l=l.float()
	v=l.reshape(1,1);return v
def Dy(lg):return lg.reshape(-1)[0:1]
class Nv(torch.autograd.Function):
	@N
	def forward(ctx,x2,x02,l1,l2):
		A=x02;N,E=x2.shape;B=JE(l1);D=JE(l2);F=C
		try:H,I,F=Dx(Ns)(Aj(x2),Aj(A),B,D);x_new,G=H.view(N,E),I.view(N,E)
		except P as err:
			Dw(BS,err)
			with torch.no_grad():x_new,G=Fq(x2,A,Dy(B),Dy(D))
		ctx.save_for_backward(x2,A,B,D,F);return x_new,G
	@N
	def backward(ctx,gn,gy):
		A,H,J,K,L=ctx.saved_tensors;N,E=A.shape
		if L is not C and not BH:
			try:
				B=gn.reshape(N,E);D=gy.reshape(N,E)
				if B.dtype!=A.dtype:B=B.to(A.dtype)
				if D.dtype!=A.dtype:D=D.to(A.dtype)
				if not B.is_contiguous():B=B.contiguous()
				if not D.is_contiguous():D=D.contiguous()
				out=Dx(Nt)(Aj(A),Aj(H),Aj(B),Aj(D),L,J,K);T,U,V=out;M=V.view(2,-1).sum(dim=1);F,G=M[0:1],M[1:2];return T.view(N,E),U.view(N,E),F,G
			except P as err:Dw(Am,err)
		with torch.enable_grad():O=A.detach().requires_grad_(I);Q=H.detach().requires_grad_(I);R=Dy(J).detach().requires_grad_(I);S=Dy(K).detach().requires_grad_(I);x_new,W=Fq(O,Q,R,S);gx,X,F,G=torch.autograd.grad((x_new,W),(O,Q,R,S),(gn,gy))
		return gx,X,F,G
class Nw(torch.autograd.Function):
	@N
	def forward(ctx,x2):
		N,B=x2.shape;A=C
		try:D,A=Dx(Nu)(Aj(x2));y=D.view(N,B)
		except P as err:
			Dw('rms forward',err)
			with torch.no_grad():y=Dr(x2)
		ctx.save_for_backward(x2,A);return y
	@N
	def backward(ctx,gy):
		B,D=ctx.saved_tensors;N,E=B.shape
		if D is not C and not BH:
			try:
				A=gy.reshape(N,E)
				if A.dtype!=B.dtype:A=A.to(B.dtype)
				if not A.is_contiguous():A=A.contiguous()
				G=Dx(JD)(Aj(B),Aj(A),D);return G.view(N,E)
			except P as err:Dw('rms backward',err)
		with torch.enable_grad():F=B.detach().requires_grad_(I);gx,=torch.autograd.grad(Dr(F),F,gy)
		return gx
def JF(x):N=x.numel()//x.shape[-1];return N>0 and N%Fp==0
def bheadk_block_head(x,x0,l1,l2):
	D=Fr and not BH and x.device.type==a and x.dim()>=2 and x.dtype in(torch.bfloat16,torch.float32)and x0.dtype==x.dtype and x0.shape==x.shape and l1.numel()==1 and l2.numel()==1 and JF(x)
	if not D:return Fq(x,x0,l1,l2)
	C=x.shape[-1];N=x.numel()//C;A=x.reshape(N,C);B=x0.reshape(N,C)
	if not A.is_contiguous():A=A.contiguous()
	if not B.is_contiguous():B=B.contiguous()
	x_new,E=Nv.apply(A,B,l1,l2);return x_new.view(x.shape),E.view(x.shape)
def bheadk_rmsnorm_fast(x):
	B=Fr and not BH and x.device.type==a and x.dim()>=2 and x.dtype in(torch.bfloat16,torch.float32)and JF(x)
	if not B:return Dr(x)
	C=x.shape[-1];A=x.reshape(-1,C)
	if not A.is_contiguous():A=A.contiguous()
	return Nw.apply(A).view(x.shape)
block_head=bheadk_block_head
USE_NKI_BLOCK_HEAD=No
USE_NKI_NORM2=bheadk_USE_NKI_NORM2
import os,torch,torch.nn.functional as F
try:Fs=bheadk_block_head;Ft=bheadk_rmsnorm_fast
except P:Fs=C;Ft=C
Fu=128
Nx=os.environ.get
JG=G(2.**-23)
Ny=I
Fv=4
Nz=Cm
N_=BX
O0=BX
O1='ts'
O2=I
Fw=B(Nx(AB,q))
O3=I
O4=BR
O5=AX
O6='unknown'
Fx={}
Dz=C
BI=O
JH=C
def BJ(x):return F.rms_norm(x,(x.size(-1),))
def D_(x,x0,l1,l2):x_new=l1.to(x.dtype)*x+l2.to(x.dtype)*x0;return x_new,BJ(x_new)
def E0(x,x0,skip,l1,l2,l3):s=x+l3.to(x.dtype)*skip;return D_(s,x0,l1,l2)
def O7(x_new,attn_out):A=x_new+attn_out;return A,BJ(A)
def JI(x,x0,l1,l2):
	if Fs is not C:return Fs(x,x0,l1,l2)
	return D_(x,x0,l1,l2)
def O8(x):
	if Ft is not C:return Ft(x)
	return BJ(x)
try:
	import nki as K,nki.language as A,nki.isa as E;Fy=E.engine.unknown
	def M(dst,src):E.dma_copy(dst=dst,src=src)
	def O9():
		if A.program_ndim()>0:return A.program_id(0),A.num_programs(0)
		return 0,1
	def Bv(NT):A,B=O9();C,D=NT//B,NT%B;E=A*C+g(A,D);n=C+(1 if A<D else 0);return E,E+n
	def E1(n):
		k=1
		for d in L(1,Fv+1):
			if n%d==0:k=d
		return k
	def U(shape,dtype):return A.ndarray(shape,dtype=dtype,buffer=A.sbuf),A.ndarray(shape,dtype=dtype,buffer=A.sbuf)
	def b(shape,dtype):return A.ndarray(shape,dtype=dtype,buffer=A.sbuf)
	def AK(lg,P):
		t=b((P,1),A.float32)
		if lg.shape[0]==1:M(dst=t,src=lg.broadcast(0,P))
		else:M(dst=t,src=lg)
		return t
	def E2(dt):return A.float32
	def JJ(dt):return A.float32
	def Ax(xb,x0b,tb,xnb,l1t,l2t):E.activation(dst=tb,op=A.copy,data=x0b,scale=l2t);E.scalar_tensor_tensor(dst=xnb,data=xb,op0=A.multiply,operand0=l1t,op1=A.add,operand1=tb)
	def Bw(dst,skb,xb,l3t):E.scalar_tensor_tensor(dst=dst,data=skb,op0=A.multiply,operand0=l3t,op1=A.add,operand1=xb)
	def Fz(xb,sq,ss,K):
		for k in L(K):E.activation_reduce(dst=sq,op=A.square,data=xb[:,k,:],reduce_op=A.add,reduce_res=ss[:,k:k+1])
	def F_(ss,inv_dst,C):E.tensor_scalar(dst=ss,data=ss,op0=A.multiply,operand0=Q/C,op1=A.add,operand1=JG);E.tensor_scalar(dst=inv_dst,data=ss,op0=A.rsqrt,operand0=S)
	def G0(dst,src,inv,K):
		for k in L(K):E.tensor_scalar(dst=dst[:,k,:],data=src[:,k,:],op0=A.multiply,operand0=inv[:,k:k+1])
	def BK(pb,dst,K):
		for k in L(K):E.activation_reduce(dst=pb[:,k,:],op=A.copy,data=pb[:,k,:],reduce_op=A.add,reduce_res=dst[:,k:k+1])
	def G1(P,K,C):return P,C
	def G2(dt):return dt
	@K.jit
	def OA(x3,x03,sk3,l1g,l2g,l3g):
		B,E,C=x3.shape;dt=x3.dtype;F=A.float32;P=A.ndarray((B,E,C),dtype=dt,buffer=A.shared_hbm);Q=A.ndarray((B,E,C),dtype=dt,buffer=A.shared_hbm);R=A.ndarray((B,E),dtype=F,buffer=A.shared_hbm);D,S=Bv(E);n=S-D
		if n>0:
			X=AK(l1g,B);Y=AK(l2g,B);Z=AK(l3g,B);K=E1(n);T=n//K;c=E2(dt);H=b((B,n),F);I=U((B,K,C),dt);J=U((B,K,C),dt);N=U((B,K,C),dt);d=U((B,K,C),F);e=U((B,K,C),c);f=U((B,K,C),dt);g=U((B,K,C),dt);h=U(G1(B,K,C),G2(dt));M(dst=I[0],src=x3[:,D:D+K,:]);M(dst=J[0],src=x03[:,D:D+K,:]);M(dst=N[0],src=sk3[:,D:D+K,:])
			for j in L(T):
				a=D+j*K;s=j%2;xb,i,k,V,tb,G,yb,sq=I[s],J[s],N[s],d[s],e[s],f[s],g[s],h[s]
				if j+1<T:O=(j+1)%2;M(dst=I[O],src=x3[:,a+K:a+2*K,:]);M(dst=J[O],src=x03[:,a+K:a+2*K,:]);M(dst=N[O],src=sk3[:,a+K:a+2*K,:])
				Bw(V,k,xb,Z);Ax(V,i,tb,G,X,Y);W=b((B,K),F);Fz(G,sq,W,K);F_(W,H[:,j*K:(j+1)*K],C);G0(yb,G,H[:,j*K:(j+1)*K],K);M(dst=P[:,a:a+K,:],src=G);M(dst=Q[:,a:a+K,:],src=yb)
			M(dst=R[:,D:S],src=H)
		return P,Q,R
	@K.jit
	def OB(x3,x03,sk3,gn3,gy3,inv2,l1g,l2g,l3g):
		B,J,D=x3.shape;dt=x3.dtype;F=A.float32;T=JJ(dt);e=E2(dt);f=A.ndarray((B,J,D),dtype=dt,buffer=A.shared_hbm);g=A.ndarray((B,J,D),dtype=dt,buffer=A.shared_hbm);h=A.ndarray((B,J,D),dtype=dt,buffer=A.shared_hbm);r3=A.ndarray((3,B,J),dtype=F,buffer=A.shared_hbm);C,N=Bv(J);n=N-C
		if n>0:
			P=AK(l1g,B);i=AK(l2g,B);V=AK(l3g,B);l=b((B,1),F);E.tensor_tensor(dst=l,data1=P,data2=V,op=A.multiply);K=E1(n);m=n//K;o=b((B,n),F);M(dst=o,src=inv2[:,C:N]);p=b((B,n),F);W=b((B,n),F);q=b((B,n),F);r=b((B,n),F);t=b((B,n),F);X=U((B,K,D),dt);R=U((B,K,D),dt);Y=U((B,K,D),dt);S=U((B,K,D),dt);Z=U((B,K,D),dt);u=U((B,K,D),e);v=U((B,K,D),dt);A5=U((B,K,D),F);A6=U((B,K,D),T)
			if e!=T:w=U((B,K,D),T)
			else:w=u
			A7,A8,A9=v,S,R;M(dst=X[0],src=x3[:,C:C+K,:]);M(dst=R[0],src=x03[:,C:C+K,:]);M(dst=Y[0],src=sk3[:,C:C+K,:]);M(dst=S[0],src=gn3[:,C:C+K,:]);M(dst=Z[0],src=gy3[:,C:C+K,:])
			for j in L(m):
				a=C+j*K;s=j%2;xb,x,y,AA,z=X[s],R[s],Y[s],S[s],Z[s];tb,c,G,H,A0=u[s],v[s],A5[s],A6[s],w[s];A1,A2,A3=A7[s],A8[s],A9[s]
				if j+1<m:O=(j+1)%2;M(dst=X[O],src=x3[:,a+K:a+2*K,:]);M(dst=R[O],src=x03[:,a+K:a+2*K,:]);M(dst=Y[O],src=sk3[:,a+K:a+2*K,:]);M(dst=S[O],src=gn3[:,a+K:a+2*K,:]);M(dst=Z[O],src=gy3[:,a+K:a+2*K,:])
				d=o[:,j*K:(j+1)*K];Bw(G,y,xb,V);Ax(G,x,tb,c,P,i);E.tensor_tensor(dst=H,data1=z,data2=c,op=A.multiply);A4=b((B,K),F);BK(H,A4,K);I=b((B,K),F);E.tensor_tensor(dst=I,data1=d,data2=d,op=A.multiply);E.tensor_tensor(dst=I,data1=I,data2=A4,op=A.multiply);E.tensor_scalar(dst=I,data=I,op0=A.multiply,operand0=-Q/D)
				for k in L(K):E.scalar_tensor_tensor(dst=G[:,k,:],data=c[:,k,:],op0=A.multiply,operand0=I[:,k:k+1],op1=A.add,operand1=z[:,k,:]);E.scalar_tensor_tensor(dst=G[:,k,:],data=G[:,k,:],op0=A.multiply,operand0=d[:,k:k+1],op1=A.add,operand1=AA[:,k,:])
				E.tensor_tensor(dst=H,data1=G,data2=xb,op=A.multiply);E.tensor_tensor(dst=A0,data1=G,data2=x,op=A.multiply);BK(H,p[:,j*K:(j+1)*K],K);BK(A0,r[:,j*K:(j+1)*K],K);E.tensor_tensor(dst=H,data1=G,data2=y,op=A.multiply);BK(H,W[:,j*K:(j+1)*K],K);E.tensor_scalar(dst=A1,data=G,op0=A.multiply,operand0=P);E.tensor_scalar(dst=A2,data=G,op0=A.multiply,operand0=i);E.tensor_scalar(dst=A3,data=G,op0=A.multiply,operand0=l);M(dst=f[:,a:a+K,:],src=A1);M(dst=g[:,a:a+K,:],src=A2);M(dst=h[:,a:a+K,:],src=A3)
			E.scalar_tensor_tensor(dst=q,data=W,op0=A.multiply,operand0=V,op1=A.add,operand1=p);E.tensor_scalar(dst=t,data=W,op0=A.multiply,operand0=P);M(dst=r3[0,:,C:N],src=q);M(dst=r3[1,:,C:N],src=r);M(dst=r3[2,:,C:N],src=t)
		return f,g,h,r3
	@K.jit
	def JK(xn3,ao3):
		F=xn3;B,G,C=F.shape;dt=F.dtype;I=A.float32;P=A.ndarray((B,G,C),dtype=dt,buffer=A.shared_hbm);Q=A.ndarray((B,G,C),dtype=dt,buffer=A.shared_hbm);R=A.ndarray((B,G),dtype=I,buffer=A.shared_hbm);D,S=Bv(G);n=S-D
		if n>0:
			K=E1(n);T=n//K;J=b((B,n),I);N=U((B,K,C),dt);O=U((B,K,C),dt);X=U((B,K,C),dt);Y=U((B,K,C),dt);Z=U(G1(B,K,C),G2(dt));M(dst=N[0],src=F[:,D:D+K,:]);M(dst=O[0],src=ao3[:,D:D+K,:])
			for j in L(T):
				a=D+j*K;s=j%2;xb,ab,H,yb,sq=N[s],O[s],X[s],Y[s],Z[s]
				if j+1<T:V=(j+1)%2;M(dst=N[V],src=F[:,a+K:a+2*K,:]);M(dst=O[V],src=ao3[:,a+K:a+2*K,:])
				E.tensor_tensor(dst=H,data1=xb,data2=ab,op=A.add,engine=Fy);W=b((B,K),I);Fz(H,sq,W,K);F_(W,J[:,j*K:(j+1)*K],C);G0(yb,H,J[:,j*K:(j+1)*K],K);M(dst=P[:,a:a+K,:],src=H);M(dst=Q[:,a:a+K,:],src=yb)
			M(dst=R[:,D:S],src=J)
		return P,Q,R
	@K.jit
	def OC(s3,gy3,ge3,inv2):
		B,P,F=s3.shape;dt=s3.dtype;G=A.float32;d=JJ(dt);R=A.ndarray((B,P,F),dtype=dt,buffer=A.shared_hbm);C,S=Bv(P);n=S-C
		if n>0:
			K=E1(n);T=n//K;V=b((B,n),G);M(dst=V,src=inv2[:,C:S]);J=U((B,K,F),dt);H=U((B,K,F),dt);N=U((B,K,F),dt);e=U((B,K,F),d);f=U((B,K,F),G);g=H;M(dst=J[0],src=s3[:,C:C+K,:]);M(dst=H[0],src=gy3[:,C:C+K,:]);M(dst=N[0],src=ge3[:,C:C+K,:])
			for j in L(T):
				a=C+j*K;s=j%2;W,X,h,Y,Z,gb=J[s],H[s],N[s],e[s],f[s],g[s]
				if j+1<T:O=(j+1)%2;M(dst=J[O],src=s3[:,a+K:a+2*K,:]);M(dst=H[O],src=gy3[:,a+K:a+2*K,:]);M(dst=N[O],src=ge3[:,a+K:a+2*K,:])
				I=V[:,j*K:(j+1)*K];E.tensor_tensor(dst=Y,data1=X,data2=W,op=A.multiply);c=b((B,K),G);BK(Y,c,K);D=b((B,K),G);E.tensor_tensor(dst=D,data1=I,data2=I,op=A.multiply);E.tensor_tensor(dst=D,data1=D,data2=I,op=A.multiply);E.tensor_tensor(dst=D,data1=D,data2=c,op=A.multiply);E.tensor_scalar(dst=D,data=D,op0=A.multiply,operand0=-Q/F)
				for k in L(K):E.scalar_tensor_tensor(dst=Z[:,k,:],data=X[:,k,:],op0=A.multiply,operand0=I[:,k:k+1],op1=A.add,operand1=h[:,k,:])
				for k in L(K):E.scalar_tensor_tensor(dst=gb[:,k,:],data=W[:,k,:],op0=A.multiply,operand0=D[:,k:k+1],op1=A.add,operand1=Z[:,k,:])
				M(dst=R[:,a:a+K,:],src=gb)
		return R
	Cb=I
except P:Cb=O
def E3(where,err):
	A=where;global BI,JH
	if not BI:BI=I;JH=f"{A}: {W(err).__name__}: {err}";D(f"[nki_megablock] kernel disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
def BL(kernel):A=kernel;return A[Fw]if Fw==2 else A
def T(x2):N,A=x2.shape;return x2.view(Fu,N//Fu,A)
def Bx(l):
	if l.dtype!=torch.float32:l=l.float()
	v=l.reshape(1,1);return v
def AL(lg):return lg.reshape(-1)[0:1]
def Ay(g,like):
	A=like;N,B=A.shape
	if g is C:return torch.zeros_like(A)
	g2=g.reshape(N,B)
	if g2.dtype!=A.dtype:g2=g2.to(A.dtype)
	if not g2.is_contiguous():g2=g2.contiguous()
	return g2
class OD(torch.autograd.Function):
	@N
	def forward(ctx,x2,x02,sk2,l1,l2,l3):
		B=sk2;A=x02;N,G=x2.shape;D,E,F=Bx(l1),Bx(l2),Bx(l3);H=C
		try:J,K,H=BL(OA)(T(x2),T(A),T(B),D,E,F);x_new,I=J.view(N,G),K.view(N,G)
		except P as err:
			E3('head_skip forward',err)
			with torch.no_grad():x_new,I=E0(x2,A,B,AL(D),AL(E),AL(F))
		ctx.save_for_backward(x2,A,B,D,E,F,H);return x_new,I
	@N
	def backward(ctx,gn,gy):
		A,E,F,G,H,J,K=ctx.saved_tensors;N,B=A.shape
		if K is not C and not BI:
			try:V,W=Ay(gn,A),Ay(gy,A);X,Y,Z,r3=BL(OB)(T(A),T(E),T(F),T(V),T(W),K,G,H,J);D=r3.view(3,-1).sum(dim=1);return X.view(N,B),Y.view(N,B),Z.view(N,B),D[0:1],D[1:2],D[2:3]
			except P as err:E3('head_skip backward',err)
		with torch.enable_grad():L=A.detach().requires_grad_(I);M=E.detach().requires_grad_(I);O=F.detach().requires_grad_(I);Q=AL(G).detach().requires_grad_(I);R=AL(H).detach().requires_grad_(I);S=AL(J).detach().requires_grad_(I);x_new,U=E0(L,M,O,Q,R,S);a=torch.zeros_like(x_new)if gn is C else gn;b=torch.zeros_like(U)if gy is C else gy;c=torch.autograd.grad((x_new,U),(L,M,O,Q,R,S),(a,b))
		return c
class OE(torch.autograd.Function):
	@N
	def forward(ctx,xn2,ao2):
		N,B=xn2.shape;D=C
		try:E,F,D=BL(JK)(T(xn2),T(ao2));A,y=E.view(N,B),F.view(N,B)
		except P as err:
			E3('tail forward',err)
			with torch.no_grad():A,y=O7(xn2,ao2)
		ctx.save_for_backward(A,D);return A,y
	@N
	def backward(ctx,ge,gy):
		A,B=ctx.saved_tensors;N,F=A.shape
		if B is not C and not BI:
			try:G,H=Ay(ge,A),Ay(gy,A);J=BL(OC)(T(A),T(H),T(G),B);g=J.view(N,F);return g,g
			except P as err:E3('tail backward',err)
		with torch.enable_grad():D=A.detach().requires_grad_(I);K=torch.zeros_like(A)if gy is C else gy;E,=torch.autograd.grad(BJ(D),D,K)
		g=E if ge is C else E+ge;return g,g
def E4(x):N=x.numel()//x.shape[-1];return N>0 and N%Fu==0
def E5(x):return x.dtype in(torch.bfloat16,torch.float32)
def A4(x,C):
	A=x.reshape(-1,C)
	if not A.is_contiguous():A=A.contiguous()
	return A
def mblk_head_skip(x,x0,skip,l1,l2,l3):
	if skip is C:return JI(x,x0,l1,l2)
	B=Cb and not BI and x.device.type==a and x.dim()>=2 and E5(x)and x0.dtype==x.dtype and skip.dtype==x.dtype and x0.shape==x.shape and skip.shape==x.shape and l1.numel()==1 and l2.numel()==1 and l3.numel()==1 and E4(x)
	if not B:return JI(x+l3.to(x.dtype)*skip,x0,l1,l2)
	A=x.shape[-1];x_new,D=OD.apply(A4(x,A),A4(x0,A),A4(skip,A),l1,l2,l3);return x_new.view(x.shape),D.view(x.shape)
def mblk_tail_norm2(x_new,attn_out):
	C=Cb and not BI and x_new.device.type==a and x_new.dim()>=2 and E5(x_new)and attn_out.dtype==x_new.dtype and attn_out.shape==x_new.shape and E4(x_new)
	if not C:A=x_new+attn_out;return A,O8(A)
	B=x_new.shape[-1];A,y=OE.apply(A4(x_new,B),A4(attn_out,B));return A.view(x_new.shape),y.view(x_new.shape)
OF=I
OG=B(v(Fv))
OH=B(v(200*1024))
o,c,h,p=0,1,2,3
class OI:
	def __repr__(self):return'SELF_SKIP'
mblk_SELF_SKIP=OI()
BM=O
JL=C
def E6(where,err):
	A=where;global BM,JL
	if not BM:BM=I;JL=f"{A}: {W(err).__name__}: {err}";D(f"[nki_megablock] phase-2 kernels disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
def JM(mode,s1,mo,x0,sk,l1,l2,l3):
	B=mode
	if B==o:x_new,A=D_(x0,x0,l1,l2);return x_new,A,x0
	x_in=s1+mo
	if B==c:x_new,A=D_(x_in,x0,l1,l2)
	elif B==h:x_new,A=E0(x_in,x0,x_in,l1,l2,l3)
	else:x_new,A=E0(x_in,x0,sk,l1,l2,l3)
	return x_new,A,x_in
E7=O
try:
	if not Cb:raise ImportError(KM)
	try:Cc=JD
	except P:Cc=C
	def OJ(dt):
		if dt==A.float32:return 4
		return 2
	def JN(n,C,dt,n_in,n_ext,fwd,mode):
		A=OJ(dt);k=1
		for d in L(1,OG+1):
			if n%d==0:
				if fwd:B=d*C*(A*n_in+4+4+A+A)+C*A
				else:B=d*C*(A*(n_in+2)+4+A+4);B=B+d*C*4
				D=2*B+n_ext*d*C*A+4096
				if D<=OH:k=d
		return k
	def G3(ub,other,pb,dst_block,K):E.tensor_tensor(dst=pb,data1=ub,data2=other,op=A.multiply);BK(pb,dst_block,K)
	def Cd(dst,ub,lt,extb):
		if extb is C:E.tensor_scalar(dst=dst,data=ub,op0=A.multiply,operand0=lt)
		else:E.scalar_tensor_tensor(dst=dst,data=ub,op0=A.multiply,operand0=lt,op1=A.add,operand1=extb)
	def JO(mode):
		if mode==o:return 1
		if mode==p:return 4
		return 3
	def Ce(mode,emit,s13,mo3,x03,sk3,l1g,l2g,l3g,xn3,y3,inv2,xin3):
		H=x03;C=mode;B,f,D=H.shape;dt=H.dtype;P=A.float32;F,Y=Bv(f);n=Y-F
		if n>0:
			Q=AK(l1g,B);R=AK(l2g,B)
			if C>=h:Z=AK(l3g,B)
			K=JN(n,D,dt,JO(C),0,I,C);d=n//K;g=E2(dt);S=b((B,n),P);T=U((B,K,D),dt)
			if C>=c:V=U((B,K,D),dt);W=U((B,K,D),dt)
			if C==p:X=U((B,K,D),dt)
			if C>=h:J=U((B,K,D),P)
			i=U((B,K,D),g);k=U((B,K,D),dt);l=U((B,K,D),dt);m=U(G1(B,K,D),G2(dt));M(dst=T[0],src=H[:,F:F+K,:])
			if C>=c:M(dst=V[0],src=s13[:,F:F+K,:]);M(dst=W[0],src=mo3[:,F:F+K,:])
			if C==p:M(dst=X[0],src=sk3[:,F:F+K,:])
			for j in L(d):
				a=F+j*K;s=j%2;N,tb,G,yb,sq=T[s],i[s],k[s],l[s],m[s]
				if j+1<d:
					O=(j+1)%2;M(dst=T[O],src=H[:,a+K:a+2*K,:])
					if C>=c:M(dst=V[O],src=s13[:,a+K:a+2*K,:]);M(dst=W[O],src=mo3[:,a+K:a+2*K,:])
					if C==p:M(dst=X[O],src=sk3[:,a+K:a+2*K,:])
				if C==o:xb=N
				else:
					xb=V[s];E.tensor_tensor(dst=xb,data1=xb,data2=W[s],op=A.add,engine=Fy)
					if emit:M(dst=xin3[:,a:a+K,:],src=xb)
				if C==h:Bw(J[s],xb,xb,Z);Ax(J[s],N,tb,G,Q,R)
				elif C==p:Bw(J[s],X[s],xb,Z);Ax(J[s],N,tb,G,Q,R)
				else:Ax(xb,N,tb,G,Q,R)
				e=b((B,K),P);Fz(G,sq,e,K);F_(e,S[:,j*K:(j+1)*K],D);G0(yb,G,S[:,j*K:(j+1)*K],K);M(dst=xn3[:,a:a+K,:],src=G);M(dst=y3[:,a:a+K,:],src=yb)
			M(dst=inv2[:,F:Y],src=S)
	def Ad(mode,s13,mo3,x03,sk3,gn3,gy3,inv2,x0e3,xine3,l1g,l2g,l3g,gxin3,gx03,gsk3,r3):
		d=x03;D=mode;B,AC,I=d.shape;dt=d.dtype;H=A.float32;A2=E2(dt);m=x0e3 is not C;q=xine3 is not C;F,J=Bv(AC);n=J-F
		if n>0:
			N=AK(l1g,B);W=AK(l2g,B)
			if D==o:A3=b((B,1),H);E.tensor_tensor(dst=A3,data1=N,data2=W,op=A.add)
			if D>=h:X=AK(l3g,B);r=b((B,1),H);E.tensor_tensor(dst=r,data1=N,data2=X,op=A.multiply)
			if D==h:lst=b((B,1),H);E.tensor_tensor(dst=lst,data1=r,data2=N,op=A.add);A4=b((B,1),H);E.tensor_scalar(dst=A4,data=X,op0=A.add,operand0=Q)
			Y=0
			if m:Y=Y+1
			if q:Y=Y+1
			K=JN(n,I,dt,JO(D),Y,O,D);A5=n//K;A6=b((B,n),H);M(dst=A6,src=inv2[:,F:J]);R=b((B,n),H)
			if D>=c:e=b((B,n),H)
			if D==p:t=b((B,n),H)
			if D>=h:f=b((B,n),H);g=b((B,n),H)
			u=U((B,K,I),dt);v=U((B,K,I),dt);w=U((B,K,I),dt)
			if D>=c:x=U((B,K,I),dt);y=U((B,K,I),dt)
			if D==p:i=U((B,K,I),dt)
			AD=U((B,K,I),A2);AE=U((B,K,I),dt);AF=U((B,K,I),H);z=U((B,K,I),H)
			if m:l=b((B,K,I),dt)
			else:l=C
			if q:A0=b((B,K,I),dt)
			else:A0=C
			M(dst=u[0],src=d[:,F:F+K,:]);M(dst=v[0],src=gn3[:,F:F+K,:]);M(dst=w[0],src=gy3[:,F:F+K,:])
			if D>=c:M(dst=x[0],src=s13[:,F:F+K,:]);M(dst=y[0],src=mo3[:,F:F+K,:])
			if D==p:M(dst=i[0],src=sk3[:,F:F+K,:])
			for j in L(A5):
				a=F+j*K;s=j%2;S,Z,A7,tb,P,G=u[s],v[s],w[s],AD[s],AE[s],AF[s]
				if m:M(dst=l,src=x0e3[:,a:a+K,:])
				if q:M(dst=A0,src=xine3[:,a:a+K,:])
				if j+1<A5:
					T=(j+1)%2;M(dst=u[T],src=d[:,a+K:a+2*K,:]);M(dst=v[T],src=gn3[:,a+K:a+2*K,:]);M(dst=w[T],src=gy3[:,a+K:a+2*K,:])
					if D>=c:M(dst=x[T],src=s13[:,a+K:a+2*K,:]);M(dst=y[T],src=mo3[:,a+K:a+2*K,:])
					if D==p:M(dst=i[T],src=sk3[:,a+K:a+2*K,:])
				A1=A6[:,j*K:(j+1)*K]
				if D==o:xb=S
				else:xb=x[s];E.tensor_tensor(dst=xb,data1=xb,data2=y[s],op=A.add,engine=Fy)
				if D==h:Bw(G,xb,xb,X);Ax(G,S,tb,P,N,W)
				elif D==p:Bw(G,i[s],xb,X);Ax(G,S,tb,P,N,W)
				else:Ax(xb,S,tb,P,N,W)
				A8=z[s];A9=z[s]
				if A2==H:AA=tb
				else:AA=z[s]
				E.tensor_tensor(dst=A8,data1=A7,data2=P,op=A.multiply);AB=b((B,K),H);BK(A8,AB,K);V=b((B,K),H);E.tensor_tensor(dst=V,data1=A1,data2=A1,op=A.multiply);E.tensor_tensor(dst=V,data1=V,data2=AB,op=A.multiply);E.tensor_scalar(dst=V,data=V,op0=A.multiply,operand0=-Q/I)
				for k in L(K):E.scalar_tensor_tensor(dst=G[:,k,:],data=P[:,k,:],op0=A.multiply,operand0=V[:,k:k+1],op1=A.add,operand1=A7[:,k,:]);E.scalar_tensor_tensor(dst=G[:,k,:],data=G[:,k,:],op0=A.multiply,operand0=A1[:,k:k+1],op1=A.add,operand1=Z[:,k,:])
				G3(G,xb,A9,R[:,j*K:(j+1)*K],K)
				if D>=c:G3(G,S,AA,e[:,j*K:(j+1)*K],K)
				if D==p:G3(G,i[s],A9,t[:,j*K:(j+1)*K],K)
				if D==o:Cd(Z,G,A3,l);M(dst=gx03[:,a:a+K,:],src=Z)
				else:
					if D==c:Cd(P,G,N,A0)
					elif D==h:Cd(P,G,lst,C)
					else:Cd(P,G,N,C)
					M(dst=gxin3[:,a:a+K,:],src=P);Cd(Z,G,W,l);M(dst=gx03[:,a:a+K,:],src=Z)
					if D==p:E.tensor_scalar(dst=S,data=G,op0=A.multiply,operand0=r);M(dst=gsk3[:,a:a+K,:],src=S)
			if D==o:M(dst=r3[0,:,F:J],src=R);M(dst=r3[1,:,F:J],src=R)
			elif D==c:M(dst=r3[0,:,F:J],src=R);M(dst=r3[1,:,F:J],src=e)
			elif D==h:E.tensor_scalar(dst=f,data=R,op0=A.multiply,operand0=A4);E.tensor_scalar(dst=g,data=R,op0=A.multiply,operand0=N);M(dst=r3[0,:,F:J],src=f);M(dst=r3[1,:,F:J],src=e);M(dst=r3[2,:,F:J],src=g)
			else:E.scalar_tensor_tensor(dst=f,data=t,op0=A.multiply,operand0=X,op1=A.add,operand1=R);E.tensor_scalar(dst=g,data=t,op0=A.multiply,operand0=N);M(dst=r3[0,:,F:J],src=f);M(dst=r3[1,:,F:J],src=e);M(dst=r3[2,:,F:J],src=g)
	def Cf(x03,emit):
		B=x03;D,E,F=B.shape;H=A.ndarray((D,E,F),dtype=B.dtype,buffer=A.shared_hbm);I=A.ndarray((D,E,F),dtype=B.dtype,buffer=A.shared_hbm);J=A.ndarray((D,E),dtype=A.float32,buffer=A.shared_hbm)
		if emit:G=A.ndarray((D,E,F),dtype=B.dtype,buffer=A.shared_hbm)
		else:G=C
		return H,I,J,G
	@K.jit
	def OK(x03,l1g,l2g):A,B,D,_=Cf(x03,O);Ce(o,O,C,C,x03,C,l1g,l2g,C,A,B,D,C);return A,B,D
	@K.jit
	def OL(s13,mo3,x03,l1g,l2g):A,B,D,_=Cf(x03,O);Ce(c,O,s13,mo3,x03,C,l1g,l2g,C,A,B,D,C);return A,B,D
	@K.jit
	def OM(s13,mo3,x03,l1g,l2g):A,B,D,E=Cf(x03,I);Ce(c,I,s13,mo3,x03,C,l1g,l2g,C,A,B,D,E);return A,B,D,E
	@K.jit
	def ON(s13,mo3,x03,l1g,l2g,l3g):A,B,D,_=Cf(x03,O);Ce(h,O,s13,mo3,x03,C,l1g,l2g,l3g,A,B,D,C);return A,B,D
	@K.jit
	def OO(s13,mo3,x03,sk3,l1g,l2g,l3g):A,B,D,_=Cf(x03,O);Ce(p,O,s13,mo3,x03,sk3,l1g,l2g,l3g,A,B,D,C);return A,B,D
	def Ae(x03,mode):
		E=mode;B,D,F=x03.shape;dt=x03.dtype;I=A.ndarray((B,D,F),dtype=dt,buffer=A.shared_hbm)
		if E==o:G=C
		else:G=A.ndarray((B,D,F),dtype=dt,buffer=A.shared_hbm)
		if E==p:H=A.ndarray((B,D,F),dtype=dt,buffer=A.shared_hbm)
		else:H=C
		if E>=h:r3=A.ndarray((3,B,D),dtype=A.float32,buffer=A.shared_hbm)
		else:r3=A.ndarray((2,B,D),dtype=A.float32,buffer=A.shared_hbm)
		return G,I,H,r3
	@K.jit
	def OP(x03,gn3,gy3,inv2,x0e3,l1g,l2g):_,A,_,r3=Ae(x03,o);Ad(o,C,C,x03,C,gn3,gy3,inv2,x0e3,C,l1g,l2g,C,C,A,C,r3);return A,r3
	@K.jit
	def OQ(x03,gn3,gy3,inv2,l1g,l2g):_,A,_,r3=Ae(x03,o);Ad(o,C,C,x03,C,gn3,gy3,inv2,C,C,l1g,l2g,C,C,A,C,r3);return A,r3
	@K.jit
	def OR(s13,mo3,x03,gn3,gy3,inv2,x0e3,xine3,l1g,l2g):A,B,_,r3=Ae(x03,c);Ad(c,s13,mo3,x03,C,gn3,gy3,inv2,x0e3,xine3,l1g,l2g,C,A,B,C,r3);return A,B,r3
	@K.jit
	def OS(s13,mo3,x03,gn3,gy3,inv2,x0e3,l1g,l2g):A,B,_,r3=Ae(x03,c);Ad(c,s13,mo3,x03,C,gn3,gy3,inv2,x0e3,C,l1g,l2g,C,A,B,C,r3);return A,B,r3
	@K.jit
	def OT(s13,mo3,x03,gn3,gy3,inv2,xine3,l1g,l2g):A,B,_,r3=Ae(x03,c);Ad(c,s13,mo3,x03,C,gn3,gy3,inv2,C,xine3,l1g,l2g,C,A,B,C,r3);return A,B,r3
	@K.jit
	def OU(s13,mo3,x03,gn3,gy3,inv2,l1g,l2g):A,B,_,r3=Ae(x03,c);Ad(c,s13,mo3,x03,C,gn3,gy3,inv2,C,C,l1g,l2g,C,A,B,C,r3);return A,B,r3
	@K.jit
	def OV(s13,mo3,x03,gn3,gy3,inv2,x0e3,l1g,l2g,l3g):A,B,_,r3=Ae(x03,h);Ad(h,s13,mo3,x03,C,gn3,gy3,inv2,x0e3,C,l1g,l2g,l3g,A,B,C,r3);return A,B,r3
	@K.jit
	def OW(s13,mo3,x03,gn3,gy3,inv2,l1g,l2g,l3g):A,B,_,r3=Ae(x03,h);Ad(h,s13,mo3,x03,C,gn3,gy3,inv2,C,C,l1g,l2g,l3g,A,B,C,r3);return A,B,r3
	@K.jit
	def OX(s13,mo3,x03,sk3,gn3,gy3,inv2,x0e3,l1g,l2g,l3g):A,B,D,r3=Ae(x03,p);Ad(p,s13,mo3,x03,sk3,gn3,gy3,inv2,x0e3,C,l1g,l2g,l3g,A,B,D,r3);return A,B,D,r3
	@K.jit
	def OY(s13,mo3,x03,sk3,gn3,gy3,inv2,l1g,l2g,l3g):A,B,D,r3=Ae(x03,p);Ad(p,s13,mo3,x03,sk3,gn3,gy3,inv2,C,C,l1g,l2g,l3g,A,B,D,r3);return A,B,D,r3
	JP={(o,O):OK,(c,O):OL,(c,I):OM,(h,O):ON,(p,O):OO};Cg={(o,I,O):OP,(o,O,O):OQ,(c,I,I):OR,(c,I,O):OS,(c,O,I):OT,(c,O,O):OU,(h,I,O):OV,(h,O,O):OW,(p,I,O):OX,(p,O,O):OY};E7=I
except P:E7=O;Cc=C;JP={};Cg={}
def Ch(key,table):return BL(table[key])
def JQ(g,like):
	if g is C:return C
	return Ay(g,like)
def OZ(mode,ts):
	E=mode
	if E==o:x0,l1,A=ts;return C,C,x0,C,l1,A,C
	if E==c:B,D,x0,l1,A=ts;return B,D,x0,C,l1,A,C
	if E==h:B,D,x0,l1,A,F=ts;return B,D,x0,C,l1,A,F
	B,D,x0,G,l1,A,F=ts;return B,D,x0,G,l1,A,F
def E8(ctx,r3,k):
	if ctx.key is not C:Fx[ctx.key]=r3;return(C,)*k
	A=r3.view(k,-1).sum(dim=1);return e(A[i:i+1]for i in L(k))
def mblk_gain_partials_take():out=AA(Fx);Fx.clear();return out
class Oa(torch.autograd.Function):
	@N
	def forward(ctx,mode,emit,*R):
		A=mode;ctx.set_materialize_grads(O);E,F,x0,I,l1,S,L=OZ(A,R);N,J=x0.shape;B,D=Bx(l1),Bx(S);G=Bx(L)if L is not C else C;H=C;x_in=C
		try:
			if A==o:args=T(x0),B,D
			elif A==c:args=T(E),T(F),T(x0),B,D
			elif A==h:args=T(E),T(F),T(x0),B,D,G
			else:args=T(E),T(F),T(x0),T(I),B,D,G
			out=Ch((A,z(emit)),JP)(*args)
			if emit:M,Q,H,U=out;x_in=U.view(N,J)
			else:M,Q,H=out
			x_new,K=M.view(N,J),Q.view(N,J)
		except P as err:
			E6('head2 forward',err);H=C
			with torch.no_grad():x_new,K,V=JM(A,E,F,x0,I,AL(B),AL(D),AL(G)if G is not C else C);x_in=V if emit else C
		ctx.mode,ctx.emit=A,z(emit);ctx.key=Dz;ctx.save_for_backward(E,F,x0,I,B,D,G,H)
		if emit:return x_new,K,x_in,x0
		return x_new,K,x0
	@N
	def backward(ctx,*n):
		W,X,x0,q,O,Q,d,R=ctx.saved_tensors;B,emit=ctx.mode,ctx.emit
		if emit:Y,gy,Z,D=n
		else:Y,gy,D=n;Z=C
		N,F=x0.shape
		if R is not C and not BM:
			try:
				a,b=Ay(Y,x0),Ay(gy,x0);E,e=JQ(D,x0),JQ(Z,x0);key=B,E is not C,e is not C
				if B==o:args=(T(x0),T(a),T(b),R)+((T(E),)if E is not C else())+(O,Q);H,r3=Ch(key,Cg)(*args);J,g2=E8(ctx,r3,2);return C,C,H.view(N,F),J,g2
				if B==c:
					args=T(W),T(X),T(x0),T(a),T(b),R
					if E is not C:args=args+(T(E),)
					if e is not C:args=args+(T(e),)
					args=args+(O,Q);S,H,r3=Ch(key,Cg)(*args);J,g2=E8(ctx,r3,2);G=S.view(N,F);return C,C,G,G,H.view(N,F),J,g2
				if B==h:args=(T(W),T(X),T(x0),T(a),T(b),R)+((T(E),)if E is not C else())+(O,Q,d);S,H,r3=Ch(key,Cg)(*args);J,g2,f=E8(ctx,r3,3);G=S.view(N,F);return C,C,G,G,H.view(N,F),J,g2,f
				args=(T(W),T(X),T(x0),T(q),T(a),T(b),R)+((T(E),)if E is not C else())+(O,Q,d);S,H,w,r3=Ch(key,Cg)(*args);J,g2,f=E8(ctx,r3,3);G=S.view(N,F);return C,C,G,G,H.view(N,F),w.view(N,F),J,g2,f
			except P as err:E6('head2 backward',err)
		with torch.enable_grad():
			r=x0.detach().requires_grad_(I);s=AL(O).detach().requires_grad_(I);t=AL(Q).detach().requires_grad_(I);M=[r,s,t];i=j=k=l=C
			if B!=o:i=W.detach().requires_grad_(I);j=X.detach().requires_grad_(I);M=[i,j]+M
			if B==p:k=q.detach().requires_grad_(I);M.append(k)
			if B>=h:l=AL(d).detach().requires_grad_(I);M.append(l)
			x_new,u,x_in=JM(B,i,j,r,k,s,t,l);outs=[x_new,u];v=[torch.zeros_like(x_new)if Y is C else Y,torch.zeros_like(u)if gy is C else gy]
			if Z is not C and B!=o:outs.append(x_in);v.append(Z)
			g=A0(torch.autograd.grad(outs,M,v,allow_unused=I))
		g=[torch.zeros_like(t)if A is C else A for(A,t)in zip(g,M)]
		if B==o:
			A,K,L=g
			if D is not C:A=A+D
			return C,C,A,K,L
		if B==c:
			U,V,A,K,L=g
			if D is not C:A=A+D
			return C,C,U,V,A,K,L
		if B==h:
			U,V,A,K,L,m=g
			if D is not C:A=A+D
			return C,C,U,V,A,K,L,m
		U,V,A,K,L,x,m=g
		if D is not C:A=A+D
		return C,C,U,V,A,x,K,L,m
class Ob(torch.autograd.Function):
	@N
	def forward(ctx,s12,mo2):
		A=s12;N,D=A.shape;B=C;E=C
		try:E,F,B=BL(JK)(T(A),T(mo2));y=F.view(N,D);s=E.view(N,D)
		except P as err:
			E6('final_norm forward',err);B=C
			with torch.no_grad():s=A+mo2;y=BJ(s)
		ctx.save_for_backward(s,B);return y
	@N
	def backward(ctx,gy):
		s,A=ctx.saved_tensors;N,D=s.shape
		if A is not C and not BM and Cc is not C:
			try:E=Ay(gy,s);F=BL(Cc)(T(s),T(E),A);g=F.view(N,D);return g,g
			except P as err:E6('final_norm backward',err)
		with torch.enable_grad():B=s.detach().requires_grad_(I);G=torch.zeros_like(s)if gy is C else gy;g,=torch.autograd.grad(BJ(B),B,G)
		return g,g
def JR(t):return t.device.type==a
def E9(a,b):return a.dtype==b.dtype and a.shape==b.shape
def mblk_head2(x,x0,skip,l1,l2,l3,emit=O,key=C):
	G=l3;D=l2;J=AP(x,(e,A0));L=skip is mblk_SELF_SKIP
	if J:
		E,F=x;base=E
		if L:B=h
		elif skip is C:B=c
		else:B=p
	else:E=F=C;base=x;B=o if x is x0 and skip is C else-1
	N=E7 and not BM and B>=0 and JR(base)and base.dim()>=2 and E5(base)and E4(base)and E9(x0,base)and(not J or E9(F,base))and(B!=p or E9(skip,base))and l1.numel()==1 and D.numel()==1 and(B<h or G is not C and G.numel()==1)
	if not N:K=E+F if J else x;O=K if L else skip;x_new,H=mblk_head_skip(K,x0,O,l1,D,G);return x_new,H,K if emit else C,x0
	A=base.shape[-1]
	if B==o:P=A4(x0,A);I=P,l1,D
	elif B==c:I=A4(E,A),A4(F,A),A4(x0,A),l1,D
	elif B==h:I=A4(E,A),A4(F,A),A4(x0,A),l1,D,G
	else:I=A4(E,A),A4(F,A),A4(x0,A),A4(skip,A),l1,D,G
	global Dz;Dz=key;outs=Oa.apply(B,z(emit),*I);Dz=C
	if emit:x_new,H,x_in,M=outs;x_in=x_in.view(base.shape)
	else:x_new,H,M=outs;x_in=C
	return x_new.view(base.shape),H.view(base.shape),x_in,M.view(x0.shape)
def mblk_final_norm(s1,mo):
	A=s1;D=Cb and E7 and not BM and Cc is not C and JR(A)and A.dim()>=2 and E5(A)and E9(mo,A)and E4(A)
	if not D:return BJ(A+mo)
	B=A.shape[-1];return Ob.apply(A4(A,B),A4(mo,B)).view(A.shape)
head_skip=mblk_head_skip
tail_norm2=mblk_tail_norm2
USE_NKI_MEGABLOCK=Ny
head2=mblk_head2
final_norm=mblk_final_norm
SELF_SKIP=mblk_SELF_SKIP
USE_NKI_MEGABLOCK_TAIL=OF
try:gain_partials_take=mblk_gain_partials_take
except P:gain_partials_take=C
import os,torch
Pb=object()
A8=128
JS=JG
Oc=Fv
EA=Fw
EB=O0
Od=O1
Oe=N_
JT=Nz
G4=O2
Of=O3
JU=O4
Og=O5
JV=O6
Oh=200*1024
Oi=.5
JW=2.**-4
By=O
EC=C
ED=0
G5=0
G6=0
JX=set()
G7=O
Bz=CA
JY={}
JZ={}
def Oj():return os.environ.get('RANK',A1)==A1
def G8(key,msg,everywhere=O):
	if key in JX:return
	JX.add(key)
	if everywhere or Oj():D(msg,flush=I)
def EE(where,err):
	A=where;global By,EC
	if not By:By=I;EC=f"{A}: {W(err).__name__}: {err}";D(f"[mixfuse] fused mixer DISABLED after failure in {A}: {W(err).__name__}: {err} -- the unfused mixer runs from here on (nki_megablock untouched)",flush=I)
def Ja(dt):return 4 if dt==torch.float32 else 2
def EF(n,C,isz,fwd):
	A=isz;D=4 if JT==Cm else A;k=1
	for d in L(1,Oc+1):
		if n%d==0:
			if fwd:sq=d*C*4 if EB==AX else C*A;B=d*C*A*4+sq;E=C*A+8*n+2048
			else:
				B=d*C*(A*4+D+4)
				if not G4:B=B+d*C*A
				E=C*A+C*(A*4+D+4)+8*n+2048
			if 2*B+E<=Oh:k=d
	return k
def Ok(NT,pid,nprg):A=pid;B,C=NT//nprg,NT%nprg;D=A*B+g(A,C);return D,D+B+(1 if A<C else 0)
try:
	import nki as K,nki.language as A,nki.isa as E;Jb=C if JU==BR else n(E.dge_mode,JU)
	if JV=='gpsimd':BN=E.engine.gpsimd
	elif JV==C5:BN=E.engine.vector
	else:BN=E.engine.unknown
	def f(dst,src):
		if Jb is C:E.dma_copy(dst=dst,src=src)
		else:E.dma_copy(dst=dst,src=src,dge_mode=Jb)
	def Ol():
		if Of and A.program_ndim()>0:return A.program_id(0),A.num_programs(0)
		return 0,1
	def Jc(NT):A,B=Ol();C,D=NT//B,NT%B;E=A*C+g(A,D);n=C+(1 if A<D else 0);return E,E+n
	def AM(shape,dtype):return A.ndarray(shape,dtype=dtype,buffer=A.sbuf),A.ndarray(shape,dtype=dtype,buffer=A.sbuf)
	def A5(shape,dtype):return A.ndarray(shape,dtype=dtype,buffer=A.sbuf)
	def Om(xb,sq,ss,K):
		if EB==AX:E.activation(dst=sq,op=A.square,data=xb);E.tensor_reduce(dst=ss,op=A.add,data=sq,axis=2)
		else:
			for k in L(K):E.activation_reduce(dst=sq,op=A.square,data=xb[:,k,:],reduce_op=A.add,reduce_res=ss[:,k:k+1])
	def On(ss,inv_dst,C):
		B=inv_dst
		if Od==BX:E.activation(dst=B,op=A.rsqrt,data=ss,scale=Q/C,bias=JS)
		else:E.tensor_scalar(dst=ss,data=ss,op0=A.multiply,operand0=Q/C,op1=A.add,operand1=JS);E.tensor_scalar(dst=B,data=ss,op0=A.rsqrt,operand0=S)
	def Oo(dst,src,inv,K):
		for k in L(K):E.tensor_scalar(dst=dst[:,k,:],data=src[:,k,:],op0=A.multiply,operand0=inv[:,k:k+1])
	def Jd(pb,dst,K):
		if Oe==AX:E.tensor_reduce(dst=dst,op=A.add,data=pb,axis=2)
		else:
			for k in L(K):E.activation_reduce(dst=pb[:,k,:],op=A.copy,data=pb[:,k,:],reduce_op=A.add,reduce_res=dst[:,k:k+1])
	def Op(P,K,C):
		if EB==AX:return P,K,C
		return P,C
	def Oq(dt):
		if EB==AX:return A.float32
		return dt
	def Or(dt):return A.float32 if JT==Cm else dt
	def Je(dt):
		if dt==A.float32:return 4
		return 2
	def Jf(lam2,P,dt):
		B=A.float32;t=A5((P,1),dt);f(dst=t,src=lam2.broadcast(0,P))
		if dt==B:return t
		C=A5((P,1),B);E.activation(dst=C,op=A.copy,data=t);return C
	def G9(hm2,P):t=A5((P,1),A.float32);f(dst=t,src=hm2);return t
	def Os(cr,x3,h3,hm2,i0,P):
		if i0>0:f(dst=cr,src=x3[:,i0-1:i0,:])
		else:f(dst=cr,src=h3);B=G9(hm2,P);E.tensor_scalar(dst=cr[:,0,:],data=cr[:,0,:],op0=A.multiply,operand0=B)
	def Jg(dst,x3,h3,hm2,a,K,P):
		B=dst
		if a>0:f(dst=B,src=x3[:,a-1:a-1+K,:])
		else:
			f(dst=B[:,0:1,:],src=h3)
			if K>1:f(dst=B[:,1:K,:],src=x3[:,0:K-1,:])
			C=G9(hm2,P);E.tensor_scalar(dst=B[:,0,:],data=B[:,0,:],op0=A.multiply,operand0=C)
	def Jh(sb,gyb,geb,pb,gib,gb,invs,K,C,P):
		G=gyb;F=invs;D=gib;H=A.float32;E.tensor_tensor(dst=pb,data1=G,data2=sb,op=A.multiply);I=A5((P,K),H);Jd(pb,I,K);B=A5((P,K),H);E.tensor_tensor(dst=B,data1=F,data2=F,op=A.multiply);E.tensor_tensor(dst=B,data1=B,data2=F,op=A.multiply);E.tensor_tensor(dst=B,data1=B,data2=I,op=A.multiply);E.tensor_scalar(dst=B,data=B,op0=A.multiply,operand0=-Q/C)
		if Og==BX:
			for k in L(K):E.activation(dst=D[:,k,:],op=A.copy,data=G[:,k,:],scale=F[:,k:k+1])
			E.tensor_tensor(dst=D,data1=D,data2=geb,op=A.add,engine=BN)
		else:
			for k in L(K):E.scalar_tensor_tensor(dst=D[:,k,:],data=G[:,k,:],op0=A.multiply,operand0=F[:,k:k+1],op1=A.add,operand1=geb[:,k,:])
		for k in L(K):E.scalar_tensor_tensor(dst=gb[:,k,:],data=sb[:,k,:],op0=A.multiply,operand0=B[:,k:k+1],op1=A.add,operand1=D[:,k,:])
	def Ot(s3,gy3,ge3,inv2,hmn2,lamt,i1,NT,P,C,dt,pdt):
		B=i1;I=A.float32;D=A5((P,1,C),dt);F=A5((P,1,C),dt);G=A5((P,1,C),dt);H=A5((P,1),I)
		if B<NT:f(dst=D,src=s3[:,B:B+1,:]);f(dst=F,src=gy3[:,B:B+1,:]);f(dst=G,src=ge3[:,B:B+1,:]);f(dst=H,src=inv2[:,B:B+1])
		else:E.memset(dst=D,value=S);E.memset(dst=F,value=S);E.memset(dst=G,value=S);E.memset(dst=H,value=S);f(dst=D[0:P-1,:,:],src=s3[1:P,0:1,:]);f(dst=F[0:P-1,:,:],src=gy3[1:P,0:1,:]);f(dst=G[0:P-1,:,:],src=ge3[1:P,0:1,:]);f(dst=H[0:P-1,:],src=inv2[1:P,0:1])
		K=A5((P,1,C),pdt);L=A5((P,1,C),I);J=F if G4 else A5((P,1,C),dt);Jh(D,F,G,K,L,J,H,1,C,P);E.activation(dst=D,op=A.copy,data=J,scale=lamt)
		if B==NT:M=G9(hmn2,P);E.tensor_scalar(dst=D[:,0,:],data=D[:,0,:],op0=A.multiply,operand0=M)
		return D
	@K.jit
	def Ou(x3,h3,hm2,lam2):
		F=x3;B,G,C=F.shape;dt=F.dtype;J=A.float32;R=A.ndarray((B,G,C),dtype=dt,buffer=A.shared_hbm);S=A.ndarray((B,G,C),dtype=dt,buffer=A.shared_hbm);T=A.ndarray((B,G),dtype=J,buffer=A.shared_hbm);D,U=Jc(G);n=U-D
		if n>0:
			K=EF(n,C,Je(dt),I);M=n//K;V=Jf(lam2,B,dt);N=A5((B,n),J);O=AM((B,K,C),dt);X=AM((B,K,C),dt);Y=AM((B,K,C),dt);Z=AM((B,K,C),dt);b=AM(Op(B,K,C),Oq(dt));P=A5((B,1,C),dt);Os(P,F,h3,hm2,D,B);f(dst=O[0],src=F[:,D:D+K,:])
			for j in L(M):
				a=D+j*K;s=j%2;xb,Q,H,yb,sq=O[s],X[s],Y[s],Z[s],b[s]
				if j+1<M:f(dst=O[(j+1)%2],src=F[:,a+K:a+2*K,:])
				E.activation(dst=Q[:,0:1,:],op=A.copy,data=P,scale=V)
				if K>1:E.activation(dst=Q[:,1:K,:],op=A.copy,data=xb[:,0:K-1,:],scale=V)
				if j+1<M:E.activation(dst=P,op=A.copy,data=xb[:,K-1:K,:])
				E.tensor_tensor(dst=H,data1=xb,data2=Q,op=A.add,engine=BN);W=A5((B,K),J);Om(H,sq,W,K);On(W,N[:,j*K:(j+1)*K],C);Oo(yb,H,N[:,j*K:(j+1)*K],K);f(dst=R[:,a:a+K,:],src=H);f(dst=S[:,a:a+K,:],src=yb)
			f(dst=T[:,D:U],src=N)
		return R,S,T
	@K.jit
	def Ov(s3,gy3,ge3,x3,h3,inv2,hm2,hmn2,lam2):
		T=ge3;S=gy3;F=s3;B,G,C=F.shape;dt=F.dtype;H=A.float32;Y=Or(dt);I=A.ndarray((B,G,C),dtype=dt,buffer=A.shared_hbm);Z=A.ndarray((B,G),dtype=H,buffer=A.shared_hbm);D,J=Jc(G);n=J-D
		if n>0:
			K=EF(n,C,Je(dt),O);M=n//K;b=Jf(lam2,B,dt);c=A5((B,n),H);f(dst=c,src=inv2[:,D:J]);d=A5((B,n),H);U=AM((B,K,C),dt);N=AM((B,K,C),dt);V=AM((B,K,C),dt);W=AM((B,K,C),dt);e=AM((B,K,C),Y);g=AM((B,K,C),H);h=N if G4 else AM((B,K,C),dt);P=A5((B,1,C),dt);f(dst=U[0],src=F[:,D:D+K,:]);f(dst=N[0],src=S[:,D:D+K,:]);f(dst=V[0],src=T[:,D:D+K,:]);Jg(W[0],x3,h3,hm2,D,K,B)
			for j in L(M):
				a=D+j*K;s=j%2;Q,i,X,k,l,m,gb=U[s],N[s],V[s],W[s],e[s],g[s],h[s]
				if j+1<M:R=(j+1)%2;f(dst=U[R],src=F[:,a+K:a+2*K,:]);f(dst=N[R],src=S[:,a+K:a+2*K,:]);f(dst=V[R],src=T[:,a+K:a+2*K,:]);Jg(W[R],x3,h3,hm2,a+K,K,B)
				o=c[:,j*K:(j+1)*K];Jh(Q,i,X,l,m,gb,o,K,C,B);E.tensor_tensor(dst=X,data1=gb,data2=k,op=A.multiply);Jd(X,d[:,j*K:(j+1)*K],K);E.activation(dst=Q,op=A.copy,data=gb,scale=b)
				if j>0:E.tensor_tensor(dst=P,data1=P,data2=Q[:,0:1,:],op=A.add,engine=BN);f(dst=I[:,a-1:a,:],src=P)
				if j+1<M:E.activation(dst=P,op=A.copy,data=gb[:,K-1:K,:])
				if K>1:E.tensor_tensor(dst=gb[:,0:K-1,:],data1=gb[:,0:K-1,:],data2=Q[:,1:K,:],op=A.add,engine=BN)
				if j+1<M:
					if K>1:f(dst=I[:,a:a+K-1,:],src=gb[:,0:K-1,:])
				else:p=Ot(F,S,T,inv2,hmn2,b,J,G,B,C,dt,Y);E.tensor_tensor(dst=gb[:,K-1:K,:],data1=gb[:,K-1:K,:],data2=p,op=A.add,engine=BN);f(dst=I[:,a:a+K,:],src=gb)
			f(dst=Z[:,D:J],src=d)
		return I,Z
	Ji=I
except P:Ji=O
def Jj():
	if Ji:return Ou,Ov
	return C,C
Ci,GA=Jj()
def Ow(t):return t.device.type==a
def EG(kernel):
	A=kernel
	if EA==2 and EP(A,'__getitem__'):return A[EA]
	return A
def GB(NT,T,device):
	key=NT,T,v(device);m=JZ.get(key)
	if m is C:A=[S if p*NT%T==0 else Q for p in L(A8)];B=A[1:]+[S];m=torch.tensor(A,dtype=torch.float32).view(A8,1).to(device),torch.tensor(B,dtype=torch.float32).view(A8,1).to(device);JZ[key]=m
	return m
def Jk(x3):A=x3.shape[1];return torch.cat((x3[0:1,A-1:A,:],x3[0:A8-1,A-1:A,:]),dim=0)
def Ox(meta):
	if meta is C:return C
	z=JY.get(meta)
	if z is C:shape,dtype,device=meta;z=torch.zeros(shape,dtype=dtype,device=device);JY[meta]=z
	return z
def Jl(g,like3):
	B=like3
	if g is C:return torch.zeros_like(B)
	A=g.reshape(B.shape)
	if A.dtype!=B.dtype:A=A.to(B.dtype)
	if not A.is_contiguous():A=A.contiguous()
	return A
def Jm(x):return torch.cat((x.new_zeros(x.shape[0],1,x.shape[2]),x[:,:-1,:]),dim=1)
def EH(x,lam):mix_out=lam*Jm(x);return mblk_tail_norm2(x,mix_out)
def mixk_applies(x):
	global G6;A=C
	if Ci is C or GA is C:A=KM
	elif By:A=f"disabled ({EC})"
	elif not Ow(x):A=f"device {x.device.type}"
	elif x.dim()!=3 or x.dtype not in(torch.bfloat16,torch.float32)or not x.is_contiguous():A=f"tensor {e(x.shape)} {x.dtype} contiguous={x.is_contiguous()}"
	else:
		B,T,E=x.shape;N=B*T
		if N%A8:A=f"rows {N} % {A8}"
		else:
			D=N//A8
			if D>T or T%D:A=f"layout NT={D} T={T} (needs NT <= T and T % NT == 0)"
	if A is C:return I
	G6+=1;G8((C_,A),f"[mixfuse] fused mixer not applicable: {A} -- unfused mixer for such calls");return O
def mixk_mix_tail(x,xn_dead,lam):
	A=lam;B,T,C=x.shape
	if not G7:
		if not torch.is_grad_enabled()or torch.is_inference_mode_enabled():return EH(x,A)
		Oz(x,T)
		if By:return EH(x,A)
	D,y=Oy.apply(x.reshape(B*T,C),xn_dead,A,T);return D.view(B,T,C),y.view(B,T,C)
class Oy(torch.autograd.Function):
	@N
	def forward(ctx,x2,xn_dead,lam,T):
		F=lam;E=xn_dead;B=x2;global ED;ctx.set_materialize_grads(O);N,A=B.shape;D=N//A8;ctx.T=T;ctx.xn_meta=(e(E.shape),E.dtype,E.device)if E is not C else C;G=B.view(A8,D,A);S,_=GB(D,T,B.device);H=F.reshape(1,1)
		try:J=Jk(G);K,U,V=EG(Ci)(G,J,S,H)
		except P as err:
			EE(BS,err);ctx.fused=O;ctx.save_for_backward(B,F)
			with torch.no_grad():W,y=EH(B.view(N//T,T,A),F)
			return W.reshape(N,A),y.reshape(N,A)
		ED+=1
		if ED==1:M=EA if EA==2 else 1;Q,R=Ok(D,0,M);G8('fired_fwd',f"mixfuse:           FIRED fwd x=[{N}, {A}] T={T} NT={D} nprg={M} K={EF(R-Q,A,Ja(B.dtype),I)}/{EF(R-Q,A,Ja(B.dtype),O)} halo=[{A8}, 1, {A}] seq_start_partitions={BP(1 for p in L(A8)if p*D%T==0)}")
		ctx.fused=I;ctx.save_for_backward(K,G,J,V,H);return K.view(N,A),U.view(N,A)
	@N
	def backward(ctx,ge,gy):
		global G5;D=Ox(ctx.xn_meta);T=ctx.T
		if ctx.fused:
			A,J,R,S,E=ctx.saved_tensors;K,F,B=A.shape
			if not By:
				try:U,V=GB(F,T,A.device);W,X=EG(GA)(A,Jl(gy,A),Jl(ge,A),J,R,S,U,V,E);G5+=1;return W.view(K*F,B),D,X.sum().to(E.dtype).reshape(1),C
				except P as err:EE(Am,err)
			G,L=J.reshape(K*F,B),E.reshape(1)
		else:G,L=ctx.saved_tensors
		N,B=G.shape
		with torch.enable_grad():
			M=G.detach().requires_grad_(I);lr=L.detach().requires_grad_(I);O,Q=EH(M.view(N//T,T,B),lr);outs,H=[],[]
			if ge is not C:outs.append(O);H.append(ge.reshape(O.shape))
			if gy is not C:outs.append(Q);H.append(gy.reshape(Q.shape))
			if not outs:return C,D,C,C
			gx,Y=torch.autograd.grad(outs,(M,lr),H,allow_unused=I)
		return gx,D,Y,C
def GC(a,b):d=(a.float()-b.float()).abs().max();return G(d/b.float().abs().max().clamp_min(1e-30))
def Oz(x,T,lam_test=Oi):
	H=lam_test;F='selfcheck';global G7,Bz;G7=I
	try:
		B,_,A=x.shape;N=B*T;C=N//A8
		with torch.no_grad():D=x.detach().reshape(N,A);J=torch.full((1,),H,dtype=x.dtype,device=x.device);K=D.clone();L=D*-.25;E=D.view(A8,C,A);M,c=GB(C,T,x.device);O=Jk(E);Q=J.reshape(1,1);R,d,e=EG(Ci)(E,O,M,Q);f,g=EG(GA)(R,K.view(A8,C,A),L.view(A8,C,A),E,O,e,M,c,Q);h=g.sum()
		with torch.enable_grad():S=D.detach().clone().requires_grad_(I);lr=J.detach().clone().requires_grad_(I);U=S.view(B,T,A);sh=Jm(U);m=lr*sh;V,X=mblk_tail_norm2(U,m);j,k,l=torch.autograd.grad((V,X),(S,lr,m),(L.view(B,T,A),K.view(B,T,A)))
		with torch.no_grad():scale=(l.float()*sh.float()).abs().sum().clamp_min(1e-30);Y=GC(R.view(N,A),V.reshape(N,A));Z=GC(d.view(N,A),X.reshape(N,A));a=GC(f.view(N,A),j);b=G((h.float()-k.float().reshape(())).abs()/scale)
		ok=i(Y,Z,a,b)<=JW;Bz=f"lam={H} s1 {Y:.2e} y {Z:.2e} gx {a:.2e} dlam {b:.2e} (tol {JW:.0e}) -> {"PASS"if ok else"FAIL"}";G8(F,f"mixfuse:           selfcheck {Bz}",everywhere=not ok)
		if not ok:EE(F,Al(Bz))
	except P as err:Bz=f"ERROR {W(err).__name__}: {err}";EE(F,err)
def O_():A=AW if Ci is C else'nki'if Ci is Jj()[0]else'substituted';return f"kernel={A} fired_fwd={ED} fired_bwd={G5} declined={G6} disabled={EC!r} selfcheck={Bz!r}"
nki_mixtail=object()
try:from nkilib_attention import nki_causal_attention,USE_NKILIB_ATTN
except P:USE_NKILIB_ATTN=O
import math,os,torch,torch.nn.functional as F
J=128
def P0(name,default):
	A=default
	try:return B(os.environ.get(name,Y)or A)
	except AF:return A
attnk_USE_NKI_ATTN=q
Jn=P0(AB,1)
AN=i(1,g(4,4))
B_=i(1,g(8,4))
P1=G(A1)
Jo=A1
attnk_USE_NKI_ATTN_ROPE=q
Jp=q
P2=2.**-23
C0=O
Jq=C
P3=4096
def GD(q,k,v,scale=C):y=F.scaled_dot_product_attention(q,k,v,is_causal=I,scale=scale);return y.transpose(1,2).contiguous()
def Jr(q,k,v,gy_bthd,scale):
	with torch.enable_grad():A=q.detach().requires_grad_(I);B=k.detach().requires_grad_(I);C=v.detach().requires_grad_(I);y=F.scaled_dot_product_attention(A,B,C,is_causal=I,scale=scale);return torch.autograd.grad(y,(A,B,C),gy_bthd.transpose(1,2))
try:
	import nki as K,nki.language as A,nki.isa as E
	def EI():
		if A.program_ndim()>0:return A.program_id(0),A.num_programs(0)
		return 0,1
	def EJ():return E.engine.unknown
	def GE(x3,base,T,D,G):B=A.ndarray((J,G*D),dtype=x3.dtype,buffer=A.sbuf);E.dma_copy(dst=B,src=x3.ap(pattern=[[D,J],[J*D,G],[1,D]],offset=base));return B
	def EK(x3,b,h,T,C,D,G):B=A.ndarray((J,G*D),dtype=x3.dtype,buffer=A.sbuf);E.dma_copy(dst=B,src=x3.ap(pattern=[[C,J],[J*C,G],[1,D]],offset=b*T*C+h*D));return B
	def C1(nat,T,D,G):
		F=nat;I=A.ndarray((D,T),dtype=F.dtype,buffer=A.sbuf)
		for C in L(0,G,4):
			B=G-C
			if B>4:B=4
			K=A.ndarray((D,B*J),dtype=F.dtype,buffer=A.psum)
			for H in L(B):g=C+H;E.nc_transpose(dst=K[:,H*J:(H+1)*J],data=F[:,g*D:(g+1)*D])
			E.tensor_copy(dst=I[:,C*J:(C+B)*J],src=K,engine=EJ())
		return I
	def GF(x3,base,T,D,G):B=A.ndarray((D,T),dtype=x3.dtype,buffer=A.sbuf);E.dma_transpose(dst=B.ap([[T,D],[1,1],[1,1],[1,T]]),src=x3.ap([[D,T],[1,1],[1,1],[1,D]],offset=base));return B;return C1(GE(x3,base,T,D,G),T,D,G)
	@K.jit
	def P4(q4,k4,v4,scale,shift):
		W=shift;B,K,T,C=q4.shape;X=B*K;G=T//J;U=K*C;H=C+1;I=A.float32;dt=q4.dtype;Y=A.ndarray((B,T,U),dtype=dt,buffer=A.shared_hbm);Z=A.ndarray((X,J,G),dtype=I,buffer=A.shared_hbm);e,f=EI();a=A.ndarray((J,1),dtype=I,buffer=A.sbuf);E.memset(a,value=-W)
		for i in L(e,X,f):
			b=i//K;h=i%K;base=i*T*C;k=GF(q4,base,T,C,G);l=GF(k4,base,T,C,G);V=A.ndarray((J,G*H),dtype=dt,buffer=A.sbuf);E.memset(V,value=Q)
			for j in L(G):E.dma_copy(dst=V[:,j*H:j*H+C],src=v4.ap(pattern=[[C,J],[1,C]],offset=base+j*J*C))
			M=A.ndarray((J,G),dtype=I,buffer=A.sbuf)
			for g in L(G):
				N=A.ndarray((J,H),dtype=I,buffer=A.psum)
				for P in L(0,g+1,AN):
					F=g+1-P
					if F>AN:F=AN
					st=A.ndarray((J,F*J),dtype=I,buffer=A.psum)
					for D in L(F):j=P+D;E.nc_matmul(dst=st[:,D*J:(D+1)*J],stationary=l[:,j*J:(j+1)*J],moving=k[:,g*J:(g+1)*J],accumulate=O)
					R=A.ndarray((J,F*J),dtype=dt,buffer=A.sbuf);E.activation(dst=R,op=A.exp,data=st,bias=a,scale=scale)
					if P+F-1==g:D=F-1;E.affine_select(dst=R[:,D*J:(D+1)*J],pattern=[[1,J]],channel_multiplier=-1,offset=0,cmp_op=A.greater_equal,on_true_tile=R[:,D*J:(D+1)*J],on_false_value=S)
					for D in L(F):j=P+D;E.nc_matmul(dst=N,stationary=R[:,D*J:(D+1)*J],moving=V[:,j*H:(j+1)*H],accumulate=j>0)
				c=A.ndarray((J,1),dtype=I,buffer=A.sbuf);E.reciprocal(dst=c,data=N[:,C:C+1]);d=A.ndarray((J,C),dtype=dt,buffer=A.sbuf);E.tensor_scalar(dst=d,data=N[:,0:C],op0=A.multiply,operand0=c);E.dma_copy(dst=Y.ap(pattern=[[U,J],[1,C]],offset=(b*T+g*J)*U+h*C),src=d);E.activation(dst=M[:,g:g+1],op=A.log,data=N[:,C:C+1])
			E.tensor_scalar(dst=M,data=M,op0=A.add,operand0=W);E.dma_copy(dst=Z[i,:,:],src=M)
		return Y,Z
	@K.jit
	def P5(q4,k4,v4,o3,do3,lse3,scale):
		B,U,T,C=q4.shape;A0=B*U;G=T//J;W=U*C;F=A.float32;dt=q4.dtype;e=A.ndarray((B,U,T,C),dtype=dt,buffer=A.shared_hbm);f=A.ndarray((B,U,T,C),dtype=dt,buffer=A.shared_hbm);k=A.ndarray((B,T,W),dtype=dt,buffer=A.shared_hbm);A1,A2=EI()
		for i in L(A1,A0,A2):
			b=i//U;h=i%U;base=i*T*C;l=GE(q4,base,T,C,G);m=GE(k4,base,T,C,G);A3=EK(o3,b,h,T,W,C,G);c=EK(do3,b,h,T,W,C,G);A4=C1(l,T,C,G);A5=C1(m,T,C,G);A6=C1(c,T,C,G);A7=GF(v4,base,T,C,G);X=A.ndarray((J,G),dtype=F,buffer=A.sbuf);E.dma_copy(dst=X,src=lse3[i,:,:]);E.tensor_scalar(dst=X,data=X,op0=A.multiply,operand0=-Q);n=A.ndarray((J,G),dtype=F,buffer=A.sbuf)
			for g in L(G):o=A.ndarray((J,C),dtype=F,buffer=A.sbuf);E.tensor_tensor(dst=o,data1=c[:,g*C:(g+1)*C],data2=A3[:,g*C:(g+1)*C],op=A.multiply);E.tensor_reduce(dst=n[:,g:g+1],op=A.add,data=o,axis=1)
			Y=A.ndarray((J,G*C),dtype=F,buffer=A.sbuf);Z=A.ndarray((J,G*C),dtype=F,buffer=A.sbuf);E.memset(Y,value=S);E.memset(Z,value=S)
			for N in L(0,G,B_):
				H=G-N
				if H>B_:H=B_
				P=(N+H)*J;pg=A.ndarray((J,H*P),dtype=dt,buffer=A.sbuf);d=A.ndarray((J,H*P),dtype=dt,buffer=A.sbuf)
				for D in L(H):
					g=N+D;p=A.ndarray((J,C),dtype=F,buffer=A.psum)
					for K in L(0,g+1,AN):
						R=g+1-K
						if R>AN:R=AN
						w=R*J;I=D*P+K*J;s=A.ndarray((J,w),dtype=F,buffer=A.psum);E.nc_matmul(dst=s,stationary=A4[:,g*J:(g+1)*J],moving=A5[:,K*J:K*J+w],accumulate=O);E.activation(dst=pg[:,I:I+w],op=A.exp,data=s,bias=X[:,g:g+1],scale=scale)
						if K+R-1==g:a=I+(R-1)*J;E.affine_select(dst=pg[:,a:a+J],pattern=[[-1,J]],channel_multiplier=1,offset=0,cmp_op=A.greater_equal,on_true_tile=pg[:,a:a+J],on_false_value=S)
						q=A.ndarray((J,w),dtype=F,buffer=A.psum);E.nc_matmul(dst=q,stationary=A6[:,g*J:(g+1)*J],moving=A7[:,K*J:K*J+w],accumulate=O);E.scalar_tensor_tensor(dst=d[:,I:I+w],data=q,op0=A.subtract,operand0=n[:,g:g+1],op1=A.multiply,operand1=pg[:,I:I+w]);r=A.ndarray((J,w),dtype=dt,buffer=A.psum)
						for M in L(R):E.nc_transpose(dst=r[:,M*J:(M+1)*J],data=d[:,I+M*J:I+(M+1)*J])
						t=A.ndarray((J,w),dtype=dt,buffer=A.sbuf);E.tensor_copy(dst=t,src=r,engine=EJ())
						for M in L(R):j=K+M;E.nc_matmul(dst=p,stationary=t[:,M*J:(M+1)*J],moving=m[:,j*C:(j+1)*C],accumulate=j>0)
					u=A.ndarray((J,C),dtype=dt,buffer=A.sbuf);E.tensor_scalar(dst=u,data=p,op0=A.multiply,operand0=scale);E.dma_copy(dst=e[b,h,g*J:(g+1)*J,:],src=u)
				for j in L(N+H):
					V=j-N
					if V<0:V=0
					v=A.ndarray((J,C),dtype=F,buffer=A.psum)
					for D in L(V,H):g=N+D;E.nc_matmul(dst=v,stationary=pg[:,D*P+j*J:D*P+(j+1)*J],moving=c[:,g*C:(g+1)*C],accumulate=D>V)
					E.tensor_tensor(dst=Z[:,j*C:(j+1)*C],data1=Z[:,j*C:(j+1)*C],data2=v,op=A.add);x=A.ndarray((J,C),dtype=F,buffer=A.psum)
					for D in L(V,H):g=N+D;E.nc_matmul(dst=x,stationary=d[:,D*P+j*J:D*P+(j+1)*J],moving=l[:,g*C:(g+1)*C],accumulate=D>V)
					E.scalar_tensor_tensor(dst=Y[:,j*C:(j+1)*C],data=x,op0=A.multiply,operand0=scale,op1=A.add,operand1=Y[:,j*C:(j+1)*C])
			for j in L(G):y=A.ndarray((J,C),dtype=dt,buffer=A.sbuf);E.tensor_copy(dst=y,src=Y[:,j*C:(j+1)*C]);E.dma_copy(dst=f[b,h,j*J:(j+1)*J,:],src=y);z=A.ndarray((J,C),dtype=dt,buffer=A.sbuf);E.tensor_copy(dst=z,src=Z[:,j*C:(j+1)*C]);E.dma_copy(dst=k.ap(pattern=[[W,J],[1,C]],offset=(b*T+j*J)*W+h*C),src=z)
		return e,f,k
	def Js(cos2,sin2,G,Dh):
		D=sin2;C=cos2;B=Dh;F=A.ndarray((J,G,1,B),dtype=C.dtype,buffer=A.sbuf);H=A.ndarray((J,G,1,B),dtype=D.dtype,buffer=A.sbuf);E.dma_copy(dst=F,src=C.ap(pattern=[[B,J],[J*B,G],[1,B]],offset=0));E.dma_copy(dst=H,src=D.ap(pattern=[[B,J],[J*B,G],[1,B]],offset=0));return F.broadcast(2,2),H.broadcast(2,2);I=A.ndarray((J,G,2,B),dtype=C.dtype,buffer=A.sbuf);K=A.ndarray((J,G,2,B),dtype=D.dtype,buffer=A.sbuf)
		for s in L(2):E.tensor_copy(dst=I[:,:,s,:],src=F[:,:,0,:]);E.tensor_copy(dst=K[:,:,s,:],src=H[:,:,0,:])
		return I,K
	def Jt(dst,src,cb,sb,tmp_a,tmp_b,Dh,transpose):F=transpose;D=tmp_b;C=tmp_a;B=Dh;G=2*B;H,I=src[:,:,:,0:B],src[:,:,:,B:G];J,K=dst[:,:,:,0:B],dst[:,:,:,B:G];E.tensor_tensor(dst=C,data1=H,data2=cb,op=A.multiply);E.tensor_tensor(dst=D,data1=I,data2=sb,op=A.multiply);E.tensor_tensor(dst=J,data1=C,data2=D,op=A.subtract if F else A.add,engine=E.engine.unknown);E.tensor_tensor(dst=C,data1=I,data2=cb,op=A.multiply);E.tensor_tensor(dst=D,data1=H,data2=sb,op=A.multiply);E.tensor_tensor(dst=K,data1=C,data2=D,op=A.add if F else A.subtract,engine=E.engine.unknown)
	def P6(r,sq,G,D):B=A.ndarray((J,G,2,1),dtype=A.float32,buffer=A.sbuf);E.activation(dst=sq,op=A.square,data=r);E.tensor_reduce(dst=B,op=A.add,data=sq,axis=3,keepdims=I);C=A.ndarray((J,G,2,1),dtype=A.float32,buffer=A.sbuf);E.tensor_scalar(dst=B,data=B,op0=A.multiply,operand0=Q/D,op1=A.add,operand1=P2);E.tensor_scalar(dst=C,data=B,op0=A.rsqrt,operand0=S);return C
	def Ju(tile,qkv3,b,h,T,C,G,D,store):
		A=3*C;base=b*T*A+h*D
		for s in L(2):
			B=qkv3.ap(pattern=[[A,J],[J*A,G],[1,D]],offset=base+s*C)
			if store:E.dma_copy(dst=B,src=tile[:,:,s,:])
			else:E.dma_copy(dst=tile[:,:,s,:],src=B)
	def Jv(tile,r5,b,h,B,H,T,G,D,store):
		base=(b*H+h)*T*D;C=B*H*T*D
		for s in L(2):
			A=r5.ap(pattern=[[D,J],[J*D,G],[1,D]],offset=base+s*C)
			if store:E.dma_copy(dst=A,src=tile[:,:,s,:])
			else:E.dma_copy(dst=tile[:,:,s,:],src=A)
	def Jw(yn,s,T,D,G):
		H=A.ndarray((D,T),dtype=yn.dtype,buffer=A.sbuf)
		for C in L(0,G,4):
			B=G-C
			if B>4:B=4
			I=A.ndarray((D,B*J),dtype=yn.dtype,buffer=A.psum)
			for F in L(B):g=C+F;E.nc_transpose(dst=I[:,F*J:(F+1)*J],data=yn[:,g,s,:])
			E.tensor_copy(dst=H[:,C*J:(C+B)*J],src=I,engine=EJ())
		return H
	def Jx(yn,s,T,D,G):
		H=A.ndarray((D,T),dtype=yn.dtype,buffer=A.sbuf)
		for C in L(0,G,4):
			B=G-C
			if B>4:B=4
			I=A.ndarray((D,B*J),dtype=yn.dtype,buffer=A.psum)
			for F in L(B):g=C+F;E.nc_transpose(dst=I[:,F*J:(F+1)*J],data=yn[:,g,s,:])
			E.tensor_copy(dst=H[:,C*J:(C+B)*J],src=I,engine=E.engine.vector)
		return H
	def P7(qkv3,b,h,T,C,G,D):
		B=3*C;F=D+1;I=b*T*B+2*C+h*D;H=A.ndarray((J,G*F),dtype=qkv3.dtype,buffer=A.sbuf);E.memset(H,value=Q)
		for j in L(G):E.dma_copy(dst=H[:,j*F:j*F+D],src=qkv3.ap(pattern=[[B,J],[1,D]],offset=I+j*J*B))
		return H
	@K.jit
	def P8(qkv3,cos2,sin2,scale,shift):
		Z=shift;N=qkv3;B,T,o=N.shape;K=o//3;C=J;M=K//C;P=C//2;W=B*M;G=T//J;X=C+1;F=A.float32;dt=N.dtype;a=A.ndarray((B,T,K),dtype=dt,buffer=A.shared_hbm);c=A.ndarray((W,J,G),dtype=F,buffer=A.shared_hbm);d=A.ndarray((W,J,G,2,1),dtype=F,buffer=A.shared_hbm);e=A.ndarray((2,B,M,T,C),dtype=dt,buffer=A.shared_hbm);p,q=EI();s,t=Js(cos2,sin2,G,P);f=A.ndarray((J,1),dtype=F,buffer=A.sbuf);E.memset(f,value=-Z)
		for i in L(p,W,q):
			b=i//M;h=i%M;k=A.ndarray((J,G,2,C),dtype=dt,buffer=A.sbuf);Ju(k,N,b,h,T,K,G,C,O);u=A.ndarray((J,G,2,P),dtype=dt,buffer=A.sbuf);v=A.ndarray((J,G,2,P),dtype=dt,buffer=A.sbuf);r=A.ndarray((J,G,2,C),dtype=dt,buffer=A.sbuf);Jt(r,k,s,t,u,v,P,O);sq=A.ndarray((J,G,2,C),dtype=F,buffer=A.sbuf);l=P6(r,sq,G,C);Y=A.ndarray((J,G,2,C),dtype=dt,buffer=A.sbuf);E.tensor_tensor(dst=Y,data1=r,data2=l.broadcast(3,C),op=A.multiply);Jv(r,e,b,h,B,M,T,G,C,I);E.dma_copy(dst=d[i,:,:,:,:],src=l);w=Jx(Y,0,T,C,G);x=Jx(Y,1,T,C,G);y=P7(N,b,h,T,K,G,C);Q=A.ndarray((J,G),dtype=F,buffer=A.sbuf);sums=A.ndarray((J,G),dtype=F,buffer=A.sbuf)
			for g in L(G):
				R=A.ndarray((J,X),dtype=F,buffer=A.psum)
				for U in L(0,g+1,AN):
					H=g+1-U
					if H>AN:H=AN
					st=A.ndarray((J,H*J),dtype=F,buffer=A.psum)
					for D in L(H):j=U+D;E.nc_matmul(dst=st[:,D*J:(D+1)*J],stationary=x[:,j*J:(j+1)*J],moving=w[:,g*J:(g+1)*J],accumulate=O)
					V=A.ndarray((J,H*J),dtype=dt,buffer=A.sbuf);E.activation(dst=V,op=A.exp,data=st,bias=f,scale=scale)
					if U+H-1==g:D=H-1;E.affine_select(dst=V[:,D*J:(D+1)*J],pattern=[[1,J]],channel_multiplier=-1,offset=0,cmp_op=A.greater_equal,on_true_tile=V[:,D*J:(D+1)*J],on_false_value=S)
					for D in L(H):j=U+D;E.nc_matmul(dst=R,stationary=V[:,D*J:(D+1)*J],moving=y[:,j*X:(j+1)*X],accumulate=j>0)
				m=A.ndarray((J,1),dtype=F,buffer=A.sbuf);E.reciprocal(dst=m,data=R[:,C:C+1]);n=A.ndarray((J,C),dtype=dt,buffer=A.sbuf);E.tensor_scalar(dst=n,data=R[:,0:C],op0=A.multiply,operand0=m,engine=E.engine.vector);E.tensor_copy(dst=sums[:,g:g+1],src=R[:,C:C+1],engine=E.engine.vector);E.dma_copy(dst=a.ap(pattern=[[K,J],[1,C]],offset=(b*T+g*J)*K+h*C),src=n)
			E.activation(dst=Q,op=A.log,data=sums);E.tensor_scalar(dst=Q,data=Q,op0=A.add,operand0=Z,engine=E.engine.vector);E.dma_copy(dst=c[i,:,:],src=Q)
		return a,c,e,d
	@K.jit
	def P9(qkv3,r5,inv3,o3,do3,lse3,cos2,sin2,scale):
		o=qkv3;B,T,N=o.shape;W=N//3;C=J;c=W//C;d=C//2;A8=B*c;G=T//J;D=A.float32;dt=o.dtype;p=A.ndarray((B,T,N),dtype=dt,buffer=A.shared_hbm);A9,AA=EI();AB,AC=Js(cos2,sin2,G,d)
		for i in L(A9,A8,AA):
			b=i//c;h=i%c;e=A.ndarray((J,G,2,C),dtype=dt,buffer=A.sbuf);Jv(e,r5,b,h,B,c,T,G,C,O);Z=A.ndarray((J,G,2,1),dtype=D,buffer=A.sbuf);E.dma_copy(dst=Z,src=inv3[i,:,:,:,:]);a=A.ndarray((J,G,2,C),dtype=dt,buffer=A.sbuf);E.tensor_tensor(dst=a,data1=e,data2=Z.broadcast(3,C),op=A.multiply);AD=EK(o3,b,h,T,W,C,G);q=EK(do3,b,h,T,W,C,G);AE=Jw(a,0,T,C,G);AF=Jw(a,1,T,C,G);AG=C1(q,T,C,G);AH=b*T*N+2*W+h*C;x=A.ndarray((J,G*C),dtype=dt,buffer=A.sbuf);E.dma_copy(dst=x,src=o.ap(pattern=[[N,J],[J*N,G],[1,C]],offset=AH));AI=C1(x,T,C,G);f=A.ndarray((J,G),dtype=D,buffer=A.sbuf);E.dma_copy(dst=f,src=lse3[i,:,:]);E.tensor_scalar(dst=f,data=f,op0=A.multiply,operand0=-Q);y=A.ndarray((J,G),dtype=D,buffer=A.sbuf);z=A.ndarray((J,G*C),dtype=D,buffer=A.sbuf);E.tensor_tensor(dst=z,data1=q,data2=AD,op=A.multiply);E.tensor_reduce(dst=y,op=A.add,data=z.reshape_dim(1,(G,C)),axis=2);k=A.ndarray((J,G*C),dtype=D,buffer=A.sbuf);l=A.ndarray((J,G*C),dtype=D,buffer=A.sbuf);m=A.ndarray((J,G,2,C),dtype=dt,buffer=A.sbuf)
			for H in L(0,G,B_):
				K=G-H
				if K>B_:K=B_
				U=(H+K)*J;pg=A.ndarray((J,K*U),dtype=dt,buffer=A.sbuf);r=A.ndarray((J,K*U),dtype=dt,buffer=A.sbuf)
				for F in L(K):
					g=H+F;A0=A.ndarray((J,C),dtype=D,buffer=A.psum)
					for P in L(0,g+1,AN):
						V=g+1-P
						if V>AN:V=AN
						w=V*J;M=F*U+P*J;s=A.ndarray((J,w),dtype=D,buffer=A.psum);E.nc_matmul(dst=s,stationary=AE[:,g*J:(g+1)*J],moving=AF[:,P*J:P*J+w],accumulate=O);E.activation(dst=pg[:,M:M+w],op=A.exp,data=s,bias=f[:,g:g+1],scale=scale)
						if P+V-1==g:n=M+(V-1)*J;E.affine_select(dst=pg[:,n:n+J],pattern=[[-1,J]],channel_multiplier=1,offset=0,cmp_op=A.greater_equal,on_true_tile=pg[:,n:n+J],on_false_value=S)
						A1=A.ndarray((J,w),dtype=D,buffer=A.psum);E.nc_matmul(dst=A1,stationary=AG[:,g*J:(g+1)*J],moving=AI[:,P*J:P*J+w],accumulate=O);E.scalar_tensor_tensor(dst=r[:,M:M+w],data=A1,op0=A.subtract,operand0=y[:,g:g+1],op1=A.multiply,operand1=pg[:,M:M+w]);A2=A.ndarray((J,w),dtype=dt,buffer=A.psum)
						for R in L(V):E.nc_transpose(dst=A2[:,R*J:(R+1)*J],data=r[:,M+R*J:M+(R+1)*J])
						A3=A.ndarray((J,w),dtype=dt,buffer=A.sbuf);E.tensor_copy(dst=A3,src=A2,engine=EJ())
						for R in L(V):j=P+R;E.nc_matmul(dst=A0,stationary=A3[:,R*J:(R+1)*J],moving=a[:,j,1,:],accumulate=j>0)
					E.tensor_scalar(dst=m[:,g,0,:],data=A0,op0=A.multiply,operand0=scale)
				for j in L(H+K):
					X=j-H
					if X<0:X=0
					t=A.ndarray((J,C),dtype=D,buffer=A.psum)
					for F in L(X,K):g=H+F;E.nc_matmul(dst=t,stationary=pg[:,F*U+j*J:F*U+(j+1)*J],moving=q[:,g*C:(g+1)*C],accumulate=F>X)
					if Jp and j>=H:E.tensor_copy(dst=l[:,j*C:(j+1)*C],src=t,engine=E.engine.scalar)
					else:E.tensor_tensor(dst=l[:,j*C:(j+1)*C],data1=l[:,j*C:(j+1)*C],data2=t,op=A.add)
					v=A.ndarray((J,C),dtype=D,buffer=A.psum)
					for F in L(X,K):g=H+F;E.nc_matmul(dst=v,stationary=r[:,F*U+j*J:F*U+(j+1)*J],moving=a[:,g,0,:],accumulate=F>X)
					if Jp and j>=H:E.tensor_scalar(dst=k[:,j*C:(j+1)*C],data=v,op0=A.multiply,operand0=scale,engine=E.engine.scalar)
					else:E.scalar_tensor_tensor(dst=k[:,j*C:(j+1)*C],data=v,op0=A.multiply,operand0=scale,op1=A.add,operand1=k[:,j*C:(j+1)*C])
			E.tensor_copy(dst=m[:,:,1,:],src=k.reshape_dim(1,(G,C)),engine=E.engine.scalar);A4=A.ndarray((J,G,C),dtype=dt,buffer=A.sbuf);E.tensor_copy(dst=A4,src=l.reshape_dim(1,(G,C)),engine=E.engine.scalar);E.dma_copy(dst=p.ap(pattern=[[N,J],[J*N,G],[1,C]],offset=b*T*N+2*W+h*C),src=A4);A5=dt;sq=A.ndarray((J,G,2,C),dtype=A5,buffer=A.sbuf);E.tensor_tensor(dst=sq,data1=m,data2=e,op=A.multiply);A6=A.ndarray((J,G,2,1),dtype=D,buffer=A.sbuf);E.tensor_reduce(dst=A6,op=A.add,data=sq,axis=3,keepdims=I);Y=A.ndarray((J,G,2,1),dtype=D,buffer=A.sbuf);E.tensor_tensor(dst=Y,data1=Z,data2=Z,op=A.multiply);E.tensor_tensor(dst=Y,data1=Y,data2=A6,op=A.multiply);E.tensor_scalar(dst=Y,data=Y,op0=A.multiply,operand0=Q/C);u=A.ndarray((J,G,2,C),dtype=A5,buffer=A.sbuf);E.tensor_tensor(dst=u,data1=e,data2=Y.broadcast(3,C),op=A.multiply);E.tensor_tensor(dst=u,data1=m,data2=u,op=A.subtract,engine=E.engine.unknown);A7=A.ndarray((J,G,2,C),dtype=dt,buffer=A.sbuf);E.tensor_tensor(dst=A7,data1=u,data2=Z.broadcast(3,C),op=A.multiply);gx=A.ndarray((J,G,2,C),dtype=dt,buffer=A.sbuf);AJ=A.ndarray((J,G,2,d),dtype=dt,buffer=A.sbuf);AK=A.ndarray((J,G,2,d),dtype=dt,buffer=A.sbuf);Jt(gx,A7,AB,AC,AJ,AK,d,I);Ju(gx,p,b,h,T,W,G,C,I)
		return p
	Jy=I
except P:Jy=O
def Jz(device):return(Jy and device.type==a)and not C0
def EL(where,err):
	A=where;global C0,Jq
	if not C0:C0=I;Jq=f"{A}: {W(err).__name__}: {err}";D(f"[nki_attn] kernel disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
def EM(kernel):A=kernel;return A[Jn]if Jn==2 else A
def PA():return EM(P4)
def PB():return EM(P5)
def J_(B,H,T,D):return B>0 and H>0 and 0<D<=J and T>0 and T%J==0 and T<=P3
def K0(scale,D):return G(scale)*D+P1
class PC(torch.autograd.Function):
	@N
	def forward(ctx,q,k,v,scale,shift):
		A=shift;B,D,T,E=q.shape;ctx.scale,ctx.shift=scale,A;ctx.fallback=O
		try:F,H=PA()(q,k,v,G(scale),G(A));C=F.view(B,T,D,E)
		except P as err:
			EL(BS,err);ctx.fallback=I;ctx.save_for_backward(q,k,v)
			with torch.no_grad():return GD(q,k,v,scale)
		ctx.save_for_backward(q,k,v,C,H);return C
	@N
	def backward(ctx,gy):
		if ctx.fallback:q,k,v=ctx.saved_tensors;A,D,E=Jr(q,k,v,gy,ctx.scale);return A,D,E,C,C
		q,k,v,J,K=ctx.saved_tensors;B,F,T,H=q.shape;I=F*H
		if not C0:
			try:L=gy.contiguous().view(B,T,I);M=J.view(B,T,I);N,O,Q=PB()(q,k,v,M,L,K,G(ctx.scale));R=Q.view(B,T,F,H).permute(0,2,1,3);return N,O,R,C,C
			except P as err:EL(Am,err)
		A,D,E=Jr(q,k,v,gy,ctx.scale);return A,D,E,C,C
def PD(q,k,v):
	if not Jz(q.device):return O
	if q.dim()!=4 or q.shape!=k.shape or q.shape!=v.shape:return O
	if not q.dtype==k.dtype==v.dtype or q.dtype not in(torch.bfloat16,torch.float32):return O
	if q.dtype==torch.float32 and not Jo:return O
	B,A,T,C=q.shape;return J_(B,A,T,C)
def attnk_nki_causal_attention(q,k,v,scale=C):
	A=G(scale)if scale is not C else Q/math.sqrt(q.shape[-1])
	if not PD(q,k,v):return GD(q,k,v,A if scale is not C else C)
	q=q if q.is_contiguous()else q.contiguous();k=k if k.is_contiguous()else k.contiguous();v=v if v.is_contiguous()else v.contiguous();return PC.apply(q,k,v,A,K0(A,q.shape[-1]))
def PE():return EM(P8)
def PF():return EM(P9)
def PG(cos,sin,T,Dh,dtype):
	A=cos.reshape(T,Dh);B=sin.reshape(T,Dh)
	if A.dtype!=dtype:A=A.to(dtype)
	if B.dtype!=dtype:B=B.to(dtype)
	if not A.is_contiguous():A=A.contiguous()
	if not B.is_contiguous():B=B.contiguous()
	return A,B
def K1(x,cos2,sin2):B,T,G,C=x.shape;A=C//2;c=cos2.reshape(1,T,1,A);s=sin2.reshape(1,T,1,A);D,E=x[...,:A],x[...,A:];r=torch.cat((D*c+E*s,D*-s+E*c),dim=-1);return F.rms_norm(r,(C,)).transpose(1,2)
def K2(qkv,cos2,sin2,H,scale):B,T,D=qkv.shape;C=D//3;A=C//H;q,k,v=qkv.split(C,dim=-1);q=K1(q.reshape(B,T,H,A),cos2,sin2);k=K1(k.reshape(B,T,H,A),cos2,sin2);v=v.reshape(B,T,H,A).transpose(1,2);return GD(q,k,v,scale)
def K3(qkv,cos2,sin2,H,scale,gy):
	with torch.enable_grad():A=qkv.detach().requires_grad_(I);y=K2(A,cos2,sin2,H,scale);B,=torch.autograd.grad(y,A,gy)
	return B
def PH(qkv,cos2,sin2,H,scale,shift):B=qkv.shape[0];return PE()(qkv,cos2,sin2,G(scale),G(shift))
def PI(qkv,r5,inv3,o3,go,lse3,cos2,sin2,H,scale):B=qkv.shape[0];return PF()(qkv,r5,inv3,o3,go,lse3,cos2,sin2,G(scale))
class PJ(torch.autograd.Function):
	@N
	def forward(ctx,qkv,cos2,sin2,H,scale,shift):
		D=shift;C=sin2;A=cos2;B,T,F=qkv.shape;G=F//3;J=G//H;ctx.scale,ctx.shift,ctx.H=scale,D,H;ctx.fallback=O
		try:K,L,M,N=PH(qkv,A,C,H,scale,D);E=K.view(B,T,H,J)
		except P as err:
			EL('rope forward',err);ctx.fallback=I;ctx.save_for_backward(qkv,A,C)
			with torch.no_grad():return K2(qkv,A,C,H,scale)
		ctx.save_for_backward(qkv,A,C,M,N,E,L);return E
	@N
	def backward(ctx,gy):
		if ctx.fallback:qkv,A,D=ctx.saved_tensors;return K3(qkv,A,D,ctx.H,ctx.scale,gy),C,C,C,C,C
		qkv,A,D,G,H,I,J=ctx.saved_tensors;B,T,E=qkv.shape;F=E//3
		if not C0:
			try:K=gy.contiguous().view(B,T,F);L=I.view(B,T,F);M=PI(qkv,G,H,L,K,J,A,D,ctx.H,ctx.scale);return M.view(B,T,E),C,C,C,C,C
			except P as err:EL('rope backward',err)
		return K3(qkv,A,D,ctx.H,ctx.scale,gy),C,C,C,C,C
def PK(qkv,cos,sin,n_head):
	if not Jz(qkv.device):return O
	if qkv.dim()!=3 or qkv.dtype not in(torch.bfloat16,torch.float32):return O
	if qkv.dtype==torch.float32 and not Jo:return O
	B,T,A=qkv.shape
	if A%3 or n_head<=0:return O
	C=A//3
	if C%n_head or C//n_head!=J:return O
	if not J_(B,n_head,T,J)or not qkv.is_contiguous():return O
	D=J//2;return cos.numel()==T*D and sin.numel()==T*D
def attnk_nki_rope_attention(qkv,cos,sin,n_head,scale=C):
	if not PK(qkv,cos,sin,n_head):return C
	A=J;B=G(scale)if scale is not C else Q/math.sqrt(A);D,E=PG(cos,sin,qkv.shape[1],A//2,qkv.dtype);return PJ.apply(qkv,D,E,n_head,B,K0(B,A))
nki_attn_fn=attnk_nki_causal_attention
USE_NKI_ATTN=attnk_USE_NKI_ATTN
nki_rope_attention=attnk_nki_rope_attention
USE_NKI_ATTN_ROPE=attnk_USE_NKI_ATTN_ROPE
import os,torch,torch.nn.functional as F
A9=128
embk_USE_NKI_EMB_BWD=I
Ak=4
BO=i(1,g(4,4))
PL=O
Az=512
GG=B(os.environ.get(AB,q))
EN=O
K4=C
K5=Ak*A9
def K6(idx,weight,out_dtype):return F.embedding(idx,weight).to(out_dtype)
def PM(dy,idx,V,C,wdtype):
	A=wdtype;g=dy.reshape(-1,C).to(A);ids=idx.reshape(-1)
	try:return torch.ops.aten.embedding_dense_backward(g,ids,V,-1,O)
	except P:dw=torch.zeros(V,C,dtype=A,device=dy.device);dw.index_add_(0,ids.long(),g);return dw
try:
	import nki as K,nki.language as A,nki.isa as E
	def EO(dst,src):E.dma_copy(dst=dst,src=src)
	def PN():
		if A.program_ndim()>0:return A.program_id(0),A.num_programs(0)
		return 0,1
	def PO(P,W):B=A.ndarray((P,W),dtype=A.float32,buffer=A.sbuf);E.iota(dst=B,pattern=[[1,W]],offset=0,channel_multiplier=0);return B
	def PP(idx3):
		A=idx3
		if V(A.shape)==3:return A[:,:,0]
		return A
	def GH(n,shape,dtype,buf):
		B=buf
		if n==1:return A.ndarray(shape,dtype=dtype,buffer=B),
		if n==2:return A.ndarray(shape,dtype=dtype,buffer=B),A.ndarray(shape,dtype=dtype,buffer=B)
		if n==3:return A.ndarray(shape,dtype=dtype,buffer=B),A.ndarray(shape,dtype=dtype,buffer=B),A.ndarray(shape,dtype=dtype,buffer=B)
		if n==4:return A.ndarray(shape,dtype=dtype,buffer=B),A.ndarray(shape,dtype=dtype,buffer=B),A.ndarray(shape,dtype=dtype,buffer=B),A.ndarray(shape,dtype=dtype,buffer=B)
		return A.ndarray(shape,dtype=dtype,buffer=B),A.ndarray(shape,dtype=dtype,buffer=B),A.ndarray(shape,dtype=dtype,buffer=B),A.ndarray(shape,dtype=dtype,buffer=B),A.ndarray(shape,dtype=dtype,buffer=B),A.ndarray(shape,dtype=dtype,buffer=B),A.ndarray(shape,dtype=dtype,buffer=B),A.ndarray(shape,dtype=dtype,buffer=B)
	def PQ(oh,io,idxb,ids,t,v0):E.tensor_scalar(dst=oh,data=io,op0=A.equal,operand0=ids[:,t:t+1])
	def PR(accs,oh,dyb,CC,acc_flag):
		for v in L(Ak):
			for c in L(CC):E.nc_matmul(dst=accs[v*CC+c],stationary=oh[:,v*A9:(v+1)*A9],moving=dyb[:,c*Az:(c+1)*Az],accumulate=acc_flag,is_stationary_onezero=PL)
	def PS(first):return first
	def PT(dw,accs,CC,C,v0):
		for v in L(Ak):
			B=A.ndarray((A9,C),dtype=A.float32,buffer=A.sbuf)
			for c in L(CC):E.tensor_copy(dst=B[:,c*Az:(c+1)*Az],src=accs[v*CC+c])
			EO(dst=dw[v0+v*A9:v0+(v+1)*A9,:],src=B)
	@K.jit
	def PU(idx3,dy3,V):
		D=dy3;B,C,F=D.shape;G,R=A.float32,A.bfloat16;dw=A.ndarray((V,F),dtype=G,buffer=A.shared_hbm);S,T=PN();N=V//B//T;U=N//Ak;H=F//Az;W=Ak*H;X=PO(B,K5);I=A.ndarray((B,C),dtype=G,buffer=A.sbuf);EO(dst=I,src=PP(idx3))
		for Y in L(U):
			J=(S*N+Y*Ak)*B;ids=A.ndarray((B,C),dtype=G,buffer=A.sbuf);E.tensor_scalar(dst=ids,data=I,op0=A.subtract,operand0=J*Q);O=GH(W,(B,Az),G,A.psum);K=GH(BO,(B,F),D.dtype,A.sbuf);Z=GH(2,(B,K5),R,A.sbuf)
			for t in L(g(BO-1,C)):EO(dst=K[t%BO],src=D[:,t,:])
			for t in L(C):
				M=t+BO-1
				if M<C and BO>1:EO(dst=K[M%BO],src=D[:,M,:])
				P=Z[t%2];PQ(P,X,I,ids,t,J);PR(O,P,K[t%BO],H,PS(t>0))
			PT(dw,O,H,F,J)
		return dw
	K7=I
except P:K7=O
def PV(where,err):
	A=where;global EN,K4
	if not EN:EN=I;K4=f"{A}: {W(err).__name__}: {err}";D(f"[nki_embedding] kernel disabled after failure in {A}: {W(err).__name__}: {err}",flush=I)
def PW(kernel):A=kernel;return A[GG]if GG==2 else A
def PX(idx):
	N=idx.numel();A=N//A9;flat=idx.reshape(N)
	if flat.dtype==torch.int32:return flat.contiguous().view(A9,A)
	if flat.dtype==torch.int64:
		try:return flat.contiguous().view(torch.int32).view(A9,A,2)
		except P:pass
	return flat.to(torch.int32).view(A9,A)
def PY(dy,C):N=dy.numel()//C;return dy.reshape(N,C).contiguous().view(A9,N//A9,C)
def PZ(idx_k,dy3,V):return PW(PU)(idx_k,dy3,B(V))
def K8():return K7 and not EN
def K9(t):return t.device.type==a
def Pa(idx,weight,out_dtype):A=weight;C,B=A.shape;N=idx.numel();D=2 if GG==2 else 1;return out_dtype==torch.bfloat16 and A.dtype==torch.float32 and idx.dtype in(torch.int32,torch.int64)and N>0 and N%A9==0 and B%Az==0 and B//Az*Ak in(1,2,4,8)and C%(A9*Ak*D)==0 and Ak in(1,2,4)
class KA(torch.autograd.Function):
	@N
	def forward(ctx,idx,weight,out_dtype,shadow=C):A=weight;out=F.embedding(idx,shadow)if shadow is not C else K6(idx,A,out_dtype);ctx.save_for_backward(idx);ctx.wshape=e(A.shape);ctx.wdtype=A.dtype;return out
	@N
	def backward(ctx,dy):
		idx,=ctx.saved_tensors;A,B=ctx.wshape
		if K8()and dy.dtype==torch.bfloat16 and K9(dy):
			try:dw=PZ(PX(idx),PY(dy,B),A);return C,dw,C,C
			except P as err:PV(Am,err)
		return C,PM(dy,idx,A,B,ctx.wdtype),C,C
def embk_embedding_fwd(idx,weight,out_dtype,shadow=C):
	B=out_dtype;A=weight
	if K8()and K9(A)and Pa(idx,A,B):
		if shadow is not C:return KA.apply(idx,A,B,shadow)
		return KA.apply(idx,A,B)
	return K6(idx,A,B)
embedding_fwd=embk_embedding_fwd
USE_NKI_EMB_BWD=embk_USE_NKI_EMB_BWD
class Linear(nn.Linear):
	def forward(self,x):
		inplace=_accum_inplace(self,x,'sq'if self.in_features==self.out_features else'wide')
		if USE_NKI_GEMM_FP8_NN and USE_NKI_GEMM and nki_gemm_fp8_nn.proj_active(self.training)and n(self,D0,C)==Gd:return nki_gemm_fp8_nn.c_proj(x,self.weight if inplace else _weight_for(self,x),inplace)
		if USE_NKI_GEMM and n(self,D0,C)==Gd:
			if USE_NKI_GEMM_FP8_NN:x=nki_gemm_fp8_nn.row2_cap(x)
			return _LinearGemm.apply(x,self.weight if inplace else _weight_for(self,x),O,inplace)
		if inplace:return _LinearAccum.apply(x,self.weight,O)
		return F.linear(x,_weight_for(self,x),C)
class LinearT(Linear):
	def __init__(self,in_features,out_features,bias=O):super().__init__(in_features,out_features,bias=O);self.weight=nn.Parameter(torch.empty(in_features,out_features))
	def forward(self,x):
		inplace=_accum_inplace(self,x,'t')
		if USE_NKI_GEMM and n(self,D0,C)=='fc':return _LinearGemm.apply(x,self.weight if inplace else _weight_for(self,x),I,inplace)
		if n(self,D0,C)=='head'and HEAD_DW_NSPLIT>1 and self.training and torch.is_grad_enabled()and self.weight.requires_grad:return _LmHeadDwSplit.apply(x,_weight_for(self,x))
		if inplace:return _LinearAccum.apply(x,self.weight,I)
		w=_weight_for(self,x);return torch.matmul(x.reshape(-1,x.shape[-1]),w).view(*x.shape[:-1],w.shape[1])
def _accum_inplace(m,x,tag):
	if not _ACCUM_TAGS or not(ES in _ACCUM_TAGS or tag in _ACCUM_TAGS):return O
	w=m.weight;g=w.grad;return g is not C and torch.is_grad_enabled()and w.requires_grad and w.dtype==x.dtype and g.dtype==w.dtype and g.shape==w.shape and g.is_contiguous()
def _weight_for(m,x):
	w=m.weight
	if w.dtype==x.dtype:return w
	if CAST_CACHE!=A1 and m.training and torch.is_grad_enabled()and w.requires_grad:return _MasterCast.apply(w,m,x.dtype)
	return w.to(dtype=x.dtype)
class _LinearAccum(torch.autograd.Function):
	@N
	def forward(ctx,x,w,transposed):
		ctx.save_for_backward(x);ctx.w=w;ctx.transposed=transposed
		if transposed:return torch.matmul(x.reshape(-1,x.shape[-1]),w).view(*x.shape[:-1],w.shape[1])
		return F.linear(x,w,C)
	@N
	def backward(ctx,dy):
		x,=ctx.saved_tensors;w=ctx.w;x2d=x.reshape(-1,x.shape[-1]);dy2d=dy.reshape(-1,dy.shape[-1])
		if ctx.transposed:dx=torch.matmul(dy2d,w.t());w.grad.addmm_(x2d.t(),dy2d)
		else:dx=torch.matmul(dy2d,w);w.grad.addmm_(dy2d.t(),x2d)
		return dx.view(*x.shape[:-1],dx.shape[-1]),C,C
class _LinearGemm(torch.autograd.Function):
	@N
	def forward(ctx,x,w,transposed,inplace):
		x2d=x.reshape(-1,x.shape[-1]);ctx.save_for_backward(x2d,w);ctx.w=w;ctx.transposed=transposed;ctx.inplace=inplace;ctx.lead=x.shape[:-1]
		if transposed:y=torch.matmul(x2d,w)
		else:y=nki_matmul(x2d,w,trans_b=I)
		return y.view(*x.shape[:-1],y.shape[-1])
	@N
	def backward(ctx,dy):
		x2d,w=ctx.saved_tensors;dy2d=dy.reshape(-1,dy.shape[-1])
		if not dy2d.is_contiguous():dy2d=dy2d.contiguous()
		if ctx.transposed:
			dx=nki_matmul(dy2d,w,trans_b=I)
			if ctx.inplace:ctx.w.grad.addmm_(x2d.t(),dy2d);dw=C
			else:dw=nki_matmul(x2d,dy2d,trans_a=I)
		else:
			dx=torch.matmul(dy2d,w)
			if ctx.inplace:ctx.w.grad.addmm_(dy2d.t(),x2d);dw=C
			else:dw=torch.matmul(dy2d.t(),x2d)
		return dx.view(*ctx.lead,dx.shape[-1]),dw,C,C
class _MasterCast(torch.autograd.Function):
	@N
	def forward(ctx,w,m,dtype):
		cache=n(m,'_cast_cache',C)
		if cache is C or m._cast_ver!=w._version or cache.dtype!=dtype:cache=w.detach().to(dtype=dtype);m._cast_cache,m._cast_ver=cache,w._version
		ctx.w=w;return cache
	@N
	def backward(ctx,dw):
		w=ctx.w;g=w.grad
		if g is not C and g.shape==dw.shape and g.dtype==w.dtype and g.is_contiguous():g.add_(dw);return C,C,C
		return dw.to(dtype=w.dtype),C,C
class _LinearSplitBwd(torch.autograd.Function):
	@N
	def forward(ctx,x,w):x2d=x.reshape(-1,x.shape[-1]);ctx.save_for_backward(x2d,w);ctx.lead=x.shape[:-1];return F.linear(x2d,w).view(*x.shape[:-1],w.shape[0])
	@N
	def backward(ctx,dy):x2d,w=ctx.saved_tensors;dy2d=dy.reshape(-1,dy.shape[-1]);dx=torch.matmul(dy2d,w).view(*ctx.lead,w.shape[1]);dw=torch.matmul(dy2d.t(),x2d);return dx,dw
class LinearSplitBwd(Linear):
	def forward(self,x):return _LinearSplitBwd.apply(x,self.weight.to(dtype=x.dtype))
_HEAD_DW_FIRED=0
_HEAD_DW_SAID=O
def _headdw_line(model,pad=O):head=n(model,KN,C);w=n(head,'weight',C);n_out=B(w.shape[-1])if w is not C else 0;return f"headdw:{"            "if pad else Y} nsplit={HEAD_DW_NSPLIT} fired={_HEAD_DW_FIRED} head_class={n(head,D0,C)!r} head_out={n_out} divides={HEAD_DW_NSPLIT<=1 or n_out>0 and n_out%HEAD_DW_NSPLIT==0}"
def _mixfuse_line(model,pad=O):blocks=A0(n(model,'mixer_layers',()));mod='loaded'if nki_mixtail is not C else'MISSING';return f"mixfuse:{"           "if pad else Y} fused={B(MIXER_FUSED)} module={mod} mixer_blocks={blocks}"+(f" {O_()}"if nki_mixtail is not C else Y)
def _head_dw(x2d,dy2d):
	global _HEAD_DW_FIRED,_HEAD_DW_SAID;k=HEAD_DW_NSPLIT;n_out=dy2d.shape[1]
	if k<=1 or n_out%k:
		if k>1 and not _HEAD_DW_SAID:_HEAD_DW_SAID=I;D(f"NOTE: R1_HEAD_DW_NSPLIT={k} does not divide the lm_head output {n_out} -- single GEMM, switch INERT")
		return torch.matmul(x2d.t(),dy2d)
	blk=n_out//k;_HEAD_DW_FIRED+=1
	if not _HEAD_DW_SAID:_HEAD_DW_SAID=I;acc_kib=x2d.shape[1]//128*((blk+1023)//1024)*512*4//1024;D(f"headdw:            nsplit={k} FIRED  x2d={A0(x2d.shape)} dy2d={A0(dy2d.shape)} -> {k} blocks of [{x2d.shape[1]},{blk}]  acc {acc_kib} KiB/partition (was {x2d.shape[1]//128*((n_out+1023)//1024)*512*4//1024})")
	return torch.cat([torch.matmul(x2d.t(),dy2d[:,i*blk:(i+1)*blk])for i in L(k)],dim=1)
class _LmHeadDwSplit(torch.autograd.Function):
	@N
	def forward(ctx,x,w):x2d=x.reshape(-1,x.shape[-1]);ctx.save_for_backward(x2d,w);ctx.lead=x.shape[:-1];y=torch.matmul(x2d,w);return y.view(*x.shape[:-1],y.shape[-1])
	@N
	def backward(ctx,dy):
		x2d,w=ctx.saved_tensors;dy2d=dy.reshape(-1,dy.shape[-1])
		if not dy2d.is_contiguous():dy2d=dy2d.contiguous()
		dx=torch.matmul(dy2d,w.t());dw=_head_dw(x2d,dy2d);return dx.view(*ctx.lead,dx.shape[-1]),dw
class _FusedQKV(torch.autograd.Function):
	@N
	def forward(ctx,x,wq,wk,wv,w3):ctx.save_for_backward(x,w3);return F.linear(x,w3,C)
	@N
	def backward(ctx,gy):x,w3=ctx.saved_tensors;n=w3.shape[0]//3;g2=gy.reshape(-1,gy.shape[-1]);gx=torch.matmul(g2,w3).view_as(x);gw3=torch.matmul(g2.t(),x.reshape(-1,x.shape[-1]));return gx,gw3[0:n],gw3[n:2*n],gw3[2*n:],C
class CausalSelfAttention(nn.Module):
	def __init__(self,config):
		super().__init__();self.n_head=config.n_head;self.n_embd=config.n_embd;self.head_dim,self.width=attn_geometry(config);self.c_q=Linear(config.n_embd,self.width,bias=O);self.c_k=Linear(config.n_embd,self.width,bias=O);self.c_v=Linear(config.n_embd,self.width,bias=O);self.c_proj=(LinearT if self.width<config.n_embd else Linear)(self.width,config.n_embd,bias=O);self.attn_gate=config.attn_gate;self.qk_scale=config.qk_scale if config.qk_scale else QK_SCALE
		if self.attn_gate:self.c_gate=Linear(config.n_embd,config.n_embd,bias=O);torch.nn.init.zeros_(self.c_gate.weight)
	def qkv_weight(self,dtype):
		v=n(self,Ge,C)
		if v is not C and v.dtype==dtype:return v
		return torch.cat((self.c_q.weight,self.c_k.weight,self.c_v.weight),dim=0).to(dtype=dtype)
	def forward(self,x,cos,sin,ve=C,ve_lambda=C,attn_scale=C):
		b,t,c=x.shape;c=self.width;w=self.qkv_weight(x.dtype)
		if w is n(self,Ge,C):qkv=_FusedQKV.apply(x,self.c_q.weight,self.c_k.weight,self.c_v.weight,w)
		else:qkv=F.linear(x,w,C)
		if USE_NKI_ATTN_ROPE and USE_NKI_ATTN and ve is C and attn_scale is C:
			y=nki_rope_attention(qkv,cos,sin,self.n_head,scale=self.qk_scale if self.qk_scale>S else C)
			if y is not C:
				_NARROW[ER]+=1;y=y.view(b,t,c)
				if self.attn_gate:g=torch.sigmoid(self.c_gate(x));y=y*(g if self.attn_gate==1 else 2.*g)
				return self.c_proj(y)
		q,k,v=qkv.split(self.width,dim=-1);q=q.reshape(b,t,self.n_head,self.head_dim);k=k.reshape(b,t,self.n_head,self.head_dim);v=v.reshape(b,t,self.n_head,self.head_dim)
		if ve is not C:v=v+ve_lambda.to(v.dtype)*ve.view(b,t,self.n_head,self.head_dim)
		q=rope_norm(q,cos,sin);k=rope_norm(k,cos,sin)
		if attn_scale is not C:q=q*attn_scale.view(1,self.n_head,1,1).to(q.dtype)
		v=v.transpose(1,2)
		if USE_NKI_ATTN and attn_scale is C:y=nki_attn_fn(q,k,v,scale=self.qk_scale if self.qk_scale>S else C).view(b,t,c)
		elif USE_NKILIB_ATTN:
			if self.qk_scale>S:q=q*(self.qk_scale*math.sqrt(self.head_dim))
			y=nki_causal_attention(q,k,v,'bhdt');y=y.view(b,c,t).transpose(1,2).contiguous()
		elif self.qk_scale>S:y=F.scaled_dot_product_attention(q,k,v,is_causal=I,scale=self.qk_scale);y=y.transpose(1,2).contiguous().view(b,t,c)
		else:y=F.scaled_dot_product_attention(q,k,v,is_causal=I);y=y.transpose(1,2).contiguous().view(b,t,c)
		if self.attn_gate:g=torch.sigmoid(self.c_gate(x));y=y*(g if self.attn_gate==1 else 2.*g)
		return self.c_proj(y)
class MLP(nn.Module):
	def __init__(self,config):super().__init__();hidden=config.mlp_ratio*config.n_embd;self.c_fc=(LinearT if config.wkn else Linear)(config.n_embd,hidden,bias=O);self.c_proj=Linear(hidden,config.n_embd,bias=O);self.c_proj.gemm_class=Gd
	def forward(self,x):
		if(USE_NKI_GEMM_FP8_NN or USE_NKI_GEMM_FP8_DW)and USE_NKI_FC_RELU2 and nki_gemm_fp8_nn.fc_active(self.training)and AP(self.c_fc,LinearT):inplace=_accum_inplace(self.c_fc,x,'t');h=nki_gemm_fp8_nn.fused_fc_relu2_nn(x,self.c_fc.weight if inplace else _weight_for(self.c_fc,x),inplace);return self.c_proj(h)
		if USE_NKI_GEMM_FP8 and nki_gemm_fp8.active(self.training)and AP(self.c_fc,LinearT):inplace=_accum_inplace(self.c_fc,x,'t');h=nki_gemm_fp8.fused_fc_relu2_fp8(x,self.c_fc.weight if inplace else _weight_for(self.c_fc,x),inplace);return self.c_proj(h)
		if USE_NKI_FC_RELU2 and AP(self.c_fc,LinearT):inplace=_accum_inplace(self.c_fc,x,'t');h=fused_fc_relu2(x,self.c_fc.weight if inplace else _weight_for(self.c_fc,x),inplace);return self.c_proj(h)
		return self.c_proj(relu_squared(self.c_fc(x)))
class Block(nn.Module):
	def __init__(self,config,layer_idx=0):super().__init__();self.has_attn=layer_idx in attn_block_ids(config);self.attn=CausalSelfAttention(config)if self.has_attn else C;self.mlp=MLP(config);self.layer_idx=layer_idx;self.resid_lambda=nn.Parameter(torch.ones(1));self.x0_lambda=nn.Parameter(torch.zeros(1));use_ve=self.has_attn and(config.value_emb==ES or config.value_emb==An and layer_idx%2==0);self.ve_lambda=nn.Parameter(torch.full((1,),.5))if use_ve else C;self.attn_scale=nn.Parameter(torch.ones(config.n_head))if self.has_attn and config.attn_scale=='learn'else C;use_bigram=config.bigram_buckets>0 and config.bigram_mode=='layers';self.bigram_lambda=nn.Parameter(torch.full((1,),.1))if use_bigram else C;use_mixer=not self.has_attn and mixer_mode(config)==Ck;self.mix_lambda=nn.Parameter(torch.zeros(1))if use_mixer else C
	def forward(self,x,x0,cos,sin,ve=C,xb=C,skip=C,skip_lambda=C,emit=O):
		if USE_NKI_MEGABLOCK_TAIL and xb is C:
			x_new,xn,x_in,x0=head2(x,x0,skip,self.resid_lambda,self.x0_lambda,skip_lambda,emit,key=self.layer_idx)
			if self.attn is C:
				if self.mix_lambda is C:xn2=xn
				elif MIXER_FUSED and nki_mixtail is not C and not(RESID_DROPOUT>S and self.training)and mixk_applies(x_new):x_new,xn2=mixk_mix_tail(x_new,xn,self.mix_lambda.to(x_new.dtype))
				else:mix_out=self.mix_lambda.to(x_new.dtype)*shift1(x_new);x_new,xn2=tail_norm2(x_new,mix_out)
			else:attn_out=self.attn(xn,cos,sin,ve if self.ve_lambda is not C else C,self.ve_lambda,self.attn_scale);x_new,xn2=tail_norm2(x_new,attn_out)
			mlp_out=self.mlp(xn2);return(x_new,mlp_out),x_in,x0
		fold_skip=USE_NKI_MEGABLOCK and xb is C
		if skip is not C and not fold_skip:x=x+skip_lambda.to(x.dtype)*skip
		if fold_skip:x,xn=head_skip(x,x0,skip,self.resid_lambda,self.x0_lambda,skip_lambda)
		elif USE_NKI_BLOCK_HEAD and xb is C:x,xn=block_head(x,x0,self.resid_lambda,self.x0_lambda)
		elif USE_NKI_GAIN_NORM and xb is C:x,xn=gain_residual_norm(x,x0,self.resid_lambda,self.x0_lambda)
		else:
			x=self.resid_lambda.to(x.dtype)*x+self.x0_lambda.to(x.dtype)*x0
			if xb is not C:x=x+self.bigram_lambda.to(x.dtype)*xb
			xn=norm(x)
		if self.attn is C:
			if self.mix_lambda is C:xn2=xn
			else:mix_out=self.mix_lambda.to(x.dtype)*shift1(x);x,xn2=tail_norm2(x,mix_out)
		else:attn_out=self.attn(xn,cos,sin,ve if self.ve_lambda is not C else C,self.ve_lambda,self.attn_scale);x,xn2=tail_norm2(x,attn_out)
		mlp_out=self.mlp(xn2);x=x+mlp_out;return x
class GPT(nn.Module):
	def __init__(self,config):
		A='tie_share';super().__init__();self.config=config;self.attn_layers=attn_block_ids(config);self.attn_dims=attn_dims(config);self.narrow=self.attn_dims[1]!=config.n_embd;self.mixer_layers=e(i for i in L(config.n_layer)if i not in self.attn_layers)if mixer_mode(config)==Ck else();padded_vocab_size=(config.vocab_size+63)//64*64;self.transformer=nn.ModuleDict({'wte':nn.Embedding(padded_vocab_size,config.n_embd),'h':nn.ModuleList([Block(config,layer_idx=i)for i in L(config.n_layer)])});self.tied_head=z(config.tie_embeddings);head_cls=LinearSplitBwd if self.tied_head else LinearT if config.wkn else Linear;self.lm_head=head_cls(config.n_embd,padded_vocab_size,bias=O)
		if self.tied_head:wte_w=self.transformer.wte.weight;assert self.lm_head.weight.shape==wte_w.shape,f"tie_embeddings: head {e(self.lm_head.weight.shape)} != wte {e(wte_w.shape)}";self.lm_head.weight=wte_w;_QL_FIRED[A]=_QL_FIRED.get(A,0)+1
		self.lm_head.gemm_class='head';self.value_emb_on=config.value_emb!=C2
		if self.value_emb_on:self.transformer['value_emb']=nn.Embedding(padded_vocab_size,config.n_embd)
		self.n_skip=config.n_layer//2 if config.unet_skip else 0;self.skip_lambda=nn.Parameter(torch.ones(self.n_skip))if self.n_skip>0 else C;self.bigram_on=config.bigram_buckets>0
		if self.bigram_on:assert config.bigram_buckets&config.bigram_buckets-1==0,'bigram_buckets must be a power of two';self.transformer['bigram']=nn.Embedding(config.bigram_buckets+1,config.n_embd)
		self.bigram_lambda=nn.Parameter(torch.ones(1))if self.bigram_on and config.bigram_mode==BQ else C;cos,sin=self._precompute_rotary_embeddings(config.sequence_len,self.attn_dims[0]);self.register_buffer('cos',cos,persistent=O);self.register_buffer('sin',sin,persistent=O)
	@torch.no_grad()
	def init_weights(self):
		A='tie_init'
		if self.tied_head:torch.nn.init.normal_(self.lm_head.weight,mean=S,std=TIE_INIT_STD);_QL_FIRED[A]=_QL_FIRED.get(A,0)+1
		else:
			torch.nn.init.normal_(self.transformer.wte.weight,mean=S,std=.8)
			if AP(self.lm_head,LinearT):w=self.lm_head.weight;w.copy_(torch.empty(w.shape[1],w.shape[0],dtype=w.dtype,device=w.device).normal_(mean=S,std=.001).t())
			else:torch.nn.init.normal_(self.lm_head.weight,mean=S,std=.001)
		if self.value_emb_on:torch.nn.init.normal_(self.transformer.value_emb.weight,mean=S,std=.8)
		if self.bigram_on:torch.nn.init.normal_(self.transformer.bigram.weight,mean=S,std=.8)
		s=math.sqrt(3.)*self.config.n_embd**-.5;n_layer=self.config.n_layer
		for(i,block)in A7(self.transformer.h):
			if block.attn is not C:torch.nn.init.uniform_(block.attn.c_q.weight,-s,s);torch.nn.init.uniform_(block.attn.c_k.weight,-s,s);torch.nn.init.uniform_(block.attn.c_v.weight,-s,s);torch.nn.init.zeros_(block.attn.c_proj.weight)
			if AP(block.mlp.c_fc,LinearT):w=block.mlp.c_fc.weight;w.copy_(torch.empty(w.shape[1],w.shape[0],dtype=w.dtype,device=w.device).uniform_(-.4*s,.4*s).t())
			else:torch.nn.init.uniform_(block.mlp.c_fc.weight,-.4*s,.4*s)
			torch.nn.init.zeros_(block.mlp.c_proj.weight);block.resid_lambda.fill_(INIT_RESID);block.x0_lambda.fill_(INIT_X0)
			if block.ve_lambda is not C:block.ve_lambda.fill_(.5)
			if block.attn_scale is not C:block.attn_scale.fill_(Q)
			if block.bigram_lambda is not C:block.bigram_lambda.fill_(.1)
			if block.mix_lambda is not C:block.mix_lambda.fill_(S)
		if self.skip_lambda is not C:self.skip_lambda.fill_(INIT_SKIP)
		if self.bigram_lambda is not C:self.bigram_lambda.fill_(Q)
		cos,sin=self._precompute_rotary_embeddings(self.config.sequence_len,self.attn_dims[0]);self.cos.copy_(cos);self.sin.copy_(sin)
	def _precompute_rotary_embeddings(self,seq_len,head_dim,base=C):
		if base is C:base=self.config.rope_base
		self._rope_base_used=base;device=self.transformer.wte.weight.device;freqs=torch.arange(0,head_dim,2,dtype=torch.float32,device=device);inv_freq=Q/base**(freqs/head_dim);t=torch.arange(seq_len,dtype=torch.float32,device=device);table=torch.outer(t,inv_freq);cos=table.cos().to(COMPUTE_DTYPE)[C,:,C,:];sin=table.sin().to(COMPUTE_DTYPE)[C,:,C,:];return cos,sin
	def estimate_flops(self,seq_len):
		nparams=BP(p.numel()for p in self.parameters());non_matmul=(0 if self.tied_head else self.transformer.wte.weight.numel())+2*self.config.n_layer
		if self.value_emb_on:non_matmul+=self.transformer.value_emb.weight.numel()
		if self.bigram_on:non_matmul+=self.transformer.bigram.weight.numel()
		non_matmul+=BP(p.numel()for p in self._arch_gain_params());h=self.config.n_head;d=self.attn_dims[0];attn_flops=V(self.attn_layers)*12*h*d*(seq_len//2);return 6*(nparams-non_matmul)+attn_flops
	def _arch_gain_params(self):
		gains=[]
		for block in self.transformer.h:
			if block.ve_lambda is not C:gains.append(block.ve_lambda)
			if block.attn_scale is not C:gains.append(block.attn_scale)
			if block.bigram_lambda is not C:gains.append(block.bigram_lambda)
			if block.mix_lambda is not C:gains.append(block.mix_lambda)
		if self.skip_lambda is not C:gains.append(self.skip_lambda)
		if self.bigram_lambda is not C:gains.append(self.bigram_lambda)
		return gains
	def setup_optimizer(self):
		A='tie_group';scalar_params=[];x0_params=[];matrix_params=[]
		for block in self.transformer.h:
			scalar_params.extend([block.resid_lambda,block.x0_lambda])
			if block.attn is not C:matrix_params.extend(block.attn.parameters())
			matrix_params.extend(block.mlp.parameters())
		scalar_params.extend(self._arch_gain_params());embedding_params=A0(self.transformer.wte.parameters())
		if self.value_emb_on:embedding_params.extend(self.transformer.value_emb.parameters())
		if self.bigram_on:embedding_params.extend(self.transformer.bigram.parameters())
		lm_head_params=A0(self.lm_head.parameters());tied_params=[]
		if self.tied_head:shared=self.lm_head.weight;embedding_params=[p for p in embedding_params if p is not shared];lm_head_params=[p for p in lm_head_params if p is not shared];lm_head_params=[shared]+lm_head_params;n_groups=BP(1 for lst in(embedding_params,lm_head_params,tied_params,scalar_params,x0_params,matrix_params)for p in lst if p is shared);assert n_groups==1,f"tie_embeddings: the shared tensor is in {n_groups} optimizer groups, must be exactly 1";_QL_FIRED[A]=_QL_FIRED.get(A,0)+1
		dmodel_lr_scale=(self.config.n_embd/768)**-.5;emb_betas,head_betas=(.8,.995),(.8,.96);param_groups=[AA(kind=Af,params=embedding_params,lr=EMBEDDING_LR*dmodel_lr_scale,betas=emb_betas,eps=1e-10,weight_decay=ADAMW_WD_EMB),AA(kind=Af,params=lm_head_params,lr=UNEMBEDDING_LR*dmodel_lr_scale,betas=head_betas,eps=1e-10,weight_decay=ADAMW_WD_HEAD),AA(kind=Af,params=scalar_params,lr=SCALAR_LR,betas=(.9,.95),eps=1e-10,weight_decay=S)]
		if x0_params:param_groups.append(AA(kind=Af,params=x0_params,lr=SCALAR_LR,betas=(.96,.95),eps=1e-10,weight_decay=S))
		if tied_params:param_groups.append(AA(kind=Af,params=tied_params,lr=TIE_LR*dmodel_lr_scale,betas=head_betas,eps=1e-10,weight_decay=.01))
		param_groups=[g for g in param_groups if g[l]]
		for shape in AO({p.shape for p in matrix_params}):params=[p for p in matrix_params if p.shape==shape];param_groups.append(AA(kind=AY,params=params,lr=MATRIX_LR,momentum=.95,beta2=MUON_BETA2,ns_steps=MUON_NS_STEPS,weight_decay=WEIGHT_DECAY))
		opt=MuonAdamW(param_groups);device=self.transformer.wte.weight.device
		for group in opt.param_groups:
			group[KO]=group[B0];group[CD]=torch.full((1,),G(group[B0]),dtype=torch.float32,device=device);group[Ba]=torch.full((1,),G(group[B0]),dtype=torch.float32)
			if group[AC]==AY:group[CE]=torch.full((1,),G(group[D1]),dtype=torch.float32,device=device);group[CF]=torch.full((1,),G(group[D1]),dtype=torch.float32)
			else:group[Gf]=torch.ones(1,dtype=torch.float32,device=device);group[Gg]=torch.ones(1,dtype=torch.float32,device=device);group[Ec]=torch.ones(1,dtype=torch.float32);group[Ed]=torch.ones(1,dtype=torch.float32)
		if OPT_TAIL_STAGE:opt.build_scalar_stage(device)
		for group in opt.param_groups:
			if group[AC]==Af:opt.stack_scalars(group)
		for group in opt.param_groups:
			if group[AC]==AY:opt.persist_muon_views(group)
		for group in opt.param_groups:
			if group[AC]!=AY:continue
			params=group[l];stack=opt.state[params[0]].get(Ag)
			if stack is C:continue
			for block in self.transformer.h:
				a=block.attn
				if a is C:continue
				try:i=Cj(j for(j,p)in A7(params)if p is a.c_q.weight)
				except StopIteration:continue
				same_order=params[i+1]is a.c_k.weight and params[i+2]is a.c_v.weight and stack[i].data_ptr()==a.c_q.weight.data_ptr()and stack[i+2].data_ptr()==a.c_v.weight.data_ptr()
				if same_order:a.qkv_view=stack[i:i+3].view(3*stack.shape[1],stack.shape[2])
		return opt
	def _wte_shadow(self,dtype):
		w=self.transformer.wte.weight
		if not EMB_SHADOW or w.dtype==dtype or not self.training:return C
		sh=n(self,'_shadow_cache',C)
		if sh is C or self._shadow_ver!=w._version or sh.dtype!=dtype:
			with torch.no_grad():sh=w.detach().to(dtype=dtype)
			self._shadow_cache,self._shadow_ver=sh,w._version
		return sh
	@torch.no_grad()
	def flush_gain_grads(self):
		if gain_partials_take is C:return
		parts=gain_partials_take()
		if not parts:return
		n_layer,n_skip=self.config.n_layer,self.n_skip;ones,rows,skip_rows,skip_j=[],[],[],[]
		for(i,block)in A7(self.transformer.h):
			r3=parts.pop(i,C)
			if r3 is C:continue
			rows.append(r3[0].reshape(1,-1));ones.append(block.resid_lambda);rows.append(r3[1].reshape(1,-1));ones.append(block.x0_lambda)
			if r3.shape[0]>2:skip_rows.append(r3[2].reshape(1,-1));skip_j.append(i-(n_layer-n_skip))
		if parts:raise Al(f"gain partials stashed for unknown blocks {AO(parts)}")
		sums=torch.cat(rows+skip_rows).sum(dim=1);n1=V(ones);skip=self.skip_lambda
		if n1==2*n_layer and V(skip_rows)==n_skip:
			views=n(self,'_gain_views',C);same=views is not C and all(p.grad is not C and p.grad.data_ptr()==v.data_ptr()for(p,v)in zip(ones,views))
			if same and(n_skip==0 or skip.grad is not C and skip.grad.data_ptr()==self._skip_grad.data_ptr()):
				self._gain_buf.add_(sums)
				if n_skip:self._skip_grad.add_(sums[n1:])
				return
			if all(p.grad is C for p in ones)and(n_skip==0 or skip.grad is C):
				self._gain_buf=sums;self._gain_views=[sums[k:k+1]for k in L(n1)]
				for(p,v)in zip(ones,self._gain_views):p.grad=v
				if n_skip:self._skip_grad=sums[n1:].clone();skip.grad=self._skip_grad
				return
		for(k,p)in A7(ones):
			g=sums[k:k+1]
			if p.grad is C:p.grad=g.clone()
			else:p.grad.add_(g)
		if skip_rows:
			if skip.grad is C:skip.grad=torch.zeros_like(skip)
			for(j,k)in zip(skip_j,L(n1,n1+V(skip_rows))):skip.grad[j:j+1].add_(sums[k:k+1])
	def bigram_code(self,idx):b=idx.shape[0];buckets=self.config.bigram_buckets;h=idx.to(torch.int32);prev=torch.cat((h.new_full((b,1),buckets),h[:,:-1]),dim=1);code=torch.bitwise_and(torch.bitwise_xor(h*36313,prev*27191),buckets-1);return torch.cat((code.new_full((b,1),buckets),code[:,1:]),dim=1)
	def forward(self,idx,targets=C,loss_reduction=Be):
		_b,t=idx.shape;assert t<=self.config.sequence_len;cos,sin=self.cos[:,:t],self.sin[:,:t];shadow=self._wte_shadow(COMPUTE_DTYPE);x=embedding_fwd(idx,self.transformer.wte.weight,COMPUTE_DTYPE)if shadow is C else embedding_fwd(idx,self.transformer.wte.weight,COMPUTE_DTYPE,shadow);xb=self.transformer.bigram(self.bigram_code(idx)).to(COMPUTE_DTYPE)if self.bigram_on else C
		if self.bigram_lambda is not C:x=x+self.bigram_lambda.to(COMPUTE_DTYPE)*xb;xb=C
		x=norm(x);x0=x;ve=self.transformer.value_emb(idx).to(COMPUTE_DTYPE)if self.value_emb_on else C;n_layer,n_skip=self.config.n_layer,self.n_skip;skips=[]
		if USE_NKI_MEGABLOCK_TAIL and xb is C:
			cur=x
			for(i,block)in A7(self.transformer.h):
				skip,skip_lambda=C,C
				if n_skip and i>=n_layer-n_skip:j=i-(n_layer-n_skip);skip,skip_lambda=skips.pop(),self.skip_lambda[j:j+1]
				emit=z(skips)and skips[-1]is SELF_SKIP;cur,x_in,x0=block(cur,x0,cos,sin,ve,xb,skip,skip_lambda,emit)
				if emit:skips[-1]=x_in
				if n_skip and i<n_skip:skips.append(SELF_SKIP)
			x=final_norm(cur[0],cur[1])
		else:
			for(i,block)in A7(self.transformer.h):
				skip,skip_lambda=C,C
				if n_skip and i>=n_layer-n_skip:j=i-(n_layer-n_skip);skip,skip_lambda=skips.pop(),self.skip_lambda[j:j+1]
				x=block(x,x0,cos,sin,ve,xb,skip,skip_lambda)
				if n_skip and i<n_skip:skips.append(x)
			x=norm(x)
		logits_raw=self.lm_head(x)
		if USE_NKI_LOSS_HEAD and targets is not C and CE_PATH==KC and logits_raw.shape[-1]==self.config.vocab_size:
			if MTP_SHIP and loss_reduction==Be and any(_MTP[AS])and torch.is_grad_enabled()and multi_available(logits_raw.device):
				loss=mtp_ship(logits_raw,targets)
				if loss is not C:return loss
			if BOS_EXCL=='post'and _BOSX[AS]and loss_reduction==Be and torch.is_grad_enabled():
				loss=bosx_loss(logits_raw,targets)
				if loss is not C:return loss
			loss=fused_loss(logits_raw,targets,reduction=loss_reduction,softcap=LOGIT_SOFTCAP);return loss
		logits=logits_raw[...,:self.config.vocab_size].float();logits=LOGIT_SOFTCAP*torch.tanh(logits/LOGIT_SOFTCAP)
		if targets is C:return logits
		vocab=logits.size(-1);logp=F.log_softmax(logits.reshape(-1,vocab),dim=-1);onehot=(targets.reshape(-1,1)==torch.arange(vocab,device=logits.device).view(1,-1)).to(logp.dtype);nll=-(logp*onehot).sum(dim=-1)
		if loss_reduction==AW:return nll.view_as(targets)
		return nll.sum()if loss_reduction==Gc else nll.mean();loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),ignore_index=-1,reduction=loss_reduction)
		if loss_reduction==AW:return loss.view_as(targets)
		return loss
def build_config(depth=DEPTH,n_embd=0):
	if not n_embd:base_dim=depth*ASPECT_RATIO;n_embd=(base_dim+HEAD_DIM-1)//HEAD_DIM*HEAD_DIM
	assert n_embd%HEAD_DIM==0,f"n_embd={n_embd} must be a multiple of HEAD_DIM={HEAD_DIM}";n_head=N_HEAD;return GPTConfig(sequence_len=ROPE_TABLE_LEN,vocab_size=TOKENIZER_VOCAB_SIZE,n_layer=depth,n_head=n_head,n_embd=n_embd,mlp_ratio=MLP_RATIO,value_emb=VALUE_EMB,unet_skip=UNET_SKIP,attn_gate=ATTN_GATE,attn_blocks=ATTN_BLOCKS,mixer=MIXER,qk_scale=QK_SCALE,rope_base=ROPE_BASE,tie_embeddings=TIE_EMBEDDINGS,attn_scale=ATTN_SCALE,bigram_buckets=BIGRAM_BUCKETS,bigram_mode=BIGRAM_MODE,wkn=WKN,head_dim=HEAD_DIM if n_head*HEAD_DIM!=n_embd else 0)
POLAR_EXPRESS_COEFFS=[(8.156554524902461,-22.48329292557795,15.878769915207462),(4.042929935166739,-2.808917465908714,.5000178451051316),(3.8916678022926607,-2.772484153217685,.5060648178503393),(3.285753657755655,-2.3681294933425376,.46449024233003106),(2.3465413258596377,-1.7097828382687081,.42323551169305323)]
class _ScalarStage:
	def __init__(self,device,slots):
		self.device=device;self.slots=slots;self.host,self.dev,self.sent={},{},{};self.views=I;self.copies=0
		for dt in AA.fromkeys(s[3]for s in slots):
			rows=[s for s in slots if s[3]==dt];self.host[dt]=torch.zeros(V(rows),dtype=dt);self.dev[dt]=torch.zeros(V(rows),dtype=dt,device=device)
			for(i,(owner,kd,kh,_,_))in A7(rows):owner[kh]=self.host[dt][i:i+1];owner[kd]=self.dev[dt][i:i+1]
		self._check()
		for(owner,_,kh,_,init)in slots:owner[kh].fill_(init)
		self.send()
	@torch.no_grad()
	def _check(self):
		err='R1_FUSE_DENY=stage_views'if'stage_views'in FUSE_DENY else C
		if err is C:
			try:
				for(dt,host)in self.host.items():
					n=host.numel();host.copy_(torch.arange(1,n+1,dtype=torch.float32).mul_(.125).to(dt));self.dev[dt].copy_(host);ones=torch.ones(2,dtype=dt,device=self.device);outs=[]
					for i in L(n):v=self.dev[dt][i:i+1];outs.append(ones*v);outs.append(torch.zeros(2,dtype=dt,device=self.device).lerp_(ones,v))
					got=torch.cat(outs).float().cpu();want=host.float().repeat_interleave(4)
					if not torch.equal(got,want):err=f"{dt} views read back {got.tolist()} != {want.tolist()}";break
			except P as e:err=f"{W(e).__name__}: {e}"
		if err is not C:
			print0(f"opttail: stage views FAILED their check ({err}); standalone (1,) tensors, one copy per scalar (the base form)");self.views=O
			for(owner,kd,_,dt,_)in self.slots:owner[kd]=torch.zeros(1,dtype=dt,device=self.device)
	@torch.no_grad()
	def send(self):
		if self.views:
			for(dt,host)in self.host.items():self.dev[dt].copy_(host);self.copies+=1
		else:
			for(owner,kd,kh,_,_)in self.slots:owner[kd].copy_(owner[kh]);self.copies+=1
		self.sent={dt:h.clone()for(dt,h)in self.host.items()};_OPTTAIL_FIRED[Cl]+=1
	def ensure_sent(self):
		if any(not torch.equal(h,self.sent[dt])for(dt,h)in self.host.items()):self.send()
	def desc(self):rows=AG.join(f"{v(dt).replace("torch.",Y)}:{h.numel()}"for(dt,h)in self.host.items());return f"views={self.views} rows={rows} copies_per_send={V(self.host)if self.views else V(self.slots)}"
class MuonAdamW(torch.optim.Optimizer):
	def __init__(self,param_groups):super().__init__(param_groups,defaults={});self.use_reduce_scatter=MUON_SHARD_RS!=A1;self.use_all_gather_into_tensor=MUON_SHARD_AG!=A1;self._op_ok={};self._ns_coefs={};self._grad_views=O;self._stage=C
	def op_supported(self,name,device):
		key=name,device.type
		if key not in self._op_ok:
			ok=name not in FUSE_DENY
			if ok:
				try:
					ok=self._probe_op(name,device)
					if not ok:print0(f"optim2: {name} gives wrong values on {device.type}; using the eager form")
				except P as err:print0(f"optim2: {name} unavailable on {device.type} ({W(err).__name__}: {err}); using the eager form");ok=O
			self._op_ok[key]=ok
		return self._op_ok[key]
	@N
	@torch.no_grad()
	def _probe_op(name,device):
		a=(torch.arange(Q,9.,device=device).view(2,2,2)/8.).contiguous();k=torch.full((1,),.25,dtype=torch.float32,device=device)
		if name==En:got=torch.cat((torch.addcmul(a,a,k).flatten(),torch.addcmul(a,a,a>=k).flatten()));want=torch.cat(((a+a*k).flatten(),(a+a*(a>=k)).flatten()))
		elif name==KP:got,want=torch.baddbmm(a,a,a),a+a@a
		elif name==KQ:got,want=torch.maximum(a,k),a.clamp_min(.25)
		elif name=='foreach_norm':got=torch.stack(torch._foreach_norm([a,a.to(torch.bfloat16),k],dtype=torch.float32));want=torch.stack([a.norm(),a.to(torch.bfloat16).float().norm(),k.norm()])
		elif name==KR:ab=a.to(torch.bfloat16);got=torch.linalg.vector_norm(ab,2,dim=-2,keepdim=I,dtype=torch.float32).square();want=ab.float().square().sum(dim=-2,keepdim=I)
		elif name==KS:ab=(a-.5).to(torch.bfloat16);got=(ab>=torch.zeros(1,dtype=torch.bfloat16,device=device)).float();want=(ab.float()>=0).float()
		elif name==EX:got=(a+(torch.eye(2)*.5).to(device))@a;want=.5*a+a@a
		elif name==KT:got,want=torch.add(a,a,alpha=.25),a+.25*a
		else:raise AF(f"unknown probe {name}")
		return z(torch.allclose(got.float().cpu(),want.float().cpu(),rtol=.01,atol=1e-06))
	def _ns_coefs_for(self,ns_steps,device,dtype):
		key=device.type,dtype,ns_steps;coefs=self._ns_coefs.get(key)
		if coefs is C:coefs=[e(torch.full((1,),G(v),dtype=dtype,device=device)for v in abc)for abc in POLAR_EXPRESS_COEFFS[:ns_steps]];self._ns_coefs[key]=coefs
		return coefs
	def _ns_fold_consts(self,ns_steps,n,device,dtype,want_eye):
		key=GM,device.type,dtype,ns_steps,n if want_eye else 0;got=self._ns_coefs.get(key)
		if got is C:vals=[e(G(t.detach().cpu().float()[0])for t in abc)for abc in self._ns_coefs_for(ns_steps,device,dtype)];eye=torch.eye(n,dtype=torch.float32);a_eyes=[(eye*a).to(dtype).to(device)for(a,_,_)in vals]if want_eye else C;got=a_eyes,[c for(_,_,c)in vals];self._ns_coefs[key]=got
		return got
	def _newton_schulz(self,group,x):
		tall=x.shape[-2]>x.shape[-1];coefs=self._ns_coefs_for(group[Gh],x.device,x.dtype);merged=NS_TENSOR_COEFS=='2';use_addcmul=merged and self.op_supported(En,x.device);use_baddbmm=merged and self.op_supported(KP,x.device);return self._newton_schulz_fold(group,x,tall)
		for(a_t,b_t,c_t)in coefs:
			if tall:xtx=x.mT@x;xx=xtx@xtx;inner=torch.addcmul(b_t*xtx,xx,c_t)if use_addcmul else b_t*xtx+c_t*xx;x=torch.baddbmm(a_t*x,x,inner)if use_baddbmm else a_t*x+x@inner
			else:xxt=x@x.mT;xx=xxt@xxt;inner=torch.addcmul(b_t*xxt,xx,c_t)if use_addcmul else b_t*xxt+c_t*xx;x=torch.baddbmm(a_t*x,inner,x)if use_baddbmm else a_t*x+inner@x
		return x
	def _newton_schulz_fold(self,group,x,tall):
		dev=x.device;afold=MUON_NS_AFOLD and self.op_supported(EX,dev);alpha=MUON_NS_ALPHA and self.op_supported(KT,dev);coefs=self._ns_coefs_for(group[Gh],dev,x.dtype);n=x.shape[-1]if tall else x.shape[-2];a_eyes,cvals=self._ns_fold_consts(group[Gh],n,dev,x.dtype,afold)if afold or alpha else(C,C)
		for(i,(a_t,b_t,c_t))in A7(coefs):
			if tall:xtx=x.mT@x;xx=xtx@xtx;inner=torch.add(b_t*xtx,xx,alpha=cvals[i])if alpha else b_t*xtx+c_t*xx;x=x@(inner+a_eyes[i])if afold else a_t*x+x@inner
			else:xxt=x@x.mT;xx=xxt@xxt;inner=torch.add(b_t*xxt,xx,alpha=cvals[i])if alpha else b_t*xxt+c_t*xx;x=(inner+a_eyes[i])@x if afold else a_t*x+inner@x
		_OPTTAIL_FIRED[EX]+=V(coefs)if afold else 0;_OPTTAIL_FIRED['alpha']+=V(coefs)if alpha else 0;return x
	def _fused_consts(self,group,dtype,device):
		key=f"fused_consts_{dtype}";consts=group.get(key)
		if consts is C:consts={Gi:torch.full((1,),1.01,dtype=dtype,device=device),Gj:torch.full((1,),1e-06,dtype=dtype,device=device),Gk:torch.full((1,),1e-10,dtype=torch.float32,device=device),KU:torch.full((1,),1-group[Eb],dtype=torch.float32,device=device),B6:torch.zeros(1,dtype=dtype if OPT_TAIL_MASK and dtype!=torch.float32 and self.op_supported(KS,device)else torch.float32,device=device)};group[key]=consts
		return consts
	@torch.no_grad()
	def _refresh_fused_scalars(self,group,dtype):
		if self._stage is not C:self._refresh_fused_host(group,dtype);self._stage.ensure_sent();return
		if CG not in group:
			dev=group[CD].device;shape=group[l][0].shape;group[Eo]=torch.full((1,),i(Q,shape[-2]/shape[-1])**.5,dtype=torch.float32);group[CH]=torch.zeros(1,dtype=torch.float32);group[Bg]=torch.zeros(1,dtype=torch.float32);group[D2]=torch.zeros(1,dtype=torch.float32);group[CG]=torch.zeros(1,dtype=torch.float32,device=dev);group[D3]=torch.zeros(1,dtype=torch.float32,device=dev);group[D4]=torch.zeros(1,dtype=dtype);group[Ep]=torch.zeros(1,dtype=dtype,device=dev)
			if dtype==torch.float32:group[CI]=group[CE]
			else:group[D5]=torch.zeros(1,dtype=dtype);group[CI]=torch.zeros(1,dtype=dtype,device=dev)
		group[CH].fill_(G(group[B1]));torch.mul(group[Ba],group[Eo],out=group[Bg]);torch.mul(group[Bg],group[CH],out=group[D2]);group[CG].copy_(group[Bg]);group[D3].copy_(group[D2]);m_lo=group[CF].to(dtype);group[D4].copy_(Q-m_lo);group[Ep].copy_(group[D4])
		if dtype!=torch.float32:group[D5].copy_(m_lo);group[CI].copy_(group[D5])
	@torch.no_grad()
	def _refresh_fused_host(self,group,dtype):
		assert group.get(D6)==dtype,f"opttail stage: group staged for {group.get(D6)}, refreshed as {dtype}";group[CH].fill_(G(group[B1]));torch.mul(group[Ba],group[Eo],out=group[Bg]);torch.mul(group[Bg],group[CH],out=group[D2]);m_lo=group[CF].to(dtype);group[D4].copy_(Q-m_lo)
		if dtype!=torch.float32:group[D5].copy_(m_lo)
	@torch.no_grad()
	def build_scalar_stage(self,device):
		muon=[g for g in self.param_groups if g[AC]==AY and g[l]];slots=[]
		for g in self.param_groups:
			slots.append((g,CD,Ba,torch.float32,G(g[B0])))
			if g[AC]==AY:slots.append((g,CE,CF,torch.float32,G(g[D1])))
		for g in muon:
			p0=g[l][0];dt=p0.dtype;g[D6]=dt;g[Eo]=torch.full((1,),i(Q,p0.shape[-2]/p0.shape[-1])**.5,dtype=torch.float32);g[CH]=torch.zeros(1,dtype=torch.float32);slots+=[(g,CG,Bg,torch.float32,S),(g,D3,D2,torch.float32,S),(g,Ep,D4,dt,S)]
			if dt!=torch.float32:slots.append((g,CI,D5,dt,S))
		self._stage=_ScalarStage(device,slots)
		for g in muon:
			if g[D6]==torch.float32:g[CI]=g[CE]
	@torch.no_grad()
	def stage_step_scalars(self):
		for g in self.param_groups:
			if g[AC]==AY and g[l]:self._refresh_fused_host(g,g[D6])
		self._stage.ensure_sent()
	def _fused_normalize(self,group,x):
		c=self._fused_consts(group,x.dtype,x.device);nrm=x.norm(dim=(-2,-1),keepdim=I)
		if self.op_supported(En,x.device):denom=torch.addcmul(c[Gj],nrm,c[Gi])
		else:denom=nrm*c[Gi]+c[Gj]
		return x/denom
	def _fused_tail(self,group,g,w,second,red_dim):
		A='red_size_t';dev=g.device;c=self._fused_consts(group,g.dtype,dev);use_max=self.op_supported(KQ,dev);use_addcmul=self.op_supported(En,dev)
		if A not in group:group[A]=torch.full((1,),G(g.size(red_dim)),dtype=torch.float32,device=dev)
		if OPT_TAIL_VNORM and self.op_supported(KR,dev):v_sum=torch.linalg.vector_norm(g,2,dim=red_dim,keepdim=I,dtype=torch.float32).square();v_mean=v_sum/group[A];_OPTTAIL_FIRED[EV]+=1
		else:v_mean=g.float().square().mean(dim=red_dim,keepdim=I);v_sum=v_mean*group[A]
		v_norm=v_sum.sum(dim=(-2,-1),keepdim=I).sqrt();second.lerp_(v_mean,c[KU]);floored=torch.maximum(second,c[Gk])if use_max else second.clamp_min(1e-10);step_size=floored.rsqrt();scaled=v_sum*step_size.square();v_norm_new=scaled.sum(dim=(-2,-1),keepdim=I).sqrt();v_norm_new=torch.maximum(v_norm_new,c[Gk])if use_max else v_norm_new.clamp_min(1e-10);g=g*(step_size*(v_norm/v_norm_new)).to(g.dtype);mask=g*w>=c[B6]
		if OPT_TAIL_MASK and mask is not C and c[B6].dtype!=torch.float32:_OPTTAIL_FIRED[EU]+=1
		if mask is C:return group[CG]*g+group[D3]*w
		t1=group[CG]*g;t2=group[D3]*w;return torch.addcmul(t1,t2,mask)if use_addcmul else t1+t2*mask
	@torch.no_grad()
	def persist_muon_views(self,group):
		params=group[l]
		if not params:return
		p0=params[0];state=self.state[p0];n,shape=V(params),p0.shape;world=dist.get_world_size()if dist.is_available()and dist.is_initialized()else 1;n_rows=(n+world-1)//world*world;w=torch.zeros(n_rows,*shape,dtype=p0.dtype,device=p0.device)
		for(i,p)in A7(params):w[i].copy_(p);p.data=w[i]
		state[Ag]=w;state[Cu]=I;g=torch.zeros(n_rows,*shape,dtype=p0.dtype,device=p0.device)
		for(i,p)in A7(params):p.grad=g[i]
		state[AZ]=g;state[Eq]=I;group[KV]=g;self._grad_views=I
	@torch.no_grad()
	def stage_muon_stack(self,group):
		params=group[l];p0=params[0];state=self.state[p0]
		if AZ not in state:state[AZ]=torch.empty(V(params),*p0.shape,dtype=p0.dtype,device=p0.device)
		g=state[AZ]
		if not state.get(Eq):
			for(i,p)in A7(params):
				if p.grad is C:g[i].zero_()
				else:g[i].copy_(p.grad)
		state[KW]=I;return g
	@torch.no_grad()
	def zero_grad(self,set_to_none=I):
		for group in self.param_groups:
			stack=group.get(KV)
			if stack is not C:stack.zero_();continue
			for p in group[l]:
				if p.grad is C:continue
				p.grad=C
	@N
	def _adamw_update(p,grad,exp_avg,exp_avg_sq,bias1,bias2,lr_t,beta1,beta2,eps,weight_decay):
		if weight_decay>0:p.mul_(1-lr_t*weight_decay)
		exp_avg.lerp_(grad,1-beta1);exp_avg_sq.lerp_(grad.square(),1-beta2);update=exp_avg/bias1/((exp_avg_sq/bias2).sqrt()+eps);p.sub_(update*lr_t)
	@torch.no_grad()
	def _adamw_step(self,group):
		beta1,beta2=group[GO];group[Ao]=group.get(Ao,0)+1;group[Ec].fill_(1-beta1**group[Ao]);group[Ed].fill_(1-beta2**group[Ao]);group[Gf].copy_(group[Ec]);group[Gg].copy_(group[Ed]);bias1=group[Gf];bias2=group[Gg];lr_t=group[CD];stacked=group.get(BY)
		if stacked is not C:
			if not group.get(BZ):self.stage_scalar_grads(group,distributed=O)
			group[BZ]=O;self._adamw_update(group[C6],group[C7],group[GP],group[GQ],bias1,bias2,lr_t,beta1,beta2,group[Cr],group[B1])
		skip=group.get(GN,())
		for p in group[l]:
			if p.grad is C or x(p)in skip:continue
			grad=p.grad;state=self.state[p]
			if not state:state[Cs]=torch.zeros_like(p);state[Ct]=torch.zeros_like(p)
			self._adamw_update(p,grad,state[Cs],state[Ct],bias1,bias2,lr_t,beta1,beta2,group[Cr],group[B1])
	@torch.no_grad()
	def stack_scalars(self,group):
		small=nki_optim.ADAMW_STACK_SMALL if nki_optim is not C else 0;ones=[p for p in group[l]if p.numel()==1 or 1<p.numel()<=small]
		if V(ones)<2:return
		master=torch.cat([p.detach().reshape(-1)for p in ones]).contiguous();group[BY]=ones;group[GN]=A_(x(p)for p in ones);group[C6]=master;group[C7]=torch.zeros_like(master);group[GP]=torch.zeros_like(master);group[GQ]=torch.zeros_like(master);group[BZ]=O;off=0
		for p in ones:p.data=master[off:off+p.numel()].view(p.shape);off+=p.numel()
	@torch.no_grad()
	def stage_scalar_grads(self,group,distributed):
		flat=torch.cat([(p.grad if p.grad is not C else torch.zeros_like(p)).reshape(-1)for p in group[BY]])
		if distributed:dist.all_reduce(flat,op=dist.ReduceOp.AVG)
		group[C7]=flat;group[BZ]=I;return flat
	def configure_sharding(self,device):
		distributed=dist.is_available()and dist.is_initialized()and dist.get_world_size()>1
		if not(MUON_SHARD and distributed):self.use_reduce_scatter=O;self.use_all_gather_into_tensor=O;return'single process: the shard is the whole stack, no collectives'
		world,rank=dist.get_world_size(),dist.get_rank();base=torch.arange(4*world,dtype=torch.float32,device=device)
		try:
			out=torch.zeros(4,dtype=torch.float32,device=device);dist.reduce_scatter_tensor(out,(base+G(rank)).contiguous(),op=dist.ReduceOp.AVG);want=torch.arange(4*rank,4*rank+4,dtype=torch.float32)+(world-1)/2;self.use_reduce_scatter=z(torch.allclose(out.cpu(),want))
			if not self.use_reduce_scatter:print0(f"muon shard: reduce_scatter_tensor returned {out.cpu().tolist()} (wanted {want.tolist()}); using all_reduce + slice")
		except P as err:print0(f"muon shard: reduce_scatter_tensor unavailable ({W(err).__name__}: {err}); using all_reduce + slice");self.use_reduce_scatter=O
		try:
			full=torch.zeros(4*world,dtype=torch.float32,device=device);dist.all_gather_into_tensor(full,(base[:4]+G(rank)).contiguous());want=torch.cat([torch.arange(4,dtype=torch.float32)+r for r in L(world)]);self.use_all_gather_into_tensor=z(torch.allclose(full.cpu(),want))
			if not self.use_all_gather_into_tensor:print0(f"muon shard: all_gather_into_tensor returned {full.cpu().tolist()} (wanted {want.tolist()}); using list-form all_gather")
		except P as err:print0(f"muon shard: all_gather_into_tensor unavailable ({W(err).__name__}: {err}); using list-form all_gather");self.use_all_gather_into_tensor=O
		return f"world {world}: reduce={"reduce_scatter_tensor"if self.use_reduce_scatter else"all_reduce + local slice"} gather={"all_gather_into_tensor"if self.use_all_gather_into_tensor else"all_gather (list)"} payload={MUON_SHARD_GATHER} slab={MUON_SHARD_SLAB}"
	def _all_gather(self,out_full,shard):
		if not(dist.is_available()and dist.is_initialized())or dist.get_world_size()==1:out_full.copy_(shard);return
		if self.use_all_gather_into_tensor:dist.all_gather_into_tensor(out_full,shard.contiguous())
		else:dist.all_gather(A0(out_full.chunk(dist.get_world_size(),dim=0)),shard.contiguous())
	@N
	def _owned_index(j,s,world,rank):G,t=divmod(j,s);return G*world*s+rank*s+t
	@torch.no_grad()
	def stage_muon_grads(self,group,distributed,norm=I):
		params=group[l];p0=params[0];state=self.state[p0];shape=p0.shape;world=dist.get_world_size()if distributed else 1;rank=dist.get_rank()if distributed else 0;n=V(params);k=(n+world-1)//world;n_pad=k*world
		if Bb not in state:
			s=k;state[KX],state[Gl],state[KY],state[KZ]=k,s,world,rank
			if AZ not in state:state[AZ]=torch.zeros(n_pad,*shape,dtype=p0.dtype,device=p0.device)
			state[Bb]=torch.zeros(k,*shape,dtype=p0.dtype,device=p0.device);state[Ka]=torch.zeros(k,*shape,dtype=p0.dtype,device=p0.device);state[D7]=torch.zeros(k,*shape,dtype=p0.dtype,device=p0.device);state_shape=(k,shape[-2],1)if shape[-2]>=shape[-1]else(k,1,shape[-1]);state[D8]=torch.zeros(state_shape,dtype=torch.float32,device=p0.device)
			if Ag not in state:state[Ag]=torch.zeros(n_pad,*shape,dtype=p0.dtype,device=p0.device)
		g=state[AZ]
		if not state.get(Eq):
			for(i,p)in A7(params):
				if p.grad is C:g[i].zero_()
				else:g[i].copy_(p.grad)
		shard=state[Bb];s=state[Gl]
		if world==1:shard.copy_(g)
		elif self.use_reduce_scatter:
			for G in L(k//s):dist.reduce_scatter_tensor(shard[G*s:(G+1)*s],g[G*world*s:(G+1)*world*s],op=dist.ReduceOp.AVG)
		else:
			for G in L(k//s):dist.all_reduce(g[G*world*s:(G+1)*world*s],op=dist.ReduceOp.AVG)
			for j in L(k):shard[j].copy_(g[self._owned_index(j,s,world,rank)])
		state[Gm]=I;return shard.float().square().sum()if norm else C
	@torch.no_grad()
	def _muon_update(self,group,g,w,momentum_buffer,second,shape):momentum_t=group[CE];momentum_buffer.lerp_(g,group[Ep]);g=g.lerp_(momentum_buffer,group[CI]);x=g.bfloat16()if COMPUTE_DTYPE==torch.bfloat16 else g;x=self._fused_normalize(group,x);x=self._newton_schulz(group,x);g=x;red_dim=-1 if shape[-2]>=shape[-1]else-2;return self._fused_tail(group,g,w,second,red_dim);v_mean=g.float().square().mean(dim=red_dim,keepdim=I);red_dim_size=g.size(red_dim);v_norm=(v_mean.sum(dim=(-2,-1),keepdim=I)*red_dim_size).sqrt();second.lerp_(v_mean,1-group[Eb]);step_size=second.clamp_min(1e-10).rsqrt();scaled=v_mean*red_dim_size*step_size.square();v_norm_new=scaled.sum(dim=(-2,-1),keepdim=I).sqrt();g=g*(step_size*(v_norm/v_norm_new.clamp_min(1e-10))).to(g.dtype);lr=group[CD]*i(Q,shape[-2]/shape[-1])**.5;wd=group[B1];mask=g*w>=0;return lr*g+lr*wd*w*mask
	@torch.no_grad()
	def _muon_step_sharded(self,group,state,shape):
		params=group[l];n=V(params)
		if not state.get(Gm):self.stage_muon_grads(group,distributed=O)
		state[Gm]=O;k,s,world,rank=state[KX],state[Gl],state[KY],state[KZ];g=state[Bb];w=state[Ka]
		if nki_optim is not C and nki_optim.MUON_SHARD_VIEW!=A1 and MUON_SHARD_GATHER==l:self._refresh_fused_scalars(group,params[0].dtype);self._muon_finish_fp32(group,state,g,w,k,s,world,rank,shape);return
		raise Al('fp32master: the sharded Muon step missed the slab-0 view finish: refusing the per-matrix finish, which has no fp32 master')
	@torch.no_grad()
	def _muon_finish_fp32(self,group,state,g,w,k,s,world,rank,shape):
		A='master32'
		if s!=k or not state.get(Cu):raise Al(f"fp32master: needs the slab-0 persistent-view finish: s={s} k={k} param_views={state.get(Cu)}")
		stack=state[Ag];w_view=stack[rank*k:(rank+1)*k];master=state.get(A)
		if master is C:master=state[A]=w_view.to(torch.float32,copy=I);_FP32M[GK]+=master.numel()*4;D(f"fp32master[rank {rank}]: group{e(shape)} rows [{rank*k}, {(rank+1)*k}) of {world*k}: fp32 master <- the {w_view.dtype} rows ({master.numel()*4:,} B)",flush=I)
		upd=self._muon_update(group,g,w_view,state[D7],state[D8],shape);assert upd.dtype==torch.float32 and upd.shape==master.shape,f"fp32master: the update is {upd.dtype} {e(upd.shape)}";master.sub_(upd);w.copy_(master);self._all_gather(stack[0:world*k],w);_FP32M[AV]+=1
	@torch.no_grad()
	def _muon_step(self,group):
		params=group[l]
		if not params:return
		p0=params[0];state=self.state[p0];shape=p0.shape;self._muon_step_sharded(group,state,shape);return
		if D7 not in state:state[D7]=torch.zeros(V(params),*shape,dtype=p0.dtype,device=p0.device)
		if D8 not in state:state_shape=(V(params),shape[-2],1)if shape[-2]>=shape[-1]else(V(params),1,shape[-1]);state[D8]=torch.zeros(state_shape,dtype=torch.float32,device=p0.device)
		if AZ not in state:state[AZ]=torch.empty(V(params),*shape,dtype=p0.dtype,device=p0.device)
		if Ag not in state:state[Ag]=torch.empty(V(params),*shape,dtype=p0.dtype,device=p0.device)
		g=state[AZ];w=state[Ag];copy_g=not(state.get(Eq)or state.pop(KW,O));copy_w=not state.get(Cu)
		for(i,p)in A7(params):
			if copy_g:
				if p.grad is C:g[i].zero_()
				else:g[i].copy_(p.grad)
			if copy_w:w[i].copy_(p)
		self._refresh_fused_scalars(group,p0.dtype);upd=self._muon_update(group,g,w,state[D7],state[D8],shape);w.sub_(upd if upd.dtype==w.dtype else upd.to(w.dtype))
		if copy_w:torch._foreach_copy_(params,A0(w.unbind(0)))
	@torch.no_grad()
	def step(self):
		fused_adamw=nki_optim is not C and nki_optim.adamw_fused_step(self)
		for group in self.param_groups:
			if group[AC]==Af:
				if fused_adamw:continue
				self._adamw_step(group)
			elif group[AC]==AY:self._muon_step(group)
			else:raise AF(f"unknown optimizer kind: {group[AC]}")
class GradReducer:
	def __init__(self,params,device,optimizer=C):
		self.params=A0(params);self.bad_steps=torch.zeros((),dtype=torch.float32,device=device);self.last_gnorm=torch.zeros((),dtype=torch.float32,device=device);self.host_gate=OPT_TAIL_HOSTGATE;self.gnorm_host=S;self.bad_host=0;self.optimizer=optimizer;self.muon_groups=[g for g in optimizer.param_groups if g[AC]==AY]if optimizer is not C and MUON_SHARD else[];self.stack_groups=[g for g in optimizer.param_groups if g.get(BY)]if optimizer is not C else[];self.skip_ids=A_([x(p)for g in self.muon_groups for p in g[l]]+[x(p)for g in self.stack_groups for p in g[BY]]);self.shard_sq=torch.zeros(1,dtype=torch.float32,device=device);self.norm_muon_groups=[];self.norm_muon_ids=A_();self.norm_scalar_params=[];self.norm_ids=A_();self.use_foreach_norm=O
		if self.host_gate and nki_optim is not C and nki_optim.USE_NKI_GRADNORM and self.muon_groups and GRADNORM_STACK==A1 and not n(nki_optim,'REDUCE_HOST_GATE',O):print0('opttail: hostgate REFUSED -- this nki_optim.reduce_fused predates it (no REDUCE_HOST_GATE); the device gate stays');self.host_gate=O
	@torch.no_grad()
	def reduce(self,clip):
		if nki_optim is not C and nki_optim.USE_NKI_GRADNORM and self.muon_groups and GRADNORM_STACK==A1:return nki_optim.reduce_fused(self,clip)
		distributed=dist.is_available()and dist.is_initialized();sq=C;foreach_grads=[]if self.use_foreach_norm else C
		for p in self.params:
			if p.grad is C or x(p)in self.skip_ids:continue
			if distributed:dist.all_reduce(p.grad,op=dist.ReduceOp.AVG)
			if foreach_grads is not C:foreach_grads.append(p.grad);continue
			if x(p)in self.norm_ids:continue
			s=p.grad.float().square().sum();sq=s if sq is C else sq+s
		if foreach_grads:s=torch.stack(torch._foreach_norm(foreach_grads,dtype=torch.float32)).square().sum();sq=s if sq is C else sq+s
		for group in self.norm_muon_groups:g=self.optimizer.stage_muon_stack(group);s=g.float().square().sum();sq=s if sq is C else sq+s
		if self.norm_scalar_params:flat=torch.cat([p.grad if p.grad is not C else torch.zeros_like(p)for p in self.norm_scalar_params]);s=flat.float().square().sum();sq=s if sq is C else sq+s
		for group in self.stack_groups:flat=self.optimizer.stage_scalar_grads(group,distributed);s=flat.float().square().sum();sq=s if sq is C else sq+s
		if self.muon_groups:
			shard_sq=C
			for group in self.muon_groups:s=self.optimizer.stage_muon_grads(group,distributed);shard_sq=s if shard_sq is C else shard_sq+s
			self.shard_sq.copy_(shard_sq)
			if distributed:dist.all_reduce(self.shard_sq,op=dist.ReduceOp.SUM)
			s=self.shard_sq.sum();sq=s if sq is C else sq+s
		gnorm=sq.sqrt();finite=C if self.host_gate else torch.isfinite(gnorm)
		if clip>0:
			scale=torch.clamp(clip/(gnorm+1e-06),max=Q)
			for p in self.params:
				if p.grad is not C and x(p)not in self.skip_ids and x(p)not in self.norm_muon_ids:p.grad.mul_(scale)
			for group in self.norm_muon_groups:self.optimizer.state[group[l][0]][AZ].mul_(scale)
			for group in self.stack_groups:group[C7].mul_(scale)
			for group in self.muon_groups:self.optimizer.state[group[l][0]][Bb].mul_(scale)
		if self.host_gate:return gnorm
		self.last_gnorm.copy_(torch.where(finite,gnorm,torch.zeros_like(gnorm)));self.bad_steps+=(~finite).to(torch.float32);return finite
	def gate_host(self,gnorm):assert gnorm.dtype==torch.float32 and gnorm.numel()==1,f"opttail hostgate: reduce returned {gnorm.dtype}{e(gnorm.shape)}, not the fp32 gnorm (a reduce_fused that ignores host_gate returns the bool finite)";g=G(gnorm.item());ok=math.isfinite(g);self.gnorm_host=g if ok else S;self.bad_host+=0 if ok else 1;_OPTTAIL_FIRED[EW]+=1;return ok
	def gnorm_value(self):return self.gnorm_host if self.host_gate else G(self.last_gnorm.item())
	def bad_steps_value(self):return self.bad_host if self.host_gate else B(self.bad_steps.item())
def _opttail_desc(optimizer,reducer):st=n(optimizer,'_stage',C);return f"items={AG.join(AO(_OPT_TAIL_TAGS))or"-"} afold={MUON_NS_AFOLD} alpha={MUON_NS_ALPHA} hostgate_live={z(n(reducer,KJ,O))} stage={st.desc()if st is not C else C2}"
TAIL_BLEND_G=-.075
TAIL_BLEND_REF=.8
TAIL_BLEND_SAVE='final'
def tail_blend_apply(model,ref,g):
	sd=model.state_dict();live={k:v.detach().to(C4,copy=I)for(k,v)in sd.items()};n=0
	with torch.no_grad():
		for(k,v)in sd.items():
			t,r=live[k],ref[k]
			if t.is_floating_point()and not torch.equal(t,r):tf=t.float();v.copy_((tf+g*(tf-r.float())).to(t.dtype));n+=1
	return n
def save_checkpoint(path,model,meta):path.parent.mkdir(parents=I,exist_ok=I);payload={Kb:{k:v.detach().cpu()for(k,v)in model.state_dict().items()},Kc:asdict(model.config),'meta':meta};torch.save(payload,path)
def torch_load_weights(path,map_location):return torch.load(path,map_location=map_location,weights_only=I)
def load_for_eval(checkpoint_path,device):
	global COMPUTE_DTYPE;COMPUTE_DTYPE=compute_dtype_for(device);payload=torch_load_weights(checkpoint_path,map_location=C4);config=GPTConfig(**payload[Kc]);model=GPT(config);model.init_weights();model.load_state_dict(payload[Kb],strict=I);model.to(device)
	if COMPUTE_DTYPE!=torch.float32:model.to(COMPUTE_DTYPE)
	model.eval();return model
def lr_multiplier(step,progress,warmdown_ratio,final_lr_frac):
	if step<WARMUP_STEPS:return(step+1)/i(1,WARMUP_STEPS)
	warmdown_start=Q-warmdown_ratio
	if warmdown_ratio<=S or progress<=warmdown_start:return Q
	frac=i(S,(Q-progress)/warmdown_ratio);return final_lr_frac+(Q-final_lr_frac)*frac
def grad_accum_for_step(progress,grad_accum_steps):
	if BATCH_RAMP and progress<BATCH_RAMP_FRAC:return grad_accum_steps//2
	return grad_accum_steps
def muon_momentum(step,sched_progress=S,warmdown_ratio=S,steps_left=C):
	warmdown_start=Q-warmdown_ratio
	if warmdown_ratio>S and sched_progress>warmdown_start:return .97-.07*g(Q,(sched_progress-warmdown_start)/warmdown_ratio)
	frac=g(step/300,Q);return(1-frac)*.85+frac*.97;frac=g(step/300,Q);momentum=(1-frac)*.85+frac*.95;return momentum
def ramp_lr_factor(accum,grad_accum_steps):return Q
def main():
	J='raw';H='_rope_base_used';F='store_true';E='unavailable';A='tail_blend';parser=argparse.ArgumentParser(description='Neuron Competition R1 -- team working copy');env=os.environ.get;parser.add_argument('--device-type',type=v,default=Y);parser.add_argument('--num-steps',type=B,default=B(env('NEURON_COMPETITION_R1_NUM_STEPS',NUM_STEPS)));parser.add_argument('--max-train-seconds',type=B,default=B(env('NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS',MAX_TRAIN_SECONDS)));parser.add_argument('--out-dir',type=v,default=env('NEURON_COMPETITION_R1_OUT_DIR',Cp));parser.add_argument('--eval-public',action=argparse.BooleanOptionalAction,default=env('NEURON_COMPETITION_R1_EVAL_PUBLIC',q)!=A1);parser.add_argument('--eval-tokens',type=B,default=B(env('NEURON_COMPETITION_R1_EVAL_TOKENS',v(4*524288))),help='tokens for the in-run public eval; <= 0 = the whole public val shard');parser.add_argument('--compile',action=F,default=env('NEURON_COMPETITION_R1_COMPILE',A1)!=A1,help='enable torch.compile on the full model (compiles during the excluded startup steps)');parser.add_argument('--nki-relu2',action=F,default=env('NEURON_COMPETITION_R1_NKI_RELU2',A1)!=A1,help='use the example fused NKI relu(x)**2 kernel in the MLP (Neuron only; illustration)');parser.add_argument('--depth',type=B,default=B(DEPTH));parser.add_argument('--n-embd',type=B,default=1024,help='0 = derive from depth (baseline rule)');parser.add_argument('--device-batch-size',type=B,default=B(DEVICE_BATCH_SIZE),help='0 = auto');parser.add_argument('--total-batch-size',type=B,default=B(TOTAL_BATCH_SIZE));parser.add_argument('--pack-windows',type=B,default=B(PACK_WINDOWS),help='1 = shipped packer');parser.add_argument('--warmdown-ratio',type=G,default=G(WARMDOWN_RATIO));parser.add_argument('--final-lr-frac',type=G,default=G(FINAL_LR_FRAC));parser.add_argument('--grad-clip',type=G,default=G(GRAD_CLIP));parser.add_argument('--seed',type=B,default=91);args=parser.parse_args();global USE_NKI_RELU2;USE_NKI_RELU2=args.nki_relu2;device_type=detect_device_type(args.device_type)
	if I:os.environ[EQ]=q
	ddp,rank,_local_rank,world_size,device=init_runtime(device_type);torch.manual_seed(args.seed);global COMPUTE_DTYPE;COMPUTE_DTYPE=compute_dtype_for(device);print0(f"device={device} dtype={COMPUTE_DTYPE} world_size={world_size} lnc={os.environ.get(AB,Y)}");print0(f"NEURON_CC_FLAGS={os.environ.get(KB,Y)!r} stochastic_rounding={os.environ.get(EQ,Y)}");print0(f"budget: official={TRAIN_TIME_BUDGET_SECONDS}s max_train_seconds={args.max_train_seconds} excluded_startup_steps={TRAIN_STARTUP_STEPS_EXCLUDED} (cap {TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS}s) checkpoint_reserve={CHECKPOINT_RESERVE_SECONDS}s step_cap={args.num_steps} sched_endpoint={SCHED_ENDPOINT}");tokenizer=ensure_tokenizer(build_if_missing=O);config=build_config(args.depth,args.n_embd);print0(f"model_config={json.dumps(asdict(config),indent=2)}  (training window {SEQ_LEN}, RoPE table {ROPE_TABLE_LEN})");model=GPT(config);model.init_weights();model.to(device);num_params=BP(p.numel()for p in model.parameters());flops_per_token=model.estimate_flops(SEQ_LEN);print0(f"{num_params=:,}{ flops_per_token=:e}")
	if BF16_MASTER and COMPUTE_DTYPE==torch.bfloat16:
		n_cast=0;head=model.lm_head if EP(model,KN)else C
		for m in model.modules():
			if AP(m,Linear)and m is not head:m.weight.data=m.weight.data.to(torch.bfloat16);n_cast+=1
		print0(f"bf16 master weights enabled for {n_cast} Linear layers (NEURON_RT_STOCHASTIC_ROUNDING_EN="+os.environ.get(EQ,Y)+')')
	orig_model=model
	if args.compile:compile_backend=a if device_type==a else'inductor';model=torch.compile(model,backend=compile_backend,dynamic=O);print0(f"torch.compile enabled (backend={compile_backend})")
	total_batch=args.total_batch_size;device_batch=args.device_batch_size
	if device_batch<=0:device_batch=i(1,g(16,total_batch//(SEQ_LEN*world_size)))
	world_tokens_per_micro=device_batch*SEQ_LEN*world_size
	if total_batch%world_tokens_per_micro!=0:raise AF(f"TOTAL_BATCH_SIZE={total_batch} must be a multiple of micro tokens={world_tokens_per_micro} (device_batch={device_batch} x seq={SEQ_LEN} x world={world_size})")
	grad_accum_steps=total_batch//world_tokens_per_micro
	if BATCH_RAMP and grad_accum_steps%2!=0:raise AF(f"R1_BATCH_RAMP=1 needs an even grad_accum_steps, got {grad_accum_steps} (device_batch={device_batch}, total_batch={total_batch}, world={world_size})")
	pack_windows=i(1,args.pack_windows)
	if device_batch%pack_windows!=0:raise AF(f"device_batch={device_batch} must be a multiple of pack_windows={pack_windows}")
	loader_rows=device_batch//pack_windows;print0(f"device_batch={device_batch} seq={SEQ_LEN} world={world_size} micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps} total_batch={total_batch:,} pack_windows={pack_windows} loader_rows={loader_rows}");print0(f"arch: attn_blocks={ATTN_BLOCKS!r} -> {V(orig_model.attn_layers)}/{config.n_layer} blocks {A0(orig_model.attn_layers)} mixer={MIXER!r} -> {V(orig_model.mixer_layers)} blocks {A0(orig_model.mixer_layers)} value_emb={VALUE_EMB} unet_skip={UNET_SKIP} attn_scale={ATTN_SCALE} mlp_ratio={MLP_RATIO} head_dim={HEAD_DIM} bigram_buckets={BIGRAM_BUCKETS} bigram_mode={BIGRAM_MODE} bigram_init={BIGRAM_INIT} init_resid={INIT_RESID} init_x0={INIT_X0} init_skip={INIT_SKIP} batch_ramp={BATCH_RAMP}"+f" (grad_accum {grad_accum_steps//2} -> {grad_accum_steps} at progress {BATCH_RAMP_FRAC})"+(f" attention: {narrow_desc(orig_model)}{ num_params=:,}"if orig_model.narrow else Y));print0(f"arch2: rope_base={ROPE_BASE} (tables built from {n(orig_model,H,C)}) tie_embeddings={TIE_EMBEDDINGS}"+Y+f" fired={_QL_FIRED}");print0(f"optim: muon_shard={MUON_SHARD} (rs={MUON_SHARD_RS} ag={MUON_SHARD_AG} payload={MUON_SHARD_GATHER}) adamw_stack={ADAMW_STACK} momentum_sched={MUON_MOMENTUM_SCHED} ramp_lr_coupling={RAMP_LR_COUPLING} adam_betas_bundle={ADAM_BETAS_BUNDLE} qk_scale={QK_SCALE} (cfg {config.qk_scale} attn {Cj((b.attn.qk_scale for b in orig_model.transformer.h if b.attn is not C),C)}) wd_emb={ADAMW_WD_EMB} wd_head={ADAMW_WD_HEAD}");print0(f"optim2: gradnorm_stack={GRADNORM_STACK} ns_tensor_coefs={NS_TENSOR_COEFS} persistent_views={MUON_PERSISTENT_VIEWS} zero_grad_none={ZERO_GRAD_NONE} elementwise_fuse={MUON_ELEMENTWISE_FUSE}"+(f" fuse_deny={AO(FUSE_DENY)}"if FUSE_DENY else Y));print0(_headdw_line(orig_model));print0(_mixfuse_line(orig_model))
	class _RowChunkedTokenizer:
		def __init__(self,inner,chunk_len):self._inner=inner;self._chunk_len=chunk_len
		def get_bos_token_id(self):return self._inner.get_bos_token_id()
		def encode(self,text,prepend=C,num_threads=8):
			rows=self._inner.encode(text,prepend=prepend,num_threads=num_threads)
			if AP(text,v):return rows
			n=self._chunk_len;return[ids[i:i+n]for ids in rows for i in L(0,V(ids),n)]
	loader_tokenizer=tokenizer;train_loader=make_dataloader(loader_tokenizer,loader_rows,SEQ_LEN*pack_windows,Ee,device);replay=C
	if replay is not C:print0(f"replay: {replay.describe()}")
	def next_batch():
		if replay is not C:return replay.next()
		x,y,state=Cj(train_loader)
		if pack_windows>1:x=x.view(-1,SEQ_LEN);y=y.view(-1,SEQ_LEN)
		return x,y,state
	x,y,loader_state=next_batch();optimizer=orig_model.setup_optimizer();reducer=GradReducer(orig_model.parameters(),device,optimizer);print0(f"muon shard: {optimizer.configure_sharding(device)}");print0(f"opttail: {_opttail_desc(optimizer,reducer)}");_fm=[(e(pg[l][0].shape),V(pg[l]))for pg in optimizer.param_groups if pg[AC]==AY and pg[l]];print0(f"fp32master: ARMED: the Muon update lands on an fp32 master of each rank's owned rows and the bf16 rows are its one cast, all-gathered as today; groups (shape, n, owned k) {[(s_,n_,-(-n_//world_size))for(s_,n_)in _fm]}, {BP(-(-n_//world_size)*math.prod(s_)*4 for(s_,n_)in _fm):,} B of masters per rank");ema=C;tb_ref=C;print0(f"tailblend: ARMED ref={TAIL_BLEND_REF} g={TAIL_BLEND_G} save={TAIL_BLEND_SAVE}: final.pt = theta_T + g (theta_T - theta_ref)");_MTP['w']=torch.zeros(2,device=device);print0('mtp: on w(t+2,t+3): clock [0,1/4) (.5,.25->0) [1/4,1/2) (.5->0,0) then next-token only');print0(f"mtp: ship ON: next-token + aux targets in ONE multi-target kernel pair; available={multi_available(device)} {multi_status()}");_BOSX[Cz]=tokenizer.get_bos_token_id()
	if not multi_available(device):raise Al(f"bosx: no multi-target kernel pair here ({multi_status()})")
	print0(f"bosx: ARMED post: from clock 0.5 (the MTP end) the next-token loss is fused_loss_multi K=1 with row weight 1/N and 0 at BOS targets (id {_BOSX[Cz]}); the MTP window runs the base program");loop_started=time.monotonic();budget_started=loop_started;smooth_loss=S;ema_updates=0;total_train_tokens=0;step=0;clock_start_step=0;startup_allowance_est=S;recent_step_times=[];last_lrm=Q;last_loss=C;consecutive_nan_logs=0;t_log=time.monotonic();gc.collect();gc.freeze();gc.disable()
	while step<args.num_steps:
		t0=time.monotonic();progress=S if step<TRAIN_STARTUP_STEPS_EXCLUDED else g((t0-budget_started)/args.max_train_seconds,Q);sched_progress=progress;accum=grad_accum_for_step(progress,grad_accum_steps);mtp_set(progress);_BOSX[AS]=_MTP[CB]==2
		if _BOSX[AS]and _BOSX[CC]is C:_BOSX[CC]=step,progress
		step_tokens=accum*world_tokens_per_micro;steps_left=C;optimizer.zero_grad(set_to_none=O)
		for _ in L(accum):
			if replay is not C:tok_loss=model(x,y,loss_reduction=AW);win_loss=tok_loss.sum(dim=1);replay.record(win_loss);loss=win_loss.sum()/tok_loss.numel()
			else:loss=model(x,y)
			last_loss=loss.detach();(loss/accum).backward();orig_model.flush_gain_grads();x,y,loader_state=next_batch()
		finite=reducer.reduce(args.grad_clip);last_lrm=lr_multiplier(step,sched_progress,args.warmdown_ratio,args.final_lr_frac)*ramp_lr_factor(accum,grad_accum_steps);_stage=optimizer._stage
		for group in optimizer.param_groups:
			group[B0]=group[KO]*last_lrm;group[Ba].fill_(group[B0])
			if _stage is C:group[CD].copy_(group[Ba])
			if group[AC]==AY:
				group[D1]=muon_momentum(step,sched_progress,args.warmdown_ratio,steps_left);group[CF].fill_(group[D1])
				if _stage is C:group[CE].copy_(group[CF])
				group[B1]=WEIGHT_DECAY
		if _stage is not C:optimizer.stage_step_scalars()
		if reducer.host_gate:applied=reducer.gate_host(finite)
		else:applied=z(finite.item())
		if applied:optimizer.step()
		else:optimizer.zero_grad(set_to_none=O)
		if USE_NKI_GEMM_FP8:nki_gemm_fp8.begin_step()
		if USE_NKI_GEMM_FP8_ANY:nki_gemm_fp8_nn.begin_step()
		synchronize(device)
		if replay is not C:replay.end_step()
		record_step_boundary(step);dt=time.monotonic()-t0
		if step>=TRAIN_STARTUP_STEPS_EXCLUDED:recent_step_times.append(dt)
		if step+1==TRAIN_STARTUP_STEPS_EXCLUDED:now=time.monotonic();observed_startup=now-_PROCESS_STARTED;startup_allowance_est=g(observed_startup,TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS);excess=i(S,observed_startup+STARTUP_LAUNCH_MARGIN_SECONDS-TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS);budget_started=now-excess;clock_start_step=step+1;print0(f"startup allowance boundary: {step+1} steps in {observed_startup:.1f}s since process start (excess charged {excess:.1f}s). Training clock starts now.")
		elapsed_after_step=time.monotonic()-budget_started;reached_min_steps=step+1>=TRAIN_STARTUP_STEPS_EXCLUDED;predicted_end=elapsed_after_step+(i(recent_step_times[-20:])if recent_step_times else dt)+CHECKPOINT_RESERVE_SECONDS;stop_after_step=step+1>=args.num_steps or reached_min_steps and(elapsed_after_step>=args.max_train_seconds or predicted_end>=args.max_train_seconds);control_vals=[1 if stop_after_step else 0]
		if ema is not C:control_vals.append(1 if applied and progress>=EMA_START_FRAC else 0)
		control_vals.append(1 if tb_ref is C and progress>=TAIL_BLEND_REF else 0);control=torch.tensor(control_vals,device=device,dtype=torch.int32)
		if dist.is_available()and dist.is_initialized():dist.all_reduce(control,op=dist.ReduceOp.MAX)
		if ema is not C and B(control[1].item()):ema.update()
		if tb_ref is C and B(control[1].item()):_ts=time.monotonic();tb_ref=step+1,progress,{k:v.detach().to(C4,copy=I)for(k,v)in orig_model.state_dict().items()};print0(f"tailblend: reference COPIED at step {step+1}{ progress=:.4f} ({time.monotonic()-_ts:.2f}s)")
		if step>=TRAIN_STARTUP_STEPS_EXCLUDED:total_train_tokens+=step_tokens
		step+=1;log_step=step%LOG_EVERY==0 or step<=TRAIN_STARTUP_STEPS_EXCLUDED+3
		if log_step:
			train_loss=G(last_loss.item())
			if math.isnan(train_loss):
				consecutive_nan_logs+=1
				if consecutive_nan_logs>=3:raise Al('loss has been NaN for 3 consecutive log intervals')
			else:consecutive_nan_logs=0;smooth_loss=.9*smooth_loss+.1*train_loss;ema_updates+=1
			debiased=smooth_loss/(1-.9**i(ema_updates,1));now=time.monotonic();interval_steps=LOG_EVERY if step%LOG_EVERY==0 else 1;tok_per_sec=interval_steps*step_tokens/i(now-t_log,1e-06);t_log=now
			if rank==0:D(f"step {step:05d} | loss {debiased:.4f} | lrm {last_lrm:.3f} | dt {dt:.2f}s | tok/s {tok_per_sec:,.0f} | gnorm {reducer.gnorm_value():.3f} | bad_steps {reducer.bad_steps_value()} | elapsed {now-budget_started:.0f}s | remaining {i(S,args.max_train_seconds-(now-budget_started)):.0f}s | epoch {loader_state["epoch"]}",flush=I)
		if B(control[0].item())!=0:break
	budget_elapsed_before_save=time.monotonic()-budget_started;_MTP[AS]=(O,)*2;_BOSX[AS]=O;checkpoint_path=Path(args.out_dir)/'final.pt';raw_checkpoint_path=Path(args.out_dir)/'final_raw.pt';meta={GS:step,'clocked_steps':step-clock_start_step,'startup_allowance_est':startup_allowance_est,'total_train_tokens':total_train_tokens,'budget_elapsed_before_save':budget_elapsed_before_save,'world_size':world_size,'device_batch':device_batch,'total_batch':total_batch,'pack_windows':pack_windows,'long_doc_chunks':LONG_DOC_CHUNKS,'replay':REPLAY,'batch_ramp':BATCH_RAMP,'compiled':args.compile,'seed':args.seed};ema_applied=O
	if ema is not C:
		meta[ET]={'decay':EMA_DECAY,'start_frac':EMA_START_FRAC,'warmup':ema.warmup,'updates':ema.updates,'save':EMA_SAVE,'path':ema.path_exact+(f"/{ema.path_cast}"if ema.cast_p else Y),B4:J}
		if ema.updates==0:print0(f"WARNING: R1_EMA_DECAY={EMA_DECAY} but the clock never reached R1_EMA_START_FRAC={EMA_START_FRAC} (no EMA update ran): the checkpoint holds the raw weights")
		else:ema.copy_to_model();meta[ET][B4]=ET;ema_applied=I
	tb=C;meta[A]={'ref':TAIL_BLEND_REF,'g':TAIL_BLEND_G,B4:J,'ref_step':tb_ref and tb_ref[0]}
	if tb_ref is C:print0(f"WARNING: tailblend: the clock never reached R1_TAIL_BLEND_REF={TAIL_BLEND_REF}: final.pt holds the raw weights")
	else:_t0=time.monotonic();tb=tail_blend_apply(orig_model,tb_ref[2],TAIL_BLEND_G);meta[A].update(weights='blend',blended=tb,seconds=GJ(time.monotonic()-_t0,2));print0(f"tailblend: FIRED ref={TAIL_BLEND_REF} (step {tb_ref[0]}) g={TAIL_BLEND_G} blended {tb} of {V(tb_ref[2])} tensors ({meta[A]["seconds"]}s): final.pt = theta_T + g (theta_T - theta_ref)")
	if rank==0:save_checkpoint(checkpoint_path,orig_model,meta)
	if dist.is_available()and dist.is_initialized():dist.barrier()
	budget_elapsed_after_save=time.monotonic()-budget_started;public_eval=C;causal=C
	if args.eval_public:
		eval_batch=B(g(device_batch,16))
		if eval_batch!=device_batch:print0(f"eval batch {eval_batch} (training device batch {device_batch})")
		causal=causality_check(orig_model,tokenizer,eval_batch,SEQ_LEN,device);public_eval=evaluate_bpb(orig_model,tokenizer,eval_batch,SEQ_LEN,device,split='public_val',max_tokens=args.eval_tokens if args.eval_tokens>0 else C,timeout_seconds=EVAL_TIMEOUT_SECONDS)
	if rank==0:
		D('---');D(f"checkpoint_path:    {checkpoint_path}");D(f"training_seconds:  {budget_elapsed_before_save:.1f}  (clock restarted after step {TRAIN_STARTUP_STEPS_EXCLUDED})");D(f"budget_seconds:    {TRAIN_TIME_BUDGET_SECONDS}");D(f"max_train_seconds: {args.max_train_seconds}");D(f"startup_allowance_est: {startup_allowance_est:.1f}");D(f"charged_seconds_est:   {budget_elapsed_after_save:.1f}");D(f"over_budget:       {budget_elapsed_after_save>TRAIN_TIME_BUDGET_SECONDS}");D(f"total_seconds:     {time.monotonic()-loop_started:.1f}");D(f"final_lrm:         {last_lrm:.4f}  (target floor {args.final_lr_frac})");D(f"sched_endpoint:    {SCHED_ENDPOINT}");D(f"bad_steps:         {reducer.bad_steps_value()}");D(f"opttail:           {_opttail_desc(optimizer,reducer)} fired={_OPTTAIL_FIRED}");D(f"fp32master:        finishes={_FP32M[AV]} master_bytes={_FP32M[GK]:,} (rank 0)");D('mtp:               fired '+' | '.join(f"phase{i+1} steps={a} t+2={b} t+3={c}"for(i,(a,b,c))in A7(_MTP[AV])));D(f"mtp ship:          calls phase1={_MTP[Em][0]} phase2={_MTP[Em][1]} phase3={_MTP[Em][2]} {multi_status()}");D(f"bosx:              k1_calls={_BOSX[Cw]} declined={_BOSX[C_]} hand_over={_BOSX[CC]} zero_rows_first={_BOSX[B6]} selfcheck_k1={_BOSX["chk"]!r}");D(f"compiled:          {args.compile}");D(f"nki_relu2:         {args.nki_relu2}");D(f"nki_rope_norm:     {USE_NKI_ROPE_NORM}");D(f"nki_relu2_opt:     {USE_NKI_RELU2_OPT}");D(f"nki_gemm:          {USE_NKI_GEMM}  fused fc+relu2: {USE_NKI_FC_RELU2}");D(f"nki_gain_norm:     {USE_NKI_GAIN_NORM}");D(f"nki_block_head:    {USE_NKI_BLOCK_HEAD}  (norm2 kernel: {USE_NKI_NORM2})");D(f"nki_megablock:     {USE_NKI_MEGABLOCK}  tail(phase2): {USE_NKI_MEGABLOCK_TAIL}");D(f"nki_emb_bwd:       {USE_NKI_EMB_BWD}");D(f"nki_loss_head:     {USE_NKI_LOSS_HEAD}");D(f"nkilib_attn:       {USE_NKILIB_ATTN}");D(f"nki_attn:          {USE_NKI_ATTN}  fused rope+attn: {USE_NKI_ATTN_ROPE}  qkv stack view: {QKV_STACK_VIEW}");D(f"glue:              grad_accum_inplace={GRAD_ACCUM_INPLACE!r} gain_batch={GAIN_BATCH} cast_cache={CAST_CACHE} emb_shadow={EMB_SHADOW}");D(f"nki_optim:         {nki_optim.config_string()if nki_optim is not C else E}");D(f"nki_gemm_fp8:      {nki_gemm_fp8.config_string()if nki_gemm_fp8 is not C else E}");D(f"nki_gemm_fp8_nn:   {nki_gemm_fp8_nn.config_string()if nki_gemm_fp8_nn is not C else E}");D(f"num_steps:         {step}  (charged {step-clock_start_step})");D(f"total_tokens_M:    {total_train_tokens/1e6:.1f}  (charged steps only)");D(f"world_size:        {world_size}");D(f"lnc:               {os.environ.get(AB,Y)}");D(f"device_batch:      {device_batch}  grad_accum {grad_accum_steps}  total_batch {total_batch}");D(f"pack_windows:      {pack_windows}");D(f"replay:            {REPLAY}"+(f"  alpha {REPLAY_ALPHA} floor {REPLAY_FLOOR} passes {replay.passes} windows {replay.n:,}"if replay is not C else Y));D(f"arch:              attn_blocks={ATTN_BLOCKS!r} -> {V(orig_model.attn_layers)}/{config.n_layer} blocks {A0(orig_model.attn_layers)} value_emb={VALUE_EMB} unet_skip={UNET_SKIP} attn_scale={ATTN_SCALE} mlp_ratio={MLP_RATIO} head_dim={HEAD_DIM} bigram_buckets={BIGRAM_BUCKETS} bigram_mode={BIGRAM_MODE} bigram_init={BIGRAM_INIT} init_resid={INIT_RESID} init_x0={INIT_X0} init_skip={INIT_SKIP} batch_ramp={BATCH_RAMP}");D(f"arch3:             mixer={MIXER!r} -> {V(orig_model.mixer_layers)} blocks {A0(orig_model.mixer_layers)}"+(f"  mix_lambda={[GJ(G(b.mix_lambda),6)for b in orig_model.transformer.h if b.mix_lambda is not C]}"if orig_model.mixer_layers else Y))
		if orig_model.narrow:D(f"narrow:            {narrow_desc(orig_model)} fused={_NARROW[ER]} qkv_views={BP(1 for b in orig_model.transformer.h if b.attn is not C and n(b.attn,Ge,C)is not C)}/{V(orig_model.attn_layers)}")
		D(f"arch2:             rope_base={ROPE_BASE} (tables {n(orig_model,H,C)}) tie_embeddings={TIE_EMBEDDINGS}"+Y+f" fired={_QL_FIRED}");D(f"optim:             muon_shard={MUON_SHARD} adamw_stack={ADAMW_STACK} momentum_sched={MUON_MOMENTUM_SCHED} ramp_lr_coupling={RAMP_LR_COUPLING} adam_betas_bundle={ADAM_BETAS_BUNDLE} qk_scale={QK_SCALE} (cfg {config.qk_scale} attn {Cj((b.attn.qk_scale for b in orig_model.transformer.h if b.attn is not C),C)}) wd_emb={ADAMW_WD_EMB} wd_head={ADAMW_WD_HEAD}");D(f"optim2:            gradnorm_stack={GRADNORM_STACK} ns_tensor_coefs={NS_TENSOR_COEFS} persistent_views={MUON_PERSISTENT_VIEWS} zero_grad_none={ZERO_GRAD_NONE} elementwise_fuse={MUON_ELEMENTWISE_FUSE}");D(_headdw_line(orig_model,pad=I));D(_mixfuse_line(orig_model,pad=I))
		if ema is not C:D(f"ema:               decay={EMA_DECAY} start_frac={EMA_START_FRAC} save={EMA_SAVE} updates={ema.updates} applied={ema_applied} path={ema.path_exact}"+(f"/{ema.path_cast}"if ema.cast_p else Y))
		D(f"tailblend:         ref={TAIL_BLEND_REF} g={TAIL_BLEND_G} save={TAIL_BLEND_SAVE} applied={tb is not C} blended={tb} ref_step={tb_ref and tb_ref[0]}");D(f"seq_len:           {SEQ_LEN}");D(f"num_params_M:      {num_params/1e6:.1f}")
		if causal is not C:D(f"causality_passed:  {causal["passed"]}");D(f"causality_max_abs: {causal["max_abs_diff"]:.6f}")
		if public_eval is not C:D(f"public_val_bpb:    {public_eval["val_bpb"]:.6f}");D(f"eval_seconds:      {public_eval["eval_seconds"]:.1f}");D(f"eval_tokens:       {public_eval["num_tokens"]}  (requested {args.eval_tokens if args.eval_tokens>0 else"whole shard"})");D(f"eval_timeout:      {public_eval["eval_timeout"]}");D(f"eval_over_timeout: {public_eval["eval_over_timeout"]}")
	cleanup_runtime()
if __name__=='__main__':main()
