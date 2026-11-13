"""
Round 2 training script: the kit's baseline with the changes below. Both host tables start at zero and are trained
only by the model's loss gradient on the tokens it processes, with no loss of their own (the organizers' rulings of
Oct 17, 20, 21 (UPDATED_RULES.md), 22 and 24); the re-laid text uses only the provided documents' own characters
with fixed generic markers (rulings of Oct 21, 23, 24, 25 and 27).

- Shape: DEPTH = 18, ASPECT_RATIO = 85 (the kit's 22, 64): width 1,536, 12 heads of 128; DEVICE_BATCH_SIZE = 2 (the
  kit's 1), four micro-steps a 1,048,576-token step on 64 ranks; 3,684,532,273 parameters plus the two host tables.
- Data: one packed row of the kit's make_dataloader (shards 0 to 6499) a rank-step, its 16,384 tokens cut into 8
  windows that can start mid-document (the kit packs 8 rows from document starts). A kit document whose last lines
  are whole "Question: <q> Answer: <a>" pairs has the whitespace before each "Answer:" made one line break, two or
  left as is (a third of such documents each, by a hash of its text mod 3). Only its tokens change: the selection
  and the re-laid text read it as written, and every other byte stays.
- Selection: from 0.1 of the clock a fifth of document tokens come (each at most twice) from 384 shards (ids 512 to
  6,499) that 16 host processes scan from step 5: the 8 percent of each most narrative and 8 percent most
  entity-dense by plain counts over their first and middle 1,024 bytes. It is off if a scan shard is missing or its
  folder exists at launch.
- Attention: our NKI kernel pair (rotary embedding and QK norm inside; the forward loads a head's values and stores
  its output in one transfer each) at NEURON_LOGICAL_NC_CONFIG=2, in training and in load_for_eval, else the kit's;
  QK_SCALE = 0.11 (the kit's 1/sqrt(128)). Layers but 3, 7, 11, 15, 17: a query sees only its 128-token tile and the
  3 prior (same pair; SDPA mask to score).
- Value embeddings: VALUE_EMBEDS = 2 tables on layers 15 and 17 add ve_lambda x table[token] to the values. U-net
  skips: each of the last 9 blocks adds skip_lambda[j] x the output of one of the first 9.
- N-gram table (nanochat PR #854): NGRAM_ROWS = 2^30 host rows x 128 bf16 (/dev/shm, or memfd where it lacks room:
  see Save), zero at the start; each position's 2- and 3-gram hash to a row each, and the rows x a learned gain,
  each projected by a Muon matrix, are added to the token embedding. The rows train by row-wise RMSprop (decay 1e-3
  a use, stochastic rounding to bf16), each rank updating the rows it owns with the batch's mean gradient. Every
  read of a row (training's and both scorers') takes its 4-bit view: s = the row's largest magnitude / 7 rounded up
  to bf16, q = round(row / s) in [-7, 7], q x s rounded once to bf16; the RMSprop updates the bf16 rows by the
  view's gradient (straight through).
- Output-site table: OS1_ROWS = 2^32 rows x 128, an int8 master with a bf16 scale a row, zero at the start; every
  read (training's and the scorers') takes a row's 1-bit view sign(m) x mean|m|. The 16 token ids ending at a
  position (in the window, none after a BOS) hash to a row and 128 signs; u = signs x row travels with the n-gram
  rows but enters only the head, adding to token v's logit glin s_v + G beta relu(sg(s_v) - thr) / (1 - thr), s_v =
  u . C_v / a0 (sg a stop-gradient, C a fixed random codebook, G a sigmoid of the previous 32 positions' mean read
  of their next input token). Update: row -= lr g / (rms(g) + eps) (OS1_LR = 0.125, rms(row) at most 4 lr) on the
  dequantised master (the view's gradient straight through), then seeded stochastic rounding.
- Both tables update one step stale, on worker threads. final.pt holds the n-gram table's 4-bit view (ngram_table
  uint8 [2^30, 64], q two to a byte in 4-bit two's complement, and ngram_scale bf16 [2^30], s; model_config's
  ngram_bits = 4) and the output-site table's 1-bit view as sign bits (os1_table uint8 [2^32, 16]) and bf16 scales
  (os1_scale [2^32]; os1_bits = 1); load_for_eval keeps both in host memory as parameters of the model it returns
  and reads the same values for both scorers.
- Loss: BOS targets ignored; at LNC 2 our NKI pair computes the head [x | u] [W | glin C / a0]^T, the gated read,
  the softcap and the cross-entropy, 4,096 vocabulary entries a chunk (on the CPU in torch).
- MLP: every dense MLP, each MoE block's shared half and each routed expert run our fp8 NKI kernel pair (e4m3 and
  e5m2 operands, fp32 accumulation, bf16 weight gradients but the dense c_fc's and c_proj's; power-of-two scales
  recomputed after each optimizer step and in load_for_eval); off Neuron the kernels' torch emulation. In each dense
  MLP c_fc's and c_proj's weight gradients are fp8 kernels. Each shared half, and in training each rank's expert,
  runs the pair on its own e4m3 weight tiles, written by the pre-cast (not saved; at scoring, an expert's at each of
  its calls).
- Attention projections: QKV and c_proj run our fp8 NKI kernels in training and in load_for_eval (each weight
  gradient one fp8 kernel); the attention core and lm_head stay bf16.
- Mixture of experts in blocks 9, 11, 13, 15 and 17: a shared half plus 64 experts (MOE_HIDDEN = 3,072), one a token
  by an fp32 softmax router, scaled by its probability, at most 40 tokens an expert a sequence (causal); rank e
  holds expert e (all_to_all exchanges, a local Muon update), each on the fp8 pair; load-balance loss 0.01, z-loss
  0.001. final.pt holds the experts in bf16 (model_config moe_expert8 = 1); scoring holds every expert on one
  device.
- Optimizer: the kit's Muon and AdamW; WEIGHT_DECAY = 0.05 (the kit's 0.10); resid_lambda at SCALAR_LR / 28; Muon
  momentum (the kit's 0.95) 0.85 to 0.97 over 300 steps, then to 0.90 from 0.35 of the run; gradients summed in FP32
  (Muon's in bf16, reduce-scattered); FP32 masters; the embeddings' gradients by our NKI kernel at LNC 2; the update
  math compiled, each step's gradient reduce-scatters and all-reduces in one compiled graph with the experts'
  update, so the device can run them together. Schedules on the training clock: warmup 40 steps, hold to 0.15, then
  linear to 0.01 over DENSE_WARMDOWN_RATIO = 0.85 (the n-gram rows and the momentum's window keep the kit's 0.65 and
  0.05).
- Save: synchronous. Training stops when its clock reaches the budget less CHECKPOINT_RESERVE_SECONDS = 1,050 s (90
  + 600 + 360, from measured saves); every rank packs its share of both tables' views (giving the master's pages
  back as it packs) and rank 0 writes final.pt (about 156.9 GB). The tables' files use /dev/shm when it holds the
  save's peak + 20 GB (a 1 MiB hole each rank punches there at startup decides whether the view's 77.3 GB counts).
  Where /dev/shm lacks that room, the same files are anonymous shared memory (memfd) of the same machine, made by
  rank 0 and mapped by every rank through its /proc/<pid>/fd path, when the machine's available memory and any
  memory limit of the run's container hold that amount, and otherwise the run stops before training; either way
  nothing of them is written to disk, and where they live changes no value and nothing in final.pt.
- Re-laid text: the stream also carries short blocks of the loader's documents' own text, 11 examples a block, a BOS
  before each block; code selects and cuts the text and writes only fixed markers and separators, no marker chosen
  from the text; each block family below was built for one kind of CORE task, and no CORE item is used. Bracket
  blocks, in the CORE language-modelling layout: a document's brackets ()[]{}<> in their order (at most 96) up to a
  cut before a closing run, and as its answer the document's own next 1 to 3 closers, each after a marker: "Input:"
  "Output:" in half the blocks, else one of RELAY_PAIRS, 300 plain noun pairs written into the file ("Brackets:"
  "Closers:", "Series:" "Remainder:", ...), the second after a space or a line break (602 layouts), one a block,
  drawn by a seeded generator (two unused draws keep the trained run's sequence). In 0.75 of blocks examples open
  with the document's own text just before their first bracket, as it stands (up to 30 words). The leads with their
  spaces, the markers, the separators and, before a line-opening second marker, the last input bracket are out of
  the loss. Letter blocks: a document's 4-option quiz: its stem, each option a line under a marker style drawn a
  block ("A." "A)" "(A)" "A:"), "Answer:" and its own key letter, the one token in the loss; keys balanced by
  selection. Equation blocks: from a document's text of the form <expression> = <number> (or "equals"), the
  expression being 2 to 4 integer operands of 1 to 3 digits joined by + - * / x (spaced) or the multiplication and
  division signs, optionally parenthesized: the expression as written (an enclosing parenthesis dropped) after the
  marker "Input:", and the number the document writes after the "=" after "Output:" (after a space or opening a
  line, one a block); only that number, with its space, is in the loss; one equation a document. The code selects
  and cuts by the text alone and evaluates nothing, so about 13 to 17 percent of the examples do not state the
  expression's value: the documents' own errors and jokes, and number labels (score ranges, part numbers, ratios)
  that the text rule reads as equations. An audit of the first five training shards (843 equations against the items
  of eleven CORE tasks, counts only): NOT HELD; no equation stands inside an item and none pairs an expression found
  in an item with that item's answer; the selection never reads the CORE bundle. Yes/no blocks: a paragraph of 25 to
  250 words, a line break and the polar question opening the next line (sharing key words with it) after "Input:",
  and the answer sentence's opening Yes, No, yes or no as written after "Output:"; RELAY_YNCAP_YES = 0.58 of the
  examples from the yes pool. The yes/no pools also take the cuts of RELAY_YQ_WIDE = 10 more training shards a rank
  (kit list positions 5,824 + rank + 64 j) and the letter queues the quizzes of RELAY_MCQ_WIDE = 40 more (positions
  512 + rank + 64 j). RELAY_PS_WORKERS = 16 processes (nice 19, no kernels loaded) that global rank 0 starts at its
  first charged refill read them once; ranks take results from batch 120, the letters' 4 row groups at a time; a
  letter pool keeps 8,192. These layouts' text is the document's own, selected and cut by these rules (the CORE
  bundle is never read); each takes one cut a document, drawn on its own seeded generator (the yes/no wide pool's on
  another). RELAY_SHARE = 0.02 of the kit's tokens and the shares s: RELAY_MCQ = 0.006086 (0.003043 x 2),
  RELAY_ARITH = 0.0001 and RELAY_YNCAP = 0.005; RELAY_SEED = 62. Each layout's debt grows k s n / (1 - k t) a
  refill, capped at 65,536 tokens (brackets, letters) or 8,192 (the others); n is the refill's document tokens (the
  kit's and the selected), k 0.75 to 0.8 of the run and 2 after (the letters' k: 2 throughout). A layout's t is the
  shares of its group and the groups before it: (RELAY_SHARE), (RELAY_MCQ), (RELAY_YNCAP), (RELAY_ARITH).
- Execution (numbers unchanged): NEURON_RT_DISABLE_EXECUTION_BARRIER=1, NEURON_RT_EXEC_TIMEOUT=300 and
  NEURON_USE_SPLIT_REDUCE_SCATTER=0 set before torch is imported (an environment value wins); the row copied to the
  device once, the step boundary read one step late from step 5 (synchronous every 100th step and near the end), the
  accumulation's adds compiled.
- Rescue: from 0.85 of the budget a host copy of the weights and experts, saved with the tables as they stand if the
  kit's bad-loss check fires. Removed: the kit's --nki-relu2 example kernel and in-run evaluation; --compile-only is
  refused.
- Compilation and scoring: torch.compile by default, every compilation inside the five excluded startup steps;
  load_for_eval builds the trained network (the kernels at LNC 2, both tables, every expert, the 2,048-token window
  final.pt declares).
Launch: NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=64 train.py. Every weight starts from a
seeded random draw or a constant (both tables at zero); the other optimizer settings, the checkpoint layout and seed
42 are the kit's; prepare.py holds the fixed data, tokenizer and eval utilities.

To fit the 262,144-byte limit, comments, blank lines and the type annotations nothing reads are removed (the custom
ops' signatures, which torch reads, and GPTConfig's fields keep theirs), adjacent string literals are written as
one, and every function, class and try block is whitespace-compacted (one-space indents, tokens joined, a bracketed
statement on one line, a block's simple statements on one line separated by semicolons, a suite of simple statements
on its header's line). The training loop, main(), keeps one statement a line. The names inside the NKI kernels and
their torch glue (the ops' registration and launches, fp8 scales, pre-casts, plans, the CPU emulation) are
shortened, all but the kernels', the ops' and those the rest of the file calls. Every name the model, loss,
optimizer, data, re-laid text and training loop use, the strings and the statements are unchanged, so the file runs
as its readable source would.
"""
from __future__ import annotations
import argparse
import bisect
import collections
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from collections import OrderedDict
from dataclasses import asdict,dataclass
from pathlib import Path
_PROCESS_STARTED = time.monotonic()
os.environ.setdefault("NEURON_USE_SPLIT_REDUCE_SCATTER","0")
os.environ.setdefault("NEURON_RT_DISABLE_EXECUTION_BARRIER","1")
os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT","300")
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from prepare import(TOKENIZER_VOCAB_SIZE,TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS,TRAIN_STARTUP_STEPS_EXCLUDED,TRAIN_TIME_BUDGET_SECONDS,Tokenizer,ensure_tokenizer,get_dist_info,make_dataloader,print0,record_step_boundary,)
SEQ_LEN = 2048
DEPTH = 18
ASPECT_RATIO = 85
HEAD_DIM = 128
VALUE_EMBEDS = 2
QK_SCALE = 0.11
ATTN_WINDOW = 512
UNET_INIT = 1.0
DEVICE_BATCH_SIZE = 2
TOTAL_BATCH_SIZE = 1_048_576
NUM_STEPS = -1
TARGET_PARAM_DATA_RATIO = 10.30
MAX_TRAIN_SECONDS = TRAIN_TIME_BUDGET_SECONDS
STARTUP_LAUNCH_MARGIN_SECONDS = 30.0
LOGIT_SOFTCAP = 15.0
EMBEDDING_LR = 0.30
UNEMBEDDING_LR = 0.006
MATRIX_LR = 0.025
SCALAR_LR = 0.20
RESID_LR_DIV = 28.0
WEIGHT_DECAY = 0.05
NGRAM_ORDERS = (2, 3)
NGRAM_ROWS = 1 << 30
NGRAM_DIM = 128
NGRAM_GAIN = 1.0
NGRAM_LR = 0.2
NGRAM_BETA = 0.99
NGRAM_EPS = 1e-8
NGRAM_DECAY = 1e-3
NGRAM_PATH = "/dev/shm"
NGRAM_SAVE_SECONDS = 600.0
NGRAM_HASH_SEED = 20261013
NGRAM_HASH_BITS = 30
NGRAM_HASH_MASK = (1 << NGRAM_HASH_BITS) - 1
NGRAM_BITS = 4
OS1_ROWS = 1 << 32
OS1_ORDER = 16
OS1_LR = 0.125
OS1_EPS_REL = 1e-6
OS1_CAP = 4.0
OS1_A0 = 5.517
OS1_THR = 0.7
OS1_BETA = 6.0
OS1_GLIN = 0.01
OS1_GATE_W = 32
OS1_GATE_H0 = 0.2
OS1_GATE_SLOPE = 30.0
OS1_SAVE_SECONDS = 360.0
OS1_HASH_SEED = 20261059
OS1_CODE_SEED = 59020261
OS1_LINE_EVERY = 60
OS1_MASK62 = (1 << 62) - 1
OS1_TT = 1
WARMUP_STEPS = 40
WARMDOWN_RATIO = 0.65
FINAL_LR_FRAC = 0.05
DENSE_WARMDOWN_RATIO = 0.85
DENSE_FINAL_LR_FRAC = 0.01
DENSE_SCHED = (DENSE_WARMDOWN_RATIO, DENSE_FINAL_LR_FRAC)
ROWS_SCHED = (WARMDOWN_RATIO, FINAL_LR_FRAC)
CHECKPOINT_RESERVE_SECONDS = 90.0 + NGRAM_SAVE_SECONDS + OS1_SAVE_SECONDS
RESCUE_START_FRACTION = 0.85
RESCUE_EVERY_STEPS = 200
GRAD_CLIP = 0.0
MOE_HIDDEN = 3072
MOE_TOPK = 1
MOE_CF = 1.25
MOE_ROUTER_STD = 0.02
MOE_AUX = 0.01
MOE_ZLOSS = 0.001
BOUNDARY_SYNC_EVERY = 100
def neuron_available():
 try:
  torch.device("neuron");return True
 except Exception:return False
def detect_device_type(requested=""):
 if requested:return requested
 if torch.cuda.is_available():return "cuda"
 if neuron_available():return "neuron"
 return "cpu"
def init_runtime(device_type):
 assert device_type in{"cpu","cuda","neuron"};torch.manual_seed(42);ddp,rank,local_rank,world_size=get_dist_info()
 if ddp and device_type=="cuda":device=torch.device("cuda",local_rank);torch.cuda.set_device(device);dist.init_process_group("nccl",device_id=device)
 elif ddp and device_type=="neuron":dist.init_process_group("neuron");device=torch.device("neuron")
 elif ddp:dist.init_process_group("gloo");device=torch.device("cpu")
 else:device=torch.device(device_type)
 if dist.is_available()and dist.is_initialized():dist.barrier()
 return ddp,rank,local_rank,world_size,device
def cleanup_runtime():
 if dist.is_available()and dist.is_initialized():dist.destroy_process_group()
def synchronize(device):
 if device.type=="cuda":torch.cuda.synchronize()
 elif device.type=="neuron" and hasattr(torch,"neuron"):torch.neuron.synchronize()
def compute_dtype_for(device):
 if device.type=="cuda" and torch.cuda.is_bf16_supported():return torch.bfloat16
 if device.type=="neuron":return torch.bfloat16
 return torch.float32
@dataclass
class GPTConfig:sequence_len:int;vocab_size:int;n_layer:int;n_head:int;n_embd:int;unet:bool=False;value_embeds:int=0;qk_scale:float=0.0;ngram_orders:tuple=();ngram_rows:int=0;ngram_dim:int=0;ngram_proj:str="";ngram_seed:int=0;ngram_bits:int=0;attn_window:int=0;attn_long:tuple=();os1_rows:int=0;os1_dim:int=0;os1_order:int=0;os1_seed:int=0;os1_code_seed:int=0;os1_bos:int=-1;os1_a0:float=0.0;os1_thr:float=0.0;os1_beta:float=0.0;os1_glin:float=0.0;os1_gate_w:int=0;os1_gate_h0:float=0.0;os1_gate_slope:float=0.0;os1_bits:int=0;moe_layers:tuple=();moe_experts:int=0;moe_topk:int=0;moe_capacity:int=0;moe_hidden:int=0;moe_shared:int=0;moe_dispatch:str="";moe_shared_mlp:str="";moe_expert8:int=0
def value_embed_layers(n_layer,n_tables):
 if n_tables>(n_layer+1)//2:raise ValueError(f"value_embeds={n_tables}: at most {(n_layer + 1) // 2} alternating layers at depth {n_layer}")
 return[n_layer-1-2*j for j in range(n_tables)][::-1]
def ngram_hash_tables(seed,n_tables,size):
 m64=(1<<64)-1;x=seed&m64;out=[]
 for _ in range(n_tables*size):x=(x+0x9E3779B97F4A7C15)&m64;z=((x^(x>>30))*0xBF58476D1CE4E5B9)&m64;z=((z^(z>>27))*0x94D049BB133111EB)&m64;out.append((z^(z>>31))>>(64-NGRAM_HASH_BITS))
 return torch.tensor(out,dtype=torch.int32).view(n_tables,size)
def ngram_hash_ids(idx,hash_tab,orders,rows):
 z=idx.to(torch.int32);b,t=z.shape;pad=hash_tab.size(1)-1;shifted=[z]
 for k in range(1,max(orders)):fill=torch.full((b,k),pad,dtype=torch.int32,device=z.device);shifted.append(torch.cat([fill,z],dim=1)[:,:t])
 out=[];r=0
 for n in orders:
  h=hash_tab[r][shifted[0]]
  for k in range(1,n):h=torch.bitwise_and(h+hash_tab[r+k][shifted[k]],NGRAM_HASH_MASK)
  out.append(torch.bitwise_and(h,rows-1));r+=n
 return out
def ngram_bf16_round_up(x):bits=x.contiguous().view(torch.int32);trunc=torch.bitwise_and(bits,-65536);return torch.where(bits!=trunc,trunc+65536,trunc).view(torch.float32)
def ngram_round_bf16(x,noise):return((x.contiguous().view(torch.int32)+noise)&-65536).view(torch.float32).to(torch.bfloat16)
def os1_hash_tables(seed,order,size):
 import numpy as np;n=2*order*size
 with np.errstate(over="ignore"):x=np.uint64(seed%(1<<64))+np.arange(1,n+1,dtype=np.uint64)*np.uint64(0x9E3779B97F4A7C15);z=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9);z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB);z=z^(z>>np.uint64(31))
 return torch.from_numpy((z>>np.uint64(2)).astype(np.int64).reshape(2,order,size))
def os1_keys(tokens,tabs,order,bos,rows):
 import numpy as np;z=tokens.to(torch.int64).numpy();b,t=z.shape;kn=min(order,t);pad=tabs.size(2)-1;tu=tabs.numpy().view(np.uint64);zp=np.concatenate([np.full((b,kn-1),pad,dtype=np.int64),z],axis=1);src=np.lib.stride_tricks.sliding_window_view(zp,t,axis=1)[:,::-1];g=tu[np.arange(2).reshape(2,1,1,1),np.arange(kn).reshape(1,1,kn,1),src[None]];h=np.bitwise_and(g.sum(axis=2,dtype=np.uint64),np.uint64(OS1_MASK62));c=np.cumsum(z==bos,axis=1,dtype=np.int64);prev=np.zeros_like(c)
 if order-1<t:prev[:,order-1:]=c[:,:t-order+1]
 valid=(np.arange(t)>=order-1).reshape(1,t)&(c-prev==0);row=np.where(valid,np.bitwise_and(h[0],np.uint64(rows-1)).astype(np.int64),np.int64(-1));return torch.from_numpy(row),torch.from_numpy(np.ascontiguousarray(h[1]).view(np.int64))
def os1_signs(hsign,dim):
 import numpy as np;x=hsign.reshape(-1).numpy().astype(np.uint64);golds=np.array([((k+1)*0x9E3779B97F4A7C15)%(1<<64)for k in range((dim+63)//64)],dtype=np.uint64)
 with np.errstate(over="ignore"):z=x.reshape(-1,1)+golds.reshape(1,-1);z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9);z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB);z=z^(z>>np.uint64(31))
 w=np.ascontiguousarray(z.astype("<u8"));bits=np.unpackbits(w.view(np.uint8),axis=1,bitorder="little")[:,:dim];return torch.from_numpy((1-2*bits.astype(np.int8)).astype(np.int8))
def os1_codebook(vocab_pad,dim,seed):g=torch.Generator().manual_seed(seed);c=torch.randn(vocab_pad,dim,generator=g,dtype=torch.float32);return c/c.norm(dim=1,keepdim=True)
def os1_band(seq,w):j=torch.arange(seq).unsqueeze(1);t=torch.arange(seq).unsqueeze(0);m=((j<=t-1)&(j>=t-w)).to(torch.float32);return m/m.sum(0,keepdim=True).clamp_min(1.0)
def os1_typ_rms(glin,a0,dim,tokens,world):return glin/(a0*math.sqrt(dim)*max(1,tokens)*max(1,world))
def os1_row_step(x,g,lr,eps,cap):r=g.square().mean(1,keepdim=True).sqrt();x.sub_(g*(lr/(r+eps)));q=x.square().mean(1,keepdim=True).sqrt();f=torch.clamp((cap*lr)/q.clamp_min(1e-30),max=1.0);x.mul_(f);return int((f<1.0).sum())
def os1_owner_index(counts,r,max_rows):import numpy as np;c=np.array(counts,dtype=np.int64);cnt=c[:,r];starts=c[:,:r].sum(axis=1);base=np.arange(c.shape[0],dtype=np.int64)*max_rows+starts-(np.cumsum(cnt)-cnt);return torch.from_numpy(np.repeat(base,cnt)+np.arange(int(cnt.sum()),dtype=np.int64))
def ngram_view4(m):x=m.float();s=ngram_bf16_round_up(x.abs().amax(1)/7.0);safe=torch.where(s>0,s,torch.ones_like(s));q=torch.round(x/safe.unsqueeze(1)).clamp_(-7,7).to(torch.int8);return q,s.to(torch.bfloat16)
def ngram_view4_rows(q,s):return(q.to(torch.float32)*s.to(torch.float32).unsqueeze(1)).to(torch.bfloat16)
def ngram_pack4(q):u=q.to(torch.int16)&15;return(u[:,0::2]|(u[:,1::2]<<4)).to(torch.uint8)
def ngram_unpack4(pk,dim):b=pk.to(torch.int16);v=torch.stack([b&15,(b>>4)&15],dim=2).reshape(pk.size(0),dim);return torch.where(v>=8,v-16,v).to(torch.int8)
def host_meminfo(key):
 """A /proc/meminfo field in bytes (0 if unreadable)."""
 try:
  with open("/proc/meminfo")as fh:
   for ln in fh:
    if ln.startswith(key+":"):return int(ln.split()[1])*1024
 except(OSError,ValueError,IndexError):
  pass
 return 0
class MemPath(type(Path())):
 """A host table's file as rank 0's memfd: its path /proc/<pid>/fd/<n> opens it on every rank of the host; name,
    suffix and parent are the logical file's; unlink() closes rank 0's descriptor (the memory goes with the last
    mapping, as an unlinked tmpfs file's; no use by path after it: the number may be reused)."""
 def with_segments(self,*args):return Path(*args)
 @property
 def name(self):return self.__dict__.get("_logical")or super().name
 @property
 def suffix(self):return Path(self.name).suffix
 @property
 def parent(self):return Path("memfd")
 def exists(self,*args,**kwargs):return True
 def unlink(self,missing_ok=False):
  fd=self.__dict__.get("_fd");self._fd=None
  if fd is not None:
   try:
    os.close(fd)
   except OSError:
    pass
def memfd_files(base_name,names,rank,world,group):
 """Collective on group: rank 0 makes the memfds (zeros at their sizes) and sends their paths to every rank; every
    rank opens each once, so a refusal raises on every rank before training."""
 import numpy as np;got,fds=[None],[]
 if rank==0:
  try:
   for k,(dt,shape)in names.items():fds.append(os.memfd_create(f"{base_name}.{k}", os.MFD_CLOEXEC));os.ftruncate(fds[-1],np.dtype(dt).itemsize*math.prod(shape))
   got[0] = [(f"/proc/{os.getpid()}/fd/{fd}", f"{base_name}.{k}") for fd, k in zip(fds, names)]
  except Exception as e:got[0] = f"rank 0 could not make the memfds: {e!r}"
 if world>1:dist.broadcast_object_list(got,src=0,group=group)
 err=got[0]if isinstance(got[0],str)else ""
 for p,_logical in[]if err else got[0]:
  try:
   os.close(os.open(p,os.O_RDWR))
  except OSError as e:err = f"rank {rank} cannot open {p}: {e!r}";break
 bad=torch.tensor([int(bool(err))],dtype=torch.int64)
 if world>1:dist.all_reduce(bad,group=group)
 if int(bad):
  for fd in fds:os.close(fd)
  raise RuntimeError(f"the host tables' memfds failed on {int(bad)} of {world} ranks ({err or 'this rank opened them'})")
 out=[]
 for i,(p,logical)in enumerate(got[0]):m=MemPath(p);m._logical,m._fd=logical,(fds[i]if rank==0 else None);out.append(m)
 return out
class NgramHostTable:
 def __init__(self,rows,dim,orders,seed,vocab_pad,*,path="",tag="",rank=0,world=1,group=None,max_rows=0,beta=0.99,eps=1e-8,decay=1e-3,quant=False):
  import numpy as np;self.rows,self.dim,self.orders=rows,dim,tuple(orders);self.rank,self.world,self.group=rank,world,group;self.beta,self.eps,self.decay,self.seed=beta,eps,decay,seed;self.hash_tab=ngram_hash_tables(seed,sum(self.orders),vocab_pad+1);self.q8=self.qs=self.scale=None;self._mapd={};self.bits,self.packed=NGRAM_BITS,False;self.files,self._maps=[],[];self.stats={"steps":0,"rows_updated":0,"skipped":0,"host_seconds":0.0};self.noise_buf=torch.randint(0,1<<16,(1<<23,),generator=torch.Generator().manual_seed(seed),dtype=torch.int32)
  if not path:
   if world>1:raise ValueError("the n-gram table on more than one rank needs a folder for its shared files")
   self.table=torch.zeros(rows,dim,dtype=torch.bfloat16);self.acc=torch.zeros(rows,dtype=torch.float32);return
  base = Path(path) / f"r2_ngram_{tag}";names={"table":(np.int16,(rows,dim)),"acc":(np.float32,(rows,))}
  if world>1:names.update(st_counts=(np.int64,(world,world)),st_ids=(np.int64,(world,max_rows)),st_grads=(np.float32,(world,max_rows,dim)))
  if quant:names.update(q8=(np.uint8,(rows,dim*self.bits//8)),qs=(np.int16,(rows,)))
  self.files = [Path(f"{base}.{k}") for k in names]
  if path=="memfd":self.files=[]
  if rank==0:
   for f in self.files:
    if f.exists():f.unlink()
   for f,(dt,shape)in zip(self.files,names.values()):np.memmap(f,dtype=dt,mode="w+",shape=shape).flush()
  if path=="memfd":self.files = memfd_files(f"r2_ngram_{tag}", names, rank, world, self.group)
  self.barrier();maps={k:np.memmap(f,dtype=dt,mode="r+",shape=shape)for f,(k,(dt,shape))in zip(self.files,names.items())};self._maps=list(maps.values());self._mapd=maps
  if quant:self.q8=torch.from_numpy(maps["q8"]);self.qs=torch.from_numpy(maps["qs"]).view(torch.bfloat16)
  self.table=torch.from_numpy(maps["table"]).view(torch.bfloat16);self.acc=torch.from_numpy(maps["acc"])
  if world>1:self.st_counts=torch.from_numpy(maps["st_counts"]);self.st_ids=torch.from_numpy(maps["st_ids"]);self.st_grads=torch.from_numpy(maps["st_grads"])
  self.barrier()
  if rank==0:self.close(remove=True)
 @classmethod
 def read_only(cls,table,scale,orders,seed,vocab_pad):self=cls.__new__(cls);self.rows,self.dim,self.orders=table.size(0),table.size(1),tuple(orders);self.rank,self.world,self.group=0,1,None;self.hash_tab=ngram_hash_tables(seed,sum(self.orders),vocab_pad+1);self.table,self.scale,self.acc=table,scale,None;self.files,self._maps=[],[];self.bits,self.packed,self.dim=NGRAM_BITS,True,table.size(1)*8//NGRAM_BITS;return self
 def barrier(self):
  if self.world>1:dist.barrier(group=self.group)
 def ids(self,tokens):return torch.stack(ngram_hash_ids(tokens,self.hash_tab,self.orders,self.rows),dim=-1)
 def gather(self,ids):
  flat=ids.reshape(-1).long()
  if self.packed:q,s=ngram_unpack4(self.table.index_select(0,flat),self.dim),self.scale.index_select(0,flat)
  else:q,s=ngram_view4(self.table.index_select(0,flat))
  return ngram_view4_rows(q,s).view(*ids.shape,self.dim)
 def _exchange(self,ids,grads):W,r,n=self.world,self.rank,ids.numel();owner=ids%W;order=torch.argsort(owner,stable=True);self.st_counts[r].copy_(torch.bincount(owner,minlength=W));self.st_ids[r,:n].copy_(ids.index_select(0,order));self.st_grads[r,:n].copy_(grads.index_select(0,order));self.barrier();flat=os1_owner_index(self.st_counts.numpy(),r,self.st_ids.size(1));return self.st_ids.view(-1).index_select(0,flat),self.st_grads.view(-1,self.dim).index_select(0,flat)
 @torch.no_grad()
 def prefault(self):
  t0=time.perf_counter();W,r=self.world,self.rank;views=[t.view(torch.uint8).reshape(-1)for t in(self.table,self.acc)if t is not None];views+=[getattr(self,k).view(torch.uint8).reshape(-1)for k in("st_counts","st_ids","st_grads")if getattr(self,k,None)is not None]
  for b in views:
   pages=b.numel()//4096;lo,hi=pages*r//W,pages*(r+1)//W
   if hi>lo:b[lo*4096:hi*4096:4096].sum()
  self.barrier()
  for b in views[:2]:
   pages=b.numel()//4096;start=(pages*r//W)*4096;b[start::4096].sum()
   if start:b[:start:4096].sum()
  self.barrier();return time.perf_counter()-t0
 @torch.no_grad()
 def step(self,ids,grads,lr,step):
  t0=time.perf_counter();ids=ids.reshape(-1).long();grads=grads.reshape(-1,self.dim).float();own_ids,own_grads=(ids,grads)if self.world==1 else self._exchange(ids,grads)
  if own_ids.numel()and bool(torch.isfinite(own_grads).all()):
   uniq,inv=torch.unique(own_ids,sorted=True,return_inverse=True);g=torch.zeros(uniq.numel(),self.dim,dtype=torch.float32).index_add_(0,inv,own_grads)
   if self.world>1:g.div_(self.world)
   acc=self.acc.index_select(0,uniq).mul_(self.beta).add_(g.square().mean(1),alpha=1.0-self.beta);self.acc.index_copy_(0,uniq,acc);x=self.table.index_select(0,uniq).float().mul_(1.0-self.decay);x.sub_(g.mul_((lr/(acc.sqrt()+self.eps)).unsqueeze(1)));self.table.index_copy_(0,uniq,ngram_round_bf16(x,self.noise(x.numel(),step).view(x.shape)));self.stats["rows_updated"]+=uniq.numel()
  elif own_ids.numel():self.stats["skipped"]+=1
  self.barrier();self.stats["steps"]+=1;self.stats["host_seconds"]+=time.perf_counter()-t0
 def state_for_checkpoint(self):
  if self.q8 is None or self.scale is None:raise RuntimeError("the save needs quantize_for_save on every rank first")
  return{"ngram_table":self.q8,"ngram_scale":self.qs}
 @torch.no_grad()
 def quantize_for_save(self):
  t0=time.perf_counter();R,W,r=self.rows,self.world,self.rank;lo,hi=R*r//W,R*(r+1)//W
  if self.q8 is None:self.q8=torch.zeros(R,self.dim*self.bits//8,dtype=torch.uint8);self.qs=torch.zeros(R,dtype=torch.bfloat16)
  with_state=0
  for a in range(lo,hi,1<<18):b=min(hi,a+(1<<18));m=self.table[a:b];q,s=ngram_view4(m);self.q8[a:b]=ngram_pack4(q);self.qs[a:b]=s;with_state+=int((self.acc[a:b]>0).sum())
  counted=torch.tensor([with_state],dtype=torch.int64)
  if self.world>1:dist.all_reduce(counted,group=self.group)
  self.stats["rows_with_state"]=int(counted.item());self.barrier();self.table,self.scale,self.packed=self.q8,self.qs,True;self.acc=None
  for k in("st_counts","st_ids","st_grads"):
   if hasattr(self,k):setattr(self,k,None)
  for k in("table","acc","st_counts","st_ids","st_grads"):self._mapd.pop(k,None)
  self._maps=list(self._mapd.values());gc.collect();self.barrier();self.stats["quantize_seconds"]=time.perf_counter()-t0
 def noise(self,n,step):
  buf=self.noise_buf;size=buf.numel();o=(step*2654435761+self.rank*40503+12345)%size
  if o+n<=size:return buf[o:o+n]
  if n<=size:return torch.cat([buf[o:],buf[:o+n-size]])
  return buf.repeat((o+n)//size+1)[o:o+n]
 def close(self,remove):
  if remove:
   for f in self.files:
    try:
     f.unlink()
    except OSError:
     pass
def os1_view_scale(q,ms):s=q.abs().sum(1,dtype=torch.int32).to(torch.float32);return(s*ms.to(torch.float32)/q.size(1)).to(torch.bfloat16)
def os1_unbind_bits(neg,vs,signs,valid):import numpy as np;hi=(neg.numpy()^(signs.numpy()<0)).astype(np.uint16)<<np.uint16(15);mag=vs.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16);out=np.where(valid.numpy()[:,None],hi|mag[:,None],hi);return torch.from_numpy(np.ascontiguousarray(out).view(np.int16)).view(torch.bfloat16)
def os1_pack_bits(neg):import numpy as np;return torch.from_numpy(np.packbits(neg.numpy(),axis=1,bitorder="little"))
def os1_unpack_bits(pk,dim):import numpy as np;return torch.from_numpy(np.unpackbits(pk.numpy(),axis=1,count=dim,bitorder="little").view(np.bool_))
def os1_requant(x,noise):s=ngram_bf16_round_up(x.abs().amax(1)/127.0);safe=torch.where(s>0,s,torch.ones_like(s));y=x/safe.unsqueeze(1)+noise.to(torch.float32)*(1.0/65536.0);return torch.floor(y).clamp_(-127,127).to(torch.int8),s.to(torch.bfloat16)
_OS1_LIBC = []
def os1_punch(fd,offset,length):
 if length<=0:return True
 try:
  if not _OS1_LIBC:import ctypes;libc=ctypes.CDLL(None,use_errno=True);libc.fallocate.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_longlong,ctypes.c_longlong];libc.fallocate.restype=ctypes.c_int;_OS1_LIBC.append(libc)
  return _OS1_LIBC[0].fallocate(fd,3,offset,length)==0
 except Exception:return False
class Os1ShmWatch:
 def __init__(self,path):import threading;self.path,self.t0,self.marks=str(path),time.perf_counter(),[];self.start_b=self.peak_b=self._used();self._halt=threading.Event();self._th=threading.Thread(target=self._run,name="r2-os1-shm",daemon=True);self._th.start()
 def _used(self):
  if self.path=="memfd":return host_meminfo("Shmem")
  try:
   st=os.statvfs(self.path);return(st.f_blocks-st.f_bfree)*st.f_frsize
  except(OSError,AttributeError):return 0
 def _run(self):
  while not self._halt.wait(0.2):
   self.peak_b=max(self.peak_b,self._used())
 def mark(self,name):u=self._used();self.peak_b=max(self.peak_b,u);self.marks.append((name,u,time.perf_counter()-self.t0))
 def stop(self):self._halt.set();self._th.join();self.mark("end");return (f"os1_shm: path={self.path} start_GB={self.start_b / 1e9:.1f} peak_GB={self.peak_b / 1e9:.1f} "+ " ".join(f"{n}_GB={u / 1e9:.1f} {n}_s={s:.1f}" for n, u, s in self.marks)+ f" seconds={time.perf_counter() - self.t0:.1f} order=os1_pack_first,ngram_quantize")
def os1_punch_works(folder):
 """The room rule's check (the form's own): one 1 MiB hole punched in a 2 MiB file of this rank's in the tables'
    folder (for "memfd", a memfd of its own), its blocks counted before and after. The synchronous save's pack gives
    the master's pages back by the same call (os1_punch); where it is refused the packed view is written beside the
    whole master."""
 p = Path(folder) / f".os1_punch_test_{os.getpid()}";fd=-1
 try:
  if folder=="memfd":fd=os.memfd_create(p.name,os.MFD_CLOEXEC)
  else:fd=os.open(str(p),os.O_RDWR|os.O_CREAT|os.O_TRUNC,0o600)
  os.write(fd,bytes(2<<20));os.fsync(fd);before=getattr(os.fstat(fd),"st_blocks",None)
  if before is None or not os1_punch(fd,0,1<<20):return False
  return os.fstat(fd).st_blocks<before
 except(OSError,AttributeError):return False
 finally:
  if fd>=0:os.close(fd)
  try:
   if folder!="memfd":p.unlink()
  except OSError:
   pass
def os1_table_threads(timing,key):
 import threading
 if threading.current_thread()is threading.main_thread():return
 torch.set_num_threads(OS1_TT);timing[key]=torch.get_num_threads()
class Os1Record:__slots__=("row","signs","y","nxt","n_win","seq")
class Os1HostTable:
 def __init__(self,rows,dim,order,seed,code_seed,bos,vocab_pad,*,path="",tag="",rank=0,world=1,group=None,max_rows=0,lr=0.125,eps=0.0,cap=4.0,a0=1.0,thr=0.7,gate_w=32,gate_h0=0.3,gate_slope=30.0,diag=False):
  import numpy as np;self.rows,self.dim,self.order,self.seed,self.code_seed,self.bos=rows,dim,order,seed,code_seed,bos;self.packed=False;self.noise_step=0;self.rank,self.world,self.group,self.max_rows=rank,world,group,max_rows;self.lr,self.eps,self.cap,self.a0,self.thr=lr,eps,cap,a0,thr;self.gate_w,self.gate_h0,self.gate_slope,self.diag=gate_w,gate_h0,gate_slope,diag;self.tabs=os1_hash_tables(seed,order,vocab_pad+1);self.code=os1_codebook(vocab_pad,dim,code_seed)if diag else None;self.first=None;self.stats={"update_steps":0,"update_s":0.0,"gather_s":0.0,"gathers":0,"entries":0,"updated_rows":0,"capped":0,"skipped":0,"quantize_s":0.0};self.mix=self._mix_zero();self.record_dict={"rows":rows,"dim":dim,"order":order,"lr":lr,"eps":eps,"cap":cap,"a0":a0,"thr":thr,"gate_w":gate_w,"gate_h0":gate_h0,"gate_slope":gate_slope,"hash_seed":seed,"code_seed":code_seed,"bos":bos,"rule":"normalized step + budget","counts":"none"};names={"table":(np.int8,(rows,dim)),"ms":(np.int16,(rows,)),"pk":(np.uint8,(rows,dim//8)),"vs":(np.int16,(rows,))};self.record_dict.update(bits=1,master="int8 rows, bf16 row scales (max|x| / 127 rounded up)",view="sign(m) x mean|m| (bf16), sign(0) = +1",rounding="stochastic, 16-bit",save="sign bits packed 8 a byte along the width + bf16 view scales")
  if world>1:names.update(st_counts=(np.int64,(world,world)),st_ids=(np.int64,(world,max_rows)),st_grads=(np.float32,(world,max_rows,dim)))
  self.files,self._mapd=[],{};base = Path(path) / f"r2_ngram_{tag}";self.files = [Path(f"{base}.{k}") for k in names]
  if path=="memfd":self.files=[]
  if rank==0:
   for f in self.files:
    if f.exists():f.unlink()
   for f,(dt,shape)in zip(self.files,names.values()):np.memmap(f,dtype=dt,mode="w+",shape=shape).flush()
  if path=="memfd":self.files = memfd_files(f"r2_ngram_{tag}", names, rank, world, self.group)
  self.barrier();self._mapd={k:np.memmap(f,dtype=dt,mode="r+",shape=shape)for f,(k,(dt,shape))in zip(self.files,names.items())};maps={k:torch.from_numpy(v)for k,v in self._mapd.items()};self.table,self.mscale=maps["table"],maps["ms"].view(torch.bfloat16);self.pk,self.vs=maps["pk"],maps["vs"].view(torch.bfloat16);self.noise_buf=torch.randint(0,1<<16,(1<<23,),generator=torch.Generator().manual_seed(seed+1),dtype=torch.int32)
  for k in("st_counts","st_ids","st_grads"):setattr(self,k,maps.get(k))
 @classmethod
 def read_only(cls,table,scale,config,vocab_pad):self=cls.__new__(cls);self.rows,self.dim,self.order=table.size(0),config.os1_dim,config.os1_order;self.seed,self.code_seed,self.bos=config.os1_seed,config.os1_code_seed,config.os1_bos;self.rank,self.world,self.group=0,1,None;self.tabs=os1_hash_tables(config.os1_seed,config.os1_order,vocab_pad+1);self.code,self.diag=None,False;self.pk,self.vs,self.packed=table,scale,True;self.files,self._mapd=[],{};self.stats={"gather_s":0.0,"gathers":0};self.mix=cls._mix_zero();return self
 def barrier(self):
  if self.world>1:dist.barrier(group=self.group)
 def record(self,x,y):
  b,t=x.shape;row,hs=os1_keys(x,self.tabs,self.order,self.bos,self.rows);rec=Os1Record();rec.row=row.reshape(-1);rec.signs=os1_signs(hs.reshape(-1),self.dim);rec.y=None if y is None else y.reshape(-1).to(torch.int64).clone();rec.nxt=None
  if self.diag and y is not None:z=x.to(torch.int64);nx=torch.full_like(z,self.bos);nx[:,:-1]=z[:,1:];rec.nxt=nx.reshape(-1)
  rec.n_win,rec.seq=b,t;return rec
 def first_record(self,x,y):self.first=self.record(x,y)
 def gather_u(self,rec):
  t0=time.perf_counter();valid=rec.row>=0;idx=rec.row.clamp(min=0);u=os1_unbind_bits(*self.view_bits(idx),rec.signs,valid)
  if self.diag and rec.y is not None and rec.nxt is not None:self._mix_add(u,rec)
  self.stats["gather_s"]+=time.perf_counter()-t0;self.stats["gathers"]+=1;return u.view(rec.n_win,rec.seq,self.dim)
 def view_bits(self,idx):
  if self.packed:return os1_unpack_bits(self.pk.index_select(0,idx),self.dim),self.vs.index_select(0,idx)
  q=self.table.index_select(0,idx);return q<0,os1_view_scale(q,self.mscale.index_select(0,idx))
 def lookup_u(self,idx):return self.gather_u(self.record(idx,None))
 @staticmethod
 def _mix_zero():return{"n":0,"read_sum":0.0,"hit":0,"gate":0,"gate_hit":0}
 def _mix_add(self,u,rec):uu=u.float();sel=(rec.row>=0)&(rec.y>=0);sy=(uu*self.code.index_select(0,rec.y.clamp(min=0))).sum(1)/self.a0;z=(uu*self.code.index_select(0,rec.nxt)).sum(1)/self.a0;z=torch.where(rec.nxt==self.bos,torch.zeros_like(z),z).view(rec.n_win,rec.seq);cz=torch.cumsum(z,dim=1);ex=torch.cat([torch.zeros(rec.n_win,1),cz[:,:-1]],dim=1);t=torch.arange(rec.seq);lo=torch.clamp(t-self.gate_w,min=0);h=(ex-ex[:,lo])/torch.clamp(t-lo,min=1).to(torch.float32);g=torch.sigmoid(self.gate_slope*(h-self.gate_h0)).reshape(-1);m=self.mix;m["n"]+=int(sel.sum());m["read_sum"]+=float(sy[sel].sum());m["hit"]+=int((sy[sel]>self.thr).sum());m["gate"]+=int((g[sel]>0.5).sum());m["gate_hit"]+=int(((g[sel]>0.5)&(sy[sel]>self.thr)).sum())
 def _exchange(self,ids,grads):
  W,r,n=self.world,self.rank,ids.numel()
  if n>self.max_rows:raise RuntimeError(f"the output-site table: {n} entries for a staging slot of {self.max_rows}")
  owner=ids%W;order=torch.argsort(owner,stable=True);self.st_counts[r].copy_(torch.bincount(owner,minlength=W));self.st_ids[r,:n].copy_(ids.index_select(0,order));self.st_grads[r,:n].copy_(grads.index_select(0,order));self.barrier();flat=os1_owner_index(self.st_counts.numpy(),r,self.st_ids.size(1));return self.st_ids.view(-1).index_select(0,flat),self.st_grads.view(-1,self.dim).index_select(0,flat)
 def job(self,rec,grads,step):
  self.noise_step=step;self.update_job(rec,grads)
  if step%OS1_LINE_EVERY==0 or self.stats["update_steps"]==1:self.line(step)
 @torch.no_grad()
 def update_job(self,rec,gu):
  t0=time.perf_counter();gu=gu.reshape(-1,self.dim).float();nz=(rec.row>=0)&(gu!=0).any(1);ids=rec.row[nz];gr=gu[nz]*rec.signs[nz].to(torch.float32);own_ids,own_g=(ids,gr)if self.world==1 else self._exchange(ids,gr)
  if own_ids.numel()and bool(torch.isfinite(own_g).all()):
   uniq,inv=torch.unique(own_ids,sorted=True,return_inverse=True);g=torch.zeros(uniq.numel(),self.dim,dtype=torch.float32).index_add_(0,inv,own_g)
   if self.world>1:g.div_(self.world)
   x=self.table.index_select(0,uniq).float()*self.mscale.index_select(0,uniq).float().unsqueeze(1);self.stats["capped"]+=os1_row_step(x,g,self.lr,self.eps,self.cap);q,s=os1_requant(x,self.noise(x.numel(),self.noise_step).view(x.shape));self.table.index_copy_(0,uniq,q);self.mscale.index_copy_(0,uniq,s);self.stats["updated_rows"]+=int(uniq.numel());self.stats["entries"]+=int(own_ids.numel())
  elif own_ids.numel():self.stats["skipped"]+=1
  self.barrier();self.stats["update_steps"]+=1;self.stats["update_s"]+=time.perf_counter()-t0
 def noise(self,n,step):
  buf=self.noise_buf;size=buf.numel();o=(step*2654435761+self.rank*40503+12345)%size
  if o+n<=size:return buf[o:o+n]
  if n<=size:return torch.cat([buf[o:],buf[:o+n-size]])
  return buf.repeat((o+n)//size+1)[o:o+n]
 def line(self,step):
  if self.rank!=0:return
  m,s=self.mix,self.stats;n=max(1,m["n"]);print(f"os1 step {step}: rows={self.rows} update_steps={s['update_steps']} entries_rank0={s['entries']} updated_rows_rank0={s['updated_rows']} capped_rank0={s['capped']} skipped_rank0={s['skipped']} mix_n={m['n']} read_mean={m['read_sum'] / n:.6f} read_hit={m['hit'] / n:.6f} gate_open={m['gate'] / n:.6f} gate_hit={m['gate_hit'] / n:.6f} update_ms={1000 * s['update_s'] / max(1, s['update_steps']):.2f} gather_ms={1000 * s['gather_s'] / max(1, s['gathers']):.2f}", flush=True);self.mix=self._mix_zero()
 def split_pending(self,pending,worker):ids,grads,lr,step=pending[:4];n1=ids.shape[-1];return(ids,grads[:,:,:n1],lr,step)+tuple(pending[4:]),(worker.recs[worker.cur],grads[:,:,n1],step)
 @torch.no_grad()
 def pack_for_save(self):
  import gc;t0=time.perf_counter();R,W,r,d=self.rows,self.world,self.rank,self.dim;lo,hi=R*r//W,R*(r+1)//W;written,at_cap,rms_sum,punched,chunks=0,0,0.0,0,0;bound=0.999*self.cap*self.lr;fds=[os.open(f,os.O_RDWR)for f in self.files if f.suffix in(".table",".ms")]
  try:
   for a in range(lo,hi,1<<18):
    b=min(hi,a+(1<<18));q,ms=self.table[a:b],self.mscale[a:b];self.pk[a:b]=os1_pack_bits(q<0);self.vs[a:b]=os1_view_scale(q,ms);rms=q.to(torch.float32).square().mean(1).sqrt()*ms.to(torch.float32);nzr=(q!=0).any(1);written+=int(nzr.sum());at_cap+=int((rms>=bound).sum());rms_sum+=float(rms[nzr].double().sum());chunks+=1
    if len(fds)==2:punched+=int(os1_punch(fds[0],a*d,(b-a)*d)and os1_punch(fds[1],2*a,2*(b-a)))
  finally:
   for fd in fds:os.close(fd)
  tot=torch.tensor([written,at_cap,self.stats["updated_rows"],self.stats["capped"],self.stats["skipped"],punched,chunks],dtype=torch.int64);rs=torch.tensor([rms_sum],dtype=torch.float64)
  if W>1:dist.all_reduce(tot,group=self.group);dist.all_reduce(rs,group=self.group)
  self.barrier();self.table=self.mscale=None;self.packed=True
  for k in("st_counts","st_ids","st_grads"):setattr(self,k,None)
  for k in("table","ms","st_counts","st_ids","st_grads"):self._mapd.pop(k,None)
  gc.collect();self.barrier()
  if self.rank==0:
   for f in self.files:
    if f.suffix not in(".pk",".vs"):
     try:
      f.unlink()
     except OSError:
      pass
  self.stats["quantize_s"]=time.perf_counter()-t0;written,at_cap,updated,capped,skipped,punched,chunks=tot.tolist();mean_rms=float(rs[0])/max(1,written);self.record_dict.update(written=written,written_frac=written/R,at_cap=at_cap,mean_rms_over_lr=mean_rms/self.lr,updated_row_steps=updated,capped_row_steps=capped,skipped_updates=skipped,update_steps=self.stats["update_steps"],quantize_seconds=round(self.stats["quantize_s"],1),chunks=chunks,punched_chunks=punched)
  if self.rank==0:print(f"os1_save: rows={R} written={written} written_frac={written / R:.6f} at_cap={at_cap} mean_rms_over_lr={mean_rms / self.lr:.4f} updated_row_steps={updated} capped_row_steps={capped} skipped_updates={skipped} update_steps={self.stats['update_steps']} quantize_s={self.stats['quantize_s']:.1f} packed=1bit chunks={chunks} punched_chunks={punched}",flush=True)
 def save_tables(self,ngram):
  watch=Os1ShmWatch(self.files[0].parent)if self.rank==0 and self.files else None
  try:
   self.pack_for_save()
   if watch is not None:watch.mark("packed")
   ngram.quantize_for_save()
  finally:
   if watch is not None:print(watch.stop(),flush=True)
 def state_for_checkpoint(self):
  if not self.packed:raise RuntimeError("the output-site table's save needs pack_for_save on every rank first")
  return{"os1_table":self.pk,"os1_scale":self.vs}
 @torch.no_grad()
 def prefault(self):
  t0=time.perf_counter();W,r=self.world,self.rank;views=[self.table.view(torch.uint8).reshape(-1)];views.append(self.mscale.view(torch.uint8).reshape(-1));views+=[getattr(self,k).view(torch.uint8).reshape(-1)for k in("st_counts","st_ids","st_grads")if getattr(self,k,None)is not None]
  for b in views:
   pages=b.numel()//4096;lo,hi=pages*r//W,pages*(r+1)//W
   if hi>lo:b[lo*4096:hi*4096:4096].sum()
  self.barrier();b=views[0];pages=b.numel()//4096;start=(pages*r//W)*4096;b[start::4096].sum()
  if start:b[:start:4096].sum()
  b=views[1];pages=b.numel()//4096;start=(pages*r//W)*4096;b[start::4096].sum()
  if start:b[:start:4096].sum()
  self.barrier();return time.perf_counter()-t0
 def close(self,remove):
  if remove:
   for f in self.files:
    try:
     f.unlink()
    except OSError:
     pass
 def setup_line(self,reserve_s,typ):R,d,W=self.rows,self.dim,self.world;staging=(W*W*8+W*self.max_rows*(8+4*d))/1e9 if W>1 else 0.0;tgb=R*(d+2)/1e9;sgb=R*(d//8+2)/1e9;return (f"os1: rows={R} dim={d} order={self.order} lr={self.lr:g} a0={self.a0:g} thr={self.thr:g} beta={OS1_BETA:g} glin={OS1_GLIN:g} gate_w={self.gate_w} gate_h0={self.gate_h0:g} gate_slope={self.gate_slope:g} cap={self.cap:g} eps_rel={OS1_EPS_REL:g} eps={self.eps:.6e} typ={typ:.6e} rule=normalized_step+budget counts=none init=zero thread=1 hash_seed={self.seed} code_seed={self.code_seed} bos={self.bos} table_GB={tgb:.3f} staging_GB={staging:.3f} save_GB={sgb:.3f} reserve_s={reserve_s:g} rows_per_rank_step={self.max_rows} line_every={OS1_LINE_EVERY} host_cuts=C1,C2,C3 ngram_exchange=flat bits=1 master=int8+bf16_scale view=sign_x_mean_abs save=packed_1bit+bf16_view_scale save_order=os1_first_chunked")
 def end_line(self,timing):s=self.stats;n=max(1,s["update_steps"]);return (f"os1_host rank {self.rank}: update_steps={s['update_steps']} update_ms={1000 * s['update_s'] / n:.2f} gather_ms={1000 * s['gather_s'] / max(1, s['gathers']):.2f} worker_job_ms={1000 * timing.get('os1_job', 0.0) / n:.2f} worker_gather_ms={1000 * timing.get('os1_gather', 0.0) / n:.2f} worker_wait_ms={1000 * timing.get('os1_wait', 0.0) / n:.2f} entries={s['entries']} updated_rows={s['updated_rows']} capped={s['capped']} skipped={s['skipped']} quantize_s={s['quantize_s']:.1f}")
class NgramWorker:
 def __init__(self,os1,fetch,window_x,window_y,lean,vocab,ngram,window,ids0,shape,timing):self.fetch,self.lean,self.vocab=fetch,lean,vocab;self.bx=[window_x,torch.empty_like(window_x)];self.by=[window_y,torch.empty_like(window_y)];self.ngram,self.shape,self.timing=ngram,shape,timing;self.bn=[window,torch.empty_like(window)];self.ids=[ids0,None];self.cur=0;self.pending=None;self.thread=None;self.result=None;self.error=None;self.timing.setdefault("wait",0.0);self.timing.setdefault("event_wait",0.0);self.timing.setdefault("write",0.0);self.os1=os1;self.recs=[os1.first,None];self.os1_pending=None;self.os1_thread=None;self.os1_error=None;self._os1_in=self._os1_out=self._os1_go=self._os1_done=self._grad_ready=self._os1_rel=None;self.timing.setdefault("os1_gather",0.0);self.timing.setdefault("os1_job",0.0);self.timing.setdefault("os1_wait",0.0)
 def _apply_pending(self):
  if self.pending is not None:
   t=time.perf_counter()
   if self.pending[4]is not None:self.pending[4].synchronize()
   self.pending=self.pending[:4];self.timing["event_wait"]+=time.perf_counter()-t
  if self._grad_ready is not None:self._grad_ready.set()
  if self.pending is not None:t=time.perf_counter();self.ngram.step(*self.pending);self.pending=None;self.timing["update"]+=time.perf_counter()-t
 def _apply_os1_pending(self):
  if self.os1_pending is not None:t=time.perf_counter();self.os1.job(*self.os1_pending);self.os1_pending=None;self.timing["os1_job"]+=time.perf_counter()-t
 def _os1_rows(self,target,x,y):t=time.perf_counter();self.recs[target]=self.os1.record(x.view(self.shape),y.view(self.shape));u=self.os1.gather_u(self.recs[target]);self.timing["os1_gather"]+=time.perf_counter()-t;return u
 def _os1_work(self):
  try:
   os1_table_threads(self.timing,"tt_os1");self._grad_ready.wait();self._os1_rel.wait();self._apply_os1_pending();self._os1_go.wait()
   if self._os1_in is not None:self._os1_out=self._os1_rows(*self._os1_in)
  except BaseException as err:self.os1_error=err
  finally:
   self._os1_done.set()
 def _os1_take(self):
  t=time.perf_counter();self._os1_done.wait();self.timing["os1_wait"]+=time.perf_counter()-t
  if self.os1_error is not None or self._os1_out is None:raise RuntimeError("the output-site table's thread failed")from self.os1_error
  return self._os1_out
 def os1_release(self):
  if self._os1_rel is not None:self._os1_rel.set()
 def _join_os1(self):
  if self.os1_thread is not None:self._grad_ready.set();self._os1_rel.set();self._os1_go.set();self.os1_thread.join();self.os1_thread=None
 def _work(self,target):
  try:
   os1_table_threads(self.timing,"tt_worker");self._apply_pending();x,y,state=self.fetch();self._os1_in=(target,x,y);self._os1_go.set();host={};host["x"],host["y"]=x.view(self.bx[target].shape).clone(),y.view(self.by[target].shape).clone();host_ids=None;relabel=lean_relabel(x,self.vocab)
   if relabel is None:host_ids=x.clone()
   else:host["cid"],host["rows"]=relabel[0],relabel[1]
   t=time.perf_counter();self.ids[target]=self.ngram.ids(x.view(self.shape));rows=self.ngram.gather(self.ids[target]);self.timing["gather"]+=time.perf_counter()-t;host["bn"]=torch.cat([rows,self._os1_take().unsqueeze(2)],dim=2);self.result=(target,state,host_ids,host)
  except BaseException as err:self.error=err
 def start(self):
  import threading
  if self.os1_thread is not None:raise RuntimeError("the output-site table's thread is already running")
  self._os1_in=self._os1_out=self.os1_error=None;self._os1_go,self._os1_done,self._grad_ready=threading.Event(),threading.Event(),threading.Event();self._os1_rel=threading.Event();self.os1_thread=threading.Thread(target=self._os1_work,name="os1-table",daemon=True);self.os1_thread.start()
  if self.thread is not None:raise RuntimeError("the n-gram worker is already running")
  self.result,self.error=None,None;self.thread=threading.Thread(target=self._work,args=(1-self.cur,),name="ngram-worker",daemon=True);self.thread.start()
 def finish(self):
  self.os1_release()
  try:
   out=self._finish()
  finally:
   self._join_os1()
  if self.os1_error is not None:raise RuntimeError("the output-site table's thread failed")from self.os1_error
  return out
 def _finish(self):
  t=time.perf_counter()
  if self.thread is not None:self.thread.join();self.thread=None
  if self.error is not None:raise RuntimeError("the n-gram worker's job failed")from self.error
  target,state,host_ids,host=self.result;self.timing["wait"]+=time.perf_counter()-t;t=time.perf_counter();self.bx[target].copy_(host["x"],non_blocking=True);self.by[target].copy_(host["y"],non_blocking=True);cid,rows,hostref=self.lean;hostref[0]=host_ids
  if host_ids is None:cid.copy_(host["cid"],non_blocking=True);rows.copy_(host["rows"],non_blocking=True)
  self.bn[target].copy_(host["bn"],non_blocking=True);self.timing["write"]+=time.perf_counter()-t;self.cur=target;return self.bx[target],self.by[target],state
 def close(self):
  self.os1_release()
  if self.thread is not None:self.thread.join();self.thread=None
  self._join_os1()
 def flush(self):
  self.close()
  if(self.error is not None or self.os1_error is not None)and(self.pending is not None or self.os1_pending is not None):raise RuntimeError("a table worker failed before its updates")from(self.error or self.os1_error)
  self._apply_pending();self._apply_os1_pending()
class BoundaryS11:
 def __init__(self,device,group,rank):
  self.group,self.rank=group,rank;self.event_cls=None
  if device.type=="neuron":
   self.event_cls=getattr(getattr(torch,"neuron",None),"Event",None)
   if self.event_cls is None:import torch_neuronx;self.event_cls=torch_neuronx.Event
  self.prev_obs=time.monotonic();self.near_end=False;self.n={"async":0,"sync":0,"tails":0};self.ms={"wait":0.0,"ctrl":0.0,"read":0.0}
 def record(self):
  if self.event_cls is None:return None
  ev=self.event_cls();ev.record();return ev
 def wait(self,ev):
  t=time.perf_counter()
  if ev is not None:ev.synchronize()
  self.ms["wait"]+=1000.0*(time.perf_counter()-t)
 def control(self,values):
  t=time.perf_counter();x=torch.tensor(values,dtype=torch.int64)
  if self.group is not None:dist.all_reduce(x,op=dist.ReduceOp.MAX,group=self.group)
  self.ms["ctrl"]+=1000.0*(time.perf_counter()-t);return[int(v)for v in x.tolist()]
 def exit_async(self,step,explicit_last):ok=(step>=TRAIN_STARTUP_STEPS_EXCLUDED and not explicit_last and not self.near_end and(step+1)%BOUNDARY_SYNC_EVERY!=0);self.n["async" if ok else "sync"]+=1;return ok
 def line(self,rank,timing):n=max(1,self.n["tails"]);m=max(1,self.n["async"]+self.n["sync"]);return (f"S11 boundary rank {rank}: async_exits={self.n['async']} sync_exits={self.n['sync']} tails={self.n['tails']} wait_ms={self.ms['wait'] / n:.2f} ctrl_ms={self.ms['ctrl'] / n:.2f} read_ms={self.ms['read'] / m:.2f} write_ms={1000.0 * timing.get('write', 0.0) / m:.2f} worker_event_wait_ms={1000.0 * timing.get('event_wait', 0.0) / m:.2f} near_end={int(self.near_end)}")
def ngram_table_folder(preferred,needed,punch_ok,group,rank,world):
 """Rank 0's choice for every rank: preferred (/dev/shm) if its free bytes hold the peak + 20 GB, else "memfd"
    (memfd_files) if MemAvailable and any cgroup memory limit do (the punch checked on a memfd), else an error;
    needed maps the punch's result to the peak. Nothing of the tables goes to disk."""
 def free(folder):
  try:
   return shutil.disk_usage(folder).free if Path(folder).is_dir()else 0
  except OSError:return 0
 def text(p):
  try:
   return Path(p).read_text().strip()
  except OSError:return ""
 def mem_limit():
  rel=next((ln.split(":",2)[2]for ln in text("/proc/self/cgroup").splitlines()if ln.startswith("0::")),"")
  for p in (f"/sys/fs/cgroup{rel}/memory.max", "/sys/fs/cgroup/memory.max","/sys/fs/cgroup/memory/memory.limit_in_bytes"):
   v=text(p)
   if v:return int(v)if v.isdigit()and int(v)<1<<60 else 0
  return 0
 choice=["",needed[punch_ok]]
 if rank==0:
  if free(preferred)>=needed[punch_ok]+20_000_000_000:choice[0]=preferred
  else:
   punch_mem=os1_punch_works("memfd");avail,lim=host_meminfo("MemAvailable"),mem_limit();choice[1]=needed[punch_mem]
   if min(avail,lim or avail)>=choice[1]+20_000_000_000:choice[0]="memfd"
   thp=text("/sys/kernel/mm/transparent_hugepage/shmem_enabled");print(f"os1_room: punch={'works' if punch_mem else 'refused'} folder={choice[0] or 'none'} mem_available={avail:,} cgroup_limit={lim:,} needed={choice[1]:,} ({preferred} free {free(preferred):,}) shmem_thp={thp[thp.find('[') + 1:thp.find(']')] if '[' in thp else thp}",flush=True)
 if world>1:dist.broadcast_object_list(choice,src=0,group=group)
 if not choice[0]:raise RuntimeError(f"no room for the host tables ({choice[1]:,} bytes + 20 GB; {preferred} has {free(preferred):,} free)")
 return choice[0]
COMPUTE_DTYPE: torch.dtype = torch.bfloat16
FUSED_ATTENTION = False
ATTN_BAND_KERNEL = False
TRAINING_KERNELS = False
def norm(x):return F.rms_norm(x,(x.size(-1),))
class Linear(nn.Linear):
 def forward(self,x):return F.linear(x,self.weight.to(dtype=x.dtype),None)
def apply_rotary_emb(x,cos,sin):d=x.shape[-1]//2;x1,x2=x[...,:d],x[...,d:];y1=x1*cos+x2*sin;y2=x1*(-sin)+x2*cos;return torch.cat((y1,y2),dim=-1)
def attn_long_layers(depth):return tuple(i for i in range(depth)if i%4==3 or i==depth-1)
def attn_band_mask(t,nw):i=torch.arange(t,dtype=torch.int64);return(i[None,:]<=i[:,None])&(i[None,:]//128>i[:,None]//128-nw)
def attn_band_attach(model):
 cfg=model.config
 if not cfg.attn_window:return 0
 mask=attn_band_mask(cfg.sequence_len,cfg.attn_window//128).to(model.cos.device);n=0
 for blk in model.transformer.h:
  if blk.attn.attn_nw:blk.attn.register_buffer("attn_band",mask,persistent=False);n+=1
 return n
ATTN_HEAD_DIM = 128
dc = 1.0 / math.sqrt(ATTN_HEAD_DIM)
dS = dc * ATTN_HEAD_DIM
_P = 128
cV = 4
di = 4
eL = 2.0 ** -23
try:
 if __name__=="__mp_main__":raise ImportError
 import nki;import nki.language as nl;import nki.isa as nisa;from torch_neuronx import nki_op
 def _shard():
  if nl.program_ndim()>0:return nl.program_id(0),nl.num_programs(0)
  return 0,1
 def aK():return nisa.engine.unknown
 def aQ(a,b,h,T,C,D,G):l=nl.ndarray((_P,G*D),dtype=a.dtype,buffer=nl.sbuf);nisa.dma_copy(dst=l,src=a.ap(pattern=[[C,_P],[_P*C,G],[1,D]],offset=b*T*C+h*D));return l
 def aS(a,T,D,G):
  o=nl.ndarray((D,T),dtype=a.dtype,buffer=nl.sbuf)
  for d in range(0,G,4):
   l=G-d
   if l>4:l=4
   m=nl.ndarray((D,l*_P),dtype=a.dtype,buffer=nl.psum)
   for f in range(l):g=d+f;nisa.nc_transpose(dst=m[:,f*_P:(f+1)*_P],data=a[:,g*D:(g+1)*D])
   nisa.tensor_copy(dst=o[:,d*_P:(d+l)*_P],src=m,engine=aK())
  return o
 def a5(d,f,G,a):l=nl.ndarray((_P,G,1,a),dtype=d.dtype,buffer=nl.sbuf);k=nl.ndarray((_P,G,1,a),dtype=f.dtype,buffer=nl.sbuf);nisa.dma_copy(dst=l,src=d.ap(pattern=[[a,_P],[_P*a,G],[1,a]],offset=0));nisa.dma_copy(dst=k,src=f.ap(pattern=[[a,_P],[_P*a,G],[1,a]],offset=0));return l.broadcast(2,2),k.broadcast(2,2)
 def a4(m,o,p,x,d,f,k,a):D=2*k;l,v=o[:,:,:,0:k],o[:,:,:,k:D];z,L=m[:,:,:,0:k],m[:,:,:,k:D];nisa.tensor_tensor(dst=d,data1=l,data2=p,op=nl.multiply);nisa.tensor_tensor(dst=f,data1=v,data2=x,op=nl.multiply);nisa.tensor_tensor(dst=z,data1=d,data2=f,op=nl.subtract if a else nl.add,engine=nisa.engine.unknown);nisa.tensor_tensor(dst=d,data1=v,data2=p,op=nl.multiply);nisa.tensor_tensor(dst=f,data1=l,data2=x,op=nl.multiply);nisa.tensor_tensor(dst=L,data1=d,data2=f,op=nl.add if a else nl.subtract,engine=nisa.engine.unknown)
 def bK(r,d,G,D):l=nl.ndarray((_P,G,2,1),dtype=nl.float32,buffer=nl.sbuf);nisa.activation(dst=d,op=nl.square,data=r);nisa.tensor_reduce(dst=l,op=nl.add,data=d,axis=3,keepdims=True);a=nl.ndarray((_P,G,2,1),dtype=nl.float32,buffer=nl.sbuf);nisa.tensor_scalar(dst=l,data=l,op0=nl.multiply,operand0=1.0/D,op1=nl.add,operand1=eL);nisa.tensor_scalar(dst=a,data=l,op0=nl.rsqrt,operand0=0.0);return a
 def a3(a,f,b,h,T,C,G,D,d):
  l=3*C;m=b*T*l+h*D
  for s in range(2):
   k=f.ap(pattern=[[l,_P],[_P*l,G],[1,D]],offset=m+s*C)
   if d:nisa.dma_copy(dst=k,src=a[:,:,s,:])
   else:nisa.dma_copy(dst=a[:,:,s,:],src=k)
 def a7(a,m,b,h,B,H,T,G,D,d):
  l=(b*H+h)*T*D;p=B*H*T*D
  for s in range(2):
   k=m.ap(pattern=[[D,_P],[_P*D,G],[1,D]],offset=l+s*p)
   if d:nisa.dma_copy(dst=k,src=a[:,:,s,:])
   else:nisa.dma_copy(dst=a[:,:,s,:],src=k)
 def aN(f,s,T,D,G):
  o=nl.ndarray((D,T),dtype=f.dtype,buffer=nl.sbuf)
  for a in range(0,G,4):
   l=G-a
   if l>4:l=4
   m=nl.ndarray((D,l*_P),dtype=f.dtype,buffer=nl.psum)
   for d in range(l):g=a+d;nisa.nc_transpose(dst=m[:,d*_P:(d+1)*_P],data=f[:,g,s,:])
   nisa.tensor_copy(dst=o[:,a*_P:(a+l)*_P],src=m,engine=aK())
  return o
 def Y(f,s,T,D,G):
  o=nl.ndarray((D,T),dtype=f.dtype,buffer=nl.sbuf)
  for a in range(0,G,4):
   l=G-a
   if l>4:l=4
   m=nl.ndarray((D,l*_P),dtype=f.dtype,buffer=nl.psum)
   for d in range(l):g=a+d;nisa.nc_transpose(dst=m[:,d*_P:(d+1)*_P],data=f[:,g,s,:])
   nisa.tensor_copy(dst=o[:,a*_P:(a+l)*_P],src=m,engine=nisa.engine.vector)
  return o
 def bT(a,d,b,h,T,C,G,D):l=3*C;W=D+1;nisa.dma_copy(dst=a.reshape_dim(1,(G,W))[:,:,0:D],src=d.ap(pattern=[[l,_P],[_P*l,G],[1,D]],offset=b*T*l+2*C+h*D),dge_mode=nisa.dge_mode.hwdge,engine=nisa.engine.scalar);return a
 def bQ(a,f,b,h,T,C,G,D):nisa.dma_copy(dst=f.ap(pattern=[[C,_P],[_P*C,G],[1,D]],offset=b*T*C+h*D),src=a,dge_mode=nisa.dge_mode.hwdge,engine=nisa.engine.scalar)
 def bY(a,G,D):W=D+1;l=nl.ndarray((_P,G*W),dtype=a,buffer=nl.sbuf);d=nl.ndarray((_P,G*W),dtype=a,buffer=nl.sbuf);nisa.memset(l,value=1.0);nisa.memset(d,value=1.0);m=nl.ndarray((_P,G,D),dtype=a,buffer=nl.sbuf);o=nl.ndarray((_P,G,D),dtype=a,buffer=nl.sbuf);return l,d,m,o
 @nki.jit
 def _attn_rope_fwd_kernel_band(o,br,bx,a2,v,nw):
  B,T,C3=o.shape;C=C3//3;D=_P;H=C//D;x=D//2;bs=B*H;G=T//_P;W=D+1;k=nl.float32;m=o.dtype;o3=nl.ndarray((B,T,C),dtype=m,buffer=nl.shared_hbm);ao=nl.ndarray((bs,_P,G),dtype=k,buffer=nl.shared_hbm);am=nl.ndarray((bs,_P,G,2,1),dtype=k,buffer=nl.shared_hbm);r5=nl.ndarray((2,B,H,T,D),dtype=m,buffer=nl.shared_hbm);bq,aj=_shard();cb,sb=a5(br,bx,G,x);al=nl.ndarray((_P,1),dtype=k,buffer=nl.sbuf);nisa.memset(al,value=-v);bw,bz,bo,bp=bY(m,G,D)
  for i in range(bq,bs,aj):
   b=i//H;h=i%H;bd=nl.ndarray((_P,G,2,D),dtype=m,buffer=nl.sbuf);a3(bd,o,b,h,T,C,G,D,False);az=nl.ndarray((_P,G,2,x),dtype=m,buffer=nl.sbuf);ba=nl.ndarray((_P,G,2,x),dtype=m,buffer=nl.sbuf);r=nl.ndarray((_P,G,2,D),dtype=m,buffer=nl.sbuf);a4(r,bd,cb,sb,az,ba,x,False);sq=nl.ndarray((_P,G,2,D),dtype=k,buffer=nl.sbuf);bb=bK(r,sq,G,D);yn=nl.ndarray((_P,G,2,D),dtype=m,buffer=nl.sbuf);nisa.tensor_tensor(dst=yn,data1=r,data2=bb.broadcast(3,D),op=nl.multiply);a7(r,r5,b,h,B,H,T,G,D,True);nisa.dma_copy(dst=am[i,:,:,:,:],src=bb);qT=Y(yn,0,T,D,G);kT=Y(yn,1,T,D,G);bg=bT(bw if(i//aj)%2==0 else bz,o,b,h,T,C,G,D);aq=bo if(i//aj)%2==0 else bp;l=nl.ndarray((_P,G),dtype=k,buffer=nl.sbuf);ar=nl.ndarray((_P,G),dtype=k,buffer=nl.sbuf)
   for g in range(G):
    L=nl.ndarray((_P,W),dtype=k,buffer=nl.psum);R=g-nw+1
    if R<0:R=0
    for j0 in range(R,g+1,cV):
     p=g+1-j0
     if p>cV:p=cV
     st=nl.ndarray((_P,p*_P),dtype=k,buffer=nl.psum)
     for a in range(p):j=j0+a;nisa.nc_matmul(dst=st[:,a*_P:(a+1)*_P],stationary=kT[:,j*_P:(j+1)*_P],moving=qT[:,g*_P:(g+1)*_P],accumulate=False)
     S=nl.ndarray((_P,p*_P),dtype=m,buffer=nl.sbuf);nisa.activation(dst=S,op=nl.exp,data=st,bias=al,scale=a2)
     if j0+p-1==g:a=p-1;nisa.affine_select(dst=S[:,a*_P:(a+1)*_P],pattern=[[1,_P]],channel_multiplier=-1,offset=0,cmp_op=nl.greater_equal,on_true_tile=S[:,a*_P:(a+1)*_P],on_false_value=0.0)
     for a in range(p):j=j0+a;nisa.nc_matmul(dst=L,stationary=S[:,a*_P:(a+1)*_P],moving=bg[:,j*W:(j+1)*W],accumulate=(j>R))
    rs=nl.ndarray((_P,1),dtype=k,buffer=nl.sbuf);nisa.reciprocal(dst=rs,data=L[:,D:D+1]);ot=aq[:,g,:];nisa.tensor_scalar(dst=ot,data=L[:,0:D],op0=nl.multiply,operand0=rs,engine=nisa.engine.vector);nisa.tensor_copy(dst=ar[:,g:g+1],src=L[:,D:D+1],engine=nisa.engine.vector)
   bQ(aq,o3,b,h,T,C,G,D);nisa.activation(dst=l,op=nl.log,data=ar);nisa.tensor_scalar(dst=l,data=l,op0=nl.add,operand0=v,engine=nisa.engine.vector);nisa.dma_copy(dst=ao[i,:,:],src=l)
  return o3,ao,r5,am
 @nki.jit
 def _attn_rope_bwd_kernel_band(a2,r5,cv,o3,cL,cw,cu,cz,o,nw):
  B,T,R=a2.shape;C=R//3;D=_P;H=C//D;Dh=D//2;bs=B*H;G=T//_P;k=nl.float32;f=a2.dtype;ar=nl.ndarray((B,T,R),dtype=f,buffer=nl.shared_hbm);de,cq=_shard();cb,sb=a5(cu,cz,G,Dh)
  for i in range(de,bs,cq):
   b=i//H;h=i%H;rr=nl.ndarray((_P,G,2,D),dtype=f,buffer=nl.sbuf);a7(rr,r5,b,h,B,H,T,G,D,False);az=nl.ndarray((_P,G,2,1),dtype=k,buffer=nl.sbuf);nisa.dma_copy(dst=az,src=cv[i,:,:,:,:]);yn=nl.ndarray((_P,G,2,D),dtype=f,buffer=nl.sbuf);nisa.tensor_tensor(dst=yn,data1=rr,data2=az.broadcast(3,D),op=nl.multiply);on=aQ(o3,b,h,T,C,D,G);bl=aQ(cL,b,h,T,C,D,G);qT=aN(yn,0,T,D,G);kT=aN(yn,1,T,D,G);cy=aS(bl,T,D,G);ck=b*T*R+2*C+h*D;vn=nl.ndarray((_P,G*D),dtype=f,buffer=nl.sbuf);nisa.dma_copy(dst=vn,src=a2.ap(pattern=[[R,_P],[_P*R,G],[1,D]],offset=ck));vT=aS(vn,T,D,G);nb=nl.ndarray((_P,G),dtype=k,buffer=nl.sbuf);nisa.dma_copy(dst=nb,src=cw[i,:,:]);nisa.tensor_scalar(dst=nb,data=nb,op0=nl.multiply,operand0=-1.0);bo=nl.ndarray((_P,G),dtype=k,buffer=nl.sbuf);ao=nl.ndarray((_P,G*D),dtype=k,buffer=nl.sbuf);nisa.tensor_tensor(dst=ao,data1=bl,data2=on,op=nl.multiply);nisa.tensor_reduce(dst=bo,op=nl.add,data=ao.reshape_dim(1,(G,D)),axis=2);ai=nl.ndarray((_P,G*D),dtype=k,buffer=nl.sbuf);aj=nl.ndarray((_P,G*D),dtype=k,buffer=nl.sbuf);bh=nl.ndarray((_P,G,2,D),dtype=f,buffer=nl.sbuf)
   for p in range(0,G,di):
    z=G-p
    if z>di:z=di
    wg=(p+z)*_P;pg=nl.ndarray((_P,z*wg),dtype=f,buffer=nl.sbuf);bm=nl.ndarray((_P,z*wg),dtype=f,buffer=nl.sbuf)
    for m in range(z):
     g=p+m;bg=nl.ndarray((_P,D),dtype=k,buffer=nl.psum);bi=g-nw+1
     if bi<0:bi=0
     for S in range(bi,g+1,cV):
      nj=g+1-S
      if nj>cV:nj=cV
      w=nj*_P;x=m*wg+S*_P;s=nl.ndarray((_P,w),dtype=k,buffer=nl.psum);nisa.nc_matmul(dst=s,stationary=qT[:,g*_P:(g+1)*_P],moving=kT[:,S*_P:S*_P+w],accumulate=False);nisa.activation(dst=pg[:,x:x+w],op=nl.exp,data=s,bias=nb[:,g:g+1],scale=o)
      if S+nj-1==g:db=x+(nj-1)*_P;nisa.affine_select(dst=pg[:,db:db+_P],pattern=[[-1,_P]],channel_multiplier=1,offset=0,cmp_op=nl.greater_equal,on_true_tile=pg[:,db:db+_P],on_false_value=0.0)
      dp=nl.ndarray((_P,w),dtype=k,buffer=nl.psum);nisa.nc_matmul(dst=dp,stationary=cy[:,g*_P:(g+1)*_P],moving=vT[:,S*_P:S*_P+w],accumulate=False);nisa.scalar_tensor_tensor(dst=bm[:,x:x+w],data=dp,op0=nl.subtract,operand0=bo[:,g:g+1],op1=nl.multiply,operand1=pg[:,x:x+w]);ay=nl.ndarray((_P,w),dtype=f,buffer=nl.psum)
      for U in range(nj):nisa.nc_transpose(dst=ay[:,U*_P:(U+1)*_P],data=bm[:,x+U*_P:x+(U+1)*_P])
      co=nl.ndarray((_P,w),dtype=f,buffer=nl.sbuf);nisa.tensor_copy(dst=co,src=ay,engine=aK())
      for U in range(nj):j=S+U;nisa.nc_matmul(dst=bg,stationary=co[:,U*_P:(U+1)*_P],moving=yn[:,j,1,:],accumulate=(j>bi))
     nisa.tensor_scalar(dst=bh[:,g,0,:],data=bg,op0=nl.multiply,operand0=o)
    bp=p-nw+1
    if bp<0:bp=0
    for j in range(bp,p+z):
     fg=j-p
     if fg<0:fg=0
     lg=j+nw-p
     if lg>z:lg=z
     au=nl.ndarray((_P,D),dtype=k,buffer=nl.psum)
     for m in range(fg,lg):g=p+m;nisa.nc_matmul(dst=au,stationary=pg[:,m*wg+j*_P:m*wg+(j+1)*_P],moving=bl[:,g*D:(g+1)*D],accumulate=(m>fg))
     if j>=p:nisa.tensor_copy(dst=aj[:,j*D:(j+1)*D],src=au,engine=nisa.engine.scalar)
     else:nisa.tensor_tensor(dst=aj[:,j*D:(j+1)*D],data1=aj[:,j*D:(j+1)*D],data2=au,op=nl.add)
     aq=nl.ndarray((_P,D),dtype=k,buffer=nl.psum)
     for m in range(fg,lg):g=p+m;nisa.nc_matmul(dst=aq,stationary=bm[:,m*wg+j*_P:m*wg+(j+1)*_P],moving=yn[:,g,0,:],accumulate=(m>fg))
     if j>=p:nisa.tensor_scalar(dst=ai[:,j*D:(j+1)*D],data=aq,op0=nl.multiply,operand0=o,engine=nisa.engine.scalar)
     else:nisa.scalar_tensor_tensor(dst=ai[:,j*D:(j+1)*D],data=aq,op0=nl.multiply,operand0=o,op1=nl.add,operand1=ai[:,j*D:(j+1)*D])
   nisa.tensor_copy(dst=bh[:,:,1,:],src=ai.reshape_dim(1,(G,D)),engine=nisa.engine.scalar);am=nl.ndarray((_P,G,D),dtype=f,buffer=nl.sbuf);nisa.tensor_copy(dst=am,src=aj.reshape_dim(1,(G,D)),engine=nisa.engine.scalar);nisa.dma_copy(dst=ar.ap(pattern=[[R,_P],[_P*R,G],[1,D]],offset=b*T*R+2*C+h*D),src=am);cp=f;sq=nl.ndarray((_P,G,2,D),dtype=cp,buffer=nl.sbuf);nisa.tensor_tensor(dst=sq,data1=bh,data2=rr,op=nl.multiply);cm=nl.ndarray((_P,G,2,1),dtype=k,buffer=nl.sbuf);nisa.tensor_reduce(dst=cm,op=nl.add,data=sq,axis=3,keepdims=True);ag=nl.ndarray((_P,G,2,1),dtype=k,buffer=nl.sbuf);nisa.tensor_tensor(dst=ag,data1=az,data2=az,op=nl.multiply);nisa.tensor_tensor(dst=ag,data1=ag,data2=cm,op=nl.multiply);nisa.tensor_scalar(dst=ag,data=ag,op0=nl.multiply,operand0=1.0/D);u=nl.ndarray((_P,G,2,D),dtype=cp,buffer=nl.sbuf);nisa.tensor_tensor(dst=u,data1=rr,data2=ag.broadcast(3,D),op=nl.multiply);nisa.tensor_tensor(dst=u,data1=bh,data2=u,op=nl.subtract,engine=nisa.engine.unknown);gr=nl.ndarray((_P,G,2,D),dtype=f,buffer=nl.sbuf);nisa.tensor_tensor(dst=gr,data1=u,data2=az.broadcast(3,D),op=nl.multiply);gx=nl.ndarray((_P,G,2,D),dtype=f,buffer=nl.sbuf);ce=nl.ndarray((_P,G,2,Dh),dtype=f,buffer=nl.sbuf);cj=nl.ndarray((_P,G,2,Dh),dtype=f,buffer=nl.sbuf);a4(gx,gr,cb,sb,ce,cj,Dh,True);a3(gx,ar,b,h,T,C,G,D,True)
  return ar
 _HAS_ATTN_KERNEL=True
except ImportError:_HAS_ATTN_KERNEL=False
c2: dict = {}
def cZ():
 if hasattr(torch.ops.fused_attention,"rope_attn_fwd"):return{"fwd":torch.ops.fused_attention.rope_attn_fwd,"bwd":torch.ops.fused_attention.rope_attn_bwd,"band":torch.ops.fused_attention.rope_attn_fwd_band}
 fwd_kernel=_attn_rope_fwd_kernel_band[2];bwd_kernel=_attn_rope_bwd_kernel_band[2];nw=ATTN_WINDOW//128
 @nki_op("fused_attention::rope_attn_fwd",mutates_args=())
 def rope_attn_fwd(qkv:torch.Tensor,cos2:torch.Tensor,sin2:torch.Tensor,scale:float,shift:float)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:o3,ab,r5,aa=fwd_kernel(qkv,cos2,sin2,scale,shift,qkv.shape[1]//128);return o3,ab,r5,aa
 @nki_op("fused_attention::rope_attn_bwd",mutates_args=())
 def rope_attn_bwd(qkv:torch.Tensor,r5:torch.Tensor,inv3:torch.Tensor,o3:torch.Tensor,do3:torch.Tensor,lse3:torch.Tensor,cos2:torch.Tensor,sin2:torch.Tensor,scale:float)->torch.Tensor:return bwd_kernel(qkv,r5,inv3,o3,do3,lse3,cos2,sin2,scale,qkv.shape[1]//128)
 @nki_op("fused_attention::rope_attn_fwd_band",mutates_args=())
 def rope_attn_fwd_band(qkv:torch.Tensor,cos2:torch.Tensor,sin2:torch.Tensor,scale:float,shift:float)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:o3,ab,r5,aa=fwd_kernel(qkv,cos2,sin2,scale,shift,nw);return o3,ab,r5,aa
 @nki_op("fused_attention::rope_attn_bwd_band",mutates_args=())
 def rope_attn_bwd_band(qkv:torch.Tensor,r5:torch.Tensor,inv3:torch.Tensor,o3:torch.Tensor,do3:torch.Tensor,lse3:torch.Tensor,cos2:torch.Tensor,sin2:torch.Tensor,scale:float)->torch.Tensor:return bwd_kernel(qkv,r5,inv3,o3,do3,lse3,cos2,sin2,scale,nw)
 def rope_attn_fwd_fake(qkv,cos2,sin2,scale,shift):b,t,c3=qkv.shape;h,g=c3//3//ATTN_HEAD_DIM,t//128;return(qkv.new_empty(b,t,c3//3),qkv.new_empty(b*h,128,g,dtype=torch.float32),qkv.new_empty(2,b,h,t,ATTN_HEAD_DIM),qkv.new_empty(b*h,128,g,2,1,dtype=torch.float32))
 def rope_attn_bwd_fake(qkv,r5,inv3,o3,do3,lse3,cos2,sin2,scale):return torch.empty_like(qkv)
 for ab,aa in(("fused_attention::rope_attn_fwd",rope_attn_fwd_fake),("fused_attention::rope_attn_bwd",rope_attn_bwd_fake),("fused_attention::rope_attn_fwd_band",rope_attn_fwd_fake),("fused_attention::rope_attn_bwd_band",rope_attn_bwd_fake)):
  try:
   torch.library.register_fake(ab)(aa)
  except RuntimeError:
   pass
 def setup_context(ctx,inputs,output):bf,a2,a8,ap,bc=inputs;o3,ab,r5,aa=output;ctx.save_for_backward(bf,a2,a8,o3,ab,r5,aa);ctx.scale=ap;ctx.mark_non_differentiable(ab,r5,aa);ctx.set_materialize_grads(False)
 def backward(ctx,d_o3,_d_lse3,_d_r5,_d_inv3):bc,aa,a8,o3,a2,r5,ap=ctx.saved_tensors;ab=rope_attn_bwd(bc,r5,ap,o3,d_o3.contiguous(),a2,aa,a8,ctx.scale);return ab,None,None,None,None
 torch.library.register_autograd("fused_attention::rope_attn_fwd",backward,setup_context=setup_context)
 def backward_band(ctx,d_o3,_d_lse3,_d_r5,_d_inv3):bc,aa,a8,o3,a2,r5,ap=ctx.saved_tensors;ab=rope_attn_bwd_band(bc,r5,ap,o3,d_o3.contiguous(),a2,aa,a8,ctx.scale);return ab,None,None,None,None
 torch.library.register_autograd("fused_attention::rope_attn_fwd_band",backward_band,setup_context=setup_context);return{"fwd":rope_attn_fwd,"bwd":rope_attn_bwd,"band":rope_attn_fwd_band}
def fused_attention_for(aa):
 if aa.type!="neuron":return False
 if os.environ.get("NEURON_LOGICAL_NC_CONFIG","2")!="2":return False
 if not _HAS_ATTN_KERNEL:raise RuntimeError("the fused attention kernels need nki and torch_neuronx, which a Neuron host provides")
 if not c2:c2.update(cZ())
 return True
def fused_attention(qkv,cos,sin,n_head,qk_scale=0.0):
 b,t,_=qkv.shape
 if t%128 or t>4096 or not qkv.is_contiguous()or cos.dtype!=qkv.dtype:raise ValueError(f"fused attention: t={t}, {qkv.dtype}")
 if qk_scale>0.0:aa,ab=qk_scale,qk_scale*ATTN_HEAD_DIM
 else:aa,ab=dc,dS
 ap=cos.reshape(t,ATTN_HEAD_DIM//2);a2=sin.reshape(t,ATTN_HEAD_DIM//2);o3=c2["fwd"](qkv,ap,a2,aa,ab)[0];return o3.view(b,t,n_head,ATTN_HEAD_DIM)
def fused_attention_band(qkv,cos,sin,n_head,qk_scale=0.0):
 b,t,_=qkv.shape
 if qk_scale>0.0:aa,ab=qk_scale,qk_scale*ATTN_HEAD_DIM
 else:aa,ab=dc,dS
 ap=cos.reshape(t,ATTN_HEAD_DIM//2);a2=sin.reshape(t,ATTN_HEAD_DIM//2);o3=c2["band"](qkv,ap,a2,aa,ab)[0];return o3.view(b,t,n_head,ATTN_HEAD_DIM)
_LH_VCHUNK = 8192
dj = 2
_OS1_VCHUNK = 4096
c8 = 512
cQ = 128
_EB_VPP = 2
ad = 4
_EB_MOVING = 352
_EB_MOVING_W = 384
dI = False
ea = 256
try:
 if __name__=="__mp_main__":raise ImportError
 import nki;import nki.language as nl;import nki.isa as nisa;from torch_neuronx import nki_op;J=nisa.engine.vector
 def bX():
  if nl.program_ndim()>0:return nl.program_id(0),nl.num_programs(0)
  return 0,1
 def aO(d):a,l=bX();f,k=d//l,d%l;m=a*f+min(a,k);n=f+(1 if a<k else 0);return m,m+n
 def aH(d,a):return(nl.ndarray(d,dtype=a,buffer=nl.sbuf),nl.ndarray(d,dtype=a,buffer=nl.sbuf))
 def aM(a):
  D,V=a.shape;k=V//_OS1_VCHUNK;d=nl.ndarray((D,V),dtype=a.dtype,buffer=nl.sbuf)
  for c in range(k):l=c*_OS1_VCHUNK;f=l+_OS1_VCHUNK;nisa.dma_copy(dst=d[:,l:f],src=a[:,l:f])
  return d
 def aZ(a,v,n,d,D):
  P=a.shape[0];p=d+D;k=nl.ndarray((D,n,P),dtype=a.dtype,buffer=nl.sbuf)
  for j in range(n):i=v+j;f=nl.ndarray((P,D),dtype=a.dtype,buffer=nl.sbuf);nisa.dma_copy(dst=f,src=a[:,i,d:p]);l=nl.ndarray((D,P),dtype=a.dtype,buffer=nl.psum);nisa.nc_transpose(dst=l,data=f);nisa.tensor_copy(dst=k[:,j,:],src=l,engine=J)
  return k
 def bH(a,d,f):P=a.shape[0];n=f-d;l=nl.ndarray((P,n),dtype=nl.float32,buffer=nl.sbuf);nisa.dma_copy(dst=l,src=a[:,d:f]);return l
 def bI(f,o,xb,ut,L,d,p,j,c):
  P=f.shape[0];x=_OS1_VCHUNK//c8;S=j+1;z=d[:,j:S];l=c*_OS1_VCHUNK
  for m in range(x):a=m*c8;U=a+c8;v=l+a;v1=v+c8;k=nl.ndarray((P,c8),dtype=nl.float32,buffer=nl.psum);nisa.nc_matmul(dst=k,stationary=ut[:,j,:],moving=L[:,v:v1],accumulate=False);nisa.tensor_scalar(dst=o[:,a:U],data=k,op0=nl.subtract,operand0=p,op1=nl.maximum,operand1=0.0,engine=J)
  nisa.scalar_tensor_tensor(dst=f,data=o,op0=nl.multiply,operand0=z,op1=nl.add,operand1=xb)
 @nki.jit
 def _lh_os1_fwd_kernel(U,bc,b3,cl,a2,cc,a,ch,b2):
  P,NT,V=U.shape;K=bc.shape[0];D=a2.shape[0];f=V//_OS1_VCHUNK;l=nl.float32;ah=nl.ndarray((P,NT),dtype=l,buffer=nl.shared_hbm);aw=nl.ndarray((P,NT),dtype=l,buffer=nl.shared_hbm);d,o=aO(NT);n=o-d
  if n>0:
   ct=aM(a2);ut=aZ(cl,d,n,b2,D);bd=bH(cc,d,o);zc=nl.ndarray((P,n,f),dtype=l,buffer=nl.sbuf);ay=nl.ndarray((P,K,n),dtype=nl.uint32,buffer=nl.sbuf);az=nl.ndarray((P,K,f,n),dtype=l,buffer=nl.sbuf)
   for k in range(K):nisa.dma_copy(dst=ay[:,k,:],src=bc[k,:,d:o])
   ao=aH((P,_OS1_VCHUNK),U.dtype);be=aH((P,_OS1_VCHUNK),l);es=aH((P,_OS1_VCHUNK),l);ai=n*f;x0=ao[0];nisa.dma_copy(dst=x0,src=U[:,d,0:_OS1_VCHUNK])
   for s in range(ai):
    j=s//f;c=s%f;b=s%dj;xb=ao[b];L=be[b];e=es[b];S=s+1
    if S<ai:jn=S//f;cn=S%f;bn=S%dj;bg=d+jn;w0=cn*_OS1_VCHUNK;w1=w0+_OS1_VCHUNK;xn=ao[bn];nisa.dma_copy(dst=xn,src=U[:,bg,w0:w1])
    bI(e,L,xb,ut,ct,bd,ch,j,c);j1=j+1;c1=c+1
    for k in range(K):nisa.nc_n_gather(dst=az[:,k,c,j:j1],data=e,indices=ay[:,k,j:j1])
    nisa.activation(dst=L,op=nl.tanh,data=e,scale=1.0/a);nisa.activation_reduce(dst=L,op=nl.exp,data=L,reduce_op=nl.add,reduce_res=zc[:,j,c:c1],scale=a,bias=-a)
   zb=nl.ndarray((P,n),dtype=l,buffer=nl.sbuf);nisa.tensor_reduce(dst=zb,op=nl.add,data=zc,axis=2);al=nl.ndarray((P,n),dtype=l,buffer=nl.sbuf);nisa.activation(dst=al,op=nl.log,data=zb);nisa.dma_copy(dst=aw[:,d:o],src=al);ax=nl.ndarray((P,n),dtype=l,buffer=nl.sbuf);nisa.tensor_scalar(dst=ax,data=al,op0=nl.add,operand0=a);am=nl.ndarray((P,n),dtype=l,buffer=nl.sbuf);tb=nl.ndarray((P,n),dtype=l,buffer=nl.sbuf);x=nl.ndarray((P,n),dtype=l,buffer=nl.sbuf);z=nl.ndarray((P,n),dtype=l,buffer=nl.sbuf)
   for k in range(K):
    for c in range(f):
     nisa.dma_copy(dst=am,src=b3[k,c,:,d:o]);nisa.activation(dst=tb,op=nl.tanh,data=az[:,k,c,:],scale=1.0/a);nisa.scalar_tensor_tensor(dst=x,data=tb,op0=nl.multiply,operand0=-a,op1=nl.add,operand1=ax)
     if k==0 and c==0:nisa.tensor_tensor(dst=z,data1=x,data2=am,op=nl.multiply,engine=J)
     else:nisa.tensor_tensor(dst=x,data1=x,data2=am,op=nl.multiply,engine=J);nisa.tensor_tensor(dst=z,data1=z,data2=x,op=nl.add,engine=J)
   nisa.dma_copy(dst=ah[:,d:o],src=z)
  return ah,aw
 @nki.jit
 def _lh_os1_bwd_kernel(x3,b5,aa,cl,g2,cu,b3,ch,a,cs,b8):
  P,NT,V=x3.shape;K=b5.shape[0];D=b3.shape[0];m=V//_OS1_VCHUNK;f=nl.float32;d3=nl.ndarray((P,NT,V),dtype=x3.dtype,buffer=nl.shared_hbm);d,p=aO(NT);n=p-d
  if n>0:
   ct=aM(b3);ut=aZ(cu,d,n,b8,D);bi=bH(ch,d,p);U=nl.ndarray((P,K,m,n),dtype=f,buffer=nl.sbuf);l=nl.ndarray((P,K,n),dtype=f,buffer=nl.sbuf);ba=nl.ndarray((P,n),dtype=f,buffer=nl.sbuf)
   for k in range(K):
    nisa.dma_copy(dst=U[:,k,0,:],src=b5[k,:,d:p])
    for c in range(1,m):bl=-float(c*_OS1_VCHUNK);nisa.tensor_scalar(dst=U[:,k,c,:],data=U[:,k,0,:],op0=nl.add,operand0=bl)
    nisa.dma_copy(dst=l[:,k,:],src=aa[k,0,:,d:p])
    for c in range(1,m):nisa.dma_copy(dst=ba,src=aa[k,c,:,d:p]);nisa.tensor_tensor(dst=l[:,k,:],data1=l[:,k,:],data2=ba,op=nl.add,engine=J)
   x=nl.ndarray((P,n),dtype=f,buffer=nl.sbuf);nisa.dma_copy(dst=x,src=cl[:,d:p]);S=nl.ndarray((P,n),dtype=f,buffer=nl.sbuf);nisa.dma_copy(dst=S,src=g2[:,d:p]);nisa.tensor_tensor(dst=S,data1=S,data2=l[:,0,:],op=nl.multiply,engine=J);sq=nl.ndarray((P,n),dtype=f,buffer=nl.sbuf);nisa.activation(dst=sq,op=nl.sqrt,data=S);nisa.tensor_scalar(dst=x,data=x,op0=nl.multiply,operand0=-1.0,op1=nl.add,operand1=-a)
   if K>1:
    nisa.reciprocal(dst=l[:,0,:],data=l[:,0,:])
    for k in range(1,K):nisa.tensor_tensor(dst=l[:,k,:],data1=l[:,k,:],data2=l[:,0,:],op=nl.multiply,engine=J)
    z=nl.ndarray((P,n),dtype=f,buffer=nl.sbuf);nisa.tensor_scalar(dst=z,data=l[:,1,:],op0=nl.add,operand0=1.0)
    for k in range(2,K):nisa.tensor_tensor(dst=z,data1=z,data2=l[:,k,:],op=nl.add,engine=J)
    nisa.activation(dst=z,op=nl.log,data=z);nisa.tensor_tensor(dst=x,data1=x,data2=z,op=nl.add,engine=J)
   io=nl.ndarray((P,_OS1_VCHUNK),dtype=f,buffer=nl.sbuf);nisa.iota(dst=io,pattern=[[1,_OS1_VCHUNK]],offset=0,channel_multiplier=0);au=aH((P,_OS1_VCHUNK),x3.dtype);ts=aH((P,_OS1_VCHUNK),f);qs=aH((P,_OS1_VCHUNK),f);ai=n*m;x0=au[0];nisa.dma_copy(dst=x0,src=x3[:,d,0:_OS1_VCHUNK])
   for s in range(ai):
    j=s//m;c=s%m;i=d+j;b=s%dj;xb=au[b];t=ts[b];q=qs[b];sn=s+1
    if sn<ai:jn=sn//m;cn=sn%m;bn=sn%dj;bk=d+jn;w0=cn*_OS1_VCHUNK;w1=w0+_OS1_VCHUNK;xn=au[bn];nisa.dma_copy(dst=xn,src=x3[:,bk,w0:w1])
    bI(q,t,xb,ut,ct,bi,cs,j,c);v=j+1;nisa.activation(dst=t,op=nl.tanh,data=q,scale=1.0/a);nisa.activation(dst=q,op=nl.exp,data=t,scale=a,bias=x[:,j:v]);nisa.scalar_tensor_tensor(dst=q,data=io,op0=nl.equal,operand0=U[:,0,c,j:v],op1=nl.subtract,operand1=q)
    for k in range(1,K):nisa.tensor_scalar(dst=xb,data=io,op0=nl.equal,operand0=U[:,k,c,j:v]);nisa.scalar_tensor_tensor(dst=q,data=xb,op0=nl.multiply,operand0=l[:,k,j:v],op1=nl.add,operand1=q)
    nisa.activation(dst=t,op=nl.square,data=t,scale=sq[:,j:v]);nisa.scalar_tensor_tensor(dst=xb,data=t,op0=nl.subtract,operand0=S[:,j:v],op1=nl.multiply,operand1=q);w0=c*_OS1_VCHUNK;w1=w0+_OS1_VCHUNK;nisa.dma_copy(dst=d3[:,i,w0:w1],src=xb)
  return d3
 def Q(dst,src):nisa.dma_copy(dst=dst,src=src)
 def bG():
  if nl.program_ndim()>0:return nl.program_id(0),nl.num_programs(0)
  return 0,1
 def aY(P,W):l=nl.ndarray((P,W),dtype=nl.float32,buffer=nl.sbuf);nisa.iota(dst=l,pattern=[[1,W]],offset=0,channel_multiplier=0);return l
 def a6(a):
  if len(a.shape)==3:return a[:,:,0]
  return a
 def aD(n,d,a,f):
  if n==1:return(nl.ndarray(d,dtype=a,buffer=f),)
  if n==2:return(nl.ndarray(d,dtype=a,buffer=f),nl.ndarray(d,dtype=a,buffer=f))
  if n==3:return(nl.ndarray(d,dtype=a,buffer=f),nl.ndarray(d,dtype=a,buffer=f),nl.ndarray(d,dtype=a,buffer=f))
  if n==4:return(nl.ndarray(d,dtype=a,buffer=f),nl.ndarray(d,dtype=a,buffer=f),nl.ndarray(d,dtype=a,buffer=f),nl.ndarray(d,dtype=a,buffer=f))
  return(nl.ndarray(d,dtype=a,buffer=f),nl.ndarray(d,dtype=a,buffer=f),nl.ndarray(d,dtype=a,buffer=f),nl.ndarray(d,dtype=a,buffer=f),nl.ndarray(d,dtype=a,buffer=f),nl.ndarray(d,dtype=a,buffer=f),nl.ndarray(d,dtype=a,buffer=f),nl.ndarray(d,dtype=a,buffer=f))
 def bE(k,f,d,a,t,o):nisa.tensor_scalar(dst=k,data=f,op0=nl.equal,operand0=a[:,t:t+1])
 def bN(d,o,k,f,a):
  for v in range(_EB_VPP):
   for c in range(f):nisa.nc_matmul(dst=d[v*f+c],stationary=o[:,v*cQ:(v+1)*cQ],moving=k[:,c*_EB_MOVING:(c+1)*_EB_MOVING],accumulate=a,is_stationary_onezero=dI)
 def aX(a):return a
 def bF(m,a,d,C,f):
  for v in range(_EB_VPP):
   l=nl.ndarray((cQ,C),dtype=nl.float32,buffer=nl.sbuf)
   for c in range(d):nisa.tensor_copy(dst=l[:,c*_EB_MOVING:(c+1)*_EB_MOVING],src=a[v*d+c])
   Q(dst=m[f+v*cQ:f+(v+1)*cQ,:],src=l)
 @nki.jit
 def _eb_bwd_kernel(v,a,V):
  P,d,C=a.shape;f,ao=nl.float32,nl.bfloat16;U=nl.ndarray((V,C),dtype=f,buffer=nl.shared_hbm);ay,aq=bG();R=(V//P)//aq;ag=R//_EB_VPP;o=C//_EB_MOVING;am=_EB_VPP*o;io=aY(P,ea);p=nl.ndarray((P,d),dtype=f,buffer=nl.sbuf);Q(dst=p,src=a6(v))
  for ps in range(ag):
   z=(ay*R+ps*_EB_VPP)*P;S=nl.ndarray((P,d),dtype=f,buffer=nl.sbuf);nisa.tensor_scalar(dst=S,data=p,op0=nl.subtract,operand0=z*1.0);L=aD(am,(P,_EB_MOVING),f,nl.psum);l=aD(ad,(P,C),a.dtype,nl.sbuf);ax=aD(2,(P,ea),ao,nl.sbuf)
   for t in range(min(ad-1,d)):Q(dst=l[t%ad],src=a[:,t,:])
   for t in range(d):
    x=t+ad-1
    if x<d and ad>1:Q(dst=l[x%ad],src=a[:,x,:])
    if ad==1:Q(dst=l[0],src=a[:,t,:])
    oh=ax[t%2];bE(oh,io,p,S,t,z);bN(L,oh,l[t%ad],o,aX(t>0))
   bF(U,L,o,C,z)
  return U
 def bJ(f,p,m,k,a,d):
  for v in range(_EB_VPP):
   for c in range(k):nisa.nc_matmul(dst=f[v*k+c],stationary=p[:,v*cQ:(v+1)*cQ],moving=m[:,c*d:(c+1)*d],accumulate=a,is_stationary_onezero=dI)
 def bB(o,d,f,C,k,a):
  for v in range(_EB_VPP):
   l=nl.ndarray((cQ,C),dtype=nl.float32,buffer=nl.sbuf)
   for c in range(f):nisa.tensor_copy(dst=l[:,c*a:(c+1)*a],src=d[v*f+c])
   Q(dst=o[k+v*cQ:k+(v+1)*cQ,:],src=l)
 @nki.jit
 def _eb_bwd_w_kernel(x,a,V,d):
  P,f,C=a.shape;k,ao=nl.float32,nl.bfloat16;dw=nl.ndarray((V,C),dtype=k,buffer=nl.shared_hbm);ay,aq=bG();S=(V//P)//aq;ag=S//_EB_VPP;p=C//d;am=_EB_VPP*p;io=aY(P,ea);v=nl.ndarray((P,f),dtype=k,buffer=nl.sbuf);Q(dst=v,src=a6(x))
  for ps in range(ag):
   L=(ay*S+ps*_EB_VPP)*P;U=nl.ndarray((P,f),dtype=k,buffer=nl.sbuf);nisa.tensor_scalar(dst=U,data=v,op0=nl.subtract,operand0=L*1.0);R=aD(am,(P,d),k,nl.psum);l=aD(ad,(P,C),a.dtype,nl.sbuf);ax=aD(2,(P,ea),ao,nl.sbuf)
   for t in range(min(ad-1,f)):Q(dst=l[t%ad],src=a[:,t,:])
   for t in range(f):
    z=t+ad-1
    if z<f and ad>1:Q(dst=l[z%ad],src=a[:,z,:])
    if ad==1:Q(dst=l[0],src=a[:,t,:])
    oh=ax[t%2];bE(oh,io,v,U,t,L);bJ(R,oh,l[t%ad],p,aX(t>0),d)
   bB(dw,R,p,C,L,d)
  return dw
 _HAS_TRAINING_KERNELS=True;_TRAINING_KERNELS_ERROR=""
except Exception as exc:_HAS_TRAINING_KERNELS=False;_TRAINING_KERNELS_ERROR = f"{type(exc).__name__}: {exc}"
_TRAIN_OPS: dict = {}
LEAN_COMPACT_ROWS = 8192
def c6():
 if hasattr(torch.ops.fused_loss,"loss_head_os1_fwd"):return{"loss_fwd":torch.ops.fused_loss.loss_head_os1_fwd,"loss_bwd":torch.ops.fused_loss.loss_head_os1_bwd,"emb_grad":torch.ops.lean_embedding.compact_grad,"emb_grad_w":torch.ops.lean_embedding.compact_grad_w}
 lh_fwd=_lh_os1_fwd_kernel[2];lh_bwd=_lh_os1_bwd_kernel[2];eb_bwd=_eb_bwd_kernel[2];eb_bwd_w=_eb_bwd_w_kernel[2]
 @nki_op("fused_loss::loss_head_os1_fwd",mutates_args=())
 def loss_head_os1_fwd(x3:torch.Tensor,tl3:torch.Tensor,rwc4:torch.Tensor,xu3:torch.Tensor,ct2:torch.Tensor,cg2:torch.Tensor,softcap:float,thr:float)->tuple[torch.Tensor,torch.Tensor]:ab=int(xu3.shape[2])-int(ct2.shape[0]);aa,ap=lh_fwd(x3,tl3,rwc4,xu3,ct2,cg2,softcap,thr,ab);return aa,ap
 @nki_op("fused_loss::loss_head_os1_bwd",mutates_args=())
 def loss_head_os1_bwd(x3:torch.Tensor,tg3:torch.Tensor,rwc4:torch.Tensor,lz2:torch.Tensor,g2:torch.Tensor,xu3:torch.Tensor,ct2:torch.Tensor,cg2:torch.Tensor,softcap:float,thr:float)->torch.Tensor:aa=int(xu3.shape[2])-int(ct2.shape[0]);return lh_bwd(x3,tg3,rwc4,lz2,g2,xu3,ct2,cg2,softcap,thr,aa)
 @nki_op("lean_embedding::compact_grad",mutates_args=())
 def compact_grad(cid2:torch.Tensor,dy3:torch.Tensor)->torch.Tensor:return eb_bwd(cid2,dy3,LEAN_COMPACT_ROWS)
 @nki_op("lean_embedding::compact_grad_w",mutates_args=())
 def compact_grad_w(cid2:torch.Tensor,dy3:torch.Tensor)->torch.Tensor:return eb_bwd_w(cid2,dy3,LEAN_COMPACT_ROWS,lean_moving_width(dy3.shape[-1]))
 aa={"fused_loss::loss_head_os1_fwd":lambda x3,tl3,rwc4,xu3,ct2,cg2,softcap,thr:(torch.empty_like(x3[:,:,0],dtype=torch.float32,memory_format=torch.contiguous_format),torch.empty_like(x3[:,:,0],dtype=torch.float32,memory_format=torch.contiguous_format)),"fused_loss::loss_head_os1_bwd":lambda x3,tg3,rwc4,lz2,g2,xu3,ct2,cg2,softcap,thr:torch.empty_like(x3,memory_format=torch.contiguous_format),"lean_embedding::compact_grad":lambda cid2,dy3:dy3.new_empty((LEAN_COMPACT_ROWS,dy3.shape[-1]),dtype=torch.float32),"lean_embedding::compact_grad_w":lambda cid2,dy3:dy3.new_empty((LEAN_COMPACT_ROWS,dy3.shape[-1]),dtype=torch.float32),}
 for ap,ab in aa.items():
  try:
   torch.library.register_fake(ap)(ab)
  except RuntimeError:
   pass
 return{"loss_fwd":loss_head_os1_fwd,"loss_bwd":loss_head_os1_bwd,"emb_grad":compact_grad,"emb_grad_w":compact_grad_w}
def training_kernels_for(aa):
 if aa.type!="neuron":return False
 if os.environ.get("NEURON_LOGICAL_NC_CONFIG","2")!="2":return False
 if not _HAS_TRAINING_KERNELS:raise RuntimeError(f"the training kernels need nki and torch_neuronx, which a Neuron host provides ({_TRAINING_KERNELS_ERROR})")
 if not _TRAIN_OPS:_TRAIN_OPS.update(c6())
 return True
def loss_head_applies(x,loss_reduction,vocab_size,padded_vocab):return(TRAINING_KERNELS and x.dtype==torch.bfloat16 and loss_reduction=="mean" and vocab_size==padded_vocab and vocab_size%_LH_VCHUNK==0 and(x.size(0)*x.size(1))%128==0)
class Os1FoldHeadR(torch.autograd.Function):
 @staticmethod
 def forward(ctx,xu,w,cw,wcat,x2):ctx.save_for_backward(wcat,cw,x2);return F.linear(xu,wcat)
 @staticmethod
 def backward(ctx,d_lin):wf,cw,xs=ctx.saved_tensors;d_xu=d_lin.matmul(wf);d_w=d_lin.t().matmul(xs);return d_xu,d_w,None,None,None
OS1_WCAT_FN = None
@torch.no_grad()
def os1_wcat_into(w,cg,out):out.copy_(torch.cat([w.to(COMPUTE_DTYPE),cg],dim=1))
def set_os1_wcat_fn(fn):global OS1_WCAT_FN;OS1_WCAT_FN=fn
def os1_wcat_refresh(model):fn=OS1_WCAT_FN or os1_wcat_into;fn(model.lm_head.weight,model.os1_codeg,model.os1_wcat)
class ChunkedLossHeadOS1(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x3,tg3,tl3,rwc4,xu3,ct2,cg2,thr):loss2,lz2=_TRAIN_OPS["loss_fwd"](x3,tl3,rwc4,xu3,ct2,cg2,LOGIT_SOFTCAP,thr);ctx.save_for_backward(x3,tg3,rwc4,lz2,xu3,ct2,cg2);ctx.thr=thr;return loss2
 @staticmethod
 def backward(ctx,d_loss2):x3,tg3,rwc4,lz2,xu3,ct2,cg2=ctx.saved_tensors;d3=_TRAIN_OPS["loss_bwd"](x3,tg3,rwc4,lz2,d_loss2.contiguous(),xu3,ct2,cg2,LOGIT_SOFTCAP,ctx.thr);return d3,None,None,None,None,None,None,None
def chunked_loss_mean_os1(lin,targets,xu,ct2,cg,thr):n,v=lin.shape;nt=n//128;nc=v//_OS1_VCHUNK;y=targets.reshape(n);valid=y>=0;w=valid.to(torch.float32)/valid.sum().clamp_min(1).to(torch.float32);tg=y.clamp(0,v-1);chunk=torch.div(tg,_OS1_VCHUNK,rounding_mode="floor");rwc=torch.stack([w*(chunk==c).to(torch.float32)for c in range(nc)]);tl=tg-chunk*_OS1_VCHUNK;loss2=ChunkedLossHeadOS1.apply(lin.view(128,nt,v),tg.to(torch.int32).view(1,128,nt),tl.to(torch.int32).view(1,128,nt),rwc.view(1,nc,128,nt),xu.view(128,nt,xu.size(-1)),ct2,cg.reshape(128,nt),thr);return loss2.sum()
class LeanLookup(torch.autograd.Function):
 @staticmethod
 def forward(ctx,idx,weight,sink):return F.embedding(idx,weight).to(sink.dtype)
 @staticmethod
 def backward(ctx,dy):return None,None,dy
def lean_relabel(ids,vocab):
 uniq,inv=torch.unique(ids.reshape(-1),sorted=True,return_inverse=True);u=uniq.numel()
 if u>LEAN_COMPACT_ROWS:return None
 touched=torch.zeros(vocab,dtype=torch.bool);touched[uniq.long()]=True;pad=torch.nonzero(~touched).flatten()[:LEAN_COMPACT_ROWS-u];return inv.to(torch.int32),torch.cat([uniq.long(),pad])
def lean_embedding_grad(dys,cid,rows,vocab,kernel):
 c=dys[0].size(-1);dy=torch.cat([d.reshape(-1,c)for d in dys]);n=dy.size(0)
 if kernel:op=_TRAIN_OPS["emb_grad"]if c%_EB_MOVING==0 else _TRAIN_OPS["emb_grad_w"];dwc=op(cid.view(128,n//128),dy.view(128,n//128,c))
 else:dwc=torch.zeros(LEAN_COMPACT_ROWS,c,dtype=torch.float32,device=dy.device).index_add_(0,cid.long(),dy.float())
 return torch.zeros(vocab,c,dtype=torch.float32,device=dy.device).index_copy(0,rows,dwc)
def lean_host_grad(dys,ids,vocab):c=dys[0].size(-1);dy=torch.cat([d.reshape(-1,c)for d in dys]).to("cpu",torch.float32);grad=torch.zeros(vocab,c,dtype=torch.float32).index_add_(0,ids.reshape(-1).long(),dy);return grad.to(dys[0].device)
def lean_kernel_applies(tokens,n_embd):return(TRAINING_KERNELS and tokens%128==0 and n_embd%_EB_MOVING==0 and(n_embd//_EB_MOVING)*_EB_VPP in(1,2,4,8))or(TRAINING_KERNELS and tokens%128==0 and lean_moving_width(n_embd)>0)
def lean_moving_width(n_embd):
 for mov in(_EB_MOVING,_EB_MOVING_W):
  if n_embd%mov==0 and(n_embd//mov)*_EB_VPP in(1,2,4,8):return mov
 return 0
def set_lean_embedding_grad(optimizer,weight,grad):
 if hasattr(optimizer,"set_deferred_grad"):optimizer.set_deferred_grad(weight,grad)
 else:weight.grad=grad
MLP_TB = 1024
c5 = 1
eF = 0
ec = 3
eX = 1
an = 512
eh = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
c7 = 256
cF = 240.0
dM = 57344.0
cU = 4.0
em = 15.375
dY = 1.3
ei = 2.0
dU = 36.16
E_W = (-24, 40)
E_H = (-16, 8)
ek = (-40, 60)
cO,ef,eH,cI,dV=0,1,2,3,4
dA,d7,d0,ed=5,6,7,8
dZ,eJ,eP,eu,eK,e7,eW=9,10,11,12,13,14,15
NSC = 16
_HAS_MLP_KERNEL = False
_MLP_KERNEL_ERROR = None
try:
 if __name__=="__mp_main__":raise ImportError
 import nki;import nki.language as nl;import nki.isa as nisa;bA=nisa.engine.vector;bU=nisa.matmul_perf_mode.double_row;X=nl.float8_e4m3;aT=nl.float8_e5m2
 def _shard():
  if nl.program_ndim()>0:return nl.program_id(0),nl.num_programs(0)
  return 0,1
 def aC(n,a,d):
  k=n//d;m=n-k*d;l=a*k
  if a<m:l=l+a
  else:l=l+m
  f=k
  if a<m:f=f+1
  return l,l+f
 def A(a,b):return(a+b-1)//b
 def aR(w,K,a,c,d):
  l=nl.ndarray((_P,a,_P),dtype=d,buffer=nl.sbuf)
  for s in range(_P//16):nisa.dma_transpose(dst=l.ap([[a*_P,_P],[_P,a],[1,1],[1,16]],offset=s*16),src=w.ap([[K,16],[_P,a],[1,1],[1,_P]],offset=(c*_P+s*16)*K),dge_mode=nisa.dge_mode.hwdge)
  return l
 def aB(w,N,a,c,d):l=nl.ndarray((_P,a,_P),dtype=d,buffer=nl.sbuf);nisa.dma_copy(dst=l,src=w.ap([[N,_P],[_P*N,a],[1,_P]],offset=c*_P));return l
 def aU(a,d):
  l=a
  if l>an:l=an
  f=nl.ndarray((_P,l),dtype=d,buffer=nl.psum)
  if a<=an:return(f,)
  m=nl.ndarray((_P,a-an),dtype=d,buffer=nl.psum);return(f,m)
 def I(f,j,o,L,S):
  x=j*_P;U=x+_P
  for k in range(0,L,4):
   a=L-k
   if a>4:a=4
   v=nl.ndarray((_P,a*_P),dtype=S,buffer=nl.psum)
   for l in range(a):p=k+l;d=l*_P;m=d+_P;R=p*_P;s1=R+_P;nisa.nc_transpose(dst=v[:,d:m],data=o[:,R:s1])
   for l in range(a):p=k+l;d=l*_P;m=d+_P;nisa.tensor_copy(dst=f[:,p,x:U],src=v[:,d:m],engine=bA)
 def bu(f,N,d,m,a):nisa.dma_copy(dst=f.ap([[N,_P],[_P*N,m],[1,N]],offset=d*N),src=a)
 def a9(f,N,d,m,c,a):nisa.dma_copy(dst=f.ap([[N,_P],[_P*N,m],[1,_P]],offset=d*N+c*_P),src=a)
 def bC(a):s=nl.ndarray((_P,NSC),dtype=nl.float32,buffer=nl.sbuf);nisa.dma_copy(dst=s,src=a);return s
 def bD(d,k,a,f,m):l=f;o=l+1;nisa.activation(dst=d,op=nl.copy,data=k,scale=a[:,l:o])
 def aE(k,m,f,v,a):
  p=a
  if p>an:p=an
  for o in range(v):
   l=o*2;d=l+2;nisa.nc_matmul(dst=k[0],stationary=m[:,l:d,:],moving=f[:,l:d,0:p],accumulate=(o>0),perf_mode=bU)
   if a>an:nisa.nc_matmul(dst=k[1],stationary=m[:,l:d,:],moving=f[:,l:d,an:a],accumulate=(o>0),perf_mode=bU)
 def bP(m,L,hs,h8,f,z,a,d,R):
  p=ef;S=p+1;v=eH;b1=v+1;U=A(a,an)
  for x in range(U):
   k=x*an;l=k+an
   if l>a:l=a
   nisa.tensor_scalar(dst=m[:,k:l],data=z[x],op0=nl.multiply,operand0=d[:,p:S],op1=nl.maximum,operand1=0.0,engine=bA);nisa.tensor_scalar(dst=L[:,k:l],data=m[:,k:l],op0=nl.multiply,operand0=d[:,v:b1],op1=nl.minimum,operand1=em,engine=bA);nisa.activation(dst=h8[:,f,k:l],op=nl.square,data=L[:,k:l])
 def aJ(f,p,a,d,o):
  v=o;L=v+1;x=A(a,an)
  for m in range(x):
   k=m*an;l=k+an
   if l>a:l=a
   nisa.tensor_scalar(dst=f[:,k:l],data=p[m],op0=nl.multiply,operand0=d[:,v:L],engine=bA)
 def bO(d,m,p,a,f):
  v=d7;L=v+1;z=A(a,an)
  for o in range(z):
   k=o*an;l=k+an
   if l>a:l=a
   nisa.scalar_tensor_tensor(dst=d[:,k:l],data=p[o],op0=nl.multiply,operand0=f[:,v:L],op1=nl.multiply,operand1=m[:,k:l])
 def bV(k,f,a,p,d,m):l=d0;o=l+1;nisa.tensor_scalar(dst=k[:,f,:],data=a,op0=nl.multiply,operand0=d[:,l:o],engine=bA)
 def Z(d,m,t,c,a,x):l=t*m+c*_P*a*_P;k=nl.ndarray((_P,a,_P),dtype=nl.uint8,buffer=nl.sbuf);nisa.dma_copy(dst=k,src=d.ap([[a*_P,_P],[_P,a],[1,_P]],offset=l));return k.view(X);f=nl.ndarray((_P,a,_P),dtype=X,buffer=nl.sbuf);nisa.dma_copy(dst=f,src=d.ap([[a*_P,_P],[_P,a],[1,_P]],offset=l));return f
 def aF(p,o,t,c,a,v,f,k,d,R):l=t*o+c*_P*a*_P;m=nl.ndarray((_P,a,_P),dtype=nl.uint8,buffer=nl.sbuf);bD(m.view(X),v,f,k,d);nisa.dma_copy(dst=p.ap([[a*_P,_P],[_P,a],[1,_P]],offset=l),src=m)
 @nki.jit
 def _precast8_kernel(d,f,sc,b3,a,R):
  F=d.shape[0];D=d.shape[1];x=d.dtype;k=F*D;m=D//_P;o=F//_P;v=nl.ndarray((4,k),dtype=nl.uint8,buffer=nl.shared_hbm);l=bC(sc);al,ag=_shard();z,fb=aC(o,al,ag);L,jb=aC(m,al,ag)
  for i in range(fb-z):c=z+i;wt=aR(d,D,m,c,x);aF(v,k,0,c,m,wt,l,cO,a,R)
  for i in range(jb-L):c=L+i;wt=aR(f,F,o,c,x);aF(v,k,1,c,o,wt,l,cI,a,R)
  for i in range(fb-z):c=z+i;U=aB(f,F,m,c,x);aF(v,k,2,c,m,U,l,cI,a,R)
  for i in range(jb-L):c=L+i;U=aB(d,D,o,c,x);aF(v,k,3,c,o,U,l,cO,a,R)
  return v
 def aV(U,S,K,f,k,nt,v,d):
  for t in range(nt):
   L=k+t*_P;r1=L+_P;o=nl.ndarray((_P,K),dtype=v,buffer=nl.sbuf);nisa.dma_copy(dst=o,src=S[L:r1,0:K]);p=t*_P;c1=p+_P
   for a in range(0,f,aG):
    l=f-a
    if l>aG:l=aG
    m=nl.ndarray((_P,l,_P),dtype=v,buffer=nl.psum)
    for x in range(l):R=(a+x)*_P;s1=R+_P;nisa.nc_transpose(dst=m[:,x,:],data=o[:,R:s1])
    g1=a+l;nisa.activation(dst=U[:,a:g1,p:c1],op=nl.copy,data=m,scale=d)
 @nki.jit
 def _mlp8_fwd_pc_kernel(x,w8,sc,tb,v,u8):
  M=x.shape[0];D=x.shape[1];L=w8.shape[1];F=L//D;a=x.dtype;aq=nl.float32;y=nl.ndarray((M,D),dtype=a,buffer=nl.shared_hbm);rT=nl.ndarray((F,M),dtype=a,buffer=nl.shared_hbm);o=D//_P;R=F//_P;ba=D//c7;bb=F//c7;ag=bC(sc);bg,ax=_shard();t0,t1=aC(M//_P,bg,ax);al=tb//_P;aw=A(t1-t0,al)
  for b in range(aw):
   S=t0+b*al;U=S+al
   if U>t1:U=t1
   m=U-S;l=m*_P;p=S*_P;ay=p+l;x8=nl.ndarray((_P,o,l),dtype=X,buffer=nl.sbuf);aV(x8,x,D,o,p,m,a,cU);h8=nl.ndarray((_P,R,l),dtype=X,buffer=nl.sbuf)
   for f in range(R):f0=f*_P;f1=f0+_P;bk=Z(w8,L,0,f,o,u8);k=aU(l,aq);aE(k,bk,x8,ba,l);am=nl.ndarray((_P,l),dtype=a,buffer=nl.sbuf);rc=nl.ndarray((_P,l),dtype=a,buffer=nl.sbuf);hs=rc;bP(am,rc,hs,h8,f,k,l,ag,v);nisa.dma_copy(dst=rT[f0:f1,p:ay],src=am)
   af=nl.ndarray((_P,m,D),dtype=a,buffer=nl.sbuf)
   for j in range(o):bl=Z(w8,L,1,j,R,u8);k=aU(l,aq);aE(k,bl,h8,bb,l);ah=nl.ndarray((_P,l),dtype=a,buffer=nl.sbuf);aJ(ah,k,l,ag,dV);I(af,j,ah,m,a)
   bu(y,D,p,m,af)
  return y,rT
 @nki.jit
 def _mlp8_bwd_pc_kernel(o,rT,w8,sc,tb,k,a2,u8):
  M=o.shape[0];D=o.shape[1];F=rT.shape[0];FD=F*D;a=o.dtype;bd=nl.float32;dx=nl.ndarray((M,D),dtype=a,buffer=nl.shared_hbm)
  if k==1:S=nl.ndarray((F,M),dtype=a,buffer=nl.shared_hbm);R=nl.ndarray((F,M),dtype=a,buffer=nl.shared_hbm)
  else:S=nl.ndarray((M,F),dtype=a,buffer=nl.shared_hbm);R=nl.ndarray((M,F),dtype=a,buffer=nl.shared_hbm)
  L=D//_P;NF=F//_P;bp=D//c7;bq=F//c7;U=bC(sc);bw,bk=_shard();t0,t1=aC(M//_P,bw,bk);aw=tb//_P;bj=A(t1-t0,aw)
  for b in range(bj):
   ar=t0+b*aw;au=ar+aw
   if au>t1:au=t1
   d=au-ar;l=d*_P;m=ar*_P;aq=m+l;bb=nl.ndarray((_P,L,l),dtype=aT,buffer=nl.sbuf);q0=dA;q1=q0+1;aV(bb,o,D,L,m,d,a,U[:,q0:q1]);ba=nl.ndarray((_P,NF,l),dtype=aT,buffer=nl.sbuf)
   for f in range(NF):
    f0=f*_P;f1=f0+_P;cj=Z(w8,FD,2,f,L,u8);x=aU(l,bd);aE(x,cj,bb,bp,l);ao=nl.ndarray((_P,l),dtype=a,buffer=nl.sbuf);nisa.dma_copy(dst=ao,src=rT[f0:f1,m:aq]);v=nl.ndarray((_P,l),dtype=a,buffer=nl.sbuf);bO(v,ao,x,l,U);am=nl.ndarray((_P,l),dtype=a,buffer=nl.sbuf);nisa.activation(dst=am,op=nl.square,data=ao)
    if k==1:nisa.dma_copy(dst=S[f0:f1,m:aq],src=am);nisa.dma_copy(dst=R[f0:f1,m:aq],src=v)
    else:be=nl.ndarray((_P,d,_P),dtype=a,buffer=nl.sbuf);I(be,0,am,d,a);a9(S,F,m,d,f,be);av=nl.ndarray((_P,d,_P),dtype=a,buffer=nl.sbuf);I(av,0,v,d,a);a9(R,F,m,d,f,av)
    bt=v;bV(ba,f,v,bt,U,a2)
   aj=nl.ndarray((_P,d,D),dtype=a,buffer=nl.sbuf)
   for j in range(L):ce=Z(w8,FD,3,j,NF,u8);x=aU(l,bd);aE(x,ce,ba,bq,l);ak=nl.ndarray((_P,l),dtype=a,buffer=nl.sbuf);aJ(ak,x,l,U,ed);I(aj,j,ak,d,a)
   bu(dx,D,m,d,aj)
  return dx,S,R
 def aI(d,a):s=nl.ndarray((_P,a),dtype=nl.float32,buffer=nl.sbuf);nisa.dma_copy(dst=s,src=d);return s
 def aA(w,K,d,k,a,f):
  l=nl.ndarray((_P,d,a),dtype=f,buffer=nl.sbuf)
  for s in range(a//16):nisa.dma_transpose(dst=l.ap([[d*a,_P],[a,d],[1,1],[1,16]],offset=s*16),src=w.ap([[K,16],[_P,d],[1,1],[1,_P]],offset=(k+s*16)*K),dge_mode=nisa.dge_mode.hwdge)
  return l
 def bv(z,U,w8,d,cd,t,m,o,k,x,a,dt):
  L=nl.ndarray((_P,a),dtype=nl.float32,buffer=nl.psum)
  for f in range(o):p=t*m+f*2;j1=p+2;v=f*2;k1=v+2;nisa.nc_matmul(dst=L,stationary=U[:,p:j1,:],moving=w8[:,v:k1,:],accumulate=(f>0),perf_mode=bU)
  R=cd;q1=R+1;l=nl.ndarray((_P,a),dtype=dt,buffer=nl.sbuf);nisa.tensor_scalar(dst=l,data=L,op0=nl.multiply,operand0=d[:,R:q1],engine=bA);S=k+t*_P;r1=S+_P;o1=x+a;nisa.dma_copy(dst=z[S:r1,x:o1],src=l)
 def aW(k,f,d,a,p):m=nl.ndarray((_P,d,a),dtype=nl.uint8,buffer=nl.sbuf);nisa.dma_copy(dst=m,src=k.ap([[d*a,_P],[a,d],[1,a]],offset=f));return m.view(X);l=nl.ndarray((_P,d,a),dtype=X,buffer=nl.sbuf);nisa.dma_copy(dst=l,src=k.ap([[d*a,_P],[a,d],[1,a]],offset=f));return l
 def E(v,p,d,m,a,x,k,o,f,L):l=nl.ndarray((_P,m,a),dtype=nl.uint8,buffer=nl.sbuf);bD(l.view(X),x,k,o,f);nisa.dma_copy(dst=v.ap([[d,_P],[a,m],[1,a]],offset=p),src=l)
 @nki.jit
 def _attn8_precast_kernel(wq,wk,wv,wp,sc,cv,f,o,a,d,k,u8):
  D=wq.shape[1];DD=D*D;L=wq.dtype;l=D//_P;NN=3*l;S=nl.ndarray((8,DD),dtype=nl.uint8,buffer=nl.shared_hbm);ac=aI(sc,sc.shape[1]);ah,ae=_shard();ak=D//f;a0,a1=aC(3*ak,ah,ae)
  for i in range(a1-a0):
   c=a0+i;v=c//ak;n0=(c-v*ak)*f
   if v==0:U=aA(wq,D,l,n0,f,L)
   elif v==1:U=aA(wk,D,l,n0,f,L)
   else:U=aA(wv,D,l,n0,f,L)
   E(S,c*D*f,l*f,l,f,U,ac,cR,k,u8)
  aw=D//o;b0,b1=aC(aw,ah,ae)
  for i in range(b1-b0):c=b0+i;U=aA(wp,D,l,c*o,o,L);E(S,3*DD+c*D*o,l*o,l,o,U,ac,cD,k,u8)
  av=D//a;e0,e1=aC(3*av,ah,ae)
  for i in range(e1-e0):
   u=e0+i;c=u//3;v=u-c*3;R=c*a;x=nl.ndarray((_P,l,a),dtype=L,buffer=nl.sbuf)
   if v==0:nisa.dma_copy(dst=x,src=wq.ap([[D,_P],[_P*D,l],[1,a]],offset=R))
   elif v==1:nisa.dma_copy(dst=x,src=wk.ap([[D,_P],[_P*D,l],[1,a]],offset=R))
   else:nisa.dma_copy(dst=x,src=wv.ap([[D,_P],[_P*D,l],[1,a]],offset=R))
   be=4*DD+c*_P*NN*a+v*l*a;E(S,be,NN*a,l,a,x,ac,cR,k,u8)
  ax=D//d;f0,f1=aC(ax,ah,ae)
  for i in range(f1-f0):c=f0+i;R=c*d;x=nl.ndarray((_P,l,d),dtype=L,buffer=nl.sbuf);nisa.dma_copy(dst=x,src=wp.ap([[D,_P],[_P*D,l],[1,d]],offset=R));E(S,7*DD+c*D*d,l*d,l,d,x,ac,cD,k,u8)
  return S
 aG=4
 def aP(a8,a2,K,a,k,nt,v,d,p,U):
  for t in range(nt):
   R=k+t*_P;r1=R+_P;o=nl.ndarray((_P,K),dtype=v,buffer=nl.sbuf);nisa.dma_copy(dst=o,src=a2[R:r1,0:K])
   for f in range(0,a,aG):
    l=a-f
    if l>aG:l=aG
    m=nl.ndarray((_P,l,_P),dtype=v,buffer=nl.psum)
    for x in range(l):S=(f+x)*_P;s1=S+_P;nisa.nc_transpose(dst=m[:,x,:],data=o[:,S:s1])
    z=t*a+f;j1=z+l;bD(a8[:,z:j1,:],m,d,p,0)
 def bM(bc,f,w8,sc,nw,v,ca,cd,tb,a,k,u8):
  M=f.shape[0];K=f.shape[1];L=w8.shape[1];Nw=L//K;dt=f.dtype;d=K//_P;ao=K//c7;NC=(Nw//a)*nw;l=aI(sc,sc.shape[1]);au,ai=_shard();t0,z=aC(M//_P,au,ai);p=tb//_P;ah=A(z-t0,p)
  for b in range(ah):
   m=t0+b*p;o=m+p
   if o>z:o=z
   x=o-m;U=m*_P;S=nl.ndarray((_P,x*d,_P),dtype=X,buffer=nl.sbuf);aP(S,f,K,d,U,x,dt,l,ca,k)
   for c in range(NC):
    aw=aW(w8,v*L+c*K*a,d,a,u8);o0=c*a
    for t in range(x):bv(bc,S,aw,l,cd,t,d,ao,U,o0,a,dt)
 def bL(dx,k,w8,sc,nw,v,ca,cd,tb,a,f,u8):
  M=k.shape[0];N=k.shape[1];Nw=N//nw;L=w8.shape[1];K=L//Nw;S=k.dtype;d=N//_P;am=N//c7;NC=K//a;l=aI(sc,sc.shape[1]);ar,ai=_shard();t0,z=aC(M//_P,ar,ai);p=tb//_P;ah=A(z-t0,p)
  for b in range(ah):
   m=t0+b*p;o=m+p
   if o>z:o=z
   x=o-m;R=m*_P;U=nl.ndarray((_P,x*d,_P),dtype=aT,buffer=nl.sbuf);aP(U,k,N,d,R,x,S,l,ca,f)
   for c in range(NC):
    k0=c*a;av=aW(w8,v*L+c*_P*d*a,d,a,u8)
    for t in range(x):bv(dx,U,av,l,cd,t,d,am,R,k0,a,S)
 @nki.jit
 def _proj8_fwd_tt_kernel(a,o,z,m,f,v,x,L,k,d,R):p=m*(o.shape[1]//a.shape[1]);l=nl.ndarray((a.shape[0],p),dtype=a.dtype,buffer=nl.shared_hbm);bM(l,a,o,z,m,f,v,x,L,k,d,R);return l
 @nki.jit
 def _proj8_bwd_tt_kernel(d,p,z,o,f,v,x,L,k,a,R):l=p.shape[1]//(d.shape[1]//o);m=nl.ndarray((d.shape[0],l),dtype=d.dtype,buffer=nl.shared_hbm);bL(m,d,p,z,o,f,v,x,L,k,a,R);return m
 def bZ(f,a,d,k,m):l=k;o=l+1;nisa.tensor_scalar(dst=a,data=a,op0=nl.multiply,operand0=d[:,l:o],op1=nl.minimum,operand1=cF,engine=bA);nisa.activation(dst=f,op=nl.copy,data=a,scale=1.0)
 def aL(f,m,d,k,o,a):
  if o==2:bZ(f,m,d,k,a)
  else:bD(f,m,d,k,a)
 def bR(a,d):
  k=a
  if k>an:k=an
  l=nl.ndarray((_P,k),dtype=d,buffer=nl.psum)
  if a<=an:return(l,)
  m=a-an
  if m>an:m=an
  f=nl.ndarray((_P,m),dtype=d,buffer=nl.psum)
  if a<=2*an:return(l,f)
  o=a-2*an
  if o>an:o=an
  p=nl.ndarray((_P,o),dtype=d,buffer=nl.psum)
  if a<=3*an:return(l,f,p)
  z=a-3*an;x=nl.ndarray((_P,z),dtype=d,buffer=nl.psum);return(l,f,p,x)
 def bW(b6,k,R,sc,fg,fa,d):
  M=k.shape[0];O=k.shape[1];D=R.shape[1];x=k.dtype;ax=nl.float32;a=M//_P;aq=M//c7;ar=O//_P;bb=A(D,an);l=aI(sc,eY);bh,ao=_shard();b0,b1=aC(ar,bh,ao)
  if fa==1:p=nl.ndarray((_P,a,D),dtype=aT,buffer=nl.sbuf)
  else:p=nl.ndarray((_P,a,D),dtype=X,buffer=nl.sbuf)
  ba=A(a,AB)
  for ab in range(ba):
   m=ab*AB;o=m+AB
   if o>a:o=a
   nt=o-m;at=nl.ndarray((_P,nt,D),dtype=x,buffer=nl.sbuf);nisa.dma_copy(dst=at,src=R.ap([[D,_P],[_P*D,nt],[1,D]],offset=m*_P*D));aL(p[:,m:o,:],at,l,eD,fa,d)
  for ob in range(b0,b1):
   o0=ob*_P;o1=o0+_P;gt=aB(k,O,a,ob,x)
   if fg==1:z=nl.ndarray((_P,a,_P),dtype=aT,buffer=nl.sbuf)
   else:z=nl.ndarray((_P,a,_P),dtype=X,buffer=nl.sbuf)
   aL(z,gt,l,eE,fg,d);U=bR(D,ax)
   for kd in range(aq):
    L=kd*2;j1=L+2
    for ci in range(bb):
     c0=ci*an;v=c0+an
     if v>D:v=D
     nisa.nc_matmul(dst=U[ci],stationary=z[:,L:j1,:],moving=p[:,L:j1,c0:v],accumulate=(kd>0),perf_mode=bU)
   S=nl.ndarray((_P,D),dtype=x,buffer=nl.sbuf);aJ(S,U,D,l,eS);nisa.dma_copy(dst=b6[o0:o1,:],src=S)
 @nki.jit
 def _dw8_kernel(d,f,o,m,k,a):l=nl.ndarray((d.shape[1],f.shape[1]),dtype=d.dtype,buffer=nl.shared_hbm);bW(l,d,f,o,m,k,a);return l
 def bS(b7,k,S,sc,fg,fa,f,a):
  M=k.shape[0];O=k.shape[1];D=S.shape[1];z=k.dtype;ay=nl.float32;d=M//_P;ar=M//c7;au=O//_P;NG=D//a;be=A(a,an);bd=A(d,AB);l=aI(sc,eY);bj,aq=_shard();q0,q1=aC(NG,bj,aq)
  for gi in range(q0,q1):
   v=gi*a;az=v+a
   if fa==1:p=nl.ndarray((_P,d,a),dtype=aT,buffer=nl.sbuf)
   else:p=nl.ndarray((_P,d,a),dtype=X,buffer=nl.sbuf)
   for ab in range(bd):
    m=ab*AB;o=m+AB
    if o>d:o=d
    nt=o-m;at=nl.ndarray((_P,nt,a),dtype=z,buffer=nl.sbuf);aj=m*_P*D+v;nisa.dma_copy(dst=at,src=S.ap([[D,_P],[_P*D,nt],[1,a]],offset=aj));aL(p[:,m:o,:],at,l,eD,fa,f)
   for ob in range(au):
    o0=ob*_P;o1=o0+_P;gt=aB(k,O,d,ob,z)
    if fg==1:L=nl.ndarray((_P,d,_P),dtype=aT,buffer=nl.sbuf)
    else:L=nl.ndarray((_P,d,_P),dtype=X,buffer=nl.sbuf)
    aL(L,gt,l,eE,fg,f);ah=bR(a,ay)
    for kd in range(ar):
     R=kd*2;j1=R+2
     for ci in range(be):
      c0=ci*an;x=c0+an
      if x>a:x=a
      nisa.nc_matmul(dst=ah[ci],stationary=L[:,R:j1,:],moving=p[:,R:j1,c0:x],accumulate=(kd>0),perf_mode=bU)
    U=nl.ndarray((_P,a),dtype=z,buffer=nl.sbuf);aJ(U,ah,a,l,eS);nisa.dma_copy(dst=b7[o0:o1,v:az],src=U)
 @nki.jit
 def _dw8w_kernel(d,f,p,m,k,a,o):l=nl.ndarray((d.shape[1],f.shape[1]),dtype=d.dtype,buffer=nl.shared_hbm);bS(l,d,f,p,m,k,a,o);return l
 _HAS_MLP_KERNEL=True
except Exception as exc:_MLP_KERNEL_ERROR = f"{type(exc).__name__}: {exc}"
def cr(aa):return aa[eh]if eh==2 else aa
def dH(m,d,f,tb=MLP_TB):return(m%_P==0 and d%c7==0 and f%c7==0 and m>0 and tb%_P==0 and 0<tb<=1024)
def cC(v,a8,ab,a2):
 v=v.float();s=torch.full_like(v,2.0**ab);ap=torch.full_like(v,2.0**-ab);bc=a2-ab
 for b in(64,32,16,8,4,2,1):
  if b>bc:continue
  aa=s*(2.0**b);ok=(v*aa<=a8)&(aa<=2.0**a2);s=torch.where(ok,aa,s);ap=torch.where(ok,ap*(2.0**-b),ap)
 return s,ap
def eQ(v,bc,aa,a2):
 v=v.float();s=torch.full_like(v,4.0**aa);ap=torch.full_like(v,4.0**-aa);c=torch.full_like(v,2.0**aa);ic=torch.full_like(v,2.0**-aa);br=a2-aa
 for b in(32,16,8,4,2,1):
  if b>br:continue
  ab=s*(4.0**b);ok=(v*ab<=bc)&(ab<=4.0**a2);s=torch.where(ok,ab,s);ap=torch.where(ok,ap*(4.0**-b),ap);c=torch.where(ok,c*(2.0**b),c);ic=torch.where(ok,ic*(2.0**-b),ic)
 return s,ap,c,ic
def d6(br,bx):wf=br.detach().to(torch.bfloat16).float();wp=bx.detach().to(torch.bfloat16).float();aa=wf.abs().amax();ab=wp.abs().amax();by,a2=cC(aa,cF,E_W[0],E_W[1]);b2,a8=cC(ab,cF,E_W[0],E_W[1]);bf=(wf*wf).sum(dim=1).amax();bc=(wp*wp).sum(dim=0).amax();b8,b7,b6,ca=eQ(bf*(dY*br.shape[1]),cF,E_H[0],E_H[1]);ap=torch.zeros_like(aa);return torch.stack([by,a2*(1.0/cU),b6,b2,b7*a8,ap,ap,ap,ap,bf,bc,a2,a8,b8,aa,ab])
def cE(a2,ab):aa=a2.detach().abs().amax().float();br,bc=cC(aa,dM/ei,ek[0],ek[1]);ap=torch.sqrt(ab[dZ]*ab[eJ])*aa*(dU*a2.shape[1]);bf,a8=cC(ap,dM,ek[0],ek[1]);bx=torch.stack([br,bc*ab[eu]*2.0,bf,a8*ab[eP]]);return torch.cat([ab[:dA],bx,ab[dZ:]])
def dB(aa):return aa.float().reshape(1,NSC).expand(_P,NSC).contiguous()
def q8(t,aa):return t.float().to(torch.float8_e4m3fn if aa==0 else torch.float8_e5m2).float()
def cW(x2,a2,ab,aa):dt=x2.dtype;x8=q8(x2.float()*cU,0);w8=q8(a2.float()*aa[cO],0);r=torch.clamp_min((x8@w8.t())*aa[ef],0.0).to(dt);rc=torch.clamp_max(r.float()*aa[eH],em).to(dt);h8=q8(rc.float()*rc.float(),0);bc=q8(ab.float()*aa[cI],0);y=((h8@bc.t())*aa[dV]).to(dt);return y,r.t().contiguous()
def cT(a8,rT,bf,a2,aa,ap):
 dt=a8.dtype;by=q8(a8.float()*aa[dA],1);b3=q8(a2.float()*aa[cI],0);r=rT.t().float();ab=(((by@b3)*aa[d7])*r).to(dt);h=(r*r).to(dt);br=q8(ab.float()*aa[d0],1);b2=q8(bf.float()*aa[cO],0);dx=((br@b2)*aa[ed]).to(dt)
 if ap==1:return dx,h.t().contiguous(),ab.t().contiguous()
 return dx,h.contiguous(),ab.contiguous()
def b9(t):return t.float().to(torch.float8_e4m3fn).view(torch.uint8)
def ej(ap,a2,bf,ab=3):
 F,D=ap.shape;nd,nf=D//_P,F//_P;wf=ap.to(torch.bfloat16).float()*bf[cO];wp=a2.to(torch.bfloat16).float()*bf[cI];aa=torch.zeros(4,F*D,dtype=torch.uint8)
 if ab%2==1:t0=wf.reshape(nf,_P,nd,_P).permute(0,3,2,1);aa[0]=b9(t0.contiguous()).reshape(-1);t1=wp.reshape(nd,_P,nf,_P).permute(0,3,2,1);aa[1]=b9(t1.contiguous()).reshape(-1)
 if ab>=2:t2=wp.reshape(nd,_P,nf,_P).permute(2,1,0,3);aa[2]=b9(t2.contiguous()).reshape(-1);t3=wf.reshape(nf,_P,nd,_P).permute(2,1,0,3);aa[3]=b9(t3.contiguous()).reshape(-1)
 return aa
def cY(a2,x2,ab,aa,ap):
 if ap==1:return aa.mm(x2),ab.mm(a2).t()
 return aa.t().mm(x2),a2.t().mm(ab)
class EmulatedFusedMlp8(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x2,w_fc,w_proj,sc):y,rT=cW(x2,w_fc,w_proj,sc[0]);ctx.save_for_backward(x2,rT,w_fc,w_proj,sc);return y
 @staticmethod
 def backward(ctx,dy2):x2,rT,bc,ap,sc=ctx.saved_tensors;dy2=dy2.contiguous();dx,a8,ab=cT(dy2,rT,bc,ap,cE(dy2,sc[0]),c5);a2,aa=cY(dy2,x2,a8,ab,c5);return dx,a2,aa,None
def cX(by,x2,bc,ab,bx):ap=bx[0];aa=ap[d0];br=torch.full_like(aa,cU);a8=dP(ab,x2,aa,cP(aa),br,cP(br),cG,df);a2,bf=ap[dA],ap[eK];return a8,dP(by,bc,a2,cP(a2),bf,cP(bf),cG,cN)
class dr(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x2,w_fc,w_proj,sc):y,rT=cW(x2,w_fc,w_proj,sc[0]);ctx.save_for_backward(x2,rT,w_fc,w_proj,sc);return y
 @staticmethod
 def backward(ctx,dy2):x2,rT,bf,a2,sc=ctx.saved_tensors;dy2=dy2.contiguous();ap=dB(cE(dy2,sc[0]));dx,bc,ab=cT(dy2,rT,bf,a2,ap[0],0);a8,aa=cX(dy2,x2,bc,ab,ap);return dx,a8,aa,None
MLP_READY = False
MLP_OPS: dict = {}
dg: dict = {}
cH = {"fwd_bodies": 0, "bwd_bodies": 0, "precast_bodies": 0}
MLP_SCALE_FN = None
MLP_PRECAST_FN = None
def dD():
 global MLP_OPS,dg;from torch_neuronx import nki_op as make_op
 if hasattr(torch.ops.fused_mlp8pc,"fwd"):MLP_OPS={"fwd":torch.ops.fused_mlp8pc.fwd,"bwd":torch.ops.fused_mlp8pc.bwd,"precast":torch.ops.fused_mlp8pc.precast};dg={"fwd":torch.ops.fused_mlp8k.fwd,"bwd":torch.ops.fused_mlp8k.bwd};return("custom ops fused_mlp8pc::fwd / bwd / precast and fused_mlp8k::fwd / bwd (registered earlier in this " "process)")
 fwd_kernel=cr(_mlp8_fwd_pc_kernel);bwd_kernel=cr(_mlp8_bwd_pc_kernel);pre_kernel=cr(_precast8_kernel);tb,dw_mode,forms,need,u8=MLP_TB,c5,eF,ec,eX
 @make_op("fused_mlp8pc::fwd",mutates_args=())
 def mlp8_fwd(x2:torch.Tensor,w_fc:torch.Tensor,w_proj:torch.Tensor,sc:torch.Tensor,w8:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:cH["fwd_bodies"]+=1;y,rT=fwd_kernel(x2,w8,sc,tb,forms,u8);return y,rT
 @make_op("fused_mlp8pc::bwd",mutates_args=())
 def mlp8_bwd(dy2:torch.Tensor,rT:torch.Tensor,x2:torch.Tensor,w_fc:torch.Tensor,w_proj:torch.Tensor,sc:torch.Tensor,w8:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:cH["bwd_bodies"]+=1;dx,ab,aa=bwd_kernel(dy2,rT,w8,sc,tb,dw_mode,forms,u8);return dx,ab,aa
 @make_op("fused_mlp8pc::precast",mutates_args=())
 def mlp8_precast(w_fc:torch.Tensor,w_proj:torch.Tensor,sc:torch.Tensor)->torch.Tensor:cH["precast_bodies"]+=1;return pre_kernel(w_fc,w_proj,sc,need,forms,u8)
 def mlp8_fwd_fake(x2,w_fc,w_proj,sc,w8):return x2.new_empty(x2.shape[0],x2.shape[1]),x2.new_empty(w_fc.shape[0],x2.shape[0])
 def mlp8_bwd_fake(dy2,rT,x2,w_fc,w_proj,sc,w8):m,f=dy2.shape[0],rT.shape[0];return dy2.new_empty(m,dy2.shape[1]),dy2.new_empty(f,m),dy2.new_empty(f,m)
 def mlp8_precast_fake(w_fc,w_proj,sc):return w_fc.new_empty(4,w_fc.shape[0]*w_fc.shape[1],dtype=torch.uint8)
 aa=[]
 for ap,ab in(("fused_mlp8pc::fwd",mlp8_fwd_fake),("fused_mlp8pc::bwd",mlp8_bwd_fake),("fused_mlp8pc::precast",mlp8_precast_fake)):
  try:
   torch.library.register_fake(ap)(ab);aa.append("ours")
  except Exception as err:aa.append(f"the decorator's ({type(err).__name__})")
 def setup_context(ctx,inputs,output):x2,ab,aa,sc,w8=inputs;_y,rT=output;ctx.save_for_backward(x2,rT,ab,aa,sc,w8);ctx.mark_non_differentiable(rT);ctx.set_materialize_grads(False)
 def backward(ctx,dy2,_d_rT):x2,rT,bf,ap,sc,w8=ctx.saved_tensors;dy2=dy2.contiguous();bc=dB(cE(dy2,sc[0]));dx,a8,ab=mlp8_bwd(dy2,rT,x2,bf,ap,bc,w8);a2,aa=cY(dy2,x2,a8,ab,dw_mode);return dx,a2,aa,None,None
 torch.library.register_autograd("fused_mlp8pc::fwd",backward,setup_context=setup_context);MLP_OPS={"fwd":mlp8_fwd,"bwd":mlp8_bwd,"precast":mlp8_precast}
 @make_op("fused_mlp8k::fwd",mutates_args=())
 def mlp8k_fwd(x2:torch.Tensor,w_fc:torch.Tensor,w_proj:torch.Tensor,sc:torch.Tensor,w8:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:cH["fwd_bodies"]+=1;y,rT=fwd_kernel(x2,w8,sc,tb,forms,u8);return y,rT
 @make_op("fused_mlp8k::bwd",mutates_args=())
 def mlp8k_bwd(dy2:torch.Tensor,rT:torch.Tensor,sc:torch.Tensor,w8:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:cH["bwd_bodies"]+=1;dx,ab,aa=bwd_kernel(dy2,rT,w8,sc,tb,0,forms,u8);return dx,ab,aa
 def mlp8k_bwd_fake(dy2,rT,sc,w8):m,f=dy2.shape[0],rT.shape[0];return dy2.new_empty(m,dy2.shape[1]),dy2.new_empty(m,f),dy2.new_empty(m,f)
 for ap,ab in(("fused_mlp8k::fwd",mlp8_fwd_fake),("fused_mlp8k::bwd",mlp8k_bwd_fake)):
  try:
   torch.library.register_fake(ap)(ab);aa.append("ours")
  except Exception as err:aa.append(f"the decorator's ({type(err).__name__})")
 def setup_context_k(ctx,inputs,output):x2,ap,aa,sc,w8=inputs;_y,rT=output;ctx.save_for_backward(x2,rT,sc,w8);ctx.mark_non_differentiable(rT);ctx.set_materialize_grads(False)
 def backward_k(ctx,dy2,_d_rT):x2,rT,sc,w8=ctx.saved_tensors;dy2=dy2.contiguous();ap=dB(cE(dy2,sc[0]));dx,a8,ab=mlp8k_bwd(dy2,rT,ap,w8);a2,aa=cX(dy2,x2,a8,ab,ap);return dx,a2,aa,None,None
 torch.library.register_autograd("fused_mlp8k::fwd",backward_k,setup_context=setup_context_k);dg={"fwd":mlp8k_fwd,"bwd":mlp8k_bwd};return (f"custom ops fused_mlp8pc::fwd / bwd / precast by nki_op, fake impls {aa[0]} / {aa[1]} / {aa[2]}, autograd by torch.library.register_autograd; fp8=both forms=0 tb={tb} dw=fmajor precast=1 need={need} u8={u8}; the dense MLPs' custom ops fused_mlp8k::fwd / bwd, fake impls {aa[3]} / {aa[4]}: the backward in token mode, c_fc's weight gradient on the fp8 weight-gradient kernel")
def setup_mlp_kernel(aa):
 global MLP_READY;ab=("the inline nki_mlp_fused8.py (md5 18369d38) and nki_mlp_l4t.py (md5 c36b6972), fp8=both forms=0 tb=1024 " "dw=fmajor precast=1")
 if aa.type!="neuron":return f"emulation (not a Neuron device): {ab}"
 if not _HAS_MLP_KERNEL:raise RuntimeError(f"the fp8 MLP needs nki on a Neuron device ({_MLP_KERNEL_ERROR})")
 if MLP_READY:return "on (registered earlier in this process)"
 ap=dD();MLP_READY=True;return f"on: {ap}; {ab}"
def mlp_fused_applies(x2,aa):return(MLP_READY and x2.device.type=="neuron" and x2.dtype==torch.bfloat16 and dH(128,x2.size(1),aa,MLP_TB))
def mlp_fused(aa,x):
 b,t,c=x.shape;x2=x.reshape(b*t,c).contiguous();ab=aa.c_fc.weight.to(dtype=x.dtype);a2=aa.c_proj.weight.to(dtype=x.dtype);m=x2.size(0)
 if mlp_fused_applies(x2,ab.size(0)):
  a8=(-m)%128
  if a8:x2=F.pad(x2,(0,0,0,a8))
  y,br=(MLP_OPS if isinstance(aa,MoeShared)else dg)["fwd"](x2,ab,a2,aa.mlp8_sc,aa.mlp8_w8)
  if a8:y=y[:m]
 elif x2.device.type!="neuron":y=(EmulatedFusedMlp8 if isinstance(aa,MoeShared)else dr).apply(x2,ab,a2,aa.mlp8_sc)
 else:raise RuntimeError(f"fp8 MLP: rows {tuple(x2.shape)} {x2.dtype}")
 return y.view(b,t,c)
@torch.no_grad()
def mlp_scales_into(w_fc_list,w_proj_list,sc_list):
 for ab,aa,sc in zip(w_fc_list,w_proj_list,sc_list):sc.copy_(dB(d6(ab,aa)))
@torch.no_grad()
def mlp_precast_into(w_fc_list,w_proj_list,sc_list,w8_list):
 for ap,aa,sc,w8 in zip(w_fc_list,w_proj_list,sc_list,w8_list):
  wf,wp=ap.to(torch.bfloat16),aa.to(torch.bfloat16)
  if MLP_READY and w8.device.type=="neuron":w8.copy_(MLP_OPS["precast"](wf,wp,sc))
  else:w8.copy_(ej(wf,wp,sc[0].float(),ec))
def mlp_refresh_scales(aa):
 units=mlp8_units(aa);fn=MLP_SCALE_FN or mlp_scales_into;fn([u.c_fc.weight for u in units],[u.c_proj.weight for u in units],[u.mlp8_sc for u in units]);ab=MLP_PRECAST_FN or mlp_precast_into;ab([u.c_fc.weight for u in units],[u.c_proj.weight for u in units],[u.mlp8_sc for u in units],[u.mlp8_w8 for u in units]);moe=moe_layers_of(aa);ep=[u for u in moe if u.ep]
 if ep:fn([u.w_fc[0]for u in ep],[u.w_proj[0]for u in ep],[u.exp8_sc for u in ep]);ab([u.w_fc[0]for u in ep],[u.w_proj[0]for u in ep],[u.exp8_sc for u in ep],[u.exp8_w8 for u in ep])
 for u in moe:
  if not u.ep:mlp_scales_into(u.w_fc,u.w_proj,u.exp8_sc)
def mlp_scales_line(ab):
 units=mlp8_units(ab);rows=torch.stack([u.mlp8_sc[0].detach().float().cpu()for u in units])
 def log2s(col):return[int(v)for v in torch.log2(rows[:,col]).round().tolist()]
 br=[round(v,3)for v in rows[:,dZ].sqrt().tolist()];w=units[0].c_fc.weight;nd,nf,tb=w.shape[1]//_P,w.shape[0]//_P,MLP_TB;ap=2*(nd*_P*3+nf*_P*3);bf=nd*tb*3+nf*tb+ap+2*tb*4+2*tb+(tb//_P)*w.shape[1]*2;bc=nd*tb*3+nf*tb+ap+2*tb*6+4*(tb//_P)*_P+2*tb+(tb//_P)*w.shape[1]*2;return (f"log2_s_wfc={log2s(cO)} log2_s_wproj={log2s(cI)} log2_s_h={log2s(eK)} rho_fc={br} sbuf_kib={bf / 1024.0:.0f}/{bc / 1024.0:.0f}")
def mlp_summary():return " ".join(f"{k}={v}" for k, v in cH.items()) + (" registered" if MLP_READY else " not registered")
c4 = 0
eb = 2048
dR = 3
eO = 1
d1 = 1.3
et = 2.0
dh = 168.0
ey = (2048, 1024, 512, 256, 128)
er = (512, 256, 128)
eg,cR,d8,dW,cD,dl=0,1,2,3,4,5
eI,d9,e9,eZ,eN,e3,e8=6,7,8,9,10,11,12
dT,fe,eq=0,1,2
eG = 8
dz,dO,dv,dN=0,3,4,7
def eR(nk,bf,tb,aa,bc=0):nt=tb//_P;ap=nt*nk*_P*3;a8=2*nk*aa*3;ab=2*aa*2;return(ap+a8+ab+NSC*4)/1024.0
def plan(nk,n_out,forms=0,tb_max=2048):
 for tb in ey:
  if tb>tb_max:continue
  for aa in er:
   if n_out%aa==0 and eR(nk,n_out,tb,aa,forms)<=dh:return tb,aa
 raise ValueError(f"fp8 attention projection planner: no (tb, cwid) fits {dh} KiB at {nk} slabs, {n_out} columns")
def eV(aa,d,vb):return d1*(math.sqrt(d)*torch.sqrt(aa)+vb)
def dk(wq,wk,wv,wp,vb):f=[w.detach().to(torch.bfloat16).float()for w in(wq,wk,wv,wp)];aa=torch.stack([f[0].abs().amax(),f[1].abs().amax(),f[2].abs().amax()]).amax();ab=f[3].abs().amax();bf,ap=cC(aa,cF,E_W[0],E_W[1]);by,bc=cC(ab,cF,E_W[0],E_W[1]);a2=(f[2]*f[2]).sum(dim=1).amax();bx=vb.detach().float().reshape(());br=eV(a2,wq.shape[1],bx);b6,b3=cC(br,cF,E_W[0],E_W[1]);a8=torch.zeros_like(aa);b5=torch.full_like(aa,cU);return torch.stack([b5,bf,ap*(1.0/cU),b6,by,b3*bc,ap,bc,a2,aa,ab,br,bx,a8,a8,a8])
def cS(aa):n=aa.numel();return aa.float().reshape(1,n).expand(_P,n).contiguous()
def dn(g2,a8,ap):aa=g2.detach().abs().amax().float();a2,ab=cC(aa,dM/et,ek[0],ek[1]);return cS(torch.stack([a2,a8.float().reshape(()),(ab*ap).float().reshape(())]))
def dX(m,d):return m>0 and m%_P==0 and d%c7==0
def dK(w,cw):r,d=w.shape;return w.reshape(r//cw,cw,d//_P,_P).permute(0,3,2,1).contiguous().reshape(-1)
def dJ(w,cw):r,d=w.shape;return w.reshape(r//_P,_P,d//cw,cw).permute(2,1,0,3).contiguous().reshape(-1)
def dE(wq,wk,wv,wp,a2,bx,by,br,bc,ab=3):
 d=wq.shape[1];sq,sp=a2[cR].float(),a2[cD].float();ap=torch.cat([w.detach().to(torch.bfloat16).float()for w in(wq,wk,wv)])*sq;a8=wp.detach().to(torch.bfloat16).float()*sp;aa=torch.zeros(eG,d*d,dtype=torch.uint8)
 if ab%2==1:aa[dz:dz+3]=b9(dK(ap,bx)).reshape(3,d*d);aa[dO]=b9(dK(a8,by))
 if ab>=2:aa[dv:dv+3]=b9(dJ(ap,br)).reshape(3,d*d);aa[dN]=b9(dJ(a8,bc))
 return aa
def eo(b):return b.contiguous().view(torch.float8_e4m3fn).float()
def eB(ap,aa,d,cw):nw=3 if aa==0 else 1;r0=dz if aa==0 else dO;v=eo(ap[r0:r0+nw].reshape(-1));return v.reshape(nw*d//cw,_P,d//_P,cw).permute(0,3,2,1).contiguous().reshape(nw*d,d)
def eA(ap,aa,d,cw):nw=3 if aa==0 else 1;r0=dv if aa==0 else dN;v=eo(ap[r0:r0+nw].reshape(-1));return v.reshape(d//cw,_P,nw*d//_P,cw).permute(2,1,0,3).contiguous().reshape(nw*d,d)
def da(a2,ap,ab,bc,dq,cw):w8=eB(ap,ab,a2.shape[1],cw);a8=q8(a2.float()*bc,0);return((a8@w8.t())*dq).to(a2.dtype)
def d5(g2,ap,aa,a2,dq,cw):nw=3 if aa==0 else 1;w8=eA(ap,aa,g2.shape[1]//nw,cw);g8=q8(g2.float()*a2,1);return((g8@w8)*dq).to(g2.dtype)
cA = False
cK: dict = {}
cB = {"fwd_bodies": 0, "bwd_bodies": 0, "precast_bodies": 0}
ATTN8_SCALE_FN = None
ATTN8_PRECAST_FN = None
def b4(k,aa,nw,ab):
 if ab:return plan(k//128,aa,c4,eb)
 return plan(nw*aa//128,k,c4,eb)
def cJ(d):return(b4(d,d,3,True)[1],b4(d,d,1,True)[1],b4(d,d,3,False)[1],b4(d,d,1,False)[1])
def do():
 global cK;from torch_neuronx import nki_op as make_op
 if hasattr(torch.ops.fused_attn8tt,"qkv_fwd"):cK={n:getattr(torch.ops.fused_attn8tt,n)for n in("qkv_fwd","qkv_bwd","proj_fwd","proj_bwd","precast")};return("custom ops fused_attn8tt::qkv_fwd / qkv_bwd / proj_fwd / proj_bwd / precast (registered earlier " "in this process)")
 fwd_kernel=cr(_proj8_fwd_tt_kernel);bwd_kernel=cr(_proj8_bwd_tt_kernel);pre_kernel=cr(_attn8_precast_kernel);forms,u8,need=c4,eO,dR
 @make_op("fused_attn8tt::qkv_fwd",mutates_args=())
 def qkv_fwd(x2:torch.Tensor,w8:torch.Tensor,sc:torch.Tensor)->torch.Tensor:cB["fwd_bodies"]+=1;tb,aa=b4(x2.shape[1],x2.shape[1],3,True);return fwd_kernel(x2,w8,sc,3,dz,eg,d8,tb,aa,forms,u8)
 @make_op("fused_attn8tt::qkv_bwd",mutates_args=())
 def qkv_bwd(g2:torch.Tensor,w8:torch.Tensor,bc:torch.Tensor)->torch.Tensor:cB["bwd_bodies"]+=1;d=g2.shape[1]//3;tb,aa=b4(d,d,3,False);return bwd_kernel(g2,w8,bc,3,dv,dT,eq,tb,aa,forms,u8)
 @make_op("fused_attn8tt::proj_fwd",mutates_args=())
 def proj_fwd(y2:torch.Tensor,w8:torch.Tensor,sc:torch.Tensor)->torch.Tensor:cB["fwd_bodies"]+=1;tb,aa=b4(y2.shape[1],y2.shape[1],1,True);return fwd_kernel(y2,w8,sc,1,dO,dW,dl,tb,aa,forms,u8)
 @make_op("fused_attn8tt::proj_bwd",mutates_args=())
 def proj_bwd(g2:torch.Tensor,w8:torch.Tensor,bc:torch.Tensor)->torch.Tensor:cB["bwd_bodies"]+=1;tb,aa=b4(g2.shape[1],g2.shape[1],1,False);return bwd_kernel(g2,w8,bc,1,dN,dT,eq,tb,aa,forms,u8)
 @make_op("fused_attn8tt::precast",mutates_args=())
 def precast(wq:torch.Tensor,wk:torch.Tensor,wv:torch.Tensor,wp:torch.Tensor,sc:torch.Tensor)->torch.Tensor:cB["precast_bodies"]+=1;ap,a2,ab,aa=cJ(wq.shape[1]);return pre_kernel(wq,wk,wv,wp,sc,need,ap,a2,ab,aa,forms,u8)
 aa={"fused_attn8tt::qkv_fwd":lambda x2,w8,sc:x2.new_empty(x2.shape[0],3*x2.shape[1]),"fused_attn8tt::qkv_bwd":lambda g2,w8,bc:g2.new_empty(g2.shape[0],g2.shape[1]//3),"fused_attn8tt::proj_fwd":lambda y2,w8,sc:y2.new_empty(y2.shape[0],y2.shape[1]),"fused_attn8tt::proj_bwd":lambda g2,w8,bc:g2.new_empty(g2.shape[0],g2.shape[1]),"fused_attn8tt::precast":lambda wq,wk,wv,wp,sc:wq.new_empty(8,wq.shape[1]*wq.shape[1],dtype=torch.uint8),};ab=[]
 for a2,ap in aa.items():
  try:
   torch.library.register_fake(a2)(ap);ab.append("ours")
  except Exception as err:ab.append(f"the decorator's ({type(err).__name__})")
 cK={"qkv_fwd":qkv_fwd,"qkv_bwd":qkv_bwd,"proj_fwd":proj_fwd,"proj_bwd":proj_bwd,"precast":precast};return (f"custom ops fused_attn8tt::qkv_fwd / qkv_bwd / proj_fwd / proj_bwd / precast by nki_op, fake impls {'/'.join(ab)}, autograd by AttnProj8; need={need} forms={forms} u8={u8}")
def setup_attn8_kernel(aa):
 global cA;ab=("the inline nki_attn_proj8.py (md5 275376c2) and nki_attn_tt8g.py (md5 2f0d0c02), fp8=both set=qkv+proj " "forms=0 tb_max=2048 precast=1 need=3 tt=1")
 if aa.type!="neuron":return f"emulation (not a Neuron device): {ab}"
 if not _HAS_MLP_KERNEL:raise RuntimeError(f"the fp8 attention projections need nki on a Neuron device ({_MLP_KERNEL_ERROR})")
 if cA:return "on (registered earlier in this process)"
 ap=do();cA=True;return f"on: {ap}; {ab}"
def d4(a2,w8,sc,ap):
 if a2.device.type=="neuron":return cK["qkv_fwd" if ap==0 else "proj_fwd"](a2,w8,sc)
 aa,d=sc[0],a2.shape[1];bc,bf,by,bx=cJ(d)
 if ap==0:return da(a2,w8,0,aa[eg],aa[d8],bc)
 return da(a2,w8,1,aa[dW],aa[dl],bf)
def d2(g2,w8,bc,aa):
 if g2.device.type=="neuron":return cK["qkv_bwd" if aa==0 else "proj_bwd"](g2,w8,bc)
 ap=bc[0];d=g2.shape[1]//(3 if aa==0 else 1);bx,by,bf,a2=cJ(d);return d5(g2,w8,aa,ap[dT],ap[eq],bf if aa==0 else a2)
class AttnProj8(torch.autograd.Function):
 @staticmethod
 def forward(ctx,a2,sc,w8,kind,w0,w1,w2):ws=(w0,w1,w2)if kind==0 else(w0,);aa=d4(a2,w8,sc,kind);ctx.save_for_backward(a2,sc,w8,*ws);ctx.kind=kind;return aa
 @staticmethod
 def backward(ctx,g):
  aa=ctx.saved_tensors;a2,sc,w8,ws=aa[0],aa[1],aa[2],tuple(aa[3:]);ab=ctx.kind;g=g.contiguous();ap=sc[0]
  if ab==0:bc=dn(g,ap[cR],ap[eI])
  else:bc=dn(g,ap[cD],ap[d9])
  dx=d2(g,w8,bc,ab);dw=dF(g,a2,sc,bc,ab)
  if ab==0:n=ws[0].shape[0];return dx,None,None,None,dw[:n],dw[n:2*n],dw[2*n:]
  return dx,None,None,None,dw,None,None
def dm(a,sc,bc,ws,w8):
 b,t,k=a.shape;a2=a.reshape(b*t,k).contiguous();m=a2.size(0);ap=0
 if a2.dtype!=torch.bfloat16 or ws[0].shape!=(k,k):raise RuntimeError(f"the pre-cast projections take bf16 rows and square weights, not {tuple(a2.shape)} {a2.dtype} with {tuple(ws[0].shape)}")
 if a2.device.type=="neuron":
  if not(cA and dX(m+(-m)%128,k)):raise RuntimeError(f"the pre-cast projection kernels do not take rows {tuple(a2.shape)} (registered: {cA})")
  ap=(-m)%128
  if ap:a2=F.pad(a2,(0,0,0,ap))
 if bc==0:ab=AttnProj8.apply(a2,sc,w8,0,ws[0],ws[1],ws[2])
 else:ab=AttnProj8.apply(a2,sc,w8,1,ws[0],None,None)
 if ap:ab=ab[:m]
 return ab.view(b,t,ab.size(-1))
def attn8_qkv(attn,x):ws=tuple(lin.weight.to(dtype=x.dtype)for lin in(attn.c_q,attn.c_k,attn.c_v));return dm(x,attn.attn8_sc,0,ws,attn.attn8_w8)
def attn8_proj(aa,y):return dm(y,aa.attn8_sc,1,(aa.c_proj.weight.to(dtype=y.dtype),),aa.attn8_w8)
@torch.no_grad()
def attn8_scales_into(wq_list,wk_list,wv_list,wp_list,ve_list,lam_list,sc_list):
 for wq,wk,wv,wp,ve,aa,sc in zip(wq_list,wk_list,wv_list,wp_list,ve_list,lam_list,sc_list):
  if ve is None:vb=torch.zeros((),dtype=torch.float32,device=wq.device)
  else:vb=aa.detach().float().abs().amax()*ve.detach().float().abs().amax()
  sc.copy_(cS(dk(wq,wk,wv,wp,vb)))
@torch.no_grad()
def attn8_precast_into(wq_list,wk_list,wv_list,wp_list,sc_list,w8_list):
 for wq,wk,wv,wp,sc,w8 in zip(wq_list,wk_list,wv_list,wp_list,sc_list,w8_list):
  bf=[w.to(torch.bfloat16)for w in(wq,wk,wv,wp)]
  if cA and w8.device.type=="neuron":w8.copy_(cK["precast"](bf[0],bf[1],bf[2],bf[3],sc))
  else:a8,bc,a2,ab=cJ(wq.shape[1]);w8.copy_(dE(bf[0],bf[1],bf[2],bf[3],sc[0].float(),a8,bc,a2,ab,dR))
def c9(aa):blocks=aa.transformer.h;fn=ATTN8_PRECAST_FN or attn8_precast_into;fn([blk.attn.c_q.weight for blk in blocks],[blk.attn.c_k.weight for blk in blocks],[blk.attn.c_v.weight for blk in blocks],[blk.attn.c_proj.weight for blk in blocks],[blk.attn.attn8_sc for blk in blocks],[blk.attn.attn8_w8 for blk in blocks])
def attn8_refresh_scales(aa):blocks=aa.transformer.h;tables=aa.value_embeds;ab=[tables[str(i)].weight if tables is not None and str(i)in tables else None for i in range(len(blocks))];fn=ATTN8_SCALE_FN or attn8_scales_into;fn([blk.attn.c_q.weight for blk in blocks],[blk.attn.c_k.weight for blk in blocks],[blk.attn.c_v.weight for blk in blocks],[blk.attn.c_proj.weight for blk in blocks],ab,[blk.ve_lambda for blk in blocks],[blk.attn.attn8_sc for blk in blocks]);c9(aa)
def attn8_scales_line(model):
 rows=torch.stack([blk.attn.attn8_sc[0].detach().float().cpu()for blk in model.transformer.h])
 def log2s(col):return[int(v)for v in torch.log2(rows[:,col]).round().tolist()]
 w=model.transformer.h[0].attn.c_q.weight;k,n=w.shape[1],w.shape[0];aa=[]
 for ap,nw,a8 in(("qkv_fwd",3,True),("qkv_bwd",3,False),("proj_fwd",1,True),("proj_bwd",1,False)):tb,ab=b4(k,n,nw,a8);aa.append(f"{ap}:{tb}x{ab}")
 a2=[round(v,2)for v in rows[:,e3].tolist()];vb=[round(v,3)for v in rows[:,e8].tolist()];return (f"log2_s_wqkv={log2s(cR)} log2_s_y={log2s(dW)} log2_s_wproj={log2s(cD)} b_y={a2} vb={vb} plans={','.join(aa)}")
def attn8_summary():return " ".join(f"{k}={v}" for k, v in cB.items()) + (" registered" if cA else " not registered")
eC = 0
AB = 8
eT = 2048
eE,eD,eS=0,1,2
eY = 3
df,cG,cN=0,1,2
eU = (-60, 60)
def ds(m,d):mt=m//_P;a8=mt*d;at=2*AB*d*2;gt=2*mt*_P*3;o=2*d*2;return(a8+at+gt+o+eY*4)/1024.0
def dQ(m,o,d):return(m>0 and m%c7==0 and o>0 and o%_P==0 and 0<d<=eT and d%_P==0 and ds(m,d)<=dh)
def dL(d):
 if d<=0 or d%_P!=0:return 0
 aa=0
 for gw in range(eT,0,-_P):
  if d%gw==0:
   if(d//gw)%2==0:return gw
   if aa==0:aa=gw
 return aa
def dG(m,o,d):gw=dL(d);return(m>0 and m%c7==0 and o>0 and o%_P==0 and gw>0 and d%gw==0 and ds(m,gw)<=dh)
def cP(s):return cC(s.float(),1.0,-eU[1],-eU[0])[0]
def e6(a2,ab,ap,aa):return torch.stack([a2.float().reshape(()),ap.float().reshape(()),(ab.float()*aa.float()).reshape(())])
def eM(t,aa):
 t=t.float()
 if aa==cN:t=torch.clamp_max(t,cF)
 return t.to(torch.float8_e5m2 if aa==cG else torch.float8_e4m3fn).float()
def ez(g2,a2,ap,ab,dq,fg,fa):g8=eM(g2.float()*ap,fg);a8=eM(a2.float()*ab,fa);return((g8.t()@a8)*dq).to(g2.dtype)
cM = False
du: dict = {}
dd = {"g5a4_bodies": 0, "g4ca5_bodies": 0}
def dC():
 global du;from torch_neuronx import nki_op as make_op
 if hasattr(torch.ops.fused_dw8,"g5a4"):du={(cG,df):torch.ops.fused_dw8.g5a4,(cN,cG):torch.ops.fused_dw8.g4ca5,(cG,cN):torch.ops.fused_dw8.g5a4c};return "custom ops fused_dw8::g5a4 / g4ca5 (registered earlier in this process)"
 kernel=cr(_dw8_kernel);wkernel=cr(_dw8w_kernel);forms=eC;e4,e5,e4c=df,cG,cN
 @make_op("fused_dw8::g5a4",mutates_args=())
 def dw_g5a4(g2:torch.Tensor,a2:torch.Tensor,sc:torch.Tensor)->torch.Tensor:dd["g5a4_bodies"]+=1;return kernel(g2,a2,sc,e5,e4,forms)
 @make_op("fused_dw8::g4ca5",mutates_args=())
 def dw_g4ca5(g2:torch.Tensor,a2:torch.Tensor,sc:torch.Tensor)->torch.Tensor:dd["g4ca5_bodies"]+=1;return kernel(g2,a2,sc,e4c,e5,forms)
 @make_op("fused_dw8::g5a4c",mutates_args=())
 def dw_g5a4c(g2:torch.Tensor,a2:torch.Tensor,sc:torch.Tensor)->torch.Tensor:return wkernel(g2,a2,sc,e5,e4c,forms,dL(a2.shape[1]))
 aa=[]
 for ab in("fused_dw8::g5a4","fused_dw8::g4ca5","fused_dw8::g5a4c"):
  try:
   torch.library.register_fake(ab)(lambda g2,a2,sc:g2.new_empty(g2.shape[1],a2.shape[1]));aa.append("ours")
  except Exception as err:aa.append(f"the decorator's ({type(err).__name__})")
 du={(e5,e4):dw_g5a4,(e4c,e5):dw_g4ca5,(e5,e4c):dw_g5a4c};return (f"custom ops fused_dw8::g5a4 / g4ca5 by nki_op, fake impls {'/'.join(aa)}, called from the backward formulas of the MLP and the projections; forms={forms}")
def setup_dw8_kernel(aa):
 global cM;ab="the inline nki_dw8.py (md5 0420bca1), set=qkv,proj forms=0 mlp_dw_mode=unchanged"
 if aa.type!="neuron":return f"emulation (not a Neuron device): {ab}"
 if not _HAS_MLP_KERNEL:raise RuntimeError(f"the fp8 weight gradients need nki on a Neuron device ({_MLP_KERNEL_ERROR})")
 if cM:return "on (registered earlier in this process)"
 ap=dC();cM=True;return f"on: {ap}; {ab}"
def dP(g2,a2,b3,by,b2,bx,fg,fa):
 bc=e6(b3,by,b2,bx);g2=g2.contiguous();a2=a2.contiguous()
 if g2.device.type=="neuron":
  ap=g2.size(0);a8=(-ap)%256
  if not(cM and g2.dtype==torch.bfloat16 and a2.dtype==torch.bfloat16 and a2.size(0)==ap and(dG if fa==cN else dQ)(ap+a8,g2.size(1),a2.size(1))):raise RuntimeError(f"the fp8 weight-gradient kernel does not take {tuple(g2.shape)} x {tuple(a2.shape)} {g2.dtype} (registered: {cM})")
  if a8:g2=F.pad(g2,(0,0,0,a8));a2=F.pad(a2,(0,0,0,a8))
  return du[(fg,fa)](g2,a2,cS(bc))
 return ez(g2,a2,bc[0],bc[1],bc[2],fg,fa)
def dF(g,a2,sc,bc,a8):aa=sc[0];ap=bc[0][dT];ab=aa[eg]if a8==0 else aa[dW];return dP(g,a2,ap,cP(ap),ab,cP(ab),cG,df)
def dw8_summary():return (" ".join(f"{k}={v}" for k, v in dd.items()) + (" registered" if cM else " not registered")+" set=qkv,proj")
class CausalSelfAttention(nn.Module):
 attn_nw=0
 def __init__(self,config):
  super().__init__();self.n_head=config.n_head;self.n_embd=config.n_embd;self.head_dim=config.n_embd//config.n_head;assert self.head_dim*self.n_head==self.n_embd;self.c_q=Linear(config.n_embd,config.n_embd,bias=False);self.c_k=Linear(config.n_embd,config.n_embd,bias=False);self.c_v=Linear(config.n_embd,config.n_embd,bias=False);self.c_proj=Linear(config.n_embd,config.n_embd,bias=False);self.qk_scale=config.qk_scale;self.fused=FUSED_ATTENTION;self.band_kernel=ATTN_BAND_KERNEL
  if self.fused and self.head_dim!=ATTN_HEAD_DIM:raise ValueError(f"the fused attention kernels take head_dim {ATTN_HEAD_DIM}, not {self.head_dim}")
  self.register_buffer("attn8_sc",torch.full((128,NSC),float("nan")),persistent=False);self.register_buffer("attn8_w8",torch.zeros((8,config.n_embd*config.n_embd),dtype=torch.uint8),persistent=False)
 def forward(self,x,cos,sin,ve=None,ve_lambda=None,):
  b,t,c=x.shape;band=0<self.attn_nw*128<t
  if self.fused and(self.band_kernel or not band):
   qkv=attn8_qkv(self,x)
   if ve is not None:qkv=torch.cat([qkv[...,:2*c],qkv[...,2*c:]+ve_lambda.to(qkv.dtype)*ve],dim=-1)
   y=(fused_attention_band if band else fused_attention)(qkv,cos,sin,self.n_head,self.qk_scale);return attn8_proj(self,y.view(b,t,c))
  q,k,v=attn8_qkv(self,x).split(c,dim=-1);q=q.view(b,t,self.n_head,self.head_dim);k=k.view(b,t,self.n_head,self.head_dim);v=v.view(b,t,self.n_head,self.head_dim)
  if ve is not None:v=v+ve_lambda.to(v.dtype)*ve.view(b,t,self.n_head,self.head_dim)
  q=norm(apply_rotary_emb(q,cos,sin));k=norm(apply_rotary_emb(k,cos,sin));q=q.transpose(1,2);k=k.transpose(1,2);v=v.transpose(1,2)
  if band:y=F.scaled_dot_product_attention(q,k,v,attn_mask=self.attn_band[:t,:t],scale=self.qk_scale if self.qk_scale>0.0 else None)
  elif self.qk_scale>0.0:y=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=self.qk_scale)
  else:y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
  y=y.transpose(1,2).contiguous().view(b,t,c);return attn8_proj(self,y)
class MLP(nn.Module):
 def __init__(self,config):super().__init__();self.c_fc=Linear(config.n_embd,4*config.n_embd,bias=False);self.c_proj=Linear(4*config.n_embd,config.n_embd,bias=False);self.register_buffer("mlp8_sc",torch.full((128,NSC),float("nan")),persistent=False);self.register_buffer("mlp8_w8",torch.zeros((4,4*config.n_embd*config.n_embd),dtype=torch.uint8),persistent=False)
 def forward(self,x):return mlp_fused(self,x)
MOE_EP = {"rank": 0, "world": 1, "attach": False}
MOE_BLOCK = 128
MOE_DIGIT = 64
MOE_B_MAX = 256
MOE_RESCUE = {"copies": 0, "seconds": 0.0}
MOE_FUNCOL = None
def moe_capacity_for(cf,seq_len,topk,n_exp):return max(1,int(round(cf*seq_len*topk/n_exp)))
def moe_config_fields(depth,n_embd,sequence_len):n_exp=MOE_EP["world"];return{"moe_layers":tuple(i for i in range(depth//2,depth)if i%2),"moe_experts":n_exp,"moe_topk":MOE_TOPK,"moe_capacity":moe_capacity_for(MOE_CF,sequence_len,MOE_TOPK,n_exp),"moe_hidden":MOE_HIDDEN,"moe_shared":2*n_embd,"moe_dispatch":"gather","moe_shared_mlp":"base","moe_expert8":1}
def moe_seed(*parts):return int(hashlib.md5(":".join(str(p)for p in("moe",)+parts).encode("ascii")).hexdigest()[:15],16)
def moe_tournament(v,iota):
 n=v.size(-1);i=iota.expand_as(v)
 while n>1:
  h=n//2;v2=v.reshape(*v.shape[:-1],h,2);i2=i.reshape(*i.shape[:-1],h,2);up=v2[...,1]>v2[...,0];v=torch.where(up,v2[...,1],v2[...,0]);i=torch.where(up,i2[...,1],i2[...,0]);n=h
 return v[...,0],i[...,0]
def moe_route(logits,iota_e):p=torch.softmax(logits.float(),dim=-1);v1,i1=moe_tournament(p,iota_e);m1=(iota_e-i1.unsqueeze(-1)).abs()<0.5;return{"p":p,"m":(m1,),"i":(i1,),"g":(v1,)}
def moe_positions(mall,b,t,tril_in,tril_ex):
 n_exp=mall.size(-1);blk=MOE_BLOCK;tp=-(-t//blk)*blk;m=mall.reshape(b,t,n_exp)
 if tp!=t:m=F.pad(m,(0,0,0,tp-t))
 nb=tp//blk;mb=m.reshape(b,nb,blk,n_exp);incl=torch.matmul(tril_in,mb);prefix=torch.matmul(tril_ex[:nb,:nb],incl[:,:,blk-1,:]);excl=incl-mb+prefix.unsqueeze(2);return excl.reshape(b,tp,n_exp)[:,:t]
def moe_slot_onehot(excl,mall3,cap,iota_c,dtype):oc=(iota_c-excl.unsqueeze(-1)).abs()<0.5;return oc.to(dtype)*mall3.to(dtype).unsqueeze(-1)
def moe_gather_tables(route,excl,mall,b,t,cap,c):
 n_exp=c["iota_e"].numel();n=b*t;s_total=n_exp*b*cap;rowf=c["iota_b"][:b];row_tok=rowf.view(b,1).expand(b,t).reshape(n);ex=excl.reshape(n,n_exp);slots=[]
 for mk,ik in zip(route["m"],route["i"]):pos=(ex*mk.to(ex.dtype)).sum(-1);keep=pos<(cap-0.5);slot=torch.where(keep,(ik*b+row_tok)*cap+pos,torch.full_like(pos,float(s_total)));slots.append(slot.to(torch.int32))
 d=moe_slot_onehot(excl,mall.reshape(b,t,n_exp),cap,c["iota_c"],torch.bfloat16).reshape(b,t,n_exp*cap);hi=torch.matmul(c["tok_hi"][:,:t].to(torch.bfloat16),d).float();lo=torch.matmul(c["tok_lo"][:,:t].to(torch.bfloat16),d).float();occ=torch.matmul(c["ones_t"][:,:t].to(torch.bfloat16),d).float();tok=hi*MOE_DIGIT+lo+(rowf*t).view(b,1,1);p=torch.where(occ>0.5,tok,torch.full_like(tok,float(n)));p=p.reshape(b,n_exp,cap).transpose(0,1).reshape(s_total).to(torch.int32);return{"s":slots,"p":p}
class MoeGatherRows(torch.autograd.Function):
 @staticmethod
 def forward(ctx,src,idx,inv):ctx.save_for_backward(inv);pad=torch.cat([src,src.new_zeros(1,src.size(1))],0);return pad.index_select(0,idx)
 @staticmethod
 def backward(ctx,g):(inv,)=ctx.saved_tensors;gpad=torch.cat([g,g.new_zeros(1,g.size(1))],0);return gpad.index_select(0,inv),None,None
def moe_all_to_all(t):return MOE_FUNCOL.all_to_all_single(t.contiguous(),None,None,dist.group.WORLD)
def moe_expert8_mlp(x2,w_fc,w_proj,sc,w8=None):
 dt=x2.dtype;wf,wp=w_fc.to(dt),w_proj.to(dt);m=x2.size(0)
 if mlp_fused_applies(x2,wf.size(0)):
  pad=(-m)%128
  if pad:x2=F.pad(x2,(0,0,0,pad))
  y,_rT=MLP_OPS["fwd"](x2.contiguous(),wf,wp,sc,MLP_OPS["precast"](wf,wp,sc)if w8 is None else w8);return y[:m]if pad else y
 if x2.device.type!="neuron":return EmulatedFusedMlp8.apply(x2.contiguous(),wf,wp,sc)
 raise RuntimeError(f"fp8 MLP: expert rows {tuple(x2.shape)} {dt}")
def moe_experts8_local(xs3,w_fc,w_proj,sc):return torch.stack([moe_expert8_mlp(xs3[e],w_fc[e],w_proj[e],sc[e])for e in range(xs3.size(0))])
def moe_aux_terms(route,logits,mall,b,t,n_exp,topk):p=route["p"];f=mall.mean(0)/topk;lb=n_exp*(f*p.mean(0)).sum();z=torch.logsumexp(logits,dim=-1).square().mean();aux=MOE_AUX*lb+MOE_ZLOSS*z;return aux,lb,z,f
class MoeShared(MLP):
 def __init__(self,config,hidden):nn.Module.__init__(self);self.c_fc=torch.nn.utils.skip_init(Linear,config.n_embd,hidden,bias=False);self.c_proj=torch.nn.utils.skip_init(Linear,hidden,config.n_embd,bias=False);self.register_buffer("mlp8_sc",torch.full((128,NSC),float("nan")),persistent=False);self.register_buffer("mlp8_w8",torch.zeros((4,hidden*config.n_embd),dtype=torch.uint8),persistent=False)
class MoeMLP(nn.Module):
 def __init__(self,config,layer):
  super().__init__();global MOE_FUNCOL;d,n_exp,hid=config.n_embd,config.moe_experts,config.moe_hidden;self.layer,self.n_exp,self.hidden=layer,n_exp,hid;self.topk,self.cap=config.moe_topk,config.moe_capacity;rank,world=MOE_EP["rank"],MOE_EP["world"]
  if n_exp<2 or n_exp&(n_exp-1):raise ValueError(f"{n_exp} experts is not a power of two from 2 (the tournament pairs them)")
  if world>1 and n_exp!=world:raise ValueError(f"{n_exp} experts on {world} ranks: the expert-parallel form holds one a rank")
  if not 1<=self.cap<=256:raise ValueError(f"capacity {self.cap} outside 1 to 256 (the bf16-exact count bound)")
  if(config.moe_dispatch!="gather" or config.moe_topk!=1 or config.moe_shared_mlp!="base" or config.moe_shared<1 or hid<1 or config.moe_expert8!=1):raise ValueError("the MoE here is top-1, gather, a shared half and fp8 experts")
  MLP(config);self.ep=world>1;self.e0=rank if self.ep else 0;self.n_local=1 if self.ep else n_exp
  if self.ep and MOE_FUNCOL is None:import torch.distributed._functional_collectives as funcol;MOE_FUNCOL=funcol
  self.router=nn.Parameter(torch.empty(n_exp,d));self.shared=MoeShared(config,config.moe_shared)
  if MOE_EP["attach"]:self.register_parameter("w_fc",None);self.register_parameter("w_proj",None)
  else:self.w_fc=nn.Parameter(torch.empty(self.n_local,hid,d));self.w_proj=nn.Parameter(torch.empty(self.n_local,d,hid))
  if self.ep:self.register_buffer("exp8_sc",torch.full((128,NSC),float("nan")),persistent=False);self.register_buffer("exp8_w8",torch.zeros((4,hid*d),dtype=torch.uint8),persistent=False)
  else:self.register_buffer("exp8_sc",torch.full((self.n_local,128,NSC),float("nan")),persistent=False)
  nb=-(-config.sequence_len//MOE_BLOCK)
  if nb*MOE_BLOCK>MOE_DIGIT*MOE_DIGIT:raise ValueError(f"sequence_len {config.sequence_len} above the two-digit position bound {MOE_DIGIT * MOE_DIGIT}")
  pos=torch.arange(nb*MOE_BLOCK,dtype=torch.int64);consts={"iota_e":torch.arange(n_exp,dtype=torch.float32),"iota_c":torch.arange(self.cap,dtype=torch.float32),"iota_b":torch.arange(MOE_B_MAX,dtype=torch.float32),"tril_in":torch.tril(torch.ones(MOE_BLOCK,MOE_BLOCK,dtype=torch.float32)),"tril_ex":torch.tril(torch.ones(nb,nb,dtype=torch.float32),diagonal=-1),"tok_hi":(pos//MOE_DIGIT).to(torch.float32).view(1,-1),"tok_lo":(pos%MOE_DIGIT).to(torch.float32).view(1,-1),"ones_t":torch.ones(1,nb*MOE_BLOCK,dtype=torch.float32)};self.const_names=tuple(consts)
  for k,v in consts.items():self.register_buffer("moe_"+k,v,persistent=False)
 def trunk_parameters(self):return[self.router]+list(self.shared.parameters())
 def expert_parameters(self):return[p for p in(self.w_fc,self.w_proj)if p is not None]
 @torch.no_grad()
 def init_stream(self,s):
  d=self.router.size(1);dense=torch.empty(4*d,d,dtype=torch.float32,device=self.router.device);torch.nn.init.uniform_(dense,-0.4*s,0.4*s);w=self.shared.c_fc.weight
  if w.size(0)>dense.size(0):raise ValueError(f"a shared half of {w.size(0)} is wider than the dense MLP's {dense.size(0)}")
  w.copy_(dense[:w.size(0)]);torch.nn.init.zeros_(self.shared.c_proj.weight)
 @torch.no_grad()
 def init_own(self,s):
  g=torch.Generator().manual_seed(moe_seed(42,self.layer,"router"));self.router.copy_(torch.randn(tuple(self.router.shape),generator=g)*MOE_ROUTER_STD)
  if self.w_fc is None:return
  for j in range(self.n_local):g=torch.Generator().manual_seed(moe_seed(42,self.layer,self.e0+j,"fc"));self.w_fc[j].copy_(torch.empty(tuple(self.w_fc.shape[1:])).uniform_(-0.4*s,0.4*s,generator=g))
  self.w_proj.zero_()
 def attach_experts(self,w_fc,w_proj):
  want=(self.n_exp,self.hidden,self.router.size(1))
  if tuple(w_fc.shape)!=want or tuple(w_proj.shape)!=(want[0],want[2],want[1]):raise ValueError(f"block {self.layer}: expert shapes {tuple(w_fc.shape)}")
  self.w_fc=nn.Parameter(w_fc,requires_grad=False);self.w_proj=nn.Parameter(w_proj,requires_grad=False)
 def forward(self,x,want_stats=False):
  b,t,d=x.shape;n_tok_max=self.moe_tok_hi.size(1)
  if b>MOE_B_MAX or t>n_tok_max:raise ValueError(f"a [{b}, {t}] input exceeds the layer's {MOE_B_MAX} rows or {n_tok_max} positions")
  n_exp,cap,k=self.n_exp,self.cap,self.topk;c={name:getattr(self,"moe_"+name)for name in self.const_names};h=x.reshape(b*t,d);logits=F.linear(h.float(),self.router.float());route=moe_route(logits,c["iota_e"]);mall=route["m"][0].to(torch.float32);excl=moe_positions(mall,b,t,c["tril_in"],c["tril_ex"]);s_total=n_exp*b*cap;tab=moe_gather_tables(route,excl,mall,b,t,cap,c);xs=MoeGatherRows.apply(h,tab["p"],tab["s"][0])
  if self.ep:ys=moe_all_to_all(moe_expert8_mlp(moe_all_to_all(xs),self.w_fc[0],self.w_proj[0],self.exp8_sc,self.exp8_w8))
  else:ys=moe_experts8_local(xs.reshape(n_exp,b*cap,d),self.w_fc,self.w_proj,self.exp8_sc).reshape(s_total,d)
  dt=ys.dtype;out=route["g"][0].to(dt).unsqueeze(-1)*MoeGatherRows.apply(ys,tab["s"][0],tab["p"]);out=out.reshape(b,t,d);out=out+self.shared(x)
  if not want_stats:return out
  aux,lb,z,f=moe_aux_terms(route,logits,mall,b,t,n_exp,k);kept=(mall*(excl.reshape(b*t,n_exp)<cap-0.5).to(torch.float32)).sum()/(b*t*k);stats=torch.cat([torch.stack([kept,lb.detach(),z.detach()]),f]).detach();return out,aux,stats
def mix_residual(x,x0,resid_lambda,x0_lambda,):return resid_lambda.to(dtype=x.dtype)*x+x0_lambda.to(dtype=x.dtype)*x0
class Block(nn.Module):
 def __init__(self,config,use_ve=False,layer=-1):super().__init__();self.attn=CausalSelfAttention(config);self.mlp=MoeMLP(config,layer)if layer in config.moe_layers else MLP(config);self.resid_lambda=nn.Parameter(torch.ones(1));self.x0_lambda=nn.Parameter(torch.zeros(1));self.ve_lambda=nn.Parameter(torch.full((1,),0.5))if use_ve else None
 def forward(self,x,x0,cos,sin,ve=None,moe_aux=None,):
  x=mix_residual(x,x0,self.resid_lambda,self.x0_lambda);x=x+self.attn(norm(x),cos,sin,ve,self.ve_lambda)
  if moe_aux is not None and isinstance(self.mlp,MoeMLP):y,aux,stats=self.mlp(norm(x),want_stats=True);moe_aux.append((aux,stats));return x+y
  x=x+self.mlp(norm(x));return x
class GPT(nn.Module):
 def __init__(self,config):
  super().__init__();self.config=config;padded_vocab_size=((config.vocab_size+63)//64)*64;ve_layers=value_embed_layers(config.n_layer,config.value_embeds);self.transformer=nn.ModuleDict({"wte":nn.Embedding(padded_vocab_size,config.n_embd),"h":nn.ModuleList([Block(config,use_ve=i in ve_layers,layer=i)for i in range(config.n_layer)]),});self.has_moe=bool(config.moe_layers)
  for i,blk in enumerate(self.transformer.h):blk.attn.attn_nw=0 if not config.attn_window or i in config.attn_long else config.attn_window//128
  self.lm_head=Linear(config.n_embd,padded_vocab_size,bias=False);cos,sin=self._precompute_rotary_embeddings(config.sequence_len,config.n_embd//config.n_head);self.register_buffer("cos",cos,persistent=False);self.register_buffer("sin",sin,persistent=False);self.value_embeds=nn.ModuleDict({str(i):nn.Embedding(padded_vocab_size,config.n_embd,_weight=torch.empty(padded_vocab_size,config.n_embd))for i in ve_layers})if ve_layers else None
  if config.unet and config.n_layer<2:raise ValueError("unet needs at least 2 layers")
  self.n_skip=config.n_layer//2 if config.unet else 0;self.skip_lambda=nn.Parameter(torch.zeros(self.n_skip))if self.n_skip else None;self.ngram_orders=tuple(config.ngram_orders);self._ngram_host=None
  if self.ngram_orders:self.ngram_proj=nn.ParameterList([nn.Parameter(torch.empty(config.n_embd,config.ngram_dim))for _ in self.ngram_orders]);self.ngram_lambda=nn.Parameter(torch.zeros(len(self.ngram_orders)))
  else:self.ngram_proj=None;self.ngram_lambda=None
  self._os1_host=None;code=os1_codebook(padded_vocab_size,config.os1_dim,config.os1_code_seed)/config.os1_a0;gcode=code.clone();gcode[config.os1_bos]=0.0;self.register_buffer("os1_code",code.to(COMPUTE_DTYPE),persistent=False);self.register_buffer("os1_gcode",gcode,persistent=False);self.register_buffer("os1_band",os1_band(config.sequence_len,config.os1_gate_w),persistent=False);self.register_buffer("os1_codeT",self.os1_code.t().contiguous(),persistent=False);self.register_buffer("os1_codeg",(self.os1_code.float()*config.os1_glin).to(COMPUTE_DTYPE),persistent=False);wv,wd=self.os1_code.size(0),config.n_embd+self.os1_code.size(1);self.register_buffer("os1_wcat",torch.zeros(wv,wd,dtype=COMPUTE_DTYPE),persistent=False)
 @torch.no_grad()
 def init_weights(self):
  torch.nn.init.normal_(self.transformer.wte.weight,mean=0.0,std=0.8);torch.nn.init.normal_(self.lm_head.weight,mean=0.0,std=0.001);s=math.sqrt(3.0)*self.config.n_embd**-0.5
  for block in self.transformer.h:
   torch.nn.init.uniform_(block.attn.c_q.weight,-s,s);torch.nn.init.uniform_(block.attn.c_k.weight,-s,s);torch.nn.init.uniform_(block.attn.c_v.weight,-s,s);torch.nn.init.zeros_(block.attn.c_proj.weight)
   if isinstance(block.mlp,MoeMLP):block.mlp.init_stream(s)
   else:torch.nn.init.uniform_(block.mlp.c_fc.weight,-0.4*s,0.4*s);torch.nn.init.zeros_(block.mlp.c_proj.weight)
   block.resid_lambda.fill_(1.0);block.x0_lambda.fill_(0.05)
   if block.ve_lambda is not None:block.ve_lambda.fill_(0.5)
  if self.value_embeds is not None:
   for table in self.value_embeds.values():torch.nn.init.normal_(table.weight,mean=0.0,std=0.8)
  if self.skip_lambda is not None:self.skip_lambda.fill_(UNET_INIT)
  if self.ngram_proj is not None:
   for w in self.ngram_proj:torch.nn.init.orthogonal_(w)
   self.ngram_lambda.fill_(NGRAM_GAIN)
  for block in self.transformer.h:
   if isinstance(block.mlp,MoeMLP):block.mlp.init_own(s)
  cos,sin=self._precompute_rotary_embeddings(self.config.sequence_len,self.config.n_embd//self.config.n_head);self.cos.copy_(cos);self.sin.copy_(sin)
 def _precompute_rotary_embeddings(self,seq_len,head_dim,base=100000):device=self.transformer.wte.weight.device;freqs=torch.arange(0,head_dim,2,dtype=torch.float32,device=device);inv_freq=1.0/(base**(freqs/head_dim));t=torch.arange(seq_len,dtype=torch.float32,device=device);table=torch.outer(t,inv_freq);cos=table.cos().to(COMPUTE_DTYPE)[None,:,None,:];sin=table.sin().to(COMPUTE_DTYPE)[None,:,None,:];return cos,sin
 def estimate_flops(self):
  nparams=sum(p.numel()for p in self.parameters());non_matmul=self.transformer.wte.weight.numel()+2*self.config.n_layer
  if self.value_embeds is not None:non_matmul+=sum(table.weight.numel()for table in self.value_embeds.values())
  non_matmul+=sum(p.numel()for p in self.gain_params());non_matmul+=sum(p.numel()for n,p in self.named_parameters()if n in("ngram_table","ngram_scale"));non_matmul+=sum(p.numel()for n,p in self.named_parameters()if n in("os1_table","os1_scale"));h=self.config.n_head;d=self.config.n_embd//self.config.n_head;t=self.config.sequence_len;attn_flops=self.config.n_layer*12*h*d*t;return 6*(nparams-non_matmul)+attn_flops
 def num_scaling_params(self):transformer_matrices=sum(p.numel()for block in self.transformer.h for p in block.parameters()if p.ndim==2);lm_head=self.lm_head.weight.numel();return{"transformer_matrices":transformer_matrices,"lm_head":lm_head,"total":transformer_matrices+lm_head,}
 def gain_params(self):
  gains=[block.ve_lambda for block in self.transformer.h if block.ve_lambda is not None]
  if self.skip_lambda is not None:gains.append(self.skip_lambda)
  if self.ngram_lambda is not None:gains.append(self.ngram_lambda)
  return gains
 def set_ngram_host(self,host):self._ngram_host=host
 def attach_ngram_table(self,table,scale):
  for t in(table,scale):
   if t is not None:t.view(torch.uint8).reshape(-1)[::4096].sum()
  self.ngram_table=nn.Parameter(table,requires_grad=False)
  if scale is not None:self.ngram_scale=nn.Parameter(scale,requires_grad=False)
  self._ngram_host=NgramHostTable.read_only(self.ngram_table.data,None if scale is None else self.ngram_scale.data,self.ngram_orders,self.config.ngram_seed,((self.config.vocab_size+63)//64)*64)
 def set_os1_host(self,host):self._os1_host=host
 def attach_os1_table(self,table,scale):
  for t in(table,scale):t.view(torch.uint8).reshape(-1)[::4096].sum()
  self.os1_table=nn.Parameter(table,requires_grad=False);self.os1_scale=nn.Parameter(scale,requires_grad=False);self._os1_host=Os1HostTable.read_only(self.os1_table.data,self.os1_scale.data,self.config,((self.config.vocab_size+63)//64)*64)
 @torch.compiler.disable
 def ngram_lookup(self,idx):
  host=self._ngram_host
  if host is None:raise RuntimeError("no n-gram table is attached (set_ngram_host or attach_ngram_table)")
  cpu_idx=idx.detach().to("cpu");rows=host.gather(host.ids(cpu_idx));return torch.cat([rows,self._os1_host.lookup_u(cpu_idx).unsqueeze(-2)],dim=-2).to(idx.device)
 def ngram_input(self,e):
  e=e.to(COMPUTE_DTYPE);es=[self.ngram_lambda[j:j+1].to(e.dtype)*e[:,:,j]for j in range(len(self.ngram_orders))];out=None
  for x,w in zip(es,self.ngram_proj):y=F.linear(x,w.to(x.dtype));out=y if out is None else out+y
  return out
 def os1_bias(self,u,idx):cfg=self.config;u=u.to(COMPUTE_DTYPE);s=F.linear(u,self.os1_code.to(COMPUTE_DTYPE));b,t=idx.shape;ud=u.detach().float();cn=F.embedding(idx[:,1:],self.os1_gcode);z=torch.cat([(ud[:,:t-1]*cn).sum(-1),ud.new_zeros(b,1)],dim=1);h=z@self.os1_band[:t,:t];g=torch.sigmoid(cfg.os1_gate_slope*(h-cfg.os1_gate_h0)).to(s.dtype).unsqueeze(-1);read=(cfg.os1_beta/(1.0-cfg.os1_thr))*F.relu(s.detach()-cfg.os1_thr)*g;return cfg.os1_glin*s+read
 def os1_tail(self,x,ngram_emb,idx,targets,loss_reduction,loss_div,moe_aux=None):
  u=ngram_emb[:,:,len(self.ngram_orders)]
  if targets is not None and loss_reduction=="mean":loss=self.os1_fused_loss(x,u,idx,targets);return loss_scaled(loss,loss_div)if moe_aux is None else moe_loss_out(loss,moe_aux,loss_div)
  head=self.lm_head(x)+self.os1_bias(u,idx);logits=head[...,:self.config.vocab_size].float();logits=LOGIT_SOFTCAP*torch.tanh(logits/LOGIT_SOFTCAP)
  if targets is None:return logits
  loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),ignore_index=-1,reduction=loss_reduction)
  if loss_reduction=="none":return loss.view_as(targets)
  return loss_scaled(loss,loss_div)if moe_aux is None else moe_loss_out(loss,moe_aux,loss_div)
 def os1_read_scale(self,u,idx):cfg=self.config;b,t=idx.shape;ud=u.detach().float();cn=F.embedding(idx[:,1:],self.os1_gcode);z=F.pad((ud[:,:t-1]*cn).sum(-1),(0,1));h=z@self.os1_band[:t,:t];g=torch.sigmoid(cfg.os1_gate_slope*(h-cfg.os1_gate_h0)).to(COMPUTE_DTYPE);return((cfg.os1_beta/(1.0-cfg.os1_thr))*g.float()).reshape(b*t)
 def os1_fused_loss(self,x,u,idx,targets):
  cfg=self.config;b,t,c=x.shape;xu=torch.cat([x,u.to(x.dtype)],dim=-1).reshape(b*t,-1);lin=Os1FoldHeadR.apply(xu,self.lm_head.weight.to(dtype=x.dtype),self.os1_codeg,self.os1_wcat,x.reshape(b*t,c));cg=self.os1_read_scale(u,idx)
  if loss_head_applies(x,"mean",cfg.vocab_size,self.lm_head.weight.size(0)):return chunked_loss_mean_os1(lin,targets,xu.detach(),self.os1_codeT,cg,cfg.os1_thr)
  if x.device.type!="cpu":raise RuntimeError("off the CPU the readout trains in the loss head kernels")
  cgc=cg.reshape(-1,1);s=F.linear(xu.detach()[:,c:].float(),self.os1_code.float());head=torch.clamp_min(s-cfg.os1_thr,0.0)*cgc+lin.float();logits=LOGIT_SOFTCAP*torch.tanh(head[:,:cfg.vocab_size]/LOGIT_SOFTCAP);return F.cross_entropy(logits,targets.reshape(-1),ignore_index=-1,reduction="mean")
 def setup_optimizer(self):
  scalar_params=[];resid_params=[];matrix_params=[]
  for block in self.transformer.h:resid_params.append(block.resid_lambda);scalar_params.append(block.x0_lambda);matrix_params.extend(block.attn.parameters());matrix_params.extend(block.mlp.trunk_parameters()if isinstance(block.mlp,MoeMLP)else block.mlp.parameters())
  if self.ngram_proj is not None:matrix_params.extend(self.ngram_proj)
  scalar_params.extend(self.gain_params());wte_params=list(self.transformer.wte.parameters());embedding_params=list(wte_params)
  if self.value_embeds is not None:embedding_params.extend(self.value_embeds.parameters())
  lm_head_params=list(self.lm_head.parameters());dmodel_lr_scale=(self.config.n_embd/768)**-0.5;param_groups=[dict(kind="adamw",params=embedding_params,lr=EMBEDDING_LR*dmodel_lr_scale,betas=(0.8,0.995),eps=1e-10,weight_decay=0.001),dict(kind="adamw",params=lm_head_params,lr=UNEMBEDDING_LR*dmodel_lr_scale,betas=(0.8,0.96),eps=1e-10,weight_decay=0.01),dict(kind="adamw",params=scalar_params,lr=SCALAR_LR,betas=(0.9,0.95),eps=1e-10,weight_decay=0.0),dict(kind="adamw",params=resid_params,lr=SCALAR_LR/RESID_LR_DIV,betas=(0.9,0.95),eps=1e-10,weight_decay=0.0),]
  for shape in sorted({p.shape for p in matrix_params}):params=[p for p in matrix_params if p.shape==shape];param_groups.append(dict(kind="muon",params=params,lr=MATRIX_LR,momentum=0.95,beta2=0.90,ns_steps=5,weight_decay=WEIGHT_DECAY))
  experts=[p for block in self.transformer.h if isinstance(block.mlp,MoeMLP)for p in block.mlp.expert_parameters()]
  for shape in sorted({tuple(p.shape[1:])for p in experts}):params=[p for p in experts if tuple(p.shape[1:])==shape];param_groups.append(dict(kind="muon",params=params,lr=MATRIX_LR,momentum=0.95,beta2=0.90,ns_steps=5,weight_decay=WEIGHT_DECAY,moe_local=True))
  distributed=dist.is_available()and dist.is_initialized()and dist.get_world_size()>1
  if not distributed:raise ValueError("the experts are one a rank and train in the lean optimizer: more than one rank is needed")
  if GRAD_CLIP>0:raise ValueError("GRAD_CLIP needs autograd-held gradients; LeanDistMuonAdamW.accumulate() takes them")
  opt=LeanDistMuonAdamW(param_groups,replica_dtype=COMPUTE_DTYPE,fp32_replicas=embedding_params,deferred=wte_params+(list(self.value_embeds.parameters())if self.value_embeds is not None else[]))
  for group in opt.param_groups:group["initial_lr"]=group["lr"]
  return opt
 def forward(self,idx,targets=None,loss_reduction="mean",emb_grad_sink=None,ngram_emb=None,ve_grad_sinks=None,loss_div=None):
  _b,t=idx.shape;assert t<=self.config.sequence_len;cos,sin=self.cos[:,:t],self.sin[:,:t]
  if emb_grad_sink is not None:x=LeanLookup.apply(idx,self.transformer.wte.weight,emb_grad_sink)
  else:x=self.transformer.wte(idx).to(COMPUTE_DTYPE)
  if self.ngram_proj is not None:
   if ngram_emb is None:ngram_emb=self.ngram_lookup(idx)
   x=x+self.ngram_input(ngram_emb)
  x=norm(x);x0=x;n_layer,n_skip=self.config.n_layer,self.n_skip;skips=[];moe_aux=[]if(self.has_moe and targets is not None and loss_reduction=="mean")else None
  for i,block in enumerate(self.transformer.h):
   if n_skip and i>=n_layer-n_skip:j=i-(n_layer-n_skip);x=x+self.skip_lambda[j:j+1].to(x.dtype)*skips.pop()
   ve=None
   if self.value_embeds is not None and str(i)in self.value_embeds:
    if ve_grad_sinks is not None:ve=LeanLookup.apply(idx,self.value_embeds[str(i)].weight,ve_grad_sinks[str(i)])
    else:ve=self.value_embeds[str(i)](idx).to(COMPUTE_DTYPE)
   x=block(x,x0,cos,sin,ve)if moe_aux is None else block(x,x0,cos,sin,ve,moe_aux=moe_aux)
   if n_skip and i<n_skip:skips.append(x)
  x=norm(x);return self.os1_tail(x,ngram_emb,idx,targets,loss_reduction,loss_div,moe_aux)
def loss_scaled(loss,loss_div):return loss if loss_div is None else loss/loss_div
def build_config(depth=DEPTH,aspect_ratio=ASPECT_RATIO,head_dim=HEAD_DIM,sequence_len=SEQ_LEN,):base_dim=depth*aspect_ratio;n_embd=((base_dim+head_dim-1)//head_dim)*head_dim;n_head=n_embd//head_dim;return GPTConfig(sequence_len=sequence_len,vocab_size=TOKENIZER_VOCAB_SIZE,n_layer=depth,n_head=n_head,n_embd=n_embd,unet=True,value_embeds=VALUE_EMBEDS,qk_scale=QK_SCALE,ngram_orders=NGRAM_ORDERS,ngram_rows=NGRAM_ROWS,ngram_dim=NGRAM_DIM,ngram_proj="separate",ngram_seed=NGRAM_HASH_SEED,ngram_bits=NGRAM_BITS,attn_window=ATTN_WINDOW,attn_long=attn_long_layers(depth),os1_rows=OS1_ROWS,os1_dim=NGRAM_DIM,os1_order=OS1_ORDER,os1_seed=OS1_HASH_SEED,os1_code_seed=OS1_CODE_SEED,os1_a0=OS1_A0*OS1_LR,os1_thr=OS1_THR,os1_beta=OS1_BETA,os1_glin=OS1_GLIN,os1_gate_w=OS1_GATE_W,os1_gate_h0=OS1_GATE_H0,os1_gate_slope=OS1_GATE_SLOPE,os1_bits=1,**moe_config_fields(depth,n_embd,sequence_len),)
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]
def ns_flops(r,c,iters):
 if r>c:per=2*c*r*c+2*c*c*c+2*r*c*c
 else:per=2*r*c*r+2*r*r*r+2*r*r*c
 return iters*per
def plan_layout(shapes,counts,world,ns_steps=5):
 numel=[r*c for r,c in shapes];U=max(numel);per=[U//n for n in numel];units=[]
 for g,cnt in enumerate(counts):
  for s in range(0,cnt,per[g]):units.append((g,s,min(per[g],cnt-s)))
 cost_m=[ns_flops(r,c,ns_steps)for r,c in shapes];ucost=[cost_m[g]*n for g,_s,n in units];n_units=len(units);c=max(1,n_units//world);n_main=min(n_units,c*world);order=sorted(range(n_units),key=lambda i:(-units[i][2],-i));left_ids=sorted(order[:n_units-n_main]);left_set=set(left_ids);main_ids=[i for i in range(n_units)if i not in left_set];deal=sorted(main_ids,key=lambda i:(-ucost[i],i));main=[[]for _ in range(world)]
 for j,uid in enumerate(deal):main[j%world].append(uid)
 for r in range(world):main[r].sort(key=lambda i:(units[i][0],i))
 load=[sum(ucost[i]for i in main[r])for r in range(world)];left=[(units[i][0],units[i][1]+k)for i in left_ids for k in range(units[i][2])];n_left=len(left);rounds=-(-n_left//world)if n_left else 0;single_shape=len({g for g,_m in left})<=1
 if n_left==0:route="none"
 else:route="allreduce" if(2*n_left<rounds*world or not single_shape)else "slots"
 cap=rounds if route=="slots" else n_left;left_owner,left_round,cnt=[],[],[0]*world
 for g,_m in left:r=min((q for q in range(world)if cnt[q]<cap),key=lambda q:(load[q],q));left_owner.append(r);left_round.append(cnt[r]);cnt[r]+=1;load[r]+=cost_m[g]
 return dict(U=U,per=per,units=units,c=c,main=main,left=left,route=route,rounds=rounds,left_owner=left_owner,left_round=left_round,load=load,cost_m=cost_m,numel=numel)
class LeanDistMuonAdamW(torch.optim.Optimizer):
 handles_gradient_sync=True
 def __init__(self,param_groups,*,replica_dtype,fp32_replicas=(),deferred=()):
  super().__init__(param_groups,defaults={});self.rank=dist.get_rank();self.world=dist.get_world_size();self.replica_dtype=replica_dtype;self._fp32_ids={id(p)for p in fp32_replicas};self._deferred_ids={id(p)for p in deferred};self._deferred_got=set();self._acc_count=0
  for group in self.param_groups:
   group["_lr_t"]=torch.tensor(float(group["lr"]),dtype=torch.float32);group["_wd_t"]=torch.tensor(float(group["weight_decay"]),dtype=torch.float32)
   if group["kind"]=="adamw":beta1,beta2=group["betas"];group["_one_minus_beta1_t"]=torch.tensor(1-beta1,dtype=torch.float32);group["_one_minus_beta2_t"]=torch.tensor(1-beta2,dtype=torch.float32);group["_eps_t"]=torch.tensor(float(group["eps"]),dtype=torch.float32)
   elif group["kind"]=="muon":group["_momentum_t"]=torch.tensor(float(group["momentum"]),dtype=torch.float32);group["_one_minus_momentum_t"]=torch.tensor(1-float(group["momentum"]),dtype=torch.float32);group["_one_minus_beta2_t"]=torch.tensor(1-float(group["beta2"]),dtype=torch.float32)
  self._build();self._build_local();Gflat=self.Gm.view(-1);self._fold_bases=[Gflat[run["start"]:run["end"]]for run in self._runs];self._fold_params=[list(run["params"])for run in self._runs];slab={id(p)for run in self._runs for p in run["params"]};rest=[(p,t)for p,t in zip(self._acc_params,self._acc_targets)if id(p)not in slab];self._rest_params=[p for p,_t in rest];self._rest_targets=[t for _p,t in rest];self._fold=lean_fold_slab;self._rest_fn=lean_acc_rest;self._s12_n={"copies":0,"adds":0,"flags":0};self._math={"muon":muon_math_body,"adamw":adamw_math_body};self._sched_bufs={};self._ovl=opt_overlap_rs
 def compile_parts(self,backend):self._fold=torch.compile(lean_fold_slab,backend=backend,dynamic=False);self._rest_fn=torch.compile(lean_acc_rest,backend=backend,dynamic=False);self._math={"muon":torch.compile(muon_math_body,backend=backend,dynamic=False),"adamw":torch.compile(adamw_math_body,backend=backend,dynamic=False)};self._ovl=torch.compile(opt_overlap_rs,backend=backend,dynamic=False)
 def enable_stage(self):self._staged={};self._stage_dev=None;self._stage_n={"steps":0,"keys":0,"hits":0,"misses":0}
 @torch.no_grad()
 def stage_schedule(self,non_blocking=False):
  f32=torch.float32;keys,vecs=[],[]
  def adamw(st,group):
   if st and("adamw",id(st))in self._sched_bufs:beta1,beta2=group["betas"];n=st["step"]+1;vecs.append(torch.stack([torch.tensor(float(group["lr"]),dtype=f32),torch.tensor(float(group["weight_decay"]),dtype=f32),group["_one_minus_beta1_t"].reshape(()),group["_one_minus_beta2_t"].reshape(()),group["_eps_t"].reshape(()),torch.tensor(1-beta1**n,dtype=f32),torch.tensor(1-beta2**n,dtype=f32)]));keys.append(("adamw",id(st)))
  def muon(group,st,shape):
   if("muon",id(st))in self._sched_bufs:vecs.append(torch.stack([torch.tensor(float(group["momentum"]),dtype=f32),torch.tensor(float(1-group["momentum"]),dtype=f32),torch.tensor(float(1-group["beta2"]),dtype=f32),torch.tensor(float(group["lr"]*max(1.0,shape[-2]/shape[-1])**0.5),dtype=f32),torch.tensor(float(group["weight_decay"]),dtype=f32)]));keys.append(("muon",id(st)))
  for S in self._small:adamw(S["st"],S["group"])
  for r in self._rows:adamw(r["st"],r["group"])
  for seg in self._own:muon(self._mg[seg["g"]],seg["st"],seg["shape"])
  for L in self._left:
   if L["own"]if L["kind"]=="allreduce" else L["owns"]:muon(self._mg[L["g"]],L["st"],L["shape"])
  for L in self._local:muon(L["group"],L["st"],L["shape"])
  self._staged=dict(zip(keys,vecs))
  if keys:
   cpu=torch.cat(vecs);dev=self._stage_dev
   if dev is None or dev.numel()!=cpu.numel():dev=self._stage_dev=torch.empty(cpu.shape,dtype=f32,device=self._sched_bufs[keys[0]].device)
   dev.copy_(cpu,non_blocking=non_blocking);torch._foreach_copy_([self._sched_bufs[k]for k in keys],list(dev.split([v.numel()for v in vecs])))
  self._stage_n["steps"]+=1;self._stage_n["keys"]=len(keys)
 def _sched(self,key,vals,device):
  cpu=torch.stack([v.reshape(())for v in vals]);buf=self._sched_bufs.get(key)
  if buf is None:buf=torch.empty(cpu.shape,dtype=cpu.dtype,device=device);self._sched_bufs[key]=buf
  staged=getattr(self,"_staged",None)
  if staged is not None:
   want=staged.get(key)
   if want is not None and torch.equal(want,cpu):self._stage_n["hits"]+=1;return buf
   self._stage_n["misses"]+=1;staged.pop(key,None)
  buf.copy_(cpu);return buf
 def _adopt(self,p):
  if p.dtype!=self.replica_dtype:p.data=p.data.to(self.replica_dtype)
 def _build(self):
  f32=torch.float32;W=self.world;self._small,self._rows,self._left,self._own,self._runs=[],[],[],[],[];self._acc_params,self._acc_targets=[],[]
  for g in self.param_groups:
   if g["kind"]!="adamw":continue
   small=[p for p in g["params"]if p.numel()<1024]
   if small:
    sizes=[p.numel()for p in small];n=sum(sizes);master=torch.cat([p.detach().reshape(-1).to(f32)for p in small]);pay=torch.zeros(n,dtype=f32,device=master.device);o=0
    for p,s in zip(small,sizes):self._acc_params.append(p);self._acc_targets.append(pay[o:o+s].view(p.shape));o+=s
    self._small.append(dict(group=g,params=small,sizes=sizes,master=master,pay=pay,st={}))
   for p in g["params"]:
    if p.numel()<1024:continue
    if p.shape[0]%W:raise ValueError(f"AdamW parameter first dimension {p.shape[0]} must be divisible by world size {W}")
    rows=p.shape[0]//W;start=self.rank*rows;master=p.detach()[start:start+rows].to(f32,copy=True);rd=f32 if id(p)in self._fp32_ids else self.replica_dtype
    if p.dtype!=rd:p.data=p.data.to(rd)
    tail=tuple(p.shape[1:]);out=None if rd==f32 else torch.empty((rows,*tail),dtype=rd,device=p.device);self._rows.append(dict(group=g,p=p,rows=rows,start=start,master=master,out=out,st={},acc=None,grad_slice=torch.empty((rows,*tail),dtype=f32,device=p.device)))
  self._mg=[g for g in self.param_groups if g["kind"]=="muon" and g["params"]and not g.get("moe_local")];shapes=[tuple(g["params"][0].shape)for g in self._mg];plan=plan_layout(shapes,[len(g["params"])for g in self._mg],W,ns_steps=self._mg[0]["ns_steps"]);U,c=plan["U"],plan["c"];CU=c*U;dev=self._mg[0]["params"][0].device;big=next(s for s in shapes if s[0]*s[1]==U);self._coll_in=(W*c,*big);self._coll_out=(c,*big);bf16=torch.bfloat16;self.Gm=torch.zeros(W,CU,dtype=bf16,device=dev);self.mine_g=torch.empty(1,CU,dtype=bf16,device=dev);self.Mm=torch.zeros(1,CU,dtype=f32,device=dev);reuse=self.replica_dtype==bf16;self.Wm=self.Gm if reuse else torch.zeros(W,CU,dtype=self.replica_dtype,device=dev);self.mine_out=None if self.replica_dtype==f32 else torch.empty(1,CU,dtype=self.replica_dtype,device=dev);Gflat=self.Gm.view(-1);items=[]
  for r in range(W):
   seg=None
   for j,uid in enumerate(plan["main"][r]):
    g,s,n=plan["units"][uid];ne=plan["numel"][g]
    for k in range(n):
     p=self._mg[g]["params"][s+k];items.append((r*CU+j*U+k*ne,p))
     if r==self.rank:self.Mm.view(-1)[j*U+k*ne:j*U+(k+1)*ne].copy_(p.detach().reshape(-1))
    if r==self.rank:
     if seg is not None and seg["g"]==g and seg["b"]==j*U and seg["full"]:seg["b"]=j*U+n*ne;seg["k"]+=n
     else:seg=dict(g=g,a=j*U,b=j*U+n*ne,k=n,shape=shapes[g],st={});self._own.append(seg)
     seg["full"]=(n==plan["per"][g])
  items.sort(key=lambda t:t[0])
  for off,p in items:
   ne=p.numel();self._adopt(p);self._acc_params.append(p);self._acc_targets.append(Gflat[off:off+ne].view(p.shape))
   if self._runs and self._runs[-1]["end"]==off:run=self._runs[-1];run["end"]+=ne;run["params"].append(p);run["sizes"].append(ne)
   else:self._runs.append(dict(start=off,end=off+ne,params=[p],sizes=[ne]))
  route=plan["route"]
  if route=="allreduce":
   by_g=OrderedDict()
   for j,(g,m)in enumerate(plan["left"]):by_g.setdefault(g,[]).append((m,plan["left_owner"][j]))
   for g,lst in by_g.items():
    shape=shapes[g];params=[self._mg[g]["params"][m]for m,_o in lst];own=[i for i,(_m,o)in enumerate(lst)if o==self.rank];master=torch.stack([params[i].detach().to(f32)for i in own])if own else None;Gl=torch.zeros(len(lst),*shape,dtype=bf16,device=dev);Wl=Gl if reuse else torch.zeros(len(lst),*shape,dtype=self.replica_dtype,device=dev)
    for i,p in enumerate(params):self._adopt(p);self._acc_params.append(p);self._acc_targets.append(Gl[i])
    self._left.append(dict(kind="allreduce",g=g,shape=shape,params=params,own=own,Gl=Gl,Wl=Wl,master=master,st={},idx=torch.tensor(own,dtype=torch.long,device=dev)if own else None))
  elif route=="slots":
   g=plan["left"][0][0];shape=shapes[g]
   for q in range(plan["rounds"]):
    slots=[(plan["left_owner"][j],self._mg[g]["params"][m])for j,(_g,m)in enumerate(plan["left"])if plan["left_round"][j]==q];mine_p=next((p for o,p in slots if o==self.rank),None);master=torch.zeros(1,*shape,dtype=f32,device=dev)
    if mine_p is not None:master[0].copy_(mine_p.detach())
    Gl=torch.zeros(W,*shape,dtype=bf16,device=dev);Wl=Gl if reuse else torch.zeros(W,*shape,dtype=self.replica_dtype,device=dev)
    for o,p in slots:self._adopt(p);self._acc_params.append(p);self._acc_targets.append(Gl[o])
    self._left.append(dict(kind="slots",g=g,shape=shape,slots=slots,owns=mine_p is not None,Gl=Gl,Wl=Wl,master=master,st={},mine=torch.empty(1,*shape,dtype=bf16,device=dev),out=None if self.replica_dtype==f32 else torch.empty(1,*shape,dtype=self.replica_dtype,device=dev)))
 def _build_local(self):
  f32=torch.float32;self._local=[]
  for g in self.param_groups:
   if not g.get("moe_local")or not g["params"]:continue
   ps=g["params"];shape=tuple(ps[0].shape[1:])
   if any(p.dim()!=3 or tuple(p.shape[1:])!=shape for p in ps):raise ValueError("a local group holds expert stacks [n, r, c] of one matrix shape")
   counts=[p.shape[0]for p in ps];master=torch.cat([p.detach().to(f32)for p in ps]);acc=torch.zeros_like(master)
   for p,view in zip(ps,acc.split(counts)):
    if p.dtype!=self.replica_dtype:p.data=p.data.to(self.replica_dtype)
    self._acc_params.append(p);self._acc_targets.append(view)
   self._local.append(dict(group=g,params=ps,counts=counts,shape=shape,master=master,acc=acc,st={},scale=1.0/self.world))
 @torch.no_grad()
 def accumulate(self):
  first=self._acc_count==0;self._fold(self._fold_bases,[[p.grad if p.grad is not None else torch.zeros_like(p)for p in ps]for ps in self._fold_params],first);grads=[p.grad if p.grad is not None else torch.zeros_like(p)for p in self._rest_params];self._acc_rest_s12(self._rest_targets,grads,first)
  for r in self._rows:
   if id(r["p"])in self._deferred_ids:continue
   g=r["p"].grad
   if g is None:g=torch.zeros_like(r["p"])
   if first:
    if r["acc"]is None:r["acc"]=torch.empty(r["p"].shape,dtype=torch.float32,device=g.device)
    r["acc"].copy_(g)
   else:r["acc"].add_(g)
  for group in self.param_groups:
   for p in group["params"]:p.grad=None
  self._acc_count+=1
 def _acc_rest_s12(self,targets,grads,first):
  if targets:self._rest_fn(targets,grads,first,None,None)
  self._s12_n["copies" if first else "adds"]+=1
 def set_deferred_grad(self,p,grad):
  for r in self._rows:
   if r["p"]is p:r["acc"]=grad
  self._deferred_got.add(id(p))
 def zero_grad(self,set_to_none=True):
  super().zero_grad(set_to_none=set_to_none)
  for r in self._rows:
   if id(r["p"])in self._deferred_ids:r["acc"]=None
  self._deferred_got=set()
 def _adamw_math(self,p,grad,st,group):
  beta1,beta2=group["betas"];lr_t,wd_t=group["_lr_t"],group["_wd_t"];lr_t.fill_(group["lr"]);wd_t.fill_(group["weight_decay"])
  if not st:st["step"]=0;st["exp_avg"]=torch.zeros_like(p);st["exp_avg_sq"]=torch.zeros_like(p);st["bias1_t"]=torch.tensor(0.0,dtype=torch.float32);st["bias2_t"]=torch.tensor(0.0,dtype=torch.float32)
  st["step"]+=1;st["bias1_t"].fill_(1-beta1**st["step"]);st["bias2_t"].fill_(1-beta2**st["step"]);sched=self._sched(("adamw",id(st)),(lr_t,wd_t,group["_one_minus_beta1_t"],group["_one_minus_beta2_t"],group["_eps_t"],st["bias1_t"],st["bias2_t"]),p.device);self._math["adamw"](p,grad,st["exp_avg"],st["exp_avg_sq"],sched)
 def _muon_math(self,group,g,w,st,shape):self._math["muon"](*self._mu_args(group,g,w,st,shape))
 def _mu_args(self,group,g,w,st,shape):
  k=g.shape[0]
  if "momentum_buffer" not in st:st["momentum_buffer"]=torch.zeros(k,*shape,dtype=torch.float32,device=g.device);st_shape=(k,shape[-2],1)if shape[-2]>=shape[-1]else(k,1,shape[-1]);st["second_momentum_buffer"]=torch.zeros(st_shape,dtype=torch.float32,device=g.device)
  momentum_t=group["_momentum_t"];one_minus_momentum_t=group["_one_minus_momentum_t"];one_minus_beta2_t=group["_one_minus_beta2_t"];lr_t,wd_t=group["_lr_t"],group["_wd_t"];momentum_t.fill_(group["momentum"]);one_minus_momentum_t.fill_(1-group["momentum"]);one_minus_beta2_t.fill_(1-group["beta2"]);lr_t.fill_(group["lr"]*max(1.0,shape[-2]/shape[-1])**0.5);wd_t.fill_(group["weight_decay"]);sched=self._sched(("muon",id(st)),(momentum_t,one_minus_momentum_t,one_minus_beta2_t,lr_t,wd_t),g.device);return(g,w,st["momentum_buffer"],st["second_momentum_buffer"],sched,tuple(tuple(c)for c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]),COMPUTE_DTYPE==torch.bfloat16,tuple(shape))
 @torch.no_grad()
 def step(self):
  missing=self._deferred_ids-self._deferred_got
  if missing:raise RuntimeError(f"{len(missing)} deferred parameter(s) without this step's gradient (set_deferred_grad was not called for them)")
  AVG,SUM=dist.ReduceOp.AVG,dist.ReduceOp.SUM;units=[]
  for L in self._local:
   g=L["acc"]
   if L["scale"]!=1.0:g.mul_(L["scale"])
   units.append(self._mu_args(L["group"],g,L["master"],L["st"],L["shape"]))
  self._ovl([r["grad_slice"]for r in self._rows],[r["acc"]for r in self._rows],self.mine_g.view(self._coll_out),self.Gm.view(self._coll_in),[S["pay"]for S in self._small],units);row_grads=[r["grad_slice"]for r in self._rows];g_main=self.mine_g.float().div_(self.world);left_grads=[]
  for L in self._left:
   if L["kind"]=="allreduce":dist.all_reduce(L["Gl"],op=SUM);left_grads.append(L["Gl"].float().div_(self.world))
   else:dist.reduce_scatter_tensor(L["mine"],L["Gl"],op=SUM);left_grads.append(L["mine"].float().div_(self.world))
  for S in self._small:self._adamw_math(S["master"],S["pay"],S["st"],S["group"]);torch.split_with_sizes_copy(S["master"],S["sizes"],out=[p.data.view(-1)for p in S["params"]])
  for r,rg in zip(self._rows,row_grads):
   self._adamw_math(r["master"],rg,r["st"],r["group"])
   if r["out"]is None:dist.all_gather_into_tensor(r["p"].data,r["master"])
   else:r["out"].copy_(r["master"]);dist.all_gather_into_tensor(r["p"].data,r["out"])
  for seg in self._own:g=g_main[0,seg["a"]:seg["b"]].view(seg["k"],*seg["shape"]);w=self.Mm[0,seg["a"]:seg["b"]].view(seg["k"],*seg["shape"]);self._muon_math(self._mg[seg["g"]],g,w,seg["st"],seg["shape"])
  src=self.Mm
  if self.mine_out is not None:self.mine_out.copy_(self.Mm);src=self.mine_out
  dist.all_gather_into_tensor(self.Wm.view(self._coll_in),src.view(self._coll_out));Wflat=self.Wm.view(-1)
  for run in self._runs:torch._foreach_copy_([p.data.view(-1)for p in run["params"]],list(Wflat[run["start"]:run["end"]].split(run["sizes"])))
  for L,lg in zip(self._left,left_grads):
   group=self._mg[L["g"]]
   if L["kind"]=="allreduce":
    if L["own"]:self._muon_math(group,lg.index_select(0,L["idx"]),L["master"],L["st"],L["shape"])
    L["Wl"].zero_()
    if L["own"]:L["Wl"].index_copy_(0,L["idx"],L["master"].to(L["Wl"].dtype))
    dist.all_reduce(L["Wl"],op=SUM);n=L["Wl"][0].numel();torch._foreach_copy_([p.data.view(-1)for p in L["params"]],list(L["Wl"].view(-1).split([n]*len(L["params"]))))
   else:
    if L["owns"]:self._muon_math(group,lg,L["master"],L["st"],L["shape"])
    src=L["master"]
    if L["out"]is not None:L["out"].copy_(L["master"]);src=L["out"]
    dist.all_gather_into_tensor(L["Wl"],src);torch._foreach_copy_([p.data for _o,p in L["slots"]],[L["Wl"][o]for o,_p in L["slots"]])
  for L in self._local:torch._foreach_copy_([p.data for p in L["params"]],list(L["master"].split(L["counts"])))
  for group in self.param_groups:
   for p in group["params"]:p.grad=None
  self._acc_count=0
def opt_overlap_rs(row_outs,row_srcs,slab_out,slab_in,pays,units):
 fc,grp=MOE_FUNCOL,dist.group.WORLD;rows=[fc.reduce_scatter_tensor(s,"avg",0,grp)for s in row_srcs];slab=fc.reduce_scatter_tensor(slab_in,"sum",0,grp);small=[fc.all_reduce(p,"avg",grp)for p in pays]
 for u in units:muon_math_body(*u)
 for o,r in zip(row_outs,rows):o.copy_(r)
 slab_out.copy_(slab)
 for p,r in zip(pays,small):p.copy_(r)
def lean_fold_slab(bases,groups,first):
 for base,grads in zip(bases,groups):
  flat=torch.cat([g.reshape(-1)for g in grads])
  if first:base.copy_(flat)
  else:base.add_(flat)
def lean_acc_rest(targets,grads,first,flag,value):
 for t,g in zip(targets,grads):
  if first:t.copy_(g)
  else:t.add_(g)
 if flag is not None:flag.copy_(value)
def muon_math_body(g,w,momentum,second,sched,coeffs,ns_bf16,shape):
 momentum_t,omm,omb2,lr,wd_t=sched[0],sched[1],sched[2],sched[3],sched[4];momentum.lerp_(g,omm);g=g.lerp_(momentum,momentum_t);x=g.bfloat16()if ns_bf16 else g;x=x/(x.norm(dim=(-2,-1),keepdim=True)*1.01+1e-6);squeezed=x.dim()==3 and x.size(0)==1
 if squeezed:x=x.squeeze(0)
 if x.shape[-2]>x.shape[-1]:
  for a,b,c in coeffs:xtx=x.mT@x;x=a*x+x@(b*xtx+c*(xtx@xtx))
 else:
  for a,b,c in coeffs:xxt=x@x.mT;x=a*x+(b*xxt+c*(xxt@xxt))@x
 if squeezed:x=x.unsqueeze(0)
 g=x;red_dim=-1 if shape[-2]>=shape[-1]else-2;v_mean=g.float().square().mean(dim=red_dim,keepdim=True);red_dim_size=g.size(red_dim);v_norm=(v_mean.sum(dim=(-2,-1),keepdim=True)*red_dim_size).sqrt();second.lerp_(v_mean,omb2);step_size=second.clamp_min(1e-10).rsqrt();scaled=(v_mean*red_dim_size)*step_size.square();v_norm_new=scaled.sum(dim=(-2,-1),keepdim=True).sqrt();g=g*(step_size*(v_norm/v_norm_new.clamp_min(1e-10))).to(g.dtype);mask=(g*w)>=0;w.sub_(lr*g+lr*wd_t*w*mask)
def adamw_math_body(p,grad,exp_avg,exp_avg_sq,sched):lr,wd_t,omb1,omb2=sched[0],sched[1],sched[2],sched[3];eps_t,bias1_t,bias2_t=sched[4],sched[5],sched[6];p.mul_(1-lr*wd_t);exp_avg.lerp_(grad,omb1);exp_avg_sq.lerp_(grad.square(),omb2);update=exp_avg/bias1_t/((exp_avg_sq/bias2_t).sqrt()+eps_t);p.sub_(lr*update)
def accumulate_gradients(optimizer):
 accumulate=getattr(optimizer,"accumulate",None)
 if accumulate is not None:accumulate()
def moe_layers_of(model):return[blk.mlp for blk in model.transformer.h if isinstance(blk.mlp,MoeMLP)]
def mlp8_units(model):return[blk.mlp.shared if isinstance(blk.mlp,MoeMLP)else blk.mlp for blk in model.transformer.h]
def moe_check_launch(world_size,args):
 if world_size<2:raise ValueError("the experts are one a rank: the mixture of experts needs at least two ranks")
 cap=moe_capacity_for(MOE_CF,args.seq_len,MOE_TOPK,world_size)
 if cap>256:raise ValueError(f"capacity {cap} above 256, the bf16-exact count bound")
 if args.device_batch_size>MOE_B_MAX:raise ValueError(f"device batch {args.device_batch_size} above {MOE_B_MAX}")
def moe_param_counts(model,num_params_local):layers=moe_layers_of(model);per_exp=sum(2*m.hidden*m.router.size(1)for m in layers);n_local=layers[0].n_local if layers else 0;n_exp=layers[0].n_exp if layers else 0;topk=layers[0].topk if layers else 0;trunk=num_params_local-n_local*per_exp;return{"total":trunk+n_exp*per_exp,"active":trunk+topk*per_exp,"routed":n_exp*per_exp,"routed_on_rank":n_local*per_exp,"trunk":trunk}
def moe_loss_out(loss,moe_aux,loss_div):
 aux=moe_aux[0][0]
 for a,_st in moe_aux[1:]:aux=aux+a
 stats=torch.cat([loss_scaled(loss.detach(),loss_div).float().reshape(1)]+[st for _a,st in moe_aux]);return loss_scaled(loss+aux,loss_div),stats
def moe_expert_items(model):names={id(p):n for n,p in model.named_parameters()};return[(names[id(p)],p)for m in moe_layers_of(model)for p in m.expert_parameters()]
@torch.no_grad()
def moe_gather_experts(model,rank,world,source=None):
 items=moe_expert_items(model);srcs=source if source is not None else[p.detach()for _n,p in items];out={}
 for(name,_p),src in zip(items,srcs):
  full=torch.empty((world*src.size(0),)+tuple(src.shape[1:]),dtype=src.dtype,device=src.device);dist.all_gather_into_tensor(full,src.contiguous())
  if rank==0:out[name]=full.to("cpu")
  del full
 return out if rank==0 else None
@torch.no_grad()
def moe_rescue_refresh(model,copies):
 items=moe_expert_items(model)
 if copies is None:copies=[torch.empty_like(p)for _n,p in items]
 torch._foreach_copy_(copies,[p.detach()for _n,p in items]);return copies
@torch.no_grad()
def moe_stats_line(model,stats,step,world):layers=moe_layers_of(model);n_exp=layers[0].n_exp;per=stats[1:].detach().float().reshape(len(layers),3+n_exp).clone();dist.all_reduce(per,op=dist.ReduceOp.SUM);per=per/world;per=per.cpu();keep,lb,z,f=per[:,0],per[:,1],per[:,2],per[:,3:].clamp_min(1e-12);ent=-(f*f.log()).sum(-1)/math.log(n_exp);peak=f.max(-1).values*n_exp;return (f"moe step {step}: drop={1.0 - float(keep.mean()):.5f} drop_max={1.0 - float(keep.min()):.5f} lb={float(lb.mean()):.4f} z={float(z.mean()):.3f} load_entropy={float(ent.mean()):.4f} load_entropy_min={float(ent.min()):.4f} load_peak={float(peak.max()):.3f} drop_by_block=[{','.join(f'{1.0 - v:.4f}' for v in keep.tolist())}] blocks=[{','.join(str(m.layer) for m in layers)}]")
def moe_record(config):return{"layers":list(config.moe_layers),"experts":config.moe_experts,"topk":config.moe_topk,"cf":MOE_CF,"capacity":config.moe_capacity,"hidden":config.moe_hidden,"shared":config.moe_shared,"dispatch":config.moe_dispatch,"aux":MOE_AUX,"zloss":MOE_ZLOSS,"lr":MATRIX_LR,"router_std":MOE_ROUTER_STD,"shared_mlp":config.moe_shared_mlp}
def save_checkpoint(path,model,meta,extra=None):path.parent.mkdir(parents=True,exist_ok=True);payload={"model_state":{k:v.detach().to("cpu",torch.float32)for k,v in model.state_dict().items()}|(extra or{}),"model_config":asdict(model.config),"meta":meta,};torch.save(payload,path)
def save_rescue_checkpoint(path,model,state,meta,extra=None):path.parent.mkdir(parents=True,exist_ok=True);payload={"model_state":{k:v.detach().to("cpu",torch.float32)for k,v in state.items()}|(extra or{}),"model_config":asdict(model.config),"meta":meta,};tmp_path=path.with_name(path.name+".tmp");torch.save(payload,tmp_path);os.replace(tmp_path,path)
def check_declared_length(model,trained_seq_len):
 declared=asdict(model.config)["sequence_len"]
 if declared!=trained_seq_len:raise RuntimeError(f"model_config.sequence_len={declared} differs from the trained window length {trained_seq_len}")
def load_for_eval(checkpoint_path,device):
 global COMPUTE_DTYPE,FUSED_ATTENTION,ATTN_BAND_KERNEL;COMPUTE_DTYPE=compute_dtype_for(device);FUSED_ATTENTION=fused_attention_for(device);ATTN_BAND_KERNEL=False;setup_mlp_kernel(torch.device(device));setup_attn8_kernel(torch.device(device));payload=torch.load(checkpoint_path,map_location="cpu",weights_only=True,mmap=True);config=GPTConfig(**payload["model_config"])
 if config.moe_layers:MOE_EP.update(rank=0,world=1,attach=True)
 try:
  model=GPT(config)
 finally:
  MOE_EP["attach"]=False
 model.init_weights();state=dict(payload["model_state"]);experts = {i: (state.pop(f"transformer.h.{i}.mlp.w_fc"), state.pop(f"transformer.h.{i}.mlp.w_proj"))for i in config.moe_layers};ngram_table,ngram_scale=state.pop("ngram_table",None),state.pop("ngram_scale",None);os1_table,os1_scale=state.pop("os1_table"),state.pop("os1_scale");model.load_state_dict(state,strict=True)
 for i,(w_fc,w_proj)in experts.items():model.transformer.h[i].mlp.attach_experts(w_fc,w_proj)
 model.to(device)
 if config.ngram_orders:model.attach_ngram_table(ngram_table,ngram_scale)
 model.attach_os1_table(os1_table,os1_scale);model.eval();attn_band_attach(model);os1_wcat_refresh(model);mlp_refresh_scales(model);attn8_refresh_scales(model);return model
def lr_multiplier(step,target_steps,warmdown,final):
 if step<WARMUP_STEPS:return(step+1)/max(1,WARMUP_STEPS)
 warmdown_steps=round(warmdown*target_steps)
 if step<=target_steps-warmdown_steps:return 1.0
 progress=(target_steps-step)/max(1,warmdown_steps);return progress+(1.0-progress)*final
def lr_multiplier_by_clock(step,elapsed_fraction,warmdown,final):
 if step<WARMUP_STEPS:return(step+1)/max(1,WARMUP_STEPS)
 if elapsed_fraction<=1.0-warmdown:return 1.0
 progress=max(0.0,1.0-elapsed_fraction)/warmdown;return progress+(1.0-progress)*final
def muon_momentum_schedule(step,sched_progress):
 warmdown_start=1.0-WARMDOWN_RATIO
 if sched_progress>warmdown_start:return 0.97-0.07*min(1.0,(sched_progress-warmdown_start)/WARMDOWN_RATIO)
 frac=min(step/300,1.0);return(1-frac)*0.85+frac*0.97
def sched_line(label,values):step,progress,lrm,rows_lrm,momentum=values;return (f"sched1 {label} {step}: progress {progress:.6f} dense_lrm {lrm:.6f} rows_lrm {rows_lrm:.6f} momentum {momentum:.6f}")
RELAY_SHARE = 0.02
RELAY_CAP = 65536.0
RELAY_SEED = 62
RELAY_OPEN, RELAY_PAIR = "([{<", {")": "(", "]": "[", "}": "{", ">": "<"}
RELAY_BR_RE = re.compile(r"[()\[\]{}<>]")
RELAY_SHOTS, RELAY_POOL = (11, 11), 2048
RELAY_DYCK_MAX2, RELAY_CLASS_W = 96, (1, 2, 2)
RELAY_CLASS_FLOOR = (1, 1, 64)
RELAY_SEP3, RELAY_TRIES3, RELAY_K2_SHOTS = " \n\n", 32, 2
RELAY_PAIRS = tuple(tuple(w + ":" for w in p.split("/")) for p in """
Brackets/Closers Symbols/Ending Series/Remainder String/Tail Pattern/Suffix Bakery/Napkin Backpack/Racket
Bottle/Charcoal Melon/Shark Breeze/Fork Peninsula/Scooter Planet/Moor Squid/Apron Atlas/Bamboo
Pickle/Cushion Mountain/Sled Cobalt/Violin Ocean/Robin Flame/Savanna Tapestry/Puddle Trophy/Cricket
Desert/Easel Moss/Saddle Pendant/Raven Broom/Tractor Pinecone/Tomato Grater/Sculpture Tin/Glass Oak/Jackal
Ribbon/Button Inlet/Island Chord/Plaster Quartz/Drum Mop/Gear Tunic/Badge Crab/Glove Sonata/Sponge
Pulley/Bowl Oven/Snow Cotton/Meteor Cardigan/Rapids Hail/Cocoa Lion/Scarf Baboon/Isthmus Ladle/River
Trumpet/Tambourine Sparrow/Paddle Plate/Llama Bugle/Corn Chisel/Hoe Gate/Cheetah Mango/Creek
Castle/Cardboard Waterfall/Barge Xylophone/Comb Fjord/Temple Pitcher/Copper Beret/Mitten Curtain/Hammer
Bicycle/Van Badger/Storm Gull/Tide Bolt/Timber Basin/Teapot Anthem/Rainbow Avenue/Nebula Jade/Skillet
Rice/Rattle Cavern/Soap Volcano/Iron Cedar/Kangaroo Postcard/Banner Spring/Umbrella Drawer/Willow
Pearl/Paper Cabinet/Cathedral Bonnet/Ticket Blender/Theater Sandal/Melody Ridge/Butte Drill/Bus Zinc/Goblet
Opera/Arch Trout/Reef Hollow/Flute Toaster/Cellar Thunder/Plateau Juice/Yogurt Guitar/Gown Bay/Violet
Steam/Acorn Shelf/Rope Cabin/Cloak Ivy/Delta Axe/Emerald Terrace/Sapphire Stew/Jungle Brush/Yacht
Pump/Garage Mill/Porch Juniper/Dew Clam/Pumpkin Whisk/Lighthouse Gulf/Skunk Vanilla/Ferret Rhythm/Heath
Tundra/Orchard Swamp/Workshop Boot/Daisy Stove/Tiger Dolphin/Stadium Rabbit/Hedgehog Window/Oyster
Panda/Telescope Coast/Zebra Tiara/Colander Salmon/Candle Cinnamon/Oasis Balcony/Mahogany Ginger/Deer
Plum/Pine Saucer/Microscope Penguin/Olive Alley/Carrot Birch/Coat Woodland/Lettuce Eraser/Whistle
Platypus/Necklace Ruler/Poster Sleeve/Lemon Statue/Pepper Swan/Rug Door/Dome Onion/Bridle Pillow/Bean
Helmet/Coin Cove/Beaver Sand/Cider Orchid/Dock Lunchbox/Thermos Sunrise/Pharmacy Boulder/Bread
Squirrel/Barley Valve/Crater Robe/Library Salad/Ruby Domino/Chalk Knoll/Gully Maple/Rain Denim/Wolf
Gravel/Coral Galaxy/Eclipse Clock/Hyena Sandwich/Buckle Monument/Needle Garden/Lynx Velvet/Slate Basket/Elk
Waffle/Clover Spoon/Calendar Owl/Herring Hare/Beach Vase/Plaza Tortoise/Glider Lentil/Sequoia Brooch/Almond
Wheat/Chair Brook/Geyser Moon/Falcon Sweater/Locket Panther/Jaguar Syrup/Canvas Lotus/Museum Wharf/Rocket
Apple/Nickel Pebble/Bucket Cup/Summit Harmony/Marble Bluff/Elm Taxi/Flag Peach/Carpet Tram/Ballad
Clay/Symphony Mud/Thicket Tassel/Tunnel Harp/Whale Journal/Cypress Aspen/Gorilla Cherry/Fountain
Mosaic/Puma Canteen/Stamp Wallet/Cement Bee/Bear Nail/Steppe Earring/Cable Kimono/Poppy Butterfly/Muffin
Glade/Globe Bronze/Purse Walnut/Silk Garlic/Tulip Kite/Silver Graphite/Saw Koala/Thistle Palette/Canyon
Nutmeg/Tower Giraffe/Ripple Radish/Station Moose/Twilight Flask/Spider Seal/Fence Pencil/Lantern
Amber/Chorus Spark/Potato Celery/Ferry Pancake/Cabbage Moth/Prairie Beetle/Shovel Lagoon/Ravine Vest/Atoll
Wax/Mist Hat/Bog Screw/Wheel Cottage/Anvil Javelin/Linen Envelope/Compass Marsh/Butter Barn/Lily Mink/Steel
Quilt/Pottery Camel/Frog Lobster/Walrus Crown/Wire Cloud/Trolley Oat/Turtle Belt/Chapel Mesa/Biscuit
Frame/Orbit Towel/Grape Kayak/Raft Forest/Valley Pelican/Possum Gecko/Cello Pond/Grove Truck/Estuary
Soil/Mug Canoe/Toad Magnet/Lava Wool/Weasel Slipper/Raccoon Ladder/Meadow Collar/Hinge Soup/Spire
Hurdle/Feather Padlock/Banjo Ember/Shawl Organ/Lamp Wombat/Pear Sunset/Noodle Turnip/Satellite Harbor/Rake
Pocket/Overture Hedge/Lens Strait/Sock Wrench/Eagle Honey/Bridge Courtyard/Jam Tray/Coyote Axle/Comet
Ink/Turban Pier/Kettle Wagon/Hill Stocking/Snail Dragonfly/Zipper Piano/Harness Dune/Jacket Coffee/Diary
Lizard/Bison Prism/Glacier Latch/Gallery Gorge/Wasp Lemur/Medal Lake/Funnel Balloon/Bench Frost/Jar
Notebook/Heron Parrot/Shore Limestone/Hook Crayon/Knife Cheese/Fern Anchor/Palace Fox/Shrimp
Suitcase/Granite Pasture/Attic Blanket/Otter Fog/Smoke Cliff/Octopus Leather/Ant Tempo/Leopard
Carriage/Rubber""".split())
RELAY_IO_PAIR = ("Input:", "Output:")
RELAY_LAYOUTS = tuple((m1, s + m2) for m1, m2 in RELAY_PAIRS + (RELAY_IO_PAIR,) for s in (" ", "\n"))
RELAY_IO_P = 0.5
RELAY_LEAD_P, RELAY_LEAD_W, RELAY_LEAD_MIN = 0.75, 30, 4
RELAY_LEAD_WORD, RELAY_LEAD_SENT = re.compile(r"\S+"), re.compile(r"[.!?]\s+")
RELAY_LEAD_STOP = "\n\r\u2028\u2029()[]{}<>"
def relay_lead(d,a):
 w0=max(0,a-320);s=d[w0:a];cut=max([s.rfind(ch)for ch in RELAY_LEAD_STOP])+1;head=cut==0 and w0==0 or cut>0 and s[cut-1]in RELAY_LEAD_STOP[:4];s=s[cut:].rstrip();st=[m.start()for m in RELAY_LEAD_WORD.finditer(s)]
 if cut==0 and w0>0:st=st[1:]
 if len(st)>RELAY_LEAD_W:st,head=st[-RELAY_LEAD_W:],False
 if len(st)<RELAY_LEAD_MIN:return None
 t=s[st[0]:]
 if not head:
  for m in RELAY_LEAD_SENT.finditer(t):
   if len(RELAY_LEAD_WORD.findall(t,m.end()))>=RELAY_LEAD_MIN:t=t[m.end():];break
 return t if any(ch.isalpha()for ch in t)else None
def relay_dyck_cuts(text,rng):
 br=[(m.group(),m.start())for m in RELAY_BR_RE.finditer(text)];cuts,i=[],0
 while i<len(br):
  stack,starts,j=[],[],i
  while j<len(br):
   ch=br[j][0]
   if ch in RELAY_OPEN:
    if not stack:starts.append(j)
    stack.append(j)
   elif stack and br[stack[-1]][0]==RELAY_PAIR[ch]:
    o=stack.pop()
    if len(stack)>=1 or j-o>=2:cuts.append((j,o,j-o-1,starts,len(starts)))
   else:break
   j+=1
  run=0
  for x in range(j-1,i-1,-1):run=run+1 if br[x][0]not in RELAY_OPEN else 0;br[x]=(br[x][0],br[x][1],run)
  i=j+1
 by,out=([],[],[]),[]
 for j,o,dist,starts,ns in cuts:
  lo=bisect.bisect_left(starts,j+3-RELAY_DYCK_MAX2,0,ns)
  if lo<ns:by[2 if dist>=5 or br[o][0]=="<" else int(dist>=1)].append((j,starts,lo,ns))
 for k in(2,1,0):
  q=by[k]if len(by[k])<=8-len(out)else rng.sample(by[k],8-len(out))
  for j,starts,lo,ns in q:fs=starts[lo:ns];s=rng.choice((fs[0],rng.choice(fs)));out.append((br[s][1],br[j][1],br[j][2],k))
 return out
RELAY_MCQ = 0.006086
RELAY_MCQ_EARLY, RELAY_MCQ_POOL, RELAY_MCQ_WIDE, RELAY_MCQ_AT, RELAY_MCQ_RG = 2.0, 8192, 40, 512, 4
RELAY_MCQ_STYLES = tuple((re.compile(r"(?:(?<=\s)|^)" + p + r"[ \t]+(?=\S)", re.M), c, x, y) for p, c, x, y in (
    (r"([A-E])\)", 0, "A)", "D)"), (r"\(([A-E])\)", 0, "(A)", "(D)"), (r"([A-E])\.", 1, "A.", "D."),
    (r"([A-E]):", 1, "A:", "D:")))
RELAY_MCQ_KEY = re.compile(r"(?:\b(?:[Cc]orrect\s{1,40})?(?:[Aa]nswer|ANSWER|Ans|ANS)\b\.?(?:\s{1,40}[Kk]ey)?"
                           r"(?:\s{1,40}is)?\s{0,40}(?:[:-]\s{0,40})?|\b[Tt]he\s{1,40}(?:correct\s{1,40})?answer"
                           r"\s{1,40}is\s{0,40}(?::\s{0,40})?)\(?([A-E])(?:\)|(?=[.:,;]|[ \t]{0,40}(?:\n|$)))")
RELAY_MCQ_MULTI = re.compile(r"(?:\band\b|&|,|/|\bor\b)\s*\(?[A-E](?:[).:]|\b)")
RELAY_MCQ_SENT, RELAY_MCQ_END = re.compile(r"(?<=[.!?])\s+|\n"), re.compile(r"(?<=[a-z0-9)][.?!])\s+(?=[A-Z])")
RELAY_MCQ_QLAB0 = re.compile(r"(?:(?:Question|QUESTION|Q)\s*\d{0,3}\s*[:.)]|(?:Question|QUESTION)\s+\d{1,3}(?=\s))\s*")
RELAY_MCQ_QLAB1 = re.compile(r"(?:(?:Question|QUESTION|Q)\s*\d{0,3}|\d{1,3})\s*[:.)]\s*$")
RELAY_MCQ_NEG = re.compile(r"\b(?:NOT|[Nn]ot|EXCEPT|[Ee]xcept|LEAST|[Ll]east|FALSE|[Ff]alse|INCORRECT|[Ii]ncorrect)\b")
RELAY_MCQ_KEYSUB = ("nswer", "NSWER", "Ans", "ANS")
RELAY_MCQ_OPTW, RELAY_MCQ_STEMW, RELAY_MCQ_WIN, RELAY_MCQ_CAP, RELAY_MCQ_HEAD = 40, 60, 1500, 8, 600
RELAY_MCQ_WAIT, RELAY_MCQ_TRIES = 256, 32
RELAY_MCQ_MARKS = ("{}.", "{})", "({})", "{}:")
def relay_mcq_lists(d,styles):
 out=[]
 for si in styles:
  ms=list(RELAY_MCQ_STYLES[si][0].finditer(d));i=0
  while i<len(ms):
   if ms[i].group(1)!="A":i+=1;continue
   run=[ms[i]]
   for j in range(i+1,len(ms)):
    cur=ms[j];seg=d[run[-1].end():cur.start()]
    if(len(run)==5 or cur.group(1)!="ABCDE"[len(run)]or "\n" in seg.rstrip()or seg.count("\n")>2 or not seg.strip()):break
    run.append(cur)
   if len(run)>=2:out.append((si,run))
   i+=len(run)
 out.sort(key=lambda x:x[1][0].start());return out
def relay_mcq_cuts(d,rng):
 if not any(x in d for x in RELAY_MCQ_KEYSUB):return[]
 styles=[i for i,(_r,_c,x,y)in enumerate(RELAY_MCQ_STYLES)if x in d and y in d]
 if not styles:return[]
 keys=list(RELAY_MCQ_KEY.finditer(d))
 if not keys:return[]
 k0=[m.start()for m in keys];lists=relay_mcq_lists(d,styles);out,lo=[],0
 for n_l,(si,run)in enumerate(lists):
  s0,e0=run[0].start(),run[-1].end()
  if s0<lo or len(run)!=4:continue
  nxt=min(lists[n_l+1][1][0].start()if n_l+1<len(lists)else len(d),e0+RELAY_MCQ_WIN);j=bisect.bisect_left(k0,e0);key=keys[j]if j<len(keys)and k0[j]<nxt and keys[j].group(1)!="E" else None
  if key is None:continue
  le=d.find("\n",e0);b=min(len(d)if le<0 else le,key.start())
  if d[d.rfind("\n",0,run[-1].start())+1:run[-1].start()].strip():m=RELAY_MCQ_END.search(d,e0,b);b=m.start()if m else b
  b=e0+len(d[e0:b].rstrip());opts=[d[x.end():y.start()].strip()for x,y in zip(run,run[1:])]+[d[e0:b]]
  if not all(1<=len(RELAY_LEAD_WORD.findall(o))<=RELAY_MCQ_OPTW for o in opts):continue
  kl=d.find("\n",key.end());rest=d[key.end():len(d)if kl<0 else kl];r=rest.lstrip(" ).:-");o=opts["ABCD".index(key.group(1))].rstrip(".,;");te=key.start(1)+1
  if o and r.startswith(o):te=key.end()+len(rest)-len(r)+len(o);r=r[len(o):]
  if RELAY_MCQ_MULTI.search(r,0,80):continue
  h0=max(lo,s0-RELAY_MCQ_HEAD);hs=d[h0:s0].rstrip();p=RELAY_MCQ_SENT.split(hs)[-1].strip()if hs else "";he=h0+len(hs);a=la=he-len(p);m=RELAY_MCQ_QLAB0.match(d,a,he)
  if m and m.end()<he:a=m.end()
  else:m=RELAY_MCQ_QLAB1.search(d,max(h0,d.rfind("\n",0,a)+1),a);la=m.start()if m else a
  if not(p.endswith(("?",":"))and 3<=len(RELAY_LEAD_WORD.findall(d,a,he))<=RELAY_MCQ_STEMW):continue
  sp=tuple((x.end(),y.start())for x,y in zip(run,run[1:]))+((e0,b),);out.append((2*RELAY_MCQ_STYLES[si][1]+(RELAY_MCQ_NEG.search(d,a,he)is not None),"ABCD".index(key.group(1)),(d,la,a,key.start(1),lo,he,sp)));lo=max(key.end(),te)
 if len(out)>RELAY_MCQ_CAP:out=rng.sample(out,RELAY_MCQ_CAP)
 return out
RELAY_ARITH = 0.0001
RELAY_EQ_OP = "+-*/" + chr(0xd7) + chr(0xf7)
RELAY_EQ_RE = re.compile(r"(\(?\d{1,3}\)?(?:(?:[ \t]*[-+*/" + chr(0xd7) + chr(0xf7) + r"][ \t]*| x )\(?\d{1,3}\)?)"
                         r"{1,3})[ \t]*(=|equals)[ \t]*(-?\d{1,7}(?:\.\d{1,4})?)")
RELAY_EQ_PRE = frozenset(".,/$:^=+*%(-x_" + "".join(map(chr, (0xd7, 0xf7, 0x2013, 0x2014, 0x2212, 0x2022, 0xb7))))
RELAY_EQ_CHAIN = re.compile(r"[ \t]*(?:[-+*/^(" + chr(0xd7) + chr(0xf7) + r"]| x )[ \t]*[\d(]")
RELAY_EQ_LAYOUTS = (("Input:", " Output:"), ("Input:", "\nOutput:"))
def relay_eq_cuts(d):
 out=[]
 if "=" not in d and "equals" not in d:return out
 for m in RELAY_EQ_RE.finditer(d):
  a,b=m.start(1),m.end(1);ex=d[a:b]
  if ex.count("(")>ex.count(")")and ex[0]=="(":a+=1
  elif ex.count(")")>ex.count("(")and ex[-1]==")":b-=1
  ex=d[a:b]
  if ex.count("(")!=ex.count(")")or not any(o in ex for o in RELAY_EQ_OP+"x"):continue
  p=m.start(1)-1
  if p>=0 and(d[p].isalnum()or d[p]in RELAY_EQ_PRE):continue
  q=p
  while q>=0 and d[q]in " \t":
   q-=1
  if q<p and q>=0 and(d[q]in RELAY_EQ_OP or d[q]in RELAY_EQ_PRE and d[q]not in ",:("):continue
  e=m.end(3)
  if e<len(d)and(d[e].isalnum()or d[e]in "%/" or d[e]in ".," and e+1<len(d)and d[e+1].isdigit()):continue
  if RELAY_EQ_CHAIN.match(d,e):continue
  out.append((a,b,m.start(3),e))
 return out
RELAY_YNCAP = 0.005
RELAY_YNCAP_YES = 0.58
RELAY_Q2_T = RELAY_SHARE + RELAY_MCQ + RELAY_YNCAP
RELAY_YQ_WIDE = 10
RELAY_YQ_AT = 5824
RELAY_PS_WORKERS = 16
RELAY_PS_FROM = 120
RELAY_PS_PATH = '/dev/shm'
RELAY_PS_KINDS = ("y", "m")
RELAY_PS = {"root": None, "world": None, "proc": None, "box": None, "told": False}
RELAY_NQA_SENT = re.compile(r"[.!?]+(?=\s|$)")
RELAY_NQA_QLAB = re.compile(r"(?:Question|Q)(?: ?\d{1,3})?\s*[:.]\s*")
RELAY_NQA_ALAB = re.compile(r"(?:Answer|A)(?: ?\d{1,3})?\s*[:.]\s*")
RELAY_NQA_KEY = re.compile(r"[a-z0-9]{4,}")
RELAY_NQA_AUX = frozenset("is are am was were do does did can could will would shall should has have had may might "
                          "must".split())
RELAY_NQA_QW = (4, 40)
def relay_nqa_sents(d):
 out,pos=[],0
 for li,line in enumerate(d.split("\n")):
  start=0
  for m in RELAY_NQA_SENT.finditer(line):
   seg=line[start:m.end()]
   if seg.strip():out.append((pos+start+len(seg)-len(seg.lstrip()),pos+m.end(),li))
   start=m.end()
  seg=line[start:]
  if seg.strip():out.append((pos+start+len(seg)-len(seg.lstrip()),pos+len(line.rstrip()),li))
  pos+=len(line)+1
 return out
def relay_nqa_polar(q):
 w=q.split()
 if w[0].lower().strip(",")in RELAY_NQA_AUX:return True
 for i,x in enumerate(w[:-1]):
  if x.endswith(","):return w[i+1].lower()in RELAY_NQA_AUX
 return False
RELAY_Q2_YNP = (25, 250)
RELAY_Q2_YNK = 2
RELAY_Q2_YNG = 2
RELAY_Q2_BAR = re.compile("[" + re.escape("{}[]<>|@#*=_~^`" + "".join(map(chr, (92, 9, 11, 12, 13, 0x85, 0x2028,
                                                                                0x2029)))) + "]")
RELAY_Q2_ART = (re.compile(r"[.!?][A-Z]"), re.compile(r"[,;][A-Za-z]"), re.compile(r" [.,;:!?]"))
RELAY_Q2_YN = re.compile(r"(Yes|No|yes|no)(?:[,.!;:]|$)")
RELAY_Q2_LAYOUTS = (("Input:", " Output:"), ("Input:", "\nOutput:"))
RELAY_YQ_PRE = r"\?\s*(?:(?:Answer|A)(?: ?\d{1,3})?\s*[:.]\s*)?(?:Yes|No|yes|no)"
def relay_yq_rows(path,group):import pyarrow.compute as relay_pc;import pyarrow.parquet as relay_pq;col=relay_pq.ParquetFile(path).read_row_group(group).column("text");keep=relay_pc.match_substring_regex(col,RELAY_YQ_PRE);return len(col),col.filter(keep).to_pylist()
def relay_q2_ok(t):return("http" not in t and "www." not in t and not RELAY_Q2_BAR.search(t)and not any(x.search(t)for x in RELAY_Q2_ART))
def relay_q2_yncap_cuts(d,ss):
 out=[]
 for i in range(1,len(ss)-1):
  qa,qb,ql=ss[i];pl=ss[i-1][2]
  if d[qb-1]!="?" or pl==ql or ql-pl>2:continue
  aa,ab,_l=ss[i+1];ma=RELAY_NQA_ALAB.match(d,aa,ab);aa=ma.end()if ma else aa;m=RELAY_Q2_YN.match(d[aa:ab])
  if not m:continue
  mq=RELAY_NQA_QLAB.match(d,qa,qb);qa=mq.end()if mq else qa;q,j=d[qa:qb],i-1
  while j and ss[j-1][2]==pl:
   j-=1
  pa,pb=ss[j][0],ss[i-1][1];p=d[pa:pb]
  if not(RELAY_NQA_QW[0]<=len(q.split())<=RELAY_NQA_QW[1]and "A"<=q[0]<="Z" and relay_nqa_polar(q)and d[pb-1]in ".!" and "?" not in p and RELAY_Q2_YNP[0]<=len(p.split())<=RELAY_Q2_YNP[1]and not(RELAY_NQA_QLAB.match(p)or RELAY_NQA_ALAB.match(p)or RELAY_Q2_YN.match(p))and relay_q2_ok(q)and relay_q2_ok(p)):continue
  kq,kp=set(RELAY_NQA_KEY.findall(q.lower())),set(RELAY_NQA_KEY.findall(p.lower()));ka=set(RELAY_NQA_KEY.findall(d[aa+len(m.group(1)):ab].lower()))-kq
  if len(kq&kp)>=RELAY_Q2_YNK and len(ka&kp)>=RELAY_Q2_YNG:out.append((pa,pb,qa,qb,aa,aa+len(m.group(1)),int(d[aa]in "Yy")))
 return out
class RelayPsDefer(list):
 pass
class RelayPsRng:
 def sample(self,pop,k):return RelayPsDefer(pop)
def relay_mcq_rows(path,group):import pyarrow.compute as relay_pc;import pyarrow.parquet as relay_pq;col=relay_pq.ParquetFile(path).read_row_group(group).column("text");keep=relay_pc.match_substring_regex(col,"|".join(RELAY_MCQ_KEYSUB));return len(col),col.filter(keep).to_pylist()
def relay_ps_slice(kind,rank,world,paths=None):
 from prepare import PUBLIC_TRAIN_SHARDS,list_shard_paths
 if paths is None:paths=list_shard_paths(PUBLIC_TRAIN_SHARDS,require_all=False)
 if kind=="y":n,at=RELAY_YQ_WIDE,min(RELAY_YQ_AT,max(0,len(paths)-world*RELAY_YQ_WIDE))
 else:n,at=RELAY_MCQ_WIDE,min(RELAY_MCQ_AT,len(paths)//2)
 return[paths[at+rank+world*j]for j in range(n)if at+rank+world*j<len(paths)]
def relay_ps_job(cfg,ctr,done,tasks):
 import multiprocessing as mp;import pickle;import pyarrow as pa;import pyarrow.parquet as pq
 if hasattr(os,"nice"):os.nice(19)
 pa.set_cpu_count(1);pa.set_io_thread_count(1);root,parent,rng=Path(cfg["root"]),mp.parent_process(),RelayPsRng()
 while parent is None or parent.is_alive():
  with ctr.get_lock():i=ctr.value;ctr.value+=1
  if i>=len(tasks):return
  kind,rank,j,path=tasks[i]
  try:
   ng=pq.ParquetFile(path).num_row_groups
  except Exception:ng=None
  rgs=[]
  for g in range(ng or 0):
   try:
    n_docs,rows=(relay_yq_rows if kind=="y" else relay_mcq_rows)(path,g)
   except Exception:rgs.append(None);continue
   ents=[]
   for d in rows:
    if kind=="y":
     cuts=relay_q2_yncap_cuts(d,relay_nqa_sents(d))
     if cuts:ents.append((d,cuts))
    else:
     cuts=relay_mcq_cuts(d,rng)
     if cuts:ents.append((d,list(cuts),isinstance(cuts,RelayPsDefer)))
   rgs.append((n_docs,len(rows),ents))
  name="%s_%d_%d"%(kind,rank,j)
  with open(root/(name+".tmp"),"wb")as fh:pickle.dump({"ok":ng is not None,"rg":rgs},fh,protocol=4)
  os.replace(root/(name+".tmp"),root/(name+".pkl"))
  with done.get_lock():done.value+=1
def relay_ps_rmdir(root):
 try:
  for f in os.listdir(root):os.remove(os.path.join(root,f))
  os.rmdir(root)
 except OSError:
  pass
def relay_ps_coordinator(cfg,box):
 import multiprocessing as mp;from prepare import PUBLIC_TRAIN_SHARDS,list_shard_paths
 if hasattr(os,"nice"):os.nice(19)
 t0,ctx,parent=time.monotonic(),mp.get_context("spawn"),mp.parent_process();world,ps,msg=cfg["world"],[],"done"
 try:
  paths=list_shard_paths(PUBLIC_TRAIN_SHARDS,require_all=False);sl={(k,r):relay_ps_slice(k,r,world,paths)for k in cfg["kinds"]for r in range(world)};tasks=[(k,r,j,str(sl[(k,r)][j]))for j in range(max(map(len,sl.values()),default=0))for k in cfg["kinds"]for r in range(world)if j<len(sl[(k,r)])]
  if any(int(Path(t[3]).stem.split("_")[-1])>=6500 for t in tasks):raise RuntimeError("a shard at or above 6500")
  ctr,done,last=ctx.Value("i",0),ctx.Value("i",0),time.monotonic();ps=[ctx.Process(target=relay_ps_job,args=(cfg,ctr,done,tasks),daemon=True)for _ in range(min(cfg["workers"],max(1,len(tasks))))]
  for p in ps:p.start()
  print("prescan: started "+json.dumps({"tasks":len(tasks),"workers":len(ps),"world":world,"kinds":list(cfg["kinds"])}),flush=True)
  while any(p.is_alive()for p in ps):
   time.sleep(0.5)
   if box[0]or not parent.is_alive():raise RuntimeError("stopped")
   if any(p.exitcode for p in ps):raise RuntimeError("a worker died")
   if time.monotonic()-last>=60.0:last=time.monotonic();print("prescan: "+json.dumps({"done":done.value,"of":len(tasks),"elapsed_s":round(last-t0,1)}),flush=True)
  if done.value!=len(tasks):raise RuntimeError("%d of %d tasks done"%(done.value,len(tasks)))
 except Exception as e:msg="failed "+str(e)[:200]
 finally:
  for p in ps:
   if p.is_alive():p.kill()
   p.join()
 print("prescan: %s %s"%(msg,json.dumps({"elapsed_s":round(time.monotonic()-t0,1)})),flush=True)
 while not box[0]and parent.is_alive():
  time.sleep(1.0)
 if not parent.is_alive():relay_ps_rmdir(cfg["root"])
def relay_ps_close(proc,box,root):
 box[0]=1.0;proc.join(30)
 if proc.is_alive():proc.kill();proc.join(10)
 relay_ps_rmdir(root)
def relay_ps_open(rank):
 import atexit;import multiprocessing as mp;_ddp,grank,_local,world=get_dist_info();tag=hashlib.md5(("%s|%d|%d"%(os.environ.get("TORCHELASTIC_RUN_ID",""),os.getppid(),world)).encode());RELAY_PS.update(root=Path(RELAY_PS_PATH)/("r2_prescan_"+tag.hexdigest()[:12]),world=world)
 if grank!=0 or rank!=0:return
 RELAY_PS["root"].mkdir(parents=True);ctx,old=mp.get_context("spawn"),os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD");box=ctx.RawArray("d",1);cfg={"root":str(RELAY_PS["root"]),"world":world,"workers":RELAY_PS_WORKERS,"kinds":RELAY_PS_KINDS};os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"]="0"
 try:
  proc=ctx.Process(target=relay_ps_coordinator,args=(cfg,box),name="r2_prescan");proc.start()
 finally:
  if old is None:del os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"]
  else:os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"]=old
 RELAY_PS.update(proc=proc,box=box);atexit.register(relay_ps_close,proc,box,RELAY_PS["root"]);print("prescan: pool started "+json.dumps(dict(cfg,pid=proc.pid)),flush=True)
def relay_ps_take(L,kind):
 import pickle
 if RELAY_PS["root"]is None:relay_ps_open(L.rank)
 p=RELAY_PS["proc"]
 if p is not None and not RELAY_PS["told"]and p.exitcode is not None and not RELAY_PS["box"][0]:RELAY_PS["told"]=True;print(f"prescan: coordinator exited {p.exitcode} at batch {L.st.rows} (the ranks keep what landed)", flush=True)
 if L.ps[0]is None:L.ps[0]=relay_ps_slice(kind,L.rank,RELAY_PS["world"])
 if L.ps[2]>=0 or L.st.rows<RELAY_PS_FROM:return[]
 out=[]
 while L.ps[1]<len(L.ps[0]):
  f=RELAY_PS["root"]/("%s_%d_%d.pkl"%(kind,L.rank,L.ps[1]))
  try:
   with open(f,"rb")as fh:res=pickle.load(fh)
  except FileNotFoundError:break
  os.remove(f);L.ps[1]+=1
  if res["ok"]:out+=res["rg"]
 if L.ps[1]>=len(L.ps[0]):L.ps[2]=L.st.rows
 return out
DSEL_SHARE,DSEL_QN,DSEL_QE,DSEL_FROM,DSEL_REP=0.2,0.08,0.08,0.1,2
DSEL_SHARDS,DSEL_WORKERS,DSEL_DEADLINE_S,DSEL_PATH=384,16,900.0,"/dev/shm"
DSEL_FA,DSEL_FB=1024,1024
DSEL_NF=13
DSEL_SEED=0x5EED0070
DSEL_POOL_SEED=0x5EED0062
DSEL_MIN_BYTES,DSEL_MIN_WORDS=512,64
DSEL_PROSE=8
def dsel_pack(w):return int.from_bytes(w.lower().ljust(8,b"\0"),"little")
DSEL_LUT=bytes((3 if 65<=b<=90 else 1 if 97<=b<=122 else 0)|(4 if 48<=b<=57 else 0)|(8 if b in(46,33,63,58,34,10)else 0)|(16 if b in(60,62,123,125,91,93,124,61,92,94,126,95,96)else 0)|(32 if b==34 else 0)|(64 if b==10 else 0)|(128 if b==0xE2 else 0)for b in range(256))
DSEL_MASK=tuple((1<<(8*k))-1 for k in range(8))+((1<<64)-1,)
DSEL_PRON=tuple(dsel_pack(w)for w in(b"he",b"she",b"they",b"we",b"him",b"her"))
DSEL_W={k:dsel_pack(k.encode())for k in("said","born","located","founded","known","for","is","was","a","an","http","https","www")}
def dsel_counts(col):
 import numpy as np;import pyarrow as pa;u8,i64,u64=np.uint8,np.int64,np.uint64;ch=col.combine_chunks()
 if not(pa.types.is_string(ch.type)or pa.types.is_large_string(ch.type)):raise ValueError("DSEL: the text column is %s, not a string column"%ch.type)
 n=len(ch);out=np.zeros((n,DSEL_NF),dtype=i64)
 if not n:return out
 off,data=ch.buffers()[1:3];off=np.frombuffer(off,dtype=i64 if pa.types.is_large_string(ch.type)else np.int32);off=off[ch.offset:ch.offset+n+1].astype(i64);data=np.frombuffer(data,dtype=u8)if data is not None else np.zeros(0,dtype=u8);ln=np.diff(off);mid=np.maximum(DSEL_FA,(ln-DSEL_FB)//2);two=mid>DSEL_FA;e1=np.where(two,DSEL_FA,np.minimum(ln,DSEL_FA+DSEL_FB));sd=np.repeat(np.arange(n),1+two);ss=np.repeat(off[:-1],1+two);sl=np.repeat(e1,1+two);k2=np.flatnonzero(two)+np.arange(int(two.sum()))+1;ss[k2]+=mid[two];sl[k2]=DSEL_FB;keep=sl>0;sd,ss,sl=sd[keep],ss[keep],sl[keep];tot=int(sl.sum())
 if not tot:return out
 so=np.zeros(len(sl),dtype=i64);np.cumsum(sl[:-1],out=so[1:]);se=so+sl;x=np.concatenate([data[a:a+b]for a,b in zip(ss.tolist(),sl.tolist())]);xp=np.concatenate([x,np.zeros(8,dtype=u8)]);c=np.frombuffer(DSEL_LUT,dtype=u8)[x];out[:,0]=np.bincount(sd,weights=sl,minlength=n).astype(i64)
 def sg(pos):return np.searchsorted(so,pos,"right")-1
 def at(pos):return np.bincount(sd[sg(pos)],minlength=n)
 sp=np.flatnonzero(c&u8(252));cs=c[sp];out[:,7]=at(sp[(cs&u8(4))>0]);out[:,8]=at(sp[(cs&u8(16))>0]);e2=sp[(cs&u8(128))>0];e2=e2[e2+2<se[sg(e2)]];e2=e2[(x[e2+1]==0x80)&((x[e2+2]==0x9C)|(x[e2+2]==0x9D))];out[:,2]=at(sp[(cs&u8(32))>0])+at(e2);nlp=sp[(cs&u8(64))>0]+1;ls=np.union1d(so,nlp[nlp<tot]);room=se[sg(ls)]-ls;b0,b1,b2=xp[ls],xp[ls+1],xp[ls+2];dg0,dg1=(b0>=48)&(b0<=57),(b1>=48)&(b1<=57);lm=(np.isin(b0,np.frombuffer(b"-*#>|",dtype=u8))|((room>=3)&(b0==0xE2)&(b1==0x80)&(b2==0xA2))|((room>=2)&dg0&((b1==46)|(b1==41)))|((room>=3)&dg0&dg1&((b2==46)|(b2==41))));out[:,8]+=at(ls[lm]);let=(c&u8(1)).view(bool);sw=np.empty(tot,dtype=bool);sw[0]=let[0];np.greater(let[1:],let[:-1],out=sw[1:]);sw[so]=let[so];ew=np.empty(tot,dtype=bool);ew[-1]=let[-1];np.greater(let[:-1],let[1:],out=ew[:-1]);ew[se-1]=let[se-1];st=np.flatnonzero(sw);en=np.flatnonzero(ew);nw=len(st)
 if not nw:return out
 wseg=sg(st);wd=sd[wseg]
 def cnt(f):return np.bincount(wd[f],minlength=n)
 out[:,1]=np.bincount(wd,minlength=n);wl=en-st+1;key=((np.lib.stride_tricks.sliding_window_view(xp,8)[st].view(u64).ravel()|u64(0x2020202020202020))&np.array(DSEL_MASK,dtype=u64)[np.minimum(wl,8)]);key[wl>8]=0;cap=(c[st]&u8(2))>0;ii=(wl==1)&(x[st]==73);nxt=np.where(en+1>=se[wseg],-1,xp[en+1].astype(i64));pron=ii.copy()
 for k in DSEL_PRON:pron|=key==k
 out[:,3]=cnt(pron);out[:,4]=cnt((key==DSEL_W["said"])|((wl>=4)&((x[en]|u8(32))==100)&((x[en-1]|u8(32))==101)&(nxt==32)));sent=np.zeros(nw,dtype=bool);sent[0]=True;sent[1:]=wseg[1:]!=wseg[:-1];km=np.searchsorted(st,sp[(cs&u8(8))>0],"right");sent[km[km<nw]]=True;out[:,5]=cnt(sent);s0=np.flatnonzero(sent);slen=np.diff(np.append(s0,nw));out[:,6]=np.bincount(wd[s0],weights=slen.astype(np.float64)**2,minlength=n).astype(i64);out[:,9]=cnt((((key==DSEL_W["http"])|(key==DSEL_W["https"]))&(nxt==58))|((key==DSEL_W["www"])&(nxt==46)));pro=np.repeat(slen>=DSEL_PROSE,slen);midc=cap&~sent&~ii;out[:,12]=cnt(midc&pro);one=np.zeros(nw,dtype=bool);one[1:]=(wseg[1:]==wseg[:-1])&(st[1:]-en[:-1]==2)&(nxt[:-1]==32);pair=np.zeros(nw,dtype=bool);pair[1:]=one[1:]&cap[1:]&midc[:-1];out[:,10]=cnt(pair&pro);cop=np.zeros(nw,dtype=bool)
 if nw>2:cop[1:-1]=(one[1:-1]&one[2:]&cap[:-2]&((key[1:-1]==DSEL_W["is"])|(key[1:-1]==DSEL_W["was"]))&((key[2:]==DSEL_W["a"])|(key[2:]==DSEL_W["an"])))
 kf=np.zeros(nw,dtype=bool);kf[:-1]=(key[:-1]==DSEL_W["known"])&one[1:]&(key[1:]==DSEL_W["for"]);out[:,11]=cnt((cop|kf|(key==DSEL_W["born"])|(key==DSEL_W["located"])|(key==DSEL_W["founded"]))&pro);return out
def dsel_rank2(v):
 import numpy as np
 if not len(v):return np.zeros(0,dtype=np.int64)
 o=np.argsort(v,kind="stable");s=v[o];b=np.flatnonzero(np.concatenate(([True],s[1:]!=s[:-1],[True])));r=np.empty(len(v),dtype=np.int64);r[o]=np.repeat(b[:-1]+1+b[1:],np.diff(b));return r
def dsel_scores(cnt):import numpy as np;el=(cnt[:,0]>=DSEL_MIN_BYTES)&(cnt[:,1]>=DSEL_MIN_WORDS);a=cnt[el];w=a[:,1].astype(np.float64);r={k:dsel_rank2(a[:,i]/w)for k,i in(("Q",2),("P",3),("T",4),("M",8),("U",9),("C",10),("K",11),("R",12))};r["D"]=dsel_rank2(a[:,7]/a[:,0].astype(np.float64));v=np.where(a[:,5]>=3,(a[:,5]*a[:,6]-a[:,1]*a[:,1])/(w*w),0.0);r["V"]=dsel_rank2(v);neg=r["D"]+r["M"]+r["U"];nn,ee=np.zeros(len(cnt),dtype=np.int64),np.zeros(len(cnt),dtype=np.int64);nn[el]=r["Q"]+r["P"]+r["T"]+r["V"]-neg;ee[el]=r["C"]+r["K"]+r["R"]-neg;return el,nn,ee
def dsel_hash(gid):import numpy as np;u=np.uint64;z=np.asarray(gid,dtype=u)^u(DSEL_SEED);z=(z^(z>>u(30)))*u(0xBF58476D1CE4E5B9);z=(z^(z>>u(27)))*u(0x94D049BB133111EB);return z^(z>>u(31))
def dsel_pick(cnt,shard,qn,qe):
 import numpy as np;el,nn,ee=dsel_scores(cnt);rows=np.arange(len(cnt),dtype=np.int64);h=dsel_hash((np.uint64(shard)<<np.uint64(32))|rows.astype(np.uint64));flag=np.zeros(len(cnt),dtype=np.int64)
 for bit,s,q in((1,nn,qn),(2,ee,qe)):k=min(int(q*len(cnt)),int(el.sum()));e=rows[el];top=e[np.lexsort((h[e],-s[e]))[:k]];flag[top]|=bit
 sel=np.flatnonzero(flag);return sel,flag[sel]
DSEL_LOW=512
DSEL_FLOOR=160
DSEL_EVERY=60.0
DSEL_AUTO="TORCH_DEVICE_BACKEND_AUTOLOAD"
def dsel_root(out_dir):return Path(DSEL_PATH)/("r2_dsel_"+hashlib.md5(str(Path(out_dir).resolve()).encode()).hexdigest()[:12])
def dsel_shards(ids):import numpy as np;u=np.uint64;a=np.array(sorted({int(i)for i in ids if DSEL_LOW<=int(i)<6500}),dtype=u);z=a^u(DSEL_POOL_SEED);z=(z^(z>>u(30)))*u(0xBF58476D1CE4E5B9);z=(z^(z>>u(27)))*u(0x94D049BB133111EB);return[int(i)for i in a[np.argsort(z^(z>>u(31)),kind="stable")][:DSEL_SHARDS]]
def dsel_clear(root,keep=None):
 for d in(root/"t",root):
  for f in(list(d.iterdir())if d.is_dir()else[]):
   if f.suffix in(".b",".npy",".json",".bin",".tmp")and f.name not in(keep or()):
    try:
     f.unlink()
    except OSError:
     pass
  if d!=root or keep is None:
   try:
    d.rmdir()
   except OSError:
    pass
def dsel_put(path,data):
 import numpy as np
 with open(str(path)+".tmp","wb")as fh:
  if isinstance(data,(bytes,bytearray)):fh.write(data)
  elif isinstance(data,dict):fh.write(json.dumps(data).encode())
  else:np.save(fh,data)
 os.replace(str(path)+".tmp",path)
def dsel_job(cfg,ctr,done):
 import multiprocessing as mp;import numpy as np;import pyarrow as pa;import pyarrow.parquet as pq;import prepare as P
 if hasattr(os,"nice"):os.nice(19)
 pa.set_cpu_count(1);pa.set_io_thread_count(1);root,parent,ids=Path(cfg["root"]),mp.parent_process(),[int(i)for i in cfg["shards"]]
 if not set(ids)<={i for i in P.PUBLIC_TRAIN_SHARDS if DSEL_LOW<=i<P.RESERVED_EVAL_SHARD_START}:raise ValueError("dsel: a scan shard outside the kit's list from 512 to 6,499")
 while parent.is_alive():
  with ctr.get_lock():i=ctr.value;ctr.value+=1
  if i>=len(ids):return
  s,t0=ids[i],time.perf_counter();pf=pq.ParquetFile(P.DATA_DIR/P.shard_filename(s));cols=[pf.read_row_group(r,columns=["text"]).column(0)for r in range(pf.num_row_groups)];cnt=np.concatenate([dsel_counts(c)for c in cols])if cols else np.zeros((0,DSEL_NF),dtype=np.int64);rows,flag=dsel_pick(cnt,s,cfg["qn"],cfg["qe"]);txt=[]
  if len(rows):allc=pa.chunked_array([k for c in cols for k in c.chunks],type=cols[0].type);txt=[t.encode()for t in allc.take(pa.array(rows)).to_pylist()]
  rec=np.zeros(len(rows),dtype=[("g","<u8"),("n","<i8"),("f","<i1")]);rec["g"]=(np.uint64(s)<<np.uint64(32))|rows.astype(np.uint64);rec["n"]=[len(t)for t in txt];rec["f"]=flag;el=(cnt[:,0]>=DSEL_MIN_BYTES)&(cnt[:,1]>=DSEL_MIN_WORDS);dsel_put(root/"t"/("%d.b"%s),b"".join(txt));dsel_put(root/"t"/("%d.npy"%s),rec);dsel_put(root/"t"/("%d.json"%s),{"docs":len(cnt),"eligible":int(el.sum()),"N":int(((flag&1)>0).sum()),"E":int(((flag&2)>0).sum()),"both":int((flag==3).sum()),"bytes":int(rec["n"].sum()),"s":round(time.perf_counter()-t0,2)})
  with done.get_lock():done.value+=1
def dsel_coordinator(cfg,box):
 import multiprocessing as mp;import numpy as np;import prepare as P
 if hasattr(os,"nice"):os.nice(19)
 t0,root,ctx,parent=time.monotonic(),Path(cfg["root"]),mp.get_context("spawn"),mp.parent_process();ids,sec,cut,ps=[int(i)for i in cfg["shards"]],{},-1,[]
 try:
  if not ids or min(ids)<DSEL_LOW or max(ids)>=P.RESERVED_EVAL_SHARD_START:raise RuntimeError("no scan shard, or one outside 512 to 6,499")
  ctr,done=ctx.Value("i",0),ctx.Value("i",0);ps=[ctx.Process(target=dsel_job,args=(cfg,ctr,done),daemon=True)for _ in range(min(cfg["workers"],len(ids)))]
  for p in ps:p.start()
  last,done0=time.monotonic(),0
  while any(p.is_alive()for p in ps):
   time.sleep(0.5)
   if box[0]or not parent.is_alive():raise RuntimeError("stopped")
   if any(p.exitcode for p in ps):raise RuntimeError("a worker died")
   if cut<0 and time.monotonic()-t0>=cfg["deadline"]:
    with ctr.get_lock():cut=min(ctr.value,len(ids));ctr.value=len(ids)
    print("dsel_scan: cut "+json.dumps({"handed_out":cut,"of":len(ids),"elapsed_s":round(time.monotonic()-t0)}),flush=True)
   if time.monotonic()-last>=DSEL_EVERY:now,d=time.monotonic(),done.value;print("dsel_scan: "+json.dumps({"shards":d,"of":len(ids),"elapsed_s":round(now-t0),"window_shards":d-done0,"window_s":round(now-last,1)}),flush=True);last,done0=now,d
  if any(p.exitcode for p in ps):raise RuntimeError("a worker died")
  sec["scan"]=round(time.monotonic()-t0,1);got=[s for s in ids if(root/"t"/("%d.json"%s)).is_file()]
  if got!=ids[:len(got)]or len(got)<min(len(ids),DSEL_FLOOR):raise RuntimeError("the scan kept %d of %d shards (floor %d, or not a prefix)"%(len(got),len(ids),DSEL_FLOOR))
  tally,recs={k:0 for k in("docs","eligible","N","E","both","bytes")},[]
  with open(root/"store.bin.tmp","wb")as fh:
   for s in got:
    fh.write((root/"t"/("%d.b"%s)).read_bytes());recs.append(np.load(root/"t"/("%d.npy"%s)))
    for k,v in json.loads((root/"t"/("%d.json"%s)).read_text()).items():
     if k in tally:tally[k]+=v
    for x in("b","npy","json"):
     try:
      (root/"t"/("%d.%s"%(s,x))).unlink()
     except OSError:
      pass
  rec=np.concatenate(recs);dsel_put(root/"off.npy",np.concatenate(([0],np.cumsum(rec["n"]))).astype(np.int64));dsel_put(root/"gid.npy",np.ascontiguousarray(rec["g"]));dsel_put(root/"flag.npy",np.ascontiguousarray(rec["f"]));os.replace(root/"store.bin.tmp",root/"store.bin");dsel_clear(root,keep=("store.bin","off.npy","gid.npy","flag.npy"));sec["total"]=round(time.monotonic()-t0,1);info=dict(tally,pool=int(len(rec)),shards=len(got),planned=len(ids),cut=len(got)<len(ids),shards_md5=hashlib.md5(repr(got).encode()).hexdigest()[:12],seconds=sec);dsel_put(root/"READY.json",info);print("dsel_scan: ready "+json.dumps(info),flush=True)
 except Exception as e:
  for p in ps:
   if p.is_alive():p.kill()
   p.join()
  sec["total"]=round(time.monotonic()-t0,1);dsel_clear(root,keep=())
  try:
   dsel_put(root/"FAILED.json",{"failed":str(e)[:300],"seconds":sec})
  except OSError:
   pass
  print("dsel_scan: failed",e,sec,flush=True)
 while not box[0]and parent.is_alive():
  time.sleep(1.0)
 if not parent.is_alive():dsel_clear(root)
def dsel_check(args):
 import prepare as P;root,sh=dsel_root(args.out_dir),dsel_shards(P.PUBLIC_TRAIN_SHARDS);have=len(P.list_shard_paths(sh,require_all=False))
 if(len(sh)!=DSEL_SHARDS or min(sh)<DSEL_LOW or max(sh)>=P.RESERVED_EVAL_SHARD_START or have!=len(sh)or root.exists()):print0("dsel: off (%d of %d scan shards, folder %s exists %s)"%(have,DSEL_SHARDS,root,root.exists()));return False
 print0("dsel: on: %d shards (ids %d to %d, md5 %s), folder %s"%(len(sh),min(sh),max(sh),hashlib.md5(repr(sh).encode()).hexdigest()[:12],root));return True
def dsel_start(root,world,step):
 import atexit;import multiprocessing as mp;import prepare as P;(root/"t").mkdir(parents=True);ctx,old=mp.get_context("spawn"),os.environ.get(DSEL_AUTO);box=ctx.RawArray("d",1);cfg={"root":str(root),"workers":DSEL_WORKERS,"qn":DSEL_QN,"qe":DSEL_QE,"deadline":DSEL_DEADLINE_S,"shards":dsel_shards(P.PUBLIC_TRAIN_SHARDS)};os.environ[DSEL_AUTO]="0"
 try:
  proc=ctx.Process(target=dsel_coordinator,args=(cfg,box),name="r2_dsel");proc.start()
 finally:
  if old is None:del os.environ[DSEL_AUTO]
  else:os.environ[DSEL_AUTO]=old
 atexit.register(dsel_close,(proc,box),root);md5=hashlib.md5(repr(cfg["shards"]).encode()).hexdigest()[:12];print("dsel: scan started at step %d (the charged part: after the %d excluded startup steps)"%(step,TRAIN_STARTUP_STEPS_EXCLUDED),proc.pid,world,json.dumps(dict(cfg,shards=[len(cfg["shards"]),md5])),flush=True);return proc,box
def dsel_close(scan,root):
 scan[1][0]=1.0;scan[0].join(60)
 if scan[0].is_alive():scan[0].kill();scan[0].join(10)
 dsel_clear(root)
class DselStream:
 def __init__(self,tokenizer,root,rank,world):import collections;self.tok,self.root,self.rank,self.world=tokenizer,root,rank,world;self.bos=tokenizer.get_bos_token_id();self.phase,self.decided,self.active,self.at=0.0,0,0,-1;self.order,self.next,self.passes,self.q=[],0,0,collections.deque();self.kit=self.debt=0.0;self.added=[[0,0]for _ in range(4)];self.short=self.calls=0;self.secs=0.0;self.data=self.off=self.flag=self.gid=self.mine=None
 def launch_line(self):return "dsel: loader: share %g from fraction %g, at most %d draws a document"%(DSEL_SHARE,DSEL_FROM,DSEL_REP)
 def turn(self,group,scan,step):
  import numpy as np;ok=torch.tensor([int((self.root/"READY.json").is_file())],dtype=torch.int32)
  if group is not None:dist.all_reduce(ok,op=dist.ReduceOp.MIN,group=group)
  on=int(ok.item())
  if on:self.gid,self.off=np.load(self.root/"gid.npy"),np.load(self.root/"off.npy");self.flag=np.load(self.root/"flag.npy");self.mine=np.flatnonzero(dsel_hash(self.gid)%np.uint64(self.world)==np.uint64(self.rank));self.order=self.mine[np.argsort(dsel_hash(self.gid[self.mine]),kind="stable")];store=self.root/"store.bin";self.data=memoryview(np.memmap(store,dtype=np.uint8,mode="r"))if self.off[-1]else b""
  self.decided,self.active,self.at=1,on,step
  if scan is not None:scan[1][0]=1.0;f=[p for p in(self.root/"READY.json",self.root/"FAILED.json")if p.is_file()];print("dsel: step",step,"decision","on" if self.active else "fallback",f[0].read_text().strip()if f else "-",flush=True)
 def fill(self):
  import numpy as np
  if self.next>=len(self.order):
   if self.passes+1>=DSEL_REP or not len(self.order):return False
   self.passes+=1;self.order=self.mine[np.argsort(dsel_hash(self.gid[self.mine]^np.uint64(self.passes)),kind="stable")];self.next=0
  m=self.order[self.next:self.next+128];self.next+=len(m);texts=[str(self.data[self.off[i]:self.off[i+1]],"utf-8")for i in m];ids=self.tok.encode(texts,prepend=self.bos,num_threads=4);self.q.extend(zip(texts,ids,self.flag[m].tolist()));return True
 def mix(self,docs,token_lists):
  if not self.active:return docs,token_lists
  t0=time.perf_counter();self.calls+=1;n=sum(len(t)for t in token_lists);self.kit+=n;self.debt+=DSEL_SHARE/(1.0-DSEL_SHARE)*n;docs,token_lists=list(docs),list(token_lists)
  while self.debt>0:
   if not self.q and not self.fill():self.short+=1;self.debt=0.0;break
   text,ids,f=self.q.popleft();docs.append(text);token_lists.append(ids);self.debt-=len(ids);self.added[f][0]+=1;self.added[f][1]+=len(ids)
  self.secs+=time.perf_counter()-t0;return docs,token_lists
 def line(self,head):tok=sum(a[1]for a in self.added);return "dsel %s: %s"%(head,json.dumps({"on":self.active,"at":self.at,"kit":int(self.kit),"docs":sum(a[0]for a in self.added),"tokens":tok,"share":round(tok/max(1.0,tok+self.kit),5),"by_flag":{"N":self.added[1],"E":self.added[2],"both":self.added[3]},"list":0 if self.mine is None else int(len(self.mine)),"next":self.next,"passes":self.passes+1,"short":self.short,"calls":self.calls,"mix_ms":round(1e3*self.secs/max(1,self.calls),3)}))
class RelayBrackets:
 key,share,den,cap,early="dyck",RELAY_SHARE,RELAY_SHARE,RELAY_CAP,0.75
 def __init__(self,rank):s=RELAY_SEED*1000003+rank;self.rng,self.frng,self.crng=random.Random(s),random.Random(s+15485863),random.Random(s+15486863);self.pools=[collections.deque(maxlen=RELAY_POOL)for _ in range(3)];self.debt,self.n=0.0,[0,0,0]
 def take(self,docs):
  for d in docs:
   for a,c,r,k in relay_dyck_cuts(d,self.rng):self.pools[k].append((d,a,c,r))
 def draw(self,w,need_k2):
  rng=self.rng
  for soft in(True,False):
   for _ in range(RELAY_TRIES3):
    q=self.pools[rng.choices(range(3),w)[0]];d,a,c,r=q[rng.randrange(len(q))]
    if soft and need_k2 and r<2:continue
    k=rng.randint(2 if soft and need_k2 else 1,min(3,r));return d,a,c,k,[d[m.start()]for _,m in zip(range(k),RELAY_BR_RE.finditer(d,c))]
 def block(self):
  rng,frng=self.rng,self.frng;w=[cw if len(q)>=f else 0 for cw,q,f in zip(RELAY_CLASS_W,self.pools,RELAY_CLASS_FLOOR)]
  if sum(len(q)for q in self.pools)<16 or not any(w):return None
  rng.randrange(10);frng.randrange(len(RELAY_LAYOUTS));n_io=len(RELAY_LAYOUTS)-2;l1,l2=RELAY_LAYOUTS[n_io+frng.randrange(2)if frng.random()<RELAY_IO_P else frng.randrange(n_io)];n_ex=rng.randint(*RELAY_SHOTS);item=self.draw(w,False);picks,k2=[],0
  for q in range(n_ex-1):picks.append(self.draw(w,k2<RELAY_K2_SHOTS and n_ex-1-q<=RELAY_K2_SHOTS-k2));k2+=picks[-1][3]>=2
  led=self.crng.random()<RELAY_LEAD_P;exs=[]
  for d,a,c,k,ans in picks+[item]:
   inp=[d[m.start()]for m in RELAY_BR_RE.finditer(d,a,c)];t=relay_lead(d,a)if led else None;ex=[]if t is None else[(t+" ",0)]
   if l2.startswith("\n"):ex+=[(l1,0),("".join(" "+x for x in inp[:-1]),1),(" "+inp[-1]+l2,0)]
   else:ex+=[(l1,0),("".join(" "+x for x in inp),1),(l2,0)]
   exs.append(ex+[("".join(" "+x for x in ans),1)])
  return exs
class RelayLetters:
 key,share,den,cap,early="mc",RELAY_MCQ,RELAY_SHARE+RELAY_MCQ,RELAY_CAP,RELAY_MCQ_EARLY
 def __init__(self,rank,st):self.rng=random.Random(RELAY_SEED*1000003+rank+15491237);self.pools=[[collections.deque(maxlen=RELAY_MCQ_POOL)for _ in range(4)]for _ in range(4)];self.wait=[[collections.deque(maxlen=RELAY_MCQ_WAIT)for _ in range(4)]for _ in range(4)];self.admit=[0]*4;self.debt,self.n=0.0,[0,0,0];self.wrng=random.Random(RELAY_SEED*1000003+rank+15531001);self.ps,self.buf,self.st,self.rank=[None,0,-1],[],st,rank
 def take(self,docs):
  if self.st.rows>TRAIN_STARTUP_STEPS_EXCLUDED:
   self.buf+=relay_ps_take(self,"m")
   while len(self.buf)>=RELAY_MCQ_RG or self.ps[2]>=0 and self.buf:
    for rg in self.buf[:RELAY_MCQ_RG]:
     for d,cuts,drawn in rg[2]if rg else():
      for s,k,cut in self.wrng.sample(cuts,RELAY_MCQ_CAP)if drawn else cuts:self.wait[s][k].append(cut)
    del self.buf[:RELAY_MCQ_RG];self.admit_all()
  for d in docs:
   for s,k,cut in relay_mcq_cuts(d,self.rng):self.wait[s][k].append(cut)
  self.admit_all()
 def admit_all(self):
  for s in range(4):
   q4=self.wait[s]
   while all(q4):
    for k in range(4):self.pools[s][k].append(q4[k].pop())
    self.admit[s]+=1
 def block(self):
  rng,need=self.rng,RELAY_SHOTS[1];ok=[s for s in range(4)if all(len(q)>=RELAY_MCQ_CAP*need or len({id(x[0])for x in q})>=need for q in self.pools[s])]
  if not ok:return None
  if rng.random()>=0.5:rng.randrange(len(RELAY_LAYOUTS)-2)
  led=rng.random()<RELAY_LEAD_P;mk=RELAY_MCQ_MARKS[rng.randrange(4)];w=[self.admit[s]for s in ok];exs,used=[],set()
  for _ in range(need):
   q=self.pools[rng.choices(ok,w)[0]][rng.randrange(4)]
   for _t in range(RELAY_MCQ_TRIES):
    d,la,a,ks,lo,he,sp=q[rng.randrange(len(q))]
    if id(d)not in used:break
   else:return None
   used.add(id(d));t=relay_lead(d,la)if led else None
   if t is not None and d.find(t,lo,la)<0:t=None
   exs.append([(("" if t is None else t+" ")+d[a:he]+"".join("\n"+mk.format(L)+" "+d[x:y].strip()for L,(x,y)in zip("ABCD",sp))+"\nAnswer:",0),(" "+d[ks],1)])
  return exs
class RelayEquations:
 key,share,den,cap,early="eq",RELAY_ARITH,RELAY_SHARE+RELAY_MCQ+RELAY_ARITH+(RELAY_YNCAP),8192.0,0.75
 def __init__(self,rank):self.rng=random.Random(RELAY_SEED*1000003+rank+15523001);self.pool=collections.deque(maxlen=RELAY_POOL);self.debt,self.n=0.0,[0,0,0]
 def take(self,docs):
  for d in docs:
   cuts=relay_eq_cuts(d)
   if cuts:self.pool.append((d,)+cuts[self.rng.randrange(len(cuts))])
 def block(self):
  rng,pool=self.rng,self.pool
  if len(pool)<32:return None
  l1,l2=RELAY_EQ_LAYOUTS[rng.randrange(2)];exs,used=[],set()
  for _ in range(RELAY_SHOTS[1]):
   for _t in range(32):
    e=pool[rng.randrange(len(pool))]
    if id(e[0])not in used:break
   else:return None
   used.add(id(e[0]));d=e[0];exs.append([(l1,0),(" "+d[e[1]:e[2]],0),(l2,0),(" "+d[e[3]:e[4]],1)])
  return exs
class RelayYesNo:
 key,share,den,cap,early="yn",RELAY_YNCAP,RELAY_Q2_T,8192.0,0.75
 def __init__(self,rank,st):s=RELAY_SEED*1000003+rank;self.rng,self.wrng=random.Random(s+15507919),random.Random(s+15533003);self.pools=[collections.deque(maxlen=RELAY_POOL)for _ in range(2)];self.debt,self.n,self.st,self.rank=0.0,[0,0,0],st,rank;self.ps=[None,0,-1]
 def put(self,d,cuts,rng):cut=cuts[rng.randrange(len(cuts))];self.pools[1-cut[6]].append((d,)+cut)
 def take(self,docs):
  if self.st.rows>TRAIN_STARTUP_STEPS_EXCLUDED:
   for rg in relay_ps_take(self,"y"):
    for d,cuts in rg[2]if rg else():self.put(d,cuts,self.wrng)
  for d in docs:
   cuts=relay_q2_yncap_cuts(d,relay_nqa_sents(d))
   if cuts:self.put(d,cuts,self.rng)
 def block(self):
  rng,(yes,no)=self.rng,self.pools
  if min(len(yes),len(no))<32:return None
  l1,l2=RELAY_Q2_LAYOUTS[rng.randrange(2)];exs,used=[],set()
  for _ in range(RELAY_SHOTS[1]):
   pool=yes if rng.random()<RELAY_YNCAP_YES else no
   for _t in range(32):
    e=pool[rng.randrange(len(pool))]
    if id(e[0])not in used:break
   else:return None
   used.add(id(e[0]));d=e[0];exs.append([(l1,0),(" "+d[e[1]:e[2]]+"\n"+d[e[3]:e[4]],0),(l2,0),(" "+d[e[5]:e[6]],1)])
  return exs
QA_BOUND_RE = re.compile(r"Question:[ \t]*\S(?:.*\S)?([ \t]+)Answer:[ \t]*\S.*")
def qa_bound(d):
 if "Answer:" not in d:return d
 L=d.split("\n");n=k=len(L)
 while k>1 and(not L[k-1]or QA_BOUND_RE.fullmatch(L[k-1])is not None and L[k-1].count("Question:")==1==L[k-1].count("Answer:")):k-=1
 g=(0x5EED00B0^int.from_bytes(hashlib.md5(d.encode("utf-8","surrogatepass")).digest()[:8],"little"))%3
 if not g or all(not L[i]for i in range(k,n))or"Question:"in L[k-1]or"Answer:"in L[k-1]or not L[k-1].strip():return d
 for i in range(k,n):
  if L[i]:m=QA_BOUND_RE.fullmatch(L[i]);L[i]=L[i][:m.start(1)]+"\n"*g+L[i][m.end(1):]
 return "\n".join(L)
class RelayStream:
 def __init__(self,tokenizer,rank):self.tok,self.bos,self.kit,self.phase,self.rows=tokenizer,tokenizer.get_bos_token_id(),0,0.0,0;self.dsel=None;self.lays=[RelayBrackets(rank),RelayYesNo(rank,self),RelayLetters(rank,self),RelayEquations(rank)]
 def get_bos_token_id(self):return self.bos
 def batches(self,loader):
  for b in loader:self.rows+=1;yield b
 def encode(self,docs,**kw):
  out=self.tok.encode([qa_bound(d)for d in docs],**kw)
  if self.dsel is not None:docs,out=self.dsel.mix(docs,out)
  n=sum(len(t)for t in out);self.kit+=n
  for L in self.lays:
   m=2.0 if self.phase>=0.8 else L.early;L.take(docs);L.debt=min(L.debt+m*L.share*n/(1.0-m*L.den),L.cap)
   while L.debt>0:
    exs=L.block()
    if exs is None:break
    out.append(self.emit(exs,L.n));L.debt-=len(out[-1])
  return out
 def emit(self,exs,cx):
  segs=[]
  for i,ex in enumerate(exs):segs+=([(RELAY_SEP3,0)]if i else[])+ex
  ids,out,ends,pos,j=self.tok.encode("".join(s for s,_ in segs)),[self.bos],[],0,0
  for s,t in segs:pos+=len(s.encode("utf-8"));ends.append((pos,t))
  pos=0
  for t in ids:
   while ends[j][0]<=pos:
    j+=1
   out.append(t if ends[j][1]else-t-1);pos+=len(self.tok.enc.decode_single_token_bytes(t))
  cx[0]+=1;cx[1]+=len(exs);cx[2]+=len(out);return out
 @staticmethod
 def decode(x,y):neg=x<0;x[neg]=-x[neg]-1;y[y<0]=-1
 def line(self,head):
  e={"kit":self.kit,"phase":round(self.phase,6)}
  for L in self.lays:e[L.key]=L.n;e[L.key+"_share"]=round(L.n[2]/max(1,self.kit),6)
  return "relay %s: %s"%(head,json.dumps(e))
def main():
 parser=argparse.ArgumentParser(description="Neuron Competition Round 2 baseline")
 parser.add_argument("--device-type",type=str,default="")
 parser.add_argument("--depth",type=int,default=DEPTH)
 parser.add_argument("--aspect-ratio",type=int,default=ASPECT_RATIO)
 parser.add_argument("--head-dim",type=int,default=HEAD_DIM)
 parser.add_argument("--seq-len",type=int,default=SEQ_LEN)
 parser.add_argument("--device-batch-size",type=int,default=DEVICE_BATCH_SIZE)
 parser.add_argument("--total-batch-size",type=int,default=TOTAL_BATCH_SIZE)
 parser.add_argument("--num-steps",type=int,default=int(os.environ.get("NEURON_COMPETITION_R2_NUM_STEPS",NUM_STEPS)),help="explicit step override; <=0 derives steps from --target-param-data-ratio")
 parser.add_argument("--target-param-data-ratio",type=float,default=float(os.environ.get("NEURON_COMPETITION_R2_TARGET_PARAM_DATA_RATIO",TARGET_PARAM_DATA_RATIO)),help="target token IDs per scaling parameter (Transformer matrices + lm_head)")
 parser.add_argument("--max-train-seconds",type=int,default=int(os.environ.get("NEURON_COMPETITION_R2_MAX_TRAIN_SECONDS",MAX_TRAIN_SECONDS)))
 parser.add_argument("--out-dir",type=str,default=os.environ.get("NEURON_COMPETITION_R2_OUT_DIR","out"))
 parser.add_argument("--compile",action="store_true",default=os.environ.get("NEURON_COMPETITION_R2_COMPILE","1")!="0",help="enable torch.compile on the full model")
 parser.add_argument("--compile-only",action="store_true",help="not supported in this file (no synthetic step is built for the output-site table)")
 args=parser.parse_args()
 dsel_on,dsel,dsel_scan,dsel_group=dsel_check(args),None,None,None
 device_type=detect_device_type(args.device_type)
 ddp,rank,_local_rank,world_size,device=init_runtime(device_type)
 global COMPUTE_DTYPE,FUSED_ATTENTION,TRAINING_KERNELS,ATTN_BAND_KERNEL
 COMPUTE_DTYPE=compute_dtype_for(device)
 FUSED_ATTENTION=fused_attention_for(device)
 ATTN_BAND_KERNEL=FUSED_ATTENTION
 TRAINING_KERNELS=training_kernels_for(device)
 print0(f"device={device} dtype={COMPUTE_DTYPE} world_size={world_size} fused_attention={FUSED_ATTENTION} training_kernels={TRAINING_KERNELS}")
 print0(f"mlp8_kernel: {setup_mlp_kernel(device)}")
 print0(f"attn8_kernel: {setup_attn8_kernel(device)}")
 print0(f"dw8_kernel: {setup_dw8_kernel(device)}")
 moe_check_launch(world_size,args)
 if args.compile_only:
  raise ValueError("--compile-only is not supported in this file")
 MOE_EP.update(rank=rank,world=world_size)
 print0(f"moe_kernel: fp8, the routed experts on the MLPs' pc pair (registered: {MLP_READY})")
 tokenizer=ensure_tokenizer(build_if_missing=False)
 config=build_config(depth=args.depth,aspect_ratio=args.aspect_ratio,head_dim=args.head_dim,sequence_len=args.seq_len,)
 config.os1_bos=tokenizer.get_bos_token_id()
 print0(f"model_config={json.dumps(asdict(config), indent=2)}")
 model=GPT(config)
 model.init_weights()
 model.to(device)
 if not ATTN_BAND_KERNEL:
  attn_band_attach(model)
 num_params=sum(p.numel()for p in model.parameters())
 scaling_param_counts=model.num_scaling_params()
 num_scaling_params=scaling_param_counts["total"]
 flops_per_token=model.estimate_flops()
 print0(f"num_params={num_params:,} scaling_params={num_scaling_params:,} flops_per_token={flops_per_token:e}")
 mc=moe_param_counts(model,num_params)
 print0(f"moe: blocks={','.join(str(i) for i in config.moe_layers)} experts={config.moe_experts} topk={config.moe_topk} capacity={config.moe_capacity} hidden={config.moe_hidden} shared={config.moe_shared} dispatch={config.moe_dispatch} expert_mlp=fused shared_mlp={config.moe_shared_mlp} params_total={mc['total']:,} params_active={mc['active']:,} routed={mc['routed']:,} routed_on_rank={mc['routed_on_rank']:,} trunk={mc['trunk']:,} aux={MOE_AUX:g} aux_seq=0 zloss={MOE_ZLOSS:g} opt=muon lr={MATRIX_LR:g} router_opt=muon router_std={MOE_ROUTER_STD:g} ep=world")
 world_tokens_per_micro=args.device_batch_size*args.seq_len*world_size
 if args.total_batch_size%world_tokens_per_micro!=0:
  raise ValueError(f"total_batch_size={args.total_batch_size} must divide micro tokens={world_tokens_per_micro}")
 grad_accum_steps=args.total_batch_size//world_tokens_per_micro
 if grad_accum_steps&(grad_accum_steps-1):
  raise ValueError(f"grad_accum_steps must be a power of two (got {grad_accum_steps})")
 loss_seed=None
 local_step_batch_size=args.device_batch_size*grad_accum_steps
 if args.num_steps>0:
  target_steps=args.num_steps
  horizon_source="explicit steps"
 else:
  if args.target_param_data_ratio<=0:
   raise ValueError("--target-param-data-ratio must be positive when --num-steps is not set")
  target_tokens=int(args.target_param_data_ratio*num_scaling_params)
  target_steps=target_tokens//args.total_batch_size
  if target_steps<=0:
   raise ValueError("target ratio produces zero optimizer steps")
  horizon_source="data:param ratio"
 target_tokens=target_steps*args.total_batch_size
 actual_target_ratio=target_tokens/num_scaling_params
 clock_schedule=args.num_steps<=0
 if clock_schedule:
  horizon_source="clock (data:param ratio steps reported only)"
 schedule_seconds=max(1.0,args.max_train_seconds-CHECKPOINT_RESERVE_SECONDS)
 print0(f"micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps} local_step_batch_size={local_step_batch_size}")
 print0(f"training_horizon={horizon_source} target_steps={target_steps:,} target_tokens={target_tokens:,} target_scaling_ratio={actual_target_ratio:.4f} max_seconds={args.max_train_seconds} official_budget={TRAIN_TIME_BUDGET_SECONDS}s")
 print0(f"clock_schedule={clock_schedule} stop_reserve={CHECKPOINT_RESERVE_SECONDS:.0f}s rescue_from={RESCUE_START_FRACTION} compile={args.compile}")
 span,unit=(schedule_seconds,"s")if clock_schedule else(target_steps,"steps")
 print0(f"sched1: scope=dense groups: warmdown {DENSE_WARMDOWN_RATIO:g} from {(1 - DENSE_WARMDOWN_RATIO) * span:.1f} {unit}, floor {DENSE_FINAL_LR_FRAC:g}; table: warmdown {WARMDOWN_RATIO:g} from {(1 - WARMDOWN_RATIO) * span:.1f} {unit}, floor {FINAL_LR_FRAC:g}; momentum window {WARMDOWN_RATIO:g}; warmup {WARMUP_STEPS} steps")
 orig_model=model
 global MLP_SCALE_FN,MLP_PRECAST_FN,ATTN8_SCALE_FN,ATTN8_PRECAST_FN
 lean_grad_fn=lean_embedding_grad
 if args.compile:
  compile_backend="neuron" if device_type=="neuron" else "inductor"
  model=torch.compile(model,backend=compile_backend,dynamic=False)
  print0(f"torch.compile enabled (backend={compile_backend})")
  MLP_SCALE_FN=torch.compile(mlp_scales_into,backend=compile_backend,dynamic=False)
  MLP_PRECAST_FN=torch.compile(mlp_precast_into,backend=compile_backend,dynamic=False)
  ATTN8_SCALE_FN=torch.compile(attn8_scales_into,backend=compile_backend,dynamic=False)
  ATTN8_PRECAST_FN=torch.compile(attn8_precast_into,backend=compile_backend,dynamic=False)
  lean_grad_fn=torch.compile(lean_embedding_grad,backend=compile_backend,dynamic=False)
  set_os1_wcat_fn(torch.compile(os1_wcat_into,backend=compile_backend,dynamic=False))
 table=orig_model.transformer.wte.weight
 lean_kernel=lean_kernel_applies(local_step_batch_size*args.seq_len,config.n_embd)
 lean_sink=torch.zeros((args.device_batch_size,args.seq_len,config.n_embd),dtype=COMPUTE_DTYPE,device=device,requires_grad=True)
 lean_dys=[None]*grad_accum_steps
 ve_sinks={k:torch.zeros((args.device_batch_size,args.seq_len,config.n_embd),dtype=COMPUTE_DTYPE,device=device,requires_grad=True)for k in orig_model.value_embeds}
 ve_dys={k:[None]*grad_accum_steps for k in ve_sinks}
 if device.type!="cpu":
  torch.set_num_threads(max(1,(os.cpu_count()or 1)//int(os.environ.get("LOCAL_WORLD_SIZE","1"))))
 ngram_group=dist.new_group(backend="gloo")if world_size>1 else None
 s11_group=dist.new_group(backend="gloo")if world_size>1 else None
 ngram_tag=hashlib.md5(str(Path(args.out_dir).resolve()).encode("utf-8")).hexdigest()[:12]
 n_ord=len(config.ngram_orders)
 ngram_max_rows=local_step_batch_size*args.seq_len*n_ord
 ngram_bytes=config.ngram_rows*(config.ngram_dim*2+4)+config.ngram_rows*(config.ngram_dim+2)+(world_size*world_size*8+world_size*ngram_max_rows*(8+4*config.ngram_dim)if world_size>1 else 0)
 os1_tokens=local_step_batch_size*args.seq_len
 os1_bytes=config.os1_rows*(config.os1_dim+2)+(world_size*world_size*8+world_size*os1_tokens*(8+4*config.os1_dim)if world_size>1 else 0)
 os1_view=config.os1_rows*(config.os1_dim//8+2)
 ngram_int8=config.ngram_rows*(config.ngram_dim+2)
 os1_punch_ok=os1_punch_works(NGRAM_PATH)
 print0(f"os1_room: punch={'works' if os1_punch_ok else 'refused'} folder={NGRAM_PATH}")
 ngram_path=ngram_table_folder(NGRAM_PATH,{ok:max(ngram_bytes-ngram_int8+os1_bytes+(0 if ok else os1_view),ngram_bytes+os1_view)for ok in(True,False)},os1_punch_ok,ngram_group,rank,world_size)
 ngram=NgramHostTable(config.ngram_rows,config.ngram_dim,config.ngram_orders,config.ngram_seed,table.size(0),path=ngram_path,tag=ngram_tag,rank=rank,world=world_size,group=ngram_group,max_rows=ngram_max_rows,beta=NGRAM_BETA,eps=NGRAM_EPS,decay=NGRAM_DECAY,quant=True)
 orig_model.set_ngram_host(ngram)
 os1_typ=os1_typ_rms(config.os1_glin,config.os1_a0,config.os1_dim,os1_tokens,world_size)
 os1=Os1HostTable(config.os1_rows,config.os1_dim,config.os1_order,config.os1_seed,config.os1_code_seed,config.os1_bos,table.size(0),path=ngram_path,tag=ngram_tag+"o",rank=rank,world=world_size,group=dist.new_group(backend="gloo")if world_size>1 else None,max_rows=os1_tokens,lr=OS1_LR,eps=OS1_EPS_REL*os1_typ,cap=OS1_CAP,a0=config.os1_a0,thr=config.os1_thr,gate_w=config.os1_gate_w,gate_h0=config.os1_gate_h0,gate_slope=config.os1_gate_slope,diag=rank==0)
 orig_model.set_os1_host(os1)
 print0(os1.setup_line(OS1_SAVE_SECONDS,os1_typ))
 os1.record_dict["fused"]=1
 print0(f"os1_fused: on route={'kernel' if TRAINING_KERNELS else 'torch'} vchunk={_OS1_VCHUNK} fold_k={config.n_embd}+{config.os1_dim} ops=fused_loss::loss_head_os1_fwd/bwd scorers=unfused")
 os1.record_dict["riders"] = f"tt={OS1_TT} wcat=1 dwx=1 aff=unset"
 print0(f"os1_riders: tt={OS1_TT} wcat=on dwx=on aff=unset")
 ngram_window=torch.zeros((local_step_batch_size,args.seq_len,n_ord,config.ngram_dim),dtype=torch.bfloat16,device=device)
 ngram_grads=[None]*grad_accum_steps
 ngram_step_ids=[None]
 ngram_timing={"gather":0.0,"grad_to_host":0.0,"update":0.0}
 print0(f"ngram_prefault_seconds: {ngram.prefault() + os1.prefault():.1f}")
 print0(f"ngram: orders={','.join(str(n) for n in config.ngram_orders)} rows={config.ngram_rows} dim={config.ngram_dim} lr={NGRAM_LR:g} folder={ngram_path or '(process memory)'} tag={ngram_tag} host_bytes={ngram_bytes:,} host_threads={torch.get_num_threads()}")
 print0(f"ngram_bits: {NGRAM_BITS} view=q_x_s_s_bf16up_max_abs_row_over_7 master=bf16 straight_through=1 save=packed values_GB={config.ngram_rows * config.ngram_dim * NGRAM_BITS / 8 / 1e9:.3f} scales_GB={config.ngram_rows * 2 / 1e9:.3f}")
 relay=RelayStream(tokenizer,rank)
 if dsel_on:
  dsel=relay.dsel=DselStream(tokenizer,dsel_root(args.out_dir),rank,world_size)
  print0(dsel.launch_line())
 train_loader=relay.batches(make_dataloader(relay,1,local_step_batch_size*args.seq_len,"train","cpu"))
 print0(f"relay: on: {[(L.key, L.share) for L in relay.lays]} of the kit's tokens, seed {RELAY_SEED}, {len(RELAY_LAYOUTS)} marker layouts md5 {hashlib.md5(repr(RELAY_LAYOUTS).encode()).hexdigest()[:12]}")
 window_x=torch.empty((local_step_batch_size,args.seq_len),dtype=torch.int32,device=device)
 window_y=torch.empty((local_step_batch_size,args.seq_len),dtype=torch.long,device=device)
 lean_cid=torch.empty(local_step_batch_size*args.seq_len,dtype=torch.int32,device=device)
 lean_rows=torch.empty(LEAN_COMPACT_ROWS,dtype=torch.long,device=device)
 lean_host_ids=[None]
 bos_token_id=tokenizer.get_bos_token_id()
 def next_step_batch():
  x,y,state=next(train_loader)
  RelayStream.decode(x,y)
  y.masked_fill_(y==bos_token_id,-1)
  window_x.copy_(x.view(local_step_batch_size,args.seq_len))
  window_y.copy_(y.view(local_step_batch_size,args.seq_len))
  relabel=lean_relabel(x,table.size(0))
  lean_host_ids[0]=x.clone()if relabel is None else None
  if relabel is not None:
   lean_cid.copy_(relabel[0])
   lean_rows.copy_(relabel[1])
  t_g=time.perf_counter()
  ngram_step_ids[0]=ngram.ids(x.view(local_step_batch_size,args.seq_len))
  ngram_window.copy_(ngram.gather(ngram_step_ids[0]))
  ngram_timing["gather"]+=time.perf_counter()-t_g
  return window_x,window_y,state
 step_x,step_y,loader_state=next_step_batch()
 os1.first_record(step_x.to("cpu"),step_y.to("cpu"))
 optimizer=orig_model.setup_optimizer()
 if args.compile and hasattr(optimizer,"compile_parts"):
  optimizer.compile_parts(compile_backend)
 s12_info={"targets":len(optimizer._rest_targets),"numel":sum(t.numel()for t in optimizer._rest_targets),"dtypes":",".join(sorted({str(t.dtype).replace("torch.","")for t in optimizer._rest_targets}))}
 print0(f"S12 accnowrite: on mode={'compiled' if args.compile else 'eager'} flag=False "+ " ".join(f"{k}={v}" for k, v in s12_info.items()))
 optimizer.enable_stage()
 print0("S6 stage: on (the update math's schedule tensors written once at each step's start)")
 print0(f"S11 boundary: on (a step's tail read in the next step from step {TRAIN_STARTUP_STEPS_EXCLUDED} on; control on gloo; event_class={'neuron' if device.type == 'neuron' else 'none'})")
 _gm=getattr(optimizer,"Gm",None)
 _mode=("compiled" if args.compile else "eager")if _gm is not None else "off"
 print0(f"sysoverlap: rs_unsplit=1 split_env={os.environ.get('NEURON_USE_SPLIT_REDUCE_SCATTER', 'unset')} slab_dtype={getattr(_gm, 'dtype', None)} gm_dtype={getattr(_gm, 'dtype', None)} gm_is_wm={_gm is not None and getattr(optimizer, 'Wm', None) is _gm} slab_sum={_gm is not None} acc_fold={_mode} fold_runs={len(getattr(optimizer, '_fold_bases', None) or [])} host_idle=1 loss_div={grad_accum_steps} opt_compiled={_mode} class={type(optimizer).__name__}")
 def worker_fetch():
  x,y,state=next(train_loader)
  RelayStream.decode(x,y)
  y.masked_fill_(y==bos_token_id,-1)
  return x,y,state
 ngram_worker=NgramWorker(os1,worker_fetch,window_x,window_y,(lean_cid,lean_rows,lean_host_ids),table.size(0),ngram,torch.cat([ngram_window,torch.zeros_like(ngram_window[:,:,:1])],dim=2),ngram_step_ids[0],(local_step_batch_size,args.seq_len),ngram_timing)
 ngram_next_pending=None
 os1_next_pending=None
 os1_wcat_refresh(orig_model)
 mlp_refresh_scales(orig_model)
 attn8_refresh_scales(orig_model)
 print0(f"mlp8_scales: start {mlp_scales_line(orig_model)}")
 print0(f"attn8_scales: start {attn8_scales_line(orig_model)}")
 dsel_group=dist.new_group(backend="gloo")if dsel is not None and world_size>1 else None
 loop_started=time.monotonic()
 budget_started=loop_started
 smooth_loss=0.0
 total_train_tokens=0
 step=0
 recent_step_times=[]
 startup_allowance_est=0.0
 clock_seconds=0.0
 rescue_names=[]
 rescue_copy=[]
 rescue_step=-1
 rescue_tokens=0
 rescue_meta=None
 sched_now=None
 moe_rescue=None
 moe_stats=None
 moe_last=""
 s11=BoundaryS11(device,s11_group,rank)
 s11_rec=None
 def s11_tail(rec):
  nonlocal smooth_loss,total_train_tokens,clock_seconds,rescue_names,rescue_copy,rescue_step,rescue_meta
  nonlocal rescue_tokens,moe_rescue
  k=rec["step"]
  s11.wait(rec["event"])
  record_step_boundary(k)
  now=time.monotonic()
  dt=now-s11.prev_obs
  s11.prev_obs=now
  s11.n["tails"]+=1
  recent_step_times.append(dt)
  train_loss=float(rec["loss"].item())
  train_loss*=grad_accum_steps
  elapsed_after_step=time.monotonic()-budget_started
  stop_after_step=((not clock_schedule and k+1>=target_steps)or elapsed_after_step>=args.max_train_seconds or elapsed_after_step+max(recent_step_times[-20:])+CHECKPOINT_RESERVE_SECONDS>=args.max_train_seconds)
  near=clock_schedule and(elapsed_after_step+3*max(recent_step_times[-20:])+CHECKPOINT_RESERVE_SECONDS>=args.max_train_seconds)
  vals=s11.control([1 if(math.isnan(train_loss)or train_loss>100)else 0,1 if stop_after_step else 0,int(elapsed_after_step*1000)if rank==0 and clock_schedule else 0,1 if near else 0])
  if clock_schedule:
   clock_seconds=vals[2]/1000.0
  if vals[3]:
   s11.near_end=True
  if vals[0]:
   if rescue_step<0:
    raise RuntimeError(f"bad loss detected: {train_loss}")
   rescue_meta={"bad_loss_step":k,"bad_loss_rank0":train_loss,"snapshot_step":rescue_step}
   return "bad"
  step_tokens=args.total_batch_size
  total_train_tokens+=step_tokens
  smooth_loss=0.9*smooth_loss+0.1*train_loss
  debiased=smooth_loss/(1-0.9**(k+1))
  tok_per_sec=step_tokens/max(dt,1e-6)
  if rank==0 and(k<5 or k%10==0):
   if clock_schedule:
    horizon_left = f"clock {clock_seconds / schedule_seconds:.4f}"
   else:
    horizon_left = f"remaining_steps {target_steps - k - 1}"
   remaining_time=max(0.0,args.max_train_seconds-(time.monotonic()-budget_started))
   print(f"step {k:05d} | loss {debiased:.4f} | lrm {rec['lrm']:.3f} | dt {dt:.2f}s | tok/s {tok_per_sec:,.0f} | {horizon_left} | remaining_time {remaining_time:.0f}s | epoch {loader_state['epoch']}",flush=True,)
  if clock_schedule:
   horizon_fraction=clock_seconds/args.max_train_seconds
  else:
   horizon_fraction=(k+1)/target_steps
  if horizon_fraction>=RESCUE_START_FRACTION and(rescue_step<0 or k+1-rescue_step>=RESCUE_EVERY_STEPS):
   if rank==0:
    t_r=time.perf_counter()
    state=orig_model.state_dict()
    if not rescue_copy:
     rescue_names=list(state.keys())
     rescue_copy=[torch.empty(t.shape,dtype=t.dtype,device="cpu")for t in state.values()]
    for dst,src in zip(rescue_copy,state.values()):
     dst.copy_(src.detach().to("cpu"))
    MOE_RESCUE["copies"]+=1
    MOE_RESCUE["seconds"]+=time.perf_counter()-t_r
   moe_rescue=moe_rescue_refresh(orig_model,moe_rescue)
   rescue_step=k+1
   rescue_tokens=total_train_tokens
  return "stop" if vals[1]else None
 while clock_schedule or step<target_steps:
  s11_in=s11_rec is not None
  if not s11_in:
   synchronize(device)
  t0=time.monotonic()
  relay.phase=clock_seconds/schedule_seconds if clock_schedule else step/target_steps
  if dsel is not None:
   dsel.phase=relay.phase
   if rank==0 and dsel_scan is None and not dsel.decided and step==TRAIN_STARTUP_STEPS_EXCLUDED:
    dsel_scan=dsel_start(dsel.root,world_size,step)
   if not dsel.decided and dsel.phase>=DSEL_FROM:
    dsel.turn(dsel_group,dsel_scan,step)
   if rank==0 and step%500==0:
    print(dsel.line(f"step {step} rank 0"), flush=True)
  if not s11_in:
   s6_lrm=(lr_multiplier_by_clock(step,clock_seconds/schedule_seconds,*DENSE_SCHED)if clock_schedule else lr_multiplier(step,target_steps,*DENSE_SCHED))
   s6_mom=muon_momentum_schedule(step,clock_seconds/schedule_seconds if clock_schedule else step/target_steps)
   for group in optimizer.param_groups:
    group["lr"]=group["initial_lr"]*s6_lrm
    if group["kind"]=="muon":
     group["momentum"]=s6_mom
     group["weight_decay"]=WEIGHT_DECAY
   optimizer.stage_schedule()
  ngram_worker.start()
  optimizer.zero_grad(set_to_none=True)
  last_loss=None
  for micro_step in range(grad_accum_steps):
   start=micro_step*args.device_batch_size
   end=start+args.device_batch_size
   leaf=ngram_worker.bn[ngram_worker.cur][start:end].detach().to(COMPUTE_DTYPE).requires_grad_(True)
   loss=model(step_x[start:end],step_y[start:end],emb_grad_sink=lean_sink,ngram_emb=leaf,ve_grad_sinks=ve_sinks,loss_div=grad_accum_steps)
   loss,moe_stats=loss
   last_loss=moe_stats[0]
   if loss_seed is None:
    loss_seed=torch.ones_like(loss)
   loss.backward(gradient=loss_seed)
   lean_dys[micro_step]=lean_sink.grad
   lean_sink.grad=None
   for k,s in ve_sinks.items():
    ve_dys[k][micro_step]=s.grad
    s.grad=None
   ngram_grads[micro_step]=leaf.grad
   accumulate_gradients(optimizer)
  if lean_host_ids[0]is None:
   lean_grad=lean_grad_fn(lean_dys,lean_cid,lean_rows,table.size(0),lean_kernel)
  else:
   lean_grad=lean_host_grad(lean_dys,lean_host_ids[0],table.size(0))
   print(f"rank {rank}: step {step} has more than {LEAN_COMPACT_ROWS} distinct tokens; its embedding gradient was summed on the host", flush=True)
  set_lean_embedding_grad(optimizer,table,lean_grad)
  for k in ve_sinks:
   if lean_host_ids[0]is None:
    ve_grad=lean_grad_fn(ve_dys[k],lean_cid,lean_rows,table.size(0),lean_kernel)
   else:
    ve_grad=lean_host_grad(ve_dys[k],lean_host_ids[0],table.size(0))
   set_lean_embedding_grad(optimizer,orig_model.value_embeds[k].weight,ve_grad)
  if GRAD_CLIP>0:
   torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
  if s11_rec is not None:
   s11_out=s11_tail(s11_rec)
   s11_rec=None
   t0=s11.prev_obs
   if s11_out is not None:
    break
  if clock_schedule:
   lrm=lr_multiplier_by_clock(step,clock_seconds/schedule_seconds,*DENSE_SCHED)
   rows_lrm=lr_multiplier_by_clock(step,clock_seconds/schedule_seconds,*ROWS_SCHED)
  else:
   lrm=lr_multiplier(step,target_steps,*DENSE_SCHED)
   rows_lrm=lr_multiplier(step,target_steps,*ROWS_SCHED)
  sched_progress=clock_seconds/schedule_seconds if clock_schedule else step/target_steps
  muon_momentum=muon_momentum_schedule(step,sched_progress)
  if rank==0:
   sched_now=(step,sched_progress,lrm,rows_lrm,muon_momentum)
   if step==100 or step%500==0:
    print(sched_line("step",sched_now),flush=True)
  for group in optimizer.param_groups:
   group["lr"]=group["initial_lr"]*lrm
   if group["kind"]=="muon":
    group["momentum"]=muon_momentum
    group["weight_decay"]=WEIGHT_DECAY
  if s11_in:
   optimizer.stage_schedule(non_blocking=True)
  optimizer.step()
  t_g=time.perf_counter()
  ngram_grad_host=torch.cat(ngram_grads).to("cpu",non_blocking=True)
  s11_loss_host=last_loss.to("cpu",non_blocking=True)
  s11_event=s11.record()
  s11.ms["read"]+=1000.0*(time.perf_counter()-t_g)
  ngram_worker.os1_release()
  if step==TRAIN_STARTUP_STEPS_EXCLUDED:
   print0(f"S11 pinned: grad={ngram_grad_host.is_pinned()} loss={s11_loss_host.is_pinned()}")
  ngram_timing["grad_to_host"]+=time.perf_counter()-t_g
  ngram_ids_now=ngram_worker.ids[ngram_worker.cur]
  ngram_next_pending=(ngram_ids_now,ngram_grad_host,NGRAM_LR*rows_lrm,step)
  ngram_next_pending+=(s11_event,)
  ngram_next_pending,os1_next_pending=os1.split_pending(ngram_next_pending,ngram_worker)
  s11_out_async=s11.exit_async(step,not clock_schedule and step+1>=target_steps)
  if not s11_out_async:
   os1_wcat_refresh(orig_model)
   mlp_refresh_scales(orig_model)
   attn8_refresh_scales(orig_model)
   synchronize(device)
   record_step_boundary(step)
  step_x,step_y,loader_state=ngram_worker.finish()
  ngram_worker.pending=ngram_next_pending
  ngram_worker.os1_pending=os1_next_pending
  if s11_out_async:
   os1_wcat_refresh(orig_model)
   mlp_refresh_scales(orig_model)
   attn8_refresh_scales(orig_model)
   s11_rec={"step":step,"event":s11_event,"loss":s11_loss_host,"lrm":lrm}
   step+=1
   continue
  dt=time.monotonic()-t0
  s11.prev_obs=time.monotonic()
  recent_step_times.append(dt)
  train_loss=float(last_loss.item())*grad_accum_steps
  if step+1==TRAIN_STARTUP_STEPS_EXCLUDED:
   now=time.monotonic()
   observed_startup=now-_PROCESS_STARTED
   startup_allowance_est=min(observed_startup,TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
   excess=max(0.0,observed_startup+STARTUP_LAUNCH_MARGIN_SECONDS-TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS,)
   budget_started=now-excess
  elapsed_after_step=time.monotonic()-budget_started
  reached_min_steps=step+1>=TRAIN_STARTUP_STEPS_EXCLUDED
  stop_after_step=((not clock_schedule and step+1>=target_steps)or(reached_min_steps and(elapsed_after_step>=args.max_train_seconds or elapsed_after_step+max(recent_step_times[-20:])+CHECKPOINT_RESERVE_SECONDS>=args.max_train_seconds)))
  control_values=[1 if(math.isnan(train_loss)or train_loss>100)else 0,1 if stop_after_step else 0,]
  if clock_schedule:
   control_values.append(int(elapsed_after_step*1000)if rank==0 and reached_min_steps else 0)
  control=torch.tensor(control_values,device=device,dtype=torch.int32)
  if dist.is_available()and dist.is_initialized():
   dist.all_reduce(control,op=dist.ReduceOp.MAX)
  if clock_schedule:
   clock_seconds=int(control[2].item())/1000.0
  if int(control[0].item())!=0:
   if rescue_step<0:
    raise RuntimeError(f"bad loss detected: {train_loss}")
   rescue_meta={"bad_loss_step":step,"bad_loss_rank0":train_loss,"snapshot_step":rescue_step}
   step+=1
   break
  step_tokens=args.total_batch_size
  total_train_tokens+=step_tokens
  smooth_loss=0.9*smooth_loss+0.1*train_loss
  debiased=smooth_loss/(1-0.9**(step+1))
  tok_per_sec=step_tokens/max(dt,1e-6)
  if rank==0 and(step<5 or step%10==0):
   if clock_schedule:
    horizon_left = f"clock {clock_seconds / schedule_seconds:.4f}"
   else:
    horizon_left = f"remaining_steps {target_steps - step - 1}"
   remaining_time=max(0.0,args.max_train_seconds-(time.monotonic()-budget_started))
   print(f"step {step:05d} | loss {debiased:.4f} | lrm {lrm:.3f} | dt {dt:.2f}s | tok/s {tok_per_sec:,.0f} | {horizon_left} | remaining_time {remaining_time:.0f}s | epoch {loader_state['epoch']}",flush=True,)
  if step<3 or step%100==0 or(step+1)%BOUNDARY_SYNC_EVERY==0:
   moe_last=moe_stats_line(orig_model,moe_stats,step,world_size)
   if rank==0:
    print(moe_last,flush=True)
  if clock_schedule:
   horizon_fraction=clock_seconds/args.max_train_seconds
  else:
   horizon_fraction=(step+1)/target_steps
  if step==0 or(horizon_fraction>=RESCUE_START_FRACTION and(rescue_step<0 or step+1-rescue_step>=RESCUE_EVERY_STEPS)):
   if rank==0:
    t_r=time.perf_counter()
    state=orig_model.state_dict()
    if not rescue_copy:
     rescue_names=list(state.keys())
     rescue_copy=[torch.empty(t.shape,dtype=t.dtype,device="cpu")for t in state.values()]
    for dst,src in zip(rescue_copy,state.values()):
     dst.copy_(src.detach().to("cpu"))
    MOE_RESCUE["copies"]+=1
    MOE_RESCUE["seconds"]+=time.perf_counter()-t_r
   moe_rescue=moe_rescue_refresh(orig_model,moe_rescue)
   if step>0:
    rescue_step=step+1
    rescue_tokens=total_train_tokens
  if step==0:
   gc.collect()
   gc.freeze()
   gc.disable()
  step+=1
  if int(control[1].item())!=0:
   break
 if moe_stats is not None:
  moe_last=moe_stats_line(orig_model,moe_stats,step,world_size)
 ngram_worker.flush()
 if rank==0 and sched_now is not None:
  print(sched_line("last step",sched_now),flush=True)
 checkpoint_path=Path(args.out_dir)/"final.pt"
 budget_elapsed_before_save=time.monotonic()-budget_started
 check_declared_length(orig_model,args.seq_len)
 os1.save_tables(ngram)
 print0(f"ngram_quantize_seconds: {ngram.stats['quantize_seconds']:.1f}")
 moe_saved=moe_gather_experts(orig_model,rank,world_size,moe_rescue if rescue_meta is not None else None)
 if rank==0 and rescue_meta is None:
  save_checkpoint(checkpoint_path,orig_model,{"step":step,"total_train_tokens":total_train_tokens,"budget_elapsed_before_save":budget_elapsed_before_save,"world_size":world_size,"compiled":args.compile,"sched_stage":dict(optimizer._stage_n),"dense_sched":{"scope":"dense","dense":list(DENSE_SCHED),"rows":list(ROWS_SCHED),"momentum_window":WARMDOWN_RATIO},"moe":moe_record(config),"os1":os1.record_dict,},ngram.state_for_checkpoint()|os1.state_for_checkpoint()|moe_saved)
 elif rank==0:
  save_rescue_checkpoint(checkpoint_path,orig_model,dict(zip(rescue_names,rescue_copy)),{"step":rescue_step,"total_train_tokens":rescue_tokens,"budget_elapsed_before_save":budget_elapsed_before_save,"world_size":world_size,"compiled":args.compile,"rescue":rescue_meta,"sched_stage":dict(optimizer._stage_n),"dense_sched":{"scope":"dense","dense":list(DENSE_SCHED),"rows":list(ROWS_SCHED),"momentum_window":WARMDOWN_RATIO},"moe":moe_record(config),"os1":os1.record_dict,},ngram.state_for_checkpoint()|os1.state_for_checkpoint()|moe_saved)
 if dist.is_available()and dist.is_initialized():
  dist.barrier()
 budget_elapsed_after_save=time.monotonic()-budget_started
 print0(f"ngram_save_seconds: {budget_elapsed_after_save - budget_elapsed_before_save:.1f}")
 if rank==0:
  print("---")
  print(f"checkpoint_path:    {checkpoint_path}")
  print(f"training_seconds:  {budget_elapsed_before_save:.1f}")
  print(f"budget_seconds:    {TRAIN_TIME_BUDGET_SECONDS}")
  print(f"horizon_source:    {horizon_source}")
  print(f"target_steps:      {target_steps}")
  print(f"target_tokens_M:   {target_tokens / 1e6:.1f}")
  print(f"target_ratio:      {args.target_param_data_ratio:.4f}")
  print(f"actual_target_ratio: {actual_target_ratio:.4f}")
  print(f"max_train_seconds: {args.max_train_seconds}")
  print(f"startup_allowance_est: {startup_allowance_est:.1f}")
  print(f"charged_seconds_est:   {budget_elapsed_after_save:.1f}")
  print(f"over_budget:       {budget_elapsed_after_save > args.max_train_seconds}")
  print(f"total_seconds:     {time.monotonic() - loop_started:.1f}")
  print(f"compiled:          {args.compile}")
  print(f"num_steps:         {step}")
  print(f"rescue:            {rescue_meta}")
  print(f"total_tokens_M:    {total_train_tokens / 1e6:.1f}")
  print(f"world_size:        {world_size}")
  print(f"lnc:               {os.environ.get('NEURON_LOGICAL_NC_CONFIG', '')}")
  print(f"depth:             {args.depth}")
  print(f"aspect_ratio:      {args.aspect_ratio}")
  print(f"seq_len:           {args.seq_len}")
  print(f"device_batch_size: {args.device_batch_size}")
  print(f"total_batch_size:  {args.total_batch_size}")
  print(f"num_params_M:      {num_params / 1e6:.1f}")
  print(f"scaling_params_M:  {num_scaling_params / 1e6:.1f}")
  print(f"actual_ratio:      {total_train_tokens / num_scaling_params:.4f}")
  print(f"mlp8_kernel:       {mlp_summary()}")
  print(f"mlp8_scales:       end {mlp_scales_line(orig_model)}")
  print(f"attn8_kernel:      {attn8_summary()}")
  print(f"attn8_scales:      end {attn8_scales_line(orig_model)}")
  print(f"dw8_kernel:        {dw8_summary()}")
  print(f"moe_end:           {moe_last or '-'}")
  print(f"moe_kernel:        fp8 ready={MLP_READY}")
  print(f"moe_rescue:        host copies={MOE_RESCUE['copies']} seconds={MOE_RESCUE['seconds']:.2f}")
 print(f"S6 stage rank {rank}: " + " ".join(f"{k}={v}" for k, v in optimizer._stage_n.items()), flush=True)
 print(s11.line(rank,ngram_timing),flush=True)
 print(f"S12 accnowrite rank {rank}: " + " ".join(f"{k}={v}" for k, v in optimizer._s12_n.items()), flush=True)
 print(relay.line(f"rank {rank}"), flush=True)
 if dsel is not None:
  print(dsel.line(f"rank {rank}"), flush=True)
  dsel.data=None
  if dsel_group is not None:
   dist.barrier(group=dsel_group)
  if dsel_scan is not None:
   dsel_close(dsel_scan,dsel.root)
 n=max(1,ngram.stats["steps"])
 touched=ngram.stats["rows_with_state"]
 print(f"ngram_host rank {rank}: steps={ngram.stats['steps']} gather_ms={1000 * ngram_timing['gather'] / n:.2f} grad_to_host_ms={1000 * ngram_timing['grad_to_host'] / n:.2f} update_ms={1000 * ngram_timing['update'] / n:.2f} rows_updated={ngram.stats['rows_updated']} skipped_steps={ngram.stats['skipped']} rows_with_state={touched} touched_fraction={touched / config.ngram_rows:.4f} wait_ms={1000 * ngram_timing['wait'] / n:.2f}",flush=True)
 if dist.is_available()and dist.is_initialized():
  dist.barrier()
 print(os1.end_line(ngram_timing),flush=True)
 print(f"os1_riders rank {rank}: tt_worker={ngram_timing.get('tt_worker', 'unset')} tt_os1={ngram_timing.get('tt_os1', 'unset')} main_threads={torch.get_num_threads()}", flush=True)
 ngram.close(remove=rank==0)
 os1.close(remove=rank==0)
 cleanup_runtime()
if __name__ == "__main__":main()
