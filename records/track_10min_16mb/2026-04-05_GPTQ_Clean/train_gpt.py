from __future__ import annotations
import copy,glob,io,math,os,random,subprocess,sys,time,uuid
from pathlib import Path
import lzma
_COMPRESSOR='lzma'
import numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from torch import Tensor,nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:from flash_attn_interface import flash_attn_func as flash_attn_3_func;_HAS_FA3=True
except ImportError:
 try:from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_3_func;_HAS_FA3=True
 except ImportError:
  try:from flash_attn import flash_attn_func as flash_attn_3_func;_HAS_FA3=True
  except ImportError:_HAS_FA3=False;flash_attn_3_func=None
class Hyperparameters:data_path=os.environ.get('DATA_PATH','./data/datasets/fineweb10B_sp1024');train_files=os.path.join(data_path,'fineweb_train_*.bin');val_files=os.path.join(data_path,'fineweb_val_*.bin');tokenizer_path=os.environ.get('TOKENIZER_PATH','./data/tokenizers/fineweb_1024_bpe.model');run_id=os.environ.get('RUN_ID',str(uuid.uuid4()));seed=int(os.environ.get('SEED',1337));val_batch_size=int(os.environ.get('VAL_BATCH_SIZE',524288));val_loss_every=int(os.environ.get('VAL_LOSS_EVERY',4000));train_log_every=int(os.environ.get('TRAIN_LOG_EVERY',500));iterations=int(os.environ.get('ITERATIONS',20000));warmdown_iters=int(os.environ.get('WARMDOWN_ITERS',3500));warmup_steps=int(os.environ.get('WARMUP_STEPS',20));train_batch_tokens=int(os.environ.get('TRAIN_BATCH_TOKENS',786432));train_seq_len=int(os.environ.get('TRAIN_SEQ_LEN',2048));eval_seq_len=int(os.environ.get('EVAL_SEQ_LEN',2048));max_wallclock_seconds=float(os.environ.get('MAX_WALLCLOCK_SECONDS',6e2));qk_gain_init=float(os.environ.get('QK_GAIN_INIT',4.));vocab_size=int(os.environ.get('VOCAB_SIZE',1024));num_layers=int(os.environ.get('NUM_LAYERS',11));num_kv_heads=int(os.environ.get('NUM_KV_HEADS',4));model_dim=int(os.environ.get('MODEL_DIM',512));num_heads=int(os.environ.get('NUM_HEADS',8));mlp_mult=float(os.environ.get('MLP_MULT',3.));tie_embeddings=bool(int(os.environ.get('TIE_EMBEDDINGS','1')));rope_base=float(os.environ.get('ROPE_BASE',1e4));logit_softcap=float(os.environ.get('LOGIT_SOFTCAP',3e1));embed_lr=float(os.environ.get('EMBED_LR',.6));head_lr=float(os.environ.get('HEAD_LR',.008));tied_embed_lr=float(os.environ.get('TIED_EMBED_LR',.035));tied_embed_init_std=float(os.environ.get('TIED_EMBED_INIT_STD',.005));matrix_lr=float(os.environ.get('MATRIX_LR',.025));scalar_lr=float(os.environ.get('SCALAR_LR',.025));muon_momentum=float(os.environ.get('MUON_MOMENTUM',.99));muon_backend_steps=int(os.environ.get('MUON_BACKEND_STEPS',5));muon_momentum_warmup_start=float(os.environ.get('MUON_MOMENTUM_WARMUP_START',.92));muon_momentum_warmup_steps=int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS',1500));beta1=float(os.environ.get('BETA1',.9));beta2=float(os.environ.get('BETA2',.95));adam_eps=float(os.environ.get('ADAM_EPS',1e-08));grad_clip_norm=float(os.environ.get('GRAD_CLIP_NORM',.3));eval_stride=int(os.environ.get('EVAL_STRIDE',96));mtp_num_heads=int(os.environ.get('MTP_NUM_HEADS',0));mtp_loss_weight=float(os.environ.get('MTP_LOSS_WEIGHT',.2));muon_beta2=float(os.environ.get('MUON_BETA2',.95));swa_enabled=bool(int(os.environ.get('SWA_ENABLED','1')));swa_every=int(os.environ.get('SWA_EVERY',50));muon_wd=float(os.environ.get('MUON_WD',.04));adam_wd=float(os.environ.get('ADAM_WD',.04));qat_enabled=bool(int(os.environ.get('QAT_ENABLED','0')));bigram_vocab_size=int(os.environ.get('BIGRAM_VOCAB_SIZE',1024));bigram_dim=int(os.environ.get('BIGRAM_DIM',128));xsa_last_n=int(os.environ.get('XSA_LAST_N',11));rope_dims=int(os.environ.get('ROPE_DIMS',16));ln_scale=bool(int(os.environ.get('LN_SCALE','1')));dtg_enabled=bool(int(os.environ.get('DTG_ENABLED','0')));late_qat_threshold=float(os.environ.get('LATE_QAT_THRESHOLD',.15));ve_enabled=bool(int(os.environ.get('VE_ENABLED','1')));ve_dim=int(os.environ.get('VE_DIM',128));ve_layers=os.environ.get('VE_LAYERS','9,10');vrl_enabled=bool(int(os.environ.get('VRL_ENABLED','1')));ogd_enabled=bool(int(os.environ.get('OGD_ENABLED','1')));ogd_lr=float(os.environ.get('OGD_LR',.1));cache_lambda=float(os.environ.get('CACHE_LAMBDA',.02));cache_decay=float(os.environ.get('CACHE_DECAY',.995));ttt_enabled=bool(int(os.environ.get('TTT_ENABLED','1')));ttt_lr=float(os.environ.get('TTT_LR',.001));ttt_epochs=int(os.environ.get('TTT_EPOCHS',3));ttt_chunk_tokens=int(os.environ.get('TTT_CHUNK_TOKENS',32768));ttt_freeze_blocks=int(os.environ.get('TTT_FREEZE_BLOCKS',0));ttt_momentum=float(os.environ.get('TTT_MOMENTUM',.9));ttt_batch_seqs=int(os.environ.get('TTT_BATCH_SEQS',32));ttt_grad_clip=float(os.environ.get('TTT_GRAD_CLIP',1.))
def zeropower_via_newtonschulz5(G:Tensor,steps:int=10,eps:float=1e-07)->Tensor:
 D,E,F=3.4445,-4.775,2.0315;A=G.bfloat16();A/=A.norm()+eps;C=G.size(0)>G.size(1)
 if C:A=A.T
 for I in range(steps):B=A@A.T;H=E*B+F*B@B;A=D*A+H@A
 return A.T if C else A
class Muon(torch.optim.Optimizer):
 def __init__(A,params,lr:float,momentum:float,backend_steps:int,nesterov:bool=True,weight_decay:float=.0):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay))
 @torch.no_grad()
 def step(self,closure=None):
  I=closure;J=None
  if I is not None:
   with torch.enable_grad():J=I()
  F=dist.is_available()and dist.is_initialized();O=dist.get_world_size()if F else 1;P=dist.get_rank()if F else 0
  for D in self.param_groups:
   E=D['params']
   if not E:continue
   K=D['lr'];L=D['momentum'];Q=D['backend_steps'];R=D['nesterov'];S=sum(int(A.numel())for A in E);G=torch.zeros(S,device=E[0].device,dtype=torch.bfloat16);C=0
   for(T,A)in enumerate(E):
    if T%O==P and A.grad is not None:
     B=A.grad;H=self.state[A]
     if'momentum_buffer'not in H:H['momentum_buffer']=torch.zeros_like(B)
     M=H['momentum_buffer'];M.mul_(L).add_(B)
     if R:B=B.add(M,alpha=L)
     B=zeropower_via_newtonschulz5(B,steps=Q);B*=max(1,B.size(0)/B.size(1))**.5;G[C:C+A.numel()]=B.reshape(-1)
    C+=A.numel()
   if F:dist.all_reduce(G,op=dist.ReduceOp.SUM)
   N=D.get('weight_decay',.0);C=0
   for A in E:
    if N>.0:A.data.mul_(1.-K*N)
    B=G[C:C+A.numel()].view_as(A).to(dtype=A.dtype);A.add_(B,alpha=-K);C+=A.numel()
  return J
def build_sentencepiece_luts(sp:spm.SentencePieceProcessor,vocab_size:int,device:torch.device)->tuple[Tensor,Tensor,Tensor]:
 D=device;B=sp;G=int(B.vocab_size());E=max(G,vocab_size);F=np.zeros((E,),dtype=np.int16);H=np.zeros((E,),dtype=np.bool_);I=np.ones((E,),dtype=np.bool_)
 for A in range(G):
  if B.is_control(A)or B.is_unknown(A)or B.is_unused(A):continue
  I[A]=False
  if B.is_byte(A):F[A]=1;continue
  C=B.id_to_piece(A)
  if C.startswith('▁'):H[A]=True;C=C[1:]
  F[A]=len(C.encode('utf-8'))
 return torch.tensor(F,dtype=torch.int16,device=D),torch.tensor(H,dtype=torch.bool,device=D),torch.tensor(I,dtype=torch.bool,device=D)
def load_validation_tokens(pattern:str,seq_len:int)->Tensor:
 B=pattern;A=seq_len;C=[Path(A)for A in sorted(glob.glob(B))]
 if not C:raise FileNotFoundError(f"No files found for pattern: {B}")
 D=torch.cat([load_data_shard(A)for A in C]).contiguous();E=(D.numel()-1)//A*A
 if E<=0:raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={A}")
 return D[:E+1]
def eval_val(args:Hyperparameters,model:nn.Module,rank:int,world_size:int,device:torch.device,grad_accum_steps:int,val_tokens:Tensor,base_bytes_lut:Tensor,has_leading_space_lut:Tensor,is_boundary_token_lut:Tensor,eval_seq_len:int|None=None)->tuple[float,float]:
 K=val_tokens;J=grad_accum_steps;F=model;E=args;C=device;B=world_size;A=eval_seq_len or E.train_seq_len;L=E.val_batch_size//(B*J)
 if L<A:raise ValueError(f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={E.val_batch_size}, WORLD_SIZE={B}, GRAD_ACCUM_STEPS={J}, seq_len={A}")
 M=L//A;N=(K.numel()-1)//A;W=N*rank//B;O=N*(rank+1)//B;G=torch.zeros((),device=C,dtype=torch.float64);D=torch.zeros((),device=C,dtype=torch.float64);H=torch.zeros((),device=C,dtype=torch.float64);F.eval()
 with torch.inference_mode():
  for P in range(W,O,M):
   X=min(P+M,O);Y=P*A;Z=X*A+1;Q=K[Y:Z].to(device=C,dtype=torch.int64,non_blocking=True);R=Q[:-1].reshape(-1,A);I=Q[1:].reshape(-1,A)
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):a=F(R,I).detach()
   S=float(I.numel());G+=a.to(torch.float64)*S;D+=S;b=R.reshape(-1);T=I.reshape(-1);U=base_bytes_lut[T].to(dtype=torch.int16);U+=(has_leading_space_lut[T]&~is_boundary_token_lut[b]).to(dtype=torch.int16);H+=U.to(torch.float64).sum()
 if dist.is_available()and dist.is_initialized():dist.all_reduce(G,op=dist.ReduceOp.SUM);dist.all_reduce(D,op=dist.ReduceOp.SUM);dist.all_reduce(H,op=dist.ReduceOp.SUM)
 V=G/D;c=V.item()/math.log(2.);d=D.item()/H.item();F.train();return float(V.item()),float(c*d)
CONTROL_TENSOR_NAME_PATTERNS=tuple(A for A in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale').split(',')if A)
INT8_PER_ROW_SCALE_DTYPE=torch.float16
INT8_CLIP_PERCENTILE=99.99984
INT8_CLIP_Q=INT8_CLIP_PERCENTILE/1e2
def tensor_nbytes(t:Tensor)->int:return int(t.numel())*int(t.element_size())
def quantize_float_tensor(t:Tensor)->tuple[Tensor,Tensor]:
 A=t.float()
 if A.ndim==2:B=torch.quantile(A.abs(),INT8_CLIP_Q,dim=1)if A.numel()else torch.empty((A.shape[0],),dtype=torch.float32);E=torch.maximum(torch.minimum(A,B[:,None]),-B[:,None]);C=(B/127.).clamp_min(1./127.);D=torch.clamp(torch.round(E/C[:,None]),-127,127).to(torch.int8).contiguous();return D,C.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
 B=float(torch.quantile(A.abs().flatten(),INT8_CLIP_Q).item())if A.numel()else .0;C=torch.tensor(B/127. if B>0 else 1.,dtype=torch.float32);D=torch.clamp(torch.round(torch.clamp(A,-B,B)/C),-127,127).to(torch.int8).contiguous();return D,C
def load_data_shard(file:Path)->Tensor:
 A=file;D=256*np.dtype('<i4').itemsize;G=np.dtype('<u2').itemsize;B=np.fromfile(A,dtype='<i4',count=256)
 if B.size!=256 or int(B[0])!=20240520 or int(B[1])!=1:raise ValueError(f"Unexpected shard header for {A}")
 C=int(B[2]);E=D+C*G
 if A.stat().st_size!=E:raise ValueError(f"Shard size mismatch for {A}: expected {E} bytes")
 F=np.fromfile(A,dtype='<u2',count=C,offset=D)
 if F.size!=C:raise ValueError(f"Short read for {A}")
 return torch.from_numpy(F.astype(np.uint16,copy=False))
class TokenStream:
 def __init__(A,pattern:str):
  B=pattern;A.files=[Path(A)for A in sorted(glob.glob(B))]
  if not A.files:raise FileNotFoundError(f"No files found for pattern: {B}")
  A.file_idx=0;A.tokens=load_data_shard(A.files[0]);A.pos=0
 def _advance_file(A)->None:A.file_idx=(A.file_idx+1)%len(A.files);A.tokens=load_data_shard(A.files[A.file_idx]);A.pos=0
 def take(A,n:int)->Tensor:
  B:list[Tensor]=[];C=n
  while C>0:
   E=A.tokens.numel()-A.pos
   if E<=0:A._advance_file();continue
   D=min(C,E);B.append(A.tokens[A.pos:A.pos+D]);A.pos+=D;C-=D
  return B[0]if len(B)==1 else torch.cat(B)
class DistributedTokenLoader:
 def __init__(A,pattern:str,rank:int,world_size:int,device:torch.device):A.rank=rank;A.world_size=world_size;A.device=device;A.stream=TokenStream(pattern)
 def next_batch(A,global_tokens:int,seq_len:int,grad_accum_steps:int)->tuple[Tensor,Tensor]:C=seq_len;F=global_tokens//(A.world_size*grad_accum_steps);B=F+1;G=A.stream.take(B*A.world_size);D=A.rank*B;E=G[D:D+B].to(dtype=torch.int64);H=E[:-1].reshape(-1,C);I=E[1:].reshape(-1,C);return H.to(A.device,non_blocking=True),I.to(A.device,non_blocking=True)
class RMSNorm(nn.Module):
 def __init__(A,eps:float|None=None):super().__init__();A.eps=eps
 def forward(A,x:Tensor)->Tensor:return F.rms_norm(x,(x.size(-1),),eps=A.eps)
class CastedLinear(nn.Linear):
 _qat_enabled:bool=False
 def forward(A,x:Tensor)->Tensor:
  B=A.weight.to(x.dtype)
  if CastedLinear._qat_enabled and A.training and B.ndim==2:
   with torch.no_grad():C=A.weight.float();E=C.abs().amax(dim=1);D=(E/31.).clamp_min(1./31.);G=(torch.clamp(torch.round(C/D[:,None]),-32,31)*D[:,None]).to(x.dtype)
   B=B+(G-B).detach()
  H=A.bias.to(x.dtype)if A.bias is not None else None;return F.linear(x,B,H)
def restore_low_dim_params_to_fp32(module:nn.Module)->None:
 with torch.no_grad():
  for(B,A)in module.named_parameters():
   if(A.ndim<2 or any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS))and A.dtype!=torch.float32:A.data=A.data.float()
class Rotary(nn.Module):
 def __init__(A,dim:int,base:float=1e4,train_seq_len:int=1024,rope_dims:int=0):B=rope_dims;super().__init__();A.dim=dim;A.base=base;A.train_seq_len=train_seq_len;A.rope_dims=B if B>0 else dim;C=1./base**(torch.arange(0,A.rope_dims,2,dtype=torch.float32)/A.rope_dims);A.register_buffer('inv_freq',C,persistent=False);A._seq_len_cached=0;(A._cos_cached):Tensor|None=None;(A._sin_cached):Tensor|None=None
 def forward(A,seq_len:int,device:torch.device,dtype:torch.dtype)->tuple[Tensor,Tensor]:
  F=dtype;C=device;B=seq_len
  if A._cos_cached is None or A._sin_cached is None or A._seq_len_cached!=B or A._cos_cached.device!=C:
   D=A.rope_dims
   if B>A.train_seq_len:H=B/A.train_seq_len;I=A.base*H**(D/(D-2));E=1./I**(torch.arange(0,D,2,dtype=torch.float32,device=C)/D)
   else:E=A.inv_freq.to(C)
   J=torch.arange(B,device=C,dtype=E.dtype);G=torch.outer(J,E);A._cos_cached=G.cos()[None,:,None,:];A._sin_cached=G.sin()[None,:,None,:];A._seq_len_cached=B
  return A._cos_cached.to(dtype=F),A._sin_cached.to(dtype=F)
def apply_rotary_emb(x:Tensor,cos:Tensor,sin:Tensor,rope_dims:int=0)->Tensor:
 F=sin;E=cos;A=rope_dims
 if A>0 and A<x.size(-1):G,H=x[...,:A],x[...,A:];B=A//2;C,D=G[...,:B],G[...,B:];G=torch.cat((C*E+D*F,C*-F+D*E),dim=-1);return torch.cat((G,H),dim=-1)
 B=x.size(-1)//2;C,D=x[...,:B],x[...,B:];return torch.cat((C*E+D*F,C*-F+D*E),dim=-1)
class CausalSelfAttention(nn.Module):
 def __init__(A,dim:int,num_heads:int,num_kv_heads:int,rope_base:float,qk_gain_init:float):
  D=num_kv_heads;C=num_heads;B=dim;super().__init__()
  if B%C!=0:raise ValueError('model_dim must be divisible by num_heads')
  if C%D!=0:raise ValueError('num_heads must be divisible by num_kv_heads')
  A.num_heads=C;A.num_kv_heads=D;A.head_dim=B//C
  if A.head_dim%2!=0:raise ValueError('head_dim must be even for RoPE')
  E=A.num_kv_heads*A.head_dim;A.c_q=CastedLinear(B,B,bias=False);A.c_k=CastedLinear(B,E,bias=False);A.c_v=CastedLinear(B,E,bias=False);A.proj=CastedLinear(B,B,bias=False);A.proj._zero_init=True;A.q_gain=nn.Parameter(torch.full((C,),qk_gain_init,dtype=torch.float32));A.rope_dims=0;A.rotary=Rotary(A.head_dim,base=rope_base,train_seq_len=1024);A.use_xsa=False;A.vrl_gate=None
 def _xsa_efficient(K,y:Tensor,v:Tensor)->Tensor:A,B,C,D=y.shape;E=v.size(-2);I=C//E;G=y.reshape(A,B,E,I,D);H=F.normalize(v,dim=-1).unsqueeze(-2);J=(G*H).sum(dim=-1,keepdim=True)*H;return(G-J).reshape(A,B,C,D)
 def forward(A,x:Tensor,v_embed:Tensor|None=None,v_first:Tensor|None=None)->tuple[Tensor,Tensor]:
  L=v_first;K=v_embed;H,G,P=x.shape;B=A.c_q(x).reshape(H,G,A.num_heads,A.head_dim);E=A.c_k(x).reshape(H,G,A.num_kv_heads,A.head_dim);C=A.c_v(x);Q=C
  if K is not None:C=C+K
  if L is not None and A.vrl_gate is not None:M=torch.sigmoid(A.vrl_gate.to(dtype=C.dtype));C=(1-M)*C+M*L
  C=C.reshape(H,G,A.num_kv_heads,A.head_dim);B=F.rms_norm(B,(B.size(-1),));E=F.rms_norm(E,(E.size(-1),));N,O=A.rotary(G,x.device,B.dtype);B=apply_rotary_emb(B,N,O,A.rope_dims);E=apply_rotary_emb(E,N,O,A.rope_dims);B=B*A.q_gain.to(dtype=B.dtype)[None,None,:,None]
  if _HAS_FA3:D=flash_attn_3_func(B,E,C,causal=True)
  else:R=B.transpose(1,2);I=E.transpose(1,2);J=C.transpose(1,2);I=I.repeat_interleave(A.num_heads//A.num_kv_heads,dim=1);J=J.repeat_interleave(A.num_heads//A.num_kv_heads,dim=1);D=F.scaled_dot_product_attention(R,I,J,is_causal=True);D=D.transpose(1,2).contiguous()
  if A.use_xsa:D=A._xsa_efficient(D,C)
  D=D.reshape(H,G,P);return A.proj(D),Q
class SmearGate(nn.Module):
 def __init__(A,dim:int):super().__init__();A.gate=nn.Parameter(torch.zeros(dim,dtype=torch.float32))
 def forward(B,x:Tensor)->Tensor:A=torch.sigmoid(B.gate.to(dtype=x.dtype))[None,None,:];C=torch.cat([torch.zeros_like(x[:,:1]),x[:,:-1]],dim=1);return(1-A)*x+A*C
class BigramHashEmbedding(nn.Module):
 def __init__(A,bigram_vocab_size:int,bigram_dim:int,model_dim:int):
  D=model_dim;C=bigram_vocab_size;B=bigram_dim;super().__init__();A.bigram_vocab_size=C;A.embed=nn.Embedding(C,B);nn.init.zeros_(A.embed.weight);A.proj=CastedLinear(B,D,bias=False)if B!=D else None
  if A.proj is not None:nn.init.zeros_(A.proj.weight)
  A.scale=nn.Parameter(torch.tensor(.05,dtype=torch.float32))
 def bigram_hash(D,tokens:Tensor)->Tensor:A=tokens.to(torch.int32);C=D.bigram_vocab_size-1;B=torch.empty_like(A);B[...,0]=C;B[...,1:]=torch.bitwise_xor(36313*A[...,1:],27191*A[...,:-1])%C;return B.long()
 def forward(A,token_ids:Tensor)->Tensor:
  B=A.embed(A.bigram_hash(token_ids))
  if A.proj is not None:B=A.proj(B)
  return B*A.scale.to(dtype=B.dtype)
class ValueEmbedding(nn.Module):
 def __init__(A,vocab_size:int,ve_dim:int,model_dim:int):
  C=model_dim;B=ve_dim;super().__init__();A.embed=nn.Embedding(vocab_size,B);nn.init.normal_(A.embed.weight,std=.01);A.proj=CastedLinear(B,C,bias=False)if B!=C else None
  if A.proj is not None:nn.init.zeros_(A.proj.weight)
  A.scale=nn.Parameter(torch.tensor(.1,dtype=torch.float32))
 def forward(A,token_ids:Tensor)->Tensor:
  B=A.embed(token_ids)
  if A.proj is not None:B=A.proj(B)
  return B*A.scale.to(dtype=B.dtype)
class MLP(nn.Module):
 def __init__(A,dim:int,mlp_mult:int):B=dim;super().__init__();C=int(mlp_mult*B);A.fc=CastedLinear(B,C,bias=False);A.proj=CastedLinear(C,B,bias=False);A.proj._zero_init=True
 def forward(A,x:Tensor)->Tensor:x=F.leaky_relu(A.fc(x),negative_slope=.5);return A.proj(x.square())
class Block(nn.Module):
 def __init__(A,dim:int,num_heads:int,num_kv_heads:int,mlp_mult:int,rope_base:float,qk_gain_init:float,layer_idx:int=0,ln_scale:bool=False,dtg:bool=False):
  B=dim;super().__init__();A.attn_norm=RMSNorm();A.mlp_norm=RMSNorm();A.attn=CausalSelfAttention(B,num_heads,num_kv_heads,rope_base,qk_gain_init);A.mlp=MLP(B,mlp_mult);A.attn_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.mlp_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.resid_mix=nn.Parameter(torch.stack((torch.ones(B),torch.zeros(B))).float());A.ln_scale_factor=1./math.sqrt(layer_idx+1)if ln_scale else 1.
  if dtg:A.dtg_gate=nn.Linear(B,1,bias=True);nn.init.zeros_(A.dtg_gate.weight);nn.init.constant_(A.dtg_gate.bias,2.)
  else:A.dtg_gate=None
 def forward(A,x:Tensor,x0:Tensor,v_embed:Tensor|None=None,v_first:Tensor|None=None)->tuple[Tensor,Tensor]:
  D=A.resid_mix.to(dtype=x.dtype);C=D[0][None,None,:]*x+D[1][None,None,:]*x0;E,F=A.attn(A.attn_norm(C)*A.ln_scale_factor,v_embed=v_embed,v_first=v_first);B=C+A.attn_scale.to(dtype=C.dtype)[None,None,:]*E;B=B+A.mlp_scale.to(dtype=B.dtype)[None,None,:]*A.mlp(A.mlp_norm(B)*A.ln_scale_factor)
  if A.dtg_gate is not None:G=torch.sigmoid(A.dtg_gate(C.detach()));B=C+G*(B-C)
  return B,F
class GPT(nn.Module):
 def __init__(A,vocab_size:int,num_layers:int,model_dim:int,num_heads:int,num_kv_heads:int,mlp_mult:int,tie_embeddings:bool,tied_embed_init_std:float,logit_softcap:float,rope_base:float,qk_gain_init:float,mtp_num_heads:int=0,mtp_loss_weight:float=.1,bigram_vocab_size:int=0,bigram_dim:int=128,xsa_last_n:int=0,rope_dims:int=0,ln_scale:bool=False,dtg:bool=False,ve_enabled:bool=False,ve_dim:int=128,ve_layers:str='9,10',vrl_enabled:bool=False):
  O=vrl_enabled;N=xsa_last_n;M=bigram_vocab_size;L=mtp_num_heads;K=rope_base;J=tie_embeddings;I=num_kv_heads;G=rope_dims;F=logit_softcap;E=num_heads;D=vocab_size;C=num_layers;B=model_dim;super().__init__();A._ve_target_dim=I*(B//E)
  if F<=.0:raise ValueError(f"logit_softcap must be positive, got {F}")
  A.tie_embeddings=J;A.tied_embed_init_std=tied_embed_init_std;A.logit_softcap=F;A.mtp_num_heads=L;A.mtp_loss_weight=mtp_loss_weight;A.tok_emb=nn.Embedding(D,B);A.bigram=BigramHashEmbedding(M,bigram_dim,B)if M>0 else None;A.smear=SmearGate(B);A.num_encoder_layers=C//2;A.num_decoder_layers=C-A.num_encoder_layers;A.num_skip_weights=min(A.num_encoder_layers,A.num_decoder_layers);A.skip_weights=nn.Parameter(torch.ones(A.num_skip_weights,B,dtype=torch.float32));A.blocks=nn.ModuleList([Block(B,E,I,mlp_mult,K,qk_gain_init,layer_idx=A,ln_scale=ln_scale,dtg=dtg)for A in range(C)])
  if G>0:
   Q=B//E
   for P in A.blocks:P.attn.rope_dims=G;P.attn.rotary=Rotary(Q,base=K,train_seq_len=1024,rope_dims=G)
  A.ve_layer_indices=[int(A)for A in ve_layers.split(',')if A.strip()]if ve_enabled else[];R=A._ve_target_dim
  if A.ve_layer_indices:A.ve_shared=ValueEmbedding(D,ve_dim,R);A.ve_layer_scales=nn.ParameterList([nn.Parameter(torch.ones(1,dtype=torch.float32))for A in A.ve_layer_indices])
  else:A.ve_shared=None;A.ve_layer_scales=nn.ParameterList()
  A.value_embeds=nn.ModuleList();A.final_norm=RMSNorm();A.lm_head=None if J else CastedLinear(B,D,bias=False)
  if A.lm_head is not None:A.lm_head._zero_init=True
  A.mtp_heads=nn.ModuleList([CastedLinear(B,D,bias=False)for A in range(L)])
  for S in A.mtp_heads:S._zero_init=True
  if N>0:
   for H in range(max(0,C-N),C):A.blocks[H].attn.use_xsa=True
  A.vrl_enabled=O
  if O:
   for H in range(1,C):A.blocks[H].attn.vrl_gate=nn.Parameter(torch.tensor(-1.5,dtype=torch.float32))
  A._init_weights()
 def _init_weights(B)->None:
  if B.tie_embeddings:nn.init.normal_(B.tok_emb.weight,mean=.0,std=B.tied_embed_init_std)
  D=len(B.blocks)
  for(C,A)in B.named_modules():
   if isinstance(A,nn.Linear):
    if getattr(A,'_zero_init',False):nn.init.zeros_(A.weight)
    elif A.weight.ndim==2 and A.weight.shape[0]>=64 and A.weight.shape[1]>=64:
     nn.init.orthogonal_(A.weight,gain=1.)
     if'.proj.'in C or C.endswith('.proj'):
      with torch.no_grad():A.weight.mul_(1./math.sqrt(2*D))
 def _get_ve(A,layer_idx:int,input_ids:Tensor,ve_cache:dict|None=None)->Tensor|None:
  D=input_ids;C=layer_idx;B=ve_cache
  if A.ve_shared is None or C not in A.ve_layer_indices:return None
  if B is not None and've'not in B:B['ve']=A.ve_shared(D)
  E=B['ve']if B is not None else A.ve_shared(D);F=A.ve_layer_indices.index(C);return E*A.ve_layer_scales[F].to(dtype=E.dtype)
 def forward(A,input_ids:Tensor,target_ids:Tensor)->Tensor:
  L=target_ids;D=input_ids;B=A.tok_emb(D)
  if A.bigram is not None:B=B+A.bigram(D)
  B=F.rms_norm(B,(B.size(-1),));B=A.smear(B);M=B;E:list[Tensor]=[];N:dict={};G:Tensor|None=None
  for C in range(A.num_encoder_layers):
   H=A._get_ve(C,D,N);B,T=A.blocks[C](B,M,v_embed=H,v_first=G if A.vrl_enabled else None)
   if C==0 and A.vrl_enabled:G=T
   E.append(B)
  for C in range(A.num_decoder_layers):
   O=A.num_encoder_layers+C
   if E:B=B+A.skip_weights[C].to(dtype=B.dtype)[None,None,:]*E.pop()
   H=A._get_ve(O,D,N);B,U=A.blocks[O](B,M,v_embed=H,v_first=G if A.vrl_enabled else None)
  B=A.final_norm(B);P=B.reshape(-1,B.size(-1));V=L.reshape(-1)
  if A.tie_embeddings:Q=F.linear(P,A.tok_emb.weight)
  else:
   if A.lm_head is None:raise RuntimeError('lm_head is required when tie_embeddings=False')
   Q=A.lm_head(P)
  W=A.logit_softcap*torch.tanh(Q/A.logit_softcap);I=F.cross_entropy(W.float(),V,reduction='mean')
  if A.training and A.mtp_num_heads>0 and A.mtp_loss_weight>.0:
   U,X,Y=B.shape;J=B.new_zeros(());K=0
   for(R,Z)in enumerate(A.mtp_heads):
    S=X-(R+1)
    if S<=0:continue
    a=B[:,:S,:].reshape(-1,Y);b=L[:,R+1:].reshape(-1);c=Z(a);d=A.logit_softcap*torch.tanh(c/A.logit_softcap);J=J+F.cross_entropy(d.float(),b,reduction='mean');K+=1
   if K>0:I=I+A.mtp_loss_weight*(J/K)
  return I
 def forward_hidden(B,input_ids:Tensor)->Tensor:
  D=input_ids;A=B.tok_emb(D)
  if B.bigram is not None:A=A+B.bigram(D)
  A=F.rms_norm(A,(A.size(-1),));A=B.smear(A);I=A;E:list[Tensor]=[];J:dict={};G:Tensor|None=None
  for C in range(B.num_encoder_layers):
   H=B._get_ve(C,D,J);A,L=B.blocks[C](A,I,v_embed=H,v_first=G if B.vrl_enabled else None)
   if C==0 and B.vrl_enabled:G=L
   E.append(A)
  for C in range(B.num_decoder_layers):
   K=B.num_encoder_layers+C
   if E:A=A+B.skip_weights[C].to(dtype=A.dtype)[None,None,:]*E.pop()
   H=B._get_ve(K,D,J);A,M=B.blocks[K](A,I,v_embed=H,v_first=G if B.vrl_enabled else None)
  return B.final_norm(A)
 def compute_logits(A,hidden:Tensor)->Tensor:
  B=hidden
  if A.tie_embeddings:C=F.linear(B,A.tok_emb.weight)
  else:C=A.lm_head(B)
  return A.logit_softcap*torch.tanh(C/A.logit_softcap)
 def forward_logits(A,input_ids:Tensor)->Tensor:return A.compute_logits(A.forward_hidden(input_ids))
def eval_val_sliding(args:Hyperparameters,base_model:nn.Module,rank:int,world_size:int,device:torch.device,val_tokens:Tensor,base_bytes_lut:Tensor,has_leading_space_lut:Tensor,is_boundary_token_lut:Tensor,stride:int,batch_seqs:int=32,eval_seq_len:int|None=None)->tuple[float,float]:
 T=batch_seqs;S=stride;R=val_tokens;Q=world_size;I=base_model;C=device;D=eval_seq_len or args.train_seq_len;J=R.numel()-1;U=[A for A in range(0,J,S)if min(A+D,J)-A>=1];V=len(U);f=V*rank//Q;g=V*(rank+1)//Q;W=U[f:g];K=torch.zeros((),device=C,dtype=torch.float64);G=torch.zeros((),device=C,dtype=torch.float64);L=torch.zeros((),device=C,dtype=torch.float64);I.eval();h=torch.compile(I.forward_logits,dynamic=False,fullgraph=True)
 with torch.inference_mode():
  for X in range(0,len(W),T):
   M=W[X:X+T];N=len(M);O=torch.zeros(N,D,dtype=torch.int64,device=C);P=torch.zeros(N,D,dtype=torch.int64,device=C);Y:list[int]=[]
   for(B,E)in enumerate(M):Z=min(E+D,J);A=Z-E;Y.append(A);a=R[E:Z+1].to(dtype=torch.int64,device=C);O[B,:A]=a[:-1];P[B,:A]=a[1:]
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16):b=h(O)
   i=F.cross_entropy(b.reshape(-1,b.size(-1)).float(),P.reshape(-1),reduction='none').reshape(N,D)
   for(B,E)in enumerate(M):A=Y[B];H=0 if E==0 else max(A-S,0);j=i[B,H:A].to(torch.float64);K+=j.sum();G+=float(A-H);c=P[B,H:A];k=O[B,H:A];d=base_bytes_lut[c].to(torch.float64);d+=(has_leading_space_lut[c]&~is_boundary_token_lut[k]).to(torch.float64);L+=d.sum()
 if dist.is_available()and dist.is_initialized():dist.all_reduce(K,op=dist.ReduceOp.SUM);dist.all_reduce(G,op=dist.ReduceOp.SUM);dist.all_reduce(L,op=dist.ReduceOp.SUM)
 e=(K/G).item();l=e/math.log(2.);m=G.item()/L.item();I.train();return e,l*m
def eval_val_sliding_ttt(args:Hyperparameters,base_model:nn.Module,rank:int,world_size:int,device:torch.device,val_tokens:Tensor,base_bytes_lut:Tensor,has_leading_space_lut:Tensor,is_boundary_token_lut:Tensor,stride:int,batch_seqs:int=32,log0=print)->tuple[float,float]:
 'Legal score-first TTT (PR #461 recipe): score each chunk with sliding windows,\n then train on it. Every token scored BEFORE any update that could use it.';i=batch_seqs;U=log0;T=val_tokens;R=stride;Q=world_size;P=rank;I=device;G=base_model;A=args;B=A.train_seq_len;M=T.numel()-1;N=A.ttt_chunk_tokens;j=[A for A in range(0,M,R)if min(A+B,M)-A>=R or A==0];H=(M+N-1)//N;k:list[list[int]]=[[]for A in range(H)]
 for C in j:V=min(C+B,M);D=V-C;O=0 if C==0 else max(D-R,0);E=min((C+O)//N,H-1);k[E].append(C)
 U(f"ttt_sliding:start chunks={H} chunk_tokens={N} total_windows={len(j)} stride={R} ttt_lr={A.ttt_lr} ttt_epochs={A.ttt_epochs} freeze_blocks={A.ttt_freeze_blocks}");W=torch.zeros((),device=I,dtype=torch.float64);J=torch.zeros((),device=I,dtype=torch.float64);X=torch.zeros((),device=I,dtype=torch.float64);z=set(range(min(A.ttt_freeze_blocks,len(G.blocks))));S=[]
 for(A0,K)in G.named_parameters():
  l=any(f"blocks.{A}."in A0 for A in z);K.requires_grad_(not l)
  if not l:S.append(K)
 U(f"ttt_sliding:params unfrozen={sum(A.numel()for A in S)} frozen={sum(A.numel()for A in G.parameters()if not A.requires_grad)}");Z=torch.optim.SGD(S,lr=A.ttt_lr,momentum=A.ttt_momentum);m=time.perf_counter()
 for E in range(H):
  Y=k[E]
  if not Y:continue
  a=E*N;A1=min((E+1)*N,M);A2=len(Y)*P//Q;A3=len(Y)*(P+1)//Q;n=Y[A2:A3];G.eval()
  with torch.inference_mode():
   for o in range(0,len(n),i):
    b=n[o:o+i];c=len(b);d=torch.zeros(c,B,dtype=torch.int64,device=I);e=torch.zeros(c,B,dtype=torch.int64,device=I);p:list[int]=[]
    for(L,C)in enumerate(b):V=min(C+B,M);D=V-C;p.append(D);q=T[C:V+1].to(dtype=torch.int64,device=I);d[L,:D]=q[:-1];e[L,:D]=q[1:]
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):r=G.forward_logits(d)
    A4=F.cross_entropy(r.reshape(-1,r.size(-1)).float(),e.reshape(-1),reduction='none').reshape(c,B)
    for(L,C)in enumerate(b):D=p[L];O=0 if C==0 else max(D-R,0);W+=A4[L,O:D].to(torch.float64).sum();J+=float(D-O);s,A5=e[L,O:D],d[L,O:D];t=base_bytes_lut[s].to(torch.float64);t+=(has_leading_space_lut[s]&~is_boundary_token_lut[A5]).to(torch.float64);X+=t.sum()
  if E!=H-1 and A.ttt_epochs>0:
   G.train();f=(A1-a)//B
   if f>0:
    A6=A.ttt_lr*.5*(1.+math.cos(math.pi*E/max(H-1,1)))
    for A7 in Z.param_groups:A7['lr']=A6
    g=f*P//Q;A8=f*(P+1)//Q;u=A8-g
    for AG in range(A.ttt_epochs):
     for v in range(0,u,A.ttt_batch_seqs):
      A9=min(v+A.ttt_batch_seqs,u);AA=a+(g+v)*B;w=a+(g+A9)*B+1
      if w>T.numel():continue
      x=T[AA:w].to(device=I,dtype=torch.int64);AB=x[:-1].reshape(-1,B);AC=x[1:].reshape(-1,B);Z.zero_grad(set_to_none=True)
      with torch.autocast(device_type='cuda',dtype=torch.bfloat16):AD=G(AB,AC)
      AD.backward()
      if Q>1:
       for K in S:
        if K.grad is not None:dist.all_reduce(K.grad,op=dist.ReduceOp.AVG)
      torch.nn.utils.clip_grad_norm_(S,A.ttt_grad_clip);Z.step()
  if P==0 and(E%10==0 or E==H-1):AE=W.item()/max(J.item(),1);AF=AE/math.log(2.)*(J.item()/max(X.item(),1))if J.item()>0 else .0;U(f"  ttt_chunk [{E+1}/{H}] bpb={AF:.6f} time={time.perf_counter()-m:.1f}s")
 if dist.is_available()and dist.is_initialized():dist.all_reduce(W,op=dist.ReduceOp.SUM);dist.all_reduce(J,op=dist.ReduceOp.SUM);dist.all_reduce(X,op=dist.ReduceOp.SUM)
 h=(W/J).item();y=h/math.log(2.)*(J.item()/X.item())
 for K in G.parameters():K.requires_grad_(True)
 G.eval();U(f"ttt_sliding:done val_loss={h:.6f} val_bpb={y:.6f} elapsed={time.perf_counter()-m:.1f}s");return h,y
def run_ogd_sliding(args,base_model,rank,world_size,device,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut,stride,batch_seqs=32,use_seq_len=None):
 a=batch_seqs;Z=stride;Y=val_tokens;X=world_size;N=base_model;D=args;C=device;G=use_seq_len or D.train_seq_len;O=Y.numel()-1;P=[A for A in range(0,O,Z)if min(A+G,O)-A>=1];l=len(P)*rank//X;m=len(P)*(rank+1)//X;b=P[l:m];Q=torch.zeros((),device=C,dtype=torch.float64);K=torch.zeros((),device=C,dtype=torch.float64);R=torch.zeros((),device=C,dtype=torch.float64);S=torch.zeros(D.vocab_size,device=C,dtype=torch.float32);E=torch.zeros(D.vocab_size,device=C,dtype=torch.float32);N.eval();n=torch.compile(N.forward_logits,dynamic=False,fullgraph=True)
 with torch.inference_mode():
  for c in range(0,len(b),a):
   T=b[c:c+a];U=len(T);V=torch.zeros(U,G,dtype=torch.int64,device=C);H=torch.zeros(U,G,dtype=torch.int64,device=C);d=[]
   for(B,I)in enumerate(T):e=min(I+G,O);A=e-I;d.append(A);f=Y[I:e+1].to(dtype=torch.int64,device=C);V[B,:A]=f[:-1];H[B,:A]=f[1:]
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16):o=n(V)
   L=o.float()+S[None,None,:]
   if E.sum()>0:p=E/E.sum();q=torch.softmax(L,dim=-1);r=(1-D.cache_lambda)*q+D.cache_lambda*p[None,None,:];g=-torch.log(r.gather(-1,H.unsqueeze(-1)).squeeze(-1).clamp(min=1e-10))
   else:g=F.cross_entropy(L.reshape(-1,L.size(-1)),H.reshape(-1),reduction='none').reshape(U,G)
   for(B,I)in enumerate(T):
    A=d[B];J=0 if I==0 else max(A-Z,0);Q+=g[B,J:A].to(torch.float64).sum();K+=float(A-J);h=H[B,J:A];s=V[B,J:A];i=base_bytes_lut[h].to(torch.float64);i+=(has_leading_space_lut[h]&~is_boundary_token_lut[s]).to(torch.float64);R+=i.sum()
    for j in range(J,A):M=H[B,j].item();t=torch.softmax(L[B,j],dim=-1);W=t.clone();W[M]=W[M]-1.;S=S-D.ogd_lr*W;E=E*D.cache_decay;E[M]=E[M]+1.
 if dist.is_available()and dist.is_initialized():dist.all_reduce(Q,op=dist.ReduceOp.SUM);dist.all_reduce(K,op=dist.ReduceOp.SUM);dist.all_reduce(R,op=dist.ReduceOp.SUM)
 k=(Q/K).item();N.train();return k,k/math.log(2.)*(K.item()/R.item())
def _classify_param(name:str)->str:
 A=name
 if'tok_emb'in A or'lm_head'in A:return'embed'
 if'.mlp.'in A:return'mlp'
 if'.attn.'in A or'.proj.'in A and'.mlp.'not in A:return'attn'
 return'other'
def quantize_int6_per_row(t:Tensor,clip_range:int=31)->tuple[Tensor,Tensor]:
 A=clip_range;B=t.float()
 if B.ndim==2:
  E,F,G=None,None,float('inf')
  for H in[.999,.9995,.9999,.99999,1.]:
   if H<1.:I=torch.quantile(B.abs(),H,dim=1)
   else:I=B.abs().amax(dim=1)
   D=(I/A).clamp_min(1./A).to(torch.float16);C=torch.clamp(torch.round(B/D.float()[:,None]),-A,A).to(torch.int8);M=C.float()*D.float()[:,None];J=(B-M).pow(2).mean().item()
   if J<G:E,F,G=C,D,J
  return E,F
 K=B.abs().max().item();L=torch.tensor(K/A if K>0 else 1.,dtype=torch.float16);C=torch.clamp(torch.round(B/L.float()),-A,A).to(torch.int8);return C,L
def generate_autoregressive_calib(model,device,num_seqs=64,seq_len=2048,vocab_size=1024,temperature=.8,batch_size=8,seed=42):
 F=batch_size;E=num_seqs;D=device;C=model;C.eval();B=torch.Generator(device=D);B.manual_seed(seed);G=[]
 with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
  for J in range(0,E,F):
   H=min(F,E-J);A=torch.randint(0,vocab_size,(H,1),device=D,generator=B)
   for O in range(seq_len-1):K=C.forward_logits(A);L=K[:,-1,:];M=torch.softmax(L/temperature,dim=-1);N=torch.multinomial(M,1,generator=B);A=torch.cat([A,N],dim=1)
   for I in range(H):G.append(A[I:I+1])
 return G
def collect_hessians(model,token_seqs,device):
 H=device;G=token_seqs;D=model;A={};I=[]
 for(C,E)in D.named_modules():
  if isinstance(E,CastedLinear):
   J=C+'.weight';K=E.weight.shape[1];A[J]=torch.zeros(K,K,dtype=torch.float32,device='cpu')
   def M(pname):
    def B(module,input,output):
     B=input[0].detach().float()
     if B.ndim==3:B=B.reshape(-1,B.shape[-1])
     A[pname]+=(B.T@B).cpu()
    return B
   F=E.register_forward_hook(M(J));I.append(F)
 D.eval()
 with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
  for L in G:N=L[:,:-1].to(H);O=L[:,1:].to(H);D(N,O)
 for F in I:F.remove()
 for C in A:B=A[C];B/=len(G);P=.01*torch.diag(B).mean().clamp_min(1e-06);B+=P*torch.eye(B.shape[0]);A[C]=B
 return A
def quantize_int6_gptq(weight,hessian=None,clip_range=31,block_size=128):
 Q=block_size;P=hessian;G=clip_range;E=weight.float()
 if E.ndim!=2 or P is None:return quantize_int6_per_row(E,G)
 R,H=E.shape;B=P.float().clone();L=torch.diag(B)==0;B[L,L]=1;g=.01*torch.mean(torch.diag(B));B[torch.arange(H),torch.arange(H)]+=g;I=torch.argsort(torch.diag(B),descending=True);h=torch.argsort(I);J=E[:,I].clone();J[:,L[I]]=0;B=B[I][:,I];F=torch.linalg.cholesky(B);F=torch.cholesky_inverse(F);F=torch.linalg.cholesky(F,upper=True);K,S,T=None,None,float('inf')
 for U in[.999,.9995,.9999,.99999,1.]:
  if U<1.:V=torch.quantile(E.abs(),U,dim=1)
  else:V=E.abs().amax(dim=1)
  W=(V/G).clamp_min(1./G).to(torch.float16);M=W.float();N=torch.zeros_like(J,dtype=torch.int8);X=J.clone()
  for D in range(0,H,Q):
   A=min(D+Q,H);O=A-D;Y=X[:,D:A].clone();Z=torch.zeros(R,O,dtype=torch.int8);a=torch.zeros(R,O);b=F[D:A,D:A]
   for C in range(O):c=Y[:,C];i=b[C,C];d=torch.clamp(torch.round(c/M),-G,G).to(torch.int8);Z[:,C]=d;e=(c-d.float()*M)/i;Y[:,C:]-=e.unsqueeze(1)*b[C,C:].unsqueeze(0);a[:,C]=e
   N[:,D:A]=Z
   if A<H:X[:,A:]-=a@F[D:A,A:]
  j=N.float()*M[:,None];f=(J-j).pow(2).mean().item()
  if f<T:K,S,T=N,W,f
 K=K[:,h];return K,S
def mixed_quantize_int6(state_dict:dict[str,Tensor],int6_cats:set[str],hessians:dict[str,Tensor]|None=None):
 G=hessians;C:dict[str,Tensor]={};D:dict[str,object]={}
 for(A,H)in state_dict.items():
  B=H.detach().cpu().contiguous();I=_classify_param(A)
  if not B.is_floating_point()or B.numel()<=65536:C[A]=B.to(torch.float16)if B.is_floating_point()else B;D[A]='passthrough';continue
  if any(B in A for B in CONTROL_TENSOR_NAME_PATTERNS):C[A]=B.to(torch.float16);D[A]='passthrough_ctrl';continue
  if I in int6_cats and B.ndim>=1:J=G.get(A)if G is not None and B.ndim==2 else None;E,F=quantize_int6_gptq(B,hessian=J)if B.ndim==2 else quantize_int6_per_row(B);C[A+'.q']=E;C[A+'.scale']=F;D[A]={'type':'int6'}
  else:E,F=quantize_float_tensor(B);C[A+'.q']=E;C[A+'.scale']=F;D[A]={'type':'int8'}
 return C,D
def dequantize_mixed_int6(result:dict[str,Tensor],meta:dict[str,object],template_sd:dict[str,Tensor])->dict[str,Tensor]:
 F=result;B:dict[str,Tensor]={}
 for(A,I)in template_sd.items():
  H=meta.get(A)
  if H is None:continue
  C=I.dtype
  if H in('passthrough','passthrough_ctrl','passthrough_fp16'):
   D=F[A]
   if D.dtype==torch.float16 and C in(torch.float32,torch.bfloat16):D=D.to(C)
   B[A]=D;continue
  E,G=F[A+'.q'],F[A+'.scale']
  if G.ndim>0:B[A]=(E.float()*G.float().view(E.shape[0],*[1]*(E.ndim-1))).to(C)
  else:B[A]=(E.float()*float(G.item())).to(C)
 return B
def main()->None:
 global zeropower_via_newtonschulz5;g=Path(__file__).read_text(encoding='utf-8');A=Hyperparameters();zeropower_via_newtonschulz5=torch.compile(zeropower_via_newtonschulz5);I='RANK'in os.environ and'WORLD_SIZE'in os.environ;H=int(os.environ.get('RANK','0'));F=int(os.environ.get('WORLD_SIZE','1'));A3=int(os.environ.get('LOCAL_RANK','0'))
 if F<=0:raise ValueError(f"WORLD_SIZE must be positive, got {F}")
 if 8%F!=0:raise ValueError(f"WORLD_SIZE={F} must divide 8 so grad_accum_steps stays integral")
 G=8//F;A4=1./G
 if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
 D=torch.device('cuda',A3);torch.cuda.set_device(D)
 if I:dist.init_process_group(backend='nccl',device_id=D);dist.barrier()
 W=H==0;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;from torch.backends.cuda import enable_cudnn_sdp as AV,enable_flash_sdp as AW,enable_math_sdp as AX,enable_mem_efficient_sdp as AY;AV(False);AW(True);AY(False);AX(False);X=None
 if W:os.makedirs('logs',exist_ok=True);X=f"logs/{A.run_id}.txt";print(X)
 def B(msg:str,console:bool=True)->None:
  if not W:return
  if console:print(msg)
  if X is not None:
   with open(X,'a',encoding='utf-8')as A:print(msg,file=A)
 B(g,console=False);B('='*100,console=False);B(f"Running Python {sys.version}",console=False);B(f"Running PyTorch {torch.__version__}",console=False);B(subprocess.run(['nvidia-smi'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False).stdout,console=False);B('='*100,console=False);random.seed(A.seed);np.random.seed(A.seed);torch.manual_seed(A.seed);torch.cuda.manual_seed_all(A.seed)
 if not A.tokenizer_path.endswith('.model'):raise ValueError(f"Script only setup for SentencePiece .model file: {A.tokenizer_path}")
 h=spm.SentencePieceProcessor(model_file=A.tokenizer_path)
 if int(h.vocab_size())!=A.vocab_size:raise ValueError(f"VOCAB_SIZE={A.vocab_size} does not match tokenizer vocab_size={int(h.vocab_size())}")
 A5=Path(A.data_path).resolve();AZ=len(list(A5.glob('fineweb_train_*.bin')));i=A.eval_seq_len if A.eval_seq_len>0 else A.train_seq_len;Aa=max(A.train_seq_len,i);S=load_validation_tokens(A.val_files,Aa);Y,Z,a=build_sentencepiece_luts(h,A.vocab_size,D);B(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={A.tokenizer_path}");B(f"train_loader:dataset:{A5.name} train_shards:{AZ}");B(f"val_loader:shards pattern={A.val_files} tokens:{S.numel()-1}");CastedLinear._qat_enabled=A.qat_enabled;C=GPT(vocab_size=A.vocab_size,num_layers=A.num_layers,model_dim=A.model_dim,num_heads=A.num_heads,num_kv_heads=A.num_kv_heads,mlp_mult=A.mlp_mult,tie_embeddings=A.tie_embeddings,tied_embed_init_std=A.tied_embed_init_std,logit_softcap=A.logit_softcap,rope_base=A.rope_base,qk_gain_init=A.qk_gain_init,mtp_num_heads=A.mtp_num_heads,mtp_loss_weight=A.mtp_loss_weight,bigram_vocab_size=A.bigram_vocab_size,bigram_dim=A.bigram_dim,xsa_last_n=A.xsa_last_n,rope_dims=A.rope_dims,ln_scale=A.ln_scale,dtg=A.dtg_enabled,ve_enabled=A.ve_enabled,ve_dim=A.ve_dim,ve_layers=A.ve_layers,vrl_enabled=A.vrl_enabled).to(D).bfloat16()
 for A6 in C.modules():
  if isinstance(A6,CastedLinear):A6.float()
 restore_low_dim_params_to_fp32(C);j=torch.compile(C,dynamic=False,fullgraph=True);K:nn.Module=DDP(j,device_ids=[A3],broadcast_buffers=False)if I else j;A7=list(C.blocks.named_parameters());b=[A for(B,A)in A7 if A.ndim==2 and not any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS)]
 if C.mtp_num_heads>0:b.extend([A for A in C.mtp_heads.parameters()if A.ndim==2])
 O=[A for(B,A)in A7 if A.ndim<2 or any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS)]
 if C.skip_weights.numel()>0:O.append(C.skip_weights)
 O.append(C.smear.gate)
 if C.bigram is not None:O.append(C.bigram.scale)
 L=A.tied_embed_lr if A.tie_embeddings else A.embed_lr;k=[{'params':[C.tok_emb.weight],'lr':L,'base_lr':L}]
 if C.bigram is not None:
  k.append({'params':[C.bigram.embed.weight],'lr':L,'base_lr':L})
  if C.bigram.proj is not None:b.append(C.bigram.proj.weight)
 if C.ve_shared is not None:
  k.append({'params':[C.ve_shared.embed.weight],'lr':L,'base_lr':L})
  if C.ve_shared.proj is not None:b.append(C.ve_shared.proj.weight)
  O.append(C.ve_shared.scale)
  for Ab in C.ve_layer_scales:O.append(Ab)
 Ac=torch.optim.AdamW(k,betas=(A.beta1,A.beta2),eps=A.adam_eps,weight_decay=A.adam_wd,fused=True);l=Muon(b,lr=A.matrix_lr,momentum=A.muon_momentum,backend_steps=A.muon_backend_steps,weight_decay=A.muon_wd)
 for P in l.param_groups:P['base_lr']=A.matrix_lr
 Ad=torch.optim.AdamW([{'params':O,'lr':A.scalar_lr,'base_lr':A.scalar_lr}],betas=(A.beta1,A.beta2),eps=A.adam_eps,weight_decay=A.adam_wd,fused=True);M:list[torch.optim.Optimizer]=[Ac,l,Ad]
 if C.lm_head is not None:Ae=torch.optim.Adam([{'params':[C.lm_head.weight],'lr':A.head_lr,'base_lr':A.head_lr}],betas=(A.beta1,A.beta2),eps=A.adam_eps,fused=True);M.insert(1,Ae)
 Af=sum(A.numel()for A in C.parameters());Ag=sum(A.numel()for A in C.mtp_heads.parameters());B(f"model_params:{Af}");B(f"mtp_num_heads:{A.mtp_num_heads} mtp_loss_weight:{A.mtp_loss_weight} mtp_params:{Ag}");Ah=[A for(A,B)in enumerate(C.blocks)if B.attn.use_xsa];B(f"XSA:last_{A.xsa_last_n} active_layers:{Ah}");B(f"world_size:{F} grad_accum_steps:{G}");B('sdp_backends:cudnn=False flash=True mem_efficient=False math=False');B(f"attention_mode:gqa num_heads:{A.num_heads} num_kv_heads:{A.num_kv_heads}");B(f"tie_embeddings:{A.tie_embeddings} embed_lr:{L} head_lr:{A.head_lr if C.lm_head is not None else .0} matrix_lr:{A.matrix_lr} scalar_lr:{A.scalar_lr}");B(f"train_batch_tokens:{A.train_batch_tokens} train_seq_len:{A.train_seq_len} iterations:{A.iterations} warmup_steps:{A.warmup_steps} max_wallclock_seconds:{A.max_wallclock_seconds:.3f}");B(f"seed:{A.seed}");m=DistributedTokenLoader(A.train_files,H,F,D)
 def T()->None:
  for A in M:A.zero_grad(set_to_none=True)
 U=1e3*A.max_wallclock_seconds if A.max_wallclock_seconds>0 else None
 def Ai(step:int,elapsed_ms:float)->float:
  C=elapsed_ms;B=step
  if A.warmdown_iters<=0:return 1.
  if U is None:F=max(A.iterations-A.warmdown_iters,0);return max((A.iterations-B)/max(A.warmdown_iters,1),.0)if F<=B<A.iterations else 1.
  G=C/max(B,1);D=A.warmdown_iters*G;E=max(U-C,.0);return E/max(D,1e-09)if E<=D else 1.
 if A.warmup_steps>0:
  Aj={A:B.detach().cpu().clone()for(A,B)in C.state_dict().items()};Ak=[copy.deepcopy(A.state_dict())for A in M];K.train()
  for n in range(A.warmup_steps):
   T()
   for o in range(G):
    if I:K.require_backward_grad_sync=o==G-1
    p,q=m.next_batch(A.train_batch_tokens,A.train_seq_len,G)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):Al=K(p,q)
    (Al*A4).backward()
   for N in M:N.step()
   T()
   if A.warmup_steps<=20 or(n+1)%10==0 or n+1==A.warmup_steps:B(f"warmup_step:{n+1}/{A.warmup_steps}")
  C.load_state_dict(Aj,strict=True)
  for(N,Am)in zip(M,Ak,strict=True):N.load_state_dict(Am)
  T()
  if I:K.require_backward_grad_sync=True
  m=DistributedTokenLoader(A.train_files,H,F,D)
 r:dict[str,Tensor]|None=None;A8=0;A9={A:B.detach().float().clone()for(A,B)in C.state_dict().items()};AA=.997;Q=.0;R:int|None=None;torch.cuda.synchronize();c=time.perf_counter();E=0
 while True:
  AB=E==A.iterations or R is not None and E>=R;An=AB or A.val_loss_every>0 and E%A.val_loss_every==0
  if An:torch.cuda.synchronize();Q+=1e3*(time.perf_counter()-c);Ao,Ap=eval_val(A,K,H,F,D,G,S,Y,Z,a);B(f"step:{E}/{A.iterations} val_loss:{Ao:.4f} val_bpb:{Ap:.4f} train_time:{Q:.0f}ms step_avg:{Q/max(E,1):.2f}ms");torch.cuda.synchronize();c=time.perf_counter()
  if AB:
   if R is not None and E<A.iterations:B(f"stopping_early: wallclock_cap train_time:{Q:.0f}ms step:{E}/{A.iterations}")
   break
  Aq=Q+1e3*(time.perf_counter()-c);d=Ai(E,Aq)
  if A.late_qat_threshold>0 and d<A.late_qat_threshold and not CastedLinear._qat_enabled:CastedLinear._qat_enabled=True;B(f"late_qat:enabled step:{E} scale:{d:.4f}")
  T();s=torch.zeros((),device=D)
  for o in range(G):
   if I:K.require_backward_grad_sync=o==G-1
   p,q=m.next_batch(A.train_batch_tokens,A.train_seq_len,G)
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):AC=K(p,q)
   s+=AC.detach();(AC*A4).backward()
  s/=G;AD=min(E/A.muon_momentum_warmup_steps,1.)if A.muon_momentum_warmup_steps>0 else 1.;Ar=(1-AD)*A.muon_momentum_warmup_start+AD*A.muon_momentum
  for P in l.param_groups:P['momentum']=Ar
  for N in M:
   for P in N.param_groups:P['lr']=P['base_lr']*d
  if A.grad_clip_norm>0:torch.nn.utils.clip_grad_norm_(C.parameters(),A.grad_clip_norm)
  for N in M:N.step()
  T()
  with torch.no_grad():
   for(t,u)in C.state_dict().items():A9[t].mul_(AA).add_(u.detach().float(),alpha=1.-AA)
  E+=1;v=Q+1e3*(time.perf_counter()-c)
  if A.swa_enabled and d<.2 and E%A.swa_every==0:
   if r is None:r={A:B.detach().cpu().clone()for(A,B)in C.state_dict().items()};A8=1;B(f"swa:start step:{E}")
   else:
    for(t,u)in C.state_dict().items():r[t]+=u.detach().cpu()
    A8+=1
  As=A.train_log_every>0 and(E<=10 or E%A.train_log_every==0 or R is not None)
  if As:B(f"step:{E}/{A.iterations} train_loss:{s.item():.4f} train_time:{v:.0f}ms step_avg:{v/E:.2f}ms")
  w=U is not None and v>=U
  if I and U is not None:AE=torch.tensor(int(w),device=D);dist.all_reduce(AE,op=dist.ReduceOp.MAX);w=bool(AE.item())
  if R is None and w:R=E
 B(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB");B('ema:applying EMA weights');At=C.state_dict();Au={A:B.to(dtype=At[A].dtype)for(A,B)in A9.items()};C.load_state_dict(Au,strict=True);torch.cuda.synchronize();Av=time.perf_counter();Aw,Ax=eval_val(A,j,H,F,D,G,S,Y,Z,a);torch.cuda.synchronize();B(f"DIAGNOSTIC post_ema val_loss:{Aw:.4f} val_bpb:{Ax:.4f} eval_time:{1e3*(time.perf_counter()-Av):.0f}ms");B('gptq:generating calibration sequences');torch.cuda.synchronize();Ay=time.perf_counter();AF=generate_autoregressive_calib(C,D,num_seqs=64,seq_len=A.train_seq_len,vocab_size=A.vocab_size);B(f"gptq:generated {len(AF)} sequences in {1000*(time.perf_counter()-Ay):.0f}ms");B('gptq:collecting hessians');AG=collect_hessians(C,AF,D);B(f"gptq:collected {len(AG)} hessians");AH=C.state_dict();AI={A:B for(A,B)in AH.items()if'mtp_heads'not in A};AJ=sum(int(B.numel())for(A,B)in AH.items()if'mtp_heads'in A)
 if AJ>0:B(f"export_excluding_mtp_params:{AJ}")
 if W:torch.save(AI,'final_model.pt');Az=os.path.getsize('final_model.pt');e=len(g.encode('utf-8'));B(f"Serialized model: {Az} bytes");B(f"Code size: {e} bytes")
 AK={A:B.detach().cpu()for(A,B)in AI.items()};A_,B0=mixed_quantize_int6(AK,{'mlp','attn'},hessians=AG);AL=io.BytesIO();torch.save({'w':A_,'m':B0},AL);B1=AL.getvalue();AM=lzma.compress(B1,preset=6)
 if W:
  with open('final_model.int6.ptz','wb')as x:x.write(AM)
  y=len(AM);e=len(g.encode('utf-8'));B(f"Serialized model int6+{_COMPRESSOR}: {y} bytes");B(f"Total submission size int6+{_COMPRESSOR}: {y+e} bytes");B(f"Total submission size: {y+e} bytes")
 if I:dist.barrier()
 with open('final_model.int6.ptz','rb')as x:B2=x.read()
 AN=torch.load(io.BytesIO(lzma.decompress(B2)),map_location='cpu');B3=dequantize_mixed_int6(AN['w'],AN['m'],AK);J=GPT(vocab_size=A.vocab_size,num_layers=A.num_layers,model_dim=A.model_dim,num_heads=A.num_heads,num_kv_heads=A.num_kv_heads,mlp_mult=A.mlp_mult,tie_embeddings=A.tie_embeddings,tied_embed_init_std=A.tied_embed_init_std,logit_softcap=A.logit_softcap,rope_base=A.rope_base,qk_gain_init=A.qk_gain_init,mtp_num_heads=0,mtp_loss_weight=.0,bigram_vocab_size=A.bigram_vocab_size,bigram_dim=A.bigram_dim,xsa_last_n=A.xsa_last_n,rope_dims=A.rope_dims,ln_scale=A.ln_scale,dtg=A.dtg_enabled,ve_enabled=A.ve_enabled,ve_dim=A.ve_dim,ve_layers=A.ve_layers,vrl_enabled=A.vrl_enabled).to(D).bfloat16()
 for AO in J.modules():
  if isinstance(AO,CastedLinear):AO.float()
 restore_low_dim_params_to_fp32(J);J.load_state_dict(B3,strict=True);B4=torch.compile(J,dynamic=False,fullgraph=True);f=S,Y,Z,a;torch.cuda.synchronize();B5=time.perf_counter();AP,AQ=eval_val(A,B4,H,F,D,G,*f,eval_seq_len=i);torch.cuda.synchronize();B(f"final_int6_roundtrip val_loss:{AP:.4f} val_bpb:{AQ:.4f} eval_time:{1e3*(time.perf_counter()-B5):.0f}ms");B(f"final_int6_roundtrip_exact val_loss:{AP:.8f} val_bpb:{AQ:.8f}");V=i
 if A.eval_stride>0 and A.eval_stride<V:torch.cuda.synchronize();B6=time.perf_counter();z,A0=eval_val_sliding(A,J,H,F,D,*f,stride=A.eval_stride,eval_seq_len=V);torch.cuda.synchronize();B(f"final_int6_sliding_window val_loss:{z:.4f} val_bpb:{A0:.4f} stride:{A.eval_stride} eval_time:{1e3*(time.perf_counter()-B6):.0f}ms");B(f"final_int6_sliding_window_exact val_loss:{z:.8f} val_bpb:{A0:.8f}");B(f"final_int6_roundtrip_exact val_loss:{z:.8f} val_bpb:{A0:.8f}")
 if A.eval_stride!=64 and 64<V:torch.cuda.synchronize();B7=time.perf_counter();A1,A2=eval_val_sliding(A,J,H,F,D,*f,stride=64,eval_seq_len=V);torch.cuda.synchronize();B(f"final_int6_sliding_window_s64 val_loss:{A1:.4f} val_bpb:{A2:.4f} stride:64 eval_time:{1e3*(time.perf_counter()-B7):.0f}ms");B(f"final_int6_sliding_window_s64_exact val_loss:{A1:.8f} val_bpb:{A2:.8f}");B(f"final_int6_roundtrip_exact val_loss:{A1:.8f} val_bpb:{A2:.8f}")
 if A.ogd_enabled:torch._dynamo.reset();torch.cuda.synchronize();B8=time.perf_counter();AR,AS=run_ogd_sliding(A,J,H,F,D,*f,stride=A.eval_stride,use_seq_len=V);torch.cuda.synchronize();B(f"final_ogd val_loss:{AR:.4f} val_bpb:{AS:.4f} eval_time:{1e3*(time.perf_counter()-B8):.0f}ms");B(f"final_ogd_exact val_loss:{AR:.8f} val_bpb:{AS:.8f}")
 if A.ttt_enabled:torch._dynamo.reset();torch.cuda.synchronize();B9=time.perf_counter();AT,AU=eval_val_sliding_ttt(A,J,H,F,D,S,Y,Z,a,stride=64,log0=B);torch.cuda.synchronize();B(f"final_ttt val_loss:{AT:.4f} val_bpb:{AU:.4f} eval_time:{1e3*(time.perf_counter()-B9):.0f}ms");B(f"final_ttt_exact val_loss:{AT:.8f} val_bpb:{AU:.8f}")
 if I:dist.destroy_process_group()
if __name__=='__main__':main()