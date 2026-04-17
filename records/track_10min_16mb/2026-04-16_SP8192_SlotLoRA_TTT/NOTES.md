# SP8192 Slot-LoRA TTT scaffolding

Adds TTT-eval flexibility on top of the PR #1572 (1.07974 BPB) stack. Goal is to pull adaptation gain from the recurrent band where layers 3-5 are reused 3x per forward.

## What changed vs PR #1572

Two orthogonal additions, both opt-in via env vars (defaults preserve existing behavior):

### 1. Flexible TTT parameter selection

Replaces the single `TTT_SELECTIVE_LAYERS=N` knob (which only ever picked the LAST N physical layers) with:

- `TTT_LAYER_IDS` -- explicit CSV like "3,4,5". Empty falls back to legacy.
- `TTT_MODE` -- one of `control|qv|qkv|full|none`.
  - `control` selects only `attn_scale`, `mlp_scale`, `resid_mix`, `q_gain` per block (pulled from the existing `CONTROL_TENSOR_NAME_PATTERNS` set used elsewhere by the optimizer).
  - `qv` / `qkv` / `full` select the matrix weights as expected.
- `TTT_INCLUDE_GLOBAL` -- adds `skip_weights` and `skip_gates` (per-skip-position scalars, not per-block).

### 2. Per-slot LoRA bank

`SlotLoRABank` allocates per-doc, per-slot, per-projection low-rank adapters. The slot id is the position in the unrolled encoder/decoder schedule, NOT the physical block index, so reused blocks get distinct adapters at distinct call sites.

Shapes per projection:
- A: `[max_bsz, num_slots, rank, dim]`
- B: `[max_bsz, num_slots, dim_out, rank]` (kv_dim for v/k)

Forward injects `delta = einsum('btd,brd->btr', x, A) @ einsum('btr,bdr->btd', xA, B)` into `c_q`/`c_k`/`c_v`/`c_proj` (subset configurable). Frozen weight stays shared across reuse sites; only the bank carries per-call-site adaptation.

Knobs:
- `TTT_LORA_ENABLED=1` to attach.
- `TTT_LORA_RANK` (default 16).
- `TTT_LORA_PROJ` -- subset of `q,k,v,o` (default `q,v`).
- `TTT_LORA_SLOT_IDS` -- explicit slot list. Empty derives from `TTT_LAYER_IDS` (all slots whose block is in that set).
- `TTT_LORA_MAX_BSZ` (default 32) and `TTT_LORA_INIT_SCALE` (default 0.02).

Bank is allocated AFTER deserialize, lives only during the eval call, and is not part of the saved artifact.

## Slot accounting (NUM_LOOPS=2, NUM_LAYERS=11)

Schedule has 17 slots:

```
[0,1,2, 3,4,5, 3,4,5, 3,4,5, 6,7,8,9,10]
```

Slots whose block is in {3,4,5}: `[3,4,5, 6,7,8, 9,10,11]` = 9 active slots when targeting the recurrent band.

## CUDA event phase timing

`eval_val_ttt` now logs per-phase totals when `TTT_PHASE_LOG=1` (default on). Phases: `score`, `ttt_fwd`, `backward`, `all_reduce`, `optimizer`. Lets us see whether eval cost is dominated by scoring forward or by TTT update, which decides whether further kernel work or further TTT tuning is the right next axis.

## Compile interaction

When the LoRA bank is attached, the score forward falls back to eager (the dynamic per-batch slicing of the bank breaks `dynamic=False, fullgraph=True`). When the bank is NOT attached, the existing compiled path is unchanged. Future work: pad-to-fixed-bsz with masking to reclaim compile.

## First experiment plan

Eval-only on the PR #1572 checkpoint, single seed 42, fast pod. Four configs back-to-back on the same pod:

- A: baseline chunk TTT (`TTT_MODE=full TTT_LAYER_IDS=` with empty for legacy fallback)
- B: control-only on layers 3-5 (`TTT_LAYER_IDS=3,4,5 TTT_MODE=control TTT_INCLUDE_GLOBAL=1`)
- C: qv on layers 3-5 (`TTT_LAYER_IDS=3,4,5 TTT_MODE=qv`)
- D: slot-LoRA q/v rank 16 on layers 3-5 (`TTT_LAYER_IDS=3,4,5 TTT_MODE=none TTT_LORA_ENABLED=1 TTT_LORA_RANK=16 TTT_LORA_PROJ=q,v`)

Compare BPB to baseline 1.08041; capture phase timing breakdown.

## Status

Scaffolding only -- not yet run on a pod. CPU smoke tests in `test_scaffolding.py` cover shape correctness, slot guard, gradient flow into active vs inactive slots, and all five TTT_MODE selections.
