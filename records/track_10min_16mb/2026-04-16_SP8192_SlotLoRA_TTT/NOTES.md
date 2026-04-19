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

## Run 1 results (2026-04-18) and post-mortem

First paid sweep on AP-IN-1 pod (community image, 96-116 ms/step). Seed 42 single run.

| Config | Params | compile | val_bpb | eval_time | vs no-TTT |
|---|---|---|---|---|---|
| no-TTT quantized+sliding | 0 | -- | 1.08151 | 128s | -- |
| A: TTT full all 11 layers | 31.7M | 1 | 1.08020 | 381s | -0.00131 |
| B: control 3-5 + global | 14,360 | 1 | 1.08137 | 296s | -0.00014 |
| C: qv 3-5 (global leaked) | 1.19M | 1 | 1.08129 | 294s | -0.00022 |
| D: slot-LoRA q/v r16 (global leaked) | 15.6M | 0 | 1.59058 | 441s | BROKEN |
| E: qv + LoRA (multi-leak) | 16.8M | 0 | 1.37122 | 442s | BROKEN |

Baseline A matched the PR #1572 3-seed mean (1.07974) within seed variance, so the scaffolding did not damage the A path.

### Bug 1 (post-run): sweep env-var leakage -- FIXED 2026-04-19

`setattr(Hyperparameters, attr, val)` mutated the class across sweep iterations, and the apply loop never reset to baseline between configs. Observed leakage: B's `ttt_include_global=1` bled into C, D, E. D's `ttt_lora_enabled=1 rank=16 proj=q,v` bled into E.

Fix applied in `train_gpt_sota.py` sweep block: snapshot the `Hyperparameters` class attr table before the loop, restore the snapshot at the top of every iteration before overrides are parsed.

### Bug 2 (post-run): slot-LoRA D/E catastrophic regression -- DIAGNOSTIC ADDED 2026-04-19

The handoff hypothesized A=Gaussian B=Gaussian init violating the LoRA convention. Code inspection says otherwise: `SlotLoRABank.reset_state` already initializes A with Kaiming*init_scale and explicitly calls `B.zero_()`. The CPU smoke test in `test_scaffolding.py` asserts `max_abs_diff(baseline, bank_attached_B0) == 0.00e+00` for a tiny model, so the math is correct in principle.

The real regression is therefore GPU/pod-specific. Top suspect: the score forward uses `torch.compile` when `bank is None` and eager when `bank is not None`. Numerical divergence between the compiled and eager paths (bf16 accumulator differences, flash_attn_3 kernel variants, or op-fusion boundaries) could push BPB from baseline to 1.59 even when LoRA delta is exactly zero at init.

Diagnostic added in `eval_val_ttt`: immediately after `_select_ttt_params` (before any TTT update), if a bank was attached, run two forward passes on the same tiny input -- one with `ttt_lora_bsz=0` (bypasses bank via `_slot_lora` early-exit) and one with `ttt_lora_bsz=2` and B=0 (exercises the einsum path producing delta=0). Logs `max_abs_diff` and `ref_max`, plus `bank_A_norm` and `bank_B_norm` for sanity. Warning fires if diff > 1% of the reference logit scale.

Expected next-run outcomes:
1. `init_check` diff is tiny (< 1e-4 relative): the LoRA bank forward path is fine, so the regression comes from TTT training dynamics (gradient clipping, SGD step, per-row bsz mismatch between train and score). Investigate the TTT update path.
2. `init_check` diff is large: the bank-attached eager forward disagrees with the bypass path. This would localize the bug to either the `apply_lora_delta` einsum under bf16 autocast, or the control-flow difference between `lora_q=None` (skip) and `lora_q=(A,B)` with B=0. Next fix: force zero-check in `apply_lora_delta` or compile the bank-attached path with `dynamic=True`.

### Next run

With both fixes in place, rerun the same sweep spec. Expected budget: ~$10 on AP-IN-1, eval-only, same one-seed single pod. Read `ttt:init_check` lines in the log before looking at final BPB -- they answer bug 2 directly.
