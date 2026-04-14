# Record: Reproduction of PR #1530 with random-token compile warmup

**Status: pending run (logs and submission.json populated after 3-seed eval)**

This record reproduces samacqua's [PR #1530](https://github.com/openai/parameter-golf/pull/1530) (varlen attention + fused MLP + doc-independent batched LoRA TTT) on a clean branch from main, with a single targeted modification: the TTT compile warmup no longer reads validation tokens.

## Why reproduce, not port

Three prior attempts to port pieces of PR #1530 onto our SP8192 + depth-recurrence stack (varlen attention, fused MLP, batched LoRA TTT) all regressed BPB. Root cause: PR #1530's components assume a banked / explicit-weight host architecture; our module-loop recurrence host triggers torch.compile shape-variant recompilations that defeat the kernels' speed advantage and confound the eval-time TTT path.

Decision: stop porting and reproduce the whole stack, then iterate from a known-working baseline.

## The one modification

PR #1530's eval-side TTT compile warmup feeds real validation tokens into the LoRA training step purely to populate the torch.compile cache (model state is never updated; the warmup operates on a throwaway `BatchedTTTLoRA`). However, the model parameters do see validation token activations during warmup, which the PR review thread flagged as a competition-rules concern.

The fix exploits the fact that torch.compile recompilation is triggered by tensor shapes/dtypes/devices, not contents. Replacing `val_data.val_tokens[...]` slicing with `torch.randint(0, vocab_size, ...)` of identical shape yields the same compile cache while eliminating any validation token exposure.

Diff (eval-side TTT warmup, around line ~2735 of `train_gpt.py`):

```python
# Before:
val_tokens_idx = val_data.val_tokens.to(torch.int32)
...
col_w = torch.arange(ctx_len + 1)
idx_w = (ds0 + col_w).clamp_(max=val_data.val_tokens.numel() - 1)
row_w = val_tokens_idx[idx_w].to(device=device, dtype=torch.int64)
xw = row_w[:ctx_len].unsqueeze(0).expand(bsz, -1).contiguous()
yw = row_w[1 : ctx_len + 1].unsqueeze(0).expand(bsz, -1).contiguous()

# After:
warmup_gen = torch.Generator(device=device).manual_seed(0)
...
xw = torch.randint(0, h.vocab_size, (bsz, ctx_len),
                   generator=warmup_gen, device=device, dtype=torch.int64)
yw = torch.randint(0, h.vocab_size, (bsz, ctx_len),
                   generator=warmup_gen, device=device, dtype=torch.int64)
```

The train-side compile warmup (lines 2462-2515) was already safe -- it uses `train_loader.next_batch()` and restores the original model and optimizer state via `load_state_dict` after warmup completes.

## Reproduction protocol

1. Pod scout (`bash .private/pod_scout.sh`) -- reject any pod with median fwd+bwd > 120ms.
2. Three seeds (0, 1, 2) on a vetted 8xH100 SXM pod.
3. Report mean and std of BPB; verify within +-0.001 of PR #1530's reported 1.07336.

## Expected outcome

If reproduction lands in the 1.073 band: the banked architecture becomes our new submission baseline, displacing the depth-recurrence stack. Subsequent deltas (fused-softcap-ce, artifact-size tuning) layer on top of this, not on top of the prior stack.

If reproduction misses the band: investigate before any deltas. Likely culprits in priority order would be torch/triton/FA3 version drift, data shard mismatch, or a misread of the warmup semantics.
