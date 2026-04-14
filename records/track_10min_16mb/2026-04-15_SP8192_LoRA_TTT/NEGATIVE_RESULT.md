# NEGATIVE RESULT: Batched LoRA TTT port from PR #1530

**Status: DO NOT SHIP. Port regressed BPB by 0.17 vs sliding-window eval on seed 42.**

Pod session: 2026-04-14, 8xH100 SXM in AP-IN-1 (rptiaef7wx17bu).

## Seed 42 result

| Eval mode | val_bpb | eval_time |
|-----------|---------|-----------|
| pre-quantization post-ema | 1.08727 | 7s |
| quantized (raw) | 1.09989 | 9s |
| quantized_sliding_window | 1.08292 | 89s |
| **quantized_ttt (LoRA)** | **1.25076** | **910s** (non-compliant, exceeds 600s cap) |

Baseline-equivalent on same pod: 1.08041 (quantized_ttt, chunk-based).

**Delta vs this-pod baseline: +0.17034 BPB (massive regression).**

## Root cause: two LoRA application semantics bugs

My port applied LoRAs as inner projection tweaks. Samacqua's PR #1530 applies them as parallel residual-level paths. These are not equivalent.

### Bug 1: mlp_lora dimensions wrong

**Samacqua**:
```python
self.mlp_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)])
# ...
mlp_out = block.mlp(mlp_n, up_w, down_w)
if lora.mlp_loras is not None:
    mlp_out = mlp_out + lora.mlp_loras[slot](mlp_n)  # residual-level bypass
```

**My port**:
```python
self.mlp_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, hidden_dim, rank) ...])
# In MLP.forward:
h = self.fc(x)
if lora_up is not None:
    h = h + lora_up(x)  # tweaks up-projection hidden state, wrong dimension
```

### Bug 2: o_lora input wrong

**Samacqua** applies o_lora to the pre-attention normalized residual `n`, not the attention output `y`. Mine applies to `y`. Different tensors, different semantics, different effective LoRA behavior.

## Eval time also blown

910s for full LoRA TTT eval on our pod. Samacqua reports 217s. My eval is 4x slower. Without `torch.compile(forward_ttt)`, every forward is fully eager. Samacqua's fullgraph compile worked because his block.forward is structurally simpler (no depth recurrence). Ours with looping active blew through compile cache and hung, forcing us to disable compile.

## Why it regressed so hard

LoRA weights init at `B=0` so initial contribution is 0 and the TTT_LoRA starting point equals the base model. But gradient updates push LoRA in a direction that doesn't correspond to samacqua's intended semantics. Because my mlp_lora has `(bsz, dim, hidden_dim)` shape while the effective gradient signal comes from downstream `proj` which expects `(bsz, seq, hidden_dim)` as input, the LoRA ends up effectively noise-injecting into the residual stream via a dimension-mismatched projection path. Similar story for o_lora.

The correct fix requires:
1. Changing BatchedTTTLoRA.mlp_loras output dim from `hidden_dim` back to `dim`
2. Moving MLP lora application OUT of MLP.forward, back into a parallel add at residual level
3. Moving o_lora application to use pre-attention norm input

This is effectively rewriting the forward_ttt path structurally. Given the budget sink at this point in the session, we stopped here.

## Session aggregate conclusion

Three consecutive attempts to port samacqua's improvements failed or neutral-ed on our SP8192 + depth-recurrence stack:

| Attempt | Branch | Seed 42 result vs this-pod baseline 1.08041 |
|---------|--------|-------------------------------------------|
| VarLen + Fused MLP | `submission/sp8192-varlen-frontier` | Regression by 0.85 BPB (3.4x slower pod due to recompile cascades) |
| Per-layer adaptive GPTQ + INT7 embed | `submission/sp8192-per-layer-gptq` | Neutral (-0.00006, within noise) |
| Batched LoRA TTT | `submission/sp8192-lora-ttt` | Regression by 0.17 BPB (LoRA semantics bugs, eval time blown) |

Pattern: **the samacqua stack's improvements are co-tuned with VarLen architecture and depth recurrence absence**. Our depth-recurrence loop and parallel-residual structure are sufficiently different that his wins don't transfer cleanly. Each port would need deep structural rework to match his architecture, at which point we'd be reproducing samacqua's submission rather than building on PR #1572.

**Final recommendation: PR #1572 (1.07974 BPB) stays our best shipped submission.** Further gains would require either:
- Sitting on samacqua's entire stack (no depth recurrence) and rebuilding our extras on top -- essentially a new submission lineage
- Finding improvements from other PRs whose architecture matches ours more closely
- Deep own-research into what's distinctive about our stack that could be improved
