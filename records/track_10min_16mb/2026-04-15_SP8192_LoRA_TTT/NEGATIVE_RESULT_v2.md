# NEGATIVE RESULT UPDATE: Batched LoRA TTT semantic fix attempt

**Status: DO NOT SHIP. Semantic fixes did not resolve the regression.**

Following council consensus from three models + followup analysis, applied the two diagnosed semantic bugs fix:

1. `mlp_lora` shape changed from `(bsz, dim, hidden_dim)` to `(bsz, dim, dim)`, moved from inner MLP.forward tweak to Block-level parallel residual bypass (`mlp_out = block.mlp(mlp_n) + lora_mlp(mlp_n)`)
2. `o_lora` input changed from attention output `y` to pre-attention normalized residual `n`
3. Plus: pod speedgate at step 20 (POD_SPEEDGATE_MS env var), tighter clip_sigmas for looped layers 3-5 (LOOP_CLIP_SIGMAS env, default 10.0)

Seed 42 on fresh AP-IN-1 pod (passed 100.6ms/step speedgate, under 110ms threshold):

| Metric | Pre-fix (commit 6f024bd) | Post-fix (commit 2824f70) | Delta |
|---|---|---|---|
| Training steps | ~4500 | 4575 | equivalent |
| Pre-quant val_bpb | ~1.087 | 1.08736 | equivalent |
| Sliding val_bpb | 1.08292 | 1.08150 | -0.00142 (better, within noise) |
| **TTT val_bpb** | **1.25076** | **1.25268** | **+0.00192 (still regression)** |
| TTT eval time | 909s | 882s | -27s (still non-compliant, >600s cap) |
| Artifact size | 15.98 MB | 16.41 MB | +430KB (now non-compliant, >16MB cap) |

The semantic fixes did not resolve the fundamental regression. Identical failure pattern (~1.25 BPB instead of expected sub-sliding BPB ~1.075) confirms the root cause is NOT the two bugs I diagnosed.

**Implication: there is a deeper bug in forward_ttt or in the LoRA update loop that I did not identify.** Possibilities include:
- Logit computation path differs from samacqua's (tok_emb tied vs lm_head treatment)
- Loss masking / chunk boundary logic has an off-by-one
- Optimizer state reset between batches has a subtle issue
- Interaction between depth recurrence slots and LoRA slot indexing is wrong

Artifact regression (15.98 -> 16.41 MB) is from two contributions: the larger readable-Python code (77KB vs baseline minified 17KB = +60KB) and LOOP_CLIP_SIGMAS=10.0 apparently worsening brotli compressibility of the quantized weights (+370KB).

Given the depth of the remaining bug (requires careful line-by-line tracing of forward_ttt against samacqua's reference on a foreign codebase) and budget state, we stopped.

**PR #1572 (1.07974) remains the best submission.**

## Session aggregate: 4 attempts at Lineage B improvements, 4 failures

| Attempt | Branch | Seed 42 result |
|---|---|---|
| VarLen + Fused MLP | submission/sp8192-varlen-frontier | Regression 0.85 BPB (compile cascades) |
| Per-layer adaptive GPTQ | submission/sp8192-per-layer-gptq | Neutral (-0.00006, within noise) |
| Batched LoRA TTT (initial) | submission/sp8192-lora-ttt | Regression 0.17 BPB (semantic bugs + eval timing) |
| Batched LoRA TTT (semantic-fixed) | submission/sp8192-lora-ttt (2824f70) | Regression 0.17 BPB (deeper bug remains) |

This aggregate is consistent with the finding from the first negative result doc: **samacqua lineage improvements do not port piecewise onto our depth recurrence + module-based architecture stack**. Each attempt has an architectural interaction that resists surgical fix.
