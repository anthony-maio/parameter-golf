# NEGATIVE RESULT: VarLen + Fused MLP port from PR #1530

**Status: DO NOT SHIP. The port reproduces correctly but is performance-regressive vs the SP8192 frontier baseline.**

Pod session: 2026-04-14, 8xH100 SXM in US-MO-1, ~80 minutes total runtime, ~$31 burned.

## Three configurations tested at seed 42

| Config | Tok/s sustained | Steps in 588s | pre-quant val_bpb | quantized val_bpb | sliding val_bpb |
|--------|-----------------|---------------|-------------------|-------------------|-----------------|
| Baseline (records 2026-04-12) | 7.7M | 4606 | 1.087 | 1.098 | 1.082 |
| VarLen + Fused MLP (this port) | 2.3M | 1440 | 1.925 | 1.932 | 2.267 |
| Fused MLP only (no VarLen) | 3.4M | 2581 | 1.110 | 1.120 | 1.103 |
| Pure baseline reproduction | (run interrupted) | -- | -- | -- | -- |

## What we learned

### VarLen attention is incompatible with depth recurrence + parallel residuals + fullgraph compile

The DocumentPackingLoader produces cu_seqlens tensors with variable lengths (padded to 64-multiples). With depth recurrence active, the SAME blocks are called multiple times per forward, each potentially seeing different cu_seqlens shapes through the loop iterations. With `dynamic=False, fullgraph=True` torch.compile, this triggers cascading recompilations even with cache_size_limit bumped to 64.

Result: throughput dropped 3.4x (2.3M vs 7.7M tok/s sustained), step count dropped 3.2x (1440 vs 4606), and the model was massively undertrained -- val_bpb 1.93 vs baseline 1.087. The sliding-window eval came out WORSE than raw eval (2.27 vs 1.93), which suggests interaction bugs between our changes and the eval path that I didn't diagnose because the run was clearly not viable.

The same VarLen approach DID work for samacqua's PR #1530 (1.07336 BPB at 587s). The difference: samacqua's stack has NO depth recurrence. Without looping, each block is called once per forward, so cu_seqlens shape variance compiles into a bounded set of specializations (4-5 max). With our looping (NUM_LOOPS=2, loop layers 3-5), the combinatorial explosion of (loop_iter, cu_seqlen_shape) overflows the cache.

### Fused MLP Triton kernel is also regressive in this setup

Even without VarLen (USE_VARLEN=0 USE_FUSED_MLP=1), throughput dropped to 3.4M tok/s (2.3x slower than baseline's 7.7M). The Triton fused kernel uses `TensorDescriptor.from_tensor` per call, and for our small hidden dim (2048 = 4x512), the descriptor allocation + kernel launch overhead doesn't amortize. The standard PyTorch path (`F.linear` -> `F.leaky_relu` -> `.square()`) compiled by inductor appears faster for this scale.

Got 2581 steps vs 4606 baseline. Final pre-quant val_bpb 1.110 vs 1.087. Worse, not better.

### Implication for samacqua's claim of ~3% throughput improvement

Samacqua reports the fused MLP saves ~3% wallclock. That gain likely exists for:
- Larger hidden dimensions (where matmul work dominates kernel launch overhead)
- Models without fullgraph+depth-recurrence interactions

For our specific 35.9M params + depth recurrence + SP8192 stack, the kernel is regressive.

## Decision

**Do not ship this port.** PR #1572 (1.07974) remains our best submission.

## What's NOT a dead end

The conclusion is specific to combining VarLen + Fused MLP with our exact stack (depth recurrence + parallel residuals). The following are still potentially worth pursuing:

- **Move 2: per-layer adaptive GPTQ from PR #1586 (dexhunter)** -- config-level changes (clip_sigmas=12 for MLP, 13 for attn, 15 for embed; int7 embeddings; MATRIX_LR=0.026). No torch.compile interaction risk, no Triton kernel overhead. Pure post-training quantization tweaks. Expected gain: ~0.005 BPB based on dexhunter's measured 1.0749 vs baseline 1.0810.
- **Move 3: LoRA TTT from PR #1530** -- replaces our chunk-based score-first TTT. This is the largest claimed gain in samacqua's stack (~0.008 BPB). Eval-only (training path unchanged), so torch.compile recompile concerns from VarLen don't apply. Significantly more porting work than Move 2.
- **VarLen WITHOUT depth recurrence** -- would reproduce samacqua's setup but lose our depth-recurrence gain. The depth recurrence vs VarLen tradeoff hasn't been measured. Possibly: no depth recurrence + VarLen + fused MLP + LoRA TTT (i.e., samacqua's exact approach) would be a cleaner separate submission.

## Pod session summary

- Pod ID: vwpu12365pnuks (8xH100 SXM, US-MO-1, runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404)
- Started: 2026-04-14 08:23:51 UTC
- Exited (by RunPod, likely budget protection): 2026-04-14 09:42:22 UTC
- Runtime: ~78 min, cost ~$31
- Account balance: $48.10 -> $7.66

Significant fraction of pod time was spent on:
- Failed pod boots earlier in the session (Git Bash path translation munging /workspace -> C:/Program Files/Git/workspace; resolved by MSYS_NO_PATHCONV=1)
- Wrong image name first (`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` doesn't exist; correct template-id is `runpod-torch-v280` with image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`)
- Data download (~6 min for 25.6 GB sp8192 fineweb)
- Three full training runs (smoke + two failed config variations + pure baseline interrupted)
