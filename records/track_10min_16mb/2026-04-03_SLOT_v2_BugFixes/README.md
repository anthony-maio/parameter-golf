# SLOT v2 — Bug Fixes (RoPE + SWA + Logging)

Bug-fix iteration on PR #1303 (0.9462 BPB). Three fixes identified by multi-model code review:

## Fixes

1. **RoPE train_seq_len**: Hardcoded to 1024 but training at seq_len=2048, triggering unnecessary NTK extrapolation on every step. Fixed to 2048.
2. **SWA never applied**: SWA weights were collected during warmdown but never loaded back. Now blended 50/50 with EMA post-training.
3. **`_COMPRESSOR` undefined**: Logging referenced undefined variable after cleanup. Fixed.

## Architecture

Same as PR #1303: 11L, 512d, 8H/4KV GQA, LeakyReLU(0.5)^2 MLP 3x, VRL, VE128, BigramHash(1024), XSA all 11 layers, QK-Gain 4.0, SLOT-16, int6+lzma.

## Reproduction

```bash
SLOT_ENABLED=1 SLOT_STEPS=16 SLOT_LR=0.008 SLOT_LR_MIN=0.0008 \
  SEED=1337 DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- Base: PR #1303 / PR #175 (anthony-maio)
- Bug reports: Multi-model code review council (GPT-5.4, Gemini 3.1 Pro, Claude Opus/Sonnet)
