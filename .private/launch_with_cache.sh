#!/bin/bash
# Launch wrapper that persists the torch.inductor compile cache to the pod's
# network volume. First run populates; subsequent runs skip ~147s of TTT
# compile warmup plus ~30s of train-graph compile.
#
# Cache safety: the warmup path in train_gpt.py already uses torch.randint
# tokens (not validation data), so the cached compiled graphs contain no
# information about validation contents -- shapes drive inductor keys,
# not values.
#
# The cache is keyed by:
#   - torch version + inductor version
#   - GPU arch (SM_90 for H100)
#   - graph structure (shapes, dtypes, device)
# So reusing across runs on the same pod class is safe. Cache will
# invalidate automatically if any of those change.
#
# Usage:
#   bash .private/launch_with_cache.sh <SEED> [extra env vars]
# Example:
#   bash .private/launch_with_cache.sh 42 TTT_ENABLED=1 SLIDING_WINDOW_ENABLED=1

set -e

SEED="${1:-42}"
shift || true

# Volume-backed cache directory (survives pod stop/start).
CACHE_DIR="/workspace/.torch_inductor_cache"
mkdir -p "$CACHE_DIR"

export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1

# Report cache state so we can see cold-vs-warm starts in logs.
CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
CACHE_FILES=$(find "$CACHE_DIR" -type f 2>/dev/null | wc -l)
echo "=== Inductor cache state ==="
echo "dir: $CACHE_DIR"
echo "size: $CACHE_SIZE"
echo "files: $CACHE_FILES"
echo "==========================="

cd /workspace/repo
RECORD_DIR=records/track_10min_16mb/2026-04-14_1530_Repro_RandomWarmup
RUN_NAME="seed${SEED}"

rm -rf "runs/${RUN_NAME}"
mkdir -p "runs/${RUN_NAME}"

# Launch under nohup so SSH drops don't kill training.
nohup env \
    DATA_DIR=/workspace/repo/data \
    SEED="$SEED" \
    RUN_ID="$RUN_NAME" \
    ARTIFACT_DIR="/workspace/repo/runs/${RUN_NAME}" \
    TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR" \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
    "$@" \
    torchrun --standalone --nproc_per_node=8 "$RECORD_DIR/train_gpt.py" \
    > "/tmp/${RUN_NAME}.log" 2>&1 &
disown

sleep 2
echo "Launched ${RUN_NAME}, PID $(pgrep -f torchrun | head -1)"
echo "Log: /tmp/${RUN_NAME}.log"
