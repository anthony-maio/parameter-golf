#!/bin/bash
# Pod setup for PR #1530 reproduction (random-token warmup variant).
# Run on pod after SSH is available. Stages are gated -- if scout fails,
# we exit BEFORE downloading data, so a slow pod costs <2 minutes.
set -e

cd /workspace

echo "=== Environment baseline ==="
which python && python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)" 2>&1 || echo "torch missing"
python -c "import triton; print('triton:', triton.__version__)" 2>&1 || echo "triton missing"
nvidia-smi --query-gpu=name,driver_version --format=csv | head -3

# Clone repo
if [ ! -d /workspace/repo ]; then
    git clone https://github.com/anthony-maio/parameter-golf.git /workspace/repo
fi
cd /workspace/repo
git fetch origin
git checkout submission/sp8192-1530-repro
git pull

RECORD_DIR=records/track_10min_16mb/2026-04-14_1530_Repro_RandomWarmup

# === STAGE 1: Pod scout (single GPU, ~60-90s) ===
# Must pass before installing FA3 or downloading data.
echo "=== Stage 1: Pod scout ==="
python .private/pod_scout.py
SCOUT_STATUS=$?
if [ $SCOUT_STATUS -ne 0 ]; then
    echo "POD SCOUT FAILED (status $SCOUT_STATUS) - terminating without further setup"
    exit $SCOUT_STATUS
fi

# === STAGE 2: Verify deps (template *may* be missing FA3) ===
echo "=== Stage 2: Verify deps ==="
# FA3 is the one likely-missing dep -- install the wheel if import fails.
if ! python -c "from flash_attn_interface import flash_attn_varlen_func" 2>/dev/null; then
    echo "FA3 missing -- installing wheel"
    pip install --break-system-packages "flash_attn_3" \
        --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 2>&1 | tail -5
fi
# Same defensive install for record-dir requirements (mostly no-ops on official template)
pip install --break-system-packages -r $RECORD_DIR/requirements.txt 2>&1 | tail -3 || true
python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
print('FA3 varlen: OK')
import triton
print('triton:', triton.__version__)
"

# === STAGE 3: Data ===
export DATA_DIR=${DATA_DIR:-/workspace/data}
mkdir -p $DATA_DIR
if [ ! -f "$DATA_DIR/datasets/fineweb10B_sp8192/fineweb_train_000.bin" ]; then
    echo "=== Stage 3: Downloading fineweb sp8192 (128 shards) ==="
    MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
        python data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128 2>&1 | tail -20
else
    echo "=== Stage 3: Data already cached ==="
fi

du -sh $DATA_DIR/datasets/fineweb10B_sp8192/ 2>/dev/null || true
echo "=== Setup complete -- ready for training ==="
