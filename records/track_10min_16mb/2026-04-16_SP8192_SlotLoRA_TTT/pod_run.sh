#!/bin/bash
# All-in-one pod runner for the slot-LoRA TTT scaffolding sweep.
# Runs scout -> setup -> single training with TTT_SWEEP_CONFIGS spanning 4 configs.
# Designed to be invoked AFTER ssh-into-pod and `cd /workspace/repo`.
#
# Caller is expected to set:
#   DATA_DIR (default: /workspace/data, expects pgolf-data network volume mount)
#   POD_SPEEDGATE_MS (default: 110, raises RuntimeError at step 20 if too slow)
set -e

cd "$(dirname "$0")/../../.."  # back to repo root

RECORD_DIR=records/track_10min_16mb/2026-04-16_SP8192_SlotLoRA_TTT
export DATA_DIR=${DATA_DIR:-/workspace/data}
export POD_SPEEDGATE_MS=${POD_SPEEDGATE_MS:-110}

echo "=== Stage 1: Pod scout (fail-fast on slow pod) ==="
python .private/pod_scout.py
SCOUT=$?
if [ $SCOUT -ne 0 ]; then
    echo "SCOUT FAILED ($SCOUT) -- aborting before any expensive work"
    exit $SCOUT
fi

echo "=== Stage 2: Verify deps ==="
if ! python -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    echo "FA3 missing -- installing wheel"
    pip install --break-system-packages "flash_attn_3" \
        --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 2>&1 | tail -5
fi
pip install --break-system-packages -r $RECORD_DIR/requirements.txt 2>&1 | tail -5
python -c "
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
from flash_attn_interface import flash_attn_func
print('FA3: OK')
"

echo "=== Stage 3: Data check ==="
if [ ! -f "$DATA_DIR/datasets/fineweb10B_sp8192/fineweb_train_000.bin" ]; then
    echo "Data missing in $DATA_DIR -- downloading (5+ min)"
    MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
        python data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128 2>&1 | tail -10
else
    echo "Data already present at $DATA_DIR"
fi
du -sh $DATA_DIR/datasets/fineweb10B_sp8192/ 2>/dev/null || true

echo "=== Stage 4: Launch training with sweep ==="
mkdir -p logs
LOG_NAME="slotlora_$(date +%Y%m%d_%H%M%S).log"

# Sweep specification:
#   A is the existing TTT (defaults preserved) -- already produced by main eval flow as 'quantized_ttt'
#   B: control-only on layers 3-5 + global skip params
#   C: qv on layers 3-5 (matches old TTT_SELECTIVE_LAYERS=3 conceptually but on recurrent band)
#   D: slot-LoRA q,v rank 16 on layers 3-5
#   E: control + qv hybrid on layers 3-5 (combination test)
SWEEP="B_control345:TTT_LAYER_IDS=3,4,5;TTT_MODE=control;TTT_INCLUDE_GLOBAL=1|C_qv345:TTT_LAYER_IDS=3,4,5;TTT_MODE=qv|D_lora_qv345:TTT_LAYER_IDS=3,4,5;TTT_MODE=none;TTT_LORA_ENABLED=1;TTT_LORA_RANK=16;TTT_LORA_PROJ=q,v|E_control_qv345:TTT_LAYER_IDS=3,4,5;TTT_MODE=qv;TTT_INCLUDE_GLOBAL=1"

DATA_DIR=$DATA_DIR \
SEED=42 \
TTT_ENABLED=1 \
TTT_PHASE_LOG=1 \
TTT_SWEEP_CONFIGS="$SWEEP" \
torchrun --nproc_per_node 8 $RECORD_DIR/train_gpt_sota.py 2>&1 | tee logs/$LOG_NAME

echo "=== Done. Log: logs/$LOG_NAME ==="
echo "=== Summary lines: ==="
grep -E "val_bpb|sweep:|ttt:phase_ms" logs/$LOG_NAME || true
