#!/bin/bash
# Pod scout: run BEFORE downloading data or kicking off training.
# Exits 0 if pod is fast enough, 1 if too slow, 2 if no CUDA.
# Usage on a fresh pod:
#   bash .private/pod_scout.sh
# Override threshold:
#   SCOUT_THRESHOLD_MS=100 bash .private/pod_scout.sh
set -e
cd "$(dirname "$0")/.."
python .private/pod_scout.py
