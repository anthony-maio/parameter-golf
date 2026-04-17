# Runbook: slot-LoRA sweep + PR #1674 friend-check

Two paid pod jobs to run when you have a clean few hours and capacity is available. Both can share one pod if provisioned well -- details below.

## Before anything else

```bash
# 1. Balance and pod state
runpodctl user | grep clientBalance
runpodctl pod list   # must be empty or existing pods are yours

# 2. If auth is on work GitHub, keep the SCP path (no push needed). Otherwise push.
git push origin submission/sp8192-slot-lora-ttt
```

## Critical rules (from the skill, do not violate)

- **Do NOT upgrade torch.** Template ships torch 2.8.0. FA3 wheel must match that version: `pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280`. If that wheel URL is stale, build FA3 from source.
- **Scout every pod, kill at >120 ms eager median.** Real production target is sub-100 ms/step.
- **Pretokenized shards: always set `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf`.** Default repo is wrong/rate-limited. Data lives at `/workspace/repo/data/datasets/fineweb10B_<variant>/` after download. Set `DATA_DIR=/workspace/repo/data` for the train script.
- **No network volume.** Fresh download per pod is the standard flow. Don't pin to US-GA-2 for a volume that isn't worth the constraint.
- **Stop pod the moment its job ends.** `runpodctl pod remove <id>` is fine; volume is not precious.

## Pod provisioning helper

```bash
# Try DCs in order of typical speed until one has 8xH100 SXM AND scouts fast.
# The "first DC that has stock" is often slow -- be ready to kill and retry.
for DC in "US-GA-1" "US-CA-2" "US-OR-1" "EU-RO-1" "AP-JP-1" "AP-IN-1"; do
  echo "--- Trying $DC ---"
  RESULT=$(MSYS_NO_PATHCONV=1 runpodctl pod create \
    --name "pgolf-sweep" --template-id "runpod-torch-v280" \
    --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 8 \
    --container-disk-in-gb 120 --volume-in-gb 100 \
    --volume-mount-path "/workspace" --data-center-ids "$DC" \
    --ports "22/tcp,8888/http" 2>&1 | grep -E '"id"|error' | head -3)
  echo "$RESULT"
  if echo "$RESULT" | grep -q '"id":'; then
    POD_ID=$(echo "$RESULT" | grep '"id"' | head -1 | sed -E 's/.*"id": "([^"]+)".*/\1/')
    echo "GOT $POD_ID IN $DC"
    break
  fi
done
```

Then poll SSH:

```bash
# Get IP/port
runpodctl pod get $POD_ID -o json | python -c "import sys,json; d=json.load(sys.stdin); s=d['ssh']; print(s['ip'],s['port'])"
# (capture as IP, PORT)

# Wait for SSH
until ssh -i "/c/Users/Ant/.runpod/ssh/RunPod-Key-Go" -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p $PORT root@$IP "echo SSH_READY" 2>/dev/null; do sleep 5; done
```

## One-shot pod bootstrap (run after SSH is up)

SCP up the files we need (work account can't push; this bypasses that):

```bash
# From local repo root
scp -i "/c/Users/Ant/.runpod/ssh/RunPod-Key-Go" -P $PORT \
  records/track_10min_16mb/2026-04-16_SP8192_SlotLoRA_TTT/* \
  root@$IP:/tmp/slotlora/
scp -i "/c/Users/Ant/.runpod/ssh/RunPod-Key-Go" -P $PORT \
  .private/pod_scout.py \
  root@$IP:/tmp/
```

On the pod (one SSH session):

```bash
set -e
cd /workspace
git clone --depth 1 --branch main https://github.com/anthony-maio/parameter-golf.git repo
cd repo
mkdir -p records/track_10min_16mb/2026-04-16_SP8192_SlotLoRA_TTT .private
cp /tmp/slotlora/* records/track_10min_16mb/2026-04-16_SP8192_SlotLoRA_TTT/
cp /tmp/pod_scout.py .private/

# Scout -- torch 2.8.0 eager target is ~100 ms. Over 120 ms: kill pod, try again.
python .private/pod_scout.py || { echo "SLOW -- killing"; exit 1; }

# Install deps (NO torch upgrade)
pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280
pip install --break-system-packages 'fused-softcap-ce @ git+https://github.com/anthony-maio/fused-softcap-ce.git@25e7ad6292cd1e837eef592f50e4d9f5990b6c84' huggingface-hub sentencepiece zstandard brotli pyminify
python -c "from flash_attn_interface import flash_attn_func; from fused_softcap_ce import fused_softcap_cross_entropy_per_token; print('deps OK')"

# Download pretokenized data
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
```

If any of those steps fail or the scout shows slow, **stop the pod immediately** and retry from the provisioning loop:

```bash
runpodctl pod remove $POD_ID
```

## Job 1: Slot-LoRA sweep (the main experiment)

**Goal:** validate slot-LoRA TTT scaffolding end-to-end and pull signal on which TTT configuration beats the PR #1572 baseline (1.08041 on seed 42).

**Cost estimate:** ~$9-12 on a fast (~90 ms/step) pod. 10 min train + 4 eval sweep configs (~3 min each) + setup.

**On the pod:**

```bash
cd /workspace/repo
mkdir -p logs
export DATA_DIR=/workspace/repo/data

# Sweep spec: A is baseline (runs as regular quantized_ttt step), B/C/D/E are added post-TTT.
SWEEP="B_control345:TTT_LAYER_IDS=3,4,5;TTT_MODE=control;TTT_INCLUDE_GLOBAL=1|C_qv345:TTT_LAYER_IDS=3,4,5;TTT_MODE=qv|D_lora_qv345:TTT_LAYER_IDS=3,4,5;TTT_MODE=none;TTT_LORA_ENABLED=1;TTT_LORA_RANK=16;TTT_LORA_PROJ=q,v|E_control_qv345:TTT_LAYER_IDS=3,4,5;TTT_MODE=qv;TTT_INCLUDE_GLOBAL=1"

SEED=42 \
TTT_ENABLED=1 \
TTT_PHASE_LOG=1 \
TTT_SWEEP_CONFIGS="$SWEEP" \
POD_SPEEDGATE_MS=120 \
torchrun --nproc_per_node 8 records/track_10min_16mb/2026-04-16_SP8192_SlotLoRA_TTT/train_gpt_sota.py 2>&1 | tee logs/slotlora_$(date +%Y%m%d_%H%M%S).log

# When it finishes, grep the summary:
grep -E "val_bpb|sweep:|ttt:phase_ms" logs/slotlora_*.log | tail -30
```

**What to look for:**

- `quantized_ttt val_bpb` (baseline Config A) should be in the 1.078-1.082 range. If it is not, the scaffolding broke the baseline and we need to debug before interpreting B/C/D/E.
- `sweep_B_control345 val_bpb` vs baseline: tells us if control-only TTT on the recurrent band is competitive
- `sweep_C_qv345 val_bpb`: matches old `TTT_SELECTIVE_LAYERS=3` semantically but on recurrent-band layers rather than last-N
- `sweep_D_lora_qv345 val_bpb`: the slot-LoRA headline test
- `sweep_E_control_qv345 val_bpb`: combined tweaks
- `ttt:phase_ms` lines: tell us whether eval time is bottlenecked on score-fwd, ttt-fwd, backward, or reduce -- guides follow-up kernel work

**Pull logs back locally:**

```bash
scp -i "/c/Users/Ant/.runpod/ssh/RunPod-Key-Go" -P $PORT \
  root@$IP:/workspace/repo/logs/slotlora_*.log \
  .private/overnight_results/
```

**Then decide:** if D beats A by >0.003 BPB, plan a 3-seed follow-up (~$18) on a fresh branch for submission.

## Job 2: Run PR #1674 for your friend

**Goal:** evaluate openai/parameter-golf PR #1674 on our 8xH100 setup and hand back results. This is a separate submission -- no modifications from us.

**Cost estimate:** ~$5-7 on a fast pod. Single seed 42 first, then 3-seed only if they ask.

**Prerequisite:** understand what's in the PR before running. This is not a blind execute -- at minimum:

```bash
# Local: fetch and inspect
git fetch upstream pull/1674/head:pr-1674
git show pr-1674 --stat | head -30
git log pr-1674 --oneline | head -20
git diff upstream/main..pr-1674 --stat | head -20
```

Look at:
- Which record dir they added (`records/track_10min_16mb/<date>_<name>/`)
- Whether they ship a `train_gpt.py` (compressed), `train_gpt_sota.py` (readable), or both
- `requirements.txt` for non-standard deps
- Any `NEGATIVE_RESULT.md` or notes explaining their approach

If the PR looks like a normal submission (single record dir + requirements + train_gpt), proceed. If it has anything weird -- custom kernels, custom data pipeline, banned techniques from the prohibitions list -- flag it and ask before running.

**On the pod (same bootstrap as above, plus):**

```bash
cd /workspace/repo
# Fetch their branch
git remote add upstream https://github.com/openai/parameter-golf.git 2>/dev/null || true
git fetch upstream pull/1674/head:pr-1674
git checkout pr-1674

# Install any extra deps their requirements.txt needs
PR_RECORD_DIR=$(ls -d records/track_10min_16mb/*/ | grep -v 2026-04-16_SP8192_SlotLoRA_TTT | sort | tail -1)
echo "PR #1674 record dir: $PR_RECORD_DIR"
[ -f "$PR_RECORD_DIR/requirements.txt" ] && pip install --break-system-packages -r "$PR_RECORD_DIR/requirements.txt"

# Run training, seed 42 only for first pass
SEED=42 \
DATA_DIR=/workspace/repo/data \
POD_SPEEDGATE_MS=120 \
torchrun --nproc_per_node 8 "$PR_RECORD_DIR/train_gpt.py" 2>&1 | tee logs/pr1674_seed42.log

# Grep the result
grep -E "val_bpb|val_loss" logs/pr1674_seed42.log | tail -10
```

**Expected output:** `quantized_ttt val_bpb` line gives their number. Compare to what the PR claims.

**Hand-back to friend:** their seed-42 BPB + a short note on compliance (did it stay under 600s train, 600s eval, 16MB artifact). If it's within noise of their claim, that's the result. If 3-seed is needed, set SEED=1337 and SEED=2024 and rerun (~$10-14 more).

## Shared-pod path (cheaper)

If Job 1 takes ~25 min and Job 2 takes ~25 min, one pod of ~55 min = ~$20. This is cheaper than two separate pod sessions ($30+) because each pod setup eats ~10 min dead time.

Sequence on a single pod:
1. Bootstrap (deps + data)
2. Job 1 sweep on `submission/sp8192-slot-lora-ttt` branch
3. `git fetch + checkout pr-1674`
4. Job 2 on PR #1674
5. Stop pod

Watch for: pr-1674's requirements may conflict with the slot-LoRA branch's (e.g., different FA3 versions). If so, split into two pod sessions to avoid debugging package conflicts.

## Final: kill the pod

Always finish with:

```bash
runpodctl pod remove $POD_ID
runpodctl pod list    # verify nothing running
runpodctl user | grep clientBalance    # confirm spend
```

If you set up but didn't run (you got interrupted), still remove the pod. Re-download data on the next pod -- not worth leaving a $22/hr machine idle.

## Files this runbook references

- [train_gpt_sota.py](train_gpt_sota.py) -- slot-LoRA scaffolding + sweep support
- [test_scaffolding.py](test_scaffolding.py) -- CPU smoke tests (already pass; safe to skip on pod)
- [NOTES.md](NOTES.md) -- design notes on why per-slot vs per-block
- [pod_run.sh](pod_run.sh) -- older helper that assumed network volume; prefer this runbook instead
- `.private/pod_scout.py` -- scout tool (untracked, SCP separately)

## State as of 2026-04-17 pickup

- Balance: $91.59 (down from $104 across two evenings of failed pod attempts)
- Local branch `submission/sp8192-slot-lora-ttt` has commits c61274e + a39b531, not pushed yet
- All 3 pod scouts this session ran 213-220 ms (one was also torch-upgraded which probably distorted the production number). US-GA-2 was available; the others sold out
- No open pods
- No usable experimental data yet -- both jobs remain pending
