"""Pod scout: 60-90s micro-benchmark to reject slow H100 pods before real runs.

Builds a tiny stand-in transformer matching PR #1530's model shape and times
forward+backward at the per-rank training shape. Exits nonzero if median step
time exceeds the threshold (default 120ms per memory rule).

Run on the pod after env setup, BEFORE downloading data or starting training:
    python .private/pod_scout.py

Env knobs:
    SCOUT_THRESHOLD_MS   reject if median step > this (default 120)
    SCOUT_ITERS          timed iterations after warmup (default 12)
    SCOUT_BATCH          per-rank micro-batch sequences (default 48)
    SCOUT_SEQ            sequence length (default 2048)
"""
import os
import sys
import time
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F


THRESHOLD_MS = float(os.environ.get("SCOUT_THRESHOLD_MS", 120.0))
ITERS = int(os.environ.get("SCOUT_ITERS", 12))
WARMUP = int(os.environ.get("SCOUT_WARMUP", 4))
BATCH = int(os.environ.get("SCOUT_BATCH", 48))
SEQ = int(os.environ.get("SCOUT_SEQ", 2048))

# Model shape mirrors PR #1530 (records/track_10min_16mb/2026-04-10_VarLenAttn).
NUM_LAYERS = 11
MODEL_DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4
MLP_MULT = 4.0
VOCAB_SIZE = 8192
HEAD_DIM = MODEL_DIM // NUM_HEADS


class GQAAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(MODEL_DIM, NUM_HEADS * HEAD_DIM, bias=False)
        self.k = nn.Linear(MODEL_DIM, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v = nn.Linear(MODEL_DIM, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o = nn.Linear(NUM_HEADS * HEAD_DIM, MODEL_DIM, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k(x).view(B, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v(x).view(B, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        # GQA: expand kv heads
        rep = NUM_HEADS // NUM_KV_HEADS
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, NUM_HEADS * HEAD_DIM)
        return self.o(y)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.RMSNorm(MODEL_DIM)
        self.attn = GQAAttn()
        self.n2 = nn.RMSNorm(MODEL_DIM)
        hidden = int(MODEL_DIM * MLP_MULT)
        self.mlp_in = nn.Linear(MODEL_DIM, hidden, bias=False)
        self.mlp_out = nn.Linear(hidden, MODEL_DIM, bias=False)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        h = self.mlp_in(self.n2(x))
        h = F.relu(h).square()  # cheap stand-in for fused leaky-relu-square
        return x + self.mlp_out(h)


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(NUM_LAYERS))
        self.norm_f = nn.RMSNorm(MODEL_DIM)
        # tied head
        self.head_weight = self.embed.weight

    def forward(self, x, y):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_f(h)
        logits = h @ self.head_weight.t()
        return F.cross_entropy(
            logits.float().view(-1, VOCAB_SIZE), y.view(-1)
        )


def gpu_info():
    if not torch.cuda.is_available():
        return "no cuda"
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    return f"{name} | mem free {free/1e9:.1f}/{total/1e9:.1f} GB"


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA")
        sys.exit(2)

    torch.manual_seed(0)
    device = "cuda"
    print(f"GPU: {gpu_info()}")
    print(f"Shape: {NUM_LAYERS}L d={MODEL_DIM} H={NUM_HEADS}/{NUM_KV_HEADS} "
          f"vocab={VOCAB_SIZE} batch={BATCH} seq={SEQ}")
    print(f"Threshold: {THRESHOLD_MS:.0f}ms median step")

    model = TinyGPT().to(device).to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ), device=device, dtype=torch.long)
    y = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ), device=device, dtype=torch.long)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        opt.step()
        return loss

    print(f"Warming up ({WARMUP} iters)...")
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()

    times_ms = []
    print(f"Timing {ITERS} iters...")
    for i in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1e3
        times_ms.append(dt_ms)
        print(f"  iter {i:2d}: {dt_ms:6.1f} ms")

    median = statistics.median(times_ms)
    p10 = sorted(times_ms)[len(times_ms) // 10] if len(times_ms) >= 10 else min(times_ms)
    print(f"\nmedian={median:.1f}ms  p10={p10:.1f}ms  min={min(times_ms):.1f}ms  max={max(times_ms):.1f}ms")

    if median > THRESHOLD_MS:
        print(f"REJECT: median {median:.1f}ms > threshold {THRESHOLD_MS:.0f}ms")
        sys.exit(1)
    print(f"OK: median {median:.1f}ms <= threshold {THRESHOLD_MS:.0f}ms")
    sys.exit(0)


if __name__ == "__main__":
    main()
