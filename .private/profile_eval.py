"""Minimal eval-path profiler for the PR #1530 reproduction stack.

Answers: "Is CE / logit-softcap a meaningful fraction of eval runtime?"

Strategy: load a trained checkpoint, repeat the core eval op sequence
(forward_logits -> softcap -> cross_entropy) under torch.profiler on a
small batch, and dump an op-level breakdown.

This skips the sliding-eval and TTT-eval outer loops entirely -- those
loops don't contribute to per-token compute, they just feed batches
to the same forward_logits call. If CE is material in one forward pass,
it's material across the whole eval budget. If not, it's not.

Usage on pod:
    cd /workspace/repo
    python .private/profile_eval.py \\
        --checkpoint runs/seed42/final_model.pt \\
        --record-dir records/track_10min_16mb/2026-04-14_1530_Repro_RandomWarmup \\
        --output /tmp/eval_profile.txt
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path


def load_train_module(record_dir: Path):
    """Import train_gpt.py as a module so we can reach its GPT class.

    train_gpt.py reads env at import but only executes main() under
    `if __name__ == '__main__'`, so import is safe.
    """
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("DATA_DIR", "/workspace/repo/data")

    path = record_dir / "train_gpt.py"
    spec = importlib.util.spec_from_file_location("train_gpt_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--record-dir", required=True)
    ap.add_argument("--output", default="/tmp/eval_profile.txt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--active", type=int, default=20)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import torch.profiler as profiler

    print(f"Loading train_gpt.py from {args.record_dir}")
    mod = load_train_module(Path(args.record_dir))

    h = mod.Hyperparameters()
    device = "cuda"
    torch.cuda.set_device(0)

    print(f"Building model: {h.num_layers}L d={h.model_dim} vocab={h.vocab_size}")
    base_model = mod.GPT(h).to(device).to(torch.bfloat16)

    print(f"Loading checkpoint: {args.checkpoint}")
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = base_model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing: {len(missing)} keys (first 3: {missing[:3]})")
    if unexpected:
        print(f"  unexpected: {len(unexpected)} keys (first 3: {unexpected[:3]})")
    base_model.train(False)  # equivalent to nn.Module's eval mode

    # Synthetic batch matching eval shape. Contents irrelevant for profiling.
    x = torch.randint(0, h.vocab_size, (args.batch_size, args.seq_len),
                      device=device, dtype=torch.int64)
    y = torch.randint(0, h.vocab_size, (args.batch_size, args.seq_len),
                      device=device, dtype=torch.int64)

    # Warm up caches + compile before profiler window.
    print("Pre-profile sanity warmup...")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(3):
            logits = base_model.forward_logits(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, h.vocab_size),
                y.reshape(-1),
            )
    torch.cuda.synchronize()
    print(f"  sanity loss: {loss.item():.4f}")

    total_iters = args.warmup + args.active
    print(f"Profiling {total_iters} iterations ({args.warmup} warmup + {args.active} active)...")
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        schedule=profiler.schedule(wait=0, warmup=args.warmup, active=args.active),
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for i in range(total_iters):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, h.vocab_size),
                    y.reshape(-1),
                )
            torch.cuda.synchronize()
            prof.step()

    table_self = prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=40,
    )
    table_total = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=40,
    )

    CE_SUBSTR = ("cross_entropy", "nll_loss", "log_softmax", "softmax")
    SOFTCAP_SUBSTR = ("tanh",)
    ce_time_us = 0
    softcap_time_us = 0
    total_cuda_us = 0
    for e in prof.key_averages():
        name = e.key
        self_cuda = e.self_cuda_time_total
        total_cuda_us += self_cuda
        if any(s in name for s in CE_SUBSTR):
            ce_time_us += self_cuda
        if any(s in name for s in SOFTCAP_SUBSTR):
            softcap_time_us += self_cuda

    ce_frac = ce_time_us / total_cuda_us if total_cuda_us > 0 else 0.0
    softcap_frac = softcap_time_us / total_cuda_us if total_cuda_us > 0 else 0.0

    verdict_lines = [
        "=" * 60,
        "VERDICT",
        "=" * 60,
        f"total CUDA time:   {total_cuda_us/1e3:.1f} ms",
        f"CE-family time:    {ce_time_us/1e3:.1f} ms ({ce_frac*100:.1f}%)",
        f"softcap (tanh):    {softcap_time_us/1e3:.1f} ms ({softcap_frac*100:.1f}%)",
        f"CE+softcap total:  {((ce_time_us+softcap_time_us)/total_cuda_us)*100:.1f}%",
        "",
        "DECISION RULE:",
        "  >=15% CE+softcap -> fused-softcap-ce is the next delta",
        "  <5% CE+softcap   -> skip fused-CE, go to TTT rank sweep",
        "  5-15%            -> manual call, look at per-op table below",
        "",
    ]

    output = Path(args.output)
    output.write_text(
        "\n".join(verdict_lines)
        + "\n\nTOP BY SELF CUDA TIME:\n" + table_self
        + "\n\nTOP BY TOTAL CUDA TIME:\n" + table_total + "\n"
    )
    for line in verdict_lines:
        print(line)
    print(f"\nFull report: {output}")


if __name__ == "__main__":
    main()
