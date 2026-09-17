# Phase 2 report — overfit one batch

Run 2026-09-17. Script: `phase2/20_overfit.py`. Raw: `logs/phase2/*.json`, per-step JSONL under `checkpoints/phase2_overfit_*/train.jsonl`.

## Verdict: PASS (4/4) with the full-size bridge and 8-bit AdamW. Phase 3 may start.

Setup: one held-out sequence, ctx_len 512, B=32 (16 blocks), fixed mask pattern every step, LoRA r=16 on all seven
projections, full 3584×3584 bridge `out_proj` on all 28 layers (406.6M trainable), gradient checkpointing on, lr 3e-4 with
10-step warmup and cosine decay, grad clip 1.0.

| Optimizer | Loss step 1 → 30 → 200 | Optimizer state | Peak VRAM alloc / reserved | Gate |
|---|---|---|---|---|
| fp32 AdamW (`foreach=False`) | 2.87 → 0.0007 → 0.0000 | 2.88 GiB | **29.00 / 29.59 GiB** | memory FAIL (over 28) |
| **bitsandbytes AdamW8bit** | 2.87 → 5.81* → 0.0000 | 0.77 GiB | **26.85 / 27.13 GiB** | **PASS** |

1.5 s per step for one 1024-token sequence (≈ 670 training tokens/s) in both cases. The loop memorises the batch: loss is
below 0.01 by step 20, so the objective, masking, shift, cache plumbing and gradient path are all wired correctly.

\* The 8-bit run had a transient spike at steps ~25-45 (loss 0.013 → 5.8 → recovered to 0.0 by step 90) at the peak lr of 3e-4;
the fp32 run did not. Phase 3 should watch for this at its (lower) learning rate and treat it as a reason to prefer 1e-4 or
lower with 8-bit states. Nothing in Phase 2's gate depends on it.

## Decisions
- **Keep the full-size bridge** (user's call: try full first, shrink only if over the line). Memory lever used instead:
  8-bit optimizer states. `TrainConfig(optimizer="adamw8bit")` is now the default for training scripts.
- Headroom at peak: ~4.8 GiB below the card's 31.9 GiB. Longer `ctx_max` (768) means *fewer* blocks, so training memory does
  not grow with context; the only growth term is the eval/generation path, which runs without grads.

## Environment fix worth knowing (no sudo)
bitsandbytes' 8-bit optimizers (and NF4) run through Triton's XPU backend, which JIT-builds a small driver shim against the
Level Zero headers. The system has the loader (`libze1`) but not the dev headers. Fix, done in user space:
```bash
# headers matching the installed loader (libze1 1.28.6)
curl -sL https://github.com/oneapi-src/level-zero/archive/refs/tags/v1.28.6.tar.gz | tar xz -C ~/.local/src
mkdir -p ~/.local/include/level_zero ~/.local/lib
cp ~/.local/src/level-zero-1.28.6/include/*.h ~/.local/include/level_zero/
ln -sfn /usr/lib/x86_64-linux-gnu/libze_loader.so.1 ~/.local/lib/libze_loader.so
```
`twostory/util.py` sets `ZE_PATH=~/.local` (Triton's include root) and prepends `~/.local/lib` to `LIBRARY_PATH` when those
files exist. First Triton use JIT-compiles for ~10 s. This likely also unlocks `torch.compile` on XPU (untested).
