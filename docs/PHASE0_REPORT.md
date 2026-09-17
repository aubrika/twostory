# Phase 0 report — environment and premise verification

Run 2026-09-16/17 on thesia (Intel Arc Pro B70 32 GB, Ryzen 5 5600, 15 GB RAM, 20 GB swap).
Stack: Python 3.12, torch 2.14.0+xpu, transformers 4.57.6, peft 0.21.0, bitsandbytes 0.50.2 (XPU wheel), accelerate 1.15.0.
Raw reports: `logs/phase0/*.json`. Rerun everything with `phase0/run_all.sh`.

## Verdict: all eight checks PASS. Phase 1 may start.

| # | Check | Gate | Result |
|---|---|---|---|
| 00 | Environment | torch XPU, bnb int8 Linear on XPU, SDPA bool mask | PASS. int8 rel. err 0.005 vs bf16 |
| 01 | Lineage premise | tokenizer, hidden size, layers, heads, vocab identical | PASS. 16/16. Dream adds only `<|beginoftext|>`=151665 and `<|mask|>`=151666, both inside Qwen's 152064-row embedding |
| 02 | Qwen2.5-7B 8-bit on XPU | no swap, VRAM ≤ ~8 GiB | PASS. 8.12 GiB resident (8.79 peak), RSS peak 5.0 GiB, swap +0.34 GiB, 48 s |
| 03 | Both towers 8-bit resident | ≤ ~16 GiB | PASS. 16.24 GiB allocated (18.77 reserved); Dream forward finite |
| 04 | Baseline A (Dream) | deterministic, coherent | PASS. 3.19 tok/s @40-tok prompt, 1.77 tok/s @512-tok prompt (128 new tokens, 128 steps, alg=entropy, T=0) |
| 05 | Baseline B (Qwen AR) | deterministic, coherent | PASS. 4.03 tok/s @40, 4.04 tok/s @512 (greedy, 128 new) |
| 06 | LoRA on XPU | bf16 and int8+LoRA+grad-ckpt fwd/bwd | PASS. 11/11. Only LoRA params get grads |
| 07 | Pre-tokenize | held-out written and hashed | PASS. 400.5M tokens, 391,094 × 1024 seqs, 7,822 held-out (2.0%), sha256 `b73ab58c…` in `data/tokens/meta.json` |

## Decisions

- **RAM decision point: no purchase needed.** Loading streams shard-by-shard; peak RSS was 5 GiB per 7B model and swap growth
  was ≤ 1.4 GiB (the box already had ~6 GB of swap in use from other things). Not RAM-bound.
- **8-bit works on Arc via bitsandbytes' official XPU wheel.** No workaround needed. Note: `MatMul8bitLt` casts bf16 inputs
  to fp16 for the int8 matmul on XPU; outputs were coherent for both models, so fp16 range is not biting here, but keep it in mind
  if a layer ever goes non-finite.
- **Both towers resident = 16.24 GiB.** That leaves ~15 GiB for LoRA, bridge, optimizer state and activations. Possible saving
  later: drop Qwen's `lm_head` (1.09 GiB bf16) since the context tower never needs logits.
- **Hub revisions pinned** in `twostory/util.py` (Dream `6572adb…`, Qwen `d149729…`). Baseline A was produced against them.
- **Dream's generation loop, as read:** fully bidirectional (`is_causal=False`, no KV cache); every denoising step re-forwards the
  whole `prompt + max_new_tokens` sequence; logits are shifted one position (`logits[i]` scores token `i+1`); with `alg=entropy`,
  `temperature=0` it is deterministic (two runs token-identical). This is what the bridge must slot into.
- **Throughput baselines are slow in absolute terms** (both ≈ 0.25–0.3 s per forward) because bitsandbytes' int8 XPU kernels
  dominate, not the card. Every system in the final table runs the same int8 path, so the comparison is fair; but these Phase 0
  numbers are *regression targets* (`logs/phase0/baseline_A_dream_8bit.json` holds the exact token ids Phase 1 must reproduce),
  not benchmark results. The benchmark harness (`docs/BENCHMARK.md`) will re-measure with its own prompt set, warm-up, and repeats.

## Carry-forward constraints for Phase 1+ (from docs/BENCHMARK.md)

- Generation must expose **time-to-first-token and per-block commit timestamps** via a callback, so the harness can measure them
  without re-instrumenting later.
- Every results file carries `util.provenance()` (commit, dirty flag, pinned revisions, pip freeze) and `util.gpu_telemetry()`
  (xe clock, temperature, power) per run.
- The T₀ row (zero-init bridge, untrained) is the Phase 1 regression gate and stays in the final table.

## Addendum (2026-09-17): memory plan revised to bf16 denoiser + int8 context tower

**Why.** Measured, not assumed: bitsandbytes' int8 linear on XPU is 4-8x *slower* than bf16 at every shape
(0.23 ms vs 1.92 ms for the 3584→18944 MLP linear at batch 1). bf16 already runs at ~550 GB/s effective at batch 1,
i.e. at the card's bandwidth ceiling, so int8 cannot buy speed here; the unfused int8→bf16 conversion costs more than
the matmul it feeds. A naive weight-only int8 dequant is just as slow (1.75 ms), so it is the dequant, not bnb specifically.
Only a fused dequant-in-GEMV kernel would win, and none exists for XPU in PyTorch today (NF4 needs Triton-XPU, which
fails to build here for lack of the Level Zero headers).

**New plan.** `twostory/util.py`: `DENOISER_BITS = 16` (Dream runs every denoising step), `CONTEXT_BITS = 8`
(Qwen runs once per committed block at prefill shapes, where the int8 penalty is smaller and amortised).

| | int8 both (original) | bf16 denoiser + int8 context (now) |
|---|---|---|
| Both towers resident | 16.24 GiB | **22.31 GiB** (`03_load_both_ctx8_den16.json`) |
| Headroom for LoRA/bridge/Adam/activations | ~15 GiB | ~9.5 GiB (Phase 2 proves or shrinks it) |
| Baseline A: Dream 128 tok / 128 steps, short prompt | 3.19 tok/s | **21.12 tok/s** |
| Baseline A: Dream, 512-tok prompt | 1.77 tok/s | **9.61 tok/s** |
| Baseline B: Qwen greedy, short / 512 | 4.03 / 4.04 tok/s | **32.55 / 31.67 tok/s** |

Both baselines remain deterministic and coherent. **The Phase 1 regression target is now
`logs/phase0/baseline_A_dream_16bit.json`.** The 8-bit files are kept as a record.

**Open experiment (user's idea):** the bridge is trained against KV caches produced by the *int8* context tower. After
training, swap in the bf16 Qwen tower (both bf16 = 28.4 GiB, fits for inference) and measure. Per-layer int8 error is
~0.5%, so the bridge should transfer with a small shift, as QLoRA adapters do when merged onto full-precision bases, but it
is an empirical question: the benchmark harness gets an extra row, T with bf16 context tower, to answer it.
