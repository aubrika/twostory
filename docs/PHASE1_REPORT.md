# Phase 1 report — architecture wired, proven a no-op

Run 2026-09-17. Scripts: `phase1/10_regression.py`, `phase1/11_trainable.py` (`phase1/run_all.sh`). Raw: `logs/phase1/*.json`.

## Verdict: PASS (20/20 gates). Phase 2 may start.

### What was built
- `twostory/dream/` — vendored Dream modeling code (pinned revision) with a `bridge` slot per decoder layer and
  `ctx_kv`/`ctx_mask` threaded through `DreamModel.forward` → `DreamBaseModel.forward` → `DreamDecoderLayer.forward`,
  including the gradient-checkpointing call. 93-line diff in `docs/dream_modeling.diff`. An "ambient" context lets Dream's
  unmodified `diffusion_generate` loop run with bridges live.
- `twostory/bridge.py` — `CrossAttentionBridge`: RMSNorm (warm-started from the layer's input norm) → the layer's own frozen
  `q_proj` (shared weight, no copy) + LoRA delta (r=32) → RoPE at the block's absolute positions → SDPA against the context
  tower's post-RoPE K/V (GQA 4→28) → `out_proj`, **zero-initialised**. Optional low-rank `out_proj`. Inserted after
  self-attention, before the MLP, on all 28 layers by default. Trainable params fp32.
- `twostory/context_tower.py` — Qwen2.5-7B (int8) as a `DynamicCache` producer: `build(ids)` once, `extend(block)` per commit,
  `kv()` = per-layer (K, V). Skips `lm_head`. 57 KB/token (28 MiB for a 512-token prompt).
- `twostory/model.py` — `TwoStory`: `generate_full` (Dream's loop + bridge, for regression), `generate_block` (the TwoTower
  loop: denoiser input is `[anchor, B masks]`, Dream's confidence schedule over S steps, commit, extend cache; records prefill,
  TTFT, per-block seconds), `training_loss` (all blocks of a packed sequence batched as `[n_blocks, B+1]` rows, each
  cross-attending causally to cache positions before its start; masked-position CE with 1/t weighting).

### Gate results
| Gate | Result |
|---|---|
| Token-identical to Baseline A, short prompt (128 tok, 128 steps) | PASS, 3584 bridge calls (28 layers × 128 forwards) |
| Token-identical to Baseline A, 512-token prompt | PASS |
| Perturbed bridge changes output; zeroed again → identical again | PASS (wiring is live, not bypassed) |
| Trainable = bridge + LoRA only; context tower fully frozen | PASS. 366.2M bridge + 40.4M LoRA = 406.6M |
| One real training step: loss finite, every trainable param has a finite grad, no frozen param has a grad | PASS. loss 3.99 (masked CE 5.81) on a held-out sequence, ctx 512, B=32, 16 blocks |
| Train-step peak VRAM (grad checkpointing on) | 26.08 GiB allocated / 27.08 reserved |
| T₀ block-mode loop (B=32, S=8, 128 tokens) | runs; 33.4 tok/s, TTFT 0.92 s, prefill 0.27 s, 0.63 s/block; output unconditional garbage as expected |

### Design notes recorded for later phases
- **Dream's shift.** `logits[i]` predicts token `i+1`, so a block's first prediction comes from the token *before* the block.
  Block-mode denoiser input is therefore `[last committed token, B masks]` (B+1 rows), positions `T-1 … T+B-1`.
- **T₀ in block mode is unconditional.** The prompt reaches the denoiser only through the bridge, so with `out_proj = 0` the
  block is generated with no context at all (it emitted "1. 1. 0 0 0 …"). The benchmark's T₀ row is therefore a *throughput*
  control only; the quality identity T₀ = D holds in full mode (the regression gate), not block mode. `docs/BENCHMARK.md` §2
  should be read with that in mind.
- **Tokenisation.** Context tower gets Qwen-style ids (no BOS) in block mode and training; only full-mode regression uses
  Dream's BOS-prefixed prompt, fed to both towers so positions align.

### Concerns handed to Phase 2 (memory)
- 26.08 GiB peak is *before* optimizer state. Fp32 AdamW for 406.6M params adds ~3.3 GiB → ~29.4 GiB, over the 28 GiB target.
  Cheapest levers, in order: (1) low-rank `out_proj` (`BridgeConfig(out_rank=256)` → bridge ≈ 58M params, saves ~2.5 GiB of
  params+grads+Adam); (2) bridges on every other layer; (3) bf16 bridge params with fp32 Adam master copy; (4) 8-bit Adam if
  bitsandbytes' optimizer works on XPU. Phase 2's overfit test should measure (1) first.
- Block-mode denoiser forward over 33 tokens costs ~78 ms vs ~48 ms for plain Dream over 168 tokens: the fp32 bridges are
  not free. For inference/benchmarking, cast bridges to bf16 (Phase 3 throughput gate).
