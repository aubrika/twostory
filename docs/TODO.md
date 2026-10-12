# TwoStory — to-do / optimization backlog

- [ ] **Baseline: Fast-dLLM v2 7B** (NVIDIA's single-model block-diffusion conversion of Qwen2.5-7B-Instruct, ~1B tokens) on the same card, same
      harness, same judge: `bench/run_bench.py --session f`. Queued to run after the GSM8K sessions (2026-09-20). The question it answers:
      does a bridge-only recipe compete with a full fine-tune at a fraction of the training lift?

- [ ] **Merge LoRA into Dream's base weights for inference.** Dream is bf16, so `W += (alpha/r)·B·A` is exact up to rounding. Removes 2 extra fp32
      matmuls + casts on each of 196 adapted projections per forward. Expected: trained bridge goes from ~37 to ~48-49 tok/s (the T₀ row's
      speed, which runs with adapters off), identical outputs up to rounding. Add a `T merged` row to the Phase 5 table. (Noted 2026-09-20.)
- [ ] **bf16 context tower at inference** (Aubrey's swap experiment): first prompt measured 30 → 52 tok/s; int8 `extend()` per block is the cost. Full
      numbers pending in `logs/phase5/raw_swap.jsonl`.
- [ ] **Exposure bias** (Phase 5 finding): repetition falls from 0.76 at 128-token prompts to 0.09 at 4096 — the cache fills with self-generated
      blocks the bridge never saw in training. Candidate fixes: train with model-generated blocks in the cache (scheduled sampling / self-forcing),
      or noise the context tower's input for committed blocks.
- [ ] Unresolved: two NaN held-out evals early in Phase 4 (steps 1000, 2000) never recurred after per-sequence scoring; no dump captured.
- [ ] Unresolved: one silent generation hang in the Phase 5 main session (no driver timeout logged); stall guard recovered it.
- [ ] Launch long jobs via `systemd-run --user`, never from the VS Code shell (its cgroup dies with VS Code).

## Frozen 2026-09-20 (paused; pivot to the linear K/V-map bridge)
- Phase 5 sessions main, q, swap (short+medium + held-out CE), gsm, gsmq: COMPLETE in `logs/phase5/raw_*.jsonl`.
- Fast-dLLM v2 session `f`: PARTIAL (stopped mid-run; resume-by-key with `bench/run_all.sh f`).
- Distillation probe `bench/distill_probe.py`: written and preflighted in code only, NOT run. Resume with `bench/queue_probe.sh`.
- Phase 5 report not yet written; `bench/analyze.py` regenerates tables/figures from whatever raw data exists.
