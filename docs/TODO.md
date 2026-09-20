# TwoStory — to-do / optimization backlog

- [ ] **Merge LoRA into Dream's base weights for inference.** Dream is bf16, so `W += (alpha/r)·B·A` is exact up to rounding. Removes 2 extra fp32
      matmuls + casts on each of 196 adapted projections per forward. Expected: trained bridge goes from ~37 to ~48-49 tok/s (the T₀ row's
      speed, which runs with adapters off), identical outputs up to rounding. Add a `T merged` row to the Phase 5 table. (Noted 2026-09-20.)
- [ ] **bf16 context tower at inference** (user's swap experiment): first prompt measured 30 → 52 tok/s; int8 `extend()` per block is the cost. Full
      numbers pending in `logs/phase5/raw_swap.jsonl`.
- [ ] **Exposure bias** (Phase 5 finding): repetition falls from 0.76 at 128-token prompts to 0.09 at 4096 — the cache fills with self-generated
      blocks the bridge never saw in training. Candidate fixes: train with model-generated blocks in the cache (scheduled sampling / self-forcing),
      or noise the context tower's input for committed blocks.
- [ ] Unresolved: two NaN held-out evals early in Phase 4 (steps 1000, 2000) never recurred after per-sequence scoring; no dump captured.
- [ ] Unresolved: one silent generation hang in the Phase 5 main session (no driver timeout logged); stall guard recovered it.
- [ ] Launch long jobs via `systemd-run --user`, never from the VS Code shell (its cgroup dies with VS Code).
