# Phase 4 report — the run (piecewise)

2026-09-17 15:10 → 2026-09-19 05:08, four segments, 30.8 h of training, 77.8M tokens (plus the 8M-token pilot it was
warm-started from). Config locked from Phase 3: full-size bridge on all 28 layers + LoRA r=16 (406.6M trainable), 8-bit AdamW,
2 sequences/step, B=32, context 256–768, grad checkpointing. Schedule: warmup-stable-decay, lr 1e-4 flat, linear decay to 1e-5
over the final 8,000 steps. Run dir `checkpoints/phase4_run/` (`best.pt` = final, step 38000). Check-ins via `phase4/status.py`.

## Result
| | Held-out loss | Masked CE (token-weighted) | Gap closed toward plain Dream |
|---|---|---|---|
| T₀ untrained | — | 4.913 | 0% |
| Pilot end (8M tokens) | 1.236 | 3.910 | 69.3% |
| Plateau best (step 30000, 61M) | 1.220 | 3.869 | 72.1% |
| **Final (step 38000, 77.8M)** | **1.180** | **3.772** | **78.8%** |
| D = plain Dream, context visible | — | 3.465 | 100% |

The decay phase did what the pilot predicted: eight consecutive improvements, 1.220 → 1.180 (the pilot's decay gave 0.06; this gave 0.04).
**Still falling at the last eval (1.184 → 1.180 at lr 1e-5).** More training might improve it further; this alone does not
establish the cause of the remaining gap.

These are validation results: the 32-sequence slice was reused for checkpoint selection and training decisions, and a fresh test
remains to be run. The training logs and checkpoints are not included in the repository. See [reproduction notes](REPRODUCING.md).

## Timeline and decisions
- Seg 1 (3.5 h): warm start from the pilot; held-out bumped 1.236 → 1.277 as lr re-warmed to 1e-4, then recovered. Stopped once by me to
  harden the eval (two early evals had returned NaN — intermittent, weight-state dependent, never recurred after per-sequence scoring was added;
  no non-finite sequence was recorded in 26 later evals, so no dump exists to diagnose).
- Seg 2 (16.5 h): flat plateau, slow real progress (−0.008 per 10k steps against eval noise sd 0.005). Beat the pilot's 1.236 around step 16–20k.
- Seg 3 (4.7 h): **early-stop rule fired** at step 33000 (1.220 → 1.223 → 1.227 → 1.231). Judged as plateau noise catching the rule, but the
  protocol says stop and roll back, don't hope — and the plateau was buying little. Rolled back to step 30000 (`last_earlystop_step33000.pt` kept).
- Seg 4 (6.0 h): decay from the step-30000 checkpoint over 8,000 steps → final. Plan total 77.8M instead of 100M (plateau cut by 11.5k steps).
- Health across all segments: 0 watchdog restarts, 0 driver timeouts, min device-free 3.30 GiB, max reserved 26.53 GiB, GPU ≤ 86 °C.

## Notes for Phase 5
- Samples at S=16 per block: on-topic openings, repetition in the low-confidence tail of each block (the steps-per-token effect from Phase 3).
  Phase 5's S sweep is the honest way to present this; the table must not be read off S=16 alone.
- `predecay.pt` (= step 30000) exists: a longer plateau toward 300M could resume from it, though the plateau's measured yield argues for a
  different lever first (e.g. more data per step, or a lower plateau lr) if a second run is ever wanted.
