# Phase 3 report — pilot run

Run 2026-09-17, 10:04–13:12 (3.1 h), 8.0M tokens, 3907 steps × 2 sequences. Scripts: `phase3/30_pilot.py` under
`phase3/watchdog.sh`, `31_throughput.py`, `32_reference_loss.py`. Raw: `logs/phase3/*.json`, `checkpoints/phase3_pilot/{train,samples}.jsonl`.
Config: full-size bridge on all 28 layers + LoRA r=16 (406.6M trainable), 8-bit AdamW, lr 1e-4, 200-step warmup, cosine to 10%,
grad clip 1.0, B=32, context 256–768, gradient checkpointing.

## Verdict
Automated gates PASS (pilot 6/6; throughput: protocol gate PASS, my stricter equal-forwards gate PASS at 2k, narrow FAIL at 512).
**The go/no-go and the config lock are yours**; the evidence and my recommendations are below.

## 1. Is the bridge learning? Yes, steadily, and it is training-limited, not plateaued.
Held-out loss (32 fixed sequences, fixed masks), every 250 steps:
`1.595 → 1.446 → 1.391 → 1.350 → 1.327 → 1.313 → 1.295 → 1.294 → 1.276 → 1.270 → 1.260 → 1.253 → 1.250 → 1.246 → 1.242 → 1.237 → 1.236`
Sixteen evals, no regression, no NaN, no loss spikes at lr 1e-4 (the Phase 2 spike at 3e-4 did not recur). Still falling at the lr floor.

**What did the bridge cost? (masked cross-entropy on identical held-out blocks and masks)**
| System | CE |
|---|---|
| T₀ — untrained bridge, denoiser sees no context | 4.913 |
| **T — pilot best checkpoint, context only via the bridge** | **3.910** |
| D — plain Dream with the context in its own input (the ceiling) | 3.465 |

**69% of the T₀→D gap closed in 8M tokens**; 0.445 nats remain. (BOS vs no-BOS for D made no difference: 3.465 vs 3.468.)

## 2. Are generations degrading? No — they went from unconditional to on-topic — but decoding steps matter a lot.
Lighthouse prompt, block mode: step 0 `'..  1.  1. 1.  0 0 0 0'` → final `'1851, the petition was transferred to the Board of Trade, which
decided the matter to proceed to the construction of the lighthouse.'` The prompt's year, board and petition arrive through the bridge alone.
Two honest caveats:
- **Late-block repetition ("the the of the") is mostly steps-per-token, not the bridge.** Plain Dream at the same 4 tokens/step collapses the
  same way. Bridged sweep (64 new tokens, B=32): S=8 → repetition 0.36 at 30 tok/s; S=16 → 0.30 at 20 tok/s; S=32 → 0.20 at 12 tok/s, with fluent
  text (`'The man in the grey coat was wearing a grey hat and a grey scarf, and he was smoking a grey pipe.'`). Baseline A is 0.02.
- **Precise copying from long context is not there yet.** On the repetitive 512/2k prompts Dream copies verbatim; the bridged model drifts.

## 3. Does the cache pay off? Yes, growing with prompt length; crossover just above 512 tokens.
128 new tokens, median of 5 after warm-up, bridges bf16, plain Dream measured with LoRA adapters disabled:
| Prompt | Dream S=128 (Baseline A setting) | Dream S=32 | Bridged S=8,B=32 (32 forwards) | TTFT bridged |
|---|---|---|---|---|
| 40 tok | 21.0 tok/s | 83.8 | 37.5 | 0.85 s |
| 512 tok | 9.7 | 38.6 | 35.8 | 1.02 s |
| 2048 tok | 2.8 | 11.1 | **28.6** | 1.93 s |
Dream slows ~in proportion to prompt length (it re-processes the prompt every step); the bridged model barely slows. At 2k it is 2.6× Dream
at equal forwards and 10× Baseline A. At short prompts the bridge's fixed overhead loses.

## 4. The incident (cost one overnight run) and what changed
The first launch wedged at step 2 for 8 h. Causes: SDPA on XPU with stride-0 *expanded* K/V returns NaN under no_grad and selects a very slow
kernel; plus VRAM exhaustion (Phase 2 measured 16 blocks, worst case is 24; allocator fragmentation reserved 29.4 GiB vs 26.0 allocated). A single
GPU job then exceeded xe's 5 s `job_timeout_ms` (root-only, max 10 s) → engine reset → process spins forever. My monitor watched a grep-piped
log (block-buffered → silent). Fixes: blocks folded into the query dimension (no K/V expand/copy; bit-identical losses on a tiny model; Phase 1
regression still token-identical), `lm_head` only at masked positions, Qwen `lm_head` dropped, allocator guard + `dev_free_gib` logged per step,
15-min checkpoints, `--resume`, progress-based watchdog. Pilot rerun: 0 restarts, 0 driver timeouts, min device-free 3.56 GiB, 2.6 s/step (~800 tok/s).

## 5. Recommendations for the config lock
| Knob | Recommend | Why |
|---|---|---|
| LR / schedule | 1e-4, 200 warmup, cosine to 10% | clean curve, no spikes with 8-bit Adam |
| Bridge density | all 28 layers, full `out_proj` | fits (26.1 GiB peak), learning fast; every-other-layer untested, no reason to shrink |
| LoRA | r=16 all seven projections | unchanged |
| Batch | 2 sequences/step | ~800 tok/s → 100M tokens ≈ 35 h, 300M ≈ 4.3 days |
| Phase 4 length | start with 100M, decide on more from the curve | gap is training-limited; 69% closed at 8M |
| Eval additions | log gap-closed vs the fixed D reference (3.465) each eval; sample at S=16 | makes the curve interpretable |
| Run hygiene | launch via `phase3/watchdog.sh`-style driver; close Steam | the driver wedge is unrecoverable without it |
Warm-start option: Phase 4 can resume from `checkpoints/phase3_pilot/best.pt` (8M tokens already paid for) or start fresh for a clean lr schedule.
