# TwoStory — Benchmarking Protocol

**Quantitatively comparing the bridged model against its two parents (Dream-7B plain diffusion, Qwen2.5-7B autoregressive) on the Intel Arc Pro B70.**

*Companion to `twostory-protocol.md`. That document builds the thing; this one measures it honestly. The rules here exist because every benchmark table lies unless you make it hard for it to.*

---

## 0. The three questions the benchmark must answer

1. **Does the cache pay off?** Throughput of TwoStory vs plain Dream, as a function of prompt length. (The architectural claim.)
2. **Where does diffusion sit vs autoregression on this card?** Throughput and latency of TwoStory vs Qwen AR — the bandwidth-vs-compute crossover on Arc. (The hardware claim.)
3. **What did the bridge cost?** Quality of TwoStory vs plain Dream, and whether the gap is still shrinking. (The training-budget claim.)

Every metric below serves one of these. If a measurement doesn't feed one of the three, don't collect it.

---

## 1. Rigor rules — non-negotiable, in every run

These are the things that, skipped, turn a table into fiction. Bake each into the harness so they can't be forgotten.

- **Same build for every system.** Same PyTorch/XPU version, same commit of the TwoStory repo, same quantization for the shared weights. Pin and record the commit hash and the environment (`pip freeze`) in every results file. A comparison across different builds is not a comparison.
- **Identical prompts, identical decoding settings across systems.** The prompt set (§3) is fixed and hashed. Decoding is deterministic wherever possible (greedy / temperature 0); where sampling is required, fix the seed and report it.
- **Warm-up, then measure.** Discard the first N generations of every session (N = 3). The first pass pays one-time costs — kernel compilation, allocator warm-up, cache priming — that are not the steady state.
- **Repeat, and report the distribution.** Every throughput/latency number is the **median of ≥ 5 runs**, with the interquartile range (the middle 50% — a robust spread measure) reported alongside. Never a single run. Never a mean without a spread; a single hitch inflates a mean and hides in it.
- **Control thermals.** The B70 will throttle under sustained load and the numbers will drift downward over a long session. Log GPU clock and temperature with every run; interleave systems (A, B, C, A, B, C…) rather than running all of A then all of B, so drift hits all systems equally.
- **Isolate the machine.** No ComfyUI, no other model serving, no background jobs during a benchmark session. Same power/fan profile every session.
- **One variable at a time.** When sweeping, change exactly one knob per series. A grid where two things vary together is uninterpretable.
- **Record everything, decide later.** Log raw per-run timings to disk (JSONL). Aggregate in analysis, not in the harness. Raw data lets you re-cut the table without re-running the GPU.

---

## 2. The systems under test

| ID | System | Role |
|---|---|---|
| **Q** | Qwen2.5-7B, autoregressive, same quantization | Quality ceiling; AR throughput baseline |
| **D** | Dream-7B, plain masked diffusion (no bridge) | Diffusion baseline; the parent TwoStory must beat on throughput |
| **T** | TwoStory (bridged), **best checkpoint by held-out loss** | The subject |
| **T₀** | TwoStory, **zero-init bridge, untrained** | Control: must equal D exactly (regression check) and measures the pure *mechanical* throughput cost/benefit of the cache handoff before any learning |

**T₀ is the underappreciated row.** It separates "the cache mechanism itself changes speed" from "the trained bridge changes quality." If T₀ ≠ D on quality, the wiring is wrong. If T₀'s throughput already beats D's, the speed win is architectural and independent of training — an important attribution.

---

## 3. The prompt set

Build once, hash, freeze. Never regenerate mid-study.

**Length tiers** — the independent variable for question 1:
| Tier | Prompt tokens | Count |
|---|---|---|
| Short | 128 | 20 |
| Medium | 512 | 20 |
| Long | 2,048 | 20 |
| Very long | 4,096 (or the max the memory budget allows) | 10 |

**Content:** drawn from the **held-out split** of the training corpus (never trained on) plus a small set of hand-written task prompts (instructions, code, Q&A) so the numbers aren't purely in-distribution. Fixed generation length per tier: **256 output tokens**, so every system does the same amount of *output* work and only the *prompt* cost varies.

**Why length tiers are the spine:** the entire architectural claim is that TwoStory's advantage over D *grows with prompt length* (cache amortization). A benchmark without a length axis cannot test the claim.

---

## 4. Metrics — definitions, so they mean the same thing every time

### Throughput and latency (question 1 and 2)
- **tok/s (steady-state):** output tokens ÷ wall-clock from first output token to last. Excludes prompt processing. Report per length tier.
- **TTFT — time to first token:** wall-clock from request to first emitted token. For D and T this is dominated by prompt handling; for Q it's prefill. This is where the cache should show up most sharply.
- **Total wall-clock per request:** prompt + generation. The number a user actually feels.
- **Per-block latency (T and D only):** time per committed block. Lets you see the diffusion step cost directly.
- **Prompt-cost scaling:** fit total wall-clock vs prompt length per system; report the *slope*. D's slope should be steepest (pays the prompt every step); T's should be flatter; Q's in between. The slopes are the cleanest single statement of question 1.

### Quality (question 3)
- **Held-out masked-block loss** — primary. Computed on the frozen held-out split, identical for D and T (both are diffusion models scoring the same objective). This is the number you trust: never trained on, same objective, same data.
- **Held-out perplexity (Q):** Q can't be scored on the masked objective, so report its AR perplexity on the same held-out text as a *separate* quality reference — comparable in spirit, not in units. Do not put Q's perplexity and D/T's masked loss in the same column.
- **One public benchmark, small, as a sanity read:** a fixed 200-item subset of GSM8K via `lm-evaluation-harness`, same subset for all systems. Secondary — it tells you if something is badly broken, not which model is better by a point.
- **Long-context quality retention (question 3, sharpened):** held-out loss for D vs T at each prompt-length tier. The interesting result is whether T's loss degrades *less* than D's as prompts lengthen.

### Resource
- **Peak VRAM during generation**, per system per tier. T should be lower than D at long context (only one block resident in the denoiser).
- **GPU clock/temperature** per run — a control variable, logged not reported.

---

## 5. The sweep (T only)

The two knobs that define TwoStory's operating point. Sweep each **holding the other fixed at its default**, then run the full grid once at the Long tier.

| Knob | Values | Default |
|---|---|---|
| Denoising steps `S` | 4, 8, 16, 32 | 8 |
| Block size `B` | 16, 32, 64 | 32 |

For each point, record tok/s and held-out loss. This produces the **quality-vs-throughput Pareto curve** — the study's main figure. Plot held-out loss (y, lower is better) against tok/s (x, higher is better); each sweep point is a dot, and the lower-right envelope is T's frontier. Overlay D's operating points (D also has an `S` knob — sweep it identically for a fair frontier) and Q as a single point. **The claim, if it holds, is that T's frontier sits below-and-right of D's.** If T's frontier never dominates D's anywhere, the bridge didn't earn its cost — report that plainly.

---

## 6. Reporting — the table and the figure

**The table** (one per length tier, plus a summary):

| System | tok/s (med, IQR) | TTFT (med) | wall-clock (med) | held-out loss | peak VRAM |
|---|---|---|---|---|---|
| Q | | | | (perplexity, separate) | |
| D | | | | | |
| T₀ | | | | (must = D) | |
| T | | | | | |

**The figure:** the Pareto plot from §5, at the Long tier, with D's frontier overlaid. This is the one image that answers all three questions at once.

**The scaling plot:** total wall-clock vs prompt length, one line per system. Question 1 in a single picture.

**Every reported number carries:** commit hash, quantization, `S`/`B`, seed, number of runs, and the date. A number without its provenance is not a result.

---

## 7. Interpretation guide — what each outcome means

| Observation | Meaning |
|---|---|
| T₀ ≠ D on quality | Wiring bug. Stop. Fix before anything else. |
| T tok/s > D tok/s, gap grows with prompt length | Cache amortization works. Question 1: yes. |
| T tok/s > D but gap *flat* across lengths | Speed win exists but isn't the cache — find what it is (block size? step count?). |
| T tok/s < D | Bridge overhead exceeds cache savings on this card. Real finding; check per-block latency to localize. |
| T tok/s > Q tok/s | Diffusion parallelism beats AR decode on Arc at this point. Question 2: crossover favors diffusion. |
| T tok/s < Q tok/s | Compute-bound diffusion steps lose to bandwidth-bound AR on this card. Question 2: crossover favors AR here. Also a result. |
| T loss > D loss, but T's loss was still falling at end of training | Gap is training-limited. More tokens would close it. |
| T loss > D loss and plateaued | Gap is bridge-bottleneck-limited at this scale. The architectural cost showed up. |
| T's long-context loss degrades less than D's | Causal-cache context scales more gracefully than bidirectional diffusion. A capability result, not just speed. |

---

## 8. Division of labor

**Claude Code builds:** the benchmark harness — prompt-set generator (hashed), the interleaved runner with warm-up and repetition, thermal/clock logging, JSONL raw output, the sweep driver, and the analysis script that produces the table, the Pareto plot, and the scaling plot from the raw logs. Tell it: *raw timings to disk first, aggregate separately; interleave systems; median + IQR, never a bare mean.*

**You do:** freeze the prompt set (read it — make sure the hand-written prompts are what you want), run the sessions with the machine isolated, watch the thermal log, and **interpret** using §7. You also decide the one thing the harness can't: whether a surprising number is a bug or a finding. Surprising numbers get re-run before they get believed.

---

## 9. The one-sentence goal

To be able to write, with numbers you'd defend: *"On Intel Arc, TwoStory reaches X% of Dream's held-out quality at Y× Dream's throughput on 2k-token prompts, and sits [above / below] Qwen AR at Z tok/s — with the crossover at [S, B]."* Every rule in this document exists so that sentence is true when you write it.
