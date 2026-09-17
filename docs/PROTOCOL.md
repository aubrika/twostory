# TwoStory — Experimental Protocol

**TwoStory: a homelab TwoTower. Dream-7B (denoiser) + frozen Qwen2.5-7B (context tower), trained bridge-only via LoRA, on an Intel Arc Pro B70. Smaller than a tower.**

*Written Sept 2026. Designed to be executed once, correctly: every phase has a go/no-go gate, and nothing expensive starts until the cheap things have proven the setup works.*

---

## 0. The idea in three sentences

NVIDIA's TwoTower (arXiv 2606.26493) pairs a **frozen autoregressive context tower** (processes the prompt + committed tokens once, produces a reusable KV cache) with a **diffusion denoiser tower** (refines a block of masked tokens in parallel, cross-attending to that cache). They trained the denoiser from a raw AR copy on ~2.1T tokens — datacenter scale. We shortcut it: use **Dream-7B**, which *already denoises* and is *itself derived from Qwen2.5-7B*, as the denoiser, and **Qwen2.5-7B** as the frozen context tower — so the towers share a lineage and representation space — and train **only the cross-attention bridge** (plus small LoRA adapters) on ~100–300M tokens. Days on one card, not years.

**Hypothesis:** a lineage-matched LoRA bridge recovers a meaningful fraction of TwoTower's benefit (block-parallel generation *with* a reusable context cache) at homelab training cost. **We do not expect NVIDIA's 98.7% quality retention.** The result is a measured quality-vs-throughput curve for this recipe on Arc, which does not exist anywhere.

---

## 1. Why this pairing, specifically

| Choice | Reason |
|---|---|
| **Dream-7B as denoiser** | Already trained on the masked-diffusion objective. The expensive "learn to denoise" step is done. Initialized from Qwen2.5-7B, so its layers speak Qwen's representation language. |
| **Qwen2.5-7B (base) as context tower** | Dream's ancestor. Same tokenizer, same hidden size (3584), same layer count (28), same architecture. Cross-attention layer-to-layer has a natural alignment with no learned bridge for *dimensions* — only for *function*. |
| **Both pure transformers** | Nemotron's backbone is a Mamba/attention/MoE hybrid, which complicates cache handoff. Qwen/Dream are plain transformers: the context tower hands over ordinary per-layer KV. Simpler than the paper. |
| **Freeze both, train the bridge** | The towers' competence is already paid for. What TwoTower *adds* is "use a separate frozen cache instead of reprocessing the prompt." That's the only thing we need to learn. |

**Verify before anything else (Phase 0):** that Dream-7B and Qwen2.5-7B genuinely share tokenizer vocab, hidden size, and layer count. If any of these differ, the lineage-match premise is wrong and the plan changes. *(Expected: they match. Confirm, don't assume.)*

---

## 2. Hardware reality and the two constraints that will bite

**VRAM: 32 GB.** Two 7B towers at BF16 ≈ 15 GB each ≈ 30 GB before a single activation. **Does not fit for training.** The plan therefore requires the frozen towers in **8-bit** (≈ 7.5 GB each ≈ 15 GB total), leaving ~17 GB for LoRA params, bridge params, optimizer state, activations, and gradient checkpointing. This is the memory plan; it has to be *proven* on XPU in Phase 0, because 8-bit weight loading on Arc is not a given (bitsandbytes is CUDA-first).

**System RAM: 15 GB.** This is the sneakier constraint. Loading a 7B model normally stages weights through CPU RAM; two of them will exceed 15 GB and swap or OOM *before they ever reach the GPU*. Mitigation: load with `low_cpu_mem_usage=True` and `device_map` targeting XPU so safetensors stream shard-by-shard to the device. **If Phase 0 shows loading is RAM-bound, this is the moment to buy the 64 GB DDR4 kit** — it turns a multi-day fight into a non-issue, and it's the upgrade that keeps coming up.

**Known Arc/XPU friction to expect** (all Phase 0 checks):
- No FlashAttention on XPU; use PyTorch SDPA and accept the memory/speed profile.
- The ~21.7 GB single-buffer allocation ceiling you hit before — keep any one tensor well under it; gradient checkpointing and small block sizes help.
- `peft` (LoRA) is CUDA-first; it usually works on XPU via plain PyTorch but verify a LoRA forward/backward on XPU in isolation before building on it.
- Dream's own generation code is HF-Transformers-based; read it first (Phase 0) — the bridge must slot into *its* denoising loop, not a reimplementation.

---

## 3. Architecture spec (what Claude Code builds)

**Context tower (frozen):** Qwen2.5-7B, causal, 8-bit weights, `requires_grad=False`. Given clean tokens (prompt + committed blocks), run a forward pass and **capture per-layer K and V** (28 layers). This is the cache. Recomputed only when a block is committed (extend, don't rebuild).

**Denoiser tower (Dream-7B, frozen base + trainable additions):**
- Base weights 8-bit, frozen.
- **LoRA adapters** on attention projections (q, k, v, o) and MLP, rank 16–32. Trainable.
- **Cross-attention bridge:** one new cross-attention sub-block inserted per layer (or per every-other layer — a Phase 3 decision), where **queries come from the denoiser's hidden state at that layer** and **keys/values come from the context tower's cache at the same layer index.** Attention is **causal over the context** (the block may look at the past, not the future). Trainable.
- **Zero-initialize the bridge's output projection.** This is the single most important implementation detail for one-shotting: at init, the bridge contributes exactly zero, so the wired model is *behaviorally identical to plain Dream.* That gives you a regression test (Phase 1) and a stable training start (the model can't get worse than Dream at step 0).

**Generation loop (block-wise):**
1. Context tower processes the prompt → cache.
2. Denoiser takes a block of `B` mask tokens (start `B=32`), runs `S` denoising steps (start `S=8`), each step attending bidirectionally within the block and cross-attending to the cache.
3. Commit the block when confident (confidence-threshold decoding, per Dream's own scheme) → append tokens → context tower extends the cache by processing the new clean block.
4. Repeat until EOS or length limit.

**Training objective:** Dream's masked-diffusion loss, unchanged — but with the *prompt portion supplied via the context tower's cache* rather than in the denoiser's own input. Concretely: split each training sequence into (context, target-block); context goes through the frozen Qwen tower → cache; the target block is masked at a sampled ratio and the denoiser must reconstruct it using its own bidirectional view + the cache. Loss only on masked target positions. This teaches exactly one thing: *use the cache.*

---

## 4. Data

- **Corpus:** a clean, permissively-licensed web/edu text sample — **FineWeb-Edu** (HF) is the default. Pull a ~1B-token shard; you'll use 100–300M.
- **Preprocessing:** tokenize with the shared Qwen tokenizer; pack into sequences of 1024 tokens; for each, choose a split point so context is 256–768 tokens and the target block is the remainder in `B`-sized chunks. Pre-tokenize to disk once (Phase 0) — never tokenize in the training loop.
- **Held-out:** reserve 2% of sequences, never trained on. This is your eval set for the whole run. **Do not touch it for anything else.**

---

## 5. The phased plan — with go/no-go gates

The one-shot discipline is: **each phase is cheap, proves one thing, and gates the next.** Nothing multi-day starts until Phases 0–3 all pass. Budget: Phases 0–3 ≈ 1–3 days of effort but ≈ minutes-to-hours of GPU time. Phase 4 is the multi-day run.

### Phase 0 — Environment and premise verification (GPU time: minutes)
Claude Code builds a `phase0/` set of standalone scripts; you run and read each.
- [ ] Load Qwen2.5-7B on XPU at 8-bit. Report peak VRAM and peak system RAM. **Gate:** loads without swap/OOM; VRAM ≤ ~8 GB.
- [ ] Load Dream-7B on XPU at 8-bit. Same report. **Gate:** both resident simultaneously ≤ ~16 GB VRAM.
- [ ] Assert tokenizer equality (vocab hash), hidden size, layer count between the two. **Gate:** all match. *(If not: stop, reassess.)*
- [ ] Run Dream's own generation on one prompt; record output and tokens/sec. This is **Baseline A**.
- [ ] Run Qwen2.5-7B AR generation on the same prompt; record tokens/sec. This is **Baseline B**.
- [ ] LoRA forward + backward on XPU on a tiny model: proves `peft` works on Arc. **Gate:** no errors, loss.backward() completes.
- [ ] Pre-tokenize the corpus to disk. **Gate:** held-out split written and hashed.
- [ ] **RAM decision point:** if loading was swap-bound, buy the RAM kit *now*, before Phase 1.

### Phase 1 — Wire the architecture, prove it's a no-op (GPU time: minutes)
- [ ] Implement the context-tower cache capture and the zero-init bridge in Dream's forward.
- [ ] **The regression test:** with the bridge zero-initialized and LoRA at zero, generate on the Phase-0 prompt. **Gate:** output is *token-identical* to Baseline A (greedy/deterministic decoding). If it isn't, the wiring changed Dream's behavior and must be fixed before proceeding. This gate is what makes the rest one-shottable.
- [ ] Confirm gradients flow to bridge + LoRA params *only* (print trainable param count; ≈ tens of millions, not billions). **Gate:** frozen towers have no grad.

### Phase 2 — Overfit a single batch (GPU time: ~10 minutes)
- [ ] Train on **one** batch repeatedly. **Gate:** loss drives toward ~0. If it can't memorize one batch, the training loop is broken — fix before spending hours. (Classic sanity check; skipping it is how multi-day runs get wasted.)
- [ ] Record peak VRAM during training with gradient checkpointing on. **Gate:** comfortably under 32 GB with headroom (target ≤ 28 GB). If not, reduce block size / sequence length / LoRA rank *now*.

### Phase 3 — Pilot run (GPU time: 2–4 hours)
- [ ] Train on ~5–10M tokens with the real config. Eval on held-out every ~500 steps: masked-block loss, plus generate 5 fixed prompts and eyeball them.
- [ ] **Gate:** held-out loss is *decreasing* and generations are *not degrading* vs Baseline A. Two failure signatures to watch: loss flat (bridge isn't learning — check zero-init/LR), or generations getting worse (LR too high, or LoRA damaging Dream — lower LR, reduce rank).
- [ ] Decide bridge density (every layer vs every-other) and final LR from pilot behavior. Lock the config.
- [ ] Measure tokens/sec of the *bridged* generation vs Baselines A and B. Even untrained-ish, this tells you whether the cache handoff is delivering throughput. **Gate:** bridged generation is faster than Baseline A on prompts ≥ 512 tokens (the cache should pay off with long context). If it's slower, there's an implementation inefficiency to fix before the long run.

### Phase 4 — The run (GPU time: days)
- [ ] 100–300M tokens, locked config. Conservative LR (start ~1e-4 for LoRA/bridge, cosine decay). Checkpoint every ~2 hours to disk. Eval on held-out every ~2k steps; keep the best checkpoint by held-out loss.
- [ ] **Early-stop rule:** if held-out loss rises for 3 consecutive evals, stop and roll back — don't hope.
- [ ] Log everything (loss, LR, VRAM, tokens/sec, timestamps). `tmux` + a watchdog that alerts you if the process dies; the machine will be pinned for days.
- [ ] **Do not touch the config mid-run.** If something's wrong, stop, fix, restart from the last good checkpoint.

### Phase 5 — Evaluation (GPU time: hours)
The deliverable is a **quality-vs-throughput table**, four rows:
| System | Quality (held-out masked loss + a small benchmark) | tok/s @ 512-tok prompt | tok/s @ 2k-tok prompt |
|---|---|---|---|
| Baseline B: Qwen2.5-7B AR | | | |
| Baseline A: Dream-7B plain diffusion | | | |
| Bridged TwoTower, best ckpt, S=8 | | | |
| Bridged TwoTower, sweep S∈{4,8,16}, B∈{16,32,64} | | | |

- Quality: held-out loss is the honest primary metric (never trained on). Add one cheap public benchmark (e.g. a GSM8K subset via `lm-evaluation-harness`) for a sanity read — but treat it as secondary; the held-out loss is the number you trust.
- Throughput at **two prompt lengths** is the point: TwoTower's advantage should *grow* with context length (the cache amortizes). If it doesn't, that's a finding too.
- Report the step-count sweep as a curve. That curve is the contribution.

---

## 6. Division of labor

**Claude Code does:** all implementation — Phase 0 scripts, the bridge/cache wiring inside Dream's forward, the training loop with checkpointing/logging/early-stop, the eval harness and the sweep. Point it at this document and at Dream's HF modeling file. Tell it explicitly: *zero-init the bridge output projection; the Phase 1 regression gate must pass token-identically before any training code is written.* Have it write each phase as a separate runnable script with a printed PASS/FAIL against the gate.

**You do:** run each phase, **read the gate result, and decide.** You own the go/no-go at every phase boundary — that judgment is the thing that can't be delegated and is what makes it one-shot. You also own: the RAM purchase decision (Phase 0), the config lock (Phase 3), the mid-run stop decision (Phase 4), and the interpretation of the final table.

**The one rule for both of you:** no phase starts until the previous gate printed PASS. The temptation will be to skip Phase 2 ("it's obviously fine"). Don't. Phase 2 is ten minutes and it's the difference between a wasted week and a working run.

---

## 7. Expected outcomes, stated honestly

- **Likely:** bridged generation works, throughput beats plain Dream at long context, quality lands *below* Dream-alone by a small-to-moderate margin (the bridge is under-trained relative to NVIDIA's budget). This is a success — it's the measured tradeoff.
- **Possible:** quality ≈ Dream-alone with a real throughput win. Genuinely nice result.
- **Possible:** throughput gain is small because the block-parallel step is compute-bound on Arc and the cache savings are offset. Also a real finding — it's exactly the bandwidth-vs-compute crossover question, answered for this card.
- **Failure modes to name up front:** XPU friction eats the schedule (mitigated by Phase 0); RAM-bound loading (mitigated by the RAM decision); bridge doesn't learn (caught in Phase 3); LoRA damages Dream (caught in Phase 3 by generation degradation).

**What's novel regardless of outcome:** a lineage-matched (Dream + Qwen2.5) LoRA-bridge TwoTower has not been published; nor has any diffusion-LM throughput characterization on Intel Arc. Whatever the numbers, the table in Phase 5 is data nobody has.

---

## 8. Reading list (do this before Phase 0)
- TwoTower paper: https://arxiv.org/abs/2606.26493 — read §2–3 (architecture, cross-attention, block decoding).
- Dream-7B paper: https://arxiv.org/abs/2508.15487 — read the decoding section; you're plugging into its loop.
- Dream's HF modeling file — the code you're actually modifying. Read the generation function end to end.
- Kuleshov group, "How to Build a Diffusion Language Model" — the clearest tutorial on masked diffusion + block decoding.
