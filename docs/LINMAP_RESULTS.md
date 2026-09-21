# Linear K/V-map bridge — results (2026-09-20)

Script: `linmap/fit_and_eval.py --fit-seqs 400 --val-seqs 24 --eval-seqs 32` (10.5 min on the B70, zero gradient steps).
Raw: `logs/linmap/linmap_results_fit400.json`, maps in `logs/linmap/linmap_weights_fit400.pt`, figure `docs/figs/linmap_r2.png`.

**Design.** NVIDIA-TwoTower-style wiring: the context tower's per-layer K/V are concatenated into Dream's OWN self-attention (one softmax over
context + block); no separate cross-attention module. Denoiser input = `[BOS, anchor, 32-token block]`. The only "bridge" is a per-layer affine
map from Qwen's K/V space to Dream's, fitted in closed form (ridge regression) on pre-RoPE activations from 211k tokens of paired forwards.
Targets = Dream's context K/V with the block fully masked (the deployable condition).

## Stage A — how linearly predictable are Dream's context K/V from Qwen's? (held-out R², mean over 28 layers)
| Predictor | keys | values |
|---|---|---|
| identity (raw Qwen K/V) | −7.0 (≈0 in most layers; −90 at layers 0 and 27: scale drift) | 0.20 |
| linear map from Qwen K/V (512→512) | 0.63 | 0.55 |
| linear map from Qwen attention-input hidden (3584→512) | 0.67 | 0.59 |
Layer profile: 0.8–0.96 in layers 0–4, falling to 0.5–0.6 (keys) / 0.3–0.5 (values) through the middle, ~0.6 at the top.
Dream has drifted far from Qwen (identity is useless), but most of the drift is linear in the early layers. The ceiling in later layers is
expected: Dream's context representations are bidirectional, so they encode later tokens a causal Qwen cannot know at that position.

## Stage B — functional test: masked CE on the standard 32 held-out sequences, identical masks to every earlier number
| Context source fed into Dream's attention | CE | gap closed | argmax agreement with plain Dream |
|---|---|---|---|
| none (unconditional; matches the old T₀ 4.913) | 4.908 | 0% | 47.0% |
| identity: raw Qwen K/V | 6.395 | −103% | 23.1% |
| **linear K/V map (closed form)** | **4.051** | **59.5%** | 61.5% |
| **linear hidden→K/V map (closed form)** | **3.948** | **66.6%** | 64.2% |
| Phase 4 trained bridge (separate cross-attn, 86M tokens, ~34 GPU-h) | 3.771 | 79.0% | — |
| oracle: Dream's own context K/V, computed once with the block fully masked | 3.488 | 98.6% | 92.3% |
| plain Dream, context in its own input (recomputed here; Phase 3 value 3.465) | 3.468 | 100% | 100% |

## Reading
1. The concat wiring is essentially lossless (oracle 98.6%): "context processed once per block, block-only denoiser" reproduces Dream.
   The architecture is no longer the bottleneck; the quality of the context K/V is.
2. A closed-form linear map fitted on 0.2M tokens in minutes recovers 60–67% of the gap; 86M tokens of gradient training through a
   from-scratch cross-attention module recovered 79%. The linear map is a far better *starting point* than zero-init.
3. The oracle row is also a result in its own right: Dream can act as its own context tower (K/V cached per block rather than recomputed
   per denoising step) at ~no teacher-forced quality cost. That gives an S-fold cut in prompt cost without any second model, though not
   Qwen's prompt-length-independent causal cache.
