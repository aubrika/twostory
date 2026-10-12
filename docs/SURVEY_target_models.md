# Survey: Qwen-lineage diffusion LMs on the Hugging Face Hub (queried 2026-09-20)

Sizes, licenses, dimensions and download counts below were read from the Hub API and each repo's `config.json` on the date above.
The "native cache" and mechanism notes are from memory of the papers and were NOT re-verified against them.

## Reference towers (frozen AR candidates)
| Model | hidden | layers | heads / KV heads | head dim |
|---|---|---|---|---|
| Qwen2.5-1.5B | 1536 | 28 | 12 / 2 | 128 |
| Qwen2.5-7B, Qwen2.5-Coder-7B | 3584 | 28 | 28 / 4 | 128 |
| Qwen3-0.6B | 1024 | 28 | 16 / 8 | 128 |
| Qwen3-1.7B | 2048 | 28 | 16 / 8 | 128 |
| Qwen3-4B | 2560 | 36 | 32 / 8 | 128 |
| Qwen3-8B | 4096 | 36 | 32 / 8 | 128 |

Every Qwen3 size has 8 KV heads of dim 128, so a small Qwen3-lineage denoiser can cross-attend to a larger Qwen3's KV cache with no K/V
projection; only the query side and a layer mapping (28 vs 36) are new.

## Diffusion LMs
| Model | Lineage / exact tower match | Params | License | Native block KV cache | Downloads |
|---|---|---|---|---|---|
| Dream-org/Dream-v0-{Base,Instruct}-7B | Qwen2.5-7B (3584/28) | 7.6B | Apache-2.0 | no (full re-forward per step) | 9k / 100k |
| Dream-org/Dream-Coder-v0-{Base,Instruct}-7B | Qwen2.5-Coder-7B | 7.6B | Apache-2.0 | no | 1-2k |
| apple/DiffuCoder-7B-{Base,Instruct,cpGRPO} | Qwen2.5-Coder-7B, Dream architecture | 7.6B | apple-amlr (research) | no | ~1k |
| Efficient-Large-Model/Fast_dLLM_v2_7B | Qwen2.5-7B-Instruct, block 32 | 7.6B | Apache-2.0 | yes | 13k |
| Efficient-Large-Model/Fast_dLLM_v2_1.5B | Qwen2.5-1.5B-Instruct, block 32 | 1.5B | Apache-2.0 | yes | 4k |
| tencent/WeDLM-7B-{Base,Instruct} | Qwen2.5-7B dims | 7.6B | Apache-2.0 | yes (causal attention) | ~200 |
| tencent/WeDLM-8B-{Base,Instruct} | Qwen3-8B dims | 8.2B | Apache-2.0 | yes | ~300 |
| JetLM/SDAR-{1.7B,4B,8B}-Chat (+ b32/b64 variants) | Qwen3-1.7B/4B/8B dims | 2.0 / 4.4 / 8.2B | Apache-2.0 | yes | 24k / 5k / 17k |
| JetLM/SDAR-30B-A3B-Chat | Qwen3-30B-A3B | 30B MoE | Apache-2.0 | yes | 1k (too big for 32 GB bf16) |
| Gen-Verse/TraDo-{4B,8B}-Instruct, 8B-Thinking | SDAR + RL | 4.4 / 8.2B | MIT | yes | <1k |
| nvidia/Efficient-DLM-{4B,8B} | Qwen3-4B/8B dims, block 32 | 4.4 / 8.2B | "other" (NVIDIA) | yes | ~500 |
| TiDAR-LM/TiDAR-8b (+1.7b research ckpts) | Qwen3-8B dims; diffusion draft + AR verify in one pass | 8.2B | none stated | yes | 15 |
| dllm-hub/Qwen3-0.6B-diffusion-{mdlm,bd3lm}-v0.1 | Qwen3-0.6B | 0.75B | Apache-2.0 | bd3lm: yes | 8k / 1.5k |
| dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-* | Qwen2.5-Coder-0.5B | 0.5B | Apache-2.0 | bd3lm: yes | <100 |
| radicalnumerics/RND1-Base-0910 | Qwen3-30B-A3B | 30.5B MoE | Apache-2.0 | no | 1k (too big) |
| nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base | the TwoTower reference (Nemotron-H hybrid, not Qwen) | 63B both towers | "other" | yes | 548 (too big) |
| inclusionAI/LLaDA2.0/2.1-mini | not Qwen (own MoE) | 16B MoE | Apache-2.0 | yes | 120-230k |

## Speculative-decoding prior art on the Hub (the bar for "lossless speed")
| Model | What it is | Params | License | Downloads |
|---|---|---|---|---|
| z-lab/Qwen3-{4B,8B}-DFlash-b16 | 5-layer block-diffusion DRAFTER (block 16) conditioned on the target Qwen3; lossless speculative decoding | 0.54 / 1.05B | MIT | — |
| z-lab/Qwen3.6-27B-DFlash, Qwen3.6-35B-A3B-DFlash, Qwen3.8-27B-DFlash2 (+GGUF for llama.cpp) | same, for current Qwen models | — | — | 110k-700k |
| RedHatAI/Qwen3-8B-speculator.eagle3, Tengyunw/qwen3_8b_eagle3, AngelSlim/Qwen3-8B_eagle3 | EAGLE-3 autoregressive drafters for Qwen3-8B | — | — | 3k-17k |
| thoughtworks / ruipeterpan Qwen2.5-7B-Instruct EAGLE3 | EAGLE-3 drafters for Qwen2.5-7B-Instruct | — | — | <1k |
