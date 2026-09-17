# TwoStory

A homelab TwoTower: **Dream-7B** (masked-diffusion denoiser) + frozen **Qwen2.5-7B** (autoregressive context
tower), joined by a zero-initialised cross-attention bridge and LoRA adapters trained on ~100-300M tokens,
on one Intel Arc Pro B70 (32 GB). Full protocol, gates, and rationale: [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Layout

| Path | What |
|---|---|
| `docs/PROTOCOL.md` | The experimental protocol. Every phase has a go/no-go gate. |
| `docs/PHASE0_REPORT.md` | Phase 0 results and decisions. |
| `phase0/` | One standalone script per Phase 0 checklist item. Each prints `[GATE] ...: PASS/FAIL` and a verdict. |
| `twostory/` | Shared package: `util.py` (loaders, memory tracking, gates), `prompts.py` (fixed prompts). |
| `reference/` | Unmodified configs, tokenizer files and Dream's remote modeling code, for reading and diffing. |
| `logs/phase0/` | JSON reports and logs from each check. `baseline_A_dream_8bit.json` is the Phase 1 regression target. |
| `data/` | (git-ignored) FineWeb-Edu parquet shards and the pre-tokenized `tokens/{train,heldout}.bin`. |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/xpu
uv pip install --python .venv/bin/python "transformers==4.57.6" peft bitsandbytes accelerate datasets pyarrow numpy psutil safetensors
uv pip install --python .venv/bin/python -e .
```

Weights live in the default HF cache (`hf download Dream-org/Dream-v0-Base-7B`, `hf download Qwen/Qwen2.5-7B`);
revisions are pinned in `twostory/util.py`. Data: two `sample/10BT` shards of `HuggingFaceFW/fineweb-edu` under
`data/fineweb-edu/`.

## Running Phase 0

```bash
phase0/run_all.sh            # everything, in order, stops at the first FAIL
.venv/bin/python phase0/04_baseline_dream.py --bits 8   # or any single check
```

## Rules

- No phase starts until the previous gate printed PASS.
- `twostory/prompts.py` and the recorded baselines are frozen after Phase 0.
- The held-out split (`data/tokens/heldout.bin`, hash in `data/tokens/meta.json`) is never trained on.
