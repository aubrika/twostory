# TwoStory

A homelab TwoTower: **Dream-7B** (masked-diffusion denoiser) + frozen **Qwen2.5-7B** (autoregressive context
tower), joined by a zero-initialised cross-attention bridge and LoRA adapters trained on ~100-300M tokens,
on one Intel Arc Pro B70 (32 GB). Full protocol, gates, and rationale: [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Layout

| Path | What |
|---|---|
| `docs/PROTOCOL.md` | The experimental protocol. Every phase has a go/no-go gate. |
| `docs/PHASE{0,1,2,3}_REPORT.md` | Per-phase results and decisions. |
| `phase0/` … `phase3/` | One standalone script per checklist item. Each prints `[GATE] ...: PASS/FAIL` and a verdict; `run_all.sh` runs a phase in order. |
| `twostory/` | `util.py` (loaders, memory, gates, provenance, telemetry), `prompts.py` (frozen prompts), `dream/` (vendored Dream + bridge hooks), `bridge.py`, `context_tower.py`, `model.py` (`TwoStory`: full/block generation, training loss). |
| `reference/` | Unmodified configs, tokenizer files and Dream's remote modeling code, for reading and diffing. |
| `logs/phase0/` | JSON reports and logs from each check. `baseline_A_dream_16bit.json` is the Phase 1 regression target. |
| `data/` | (git-ignored) FineWeb-Edu parquet shards and the pre-tokenized `tokens/{train,heldout}.bin`. |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/xpu
uv pip install --python .venv/bin/python "transformers==4.57.6" peft bitsandbytes accelerate datasets pyarrow numpy psutil safetensors
uv pip install --python .venv/bin/python -e .
```

bitsandbytes' 8-bit optimizer needs Triton-XPU, which needs the Level Zero headers in user space: see the
"Environment fix" section of `docs/PHASE2_REPORT.md`.

Weights live in the default HF cache (`hf download Dream-org/Dream-v0-Base-7B`, `hf download Qwen/Qwen2.5-7B`);
revisions are pinned in `twostory/util.py`. Data: two `sample/10BT` shards of `HuggingFaceFW/fineweb-edu` under
`data/fineweb-edu/`.

## Running a phase

```bash
phase0/run_all.sh            # everything, in order, stops at the first FAIL
phase1/run_all.sh            # regression (token-identical to Baseline A) + gradient flow + block-mode smoke
.venv/bin/python phase2/20_overfit.py   # overfit one batch; peak VRAM with optimizer state
phase3/watchdog.sh                      # pilot under a progress watchdog (auto-resume), then throughput
.venv/bin/python phase3/32_reference_loss.py   # T0 / T / D masked CE on identical blocks + steps sweep
.venv/bin/python phase0/04_baseline_dream.py --bits 8   # or any single check
```

## Memory plan

Denoiser (Dream) in **bf16**, context tower (Qwen) in **int8** via bitsandbytes' XPU wheel: 22.3 GiB resident;
26.9 GiB peak in training with the full-size bridge (406.6M trainable) and 8-bit AdamW.
int8 on XPU is 4-8x slower than bf16 per linear, so only the tower that runs once per block is quantised.
See the addendum in `docs/PHASE0_REPORT.md`.

## Rules

- No phase starts until the previous gate printed PASS.
- `twostory/prompts.py` and the recorded baselines are frozen after Phase 0.
- The held-out split (`data/tokens/heldout.bin`, hash in `data/tokens/meta.json`) is never trained on.
