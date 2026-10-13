# Setup and reproduction

## What can be reproduced from the repository

New model measurements require Linux and Intel XPU support. Linear-map fitting
needs the two pretrained models and the token files, but no Phase 3 result JSON
or trained bridge checkpoint. The trained bridge and generation comparisons need
the training phases to be rerun, or the original checkpoints.

Later-stage raw logs and checkpoints are not included in the repository; they are
large and remain on the machine the experiments ran on.

## Environment

[The Phase 0 environment report](../logs/phase0/00_env.json) records:

| Component | Recorded version |
|---|---|
| PyTorch | 2.14.0+xpu |
| Transformers | 4.57.6 |
| PEFT | 0.21.0 |
| bitsandbytes | 0.50.2 |
| Accelerate | 1.15.0 |
| GPU | Intel Arc Pro B70 |

The original setup used Python 3.12. The report does not contain a complete lockfile
for all data and plotting dependencies, or a full driver/system package snapshot.
The commands below pin the recorded model-library versions; they are a setup
recipe, not a fully reconstructed environment.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "torch==2.14.0" --index-url https://download.pytorch.org/whl/xpu
uv pip install --python .venv/bin/python -e ".[xpu]"
```

The 8-bit optimizer also needs Triton-XPU and Level Zero headers. The
[Phase 2 environment notes](PHASE2_REPORT.md) describe the original user-space
installation. Run `bash phase0/run_all.sh` only after weights and data are prepared;
the scripts check actual XPU operations and report failures.

For offline Phase 5 aggregation with recovered raw logs, install only the
`analysis` extra. The aggregator does not import PyTorch or XPU helpers.

## Models and data

Loaders in [util.py](../twostory/util.py) pin model revisions:

```bash
hf download Dream-org/Dream-v0-Base-7B --revision 6572adb5535263e4d1a337b56942ba48b6dee2a9
hf download Qwen/Qwen2.5-7B --revision d149729398750b98c0af14eb82c78cfe92750796
hf download HuggingFaceFW/fineweb-edu --repo-type dataset --include "sample/10BT/000_00000.parquet" "sample/10BT/001_00000.parquet" --local-dir data/fineweb-edu
.venv/bin/python phase0/07_pretokenize.py
```

The tokenizer packs documents into 1,024-token sequences, separated by EOS.
Every fiftieth sequence goes into `heldout.bin`. This is a sequence-level split;
a document can span both splits. Repeated validation and possible document overlap
limit the interpretation as out-of-sample generalization.

The committed [token manifest](../data/tokens/meta.json) records the original
source filenames and token-file hashes. The dataset revision was not recorded.
Compare new token hashes with the committed manifest before describing a rerun
as using identical inputs:

```bash
sha256sum data/tokens/train.bin data/tokens/heldout.bin
git diff -- data/tokens/meta.json
```

The preprocessing script writes a new manifest. Keep that manifest with the new
run; do not silently substitute it for the original when comparing results.

## Experiment paths

The trained-bridge path is Phase 0 environment checks, Phase 1 regression and
gradients, Phase 2 overfit, Phase 3 pilot, then Phase 4 training.
Read the phase reports for configuration changes made during the recorded runs;
the original [protocol](PROTOCOL.md) contains superseded plans.

The independent linear-map path is:

```bash
.venv/bin/python -m linmap.fit_and_eval --fit-seqs 400 --val-seqs 24 --eval-seqs 32
```

The script measures its no-context and plain-Dream references in the same run.
It does not score the Phase 4 bridge. Validation uses held-out indices 0–31 and
fixed masks; the representation R² check uses indices 100 onward. These slices
must remain disjoint. The first slice was reused during model development.

Phase 4 start, status, and stop commands are separate operations, not a shell
pipeline. Long jobs were managed with shell watchdogs; the original machine also
used `systemd-run --user` to keep jobs independent of editor lifetime.

## Retaining a new run

Keep raw JSON/JSONL, model and data revisions, package versions, random seeds,
configuration, and checkpoint identifiers together. The existing provenance helper
records the source commit, dirty state, and available package information.
Use a release artifact for large checkpoints.
