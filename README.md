# GBPU-KBGAN

Cleaned and refactored Knowledge Graph adversarial training project.

## Overview
This repository trains and evaluates knowledge graph embedding models (TransE / TransD as discriminators, DistMult / ComplEx as generators) with an adversarial KBGAN setup. The `main.py` CLI unifies pretraining, adversarial training, and evaluation.

## Directory Structure (Post-Cleanup)
- `main.py` : Unified entrypoint.
- `train.py` : Pipeline logic exposed via `run_kbgan(mode, config_overrides)`.
- `pretrain.py` : Legacy single-model pretraining helper.
- `datasets.py` : Data corruption / batching utilities.
- `data_loader.py` : Loading triples and indexing entities/relations.
- `models.py` : Model definitions (TransE, TransD, DistMult, ComplEx).
- `base_model.py` : Shared evaluation metrics and utilities.
- `config/` : YAML configuration files per dataset.
- `data/` : Raw triples and optional evaluation sets.
- `output/<task>/models/` : Pretrained base models.
- `output/<task>/kbgan/` : Saved discriminator checkpoints during adversarial training.

Removed: notebook, shell scripts, utility script now obsolete.

## Quick Start
```bash
python3 main.py --mode full
```
This will:
1. Pretrain discriminator & generator.
2. Run adversarial KBGAN training.
3. Evaluate final discriminator on triple classification.

## Modes
- `--mode pretrain` : Only pretrain individual models.
- `--mode train` : Assume pretrained models exist; run adversarial training.
- `--mode evaluate` : Evaluate a pretrained model (uses logic in `train.py`'s `KBGAN.evaluate`).
- `--mode full` : Pretrain + train + evaluate.

## Configuration
Default config file: `config/config_wn18rr.yaml`. Override by passing `--config-file`.

Inline overrides (auto type-conversion) via repeated `--override` flags:
```bash
python3 main.py --mode full \
  --override TransE.n_epoch=500 \
  --override DistMult.n_epoch=500 \
  --override KBGAN.n_epoch=3000 \
  --override KBGAN.temperature=0.5
```
All override strings map to keys inside the loaded YAML (dot-separated path).

## GPU Selection
`select_gpu()` auto picks a free / least used GPU. If CUDA unavailable it falls back to CPU.

## Evaluation (Triple Classification)
`KBGAN.evaluate(model_type='DistMult')` loads the specified pretrained model from `output/<task>/models/` and computes accuracy, precision, recall, F1. A validation-driven threshold search is performed automatically.

## Saving & Artifacts
- Base models saved in `output/<task>/models/` using their configured `model_file` names.
- Adversarial checkpoints saved in `output/<task>/kbgan/` as `gan_<Dis>-dis_<Gen>-gen_<timestamp>.mdl`.

## Artifact Pruning Suggestion
You can keep only the latest checkpoint:
Latest (by timestamp) currently appears to be: `gan_TransE-dis_DistMult-gen_1117155145.mdl`.
Option: move older `.mdl` files into an archive or delete them:
```bash
mkdir -p output/wn18rr/kbgan/archive
mv output/wn18rr/kbgan/gan_TransE-dis_DistMult-gen_10*.mdl output/wn18rr/kbgan/archive/
```
(Adjust glob to exclude the latest.)

## Minimal Programmatic Use
```python
from train import run_kbgan
# full pipeline
run_kbgan(mode='full', config_overrides=["--KBGAN.n_epoch=2000"])
# evaluate only
run_kbgan(mode='evaluate')
```

## Troubleshooting
- Missing model file: ensure pretrain phase ran (`--mode pretrain` or `full`).
- CUDA OOM: lower embedding dims or batch sizes via overrides.
- Empty evaluation scores: check that validation/test label files exist when using triple classification.

## Next Ideas
- Add checkpoint selection logic (keep best instead of latest).
- Add a pruning script under `scripts/`.
- Integrate tensorboard or simple metrics CSV logging.

## License / Attribution
(Please add licensing details if required.)
