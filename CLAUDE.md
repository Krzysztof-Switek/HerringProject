# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch-based computer vision system for classifying herring otolith images by population and predicting fish age. Supports both single-task classification and multitask (classification + age regression) training.

## Commands

### Setup
```bash
# Install dependencies into the existing virtual environment
.venv/Scripts/pip install -r requirements.txt
```

### Training
```bash
# Run the full training pipeline
.venv/Scripts/python src/train.py
```

### Tests
```bash
# Run all tests
.venv/Scripts/python -m pytest tests/

# Run a single test file
.venv/Scripts/python -m pytest tests/losses_test.py

# Run a single test by name
.venv/Scripts/python -m pytest tests/losses_test.py::test_loss_computation
```

Integration tests (`losses_integration_test.py`, `population_integrity_test.py`) require the real data and metadata Excel file to be present.

## Architecture

### Training Pipeline

`src/train.py` instantiates `Trainer` (engine/trainer.py), which delegates entirely to `run_training_loop` (engine/trainer_setup.py). The loop iterates over every loss function listed in `training.loss_type` in the config, training a fresh model per loss. After training, it runs full-dataset prediction and generates a PDF report.

```
src/train.py
  -> Trainer.__init__  (loads config, PathManager, PopulationMapper, HerringDataset)
  -> Trainer.train()
       -> run_training_loop()   [trainer_setup.py]
            for each loss_type:
              -> train_epoch() / validate()  [train_loop.py]
              -> save_best_model()           [trainer_logger.py]
              -> run_full_dataset_prediction()
              -> TrainingPredictionReport.run()
```

### Configuration

All settings live in `src/config/config.yaml` (OmegaConf). Key switches:

- `multitask_model.use: true/false` — switches between `HerringModel` (classification only) and `MultiTaskHerringModel` (classification + age regression)
- `training.loss_type` — list of loss function names; training runs once per entry
- `training.stop_after_one_epoch: true` — short-circuits training for pipeline testing
- `data.active_populations` — list of population numbers (biological IDs from Excel, e.g. `[1, 2]`) to include

### Population Mapping

Population numbers in the metadata Excel file (e.g., `1`, `2`) are NOT the same as model class indices. `PopulationMapper` (utils/population_mapper.py) handles bidirectional conversion: `to_idx(pop)` and `to_pop(idx)`. All metrics, confusion matrices, and CSV outputs use biological population numbers, not 0-based indices.

### Models

`src/models/model.py::build_model(cfg)` dispatches based on `cfg.multitask_model.use`:
- `HerringModel` — replaces the backbone's final classifier with a dropout + linear head sized to `len(active_populations)`
- `MultiTaskHerringModel` — same backbone, but replaces classifier with identity and adds two heads: `classifier_head` (population) and `age_regression_head` (age as float)

Supported backbones: `resnet50`, `convnext_large`, `efficientnet_v2_l`, `regnet_y_32gf` (and `vit_h_14` is commented out).

### Loss Functions

`LossFactory` (engine/loss_utills.py) instantiates one of 10 loss classes by name. For multitask mode, the chosen classification loss is wrapped in `MultiTaskLossWrapper` alongside `MSELoss` for age regression. `MultiTaskLossWrapper` supports four loss-weighting strategies: `none`, `static`, `uncertainty` (learnable `log_vars` parameter), and `gradnorm` (learnable `weights` parameter). The learnable parameters of the wrapper must be included in the optimizer — this is handled in `trainer_setup.py`.

### Data

`HerringDataset` (data_loader/dataset.py) reads from an Excel metadata file with columns `FileName`, `Populacja`, `Wiek`, `set`. The data directory must have subdirectories named by population number:

```
data/
  train/
    1/   <- images for population 1
    2/   <- images for population 2
  val/
    1/
    2/
```

Training samples are wrapped in `AugmentWrapper`, which applies probabilistic strong augmentation to undersized (population, age) strata to balance the dataset. Validation uses `HerringValDataset` (no augmentation).

### Outputs

Per training run (one model + one loss), outputs are saved under `results/logs/<model>_<loss>_<mode>_<timestamp>/`:
- `*_training_metrics.csv` — per-epoch metrics
- `*_predictions.xlsx` — full-dataset predictions after training
- `augmentation_summary_*.csv` — augmentation counts per (population, age) group
- `*_Training_Report.pdf` — combined training + prediction report

Best model checkpoints go to `checkpoints/<run_name>/`.

### Path Resolution

`PathManager` (utils/path_manager.py) resolves all paths relative to the project root. Relative paths in the config are resolved against project root; absolute paths are used as-is. The config path is always `<project_root>/src/config/config.yaml`.