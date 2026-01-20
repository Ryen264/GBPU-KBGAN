# GBPU-KBGAN

Knowledge Graph Adversarial Training with KBGAN Framework

## Overview

This project implements KBGAN (Knowledge Base GAN), an adversarial learning framework for knowledge graph embedding. It trains discriminator models (TransE, TransD) and generator models (DistMult, ComplEx) in an adversarial setup to improve link prediction and triple classification tasks.

---

## Project Structure & File Descriptions

### Core Training Files

- **`main.py`** - Main entry point for training and evaluation
  - Supports multiple modes: `full-train`, `gan-train`, `test-only`
  - Orchestrates the complete pipeline: data loading, pretraining, adversarial training, evaluation
  - Command-line interface with config override support

- **`kbgan.py`** - KBGAN adversarial training framework
  - `Component` class: Wraps individual models (discriminator/generator)
  - `KBGAN` class: Implements adversarial training loop, reward computation, policy gradient updates
  - Handles model loading, saving, and evaluation

- **`models.py`** - Knowledge graph embedding model implementations
  - `TransE`: Translation-based model (h + r ≈ t)
  - `TransD`: TransE with projection matrices for entities and relations
  - `DistMult`: Bilinear model with diagonal weight matrix
  - `ComplEx`: Complex-valued embeddings for asymmetric relations
  - Each model class includes training and evaluation logic

- **`base_model.py`** - Base classes for all models
  - `BaseModule`: PyTorch nn.Module base with common methods (forward, dist, score, prob_logit, constraint)
  - `BaseModel`: Model wrapper with training loop, optimizer setup, device management, evaluation metrics

### Data Processing Files

- **`data_loader.py`** - Data loading and preprocessing utilities
  - Reads triple data from text files (head, relation, tail format)
  - `index_entity_relation()`: Maps entities/relations to integer indices
  - `graph_size()`: Returns number of entities and relations
  - `read_data()`: Loads train/valid/test splits

- **`datasets.py`** - Data corruption and batching utilities
  - `BernCorrupter`: Bernoulli-based negative sampling for single models
  - `BernCorrupterMulti`: Multi-model negative sampling for adversarial training
  - `sparse_heads_tails()`: Creates sparse tensor dictionaries for filtered evaluation
  - `batch_by_num()`, `batch_by_size()`: Batch data generators
  - `inplace_shuffle()`: Efficient in-place shuffling

### Evaluation & Utilities

- **`metrics.py`** - Evaluation metrics for link prediction and classification
  - `ranking_metrics()`: MR, MRR, Hits@K for link prediction tasks
  - `classification_metrics()`: Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC

- **`config.py`** - Configuration management and device selection
  - `config()`: Loads YAML configuration files
  - `overwrite_config_with_args()`: Command-line config overrides with auto type conversion
  - `select_gpu()`: Auto-selects best available GPU based on memory usage
  - `logger_init()`: Configures logging to console and file

### Configuration Files

- **`config/`** - YAML configuration files per dataset
  - `config_wn18rr.yaml`: WordNet18RR dataset config
  - `config_wn18.yaml`: WordNet18 dataset config  
  - `config_fb15k237.yaml`: Freebase FB15k-237 dataset config
  - Each config specifies model hyperparameters, training settings, paths

### Data Directories

- **`data/`** - Knowledge graph datasets
  - `wn18/`, `wn18rr/`, `fb15k237/`: Train/valid/test triple files
  - Format: `head\trelation\ttail` (tab-separated)

### Output Directories

- **`output/`** - Saved model checkpoints and training artifacts
  - `<dataset>/<task>/models/`: Pretrained discriminator/generator models
  - `<dataset>/<task>/kbgan/`: KBGAN adversarial training checkpoints

- **`logs/`** - Training logs and process management
  - `training_YYYYMMDD_HHMMSS.log`: Timestamped training logs
  - `training.pid`: Process ID for background runs

### Process Management Scripts

- **`run_process.sh`** - Start training in background with nohup
  - Usage: `./run_process.sh [mode] [extra_args]`
  - Example: `./run_process.sh full-train "--override KBGAN.n_epoch=1000"`
  - Creates timestamped log files and saves PID

- **`check_process.sh`** - Monitor running training process
  - Usage: `./check_process.sh [lines]`
  - Shows process status, runtime, and recent log output
  - Auto-cleans stale PID files

- **`stop_process.sh`** - Stop running training process
  - Usage: `./stop_process.sh [--force]`
  - Graceful shutdown (SIGTERM) or forced kill (SIGKILL)
  - Fallback to finding processes by command name

### Other Files

- **`requirements.txt`** - Python package dependencies
- **`__pycache__/`** - Python bytecode cache (auto-generated)

---

## Setup Instructions

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Option A: Install from requirements file
pip install -r requirements.txt

# Option B: Install manually
pip install torch numpy pyyaml scikit-learn

# For GPU support (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Required packages:**
- `torch` - PyTorch for deep learning
- `numpy` - Numerical computations
- `pyyaml` - YAML config file parsing
- `scikit-learn` - Evaluation metrics (PR-AUC, ROC-AUC)

### Step 3: Prepare Data

Ensure your dataset is in the `data/` directory:
```
data/
  wn18rr/
    train.txt
    valid.txt
    test.txt
```

Format: Each line should be `head\trelation\ttail` (tab-separated)

---

## Running the Code

### Option 1: Manual Foreground Execution

Run training directly in the current terminal:

```bash
# Activate virtual environment first
source .venv/bin/activate

# Full training pipeline (pretrain + adversarial + evaluation)
python main.py mode=full-train

# Only adversarial training (requires pretrained models)
python main.py mode=gan-train

# Only evaluation (requires trained model)
python main.py mode=test-only
```

**With config overrides:**
```bash
python main.py mode=full-train \
  --override TransE.n_epoch=500 \
  --override DistMult.n_epoch=500 \
  --override KBGAN.n_epoch=3000 \
  --override KBGAN.temperature=0.5
```

**Using different dataset:**
```bash
python main.py mode=full-train --config-file ./config/config_fb15k237.yaml
```

### Option 2: Background Execution with NOHUP

For long-running training sessions, use the provided shell scripts:

#### Start Training in Background

```bash
# Make scripts executable (first time only)
chmod +x run_process.sh check_process.sh stop_process.sh

# Start training with default mode (full-train)
./run_process.sh

# Start with specific mode
./run_process.sh gan-train

# Start with config overrides
./run_process.sh full-train "--override KBGAN.n_epoch=1000"
```

**What happens:**
- Training runs in background via `nohup`
- Timestamped log file created: `logs/training_YYYYMMDD_HHMMSS.log`
- Process ID saved to: `logs/training.pid`
- Terminal can be closed without stopping training

#### Monitor Training Progress

```bash
# Check status and show last 30 lines of log
./check_process.sh

# Show last 50 lines
./check_process.sh 50

# Follow live log output
tail -f logs/training_*.log  # Use the most recent log file
```

#### Stop Training

```bash
# Graceful shutdown (recommended)
./stop_process.sh

# Force kill (if graceful fails)
./stop_process.sh --force
```

---

## Configuration

Configuration files are located in `config/` directory. Key parameters:

**Model Hyperparameters:**
- `dim`: Embedding dimension
- `margin`: Margin for ranking loss
- `lr`: Learning rate
- `n_epoch`: Number of training epochs
- `n_batch`: Batch size

**KBGAN Settings:**
- `temperature`: Softmax temperature for reward distribution
- `n_epoch`: Adversarial training epochs
- `n_sample`: Number of negative samples
- `lr_gan`: Learning rate for generator

**Example override:**
```bash
python main.py mode=full-train \
  --override TransE.dim=100 \
  --override KBGAN.temperature=0.3
```

---

## Training Modes

- **`mode=full-train`**: Complete pipeline
  1. Pretrain discriminator (TransE/TransD)
  2. Pretrain generator (DistMult/ComplEx)
  3. Adversarial KBGAN training
  4. Final evaluation

- **`mode=gan-train`**: Only adversarial training
  - Requires pretrained models in `output/<dataset>/<task>/models/`

- **`mode=test-only`**: Only evaluation
  - Loads trained discriminator and evaluates performance

---

## Output & Logs

**Saved Models:**
- Pretrained models: `output/<dataset>/<task>/models/`
  - `<ModelName>_<timestamp>.pt`
- KBGAN checkpoints: `output/<dataset>/<task>/kbgan/`
  - `discriminator_epoch<N>.pt`
  - `generator_epoch<N>.pt`

**Log Files:**
- Foreground runs: Console output
- Background runs: `logs/training_YYYYMMDD_HHMMSS.log`
- Logs include: Loss values, evaluation metrics, timing info

---

## Troubleshooting

**Issue:** `scikit-learn not available. PR AUC and ROC AUC set to 0.0`
- **Fix:** `pip install scikit-learn`

**Issue:** `Could not parse nvidia-smi output. Defaulting to GPU 0.`
- **Impact:** Informational warning; GPU 0 will be used
- **Fix:** Code works fine; parser expects specific nvidia-smi format

**Issue:** `Virtual environment not found` when running scripts
- **Fix:** Create venv first: `python3 -m venv .venv`

**Issue:** Process still running after `./stop_process.sh`
- **Fix:** Use force kill: `./stop_process.sh --force`

---

## Citation

If you use this code, please cite the original KBGAN paper:
```
@inproceedings{cai2017kbgan,
  title={KBGAN: Adversarial Learning for Knowledge Graph Embeddings},
  author={Cai, Liwei and Wang, William Yang},
  booktitle={NAACL-HLT},
  year={2018}
}
```

---

## License

This project is for research purposes. Check individual dependencies for their licenses.
