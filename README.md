# MNIST Pre-labelling Pipeline

## Problem Statement

Automatic pre-labelling of the MNIST training set (60,000 images) using only
the test split (10,000 images) as a labeled seed set, via embedding-based
clustering and iterative unsupervised label correction.

The pipeline treats the 10k test split as the only labeled data available,
clusters embeddings learned from it to assign initial labels to the 60k pool,
then iteratively refines those labels using unsupervised error detection —
never accessing the pool's ground-truth labels during the labeling process.

## Results

Results vary slightly across runs due to GPU non-determinism in weight
initialization. Representative numbers from a verified full pipeline run:

| Stage | Accuracy |
|---|---|
| Initial pre-labelling (clustering + Hungarian assignment) | 96.21% |
| After single-pass correction (iteration 0) | 97.66% |
| After iteration 1 | 98.02% |
| After iteration 2 | 98.11% |
| After iteration 3 | 98.14% |
| After iteration 4 (final) | 98.10% |
| **Total improvement** | **+1.89 percentage points** |

Per-digit accuracy after final correction:

| Digit | Accuracy |
|---|---|
| 0 | 99.16% |
| 1 | 98.87% |
| 2 | 97.82% |
| 3 | 97.39% |
| 4 | 97.93% |
| 5 | 98.10% |
| 6 | 99.32% |
| 7 | 98.24% |
| 8 | 96.17% |
| 9 | 97.87% |

Digit 8 remains the hardest class (~96%) due to visual similarity with 3, 5,
and 9 in stroke structure. Accuracy peaks around iteration 3, then plateaus —
consistent with the confirmation bias effect documented in the noisy-label
literature.

## Environment

- OS: Ubuntu 22.04 (WSL2)
- Python: 3.11.15
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- CUDA Driver: 13.1 (PyTorch built against CUDA 12.6)
- PyTorch: 2.12.1+cu126

## Setup (local development with GPU)

### Create conda environment
```bash
conda create -n mnist_prelabelling python=3.11 -y
conda activate mnist_prelabelling
```

### Install PyTorch with CUDA support
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Install the package and remaining dependencies
```bash
pip install -e .
pip install scikit-learn umap-learn matplotlib scipy
```

### MNIST data
Downloaded automatically on first run via torchvision. Alternatively, download
the four IDX files from the [CVDF MNIST mirror](https://github.com/cvdfoundation/mnist)
and place them in `data/MNIST/raw/`.

## Running the pipeline

### Option 1 — Docker (recommended, CPU, no GPU required)

```bash
docker-compose up --build
```

Or without compose:
```bash
docker build -t mnist-prelabelling .
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data:/app/data \
  mnist-prelabelling
```

Each run creates a timestamped folder under `outputs/run_YYYYMMDD_HHMMSS/`
containing all generated artifacts.

> **Note:** The Docker image uses CPU-only PyTorch for portability — no CUDA
> or NVIDIA drivers required on the host. For GPU-accelerated runs on larger
> datasets, switch the base image to `nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04`,
> install CUDA-enabled PyTorch, and run with `docker run --gpus all ...`.
> This is a one-line Dockerfile change.

### Option 2 — local conda environment (GPU-accelerated)

```bash
conda activate mnist_prelabelling
python src/mnist_prelabelling/pipeline.py
# or:
mnist-prelabelling
```

### Individual pipeline steps (for debugging)
```bash
python src/mnist_prelabelling/training/generate_pool_embeddings.py
python src/mnist_prelabelling/clustering/cluster_embeddings.py
python src/mnist_prelabelling/clustering/visualize_embeddings.py
python src/mnist_prelabelling/clustering/assign_labels.py
python src/mnist_prelabelling/training/retrain_on_pool_labels.py
python src/mnist_prelabelling/clustering/detect_label_errors.py
python src/mnist_prelabelling/clustering/correct_labels.py
python src/mnist_prelabelling/evaluation/evaluate_prelabelling.py
```

### Optional verification scripts
```bash
# Verify CustomCNN against published benchmarks (conventional MNIST split)
python src/mnist_prelabelling/training/train.py

# Verify seed classifier generalizes to the pool
python src/mnist_prelabelling/training/train_seed_classifier.py
```

### Offline analysis scripts (use ground truth for strategy selection only)
```bash
python src/mnist_prelabelling/analysis/evaluate_detection_strategies.py
python src/mnist_prelabelling/analysis/evaluate_correction_strategies.py
python src/mnist_prelabelling/analysis/iterative_correction_experiment.py
```

## Project Structure

```
src/mnist_prelabelling/
├── pipeline.py      # Main orchestrator — runs the full pipeline end-to-end
├── config.py        # Hyperparameters, paths, random seed
├── models/          # CNN architecture (CustomCNN: forward() + embed())
├── training/        # Seed classifier training, embedding generation, retraining
├── embeddings/      # Embedding extraction utility
├── clustering/      # UMAP visualization, KMeans clustering, label assignment,
│                    # error detection, iterative label correction
├── evaluation/      # Final accuracy reporting (before/after correction)
├── analysis/        # Offline strategy-selection experiments (use GT)
└── utils/           # Run logging utility
```