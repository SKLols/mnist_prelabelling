# MNIST Pre-labelling Pipeline

## Problem Statement
Automatic pre-labelling of the MNIST training set (60,000 images) using only
the test split (10,000 images) as a labeled seed set, via embedding-based
clustering and unsupervised label correction.

## Results
| | Accuracy |
|---|---|
| Seed classifier (trained on 10k seed set) | 97.35% on pool |
| Initial pre-labelling (clustering + Hungarian assignment) | 96.35% |
| After unsupervised label correction | 97.66% |
| After iterative correction (peak at iteration 4) | 98.17% |

## Environment
- OS: Ubuntu 22.04 (WSL2)
- Python: 3.11.15
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- CUDA Driver: 13.1 (PyTorch built against CUDA 12.6)
- PyTorch: 2.12.1+cu126

## Setup

### Create conda environment
```bash
conda create -n mnist_prelabelling python=3.11 -y
conda activate mnist_prelabelling
```

### Install PyTorch (CUDA 12.6)
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Install the package (editable mode)
```bash
pip install -e .
```

### Install remaining dependencies
```bash
pip install scikit-learn umap-learn matplotlib scipy
```

### Download MNIST data
The pipeline auto-downloads MNIST via `torchvision` on first run. Alternatively,
download the four IDX files manually from the
[CVDF MNIST mirror](https://github.com/cvdfoundation/mnist) and place them in
`data/MNIST/raw/`.

## Pipeline

### Verification runs (optional, not part of the official pipeline)
```bash
# Verify CustomCNN against published benchmarks (conventional MNIST split)
python src/mnist_prelabelling/training/train.py

# Verify seed classifier generalizes to the pool (debug check using pool groundtruth)
python src/mnist_prelabelling/training/train_seed_classifier.py
```

### Core pipeline (run in this order)
```bash
# Train seed classifier on 10k labeled set, extract 64-dim embeddings for 60k pool
python src/mnist_prelabelling/training/generate_pool_embeddings.py

# Cluster pool embeddings into 10 groups (KMeans on 64-dim space)
python src/mnist_prelabelling/clustering/cluster_embeddings.py

# Visualize embeddings in 2D via UMAP (outputs/embeddings_2d_plot.png)
python src/mnist_prelabelling/clustering/visualize_embeddings.py

# Assign digit labels to clusters via Hungarian algorithm
python src/mnist_prelabelling/clustering/assign_labels.py

# Retrain on self-generated pool labels, track confidence trajectory
python src/mnist_prelabelling/training/retrain_on_pool_labels.py

# Detect likely label errors using confidence + distance signals
python src/mnist_prelabelling/clustering/detect_label_errors.py

# Correct flagged labels via neighbor-majority reassignment
python src/mnist_prelabelling/clustering/correct_labels.py

# Report pre-labelling accuracy before and after correction
python src/mnist_prelabelling/evaluation/evaluate_prelabelling.py
```

### Offline analysis (uses ground truth for strategy selection only)
```bash
# Compare detection strategy precision/recall
python src/mnist_prelabelling/analysis/evaluate_detection_strategies.py

# Measure net correction impact per strategy (fixed vs broken labels)
python src/mnist_prelabelling/analysis/evaluate_correction_strategies.py

# Test iterative self-training (peaked at 98.17% after 4 iterations)
python src/mnist_prelabelling/analysis/iterative_correction_experiment.py
```

## Project Structure
```
src/mnist_prelabelling/
├── models/          # CNN architecture (CustomCNN with forward() and embed())
├── training/        # Seed classifier training, pool embedding generation, retraining
├── embeddings/      # Embedding extraction utility
├── clustering/      # UMAP visualization, KMeans clustering, label assignment, correction
├── evaluation/      # Final accuracy reporting (step 8)
├── analysis/        # Offline strategy-selection experiments (use GT, not pipeline)
└── utils/           # Run logging utility
```