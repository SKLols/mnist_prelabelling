# MNIST Pre-labelling Pipeline

## Problem Statement
Automatic pre-labelling of the MNIST training set (60,000 images) using only
the test split (10,000 images) as a labeled seed set, via embedding-based
clustering.

## Environment
- OS: Ubuntu 22.04 (WSL2)
- Python: 3.11.15
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- CUDA Driver: 13.1 (PyTorch built against CUDA 12.6)
- PyTorch: 2.12.1+cu126

## Setup

### 1. Create conda environment
```bash
conda create -n mnist_prelabelling python=3.11 -y
conda activate mnist_prelabelling
```

### 2. Install PyTorch (CUDA 12.6)
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install the package (editable mode)
```bash
pip install -e .
```

### 4. Download MNIST data
Download the four IDX files from the [CVDF MNIST mirror](https://github.com/cvdfoundation/mnist)
and place them in `data/MNIST/raw/`. Alternatively, the training scripts will
auto-download via `torchvision` if the files are not already present.

### 5. Install remaining dependencies (as needed per pipeline stage)
```bash
pip install scikit-learn umap-learn matplotlib
```

## Pipeline Usage

### Sanity check — verify the model architecture (optional)
Trains on the full 60k training split, evaluates on the 10k test split,
in the conventional MNIST direction. Used to verify the CustomCNN
implementation against published benchmarks.
```bash
python src/mnist_prelabelling/training/train.py
```

### Train seed classifier + debug check against pool groundtruth (optional)
Trains on the 10k seed set, then evaluates against the 60k pool's groundtruth
labels as a debug-only check (not part of the official unsupervised pipeline).
```bash
python src/mnist_prelabelling/training/train_seed_classifier.py
```

### Generate pool embeddings
Trains the seed classifier and extracts 64-dim embeddings for all 60,000
pool images. Saves output to `outputs/pool_embeddings.npy`.
```bash
python src/mnist_prelabelling/training/generate_pool_embeddings.py
```

### Visualize embeddings (2D projection)
Projects the 64-dim embeddings to 2D via UMAP and saves a scatter plot.
```bash
python src/mnist_prelabelling/clustering/visualize_embeddings.py
```
Output: `outputs/embeddings_2d_plot.png`

### Cluster embeddings
Runs KMeans (k=10) on the original 64-dim embeddings. Saves cluster
assignments and per-point distances to centroid.
```bash
python src/mnist_prelabelling/clustering/cluster_embeddings.py
```
Output: `outputs/pool_cluster_ids.npy`, `outputs/pool_distances_to_centroid.npy`