# MNIST Pre-labelling Pipeline

## Problem Statement
Automatic pre-labelling of the MNIST dataset

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