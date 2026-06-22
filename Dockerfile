# CPU-only image for portability — no CUDA or NVIDIA drivers required on host.
# For GPU support on larger datasets, replace with:
#   FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04
# and install torch with --index-url https://download.pytorch.org/whl/cu126
FROM python:3.11-slim

# System dependencies required by umap-learn (numba/llvmlite) and matplotlib
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy packaging files first — better Docker layer caching.
# If only source code changes (not dependencies), the pip install
# layers are reused from cache, making rebuilds much faster.
COPY pyproject.toml .
COPY src/ src/

# Install CPU-only PyTorch (significantly smaller image than CUDA build)
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install the package in editable mode, then remaining dependencies
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir scikit-learn umap-learn matplotlib scipy

# Create output directory (will be overridden by volume mount at runtime)
RUN mkdir -p outputs data

# Single entry point — runs the full pipeline
CMD ["mnist-prelabelling"]