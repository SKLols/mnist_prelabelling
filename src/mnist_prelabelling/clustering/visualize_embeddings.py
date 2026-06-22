import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path
from mnist_prelabelling import config

from pathlib import Path


def run(pool_embeddings=None, cluster_ids=None, output_dir: str = "outputs"):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if pool_embeddings is None:
        pool_embeddings = np.load("outputs/pool_embeddings.npy")
    if cluster_ids is None:
        cluster_ids = np.load("outputs/pool_cluster_ids.npy")

    print(f"Loaded embeddings: {pool_embeddings.shape}")

    reducer = umap.UMAP(n_components=2, random_state=config.RANDOM_SEED)
    embeddings_2d = reducer.fit_transform(pool_embeddings)
    print(f"2D projection shape: {embeddings_2d.shape}")

    Path("outputs").mkdir(exist_ok=True)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=cluster_ids, cmap="tab10", s=2, alpha=0.5
    )
    plt.colorbar(scatter)
    plt.title("UMAP projection of pool embeddings (colored by cluster)")
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    plt.savefig(f"{output_dir}/embeddings_2d_plot.png", dpi=150)
    plt.close()
    print("Plot saved to outputs/embeddings_2d_plot.png")

    return embeddings_2d


if __name__ == "__main__":
    run()