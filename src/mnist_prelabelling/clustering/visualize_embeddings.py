import numpy as np
from mnist_prelabelling import config
import umap
import matplotlib.pyplot as plt

# load the pool embeddings generated
pool_embeddings = np.load("outputs/pool_embeddings.npy")
print(f"Loaded embeddings: {pool_embeddings.shape}")

# reduce to 2D for visualization
reducer = umap.UMAP(n_components=2, random_state=config.RANDOM_SEED)
embeddings_2d = reducer.fit_transform(pool_embeddings)
print(f"2D projection shape: {embeddings_2d.shape}")

#Plotting the 2D embeddings
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=2, alpha=0.5)
plt.title("UMAP projection of pool embeddings")
plt.xlabel("UMAP dimension 1")
plt.ylabel("UMAP dimension 2")
plt.savefig("outputs/embeddings_2d_plot.png", dpi=150)
print("Plot saved to outputs/embeddings_2d_plot.png")