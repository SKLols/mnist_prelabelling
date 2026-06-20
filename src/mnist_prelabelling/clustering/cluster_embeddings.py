import numpy as np
from sklearn.cluster import KMeans
from mnist_prelabelling import config
from mnist_prelabelling.utils.run_logger import save_run_log

# Load the pool embeddings
pool_embeddings = np.load("outputs/pool_embeddings.npy")
print(f"Loaded embeddings: {pool_embeddings.shape}")

# Perform KMeans clustering
kmeans = KMeans(n_clusters=10, n_init=10, random_state=config.RANDOM_SEED)
cluster_ids = kmeans.fit_predict(pool_embeddings)

assigned_centroids = kmeans.cluster_centers_[cluster_ids]
distances_to_centroid = np.linalg.norm(pool_embeddings - assigned_centroids, axis=1)

# Save the cluster IDs and distances to centroid
np.save("outputs/pool_cluster_ids.npy", cluster_ids)
np.save("outputs/pool_distances_to_centroid.npy", distances_to_centroid)
print("Saved cluster_ids and distances_to_centroid to outputs/")

# Print cluster size distribution
unique, counts = np.unique(cluster_ids, return_counts=True)
print("\nCluster size distribution:")
for cluster_id, count in zip(unique, counts):
    print(f"  Cluster {cluster_id}: {count} images")

run_data = {
    "script": "cluster_embeddings.py",
    "purpose": "cluster pool embeddings into 10 groups using KMeans (no labels used)",
    "config": {
        "n_clusters": 10,
        "n_init": 10,
        "random_seed": config.RANDOM_SEED,
    },
    "results": {
        "cluster_sizes": {str(c): int(n) for c, n in zip(unique, counts)},
        "mean_distance_to_centroid": float(distances_to_centroid.mean()),
    },
}
save_run_log("cluster_embeddings", run_data)