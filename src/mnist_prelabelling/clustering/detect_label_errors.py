import numpy as np
from sklearn.cluster import KMeans
from mnist_prelabelling import config

# --- Load the per-epoch confidence trajectory from retraining ---
confidence_per_epoch = np.load("outputs/pool_confidence_per_epoch.npy")
print(f"Confidence trajectory shape: {confidence_per_epoch.shape}")

early_epoch_confidence = confidence_per_epoch[:3].mean(axis=0)
final_epoch_confidence = confidence_per_epoch[-1]

print(f"Early-epoch mean confidence: {early_epoch_confidence.mean():.4f}")
print(f"Final-epoch mean confidence: {final_epoch_confidence.mean():.4f}")

# --- Signal 1: flag points with low early-training confidence on their own label ---
combined_signal = early_epoch_confidence.reshape(-1, 1)

confidence_kmeans = KMeans(n_clusters=2, n_init=10, random_state=config.RANDOM_SEED)
confidence_cluster_ids = confidence_kmeans.fit_predict(combined_signal)

cluster_means = [combined_signal[confidence_cluster_ids == c].mean() for c in range(2)]
noisy_cluster_id = int(np.argmin(cluster_means))

is_flagged_as_noisy = confidence_cluster_ids == noisy_cluster_id
num_flagged = is_flagged_as_noisy.sum()

print(f"\nCluster means (confidence): {cluster_means}")
print(f"Flagged as potentially noisy (confidence signal): {num_flagged} images ({num_flagged/60000*100:.2f}%)")

# --- Signal 2: flag points far from their assigned cluster's centroid (relative to that cluster's own spread) ---
distances_to_centroid = np.load("outputs/pool_distances_to_centroid.npy")
pool_cluster_ids = np.load("outputs/pool_cluster_ids.npy")

percentile_threshold = 95
is_flagged_by_distance = np.zeros(len(distances_to_centroid), dtype=bool)

for cluster_id in range(10):
    mask = pool_cluster_ids == cluster_id
    cluster_distances = distances_to_centroid[mask]
    cutoff = np.percentile(cluster_distances, percentile_threshold)
    is_flagged_by_distance[mask] = distances_to_centroid[mask] > cutoff

print(f"Flagged by distance (top {100-percentile_threshold}% per cluster): {is_flagged_by_distance.sum()} images")
# --- Combine both independent signals ---
is_flagged_by_either = is_flagged_as_noisy | is_flagged_by_distance
is_flagged_by_both = is_flagged_as_noisy & is_flagged_by_distance

print(f"\nFlagged by confidence only: {(is_flagged_as_noisy & ~is_flagged_by_distance).sum()}")
print(f"Flagged by distance only: {(is_flagged_by_distance & ~is_flagged_as_noisy).sum()}")
print(f"Flagged by BOTH (highest confidence errors): {is_flagged_by_both.sum()}")
print(f"Flagged by EITHER (broader candidate set): {is_flagged_by_either.sum()}")

np.save("outputs/flag_confidence.npy", is_flagged_as_noisy)
np.save("outputs/flag_distance.npy", is_flagged_by_distance)
print("\nSaved flag arrays to outputs/")