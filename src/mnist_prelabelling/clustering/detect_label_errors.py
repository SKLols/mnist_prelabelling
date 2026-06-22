import numpy as np
from sklearn.cluster import KMeans
from mnist_prelabelling import config

from pathlib import Path


def run(confidence_per_epoch=None, distances_to_centroid=None, cluster_ids=None, output_dir: str = "outputs"):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if confidence_per_epoch is None:
        confidence_per_epoch = np.load("outputs/pool_confidence_per_epoch.npy")
    if distances_to_centroid is None:
        distances_to_centroid = np.load("outputs/pool_distances_to_centroid.npy")
    if cluster_ids is None:
        cluster_ids = np.load("outputs/pool_cluster_ids.npy")

    print(f"Confidence trajectory shape: {confidence_per_epoch.shape}")

    early_conf = confidence_per_epoch[:3].mean(axis=0)
    print(f"Early-epoch mean confidence: {early_conf.mean():.4f}")
    print(f"Final-epoch mean confidence: {confidence_per_epoch[-1].mean():.4f}")

    # Signal 1: confidence-based flagging
    km = KMeans(n_clusters=2, n_init=10, random_state=config.RANDOM_SEED)
    cids = km.fit_predict(early_conf.reshape(-1, 1))
    means = [early_conf[cids == c].mean() for c in range(2)]
    flag_confidence = cids == int(np.argmin(means))
    print(f"\nFlagged by confidence: {flag_confidence.sum()} ({flag_confidence.sum()/60000*100:.2f}%)")

    # Signal 2: distance-based flagging (top 5% per cluster)
    percentile_threshold = 95
    flag_distance = np.zeros(len(distances_to_centroid), dtype=bool)
    for cid in range(10):
        mask = cluster_ids == cid
        cutoff = np.percentile(distances_to_centroid[mask], percentile_threshold)
        flag_distance[mask] = distances_to_centroid[mask] > cutoff
    print(f"Flagged by distance: {flag_distance.sum()}")

    # Combined
    flag_either = flag_confidence | flag_distance
    flag_both = flag_confidence & flag_distance
    print(f"Flagged by both: {flag_both.sum()}")
    print(f"Flagged by either: {flag_either.sum()}")

    np.save(f"{output_dir}/flag_confidence.npy", flag_confidence)
    np.save(f"{output_dir}/flag_distance.npy", flag_distance)
    print("Saved flag arrays")

    return flag_confidence, flag_distance


if __name__ == "__main__":
    run()