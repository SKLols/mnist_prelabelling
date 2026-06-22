"""
Pipeline: unsupervised label correction with iterative self-training.
Uses 'either' strategy (empirically validated in analysis/evaluate_correction_strategies.py).
No ground truth used here — GT is only read in evaluate_prelabelling.py for final reporting.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from mnist_prelabelling import config
from mnist_prelabelling.models.custom_cnn import CustomCNN
from mnist_prelabelling.utils.run_logger import save_run_log


NUM_ITERATIONS = 4
K_NEIGHBORS = 15


class PoolDatasetWithGeneratedLabels(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def compute_early_confidence(pool_images, current_labels, device):
    """Retrain for 3 epochs on current labels, return mean early-epoch confidence."""
    labels_tensor = torch.tensor(current_labels, dtype=torch.long)
    loader = DataLoader(
        PoolDatasetWithGeneratedLabels(pool_images, labels_tensor),
        batch_size=config.BATCH_SIZE, shuffle=True
    )
    model = CustomCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    confidences = []
    for epoch in range(3):
        model.train()
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), Y)
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_conf = []
        with torch.no_grad():
            for start in range(0, len(pool_images), 256):
                X = pool_images[start:start+256].to(device)
                Y = labels_tensor[start:start+256].to(device)
                probs = torch.softmax(model(X), dim=1)
                epoch_conf.append(
                    probs.gather(1, Y.unsqueeze(1)).squeeze(1).cpu().numpy()
                )
        conf_arr = np.concatenate(epoch_conf)
        confidences.append(conf_arr)
        print(f"    early-training epoch {epoch+1} mean confidence: {conf_arr.mean():.4f}")

    return np.array(confidences).mean(axis=0)


def build_flags(early_conf, distances_to_centroid, cluster_ids):
    """Recompute confidence + distance flags from current signals."""
    km = KMeans(n_clusters=2, n_init=10, random_state=config.RANDOM_SEED)
    cids = km.fit_predict(early_conf.reshape(-1, 1))
    means = [early_conf[cids == c].mean() for c in range(2)]
    flag_conf = cids == int(np.argmin(means))

    flag_dist = np.zeros(len(distances_to_centroid), dtype=bool)
    for cid in range(10):
        mask = cluster_ids == cid
        cutoff = np.percentile(distances_to_centroid[mask], 95)
        flag_dist[mask] = distances_to_centroid[mask] > cutoff

    return flag_conf | flag_dist


def apply_correction(current_labels, flagged_mask, pool_embeddings):
    """Neighbor-majority reassignment for flagged points."""
    corrected = current_labels.copy()
    clean_mask = ~flagged_mask
    nn_model = NearestNeighbors(n_neighbors=K_NEIGHBORS).fit(pool_embeddings[clean_mask])
    clean_labels = current_labels[clean_mask]
    flagged_indices = np.where(flagged_mask)[0]
    _, npos = nn_model.kneighbors(pool_embeddings[flagged_indices])
    changed = 0
    for i, pidx in enumerate(flagged_indices):
        majority = Counter(clean_labels[npos[i]].tolist()).most_common(1)[0][0]
        if majority != corrected[pidx]:
            changed += 1
        corrected[pidx] = majority
    return corrected, changed


def run(pool_embeddings=None, pool_initial_labels=None,
        flag_confidence=None, flag_distance=None,
        distances_to_centroid=None, cluster_ids=None,
        device=None, output_dir: str = "outputs"):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pool_embeddings is None:
        pool_embeddings = np.load("outputs/pool_embeddings.npy")
    if pool_initial_labels is None:
        pool_initial_labels = np.load("outputs/pool_initial_labels.npy")
    if flag_confidence is None:
        flag_confidence = np.load("outputs/flag_confidence.npy")
    if flag_distance is None:
        flag_distance = np.load("outputs/flag_distance.npy")
    if distances_to_centroid is None:
        distances_to_centroid = np.load("outputs/pool_distances_to_centroid.npy")
    if cluster_ids is None:
        cluster_ids = np.load("outputs/pool_cluster_ids.npy")

    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    raw_pool = datasets.MNIST(
        root=config.DATA_ROOT, train=True, transform=my_transform, download=True
    )
    pool_images = torch.stack([img for img, _ in raw_pool])

    current_labels = pool_initial_labels.copy()

    # Iteration 0 — single-pass correction using flags from detect_label_errors
    flagged = flag_confidence | flag_distance
    current_labels, changed = apply_correction(current_labels, flagged, pool_embeddings)
    np.save(f"{output_dir}/pool_labels_iter_0.npy", current_labels)
    print(f"\nIteration 0 (initial correction): {changed} labels changed, "
          f"{flagged.sum()} flagged")

    # Iterations 1..NUM_ITERATIONS — retrain, redetect, re-correct
    for it in range(1, NUM_ITERATIONS + 1):
        print(f"\nIteration {it}/{NUM_ITERATIONS}: retraining for confidence signal...")
        early_conf = compute_early_confidence(pool_images, current_labels, device)
        flagged = build_flags(early_conf, distances_to_centroid, cluster_ids)
        current_labels, changed = apply_correction(current_labels, flagged, pool_embeddings)
        np.save(f"{output_dir}/pool_labels_iter_{it}.npy", current_labels)
        print(f"Iteration {it}: {flagged.sum()} flagged, {changed} labels changed")

    np.save(f"{output_dir}/pool_corrected_labels.npy", current_labels)
    print(f"\nSaved final corrected labels after {NUM_ITERATIONS} iterations")

    save_run_log("correct_labels", {
        "script": "correct_labels.py",
        "purpose": f"iterative unsupervised label correction ({NUM_ITERATIONS} iterations)",
        "config": {
            "strategy": "either",
            "k_neighbors": K_NEIGHBORS,
            "num_iterations": NUM_ITERATIONS,
        },
    }, output_dir=f"{output_dir}/runs")

    return current_labels


if __name__ == "__main__":
    run()