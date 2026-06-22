"""
OFFLINE ANALYSIS SCRIPT — uses ground truth to measure the NET correction impact
(errors fixed minus errors introduced) of each detection strategy, so we can choose
which one to use in the actual pipeline. NOT part of the unsupervised pipeline.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from torchvision import datasets
from mnist_prelabelling import config

pool_embeddings = np.load("outputs/pool_embeddings.npy")
pool_initial_labels = np.load("outputs/pool_initial_labels.npy")
flag_confidence = np.load("outputs/flag_confidence.npy")
flag_distance = np.load("outputs/flag_distance.npy")

# Ground truth — analysis only
raw_pool = datasets.MNIST(root=config.DATA_ROOT, train=True, download=True)
ground_truth = np.array(raw_pool.targets)


def correct_labels(flagged_mask, k_neighbors=15):
    """Reassign each flagged point's label to the majority label among its
    k nearest NON-flagged neighbors in embedding space."""
    corrected_labels = pool_initial_labels.copy()

    clean_mask = ~flagged_mask
    clean_embeddings = pool_embeddings[clean_mask]
    clean_labels = pool_initial_labels[clean_mask]

    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(clean_embeddings)

    flagged_indices = np.where(flagged_mask)[0]
    if len(flagged_indices) == 0:
        return corrected_labels

    flagged_embeddings = pool_embeddings[flagged_indices]
    _, neighbor_positions = nn.kneighbors(flagged_embeddings)

    for i, point_idx in enumerate(flagged_indices):
        neighbor_labels = clean_labels[neighbor_positions[i]]
        majority_label = Counter(neighbor_labels.tolist()).most_common(1)[0][0]
        corrected_labels[point_idx] = majority_label

    return corrected_labels


# Baseline accuracy before any correction
baseline_correct = (pool_initial_labels == ground_truth).sum()
print(f"Baseline (no correction) accuracy: {baseline_correct/60000*100:.2f}% ({baseline_correct}/60000)\n")

flag_both = flag_confidence & flag_distance
flag_either = flag_confidence | flag_distance

strategies = {
    "confidence only": flag_confidence,
    "distance only": flag_distance,
    "both (intersection)": flag_both,
    "either (union)": flag_either,
}

print(f"{'Strategy':<22} {'Flagged':>8} {'Fixed':>7} {'Broken':>7} {'Net':>6} {'NewAcc':>8}")
print("-" * 62)
for name, flag in strategies.items():
    corrected = correct_labels(flag)

    was_wrong = pool_initial_labels != ground_truth
    now_right = corrected == ground_truth

    # Fixed: was wrong before, right after
    fixed = (was_wrong & now_right & flag).sum()
    # Broken: was right before, wrong after
    was_right = pool_initial_labels == ground_truth
    now_wrong = corrected != ground_truth
    broken = (was_right & now_wrong & flag).sum()

    net = fixed - broken
    new_accuracy = (corrected == ground_truth).mean() * 100

    print(f"{name:<22} {flag.sum():>8} {fixed:>7} {broken:>7} {net:>+6} {new_accuracy:>7.2f}%")