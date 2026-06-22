"""
Pipeline step: unsupervised label correction.

Uses the chosen strategy (validated offline in analysis/evaluate_correction_strategies.py):
flag a point if EITHER the confidence signal OR the distance signal flags it, then
reassign each flagged point's label to the majority label among its nearest
NON-flagged neighbors in embedding space.

This script uses NO ground-truth labels. It consumes only the unsupervised signals
(flags, embeddings, initial labels) produced by earlier pipeline steps, and outputs
the final corrected labels.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter

from mnist_prelabelling.utils.run_logger import save_run_log

# --- Load unsupervised inputs (no ground truth anywhere) ---
pool_embeddings = np.load("outputs/pool_embeddings.npy")
pool_initial_labels = np.load("outputs/pool_initial_labels.npy")
flag_confidence = np.load("outputs/flag_confidence.npy")
flag_distance = np.load("outputs/flag_distance.npy")

# --- Chosen strategy: flag by EITHER signal ---
flagged_mask = flag_confidence | flag_distance
print(f"Total flagged for correction: {flagged_mask.sum()} images ({flagged_mask.sum()/60000*100:.2f}%)")

# --- Neighbor-majority correction using only non-flagged (trusted) neighbors ---
K_NEIGHBORS = 15

corrected_labels = pool_initial_labels.copy()

clean_mask = ~flagged_mask
clean_embeddings = pool_embeddings[clean_mask]
clean_labels = pool_initial_labels[clean_mask]

nn = NearestNeighbors(n_neighbors=K_NEIGHBORS)
nn.fit(clean_embeddings)

flagged_indices = np.where(flagged_mask)[0]
flagged_embeddings = pool_embeddings[flagged_indices]
_, neighbor_positions = nn.kneighbors(flagged_embeddings)

num_changed = 0
for i, point_idx in enumerate(flagged_indices):
    neighbor_labels = clean_labels[neighbor_positions[i]]
    majority_label = Counter(neighbor_labels.tolist()).most_common(1)[0][0]
    if majority_label != corrected_labels[point_idx]:
        num_changed += 1
    corrected_labels[point_idx] = majority_label

print(f"Labels actually changed: {num_changed}")

# --- Save final corrected labels ---
np.save("outputs/pool_corrected_labels.npy", corrected_labels)
print("Saved final corrected labels to outputs/pool_corrected_labels.npy")

run_data = {
    "script": "correct_labels.py",
    "purpose": "unsupervised label correction (either-signal flagging + neighbor-majority reassignment)",
    "config": {
        "strategy": "either (confidence OR distance)",
        "k_neighbors": K_NEIGHBORS,
        "random_seed": 42,
    },
    "results": {
        "num_flagged": int(flagged_mask.sum()),
        "num_labels_changed": int(num_changed),
    },
}
save_run_log("correct_labels", run_data)