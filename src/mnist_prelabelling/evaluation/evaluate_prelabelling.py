"""
official pre-labelling accuracy evaluation.

Compares our generated labels against ground truth, BEFORE and AFTER correction.
This is the only pipeline-output script that reads ground-truth pool labels, and
it does so purely for final reporting (exactly what the task's step 8 specifies) -
never to influence the labels themselves.
"""

import numpy as np
from torchvision import datasets
from sklearn.metrics import confusion_matrix

from mnist_prelabelling import config
from mnist_prelabelling.utils.run_logger import save_run_log

# --- Load our generated labels (before and after correction) ---
pool_initial_labels = np.load("outputs/pool_initial_labels.npy")
pool_corrected_labels = np.load("outputs/pool_corrected_labels.npy")

# --- Ground truth, for final scoring only ---
raw_pool = datasets.MNIST(root=config.DATA_ROOT, train=True, download=True)
ground_truth = np.array(raw_pool.targets)

# --- Accuracy before and after correction ---
accuracy_before = (pool_initial_labels == ground_truth).mean()
accuracy_after = (pool_corrected_labels == ground_truth).mean()

print("=" * 50)
print("PRE-LABELLING ACCURACY (vs ground truth)")
print("=" * 50)
print(f"Before correction: {accuracy_before*100:.2f}%  ({(pool_initial_labels == ground_truth).sum()}/60000)")
print(f"After correction:  {accuracy_after*100:.2f}%  ({(pool_corrected_labels == ground_truth).sum()}/60000)")
print(f"Improvement:       {(accuracy_after - accuracy_before)*100:+.2f} percentage points")

# --- Per-digit accuracy after correction (which digits remain hardest) ---
print("\nPer-digit accuracy (after correction):")
for digit in range(10):
    mask = ground_truth == digit
    digit_acc = (pool_corrected_labels[mask] == ground_truth[mask]).mean()
    print(f"  digit {digit}: {digit_acc*100:.2f}%")

run_data = {
    "script": "evaluate_prelabelling.py",
    "purpose": "official step-8 pre-labelling accuracy, before vs after correction",
    "results": {
        "accuracy_before_correction": float(accuracy_before),
        "accuracy_after_correction": float(accuracy_after),
        "improvement_percentage_points": float((accuracy_after - accuracy_before) * 100),
    },
}
save_run_log("evaluate_prelabelling", run_data)