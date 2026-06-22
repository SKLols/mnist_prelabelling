"""
Official accuracy evaluation: before correction, per iteration, and final.
This is the only pipeline script that reads pool ground-truth labels,
used purely for final reporting as the task requires.
"""

import numpy as np
from pathlib import Path
from torchvision import datasets

from mnist_prelabelling import config
from mnist_prelabelling.utils.run_logger import save_run_log


def run(pool_initial_labels=None, pool_corrected_labels=None,
        output_dir: str = "outputs"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if pool_initial_labels is None:
        pool_initial_labels = np.load("outputs/pool_initial_labels.npy")
    if pool_corrected_labels is None:
        pool_corrected_labels = np.load("outputs/pool_corrected_labels.npy")

    raw_pool = datasets.MNIST(root=config.DATA_ROOT, train=True, download=True)
    ground_truth = np.array(raw_pool.targets)

    accuracy_before = (pool_initial_labels == ground_truth).mean()
    accuracy_after = (pool_corrected_labels == ground_truth).mean()

    # Check for per-iteration intermediate labels
    iter_accuracies = {}
    for it in range(10):
        iter_path = Path(output_dir) / f"pool_labels_iter_{it}.npy"
        if iter_path.exists():
            iter_labels = np.load(str(iter_path))
            iter_accuracies[it] = float((iter_labels == ground_truth).mean())

    print("=" * 55)
    print("PRE-LABELLING ACCURACY (vs ground truth)")
    print("=" * 55)
    print(f"Before any correction:  {accuracy_before*100:.2f}%  "
          f"({(pool_initial_labels == ground_truth).sum()}/60000)")

    if iter_accuracies:
        for it, acc in iter_accuracies.items():
            label = "initial" if it == 0 else f"iter {it}"
            print(f"After {label:>8} correction: {acc*100:.2f}%")

    print(f"After final correction:  {accuracy_after*100:.2f}%  "
          f"({(pool_corrected_labels == ground_truth).sum()}/60000)")
    print(f"Total improvement:       {(accuracy_after - accuracy_before)*100:+.2f} "
          f"percentage points")

    print("\nPer-digit accuracy (after final correction):")
    for digit in range(10):
        mask = ground_truth == digit
        acc = (pool_corrected_labels[mask] == ground_truth[mask]).mean()
        print(f"  digit {digit}: {acc*100:.2f}%")

    save_run_log("evaluate_prelabelling", {
        "script": "evaluate_prelabelling.py",
        "purpose": "official pre-labelling accuracy before vs after correction",
        "results": {
            "accuracy_before_correction": float(accuracy_before),
            "accuracy_per_iteration": iter_accuracies,
            "accuracy_after_correction": float(accuracy_after),
            "improvement_percentage_points": float(
                (accuracy_after - accuracy_before) * 100
            ),
        },
    }, output_dir=f"{output_dir}/runs")

    return accuracy_before, accuracy_after


if __name__ == "__main__":
    run()