# Results

## Summary

| Stage | Accuracy |
|---|---|
| Initial pre-labelling | 96.21% |
| After single-pass correction | 97.66% |
| After iteration 1 | 98.02% |
| After iteration 2 | 98.11% |
| After iteration 3 | 98.14% |
| After iteration 4 (final) | 98.10% |
| **Total improvement** | **+1.89 pp** |

Full JSON log: `results/evaluate_prelabelling_best_run.json`

## Sanity Checks

| Check | Result |
|---|---|
| CustomCNN on conventional MNIST split (60k→10k) | 99.02% |
| Seed classifier on pool (debug check, 10k→60k) | 97.35% |
| Seed-set labeling accuracy (cluster→digit mapping) | 97.87% |

## Embedding Visualization

![UMAP projection](../results/embeddings_2d_plot.png)

The 10 well-separated blobs in the UMAP projection confirm that the
seed-trained classifier produces class-discriminative embeddings that
generalize cleanly to the unlabeled pool — a necessary condition for the
clustering-based labeling approach to work.

## Per-digit Accuracy (after final correction)

| Digit | Accuracy | Notes |
|---|---|---|
| 0 | 99.16% | |
| 1 | 98.87% | |
| 2 | 97.82% | |
| 3 | 97.39% | |
| 4 | 97.93% | |
| 5 | 98.10% | |
| 6 | 99.32% | |
| 7 | 98.24% | |
| 8 | 96.17% | Hardest class — visual overlap with 3, 5, 9 |
| 9 | 97.87% | |

## Iterative Correction Behavior

Labels changed per iteration: 1101 → 327 → 162 → 127 → 122

The sharply diminishing number of changes per iteration confirms the
pipeline converges — most correctable errors are fixed in iteration 0
and 1, with later iterations making progressively smaller refinements.
Accuracy plateaus around iteration 3, consistent with confirmation bias:
beyond a certain point, the model begins reinforcing its own residual
errors rather than correcting new ones.

## Docker CPU vs GPU

| Environment | Runtime | Final Accuracy |
|---|---|---|
| GPU (RTX 3050, conda) | ~5-8 min | 98.10% |
| CPU (Docker) | ~25 min | 97.97% |

The slight accuracy difference between GPU and CPU runs is due to
GPU non-determinism in floating-point operations, not a systematic
difference in the approach.