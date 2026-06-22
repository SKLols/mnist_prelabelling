# Architecture

## Overview

This project implements a semi-supervised bootstrapping pipeline: a small
labeled seed set (10k images) is used to train a feature extractor, whose
learned representations are then used to automatically label a much larger
unlabeled pool (60k images) via clustering, without ever accessing the pool's
ground-truth labels during the labeling process.

## Model Architecture — CustomCNN

A LeNet-5-inspired CNN, modernized with ReLU activations and max-pooling
instead of the original's average-pooling. The architecture was chosen
specifically for this task's scale (10k seed images, 10-class simple dataset)
rather than a larger architecture like AlexNet or VGG-16, which would overfit
on such limited labeled data.

Input (1×28×28)

→ Conv2d(1→32, k=3, pad=1) → ReLU → MaxPool2d(2×2)    # 32×14×14

→ Conv2d(32→64, k=3, pad=1) → ReLU → MaxPool2d(2×2)   # 64×7×7

→ Flatten                                               # 3136

→ Linear(3136→64) → ReLU                               # embedding layer

→ Linear(64→10)                                        # classification head

The model exposes two access modes:
- `forward(x)` — returns 10-class logits for training
- `embed(x)` — returns the 64-dim penultimate layer representation for
  downstream clustering; bypasses the classification head entirely

## Clustering Approach

KMeans (k=10) on the 64-dimensional embeddings. k=10 is set as a known prior
(MNIST has exactly 10 digit classes). Clustering runs on the original 64-dim embeddings, not on the 2D
UMAP projection — UMAP is used only for visualization.

## Cluster-to-Digit Mapping

The Hungarian algorithm (scipy.optimize.linear_sum_assignment) finds the
optimal one-to-one assignment between 10 cluster IDs and 10 digit classes,
using a 10×10 agreement matrix built from seed-set labels. This guarantees
no two clusters map to the same digit — a structural guarantee that simple
per-cluster majority voting cannot provide.

## Error Detection

Two independent unsupervised signals, combined via union (empirically validated):

**Signal 1 — Confidence (CTRL-style):**
The pool-retrained model's early-training softmax probability on each image's
own assigned label, averaged across the first 3 epochs. Low early-training
confidence indicates a label the model resisted fitting — a known signature
of noisy labels (clean labels are learned quickly; wrong labels are memorized
slowly or inconsistently). Two-cluster KMeans on this signal separates
confident from suspicious points.

**Signal 2 — Distance to centroid:**
Per-cluster 95th-percentile distance cutoff. Points in the top 5% most
distant from their cluster's centroid (relative to that cluster's own spread)
are flagged as geometrically ambiguous.

The union of both signals was chosen over intersection after empirical
measurement of precision, recall, and net correction impact showed the union
strategy fixes more genuine errors with minimal damage to correct labels.

## Error Correction

Flagged points are reassigned to the majority label among their 15 nearest
non-flagged neighbors in the 64-dim embedding space (neighbor-majority
reassignment). Using only non-flagged neighbors as voters avoids the
confirmation bias risk of flagged points influencing each other's corrections.

This process is iterated 4 times, with the confidence signal recomputed from
scratch each iteration. Accuracy peaks around iteration 3-4 then plateaus,
consistent with the memorization effect documented in noisy-label literature.

## References

- LeCun et al., "Gradient-based learning applied to document recognition," IEEE 1998
  — Origin of the MNIST dataset and LeNet-5 architecture; informed our CNN design
  (conv-pool-conv-pool-FC structure) and provided the dataset's construction rationale
  (why train/test splits have matched difficulty distributions).

- Caron et al., "Deep Clustering for Unsupervised Learning of Visual Features," ECCV 2018
  — Methodological foundation for the overall pipeline: alternating between clustering
  learned embeddings and using cluster assignments as pseudo-labels for retraining.
  Our pipeline follows this paradigm with a labeled seed set added for initialization.
  [GitHub: facebookresearch/deepcluster]

- Northcutt et al., "Confident Learning: Estimating Uncertainty in Dataset Labels," JAIR 2021
  — Basis for our confidence-based error detection signal: comparing model predictions
  against assigned labels to identify likely mislabelings. Also provided empirical
  evidence that even the original MNIST training set contains genuine label errors,
  validating the need for a correction step.

- Wani et al., "Learning with Noisy Labels through Learnable Weighting and Centroid
  Similarity," arXiv 2024
  — Validated our distance-to-centroid error detection signal: samples situated
  further from their class centroid in early training stages are more likely to be
  mislabeled. Informed the design of Signal 2 in our error detection approach.

- Mughesh Kumar et al., "Comparative Analysis of Handwritten Digit Recognition using
  MLP, CNN, LeNet-5," ICDSCNC 2024
  — Empirical benchmark validating our CustomCNN architecture choice: confirmed that
  a Simple CNN (2 conv + 2 FC, ReLU, ~421k params) achieves 99.15% on MNIST,
  matching our independently-derived architecture. Used as the primary reference
  for architecture validation and hyperparameter selection (Adam, lr=0.001,
  batch_size=64).

- Chang & Jha, "CTRL: Clustering Training Losses for Label Error Detection," arXiv 2023
  — Basis for tracking per-epoch confidence trajectories during retraining rather
  than checking confidence after full training: models learn clean and noisy labels
  differently, with noisy labels showing persistently lower confidence in early
  epochs before eventual memorization. Informed our 3-epoch early-confidence signal.