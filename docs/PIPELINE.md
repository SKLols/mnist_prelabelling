# Pipeline

## Data Flow

MNIST test split (10k, labeled)     MNIST train split (60k, unlabeled pool)

│                                        │

▼                                        │

Seed Classifier                                 │

(CustomCNN, 10 epochs)                          │

│                                        │

├──── embed() ──────────────────────────▶│

│                              Pool embeddings (60k × 64)

│                                        │

│                              ┌──────────┴──────────┐

│                              ▼                     ▼

│                        KMeans (k=10)         UMAP → 2D plot

│                         Cluster IDs          (visualization only)

│                              │

│◀──── embed seed set ─────────┤

│                              │

└──── Hungarian algorithm ────▶│

cluster→digit mapping

│

▼

pool_initial_labels.npy

(initial accuracy ~96%)

│

Retrain CustomCNN

(on pool + generated labels)

│

Confidence trajectory

+ Distance to centroid

│

Error detection (either signal)

│

Neighbor-majority correction × 4

│

pool_corrected_labels.npy

(final accuracy ~98%)

│

Evaluate vs ground truth

(before / per-iteration / after)

## Intermediate Outputs (per run, saved to `outputs/run_YYYYMMDD_HHMMSS/`)

| File | Contents | Used by |
|---|---|---|
| `pool_embeddings.npy` | 60k × 64 embedding matrix | clustering, correction |
| `seed_classifier_weights.pth` | Trained seed classifier | label assignment |
| `pool_cluster_ids.npy` | Cluster ID per pool image | label assignment, correction |
| `pool_distances_to_centroid.npy` | Distance to assigned centroid | error detection |
| `embeddings_2d_plot.png` | UMAP scatter plot | visualization |
| `pool_initial_labels.npy` | Initial pseudo-labels | evaluation, retraining |
| `pool_confidence_per_epoch.npy` | 10×60k confidence trajectory | error detection |
| `pool_retrained_weights.pth` | Pool-retrained model | (reference) |
| `flag_confidence.npy` | Confidence-based error flags | correction |
| `flag_distance.npy` | Distance-based error flags | correction |
| `pool_labels_iter_{n}.npy` | Labels after iteration n | evaluation |
| `pool_corrected_labels.npy` | Final corrected labels | evaluation |
| `runs/*.json` | Per-step run logs | reproducibility |