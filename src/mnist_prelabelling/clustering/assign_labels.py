import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from mnist_prelabelling import config
from mnist_prelabelling.models.custom_cnn import CustomCNN
from mnist_prelabelling.embeddings.extractor import extract_embeddings
from mnist_prelabelling.utils.run_logger import save_run_log

my_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
cnn_model = CustomCNN().to(device)
cnn_model.load_state_dict(torch.load("outputs/seed_classifier_weights.pth", map_location=device))
cnn_model.eval()

# Load the seed set (with true labels — legitimate here, since this script's job is label assignment)
seed_dataset = datasets.MNIST(root=config.DATA_ROOT, train=False, transform=my_transform, download=True)
seed_true_labels = np.array([label for _, label in seed_dataset])

# Embed the seed set using the same trained model
seed_embeddings = extract_embeddings(cnn_model, seed_dataset, device, batch_size=config.BATCH_SIZE)
print(f"Seed embeddings shape: {seed_embeddings.shape}")

# Load the pool's cluster assignments and reconstruct centroids
pool_embeddings = np.load("outputs/pool_embeddings.npy")
pool_cluster_ids = np.load("outputs/pool_cluster_ids.npy")

centroids = np.zeros((10, pool_embeddings.shape[1]))
for cluster_id in range(10):
    centroids[cluster_id] = pool_embeddings[pool_cluster_ids == cluster_id].mean(axis=0)

# Geometrically assign each seed image to its nearest pool-centroid (Step A)
distances = cdist(seed_embeddings, centroids)
seed_cluster_ids = distances.argmin(axis=1)

print(f"Seed cluster IDs shape: {seed_cluster_ids.shape}")
print(f"Seed cluster ID distribution: {Counter(seed_cluster_ids.tolist())}")

print("\n=== Hungarian Matching Algorithm ===")

# Build a 10x10 "agreement matrix": rows = clusters, columns = true digits
agreement_matrix = np.zeros((10, 10))
for cluster_id in range(10):
    for digit in range(10):
        agreement_matrix[cluster_id, digit] = np.sum(
            (seed_cluster_ids == cluster_id) & (seed_true_labels == digit)
        )

# linear_sum_assignment MINIMIZES cost by default, so we negate to maximize agreement
cluster_indices, digit_indices = linear_sum_assignment(-agreement_matrix)

hungarian_mapping = {int(c): int(d) for c, d in zip(cluster_indices, digit_indices)}
print("Hungarian assignment:")
for cluster_id, digit in hungarian_mapping.items():
    print(f"  Cluster {cluster_id} -> digit {digit} ({int(agreement_matrix[cluster_id, digit])} seed votes)")

pool_predicted_labels = np.array([hungarian_mapping[c] for c in pool_cluster_ids])
np.save("outputs/pool_initial_labels.npy", pool_predicted_labels)
print(f"\nSaved initial labels for {len(pool_predicted_labels)} pool images")

# Evaluate: how many seed images get correctly labeled by the cluster->digit mapping?
seed_predicted_labels_hungarian = np.array([hungarian_mapping[c] for c in seed_cluster_ids])
hungarian_accuracy = (seed_predicted_labels_hungarian == seed_true_labels).mean()

print(f"\n=== Seed-set labeling accuracy (cluster-derived labels vs true labels) ===")
print(f"Hungarian accuracy:     {hungarian_accuracy:.4f}")