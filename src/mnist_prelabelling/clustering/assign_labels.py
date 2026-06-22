import numpy as np
import torch
from torchvision import datasets, transforms
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from mnist_prelabelling import config
from mnist_prelabelling.models.custom_cnn import CustomCNN
from mnist_prelabelling.embeddings.extractor import extract_embeddings
from mnist_prelabelling.utils.run_logger import save_run_log

from pathlib import Path


def run(cnn_model=None, pool_embeddings=None, cluster_ids=None, centroids=None, device=None, output_dir: str = "outputs"):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if cnn_model is None:
        cnn_model = CustomCNN().to(device)
        cnn_model.load_state_dict(
            torch.load("outputs/seed_classifier_weights.pth", map_location=device)
        )
    cnn_model.eval()

    if pool_embeddings is None:
        pool_embeddings = np.load("outputs/pool_embeddings.npy")
    if cluster_ids is None:
        cluster_ids = np.load("outputs/pool_cluster_ids.npy")

    seed_dataset = datasets.MNIST(
        root=config.DATA_ROOT, train=False, transform=my_transform, download=True
    )
    seed_true_labels = np.array([label for _, label in seed_dataset])
    seed_embeddings = extract_embeddings(cnn_model, seed_dataset, device, batch_size=config.BATCH_SIZE)
    print(f"Seed embeddings shape: {seed_embeddings.shape}")

    if centroids is None:
        centroids = np.zeros((10, pool_embeddings.shape[1]))
        for cid in range(10):
            centroids[cid] = pool_embeddings[cluster_ids == cid].mean(axis=0)

    distances = cdist(seed_embeddings, centroids)
    seed_cluster_ids = distances.argmin(axis=1)

    agreement_matrix = np.zeros((10, 10))
    for cid in range(10):
        for digit in range(10):
            agreement_matrix[cid, digit] = np.sum(
                (seed_cluster_ids == cid) & (seed_true_labels == digit)
            )

    cluster_indices, digit_indices = linear_sum_assignment(-agreement_matrix)
    cluster_to_digit = {int(c): int(d) for c, d in zip(cluster_indices, digit_indices)}

    print("\nHungarian assignment:")
    for cid, digit in cluster_to_digit.items():
        print(f"  Cluster {cid} -> digit {digit} ({int(agreement_matrix[cid, digit])} votes)")

    pool_initial_labels = np.array([cluster_to_digit[c] for c in cluster_ids])
    np.save(f"{output_dir}/pool_initial_labels.npy", pool_initial_labels)
    print(f"\nSaved initial labels for {len(pool_initial_labels)} pool images")

    seed_predicted = np.array([cluster_to_digit[c] for c in seed_cluster_ids])
    accuracy = (seed_predicted == seed_true_labels).mean()
    print(f"Seed-set labeling accuracy: {accuracy:.4f}")

    save_run_log("assign_labels", {
        "script": "assign_labels.py",
        "purpose": "cluster-to-digit label assignment via Hungarian algorithm",
        "results": {
            "cluster_to_digit": cluster_to_digit,
            "seed_set_accuracy": float(accuracy),
        },
    }, output_dir=f"{output_dir}/runs")

    return pool_initial_labels, cluster_to_digit


if __name__ == "__main__":
    run()