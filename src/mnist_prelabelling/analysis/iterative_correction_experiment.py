"""
OFFLINE ANALYSIS — iterative self-training experiment.

Tests whether repeating the label-correction loop improves accuracy, and where it
peaks/plateaus. Embeddings and the distance signal are FROZEN (from the original
seed classifier); only the confidence signal is regenerated each iteration by
retraining on the current labels.

Uses ground truth to MEASURE accuracy per iteration (for our understanding only).
Demonstrates awareness of confirmation bias: we expect gains to diminish or reverse,
not grow indefinitely.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from collections import Counter

from mnist_prelabelling import config
from mnist_prelabelling.models.custom_cnn import CustomCNN
from mnist_prelabelling.utils.run_logger import save_run_log

NUM_ITERATIONS = 10

my_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Frozen inputs (computed once, never change across iterations) ---
pool_embeddings = np.load("outputs/pool_embeddings.npy")
flag_distance = np.load("outputs/flag_distance.npy")            # frozen distance signal
raw_pool = datasets.MNIST(root=config.DATA_ROOT, train=True, transform=my_transform, download=True)
pool_images = torch.stack([img for img, _ in raw_pool])
ground_truth = np.array(raw_pool.targets)                        # GT, for measurement only

current_labels = np.load("outputs/pool_initial_labels.npy").copy()


class PoolDS(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def retrain_and_get_early_confidence(labels):
    """Retrain a fresh model on `labels`, return mean early-epoch confidence per image."""
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    loader = DataLoader(PoolDS(pool_images, labels_tensor), batch_size=config.BATCH_SIZE, shuffle=True)
    model = CustomCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    confidences = []
    for epoch in range(3):  # only need early epochs for the confidence signal
        model.train()
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), Y)
            loss.backward()
            optimizer.step()
        # record confidence on own label after this epoch
        model.eval()
        epoch_conf = []
        with torch.no_grad():
            for start in range(0, len(pool_images), 256):
                X = pool_images[start:start+256].to(device)
                Y = labels_tensor[start:start+256].to(device)
                probs = torch.softmax(model(X), dim=1)
                epoch_conf.append(probs.gather(1, Y.unsqueeze(1)).squeeze(1).cpu().numpy())
        confidences.append(np.concatenate(epoch_conf))
    return np.array(confidences).mean(axis=0)


def correct(labels, flagged_mask, k=15):
    corrected = labels.copy()
    clean = ~flagged_mask
    nn_model = NearestNeighbors(n_neighbors=k).fit(pool_embeddings[clean])
    clean_labels = labels[clean]
    idxs = np.where(flagged_mask)[0]
    _, npos = nn_model.kneighbors(pool_embeddings[idxs])
    for i, pidx in enumerate(idxs):
        corrected[pidx] = Counter(clean_labels[npos[i]].tolist()).most_common(1)[0][0]
    return corrected


accuracy_per_iteration = [(current_labels == ground_truth).mean()]
print(f"Iteration 0 (initial): {accuracy_per_iteration[0]*100:.2f}%")

for it in range(1, NUM_ITERATIONS + 1):
    early_conf = retrain_and_get_early_confidence(current_labels)

    # recompute confidence flag (KMeans 2-cluster on confidence, flag low-confidence cluster)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, n_init=10, random_state=config.RANDOM_SEED)
    cids = km.fit_predict(early_conf.reshape(-1, 1))
    means = [early_conf[cids == c].mean() for c in range(2)]
    flag_conf = cids == int(np.argmin(means))

    flagged = flag_conf | flag_distance       # 'either' strategy
    current_labels = correct(current_labels, flagged)

    acc = (current_labels == ground_truth).mean()
    accuracy_per_iteration.append(acc)
    print(f"Iteration {it}: {acc*100:.2f}%  (flagged {flagged.sum()}, changed this round)")

print("\nAccuracy per iteration:", [f"{a*100:.2f}%" for a in accuracy_per_iteration])

save_run_log("iterative_correction_experiment", {
    "script": "iterative_correction_experiment.py",
    "purpose": "test iterative self-training (frozen embeddings); measure accuracy per iteration",
    "config": {"num_iterations": NUM_ITERATIONS, "strategy": "either", "k_neighbors": 15},
    "results": {"accuracy_per_iteration": [float(a) for a in accuracy_per_iteration]},
})