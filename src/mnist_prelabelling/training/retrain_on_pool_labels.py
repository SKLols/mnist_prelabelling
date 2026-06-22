import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from mnist_prelabelling import config
from mnist_prelabelling.models.custom_cnn import CustomCNN
from mnist_prelabelling.utils.run_logger import save_run_log

from pathlib import Path


class PoolDatasetWithGeneratedLabels(Dataset):
    def __init__(self, images, generated_labels):
        self.images = images
        self.generated_labels = generated_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.generated_labels[idx]


def compute_confidence_on_own_labels(model, images, labels, device, batch_size=256):
    model.eval()
    all_confidences = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            X = images[start:start + batch_size].to(device)
            Y = labels[start:start + batch_size].to(device)
            probs = torch.softmax(model(X), dim=1)
            conf = probs.gather(1, Y.unsqueeze(1)).squeeze(1)
            all_confidences.append(conf.cpu().numpy())
    model.train()
    return np.concatenate(all_confidences, axis=0)


def run(pool_initial_labels=None, device=None, output_dir: str = "outputs"):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    raw_pool = datasets.MNIST(
        root=config.DATA_ROOT, train=True, transform=my_transform, download=True
    )
    pool_images = torch.stack([img for img, _ in raw_pool])

    if pool_initial_labels is None:
        pool_initial_labels = np.load("outputs/pool_initial_labels.npy")
    pool_labels_tensor = torch.tensor(pool_initial_labels, dtype=torch.long)

    loader = DataLoader(
        PoolDatasetWithGeneratedLabels(pool_images, pool_labels_tensor),
        batch_size=config.BATCH_SIZE, shuffle=True
    )

    cnn_model = CustomCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=config.LEARNING_RATE)

    confidence_per_epoch = []

    print("=== Retraining on pool images using self-generated labels ===")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}\n-------------------------------")
        for batch, (X, Y) in enumerate(loader):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = criterion(cnn_model(X), Y)
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                print(f"  batch {batch}, loss: {loss.item():.4f}")

        epoch_conf = compute_confidence_on_own_labels(
            cnn_model, pool_images, pool_labels_tensor, device
        )
        confidence_per_epoch.append(epoch_conf)
        print(f"  epoch {epoch+1} mean confidence: {epoch_conf.mean():.4f}")

    confidence_per_epoch = np.array(confidence_per_epoch)
    np.save(f"{output_dir}/pool_confidence_per_epoch.npy", confidence_per_epoch)
    torch.save(cnn_model.state_dict(), f"{output_dir}/pool_retrained_weights.pth")
    print("Saved confidence trajectory and retrained model weights")

    save_run_log("retrain_on_pool_labels", {
        "script": "retrain_on_pool_labels.py",
        "purpose": "retrain on self-generated pool labels, track confidence trajectory",
        "config": {
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
        },
        "results": {
            "mean_confidence_per_epoch": [float(c.mean()) for c in confidence_per_epoch],
        },
    }, output_dir=f"{output_dir}/runs")

    return confidence_per_epoch, device


if __name__ == "__main__":
    run()