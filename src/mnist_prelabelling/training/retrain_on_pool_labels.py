import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from mnist_prelabelling import config
from mnist_prelabelling.models.custom_cnn import CustomCNN
from mnist_prelabelling.utils.run_logger import save_run_log


class PoolDatasetWithGeneratedLabels(Dataset):
    def __init__(self, images, generated_labels):
        self.images = images
        self.generated_labels = generated_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.generated_labels[idx]


def compute_confidence_on_own_labels(model, images, labels, device, batch_size=256):
    """Run images through model, return softmax confidence on each image's given label."""
    model.eval()
    all_confidences = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            X = images[start:start + batch_size].to(device)
            Y = labels[start:start + batch_size].to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            conf = probs.gather(1, Y.unsqueeze(1)).squeeze(1)
            all_confidences.append(conf.cpu().numpy())
    model.train()
    return np.concatenate(all_confidences, axis=0)


my_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

raw_pool_dataset = datasets.MNIST(root=config.DATA_ROOT, train=True, transform=my_transform, download=True)
pool_images = torch.stack([img for img, _ in raw_pool_dataset])

pool_initial_labels = np.load("outputs/pool_initial_labels.npy")
pool_initial_labels_tensor = torch.tensor(pool_initial_labels, dtype=torch.long)

pool_dataset_with_generated_labels = PoolDatasetWithGeneratedLabels(pool_images, pool_initial_labels_tensor)
pool_loader = DataLoader(pool_dataset_with_generated_labels, batch_size=config.BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CustomCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=config.LEARNING_RATE)

confidence_per_epoch = []

print("=== Retraining on pool images using self-generated labels (not groundtruth) ===")
for epoch in range(config.EPOCHS):
    print(f"\nEpoch {epoch+1}\n-------------------------------")
    for batch, (X, Y) in enumerate(pool_loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        outputs = cnn_model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print(f"  batch {batch}, loss: {loss.item():.4f}")

    epoch_confidences = compute_confidence_on_own_labels(
        cnn_model, pool_images, pool_initial_labels_tensor, device
    )
    confidence_per_epoch.append(epoch_confidences)
    print(f"  epoch {epoch+1} mean confidence on own label: {epoch_confidences.mean():.4f}")

confidence_per_epoch = np.array(confidence_per_epoch)
print(f"\nConfidence trajectory shape: {confidence_per_epoch.shape}")

np.save("outputs/pool_confidence_per_epoch.npy", confidence_per_epoch)
torch.save(cnn_model.state_dict(), "outputs/pool_retrained_weights.pth")
print("Saved confidence trajectory and final model weights")

run_data = {
    "script": "retrain_on_pool_labels.py",
    "purpose": "retrain on self-generated pool labels, tracking per-epoch confidence trajectory for label error detection",
    "config": {
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.EPOCHS,
        "random_seed": config.RANDOM_SEED,
    },
    "results": {
        "mean_confidence_per_epoch": [float(c.mean()) for c in confidence_per_epoch],
    },
}
save_run_log("retrain_on_pool_labels", run_data)