import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from mnist_prelabelling import config
from mnist_prelabelling.models.custom_cnn import CustomCNN

from mnist_prelabelling.utils.run_logger import save_run_log

my_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#Swapped the datasets for seed and pool to train on the 10k test set and evaluate on the 60k training set
seed_dataset = datasets.MNIST(root=config.DATA_ROOT, train=False, transform=my_transform, download=True)
pool_dataset = datasets.MNIST(root=config.DATA_ROOT, train=True, transform=my_transform, download=True)

seed_loader = DataLoader(seed_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
pool_loader = DataLoader(pool_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CustomCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=config.LEARNING_RATE)

for epoch in range(config.EPOCHS):
    print(f"\nEpoch {epoch+1}\n-------------------------------")

    for batch, (X, Y) in enumerate(seed_loader):
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        outputs = cnn_model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"  batch {batch}, loss: {loss.item():.4f}")

# Evaluation
cnn_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X, Y in pool_loader:
        X, Y = X.to(device), Y.to(device)
        outputs = cnn_model(X)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == Y).sum().item()
        total += Y.size(0)

accuracy = correct / total

print(f"\nPool Accuracy: {accuracy:.4f} ({correct}/{total})")

run_data = {
    "script": "train_seed_classifier.py",
    "purpose": "task setup - train on 10k seed set, evaluate against 60k pool groundtruth",
    "model": config.MODEL_NAME,
    "config": {
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.EPOCHS,
        "random_seed": config.RANDOM_SEED,
    },
    "results": {
        "pool_accuracy": accuracy,
        "correct": correct,
        "total": total,
    },
}
save_run_log("seed_classifier_pool_debug_check", run_data)