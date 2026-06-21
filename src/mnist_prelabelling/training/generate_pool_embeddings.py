import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from mnist_prelabelling import config
from mnist_prelabelling.models.custom_cnn import CustomCNN
from mnist_prelabelling.embeddings.extractor import extract_embeddings
from mnist_prelabelling.utils.run_logger import save_run_log

my_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# seed set (test split, 10k) — used to train the classifier, per step 3
seed_dataset = datasets.MNIST(root=config.DATA_ROOT, train=False, transform=my_transform, download=True)
seed_loader = DataLoader(seed_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# pool (train split, 60k) — labels exist in the data but are never read until step 8
pool_dataset = datasets.MNIST(root=config.DATA_ROOT, train=True, transform=my_transform, download=True)

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

# extract embeddings for the 60k pool ---
print("\n=== extracting embeddings for 60k pool ===")
start_time = time.time()
pool_embeddings = extract_embeddings(cnn_model, pool_dataset, device, batch_size=config.BATCH_SIZE)
elapsed = time.time() - start_time

print(f"Embeddings shape: {pool_embeddings.shape}")
print(f"Extraction took {elapsed:.1f} seconds")

np.save("outputs/pool_embeddings.npy", pool_embeddings)
print("Saved to outputs/pool_embeddings.npy")

torch.save(cnn_model.state_dict(), "outputs/seed_classifier_weights.pth")
print("Model weights saved to outputs/seed_classifier_weights.pth")

run_data = {
    "script": "generate_pool_embeddings.py",
    "purpose": "train seed classifier + extract pool embeddings",
    "model": config.MODEL_NAME,
    "config": {
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.EPOCHS,
        "random_seed": config.RANDOM_SEED,
    },
    "results": {
        "embeddings_shape": list(pool_embeddings.shape),
        "extraction_time_seconds": elapsed,
    },
}
save_run_log("pool_embeddings", run_data)