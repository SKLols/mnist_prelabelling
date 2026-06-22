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

from pathlib import Path


def run(device=None, output_dir: str = "outputs"):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_dataset = datasets.MNIST(root=config.DATA_ROOT, train=False, transform=my_transform, download=True)
    seed_loader = DataLoader(seed_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    pool_dataset = datasets.MNIST(root=config.DATA_ROOT, train=True, transform=my_transform, download=True)

    cnn_model = CustomCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=config.LEARNING_RATE)

    print("=== Training seed classifier on 10k seed set ===")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}\n-------------------------------")
        for batch, (X, Y) in enumerate(seed_loader):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = criterion(cnn_model(X), Y)
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                print(f"  batch {batch}, loss: {loss.item():.4f}")

    print("\n=== Extracting embeddings for 60k pool ===")
    start_time = time.time()
    pool_embeddings = extract_embeddings(cnn_model, pool_dataset, device, batch_size=config.BATCH_SIZE)
    elapsed = time.time() - start_time
    print(f"Embeddings shape: {pool_embeddings.shape}, took {elapsed:.1f}s")

    np.save(f"{output_dir}/pool_embeddings.npy", pool_embeddings)
    torch.save(cnn_model.state_dict(), f"{output_dir}/seed_classifier_weights.pth")
    print("Saved embeddings and model weights")

    save_run_log("pool_embeddings", {
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
    }, output_dir=f"{output_dir}/runs")

    return cnn_model, pool_embeddings, device


if __name__ == "__main__":
    run()