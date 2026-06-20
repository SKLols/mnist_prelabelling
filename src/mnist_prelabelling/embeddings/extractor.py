import torch
import numpy as np
from torch.utils.data import DataLoader


def extract_embeddings(model, dataset, device, batch_size=64):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    for X, _ in loader:
        X = X.to(device)
        embeddings = model.embed(X)
        embeddings = embeddings.cpu().numpy()
        all_embeddings.append(embeddings)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings