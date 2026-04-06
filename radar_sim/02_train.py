"""
02_train.py — Train a neural network to classify radar contacts.

Architecture: 3-layer MLP  (4 → 32 → 16 → 4)   (see radar_model.py)
Saves model.pt and scaler.npz — both are needed by 03_infer.py and 04_attack.py.

Run: python radar_sim/02_train.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from radar_model import RadarMLP

# ── Hyperparameters ───────────────────────────────────────────
EPOCHS               = 50
BATCH_SIZE           = 64
LEARNING_RATE        = 1e-3
TRAIN_SPLIT          = 0.8
CONFIDENCE_THRESHOLD = 0.70   # contacts below this are flagged as uncertain
# ─────────────────────────────────────────────────────────────

# Load dataset
data        = np.load("radar_sim/dataset.npz", allow_pickle=True)
features    = torch.tensor(data["features"])
labels      = torch.tensor(data["labels"])
class_names = list(data["class_names"])

# Normalize and save scaler for inference scripts
mean          = features.mean(0)
std           = features.std(0)
features_norm = (features - mean) / (std + 1e-8)
np.savez("radar_sim/scaler.npz", mean=mean.numpy(), std=std.numpy())

# Train / validation split
dataset    = TensorDataset(features_norm, labels)
train_size = int(TRAIN_SPLIT * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

model     = RadarMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print(f"Training RadarMLP for {EPOCHS} epochs  "
      f"(train={train_size}, val={val_size})\n")
print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Accuracy':>14}")
print("─" * 38)

for epoch in range(1, EPOCHS + 1):
    # Training pass
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation pass
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds    = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total   += len(yb)

    print(f"{epoch:>6}  {total_loss / len(train_loader):>12.4f}  "
          f"{100 * correct / total:>13.1f}%")

torch.save(model.state_dict(), "radar_sim/model.pt")

print()
print(f"Final validation accuracy : {100 * correct / total:.1f}%")
print(f"Confidence threshold      : {CONFIDENCE_THRESHOLD:.0%}")
print(f"Model saved  → radar_sim/model.pt")
print(f"Scaler saved → radar_sim/scaler.npz")
