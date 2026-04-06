"""
01_simulate.py — Generate a labeled radar dataset.

Simulates an S-band surveillance radar scanning airspace.
Produces 1000 samples per class and saves them to dataset.npz.

Run: python radar_sim/01_simulate.py
"""

import os
import numpy as np
from radar_model import (
    CLASSES, CLASS_PARAMS, N_CLASSES, FEATURE_NAMES, compute_snr
)

SAMPLES_PER_CLASS = 1000
RANDOM_SEED       = 42

rng = np.random.default_rng(RANDOM_SEED)

features_list = []
labels_list   = []

for label in range(N_CLASSES):
    cls = CLASSES[label]
    p   = CLASS_PARAMS[label]

    ranges     = rng.uniform(*p["range"],    SAMPLES_PER_CLASS)
    velocities = rng.uniform(*p["velocity"], SAMPLES_PER_CLASS)
    rcss       = rng.uniform(*p["rcs"],      SAMPLES_PER_CLASS)
    snrs       = compute_snr(rcss, ranges)

    # Small measurement noise added to each feature
    batch  = np.column_stack([ranges, velocities, rcss, snrs])
    batch += rng.normal(0, 0.5, size=batch.shape)

    features_list.append(batch)
    labels_list.append(np.full(SAMPLES_PER_CLASS, label))

features = np.vstack(features_list).astype(np.float32)
labels   = np.concatenate(labels_list).astype(np.int64)

# Shuffle
idx      = rng.permutation(len(labels))
features = features[idx]
labels   = labels[idx]

os.makedirs("radar_sim", exist_ok=True)
np.savez("radar_sim/dataset.npz", features=features, labels=labels, class_names=CLASSES)

print(f"Dataset saved  → radar_sim/dataset.npz")
print(f"Total samples  : {len(labels)}")
print(f"Feature vector : {FEATURE_NAMES}")
print()
for i, cls in enumerate(CLASSES):
    print(f"  [{i}] {cls:<24}  {(labels == i).sum()} samples")
