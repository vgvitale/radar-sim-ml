"""
radar_model.py — Shared model, constants, and utilities for the radar pipeline.

Imported by: 01_simulate.py, 02_train.py, 03_infer.py, 04_attack.py
"""

from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn

# ── Attack configuration ──────────────────────────────────────
EPSILON = 2.3   # adversarial perturbation strength — change here to affect
                # both 04_attack.py and 06_compare.py

# ── Object classes ────────────────────────────────────────────
CLASSES = ["Commercial Aircraft", "Fighter Jet", "Helicopter", "Drone"]

# Physical parameter ranges per class
# Keys: range_km, velocity_kmh, rcs_dbsm  (all uniform distributions)
CLASS_PARAMS = {
    0: dict(range=(50, 400),  velocity=(200, 900),  rcs=(20, 40)),   # Commercial Aircraft
    1: dict(range=(30, 300),  velocity=(500, 2000), rcs=(0, 15)),    # Fighter Jet
    2: dict(range=(5, 100),   velocity=(0, 300),    rcs=(5, 15)),    # Helicopter
    3: dict(range=(0.5, 20),  velocity=(0, 150),    rcs=(-20, 0)),   # Drone
}

# Feature vector layout: [range_km, velocity_kmh, rcs_dbsm, snr_db]
FEATURE_NAMES = ["range_km", "velocity_kmh", "rcs_dbsm", "snr_db"]
N_FEATURES    = len(FEATURE_NAMES)
N_CLASSES     = len(CLASSES)


# ── Radar physics ─────────────────────────────────────────────
def compute_snr(rcs_dbsm, range_km):
    """
    Simplified radar range equation.

    SNR (dB) is proportional to RCS / R^4.
    A constant offset (+120) shifts values into a readable positive range.
    """
    rcs_linear = 10 ** (rcs_dbsm / 10)
    snr_linear = rcs_linear / (range_km ** 4)
    return 10 * np.log10(snr_linear + 1e-10) + 120


# ── Neural network ────────────────────────────────────────────
class RadarMLP(nn.Module):
    """
    3-layer MLP classifier.
    Input : 4 radar features  [range_km, velocity_kmh, rcs_dbsm, snr_db]
    Output: 4 class logits    [Commercial Aircraft, Fighter Jet, Helicopter, Drone]
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


# ── Helpers ───────────────────────────────────────────────────
def load_model(path="radar_sim/model.pt"):
    """Load a saved RadarMLP from disk and set to eval mode."""
    model = RadarMLP()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def load_scaler(path="radar_sim/scaler.npz"):
    """Return (mean, std) as float32 torch tensors."""
    scaler = np.load(path)
    return torch.tensor(scaler["mean"]), torch.tensor(scaler["std"])


def normalize(x, mean, std):
    return (x - mean) / (std + 1e-8)


def generate_sample(rng=None):
    """
    Generate one random radar contact.

    Returns
    -------
    features : np.ndarray  shape (4,)  [range_km, velocity_kmh, rcs_dbsm, snr_db]
    label    : int          class index
    """
    if rng is None:
        rng = np.random.default_rng()
    cls = rng.integers(0, N_CLASSES)
    p   = CLASS_PARAMS[cls]
    r   = rng.uniform(*p["range"])
    v   = rng.uniform(*p["velocity"])
    rcs = rng.uniform(*p["rcs"])
    snr = compute_snr(np.array(rcs), np.array(r)).item()
    return np.array([r, v, rcs, snr], dtype=np.float32), int(cls)


def generate_frame(n_contacts=None, rng=None):
    """
    Generate a radar sweep frame.

    Always includes one contact from each of the 4 classes so every class
    is represented, then appends 0-2 additional random contacts.
    Total contacts: 4 + extras (or exactly n_contacts if specified).

    Returns
    -------
    frame  : np.ndarray  shape (n, 4)
    labels : list[int]   true class index for each contact
    """
    if rng is None:
        rng = np.random.default_rng()

    contacts, labels = [], []

    if n_contacts is not None:
        for _ in range(n_contacts):
            feat, cls = generate_sample(rng)
            contacts.append(feat)
            labels.append(cls)
    else:
        # Guarantee one contact per class
        for cls in range(N_CLASSES):
            p   = CLASS_PARAMS[cls]
            r   = rng.uniform(*p["range"])
            v   = rng.uniform(*p["velocity"])
            rcs = rng.uniform(*p["rcs"])
            snr = compute_snr(np.array(rcs), np.array(r)).item()
            contacts.append(np.array([r, v, rcs, snr], dtype=np.float32))
            labels.append(cls)
        # Add 0-2 extra random contacts
        for _ in range(int(rng.integers(0, 3))):
            feat, cls = generate_sample(rng)
            contacts.append(feat)
            labels.append(cls)

    # Shuffle so class order is not predictable
    idx      = rng.permutation(len(contacts))
    contacts = [contacts[i] for i in idx]
    labels   = [labels[i]   for i in idx]

    return np.array(contacts, dtype=np.float32), labels
