"""
04_attack.py — General adversarial attack demo.

Simulates an attacker who manipulates their aircraft's radar signature
(speed, altitude, RCS) to be misclassified — without any knowledge of
which classifier (ML or traditional) the radar is using.

Attack strategy — centroid-directed perturbation:
  For each contact, find the nearest alternative class centroid and shift
  the feature vector toward it by EPSILON steps (in normalised space).
  This mimics a real adversary changing their flight profile to look like
  a different aircraft type.

Tuning:
  - Increase EPSILON to make perturbations larger (easier to fool).
  - Set MANUAL_TARGET to a class index (0–3) to always attack toward
    that specific class instead of the nearest alternative.
    0 = Commercial Aircraft, 1 = Fighter Jet, 2 = Helicopter, 3 = Drone

Run: python radar_sim/04_attack.py
"""

import numpy as np
import torch

from radar_model import (
    CLASSES, CLASS_PARAMS, N_CLASSES, EPSILON, compute_snr,
    load_model, load_scaler, normalize, generate_sample
)

# ── Configurable ──────────────────────────────────────────────
# EPSILON is set in radar_model.py — change it there
N_CONTACTS    = 10
MANUAL_TARGET = None     # e.g. 3 to always attack toward Drone
RANDOM_SEED   = 0
# ─────────────────────────────────────────────────────────────

model     = load_model()
mean, std = load_scaler()
rng       = np.random.default_rng(RANDOM_SEED)


# ── Build class centroids (physical parameter midpoints) ──────
def build_centroids():
    centroids = np.zeros((N_CLASSES, 4), dtype=np.float32)
    for cls in range(N_CLASSES):
        p   = CLASS_PARAMS[cls]
        r   = (p["range"][0]    + p["range"][1])    / 2
        v   = (p["velocity"][0] + p["velocity"][1]) / 2
        rcs = (p["rcs"][0]      + p["rcs"][1])      / 2
        snr = compute_snr(np.array(rcs), np.array(r)).item()
        centroids[cls] = [r, v, rcs, snr]
    scales = centroids.max(axis=0) - centroids.min(axis=0)
    scales[scales == 0] = 1.0
    return centroids, scales

CENTROIDS, SCALES = build_centroids()

# Normalised centroids (for distance comparison in ML feature space)
CENTROIDS_NORM = (CENTROIDS - mean.numpy()) / std.numpy()


# ── General centroid-directed attack ──────────────────────────
def general_attack(raw_feat, current_cls):
    """
    Shift features toward the nearest alternative class centroid.
    Returns both the attacked normalised vector and raw feature vector.
    """
    x_norm = (raw_feat - mean.numpy()) / std.numpy()

    if MANUAL_TARGET is not None:
        target_cls = MANUAL_TARGET
    else:
        # Find nearest alternative centroid in normalised space
        dists = np.linalg.norm(CENTROIDS_NORM - x_norm, axis=1)
        dists[current_cls] = np.inf   # exclude current class
        target_cls = int(np.argmin(dists))

    # Direction from current position toward target centroid
    direction = CENTROIDS_NORM[target_cls] - x_norm
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    x_attacked_norm = x_norm + EPSILON * direction
    x_attacked_raw  = x_attacked_norm * std.numpy() + mean.numpy()
    return x_attacked_norm, x_attacked_raw, target_cls


# ── Classifier helpers ────────────────────────────────────────
def ml_predict(x_norm_np):
    x_t = torch.tensor(x_norm_np, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(model(x_t.unsqueeze(0)), dim=1)[0]
    return int(probs.argmax())

def trad_predict(raw_feat):
    dists = np.linalg.norm((raw_feat - CENTROIDS) / SCALES, axis=1)
    return int(np.argmin(dists))


# ── Run attack ────────────────────────────────────────────────
print("=" * 72)
print(f"  General Adversarial Attack Report   (ε={EPSILON}, n={N_CONTACTS})")
print(f"  Strategy: centroid-directed — no knowledge of classifier internals")
if MANUAL_TARGET is not None:
    print(f"  Fixed target class: {CLASSES[MANUAL_TARGET]}")
print("=" * 72)

ml_fooled   = 0
trad_fooled = 0

for i in range(N_CONTACTS):
    raw_feat, true_label = generate_sample(rng)
    x_norm = (raw_feat - mean.numpy()) / std.numpy()

    # ── Original predictions ──────────────────────────────────
    ml_orig   = ml_predict(x_norm)
    trad_orig = trad_predict(raw_feat)

    # ── Attack (directed toward nearest alternative centroid) ─
    x_atk_norm, x_atk_raw, target_cls = general_attack(raw_feat, ml_orig)

    # ── Post-attack predictions ───────────────────────────────
    ml_adv   = ml_predict(x_atk_norm)
    trad_adv = trad_predict(x_atk_raw)

    ml_fooled_flag   = ml_adv   != ml_orig
    trad_fooled_flag = trad_adv != trad_orig
    if ml_fooled_flag:
        ml_fooled   += 1
    if trad_fooled_flag:
        trad_fooled += 1

    print(
        f"\n  Contact {i + 1}  —  True: {CLASSES[true_label]}  "
        f"→  Attack target: {CLASSES[target_cls]}"
    )
    print(f"  {'Classifier':<14}  {'Before':<22}  {'After':<22}  Fooled?")
    print(f"  {'-' * 68}")
    print(
        f"  {'ML':<14}  {CLASSES[ml_orig]:<22}  "
        f"{CLASSES[ml_adv]:<22}  {'YES ✓' if ml_fooled_flag else 'no'}"
    )
    print(
        f"  {'Traditional':<14}  {CLASSES[trad_orig]:<22}  "
        f"{CLASSES[trad_adv]:<22}  {'YES ✓' if trad_fooled_flag else 'no'}"
    )

print("\n" + "=" * 72)
print(f"  Fooled — ML Classifier  : {ml_fooled}/{N_CONTACTS} "
      f"({100 * ml_fooled / N_CONTACTS:.0f}%)")
print(f"  Fooled — Traditional    : {trad_fooled}/{N_CONTACTS} "
      f"({100 * trad_fooled / N_CONTACTS:.0f}%)")
print("=" * 72)
print()
print("Tips:")
print("  • Raise EPSILON (e.g. 1.0) to make perturbations more aggressive.")
print("  • Set MANUAL_TARGET = 3 to always push contacts toward Drone profile.")
