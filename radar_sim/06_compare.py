"""
06_compare.py — Compare ML classifier vs. Traditional (nearest-centroid) classifier.

Produces two side-by-side bar charts saved to radar_sim/comparison_results.png:

  Chart 1: Per-class accuracy on clean data
  Chart 2: Per-class accuracy under general adversarial attack
            (centroid-directed perturbation — no model knowledge assumed)

Also prints a summary table to the console.

Prerequisites: run 01_simulate.py and 02_train.py first.

Run:
  python radar_sim/06_compare.py                  # uses default epsilon
  python radar_sim/06_compare.py --epsilon 1.0    # custom attack strength
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from radar_model import (
    CLASSES, CLASS_PARAMS, N_CLASSES, EPSILON as DEFAULT_EPSILON, compute_snr,
    load_model, load_scaler, normalize
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epsilon", type=float, default=DEFAULT_EPSILON,
    help=f"Attack perturbation strength (default: {DEFAULT_EPSILON})"
)
args    = parser.parse_args()
EPSILON = args.epsilon

print(f"Attack strength (epsilon): {EPSILON}")

# ── Load held-out test set (20 %) ─────────────────────────────
data     = np.load("radar_sim/dataset.npz", allow_pickle=True)
features = data["features"]
labels   = data["labels"].astype(int)

rng      = np.random.default_rng(99)
idx      = rng.permutation(len(labels))
split    = int(0.8 * len(idx))
test_idx = idx[split:]
X_test   = features[test_idx]
y_test   = labels[test_idx]

print(f"Test set: {len(y_test)} samples")

# ── Load ML model ─────────────────────────────────────────────
model     = load_model()
mean, std = load_scaler()

# ── Build class centroids ─────────────────────────────────────
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

CENTROIDS, SCALES   = build_centroids()
mean_np             = mean.numpy()
std_np              = std.numpy()
CENTROIDS_NORM      = (CENTROIDS - mean_np) / std_np


# ── Classifiers ───────────────────────────────────────────────
def ml_predict_batch(X_raw):
    with torch.no_grad():
        x_norm = normalize(torch.tensor(X_raw), mean, std)
        return model(x_norm).argmax(1).numpy().astype(int)

def trad_predict_batch(X_raw):
    preds = []
    for feat in X_raw:
        dists = np.linalg.norm((feat - CENTROIDS) / SCALES, axis=1)
        preds.append(int(np.argmin(dists)))
    return np.array(preds)


# ── General centroid-directed attack ──────────────────────────
def attack_batch(X_raw, current_preds):
    """
    Shift each sample toward the nearest alternative class centroid.
    Returns attacked raw features.
    """
    X_atk_raw = np.zeros_like(X_raw)
    for i, (feat, cur_cls) in enumerate(zip(X_raw, current_preds)):
        x_norm = (feat - mean_np) / std_np
        dists  = np.linalg.norm(CENTROIDS_NORM - x_norm, axis=1)
        dists[cur_cls] = np.inf
        target    = int(np.argmin(dists))
        direction = CENTROIDS_NORM[target] - x_norm
        norm      = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        x_atk_norm       = x_norm + EPSILON * direction
        X_atk_raw[i]     = x_atk_norm * std_np + mean_np
    return X_atk_raw


# ── Per-class accuracy ────────────────────────────────────────
def per_class_accuracy(y_true, y_pred):
    accs = []
    for cls in range(N_CLASSES):
        mask = y_true == cls
        accs.append((y_pred[mask] == cls).mean() * 100 if mask.sum() > 0 else 0.0)
    return np.array(accs)


# ── Evaluate — clean data ─────────────────────────────────────
ml_clean   = ml_predict_batch(X_test)
trad_clean = trad_predict_batch(X_test)

ml_clean_acc   = per_class_accuracy(y_test, ml_clean)
trad_clean_acc = per_class_accuracy(y_test, trad_clean)

# ── Evaluate — under attack ───────────────────────────────────
X_atk      = attack_batch(X_test, ml_clean)
ml_atk     = ml_predict_batch(X_atk)
trad_atk   = trad_predict_batch(X_atk)

ml_atk_acc   = per_class_accuracy(y_test, ml_atk)
trad_atk_acc = per_class_accuracy(y_test, trad_atk)

# ── Console summary ───────────────────────────────────────────
SHORT = ["Comm. Aircraft", "Fighter Jet", "Helicopter", "Drone"]

print()
print("=" * 68)
print(f"  {'':22}  {'— Clean —':>20}  {'— Under Attack —':>20}")
print(f"  {'Class':<22}  {'ML':>9}  {'Traditional':>9}  {'ML':>9}  {'Traditional':>9}")
print("=" * 68)
for i, cls in enumerate(SHORT):
    print(
        f"  {cls:<22}  {ml_clean_acc[i]:>8.1f}%  {trad_clean_acc[i]:>8.1f}%"
        f"  {ml_atk_acc[i]:>8.1f}%  {trad_atk_acc[i]:>8.1f}%"
    )
print("-" * 68)
print(
    f"  {'Overall':<22}  "
    f"{ml_clean_acc.mean():>8.1f}%  {trad_clean_acc.mean():>8.1f}%"
    f"  {ml_atk_acc.mean():>8.1f}%  {trad_atk_acc.mean():>8.1f}%"
)
print("=" * 68)

# ── Charts ────────────────────────────────────────────────────
BLUE  = "#4C72B0"
GREEN = "#55A868"
x     = np.arange(N_CLASSES)
w     = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "ML Classifier vs. Traditional Classifier",
    fontsize=13, fontweight="bold", y=1.02
)

def bar_chart(ax, ml_vals, trad_vals, title):
    b1 = ax.bar(x - w / 2, ml_vals,   w, label="ML Classifier",  color=BLUE,  alpha=0.9)
    b2 = ax.bar(x + w / 2, trad_vals, w, label="Traditional",     color=GREEN, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 1,
            f"{h:.0f}%", ha="center", va="bottom", fontsize=8
        )

bar_chart(ax1, ml_clean_acc,   trad_clean_acc, "Clean Data")
bar_chart(ax2, ml_atk_acc,     trad_atk_acc,   f"Under Attack  (ε={EPSILON})")

plt.tight_layout()
out_path = "radar_sim/comparison_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nChart saved → {out_path}")
