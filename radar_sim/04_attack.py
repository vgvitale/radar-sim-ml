"""
04_attack.py — Adversarial noise injection demo.

For each contact in a generated radar frame:
  1. Show the original classification.
  2. Inject adversarial noise (FGSM or manual delta).
  3. Show the new classification and whether the model was fooled.

Tuning:
  - Increase EPSILON to make FGSM perturbations larger (easier to fool).
  - Set MANUAL_DELTA to a list of 4 floats to manually shift features
    instead of using FGSM.
    Example: [50.0, 0, 0, 0]   → shift range by +50 km
             [0, 500.0, 0, 0]  → shift velocity by +500 km/h

Run: python radar_sim/04_attack.py
"""

import numpy as np
import torch
import torch.nn as nn

from radar_model import (
    CLASSES, load_model, load_scaler, normalize, generate_sample
)

# ── Configurable ──────────────────────────────────────────────
EPSILON      = 0.3        # FGSM perturbation magnitude (normalized space)
N_CONTACTS   = 10         # number of contacts to attack
MANUAL_DELTA = None       # e.g. [50.0, 200.0, 0.0, 0.0] for manual injection
RANDOM_SEED  = 0
# ─────────────────────────────────────────────────────────────

model      = load_model()
mean, std  = load_scaler()
rng        = np.random.default_rng(RANDOM_SEED)


def predict(x_norm: torch.Tensor) -> np.ndarray:
    """Return softmax probabilities for a single normalized feature vector."""
    with torch.no_grad():
        probs = torch.softmax(model(x_norm.unsqueeze(0)), dim=1)[0]
    return probs.numpy()


def fgsm_attack(x_raw: torch.Tensor, true_label: int) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM).

    Perturbs the normalized input in the direction that maximises
    cross-entropy loss, scaled by EPSILON.
    """
    x_norm = normalize(x_raw, mean, std)
    x_adv  = x_norm.clone().detach().requires_grad_(True)

    loss = nn.CrossEntropyLoss()(
        model(x_adv.unsqueeze(0)),
        torch.tensor([true_label])
    )
    loss.backward()

    perturbation = EPSILON * x_adv.grad.sign()
    return (x_norm + perturbation).detach()


# ── Run attack ────────────────────────────────────────────────
print("=" * 72)
print(f"  Adversarial Attack Report   (ε={EPSILON}, n={N_CONTACTS})")
if MANUAL_DELTA:
    print(f"  Mode: MANUAL DELTA  Δ={MANUAL_DELTA}")
else:
    print(f"  Mode: FGSM — increase EPSILON to force more misclassifications")
print("=" * 72)
print(
    f"\n{'#':<3}  {'True Class':<22}  {'Original':>22}  "
    f"{'After Attack':>22}  {'Fooled?':>7}"
)
print("─" * 80)

successes = 0

for i in range(N_CONTACTS):
    raw_feat, true_label = generate_sample(rng)
    x_raw  = torch.tensor(raw_feat)
    x_norm = normalize(x_raw, mean, std)

    # Original prediction
    orig_prob = predict(x_norm)
    orig_cls  = int(orig_prob.argmax())
    orig_conf = float(orig_prob[orig_cls])

    # Attack
    if MANUAL_DELTA is not None:
        delta      = torch.tensor(MANUAL_DELTA, dtype=torch.float32)
        x_attacked = x_norm + delta
    else:
        x_attacked = fgsm_attack(x_raw, true_label)

    adv_prob = predict(x_attacked)
    adv_cls  = int(adv_prob.argmax())
    adv_conf = float(adv_prob[adv_cls])

    fooled = adv_cls != orig_cls
    if fooled:
        successes += 1

    marker = "YES ✓" if fooled else "no"
    print(
        f"{i + 1:<3}  {CLASSES[true_label]:<22}  "
        f"{CLASSES[orig_cls]:>14} ({orig_conf:.0%})  →  "
        f"{CLASSES[adv_cls]:>14} ({adv_conf:.0%})  "
        f"{marker:>7}"
    )

print("─" * 80)
print(f"\nAttack success rate : {successes}/{N_CONTACTS}  "
      f"({100 * successes / N_CONTACTS:.0f}%)")
print(f"Epsilon (ε)         : {EPSILON}")
print()
print("Tips:")
print("  • Raise EPSILON (e.g. 1.0) to make attacks more aggressive.")
print("  • Set MANUAL_DELTA = [50.0, 0, 0, 0] to manually shift the range feature.")
print("  • Set MANUAL_DELTA = [0, 500.0, 0, 0] to manually shift velocity.")
