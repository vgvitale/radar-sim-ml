"""
05_traditional.py — Traditional (algorithm-based) radar classifier — live inference.

Uses a nearest-centroid rule: each contact is assigned to the class whose
expected feature centroid it is closest to (after per-feature normalisation).
No training required — the centroids are derived directly from the known
physical parameter ranges in radar_model.py.

Run: python radar_sim/05_traditional.py
"""

import sys
import time
import numpy as np

from radar_model import (
    CLASSES, CLASS_PARAMS, N_CLASSES, compute_snr, generate_frame
)

# ── Configuration ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70   # minimum score to avoid LOW CONFIDENCE flag
FRAME_INTERVAL_SEC   = 1.5
# ─────────────────────────────────────────────────────────────


def build_centroids():
    """
    Derive one centroid per class from the known physical parameter ranges.

    For range, velocity, and RCS we take the midpoint of each class's range.
    For SNR we compute the expected value at the midpoint range and RCS.

    Returns
    -------
    centroids : np.ndarray  shape (N_CLASSES, 4)
    scales    : np.ndarray  shape (4,)  per-feature normalisation widths
    """
    centroids = np.zeros((N_CLASSES, 4), dtype=np.float32)

    for cls in range(N_CLASSES):
        p = CLASS_PARAMS[cls]
        r   = (p["range"][0]    + p["range"][1])    / 2
        v   = (p["velocity"][0] + p["velocity"][1]) / 2
        rcs = (p["rcs"][0]      + p["rcs"][1])      / 2
        snr = compute_snr(np.array(rcs), np.array(r)).item()
        centroids[cls] = [r, v, rcs, snr]

    # Normalisation scale = overall range of each feature across all classes
    scales = centroids.max(axis=0) - centroids.min(axis=0)
    scales[scales == 0] = 1.0   # avoid division by zero
    return centroids, scales


CENTROIDS, SCALES = build_centroids()


def classify_traditional(features):
    """
    Classify one or more contacts using nearest-centroid distance.

    Parameters
    ----------
    features : np.ndarray  shape (n, 4) or (4,)

    Returns
    -------
    predictions  : list[int]   predicted class indices
    confidences  : list[float] confidence scores in [0, 1]
    """
    single = features.ndim == 1
    if single:
        features = features[np.newaxis, :]

    predictions, confidences = [], []
    for feat in features:
        # Normalised distance to each centroid
        dists = np.linalg.norm(
            (feat - CENTROIDS) / SCALES, axis=1
        )
        cls  = int(np.argmin(dists))
        # Convert distances to a confidence-like score using softmin
        neg_d = -dists
        conf  = float(np.exp(neg_d[cls]) / np.exp(neg_d).sum())
        predictions.append(cls)
        confidences.append(conf)

    if single:
        return predictions[0], confidences[0]
    return predictions, confidences


# ── Live inference loop ───────────────────────────────────────
rng = np.random.default_rng()

print("=" * 65)
print("  Traditional Radar Classifier — Live Inference")
print("  (Nearest-Centroid Algorithm)")
print("  Press Ctrl+C to stop")
print("=" * 65)
sys.stdout.flush()

frame_num = 0
try:
    while True:
        frame_num += 1
        frame, _ = generate_frame(rng=rng)
        preds, confs = classify_traditional(frame)

        print(f"\n[ Frame {frame_num} — {len(frame)} contact(s) detected ]")
        for i, (feat, cls_idx, conf) in enumerate(zip(frame, preds, confs)):
            flag = "" if conf >= CONFIDENCE_THRESHOLD else "  ⚠ LOW CONFIDENCE"
            print(
                f"  Contact {i + 1}: {CLASSES[cls_idx]:<22} "
                f"conf={conf:.0%}{flag}\n"
                f"             range={feat[0]:>7.1f} km  "
                f"vel={feat[1]:>6.0f} km/h  "
                f"RCS={feat[2]:>5.1f} dBsm  "
                f"SNR={feat[3]:>5.1f} dB"
            )
        sys.stdout.flush()
        time.sleep(FRAME_INTERVAL_SEC)

except KeyboardInterrupt:
    print("\n\nSimulation stopped.")
