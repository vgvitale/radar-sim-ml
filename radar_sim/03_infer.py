"""
03_infer.py — Real-time radar classification loop.

Generates a new radar frame every FRAME_INTERVAL_SEC seconds, classifies
each contact, and prints the result. Press Ctrl+C to stop.

Run: python radar_sim/03_infer.py
"""

import sys
import time
import numpy as np
import torch

from radar_model import (
    CLASSES, load_model, load_scaler, normalize, generate_frame
)

# ── Configuration ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70   # flag contacts below this confidence
FRAME_INTERVAL_SEC   = 1.5    # seconds between radar sweeps
# ─────────────────────────────────────────────────────────────

model      = load_model()
mean, std  = load_scaler()
rng        = np.random.default_rng()


def classify_frame(frame: np.ndarray):
    """Return softmax probability arrays for each contact in the frame."""
    x      = torch.tensor(frame)
    x_norm = normalize(x, mean, std)
    with torch.no_grad():
        probs = torch.softmax(model(x_norm), dim=1)
    return probs.numpy()


print("=" * 65)
print("  Radar Surveillance System — Live Inference")
print("  Press Ctrl+C to stop")
print("=" * 65)
sys.stdout.flush()

frame_num = 0
try:
    while True:
        frame_num += 1
        frame, _ = generate_frame(rng=rng)
        probs    = classify_frame(frame)

        print(f"\n[ Frame {frame_num} — {len(frame)} contact(s) detected ]")
        for i, (feat, prob) in enumerate(zip(frame, probs)):
            cls_idx    = int(prob.argmax())
            confidence = float(prob[cls_idx])
            flag       = "" if confidence >= CONFIDENCE_THRESHOLD else "  ⚠ LOW CONFIDENCE"

            print(
                f"  Contact {i + 1}: {CLASSES[cls_idx]:<22} "
                f"conf={confidence:.0%}{flag}\n"
                f"             range={feat[0]:>7.1f} km  "
                f"vel={feat[1]:>6.0f} km/h  "
                f"RCS={feat[2]:>5.1f} dBsm  "
                f"SNR={feat[3]:>5.1f} dB"
            )
        sys.stdout.flush()
        time.sleep(FRAME_INTERVAL_SEC)

except KeyboardInterrupt:
    print("\n\nSimulation stopped.")
