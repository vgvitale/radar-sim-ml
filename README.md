# Radar Simulation + ML Classification Pipeline

A simple Python pipeline that simulates an airborne surveillance radar, trains a neural network to classify aircraft contacts, and tests the model against adversarial attacks.

---

## Object Classes

| Class | Range | Velocity | RCS |
|---|---|---|---|
| Commercial Aircraft | 50–400 km | 200–900 km/h | 20–40 dBsm |
| Fighter Jet | 30–300 km | 500–2000 km/h | 0–15 dBsm |
| Helicopter | 5–100 km | 0–300 km/h | 5–15 dBsm |
| Drone | 0.5–20 km | 0–150 km/h | −20–0 dBsm |

---

## Feature Vector

Each radar detection is represented as 4 values:

```
[range_km,  velocity_kmh,  rcs_dbsm,  snr_db]
```

**SNR** is derived from the simplified radar range equation: `SNR ∝ RCS / R⁴`

---

## Neural Network

3-layer MLP: **4 → 32 → 16 → 4**

- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Epochs: 50
- Features are normalized (mean/std saved in `scaler.npz`)

---

## Setup

```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
pip install -r requirements.txt
```

---

## Running the Pipeline

Run each script in order from the **project root**:

```bash
# 1. Generate the dataset
python radar_sim/01_simulate.py

# 2. Train the classifier
python radar_sim/02_train.py

# 3. Live inference loop (Ctrl+C to stop)
python radar_sim/03_infer.py

# 4. Adversarial attack demo
python radar_sim/04_attack.py
```

---

## Adversarial Attacks (04_attack.py)

The attack script uses **FGSM (Fast Gradient Sign Method)** — a one-step gradient-based perturbation that nudges each feature in the direction that maximises the model's loss.

### Tuning

At the top of `04_attack.py`:

```python
EPSILON = 0.3        # perturbation size in normalized space
                     # try 0.1 (subtle) → 1.0 (aggressive)

MANUAL_DELTA = None  # set to a list to manually shift features instead of FGSM
                     # e.g. [50.0, 0, 0, 0]  → shift range +50 km
                     #      [0, 500.0, 0, 0]  → shift velocity +500 km/h
```

### Interpreting Results

- **Fooled = YES** — the attack caused a misclassification.
- **Attack success rate** — % of contacts that were successfully fooled.
- A high success rate at low epsilon indicates the model has adversarial vulnerabilities near the decision boundary.

---

## Output Files

| File | Created by | Contents |
|---|---|---|
| `radar_sim/dataset.npz` | `01_simulate.py` | Features + labels |
| `radar_sim/scaler.npz` | `02_train.py` | Normalization mean + std |
| `radar_sim/model.pt` | `02_train.py` | Trained model weights |
