"""
src/train_model.py
==================
Synthetic dataset generation, Random Forest training, and model export.

Run directly:
    python src/train_model.py

Outputs (project root)
-----------------------
  posture_rf_model.pkl   - Trained RandomForestClassifier + label mapping
  data/synthetic_posture_data.csv - Generated dataset for reproducibility
"""

from __future__ import annotations

import os
import sys

# Allow  `python src/train_model.py`  from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.features import FEATURE_COLUMNS

# ── Reproducibility ──────────────────────────────────────────────────────────
RNG_SEED    = 42
N_SAMPLES   = 2_000          # samples per class
TEST_SPLIT  = 0.20           # 80 / 20 train-test split
MODEL_PATH  = os.path.join(PROJECT_ROOT, "posture_rf_model.pkl")
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
DATA_PATH   = os.path.join(DATA_DIR, "synthetic_posture_data.csv")

# ── Posture class definitions ────────────────────────────────────────────────
# Format: { label: { feature: (mean, std) } }
# Values are plausible biomechanical angles (degrees) + normalised coordinates.
# Coordinates are centred around a "sitting at a desk" anatomical prior.
CLASS_SPECS: dict[str, dict[str, tuple[float, float]]] = {
    # ── Good Posture ──────────────────────────────────────────────────────────
    # Neck straight (~90 deg), torso upright (~90 deg), shoulders level (~0 deg).
    "Good Posture": {
        "neck_flexion_deg":       (88.0, 3.5),
        "trunk_inclination_deg":  (88.0, 3.5),
        "shoulder_asymmetry_deg": ( 0.5, 1.5),
    },

    # ── Slouching (Trunk Inclination) ─────────────────────────────────────────
    # Torso noticeably forward; neck follows partially.
    "Slouching": {
        "neck_flexion_deg":       (78.0, 4.5),
        "trunk_inclination_deg":  (72.0, 5.0),
        "shoulder_asymmetry_deg": ( 1.5, 2.5),
    },

    # ── Forward Head Posture (Neck Flexion) ───────────────────────────────────
    # Head protrudes forward; trunk relatively upright.
    "Forward Head Posture": {
        "neck_flexion_deg":       (65.0, 5.0),
        "trunk_inclination_deg":  (84.0, 4.0),
        "shoulder_asymmetry_deg": ( 1.0, 2.0),
    },

    # ── Asymmetric Tilt (Shoulder Slope) ─────────────────────────────────────
    # Shoulders visibly tilted left or right; angles otherwise moderate.
    "Asymmetric Tilt": {
        "neck_flexion_deg":       (84.0, 4.5),
        "trunk_inclination_deg":  (84.0, 4.5),
        "shoulder_asymmetry_deg": (10.0, 3.5),  # large tilt
    },
}

# Shared coordinate priors for a person sitting ~60 cm from the camera
# (MediaPipe normalised coords, 0-1 range)
COORD_PRIORS: dict[str, tuple[float, float]] = {
    "left_ear_x":        (0.42, 0.03),
    "left_ear_y":        (0.22, 0.03),
    "right_ear_x":       (0.58, 0.03),
    "right_ear_y":       (0.22, 0.03),
    "left_shoulder_x":   (0.37, 0.03),
    "left_shoulder_y":   (0.42, 0.03),
    "right_shoulder_x":  (0.63, 0.03),
    "right_shoulder_y":  (0.42, 0.03),
    "left_hip_x":        (0.40, 0.03),
    "left_hip_y":        (0.72, 0.04),
    "right_hip_x":       (0.60, 0.03),
    "right_hip_y":       (0.72, 0.04),
}


# ── Dataset generation ───────────────────────────────────────────────────────

def generate_dataset(n_per_class: int = N_SAMPLES, seed: int = RNG_SEED) -> pd.DataFrame:
    """
    Synthesise a balanced dataset with *n_per_class* samples for each of the
    four posture classes.

    Each sample has:
      • 3 biomechanical angles (class-specific Gaussian distribution)
      • 12 normalised MediaPipe landmark coordinates (shared prior)
      • 1 'label' column (string class name)

    Returns
    -------
    pd.DataFrame
        Shuffled dataset, shape (4 * n_per_class, len(FEATURE_COLUMNS) + 1).
    """
    rng  = np.random.default_rng(seed)
    rows = []

    for label, angle_specs in CLASS_SPECS.items():
        for _ in range(n_per_class):
            row: dict[str, float | str] = {}

            # Biomechanical angle features (class-specific distribution)
            for feat, (mu, sigma) in angle_specs.items():
                # Asymmetric Tilt shoulder angle can be left or right tilt
                if label == "Asymmetric Tilt" and feat == "shoulder_asymmetry_deg":
                    sign = rng.choice([-1.0, 1.0])
                    row[feat] = float(sign * abs(rng.normal(mu, sigma)))
                else:
                    row[feat] = float(rng.normal(mu, sigma))

            # Coordinate features (shared anatomical prior)
            for feat, (mu, sigma) in COORD_PRIORS.items():
                # Clip to [0, 1] to stay within normalised image space
                val = rng.normal(mu, sigma)
                row[feat] = float(np.clip(val, 0.0, 1.0))

            row["label"] = label
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ── Training ─────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> tuple[RandomForestClassifier, list[str], dict]:
    """
    Train a Random Forest classifier and return the model, class list, and
    a dictionary of evaluation metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with FEATURE_COLUMNS + 'label'.

    Returns
    -------
    model      : Fitted RandomForestClassifier.
    classes    : Ordered list of class labels.
    metrics    : dict with accuracy, precision, recall, f1, confusion_matrix.
    """
    # Explicitly cast to numpy to avoid pandas/pyarrow indexing issue with sklearn
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy()
    classes = sorted(df["label"].unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        random_state=RNG_SEED,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,          # grow full trees
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced", # handles any residual imbalance
        random_state=RNG_SEED,
        n_jobs=-1,
    )
    X_train_df = pd.DataFrame(X_train, columns=FEATURE_COLUMNS)
    X_test_df = pd.DataFrame(X_test, columns=FEATURE_COLUMNS)
    model.fit(X_train_df, y_train)

    # ── Evaluation ──────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_df)

    metrics = {
        "accuracy":          accuracy_score(y_test, y_pred),
        "precision_macro":   precision_score(y_test, y_pred, average="macro",  zero_division=0),
        "recall_macro":      recall_score(   y_test, y_pred, average="macro",  zero_division=0),
        "f1_macro":          f1_score(       y_test, y_pred, average="macro",  zero_division=0),
        "confusion_matrix":  confusion_matrix(y_test, y_pred, labels=classes),
        "classification_report": classification_report(
            y_test, y_pred, target_names=classes, zero_division=0
        ),
        "classes": classes,
        "n_train": len(X_train),
        "n_test":  len(X_test),
    }

    return model, classes, metrics


# ── Persistence ──────────────────────────────────────────────────────────────

def save_model(model: RandomForestClassifier, classes: list[str], path: str) -> None:
    """Serialise the model and its class list to *path* using joblib."""
    artefact = {"model": model, "classes": classes}
    joblib.dump(artefact, path, compress=3)
    print(f"[[OK]] Model saved -> {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Ergonomic Posture Monitor - Model Training")
    print("=" * 60)

    # 1. Generate dataset
    print(f"\n[1/4] Generating synthetic dataset ({N_SAMPLES} samples/class)...")
    df = generate_dataset()
    print(f"      Total samples : {len(df)}")
    print(f"      Class balance :\n{df['label'].value_counts().to_string()}")

    # 2. Save dataset
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"\n[2/4] Dataset saved -> {DATA_PATH}")

    # 3. Train
    print("\n[3/4] Training Random Forest classifier...")
    model, classes, metrics = train(df)

    # 4. Print metrics
    print("\n[4/4] Evaluation metrics (held-out test set)")
    print("-" * 50)
    print(f"  Samples - train: {metrics['n_train']}  test: {metrics['n_test']}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f} %)")
    print(f"  Precision  : {metrics['precision_macro']:.4f}")
    print(f"  Recall     : {metrics['recall_macro']:.4f}")
    print(f"  F1-Score   : {metrics['f1_macro']:.4f}")
    print("\n  Confusion Matrix (rows = actual, cols = predicted):")
    print(f"  Classes: {metrics['classes']}")
    print(metrics["confusion_matrix"])
    print("\n  Per-class report:")
    print(metrics["classification_report"])

    # 5. Save model
    save_model(model, classes, MODEL_PATH)

    # Performance gate check
    if metrics["accuracy"] >= 0.90:
        print(f"\n[[OK]] Accuracy target met: {metrics['accuracy']*100:.2f}% >= 90.00%")
    else:
        print(f"\n[✗] Accuracy below target: {metrics['accuracy']*100:.2f}% < 90.00%")

    print("\nDone.")


if __name__ == "__main__":
    main()