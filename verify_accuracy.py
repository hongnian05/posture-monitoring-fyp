"""
verify_accuracy.py
==================
Simple verification script to calculate and display the exact model accuracy,
precision, recall, F1-score, and confusion matrix on the held-out test dataset.

Run:
    python verify_accuracy.py
"""

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.features import FEATURE_COLUMNS


def main():
    print("=" * 60)
    print("  VERIFYING MODEL ACCURACY ON HELD-OUT TEST DATA")
    print("=" * 60)

    # 1. Load the trained Random Forest model
    model_path = "posture_rf_model.pkl"
    print(f"\n[1] Loading trained model from: {model_path}")
    artefact = joblib.load(model_path)
    model = artefact["model"]
    classes = artefact["classes"]

    # 2. Load the dataset (8,000 samples)
    data_path = "data/synthetic_posture_data.csv"
    print(f"[2] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    print(f"    Total samples: {len(df)} across 4 posture classes")

    # 3. Create the exact same 80/20 train/test split (random_state=42)
    #    This ensures the model is tested on data it has NEVER seen during training.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[3] Test set size: {len(X_test)} samples (20% held-out)")

    # 4. Predict using the model
    print("\n[4] Running model predictions on test samples...")
    y_pred = model.predict(X_test)

    # 5. Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"  >>> CONFIRMED ACCURACY: {accuracy * 100:.2f}% <<<")
    print("=" * 60)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    print("Confusion Matrix (Rows = Actual, Columns = Predicted):")
    cm_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=classes),
        index=classes,
        columns=classes,
    )
    print(cm_df)

    correct_predictions = (y_test == y_pred).sum()
    total_test = len(y_test)
    print("\nCalculation Breakdown:")
    print(f"  Correct Predictions : {correct_predictions}")
    print(f"  Total Test Samples  : {total_test}")
    print(f"  Formula             : {correct_predictions} / {total_test} = {accuracy:.4f} ({accuracy * 100:.2f}%)")


if __name__ == "__main__":
    main()
