"""
data/training/train_model.py
===============================

Trains the edge-case classifier that src/agent/ml_reasoner.py serves at
inference time.

Pipeline:
  1. For each of the 10 dummy batches, run the REAL production deterministic
     matcher (src/deterministic/matcher.py) over that batch's
     merchant_sales.json / razorpay_payouts.json. This deliberately reuses
     the exact same code path as the live pipeline instead of a parallel
     "training-only" matcher, so the training data is generated the same
     way inference data will be.
  2. Join the resulting `unresolved` records against that batch's
     labels.json (payout_id -> ground-truth archetype). Unresolved records
     with no label are dropped (they'd only be NO_PAYOUT_FOUND / ghost-order
     cases, which are a data-integrity issue, not a reasoning-classifier
     problem).
  3. Extract features via src/agent/features.py (same function the live
     agent will call).
  4. Split by BATCH, not by row: batches 0-7 -> train, batches 8-9 -> held
     out test set. This is stricter than a random row-level split — it
     tests whether the model generalizes to a batch of orders it has never
     seen any example from at all, not just unseen rows interleaved with
     seen ones.
  5. Train a small RandomForestClassifier (kept shallow/interpretable via
     feature_importances_) and report accuracy, a full classification
     report, and a confusion matrix on the held-out batches.
  6. Save the trained model + label encoder to
     data/training/edge_case_model.joblib and write a human-readable
     data/training/model_report.md.

Run with:
    python -m data.training.train_model
"""
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from src.agent.features import FEATURE_NAMES, extract_features
from src.deterministic.matcher import run_deterministic_match
from src.models import UnresolvedRecord

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parents[1]
NUM_BATCHES = 10
TEST_BATCH_INDICES = {8, 9}  # held out entirely from training

MODEL_PATH = TRAINING_DIR / "edge_case_model.joblib"
REPORT_PATH = TRAINING_DIR / "model_report.md"


def _load_batch_examples(batch_index: int) -> Tuple[List[List[float]], List[str], List[str]]:
    """Returns (feature_rows, labels, payout_ids) for one batch, using the
    real production matcher to find its unresolved records."""
    batch_dir = TRAINING_DIR / f"batch_{batch_index:02d}"
    labels = __import__("json").loads((batch_dir / "labels.json").read_text(encoding="utf-8"))

    result = run_deterministic_match(
        batch_dir / "merchant_sales.json", batch_dir / "razorpay_payouts.json"
    )

    rows: List[List[float]] = []
    ys: List[str] = []
    ids: List[str] = []
    for record in result.unresolved:
        label = labels.get(record.payout_id)
        if label is None:
            continue  # e.g. NO_PAYOUT_FOUND ghost records — not a reasoning-model concern
        rows.append(extract_features(record))
        ys.append(label)
        ids.append(record.payout_id)
    return rows, ys, ids


def build_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    train_X, train_y = [], []
    test_X, test_y = [], []

    for i in range(NUM_BATCHES):
        rows, ys, _ids = _load_batch_examples(i)
        if i in TEST_BATCH_INDICES:
            test_X.extend(rows)
            test_y.extend(ys)
        else:
            train_X.extend(rows)
            train_y.extend(ys)

    return (
        np.array(train_X, dtype=float),
        np.array(train_y, dtype=object),
        np.array(test_X, dtype=float),
        np.array(test_y, dtype=object),
        FEATURE_NAMES,
        sorted(set(list(train_y) + list(test_y))),
    )


def train_and_evaluate() -> str:
    X_train, y_train, X_test, y_test, feature_names, all_labels = build_dataset()

    encoder = LabelEncoder()
    encoder.fit(all_labels)
    y_train_enc = encoder.transform(y_train)
    y_test_enc = encoder.transform(y_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,           # kept shallow on purpose: interpretability over squeezing accuracy
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train_enc)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train_enc, train_pred)
    test_acc = accuracy_score(y_test_enc, test_pred)

    test_report = classification_report(
        y_test_enc, test_pred, target_names=encoder.classes_, zero_division=0
    )
    cm = confusion_matrix(y_test_enc, test_pred)

    importances = sorted(
        zip(feature_names, model.feature_importances_), key=lambda t: -t[1]
    )

    joblib.dump({"model": model, "encoder": encoder, "feature_names": feature_names}, MODEL_PATH)

    lines = []
    lines.append("# Edge-Case Reasoning Model — Training Report")
    lines.append("")
    lines.append(f"- **Training examples:** {len(X_train)} (batches 0-7, labeled edge cases only)")
    lines.append(f"- **Held-out test examples:** {len(X_test)} (batches 8-9, entirely unseen during training)")
    lines.append(f"- **Classes:** {', '.join(encoder.classes_)}")
    lines.append(f"- **Train accuracy:** {train_acc:.1%}")
    lines.append(f"- **Held-out test accuracy:** {test_acc:.1%}")
    lines.append("")
    lines.append("## Held-out classification report (batches 8-9, never seen in training)")
    lines.append("")
    lines.append("```")
    lines.append(test_report)
    lines.append("```")
    lines.append("")
    lines.append("## Confusion matrix (rows = true label, columns = predicted label)")
    lines.append("")
    lines.append("| true \\\\ pred | " + " | ".join(encoder.classes_) + " |")
    lines.append("|---" * (len(encoder.classes_) + 1) + "|")
    for true_label, row in zip(encoder.classes_, cm):
        lines.append(f"| **{true_label}** | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")
    lines.append("## Feature importances")
    lines.append("")
    lines.append("| Feature | Importance |")
    lines.append("|---|---|")
    for name, imp in importances:
        lines.append(f"| {name} | {imp:.3f} |")
    lines.append("")
    lines.append("## Notes on the held-out split")
    lines.append("")
    lines.append(
        "The test set is two ENTIRE batches (0 rows of which appear in training), not a random "
        "row-level split of all 10 batches. Batches 8-9 were generated with their own independent "
        "random seeds and their own randomly-drawn archetype mix and rate ranges, so this measures "
        "whether the model generalizes to genuinely new data, not whether it memorized rows from a "
        "shuffled pool it partially saw."
    )
    lines.append("")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    report = train_and_evaluate()
    print(report)
    print(f"\nModel saved -> {MODEL_PATH}")
    print(f"Report saved -> {REPORT_PATH}")
