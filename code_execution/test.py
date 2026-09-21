from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "data"

LABELS_PATH = DATA_DIR / "submission_format.csv"
PREDICTIONS_PATH = TEST_DIR / "submission.csv"


labels = pd.read_csv(LABELS_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH)

assert list(labels.columns) == [
    "uid",
    "is_pathologic",
]

assert list(predictions.columns) == [
    "uid",
    "is_pathologic",
]

assert labels["uid"].is_unique
assert predictions["uid"].is_unique

comparison = labels.merge(
    predictions,
    on="uid",
    suffixes=("_true", "_pred"),
    validate="one_to_one",
)

assert len(comparison) == len(labels)
assert comparison["is_pathologic_pred"].notna().all()

y_true = comparison[
    "is_pathologic_true"
].to_numpy()

raw_probabilities = comparison[
    "is_pathologic_pred"
].to_numpy()


print("Liczba obrazów:", len(comparison))
print(
    "Dokładne zera:",
    np.sum(raw_probabilities == 0.0),
)
print(
    "Dokładne jedynki:",
    np.sum(raw_probabilities == 1.0),
)

print(
    "Minimalna predykcja:",
    raw_probabilities.min(),
)
print(
    "Maksymalna predykcja:",
    raw_probabilities.max(),
)

print(
    "\nLog loss bez dodatkowego clippingu:",
    log_loss(y_true, raw_probabilities),
)

print(
    "AUROC:",
    roc_auc_score(y_true, raw_probabilities),
)


print("\nPorównanie clippingu:")

for epsilon in [
    1e-6,
    1e-4,
    1e-3,
    0.005,
    0.01,
]:
    clipped_probabilities = np.clip(
        raw_probabilities,
        epsilon,
        1 - epsilon,
    )

    clipped_loss = log_loss(
        y_true,
        clipped_probabilities,
    )

    changed_count = np.sum(
        clipped_probabilities
        != raw_probabilities
    )

    print(
        f"epsilon={epsilon:<8} "
        f"log_loss={clipped_loss:.8f} "
        f"zmienione={changed_count}"
    )

