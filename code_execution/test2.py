from pathlib import Path

import pandas as pd
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import log_loss

project_dir = Path(__file__).resolve().parents[1]

submission_path = project_dir / "code_execution" / "submission.csv"
labels_path = project_dir / "data" / "submission_format.csv"
submission = pd.read_csv(submission_path)
labels = pd.read_csv(labels_path)
comparison = submission.merge(
    labels,
    on="uid",
    suffixes=("_pred", "_true"),
    validate="one_to_one",
)

val_predictions = comparison["is_pathologic_pred"].to_numpy()
y_val = comparison["is_pathologic_true"].to_numpy()


def sharpen_predictions(predictions, strength):
    predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
    return expit(strength * logit(predictions))


strengths = [0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 2.68, 3.0]

print("Liczba porównanych wyników:", len(comparison))

for strength in strengths:
    adjusted = sharpen_predictions(val_predictions, strength)
    loss = log_loss(y_val, adjusted)

    print(f"strength={strength:>4}: log loss={loss:.6f}")