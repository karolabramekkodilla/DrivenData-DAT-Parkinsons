from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from nibabel.processing import resample_to_output

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from resources.cut_off_edges import cut_off_edges
from resources.find_rotation_error import create_rotation_x_profile
from resources.standarize_image import standardize_image


def main():
    project_dir = Path(__file__).resolve().parents[1]

    niftis_dir = project_dir / "data" / "niftis"

    # Dostosuj nazwę pliku, jeżeli u Ciebie jest inna
    labels_path = project_dir / "data" / "niftis" / "train_labels_JNDlMjr.csv"

    outputs_dir = project_dir / "src" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(labels_path)

    # Jeżeli kolumna nazywa się is_pathologic, zmień tutaj nazwę
    target_column = "is_pathologic"

    labels = labels_df.set_index("uid")[target_column]

    nifti_paths = sorted(niftis_dir.glob("*.nii.gz"))

    x_profiles = []
    y_labels = []
    used_uids = []

    for index, nifti_path in enumerate(nifti_paths, start=1):
        uid = nifti_path.name.removesuffix(".nii.gz")

        if uid not in labels.index:
            print(f"Brak etykiety dla {uid}")
            continue

        print(f"{index}/{len(nifti_paths)}: {uid}")

        image = nib.load(nifti_path)

        processed_image = cut_off_edges(image)

        processed_image = cut_off_edges(
            processed_image,
            x_left=35,
            x_right=35,
            y_front=40,
            y_back=40,
            z_down=30,
            z_up=30,
        )

        # processed_image = standardize_image(
        #     processed_image,
        #     percent_top=0.02,
        # )

        processed_image = cut_off_edges(
            processed_image,
            x_left=30,
            x_right=30,
            y_front=30,
            y_back=30,
            z_down=20,
            z_up=25,
        )

        # processed_image = resample_to_output(
        #     processed_image,
        #     voxel_sizes=(2.46, 2.46, 2.46),
        #     order=1,
        # )

        profile = create_rotation_x_profile(processed_image)
        profile = np.asarray(profile, dtype=np.float32)

        # Jeżeli masz także punkt 360°, usuń duplikat punktu 0°
        if len(profile) == 361:
            profile = profile[:360]

        if len(profile) != 360:
            print(
                f"Nieprawidłowa długość profilu dla {uid}: "
                f"{len(profile)}"
            )
            continue

        x_profiles.append(profile)
        y_labels.append(int(labels.loc[uid]))
        used_uids.append(uid)

    X = np.stack(x_profiles)
    y = np.asarray(y_labels)

    print("X:", X.shape)
    print("y:", y.shape)
    print("Klasy:", np.unique(y, return_counts=True))

    # Zapis profili, żeby nie liczyć ich ponownie
    np.savez_compressed(
        outputs_dir / "rotation_profiles.npz",
        X=X,
        y=y,
        uids=np.asarray(used_uids),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_probability))
    print("Log loss:", log_loss(y_test, y_probability))

    print("\nMacierz omyłek:")
    print(confusion_matrix(y_test, y_pred))

    print("\nRaport:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()