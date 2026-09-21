import time

SCRIPT_START = time.perf_counter()

from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.processing import resample_to_output

from functions.find_pitch_error_y_y import (
    find_pitch_error_by_similarity,
)
from functions.find_rotation_error import (
    create_rotation_x_profile,
    find_rotation_error_by_similarity,
)
from functions.find_tilt_error import (
    find_tilt_error_by_similarity,
)
from functions.find_x import (
    calculate_x_profile,
    find_x_area_center,
)
from functions.find_y import (
    calculate_y_profile,
    find_y_area_center,
)
from functions.find_z import (
    calculate_z_profile,
    find_z_argmax,
)
from functions.get_template_images import (
    get_template_images,
)
from functions.preprocess_one_nifti import (
    preprocess_one_nifti,
)
from functions.cut_off_edges import cut_off_edges
from functions.rotate_image import rotate_volume
from functions.standarize_image import (
    standardize_image,
)
from model.cnn_model import (
    Simple3DCNN,
    Simple3DCNNLarge,
    ProfileMLP,
)


# ======================================================
# Ścieżki
# ======================================================

EXECUTION_DIR = Path(__file__).resolve().parent

TEMPLATE_PATH = (
    EXECUTION_DIR
    / "template_94xfri09.nii.gz"
)

DATA_DIR = (
    EXECUTION_DIR
    / "data"
)

NIFTI_DIR = (
    DATA_DIR
    / "niftis"
)

SUBMISSION_FORMAT_PATH = (
    DATA_DIR
    / "submission_format.csv"
)

OUTPUT_PATH = (
    EXECUTION_DIR
    / "submission.csv"
)


EDA6_MODEL_PATHS = [
    EXECUTION_DIR
    / "model"
    / f"2model_fold_{fold}.pth"
    for fold in range(1, 6)
]


ANTSPYX_MODEL_PATHS = [
    EXECUTION_DIR
    / "model"
    / f"ants_aug_model_fold_{fold}.pth"
    for fold in range(1, 6)
]


# Nowe modele Simple3DCNNLarge.
NEW_MODEL_PATHS = [
    EXECUTION_DIR
    / "model"
    / f"new{fold}.pth"
    for fold in range(1, 6)
]


PROFILE_MODEL_PATH = (
    EXECUTION_DIR
    / "model"
    / "profile_model.pth"
)


META_MODEL_PATH = (
    EXECUTION_DIR
    / "model"
    / "final_meta_classifier_no_native.joblib"
)


LOG_EVERY_IMAGES = 10


# ======================================================
# Szablony
# ======================================================

template_images_start = time.perf_counter()

template_images = get_template_images()

template_images_loading_time = (
    time.perf_counter()
    - template_images_start
)


# ======================================================
# Logowanie czasu
# ======================================================

def format_duration(seconds):
    total_seconds = max(
        0,
        int(round(seconds)),
    )

    hours, remainder = divmod(
        total_seconds,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def log_progress(message):
    elapsed = (
        time.perf_counter()
        - SCRIPT_START
    )

    print(
        f"[{format_duration(elapsed)}] "
        f"{message}",
        flush=True,
    )


# ======================================================
# Preprocessing
# ======================================================

def preprocess_image(image_path):
    image = nib.load(image_path)

    image_for_errors = preprocess_one_nifti(
        image
    )

    # ==================================================
    # Rotation wokół Z
    # ==================================================

    rotation_errors = [
        find_rotation_error_by_similarity(
            image_for_errors,
            template_image,
        )
        for template_image in template_images
    ]

    median_rotation_error = int(
        np.median(rotation_errors)
    )

    # ==================================================
    # Tilt wokół Y
    # ==================================================

    tilt_errors = [
        find_tilt_error_by_similarity(
            image_for_errors,
            template_image,
        )
        for template_image in template_images
    ]

    median_tilt_error = int(
        np.median(tilt_errors)
    )

    # ==================================================
    # Pitch wokół X
    # ==================================================

    pitch_errors = [
        find_pitch_error_by_similarity(
            image_for_errors,
            template_image,
        )
        for template_image in template_images
    ]

    median_pitch_error = int(
        np.median(pitch_errors)
    )

    # ==================================================
    # Główny preprocessing
    # ==================================================

    processed_image = cut_off_edges(
        image
    )

    processed_image = cut_off_edges(
        processed_image,
        x_left=35,
        x_right=35,
        y_front=40,
        y_back=40,
        z_down=30,
        z_up=30,
    )

    processed_image = standardize_image(
        processed_image,
        percent_top=0.02,
    )

    processed_image = resample_to_output(
        processed_image,
        voxel_sizes=(
            2.46,
            2.46,
            2.46,
        ),
        order=1,
    )

    processed_volume = (
        processed_image.get_fdata(
            dtype=np.float32,
        )
    )

    x_profile = calculate_x_profile(
        processed_volume
    )

    x_centre = find_x_area_center(
        x_profile
    )

    y_profile = calculate_y_profile(
        processed_volume
    )

    y_center = find_y_area_center(
        y_profile
    )

    z_profile = calculate_z_profile(
        processed_volume
    )

    z_center = find_z_argmax(
        z_profile
    )

    xyz_center = [
        x_centre,
        y_center,
        z_center,
    ]

    processed_image = rotate_volume(
        processed_image,
        center=xyz_center,
        angle=median_rotation_error,
        axis="z",
    )

    processed_image = rotate_volume(
        processed_image,
        center=xyz_center,
        angle=median_tilt_error,
        axis="y",
    )

    processed_image = rotate_volume(
        processed_image,
        center=xyz_center,
        angle=median_pitch_error,
        axis="x",
    )

    processed_image = cut_off_edges(
        processed_image,
        x_left=30,
        x_right=30,
        y_front=30,
        y_back=30,
        z_down=20,
        z_up=25,
    )

    processed_image = cut_off_edges(
        processed_image,
        x_left=25,
        x_right=25,
        y_front=30,
        y_back=30,
        z_down=10,
        z_up=20,
    )

    return processed_image


# ======================================================
# Profile
# ======================================================

PROFILE_SIZE = 495


def calculate_profiles_for_model(image):
    volume = image.get_fdata(
        dtype=np.float32
    )

    x_profile = np.asarray(
        calculate_x_profile(volume),
        dtype=np.float32,
    ).reshape(-1)

    y_profile = np.asarray(
        calculate_y_profile(volume),
        dtype=np.float32,
    ).reshape(-1)

    z_profile = np.asarray(
        calculate_z_profile(volume),
        dtype=np.float32,
    ).reshape(-1)

    rotation_x_profile = np.asarray(
        create_rotation_x_profile(image),
        dtype=np.float32,
    ).reshape(-1)

    profiles = np.concatenate(
        [
            x_profile,
            y_profile,
            z_profile,
            rotation_x_profile,
        ],
        axis=0,
    ).astype(np.float32)

    if profiles.shape != (PROFILE_SIZE,):
        raise ValueError(
            "Nieprawidłowy rozmiar profili: "
            f"{profiles.shape}; "
            f"oczekiwano ({PROFILE_SIZE},)"
        )

    return profiles


# ======================================================
# Ładowanie modeli
# ======================================================

def load_model_group(
    model_paths,
    model_class,
    device,
    group_name,
):
    models = []

    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Brak modelu {group_name}: "
                f"{model_path}"
            )

        model = model_class()

        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(
            state_dict
        )

        model.to(device)
        model.eval()

        models.append(model)

    return models


def load_models(device):
    # Pięć modeli EDA6.
    eda6_models = load_model_group(
        model_paths=EDA6_MODEL_PATHS,
        model_class=Simple3DCNNLarge,
        device=device,
        group_name="EDA6",
    )

    # Pięć modeli ANTSPYX.
    antspyx_models = load_model_group(
        model_paths=ANTSPYX_MODEL_PATHS,
        model_class=Simple3DCNN,
        device=device,
        group_name="ANTSPYX",
    )

    # Pięć nowych modeli Large.
    new_models = load_model_group(
        model_paths=NEW_MODEL_PATHS,
        model_class=Simple3DCNNLarge,
        device=device,
        group_name="NEW",
    )

    # Model profili.
    profile_model = ProfileMLP(
        input_size=PROFILE_SIZE,
    )

    profile_state_dict = torch.load(
        PROFILE_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    profile_model.load_state_dict(
        profile_state_dict
    )

    profile_model.to(device)
    profile_model.eval()

    # Końcowy klasyfikator sklearn.
    meta_bundle = joblib.load(
        META_MODEL_PATH
    )

    meta_model = meta_bundle["model"]

    feature_columns = list(
        meta_bundle["feature_columns"]
    )

    return {
        "eda6": eda6_models,
        "antspyx": antspyx_models,
        "new": new_models,
        "profile": profile_model,
        "meta": meta_model,
        "feature_columns": feature_columns,
    }


# ======================================================
# Predykcja modeli CNN
# ======================================================

def predict_cnn_group(
    models,
    volume_tensor,
):
    probabilities = []

    with torch.inference_mode():
        for model in models:
            logits = model(
                volume_tensor
            )

            probability = (
                torch.sigmoid(logits)
                .detach()
                .cpu()
                .reshape(-1)[0]
                .item()
            )

            probabilities.append(
                float(probability)
            )

    return probabilities


# ======================================================
# Predykcja końcowa
# ======================================================

def predict_probability(
    loaded_models,
    processed_image,
    device,
):
    volume = processed_image.get_fdata(
        dtype=np.float32
    )

    volume = np.ascontiguousarray(
        volume,
        dtype=np.float32,
    )

    volume_tensor = (
        torch.from_numpy(volume)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    # Pięć predykcji EDA6.
    eda6_probabilities = predict_cnn_group(
        models=loaded_models["eda6"],
        volume_tensor=volume_tensor,
    )

    # Pięć predykcji ANTSPYX.
    antspyx_probabilities = predict_cnn_group(
        models=loaded_models["antspyx"],
        volume_tensor=volume_tensor,
    )

    # Pięć predykcji nowego modelu Large.
    new_probabilities = predict_cnn_group(
        models=loaded_models["new"],
        volume_tensor=volume_tensor,
    )

    # Średnia nowych pięciu foldów.
    new_probability = float(
        np.mean(new_probabilities)
    )

    # Predykcja modelu profili.
    profiles = calculate_profiles_for_model(
        processed_image
    )

    profiles_tensor = (
        torch.from_numpy(profiles)
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        profile_logit = loaded_models[
            "profile"
        ](
            profiles_tensor
        )

        profile_probability = (
            torch.sigmoid(profile_logit)
            .detach()
            .cpu()
            .reshape(-1)[0]
            .item()
        )

    # Cechy dla istniejącego meta-klasyfikatora.
    feature_values = {}

    for fold_index, probability in enumerate(
        eda6_probabilities,
        start=1,
    ):
        feature_values[
            f"p_eda6_fold_{fold_index}"
        ] = probability

    for fold_index, probability in enumerate(
        antspyx_probabilities,
        start=1,
    ):
        feature_values[
            f"p_antspyx_fold_{fold_index}"
        ] = probability

    feature_values["p_profiles"] = float(
        profile_probability
    )

    # Zachowujemy kolejność cech wymaganą
    # przez zapisany meta-klasyfikator.
    meta_input = (
        pd.DataFrame(
            [feature_values]
        )[loaded_models["feature_columns"]]
        .to_numpy(dtype=np.float32)
    )

    meta_probabilities = loaded_models[
        "meta"
    ].predict_proba(
        meta_input
    )

    current_probability = float(
        meta_probabilities[0, 1]
    )

    # 75% średniej nowego modelu
    # oraz 25% obecnego meta-klasyfikatora.
    final_probability = (
        0.75 * new_probability
        + 0.25 * current_probability
    )

    final_probability = float(
        np.clip(
            final_probability,
            0.001,
            0.999,
        )
    )

    return final_probability


# ======================================================
# Main
# ======================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    loaded_models = load_models(
        device
    )

    submission = pd.read_csv(
        SUBMISSION_FORMAT_PATH
    )

    required_columns = {
        "uid",
        "is_pathologic",
    }

    missing_columns = (
        required_columns
        - set(submission.columns)
    )

    if missing_columns:
        raise ValueError(
            "Brak wymaganych kolumn w "
            "submission_format.csv: "
            f"{sorted(missing_columns)}"
        )



    for row_index, uid in enumerate(
        submission["uid"]
    ):
        uid = str(uid)

        image_path = (
            NIFTI_DIR
            / f"{uid}.nii.gz"
        )

        preprocessing_start = (
            time.perf_counter()
        )

        processed_image = preprocess_image(
            image_path
        )



        prediction_start = (
            time.perf_counter()
        )

        probability = predict_probability(
            loaded_models=loaded_models,
            processed_image=processed_image,
            device=device,
        )

        submission.loc[
            row_index,
            "is_pathologic",
        ] = probability


    if submission[
        "is_pathologic"
    ].isna().any():
        raise ValueError(
            "Submission zawiera brakujące "
            "predykcje."
        )

    submission["is_pathologic"] = (
        submission["is_pathologic"]
        .astype(float)
        .clip(0.001, 0.999)
    )

    save_start = time.perf_counter()

    submission.to_csv(
        OUTPUT_PATH,
        index=False,
    )


if __name__ == "__main__":
    main()