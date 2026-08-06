from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

TARGET_VOXEL_SIZE = 2.46

# Wymiary docelowe w voxelach po normalizacji do 2.46 mm.
X_LEFT = 35
X_RIGHT = 35

Y_FRONT = 40
Y_BACK = 50

Z_DOWN = 30
Z_UP = 40


def _cut_with_padding(
        volume: np.ndarray,
        centre: tuple[int, int, int],
        left_widths: tuple[int, int, int],
        right_widths: tuple[int, int, int],
) -> np.ndarray:
    x_centre, y_centre, z_centre = centre

    x_left, y_left, z_left = left_widths
    x_right, y_right, z_right = right_widths

    output_shape = (
        x_left + x_right,
        y_left + y_right,
        z_left + z_right,
    )

    output = np.zeros(output_shape, dtype=volume.dtype)

    source_starts = np.array([
        x_centre - x_left,
        y_centre - y_left,
        z_centre - z_left,
    ])

    source_ends = np.array([
        x_centre + x_right,
        y_centre + y_right,
        z_centre + z_right,
    ])

    volume_shape = np.array(volume.shape)

    clipped_starts = np.maximum(source_starts, 0)
    clipped_ends = np.minimum(source_ends, volume_shape)

    target_starts = clipped_starts - source_starts
    target_ends = target_starts + (clipped_ends - clipped_starts)

    output[
        target_starts[0]:target_ends[0],
        target_starts[1]:target_ends[1],
        target_starts[2]:target_ends[2],
    ] = volume[
        clipped_starts[0]:clipped_ends[0],
        clipped_starts[1]:clipped_ends[1],
        clipped_starts[2]:clipped_ends[2],
    ]

    return output


def _resample_nearest_to_shape(
        volume: np.ndarray,
        target_shape: tuple[int, int, int],
) -> np.ndarray:
    x_indices = np.linspace(0, volume.shape[0] - 1, target_shape[0]).round().astype(int)
    y_indices = np.linspace(0, volume.shape[1] - 1, target_shape[1]).round().astype(int)
    z_indices = np.linspace(0, volume.shape[2] - 1, target_shape[2]).round().astype(int)

    return volume[np.ix_(x_indices, y_indices, z_indices)]


def cut_image_to_one_actual_size():
    project_dir = Path(__file__).resolve().parents[2]

    csv_path = (
            project_dir
            / "src"
            / "outputs"
            / "scan_centres.csv"
    )

    nifti_dir = (
            project_dir
            / "data"
            / "niftis"
    )

    output_dir = (
            project_dir
            / "src"
            / "outputs"
            / "cut_images"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    images_data = pd.read_csv(csv_path)

    target_shape = (
        X_LEFT + X_RIGHT,
        Y_FRONT + Y_BACK,
        Z_DOWN + Z_UP,
    )

    saved_count = 0

    for index, row in images_data.iterrows():
        uid = row["uid"]
        file_path = nifti_dir / f"{uid}.nii.gz"

        if not file_path.exists():
            print(f"Brak pliku: {file_path}")
            continue

        print(f"{index + 1}/{len(images_data)}: {uid}")
        original_img = nib.load(file_path)
        volume = original_img.get_fdata(dtype=np.float32)
        # cropped, original_img = cut_off_black_segments(
        #     path=file_path,
        #     threshold=0,
        #     margin=1,
        # )

        x_voxel_size = float(row["x_voxel_size"])
        y_voxel_size = float(row["y_voxel_size"])
        z_voxel_size = float(row["z_voxel_size"])

        x_left_source = round((X_LEFT * TARGET_VOXEL_SIZE) / x_voxel_size)
        x_right_source = round((X_RIGHT * TARGET_VOXEL_SIZE) / x_voxel_size)

        y_front_source = round((Y_FRONT * TARGET_VOXEL_SIZE) / y_voxel_size)
        y_back_source = round((Y_BACK * TARGET_VOXEL_SIZE) / y_voxel_size)

        z_down_source = round((Z_DOWN * TARGET_VOXEL_SIZE) / z_voxel_size)
        z_up_source = round((Z_UP * TARGET_VOXEL_SIZE) / z_voxel_size)

        x_centre = round(float(row["x_centre"]))
        y_centre = round(float(row["y_centre"]))
        z_centre = round(float(row["z_centre"]))

        cut_volume = _cut_with_padding(
            volume=volume,
            centre=(x_centre, y_centre, z_centre),
            left_widths=(x_left_source, y_front_source, z_down_source),
            right_widths=(x_right_source, y_back_source, z_up_source),
        )

        normalized_volume = _resample_nearest_to_shape(
            volume=cut_volume,
            target_shape=target_shape,
        ).astype(np.float32)

        affine = np.diag([
            TARGET_VOXEL_SIZE,
            TARGET_VOXEL_SIZE,
            TARGET_VOXEL_SIZE,
            1.0,
        ])

        output_img = nib.Nifti1Image(
            normalized_volume,
            affine=affine,
            header=original_img.header.copy(),
        )

        output_img.header.set_zooms((
            TARGET_VOXEL_SIZE,
            TARGET_VOXEL_SIZE,
            TARGET_VOXEL_SIZE,
        ))

        output_path = output_dir / f"{uid}.nii.gz"
        nib.save(output_img, output_path)

        saved_count += 1

    print()
    print(f"Zapisano obrazy do: {output_dir}")
    print(f"Liczba zapisanych obrazów: {saved_count}")

    return output_dir
    # width x = 35 right left * 2,46 = 86,1
    # width y = 40 front * 2,46 = 98,4 50 back * 2,46 = 123
    # width z 30 down * 2,46 = 73,8 40 up * 2,46 = 98,4
    # return images_data

