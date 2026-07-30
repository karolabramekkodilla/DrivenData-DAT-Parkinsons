from pathlib import Path

import pandas as pd

from resources.cut_off_black_segments import cut_off_black_segments
from resources.find_x import calculate_x_profile, find_x_area_center_split
from resources.find_y import calculate_y_profile, find_y_argmax
from resources.find_z import calculate_z_profile, find_z_argmax


def create_centres_csv(
        project_dir: Path,
        output_file_name: str = "scan_centres.csv",
        filter_width: int = 3,
        smooth_window: int = 5,
) -> Path:
    nifti_dir = project_dir / "data" / "niftis"
    labels_path = nifti_dir / "train_labels_JNDlMjr.csv"

    output_dir = project_dir / "src" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_file_name

    labels = pd.read_csv(labels_path)

    rows = []

    for index, row in labels.iterrows():
        uid = row["uid"]
        pathological = int(row["is_pathologic"])

        file_path = nifti_dir / f"{uid}.nii.gz"

        if not file_path.exists():
            print(f"Brak pliku: {file_path}")
            continue

        print(f"{index + 1}/{len(labels)}: {uid}")

        cropped, img = cut_off_black_segments(
            path=file_path,
            threshold=0,
            margin=1,
        )

        x_profile = calculate_x_profile(
            cropped,
            filter_width=filter_width,
        )

        y_profile = calculate_y_profile(
            cropped,
            filter_width=filter_width,
        )

        z_profile = calculate_z_profile(
            cropped,
            filter_width=filter_width,
        )

        x_centre = find_x_area_center_split(
            x_profile,
            filter_width=filter_width,
            smooth_window=smooth_window,
            edge_margin_ratio=0.10,
            local_search_ratio=0.20,
        )

        y_centre = find_y_argmax(
            y_profile,
            filter_width=filter_width,
            smooth_window=smooth_window,
            edge_margin_ratio=0.10,
        )

        z_centre = find_z_argmax(
            z_profile,
            filter_width=filter_width,
            smooth_window=smooth_window,
            edge_margin_ratio=0.10,
        )

        x_voxel_size, y_voxel_size, z_voxel_size = img.header.get_zooms()[:3]

        rows.append({
            "uid": uid,
            "pathological": pathological,
            "x_centre": x_centre,
            "y_centre": y_centre,
            "z_centre": z_centre,
            "x_voxel_size": float(x_voxel_size),
            "y_voxel_size": float(y_voxel_size),
            "z_voxel_size": float(z_voxel_size),
        })

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_path, index=False)

    print()
    print(f"Zapisano plik: {output_path}")
    print(f"Liczba zapisanych rekordów: {len(output_df)}")

    return output_path