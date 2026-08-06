from pathlib import Path

import nibabel as nib
import numpy as np


def cut_off_black_segments(
    path: str | Path,
    threshold: float = 0,
    margin: int = 0,
) -> tuple[np.ndarray, nib.Nifti1Image]:
    """
    Wczytuje obraz NIfTI i usuwa zewnętrzne obszary tła.

    Args:
        path:
            Ścieżka do pliku .nii lub .nii.gz.
        threshold:
            Woksele o wartości większej od progu są uznawane
            za część obrazu.
        margin:
            Liczba dodatkowych wokseli pozostawionych wokół obrazu.

    Returns:
        cropped:
            Wycięta objętość 3D.
        img:
            Oryginalny obiekt NIfTI zawierający m.in. affine i header.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    img = nib.load(path)
    volume = img.get_fdata(dtype=np.float32)

    mask = volume > threshold
    coords = np.argwhere(mask)

    if coords.size == 0:
        raise ValueError(
            f"Obraz nie zawiera wokseli większych od progu {threshold}: {path}"
        )

    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0) + 1

    minimum = np.maximum(minimum - margin, 0)
    maximum = np.minimum(maximum + margin, volume.shape)

    x_min, y_min, z_min = minimum
    x_max, y_max, z_max = maximum

    cropped = volume[
        x_min:x_max,
        y_min:y_max,
        z_min:z_max,
    ]

    return cropped, img