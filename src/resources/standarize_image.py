import pandas as pd
import nibabel as nib
import numpy as np

def standardize_image(image: nib.Nifti1Image,
                      percent_top: float = 0.10) -> nib.Nifti1Image:
    volume = image.get_fdata(dtype=np.float32)

    flat_volume = volume.ravel()
    top_10_percent_count = max(1, int(flat_volume.size * percent_top))

    top_values = np.partition(
        flat_volume,
        -top_10_percent_count
    )[-top_10_percent_count:]

    normalization_max = top_values.mean()

    if normalization_max <= 0:
        normalized_volume = np.zeros_like(volume, dtype=np.float32)
    else:
        normalized_volume = volume / normalization_max
        # normalized_volume = np.clip(normalized_volume, 0.0, 1.0)

    return nib.Nifti1Image(
        normalized_volume.astype(np.float32),
        affine=image.affine,
        header=image.header.copy()
    )