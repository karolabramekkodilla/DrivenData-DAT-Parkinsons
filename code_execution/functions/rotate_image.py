import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform


def rotate_volume(
    image: nib.Nifti1Image,
    angle: float,
    axis: str,
    center: tuple[float, float, float],
    order: int = 1,
) -> nib.Nifti1Image:

    volume = image.get_fdata(dtype=np.float32)

    angle_rad = np.deg2rad(angle)

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    if axis == "x":
        rotation = np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a],
        ])

    elif axis == "y":
        rotation = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a],
        ])

    elif axis == "z":
        rotation = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ])

    else:
        raise ValueError("axis musi być 'x', 'y' albo 'z'")

    center = np.asarray(center, dtype=np.float64)

    # affine_transform działa odwrotnie:
    # dla każdego voxela WYJŚCIOWEGO szuka miejsca
    # w obrazie WEJŚCIOWYM.
    inverse_rotation = rotation.T

    offset = center - inverse_rotation @ center

    rotated = affine_transform(
        volume,
        matrix=inverse_rotation,
        offset=offset,
        output_shape=volume.shape,
        order=order,
        mode="constant",
        cval=0.0,
    )

    return nib.Nifti1Image(
        rotated.astype(np.float32),
        image.affine,
        image.header,
    )