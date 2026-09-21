import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation


def rotate_nifti_by_errors(
    image,
    rotation_error,
    tilt_error,
    pitch_error,
    order=1,
):
    data = image.get_fdata()

    rotation_matrix = Rotation.from_euler(
        "xyz",
        [
            rotation_error,
            tilt_error,
            pitch_error,
        ],
        degrees=True,
    ).as_matrix()

    center = (np.array(data.shape) - 1) / 2.0

    # scipy affine_transform mapuje współrzędne output -> input,
    # więc używamy macierzy odwrotnej.
    inverse_matrix = np.linalg.inv(rotation_matrix)

    offset = center - inverse_matrix @ center

    rotated_data = affine_transform(
        data,
        matrix=inverse_matrix,
        offset=offset,
        output_shape=data.shape,
        order=order,
        mode="constant",
        cval=0.0,
    )

    return nib.Nifti1Image(
        rotated_data,
        affine=image.affine,
        header=image.header,
    )