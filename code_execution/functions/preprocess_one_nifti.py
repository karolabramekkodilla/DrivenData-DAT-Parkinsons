from os import PathLike

import nibabel as nib
from nibabel.processing import resample_to_output

from functions.cut_off_edges import cut_off_edges
from functions.find_z import find_z_argmax
from functions.standarize_image import standardize_image


def preprocess_one_nifti(image):
    # Jeżeli otrzymaliśmy ścieżkę, wczytujemy obraz.
    if isinstance(
        image,
        (str, PathLike),
    ):
        image = nib.load(
            str(image)
        )

    # Jeżeli to nie ścieżka ani obraz NIfTI,
    # zgłaszamy czytelny błąd.
    elif not isinstance(
        image,
        nib.spatialimages.SpatialImage,
        ):
        raise TypeError(
            f"Oczekiwano ścieżki albo obrazu NIfTI, "
            f"otrzymano: {type(image).__name__}"
        )
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
    processed_image = cut_off_edges(
        processed_image,
        x_left=30,
        x_right=30,
        y_front=35,
        y_back=35,
        z_down=25,
        z_up=25,
        z_center_method=find_z_argmax
    )

    processed_image = standardize_image(
        processed_image,
        percent_top=0.02,
    )

    processed_image = resample_to_output(
        processed_image,
        voxel_sizes=(2.46, 2.46, 2.46),
        order=1,
    )

    return processed_image