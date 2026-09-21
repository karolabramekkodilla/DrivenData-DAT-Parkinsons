from pathlib import Path

import ants
import nibabel as nib
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = PROJECT_DIR / "template_94xfri09.nii.gz"

FIXED_TEMPLATE_ANTS = ants.image_read(str(TEMPLATE_PATH))
TEMPLATE_NIB = nib.load(TEMPLATE_PATH)


def register_to_template(image_path):
    moving = ants.image_read(str(image_path))

    registration = ants.registration(
        fixed=FIXED_TEMPLATE_ANTS,
        moving=moving,
        type_of_transform="Rigid",
        verbose=False,
    )

    aligned = registration["warpedmovout"]
    #
    # print("fixed shape:", FIXED_TEMPLATE_ANTS.shape, flush=True)
    # print("fixed spacing:", FIXED_TEMPLATE_ANTS.spacing, flush=True)
    # print("moving shape:", moving.shape, flush=True)
    # print("moving spacing:", moving.spacing, flush=True)
    # print("aligned shape:", aligned.shape, flush=True)
    # print("aligned spacing:", aligned.spacing, flush=True)

    aligned_array = aligned.numpy().astype(np.float32)

    header = TEMPLATE_NIB.header.copy()
    header.set_data_shape(aligned_array.shape)
    header.set_data_dtype(np.float32)

    return nib.Nifti1Image(
        aligned_array,
        affine=TEMPLATE_NIB.affine,
        header=header,
    )