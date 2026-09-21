from pathlib import Path

import ants
import nibabel as nib
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = PROJECT_DIR / "template_94xfri09.nii.gz"

def register_to_template(image_path, template_path=TEMPLATE_PATH):
    fixed = ants.image_read(str(template_path))
    moving = ants.image_read(str(image_path))

    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Rigid",
        verbose=False,
    )

    aligned = registration["warpedmovout"]

    # ANTs -> numpy -> NIfTI, żeby dalej działały Twoje funkcje cut_off_edges itd.
    aligned_array = aligned.numpy().astype(np.float32)

    return nib.Nifti1Image(
        aligned_array,
        affine=np.eye(4),
    )