from pathlib import Path

import numpy as np
import ants
from scipy.spatial.transform import Rotation
from pathlib import Path

import ants
import nibabel as nib
from scipy.spatial.transform import Rotation


PROJECT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = PROJECT_DIR / "template_94xfri09.nii.gz"

FIXED_TEMPLATE_ANTS = ants.image_read(str(TEMPLATE_PATH))
TEMPLATE_NIB = nib.load(TEMPLATE_PATH)


def get_rotation_errors_from_same_registration(image_path):
    moving = ants.image_read(str(image_path))

    registration = ants.registration(
        fixed=FIXED_TEMPLATE_ANTS,
        moving=moving,
        type_of_transform="Rigid",
        verbose=False,
    )

    transform_path = registration["fwdtransforms"][0]
    transform = ants.read_transform(transform_path)

    params = np.asarray(transform.parameters)

    matrix = params[:9].reshape(3, 3)

    u, _, vh = np.linalg.svd(matrix)
    rotation_matrix = u @ vh

    angles_deg = Rotation.from_matrix(rotation_matrix).as_euler(
        "xyz",
        degrees=True,
    )

    return angles_deg[0], angles_deg[1], angles_deg[2]

def get_rotation_errors_from_ants(
    nifti_path,
    random_seed=None,
):
    template_path =  Path(__file__).resolve().parent / "template_94xfri09.nii.gz"
    fixed = ants.image_read(str(template_path))

    moving = ants.image_read(str(nifti_path))

    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Rigid",
        random_seed=random_seed,
    )

    transform_path = registration["fwdtransforms"][0]

    transform = ants.read_transform(transform_path)

    params = np.array(transform.parameters)

    # ANTs / ITK affine params:
    # pierwsze 9 wartości = macierz 3x3
    # ostatnie 3 wartości = przesunięcie, którego teraz nie używamy
    matrix = params[:9].reshape(3, 3)

    # Dla bezpieczeństwa usuwamy minimalne błędy numeryczne,
    # żeby została czysta macierz obrotu.
    u, _, vh = np.linalg.svd(matrix)
    rotation_matrix = u @ vh

    angles_deg = Rotation.from_matrix(rotation_matrix).as_euler(
        "xyz",
        degrees=True,
    )

    return angles_deg[0], angles_deg[1], angles_deg[2]