from pathlib import Path

from functions.preprocess_one_nifti import preprocess_one_nifti


def get_template_images():

    functions_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    template_path_1 = (
            functions_dir
            / "template_94xfri09.nii.gz"
    )

    template_path_2 = (
            functions_dir
            / "template_09lzrk79.nii.gz"
    )

    template_path_3 = (
            functions_dir
            / "template_2c9ros7y.nii.gz"
    )

    template_paths = [
        template_path_1,
        template_path_2,
        template_path_3,
    ]

    template_images = [
        preprocess_one_nifti(template_path)
        for template_path in template_paths
    ]

    return template_images