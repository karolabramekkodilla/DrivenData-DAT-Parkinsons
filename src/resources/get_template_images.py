from resources.preprocess_one_nifti import preprocess_one_nifti


def get_template_images():
    template_path_1 = (
        "C:/Users/abram/PycharmProjects/"
        "DrivenData_DAT_Parkinsons/src/resources/"
        "template_94xfri09.nii.gz"
    )
    template_path_2 = (
        "C:/Users/abram/PycharmProjects/"
        "DrivenData_DAT_Parkinsons/src/resources/"
        "template_09lzrk79.nii.gz"
    )

    template_path_3 = (
        "C:/Users/abram/PycharmProjects/"
        "DrivenData_DAT_Parkinsons/src/resources/"
        "template_2c9ros7y.nii.gz"
    )
    template_path_4 = (
        "C:/Users/abram/PycharmProjects/"
        "DrivenData_DAT_Parkinsons/src/resources/"
        "template_0ezzf8s2.nii.gz"
    )
    template_path_5 = (
        "C:/Users/abram/PycharmProjects/"
        "DrivenData_DAT_Parkinsons/src/resources/"
        "template_33mg1zaq.nii.gz"
    )

    template_paths = [
        template_path_1,
        template_path_2,
        template_path_3,
        template_path_4,
        template_path_5,
    ]

    template_images = [
        preprocess_one_nifti(template_path)
        for template_path in template_paths
    ]

    return template_images