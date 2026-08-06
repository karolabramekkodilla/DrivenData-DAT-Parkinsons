
import nibabel as nib
import numpy as np
from resources.find_x import find_x_area_center_split,calculate_x_profile, find_x_area_center
from resources.find_y import find_y_argmax,calculate_y_profile
from resources.find_z import find_z_argmax,calculate_z_profile

def cut_off_edges(image):

    TARGET_VOXEL_SIZE = 2.46

    # Wymiary docelowe w voxelach po normalizacji do 2.46 mm.
    X_LEFT = 35
    X_RIGHT = 35

    Y_FRONT = 40
    Y_BACK = 50

    Z_DOWN = 30
    Z_UP = 40

    shape = image.shape
    x_voxel_size, y_voxel_size, z_voxel_size = image.header.get_zooms()[:3]

    x_left_source = round((X_LEFT * TARGET_VOXEL_SIZE) / x_voxel_size)
    x_right_source = round((X_RIGHT * TARGET_VOXEL_SIZE) / x_voxel_size)
    y_front_source = round((Y_FRONT * TARGET_VOXEL_SIZE) / y_voxel_size)
    y_back_source = round((Y_BACK * TARGET_VOXEL_SIZE) / y_voxel_size)
    z_down_source = round((Z_DOWN * TARGET_VOXEL_SIZE) / z_voxel_size)
    z_up_source = round((Z_UP * TARGET_VOXEL_SIZE) / z_voxel_size)

    volume = image.get_fdata(dtype=np.float32)

    x_profile = calculate_x_profile(volume)
    y_profile = calculate_y_profile(volume)
    z_profile = calculate_z_profile(volume)

    # x_centre = find_x_area_center_split(x_profile)
    x_centre = find_x_area_center(x_profile)
    y_center = find_y_argmax(y_profile)
    z_center = find_z_argmax(z_profile)

    x_left_cut = x_centre - x_left_source
    if x_left_cut < 0:
        x_left_cut = 0

    x_right_cut = x_centre + x_right_source
    if x_right_cut > shape[0]:
        x_right_cut = shape[0]

    y_front_cut = y_center - y_front_source
    if y_front_cut < 0:
        y_front_cut = 0

    y_back_cut = y_center + y_back_source
    if y_back_cut > shape[1]:
        y_back_cut = shape[1]

    z_down_cut = z_center - z_down_source
    if z_down_cut < 0:
        z_down_cut = 0

    z_up_cut = z_center + z_up_source
    if z_up_cut > shape[2]:
        z_up_cut = shape[2]

    cropped_volume = volume[
        x_left_cut:x_right_cut,
        y_front_cut:y_back_cut,
        z_down_cut:z_up_cut,
    ]

    cropped_image = nib.Nifti1Image(
        cropped_volume,
        image.affine,
        image.header.copy(),
    )

    return cropped_image