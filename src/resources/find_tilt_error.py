
from resources.find_x import find_x_area_center, calculate_x_profile, calculate_x_profile_hot_spot, find_x_argmax
from resources.find_y import find_y_area_center,calculate_y_profile
from resources.find_z import find_z_area_center,calculate_z_profile
from resources.rotate_image import rotate_volume
import numpy as np

from resources.smooth_1d import smooth_1d


def find_tilt_error(image):
    volume = image.get_fdata(dtype=np.float32)
    x_profile = calculate_x_profile(volume)
    x_centre = find_x_area_center(x_profile)
    y_profile = calculate_y_profile(volume)
    y_center = find_y_area_center(y_profile)
    # y_center = 1
    z_profile = calculate_z_profile(volume)
    z_center = find_z_area_center(z_profile)
    centre = (x_centre,y_center,z_center)
    x_best_min = x_profile[x_centre]
    error = 0
    for i in range(-30,30):
        image_loop = rotate_volume(image, i,'y',centre)
        volume = image_loop.get_fdata(dtype=np.float32)
        x_profile = calculate_x_profile(volume)
        x_centre = find_x_area_center(x_profile)
        x_minimum = x_profile[x_centre]
        if x_minimum < x_best_min:
            x_best_min = x_minimum
            error = i

    return error

def create_rotation_x_profile(image,start = 0,end = 360):
    volume = image.get_fdata(dtype=np.float32)
    x_profile = calculate_x_profile(volume)
    x_centre = find_x_area_center(x_profile)
    y_profile = calculate_y_profile(volume)
    y_center = find_y_area_center(y_profile)
    z_profile = calculate_z_profile(volume)
    z_center = find_z_area_center(z_profile)
    centre = (x_centre, y_center, z_center)
    # print('0= ',x_profile[x_centre])
    rotation_x_profile = []
    for i in range(start,end + 1):
        image_loop = rotate_volume(image, i,'y',centre)
        volume = image_loop.get_fdata(dtype=np.float32)
        # x_profile = calculate_x_profile_with_margin(volume,margin = 0.3)
        x_profile = calculate_x_profile_hot_spot(volume)
        x_centre = find_x_area_center(x_profile)
        # x_centre = find_x_argmax(x_profile)
        x_value_in_centre = x_profile[x_centre]
        rotation_x_profile.append(x_value_in_centre)
        # print(i,"= ",x_value_in_centre)
    return rotation_x_profile

def find_tilt_error_by_profile(image):
    rotation_x_profile = create_rotation_x_profile(image,60,120)
    rotation_x_profile = smooth_1d(rotation_x_profile, window_size=19)
    er1 = np.argmax(rotation_x_profile) - 30
    rotation_x_profile = create_rotation_x_profile(image, 150, 210)
    rotation_x_profile = smooth_1d(rotation_x_profile, window_size=19)
    er2 = np.argmin(rotation_x_profile) - 30
    rotation_x_profile = create_rotation_x_profile(image, 240, 300)
    rotation_x_profile = smooth_1d(rotation_x_profile, window_size=19)
    er3 = np.argmax(rotation_x_profile) - 30
    rotation_x_profile = create_rotation_x_profile(image, 330, 490)
    rotation_x_profile = smooth_1d(rotation_x_profile, window_size=19)
    er4 = np.argmin(rotation_x_profile) - 30
    errors = [er1, er2, er3, er4]
    final_error = min(errors, key=abs)
    return int(final_error)