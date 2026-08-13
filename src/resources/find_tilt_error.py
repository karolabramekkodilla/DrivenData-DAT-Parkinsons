
from resources.find_x import find_x_area_center,calculate_x_profile
from resources.find_y import find_y_area_center,calculate_y_profile
from resources.find_z import find_z_area_center,calculate_z_profile
from resources.rotate_image import rotate_volume
import numpy as np


def find_tilt_error(image):
    volume = image.get_fdata(dtype=np.float32)
    x_profile = calculate_x_profile(volume)
    x_centre = find_x_area_center(x_profile)
    y_profile = calculate_y_profile(volume)
    y_center = find_y_area_center(y_profile)
    z_profile = calculate_z_profile(volume)
    z_center = find_z_area_center(z_profile)
    centre = (x_centre,y_center,z_center)
    x_best_min = x_profile[x_centre]
    error = 0
    for i in range(-20,21):
        image_loop = rotate_volume(image, i,'y',centre)
        volume = image_loop.get_fdata(dtype=np.float32)
        x_profile = calculate_x_profile(volume)
        x_centre = find_x_area_center(x_profile)
        x_minimum = x_profile[x_centre]
        if x_minimum < x_best_min:
            x_best_min = x_minimum
            error = i

    return error