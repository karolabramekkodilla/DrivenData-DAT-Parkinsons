
from resources.find_x import find_x_area_center,calculate_x_profile,find_x_area_center_split
from resources.find_y import find_y_area_center, calculate_y_profile, find_y_argmax
from resources.find_z import find_z_area_center, calculate_z_profile, find_z_argmax, calculate_z_profile_hot_spot
from resources.rotate_image import rotate_volume
import numpy as np

def find_pitch_error(image):
    volume = image.get_fdata(dtype=np.float32)
    x_profile = calculate_x_profile(volume)
    x_centre = find_x_area_center(x_profile)
    y_profile = calculate_y_profile(volume)
    y_center = find_y_area_center(y_profile)
    z_profile = calculate_z_profile(volume)
    z_center = find_z_area_center(z_profile)
    centre = (x_centre,y_center,z_center)
    # z_best_max = z_profile[z_center]
    z_profile_hot_spot = calculate_z_profile_hot_spot(volume)
    z_pos_max = find_z_argmax(z_profile_hot_spot)
    z_best_max = z_profile_hot_spot[z_pos_max]

    # print('0= ',z_profile[z_pos_max])
    error = 0
    for i in range(-60,60):
        image_loop = rotate_volume(image, i,'x',centre)
        volume = image_loop.get_fdata(dtype=np.float32)
        z_profile = calculate_z_profile_hot_spot(volume)
        z_pos_max = find_z_argmax(z_profile)
        z_maximum = z_profile[z_pos_max]
        # print(i,"= ",z_maximum)
        if z_maximum > z_best_max:
            z_best_max = z_maximum
            error = i
    return error

def create_rotation_y_profile(image,start = 0,end = 360):
    volume = image.get_fdata(dtype=np.float32)
    x_profile = calculate_x_profile(volume)
    x_centre = find_x_area_center(x_profile)
    y_profile = calculate_y_profile(volume)
    y_center = find_y_area_center(y_profile)
    z_profile = calculate_z_profile(volume)
    z_center = find_z_area_center(z_profile)
    centre = (x_centre, y_center, z_center)
    # print('0= ',x_profile[x_centre])
    rotation_z_profile = []
    for i in range(start,end + 1):
        image_loop = rotate_volume(image, i,'x',centre)
        volume = image_loop.get_fdata(dtype=np.float32)
        # z_profile = calculate_z_profile(volume)
        z_profile = calculate_z_profile_hot_spot(volume)
        z_centre = find_z_argmax(z_profile)
        z_value_in_centre = z_profile[z_centre]
        rotation_z_profile.append(z_value_in_centre)
    return rotation_z_profile