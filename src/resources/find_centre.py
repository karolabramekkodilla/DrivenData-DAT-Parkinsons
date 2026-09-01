from resources.find_x import calculate_x_profile, find_x_area_center
from resources.find_y import calculate_y_profile, find_y_area_center
from resources.find_z import calculate_z_profile, find_z_area_center


def find_centre(volume):

        image = volume.get_fdata()
        x_profile = calculate_x_profile(image)
        x_centre = find_x_area_center(x_profile)
        y_profile = calculate_y_profile(image)
        y_center = find_y_area_center(y_profile)
        z_profile = calculate_z_profile(image)
        z_center = find_z_area_center(z_profile)

        return (x_centre,y_center,z_center)