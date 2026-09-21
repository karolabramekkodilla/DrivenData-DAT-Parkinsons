from resources.calculate_similarity import calculate_similarity
from resources.cut_off_edges import cut_off_edges
from resources.find_x import find_x_area_center, calculate_x_profile, calculate_x_profile_hot_spot, find_x_argmax
from resources.find_y import find_y_area_center,calculate_y_profile
from resources.find_z import find_z_area_center, calculate_z_profile, find_z_argmax
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
    for i in range(-15,15):
        image_loop = rotate_volume(image, i,'y',centre)
        volume = image_loop.get_fdata(dtype=np.float32)
        x_profile = calculate_x_profile(volume)
        x_centre = find_x_area_center(x_profile)
        x_minimum = x_profile[x_centre]
        if x_minimum < x_best_min:
            x_best_min = x_minimum
            error = i

    return error

def create_rotation_x_profile_tilt(image, start = 0, end = 360):
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
    rotation_x_profile = create_rotation_x_profile_tilt(image, 60, 120)
    rotation_x_profile = smooth_1d(rotation_x_profile, window_size=19)
    er1 = np.argmax(rotation_x_profile) - 30
    rotation_x_profile = create_rotation_x_profile_tilt(image, 150, 210)
    rotation_x_profile = smooth_1d(rotation_x_profile, window_size=19)
    er2 = np.argmin(rotation_x_profile) - 30
    rotation_x_profile = create_rotation_x_profile_tilt(image, 240, 300)
    rotation_x_profile = smooth_1d(rotation_x_profile, window_size=19)
    er3 = np.argmax(rotation_x_profile) - 30
    rotation_x_profile = create_rotation_x_profile_tilt(image, 330, 490)
    rotation_x_profile = smooth_1d(rotation_x_profile, window_size=19)
    er4 = np.argmin(rotation_x_profile) - 30
    errors = [er1, er2, er3, er4]
    final_error = min(errors, key=abs)
    return int(final_error)


def find_tilt_error_by_similarity(
    image,
    template_image,
):
    # Obraz przed końcowym przycięciem.
    volume = image.get_fdata(
        dtype=np.float32,
    )

    # Środek obrotu wyznaczamy na obrazie nieprzyciętym.
    x_profile = calculate_x_profile(volume)
    x_centre = find_x_area_center(x_profile)

    y_profile = calculate_y_profile(volume)
    y_centre = find_y_area_center(y_profile)

    z_profile = calculate_z_profile(volume)
    z_centre = find_z_argmax(z_profile)

    centre = (
        x_centre,
        y_centre,
        z_centre,
    )

    # Template nie jest obracany, więc przycinamy go tylko raz.
    cropped_template_image = cut_off_edges(
        template_image,
        x_left=25,
        x_right=25,
        y_front=30,
        y_back=30,
        z_down=20,
        z_up=20,
        z_center_method=find_z_argmax
    )

    template_volume = (
        cropped_template_image.get_fdata(
            dtype=np.float32,
        )
    )

    best_similarity = -np.inf
    best_angle = 0

    for angle in range(-25, 26):
        # Najpierw obracamy pełny obraz.
        rotated_image = rotate_volume(
            image=image,
            angle=angle,
            axis="y",
            center=centre,
        )

        # Dopiero później usuwamy czarne krawędzie.
        cropped_rotated_image = cut_off_edges(
            rotated_image,
            x_left=25,
            x_right=25,
            y_front=30,
            y_back=30,
            z_down=20,
            z_up=20,
            z_center_method=find_z_argmax
        )

        rotated_volume = (
            cropped_rotated_image.get_fdata(
                dtype=np.float32,
            )
        )

        if rotated_volume.shape != template_volume.shape:
            raise ValueError(
                f"Różne rozmiary po przycięciu: "
                f"image={rotated_volume.shape}, "
                f"template={template_volume.shape}"
            )

        similarity = calculate_similarity(
            rotated_volume,
            template_volume,
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_angle = angle

    return best_angle