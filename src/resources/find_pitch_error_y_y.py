from resources.calculate_similarity import calculate_similarity
from resources.cut_off_edges import cut_off_edges
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
    for i in range(-20,20):
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

def find_pitch_error_by_similarity(
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
            axis="x",
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