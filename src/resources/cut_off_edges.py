
import nibabel as nib
import numpy as np
from resources.find_x import find_x_area_center_split,calculate_x_profile, find_x_area_center
from resources.find_y import find_y_argmax, calculate_y_profile, find_y_area_center
from resources.find_z import find_z_argmax,calculate_z_profile, find_z_area_center, find_z_center


def cut_off_edges(
    image: nib.Nifti1Image,
    x_left: int = 40,
    x_right: int = 40,
    y_front: int = 50,
    y_back: int = 50,
    z_down: int = 40,
    z_up: int = 40,
    target_voxel_size: float = 2.46,
    z_center_method=find_z_area_center,
    z_filter_width: int = 3,
) -> nib.Nifti1Image:

    x_voxel_size, y_voxel_size, z_voxel_size = (
        image.header.get_zooms()[:3]
    )

    # Liczba voxeli potrzebna w obrazie źródłowym,
    # aby objąć ten sam zakres w milimetrach.
    x_left_source = round(
        x_left * target_voxel_size / x_voxel_size
    )
    x_right_source = round(
        x_right * target_voxel_size / x_voxel_size
    )

    y_front_source = round(
        y_front * target_voxel_size / y_voxel_size
    )
    y_back_source = round(
        y_back * target_voxel_size / y_voxel_size
    )

    z_down_source = round(
        z_down * target_voxel_size / z_voxel_size
    )
    z_up_source = round(
        z_up * target_voxel_size / z_voxel_size
    )

    volume = image.get_fdata(dtype=np.float32)

    x_profile = calculate_x_profile(volume)
    y_profile = calculate_y_profile(volume)

    x_centre = int(find_x_area_center(x_profile))
    # x_centre = int(find_x_area_center_split(x_profile))

    y_centre = int(find_y_area_center(y_profile))

    z_profile = calculate_z_profile(
        volume,
        filter_width=z_filter_width,
    )

    z_centre = int(
        z_center_method(
            z_profile,
            filter_width=z_filter_width,
        )
    )

    x_left_cut = x_centre - x_left_source
    x_right_cut = x_centre + x_right_source

    y_front_cut = y_centre - y_front_source
    y_back_cut = y_centre + y_back_source

    z_down_cut = z_centre - z_down_source
    z_up_cut = z_centre + z_up_source

    cropped_volume = cut_with_padding(
        volume=volume,
        x_start=x_left_cut,
        x_end=x_right_cut,
        y_start=y_front_cut,
        y_end=y_back_cut,
        z_start=z_down_cut,
        z_end=z_up_cut,
    )

    # Przesunięcie affine, ponieważ punkt [0, 0, 0]
    # nowego obrazu odpowiada pozycji x_left_cut itd.
    new_affine = image.affine.copy()

    voxel_shift = np.array([
        x_left_cut,
        y_front_cut,
        z_down_cut,
    ])

    new_affine[:3, 3] += (
        image.affine[:3, :3] @ voxel_shift
    )

    new_header = image.header.copy()
    new_header.set_data_shape(cropped_volume.shape)

    cropped_image = nib.Nifti1Image(
        cropped_volume,
        new_affine,
        new_header,
    )

    return cropped_image

def cut_with_padding(
    volume: np.ndarray,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    z_start: int,
    z_end: int,
) -> np.ndarray:

    output_shape = (
        x_end - x_start,
        y_end - y_start,
        z_end - z_start,
    )

    cropped = np.zeros(
        output_shape,
        dtype=volume.dtype,
    )

    # Zakres istniejący w obrazie źródłowym
    source_x_start = max(0, x_start)
    source_x_end = min(volume.shape[0], x_end)

    source_y_start = max(0, y_start)
    source_y_end = min(volume.shape[1], y_end)

    source_z_start = max(0, z_start)
    source_z_end = min(volume.shape[2], z_end)

    # Sprawdzenie, czy istnieje jakakolwiek część wspólna
    if (
        source_x_start >= source_x_end
        or source_y_start >= source_y_end
        or source_z_start >= source_z_end
    ):
        return cropped

    # Miejsce w obrazie wynikowym
    target_x_start = source_x_start - x_start
    target_y_start = source_y_start - y_start
    target_z_start = source_z_start - z_start

    target_x_end = target_x_start + (source_x_end - source_x_start)
    target_y_end = target_y_start + (source_y_end - source_y_start)
    target_z_end = target_z_start + (source_z_end - source_z_start)

    cropped[
        target_x_start:target_x_end,
        target_y_start:target_y_end,
        target_z_start:target_z_end,
    ] = volume[
        source_x_start:source_x_end,
        source_y_start:source_y_end,
        source_z_start:source_z_end,
    ]

    return cropped