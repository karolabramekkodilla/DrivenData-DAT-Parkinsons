import numpy as np

from resources.smooth_1d import smooth_1d


def calculate_z_profile(volume, filter_width=3):
    x_max, y_max, z_max = volume.shape
    profile = np.zeros(z_max - filter_width + 1)

    for z_pos in range(z_max - filter_width + 1):
        area = volume[:, :, z_pos:z_pos + filter_width]
        profile[z_pos] = np.sum(area)

    return profile


def find_z_argmax(
        z_profile,
        filter_width=3,
        smooth_window=9,
        edge_margin_ratio=0.10
):
    z_profile = np.asarray(z_profile, dtype=float)
    smoothed = smooth_1d(z_profile, window_size=smooth_window)

    n = len(smoothed)
    edge_margin = int(n * edge_margin_ratio)

    search_start = edge_margin
    search_end = n - edge_margin

    if search_end <= search_start:
        search_start = 0
        search_end = n

    z_peak_start = int(
        np.argmax(smoothed[search_start:search_end]) + search_start
    )
    z_peak_center = z_peak_start + filter_width // 2

    return z_peak_center

def find_z_area_center(
        z_profile,
        filter_width=3,
        smooth_window=9,
        edge_margin_ratio=0.10
):
    z_profile = np.asarray(z_profile, dtype=float)
    smoothed = smooth_1d(z_profile, window_size=smooth_window)

    n = len(smoothed)
    edge_margin = int(n * edge_margin_ratio)

    search_start = edge_margin
    search_end = n - edge_margin

    if search_end <= search_start:
        search_start = 0
        search_end = n

    search_values = smoothed[search_start:search_end]

    total_area = np.sum(search_values)

    if total_area <= 0:
        return n // 2 + filter_width // 2

    cumulative_area = np.cumsum(search_values)
    half_area = total_area / 2

    z_area_center_start = int(
        np.searchsorted(cumulative_area, half_area) + search_start
    )

    z_area_center = z_area_center_start + filter_width // 2

    return z_area_center

def calculate_z_profile_hot_spot(
    volume,
    center=None,
    filter_width=3,
    x_left_scope=0.3,
    x_right_scope=0.3,
    y_front_scope=0.3,
    y_back_scope=0.3,
    z_up_scope=0.15,
    z_down_scope=0.15,
):
    x_size, y_size, z_size = volume.shape

    if center is None:
        x_center = x_size // 2
        y_center = y_size // 2
        z_center = z_size // 2
    else:
        x_center, y_center, z_center = center

    x_min = max(0, int(x_center - x_left_scope * x_size))
    x_max = min(x_size - filter_width + 1, int(x_center + x_right_scope * x_size))

    y_min = max(0, int(y_center - y_back_scope * y_size))
    y_max = min(y_size, int(y_center + y_front_scope * y_size))

    z_min = max(0, int(z_center - z_down_scope * z_size))
    z_max = min(z_size, int(z_center + z_up_scope * z_size))

    profile = np.zeros(z_size - filter_width + 1)

    for z_pos in range(z_min, z_max):
        area = volume[
            x_min:x_max,
            y_min:y_max,
            z_pos:z_pos + filter_width,
        ]
        profile[z_pos] = np.sum(area)

    return profile