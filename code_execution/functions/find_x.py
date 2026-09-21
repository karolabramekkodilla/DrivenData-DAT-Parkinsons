import numpy as np
from functions.smooth_1d import smooth_1d

def calculate_x_profile(volume, filter_width=3):
    x_max, y_max, z_max = volume.shape
    profile = np.zeros(x_max - filter_width + 1)

    for x_pos in range(x_max - filter_width + 1):
        area = volume[x_pos:x_pos + filter_width, :, :]
        profile[x_pos] = np.sum(area)

    return profile
def calculate_x_profile_with_margin(volume, filter_width=3, margin=0.10):
    x_max, y_max, z_max = volume.shape
    y_min = int(margin * y_max)
    y_max = int((1 - margin) * y_max)
    z_min = int(margin * z_max)
    z_max = int((1 - margin) * z_max)
    profile = np.zeros(x_max - filter_width + 1)
    for x_pos in range(x_max - filter_width + 1):
        area = volume[x_pos:x_pos + filter_width, y_min:y_max, z_min:z_max]
        profile[x_pos] = np.sum(area)

    return profile

def calculate_x_profile_hot_spot(
    volume,
    center=None,
    filter_width=3,
    x_left_scope=0.3,
    x_right_scope=0.3,
    y_front_scope=0.3,
    y_back_scope=0,
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

    profile = np.zeros(x_size - filter_width + 1)

    for x_pos in range(x_min, x_max):
        area = volume[
            x_pos:x_pos + filter_width,
            y_min:y_max,
            z_min:z_max,
        ]
        profile[x_pos] = np.sum(area)

    return profile

def find_x_area_center(
        x_profile,
        filter_width=3,
        smooth_window=9,
        edge_margin_ratio=0.10
):
    x_profile = np.asarray(x_profile, dtype=float)
    smoothed = smooth_1d(x_profile, window_size=smooth_window)

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

    x_area_center_start = int(
        np.searchsorted(cumulative_area, half_area) + search_start
    )

    x_area_center = x_area_center_start + filter_width // 2

    return x_area_center

def find_x_argmax(
        x_profile,
        filter_width=3,
        smooth_window=9,
        edge_margin_ratio=0.10
):
    x_profile = np.asarray(x_profile, dtype=float)
    smoothed = smooth_1d(x_profile, window_size=smooth_window)

    n = len(smoothed)
    edge_margin = int(n * edge_margin_ratio)

    search_start = edge_margin
    search_end = n - edge_margin

    if search_end <= search_start:
        search_start = 0
        search_end = n

    x_peak_start = int(
        np.argmax(smoothed[search_start:search_end]) + search_start
    )
    x_peak_center = x_peak_start + filter_width // 2

    return x_peak_center

def find_x_area_center_split(
        x_profile,
        filter_width=3,
        smooth_window=9,
        edge_margin_ratio=0.15,
        local_search_ratio=0.10
):
    x_profile = np.asarray(x_profile, dtype=float)
    smoothed = smooth_1d(x_profile, window_size=smooth_window)

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

    x_area_center_start = int(
        np.searchsorted(cumulative_area, half_area) + search_start
    )

    local_half_width = int(n * local_search_ratio / 2)

    local_start = max(search_start, x_area_center_start - local_half_width)
    local_end = min(search_end, x_area_center_start + local_half_width + 1)

    local_minima = []

    for x_pos in range(local_start + 1, local_end - 1):
        if smoothed[x_pos] <= smoothed[x_pos - 1] and smoothed[x_pos] <= smoothed[x_pos + 1]:
            local_minima.append(x_pos)

    if local_minima:
        x_split_start = min(
            local_minima,
            key=lambda pos: abs(pos - x_area_center_start)
        )
    else:
        x_split_start = int(
            np.argmin(smoothed[local_start:local_end]) + local_start
        )

    x_split = x_split_start + filter_width // 2

    return x_split