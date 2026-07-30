import numpy as np
from resources.smooth_1d import smooth_1d

def calculate_x_profile(volume, filter_width=3):
    x_max, y_max, z_max = volume.shape
    profile = np.zeros(x_max - filter_width + 1)

    for x_pos in range(x_max - filter_width + 1):
        area = volume[x_pos:x_pos + filter_width, :, :]
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

def find_x_area_center_split(
        x_profile,
        filter_width=3,
        smooth_window=9,
        edge_margin_ratio=0.10,
        local_search_ratio=0.20
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