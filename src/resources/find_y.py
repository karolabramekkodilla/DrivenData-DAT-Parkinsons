import numpy as np

from resources.smooth_1d import smooth_1d


def calculate_y_profile(volume, filter_width=3):
    x_max, y_max, z_max = volume.shape
    profile = np.zeros(y_max - filter_width + 1)

    for y_pos in range(y_max - filter_width + 1):
        area = volume[:, y_pos:y_pos + filter_width, :]
        profile[y_pos] = np.sum(area)

    return profile


def find_y_argmax(
        y_profile,
        filter_width=3,
        smooth_window=9,
        edge_margin_ratio=0.10
):
    y_profile = np.asarray(y_profile, dtype=float)
    smoothed = smooth_1d(y_profile, window_size=smooth_window)

    n = len(smoothed)
    edge_margin = int(n * edge_margin_ratio)

    search_start = edge_margin
    search_end = n - edge_margin

    if search_end <= search_start:
        search_start = 0
        search_end = n

    y_peak_start = int(
        np.argmax(smoothed[search_start:search_end]) + search_start
    )
    y_peak_center = y_peak_start + filter_width // 2

    return y_peak_center


def find_y_geometric_center(y_profile, filter_width=3):
    y_profile = np.asarray(y_profile, dtype=float)

    y_center = len(y_profile) // 2 + filter_width // 2

    return y_center