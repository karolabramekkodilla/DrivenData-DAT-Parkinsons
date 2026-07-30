import numpy as np

def smooth_1d(values, window_size=9):
    values = np.asarray(values, dtype=float)

    if window_size <= 1:
        return values

    if window_size % 2 == 0:
        window_size += 1

    pad = window_size // 2
    padded = np.pad(values, pad_width=pad, mode="edge")

    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode="valid")