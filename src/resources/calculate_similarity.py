
import numpy as np
def calculate_similarity(volume, template_volume):
    """
    Suma iloczynów odpowiadających sobie wokseli.
    Większy wynik oznacza lepsze nałożenie aktywnych obszarów.
    """
    volume_flat = np.asarray(
        volume,
        dtype=np.float32,
    ).ravel()

    template_flat = np.asarray(
        template_volume,
        dtype=np.float32,
    ).ravel()

    if volume_flat.shape != template_flat.shape:
        raise ValueError(
            f"Różne rozmiary wektorów: "
            f"volume={volume_flat.shape}, "
            f"template={template_flat.shape}"
        )

    return float(
        np.sum(volume_flat * template_flat)
    )