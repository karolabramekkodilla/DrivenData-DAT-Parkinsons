from ipywidgets import interact, IntSlider
import matplotlib.pyplot as plt
def create_mri_view(uid, volume, diagnosis):

    @interact(
        x=IntSlider(
            min=0,
            max=volume.shape[0] - 1,
            value=volume.shape[0] // 2,
            continuous_update=False
        ),
        y=IntSlider(
            min=0,
            max=volume.shape[1] - 1,
            value=volume.shape[1] // 2,
            continuous_update=False
        ),
        z=IntSlider(
            min=0,
            max=volume.shape[2] - 1,
            value=volume.shape[2] // 2,
            continuous_update=False
        )
    )
    def show_mri(x, y, z):

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(
            volume[x, :, :].T,
            cmap="gray",
            origin="lower"
        )

        ax[1].imshow(
            volume[:, y, :].T,
            cmap="gray",
            origin="lower"
        )

        ax[2].imshow(
            volume[:, :, z].T,
            cmap="gray",
            origin="lower"
        )

        ax[0].set_title(f"Oś 0, przekrój {x}")
        ax[1].set_title(f"Oś 1, przekrój {y}")
        ax[2].set_title(f"Oś 2, przekrój {z}")

        ax[0].set_xlabel("Y")
        ax[0].set_ylabel("Z")

        ax[1].set_xlabel("X")
        ax[1].set_ylabel("Z")

        ax[2].set_xlabel("X")
        ax[2].set_ylabel("Y")

        fig.suptitle(
            f"UID: {uid} | {diagnosis} | shape={volume.shape}",
            fontsize=14
        )

        plt.tight_layout()
        plt.show()