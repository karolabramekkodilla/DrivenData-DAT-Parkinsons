from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_to_output
from resources.cut_off_edges import cut_off_edges
from resources.standarize_image import standardize_image


def main():
    project_dir = Path(__file__).resolve().parents[1]
    # create_centres_csv(
    #     project_dir=project_dir,
    #     output_file_name="scan_centres.csv",
    #     filter_width=3,
    #     smooth_window=5,
    # )
    niftis_dir = project_dir / "data" / "niftis"
    outputs_dir = project_dir / "src" / "outputs" / "cut_images"

    outputs_dir.mkdir(parents=True, exist_ok=True)

    nifti_paths = sorted(niftis_dir.glob("*.nii.gz"))

    for index, nifti_path in enumerate(nifti_paths, start=1):

        # Odczyt obrazu jako obiekt NIfTI
        image = nib.load(nifti_path)

        # Obróbka obrazu; funkcja zwraca obiekt NIfTI
        processed_image = cut_off_edges(image)
        processed_image = cut_off_edges(processed_image,x_left=35,x_right=35,y_front=40,y_back=40,z_down=30,z_up=30)
        # Standaryzacja obrazu
        processed_image = standardize_image(processed_image, percent_top=0.02)

        last_cut = cut_off_edges(processed_image, x_left=30, x_right=30, y_front=30, y_back=30, z_down=25, z_up=30)
        resampled_image = resample_to_output(
            last_cut,
            voxel_sizes=(2.46, 2.46, 2.46),
            order=1,
        )


        # Zapis pod tą samą nazwą do outputs
        output_path = outputs_dir / nifti_path.name
        nib.save(resampled_image, output_path)

        print(
            f"[{index}/{len(nifti_paths)}] "
            f"Zapisano: {output_path.name}"
        )

if __name__ == "__main__":
    main()
