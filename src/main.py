from pathlib import Path
import nibabel as nib
import pandas as pd
from resources.cut_off_edges import cut_off_edges
from resources.find_save_centre import create_centres_csv




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

        # Zapis pod tą samą nazwą do outputs
        output_path = outputs_dir / nifti_path.name
        nib.save(processed_image, output_path)

        print(
            f"[{index}/{len(nifti_paths)}] "
            f"Zapisano: {output_path.name}"
        )






if __name__ == "__main__":
    main()
