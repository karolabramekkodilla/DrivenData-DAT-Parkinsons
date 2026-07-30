from pathlib import Path

from resources.reload_data import create_centres_csv


def main():
    project_dir = Path(__file__).resolve().parents[1]

    create_centres_csv(
        project_dir=project_dir,
        output_file_name="scan_centres.csv",
        filter_width=3,
        smooth_window=5,
    )


if __name__ == "__main__":
    main()
