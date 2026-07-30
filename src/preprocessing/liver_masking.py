"""
Apply a liver segmentation mask to 3D CT volumes.

For each case, the script:

1. Loads image.nii.gz and segmentation.nii.gz.
2. Verifies that the image and segmentation have matching dimensions.
3. Retains CT intensities inside the liver segmentation.
4. Sets voxels outside the liver to -250 HU.
5. Saves the masked volume as masked_image.nii.gz.

Expected input structure:

input_directory/
├── case_001/
│   ├── image.nii.gz
│   └── segmentation.nii.gz
├── case_002/
│   ├── image.nii.gz
│   └── segmentation.nii.gz
└── ...

Example:

python liver_masking.py \
    --input-dir /path/to/input_directory \
    --output-dir /path/to/output_directory \
    --n-jobs 8
"""

import argparse
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


BACKGROUND_HU = -250.0


def apply_liver_mask(
    image_path: Path,
    segmentation_path: Path,
    output_path: Path,
) -> str:
    """
    Apply a liver segmentation mask to one CT volume.

    Parameters
    ----------
    image_path:
        Path to the input CT image.
    segmentation_path:
        Path to the corresponding liver segmentation.
    output_path:
        Path where the masked CT image will be saved.

    Returns
    -------
    str
        Processing status message.
    """
    if output_path.exists():
        return f"SKIP: {output_path.parent.name}"

    if not image_path.exists():
        return f"MISSING IMAGE: {image_path}"

    if not segmentation_path.exists():
        return f"MISSING SEGMENTATION: {segmentation_path}"

    try:
        image_nifti = nib.load(str(image_path))
        segmentation_nifti = nib.load(str(segmentation_path))

        image = image_nifti.get_fdata(dtype=np.float32)
        segmentation = segmentation_nifti.get_fdata()

        if image.shape != segmentation.shape:
            return (
                f"SHAPE MISMATCH: {image_path.parent.name} "
                f"image={image.shape}, segmentation={segmentation.shape}"
            )

        liver_mask = segmentation > 0

        masked_image = np.where(
            liver_mask,
            image,
            BACKGROUND_HU,
        ).astype(np.float32)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_nifti = nib.Nifti1Image(
            masked_image,
            affine=image_nifti.affine,
            header=image_nifti.header.copy(),
        )

        output_nifti.set_data_dtype(np.float32)

        nib.save(
            output_nifti,
            str(output_path),
        )

        return f"OK: {image_path.parent.name}"

    except Exception as error:
        return f"ERROR: {image_path.parent.name} -> {error}"


def process_case(
    case_directory: Path,
    output_directory: Path,
    image_name: str,
    segmentation_name: str,
) -> Optional[str]:
    """
    Process one case directory.
    """
    if not case_directory.is_dir():
        return None

    image_path = case_directory / image_name
    segmentation_path = case_directory / segmentation_name

    output_path = (
        output_directory
        / case_directory.name
        / "masked_image.nii.gz"
    )

    return apply_liver_mask(
        image_path=image_path,
        segmentation_path=segmentation_path,
        output_path=output_path,
    )


def run_masking(
    input_directory: Path,
    output_directory: Path,
    image_name: str,
    segmentation_name: str,
    n_jobs: int,
) -> None:
    """
    Apply liver masks to all valid case directories.
    """
    if not input_directory.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_directory}"
        )

    case_directories = sorted(
        path
        for path in input_directory.iterdir()
        if path.is_dir()
    )

    if not case_directories:
        raise RuntimeError(
            f"No case directories were found in {input_directory}."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Cases found: {len(case_directories)}")
    print(f"Background value: {BACKGROUND_HU} HU")
    print(f"Parallel jobs: {n_jobs}")

    results = Parallel(
        n_jobs=n_jobs,
        backend="threading",
    )(
        delayed(process_case)(
            case_directory=case_directory,
            output_directory=output_directory,
            image_name=image_name,
            segmentation_name=segmentation_name,
        )
        for case_directory in tqdm(
            case_directories,
            desc="Applying liver masks",
        )
    )

    results = [
        result
        for result in results
        if result is not None
    ]

    successful = [
        result
        for result in results
        if result.startswith("OK")
        or result.startswith("SKIP")
    ]

    problems = [
        result
        for result in results
        if not (
            result.startswith("OK")
            or result.startswith("SKIP")
        )
    ]

    for result in results:
        print(result)

    if problems:
        log_path = output_directory / "masking_issues.log"

        with open(
            log_path,
            "w",
            encoding="utf-8",
        ) as log_file:
            log_file.write("\n".join(problems))

        print(f"Issue log saved to: {log_path}")

    print("\nLiver masking complete.")
    print(f"Successful or skipped: {len(successful)}")
    print(f"Problems: {len(problems)}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Apply liver segmentation masks to 3D CT volumes."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing one folder per case."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory where masked CT volumes will be saved."
        ),
    )

    parser.add_argument(
        "--image-name",
        type=str,
        default="image.nii.gz",
        help=(
            "Filename of the CT image inside each case folder."
        ),
    )

    parser.add_argument(
        "--segmentation-name",
        type=str,
        default="segmentation.nii.gz",
        help=(
            "Filename of the liver segmentation inside each case folder."
        ),
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help=(
            "Number of parallel workers. "
            "Use -1 to use all available CPU cores."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Run liver masking.
    """
    arguments = parse_arguments()

    run_masking(
        input_directory=arguments.input_dir,
        output_directory=arguments.output_dir,
        image_name=arguments.image_name,
        segmentation_name=arguments.segmentation_name,
        n_jobs=arguments.n_jobs,
    )


if __name__ == "__main__":
    main()
