"""
Preprocess masked 3D liver CT volumes.

For each case, the script:

1. Loads masked_image.nii.gz.
2. Resamples the CT volume to 1 x 1 x 1 mm isotropic spacing.
3. Clips intensities to [-250, 250] HU.
4. Applies fixed min-max normalization to [0, 1].
5. Saves the processed volume as volume.npy.

Expected input structure:

input_directory/
├── case_001/
│   └── masked_image.nii.gz
├── case_002/
│   └── masked_image.nii.gz
└── ...

Example:

python ct_preprocessing.py \
    --input-dir /path/to/masked_data \
    --output-dir /path/to/preprocessed_data \
    --n-jobs 8
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from joblib import Parallel, delayed
from tqdm import tqdm


TARGET_SPACING = (1.0, 1.0, 1.0)
HU_MIN = -250.0
HU_MAX = 250.0


def resample_volume(
    image: sitk.Image,
    target_spacing: tuple[float, float, float],
) -> sitk.Image:
    """
    Resample a 3D CT volume to the requested isotropic spacing.

    Linear interpolation is used because the input is a continuous
    CT intensity image.

    Parameters
    ----------
    image:
        Input SimpleITK image.

    target_spacing:
        Target voxel spacing in x, y, z order.

    Returns
    -------
    sitk.Image
        Resampled CT image.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(
            round(
                original_size[i]
                * original_spacing[i]
                / target_spacing[i]
            )
        )
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(HU_MIN)
    resampler.SetOutputPixelType(sitk.sitkFloat32)

    return resampler.Execute(image)


def preprocess_case(
    case_directory: Path,
    output_directory: Path,
    input_filename: str,
) -> str:
    """
    Preprocess one masked CT volume.
    """
    case_id = case_directory.name

    input_path = case_directory / input_filename
    output_case_directory = output_directory / case_id
    output_path = output_case_directory / "volume.npy"

    if output_path.exists():
        return f"SKIP: {case_id}"

    if not input_path.exists():
        return f"MISSING: {case_id}"

    try:
        image = sitk.ReadImage(
            str(input_path),
            sitk.sitkFloat32,
        )

        if image.GetDimension() != 3:
            raise ValueError(
                f"Expected a 3D image, received "
                f"{image.GetDimension()} dimensions."
            )

        resampled_image = resample_volume(
            image=image,
            target_spacing=TARGET_SPACING,
        )

        # SimpleITK converts images to NumPy arrays in z, y, x order.
        volume = sitk.GetArrayFromImage(
            resampled_image
        ).astype(np.float32)

        volume = np.nan_to_num(
            volume,
            nan=HU_MIN,
            posinf=HU_MAX,
            neginf=HU_MIN,
        )

        volume = np.clip(
            volume,
            HU_MIN,
            HU_MAX,
        ).astype(np.float32)

        volume = (
            (volume - HU_MIN)
            / (HU_MAX - HU_MIN)
        ).astype(np.float32)

        output_case_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            output_path,
            volume,
        )

        return f"OK: {case_id}"

    except Exception as error:
        return f"ERROR: {case_id} -> {error}"


def run_preprocessing(
    input_directory: Path,
    output_directory: Path,
    input_filename: str,
    n_jobs: int,
) -> None:
    """
    Preprocess all case directories in parallel.
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
    print(f"Target spacing: {TARGET_SPACING} mm")
    print(f"HU range: [{HU_MIN}, {HU_MAX}]")
    print("Normalization range: [0, 1]")
    print(f"Parallel jobs: {n_jobs}")

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
    )(
        delayed(preprocess_case)(
            case_directory=case_directory,
            output_directory=output_directory,
            input_filename=input_filename,
        )
        for case_directory in tqdm(
            case_directories,
            desc="Preprocessing CT volumes",
        )
    )

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
        log_path = output_directory / "preprocessing_issues.log"

        with open(
            log_path,
            "w",
            encoding="utf-8",
        ) as log_file:
            log_file.write("\n".join(problems))

        print(f"Issue log saved to: {log_path}")

    print("\nCT preprocessing complete.")
    print(f"Successful or skipped: {len(successful)}")
    print(f"Problems: {len(problems)}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Resample, clip, and normalize masked "
            "3D liver CT volumes."
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
            "Directory where preprocessed NumPy volumes "
            "will be saved."
        ),
    )

    parser.add_argument(
        "--input-filename",
        type=str,
        default="masked_image.nii.gz",
        help=(
            "Filename of the masked CT image inside "
            "each case folder."
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
    Run CT preprocessing.
    """
    arguments = parse_arguments()

    run_preprocessing(
        input_directory=arguments.input_dir,
        output_directory=arguments.output_dir,
        input_filename=arguments.input_filename,
        n_jobs=arguments.n_jobs,
    )


if __name__ == "__main__":
    main()
