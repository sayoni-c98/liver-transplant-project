"""
Resize preprocessed 3D liver CT volumes to a fixed shape.

For each case, the script:

1. Loads volume.npy.
2. Resizes the volume to 96 x 96 x 96 voxels.
3. Uses linear interpolation.
4. Saves the resized volume as volume.npy.

Expected input structure:

input_directory/
├── case_001/
│   └── volume.npy
├── case_002/
│   └── volume.npy
└── ...

Example:

python volume_resizing.py \
    --input-dir /path/to/preprocessed_data \
    --output-dir /path/to/resized_data \
    --n-jobs 8
"""

import argparse
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import zoom
from tqdm import tqdm


TARGET_SHAPE = (96, 96, 96)


def resize_volume(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Resize a 3D volume to the requested output shape.

    Linear interpolation is used for continuous CT intensity values.

    Parameters
    ----------
    volume:
        Input three-dimensional NumPy array.

    target_shape:
        Desired output shape in z, y, x order.

    Returns
    -------
    np.ndarray
        Resized three-dimensional volume.
    """
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume, received shape {volume.shape}."
        )

    scale_factors = np.asarray(
        target_shape,
        dtype=np.float64,
    ) / np.asarray(
        volume.shape,
        dtype=np.float64,
    )

    resized_volume = zoom(
        volume,
        zoom=scale_factors,
        order=1,
        mode="nearest",
        prefilter=False,
    )

    return resized_volume.astype(np.float32)


def resize_case(
    case_directory: Path,
    output_directory: Path,
    input_filename: str,
) -> str:
    """
    Resize one preprocessed CT volume.
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
        volume = np.load(
            input_path,
            mmap_mode="r",
        )

        resized_volume = resize_volume(
            volume=volume,
            target_shape=TARGET_SHAPE,
        )

        if resized_volume.shape != TARGET_SHAPE:
            raise RuntimeError(
                f"Unexpected resized shape {resized_volume.shape}; "
                f"expected {TARGET_SHAPE}."
            )

        output_case_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            output_path,
            resized_volume,
        )

        return f"OK: {case_id}"

    except Exception as error:
        return f"ERROR: {case_id} -> {error}"


def run_resizing(
    input_directory: Path,
    output_directory: Path,
    input_filename: str,
    n_jobs: int,
) -> None:
    """
    Resize all available CT volumes in parallel.
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
    print(f"Target shape: {TARGET_SHAPE}")
    print("Interpolation: linear")
    print(f"Parallel jobs: {n_jobs}")

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
    )(
        delayed(resize_case)(
            case_directory=case_directory,
            output_directory=output_directory,
            input_filename=input_filename,
        )
        for case_directory in tqdm(
            case_directories,
            desc="Resizing CT volumes",
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
        log_path = output_directory / "resizing_issues.log"

        with open(
            log_path,
            "w",
            encoding="utf-8",
        ) as log_file:
            log_file.write("\n".join(problems))

        print(f"Issue log saved to: {log_path}")

    print("\nVolume resizing complete.")
    print(f"Successful or skipped: {len(successful)}")
    print(f"Problems: {len(problems)}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Resize preprocessed 3D liver CT volumes "
            "to 96 x 96 x 96 voxels."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing one folder per case.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where resized volumes will be saved.",
    )

    parser.add_argument(
        "--input-filename",
        type=str,
        default="volume.npy",
        help=(
            "Filename of the preprocessed NumPy volume "
            "inside each case folder."
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
    Run volume resizing.
    """
    arguments = parse_arguments()

    run_resizing(
        input_directory=arguments.input_dir,
        output_directory=arguments.output_dir,
        input_filename=arguments.input_filename,
        n_jobs=arguments.n_jobs,
    )


if __name__ == "__main__":
    main()
