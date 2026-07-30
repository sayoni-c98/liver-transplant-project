"""
Sliding-window topological feature extraction for 3D liver CT volumes.

The script computes Betti-0, Betti-1, and Betti-2 features using:

- CT intensity range: [-250, 250] HU normalized to [0, 1]
- Sliding-window width: 30 HU, equivalent to 0.06 after normalization
- Stride: 10 HU, equivalent to 0.02 after normalization
- Number of intensity windows: 50
- Output features: 150 features per CT volume
    - 50 Betti-0 features
    - 50 Betti-1 features
    - 50 Betti-2 features

Expected input structure:

input_directory/
├── patient_001/
│   └── volume.npy
├── patient_002/
│   └── volume.npy
└── ...

Example:

python sliding_window_betti_30hu.py \
    --input-dir /path/to/input_directory \
    --output-dir /path/to/output_directory \
    --n-jobs 8
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import convolve
from scipy.ndimage import label as connected_component_label
from tqdm import tqdm


# Normalized CT intensity range.
# Original HU range: [-250, 250] HU.
NORMALIZED_MIN = 0.0
NORMALIZED_MAX = 1.0

# 30 HU / 500 HU = 0.06.
WINDOW_WIDTH = 0.06

# 10 HU / 500 HU = 0.02.
STRIDE = 0.02

# Number of sliding-window intensity anchors.
N_WINDOWS = 50

# 26-connectivity for 3D foreground and background components.
CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)


def compute_euler_characteristic(binary_volume: np.ndarray) -> int:
    """
    Compute the Euler characteristic of a 3D binary cubical complex.

    The Euler characteristic is computed as:

        chi = V - E + F - C

    where:

        V = number of vertices
        E = number of edges
        F = number of faces
        C = number of cubes

    Parameters
    ----------
    binary_volume:
        Three-dimensional binary NumPy array.

    Returns
    -------
    int
        Euler characteristic of the binary volume.
    """
    binary_volume = (
        binary_volume.astype(np.uint8) == 1
    ).astype(np.uint8)

    vertex_kernel = np.ones((2, 2, 2), dtype=int)

    edge_x_kernel = np.ones((1, 2, 2), dtype=int)
    edge_y_kernel = np.ones((2, 1, 2), dtype=int)
    edge_z_kernel = np.ones((2, 2, 1), dtype=int)

    face_xy_kernel = np.ones((1, 1, 2), dtype=int)
    face_yz_kernel = np.ones((2, 1, 1), dtype=int)
    face_zx_kernel = np.ones((1, 2, 1), dtype=int)

    n_vertices = np.sum(
        convolve(
            binary_volume,
            vertex_kernel,
            mode="constant",
            cval=0,
        )
        == 1
    )

    n_edges = (
        np.sum(
            convolve(
                binary_volume,
                edge_x_kernel,
                mode="constant",
                cval=0,
            )
            == 1
        )
        + np.sum(
            convolve(
                binary_volume,
                edge_y_kernel,
                mode="constant",
                cval=0,
            )
            == 1
        )
        + np.sum(
            convolve(
                binary_volume,
                edge_z_kernel,
                mode="constant",
                cval=0,
            )
            == 1
        )
    )

    n_faces = (
        np.sum(
            convolve(
                binary_volume,
                face_xy_kernel,
                mode="constant",
                cval=0,
            )
            == 1
        )
        + np.sum(
            convolve(
                binary_volume,
                face_yz_kernel,
                mode="constant",
                cval=0,
            )
            == 1
        )
        + np.sum(
            convolve(
                binary_volume,
                face_zx_kernel,
                mode="constant",
                cval=0,
            )
            == 1
        )
    )

    n_cubes = np.sum(binary_volume)

    euler_characteristic = (
        n_vertices
        - n_edges
        + n_faces
        - n_cubes
    )

    return int(euler_characteristic)


def compute_betti_2(binary_volume: np.ndarray) -> int:
    """
    Estimate Betti-2 by counting enclosed background components.

    Background components that touch any image boundary are excluded because
    they represent the exterior background rather than enclosed cavities.

    Parameters
    ----------
    binary_volume:
        Three-dimensional binary NumPy array.

    Returns
    -------
    int
        Number of enclosed three-dimensional cavities.
    """
    inverse_volume = (1 - binary_volume).astype(np.uint8)

    labeled_background, n_components = connected_component_label(
        inverse_volume,
        structure=CONNECTIVITY,
    )

    z_max, y_max, x_max = np.array(binary_volume.shape) - 1
    betti_2 = 0

    for component_id in range(1, n_components + 1):
        coordinates = np.where(
            labeled_background == component_id
        )

        if coordinates[0].size == 0:
            continue

        z_coordinates, y_coordinates, x_coordinates = coordinates

        touches_boundary = (
            z_coordinates.min() == 0
            or z_coordinates.max() == z_max
            or y_coordinates.min() == 0
            or y_coordinates.max() == y_max
            or x_coordinates.min() == 0
            or x_coordinates.max() == x_max
        )

        if not touches_boundary:
            betti_2 += 1

    return int(betti_2)


def extract_betti_features(volume: np.ndarray) -> list[int]:
    """
    Extract sliding-window Betti features from one 3D liver CT volume.

    For each of the 50 intensity windows, the function computes:

    - Betti-0: connected foreground components
    - Betti-1: loops or tunnels
    - Betti-2: enclosed cavities

    Betti-1 is calculated using:

        chi = beta_0 - beta_1 + beta_2

    Therefore:

        beta_1 = beta_0 + beta_2 - chi

    Parameters
    ----------
    volume:
        Three-dimensional liver CT volume normalized to [0, 1].

    Returns
    -------
    list[int]
        A 150-dimensional feature vector ordered as:

        [beta_0 values, beta_1 values, beta_2 values]
    """
    volume = np.nan_to_num(
        volume.astype(np.float32),
        nan=0.0,
        posinf=NORMALIZED_MAX,
        neginf=NORMALIZED_MIN,
    )

    volume = np.clip(
        volume,
        NORMALIZED_MIN,
        NORMALIZED_MAX,
    )

    # Background outside the liver is assumed to have been assigned
    # -250 HU and normalized to zero.
    liver_mask = volume > NORMALIZED_MIN

    betti_0_features: list[int] = []
    betti_1_features: list[int] = []
    betti_2_features: list[int] = []

    for window_index in range(N_WINDOWS):
        lower_threshold = (
            NORMALIZED_MIN
            + window_index * STRIDE
        )

        upper_threshold = min(
            lower_threshold + WINDOW_WIDTH,
            NORMALIZED_MAX,
        )

        binary_volume = (
            (volume >= lower_threshold)
            & (volume < upper_threshold)
            & liver_mask
        ).astype(np.uint8)

        if binary_volume.sum() == 0:
            betti_0_features.append(0)
            betti_1_features.append(0)
            betti_2_features.append(0)
            continue

        _, betti_0 = connected_component_label(
            binary_volume,
            structure=CONNECTIVITY,
        )

        betti_2 = compute_betti_2(binary_volume)

        euler_characteristic = compute_euler_characteristic(
            binary_volume
        )

        betti_1 = max(
            int(betti_0)
            + int(betti_2)
            - int(euler_characteristic),
            0,
        )

        betti_0_features.append(int(betti_0))
        betti_1_features.append(int(betti_1))
        betti_2_features.append(int(betti_2))

    return (
        betti_0_features
        + betti_1_features
        + betti_2_features
    )


def process_case(
    case_id: str,
    input_directory: Path,
) -> Optional[dict]:
    """
    Extract topological features for one case.

    Parameters
    ----------
    case_id:
        Name of the patient or case directory.
    input_directory:
        Root input directory containing one folder per case.

    Returns
    -------
    dict or None
        Dictionary containing the case ID and extracted features.
    """
    volume_path = (
        input_directory
        / case_id
        / "volume.npy"
    )

    if not volume_path.exists():
        return None

    try:
        volume = np.load(
            volume_path,
            mmap_mode="r",
        )

        if volume.ndim != 3:
            raise ValueError(
                f"Expected a 3D volume, but received shape "
                f"{volume.shape}."
            )

        features = extract_betti_features(volume)

        row = {"uuid": case_id}

        row.update(
            {
                f"f{feature_index}": feature_value
                for feature_index, feature_value
                in enumerate(features)
            }
        )

        return row

    except Exception as error:
        return {
            "uuid": case_id,
            "error": str(error),
        }


def collect_case_ids(input_directory: Path) -> list[str]:
    """
    Find all case directories containing a volume.npy file.
    """
    return sorted(
        directory_name
        for directory_name in os.listdir(input_directory)
        if (
            input_directory
            / directory_name
            / "volume.npy"
        ).exists()
    )


def run_feature_extraction(
    input_directory: Path,
    output_directory: Path,
    n_jobs: int,
) -> None:
    """
    Extract topological features from all available CT volumes.
    """
    if not input_directory.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{input_directory}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    case_ids = collect_case_ids(input_directory)

    if not case_ids:
        raise RuntimeError(
            "No case directories containing volume.npy "
            f"were found in {input_directory}."
        )

    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Cases found: {len(case_ids)}")
    print("Window width: 30 HU")
    print("Normalized window width: 0.06")
    print("Stride: 10 HU")
    print("Normalized stride: 0.02")
    print(f"Number of windows: {N_WINDOWS}")
    print(f"Parallel jobs: {n_jobs}")

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
    )(
        delayed(process_case)(
            case_id,
            input_directory,
        )
        for case_id in tqdm(
            case_ids,
            desc="Extracting Betti features",
        )
    )

    results = [
        result
        for result in results
        if result is not None
    ]

    successful_rows = [
        result
        for result in results
        if "error" not in result
    ]

    error_rows = [
        result
        for result in results
        if "error" in result
    ]

    output_csv = (
        output_directory
        / "liver_sw_30hu_betti_features.csv"
    )

    feature_columns = [
        "uuid",
        *[
            f"f{feature_index}"
            for feature_index in range(150)
        ],
    ]

    feature_dataframe = pd.DataFrame(
        successful_rows,
        columns=feature_columns,
    )

    feature_dataframe.to_csv(
        output_csv,
        index=False,
    )

    print(f"\nFeatures saved to: {output_csv}")
    print(f"Completed cases: {len(successful_rows)}")
    print(f"Failed cases: {len(error_rows)}")

    if error_rows:
        error_csv = (
            output_directory
            / "liver_sw_30hu_errors.csv"
        )

        pd.DataFrame(error_rows).to_csv(
            error_csv,
            index=False,
        )

        print(f"Errors saved to: {error_csv}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract 30 HU sliding-window Betti features "
            "from normalized 3D liver CT volumes."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing one folder per case. "
            "Each case folder must contain volume.npy."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where output CSV files will be saved.",
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
    Run topological feature extraction.
    """
    arguments = parse_arguments()

    run_feature_extraction(
        input_directory=arguments.input_dir,
        output_directory=arguments.output_dir,
        n_jobs=arguments.n_jobs,
    )


if __name__ == "__main__":
    main()
