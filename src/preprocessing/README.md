
# CT Preprocessing

This directory contains the preprocessing pipeline used to prepare 3D liver CT volumes for downstream imaging and topological analysis.

## Pipeline

The preprocessing workflow consists of three steps:

1. **Liver masking**
   - Loads the CT volume and its corresponding liver segmentation.
   - Retains voxels inside the liver region.
   - Assigns voxels outside the liver a background value of **−250 HU**.

2. **CT standardization**
   - Resamples each volume to **1 × 1 × 1 mm** isotropic spacing using linear interpolation.
   - Clips CT intensities to the range **[−250, 250] HU**.
   - Applies fixed min-max normalization to map intensities to **[0, 1]**.

3. **Volume resizing**
   - Resizes each normalized volume to **96 × 96 × 96 voxels** using linear interpolation.
   - Saves the final volume as a NumPy array for downstream analysis.

## Scripts

```text
liver_masking.py
ct_preprocessing.py
volume_resizing.py
```

## Expected Input Structure

For liver masking:

```text
input_directory/
├── case_001/
│   ├── image.nii.gz
│   └── segmentation.nii.gz
├── case_002/
│   ├── image.nii.gz
│   └── segmentation.nii.gz
└── ...
```

## Output

The preprocessing pipeline produces one normalized and resized volume for each case:

```text
output_directory/
├── case_001/
│   └── volume.npy
├── case_002/
│   └── volume.npy
└── ...
```

Each output volume has:

- isotropic spatial resolution before resizing;
- clipped intensity range of **[−250, 250] HU**;
- normalized intensity range of **[0, 1]**;
- final dimensions of **96 × 96 × 96** voxels.

## Processing Order

Run the scripts in the following order:

```text
liver_masking.py
        ↓
ct_preprocessing.py
        ↓
volume_resizing.py
```

The resulting `volume.npy` files can then be used for downstream machine learning models and topological feature extraction.
