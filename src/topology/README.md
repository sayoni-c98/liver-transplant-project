
# Topological Feature Extraction

This module extracts sliding-window topological descriptors from 3D liver CT volumes.

## Method

For each normalized liver CT volume:

- Intensities are normalized to the range **[0, 1]**.
- A liver mask is used to exclude background voxels.
- A **30 HU sliding intensity window** (normalized width = **0.06**) moves across the intensity range with a **10 HU stride** (normalized stride = **0.02**).
- A total of **50 overlapping intensity windows** are evaluated.

For each window, the following topological descriptors are computed:

- **β₀ (Betti-0):** Number of connected components.
- **β₁ (Betti-1):** Number of loops, computed from the Euler characteristic.
- **β₂ (Betti-2):** Number of enclosed cavities.

The resulting feature vector contains:

- 50 β₀ features
- 50 β₁ features
- 50 β₂ features

for a total of **150 topological features** per CT volume.

## Output

The script generates a CSV file where each row corresponds to one liver CT volume and contains:

- Case UUID
- 150 topological features (β₀, β₁, β₂)

## Repository

This implementation provides the sliding-window topological feature extraction used for our liver CT analysis.
