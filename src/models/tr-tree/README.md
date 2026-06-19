# Stacked Ensemble Models

Stacked ensemble models for donor liver macrovesicular steatosis (≥ 30%). Three
base learners — **LightGBM + XGBoost + TabPFN** (a tabular foundation model) —
are trained on the 5-class fat bin, collapsed to the binary ≥ 30% probability,
and combined by a logistic-regression meta-learner.

Two variants:

| Script | Model | Inputs | Weights |
|--------|-------|--------|---------|
| `no_vision_model.py` | tabular ensemble | clinical / body-composition / organ-HU / HU-histogram / topology features | `no_vision.pkl` |
| `full_model.py` | multimodal ensemble | the above **+** 96³ CT volume | `full.pkl` + `full_vision.pt` |

`full_model.py` adds the SuPreM image probability (see `../vision`) as a fourth
base learner; its image branch is trained on the meta-learner's training split so
the stacked probabilities are out-of-sample. `common.py` holds the shared base
learners and helpers.

## Weights

Released weights live on the Hugging Face model repo
[`philmorekoung/liver-steatosis`](https://huggingface.co/philmorekoung/liver-steatosis):

```bash
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('philmorekoung/liver-steatosis', local_dir='weights')"
```

Keep `full.pkl` and `full_vision.pt` in the **same** folder — `full_model.py`
references the image branch by relative filename.

## Usage

```bash
# tabular, no vision (one row per donor; columns matching feature_cols in the pkl)
python no_vision_model.py predict --weights weights/no_vision.pkl \
    --csv your_donors.csv --out predictions.csv

# full multimodal (features CSV + 96^3 volumes joined on uuid)
python full_model.py predict --weights weights/full.pkl \
    --csv your_donors.csv --images your_volumes.npz --out predictions.csv
```

Output: `uuid`, `p_macrosteatosis` (probability of fat ≥ 30%), `high_risk`
(1 if `p ≥ threshold`). The operating threshold is stored in the weight file.

Regenerate the weights with the `train` subcommand (requires the training table;
`full_model.py train` also needs the 96³ volumes, the SuPreM backbone, and a GPU):

```bash
python no_vision_model.py train --csv donors.csv --out weights/no_vision.pkl
```

## Features

The models expect the **218 numeric features** used to build the cohort (clinical
and donor-history variables, body-composition and organ-HU statistics, the liver
HU histogram, and 150 topological persistence features `f0..f149`). Missing values
are allowed. The exact column order is stored in each pickle as `feature_cols`;
`predict` reorders the input columns by name and reports any that are missing.

## Requirements

```
scikit-learn==1.6.1
lightgbm==4.6.0
xgboost==3.2.0
tabpfn==2.2.1      # pin: newer releases add a license gate and change the model
torch              # full_model.py only (image branch)
monai              # full_model.py only
```

The `.pkl` files embed a fitted TabPFN object (it carries its training table — a
foundation model doing in-context learning), so they are self-contained at predict
time but large, and must be loaded with `tabpfn==2.2.1`.
