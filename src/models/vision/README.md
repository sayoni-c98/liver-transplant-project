# Vision Models

Deep 3D model trained directly on CT volumes for donor liver macrovesicular
steatosis (≥ 30%). `vision_model.py` fine-tunes a **SuPreM-pretrained SwinUNETR**
encoder (MONAI) with an ordinal head on single-channel 96³ liver volumes
(intensities in `[0, 1]`).

The released checkpoint contains the fine-tuned encoder + head, so prediction
needs only that file — the SuPreM backbone is required for training only.

## Weights

`vision_suprem.pt` on the Hugging Face model repo
[`philmorekoung/liver-steatosis`](https://huggingface.co/philmorekoung/liver-steatosis):

```bash
python -c "from huggingface_hub import hf_hub_download as d; \
d('philmorekoung/liver-steatosis', 'vision_suprem.pt', local_dir='weights')"
```

## Usage

```bash
# score your own volumes (npz with keys: uuids[str], images[N,96,96,96] in [0,1])
python vision_model.py predict --weights weights/vision_suprem.pt \
    --images your_volumes.npz --out predictions.csv

# regenerate the checkpoint (needs a GPU, the 96^3 npz, and the SuPreM backbone
# supervised_suprem_swinunetr_2100.pth from https://github.com/MrGiovanni/SuPreM)
python vision_model.py train --images images_96.npz --csv donors.csv \
    --suprem supervised_suprem_swinunetr_2100.pth --out weights/vision_suprem.pt \
    --batch-size 32
```

Training uses all visible GPUs via `nn.DataParallel` (`--batch-size` is split
across them). Output: `uuid`, `p_macrosteatosis`, `high_risk`; the threshold is
stored in the checkpoint.

## Requirements

```
torch
monai
scikit-learn
numpy
pandas
huggingface_hub
```
