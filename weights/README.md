# Model Weights

Released weights are hosted on the Hugging Face model repo
[`philmorekoung/liver-steatosis`](https://huggingface.co/philmorekoung/liver-steatosis)
(too large for git). Download them into this directory:

```bash
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('philmorekoung/liver-steatosis', local_dir='.')"
```

| File | Model | Used by |
|------|-------|---------|
| `no_vision.pkl` | tabular stacked ensemble | `src/models/tr-tree/no_vision_model.py` |
| `vision_suprem.pt` | SuPreM SwinUNETR | `src/models/vision/vision_model.py` |
| `full.pkl` + `full_vision.pt` | multimodal ensemble | `src/models/tr-tree/full_model.py` |

Notes:

- `full.pkl` and `full_vision.pt` must stay together (the ensemble references its
  image branch by relative filename).
- The `.pkl` files must be loaded with `tabpfn==2.2.1` (they embed a fitted TabPFN
  model).
- The weights are released under CC BY-NC 4.0 on Hugging Face; confirm this is
  compatible with this repository's license and your data use agreement before
  redistributing.
