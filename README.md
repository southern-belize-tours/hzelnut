# Hazelnut defect detection

This repo is for exploring hazelnut defect datasets and building models for defect detection.


## Quickstart (EDA)

1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install requirements

```bash
pip install -r requirements.txt
```

3) Create (in the root directory) a *.env* file with the following contents:
```
SEED=42
BAD_NUTS_DIR=data/bad_nuts
```

4) Open the notebook

- `eda_mvtec.ipynb`

## Project structure

- `data/hazelnut_1/` - dataset files (train/test/masks)
- `eda1.ipynb` - first-pass exploratory data analysis
- `requirements.txt` - Python dependencies for EDA

## Next steps (ideas)

- Add dataset 2+ with consistent folder structure
- Add data loading utilities and a shared config
- Decide on baseline modeling approach (e.g., supervised vs. anomaly detection)
- Look into wavelets transformation
- Look into generalized detect detection https://ieeexplore.ieee.org/document/10097961

## Dataset notes
The first dataset is in `data/hazelnut_1` and comes from MVTec AD (hazelnut).
- License: CC BY-NC-SA 4.0 (non-commercial). See `data/hazelnut_1/readme.txt`.
- Citation (from dataset README):
  - Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
    "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
    IEEE Conference on Computer Vision and Pattern Recognition, 2019