# GUIDE.md

This document describes how to set up the environment and highlights the modifications that we made in our project.

---

# Environment Setup

The most important dependency is **MMCV**, which requires compatible versions of PyTorch and other packages. Follow the installation steps below in order to avoid version mismatch issues.

## 1. Create a Conda environment

```bash
conda create -n segearth python=3.10 -y
```

## 2. Install PyTorch

```bash
conda run -n segearth pip install \
    torch==2.4.0 \
    torchvision==0.19.0
```

## 3. Install OpenMIM

```bash
conda run -n segearth pip install openmim
```

## 4. Install MMCV (using OpenMIM)

```bash
conda run -n segearth mim install "mmcv==2.2.0"
```

## 5. Install MMSegmentation (using OpenMIM)

```bash
conda run -n segearth mim install "mmsegmentation==1.2.2"
```

## 6. Install NumPy

```bash
conda run -n segearth pip install numpy==1.26.4
```

## 7. Resolve version mismatch

MMCV and MMSEG has a known version compatibility issue. Please follow the discussion in the GitHub issue below to see the solution
https://github.com/earth-insights/SegEarth-OV-3/issues/2

## 8. Install remaining dependencies
After cloning the repository and the packages above have been installed successfully, install the remaining dependencies from `requirements.txt`.

```bash
conda run -n segearth pip install -r requirements.txt
```

## 9. Download the SAM3 checkpoint


1. Download `sam3.pt` from Hugging Face.
2. Place it in your preferred checkpoint directory.
3. Update the `checkpoint_path` inside:

```
segearthov3_segmentor.py
```

The model should now be ready to run.

---

# Differences from the Original Repository

Several files have been modified or added for this project. Many of the changes are documented directly within the source code.

The main files to look at are:

```
segearthov3_segmentor.py
eval.py
segment.py

configs/
├── base_config.py
├── cfg_potsdam.py
└── cls_potsdam.txt

custom_metrics.py
```

---

# Image Segmentation

For experimenting with the model on the Hessen dataset, use:

```bash
python segment.py
```

The following parameters can be modified for each run:

- `img_path`
- `name_list`
- `COLOR_MAP`
- Model parameters (e.g. thresholds, sliding window size)

The script generates segmentation visualizations that can be used for qualitative inspection.

---

# Evaluation

Evaluation is performed using:

```bash
python eval.py <config_file>
```

This project introduces a custom evaluation metric located in:

```
custom_metrics/
```

The metric is based on the original MMSegmentation IoU metric, but excludes the **Clutter** class from the reported evaluation metrics of Potsdam Images.

---
