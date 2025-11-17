# Leaffliction

A lightweight toolkit to train, analyze, and use an image classifier for leaf diseases on apples and grapes. It includes:

- A simple TensorFlow/Keras CNN to train on a folder-based dataset
- A prediction script for single images or batches
- Image processing and analysis utilities (segmentation, ROI, color histograms, pseudo-landmarks) using PlantCV and rembg
- Basic data augmentation utilities
- Class distribution visualization

Works on Linux and supports CPU or GPU (if TensorFlow is configured).

## Features

- Train a CNN on a directory of images using automatic class discovery
- Save and load models (`.keras` format)
- Predict on a single image or a directory and print top class per image
- Visualize and export processed variants: masks, ROI, analyzed image, pseudo-landmarks, and color histograms
- Generate augmented samples: flip, rotate, skew, shear, crop, color inversion
- Plot class distribution of a dataset

## Quick summary

This README reflects the actual scripts present in the repository and their important functions/flags.

Key scripts:
- Train.py — build and train the CNN, saves `model.keras`
- Predict.py — single-image and batch prediction (`load_model`, `render`)
- eval.py — batched evaluation utilities (`get_images_with_classes`, `predict_batch`, `run_pred`)
- prepare_data.py — dataset splitting (`split_data`)
- Augmentation.py — image augmentations (`augment`, `augment_dir`)
- Transformation.py — segmentation & analysis (`process_file`, `process_directory`, `render_plot`)
- Distribution.py — class distribution plotting
- test.py — small utilities for sampling/copying
- init.sh — fetch sample dataset and create venv
- requirements.txt — Python dependencies
- model_big_e10.keras — example pretrained model included in repo

## Requirements and setup

Recommended: Python 3.10+ (adjust for your TensorFlow version). Linux desktop recommended for interactive plotting.

Use init helper:

```bash
bash init.sh
```

Or install in a virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- rembg downloads ONNX models on first run; ensure network access.
- If GPU TensorFlow causes issues, install CPU-only `tensorflow`.
- For headless environments set `MPLBACKEND=Agg` or install `PyQt6`.

## Dataset layout

Training and many tools expect a directory with one subfolder per class:

images/
  Apple_Black_rot/
  Apple_healthy/
  ...
  
Class names are inferred from subfolder names. Some scripts filter for `.JPG` — ensure images match expected extensions or update scripts.

## Training

Script: `Train.py`

- Uses tf.keras image_dataset_from_directory (default 80/20 split)
- Default image size: 256×256, batch size: 32, epochs: 10
- Default output: `model.keras` (adjust constants near bottom of the file)

Run:

```bash
source .venv/bin/activate
python Train.py
```

Helper: `get_info` prints dataset info and sample batches.

## Prediction

Script: `Predict.py`

Predict a single image:

```bash
python Predict.py path/to/image.jpg model.keras
```

Predict a directory:

```bash
python Predict.py path/to/folder/ model.keras
```

Notes:
- Uses `load_model` to read a Keras model and `render` for simple visualization (uses rembg + plantcv).
- Default class ordering is set in the script — ensure it matches the ordering used during training.
- The `-c/--classes` CLI flag exists but in the current script expects a single string; modify argparse to accept lists if needed.

## Evaluation

Script: `eval.py`

Run evaluation on a test directory (classes as subfolders):

```bash
python eval.py path/to/test_dir model.keras
```

Important helpers: `get_images_with_classes`, `predict_batch`, `get_or_load_model`, `run_pred`, `print_results`. Flags present in the script include `--classes` and `--batch-size`.

## Image transformation & analysis

Script: `Transformation.py`

Interactive single-image view:

```bash
python Transformation.py path/to/image.jpg
```

Batch processing (writes results to destination):

```bash
python Transformation.py -src path/to/images_dir -dst path/to/output_dir
```

What it does:
- Removes background (rembg)
- Builds and refines mask (PlantCV + OpenCV)
- Creates ROI, computes size metrics
- Generates pseudo-landmarks and color histograms
- Produces multiple views per image and can write prefixed files (e.g., `ROI_<original>.jpg`)

Batch mode writes outputs to destination; single-file mode shows Matplotlib windows.

## Data augmentation

Script: `Augmentation.py`

Create augmented variants of a single image and write to `augmented_directory/`:

```bash
python Augmentation.py path/to/image.jpg
```

Augmentations: flip, rotate, skew, shear, center crop+resize, HSV-based inversion. Filenames are suffixed with `_Flip`, `_Rotate`, `_Skew`, `_Shear`, `_Crop`, `_Inverted`. Use `augment_dir` to process folders.

## Class distribution visualization

Script: `Distribution.py`

Show pie and bar charts for a dataset directory:

```bash
python Distribution.py images/
```

`get_category` computes counts and normalizes some class name prefixes.

## Utilities & tests

- test.py — sampling and copying utility
- .gitignore excludes datasets, venv, and model artifacts

## Headless / troubleshooting notes

- Set `export MPLBACKEND=Agg` for non-interactive plotting.
- Fix OpenCV libGL errors on Linux by installing `libgl1` and `libglib2.0-0`.
- If rembg fails, check network and onnxruntime installation.
- If you need a non-list `-c/--classes` behavior changed, edit argparse in Predict.py and eval.py to use `nargs='+'`.

## Development tips

- Scripts are small and localized; edit constants in Train.py to change model path, dataset, image size, or epochs.
- Enable or adapt plotting helpers (`render`, `render_plot`, `visualize_result`) as needed.
- Improve CLI ergonomics by modifying argparse blocks across scripts.

## Files of interest

- Train.py
- Predict.py
- eval.py
- prepare_data.py
- Augmentation.py
- Transformation.py
- Distribution.py
- test.py
- requirements.txt
- init.sh
- model_big_e10.keras

## License

No explicit license file is included. Add a LICENSE if you plan to redistribute.

---

Happy hacking and leaf spotting!
