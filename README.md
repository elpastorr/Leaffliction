# Leaffliction

A lightweight toolkit to train, analyze, and use an image classifier for leaf diseases on apples and grapes. It includes:

- A simple TensorFlow/Keras CNN to train on a folder-based dataset
- A prediction script for single images or batches
- Image processing and analysis utilities (segmentation, ROI, color histograms, pseudo-landmarks) with PlantCV and rembg
- Basic data augmentation utilities
- Class distribution visualization

Works on Linux and supports CPU or GPU (if properly configured for TensorFlow).

## Features

- Train a CNN on a directory of images using automatic class discovery
- Save and load models (`.keras` format)
- Predict on a single image or a directory and print top class per image
- Visualize and export processed variants: masks, ROI, analyzed image, pseudo-landmarks, and color histograms
- Generate augmented samples: flip, rotate, skew, shear, crop, color inversion
- Plot class distribution of a dataset

## Project structure

- `Train.py` — build and train the CNN, saves `model.keras`
- `Predict.py` — run predictions on images using a saved model
- `Transformation.py` — segmentation + analysis + plots; works on a single image or a directory
- `Augmentation.py` — create augmented versions of a single image
- `Distribution.py` — visualize class counts across subfolders
- `init.sh` — helper to fetch a sample dataset and create a local venv
- `requirements.txt` — Python dependencies
- `images/` — example dataset organized by class (subfolders)

## Requirements

- Python 3.10+ recommended (tested with modern TensorFlow)
- Linux (X11 or Wayland) with a GUI for interactive plots; otherwise see headless notes below
- Optional: NVIDIA GPU with CUDA/cuDNN for TensorFlow GPU

Install Python deps in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- The dependency `tensorflow[and-cuda]` attempts to install GPU-enabled TensorFlow when available. If you run into CUDA issues, install plain `tensorflow` instead.
- `rembg` will download ONNX models on first run. `onnxruntime` is included.
- `PyQt6` is included to support Matplotlib interactive backends on WSL/headless-like environments.

## Dataset layout

Training expects an image folder with one subfolder per class. Example (provided):

```
images/
  Apple_Black_rot/
  Apple_healthy/
  Apple_rust/
  Apple_scab/
  Grape_Black_rot/
  Grape_Esca/
  Grape_healthy/
  Grape_spot/
```

- Class names are inferred from immediate subfolder names.
- Images should be readable JPEG/PNG; scripts currently filter for `.JPG` in some places.
- You can download a sample dataset and create a venv via the helper:

```bash
bash init.sh
```

## Training

Script: `Train.py`

- Uses `tf.keras.preprocessing.image_dataset_from_directory` with a 80/20 split.
- Image size: 256x256, batch size 32, 10 epochs by default.
- Saves the model to `model.keras` in the project root.

Run:

```bash
source .venv/bin/activate
python Train.py
```

Outputs:
- Console logs (TensorFlow logs reduced)
- Trained model saved to `model.keras`

To customize dataset or model output path, edit the constants near the bottom of `Train.py`:
- `model_file = "model.keras"`
- `dataset = "images"`

## Prediction

Script: `Predict.py`

Predict on a single image:

```bash
python Predict.py path/to/image.jpg model.keras
```

Predict on a directory of images:

```bash
python Predict.py path/to/folder/ model.keras
```

Notes:
- The default class ordering is: `['Apple_Black_rot', 'Apple_healthy', 'Apple_rust', 'Apple_scab', 'Grape_Black_rot', 'Grape_Esca', 'Grape_healthy', 'Grape_spot']`. This should match the subfolder order used at training time.
- The script prints raw model scores (logits) and then the predicted class via argmax. Example output shape per image: `[c1 c2 … c8]` then `Predicted class: <name>`.
- The `-c/--classes` flag exists but is a single-string argument in the current version. Leave it at default unless you edit the script to accept a list properly.

## Image transformation and analysis

Script: `Transformation.py`

Single image (interactive visualization):

```bash
python Transformation.py path/to/image.jpg
```

Batch mode (export processed images to a folder):

```bash
python Transformation.py -src path/to/images_dir -dst path/to/output_dir
```

What it does:
- Removes background via `rembg`
- Converts to grayscale L channel (Lab)
- Thresholds and fills to build a mask
- Gaussian blur on the mask; applies mask to original image
- Builds an ROI and filtered mask; computes PlantCV size analysis
- Creates pseudo-landmarks
- Produces six views per image: `Original`, `Gaussian Blur`, `Mask`, `ROI`, `Analyzed`, `Pseudolandmarks`
- Plots a color histogram across multiple color spaces

Outputs:
- Interactive windows when a single image is provided
- In batch mode, images are written to `-dst` with filename prefixes like `ROI_<original>.jpg`

First run will download ONNX weights for rembg; keep internet enabled.

## Data augmentation

Script: `Augmentation.py`

Create augmented variants from a single image and save into `augmented_directory/`:

```bash
python Augmentation.py path/to/image.jpg
```

Augmentations included:
- Vertical flip
- Rotation (45°, cropped)
- Skew
- Shear
- Center crop + resize
- Color inversion (HSV-based)

Filenames are suffixed: `_Flip`, `_Rotate`, `_Skew`, `_Shear`, `_Crop`, `_Inverted`.

## Class distribution visualization

Script: `Distribution.py`

Show pie + bar charts of class counts for a directory that contains class subfolders:

```bash
python Distribution.py images/
```

- Expects `images/` to contain one subfolder per class.
- Opens an interactive Matplotlib window with two subplots.

## Headless and troubleshooting notes

- No display / Qt backend errors (e.g., on servers):
  - Ensure `PyQt6` is installed (already in requirements)
  - Or set a non-interactive backend before running scripts that plot: `export MPLBACKEND=Agg`
  - Use batch modes that write files instead of showing windows
- OpenCV libGL errors on Linux: install `libgl1` and `libglib2.0-0` via your package manager
- TensorFlow GPU issues:
  - If `tensorflow[and-cuda]` fails, install `tensorflow` CPU-only
  - Match Python version supported by your TensorFlow release
- rembg model download/firewall: first call will fetch models; ensure network access
- Predict classes argument: avoid passing `-c` unless you modify the parser to accept a list (e.g., `nargs='+'`)

## Development tips

- Scripts are small and self-contained; tweak constants in `Train.py` to experiment with image size, epochs, or architecture
- Consider enabling the plotting in `Train.py` by uncommenting `# visualize_result(history, epochs)`
- For more robust CLI behavior (e.g., custom dataset/model paths), you can restore the commented argparse in `Train.py`

## Acknowledgements

- TensorFlow/Keras for model training
- PlantCV for plant phenotyping utilities
- rembg + onnxruntime for background removal

No explicit license file is included. If you plan to redistribute or publish, please add an appropriate license.

---

Happy hacking and leaf spotting!
