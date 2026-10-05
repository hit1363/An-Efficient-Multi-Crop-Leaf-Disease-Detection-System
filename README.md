# Multi-Crop Leaf Disease Detection System

Training and export pipeline for multi-crop leaf disease classification. The project trains MobileNetV2, EfficientNetB0, and EfficientNet-Lite0 models and exports TensorFlow Lite models for mobile integration. This repository does not include a Flutter application.

## Overview

- 72 classes (diseases, healthy leaves, and invalid images) across 15+ crops
- MobileNetV2, EfficientNetB0, and EfficientNet-Lite0 transfer-learning configurations
- Dynamic-range and full INT8 TensorFlow Lite export tools
- Model, label, and evaluation artifacts for offline inference integration

## Repository Structure

```
.
├── dataset/
│   ├── prepare_data.py
│   └── processed/
├── notebooks/
│   ├── colab_training_notebook.ipynb
│   ├── data_exploration.ipynb
│   ├── evaluation.ipynb
│   └── kaggle_training_notebook.ipynb
├── training/
│   ├── train.py
│   ├── evaluate.py
│   ├── model.py
│   ├── utils.py
│   ├── config_mobilenetv2.yaml
│   ├── config_efficientnet_b0.yaml
│   └── config_efficientnet_lite0.yaml
├── quantization/
│   ├── post_training_quant.py
│   ├── evaluate_tflite.py
│   └── qat.py
├── models/
├── results/
├── pyproject.toml
├── uv.lock
└── requirements.txt  # Generated install file used by hosted notebooks
```

## Setup

### Prerequisites

- Python 3.12
- uv for dependency and environment management
- CUDA-enabled GPU recommended for full training

Install runtime and development dependencies:

```bash
uv sync --group dev --group docs
```

The hosted notebooks install from `requirements.txt`, which is exported from `pyproject.toml` and `uv.lock`:

```bash
uv export --format requirements-txt --all-extras --no-dev --no-hashes --output-file requirements.txt
```

## Dataset Preparation

Place class-organized images under `dataset/raw/`, then create the train, validation, and test splits:

```bash
uv run python dataset/prepare_data.py
```

## Training

```bash
uv run python training/train.py --config training/config_mobilenetv2.yaml
uv run python training/train.py --config training/config_efficientnet_b0.yaml
uv run python training/train.py --config training/config_efficientnet_lite0.yaml
```

## Evaluation

```bash
uv run python training/evaluate.py --model <path-to-model> --config training/config_mobilenetv2.yaml
```

## Quantization

Dynamic-range conversion:

```bash
uv run python quantization/post_training_quant.py \
  --model_path <path-to-model> \
  --output_path models/exported_tflite/mobilenetv2_dynamic.tflite \
  --arch mobilenetv2
```

Full INT8 conversion and evaluation:

```bash
uv run python quantization/post_training_quant.py \
  --model_path <path-to-model> \
  --output_path models/exported_tflite/mobilenetv2_int8.tflite \
  --representative_data dataset/processed/train \
  --evaluate --test_data dataset/processed/test \
  --arch mobilenetv2
```

## Google Colab and Kaggle

Use `notebooks/colab_training_notebook.ipynb` or `notebooks/kaggle_training_notebook.ipynb`. They install from the generated `requirements.txt` and configure dataset and output paths for their hosted environments.

## Preprocessing and Mobile Integration

Training and quantization use the preprocessing expected by each backbone. A deployment client must use the same input contract and class-label order:

- MobileNetV2: pixel values scaled to `[-1, 1]`
- EfficientNetB0: raw pixel values in `[0, 255]`; the Keras model includes rescaling
- EfficientNet-Lite0: pixel values scaled to `[0, 1]`

This checkout does not contain a mobile application. The generated TFLite models and labels can be integrated into a separate Android or iOS application. Configure `export.save_dir` in the selected training YAML file to choose the output directory.

## Dataset Summary

The checked-in dataset metadata describes 115,382 processed images across 72 classes, split 70%/15%/15% between training, validation, and testing.

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@bachelorsthesis{name2026multicrop,
  title={An Efficient Multi-Crop Leaf Disease Detection System for Mobile Deployment},
  author={Md Hasibul Islam Tamim, Saidur Rahman, & Md. Musha Mia},
  year={2026},
  school={Uttara University}
}
```

## Team

- Md Hasibul Islam Tamim (2231081023)
- Saidur Rahman (2231081021)
- Md. Musha Mia (2231081009)

## Acknowledgments

- PlantVillage Dataset
- TensorFlow Team
