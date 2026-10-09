# An Efficient Multi-Crop Leaf Disease Detection System

An image-based leaf disease classification project for multiple crops. The project trains MobileNetV2 and EfficientNetB0 models, supports TensorFlow Lite export and quantization, and is developing an offline-capable mobile application.

## Development Status

- **Model training:** Complete for the current project milestone.
- **Mobile application:** In development.
- **Training logs and evaluation results:** Add the run logs and evaluation evidence before reporting model performance. This README does not claim measured accuracy, F1 score, model size, or inference speed.

The Kaggle training notebook is configured for MobileNetV2 and EfficientNetB0. Training completion is a project status update; comparative results should be reported only after the corresponding logs and evaluation files are reviewed.

## Project Overview

- 72 classes across 15+ crops, including crop-specific healthy classes and invalid-image categories
- ImageNet-pretrained MobileNetV2 and EfficientNetB0 training configurations
- Post-training quantization for dynamic range and full INT8 TensorFlow Lite models
- Mobile application development for on-device inference

## Project Goals

1. Build a unified multi-crop classifier and evaluate it against the project targets of at least 90% test accuracy and 0.80 macro F1.
2. Compare MobileNetV2 and EfficientNetB0 on classification quality, model size, and inference speed.
3. Assess whether INT8 quantization reduces model size while keeping accuracy loss below two percentage points.
4. Measure latency and resource use on low-end and mid-range Android devices.
5. Complete an offline Android application for field testing.

These are evaluation goals, not reported results. Add the training logs and evaluation evidence before drawing conclusions about whether the targets have been met.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── prepare_data.py
├── notebooks/
│   ├── kaggle_training_notebook.ipynb
│   ├── colab_training_notebook.ipynb
│   ├── data_exploration.ipynb
│   └── evaluation.ipynb
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
└── docs/
    ├── development_progress_report.docx
    └── development_progress_presentation.pptx
```

The mobile application is in development. Its source is not currently included in this repository.

## Requirements

- Python 3.12
- TensorFlow and the packages listed in `requirements.txt`
- A CUDA-enabled GPU is recommended for training
- Kaggle or Google Colab GPU runtime for the corresponding notebook workflow

## Local Setup

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Prepare the dataset so it contains `dataset/processed/train`, `dataset/processed/val`, and `dataset/processed/test`, organized by class directory.

## Training

To run training locally with the repository configurations:

```bash
python training/train.py --config training/config_mobilenetv2.yaml
python training/train.py --config training/config_efficientnet_b0.yaml
```

For the completed Kaggle workflow, use `notebooks/kaggle_training_notebook.ipynb`. The Colab workflow is in `notebooks/colab_training_notebook.ipynb`; check that notebook's model configuration before running it.

Training logs and evaluation files should be saved with the model name and run details. Add them to the project's documentation before publishing performance claims.

## Evaluation and Quantization

Evaluate a saved model using the matching configuration:

```bash
python training/evaluate.py --model <path-to-model> --config training/config_mobilenetv2.yaml
```

Export a dynamic-range TFLite model:

```bash
python quantization/post_training_quant.py \
  --model_path <path-to-model> \
  --output_path models/exported_tflite/mobilenetv2_dynamic.tflite \
  --arch mobilenetv2
```

For full INT8 quantization, provide representative training data and test data:

```bash
python quantization/post_training_quant.py \
  --model_path <path-to-model> \
  --output_path models/exported_tflite/mobilenetv2_int8.tflite \
  --representative_data dataset/processed/train \
  --evaluate --test_data dataset/processed/test \
  --arch mobilenetv2
```

Quantization-aware training requires TensorFlow Model Optimization. The dependency is limited to Python versions below 3.12 in `requirements.txt`; use post-training quantization where that dependency is unavailable.

## Preprocessing Alignment

Use the preprocessing contract that matches the selected model in both training and deployment:

- **MobileNetV2:** `mobilenet_v2.preprocess_input`, scaling RGB values to `[-1, 1]`.
- **EfficientNetB0:** Keras EfficientNet input contract, RGB values in the `0..255` range; the model includes rescaling.
- **EfficientNet-Lite0:** follow the preprocessing configured for that model if you use its separate training configuration.

## Labels and Healthy Classes

Labels use the `Crop___Disease` convention. Healthy categories remain crop-specific, such as `Tomato___healthy`, so predictions preserve which crop is healthy.

## Dataset Summary

- Classes: 72
- Crops: 15+
- Intended split: 70% training, 15% validation, and 15% test

Confirm the actual counts and split when preparing a dataset version; report the resulting dataset details alongside evaluation metrics.

## Project Team

- Md Hasibul Islam Tamim (2231081023)
- Saidur Rahman (2231081021)
- Md. Musha Mia (2231081009)

Uttara University, 2026.

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- PlantVillage dataset
- TensorFlow team
- Flutter community

**Status:** Model training complete; mobile application in development.

**Last updated:** 6 October 2026.
