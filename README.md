# Multi-Crop Leaf Disease Detection System

Lightweight, offline-capable multi-crop leaf disease detection optimized for mobile deployment. The system trains five mobile backbones, exports TensorFlow Lite models, and ships a Flutter app for on-device inference.

## Overview

- 45 classes (diseases + healthy + invalid) across 15+ crops
- MobileNetV2, EfficientNetB0, EfficientNet-Lite0, and MobileNetV3-Small with ImageNet pretraining
- Native ShuffleNetV2 0.5x implementation trained from scratch
- Post-training quantization (dynamic range and full INT8)
- Flutter app with camera/gallery input and offline inference

## Goals

1. Unified multi-crop classifier with test accuracy >= 90% and macro F1 >= 0.80
2. Compare MobileNetV2 vs EfficientNet-Lite0 on accuracy, size, and speed
3. Achieve ~4x size reduction via INT8 quantization with < 2% accuracy loss
4. Benchmark latency, memory, CPU, and battery on low-end and mid-range devices
5. Deliver an offline Android app for field testing

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── prepare_data.py
├── notebooks/
│   ├── colab_training_notebook.ipynb
│   ├── colab_<model>_training.ipynb
│   ├── kaggle_<model>_training.ipynb
│   ├── data_exploration.ipynb
│   └── evaluation.ipynb
├── training/
│   ├── train.py
│   ├── evaluate.py
│   ├── model.py
│   ├── utils.py
│   ├── config_mobilenetv2.yaml
│   ├── config_efficientnet_b0.yaml
│   ├── config_efficientnet_lite0.yaml
│   ├── config_mobilenetv3_small.yaml
│   └── config_shufflenetv2_05.yaml
├── quantization/
│   └── post_training_quant.py
├── models/
├── flutter_app/
│   ├── lib/
│   ├── assets/
│   └── pubspec.yaml
└── results/
```

## Quick Start (Local)

### Prerequisites

- Python 3.10+
- TensorFlow (from requirements.txt)
- Flutter 3.x (for the mobile app)
- CUDA-enabled GPU recommended

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset Preparation

1. Download PlantVillage (or your dataset) and place images in dataset/raw/
2. Split into train/val/test:

```bash
python dataset/prepare_data.py
```

### Training

```bash
python training/train.py --config training/config_mobilenetv2.yaml
python training/train.py --config training/config_efficientnet_b0.yaml
python training/train.py --config training/config_efficientnet_lite0.yaml
python training/train.py --config training/config_mobilenetv3_small.yaml
python training/train.py --config training/config_shufflenetv2_05.yaml
```

### Evaluation

```bash
python training/evaluate.py --model <path-to-model> --config training/config_mobilenetv2.yaml
```

### Quantization (Dynamic + INT8)

```bash
python quantization/post_training_quant.py \
  --model_path <path-to-model> \
  --output_path models/exported_tflite/mobilenetv2_dynamic.tflite \
  --arch mobilenetv2

python quantization/post_training_quant.py \
  --model_path <path-to-model> \
  --output_path models/exported_tflite/mobilenetv2_int8.tflite \
  --representative_data dataset/processed/train \
  --evaluate --test_data dataset/processed/test \
  --arch mobilenetv2
```

## Google Colab Workflow

Use one of the model-specific notebooks named `colab_<model>_training.ipynb` or `kaggle_<model>_training.ipynb`. Each standalone notebook reads the platform dataset, trains one model, evaluates it, exports dynamic-range and full INT8 TFLite models, and benchmarks CPU inference. The existing combined notebooks remain available for backward compatibility.

## Preprocessing Alignment

Training and quantization use TensorFlow preprocess_input for the selected architecture. The Flutter app must match the same normalization:

- MobileNetV2: input range [-1, 1]
- MobileNetV3-Small: input range [-1, 1]
- ShuffleNetV2 0.5x: input range [-1, 1]
- EfficientNetB0: raw input range [0, 255] (the Keras model includes rescaling)
- EfficientNet-Lite0: input range [0, 1]

Set flutter_app/lib/utils/constants.dart -> AppConstants.preprocessType to mobilenet_v2 or efficientnet to match the deployed model.

## Labels and Healthy Classes

Labels follow Crop___Disease formatting. Healthy classes are per-crop (e.g., Tomato___healthy -> tomato_healthy) so crop-specific healthy predictions are preserved.

## Mobile App

```bash
cd flutter_app
flutter pub get
flutter run
```

Place your deployed model and labels here:

- flutter_app/assets/models/model.tflite
- flutter_app/assets/labels/labels.txt

## Dataset Summary

- Total images: ~67k
- Crops: 15+ (Tomato, Potato, Corn, Rice, Wheat, Apple, Grape, etc.)
- Classes: 45
- Split: 70% train, 15% val, 15% test

## License

MIT License. See LICENSE.

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
- Flutter Community

Status: Active Development (Last updated: May 2026)
