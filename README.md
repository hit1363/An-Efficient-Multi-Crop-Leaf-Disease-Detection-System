# Multi-Crop Leaf Disease Detection System

Lightweight, offline-capable multi-crop leaf disease classification optimized for mobile deployment. The benchmark harmonizes 72 labels and compares Keras mobile classifiers using TensorFlow Lite export and CPU proxy measurements.

## Overview

- 72 canonical classes (including the explicit `Invalid` class) across 15+ crops
- MobileNet V1/V2/V3 Small/V3 Large, NASNetMobile, EfficientNetB0/EfficientNetV2-B0, and the UT EfficientNet-Lite0 reference
- Post-training quantization (dynamic range and full INT8)
- Flutter app with camera/gallery input and offline inference

## Goals

1. Unified multi-crop classifier with test accuracy >= 90% and macro F1 >= 0.80
2. Compare classifier accuracy, INT8 size, and single-thread CPU inference latency
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
│   ├── processed_harmonized/
│   ├── taxonomy.py
│   └── prepare_benchmark.py
├── notebooks/
│   ├── colab_training_notebook.ipynb
│   ├── colab_<model>_training.ipynb
│   ├── kaggle_harmonized_comparison.ipynb
│   ├── data_exploration.ipynb
│   └── evaluation.ipynb
├── training/
│   ├── train.py
│   ├── compare_models.py
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

Build the canonical 72-class benchmark from raw class folders. An existing `train/val/test` tree is also accepted, but its old boundaries are discarded and regenerated:

```bash
python dataset/prepare_benchmark.py --data-dir dataset/raw --output-dir dataset/processed_harmonized --seed 42
```

The builder applies the explicit taxonomy, retains `Invalid` as class 72, removes exact duplicates, converts images to RGB, center-crops/resamples to 224×224 with Lanczos, and writes deterministic 70/15/15 splits. It records `class_names.json`, `taxonomy.json`, `split_manifest.csv`, `dataset_info.json`, and a near-duplicate audit. It never overwrites existing output. Unknown or conflicting labels stop the run and produce a validation report.

### Training

```bash
python training/train.py --config training/config_mobilenetv2.yaml
python training/train.py --config training/config_efficientnet_b0.yaml
python training/train.py --config training/config_efficientnet_lite0.yaml
python training/train.py --config training/config_mobilenetv3_small.yaml
python training/train.py --config training/config_shufflenetv2_05.yaml
```

Run the full mobile model comparison after preparing the benchmark:

```bash
python training/compare_models.py --data-dir dataset/processed_harmonized --output-dir results/mobile_model_comparison
```

The comparison covers MobileNet V1/V2/V3 Small/V3 Large, NASNetMobile, EfficientNetB0, EfficientNetV2-B0, and the UT EfficientNet-Lite0 reference. It reports float and full INT8 classification metrics, TFLite size, single-thread CPU latency, and the Pareto frontier. CPU timing is a proxy; validate finalists on the target Android device.

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
  --representative_data dataset/processed_harmonized/train \
  --evaluate --test_data dataset/processed_harmonized/test \
  --arch mobilenetv2
```

## Google Colab Workflow

Use one of the model-specific notebooks named `colab_<model>_training.ipynb` or `kaggle_<model>_training.ipynb`. Each standalone notebook reads the platform dataset, trains one model, evaluates it, exports dynamic-range and full INT8 TFLite models, and benchmarks CPU inference. The existing combined notebooks remain available for backward compatibility.

Hosted notebooks support two runtime profiles through `TRAINING_RUN_MODE`:

- `smoke` (the default) uses a short run, fast startup settings, moderate
  execution batching, no class-aware augmentation, and no TensorBoard
  histogram work. Use it to verify the dataset, GPU, model, and export path.
- `full` preserves the model-specific epoch and fine-tuning settings for final
  accuracy measurements.

Set the mode before running the training cell:

```python
import os
os.environ["TRAINING_RUN_MODE"] = "full"  # or "smoke"
```

For a balanced workflow, run `smoke` first and then rerun the notebook from
the configuration cell with `full`. EfficientNet-Lite0 benefits from a
persistent TensorFlow Hub cache; ShuffleNetV2 0.5x is randomly initialized in
this project and therefore requires its own full-training schedule rather than
the transfer-learning assumptions used by the ImageNet-backed models.

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
