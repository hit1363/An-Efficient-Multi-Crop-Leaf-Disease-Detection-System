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
│   ├── processed/
│   ├── taxonomy.py
│   └── prepare_benchmark.py
├── notebooks/
│   ├── colab_training_notebook.ipynb
│   ├── colab_<model>_training.ipynb
│   ├── kaggle_<model>_training.ipynb
│   ├── colab_model_comparison.ipynb
│   ├── kaggle_model_comparison.ipynb
│   ├── kaggle_harmonized_comparison.ipynb
│   ├── model_comparison_common.py
│   ├── data_exploration.ipynb
│   └── evaluation.ipynb
├── training/
│   ├── train.py
│   ├── compare_models.py
│   ├── evaluate.py
│   ├── model.py
│   ├── utils.py
│   ├── config_benchmark.yaml
│   ├── config_mobilenet_v1.yaml
│   ├── config_mobilenetv2.yaml
│   ├── config_mobilenetv3_small.yaml
│   ├── config_mobilenet_v3_large.yaml
│   ├── config_nasnet_mobile.yaml
│   ├── config_efficientnet_b0.yaml
│   ├── config_efficientnet_v2_b0.yaml
│   ├── config_efficientnet_lite0.yaml
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
python dataset/prepare_benchmark.py --data-dir dataset/raw --output-dir dataset/processed --seed 42
```

The builder applies the explicit taxonomy, retains `Invalid` as class 72, removes exact duplicates, converts images to RGB, center-crops/resamples to 224×224 with Lanczos, and writes deterministic 70/15/15 splits. It records `class_names.json`, `taxonomy.json`, `split_manifest.csv`, `dataset_info.json`, and a near-duplicate audit. It never overwrites existing output. Unknown or conflicting labels stop the run and produce a validation report.

**Per-class cap.** `--max-per-class` (default `500`) bounds every class before splitting so no crop dominates training. With seed 42 the cap-500 benchmark yields ≈24.8k train / ≈5.3k val / ≈5.3k test images (≈350 train images per class for classes with ≥500 raw images). The earlier cap-300 processed tree kept only ≈15k train images and under-trained every model, so recipe v2 (below) is tuned for the cap-500 rebuild. Upload the rebuilt directory (with its `split_manifest.csv`) to Kaggle as the training dataset; notebooks detect the manifest automatically and validate splits without re-hashing. Never rebuild from an already-capped tree — that only re-caps the survivors and cannot recover discarded images. Note that the new deterministic selection and split assignment differ from the cap-300 build, so older test-set numbers are not directly comparable; re-evaluate models on the same benchmark manifest when comparing.

### Training

```bash
python training/train.py --config training/config_mobilenet_v1.yaml
python training/train.py --config training/config_mobilenetv2.yaml
python training/train.py --config training/config_mobilenetv3_small.yaml
python training/train.py --config training/config_mobilenet_v3_large.yaml
python training/train.py --config training/config_nasnet_mobile.yaml
python training/train.py --config training/config_efficientnet_b0.yaml
python training/train.py --config training/config_efficientnet_v2_b0.yaml
python training/train.py --config training/config_efficientnet_lite0.yaml
python training/train.py --config training/config_shufflenetv2_05.yaml
```

**Recipe v3 (November 2026).** All model configs share one corrected schedule: the head is frozen for 20 epochs under a full cosine decay (`lr_schedule.decay_epochs: 20`), then the last block is unfrozen for 10 fine-tuning epochs that restart the schedule at `fine_tune_learning_rate: 2e-5` via `epoch_offset` (earlier runs restarted two-thirds decayed and floored at ~5e-8, wasting the fine-tune phase). `efficientnet_b0` is the exception: it trains at its native 224×224 (the Kaggle profile no longer caps it at 192) with the backbone unfrozen from layer 140 instead of 180, and runs 16 fine-tuning epochs (36 total), because its remaining errors clustered in the fine-grained rice/sugarcane/tomato lesions rather than in the easy crops. The loss is categorical cross-entropy with label smoothing 0.1 (the capped benchmark is class-balanced; focal loss suppressed true-positive gradients without correcting any imbalance). Configs that never used focal loss (MobileNet V1/V3-Large, NASNet, EfficientNetV2-B0) keep their `reduce_on_plateau` schedule.

Training configs use bounded, normalized categorical cross-entropy. When enabled, class weights are computed from the training split and passed to `model.fit` as per-example weighting; this is intentionally visible in the run metadata because it changes the scale of reported training loss. Class-aware augmentation is enabled for underrepresented classes, and validation-loss early stopping restores the best checkpoint (patience 6). Training validates the train/validation/test class folders and split manifest before a full run, and checks exact duplicate hashes when a manifest is not available. Pretrained backbones use the frozen-head warmup before fine-tuning; ShuffleNetV2 is trained end-to-end for its full 30-epoch cosine because its backbone is randomly initialized.

If a hosted runtime disables the filesystem split scan to reduce startup time,
the training metadata marks validation/test per-class counts as unknown rather
than treating step-cardinality estimates as measured dataset counts. These
runs are not final accuracy claims: run `training/evaluate.py` against the
held-out test directory and its `class_names.json` or `dataset_info.json`
before comparing models.

Run the full mobile model comparison after preparing the benchmark:

```bash
python training/compare_models.py --data-dir dataset/processed --output-dir results/mobile_model_comparison
```

The comparison covers MobileNet V1/V2/V3 Small/V3 Large, NASNetMobile,
EfficientNetB0, EfficientNetV2-B0, and the UT EfficientNet-Lite0 reference. It
reports float and full INT8 classification metrics, TFLite size, single-thread
CPU latency, and the Pareto frontier. CPU timing is a proxy; validate
finalists on the target Android device. Do not compare runs produced from
different dataset manifests or preprocessing settings.

### Evaluation

```bash
python training/evaluate.py --model <path-to-model> --config training/config_mobilenetv2.yaml
```

Evaluation validates the complete train/validation/test layout, resolves class
order from the benchmark manifest or dataset metadata, and writes overall,
macro, weighted, and per-class metrics. Use the same benchmark manifest for
float and quantized artifacts. The TFLite evaluator also checks the exported
output width against that manifest and preserves taxonomy aliases when source
folders use legacy names.

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

Representative calibration samples are read in deterministic directory order.
For comparable accuracy and latency results, evaluate every exported artifact
with `quantization/evaluate_tflite.py` against the complete held-out test
directory rather than relying on the optional small sample check in the legacy
conversion script.

## Google Colab and Kaggle Workflows

Use a model-specific notebook to train one candidate, evaluate it, export dynamic-range and full INT8 TFLite models, and benchmark CPU inference. Each Kaggle notebook now detects a manifest-backed benchmark wherever `split_manifest.csv` sits — either a locally rebuilt benchmark uploaded as a Kaggle dataset, or one built on Kaggle by the notebook's optional preparation cell, which reads the uncapped **raw** class-folder dataset and defaults to `MAX_PER_CLASS=500` (override via environment; `RAW_DATASET_ROOT` forces the raw tree explicitly). The runtime profile tolerates genuine cross-split duplicates in raw hosted inputs (see `performance.allow_cross_split_duplicates`). When a manifest is present it is authoritative: split validation finishes in seconds and the reported validation/test counts are leakage-free. The harmonized comparison notebook reuses an existing benchmark under `/kaggle/working/harmonized_benchmark` instead of rebuilding it. The separate ShuffleNetV2 experiment and the existing combined notebooks remain available for backward compatibility.

| Model | Google Colab | Kaggle |
|---|---|---|
| MobileNet V1 | [Train](notebooks/colab_mobilenet_v1_training.ipynb) | [Train](notebooks/kaggle_mobilenet_v1_training.ipynb) |
| MobileNet V2 | [Train](notebooks/colab_mobilenetv2_training.ipynb) | [Train](notebooks/kaggle_mobilenetv2_training.ipynb) |
| MobileNetV3-Small | [Train](notebooks/colab_mobilenetv3_small_training.ipynb) | [Train](notebooks/kaggle_mobilenetv3_small_training.ipynb) |
| MobileNetV3-Large | [Train](notebooks/colab_mobilenet_v3_large_training.ipynb) | [Train](notebooks/kaggle_mobilenet_v3_large_training.ipynb) |
| NASNetMobile | [Train](notebooks/colab_nasnet_mobile_training.ipynb) | [Train](notebooks/kaggle_nasnet_mobile_training.ipynb) |
| EfficientNetB0 | [Train](notebooks/colab_efficientnet_b0_training.ipynb) | [Train](notebooks/kaggle_efficientnet_b0_training.ipynb) |
| EfficientNetV2-B0 | [Train](notebooks/colab_efficientnet_v2_b0_training.ipynb) | [Train](notebooks/kaggle_efficientnet_v2_b0_training.ipynb) |
| EfficientNet-Lite0 | [Train](notebooks/colab_efficientnet_lite0_training.ipynb) | [Train](notebooks/kaggle_efficientnet_lite0_training.ipynb) |

For artifact evaluation and ranking, use the [Colab comparison notebook](notebooks/colab_model_comparison.ipynb) or [Kaggle comparison notebook](notebooks/kaggle_model_comparison.ipynb). To rebuild and train the harmonized benchmark on Kaggle, use the [harmonized comparison notebook](notebooks/kaggle_harmonized_comparison.ipynb).

### Downloading the trained artifacts from Kaggle

Kaggle's Output panel silently refuses browser downloads of any single file larger than about 20 MB — the button simply does nothing, with no error, in every browser — while every exported model archive is larger than that. Each Kaggle notebook therefore ends its archiving step with a `FileLink` cell that prints the archive size and renders a clickable link to the zip:

```python
import os
from IPython.display import FileLink, display

os.chdir(os.path.dirname(archive_path))  # FileLink resolves paths against the kernel CWD
print(f"Archive size: {os.path.getsize(archive_path) / 1e6:.1f} MB")
display(FileLink(os.path.basename(archive_path)))
```

`FileLink` streams the file from the running kernel session, which bypasses the Output panel's limit. Three constraints are worth knowing:

- `FileLink` resolves *both* its existence check and the rendered href against the kernel's current working directory, and the training cells `chdir` into the cloned repository — hence the `os.chdir(os.path.dirname(archive_path))` above. Without it the display degrades into a `Path (efficientnet_b0_outputs.zip) doesn't exist` notice even though the zip is perfectly fine.
- Always pass the bare filename (`os.path.basename(...)`). The href is relative, so an absolute `/kaggle/working/...` path renders a link that resolves against the Kaggle page URL and returns 404.
- The link only works while the interactive session is alive. It does not work in a `Save & Run All` version, after the 20-minute idle timeout, or after the kernel restarts — click it right after the cell runs.

Real archives routinely exceed the panel's limit (a full EfficientNetB0 export is ~190 MB), so the same cell also splits the zip into `MAX_PART_BYTES = 19 * 1024 * 1024` chunks named `<stem>.part00`, `<stem>.part01`, … and renders one link per chunk. Stale parts from an earlier run are deleted first so a rejoin cannot mix two archives. Download the parts, then rejoin them:

```bash
# Windows
copy /b efficientnet_b0_outputs.part00+efficientnet_b0_outputs.part01+... efficientnet_b0_outputs.zip
# macOS / Linux
cat efficientnet_b0_outputs.part* > efficientnet_b0_outputs.zip
```

For a committed version, pull the output through the Kaggle API instead, which has no browser size limit:

```bash
kaggle kernels output <your-username>/<notebook-slug> -p ./kaggle_out
```

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

Set the deployment client's preprocessing mode to match the selected model:
MobileNet-family models use `mobilenet_v2`-style `[-1, 1]` input, while
EfficientNetB0 uses raw `[0, 255]` input and EfficientNet-Lite0 uses `[0, 1]`.
The checked-in repository does not currently include the Flutter client, so
the client-side setting must be applied in the deployment project.

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

- Total images: 115,382 (from the checked-in dataset metadata)
- Crops: 15+ (Tomato, Potato, Corn, Rice, Wheat, Apple, Grape, etc.)
- Classes: 72
- Split: 70% train, 15% val, 15% test
- Per-class cap: 500 (exact-deduplicated before selection; the earlier cap-300 build with 15,102 train images under-trained all models and is retired)

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

Status: Active Development (Last updated: November 2026)
