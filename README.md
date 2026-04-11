# Multi-Crop Leaf Disease Detection System

An efficient deep learning-based system for detecting diseases in multiple crop types, optimized for mobile deployment.

## 📌 Abstract

Early detection of crop diseases is critical for preventing yield loss in agricultural systems. However, most state-of-the-art deep learning models are computationally heavy and unsuitable for low-end smartphones commonly used in developing regions.

This project proposes a lightweight, offline-capable multi-crop disease detection system using transfer learning with **MobileNetV2** and **EfficientNet-Lite0**, optimized through post-training quantization for deployment on resource-constrained mobile devices.

### Key Achievements

- >90% classification accuracy  
- <200ms mobile inference time  
- <10MB quantized model size  
- Fully offline on-device prediction  

---

# 🎯 Research Objectives

1. Develop a unified deep learning model for multi-crop disease detection.
2. Compare MobileNetV2 and EfficientNet-Lite0 under identical experimental settings.
3. Apply model compression and quantization techniques.
4. Evaluate real-world mobile inference performance.
5. Deploy the optimized model using Flutter and TensorFlow Lite.

## 🗂️ Project Structure

```
multi-crop-leaf-disease-detection/
│
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── LICENSE                   # Project license
│
├── docs/                     # Research documentation
│   ├── architecture_diagrams/
│   ├── experimental_results/
│   └── team/
│
├── dataset/                  # Dataset management
│   ├── raw/                  # Original images
│   ├── processed/            # Preprocessed images
│   └── prepare_data.py       # Dataset split utility
│
├── notebooks/                # Jupyter notebooks
│   ├── colab_training_notebook.ipynb
│   ├── data_exploration.ipynb
│   └── evaluation.ipynb
│
├── training/                 # Training pipeline
│   ├── train.py
│   ├── evaluate.py
│   ├── model.py
│   ├── config_mobilenetv2.yaml
│   ├── config_efficientnet_lite0.yaml
│   └── utils.py
│
├── models/                   # Trained models
│   ├── mobilenetv2/
│   ├── efficientnet_lite0/
│   └── exported_tflite/
│
├── quantization/             # Model optimization
│   ├── post_training_quant.py
│   └── quantization_results.md
│
├── flutter_app/              # Mobile application
│   ├── lib/
│   ├── assets/
│   ├── pubspec.yaml
│   └── README.md
│
└── results/                  # Experimental results
  └── training_log_mobilenetv2.csv
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (3.12 compatible)
- TensorFlow (installed via `requirements.txt`)
- Flutter 3.x (for mobile app)
- CUDA-enabled GPU (recommended for training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-crop-leaf-disease-detection.git
cd multi-crop-leaf-disease-detection
```

2. Install Python dependencies:
```bash
cd "E:\An Efficient Multi-Crop Leaf Disease Detection System"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the dataset 
- Download PlantVillage dataset from Kaggle
- Place images in `dataset/raw/`
- Run preprocessing split:

```bash
python dataset/prepare_data.py
```

### Training

```bash
python training/train.py --config training/config_mobilenetv2.yaml
# Optional second run:
python training/train.py --config training/config_efficientnet_lite0.yaml
```

### Model Quantization

```bash
python quantization/post_training_quant.py --model_path models/mobilenetv2/saved_model_YYYYMMDD_HHMMSS --output_path models/exported_tflite/model_quantized.tflite --representative_data dataset/processed/train --evaluate --test_data dataset/processed/test --benchmark
```

### Mobile App Development

```bash
cd flutter_app
flutter pub get
flutter run
```

## 📊 Dataset

The project uses a curated dataset of leaf images across multiple crops:
- **Total Images**: 67,072
- **Crops**: Tomato, Potato, Corn, Rice, Wheat, and more
- **Classes**: 45 categories (diseases + healthy + invalid)
- **Split**: 70% Train, 15% Validation, 15% Test


## 📱 Mobile Application

The Flutter-based mobile app provides:
- Real-time camera interface
- Instant disease classification
- Treatment recommendations
- Offline functionality
- Scan history tracking

## 🔬 Research

This project is part of a Bachelor's thesis on efficient deep learning for agricultural applications. Key research contributions:

1. Multi-crop disease detection with unified architecture
2. Aggressive quantization with minimal accuracy loss
3. Real-time mobile inference pipeline
4. Comprehensive evaluation on resource-constrained devices

## 📄 License

[MIT License](LICENSE)

## 🎓 Research Team

This thesis project is submitted in partial fulfillment of the requirements for the **Bachelor of Science** degree in **Computer Science & Engineering** at Uttara University.


<table>
<tr>
<td align="center" width="33%">
<img src="./docs/team/tamim.jpg" width="100" style="border-radius: 50%;" alt="Md Hasibul Islam Tamim"/>
<br />
<b>Md Hasibul Islam Tamim</b>
<br />
<sub>Student ID: 2231081023</sub>
<br />
<sub>Batch 59, Section A (Day)</sub>
<br />
<a href="mailto:2231081023@uttarauniversity.edu.bd">📧 Email</a>
</td>
<td align="center" width="33%">
<img src="./docs/team/saidur.jpg" width="100" style="border-radius: 50%;" alt="Saidur Rahman"/>
<br />
<b>Saidur Rahman</b>
<br />
<sub>Student ID: 2231081021</sub>
<br />
<sub>Batch 59, Section A (Day)</sub>
<br />
<a href="mailto:2231081021@uttarauniversity.edu.bd">📧 Email</a>
</td>
<td align="center" width="33%">
<img src="./docs/team/musha.jpg" width="100" style="border-radius: 50%;" alt="Md. Musha Mia"/>
<br />
<b>Md. Musha Mia</b>
<br />
<sub>Student ID: 2231081009</sub>
<br />
<sub>Batch 59, Section A (Day)</sub>
<br />
<a href="mailto:2231081009@uttarauniversity.edu.bd">📧 Email</a>
</td>
</tr>
</table>

#### Academic Supervision

<table>
<tr>
<td align="center">
<img src="" width="120" style="border-radius: 50%;" alt="Md. Shafiul Alam Chowdhury"/>
<br />
<b>Md. Shafiul Alam Chowdhury</b>
<br />
<sub>Associate Professor</sub>
<br />
<sub>Department of Computer Science & Engineering</sub>
<br />
<sub>Uttara University</sub>
<br />
<i>Thesis Supervisor</i>
</td>
</tr>
</table>

---

### 🏛️ Institution

**Uttara University**  
Department of Computer Science & Engineering  
Dhaka, Bangladesh

**Project Type**: Undergraduate Thesis  
**Academic Year**: 2025-2026  

## 🙏 Acknowledgments

- PlantVillage Dataset
- TensorFlow Team
- Flutter Community



## 📚 Citations

If you use this work, please cite:

```bibtex
@bachelorsthesis{name2026multicrop,
  title={An Efficient Multi-Crop Leaf Disease Detection System for Mobile Deployment},
  author={Md Hasibul Islam Tamim, Saidur Rahman, & Md. Musha Mia },
  year={2026},
  school={Uttara University}
}
```

---

**Status**: 🚧 Active Development | 📅 Last Updated: April 2026
