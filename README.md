# Multi-Crop Leaf Disease Detection System

An efficient deep learning-based system for detecting diseases in multiple crop types, optimized for mobile deployment.

## 🎯 Project Overview

This project implements a mobile-ready plant disease detection system using transfer learning with MobileNetV2 and EfficientNet-Lite0 architectures. The system is optimized for real-time inference on mobile devices through TensorFlow Lite quantization.

### Key Features

- **Multi-Crop Support**: Detects diseases across multiple crop types (Tomato, Potato, Corn, Rice, Wheat, etc.)
- **Mobile-Optimized**: Lightweight models (<10MB) with inference time <200ms
- **High Accuracy**: >90% classification accuracy on test datasets
- **Offline Capability**: On-device inference without internet connectivity
- **Cross-Platform**: Flutter-based mobile application for Android/iOS

## 🏗️ Project Structure

```
multi-crop-leaf-disease-detection/
│
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── LICENSE                   # Project license
│
├── docs/                     # Research documentation
│   ├── proposal.pdf
│   ├── thesis_draft.pdf
│   ├── architecture_diagrams/
│   └── experimental_results/
│
├── dataset/                  # Dataset management
│   ├── raw/                  # Original images
│   ├── processed/            # Preprocessed images
│   └── dataset_description.md
│
├── notebooks/                # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
│
├── training/                 # Training pipeline
│   ├── train.py
│   ├── evaluate.py
│   ├── model.py
│   ├── config.yaml
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
    ├── confusion_matrix.png
    ├── f1_scores.csv
    ├── inference_time_comparison.csv
    └── performance_analysis.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+
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
pip install -r requirements.txt
```

3. Download the dataset (see `dataset/dataset_description.md`)

### Training

```bash
cd training
python train.py --config config.yaml
```

### Model Quantization

```bash
cd quantization
python post_training_quant.py --model_path ../models/mobilenetv2/saved_model
```

### Mobile App Development

```bash
cd flutter_app
flutter pub get
flutter run
```

## 📊 Dataset

The project uses a curated dataset of leaf images across multiple crops:
- **Total Images**: ~50,000
- **Crops**: Tomato, Potato, Corn, Rice, Wheat, and more
- **Classes**: 30+ disease categories + healthy leaves
- **Split**: 70% Train, 15% Validation, 15% Test

See `dataset/dataset_description.md` for detailed information.


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

## 👥 Team

### 🎓 Research Team

This thesis project is submitted in partial fulfillment of the requirements for the **Bachelor of Science** degree in **Computer Science & Engineering** at Uttara University.

#### Development Team

<table>
<tr>
<td align="center" width="33%">
<img src="https://via.placeholder.com/150/0066cc/ffffff?text=HIT" width="100" style="border-radius: 50%;" alt="Md Hasibul Islam Tamim"/>
<br />
<b>Md Hasibul Islam Tamim</b>
<br />
<sub>Student ID: 2231081023</sub>
<br />
<sub>Batch 59, Section A (Day)</sub>
<br />
<a href="mailto:2231081023@uttarauniversity.edu">📧 Email</a> • 
</td>
<td align="center" width="33%">
<img src="https://via.placeholder.com/150/00cc66/ffffff?text=SR" width="100" style="border-radius: 50%;" alt="Saidur Rahman"/>
<br />
<b>Saidur Rahman</b>
<br />
<sub>Student ID: 2231081021</sub>
<br />
<sub>Batch 59, Section A (Day)</sub>
<br />
<a href="mailto:2231081021@uttarauniversity.edu">📧 Email</a> • 
</td>
<td align="center" width="33%">
<img src="https://via.placeholder.com/150/cc6600/ffffff?text=MM" width="100" style="border-radius: 50%;" alt="Md. Musha Mia"/>
<br />
<b>Md. Musha Mia</b>
<br />
<sub>Student ID: 2231081009</sub>
<br />
<sub>Batch 59, Section A (Day)</sub>
<br />
<a href="mailto:2231081009@uttarauniversity.edu">📧 Email</a> • 
</td>
</tr>
</table>

#### Academic Supervision

<table>
<tr>
<td align="center">
<img src="https://via.placeholder.com/120/003366/ffffff?text=SAC" width="120" style="border-radius: 50%;" alt="Md. Shafiul Alam Chowdhury"/>
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

**Status**: 🚧 Active Development | 📅 Last Updated: February 2026
