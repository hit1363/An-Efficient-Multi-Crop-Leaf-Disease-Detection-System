# Dataset Description

## Multi-Crop Leaf Disease Dataset

### Overview

This dataset contains leaf images from multiple crop types with various disease conditions and healthy samples. The images are collected from public datasets and augmented for robustness.

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Images | ~50,000 |
| Training Set | 35,000 (70%) |
| Validation Set | 7,500 (15%) |
| Test Set | 7,500 (15%) |
| Image Resolution | 224x224 pixels (after preprocessing) |
| Original Resolution | Variable (resized during preprocessing) |
| Color Space | RGB |
| File Format | JPG/PNG |

### Crop Categories

1. **Tomato** (10 classes)
   - Early Blight
   - Late Blight
   - Leaf Mold
   - Septoria Leaf Spot
   - Spider Mites
   - Target Spot
   - Yellow Leaf Curl Virus
   - Mosaic Virus
   - Bacterial Spot
   - Healthy

2. **Potato** (3 classes)
   - Early Blight
   - Late Blight
   - Healthy

3. **Corn** (4 classes)
   - Common Rust
   - Gray Leaf Spot
   - Northern Leaf Blight
   - Healthy

4. **Rice** (3 classes)
   - Blast
   - Brown Spot
   - Healthy

5. **Wheat** (3 classes)
   - Leaf Rust
   - Septoria
   - Healthy

6. **Other Crops** (Additional classes as needed)

**Total Classes**: 30+ disease categories + healthy leaves

### Data Sources

1. **PlantVillage Dataset**: Primary source
   - URL: https://github.com/spMohanty/PlantVillage-Dataset
   - License: CC0 1.0 Universal

2. **Kaggle Plant Disease Datasets**
   - Multiple curated datasets
   - Combined and standardized

3. **Field Collection** (if applicable)
   - Images collected from local farms
   - Properly labeled by agricultural experts

### Data Distribution

```
Crop         | Train  | Val   | Test  | Total
-------------|--------|-------|-------|-------
Tomato       | 14,000 | 3,000 | 3,000 | 20,000
Potato       | 3,500  | 750   | 750   | 5,000
Corn         | 5,250  | 1,125 | 1,125 | 7,500
Rice         | 3,500  | 750   | 750   | 5,000
Wheat        | 3,500  | 750   | 750   | 5,000
Others       | 5,250  | 1,125 | 1,125 | 7,500
-------------|--------|-------|-------|-------
Total        | 35,000 | 7,500 | 7,500 | 50,000
```

### Directory Structure

```
dataset/
├── raw/
│   ├── tomato/
│   │   ├── early_blight/
│   │   ├── late_blight/
│   │   ├── leaf_mold/
│   │   └── healthy/
│   ├── potato/
│   │   ├── early_blight/
│   │   ├── late_blight/
│   │   └── healthy/
│   ├── corn/
│   ├── rice/
│   └── wheat/
│
└── processed/
    ├── train/
    ├── val/
    └── test/
```

### Preprocessing Pipeline

1. **Image Resizing**: All images resized to 224x224 pixels
2. **Normalization**: Pixel values scaled to [0, 1] range
3. **Format Standardization**: Converted to RGB (if grayscale)
4. **Quality Filtering**: Removed blurry or corrupted images

### Data Augmentation (Training Only)

- **Rotation**: Random rotation ±45°
- **Horizontal Flip**: 50% probability
- **Vertical Flip**: 50% probability
- **Zoom**: Random zoom 0.8-1.2x
- **Brightness**: Adjustment ±20%
- **Color Jitter**: Slight hue/saturation variations
- **Gaussian Noise**: Added for robustness

### Data Loading

**Python Example:**

```python
import tensorflow as tf

def load_dataset(data_dir, batch_size=32):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode='categorical'
    )
    return dataset

train_ds = load_dataset('dataset/processed/train')
val_ds = load_dataset('dataset/processed/val')
test_ds = load_dataset('dataset/processed/test')
```

### Class Balance

The dataset is relatively balanced, but some disease classes have fewer samples. Class weighting is applied during training to handle imbalance.

### Ethical Considerations

- All images from public sources are properly attributed
- No personally identifiable information in images
- Used for educational and research purposes
- Proper licensing followed for all data sources

### Download Instructions

**Option 1: Manual Download**
1. Download PlantVillage dataset from Kaggle
2. Place in `dataset/raw/` directory
3. Run preprocessing script: `python training/utils.py --preprocess`

**Option 2: Automated Script**
```bash
cd dataset
python download_dataset.py --source plantvillage --output raw/
```

### Data Validation

Before training, validate dataset integrity:

```bash
python training/utils.py --validate-dataset
```

Expected output:
- Class distribution report
- Image format validation
- Missing file detection
- Duplicate detection

### Citation

If you use this dataset, please cite the original sources:

```bibtex
@article{hughes2015open,
  title={An open access repository of images on plant health to enable the development of mobile disease diagnostics},
  author={Hughes, David P and Salathe, Marcel},
  journal={arXiv preprint arXiv:1511.08060},
  year={2015}
}
```

### Contact

For dataset issues or questions:
- Create an issue on the GitHub repository
- Email: your.email@university.edu

---

**Last Updated**: February 2026  
**Version**: 1.0
