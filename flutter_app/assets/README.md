# Flutter App Assets

This directory contains the assets required for the Leaf Disease Detection mobile application.

## Directory Structure

### models/
Contains the trained TensorFlow Lite models for disease detection.

**Required Files:**
- `model.tflite` - The quantized MobileNetV2 or EfficientNet-Lite model
- `model_metadata.json` - Model metadata including version, architecture, input/output specs

**To generate the model:**
```bash
# Train the model using the training scripts
cd ../training
python train.py

# Quantize the model
cd ../quantization
python post_training_quant.py

# Copy the generated .tflite file to this directory
cp model_quantized.tflite ../flutter_app/assets/models/model.tflite
```

### labels/
Contains the class labels for disease classification.

**Files:**
- `labels.txt` - List of all disease class names (one per line)

**Format:**
```
Crop___Disease_Name
Crop___healthy
```

The labels file contains 34 classes covering 5 crops:
- Apple (4 classes)
- Blueberry (1 class)
- Cherry (2 classes)
- Corn/Maize (4 classes)
- Grape (4 classes)
- Peach (2 classes)
- Pepper (2 classes)
- Potato (3 classes)
- Strawberry (2 classes)
- Tomato (10 classes)

### images/
Contains app images, icons, and placeholders.

**Suggested Files:**
- `app_logo.png` - App logo
- `splash_screen.png` - Splash screen image
- `placeholder_leaf.png` - Placeholder for missing images
- `onboarding_*.png` - Onboarding screen images

### database/
Contains database initialization scripts and seed data.

**Files:**
- `disease_info.json` - Disease information and descriptions
- `treatment_info.json` - Treatment recommendations
- `schema.sql` - Database schema (optional)

## Usage in Code

Assets are loaded using:
```dart
// Load model
final model = await Interpreter.fromAsset('assets/models/model.tflite');

// Load labels
final labels = await rootBundle.loadString('assets/labels/labels.txt');

// Load image
Image.asset('assets/images/app_logo.png')
```

## File Size Considerations

- Model file: ~3-5 MB (INT8 quantized)
- Total assets: ~10-15 MB
- APK size impact: ~15-20 MB

Ensure assets are optimized:
- Use PNG for transparency, JPEG for photos
- Compress model with quantization
- Use vector graphics (SVG) where possible

## Notes

- All asset paths must be declared in `pubspec.yaml` under `flutter.assets`
- Assets are bundled into the APK/IPA during build
- Large models can be downloaded on first run to reduce APK size
- Consider using Firebase Storage or cloud hosting for large assets

## Updating Assets

1. Add/update files in appropriate directories
2. Run `flutter pub get` to refresh asset references
3. Test with `flutter run` to verify loading
4. For models, update version in `model_metadata.json`

## Production Checklist

- [ ] Trained model file (`model.tflite`)
- [ ] Labels file (`labels.txt`)
- [ ] App logo and icons
- [ ] Disease information database
- [ ] Treatment recommendations
- [ ] Placeholder images
- [ ] Optimize all assets for size
