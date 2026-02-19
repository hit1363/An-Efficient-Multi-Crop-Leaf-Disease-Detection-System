/// ML Service
/// Handles TensorFlow Lite model loading and inference

import 'dart:io';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../utils/constants.dart';
import '../utils/image_utils.dart';
import '../models/scan_result.dart';

class MLService {
  static final MLService _instance = MLService._internal();
  factory MLService() => _instance;
  MLService._internal();
  
  Interpreter? _interpreter;
  List<String>? _labels;
  bool _isModelLoaded = false;
  
  /// Get model loading status
  bool get isModelLoaded => _isModelLoaded;
  
  /// Load TFLite model and labels
  Future<void> loadModel() async {
    if (_isModelLoaded) return;
    
    try {
      // Load model
      _interpreter = await Interpreter.fromAsset(AppConstants.modelPath);
      
      // Load labels
      final labelsData = await rootBundle.loadString(AppConstants.labelsPath);
      _labels = labelsData.split('\n').where((line) => line.isNotEmpty).toList();
      
      _isModelLoaded = true;
      print('✅ Model loaded successfully');
      print('Input shape: ${_interpreter!.getInputTensor(0).shape}');
      print('Output shape: ${_interpreter!.getOutputTensor(0).shape}');
      print('Number of classes: ${_labels!.length}');
    } catch (e) {
      print('❌ Error loading model: $e');
      _isModelLoaded = false;
      rethrow;
    }
  }
  
  /// Run inference on image
  Future<ScanResult> predict(File imageFile) async {
    if (!_isModelLoaded) {
      throw Exception('Model not loaded. Call loadModel() first.');
    }
    
    if (_interpreter == null || _labels == null) {
      throw Exception('Interpreter or labels not initialized');
    }
    
    try {
      // Preprocess image
      final input = ImageUtils.preprocessImage(imageFile);
      
      // Prepare input tensor [1, 224, 224, 3]
      final inputTensor = input.reshape([
        1,
        AppConstants.imageSize,
        AppConstants.imageSize,
        AppConstants.numChannels,
      ]);
      
      // Prepare output tensor [1, numClasses]
      final outputTensor = List.generate(
        1,
        (_) => List<double>.filled(_labels!.length, 0),
      );
      
      // Run inference
      _interpreter!.run(inputTensor, outputTensor);
      
      // Get predictions
      final output = outputTensor[0];
      
      // Find top-3 predictions
      final predictions = <Prediction>[];
      for (int i = 0; i < output.length; i++) {
        predictions.add(Prediction(
          label: _labels![i],
          confidence: output[i],
        ));
      }
      
      // Sort by confidence descending
      predictions.sort((a, b) => b.confidence.compareTo(a.confidence));
      
      // Get top prediction
      final topPrediction = predictions.first;
      
      // Extract crop name (assumes format: "Crop_Disease")
      final parts = topPrediction.label.split('_');
      final crop = parts.isNotEmpty ? parts[0] : 'Unknown';
      
      // Create scan result
      return ScanResult(
        diseaseName: topPrediction.label,
        confidence: topPrediction.confidence,
        imagePath: imageFile.path,
        timestamp: DateTime.now(),
        crop: crop,
        topPredictions: predictions.take(3).toList(),
      );
    } catch (e) {
      print('❌ Error during prediction: $e');
      rethrow;
    }
  }
  
  /// Get model input shape
  List<int> getInputShape() {
    if (_interpreter == null) {
      throw Exception('Model not loaded');
    }
    return _interpreter!.getInputTensor(0).shape;
  }
  
  /// Get model output shape
  List<int> getOutputShape() {
    if (_interpreter == null) {
      throw Exception('Model not loaded');
    }
    return _interpreter!.getOutputTensor(0).shape;
  }
  
  /// Get number of classes
  int getNumClasses() {
    return _labels?.length ?? 0;
  }
  
  /// Get all class labels
  List<String> getLabels() {
    return _labels ?? [];
  }
  
  /// Close interpreter and free resources
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _labels = null;
    _isModelLoaded = false;
    print('🔄 Model resources released');
  }
}
