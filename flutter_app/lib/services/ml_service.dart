/// ML Service
/// Handles TensorFlow Lite model loading and inference

import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
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
  String _preprocessType = AppConstants.preprocessType;
  
  /// Get model loading status
  bool get isModelLoaded => _isModelLoaded;

  /// Backward-compatible initializer used by app bootstrap.
  Future<void> initialize() async {
    await loadModel();
  }
  
  /// Load TFLite model and labels
  Future<void> loadModel() async {
    if (_isModelLoaded) return;
    
    try {
      // Load model
      _interpreter = await Interpreter.fromAsset(AppConstants.modelPath);
      
      // Load labels
      final labelsData = await rootBundle.loadString(AppConstants.labelsPath);
      _labels = labelsData.split('\n').where((line) => line.isNotEmpty).toList();

      // Optional metadata-driven preprocess override
      try {
        final metadataText =
            await rootBundle.loadString('assets/models/model_metadata.json');
        final metadata = jsonDecode(metadataText);
        if (metadata is Map<String, dynamic>) {
          final override = metadata['preprocess_type'];
          if (override is String && override.isNotEmpty) {
            _preprocessType = override;
          }
        }
      } catch (_) {
        _preprocessType = AppConstants.preprocessType;
      }

      final outputShape = _interpreter!.getOutputTensor(0).shape;
      final modelClassCount = outputShape.isNotEmpty ? outputShape.last : 0;
      if (modelClassCount > 0 && _labels!.length != modelClassCount) {
        throw Exception(
          'Label/model mismatch: labels=${_labels!.length}, model_output=$modelClassCount. '
          'Update assets/labels/labels.txt to match the deployed model.'
        );
      }
      
      _isModelLoaded = true;
      print('✅ Model loaded successfully');
      print('Input shape: ${_interpreter!.getInputTensor(0).shape}');
      print('Output shape: ${_interpreter!.getOutputTensor(0).shape}');
      print('Input type: ${_interpreter!.getInputTensor(0).type}');
      print('Output type: ${_interpreter!.getOutputTensor(0).type}');
      print(
        'Input quantization: scale=${_interpreter!.getInputTensor(0).params.scale}, '
        'zeroPoint=${_interpreter!.getInputTensor(0).params.zeroPoint}',
      );
      print(
        'Output quantization: scale=${_interpreter!.getOutputTensor(0).params.scale}, '
        'zeroPoint=${_interpreter!.getOutputTensor(0).params.zeroPoint}',
      );
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
      await loadModel();
    }
    final input = ImageUtils.preprocessImage(
      imageFile,
      preprocessType: _preprocessType,
    );
    return _predictFromInput(input, imagePath: imageFile.path);
  }

  /// Backward-compatible classify API used by existing results screen code.
  Future<ScanResult> classifyImage(dynamic imageInput) async {
    if (!_isModelLoaded) {
      await loadModel();
    }
    if (imageInput is File) {
      return predict(imageInput);
    }

    if (imageInput is String) {
      return predict(File(imageInput));
    }

    if (imageInput is Float32List) {
      return _predictFromInput(imageInput);
    }

    throw Exception('Unsupported image input type: ${imageInput.runtimeType}');
  }

  Future<ScanResult> _predictFromInput(
    Float32List input, {
    String imagePath = '',
  }) async {
    if (!_isModelLoaded) {
      throw Exception('Model not loaded. Call loadModel() first.');
    }

    if (_interpreter == null || _labels == null) {
      throw Exception('Interpreter or labels not initialized');
    }

    try {
      final inputInfo = _interpreter!.getInputTensor(0);
      final outputInfo = _interpreter!.getOutputTensor(0);
      final inputShape = inputInfo.shape;
      final outputShape = outputInfo.shape;
      final inputType = inputInfo.type;
      final outputType = outputInfo.type;
        final inputScale = inputInfo.params.scale == 0
          ? 1.0
          : inputInfo.params.scale;
        final inputZeroPoint = inputInfo.params.zeroPoint;
        final outputScale = outputInfo.params.scale == 0
          ? 1.0
          : outputInfo.params.scale;
        final outputZeroPoint = outputInfo.params.zeroPoint;

      final modelClasses =
          outputShape.isNotEmpty ? outputShape.last : _labels!.length;
      if (_labels!.length != modelClasses) {
        throw Exception(
          'Label/model mismatch at inference: labels=${_labels!.length}, model_output=$modelClasses.',
        );
      }

      dynamic inputTensor;
      if (inputType == TfLiteType.uint8) {
        final inputUint8 = Uint8List(input.length);
        for (int i = 0; i < input.length; i++) {
          final quantized = (input[i] / inputScale + inputZeroPoint).round();
          inputUint8[i] = quantized.clamp(0, 255).toInt();
        }
        inputTensor = inputUint8.toList().reshape(inputShape);
      } else if (inputType == TfLiteType.int8) {
        final inputInt8 = List<int>.filled(input.length, 0);
        for (int i = 0; i < input.length; i++) {
          final quantized = (input[i] / inputScale + inputZeroPoint).round();
          inputInt8[i] = quantized.clamp(-128, 127).toInt();
        }
        inputTensor = inputInt8.reshape(inputShape);
      } else {
        inputTensor = input.toList().reshape(inputShape);
      }

      dynamic outputTensor;
      if (outputType == TfLiteType.uint8 || outputType == TfLiteType.int8) {
        outputTensor = List.generate(1, (_) => List<int>.filled(modelClasses, 0));
      } else {
        outputTensor =
            List.generate(1, (_) => List<double>.filled(modelClasses, 0.0));
      }

      _interpreter!.run(inputTensor, outputTensor);

      List<double> output;
      if (outputType == TfLiteType.uint8) {
        final raw = List<int>.from(outputTensor[0]);
        output = raw
            .map((v) => (v - outputZeroPoint).toDouble() * outputScale)
            .toList();
      } else if (outputType == TfLiteType.int8) {
        final raw = List<int>.from(outputTensor[0]);
        output = raw
            .map((v) => (v - outputZeroPoint).toDouble() * outputScale)
            .toList();
      } else {
        output = List<double>.from(
          (outputTensor[0] as List).map((v) => (v as num).toDouble()),
        );
      }

      final predictions = <Prediction>[];
      for (int i = 0; i < output.length && i < _labels!.length; i++) {
        predictions.add(Prediction(
          label: _labels![i],
          confidence: output[i],
        ));
      }

      predictions.sort((a, b) => b.confidence.compareTo(a.confidence));
      final topPrediction = predictions.first;

      final parts = topPrediction.label.split('___');
      final crop = parts.isNotEmpty ? parts.first : 'Unknown';

      return ScanResult(
        diseaseName: topPrediction.label,
        confidence: topPrediction.confidence,
        imagePath: imagePath,
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

  /// Returns lightweight model details for settings/about screens.
  Future<Map<String, dynamic>> getModelInfo() async {
    Map<String, dynamic> metadata = {};

    try {
      final metadataText =
          await rootBundle.loadString('assets/models/model_metadata.json');
      final parsed = jsonDecode(metadataText);
      if (parsed is Map<String, dynamic>) {
        metadata = parsed;
      }
    } catch (_) {
      // Metadata is optional at runtime.
    }

    if (!_isModelLoaded) {
      try {
        await loadModel();
      } catch (_) {
        // Return best-effort info even if model file isn't present yet.
      }
    }

    final inputShape = _interpreter?.getInputTensor(0).shape;
    final outputShape = _interpreter?.getOutputTensor(0).shape;
    final classCount = _labels?.length ??
        (metadata['num_classes'] is num
            ? (metadata['num_classes'] as num).toInt()
            : 0);

    return {
      'name': metadata['name'] ?? 'Leaf Disease Model',
      'version': metadata['version'] ?? AppConstants.appVersion,
      'architecture': metadata['architecture'] ?? 'Unknown',
      'classCount': classCount,
      'inputShape': inputShape?.toString() ?? 'Unknown',
      'outputShape': outputShape?.toString() ?? 'Unknown',
      'isLoaded': _isModelLoaded,
    };
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
