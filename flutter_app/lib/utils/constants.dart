/// Application Constants
/// Contains all constant values used throughout the app

import 'package:flutter/material.dart';

class AppConstants {
  // App Information
  static const String appName = 'Leaf Disease Detector';
  static const String appVersion = '1.0.0';
  
  // Model Configuration
  static const String modelPath = 'assets/models/mobilenetv2_quantized.tflite';
  static const String labelsPath = 'assets/labels/labels.txt';
  static const int imageSize = 224;
  static const int numChannels = 3;
  static const double confidenceThreshold = 0.75;
  
  // UI Constants
  static const double borderRadius = 12.0;
  static const double padding = 16.0;
  static const double smallPadding = 8.0;
  static const double largePadding = 24.0;
  
  // Colors
  static const Color primaryColor = Color(0xFF4CAF50);
  static const Color secondaryColor = Color(0xFF8BC34A);
  static const Color accentColor = Color(0xFFFF9800);
  static const Color errorColor = Color(0xFFE53935);
  static const Color successColor = Color(0xFF43A047);
  static const Color warningColor = Color(0xFFFFA726);
  
  // Text Styles
  static const TextStyle headingStyle = TextStyle(
    fontSize: 24,
    fontWeight: FontWeight.bold,
    color: Colors.black87,
  );
  
  static const TextStyle subheadingStyle = TextStyle(
    fontSize: 18,
    fontWeight: FontWeight.w600,
    color: Colors.black87,
  );
  
  static const TextStyle bodyStyle = TextStyle(
    fontSize: 16,
    color: Colors.black87,
  );
  
  static const TextStyle captionStyle = TextStyle(
    fontSize: 14,
    color: Colors.black54,
  );
  
  // Database
  static const String dbName = 'leaf_disease.db';
  static const int dbVersion = 1;
  
  // SharedPreferences Keys
  static const String keyFirstLaunch = 'first_launch';
  static const String keyDarkMode = 'dark_mode';
  static const String keyLanguage = 'language';
  
  // Messages
  static const String msgLoadingModel = 'Loading model...';
  static const String msgProcessingImage = 'Processing image...';
  static const String msgNoImageSelected = 'No image selected';
  static const String msgLowConfidence = 'Low confidence prediction. Please retake photo with better lighting.';
  static const String msgPermissionDenied = 'Camera permission denied';
  
  // Animation Durations
  static const Duration shortAnimation = Duration(milliseconds: 200);
  static const Duration normalAnimation = Duration(milliseconds: 300);
  static const Duration longAnimation = Duration(milliseconds: 500);
}

class Routes {
  static const String home = '/';
  static const String camera = '/camera';
  static const String results = '/results';
  static const String history = '/history';
  static const String settings = '/settings';
}
