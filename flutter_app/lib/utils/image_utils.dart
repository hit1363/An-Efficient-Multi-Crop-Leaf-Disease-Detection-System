/// Image Utility Functions
/// Handles image preprocessing and manipulation

import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'constants.dart';

class ImageUtils {
  /// Preprocess image for model inference
  /// Resizes to 224x224 and normalizes pixel values to [0, 1]
  static Float32List preprocessImage(File imageFile) {
    // Read image file
    final imageBytes = imageFile.readAsBytesSync();
    final image = img.decodeImage(imageBytes);
    
    if (image == null) {
      throw Exception('Failed to decode image');
    }
    
    // Resize to model input size
    final resizedImage = img.copyResize(
      image,
      width: AppConstants.imageSize,
      height: AppConstants.imageSize,
    );
    
    // Convert to Float32List and normalize [0, 255] -> [0, 1]
    final convertedBytes = Float32List(
      AppConstants.imageSize * 
      AppConstants.imageSize * 
      AppConstants.numChannels
    );
    
    int pixelIndex = 0;
    for (int y = 0; y < resizedImage.height; y++) {
      for (int x = 0; x < resizedImage.width; x++) {
        final pixel = resizedImage.getPixel(x, y);
        
        // Extract RGB values and normalize
        convertedBytes[pixelIndex++] = img.getRed(pixel) / 255.0;
        convertedBytes[pixelIndex++] = img.getGreen(pixel) / 255.0;
        convertedBytes[pixelIndex++] = img.getBlue(pixel) / 255.0;
      }
    }
    
    return convertedBytes;
  }
  
  /// Create thumbnail from image file
  static Future<Uint8List> createThumbnail(
    File imageFile, {
    int maxWidth = 200,
    int maxHeight = 200,
  }) async {
    final imageBytes = await imageFile.readAsBytes();
    final image = img.decodeImage(imageBytes);
    
    if (image == null) {
      throw Exception('Failed to decode image');
    }
    
    final thumbnail = img.copyResize(
      image,
      width: maxWidth,
      height: maxHeight,
    );
    
    return Uint8List.fromList(img.encodeJpg(thumbnail, quality: 85));
  }
  
  /// Check if image meets quality requirements
  static bool isImageQualityAcceptable(File imageFile) {
    try {
      final imageBytes = imageFile.readAsBytesSync();
      final image = img.decodeImage(imageBytes);
      
      if (image == null) return false;
      
      // Check minimum dimensions
      const minDimension = 100;
      if (image.width < minDimension || image.height < minDimension) {
        return false;
      }
      
      // Check if image is too dark or too bright
      int totalBrightness = 0;
      int pixelCount = 0;
      
      for (int y = 0; y < image.height; y += 10) {
        for (int x = 0; x < image.width; x += 10) {
          final pixel = image.getPixel(x, y);
          final brightness = (img.getRed(pixel) + 
                            img.getGreen(pixel) + 
                            img.getBlue(pixel)) ~/ 3;
          totalBrightness += brightness;
          pixelCount++;
        }
      }
      
      final avgBrightness = totalBrightness / pixelCount;
      
      // Reject if too dark (<30) or too bright (>225)
      if (avgBrightness < 30 || avgBrightness > 225) {
        return false;
      }
      
      return true;
    } catch (e) {
      return false;
    }
  }
  
  /// Get image dimensions
  static Map<String, int> getImageDimensions(File imageFile) {
    final imageBytes = imageFile.readAsBytesSync();
    final image = img.decodeImage(imageBytes);
    
    if (image == null) {
      throw Exception('Failed to decode image');
    }
    
    return {
      'width': image.width,
      'height': image.height,
    };
  }
  
  /// Rotate image by specified angle
  static Future<File> rotateImage(File imageFile, int angle) async {
    final imageBytes = await imageFile.readAsBytes();
    final image = img.decodeImage(imageBytes);
    
    if (image == null) {
      throw Exception('Failed to decode image');
    }
    
    final rotated = img.copyRotate(image, angle: angle);
    final rotatedBytes = img.encodeJpg(rotated);
    
    final rotatedFile = File('${imageFile.path}_rotated.jpg');
    await rotatedFile.writeAsBytes(rotatedBytes);
    
    return rotatedFile;
  }
}
