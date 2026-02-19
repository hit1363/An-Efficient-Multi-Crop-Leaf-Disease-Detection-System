/// Scan Result Model
/// Represents the result of a disease detection scan

class ScanResult {
  final int? id;
  final String diseaseName;
  final double confidence;
  final String imagePath;
  final DateTime timestamp;
  final String crop;
  final List<Prediction> topPredictions;
  
  ScanResult({
    this.id,
    required this.diseaseName,
    required this.confidence,
    required this.imagePath,
    required this.timestamp,
    required this.crop,
    this.topPredictions = const [],
  });
  
  /// Create ScanResult from database map
  factory ScanResult.fromMap(Map<String, dynamic> map) {
    // Parse top predictions if stored as JSON string
    List<Prediction> predictions = [];
    if (map['top_predictions'] != null) {
      final predList = map['top_predictions'] as String;
      // Simple parsing - in production use proper JSON
      final parts = predList.split(';');
      for (var part in parts) {
        if (part.isNotEmpty) {
          final values = part.split(':');
          if (values.length == 2) {
            predictions.add(Prediction(
              label: values[0],
              confidence: double.parse(values[1]),
            ));
          }
        }
      }
    }
    
    return ScanResult(
      id: map['id'] as int?,
      diseaseName: map['disease_name'] as String,
      confidence: map['confidence'] as double,
      imagePath: map['image_path'] as String,
      timestamp: DateTime.parse(map['timestamp'] as String),
      crop: map['crop'] as String,
      topPredictions: predictions,
    );
  }
  
  /// Convert ScanResult to database map
  Map<String, dynamic> toMap() {
    // Serialize top predictions to string
    final predString = topPredictions
        .map((p) => '${p.label}:${p.confidence}')
        .join(';');
    
    return {
      'id': id,
      'disease_name': diseaseName,
      'confidence': confidence,
      'image_path': imagePath,
      'timestamp': timestamp.toIso8601String(),
      'crop': crop,
      'top_predictions': predString,
    };
  }
  
  /// Check if confidence is above threshold
  bool get isConfident => confidence >= 0.75;
  
  /// Get confidence as percentage string
  String get confidencePercentage => '${(confidence * 100).toStringAsFixed(1)}%';
  
  /// Get formatted timestamp
  String get formattedDate {
    final now = DateTime.now();
    final diff = now.difference(timestamp);
    
    if (diff.inDays == 0) {
      if (diff.inHours == 0) {
        return '${diff.inMinutes} minutes ago';
      }
      return '${diff.inHours} hours ago';
    } else if (diff.inDays == 1) {
      return 'Yesterday';
    } else if (diff.inDays < 7) {
      return '${diff.inDays} days ago';
    } else {
      return '${timestamp.day}/${timestamp.month}/${timestamp.year}';
    }
  }
  
  /// Check if disease is healthy
  bool get isHealthy => diseaseName.toLowerCase().contains('healthy');
  
  @override
  String toString() => '$diseaseName ($confidencePercentage)';
}

/// Individual Prediction
class Prediction {
  final String label;
  final double confidence;
  
  Prediction({
    required this.label,
    required this.confidence,
  });
  
  String get confidencePercentage => '${(confidence * 100).toStringAsFixed(1)}%';
  
  @override
  String toString() => '$label: $confidencePercentage';
}
