/// Disease Model
/// Represents a plant disease with information and treatment

class Disease {
  final String id;
  final String name;
  final String crop;
  final String description;
  final List<String> symptoms;
  final List<String> causes;
  final String? imageUrl;
  
  Disease({
    required this.id,
    required this.name,
    required this.crop,
    required this.description,
    required this.symptoms,
    required this.causes,
    this.imageUrl,
  });
  
  /// Create Disease from JSON
  factory Disease.fromJson(Map<String, dynamic> json) {
    return Disease(
      id: json['id'] as String,
      name: json['name'] as String,
      crop: json['crop'] as String,
      description: json['description'] as String,
      symptoms: List<String>.from(json['symptoms'] ?? []),
      causes: List<String>.from(json['causes'] ?? []),
      imageUrl: json['image_url'] as String?,
    );
  }
  
  /// Convert Disease to JSON
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'crop': crop,
      'description': description,
      'symptoms': symptoms,
      'causes': causes,
      'image_url': imageUrl,
    };
  }
  
  /// Create Disease from database map
  factory Disease.fromMap(Map<String, dynamic> map) {
    return Disease(
      id: map['id'] as String,
      name: map['name'] as String,
      crop: map['crop'] as String,
      description: map['description'] as String,
      symptoms: (map['symptoms'] as String).split('|'),
      causes: (map['causes'] as String).split('|'),
      imageUrl: map['image_url'] as String?,
    );
  }
  
  /// Convert Disease to database map
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'name': name,
      'crop': crop,
      'description': description,
      'symptoms': symptoms.join('|'),
      'causes': causes.join('|'),
      'image_url': imageUrl,
    };
  }
  
  /// Get display name (e.g., "Tomato - Early Blight")
  String get displayName => '$crop - $name';
  
  /// Check if disease is healthy (no disease)
  bool get isHealthy => name.toLowerCase().contains('healthy');
  
  @override
  String toString() => displayName;
}
