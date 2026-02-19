/// Treatment Model
/// Represents treatment recommendations for a disease

class Treatment {
  final String diseaseId;
  final String diseaseName;
  final List<String> culturalControl;
  final List<String> chemicalControl;
  final List<String> biologicalControl;
  final List<String> preventionTips;
  final String severity;
  
  Treatment({
    required this.diseaseId,
    required this.diseaseName,
    required this.culturalControl,
    required this.chemicalControl,
    required this.biologicalControl,
    required this.preventionTips,
    this.severity = 'moderate',
  });
  
  /// Create Treatment from JSON
  factory Treatment.fromJson(Map<String, dynamic> json) {
    return Treatment(
      diseaseId: json['disease_id'] as String,
      diseaseName: json['disease_name'] as String,
      culturalControl: List<String>.from(json['cultural_control'] ?? []),
      chemicalControl: List<String>.from(json['chemical_control'] ?? []),
      biologicalControl: List<String>.from(json['biological_control'] ?? []),
      preventionTips: List<String>.from(json['prevention_tips'] ?? []),
      severity: json['severity'] as String? ?? 'moderate',
    );
  }
  
  /// Convert Treatment to JSON
  Map<String, dynamic> toJson() {
    return {
      'disease_id': diseaseId,
      'disease_name': diseaseName,
      'cultural_control': culturalControl,
      'chemical_control': chemicalControl,
      'biological_control': biologicalControl,
      'prevention_tips': preventionTips,
      'severity': severity,
    };
  }
  
  /// Create Treatment from database map
  factory Treatment.fromMap(Map<String, dynamic> map) {
    return Treatment(
      diseaseId: map['disease_id'] as String,
      diseaseName: map['disease_name'] as String,
      culturalControl: (map['cultural_control'] as String).split('|'),
      chemicalControl: (map['chemical_control'] as String).split('|'),
      biologicalControl: (map['biological_control'] as String).split('|'),
      preventionTips: (map['prevention_tips'] as String).split('|'),
      severity: map['severity'] as String? ?? 'moderate',
    );
  }
  
  /// Convert Treatment to database map
  Map<String, dynamic> toMap() {
    return {
      'disease_id': diseaseId,
      'disease_name': diseaseName,
      'cultural_control': culturalControl.join('|'),
      'chemical_control': chemicalControl.join('|'),
      'biological_control': biologicalControl.join('|'),
      'prevention_tips': preventionTips.join('|'),
      'severity': severity,
    };
  }
  
  /// Get all treatment methods combined
  List<String> get allTreatments {
    return [
      ...culturalControl,
      ...chemicalControl,
      ...biologicalControl,
    ];
  }
  
  /// Check if treatment is empty
  bool get isEmpty {
    return culturalControl.isEmpty &&
           chemicalControl.isEmpty &&
           biologicalControl.isEmpty;
  }
}
