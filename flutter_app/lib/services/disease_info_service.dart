/// Disease Info Service
/// Manages disease information and treatment data

import '../models/disease.dart';
import '../models/treatment.dart';

class DiseaseInfoService {
  static final DiseaseInfoService _instance = DiseaseInfoService._internal();
  factory DiseaseInfoService() => _instance;
  DiseaseInfoService._internal();
  
  // In-memory cache for disease information
  final Map<String, Disease> _diseaseCache = {};
  final Map<String, Treatment> _treatmentCache = {};
  
  /// Initialize with sample data
  /// In production, load from database or API
  Future<void> initialize() async {
    _loadSampleData();
  }
  
  /// Load sample disease and treatment data
  void _loadSampleData() {
    // Tomato diseases
    _diseaseCache['tomato_early_blight'] = Disease(
      id: 'tomato_early_blight',
      name: 'Early Blight',
      crop: 'Tomato',
      description: 'Early blight is a common tomato disease caused by the fungus Alternaria solani.',
      symptoms: [
        'Dark brown spots with concentric rings on older leaves',
        'Yellowing around the spots',
        'Premature leaf drop',
        'Stem lesions near soil line',
      ],
      causes: [
        'Fungal infection (Alternaria solani)',
        'Warm, humid conditions',
        'Poor air circulation',
        'Infected plant debris',
      ],
    );
    
    _treatmentCache['tomato_early_blight'] = Treatment(
      diseaseId: 'tomato_early_blight',
      diseaseName: 'Tomato Early Blight',
      culturalControl: [
        'Remove and destroy infected leaves',
        'Improve air circulation by proper spacing',
        'Mulch around plants to prevent soil splash',
        'Rotate crops (3-year rotation)',
        'Water at base of plants, avoid wetting foliage',
      ],
      chemicalControl: [
        'Apply copper-based fungicides',
        'Use chlorothalonil fungicide every 7-10 days',
        'Apply mancozeb as preventive measure',
      ],
      biologicalControl: [
        'Use Bacillus subtilis biofungicide',
        'Apply compost tea as foliar spray',
        'Use neem oil as organic option',
      ],
      preventionTips: [
        'Plant resistant varieties',
        'Maintain proper plant spacing (24-36 inches)',
        'Use drip irrigation instead of overhead watering',
        'Remove weeds regularly',
        'Apply balanced fertilizer for healthy plants',
      ],
      severity: 'moderate',
    );
    
    // Potato diseases
    _diseaseCache['potato_late_blight'] = Disease(
      id: 'potato_late_blight',
      name: 'Late Blight',
      crop: 'Potato',
      description: 'Late blight is a devastating disease caused by Phytophthora infestans.',
      symptoms: [
        'Water-soaked lesions on leaves',
        'White fuzzy growth on undersides of leaves',
        'Brown to black lesions on stems',
        'Brown spots on tubers',
      ],
      causes: [
        'Oomycete pathogen (Phytophthora infestans)',
        'Cool, wet weather',
        'High humidity',
        'Infected seed potatoes',
      ],
    );
    
    _treatmentCache['potato_late_blight'] = Treatment(
      diseaseId: 'potato_late_blight',
      diseaseName: 'Potato Late Blight',
      culturalControl: [
        'Plant certified disease-free seed potatoes',
        'Hill soil around plants to protect tubers',
        'Destroy volunteer potato plants',
        'Harvest during dry weather',
        'Remove infected plants immediately',
      ],
      chemicalControl: [
        'Apply copper fungicides preventively',
        'Use systemic fungicides (metalaxyl)',
        'Apply chlorothalonil before disease appears',
      ],
      biologicalControl: [
        'Limited biological options available',
        'Focus on cultural practices',
      ],
      preventionTips: [
        'Choose resistant varieties',
        'Avoid overhead irrigation',
        'Maintain good air circulation',
        'Monitor weather for favorable conditions',
        'Inspect plants weekly during growing season',
      ],
      severity: 'severe',
    );
    
    // Healthy plants
    _diseaseCache['healthy'] = Disease(
      id: 'healthy',
      name: 'Healthy',
      crop: 'General',
      description: 'Plant shows no signs of disease. Leaves are vibrant and healthy.',
      symptoms: ['No symptoms - plant is healthy'],
      causes: ['N/A'],
    );
    
    _treatmentCache['healthy'] = Treatment(
      diseaseId: 'healthy',
      diseaseName: 'Healthy Plant',
      culturalControl: ['Continue regular care and monitoring'],
      chemicalControl: ['No treatment needed'],
      biologicalControl: ['No treatment needed'],
      preventionTips: [
        'Maintain current care practices',
        'Monitor regularly for early disease detection',
        'Ensure proper watering and fertilization',
        'Keep area clean and free of debris',
      ],
      severity: 'none',
    );
  }
  
  /// Get disease information by ID or name
  Disease? getDisease(String identifier) {
    // Try exact match first
    if (_diseaseCache.containsKey(identifier)) {
      return _diseaseCache[identifier];
    }
    
    // Try case-insensitive match
    final key = _diseaseCache.keys.firstWhere(
      (k) => k.toLowerCase() == identifier.toLowerCase(),
      orElse: () => '',
    );
    
    return key.isNotEmpty ? _diseaseCache[key] : null;
  }
  
  /// Get treatment information by disease ID
  Treatment? getTreatment(String diseaseId) {
    // Try exact match
    if (_treatmentCache.containsKey(diseaseId)) {
      return _treatmentCache[diseaseId];
    }
    
    // Try case-insensitive match
    final key = _treatmentCache.keys.firstWhere(
      (k) => k.toLowerCase() == diseaseId.toLowerCase(),
      orElse: () => '',
    );
    
    return key.isNotEmpty ? _treatmentCache[key] : null;
  }
  
  /// Get all diseases for a crop
  List<Disease> getDiseasesByCrop(String crop) {
    return _diseaseCache.values
        .where((d) => d.crop.toLowerCase() == crop.toLowerCase())
        .toList();
  }
  
  /// Search diseases by name
  List<Disease> searchDiseases(String query) {
    final lowerQuery = query.toLowerCase();
    return _diseaseCache.values
        .where((d) =>
            d.name.toLowerCase().contains(lowerQuery) ||
            d.crop.toLowerCase().contains(lowerQuery))
        .toList();
  }
  
  /// Get all crop types
  List<String> getAllCrops() {
    return _diseaseCache.values
        .map((d) => d.crop)
        .toSet()
        .toList()
      ..sort();
  }
}
