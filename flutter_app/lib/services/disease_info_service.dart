/// Disease Info Service
/// Manages disease information and treatment data

import 'dart:convert';
import 'package:flutter/services.dart';
import '../models/disease.dart';
import '../models/treatment.dart';
import '../utils/label_mapper.dart';

class DiseaseInfoService {
  static final DiseaseInfoService _instance = DiseaseInfoService._internal();
  factory DiseaseInfoService() => _instance;
  DiseaseInfoService._internal();

  static const String _diseaseAssetPath = 'assets/database/disease_info.json';
  static const String _treatmentAssetPath = 'assets/database/treatment_info.json';
  static const String _labelsAssetPath = 'assets/labels/labels.txt';
  
  // In-memory cache for disease information
  final Map<String, Disease> _diseaseCache = {};
  final Map<String, Treatment> _treatmentCache = {};
  bool _initialized = false;
  
  /// Initialize by loading metadata from app assets and ensuring label coverage.
  Future<void> initialize() async {
    if (_initialized) return;

    _diseaseCache.clear();
    _treatmentCache.clear();

    await _loadDiseaseDataFromJson();
    await _loadTreatmentDataFromJson();
    await _ensureCoverageForAllLabels();

    _initialized = true;
  }

  Future<void> _loadDiseaseDataFromJson() async {
    final content = await rootBundle.loadString(_diseaseAssetPath);
    final decoded = jsonDecode(content);
    if (decoded is! List) return;

    for (final entry in decoded) {
      if (entry is! Map<String, dynamic>) continue;

      final id = (entry['id'] as String?)?.trim();
      if (id == null || id.isEmpty) continue;

      final favorableConditions = entry['favorable_conditions'] is List
          ? List<String>.from(entry['favorable_conditions'])
          : <String>[];
      final spread = (entry['spread'] as String?)?.trim();
      final causes = entry['causes'] is List
          ? List<String>.from(entry['causes'])
          : <String>[
              ...favorableConditions,
              if (spread != null && spread.isNotEmpty) spread,
            ];

      _diseaseCache[id] = Disease(
        id: id,
        name: (entry['name'] as String?)?.trim() ?? 'Unknown Disease',
        crop: (entry['crop'] as String?)?.trim() ?? 'Unknown Crop',
        description: (entry['description'] as String?)?.trim() ??
            'No detailed description available.',
        symptoms: entry['symptoms'] is List
            ? List<String>.from(entry['symptoms'])
            : const <String>[],
        causes: causes,
        imageUrl: (entry['image_url'] as String?)?.trim(),
      );
    }
  }

  Future<void> _loadTreatmentDataFromJson() async {
    final content = await rootBundle.loadString(_treatmentAssetPath);
    final decoded = jsonDecode(content);
    if (decoded is! List) return;

    for (final entry in decoded) {
      if (entry is! Map<String, dynamic>) continue;

      final diseaseId = (entry['disease_id'] as String?)?.trim();
      if (diseaseId == null || diseaseId.isEmpty) continue;

      final linkedDisease = _diseaseCache[diseaseId];
      final diseaseName = (entry['disease_name'] as String?)?.trim() ??
          linkedDisease?.displayName ??
          diseaseId.replaceAll('_', ' ');

      _treatmentCache[diseaseId] = Treatment(
        diseaseId: diseaseId,
        diseaseName: diseaseName,
        culturalControl: entry['cultural_control'] is List
            ? List<String>.from(entry['cultural_control'])
            : const <String>[],
        chemicalControl: entry['chemical_control'] is List
            ? List<String>.from(entry['chemical_control'])
            : const <String>[],
        biologicalControl: entry['biological_control'] is List
            ? List<String>.from(entry['biological_control'])
            : const <String>[],
        preventionTips: entry['prevention_tips'] is List
            ? List<String>.from(entry['prevention_tips'])
            : const <String>[],
        severity: (entry['severity'] as String?)?.trim() ??
            _defaultSeverityForDiseaseId(diseaseId),
      );
    }
  }

  Future<void> _ensureCoverageForAllLabels() async {
    final labelsText = await rootBundle.loadString(_labelsAssetPath);
    final labels = labelsText
        .split('\n')
        .map((line) => line.trim())
        .where((line) => line.isNotEmpty)
        .toList();

    for (final label in labels) {
      final mapped = LabelMapper.fromLabel(label);

      _diseaseCache.putIfAbsent(
        mapped.id,
        () => _buildFallbackDisease(mapped),
      );

      _treatmentCache.putIfAbsent(
        mapped.id,
        () => _buildFallbackTreatment(
          mapped,
          _diseaseCache[mapped.id],
        ),
      );
    }

    _diseaseCache.putIfAbsent(
      'healthy',
      () => _buildFallbackDisease(
        const LabelMapping(
          id: 'healthy',
          crop: 'General',
          diseaseName: 'Healthy',
        ),
      ),
    );

    _treatmentCache.putIfAbsent(
      'healthy',
      () => _buildFallbackTreatment(
        const LabelMapping(
          id: 'healthy',
          crop: 'General',
          diseaseName: 'Healthy',
        ),
        _diseaseCache['healthy'],
      ),
    );
  }

  Disease _buildFallbackDisease(LabelMapping mapped) {
    final isHealthy = mapped.id == 'healthy';
    return Disease(
      id: mapped.id,
      name: mapped.diseaseName,
      crop: mapped.crop,
      description: isHealthy
          ? 'Plant appears healthy with no obvious disease symptoms.'
          : 'No curated description available yet for this class.',
      symptoms: isHealthy
          ? const <String>['No visible disease symptoms detected']
          : const <String>[
              'Visual symptoms may vary; inspect leaf color, spots, and lesions.'
            ],
      causes: isHealthy
          ? const <String>['N/A']
          : const <String>[
              'Consult local agronomy guidance to confirm root causes for this class.'
            ],
    );
  }

  Treatment _buildFallbackTreatment(LabelMapping mapped, Disease? disease) {
    final isHealthy = mapped.id == 'healthy';
    final displayName = disease?.displayName ?? '${mapped.crop} - ${mapped.diseaseName}';

    return Treatment(
      diseaseId: mapped.id,
      diseaseName: displayName,
      culturalControl: isHealthy
          ? const <String>[
              'Maintain current care routine and continue periodic monitoring.',
            ]
          : const <String>[
              'Remove severely affected leaves and keep field hygiene high.',
              'Improve airflow and avoid prolonged leaf wetness when possible.',
            ],
      chemicalControl: isHealthy
          ? const <String>['No chemical control needed for healthy plants.']
          : const <String>[
              'Use crop-appropriate products only after confirming diagnosis locally.',
            ],
      biologicalControl: isHealthy
          ? const <String>['No biological control needed.']
          : const <String>[
              'Consider locally recommended bio-control options as applicable.',
            ],
      preventionTips: isHealthy
          ? const <String>[
              'Keep monitoring regularly for early symptom detection.',
            ]
          : const <String>[
              'Use clean tools and avoid moving infected plant material between plots.',
              'Follow local integrated pest and disease management guidelines.',
            ],
      severity: isHealthy ? 'none' : _defaultSeverityForDiseaseId(mapped.id),
    );
  }

  String _defaultSeverityForDiseaseId(String diseaseId) {
    return diseaseId == 'healthy' ? 'none' : 'moderate';
  }

  String _normalizeIdentifier(String identifier) {
    final trimmed = identifier.trim();
    if (trimmed.contains('___')) {
      return LabelMapper.toDiseaseId(trimmed);
    }
    return trimmed;
  }

  T? _findCaseInsensitive<T>(Map<String, T> source, String key) {
    final lower = key.toLowerCase();
    for (final entry in source.entries) {
      if (entry.key.toLowerCase() == lower) {
        return entry.value;
      }
    }
    return null;
  }
  
  /// Get disease information by ID or name
  Disease? getDisease(String identifier) {
    final normalized = _normalizeIdentifier(identifier);

    final direct = _diseaseCache[normalized];
    if (direct != null) {
      return direct;
    }

    final byId = _findCaseInsensitive(_diseaseCache, normalized);
    if (byId != null) {
      return byId;
    }

    return _diseaseCache.values.cast<Disease?>().firstWhere(
      (d) => d != null && d.name.toLowerCase() == normalized.toLowerCase(),
      orElse: () => null,
    );
  }
  
  /// Get treatment information by disease ID
  Treatment? getTreatment(String diseaseId) {
    final normalized = _normalizeIdentifier(diseaseId);

    final direct = _treatmentCache[normalized];
    if (direct != null) {
      return direct;
    }

    final byId = _findCaseInsensitive(_treatmentCache, normalized);
    if (byId != null) {
      return byId;
    }

    final mapped = LabelMapper.fromLabel(diseaseId);
    final disease = getDisease(mapped.id);
    return _buildFallbackTreatment(mapped, disease);
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
