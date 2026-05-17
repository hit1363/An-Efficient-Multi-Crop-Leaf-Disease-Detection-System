/// Disease Info Service
/// Manages disease information and treatment data

import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/disease.dart';
import '../models/treatment.dart';
import '../utils/constants.dart';
import '../utils/label_mapper.dart';

class DiseaseInfoService {
  static final DiseaseInfoService _instance = DiseaseInfoService._internal();
  factory DiseaseInfoService() => _instance;
  DiseaseInfoService._internal();

  static const String _diseaseAssetPath = 'assets/database/disease_info.json';
  static const String _treatmentAssetPath = 'assets/database/treatment_info.json';
  static const String _labelsAssetPath = 'assets/labels/labels.txt';
  
  // Raw JSON entries keyed by disease id
  final Map<String, Map<String, dynamic>> _diseaseRaw = {};
  final Map<String, Map<String, dynamic>> _treatmentRaw = {};

  // Localized caches for current language
  final Map<String, Disease> _diseaseCache = {};
  final Map<String, Treatment> _treatmentCache = {};

  String _languageCode = 'en';
  bool _initialized = false;
  
  /// Initialize by loading metadata from app assets and ensuring label coverage.
  Future<void> initialize() async {
    if (_initialized) return;

    _diseaseCache.clear();
    _treatmentCache.clear();
    _diseaseRaw.clear();
    _treatmentRaw.clear();

    await _loadLanguagePreference();
    await _loadDiseaseDataFromJson();
    await _loadTreatmentDataFromJson();
    await _ensureCoverageForAllLabels();
    _rebuildLocalizedCaches();

    _initialized = true;
  }

  Future<void> _loadLanguagePreference() async {
    final prefs = await SharedPreferences.getInstance();
    _languageCode = prefs.getString(AppConstants.keyLanguage) ?? 'en';
  }

  Future<void> setLanguage(String languageCode) async {
    if (languageCode.isEmpty || _languageCode == languageCode) {
      return;
    }

    _languageCode = languageCode;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(AppConstants.keyLanguage, languageCode);

    _rebuildLocalizedCaches();
  }

  Future<void> _loadDiseaseDataFromJson() async {
    final content = await rootBundle.loadString(_diseaseAssetPath);
    final decoded = jsonDecode(content);
    if (decoded is! List) return;

    for (final entry in decoded) {
      if (entry is! Map<String, dynamic>) continue;

      final id = (entry['id'] as String?)?.trim();
      if (id == null || id.isEmpty) continue;

      _diseaseRaw[id] = Map<String, dynamic>.from(entry);
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

      _treatmentRaw[diseaseId] = Map<String, dynamic>.from(entry);
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

      _diseaseRaw.putIfAbsent(
        mapped.id,
        () => _buildFallbackDiseaseEntry(mapped),
      );

      _treatmentRaw.putIfAbsent(
        mapped.id,
        () => _buildFallbackTreatmentEntry(mapped),
      );
    }

    _diseaseRaw.putIfAbsent(
      'healthy',
      () => _buildFallbackDiseaseEntry(
        const LabelMapping(
          id: 'healthy',
          crop: 'General',
          diseaseName: 'Healthy',
        ),
      ),
    );

    _treatmentRaw.putIfAbsent(
      'healthy',
      () => _buildFallbackTreatmentEntry(
        const LabelMapping(
          id: 'healthy',
          crop: 'General',
          diseaseName: 'Healthy',
        ),
      ),
    );
  }

  bool _isHealthyId(String diseaseId) {
    return diseaseId.toLowerCase().endsWith('_healthy') || diseaseId == 'healthy';
  }

  Map<String, dynamic> _buildFallbackDiseaseEntry(LabelMapping mapped) {
    final isHealthy = _isHealthyId(mapped.id);

    return {
      'id': mapped.id,
      'name_en': mapped.diseaseName,
      'name_bn': mapped.diseaseName,
      'name_ur': mapped.diseaseName,
      'crop_en': mapped.crop,
      'crop_bn': mapped.crop,
      'crop_ur': mapped.crop,
      'description_en': isHealthy
          ? 'Plant appears healthy with no obvious disease symptoms.'
          : 'No curated description available yet for this class.',
      'description_bn': isHealthy
          ? 'গাছে সুস্থতার লক্ষণ রয়েছে, স্পষ্ট রোগের উপসর্গ নেই।'
          : 'এই শ্রেণির জন্য এখনো বিস্তারিত বর্ণনা নেই।',
      'description_ur': isHealthy
          ? 'پودا صحت مند نظر آتا ہے، بیماری کی واضح علامات نہیں ہیں۔'
          : 'اس کلاس کے لیے ابھی کوئی تفصیلی وضاحت دستیاب نہیں۔',
      'symptoms_en': isHealthy
          ? const <String>['No visible disease symptoms detected']
          : const <String>[
              'Visual symptoms may vary; inspect leaf color, spots, and lesions.'
            ],
      'symptoms_bn': isHealthy
          ? const <String>['দৃশ্যমান রোগের উপসর্গ দেখা যায়নি']
          : const <String>[
              'উপসর্গ ভিন্ন হতে পারে; পাতার রং, দাগ ও ক্ষত পর্যবেক্ষণ করুন।'
            ],
      'symptoms_ur': isHealthy
          ? const <String>['بیماری کی کوئی واضح علامات نظر نہیں آتیں']
          : const <String>[
              'علامات مختلف ہو سکتی ہیں؛ پتوں کے رنگ، دھبوں اور زخموں کا مشاہدہ کریں۔'
            ],
      'causes_en': isHealthy
          ? const <String>['N/A']
          : const <String>[
              'Consult local agronomy guidance to confirm root causes for this class.'
            ],
      'causes_bn': isHealthy
          ? const <String>['প্রযোজ্য নয়']
          : const <String>[
              'এই শ্রেণির প্রকৃত কারণ নিশ্চিত করতে স্থানীয় কৃষি বিশেষজ্ঞের পরামর্শ নিন।'
            ],
      'causes_ur': isHealthy
          ? const <String>['لاگو نہیں']
          : const <String>[
              'اصل وجہ کی تصدیق کے لیے مقامی زرعی ماہر سے مشورہ کریں۔'
            ],
    };
  }

  Map<String, dynamic> _buildFallbackTreatmentEntry(LabelMapping mapped) {
    final isHealthy = _isHealthyId(mapped.id);

    return {
      'disease_id': mapped.id,
      'disease_name_en': '${mapped.crop} - ${mapped.diseaseName}',
      'disease_name_bn': '${mapped.crop} - ${mapped.diseaseName}',
      'disease_name_ur': '${mapped.crop} - ${mapped.diseaseName}',
      'cultural_control_en': isHealthy
          ? const <String>[
              'Maintain current care routine and continue periodic monitoring.',
            ]
          : const <String>[
              'Remove severely affected leaves and keep field hygiene high.',
              'Improve airflow and avoid prolonged leaf wetness when possible.',
            ],
      'cultural_control_bn': isHealthy
          ? const <String>[
              'বর্তমান পরিচর্যা বজায় রাখুন এবং নিয়মিত পর্যবেক্ষণ চালিয়ে যান।',
            ]
          : const <String>[
              'অত্যন্ত আক্রান্ত পাতা সরিয়ে দিন এবং ক্ষেত পরিষ্কার রাখুন।',
              'বাতাস চলাচল বাড়ান এবং পাতার দীর্ঘ ভেজাভাব এড়ান।',
            ],
      'cultural_control_ur': isHealthy
          ? const <String>[
              'موجودہ نگہداشت جاری رکھیں اور باقاعدگی سے نگرانی کریں۔',
            ]
          : const <String>[
              'شدید متاثرہ پتے ہٹا دیں اور کھیت کی صفائی رکھیں۔',
              'ہوا کی گردش بہتر کریں اور پتوں کے زیادہ دیر گیلا رہنے سے بچیں۔',
            ],
      'chemical_control_en': isHealthy
          ? const <String>['No chemical control needed for healthy plants.']
          : const <String>[
              'Use crop-appropriate products only after confirming diagnosis locally.',
            ],
      'chemical_control_bn': isHealthy
          ? const <String>['সুস্থ গাছের জন্য রাসায়নিক প্রয়োজন নেই।']
          : const <String>[
              'স্থানীয়ভাবে রোগ নিশ্চিত হওয়ার পরই ফসল উপযোগী পণ্য ব্যবহার করুন।',
            ],
      'chemical_control_ur': isHealthy
          ? const <String>['صحت مند پودوں کے لیے کیمیائی علاج کی ضرورت نہیں۔']
          : const <String>[
              'مقامی تشخیص کے بعد ہی فصل کے مطابق مصنوعات استعمال کریں۔',
            ],
      'biological_control_en': isHealthy
          ? const <String>['No biological control needed.']
          : const <String>[
              'Consider locally recommended bio-control options as applicable.',
            ],
      'biological_control_bn': isHealthy
          ? const <String>['কোনো জৈব নিয়ন্ত্রণের প্রয়োজন নেই।']
          : const <String>[
              'প্রযোজ্য হলে স্থানীয়ভাবে সুপারিশকৃত জৈব নিয়ন্ত্রণ ব্যবহার করুন।',
            ],
      'biological_control_ur': isHealthy
          ? const <String>['کسی حیاتیاتی کنٹرول کی ضرورت نہیں۔']
          : const <String>[
              'ضرورت کے مطابق مقامی طور پر تجویز کردہ بایو کنٹرول اختیار کریں۔',
            ],
      'prevention_tips_en': isHealthy
          ? const <String>[
              'Keep monitoring regularly for early symptom detection.',
            ]
          : const <String>[
              'Use clean tools and avoid moving infected plant material between plots.',
              'Follow local integrated pest and disease management guidelines.',
            ],
      'prevention_tips_bn': isHealthy
          ? const <String>[
              'প্রাথমিক লক্ষণ শনাক্তের জন্য নিয়মিত পর্যবেক্ষণ করুন।',
            ]
          : const <String>[
              'পরিষ্কার যন্ত্রপাতি ব্যবহার করুন এবং আক্রান্ত অংশ অন্য ক্ষেতে না নিন।',
              'স্থানীয় সমন্বিত রোগ ব্যবস্থাপনা নির্দেশিকা অনুসরণ করুন।',
            ],
      'prevention_tips_ur': isHealthy
          ? const <String>[
              'ابتدائی علامات کے لیے باقاعدہ نگرانی کریں۔',
            ]
          : const <String>[
              'صاف اوزار استعمال کریں اور متاثرہ مواد کو کھیتوں کے درمیان منتقل نہ کریں۔',
              'مقامی مربوط بیماری مینجمنٹ ہدایات پر عمل کریں۔',
            ],
      'severity': isHealthy ? 'none' : _defaultSeverityForDiseaseId(mapped.id),
    };
  }

  String _getText(Map<String, dynamic> entry, String baseKey, String fallback) {
    final langKey = '${baseKey}_$_languageCode';
    if (entry[langKey] is String && (entry[langKey] as String).trim().isNotEmpty) {
      return (entry[langKey] as String).trim();
    }

    final enKey = '${baseKey}_en';
    if (entry[enKey] is String && (entry[enKey] as String).trim().isNotEmpty) {
      return (entry[enKey] as String).trim();
    }

    if (entry[baseKey] is String && (entry[baseKey] as String).trim().isNotEmpty) {
      return (entry[baseKey] as String).trim();
    }

    return fallback;
  }

  List<String> _getList(Map<String, dynamic> entry, String baseKey) {
    final langKey = '${baseKey}_$_languageCode';
    if (entry[langKey] is List) {
      return List<String>.from(entry[langKey]);
    }

    final enKey = '${baseKey}_en';
    if (entry[enKey] is List) {
      return List<String>.from(entry[enKey]);
    }

    if (entry[baseKey] is List) {
      return List<String>.from(entry[baseKey]);
    }

    return const <String>[];
  }

  Disease _buildDiseaseFromEntry(String id, Map<String, dynamic> entry) {
    final name = _getText(entry, 'name', 'Unknown Disease');
    final crop = _getText(entry, 'crop', 'Unknown Crop');
    final description = _getText(
      entry,
      'description',
      'No detailed description available.',
    );
    final symptoms = _getList(entry, 'symptoms');
    final causes = _getList(entry, 'causes');
    final favorable = _getList(entry, 'favorable_conditions');
    final spread = _getText(entry, 'spread', '');

    final derivedCauses = causes.isNotEmpty
        ? causes
        : <String>[
            ...favorable,
            if (spread.isNotEmpty) spread,
          ];

    return Disease(
      id: id,
      name: name,
      crop: crop,
      description: description,
      symptoms: symptoms,
      causes: derivedCauses,
      imageUrl: (entry['image_url'] as String?)?.trim(),
    );
  }

  Treatment _buildTreatmentFromEntry(String id, Map<String, dynamic> entry) {
    final diseaseName = _getText(entry, 'disease_name', id.replaceAll('_', ' '));

    return Treatment(
      diseaseId: id,
      diseaseName: diseaseName,
      culturalControl: _getList(entry, 'cultural_control'),
      chemicalControl: _getList(entry, 'chemical_control'),
      biologicalControl: _getList(entry, 'biological_control'),
      preventionTips: _getList(entry, 'prevention_tips'),
      severity: (entry['severity'] as String?)?.trim() ??
          _defaultSeverityForDiseaseId(id),
    );
  }

  void _rebuildLocalizedCaches() {
    _diseaseCache.clear();
    _treatmentCache.clear();

    for (final entry in _diseaseRaw.entries) {
      _diseaseCache[entry.key] = _buildDiseaseFromEntry(entry.key, entry.value);
    }

    for (final entry in _treatmentRaw.entries) {
      _treatmentCache[entry.key] = _buildTreatmentFromEntry(entry.key, entry.value);
    }
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
    final fallbackEntry = _buildFallbackTreatmentEntry(mapped);
    _treatmentRaw[mapped.id] = fallbackEntry;
    final fallback = _buildTreatmentFromEntry(mapped.id, fallbackEntry);
    _treatmentCache[mapped.id] = fallback;
    return fallback;
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
