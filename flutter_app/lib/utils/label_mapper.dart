/// Label mapping utilities for converting model labels to canonical IDs and names.
class LabelMapping {
  final String id;
  final String crop;
  final String diseaseName;

  const LabelMapping({
    required this.id,
    required this.crop,
    required this.diseaseName,
  });
}

class LabelMapper {
  /// Converts a class label (e.g. Crop___Disease_Name) into canonical fields.
  static LabelMapping fromLabel(String label) {
    final normalizedLabel = label.trim();
    final parts = normalizedLabel.split('___');

    final rawCrop = parts.isNotEmpty ? parts.first : 'Unknown';
    final rawDisease = parts.length > 1 ? parts.sublist(1).join('___') : normalizedLabel;

    final crop = _beautify(rawCrop);
    final diseaseName = _beautify(rawDisease);

    final lowerLabel = normalizedLabel.toLowerCase();
    final id = lowerLabel.contains('healthy')
        ? 'healthy'
        : _slugify('${rawCrop.toLowerCase()}_${rawDisease.toLowerCase()}');

    return LabelMapping(
      id: id,
      crop: crop,
      diseaseName: diseaseName,
    );
  }

  static String toDiseaseId(String label) {
    return fromLabel(label).id;
  }

  static String _beautify(String value) {
    return value
        .replaceAll('_', ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
  }

  static String _slugify(String value) {
    return value
        .replaceAll(RegExp(r'[^a-z0-9]+'), '_')
        .replaceAll(RegExp(r'_+'), '_')
        .replaceAll(RegExp(r'^_|_$'), '');
  }
}
