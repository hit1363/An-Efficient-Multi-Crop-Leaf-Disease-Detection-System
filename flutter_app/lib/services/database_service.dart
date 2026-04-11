/// Database Service
/// Manages SQLite database for scan history

import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import '../models/scan_result.dart';
import '../utils/constants.dart';

class DatabaseService {
  static final DatabaseService _instance = DatabaseService._internal();
  factory DatabaseService() => _instance;
  DatabaseService._internal();
  
  Database? _database;
  
  /// Get database instance
  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }
  
  /// Initialize database
  Future<Database> _initDatabase() async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, AppConstants.dbName);
    
    return await openDatabase(
      path,
      version: AppConstants.dbVersion,
      onCreate: _onCreate,
      onUpgrade: _onUpgrade,
    );
  }

  /// Backward-compatible explicit initializer.
  Future<void> initialize() async {
    await database;
  }
  
  /// Create database tables
  Future<void> _onCreate(Database db, int version) async {
    // Scan history table
    await db.execute('''
      CREATE TABLE scan_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        disease_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        image_path TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        crop TEXT NOT NULL,
        top_predictions TEXT
      )
    ''');
    
    // Disease info table
    await db.execute('''
      CREATE TABLE disease_info (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        crop TEXT NOT NULL,
        description TEXT,
        symptoms TEXT,
        causes TEXT,
        image_url TEXT
      )
    ''');
    
    // Treatment info table
    await db.execute('''
      CREATE TABLE treatment_info (
        disease_id TEXT PRIMARY KEY,
        disease_name TEXT NOT NULL,
        cultural_control TEXT,
        chemical_control TEXT,
        biological_control TEXT,
        prevention_tips TEXT,
        severity TEXT,
        FOREIGN KEY (disease_id) REFERENCES disease_info(id)
      )
    ''');
    
    print('✅ Database tables created');
  }
  
  /// Handle database upgrades
  Future<void> _onUpgrade(Database db, int oldVersion, int newVersion) async {
    // Handle future schema updates
    if (oldVersion < 2) {
      // Add upgrade logic for version 2
    }
  }
  
  /// Insert scan result
  Future<int> insertScanResult(ScanResult result) async {
    final db = await database;
    return await db.insert('scan_history', result.toMap());
  }

  /// Backward-compatible save API used by existing screens.
  Future<int> saveScanResult(ScanResult result, String imagePath) async {
    if (result.imagePath == imagePath || imagePath.isEmpty) {
      return insertScanResult(result);
    }

    final patchedResult = ScanResult(
      id: result.id,
      diseaseName: result.diseaseName,
      confidence: result.confidence,
      imagePath: imagePath,
      timestamp: result.timestamp,
      crop: result.crop,
      topPredictions: result.topPredictions,
    );

    return insertScanResult(patchedResult);
  }
  
  /// Get all scan results
  Future<List<ScanResult>> getAllScanResults() async {
    final db = await database;
    final maps = await db.query(
      'scan_history',
      orderBy: 'timestamp DESC',
    );
    
    return maps.map((map) => ScanResult.fromMap(map)).toList();
  }

  /// Backward-compatible method name used by existing screens.
  Future<List<ScanResult>> getAllScans() async {
    return getAllScanResults();
  }
  
  /// Get recent scan results (limit)
  Future<List<ScanResult>> getRecentScans({int limit = 10}) async {
    final db = await database;
    final maps = await db.query(
      'scan_history',
      orderBy: 'timestamp DESC',
      limit: limit,
    );
    
    return maps.map((map) => ScanResult.fromMap(map)).toList();
  }
  
  /// Get scans by crop
  Future<List<ScanResult>> getScansByCrop(String crop) async {
    final db = await database;
    final maps = await db.query(
      'scan_history',
      where: 'crop = ?',
      whereArgs: [crop],
      orderBy: 'timestamp DESC',
    );
    
    return maps.map((map) => ScanResult.fromMap(map)).toList();
  }
  
  /// Delete scan result
  Future<int> deleteScanResult(int id) async {
    final db = await database;
    return await db.delete(
      'scan_history',
      where: 'id = ?',
      whereArgs: [id],
    );
  }
  
  /// Clear all scan history
  Future<int> clearAllHistory() async {
    final db = await database;
    return await db.delete('scan_history');
  }

  /// Backward-compatible method name used by existing screens.
  Future<int> deleteAllScans() async {
    return clearAllHistory();
  }
  
  /// Get scan count
  Future<int> getScanCount() async {
    final db = await database;
    final result = await db.rawQuery('SELECT COUNT(*) FROM scan_history');
    return Sqflite.firstIntValue(result) ?? 0;
  }
  
  /// Get statistics
  Future<Map<String, dynamic>> getStatistics() async {
    final db = await database;
    
    final totalScans = await getScanCount();
    
    final cropStats = await db.rawQuery('''
      SELECT crop, COUNT(*) as count
      FROM scan_history
      GROUP BY crop
      ORDER BY count DESC
    ''');
    
    final diseaseStats = await db.rawQuery('''
      SELECT disease_name, COUNT(*) as count
      FROM scan_history
      GROUP BY disease_name
      ORDER BY count DESC
      LIMIT 5
    ''');

    final healthyResult = await db.rawQuery('''
      SELECT COUNT(*) as count
      FROM scan_history
      WHERE LOWER(disease_name) LIKE '%healthy%'
    ''');

    final healthyScans =
        ((healthyResult.first['count'] as num?) ?? 0).toInt();
    final diseaseScans = totalScans - healthyScans;

    final scansByCrop = <String, int>{};
    for (final row in cropStats) {
      final crop = row['crop']?.toString() ?? 'Unknown';
      final count = ((row['count'] as num?) ?? 0).toInt();
      scansByCrop[crop] = count;
    }

    final topDiseases = diseaseStats
        .map(
          (row) => {
            'name': row['disease_name']?.toString() ?? 'Unknown',
            'count': ((row['count'] as num?) ?? 0).toInt(),
          },
        )
        .toList();
    
    return {
      'total_scans': totalScans,
      'by_crop': cropStats,
      'top_diseases': diseaseStats,
      // Backward-compatible camelCase keys for existing UI.
      'totalScans': totalScans,
      'diseaseScans': diseaseScans,
      'healthyScans': healthyScans,
      'topDiseases': topDiseases,
      'scansByCrop': scansByCrop,
    };
  }
  
  /// Close database
  Future<void> close() async {
    final db = await database;
    await db.close();
    _database = null;
  }
}
