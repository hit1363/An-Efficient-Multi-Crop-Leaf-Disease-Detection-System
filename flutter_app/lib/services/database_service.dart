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
  
  /// Get all scan results
  Future<List<ScanResult>> getAllScanResults() async {
    final db = await database;
    final maps = await db.query(
      'scan_history',
      orderBy: 'timestamp DESC',
    );
    
    return maps.map((map) => ScanResult.fromMap(map)).toList();
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
    
    return {
      'total_scans': totalScans,
      'by_crop': cropStats,
      'top_diseases': diseaseStats,
    };
  }
  
  /// Close database
  Future<void> close() async {
    final db = await database;
    await db.close();
    _database = null;
  }
}
