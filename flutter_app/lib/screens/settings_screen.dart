/// Settings Screen
/// App settings, statistics, and about information

import 'package:flutter/material.dart';
import '../services/database_service.dart';
import '../services/ml_service.dart';
import '../utils/constants.dart';

class SettingsScreen extends StatefulWidget {
  final bool showStats;
  
  const SettingsScreen({
    Key? key,
    this.showStats = false,
  }) : super(key: key);
  
  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen>
    with SingleTickerProviderStateMixin {
  final _databaseService = DatabaseService();
  final _mlService = MLService();
  
  late TabController _tabController;
  Map<String, dynamic>? _statistics;
  Map<String, dynamic>? _modelInfo;
  bool _isLoading = true;
  
  @override
  void initState() {
    super.initState();
    _tabController = TabController(
      length: 3,
      vsync: this,
      initialIndex: widget.showStats ? 1 : 0,
    );
    _loadData();
  }
  
  Future<void> _loadData() async {
    setState(() => _isLoading = true);
    
    try {
      final stats = await _databaseService.getStatistics();
      final modelInfo = await _mlService.getModelInfo();
      
      setState(() {
        _statistics = stats;
        _modelInfo = modelInfo;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading data: $e')),
        );
      }
    }
  }
  
  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(icon: Icon(Icons.settings), text: 'Settings'),
            Tab(icon: Icon(Icons.bar_chart), text: 'Statistics'),
            Tab(icon: Icon(Icons.info), text: 'About'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildSettingsTab(),
          _buildStatisticsTab(),
          _buildAboutTab(),
        ],
      ),
    );
  }
  
  Widget _buildSettingsTab() {
    return ListView(
      padding: const EdgeInsets.all(AppConstants.padding),
      children: [
        // Data Management Section
        _buildSectionHeader('Data Management'),
        Card(
          child: Column(
            children: [
              ListTile(
                leading: const Icon(Icons.delete_outline),
                title: const Text('Clear History'),
                subtitle: const Text('Delete all scan history'),
                onTap: _clearHistory,
              ),
              const Divider(height: 1),
              ListTile(
                leading: const Icon(Icons.storage),
                title: const Text('Cache Size'),
                subtitle: const Text('View and manage app cache'),
                trailing: const Text('Coming soon'),
                enabled: false,
              ),
            ],
          ),
        ),
        
        const SizedBox(height: 24),
        
        // Model Settings Section
        _buildSectionHeader('Model Settings'),
        Card(
          child: Column(
            children: [
              ListTile(
                leading: const Icon(Icons.model_training),
                title: const Text('Model Version'),
                subtitle: Text(_modelInfo?['version'] ?? 'Loading...'),
              ),
              const Divider(height: 1),
              ListTile(
                leading: const Icon(Icons.info_outline),
                title: const Text('Model Details'),
                subtitle: Text(
                  'Classes: ${_modelInfo?['classCount'] ?? '—'} | '
                  'Input: ${_modelInfo?['inputShape'] ?? '—'}',
                ),
              ),
            ],
          ),
        ),
        
        const SizedBox(height: 24),
        
        // Preferences Section
        _buildSectionHeader('Preferences'),
        Card(
          child: Column(
            children: [
              SwitchListTile(
                secondary: const Icon(Icons.flash_on),
                title: const Text('Auto Flash'),
                subtitle: const Text('Automatically use flash in low light'),
                value: false,
                onChanged: (value) {
                  // TODO: Implement preference storage
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Feature coming soon')),
                  );
                },
              ),
              const Divider(height: 1),
              SwitchListTile(
                secondary: const Icon(Icons.save),
                title: const Text('Auto Save Images'),
                subtitle: const Text('Save images after scanning'),
                value: true,
                onChanged: (value) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Feature coming soon')),
                  );
                },
              ),
            ],
          ),
        ),
      ],
    );
  }
  
  Widget _buildStatisticsTab() {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }
    
    final stats = _statistics ?? {};
    final totalScans = stats['totalScans'] ?? 0;
    final diseaseScans = stats['diseaseScans'] ?? 0;
    final healthyScans = stats['healthyScans'] ?? 0;
    final topDiseases = (stats['topDiseases'] as List?) ?? [];
    final scansByCrop = (stats['scansByCrop'] as Map?) ?? {};
    
    return ListView(
      padding: const EdgeInsets.all(AppConstants.padding),
      children: [
        // Overview Cards
        Row(
          children: [
            Expanded(
              child: _buildStatCard(
                'Total Scans',
                totalScans.toString(),
                Icons.camera_alt,
                AppConstants.primaryColor,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildStatCard(
                'Diseases Found',
                diseaseScans.toString(),
                Icons.warning,
                AppConstants.errorColor,
              ),
            ),
          ],
        ),
        
        const SizedBox(height: 12),
        
        Row(
          children: [
            Expanded(
              child: _buildStatCard(
                'Healthy Plants',
                healthyScans.toString(),
                Icons.check_circle,
                AppConstants.successColor,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildStatCard(
                'Success Rate',
                totalScans > 0
                    ? '${((healthyScans / totalScans) * 100).toStringAsFixed(0)}%'
                    : '—',
                Icons.trending_up,
                Colors.blue,
              ),
            ),
          ],
        ),
        
        const SizedBox(height: 24),
        
        // Top Diseases
        if (topDiseases.isNotEmpty) ...[
          _buildSectionHeader('Most Detected Diseases'),
          Card(
            child: Column(
              children: [
                ...topDiseases.take(5).map((disease) {
                  final name = disease['name'] ?? 'Unknown';
                  final count = disease['count'] ?? 0;
                  return ListTile(
                    leading: CircleAvatar(
                      backgroundColor:
                          AppConstants.primaryColor.withOpacity(0.1),
                      child: Text(
                        count.toString(),
                        style: TextStyle(
                          color: AppConstants.primaryColor,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    title: Text(name),
                    subtitle: Text('$count ${count == 1 ? 'scan' : 'scans'}'),
                  );
                }).toList(),
              ],
            ),
          ),
          const SizedBox(height: 24),
        ],
        
        // Scans by Crop
        if (scansByCrop.isNotEmpty) ...[
          _buildSectionHeader('Scans by Crop'),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: scansByCrop.entries.map((entry) {
                  final crop = entry.key;
                  final count = entry.value as int;
                  final percentage = totalScans > 0
                      ? (count / totalScans * 100).toStringAsFixed(0)
                      : '0';
                  
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(crop, style: AppConstants.bodyStyle),
                            Text(
                              '$count ($percentage%)',
                              style: AppConstants.captionStyle,
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        LinearProgressIndicator(
                          value: totalScans > 0 ? count / totalScans : 0,
                          backgroundColor: Colors.grey[200],
                          valueColor: AlwaysStoppedAnimation<Color>(
                            AppConstants.primaryColor,
                          ),
                        ),
                      ],
                    ),
                  );
                }).toList(),
              ),
            ),
          ),
        ],
        
        if (totalScans == 0)
          Center(
            child: Padding(
              padding: const EdgeInsets.all(32),
              child: Column(
                children: [
                  Icon(
                    Icons.bar_chart,
                    size: 64,
                    color: Colors.grey[400],
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'No statistics yet',
                    style: AppConstants.subheadingStyle.copyWith(
                      color: Colors.grey[600],
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Start scanning to see statistics',
                    style: AppConstants.captionStyle.copyWith(
                      color: Colors.grey[500],
                    ),
                  ),
                ],
              ),
            ),
          ),
      ],
    );
  }
  
  Widget _buildAboutTab() {
    return ListView(
      padding: const EdgeInsets.all(AppConstants.padding),
      children: [
        // App Info Card
        Card(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              children: [
                const Icon(
                  Icons.eco,
                  size: 64,
                  color: AppConstants.primaryColor,
                ),
                const SizedBox(height: 16),
                Text(
                  'Leaf Disease Detection',
                  style: AppConstants.headingStyle,
                ),
                const SizedBox(height: 8),
                Text(
                  'Version 1.0.0',
                  style: AppConstants.captionStyle,
                ),
                const SizedBox(height: 16),
                Text(
                  'An AI-powered mobile application for detecting plant diseases using deep learning.',
                  style: AppConstants.bodyStyle,
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
        
        const SizedBox(height: 24),
        
        // Model Information
        _buildSectionHeader('Model Information'),
        Card(
          child: Column(
            children: [
              ListTile(
                leading: const Icon(Icons.architecture),
                title: const Text('Architecture'),
                subtitle: const Text('MobileNetV2 with Transfer Learning'),
              ),
              const Divider(height: 1),
              ListTile(
                leading: const Icon(Icons.label),
                title: const Text('Supported Classes'),
                subtitle: Text('${_modelInfo?['classCount'] ?? '—'} diseases'),
              ),
              const Divider(height: 1),
              ListTile(
                leading: const Icon(Icons.speed),
                title: const Text('Optimization'),
                subtitle: const Text('INT8 Quantization for faster inference'),
              ),
            ],
          ),
        ),
        
        const SizedBox(height: 24),
        
        // Resources
        _buildSectionHeader('Resources'),
        Card(
          child: Column(
            children: [
              ListTile(
                leading: const Icon(Icons.help_outline),
                title: const Text('Help & FAQ'),
                trailing: const Icon(Icons.arrow_forward_ios, size: 16),
                onTap: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Feature coming soon')),
                  );
                },
              ),
              const Divider(height: 1),
              ListTile(
                leading: const Icon(Icons.privacy_tip_outlined),
                title: const Text('Privacy Policy'),
                trailing: const Icon(Icons.arrow_forward_ios, size: 16),
                onTap: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Feature coming soon')),
                  );
                },
              ),
              const Divider(height: 1),
              ListTile(
                leading: const Icon(Icons.description_outlined),
                title: const Text('Terms of Service'),
                trailing: const Icon(Icons.arrow_forward_ios, size: 16),
                onTap: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Feature coming soon')),
                  );
                },
              ),
            ],
          ),
        ),
        
        const SizedBox(height: 24),
        
        // Credits
        _buildSectionHeader('Credits'),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Developed as part of a Master\'s thesis on efficient multi-crop leaf disease detection.',
                  style: AppConstants.bodyStyle,
                ),
                const SizedBox(height: 12),
                Text(
                  'Dataset: PlantVillage and custom collected data',
                  style: AppConstants.captionStyle.copyWith(
                    color: Colors.grey[600],
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
  
  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(
        title,
        style: AppConstants.subheadingStyle.copyWith(
          color: AppConstants.primaryColor,
        ),
      ),
    );
  }
  
  Widget _buildStatCard(
    String title,
    String value,
    IconData icon,
    Color color,
  ) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Icon(icon, color: color, size: 24),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    value,
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: color,
                      fontSize: 16,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              title,
              style: AppConstants.captionStyle,
            ),
          ],
        ),
      ),
    );
  }
  
  Future<void> _clearHistory() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear History'),
        content: const Text(
          'Are you sure you want to delete all scan history? This cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style: TextButton.styleFrom(
              foregroundColor: AppConstants.errorColor,
            ),
            child: const Text('Clear'),
          ),
        ],
      ),
    );
    
    if (confirm == true) {
      try {
        await _databaseService.deleteAllScans();
        _loadData(); // Reload statistics
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('History cleared successfully')),
          );
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error clearing history: $e')),
          );
        }
      }
    }
  }
}
