/// History Screen
/// Display all previous scan results with filtering and search

import 'dart:io';
import 'package:flutter/material.dart';
import '../services/database_service.dart';
import '../models/scan_result.dart';
import '../utils/constants.dart';
import 'results_screen.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({Key? key}) : super(key: key);
  
  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final _databaseService = DatabaseService();
  List<ScanResult> _allScans = [];
  List<ScanResult> _filteredScans = [];
  bool _isLoading = true;
  String _searchQuery = '';
  String _filterCrop = 'All';
  
  @override
  void initState() {
    super.initState();
    _loadHistory();
  }
  
  Future<void> _loadHistory() async {
    setState(() => _isLoading = true);
    
    try {
      final scans = await _databaseService.getAllScans();
      setState(() {
        _allScans = scans;
        _filteredScans = scans;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading history: $e')),
        );
      }
    }
  }
  
  void _applyFilters() {
    setState(() {
      _filteredScans = _allScans.where((scan) {
        final topPrediction = scan.predictions.first;
        
        // Filter by crop
        if (_filterCrop != 'All' && topPrediction.disease.crop != _filterCrop) {
          return false;
        }
        
        // Filter by search query
        if (_searchQuery.isNotEmpty) {
          final query = _searchQuery.toLowerCase();
          return topPrediction.disease.name.toLowerCase().contains(query) ||
                 topPrediction.disease.crop.toLowerCase().contains(query);
        }
        
        return true;
      }).toList();
    });
  }
  
  Future<void> _deleteAllHistory() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete All History'),
        content: const Text('Are you sure you want to delete all scan history? This cannot be undone.'),
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
            child: const Text('Delete All'),
          ),
        ],
      ),
    );
    
    if (confirm == true) {
      try {
        await _databaseService.deleteAllScans();
        _loadHistory();
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('History cleared')),
          );
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error deleting history: $e')),
          );
        }
      }
    }
  }
  
  @override
  Widget build(BuildContext context) {
    // Get unique crops for filter
    final crops = ['All', ..._allScans
        .map((s) => s.predictions.first.disease.crop)
        .toSet()
        .toList()
      ..sort()];
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan History'),
        actions: [
          if (_allScans.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_outline),
              onPressed: _deleteAllHistory,
            ),
        ],
      ),
      body: Column(
        children: [
          // Search and filter bar
          _buildSearchAndFilter(crops),
          
          // Results count
          if (!_isLoading)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Row(
                children: [
                  Text(
                    '${_filteredScans.length} ${_filteredScans.length == 1 ? 'scan' : 'scans'}',
                    style: AppConstants.captionStyle,
                  ),
                ],
              ),
            ),
          
          // Scan list
          Expanded(
            child: _buildScanList(),
          ),
        ],
      ),
    );
  }
  
  Widget _buildSearchAndFilter(List<String> crops) {
    return Container(
      padding: const EdgeInsets.all(AppConstants.padding),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        children: [
          // Search field
          TextField(
            decoration: InputDecoration(
              hintText: 'Search diseases...',
              prefixIcon: const Icon(Icons.search),
              suffixIcon: _searchQuery.isNotEmpty
                  ? IconButton(
                      icon: const Icon(Icons.clear),
                      onPressed: () {
                        setState(() => _searchQuery = '');
                        _applyFilters();
                      },
                    )
                  : null,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(AppConstants.borderRadius),
              ),
              contentPadding: const EdgeInsets.symmetric(horizontal: 16),
            ),
            onChanged: (value) {
              setState(() => _searchQuery = value);
              _applyFilters();
            },
          ),
          
          const SizedBox(height: 12),
          
          // Crop filter
          SizedBox(
            height: 40,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              itemCount: crops.length,
              itemBuilder: (context, index) {
                final crop = crops[index];
                final isSelected = _filterCrop == crop;
                
                return Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: ChoiceChip(
                    label: Text(crop),
                    selected: isSelected,
                    onSelected: (selected) {
                      setState(() => _filterCrop = crop);
                      _applyFilters();
                    },
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildScanList() {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }
    
    if (_filteredScans.isEmpty) {
      return _buildEmptyState();
    }
    
    return RefreshIndicator(
      onRefresh: _loadHistory,
      child: ListView.builder(
        padding: const EdgeInsets.all(AppConstants.padding),
        itemCount: _filteredScans.length,
        itemBuilder: (context, index) {
          return _buildScanCard(_filteredScans[index]);
        },
      ),
    );
  }
  
  Widget _buildEmptyState() {
    final hasSearchOrFilter = _searchQuery.isNotEmpty || _filterCrop != 'All';
    
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              hasSearchOrFilter ? Icons.search_off : Icons.history,
              size: 64,
              color: Colors.grey[400],
            ),
            const SizedBox(height: 16),
            Text(
              hasSearchOrFilter ? 'No matching scans' : 'No scan history',
              style: AppConstants.subheadingStyle.copyWith(
                color: Colors.grey[600],
              ),
            ),
            const SizedBox(height: 8),
            Text(
              hasSearchOrFilter
                  ? 'Try different search terms or filters'
                  : 'Your scan history will appear here',
              style: AppConstants.captionStyle.copyWith(
                color: Colors.grey[500],
              ),
              textAlign: TextAlign.center,
            ),
            if (hasSearchOrFilter) ...[
              const SizedBox(height: 16),
              TextButton(
                onPressed: () {
                  setState(() {
                    _searchQuery = '';
                    _filterCrop = 'All';
                  });
                  _applyFilters();
                },
                child: const Text('Clear Filters'),
              ),
            ],
          ],
        ),
      ),
    );
  }
  
  Widget _buildScanCard(ScanResult scan) {
    final topPrediction = scan.predictions.first;
    final timeAgo = _formatDateTime(scan.timestamp);
    final confidencePercent = (topPrediction.confidence * 100).toStringAsFixed(0);
    
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppConstants.borderRadius),
      ),
      child: InkWell(
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => ResultsScreen(scanResult: scan),
            ),
          );
        },
        borderRadius: BorderRadius.circular(AppConstants.borderRadius),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              // Thumbnail
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: scan.imagePath != null && File(scan.imagePath!).existsSync()
                    ? Image.file(
                        File(scan.imagePath!),
                        width: 60,
                        height: 60,
                        fit: BoxFit.cover,
                      )
                    : Container(
                        width: 60,
                        height: 60,
                        color: Colors.grey[300],
                        child: const Icon(Icons.image_not_supported),
                      ),
              ),
              
              const SizedBox(width: 12),
              
              // Details
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      topPrediction.disease.name,
                      style: AppConstants.subheadingStyle,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      topPrediction.disease.crop,
                      style: AppConstants.captionStyle,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      timeAgo,
                      style: AppConstants.captionStyle.copyWith(
                        color: Colors.grey[500],
                      ),
                    ),
                  ],
                ),
              ),
              
              // Confidence badge
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: _getConfidenceColor(topPrediction.confidence)
                      .withOpacity(0.1),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  '$confidencePercent%',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: _getConfidenceColor(topPrediction.confidence),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.9) return AppConstants.successColor;
    if (confidence >= 0.75) return AppConstants.primaryColor;
    if (confidence >= 0.5) return AppConstants.warningColor;
    return AppConstants.errorColor;
  }
  
  String _formatDateTime(DateTime dateTime) {
    final now = DateTime.now();
    final difference = now.difference(dateTime);
    
    if (difference.inDays > 7) {
      return '${dateTime.day}/${dateTime.month}/${dateTime.year}';
    } else if (difference.inDays > 0) {
      return '${difference.inDays}d ago';
    } else if (difference.inHours > 0) {
      return '${difference.inHours}h ago';
    } else if (difference.inMinutes > 0) {
      return '${difference.inMinutes}m ago';
    } else {
      return 'Just now';
    }
  }
}
