/// Results Screen
/// Display disease detection results with treatment recommendations

import 'dart:io';
import 'package:flutter/material.dart';
import '../services/ml_service.dart';
import '../services/database_service.dart';
import '../services/disease_info_service.dart';
import '../models/scan_result.dart';
import '../models/disease.dart';
import '../models/treatment.dart';
import '../widgets/disease_card.dart';
import '../widgets/confidence_bar.dart';
import '../widgets/treatment_card.dart';
import '../utils/constants.dart';
import '../utils/image_utils.dart';

class ResultsScreen extends StatefulWidget {
  final String? imagePath;
  final ScanResult? scanResult;
  
  const ResultsScreen({
    Key? key,
    this.imagePath,
    this.scanResult,
  }) : super(key: key);
  
  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  final _mlService = MLService();
  final _databaseService = DatabaseService();
  final _diseaseInfoService = DiseaseInfoService();
  
  ScanResult? _result;
  bool _isProcessing = false;
  String? _error;
  Treatment? _treatment;
  
  @override
  void initState() {
    super.initState();
    
    if (widget.scanResult != null) {
      _result = widget.scanResult;
      _loadTreatment();
    } else if (widget.imagePath != null) {
      _processImage();
    }
  }
  
  Future<void> _processImage() async {
    if (widget.imagePath == null) return;
    
    setState(() {
      _isProcessing = true;
      _error = null;
    });
    
    try {
      // Validate image
      final validation = await ImageUtils.validateImage(widget.imagePath!);
      if (!validation['isValid']) {
        throw Exception(validation['error']);
      }
      
      // Preprocess and run inference
      final processedImage = await ImageUtils.preprocessImage(widget.imagePath!);
      final result = await _mlService.classifyImage(processedImage);
      
      // Save to database
      await _databaseService.saveScanResult(result, widget.imagePath!);
      
      // Load treatment info
      await _loadTreatment();
      
      setState(() {
        _result = result;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isProcessing = false;
      });
    }
  }
  
  Future<void> _loadTreatment() async {
    if (_result == null) return;
    
    final topDisease = _result!.predictions.first.disease;
    final treatment = await _diseaseInfoService.getTreatment(topDisease.id);
    
    setState(() {
      _treatment = treatment;
    });
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan Results'),
        actions: [
          if (_result != null)
            IconButton(
              icon: const Icon(Icons.share),
              onPressed: _shareResults,
            ),
        ],
      ),
      body: _buildBody(),
    );
  }
  
  Widget _buildBody() {
    if (_isProcessing) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 24),
            Text(
              'Analyzing image...',
              style: AppConstants.subheadingStyle,
            ),
            const SizedBox(height: 8),
            Text(
              'This may take a few seconds',
              style: AppConstants.captionStyle,
            ),
          ],
        ),
      );
    }
    
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.error_outline,
                size: 64,
                color: AppConstants.errorColor,
              ),
              const SizedBox(height: 16),
              Text(
                'Error',
                style: AppConstants.headingStyle,
              ),
              const SizedBox(height: 8),
              Text(
                _error!,
                style: AppConstants.bodyStyle,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Go Back'),
              ),
            ],
          ),
        ),
      );
    }
    
    if (_result == null) {
      return const Center(child: Text('No results available'));
    }
    
    return SingleChildScrollView(
      padding: const EdgeInsets.all(AppConstants.padding),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Image preview
          _buildImagePreview(),
          
          const SizedBox(height: 24),
          
          // Top prediction
          _buildTopPrediction(),
          
          const SizedBox(height: 24),
          
          // All predictions
          _buildAllPredictions(),
          
          const SizedBox(height: 24),
          
          // Treatment recommendations
          if (_treatment != null) ...[
            _buildTreatmentSection(),
            const SizedBox(height: 24),
          ],
          
          // Action buttons
          _buildActionButtons(),
        ],
      ),
    );
  }
  
  Widget _buildImagePreview() {
    final imagePath = widget.imagePath ?? _result?.imagePath;
    if (imagePath == null) return const SizedBox.shrink();
    
    return ClipRRect(
      borderRadius: BorderRadius.circular(AppConstants.borderRadius),
      child: Image.file(
        File(imagePath),
        height: 250,
        width: double.infinity,
        fit: BoxFit.cover,
      ),
    );
  }
  
  Widget _buildTopPrediction() {
    final topPrediction = _result!.predictions.first;
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Diagnosis',
          style: AppConstants.headingStyle,
        ),
        const SizedBox(height: 12),
        DiseaseCard(
          disease: topPrediction.disease,
          confidence: topPrediction.confidence,
        ),
      ],
    );
  }
  
  Widget _buildAllPredictions() {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppConstants.borderRadius),
      ),
      child: Padding(
        padding: const EdgeInsets.all(AppConstants.padding),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Confidence Scores',
              style: AppConstants.subheadingStyle,
            ),
            const SizedBox(height: 16),
            ..._result!.predictions.map((pred) {
              return Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: ConfidenceBar(
                  confidence: pred.confidence,
                  label: '${pred.disease.crop} - ${pred.disease.name}',
                ),
              );
            }).toList(),
          ],
        ),
      ),
    );
  }
  
  Widget _buildTreatmentSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Treatment Guide',
          style: AppConstants.headingStyle,
        ),
        const SizedBox(height: 12),
        TreatmentCard(treatment: _treatment!),
      ],
    );
  }
  
  Widget _buildActionButtons() {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(Icons.home),
            label: const Text('Home'),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.all(16),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: OutlinedButton.icon(
            onPressed: () {
              // Navigate back and open camera
              Navigator.popUntil(context, (route) => route.isFirst);
            },
            icon: const Icon(Icons.camera_alt),
            label: const Text('New Scan'),
            style: OutlinedButton.styleFrom(
              padding: const EdgeInsets.all(16),
            ),
          ),
        ),
      ],
    );
  }
  
  void _shareResults() {
    // TODO: Implement sharing functionality
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Share feature coming soon')),
    );
  }
}
