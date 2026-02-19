/// Disease Card Widget
/// Displays disease information in a card format

import 'package:flutter/material.dart';
import '../models/disease.dart';
import '../utils/constants.dart';

class DiseaseCard extends StatelessWidget {
  final Disease disease;
  final double confidence;
  final VoidCallback? onTap;
  
  const DiseaseCard({
    Key? key,
    required this.disease,
    required this.confidence,
    this.onTap,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    final confidencePercent = (confidence * 100).toStringAsFixed(1);
    final isHealthy = disease.isHealthy;
    final isHighConfidence = confidence >= AppConstants.confidenceThreshold;
    
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppConstants.borderRadius),
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(AppConstants.borderRadius),
        child: Padding(
          padding: const EdgeInsets.all(AppConstants.padding),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header with crop and disease name
              Row(
                children: [
                  // Crop icon
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: isHealthy 
                          ? AppConstants.successColor.withOpacity(0.1)
                          : AppConstants.primaryColor.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Icon(
                      isHealthy ? Icons.check_circle : Icons.warning,
                      color: isHealthy 
                          ? AppConstants.successColor
                          : AppConstants.primaryColor,
                    ),
                  ),
                  const SizedBox(width: 12),
                  
                  // Disease name
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          disease.crop,
                          style: AppConstants.captionStyle.copyWith(
                            color: Colors.grey[600],
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          disease.name,
                          style: AppConstants.subheadingStyle,
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
                      color: _getConfidenceColor(confidence).withOpacity(0.1),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      '$confidencePercent%',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: _getConfidenceColor(confidence),
                      ),
                    ),
                  ),
                ],
              ),
              
              const SizedBox(height: 16),
              
              // Description
              Text(
                disease.description,
                style: AppConstants.bodyStyle.copyWith(
                  color: Colors.grey[700],
                ),
                maxLines: 3,
                overflow: TextOverflow.ellipsis,
              ),
              
              const SizedBox(height: 12),
              
              // Confidence warning if low
              if (!isHighConfidence)
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: AppConstants.warningColor.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        Icons.info_outline,
                        size: 16,
                        color: AppConstants.warningColor,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          'Low confidence. Consider retaking photo.',
                          style: TextStyle(
                            fontSize: 12,
                            color: AppConstants.warningColor,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              
              const SizedBox(height: 8),
              
              // Action button
              if (onTap != null)
                Align(
                  alignment: Alignment.centerRight,
                  child: TextButton.icon(
                    onPressed: onTap,
                    icon: const Icon(Icons.arrow_forward),
                    label: const Text('View Details'),
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
}
