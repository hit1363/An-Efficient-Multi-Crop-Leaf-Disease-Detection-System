/// Treatment Card Widget
/// Displays treatment recommendations in expandable cards

import 'package:flutter/material.dart';
import '../models/treatment.dart';
import '../utils/constants.dart';

class TreatmentCard extends StatelessWidget {
  final Treatment treatment;
  
  const TreatmentCard({
    Key? key,
    required this.treatment,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
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
            // Header
            Row(
              children: [
                Icon(
                  Icons.medical_services,
                  color: AppConstants.primaryColor,
                  size: 28,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    'Treatment Recommendations',
                    style: AppConstants.headingStyle.copyWith(fontSize: 20),
                  ),
                ),
              ],
            ),
            
            const Divider(height: 24),
            
            // Cultural Control
            if (treatment.culturalControl.isNotEmpty) ...[
              _buildSection(
                title: 'Cultural Control',
                icon: Icons.agriculture,
                color: AppConstants.primaryColor,
                items: treatment.culturalControl,
              ),
              const SizedBox(height: 16),
            ],
            
            // Chemical Control
            if (treatment.chemicalControl.isNotEmpty) ...[
              _buildSection(
                title: 'Chemical Control',
                icon: Icons.science,
                color: AppConstants.accentColor,
                items: treatment.chemicalControl,
              ),
              const SizedBox(height: 16),
            ],
            
            // Biological Control
            if (treatment.biologicalControl.isNotEmpty) ...[
              _buildSection(
                title: 'Biological Control',
                icon: Icons.eco,
                color: AppConstants.successColor,
                items: treatment.biologicalControl,
              ),
              const SizedBox(height: 16),
            ],
            
            // Prevention Tips
            if (treatment.preventionTips.isNotEmpty) ...[
              _buildSection(
                title: 'Prevention Tips',
                icon: Icons.shield,
                color: Colors.blue,
                items: treatment.preventionTips,
              ),
            ],
          ],
        ),
      ),
    );
  }
  
  Widget _buildSection({
    required String title,
    required IconData icon,
    required Color color,
    required List<String> items,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Section header
        Row(
          children: [
            Icon(icon, color: color, size: 20),
            const SizedBox(width: 8),
            Text(
              title,
              style: AppConstants.subheadingStyle.copyWith(
                color: color,
                fontSize: 16,
              ),
            ),
          ],
        ),
        
        const SizedBox(height: 8),
        
        // Items
        ...items.map((item) => Padding(
              padding: const EdgeInsets.only(
                left: 28,
                top: 6,
                bottom: 6,
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    margin: const EdgeInsets.only(top: 6),
                    width: 6,
                    height: 6,
                    decoration: BoxDecoration(
                      color: color,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      item,
                      style: AppConstants.bodyStyle.copyWith(
                        fontSize: 14,
                        color: Colors.grey[700],
                      ),
                    ),
                  ),
                ],
              ),
            )),
      ],
    );
  }
}

/// Expandable Treatment Card with collapsible sections
class ExpandableTreatmentCard extends StatefulWidget {
  final Treatment treatment;
  
  const ExpandableTreatmentCard({
    Key? key,
    required this.treatment,
  }) : super(key: key);
  
  @override
  State<ExpandableTreatmentCard> createState() =>
      _ExpandableTreatmentCardState();
}

class _ExpandableTreatmentCardState extends State<ExpandableTreatmentCard> {
  final Set<String> _expandedSections = {};
  
  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppConstants.borderRadius),
      ),
      child: Column(
        children: [
          // Header
          Container(
            padding: const EdgeInsets.all(AppConstants.padding),
            decoration: BoxDecoration(
              color: AppConstants.primaryColor.withOpacity(0.1),
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(AppConstants.borderRadius),
                topRight: Radius.circular(AppConstants.borderRadius),
              ),
            ),
            child: Row(
              children: [
                Icon(
                  Icons.medical_services,
                  color: AppConstants.primaryColor,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    'Treatment Guide',
                    style: AppConstants.subheadingStyle,
                  ),
                ),
              ],
            ),
          ),
          
          // Sections
          _buildExpansionTile(
            'Cultural Control',
            Icons.agriculture,
            widget.treatment.culturalControl,
          ),
          _buildExpansionTile(
            'Chemical Control',
            Icons.science,
            widget.treatment.chemicalControl,
          ),
          _buildExpansionTile(
            'Biological Control',
            Icons.eco,
            widget.treatment.biologicalControl,
          ),
          _buildExpansionTile(
            'Prevention Tips',
            Icons.shield,
            widget.treatment.preventionTips,
          ),
        ],
      ),
    );
  }
  
  Widget _buildExpansionTile(
    String title,
    IconData icon,
    List<String> items,
  ) {
    if (items.isEmpty) return const SizedBox.shrink();
    
    final isExpanded = _expandedSections.contains(title);
    
    return ExpansionTile(
      leading: Icon(icon, color: AppConstants.primaryColor),
      title: Text(title, style: AppConstants.bodyStyle),
      children: items
          .map((item) => ListTile(
                dense: true,
                leading: const Icon(Icons.check, size: 16),
                title: Text(item, style: const TextStyle(fontSize: 14)),
              ))
          .toList(),
    );
  }
}
