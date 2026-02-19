/// Confidence Bar Widget
/// Displays a visual confidence indicator

import 'package:flutter/material.dart';
import '../utils/constants.dart';

class ConfidenceBar extends StatelessWidget {
  final double confidence;
  final String? label;
  final bool showPercentage;
  
  const ConfidenceBar({
    Key? key,
    required this.confidence,
    this.label,
    this.showPercentage = true,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    final confidencePercent = (confidence * 100).toStringAsFixed(1);
    final color = _getConfidenceColor();
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Label and percentage
        if (label != null || showPercentage)
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              if (label != null)
                Expanded(
                  child: Text(
                    label!,
                    style: AppConstants.bodyStyle,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              if (showPercentage)
                Text(
                  '$confidencePercent%',
                  style: AppConstants.bodyStyle.copyWith(
                    fontWeight: FontWeight.bold,
                    color: color,
                  ),
                ),
            ],
          ),
        
        const SizedBox(height: confidence),
        
        // Progress bar
        ClipRRect(
          borderRadius: BorderRadius.circular(10),
          child: SizedBox(
            height: 12,
            child: LinearProgressIndicator(
              value: confidence,
              backgroundColor: Colors.grey[200],
              valueColor: AlwaysStoppedAnimation<Color>(color),
            ),
          ),
        ),
      ],
    );
  }
  
  Color _getConfidenceColor() {
    if (confidence >= 0.9) return AppConstants.successColor;
    if (confidence >= 0.75) return AppConstants.primaryColor;
    if (confidence >= 0.5) return AppConstants.warningColor;
    return AppConstants.errorColor;
  }
}

/// Animated Confidence Bar with animation
class AnimatedConfidenceBar extends StatefulWidget {
  final double confidence;
  final String? label;
  final bool showPercentage;
  final Duration duration;
  
  const AnimatedConfidenceBar({
    Key? key,
    required this.confidence,
    this.label,
    this.showPercentage = true,
    this.duration = const Duration(milliseconds: 800),
  }) : super(key: key);
  
  @override
  State<AnimatedConfidenceBar> createState() => _AnimatedConfidenceBarState();
}

class _AnimatedConfidenceBarState extends State<AnimatedConfidenceBar>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;
  
  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: widget.duration,
      vsync: this,
    );
    
    _animation = Tween<double>(
      begin: 0.0,
      end: widget.confidence,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    ));
    
    _controller.forward();
  }
  
  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return ConfidenceBar(
          confidence: _animation.value,
          label: widget.label,
          showPercentage: widget.showPercentage,
        );
      },
    );
  }
}
