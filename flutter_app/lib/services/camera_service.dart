/// Camera Service
/// Handles camera operations and image capture

import 'dart:io';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';

class CameraService {
  static final CameraService _instance = CameraService._internal();
  factory CameraService() => _instance;
  CameraService._internal();
  
  final ImagePicker _picker = ImagePicker();
  List<CameraDescription>? _cameras;
  CameraController? _controller;
  
  /// Get available cameras
  Future<List<CameraDescription>> getAvailableCameras() async {
    if (_cameras != null) return _cameras!;
    _cameras = await availableCameras();
    return _cameras!;
  }
  
  /// Initialize camera controller
  Future<CameraController> initializeCamera({
    CameraLensDirection direction = CameraLensDirection.back,
  }) async {
    final cameras = await getAvailableCameras();
    
    if (cameras.isEmpty) {
      throw CameraException('NoCameraAvailable', 'No camera found on device');
    }
    
    // Find camera with specified direction
    final camera = cameras.firstWhere(
      (cam) => cam.lensDirection == direction,
      orElse: () => cameras.first,
    );
    
    _controller = CameraController(
      camera,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    
    await _controller!.initialize();
    
    return _controller!;
  }
  
  /// Take picture with camera
  Future<File?> takePicture() async {
    if (_controller == null || !_controller!.value.isInitialized) {
      throw CameraException('NotInitialized', 'Camera not initialized');
    }
    
    try {
      final XFile image = await _controller!.takePicture();
      return File(image.path);
    } catch (e) {
      print('Error taking picture: $e');
      return null;
    }
  }
  
  /// Pick image from gallery
  Future<File?> pickImageFromGallery() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );
      
      if (image == null) return null;
      
      return File(image.path);
    } catch (e) {
      print('Error picking image: $e');
      return null;
    }
  }
  
  /// Pick image from camera (alternative to takePicture)
  Future<File?> pickImageFromCamera() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );
      
      if (image == null) return null;
      
      return File(image.path);
    } catch (e) {
      print('Error picking image from camera: $e');
      return null;
    }
  }
  
  /// Switch camera (front/back)
  Future<void> switchCamera() async {
    if (_controller == null) return;
    
    final cameras = await getAvailableCameras();
    
    if (cameras.length < 2) {
      throw CameraException('OneCameraOnly', 'Device has only one camera');
    }
    
    final currentCameraDirection = _controller!.description.lensDirection;
    final newDirection = currentCameraDirection == CameraLensDirection.back
        ? CameraLensDirection.front
        : CameraLensDirection.back;
    
    await disposeCamera();
    await initializeCamera(direction: newDirection);
  }
  
  /// Set flash mode
  Future<void> setFlashMode(FlashMode mode) async {
    if (_controller == null || !_controller!.value.isInitialized) {
      return;
    }
    
    try {
      await _controller!.setFlashMode(mode);
    } catch (e) {
      print('Error setting flash mode: $e');
    }
  }
  
  /// Get current flash mode
  FlashMode? getFlashMode() {
    return _controller?.value.flashMode;
  }
  
  /// Dispose camera controller
  Future<void> disposeCamera() async {
    await _controller?.dispose();
    _controller = null;
  }
  
  /// Check if camera is initialized
  bool get isInitialized => _controller?.value.isInitialized ?? false;
  
  /// Get camera controller
  CameraController? get controller => _controller;
}
