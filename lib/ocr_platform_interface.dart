import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'ocr_method_channel.dart';

abstract class OcrPlatform extends PlatformInterface {
  /// Constructs a OcrPlatform.
  OcrPlatform() : super(token: _token);

  static final Object _token = Object();

  static OcrPlatform _instance = MethodChannelOcr();

  /// The default instance of [OcrPlatform] to use.
  ///
  /// Defaults to [MethodChannelOcr].
  static OcrPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [OcrPlatform] when
  /// they register themselves.
  static set instance(OcrPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }

  /// Load OCR model from file paths.
  ///
  /// [detParam] - path to detection model param file
  /// [detModel] - path to detection model bin file
  /// [recParam] - path to recognition model param file
  /// [recModel] - path to recognition model bin file
  /// [sizeid] - target size index (0=320, 1=400, 2=480, 3=560, 4=640)
  /// [cpugpu] - 0=CPU, 1=GPU, 2=GPU(Turnip)
  Future<bool> loadModel({
    required String detParam,
    required String detModel,
    required String recParam,
    required String recModel,
    int sizeid = 0,
    int cpugpu = 0,
  }) {
    throw UnimplementedError('loadModel() has not been implemented.');
  }

  Future<bool> openCamera(int facing) {
    throw UnimplementedError('openCamera() has not been implemented.');
  }

  Future<bool> closeCamera() {
    throw UnimplementedError('closeCamera() has not been implemented.');
  }

  /// Toggle camera flash/torch on/off.
  Future<bool> toggleFlash() {
    throw UnimplementedError('toggleFlash() has not been implemented.');
  }

  /// Take a photo and save to the specified path. Returns the saved file path.
  Future<String?> takePhoto(String savePath) {
    throw UnimplementedError('takePhoto() has not been implemented.');
  }

  /// Get the latest OCR recognized text (newline-separated lines).
  Future<String?> getOcrText() {
    throw UnimplementedError('getOcrText() has not been implemented.');
  }

  Future<bool?> setTargetRect(double w, double h) {
    throw UnimplementedError('setTargetRect() has not been implemented.');
  }

  Future<bool?> setPhotoMode(bool isPhoto) {
    throw UnimplementedError('setPhotoMode() has not been implemented.');
  }

  /// Run OCR on a static image file. Returns recognized text.
  Future<String?> ocrFromImage(String imagePath) {
    throw UnimplementedError('ocrFromImage() has not been implemented.');
  }

  Future<String?> ocrFromImageJson(String imagePath) {
    throw UnimplementedError('ocrFromImageJson() has not been implemented.');
  }

  /// Set LED preprocessing parameters.
  /// All values at 0 = General text mode (no preprocessing).
  /// [valueThresh] - HSV Value threshold (recommended: 180-200, 0=disabled)
  /// [rThresh] - R-channel brightness threshold (recommended: 150-180, 0=disabled)
  /// [morphSize] - Morphological opening kernel size (recommended: 3 or 5, 0=disabled)
  Future<bool> setLedParams(int valueThresh, int rThresh, int morphSize) {
    throw UnimplementedError('setLedParams() has not been implemented.');
  }

  /// Set character filter for OCR output.
  ///
  /// [allowedChars] - string of allowed characters (empty string = accept all characters).
  /// Only single-byte ASCII characters are supported for filtering.
  /// See [Ocr.setCharFilter] for full documentation and examples.
  Future<bool> setCharFilter(String allowedChars) {
    throw UnimplementedError('setCharFilter() has not been implemented.');
  }

  /// Launch native image cropper for the given image path. Returns the cropped image path.
  Future<String?> cropImage(String sourcePath) {
    throw UnimplementedError('cropImage() has not been implemented.');
  }
}
