import 'ocr_platform_interface.dart';

class Ocr {
  Future<String?> getPlatformVersion() {
    return OcrPlatform.instance.getPlatformVersion();
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
    return OcrPlatform.instance.loadModel(
      detParam: detParam,
      detModel: detModel,
      recParam: recParam,
      recModel: recModel,
      sizeid: sizeid,
      cpugpu: cpugpu,
    );
  }

  Future<bool> openCamera(int facing) {
    return OcrPlatform.instance.openCamera(facing);
  }

  Future<bool> closeCamera() {
    return OcrPlatform.instance.closeCamera();
  }

  /// Toggle camera flash/torch on/off.
  Future<bool> toggleFlash() {
    return OcrPlatform.instance.toggleFlash();
  }

  /// Take a photo and save to [savePath]. Returns the saved file path.
  Future<String?> takePhoto(String savePath) {
    return OcrPlatform.instance.takePhoto(savePath);
  }

  /// Get the latest OCR recognized text (newline-separated lines).
  Future<String?> getOcrText() {
    return OcrPlatform.instance.getOcrText();
  }

  /// Sets the normalized crop area for auto-cropping in Photo mode.
  /// [w] and [h] should be between 0.0 and 1.0.
  Future<bool?> setTargetRect(double w, double h) {
    return OcrPlatform.instance.setTargetRect(w, h);
  }

  /// Sets whether the native camera is in Photo mode.
  /// When true, continuous OCR is paused to save power and only runs on capture.
  Future<bool?> setPhotoMode(bool isPhoto) {
    return OcrPlatform.instance.setPhotoMode(isPhoto);
  }

  /// Run OCR on a static image file. Returns recognized text.
  Future<String?> ocrFromImage(String imagePath) {
    return OcrPlatform.instance.ocrFromImage(imagePath);
  }

  Future<String?> ocrFromImageJson(String imagePath) {
    return OcrPlatform.instance.ocrFromImageJson(imagePath);
  }

  /// Set LED display preprocessing parameters.
  /// All values at 0 = General text mode (no preprocessing, reads any text).
  /// [valueThresh] - HSV Value threshold (0=off, 180-200 recommended for LED ghosting removal)
  /// [rThresh] - R-channel threshold (0=off, 150-180 recommended)
  /// [morphSize] - Morphological kernel size (0=off, 3 or 5 recommended)
  Future<bool> setLedParams(int valueThresh, int rThresh, int morphSize) {
    return OcrPlatform.instance.setLedParams(valueThresh, rThresh, morphSize);
  }

  /// Set character filter for OCR output.
  ///
  /// Controls which characters are accepted in the final OCR result.
  /// Any character recognized by the model that is NOT in [allowedChars]
  /// will be silently discarded from the output.
  ///
  /// [allowedChars] - A string containing all characters to accept.
  ///   - Empty string `''` = accept ALL characters (default behavior).
  ///   - Only single-byte (ASCII) characters are supported for filtering.
  ///   - Multi-byte characters (Chinese, Japanese, etc.) are always discarded
  ///     when a filter is active.
  ///
  /// This setting persists until changed again. It works independently from
  /// [setLedParams] — you can use char filter with or without LED preprocessing.
  ///
  /// Usage examples:
  /// ```dart
  /// // Default: accept all characters (no filtering)
  /// await ocr.setCharFilter('');
  ///
  /// // Digits + dot only (for LED displays, scales, meters)
  /// await ocr.setCharFilter('0123456789.');
  ///
  /// // Digits + dot + minus (for negative values)
  /// await ocr.setCharFilter('0123456789.-');
  ///
  /// // Uppercase alphanumeric (for serial numbers, license plates)
  /// await ocr.setCharFilter('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ');
  ///
  /// // Full alphanumeric + dot (general filtered)
  /// await ocr.setCharFilter('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.');
  ///
  /// // Hex characters only
  /// await ocr.setCharFilter('0123456789ABCDEFabcdef');
  /// ```
  Future<bool> setCharFilter(String allowedChars) {
    return OcrPlatform.instance.setCharFilter(allowedChars);
  }

  /// Launch native image cropper for the given image path. Returns the cropped image path.
  Future<String?> cropImage(String sourcePath) {
    return OcrPlatform.instance.cropImage(sourcePath);
  }
}
