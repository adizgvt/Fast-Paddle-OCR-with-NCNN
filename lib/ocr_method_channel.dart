import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'ocr_platform_interface.dart';

/// An implementation of [OcrPlatform] that uses method channels.
class MethodChannelOcr extends OcrPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('ocr');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>(
      'getPlatformVersion',
    );
    return version;
  }

  @override
  Future<bool> loadModel({
    required String detParam,
    required String detModel,
    required String recParam,
    required String recModel,
    int sizeid = 0,
    int cpugpu = 0,
  }) async {
    final result = await methodChannel.invokeMethod<bool>('loadModel', {
      'detParam': detParam,
      'detModel': detModel,
      'recParam': recParam,
      'recModel': recModel,
      'sizeid': sizeid,
      'cpugpu': cpugpu,
    });
    return result ?? false;
  }

  @override
  Future<bool> openCamera(int facing) async {
    final result = await methodChannel.invokeMethod<bool>('openCamera', {
      'facing': facing,
    });
    return result ?? false;
  }

  @override
  Future<bool> closeCamera() async {
    final result = await methodChannel.invokeMethod<bool>('closeCamera');
    return result ?? false;
  }

  @override
  Future<bool> toggleFlash() async {
    final result = await methodChannel.invokeMethod<bool>('toggleFlash');
    return result ?? false;
  }

  @override
  Future<String?> takePhoto(String savePath) async {
    final result = await methodChannel.invokeMethod<String>('takePhoto', {
      'savePath': savePath,
    });
    return result;
  }

  @override
  Future<String?> getOcrText() async {
    final text = await methodChannel.invokeMethod<String>('getOcrText');
    return text;
  }

  @override
  Future<bool?> setTargetRect(double w, double h) async {
    final success = await methodChannel.invokeMethod<bool>('setTargetRect', {
      'w': w,
      'h': h,
    });
    return success;
  }

  @override
  Future<bool?> setPhotoMode(bool isPhoto) async {
    final success = await methodChannel.invokeMethod<bool>('setPhotoMode', {
      'isPhoto': isPhoto,
    });
    return success;
  }

  @override
  Future<String?> ocrFromImage(String imagePath) async {
    final result = await methodChannel.invokeMethod<String>('ocrFromImage', {
      'imagePath': imagePath,
    });
    return result;
  }

  @override
  Future<String?> ocrFromImageJson(String imagePath) async {
    final result = await methodChannel.invokeMethod<String>('ocrFromImageJson', {
      'imagePath': imagePath,
    });
    return result;
  }

  @override
  Future<bool> setLedParams(int valueThresh, int rThresh, int morphSize) async {
    final result = await methodChannel.invokeMethod<bool>('setLedParams', {
      'valueThresh': valueThresh,
      'rThresh': rThresh,
      'morphSize': morphSize,
    });
    return result ?? false;
  }

  @override
  Future<bool> setCharFilter(String allowedChars) async {
    final result = await methodChannel.invokeMethod<bool>('setCharFilter', {
      'allowedChars': allowedChars,
    });
    return result ?? false;
  }

  @override
  Future<String?> cropImage(String sourcePath) async {
    final result = await methodChannel.invokeMethod<String>('cropImage', {
      'sourcePath': sourcePath,
    });
    return result;
  }
}
