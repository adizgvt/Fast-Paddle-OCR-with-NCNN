package com.iweka.ocr

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import io.flutter.plugin.common.PluginRegistry
import com.yalantis.ucrop.UCrop
import java.io.File

/** OcrPlugin */
class OcrPlugin: FlutterPlugin, MethodCallHandler, ActivityAware, PluginRegistry.ActivityResultListener {
  private lateinit var channel : MethodChannel
  private lateinit var context: Context
  private var activity: Activity? = null
  private val ppocrv5ncnn = PPOCRv5Ncnn()
  private val ocrExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()
  
  private var pendingCropResult: Result? = null

  override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
    context = flutterPluginBinding.applicationContext
    channel = MethodChannel(flutterPluginBinding.binaryMessenger, "ocr")
    channel.setMethodCallHandler(this)

    flutterPluginBinding.platformViewRegistry.registerViewFactory(
        "ocr_camera_view", OcrCameraViewFactory(ppocrv5ncnn)
    )
  }

  override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
    channel.setMethodCallHandler(null)
    ocrExecutor.shutdownNow()
  }

  override fun onAttachedToActivity(binding: ActivityPluginBinding) {
    activity = binding.activity
    binding.addActivityResultListener(this)
  }

  override fun onDetachedFromActivityForConfigChanges() {
    activity = null
  }

  override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {
    activity = binding.activity
    binding.addActivityResultListener(this)
  }

  override fun onDetachedFromActivity() {
    activity = null
  }

  override fun onMethodCall(call: MethodCall, result: Result) {
    when (call.method) {
        "getPlatformVersion" -> {
            result.success("Android ${android.os.Build.VERSION.RELEASE}")
        }
        "loadModel" -> {
            val detParam = call.argument<String>("detParam") ?: ""
            val detModel = call.argument<String>("detModel") ?: ""
            val recParam = call.argument<String>("recParam") ?: ""
            val recModel = call.argument<String>("recModel") ?: ""
            val sizeid = call.argument<Int>("sizeid") ?: 0
            val cpugpu = call.argument<Int>("cpugpu") ?: 0
            val success = ppocrv5ncnn.loadModel(detParam, detModel, recParam, recModel, sizeid, cpugpu)
            result.success(success)
        }
        "openCamera" -> {
            val facing = call.argument<Int>("facing") ?: 0
            val success = ppocrv5ncnn.openCamera(facing)
            result.success(success)
        }
        "closeCamera" -> {
            val success = ppocrv5ncnn.closeCamera()
            result.success(success)
        }
        "toggleFlash" -> {
            val success = ppocrv5ncnn.toggleFlash()
            result.success(success)
        }
        "takePhoto" -> {
            val savePath = call.argument<String>("savePath") ?: ""
            if (savePath.isEmpty()) {
                result.error("INVALID_PATH", "savePath is required", null)
                return
            }
            ocrExecutor.submit {
                try {
                    val photoPath = ppocrv5ncnn.takePhoto(savePath)
                    android.os.Handler(android.os.Looper.getMainLooper()).post {
                        result.success(photoPath)
                    }
                } catch (e: Exception) {
                    android.os.Handler(android.os.Looper.getMainLooper()).post {
                        result.error("PHOTO_ERROR", e.message, null)
                    }
                }
            }
        }
        "getOcrText" -> {
            val text = ppocrv5ncnn.getOcrText()
            result.success(text)
        }
        "setTargetRect" -> {
          val w = call.argument<Double>("w")?.toFloat() ?: 0f
          val h = call.argument<Double>("h")?.toFloat() ?: 0f
          val success = ppocrv5ncnn.setTargetRect(w, h)
          result.success(success)
        }
        "setPhotoMode" -> {
          val isPhoto = call.argument<Boolean>("isPhoto") ?: false
          val success = ppocrv5ncnn.setPhotoMode(isPhoto)
          result.success(success)
        }
        "ocrFromImage" -> {
            val imagePath = call.argument<String>("imagePath") ?: ""
            ocrExecutor.submit {
                try {
                    val text = ppocrv5ncnn.ocrFromImage(imagePath)
                    android.os.Handler(android.os.Looper.getMainLooper()).post {
                        result.success(text)
                    }
                } catch (e: Exception) {
                    android.os.Handler(android.os.Looper.getMainLooper()).post {
                        result.error("OCR_ERROR", e.message, null)
                    }
                }
            }
        }
        "ocrFromImageJson" -> {
            val imagePath = call.argument<String>("imagePath") ?: ""
            ocrExecutor.submit {
                try {
                    val json = ppocrv5ncnn.ocrFromImageJson(imagePath)
                    android.os.Handler(android.os.Looper.getMainLooper()).post {
                        result.success(json)
                    }
                } catch (e: Exception) {
                    android.os.Handler(android.os.Looper.getMainLooper()).post {
                        result.error("OCR_ERROR", e.message, null)
                    }
                }
            }
        }
        "setLedParams" -> {
            val valueThresh = call.argument<Int>("valueThresh") ?: 0
            val rThresh = call.argument<Int>("rThresh") ?: 0
            val morphSize = call.argument<Int>("morphSize") ?: 0
            val success = ppocrv5ncnn.setLedParams(valueThresh, rThresh, morphSize)
            result.success(success)
        }
        "setCharFilter" -> {
            val allowedChars = call.argument<String>("allowedChars") ?: ""
            val success = ppocrv5ncnn.setCharFilter(allowedChars)
            result.success(success)
        }
        "cropImage" -> {
            val sourcePath = call.argument<String>("sourcePath") ?: ""
            if (sourcePath.isEmpty() || activity == null) {
                result.error("INVALID_CROP", "Source path is empty or Activity is null", null)
                return
            }
            
            pendingCropResult = result
            
            val sourceUri = Uri.fromFile(File(sourcePath))
            val destFile = File(context.cacheDir, "cropped_${System.currentTimeMillis()}.jpg")
            val destUri = Uri.fromFile(destFile)
            
            val uCrop = UCrop.of(sourceUri, destUri)
                .withOptions(UCrop.Options().apply {
                    setCompressionQuality(90)
                    setFreeStyleCropEnabled(true)
                })
            
            uCrop.start(activity!!)
        }
        else -> {
            result.notImplemented()
        }
    }
  }

  override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?): Boolean {
      if (requestCode == UCrop.REQUEST_CROP) {
          if (resultCode == Activity.RESULT_OK && data != null) {
              val resultUri = UCrop.getOutput(data)
              pendingCropResult?.success(resultUri?.path)
          } else if (resultCode == UCrop.RESULT_ERROR && data != null) {
              val cropError = UCrop.getError(data)
              pendingCropResult?.error("CROP_ERROR", cropError?.message, null)
          } else {
              pendingCropResult?.success(null) // Cancelled
          }
          pendingCropResult = null
          return true
      }
      return false
  }
}
