package com.iweka.ocr

import android.view.Surface

class PPOCRv5Ncnn {
    external fun loadModel(
        detParam: String, detModel: String,
        recParam: String, recModel: String,
        sizeid: Int, cpugpu: Int
    ): Boolean
    external fun openCamera(facing: Int): Boolean
    external fun closeCamera(): Boolean
    external fun setOutputWindow(surface: Surface): Boolean
    external fun clearOutputWindow(): Boolean
    external fun toggleFlash(): Boolean
    external fun takePhoto(savePath: String): String
    external fun getOcrText(): String
    external fun ocrFromImage(imagePath: String): String
    external fun ocrFromImageJson(imagePath: String): String
    external fun setTargetRect(normW: Float, normH: Float): Boolean
    external fun setPhotoMode(isPhoto: Boolean): Boolean
    external fun setLedParams(valueThresh: Int, rThresh: Int, morphSize: Int): Boolean
    external fun setCharFilter(allowedChars: String): Boolean

    companion object {
        init {
            System.loadLibrary("ppocrv5ncnn")
        }
    }
}
