// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "ppocrv5.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ppocrv5_dict.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include <atomic>
#include <sys/time.h>

static std::atomic<bool> is_ocr_processing(false);
static std::atomic<long long> last_ocr_timestamp(0LL);
static const long long OCR_THROTTLE_MS = 400LL;

static long long get_current_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000LL + tv.tv_usec / 1000;
}

static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

static PPOCRv5* g_ppocrv5 = 0;
static ncnn::Mutex lock;

static float compute_iou(const cv::RotatedRect& a, const cv::RotatedRect& b) {
    cv::Rect rect_a = a.boundingRect();
    cv::Rect rect_b = b.boundingRect();
    
    int x1 = std::max(rect_a.x, rect_b.x);
    int y1 = std::max(rect_a.y, rect_b.y);
    int x2 = std::min(rect_a.x + rect_a.width, rect_b.x + rect_b.width);
    int y2 = std::min(rect_a.y + rect_a.height, rect_b.y + rect_b.height);
    
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    
    float intersection = (x2 - x1) * (y2 - y1);
    float area_a = rect_a.width * rect_a.height;
    float area_b = rect_b.width * rect_b.height;
    
    return intersection / (area_a + area_b - intersection);
}

class MyNdkCamera;
static void* onDetProcess(void* args);
static void* onRecProcess(void* args);

class MyNdkCamera : public NdkCameraWindow
{
public:
    MyNdkCamera();
    virtual ~MyNdkCamera();

    virtual void on_image_render(cv::Mat& rgb) const;

    void det_thread_loop();
    void rec_thread_loop();

private:
    mutable cv::Mat det_latest_rgb;
    mutable ncnn::Mutex det_lock;
    mutable ncnn::ConditionVariable det_condition;
    bool det_exiting;
    ncnn::Thread* det_thread;

    mutable cv::Mat rec_latest_rgb;
    mutable std::vector<Object> rec_pending_objects;
    mutable ncnn::Mutex rec_lock;
    mutable ncnn::ConditionVariable rec_condition;
    bool rec_exiting;
    ncnn::Thread* rec_thread;

    mutable std::vector<Object> latest_boxes;
    mutable ncnn::Mutex latest_boxes_lock;

    mutable std::vector<Object> latest_texts;
    mutable ncnn::Mutex latest_texts_lock;
};

static void* onDetProcess(void* args)
{
    MyNdkCamera* self = (MyNdkCamera*)args;
    self->det_thread_loop();
    return 0;
}

static void* onRecProcess(void* args)
{
    MyNdkCamera* self = (MyNdkCamera*)args;
    self->rec_thread_loop();
    return 0;
}

MyNdkCamera::MyNdkCamera()
{
    det_exiting = false;
    rec_exiting = false;
    det_thread = new ncnn::Thread(onDetProcess, (void*)this);
    rec_thread = new ncnn::Thread(onRecProcess, (void*)this);
}

MyNdkCamera::~MyNdkCamera()
{
    det_exiting = true;
    det_condition.signal();
    det_thread->join();
    delete det_thread;

    rec_exiting = true;
    rec_condition.signal();
    rec_thread->join();
    delete rec_thread;
}

void MyNdkCamera::det_thread_loop()
{
    while (!det_exiting)
    {
        cv::Mat rgb;
        {
            ncnn::MutexLockGuard g(det_lock);
            while (det_latest_rgb.empty() && !det_exiting)
            {
                det_condition.wait(det_lock);
            }
            if (det_exiting) break;

            rgb = det_latest_rgb;
            det_latest_rgb = cv::Mat();
        }

        if (rgb.empty()) continue;

        std::vector<Object> objects;
        {
            ncnn::MutexLockGuard g(lock);
            if (g_ppocrv5)
            {
                cv::Mat det_rgb = rgb;
                int crop_x = 0;
                int crop_y = 0;
                if (target_norm_w > 0.0f && target_norm_h > 0.0f)
                {
                    int crop_w = (int)(rgb.cols * target_norm_w);
                    int crop_h = (int)(rgb.rows * target_norm_h);
                    if (crop_w > 0 && crop_w <= rgb.cols && crop_h > 0 && crop_h <= rgb.rows)
                    {
                        crop_x = (rgb.cols - crop_w) / 2;
                        crop_y = (int)(rgb.rows * 0.375f) - crop_h / 2;
                        if (crop_y < 0) crop_y = 0;
                        if (crop_y + crop_h > rgb.rows) crop_y = rgb.rows - crop_h;
                        
                        cv::Rect crop_region(crop_x, crop_y, crop_w, crop_h);
                        det_rgb = rgb(crop_region).clone();
                    }
                }

                g_ppocrv5->detect(det_rgb, objects);

                // shift coordinates back to original rgb space
                if (crop_x > 0 || crop_y > 0)
                {
                    for (size_t i = 0; i < objects.size(); i++)
                    {
                        objects[i].rrect.center.x += crop_x;
                        objects[i].rrect.center.y += crop_y;
                    }
                }
            }
        }

        {
            ncnn::MutexLockGuard g(latest_boxes_lock);
            latest_boxes = objects;
        }

        if (!is_photo_mode && g_ppocrv5)
        {
            ncnn::MutexLockGuard g(rec_lock);
            if (rec_latest_rgb.empty())
            {
                rec_latest_rgb = rgb; // passed to Rec thread
                rec_pending_objects = objects;
                rec_condition.signal();
            }
        }

        ncnn::sleep(10);
    }
}

void MyNdkCamera::rec_thread_loop()
{
    while (!rec_exiting)
    {
        cv::Mat rgb;
        std::vector<Object> objects;
        {
            ncnn::MutexLockGuard g(rec_lock);
            while (rec_latest_rgb.empty() && !rec_exiting)
            {
                rec_condition.wait(rec_lock);
            }
            if (rec_exiting) break;

            rgb = rec_latest_rgb;
            objects = rec_pending_objects;
            rec_latest_rgb = cv::Mat();
            rec_pending_objects.clear();
        }

        if (rgb.empty() || objects.empty()) continue;

        std::string all_text;
        {
            ncnn::MutexLockGuard g(lock);
            if (g_ppocrv5)
            {
                const std::string& char_filter = g_ppocrv5->get_char_filter();
                for (size_t i = 0; i < objects.size(); i++)
                {
                    g_ppocrv5->recognize(rgb, objects[i]);

                    std::string line_text;
                    for (size_t j = 0; j < objects[i].text.size(); j++)
                    {
                        const Character& ch = objects[i].text[j];
                        if (ch.id >= 0 && ch.id < character_dict_size)
                        {
                            std::string c_str = character_dict[ch.id];
                            if (char_filter.empty())
                            {
                                // No filter: accept all characters
                                line_text += c_str;
                            }
                            else
                            {
                                // Filter: only accept single-byte chars that are in the filter string
                                if (c_str.length() == 1 && char_filter.find(c_str[0]) != std::string::npos)
                                {
                                    line_text += c_str;
                                }
                            }
                        }
                    }
                    if (!line_text.empty())
                    {
                        if (!all_text.empty()) all_text += "\n";
                        all_text += line_text;
                    }
                }
            }
        }

        const_cast<MyNdkCamera*>(this)->set_ocr_text(all_text);

        {
            ncnn::MutexLockGuard g(latest_texts_lock);
            latest_texts = objects;
        }

        ncnn::sleep(300);
    }
}

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    // 1. Feed latest frame to Det thread
    {
        ncnn::MutexLockGuard g(det_lock);
        if (det_latest_rgb.empty()) {
            det_latest_rgb = rgb.clone();
            det_condition.signal();
        }
    }

    // 2. Fetch latest objects
    std::vector<Object> current_objects;
    {
        ncnn::MutexLockGuard g(latest_boxes_lock);
        current_objects = latest_boxes;
    }

    if (!is_photo_mode)
    {
        std::vector<Object> current_texts;
        {
            ncnn::MutexLockGuard g(latest_texts_lock);
            current_texts = latest_texts;
        }

        // 3. IoU Tracker: Gabungkan latest_boxes + latest_texts
        for (size_t i = 0; i < current_objects.size(); i++)
        {
            float max_iou = 0.0f;
            int best_idx = -1;
            for (size_t j = 0; j < current_texts.size(); j++)
            {
                float iou = compute_iou(current_objects[i].rrect, current_texts[j].rrect);
                if (iou > max_iou) {
                    max_iou = iou;
                    best_idx = j;
                }
            }
            if (best_idx != -1 && max_iou > 0.3f) 
            {
                current_objects[i].text = current_texts[best_idx].text;
            }
        }
    }

    // 4. Draw objects and handle screen capture (Render Thread)
    {
        ncnn::MutexLockGuard g(lock);

        if (g_ppocrv5)
        {
            if (is_photo_mode) 
            {
                // Draw bounding boxes only for preview
                static const cv::Scalar bbox_color(80, 175, 76); // green
                for (size_t i = 0; i < current_objects.size(); i++)
                {
                    cv::Point2f corners[4];
                    current_objects[i].rrect.points(corners);
                    cv::line(rgb, corners[0], corners[1], bbox_color, 2);
                    cv::line(rgb, corners[1], corners[2], bbox_color, 2);
                    cv::line(rgb, corners[2], corners[3], bbox_color, 2);
                    cv::line(rgb, corners[3], corners[0], bbox_color, 2);
                }

                // Handle exact photo capture request
                ncnn::MutexLockGuard cg(capture_lock);
                if (capture_requested)
                {
                    cv::Mat full_frame = full_capture_rgb.empty() ? rgb.clone() : full_capture_rgb.clone();
                    
                    cv::Mat bgr_orig;
                    cv::cvtColor(full_frame, bgr_orig, cv::COLOR_RGB2BGR);
                    cv::imwrite(capture_save_path, bgr_orig);

                    std::string cropped_path = "";

                    if (target_norm_w > 0.0f && target_norm_h > 0.0f) 
                    {
                        int crop_w = (int)(rgb.cols * target_norm_w);
                        int crop_h = (int)(rgb.rows * target_norm_h);
                        
                        if (crop_w > 0 && crop_w <= rgb.cols && crop_h > 0 && crop_h <= rgb.rows) 
                        {
                            int crop_x = (rgb.cols - crop_w) / 2;
                            int crop_y = (int)(rgb.rows * 0.375f) - crop_h / 2;
                            if (crop_y < 0) crop_y = 0;
                            if (crop_y + crop_h > rgb.rows) crop_y = rgb.rows - crop_h;
                            cv::Rect crop_region(crop_x, crop_y, crop_w, crop_h);
                            cv::Mat crop_rgb = rgb(crop_region).clone();

                            // Force synchronous high-quality OCR for the final captured cropped image
                            std::vector<Object> capture_objects;
                            g_ppocrv5->detect_and_recognize(crop_rgb, capture_objects);

                            const std::string& char_filter = g_ppocrv5->get_char_filter();
                            std::string all_text;
                            for (size_t i = 0; i < capture_objects.size(); i++)
                            {
                                std::string line_text;
                                for (size_t j = 0; j < capture_objects[i].text.size(); j++)
                                {
                                    const Character& ch = capture_objects[i].text[j];
                                    if (ch.id >= 0 && ch.id < character_dict_size)
                                    {
                                        std::string c_str = character_dict[ch.id];
                                        if (char_filter.empty())
                                        {
                                            line_text += c_str;
                                        }
                                        else
                                        {
                                            if (c_str.length() == 1 && char_filter.find(c_str[0]) != std::string::npos)
                                            {
                                                line_text += c_str;
                                            }
                                        }
                                    }
                                }
                                if (!line_text.empty())
                                {
                                    if (!all_text.empty()) all_text += "\n";
                                    all_text += line_text;
                                }
                            }
                            const_cast<MyNdkCamera*>(this)->set_ocr_text(all_text);

                            int roi_offset_x = (full_frame.cols - rgb.cols) / 2;
                            int roi_offset_y = (full_frame.rows - rgb.rows) / 2;
                            for (size_t i = 0; i < capture_objects.size(); i++)
                            {
                                capture_objects[i].rrect.center.x += crop_x + roi_offset_x;
                                capture_objects[i].rrect.center.y += crop_y + roi_offset_y;
                            }

                            g_ppocrv5->draw(full_frame, capture_objects);

                            cv::Mat bgr_full;
                            cv::cvtColor(full_frame, bgr_full, cv::COLOR_RGB2BGR);
                            
                            size_t dot_pos = capture_save_path.find_last_of('.');
                            if (dot_pos != std::string::npos) {
                                cropped_path = capture_save_path.substr(0, dot_pos) + "_crop" + capture_save_path.substr(dot_pos);
                            } else {
                                cropped_path = capture_save_path + "_crop.jpg";
                            }
                            cv::imwrite(cropped_path, bgr_full);
                        }
                    }

                    if (!cropped_path.empty()) {
                        captured_photo_path = capture_save_path + "|" + cropped_path;
                    } else {
                        captured_photo_path = capture_save_path;
                    }
                    
                    const_cast<MyNdkCamera*>(this)->full_capture_rgb = cv::Mat();
                    capture_requested = false;
                }
            }
            else 
            {
                // Realtime Mode
                // Simply draw the most recently processed OCR objects from background thread
                g_ppocrv5->draw(rgb, current_objects);

                ncnn::MutexLockGuard cg(capture_lock);
                if (capture_requested)
                {
                    cv::Mat bgr_orig;
                    cv::cvtColor(rgb, bgr_orig, cv::COLOR_RGB2BGR);
                    cv::imwrite(capture_save_path, bgr_orig);

                    std::string cropped_path = "";
                    if (target_norm_w > 0.0f && target_norm_h > 0.0f) 
                    {
                        int crop_w = (int)(rgb.cols * target_norm_w);
                        int crop_h = (int)(rgb.rows * target_norm_h);
                        if (crop_w > 0 && crop_w <= rgb.cols && crop_h > 0 && crop_h <= rgb.rows) 
                        {
                            int crop_x = (rgb.cols - crop_w) / 2;
                            int crop_y = (int)(rgb.rows * 0.375f) - crop_h / 2;
                            if (crop_y < 0) crop_y = 0;
                            if (crop_y + crop_h > rgb.rows) crop_y = rgb.rows - crop_h;
                            cv::Rect crop_region(crop_x, crop_y, crop_w, crop_h);
                            cv::Mat crop_rgb = rgb(crop_region).clone();

                            cv::Mat bgr_crop;
                            cv::cvtColor(crop_rgb, bgr_crop, cv::COLOR_RGB2BGR);

                            size_t dot_pos = capture_save_path.find_last_of('.');
                            if (dot_pos != std::string::npos) {
                                cropped_path = capture_save_path.substr(0, dot_pos) + "_crop" + capture_save_path.substr(dot_pos);
                            } else {
                                cropped_path = capture_save_path + "_crop.jpg";
                            }
                            cv::imwrite(cropped_path, bgr_crop);
                        }
                    }

                    if (!cropped_path.empty()) {
                        captured_photo_path = capture_save_path + "|" + cropped_path;
                    } else {
                        captured_photo_path = capture_save_path;
                    }
                    capture_requested = false;
                }
            }
        }
        else
        {
            draw_unsupported(rgb);
        }
    }

#ifndef NDEBUG
    draw_fps(rgb);
#endif
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_ppocrv5;
        g_ppocrv5 = 0;
    }

    ncnn::destroy_gpu_instance();

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(String detParam, String detModel, String recParam, String recModel, int sizeid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_iweka_ocr_PPOCRv5Ncnn_loadModel(JNIEnv* env, jobject thiz, jstring detParam, jstring detModel, jstring recParam, jstring recModel, jint sizeid, jint cpugpu)
{
    if (sizeid < 0 || sizeid > 4 || cpugpu < 0 || cpugpu > 2)
    {
        return JNI_FALSE;
    }

    const char* det_parampath = env->GetStringUTFChars(detParam, nullptr);
    const char* det_modelpath = env->GetStringUTFChars(detModel, nullptr);
    const char* rec_parampath = env->GetStringUTFChars(recParam, nullptr);
    const char* rec_modelpath = env->GetStringUTFChars(recModel, nullptr);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel from files: %s", det_parampath);

    const int sizetypes[5] =
    {
        320,
        400,
        480,
        560,
        640
    };

    // Determine fp16 usage based on model type (heuristic: if path contains "server", disable fp16)
    std::string det_param_str(det_parampath);
    bool use_fp16 = (det_param_str.find("server") == std::string::npos);
    bool use_gpu = (int)cpugpu == 1;
    bool use_turnip = (int)cpugpu == 2;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        delete g_ppocrv5;
        g_ppocrv5 = 0;

        ncnn::destroy_gpu_instance();

        if (use_turnip)
        {
            ncnn::create_gpu_instance("libvulkan_freedreno.so");
        }
        else if (use_gpu)
        {
            ncnn::create_gpu_instance();
        }

        g_ppocrv5 = new PPOCRv5;
        g_ppocrv5->load(det_parampath, det_modelpath, rec_parampath, rec_modelpath, use_fp16, use_gpu || use_turnip);
        g_ppocrv5->set_target_size(sizetypes[(int)sizeid]);
    }

    env->ReleaseStringUTFChars(detParam, det_parampath);
    env->ReleaseStringUTFChars(detModel, det_modelpath);
    env->ReleaseStringUTFChars(recParam, rec_parampath);
    env->ReleaseStringUTFChars(recModel, rec_modelpath);

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_iweka_ocr_PPOCRv5Ncnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_iweka_ocr_PPOCRv5Ncnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_iweka_ocr_PPOCRv5Ncnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

// public native boolean clearOutputWindow();
JNIEXPORT jboolean JNICALL Java_com_iweka_ocr_PPOCRv5Ncnn_clearOutputWindow(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "clearOutputWindow");

    if (g_camera)
    {
        g_camera->clear_window();
    }

    return JNI_TRUE;
}

// public native boolean toggleFlash();
JNIEXPORT jboolean JNICALL Java_com_iweka_ocr_PPOCRv5Ncnn_toggleFlash(JNIEnv* env, jobject thiz)
{
    if (!g_camera)
        return JNI_FALSE;

    int ret = g_camera->toggle_flash();
    return ret == 0 ? JNI_TRUE : JNI_FALSE;
}

// public native String takePhoto(String savePath);
JNIEXPORT jstring JNICALL Java_com_iweka_ocr_PPOCRv5Ncnn_takePhoto(JNIEnv* env, jobject thiz, jstring savePath)
{
    if (!g_camera)
        return env->NewStringUTF("");

    const char* save_path = env->GetStringUTFChars(savePath, nullptr);
    g_camera->request_capture(save_path);
    env->ReleaseStringUTFChars(savePath, save_path);

    // wait for capture to complete (up to 2 seconds)
    std::string photo_path;
    for (int i = 0; i < 40; i++)
    {
        ncnn::sleep(50); // 50ms
        photo_path = g_camera->get_captured_photo_path();
        if (!photo_path.empty())
            break;
    }

    return env->NewStringUTF(photo_path.c_str());
}

// public native String getOcrText();
JNIEXPORT jstring JNICALL Java_com_iweka_ocr_PPOCRv5Ncnn_getOcrText(JNIEnv* env, jobject thiz)
{
    if (!g_camera)
        return env->NewStringUTF("");

    std::string text = g_camera->get_ocr_text();
    return env->NewStringUTF(text.c_str());
}

// public native boolean setTargetRect(float norm_w, float norm_h);
JNIEXPORT jboolean JNICALL Java_com_iweka_ocr_PPOCRv5Ncnn_setTargetRect(JNIEnv* env, jobject thiz, jfloat norm_w, jfloat norm_h)
{
    if (g_camera)
    {
        g_camera->set_target_rect(norm_w, norm_h);
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// public native boolean setPhotoMode(boolean isPhoto);
JNIEXPORT jboolean JNICALL Java_com_iweka_ocr_PPOCRv5Ncnn_setPhotoMode(JNIEnv* env, jobject thiz, jboolean isPhoto)
{
    if (g_camera)
    {
        g_camera->set_photo_mode(isPhoto == JNI_TRUE);
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// public native boolean setLedParams(int valueThresh, int rThresh, int morphSize);
// valueThresh: HSV Value threshold (0=disabled/general mode, 180-200 recommended for LED)
// rThresh: R-channel threshold (0=disabled, 150-180 recommended)
// morphSize: morphological kernel size (0=disabled, 3 or 5 recommended)
JNIEXPORT jboolean JNICALL Java_com_iweka_ocr_PPOCRv5Ncnn_setLedParams(JNIEnv* env, jobject thiz, jint valueThresh, jint rThresh, jint morphSize)
{
    ncnn::MutexLockGuard g(lock);
    if (g_ppocrv5)
    {
        g_ppocrv5->set_led_params((int)valueThresh, (int)rThresh, (int)morphSize);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setLedParams: value=%d, r=%d, morph=%d", (int)valueThresh, (int)rThresh, (int)morphSize);
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// public native boolean setCharFilter(String allowedChars);
// allowedChars: string of allowed characters (empty string = accept all)
// Example: "0123456789." for digits only, "0123456789ABCDEFabcdef." for hex
JNIEXPORT jboolean JNICALL Java_com_iweka_ocr_PPOCRv5Ncnn_setCharFilter(JNIEnv* env, jobject thiz, jstring allowedChars)
{
    const char* chars = env->GetStringUTFChars(allowedChars, nullptr);
    std::string filter_str(chars);
    env->ReleaseStringUTFChars(allowedChars, chars);

    ncnn::MutexLockGuard g(lock);
    if (g_ppocrv5)
    {
        g_ppocrv5->set_char_filter(filter_str);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setCharFilter: '%s' (length=%d)", filter_str.c_str(), (int)filter_str.length());
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// public native String ocrFromImage(String imagePath);
JNIEXPORT jstring JNICALL Java_com_iweka_ocr_PPOCRv5Ncnn_ocrFromImage(JNIEnv* env, jobject thiz, jstring imagePath)
{
    const char* image_path = env->GetStringUTFChars(imagePath, nullptr);
    std::string image_path_str(image_path);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "ocrFromImage: %s", image_path);

    cv::Mat bgr = cv::imread(image_path);
    env->ReleaseStringUTFChars(imagePath, image_path);

    if (bgr.empty())
    {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "ocrFromImage: failed to load image");
        return env->NewStringUTF("");
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    ncnn::MutexLockGuard g(lock);

    if (!g_ppocrv5)
    {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "ocrFromImage: model not loaded");
        return env->NewStringUTF("");
    }

    std::vector<Object> objects;
    g_ppocrv5->detect_and_recognize(rgb, objects);

    // extract OCR text - filter depends on char_filter setting
    const std::string& char_filter = g_ppocrv5->get_char_filter();
    std::string all_text;
    for (size_t i = 0; i < objects.size(); i++)
    {
        std::string line_text;
        for (size_t j = 0; j < objects[i].text.size(); j++)
        {
            const Character& ch = objects[i].text[j];
            if (ch.id >= character_dict_size)
            {
                if (!line_text.empty() && line_text.back() != ' ')
                    line_text += " "; // space token — keep it
            }
            else if (ch.id >= 0 && ch.id < character_dict_size)
            {
                std::string c_str = character_dict[ch.id];
                if (char_filter.empty())
                {
                    line_text += c_str;
                }
                else
                {
                    if (c_str.length() == 1 && char_filter.find(c_str[0]) != std::string::npos)
                    {
                        line_text += c_str;
                    }
                }
            }
        }
        if (!line_text.empty())
        {
            if (!all_text.empty())
                all_text += "\n";
            all_text += line_text;
        }
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "ocrFromImage: found %d objects, text length %d",
                        (int)objects.size(), (int)all_text.length());

    // Draw bounding boxes + text labels on the image (same as realtime)
    std::string bbox_path;
    if (!objects.empty())
    {
        g_ppocrv5->draw(rgb, objects);

        // Save annotated image as separate _bbox file (don't overwrite original)
        size_t dot_pos = image_path_str.find_last_of('.');
        if (dot_pos != std::string::npos)
        {
            bbox_path = image_path_str.substr(0, dot_pos) + "_bbox" + image_path_str.substr(dot_pos);
        }
        else
        {
            bbox_path = image_path_str + "_bbox.jpg";
        }

        cv::Mat bgr_out;
        cv::cvtColor(rgb, bgr_out, cv::COLOR_RGB2BGR);
        cv::imwrite(bbox_path, bgr_out);

        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "ocrFromImage: bbox saved to %s", bbox_path.c_str());
    }

    return env->NewStringUTF(all_text.c_str());
}

// public native String ocrFromImageJson(String imagePath);
// Like ocrFromImage, but returns JSON with per-line box + confidence:
//   [{"text":"...","prob":0.98,"points":[[x,y],[x,y],[x,y],[x,y]]}]
JNIEXPORT jstring JNICALL Java_com_iweka_ocr_PPOCRv5Ncnn_ocrFromImageJson(JNIEnv* env, jobject thiz, jstring imagePath)
{
    const char* image_path = env->GetStringUTFChars(imagePath, nullptr);
    cv::Mat bgr = cv::imread(image_path);
    env->ReleaseStringUTFChars(imagePath, image_path);
    if (bgr.empty())
        return env->NewStringUTF("[]");

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    ncnn::MutexLockGuard g(lock);
    if (!g_ppocrv5)
        return env->NewStringUTF("[]");

    std::vector<Object> objects;
    g_ppocrv5->detect_and_recognize(rgb, objects);

    __android_log_print(ANDROID_LOG_WARN, "ncnn",
        "ocrFromImageJson: dict_size=%d objects=%d img=%dx%d",
        character_dict_size, (int)objects.size(), rgb.cols, rgb.rows);

    const std::string& char_filter = g_ppocrv5->get_char_filter();
    std::string json = "[";
    bool first = true;
    for (size_t i = 0; i < objects.size(); i++)
    {
        std::string line_text;
        for (size_t j = 0; j < objects[i].text.size(); j++)
        {
            const Character& ch = objects[i].text[j];
            if (ch.id >= character_dict_size)
            {
                // Space token (the class after all dict chars) — the recognize()
                // path emits this as a space; the image path used to drop it,
                // which is why words merged. Keep it.
                if (!line_text.empty() && line_text.back() != ' ')
                    line_text += " ";
            }
            else if (ch.id >= 0)
            {
                std::string c_str = character_dict[ch.id];
                if (char_filter.empty() ||
                    (c_str.length() == 1 && char_filter.find(c_str[0]) != std::string::npos))
                    line_text += c_str;
            }
        }
        if (line_text.empty())
            continue;

        std::string esc;
        for (char c : line_text)
        {
            if (c == '"' || c == '\\') { esc += '\\'; esc += c; }
            else if (c == '\n') esc += "\\n";
            else if ((unsigned char)c >= 0x20) esc += c;
        }

        cv::Point2f pts[4];
        objects[i].rrect.points(pts);

        if (!first) json += ",";
        first = false;
        json += "{\"text\":\"" + esc + "\",\"prob\":" + std::to_string(objects[i].prob) + ",\"points\":[";
        for (int k = 0; k < 4; k++)
        {
            if (k) json += ",";
            json += "[" + std::to_string((int)pts[k].x) + "," + std::to_string((int)pts[k].y) + "]";
        }
        json += "]}";
    }
    json += "]";
    return env->NewStringUTF(json.c_str());
}

}

