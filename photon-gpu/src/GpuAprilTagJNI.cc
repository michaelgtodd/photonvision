/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// JNI over Team 971's CUDA AprilTag detector (third_party/frc971, Team 4143's
// copy), producing edu.wpi.first.apriltag.AprilTagDetection objects like
// WPILib's own detector does. One handle per camera; each carries its own
// CUDA stream and apriltag_detector_t, so handles may be used concurrently
// from different threads. The object construction follows Team 4143's
// GpuDetectorJNI (Apache-2.0).

#include <jni.h>
#include <malloc.h>
#include <cstdio>
#include <cstdlib>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "frc971/orin/971apriltag.h"
#include "apriltag.h"
#include "tag16h5.h"
#include "tag36h11.h"

namespace {

struct Handle {
  int width, height, decimate;
  apriltag_family_t *family = nullptr;
  bool family_is_16h5 = false;
  apriltag_detector_t *td = nullptr;
  frc971::apriltag::GpuDetector *gpu = nullptr;
  std::vector<uint8_t> staging;  // for non-contiguous input
  std::mutex mutex;              // a handle is used by one pipeline, but be safe
};

jclass g_detection_cls = nullptr;
jmethodID g_detection_ctor = nullptr;

bool InitClasses(JNIEnv *env) {
  if (g_detection_cls) return true;
  jclass local = env->FindClass("edu/wpi/first/apriltag/AprilTagDetection");
  if (!local) return false;
  g_detection_cls = static_cast<jclass>(env->NewGlobalRef(local));
  env->DeleteLocalRef(local);
  g_detection_ctor = env->GetMethodID(g_detection_cls, "<init>", "(Ljava/lang/String;IIF[DDD[D)V");
  return g_detection_ctor != nullptr;
}

void ThrowIllegal(JNIEnv *env, const char *msg) {
  jclass c = env->FindClass("java/lang/IllegalArgumentException");
  if (c) env->ThrowNew(c, msg);
}

jobject MakeDetection(JNIEnv *env, const apriltag_detection_t *d) {
  jstring fam = env->NewStringUTF(d->family->name);
  jdoubleArray h = env->NewDoubleArray(9);
  env->SetDoubleArrayRegion(h, 0, 9, d->H->data);
  jdoubleArray corners = env->NewDoubleArray(8);
  env->SetDoubleArrayRegion(corners, 0, 8, &d->p[0][0]);
  jobject obj = env->NewObject(g_detection_cls, g_detection_ctor, fam, static_cast<jint>(d->id),
                               static_cast<jint>(d->hamming), static_cast<jfloat>(d->decision_margin),
                               h, static_cast<jdouble>(d->c[0]), static_cast<jdouble>(d->c[1]), corners);
  env->DeleteLocalRef(fam);
  env->DeleteLocalRef(h);
  env->DeleteLocalRef(corners);
  return obj;
}

}  // namespace

extern "C" {

/*
 * Class:     org_photonvision_jni_GpuAprilTagJNI
 * Method:    create
 * Signature: (IIIIIZFLjava/lang/String;)J
 *
 * decimate must be 2: the detector's packed point encoding (10-bit x/y,
 * 20-bit union-find representatives) is sized for the half-resolution
 * image; at decimate 1 a 1920x1200 frame overflows it and tags with
 * x > 1023 are silently lost (measured). Kept as a parameter for smaller
 * frames.
 */
JNIEXPORT jlong JNICALL Java_org_photonvision_jni_GpuAprilTagJNI_create(
    JNIEnv *env, jclass, jint width, jint height, jint decimate, jint cpuThreads,
    jint minWhiteBlackDiff, jboolean refineEdges, jfloat decodeSharpening, jstring family) {
  if (!InitClasses(env)) return 0;
  if (width % 8 || height % 8 || width <= 0 || height <= 0) {
    ThrowIllegal(env, "GpuAprilTag: width and height must be multiples of 8");
    return 0;
  }
  if (decimate != 1 && decimate != 2) {
    ThrowIllegal(env, "GpuAprilTag: decimate must be 1 or 2");
    return 0;
  }
  if (decimate == 1 && (width > 1024 || height > 1024)) {
    ThrowIllegal(env, "GpuAprilTag: decimate 1 only supports frames up to 1024x1024 (packed coordinates)");
    return 0;
  }

  // Note: cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) was tried here to
  // stop the per-frame stream synchronisation spinning a core. Measured in the
  // harness it saved nothing (the detector's CPU time is launch/driver
  // overhead, not the wait), and inside the JVM, with four detectors created
  // from four threads, it left every later CUDA call failing with "invalid
  // device ordinal". Do not reintroduce it without that test.

  const char *fam = env->GetStringUTFChars(family, nullptr);
  std::string famName = fam ? fam : "tag36h11";
  if (fam) env->ReleaseStringUTFChars(family, fam);

  auto *h = new Handle();
  h->width = width;
  h->height = height;
  h->decimate = decimate;
  if (famName == "tag16h5") {
    h->family = tag16h5_create();
    h->family_is_16h5 = true;
  } else {
    h->family = tag36h11_create();
  }
  h->td = apriltag_detector_create();
  apriltag_detector_add_family_bits(h->td, h->family, 1);
  h->td->nthreads = cpuThreads > 0 ? cpuThreads : 1;
  h->td->wp = workerpool_create(h->td->nthreads);
  h->td->quad_decimate = decimate;
  h->td->quad_sigma = 0;
  h->td->refine_edges = refineEdges ? 1 : 0;
  h->td->decode_sharpening = decodeSharpening;
  h->td->qtp.min_white_black_diff = minWhiteBlackDiff;

  // Zero distortion: the detector's corner undistortion is then the identity
  // and corners come back in raw pixel coordinates, which is what the rest of
  // PhotonVision expects (it applies the calibration itself).
  frc971::apriltag::CameraMatrix cam{1000.0, width / 2.0, 1000.0, height / 2.0};
  frc971::apriltag::DistCoeffs dist{0, 0, 0, 0, 0};
  h->gpu = new frc971::apriltag::GpuDetector(width, height, h->td, cam, dist, decimate);
  return reinterpret_cast<jlong>(h);
}

/*
 * Class:     org_photonvision_jni_GpuAprilTagJNI
 * Method:    detect
 * Signature: (JJIIJ)[Ledu/wpi/first/apriltag/AprilTagDetection;
 *
 * dataPtr/width/height/step describe an 8-bit grey image (a cv::Mat's data,
 * cols, rows, step). The frame is copied to the GPU by the detector.
 */
JNIEXPORT jobjectArray JNICALL Java_org_photonvision_jni_GpuAprilTagJNI_detect(
    JNIEnv *env, jclass, jlong handle, jlong dataPtr, jint width, jint height, jlong step) {
  auto *h = reinterpret_cast<Handle *>(handle);
  if (!h || !dataPtr) {
    ThrowIllegal(env, "GpuAprilTag: null handle or image");
    return nullptr;
  }
  if (width != h->width || height != h->height) {
    ThrowIllegal(env, "GpuAprilTag: image size differs from the detector's");
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(h->mutex);

  auto *src = reinterpret_cast<const uint8_t *>(dataPtr);
  uint8_t *image = const_cast<uint8_t *>(src);
  if (step != width) {
    h->staging.resize(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; y++)
      std::memcpy(h->staging.data() + static_cast<size_t>(y) * width, src + static_cast<size_t>(y) * step, width);
    image = h->staging.data();
  }

  const long failures_before = frc971::apriltag::cuda_check_failures.load();
  h->gpu->DetectGrayHost(image);
  if (frc971::apriltag::cuda_check_failures.load() != failures_before) {
    // A CUDA call failed during this frame: its results are not trustworthy.
    // The frame is dropped; the Java side reads cudaFailures() to report it.
    return nullptr;
  }
  const zarray_t *dets = h->gpu->Detections();
  const int n = dets ? zarray_size(dets) : 0;

  jobjectArray arr = env->NewObjectArray(n, g_detection_cls, nullptr);
  if (!arr) return nullptr;
  for (int i = 0; i < n; i++) {
    apriltag_detection_t *d;
    zarray_get(dets, i, &d);
    jobject obj = MakeDetection(env, d);
    env->SetObjectArrayElement(arr, i, obj);
    env->DeleteLocalRef(obj);
  }
  return arr;
}

/*
 * Class:     org_photonvision_jni_GpuAprilTagJNI
 * Method:    destroy
 * Signature: (J)V
 */
JNIEXPORT jlong JNICALL Java_org_photonvision_jni_GpuAprilTagJNI_cudaFailures(JNIEnv *, jclass) {
  return frc971::apriltag::cuda_check_failures.load();
}

JNIEXPORT void JNICALL Java_org_photonvision_jni_GpuAprilTagJNI_destroy(JNIEnv *, jclass, jlong handle) {
  auto *h = reinterpret_cast<Handle *>(handle);
  if (!h) return;
  {
    std::lock_guard<std::mutex> lock(h->mutex);
    delete h->gpu;
    h->gpu = nullptr;
    apriltag_detector_destroy(h->td);
    if (h->family_is_16h5) tag16h5_destroy(h->family);
    else tag36h11_destroy(h->family);
  }
  delete h;
}

/*
 * Self-test without a JVM: build a detector for width x height, run `frames`
 * frames (the caller's grey image, or a synthetic square that contains no
 * tag) and report the number of CUDA check failures and the growth of the
 * malloc heap between the 2nd frame and the last (a per-frame leak shows up
 * as growth proportional to `frames`). Exported for photongpu-selftest,
 * which runs the exact library PhotonVision loads.
 */
static long HeapInUseBytes() {
  struct mallinfo2 mi = mallinfo2();
  return static_cast<long>(mi.uordblks) + static_cast<long>(mi.hblkhd);
}

__attribute__((visibility("default"))) long photongpu_selftest(int width, int height, int decimate,
                                                                const uint8_t *frame_in, int frames,
                                                                int *ndet_out, long *heap_growth_out) {
  const long before = frc971::apriltag::cuda_check_failures.load();
  apriltag_family_t *fam = tag36h11_create();
  apriltag_detector_t *td = apriltag_detector_create();
  apriltag_detector_add_family_bits(td, fam, 1);
  td->nthreads = 1;
  td->wp = workerpool_create(1);
  td->quad_decimate = decimate;
  td->qtp.min_white_black_diff = 5;
  frc971::apriltag::CameraMatrix cam{1000.0, width / 2.0, 1000.0, height / 2.0};
  frc971::apriltag::DistCoeffs dist{0, 0, 0, 0, 0};
  auto *gpu = new frc971::apriltag::GpuDetector(width, height, td, cam, dist, decimate);
  std::vector<uint8_t> frame(static_cast<size_t>(width) * height, 128);
  if (frame_in) {
    std::copy(frame_in, frame_in + frame.size(), frame.begin());
  } else {
    // a bright square with a dark border so the pipeline has a blob to work on
    for (int y = height / 3; y < 2 * height / 3; y++)
      for (int x = width / 3; x < 2 * width / 3; x++) frame[static_cast<size_t>(y) * width + x] = 230;
  }
  int ndet = 0;
  long heap_after_warmup = 0;
  if (frames < 3) frames = 3;
  for (int i = 0; i < frames; i++) {
    gpu->DetectGrayHost(frame.data());
    const zarray_t *d = gpu->Detections();
    ndet = d ? zarray_size(d) : 0;
    if (i == 1) heap_after_warmup = HeapInUseBytes();
  }
  if (ndet_out) *ndet_out = ndet;
  if (heap_growth_out) *heap_growth_out = HeapInUseBytes() - heap_after_warmup;
  delete gpu;
  apriltag_detector_destroy(td);
  tag36h11_destroy(fam);
  return frc971::apriltag::cuda_check_failures.load() - before;
}

}  // extern "C"
