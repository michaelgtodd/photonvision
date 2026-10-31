// Zero-copy hand-off of V4L2 capture buffers to CUDA.
//
// V4l2LatestFrame (photon-targeting) streams a Y16 camera on cscore's
// descriptor and keeps the newest filled buffer. The CPU reads those buffers
// (the driver's uncached mmap) at about 1 GB/s, so converting a 1920x1200
// frame to 8 bits there costs 4.5 ms even with NEON. The GPU reads the same
// buffer, once registered with CUDA, in a fraction of a millisecond.
//
// V4l2LatestFrame cannot link against CUDA, so this library hands it a table
// of plain C function pointers (GpuAprilTagJNI.captureOps()): pinned host
// allocation for the 8-bit frames it returns (CPU-cached and GPU-visible, so
// the GPU detector reads them in place and its CPU decode stage reads them
// from cache), and the conversion itself. Bench evidence:
// experiments/apriltag-rate/bench_capture_cuda.cu in the bench repository
// (cudaHostRegister on the mmap'd buffers works; 60 frames byte-identical to
// the CPU conversion; 0.28 ms per frame).
#include <cuda_runtime.h>
#include <jni.h>

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>

#include "src/GpuCapture.h"

namespace {

// Top 8 of each 16-bit sample, 8 pixels per thread: one 16-byte load, one
// 8-byte store.
__global__ void NarrowTop8(const uint4 *in, uint2 *out, size_t n8) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= n8) return;
  uint4 v = in[i];
  uint32_t a = ((v.x >> 8) & 0xff) | ((v.x >> 16) & 0xff00);
  uint32_t b = ((v.y >> 8) & 0xff) | ((v.y >> 16) & 0xff00);
  uint32_t c = ((v.z >> 8) & 0xff) | ((v.z >> 16) & 0xff00);
  uint32_t d = ((v.w >> 8) & 0xff) | ((v.w >> 16) & 0xff00);
  out[i] = make_uint2(a | (b << 16), c | (d << 16));
}

struct Registration {
  size_t length;
  const void *device;
};

std::mutex g_mutex;
// capture buffer (host address) -> its CUDA registration
std::unordered_map<const void *, Registration> g_registered;

// One stream per calling thread: each camera's vision thread converts its own
// frames without serialising on the others.
thread_local cudaStream_t t_stream = nullptr;

const void *DevicePointer(const void *capture, size_t length) {
  std::scoped_lock lock{g_mutex};
  auto it = g_registered.find(capture);
  if (it != g_registered.end() && it->second.length == length) {
    return it->second.device;
  }
  if (it != g_registered.end()) {
    cudaHostUnregister(const_cast<void *>(capture));
    g_registered.erase(it);
  }
  // V4l2LatestFrame maps the buffers read-only.
  void *host = const_cast<void *>(capture);
  cudaError_t e = cudaHostRegister(host, length, cudaHostRegisterMapped | cudaHostRegisterReadOnly);
  if (e != cudaSuccess) {
    cudaGetLastError();
    e = cudaHostRegister(host, length, cudaHostRegisterMapped);
  }
  if (e != cudaSuccess) {
    cudaGetLastError();
    std::fprintf(stderr, "photon-gpu: cudaHostRegister(capture buffer): %s\n", cudaGetErrorString(e));
    return nullptr;
  }
  void *device = nullptr;
  if (cudaHostGetDevicePointer(&device, host, 0) != cudaSuccess) {
    cudaGetLastError();
    cudaHostUnregister(host);
    return nullptr;
  }
  g_registered[capture] = {length, device};
  return device;
}

void *PinnedAlloc(size_t bytes) {
  void *p = nullptr;
  if (cudaMallocHost(&p, bytes) != cudaSuccess) {
    cudaGetLastError();
    return nullptr;
  }
  return p;
}

void PinnedFree(void *p) {
  if (p) cudaFreeHost(p);
}

int NarrowY16(const void *capture, size_t capture_length, void *out, int width, int height) {
  const size_t pixels = static_cast<size_t>(width) * height;
  if (!capture || !out || pixels % 8 != 0 || capture_length < pixels * 2) return -1;
  const void *in = DevicePointer(capture, capture_length);
  if (!in) return -2;
  void *out_device = nullptr;
  if (cudaHostGetDevicePointer(&out_device, out, 0) != cudaSuccess) {
    cudaGetLastError();
    return -3;
  }
  if (!t_stream && cudaStreamCreateWithFlags(&t_stream, cudaStreamNonBlocking) != cudaSuccess) {
    cudaGetLastError();
    t_stream = nullptr;
    return -4;
  }
  const size_t n8 = pixels / 8;
  NarrowTop8<<<static_cast<unsigned>((n8 + 255) / 256), 256, 0, t_stream>>>(
      static_cast<const uint4 *>(in), static_cast<uint2 *>(out_device), n8);
  cudaError_t e = cudaStreamSynchronize(t_stream);
  if (e == cudaSuccess) e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "photon-gpu: capture conversion: %s\n", cudaGetErrorString(e));
    return -5;
  }
  return 0;
}

void ForgetBuffer(const void *capture) {
  std::scoped_lock lock{g_mutex};
  auto it = g_registered.find(capture);
  if (it == g_registered.end()) return;
  cudaHostUnregister(const_cast<void *>(capture));
  cudaGetLastError();
  g_registered.erase(it);
}

const PhotonGpuCaptureOps g_ops = {
    kPhotonGpuCaptureOpsVersion, PinnedAlloc, PinnedFree, NarrowY16, ForgetBuffer,
};

}  // namespace

/*
 * Class:     org_photonvision_jni_GpuAprilTagJNI
 * Method:    captureOps
 * Signature: ()J
 *
 * The address of this library's PhotonGpuCaptureOps table, for
 * V4l2LatestFrame.setGpu. Valid for the life of the process.
 */
extern "C" JNIEXPORT jlong JNICALL Java_org_photonvision_jni_GpuAprilTagJNI_captureOps(JNIEnv *, jclass) {
  return reinterpret_cast<jlong>(&g_ops);
}
