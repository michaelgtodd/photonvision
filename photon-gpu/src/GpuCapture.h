// The table of functions libphotongpu hands to V4l2LatestFrame (in
// photon-targeting, which does not link CUDA) for zero-copy capture. Plain C
// types only: photon-targeting declares an identical struct, and the version
// field guards against the two drifting apart.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { kPhotonGpuCaptureOpsVersion = 1 };

typedef struct PhotonGpuCaptureOps {
  uint32_t version;
  // Pinned host memory: CPU-cached and readable by the GPU in place.
  void *(*pinned_alloc)(size_t bytes);
  void (*pinned_free)(void *p);
  // Top 8 bits of a Y16 capture buffer (the driver's mmap, registered with
  // CUDA on first use) into out (from pinned_alloc, width * height bytes,
  // continuous), on the GPU. Returns 0 on success; out is complete on return.
  int (*narrow_y16)(const void *capture, size_t capture_length, void *out, int width, int height);
  // Drop a capture buffer's CUDA registration before it is unmapped.
  void (*forget_buffer)(const void *capture);
} PhotonGpuCaptureOps;

#ifdef __cplusplus
}
#endif
