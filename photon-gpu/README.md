# photon-gpu

Team 971's CUDA AprilTag detector as a JNI library for PhotonVision on NVIDIA
Jetson (Orin, JetPack 7 / CUDA 13). Built on the device with `build.sh`;
loaded by `org.photonvision.jni.GpuAprilTagJNI` from
`/opt/photonvision/lib/libphotongpu.so` and used by
`AprilTagDetectionGpuPipe` when an AprilTag pipeline has "GPU detector"
enabled.

Measured on an Orin Nano Super at 1920x1200: ~7 ms per frame with one
camera, 4 x 50 fps with four cameras in parallel, quad decimate 2. The
detector decodes and refines corners on the full-resolution image; only quad
finding is decimated. Decimate 1 is not supported at this resolution (see
the note in `src/GpuAprilTagJNI.cc`).

| Directory | Origin | License |
|---|---|---|
| `third_party/frc971` | Team 971's `frc971/orin`, via Team 4143's `GpuDetectorJNI` (adds grey input and the decimate parameter), plus a random-access iterator fix for CCCL 3 | Apache-2.0 |
| `third_party/apriltag` | AprilRobotics apriltag, Team 4143's copy (exports `quad_decode_index`) | BSD-2-Clause |
| `compat/` | CUDA 13 / CCCL 3 replacements for two removed CUB iterators | this repository |

Local modifications to the vendored 971 sources (all marked `photon-gpu:` in
the code): `transform_output_iterator.h` random access for CCCL 3;
`971apriltag.cu` `DetectGrayHost()` decodes from the caller's host image
instead of copying the grey image back from the device; `line_fit_filter.cu`
`FitLines()`/`FitQuads()` skip the launch when there are no blob points or
candidate blobs (a 0-block grid is a launch error that the next CUDA call
reported as "invalid device ordinal", on every frame without something
tag-like in view); `cuda.h`/`cuda.cc` `CHECK_CUDA` counts failures instead
of printing each one.

## Health

- `build.sh` runs `photongpu-selftest` on the freshly built library (a
  tagless synthetic frame, and `PHOTONGPU_SELFTEST_FRAME=<raw grey
  1920x1200>` for a frame with tags) and installs only if no CUDA call
  fails. The installed copy is `/opt/photonvision/bin/photongpu-selftest
  [w h [frame.raw]]`.
- `GpuAprilTagJNI.detect()` returns null for a frame during which a CUDA
  call failed (`AprilTagDetectionGpuPipe` drops the frame and logs, without
  flooding); `GpuAprilTagJNI.cudaFailures()` is the process-wide count.
- `PHOTONGPU_SYNC=1` in the environment synchronises and checks after every
  stage (the upstream `--sync` flag), naming the failing stage. Slow.
