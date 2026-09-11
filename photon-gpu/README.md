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
