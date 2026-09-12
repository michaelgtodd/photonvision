#!/usr/bin/env bash
# Build and install libphotongpu.so on the Jetson.
#
#   photon-gpu/build.sh [install prefix]      (default /opt/photonvision)
#
# Needs nvcc (JetPack's CUDA toolkit), cmake and a JDK for jni.h
# (openjdk-25-jdk-headless). PhotonVision loads the library from
# <prefix>/lib/libphotongpu.so (system property photon.gpu.lib overrides).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PREFIX=${1:-/opt/photonvision}
export PATH=/usr/local/cuda/bin:$PATH
if [ -z "${JAVA_HOME:-}" ]; then
  JAVA_HOME=$(dirname "$(dirname "$(readlink -f "$(command -v javac 2>/dev/null || echo /usr/lib/jvm/default-java/bin/javac)")")")
  export JAVA_HOME
fi
[ -f "$JAVA_HOME/include/jni.h" ] || { echo "no jni.h under JAVA_HOME=$JAVA_HOME (install openjdk-25-jdk-headless)"; exit 1; }
cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$HERE/build" -j"$(nproc)"
sudo -n install -d "$PREFIX/lib"
# The self-test runs the freshly built library through one detector before
# anything is installed; a CUDA failure here means PhotonVision would run
# with a broken detector.
LD_LIBRARY_PATH="$HERE/build" "$HERE/build/photongpu-selftest" 1920 1200 - 300 || { echo "self-test FAILED (tagless frame); not installing"; exit 1; }
if [ -n "${PHOTONGPU_SELFTEST_FRAME:-}" ]; then
  LD_LIBRARY_PATH="$HERE/build" "$HERE/build/photongpu-selftest" 1920 1200 "$PHOTONGPU_SELFTEST_FRAME" 300 || { echo "self-test FAILED (frame); not installing"; exit 1; }
fi
sudo -n install -m 0644 "$HERE/build/libphotongpu.so" "$PREFIX/lib/libphotongpu.so"
sudo -n install -d "$PREFIX/bin" && sudo -n install -m 0755 "$HERE/build/photongpu-selftest" "$PREFIX/bin/photongpu-selftest"
echo "installed $PREFIX/lib/libphotongpu.so ($(sha256sum "$HERE/build/libphotongpu.so" | cut -c1-16))"
