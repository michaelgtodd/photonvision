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
sudo -n install -m 0644 "$HERE/build/libphotongpu.so" "$PREFIX/lib/libphotongpu.so"
echo "installed $PREFIX/lib/libphotongpu.so ($(sha256sum "$HERE/build/libphotongpu.so" | cut -c1-16))"
