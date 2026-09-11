# photon-gpu: Team 971's CUDA AprilTag detector as a JNI library for
# PhotonVision on NVIDIA Jetson. Built ON the Jetson (needs nvcc and a JDK):
#
#   photon-gpu/build.sh          # -> build/libphotongpu.so, installed to /opt/photonvision/lib
#
# Sources: third_party/frc971 (Team 971 / Team 4143, Apache-2.0),
# third_party/apriltag (University of Michigan, BSD-2; Team 4143's copy, which
# exports quad_decode_index() for the GPU detector), compat/ (CUDA 13 / CCCL 3
# shims). The apriltag library is compiled in with hidden visibility so it
# cannot clash with the copy inside WPILib's own apriltag JNI library.
cmake_minimum_required(VERSION 3.22)
project(photongpu LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)   # Orin
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(JNI REQUIRED COMPONENTS JVM)   # headless JDK: no AWT
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(TP ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
file(GLOB APRILTAG_SRC ${TP}/apriltag/*.c ${TP}/apriltag/common/*.c)
# (all of common/ is compiled: image_u8*.c reference the pnm/pam loaders)
set(FRC971_SRC
  ${TP}/frc971/orin/971apriltag.cu
  ${TP}/frc971/orin/apriltag_detect.cu
  ${TP}/frc971/orin/labeling_allegretti_2019_BKE.cu
  ${TP}/frc971/orin/line_fit_filter.cu
  ${TP}/frc971/orin/points.cu
  ${TP}/frc971/orin/threshold.cu
  ${TP}/frc971/orin/cuda.cc)

add_library(photongpu SHARED ${APRILTAG_SRC} ${FRC971_SRC} src/GpuAprilTagJNI.cc)
# The 971 sources include "frc971/orin/..." and "third_party/apriltag/..."
# relative to the vendored tree's parent, i.e. this directory.
target_include_directories(photongpu PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} ${TP} ${TP}/apriltag ${TP}/apriltag/common
  ${CMAKE_CURRENT_SOURCE_DIR}/compat
  ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS}/cccl
  ${JNI_INCLUDE_DIRS})
# jni.h only; the library never calls into the JVM beyond what JNI itself provides
target_compile_options(photongpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fvisibility=hidden>
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-psabi -fvisibility=hidden>
  $<$<COMPILE_LANGUAGE:C>:-fvisibility=hidden -Wno-unused-result>)
target_compile_definitions(photongpu PRIVATE NDEBUG)
set_target_properties(photongpu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
target_link_libraries(photongpu PRIVATE CUDA::cudart_static Threads::Threads m)
# A missing symbol must fail the build, not the JVM at first use.
target_link_options(photongpu PRIVATE -Wl,--no-undefined)

install(TARGETS photongpu LIBRARY DESTINATION lib)
