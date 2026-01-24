cmake_minimum_required(VERSION 3.16)

project(photonlib VERSION 1.0 LANGUAGES CXX)

# --- Dependencies ---
# PhotonLib strictly depends on WPILib (ntcore, cscore, wpiutil) and JSON.
# If you are on a system with these installed (e.g. via apt or make install),
# find_package should work. Otherwise, you may need to use FetchContent 
# or point CMAKE_PREFIX_PATH to your WPILib install.

find_package(wpiutil REQUIRED)
find_package(wpilibc REQUIRED)
find_package(ntcore REQUIRED)
find_package(cscore REQUIRED)
find_package(nlohmann_json REQUIRED)

# --- Source Management ---
# Standard Gradle structure for C++ is src/main/native/cpp and include
set(PHOTONLIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/cpp")
set(PHOTONLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include")

# Glob sources (or list manually for strictness)
file(GLOB_RECURSE PHOTONLIB_SOURCES "${PHOTONLIB_SRC_DIR}/*.cpp")

# --- Target Definition ---
add_library(photonlib SHARED ${PHOTONLIB_SOURCES})

# --- Include Directories ---
target_include_directories(photonlib PUBLIC
    $<BUILD_INTERFACE:${PHOTONLIB_INCLUDE_DIR}>
    ${CMAKE_CURRENT_SOURCE_DIR}/../photon-targeting/src/main/native/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../photon-targeting/src/generated/main/native/include/
    /usr/local/include/apriltag/
    $<INSTALL_INTERFACE:include>
)

# --- Link Dependencies ---
target_link_libraries(photonlib PUBLIC
    wpiutil
    ntcore
    cscore
    wpilibc
    nlohmann_json::nlohmann_json
)

# --- Compile Features ---
# PhotonLib C++ typically requires C++17 or C++20
target_compile_features(photonlib PUBLIC cxx_std_20)

# --- Definitions ---
if(MSVC)
    target_compile_options(photonlib PRIVATE /W4)
else()
    target_compile_options(photonlib PRIVATE -Wall -Wextra)
endif()

# --- Installation (Optional) ---
# Useful if you want to 'make install' this lib to /usr/local
include(GNUInstallDirs)

install(TARGETS photonlib
    EXPORT photonlibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${PHOTONLIB_INCLUDE_DIR}/photon
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
