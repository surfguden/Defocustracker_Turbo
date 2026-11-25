cmake_minimum_required(VERSION 3.24)
project(DefocustrackerTurbo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(DEFOCUS_CUDA_BUILD "Build optional CUDA modules instead of the CPU profile" OFF)
set(DEFOCUS_BINARY_ROOT "${PROJECT_SOURCE_DIR}/bin" CACHE PATH "Installed MEX binary root")

if(NOT WIN32 OR NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "This release's native build supports Windows x64 with MSVC. MATLAB source remains separate.")
endif()

# Configure CPU and CUDA profiles separately: their OpenCV installations have
# different link/runtime requirements. CPU binaries must not import CUDA DLLs.
if(DEFOCUS_CUDA_BUILD)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit 12.3 REQUIRED)
    find_package(OpenCV 4.13 REQUIRED COMPONENTS core imgproc cudaarithm cudaimgproc)
    message(STATUS "CUDA profile: also build the independent CPU profile for fast fallback.")
else()
    set(OpenCV_STATIC ON)
    find_package(OpenCV 4.13 REQUIRED COMPONENTS core imgproc)
endif()
find_package(Matlab REQUIRED COMPONENTS MX_LIBRARY)
add_subdirectory(src)
