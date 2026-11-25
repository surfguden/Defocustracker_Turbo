add_library(defocus_core STATIC core/tracker.cpp)
target_include_directories(defocus_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

function(defocus_mex name profile)
    add_library(${name} SHARED ${ARGN})
    target_include_directories(${name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/mex"
        ${Matlab_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE "${Matlab_MEX_LIBRARY}" "${Matlab_MX_LIBRARY}")
    target_link_options(${name} PRIVATE /EXPORT:mexFunction)
    set_target_properties(${name} PROPERTIES PREFIX "" SUFFIX ".${Matlab_MEX_EXTENSION}"
        RUNTIME_OUTPUT_DIRECTORY "${DEFOCUS_BINARY_ROOT}/${profile}"
        LIBRARY_OUTPUT_DIRECTORY "${DEFOCUS_BINARY_ROOT}/${profile}")
    foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(${name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${configuration} "${DEFOCUS_BINARY_ROOT}/${profile}"
            LIBRARY_OUTPUT_DIRECTORY_${configuration} "${DEFOCUS_BINARY_ROOT}/${profile}")
    endforeach()
endfunction()

if(NOT DEFOCUS_CUDA_BUILD)
    add_library(defocus_cpu STATIC cpu/correlation_cpu.cpp cpu/candidate_cpu.cpp)
    target_include_directories(defocus_cpu PRIVATE include ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(defocus_cpu PUBLIC defocus_core ${OpenCV_LIBS})
    defocus_mex(defocus_cpu_mex cpu mex/mex_bridge.cpp)
    target_link_libraries(defocus_cpu_mex PRIVATE defocus_cpu)

    # This library reuses core/tracker.cpp inside cpu_core.cpp with renamed
    # reference refinement. Keep it independent of defocus_core to avoid
    # duplicate definitions and preserve the optimized CPU implementation.
    add_library(defocus_cached_cpu STATIC cpu/cpu_core.cpp cpu/cpu_correlation.cpp
        cpu/cpu_candidates.cpp cpu/workers.cpp cpu/initial_depth.cpp)
    target_include_directories(defocus_cached_cpu PUBLIC include ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(defocus_cached_cpu PUBLIC ${OpenCV_LIBS})
    defocus_mex(cpu_refresh_mex cpu mex/cpu_refresh_mex.cpp)
    target_link_libraries(cpu_refresh_mex PRIVATE defocus_cached_cpu)

    defocus_mex(workflow_pool_mex cpu mex/workflow_mex.cpp cpu/initial_depth.cpp)
    target_link_libraries(workflow_pool_mex PRIVATE defocus_cpu)
else()
    add_library(defocus_candidates STATIC cuda/candidates.cu)
    target_include_directories(defocus_candidates PUBLIC include PRIVATE cuda
        ${OpenCV_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_options(defocus_candidates PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
    target_link_libraries(defocus_candidates PUBLIC ${OpenCV_LIBS} CUDA::cudart CUDA::cufft)
    defocus_mex(candidates_uint16_mex cuda mex/candidates_mex.cpp)
    target_link_libraries(candidates_uint16_mex PRIVATE defocus_candidates)

    # resident.cu incorporates correlation_cuda.cu and the depth kernel headers.
    # Compile that translation unit once; do not list its included .cu separately.
    add_library(defocus_resident STATIC cuda/resident.cu)
    target_include_directories(defocus_resident PUBLIC include ${OpenCV_INCLUDE_DIRS})
    target_compile_options(defocus_resident PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
    target_link_libraries(defocus_resident PUBLIC defocus_core ${OpenCV_LIBS}
        CUDA::cudart CUDA::cufft CUDA::cuda_driver)
    defocus_mex(resident_mex cuda mex/resident_mex.cpp)
    target_link_libraries(resident_mex PRIVATE defocus_resident)

    set(gpu_headers "${Matlab_ROOT_DIR}/toolbox/parallel/gpu/extern/include")
    set(gpu_binder "${Matlab_ROOT_DIR}/extern/lib/win64/microsoft/gpumexbinder.lib")
    if(NOT EXISTS "${gpu_headers}/gpu/mxGPUArray.h" OR NOT EXISTS "${gpu_binder}")
        message(FATAL_ERROR "CUDA MEX builds require the MATLAB Parallel Computing Toolbox GPU headers/library.")
    endif()
    foreach(target candidates_uint16_mex resident_mex)
        target_include_directories(${target} PRIVATE "${gpu_headers}")
        target_link_libraries(${target} PRIVATE "${gpu_binder}")
    endforeach()
endif()
