# Building the MEX modules

Source is under `src/`; compiler intermediates belong under `build/`; final
modules go under `bin/cpu` or `bin/cuda`. The ready-to-run Windows binaries
and compressed runtime package are included; intermediates and extracted caches
are ignored. There are no Python build scripts or generated-code steps.

The native build targets Windows x64 with MSVC and CMake >=3.24. Use a Visual
Studio x64 developer shell with an installed compiler supported by your CUDA
toolkit if building CUDA. MATLAB supplies the MEX headers and import libraries.
OpenCV >=4.13 supplies optimized matrix, FFT and image-processing primitives.

## CPU profile: build this first

Use a **CUDA-disabled static OpenCV** installation with `core` and `imgproc`.
Its `OpenCV_DIR` is the directory containing `OpenCVConfig.cmake`.
For example, from the repository root in the developer shell:

```powershell
cmake -S . -B build/cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DMatlab_ROOT_DIR="C:/Program Files/MATLAB/R2026a" -DOpenCV_DIR="D:/dependencies/opencv-cpu"
cmake --build build/cpu --parallel 4
```

This produces `defocus_cpu_mex.mexw64`, `cpu_refresh_mex.mexw64` and
`workflow_pool_mex.mexw64` in `bin/cpu`. No CUDA compiler, GPU or MATLAB GPU
headers are needed for this profile. Required Microsoft runtime libraries
still need to be available on the target computer.

## Optional CUDA profile

Configure it in a **different build directory**, using CUDA-enabled OpenCV
with `core`, `imgproc`, `cudaarithm` and `cudaimgproc`. Separate configurations
avoid mixing CPU static OpenCV with CUDA runtime dependencies.

```powershell
cmake -S . -B build/cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DDEFOCUS_CUDA_BUILD=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DMatlab_ROOT_DIR="C:/Program Files/MATLAB/R2026a" -DOpenCV_DIR="D:/dependencies/opencv-cuda"
cmake --build build/cuda --parallel 4
```

`120` is the development RTX5070 architecture, not a universal value. Set the
architecture(s) for your GPU and toolkit. The CUDA source uses conditional
graphs and associated device/runtime APIs; both toolkit and device support
matter. CMake requires at least CUDA12.3 for these APIs; the working development
toolchain used CUDA13.4. Older combinations are not promised to work.

The profile builds `candidates_uint16_mex.mexw64` and `resident_mex.mexw64` in
`bin/cuda`. It requires MATLAB Parallel Computing Toolbox GPU headers and
`gpumexbinder.lib`. Keep the CPU profile installed for optimized fallback.

Place required CUDA-enabled OpenCV DLLs and their redistributable dependencies
in `bin/cuda`, or ensure they are discoverable through the process DLL search
path. CUDA runtime/cuFFT DLLs must also be discoverable; the wrapper adds
existing `CUDA_PATH/bin` and `CUDA_PATH/bin/x64` directories when available.
System NVIDIA driver libraries come from the installed driver. The included
Windows package supplies OpenCV/cuFFT/JIT-link runtimes automatically; a fresh
CUDA toolkit install is not needed when using those packaged modules. When
rebuilding with different dependencies, package the corresponding runtime
libraries and update `bin/manifest.json` and its archive parts as appropriate.

Before publishing rebuilt binaries, inspect embedded OpenCV build information
and diagnostic paths for personal directories. Rebuilding from a local SDK can
reintroduce its original build paths, including its Python executable path even
though this project contains no Python code. Use neutral build directories and
tool paths, or sanitize only diagnostic strings with equal-length replacements
and verify executable sections remain byte-for-byte identical. Repackage the
runtime DLL and update all affected manifest hashes after any such change.

## MATLAB setup and source-only use

Call `setup_defocustracker_turbo` from this checkout. It adds the included
MATLAB library/tools and available binary directories for the current session.
It unpacks the bundled CUDA runtime once and registers its local DLL directory.
It does not call `savepath` or install binaries into MATLAB itself.

Without fast CPU binaries, `dtracker_process_fast` delegates to ordinary
MATLAB CPU processing. The included MATLAB `normxcorr2_mex.m` compatibility
implementation keeps the old Windows valid-only/zero-padded interface usable
without shipping the historical binary. Its floating-point behavior can differ
from the old OpenCV MEX. Normal training requires the usual MATLAB toolboxes.

MEX binaries are platform and MATLAB/runtime dependent. The included binaries
target Windows x64, MATLAB R2026a and CUDA architecture sm_120, with optimized
CPU fallback. Other MATLAB releases may require rebuilding. Linux/macOS native
build adaptation is future work. See `bin/README.md` for the package contents.
