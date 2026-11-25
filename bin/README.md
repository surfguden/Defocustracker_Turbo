# Included Windows binaries

This package runs without a compiler on a compatible Windows x64 MATLAB
installation. The binaries were built for MATLAB R2026a with MSVC19.44,
OpenCV4.13 and CUDA13.4. GPU kernels target `sm_120` (development RTX5070).
Other platforms/MATLAB releases/GPU architectures may need a rebuild.
An installed MATLAB license/toolboxes and a compatible GPU driver for CUDA
are still required. CPU processing has no CUDA dependency.

## Package contents

| File | Purpose |
|---|---|
| `cpu/defocus_cpu_mex.mexw64` | Shared CPU context and reference operations |
| `cpu/cpu_refresh_mex.mexw64` | Optimized CPU positioning pipeline |
| `cpu/workflow_pool_mex.mexw64` | CPU helper for CUDA preprocessing/depth matching |
| `cuda/candidates_uint16_mex.mexw64` | CUDA candidate identification |
| `cuda/resident_mex.mexw64` | CUDA XYZ refinement/depth walk/suppression |
| `cuda/runtime.zip.001` through `.004` | Compressed GPU runtime archive parts |
| `manifest.json` | Platform/build information, package filenames, sizes and SHA256 hashes |
| `licenses/` | License notices for redistributed third-party components |

The archive contains `opencv_core4130.dll`, `cufft64_12.dll` and
`nvJitLink_130_0.dll`. The JIT-link library is loaded dynamically by cuFFT,
so its dependency is not visible in cuFFT's ordinary DLL import table.
No CUDA toolkit installation is needed for this included build.
CPU binaries link OpenCV statically and need no extra OpenCV DLL.
OpenCV diagnostic/build-information strings in the packaged CPU modules and
OpenCV DLL use neutral build paths. These strings were sanitized with equal-length
replacements confined to read-only data; executable sections are unchanged.
MATLAB and Microsoft runtime libraries come from the target MATLAB/system
installation; NVIDIA driver libraries come from its installed driver.

## First use

From the repository root in MATLAB:

```matlab
setup_defocustracker_turbo
```

Setup joins the four archive parts, verifies the combined SHA256, extracts
the three DLLs into a generated `bin/cuda/runtime/<package-hash>/` directory,
verifies those libraries and adds that directory to MATLAB's process DLL
search path. Later setup calls reuse the cache. CUDA initialization also
prepares the package if the fast API was called without setup.

The compressed package plus modules is about245MB; extraction needs about349MB
plus temporary space for the joined archive. The joined temporary ZIP is
removed after setup. Archive parts are ordinary Git files, so cloning requires
no Git LFS installation. Keep all four parts together and unchanged.

Only the split archive parts are distributed; generated runtime caches are
ignored by Git. If the cache needs rebuilding, close MATLAB before removing only the
generated `bin/cuda/runtime/` directory, then run setup again. Do not remove
the distributed `.zip.001` ... `.zip.004` parts.

Automatic CUDA selection retains optimized CPU fallback. Missing/incompatible
CUDA modules, GPU features, toolbox support or drivers can therefore select
CPU rather than prevent processing. Standard MATLAB processing remains
available if fast native binaries cannot load.

To change native code or target other hardware, see `docs/building.md`.
