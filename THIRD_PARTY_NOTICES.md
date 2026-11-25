# Third-party notices

## DefocusTracker MATLAB

The MATLAB library under `defocustracker/` originates from DefocusTracker by
Massimiliano Rossi and Rune Barnkob, copyright2020, under the MIT license
reproduced in `LICENSE`.

Upstream: https://gitlab.com/defocustracking/defocustracker-matlab

The local source copy was based on `develop_OJ_CC`, revision
`a6c0a0e29b8334602bc30f5207c55fa39c26f106`.
The original checkout is not modified by this repository. The fast processing
adapter, training-particle picker and C++/CUDA implementations are maintained
alongside the included MATLAB workflow. The MATLAB compatibility implementation
of `normxcorr2_mex` preserves its historical interface documentation/attribution
to Daniel Eaton and its acknowledgment of OpenCV.

## OpenCV, CUDA and MATLAB

The Windows binary package includes OpenCV4.13 core, NVIDIA cuFFT12 and its
CUDA13.4 JIT-link runtime. CPU modules statically link CUDA-disabled OpenCV.
OpenCV also incorporates Intel IPP and ITT components. Corresponding notices
are included under `bin/licenses/`: OpenCV's license, NVIDIA's CUDA EULA,
Intel's IPP EULA and the Intel ITT BSD-3-Clause license. Intel ITT source
attribution is copyright2005-2019 Intel Corporation.

These third-party binaries retain their respective licenses; the project's
MIT license does not replace those terms. The runtime DLLs are stored without
modification inside a compressed split archive. MATLAB itself, its runtime
libraries/toolboxes, NVIDIA drivers and full CUDA/OpenCV development toolkits
are not redistributed in this repository.
