# How the C++ and CUDA modules work

The MATLAB-facing API is `dtracker_process_fast(model,imageset,frame_index)`.
It reads RAM-resident images, manages temporary processing contexts and builds
the ordinary dataset table. C++ implements the numerical CPU work; `.cu` files
contain CUDA host/device code. MEX is the boundary that translates MATLAB
arrays and model structures into these implementations. Core numerical code
does not load image files or call MATLAB functions.

## Processing stages

1. **Preprocess.** Apply the supported symmetric median3 operation using native
   CPU code. Keep the caller's input unchanged. Calibration was already trained
   using the normal MATLAB training functions.
2. **Find XY candidates.** Correlate the full frame with a selected calibration
   subset, usually nine templates. Normalized cross-correlation combines a
   correlation numerator with local image mean/variance normalization.
   Threshold peaks and retain candidate/template identities and scores.
3. **Estimate depth.** Match selected-template similarities against the
   calibration similarity map. Cache calibration statistics and use native
   matching rather than repeated MATLAB calls. Process the selected candidates
   in the required strongest-first/exclusion order.
4. **Refine XYZ.** Extract local candidate crops, compare against calibration
   templates, walk adaptively through depth and interpolate the final position.
   Reuse crop statistics, buffers, spectra and previously evaluated peaks.
   Handle overlapping candidates and remove accepted particles from the working
   image. Dependencies preserve ordering when suppression could affect another
   particle; independent work can run concurrently.
5. **Return to MATLAB.** Assemble XYZ/confidence into normal table columns,
   call `dtracker_postprocess` for linking if requested, then call its scaling
   operation if requested. The fast modules contain no required trajectory
   linker in this release.

Cross-correlation is not implemented as one universal operation. Full-frame
search benefits from cached FFT work and tiled execution. Local refinement
uses cached FFTs or compact spatial correlation as appropriate to the operation.
The code preserves the original full-map versus valid-only correlation
conventions where required by the depth/refinement workflow.

## CPU implementation

`src/cpu/cpu_candidates.cpp` handles candidate correlation; `initial_depth.cpp`
handles initial matching and exclusion. `cpu_correlation.cpp` caches template
spectra, per-crop means/norms and peak results. `cpu_core.cpp` reuses shared
geometry/refinement mathematics and schedules independent particle work through
`workers.cpp`. Calibration storage is shared, while workers own mutable crop
buffers. AVX2 operations are guarded with scalar alternatives in the CPU code.

`src/core/tracker.cpp`, `candidate_cpu.cpp` and `correlation_cpu.cpp` also supply
the shared/reference operations needed by context initialization and the depth
helper. These are runtime dependencies, not a second user-facing processing API.
OpenCV provides optimized CPU primitives; the CPU profile uses a separate
CUDA-disabled static OpenCV build so CPU MEX files do not require CUDA DLLs.

## CUDA implementation

`src/cuda/candidates.cu` and `candidate_engine.cuh` implement resident full-frame
search with cached calibration spectra and normalization data. The frame and
template data are reused on the GPU rather than transferred per correlation.

`resident.cu` incorporates the double-precision correlation kernels from
`correlation_cuda.cu`, plus fused crop preparation and depth-walk kernels.
Several worker streams/graphs refine independent particles concurrently.
Adaptive depth decisions happen on the GPU through CUDA conditional graphs,
reducing host synchronization. Immutable calibration buffers are shared between
workers. Only required candidate/depth/XYZ information crosses the host/device
boundary; suppression remains part of the dependency-aware refinement process.

The MATLAB wrapper uses four resident workers, fused depth walking and
two-context candidate lookahead. Next-frame preparation/search can overlap
current-frame refinement. Persistent CPU helper threads perform preprocessing
and initial depth work. There are therefore CPU stages even in the CUDA path;
CUDA availability alone does not guarantee a particular frame rate.

## MEX modules and fallback

| Binary | Role |
|---|---|
| `defocus_cpu_mex` | Shared CPU context, model validation and reference operations |
| `cpu_refresh_mex` | Updated CPU preprocessing, search, depth and refinement |
| `workflow_pool_mex` | Persistent CPU preprocessing/depth helper for the GPU pipeline |
| `candidates_uint16_mex` | CUDA full-frame candidate search and resident image descriptors |
| `resident_mex` | CUDA dependency scheduling, XYZ refinement, depth walking and suppression |

The three CPU MEX files live in `bin/cpu`; the two optional CUDA MEX files live
in `bin/cuda`. CPU initialization is independent of CUDA. If CUDA initialization
fails, the wrapper records the reason and uses the optimized CPU implementation.
If CUDA fails during a sequence, it drains pending work, retains completed
results and reprocesses unreturned frames from the original RAM images on CPU.
Invalid arguments remain errors. If the optimized CPU modules are unavailable,
the adapter delegates the complete call to normal MATLAB CPU processing.

Contexts are created once per API call and destroyed before it returns. GPU
data descriptors have explicit producer/consumer lifetimes: borrowed image
storage cannot be reused until the consumer has copied it. No calibration
context, working-image state or track identity is silently reused across calls.
