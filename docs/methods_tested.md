# Methods explored

The aim is a DefocusTracker-equivalent workflow that is faster on large
datasets, with automatic GPU selection and a CPU alternative. This is a
qualitative summary of the investigation, not a benchmark or validation record.

| Approach | What we learned / decision |
|---|---|
| Original MATLAB: serial CPU, `parfor` and GPU | Useful compatibility baselines. Parallel/GPU execution helped, but repeated correlation work and MATLAB orchestration remained costly. |
| Standard OpenCV CPU/CUDA template matching | Provided optimized comparison implementations. The custom cached implementation was more effective for our repeated image/template workload. |
| Cached, batched CUDA FFT candidate search | Worked well. Reuse template spectra/plans and share each image's transform/statistics across templates; return peaks and sampled scores instead of complete maps. Kept. |
| Cached multicore CPU processing | Native preprocessing/depth matching, reusable crop buffers, cached FFTs/statistics/peaks and parallel independent particles substantially improved CPU positioning. Kept as the CPU path and CUDA fallback. |
| GPU-resident XYZ refinement and adaptive depth walking | Reusing crops/calibration and batching correlations reduced transfers and synchronizations. Adaptive depth walking was preferred over evaluating the entire calibration stack for each candidate. Kept. |
| Dependency scheduling and frame lookahead | Parallelize independent particles while respecting suppression dependencies; overlap next-frame preparation with current-frame refinement. Kept. More workers and extra GPU warm-up did not consistently improve performance. |
| Restricted XY searches | A bounded initial search changed detections and suppression, so it was rejected. Guarded local-search variants remained heuristics and were not promoted. |
| Persistent mixed-depth GPU variant | Slower than the selected adaptive implementation on the investigated workload; not selected as the default. |
| Faster particle removal alone | Profiling showed removal was a small part of refinement cost. Correlation reuse, depth work and synchronization deserved priority. |
| Native CPU trajectory linker | Worked in the experimental pipeline, but this release uses standard MATLAB `dtracker_postprocess` for linking and scaling. |

A low-rank calibration audit did not establish an adequate small-rank accuracy
guarantee. Some additional SIMD/float-FFT ideas remained unbuilt proposals;
they are not claimed as tested runtime methods or included as selected paths.

Most investigation used WTE2 Big-data, with images already in RAM and a fixed
calibration configuration. Stage-only trials and full-sequence trials are
different workloads. The current example permits another crop/frame range and
does not interpolate the calibration stack, so historical performance does not
automatically apply to every run. Numerical agreement with saved results also
does not establish physical tracking accuracy; small inherited differences from
the MATLAB implementation remain a documented limitation.

The release contains this summary and the selected implementation. Detailed
historical sources, measurements and validation records stay in ignored local
research history, not in the first commit. See [native modules](native_modules.md)
for how the selected CPU/CUDA methods fit together.
