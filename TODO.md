# TODO

- [x] Bundle ready-to-run Windows CPU/CUDA MEX modules and required runtime libraries with dependency notices.
- [ ] Broaden native build support and document additional supported MATLAB/GPU combinations.
- [ ] Reduce standard MATLAB linking and per-call setup/cleanup overhead for realtime use.
- [ ] Provide explicit reusable-context/live-camera interfaces without hidden cross-call state.
- [ ] Broaden accelerated preprocessing/configuration coverage while preserving CPU fallback.
- [ ] Review inherited numerical differences between MATLAB and native depth interpolation.

Show an implementation plan before changing code. Keep dataset images, build
intermediates, tests and validation records outside the release. Include only
the explicitly packaged binaries listed in `bin/manifest.json`.
