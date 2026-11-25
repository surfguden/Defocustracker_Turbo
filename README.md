# Defocustracker Turbo

The aim of this repository is to provide a DefocusTracker-equivalent workflow
that runs faster on large datasets, while retaining familiar MATLAB inputs,
outputs, training and postprocessing. The fast 3-D positioning implementation
uses C++ and optional CUDA acceleration, with a CPU alternative.

The normal `dtracker_create`, `dtracker_train`, `dtracker_process` and
`dtracker_postprocess` functions are included. `dtracker_process_fast` accepts
the same model/imageset style, selects an available CUDA backend, and falls
back to CPU. Tracking between frames and physical scaling use the normal
MATLAB `dtracker_postprocess` implementation.

See [methods we explored](docs/methods_tested.md) for a short account of what
helped, what did not, and which approaches are used in this release. Detailed
old results and validation records are not part of the repository release.

## Start in MATLAB

From the repository root:

```matlab
run(fullfile('examples','example_big_data.m'))
```

The example uses the external WTE2 **Big-data** dataset. Put its `Scan-step2um`
and `Drop-UPW` directories inside `example data/Big-data`, or edit the single relative
`dataRoot` path at the top of the example to use an existing copy.

Dataset images are not included in Git. The example opens a training-particle
picker, trains a model using the chosen calibration range, loads measurement
images into RAM, positions particles, links them with normal MATLAB tracking,
then scales positions/velocities using the original WTE2 example's factors.
It processes all21 measurement frames by default; reduce `nFrames` for a shorter run.
Tracking must precede scaling: XY search bounds are in pixels and Z bounds in
normalized calibration depth. Scaling then uses the WTE2 XY pixel factors,
2um calibration step, refractive index1.33 and10s measurement-frame interval.
Depth is relative to the first selected calibration plane. These factors are
specific to this example dataset. The picker starts with a centered151x151-pixel box,
reduced to fit smaller images, and the complete calibration-frame range.

## Training-particle picker

`defocustracker/tools/pick_training_particle.m` provides a full-image view
and a crop preview. Pass a calibration folder; the picker finds and sorts its
`.tif` images. Existing image-filename lists are also accepted. Drag the
rectangle to move it, or drag any edge/corner to
resize. Numeric X/Y are pixel centres; width/height snap to odd pixel counts.
Start/Stop frame select an inclusive calibration range of at least three
planes. The slider can inspect the complete stack, including excluded planes.

Initial settings are optional: `pick_training_particle(calibrationFolder)`
starts with the centered default box. Pass a second structure to override
individual `X`, `Y`, `width`, `height`, `start_frame` or `stop_frame` fields.

The **Image LUT** menu offers Rossi-Barnkob, Percentile (1-99%), Min-max and
Manual Linear contrast. The single **Auto once** button calculates limits
from the current full image using the selected automatic algorithm. The
Rossi-Barnkob default follows the original DefocusTracker viewer, crediting
Massimiliano Rossi and Rune Barnkob: median3-filtered statistics,
black zero, white three times the median, with its low-background fallback.
Percentile contrast clips the darkest/brightest1% to reduce outlier influence.
Min-max uses the entire intensity range. Black/White fields are always editable;
editing them selects Manual Linear. Limits stay fixed across frames, and
changing the algorithm alone does not change contrast. Both views share the
same linear LUT; image/crop/training pixels are unchanged. Initial contrast is
calculated once on the selected start frame. The returned structure contains
`X`, `Y`, `width`, `height`, `start_frame`, and `stop_frame`.

## Use an existing model and RAM images

```matlab
imageset = dtracker_create('preloaded_imageset', frames); % cell array
[dat, positioning_s] = dtracker_process_fast(model, imageset);

% Optional settings; auto is the default.
model.processing.fast.backend = 'cpu'; % auto / cpu / cuda
model.processing.fast.cpu_threads = 0;  % 0 = detect, or explicitly 1..64
[dat, positioning_s] = dtracker_process_fast(model, imageset, [1 3 5]);
```

The table has the normal `fr X Y Z DX DY DZ id cm` columns. A `model.tracking`
field enables standard linking; `model.scaling` enables standard scaling and
its usual lowercase coordinate/velocity columns. Without tracking, ID and
displacement columns are zero. Calls do not retain trajectories between calls.

`dat.Properties.UserData.Fast` records the actual backend for each selected
frame, fallback reason, template indices, CPU threads and timing phases.
The second output measures RAM preparation and positioning wall time, excluding
context setup, table assembly, tracking, scaling and cleanup. The metadata's
`total_wall_s` includes those call phases. Disk loading happens before the call
when using a preloaded imageset. Overlapping CUDA stage times must not be added.

Native CPU/CUDA processing shows one updating `Processing image n/x ... done`
line as frames complete, including recovery from CUDA to CPU. Its display
overhead is included in positioning time. Calls delegated to ordinary MATLAB
processing retain that implementation's existing progress messages.

Fast backend selection is independent of the old `use_gpu`/`use_parfor` flags.
Auto currently selects usable CUDA by availability; it does not benchmark
which device would be fastest on a particular machine.

## Source and compiled code

```text
defocustracker/    Normal MATLAB library, fast API, backend wrappers and tools
src/              Uncompiled C++/CUDA source and MEX interfaces
examples/         One Big-data example
docs/             Native-module and build explanations
build/            Local compiler intermediates (ignored)
.local/           Local build SDKs, tools and research history (ignored)
bin/cpu/          Included Windows CPU MEX modules
bin/cuda/         Included CUDA MEX modules and compressed runtime libraries
bin/licenses/     Notices for bundled third-party components
example data/     Local datasets (ignored)
```

A Windows clone includes ready-to-run MEX modules built for MATLAB R2026a;
no compiler or CUDA toolkit installation is needed to use this included build.
`setup_defocustracker_turbo` unpacks the bundled GPU runtime on first use and
sets its DLL search path for the current MATLAB session. The archive is split
into four ordinary Git files; Git LFS is not required. Startup verifies archive
and extracted library hashes once, then reuses the generated local cache.
The package includes about245MB of binary payload and about349MB of extracted
runtime libraries. Both CPU and optional CUDA modules are included. See
[the binary package details](bin/README.md), [building the modules](docs/building.md) and
[how the C++/CUDA modules work](docs/native_modules.md).

## Requirements and supported mode

The original MATLAB workflow requires Image Processing, Curve Fitting and
Statistics and Machine Learning toolboxes. The fast MATLAB wrappers target
modern MATLAB (developed with R2026a); the included upstream code has its own
older-version compatibility history. Native build support in this release is
Windows x64/MSVC. The bundled CPU modules require no CUDA. The bundled CUDA
modules target `sm_120` (the development RTX5070) and require a compatible
NVIDIA GPU/driver plus MATLAB Parallel Computing Toolbox GPU support.
They include the needed OpenCV, cuFFT and JIT-link runtime libraries. CUDA
toolkit/OpenCV development installations are needed only when rebuilding.

The fast mode supports full-frame 3-D localization with median3, Gaussian
setting <=1, resize1, one iteration, nonzero depth walking, no ROI/background
subtraction, and odd calibration-template dimensions. The template count comes
from `model.processing.images_in_guess_step` (normally nine). Explicit index
vectors are delegated to ordinary processing. RAM frames must have equal
dimensions. uint16 is retained; other numeric inputs are converted only when
every value is exactly representable as uint16. Unsupported configurations or
unavailable CPU modules delegate to ordinary MATLAB CPU processing.

Source/binary compatibility and CUDA feature support depend on the machine;
automatic CPU fallback does not imply every GPU can run the CUDA path. The
package is a batch processing API, not a camera streaming application.

Local dependencies/research history under `.local/`, datasets, build
intermediates, extracted runtime caches, tests, validation records and research
outputs remain outside version control. Only the explicitly packaged binaries
under `bin/` are included.
Upstream attribution and license details are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
