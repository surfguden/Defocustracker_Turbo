%% Big-data: choose a training particle, train, and process RAM-loaded images
% Run from the examples folder, or use run('examples/example_big_data.m') from the repo root.
nFrames = 21; % All measurement frames in the example dataset.
processingDevice = 'auto'; % The library uses CPU or GPU acceleration; 'auto' selects GPU when available, otherwise CPU.

%% 1 - Setup and example data
addpath('..',genpath('../defocustracker'));
% Place the example data in <repo>/example data/Big-data/, containing Scan-step2um/ and Drop-UPW/.
dataRoot = '../example data/Big-data'; % Change this path if the data is stored elsewhere.
calibrationFolder = fullfile(dataRoot,'Scan-step2um');
imageFolder = fullfile(dataRoot,'Drop-UPW');
calibrationFiles = dir(fullfile(calibrationFolder,'*.tif'));

%% 2 - Select one fixed training crop and a calibration-frame range
trainingBox = pick_training_particle(calibrationFolder);
if isempty(trainingBox)
    disp('Training selection cancelled.');
    return
end
calibrationFiles = calibrationFiles(trainingBox.start_frame:trainingBox.stop_frame);

%% 3 - Train with normal DefocusTracker (outside the processing timer)
trainingImages = dtracker_create('imageset',calibrationFolder,{calibrationFiles.name});
trainingPoints = dtracker_create('dataset',trainingImages.n_frames);
trainingPoints.fr(:) = 1:trainingImages.n_frames;
trainingPoints.X(:) = trainingBox.X;
trainingPoints.Y(:) = trainingBox.Y;
trainingPoints.Z(:) = linspace(0,1,trainingImages.n_frames); % Normalized depth.
model = dtracker_create('model','method_1');
model.training.imwidth = trainingBox.width;
model.training.imheight = trainingBox.height;
model.training.median_filter = 3;
model.training.smoothing = .2;
model = dtracker_train(model,trainingImages,trainingPoints);
dtracker_show(model);

%% 4 - Load all selected measurement images before processing
images = cell(nFrames,1);imageFiles = cell(nFrames,1);
for frame = 1:nFrames
    imageFiles{frame} = fullfile(imageFolder,sprintf('B%05d.tif',frame));
    images{frame} = imread(imageFiles{frame});
end
imageset = dtracker_create('preloaded_imageset',images);

%% 5 - Fast particle positioning
model.processing.fast.backend = processingDevice;
[dat,process_time] = dtracker_process_fast(model,imageset);

%% 6 - Track particles before scaling
% Order is important: track BEFORE scaling, while X/Y and their bounding box are in pixels.
% Z and its bounding-box limits are still normalized calibration depth (0 to 1).
tracking = dtracker_create('tracking');
tracking.bounding_box = [-30 30 -30 30 -.12 .12]; % [DXmin DXmax DYmin DYmax DZmin DZmax]
dat = dtracker_postprocess(tracking,dat);

%% 7 - Scale positions and velocities to physical units
% WTE2 scaling: X/Y in um per pixel, calibration step 2 um, refractive index 1.33.
scaling = dtracker_create('scaling');
scaling.X_to_x = 1.2659;
scaling.Y_to_y = 1.1965;
scaling.Z_to_z = -(trainingImages.n_frames-1)*2*1.33;
scaling.dt = 10; % Seconds between measurement frames.
scaling.unit = 'um';
dat = dtracker_postprocess('scale',dat,scaling);

%% 8 - Plot particle positions
figure('Name','Big-data particle positions','NumberTitle','off');
dtracker_show(dat,'plot_3d_tracks');
