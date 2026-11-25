function setup_defocustracker_turbo
% Add this checkout's MATLAB library, tools and available MEX binary folders.
% Call once per MATLAB session. No persistent MATLAB path changes are made.
root=fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root,'defocustracker')));
for folder={fullfile(root,'bin','cpu'),fullfile(root,'bin','cuda')}
    if isfolder(folder{1}),addpath(folder{1});end
end
if ispc&&isfile(fullfile(root,'bin','manifest.json'))
    try
        prepare_defocustracker_runtime;
    catch failure
        warning('DefocusRuntime:CUDAUnavailable', ...
            'CUDA runtime preparation failed; CPU processing remains available: %s',failure.message);
    end
end
fprintf('Defocustracker Turbo: %s\n',root);
end
