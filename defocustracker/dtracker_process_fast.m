function [dat,process_time] = dtracker_process_fast(model,imageset,frame_index)
%DTRACKER_PROCESS_FAST RAM-input positioning with normal DefocusTracker outputs.
%   DAT = DTRACKER_PROCESS_FAST(MODEL,IMAGESET) processes every preloaded frame.
%   [DAT,T] = DTRACKER_PROCESS_FAST(MODEL,IMAGESET,FRAME_INDEX) selects frames.
%   Tracking and scaling use dtracker_postprocess exactly as dtracker_process.
%
%   MODEL.processing.fast.backend: 'auto' (default), 'cpu', or 'cuda'. CUDA
%   is a preference: initialization/runtime CUDA failure falls back to CPU.
%   MODEL.processing.fast.cpu_threads: 0 (default, detected) or 1..64.
%   Legacy use_gpu/use_parfor flags do not select this function's backend.
%   images_in_guess_step selects the template count (normally 9).
%
%   T measures RAM input preparation plus positioning wall time; excludes
%   context setup, table assembly, tracking and scaling. Overlapping GPU
%   stages are never summed. UserData.Fast contains the other timings and
%   actual backends. Unsupported configurations use ordinary serial CPU
%   dtracker_process, retaining its original process_time definition.
%   No trajectories or calibration contexts persist between calls.

narginchk(2,3);call_started=tic;
validateattributes(model,{'struct'},{'scalar'},mfilename,'model');
validateattributes(imageset,{'struct'},{'scalar'},mfilename,'imageset');
assert(isfield(imageset,'n_frames'),'DefocusFast:InvalidInput','imageset.n_frames is required.');
validateattributes(imageset.n_frames,{'numeric'},{'scalar','integer','nonnegative','finite'});
if nargin<3,frame_index=1:imageset.n_frames;end
if isempty(frame_index)
    validateattributes(frame_index,{'numeric'},{});
else
    validateattributes(frame_index,{'numeric'},{'vector','integer','positive','finite','real'});
end
frame_index=double(frame_index(:).');
% Normal process_1 ignores indices after its last in-range index.
last=find(frame_index<=imageset.n_frames,1,'last');
if isempty(last),frame_index=[];else,frame_index=frame_index(1:last);end
assert(all(frame_index<=imageset.n_frames),'DefocusFast:InvalidInput','An out-of-range index occurs before an in-range index.');
[backend,threads]=options(model);
meta=struct('requested_backend',backend,'actual_backend',"none", ...
    'frame_index',frame_index,'frame_backend',strings(0,1),'fallback_reason',"", ...
    'cpu_threads',threads,'template_indices',[],'setup_s',0, ...
    'input_preparation_s',0,'localization_s',0,'assembly_s',0, ...
    'tracking_s',0,'scaling_s',0,'total_wall_s',0, ...
    'timing_scope',"RAM preparation + positioning wall time; setup/table assembly/tracking/scaling excluded; no sum of overlapping stages", ...
    'implementation',"dtracker_process_fast v1; dtracker_postprocess linking");
if isempty(frame_index)
    dat=dataset_create();process_time=0;
    if isfield(model,'parameter')&&isfield(model.parameter,'images_boundary')
        dat.Properties.UserData.ParticleBoundaries=model.parameter.images_boundary;
    end
    dat=finish(dat);return
end
reason=unsupported(model,imageset);
if strlength(reason)>0,[dat,process_time]=ordinary(reason);return,end
part=tic;images=cell(numel(frame_index),1);image_size=[];
for k=1:numel(frame_index)
    image=imageset_read(imageset,frame_index(k),'raw');
    if ~ismatrix(image)||~isreal(image)||issparse(image)||isa(image,'gpuArray')
        reason="Fast path requires real full 2-D host images.";break
    end
    if ~isa(image,'uint16')
        if ~isnumeric(image)||any(~isfinite(image(:)))||any(image(:)<0|image(:)>65535)||any(image(:)~=fix(image(:)))
            reason="Input cannot be converted losslessly to uint16.";break
        end
        image=uint16(image);
    end
    if isempty(image_size),image_size=size(image);end
    if ~isequal(size(image),image_size),reason="Fast path requires equal frame dimensions.";break,end
    images{k}=image;
end
meta.input_preparation_s=toc(part);
if strlength(reason)>0,[dat,process_time]=ordinary(reason);return,end
% MATLAB class definitions must live outside private directories.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'defocustracker','backends'));
for binaryFolder={fullfile(root,'bin','cpu'),fullfile(root,'bin','cuda')}
    if isfolder(binaryFolder{1}),addpath(binaryFolder{1});end
end
part=tic;
try
    tracker=RealtimeTracker(model,image_size,Backend=backend,CpuThreads=threads, ...
        TemplateCount=model.processing.images_in_guess_step,Workers=4, ...
        FusedDepth=true,Lookahead=true,PersistentCpuThreads=min(12,threads));
catch failure
    % Invalid calibration/model data must remain an error, not an alleged
    % hardware fallback. Only missing/unloadable native modules are caught.
    if ~ismember(string(failure.identifier),["MATLAB:UndefinedFunction","MATLAB:invalidMEXFile"]),rethrow(failure),end
    meta.setup_s=toc(part);[dat,process_time]=ordinary(string(failure.message));return
end
cleanup=onCleanup(@() delete(tracker));
meta.setup_s=toc(part);info=tracker.info();
if ~info.cpu_optimized_ready
    reason=string(info.cpu_optimization_reason);clear cleanup
    [dat,process_time]=ordinary(reason);return
end
meta.template_indices=tracker.SelectedIndices;
progressWidth=fprintf('Processing image 1/%d ...',numel(frame_index));
progressCleanup=onCleanup(@()fprintf('\n'));
part=tic;[values,sequence]=tracker.localizeSequence(images,OnFrame=@reportProgress);meta.localization_s=toc(part);
clear progressCleanup
info=tracker.info();meta.frame_backend=string(sequence.frames.backend);
used=unique(meta.frame_backend,'stable');
if numel(used)==1,meta.actual_backend=used;else,meta.actual_backend="mixed";end
meta.fallback_reason=string(info.fallback_reason);
meta.sequence=sequence;process_time=meta.input_preparation_s+meta.localization_s;
part=tic;count=cellfun(@(v) size(v,1),values);dat=dataset_create(sum(count));offset=0;
for k=1:numel(values)
    rows=offset+(1:count(k));dat.fr(rows)=frame_index(k);
    dat{rows,{'X','Y','Z','cm'}}=values{k};offset=offset+count(k);
end
dat.Properties.UserData.ParticleBoundaries=model.parameter.images_boundary;
meta.assembly_s=toc(part);clear cleanup
dat=finish(dat);

    function reportProgress(index,~,~)
        fprintf(repmat('\b',1,progressWidth));
        progressWidth=fprintf('Processing image %d/%d ... done',index,numel(frame_index));
        drawnow limitrate nocallbacks;
    end
    function result=finish(result)
        part=tic;
        if isfield(model,'tracking'),result=dtracker_postprocess(model.tracking,result,frame_index);end
        meta.tracking_s=toc(part);part=tic;
        if isfield(model,'scaling'),result=dtracker_postprocess('scale',result,model.scaling);end
        meta.scaling_s=toc(part);meta.total_wall_s=toc(call_started);
        result.Properties.UserData.Fast=meta;
    end
    function [result,seconds]=ordinary(why)
        local_model=model;
        local_model.processing.use_gpu=false;local_model.processing.use_parfor=false;
        part=tic;[result,seconds]=dtracker_process(local_model,imageset,frame_index);
        meta.actual_backend="vanilla_cpu";meta.frame_backend=repmat("vanilla_cpu",numel(frame_index),1);
        meta.fallback_reason=why;meta.vanilla_call_s=toc(part);
        meta.localization_s=seconds;
        meta.timing_scope="Delegated dtracker_process process_time; vanilla call includes its own postprocessing; phase times unavailable";
        meta.tracking_s=NaN;meta.scaling_s=NaN;meta.total_wall_s=toc(call_started);
        result.Properties.UserData.Fast=meta;
    end
end

function [backend,threads]=options(model)
backend="auto";threads=0;
if isfield(model,'processing')&&isfield(model.processing,'fast')
    fast=model.processing.fast;validateattributes(fast,{'struct'},{'scalar'});
    names=fieldnames(fast);
    assert(all(ismember(names,{'backend','cpu_threads'})),'DefocusFast:InvalidInput','Unknown model.processing.fast option.');
    if isfield(fast,'backend')
        assert((ischar(fast.backend)&&isrow(fast.backend))||(isstring(fast.backend)&&isscalar(fast.backend)), ...
            'DefocusFast:InvalidInput','fast.backend must be auto, cpu, or cuda.');
        backend=string(fast.backend);
    end
    if isfield(fast,'cpu_threads'),threads=fast.cpu_threads;end
end
assert(isscalar(backend)&&ismember(backend,["auto","cpu","cuda"]),'DefocusFast:InvalidInput','fast.backend must be auto, cpu, or cuda.');
validateattributes(threads,{'numeric'},{'scalar','integer','nonnegative','finite','<=',64});
if threads==0
    threads=str2double(getenv('NUMBER_OF_PROCESSORS'));
    if ~isfinite(threads)||threads<1,threads=1;end
    threads=min(64,floor(threads));
end
threads=double(threads);
end

function reason=unsupported(model,imageset)
reason="";
if ~isfield(imageset,'im_type')||~strcmp(imageset.im_type,'preloaded_imageset')
    reason="Native acceleration requires a preloaded imageset.";return
end
if ~ismember(string(model.method),["method_1","normalized_crosscorrelation_3d"])
    reason="Native acceleration supports normalized_crosscorrelation_3d.";return
end
t=model.training;p=model.processing;
if t.median_filter~=3||t.gauss_filter>1||t.imresize~=1||p.n_iteration~=1||p.walking_step==0||p.bkg_image_mean~=0||~isempty(p.roi)
    reason="Native acceleration requires median3, no Gaussian/resize/ROI/background subtraction, and one 3-D iteration.";return
end
n=p.images_in_guess_step;
if ~isscalar(n)
    reason="Explicit template-index vectors are delegated to ordinary DefocusTracker.";return
end
validateattributes(n,{'numeric'},{'scalar','positive','integer','finite','<=',numel(model.parameter.images)});
if any(mod(size(model.parameter.images{1}),2)==0)
    reason="Native acceleration requires odd template dimensions.";
end
end
