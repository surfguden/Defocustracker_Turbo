classdef RealtimeTracker < handle
    % Experimental RAM-input pipeline; always initialize an independent CPU fallback.
    properties (SetAccess=private)
        SelectedIndices
        FallbackReason=""
        GpuReady=false
        Workers=8
        CpuThreads=24
        PersistentCpuThreads=0
        Lookahead=false
        BoundedXY=false
        MixedDepth=false
        FusedDepth=false
    end
    properties (Access=private)
        Model
        ImageSize
        CpuTracker=[]
        OptimizedCpu=[]
        CpuOptimizationReason=""
        LastCpuOptimized=false
        DepthHandle=[]
        CandidateHandle=[]
        SecondCandidateHandle=[]
        ResidentHandle=[]
        LastBackend=""
        LastCoarse=[]
        WarmupInfo=[]
        GpuWarmupSeconds=0
        FailNext=false
        CpuModule='workflow_pool_mex'
        InCallback=false
    end
    methods
        function obj=RealtimeTracker(model,imageSize,options)
            arguments
                model (1,1) struct
                imageSize (1,2) double
                options.GpuWarmupMilliseconds (1,1) double {mustBeInteger,mustBeNonnegative}=0
                options.FusedDepth (1,1) logical=false
                options.MixedDepth (1,1) logical=false
                options.BoundedXY (1,1) logical=false
                options.Workers (1,1) double {mustBeInteger,mustBePositive}=8
                options.CpuThreads (1,1) double {mustBeInteger,mustBePositive}=24
                options.PersistentCpuThreads (1,1) double {mustBeInteger,mustBeNonnegative}=0
                options.Lookahead (1,1) logical=false
                options.Backend (1,1) string {mustBeMember(options.Backend,["auto","cpu","cuda"])}="auto"
                options.TemplateCount (1,1) double {mustBeInteger,mustBePositive}=9
            end
            obj.CpuThreads=options.CpuThreads;
            obj.PersistentCpuThreads=options.PersistentCpuThreads;
            obj.Lookahead=options.Lookahead;
            root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
            obj.Model=model;obj.ImageSize=imageSize;obj.Workers=options.Workers;obj.BoundedXY=options.BoundedXY;obj.MixedDepth=options.MixedDepth;obj.FusedDepth=options.FusedDepth;
            assert(options.Workers<=16,'DefocusRealtime:InvalidInput','Workers must be1..16.');
            assert(isempty(model.processing.roi)&&model.processing.bkg_image_mean==0&&model.training.imresize==1&&model.processing.walking_step~=0, ...
                'DefocusRealtime:UnsupportedMode','This experimental entry point supports full-frame, unresized3D localization without background subtraction.');
            obj.CpuTracker=DefocusTrackerNative(model,imageSize,Backend="cpu",TemplateCount=options.TemplateCount,Threads=obj.CpuThreads);
            obj.SelectedIndices=obj.CpuTracker.SelectedIndices;
            % CPU acceleration is initialized independently of all CUDA modules.
            cpu_refresh=fullfile(root,'bin','cpu');
            if isfolder(cpu_refresh)
                addpath(cpu_refresh);
                try
                    obj.OptimizedCpu=CpuRefreshTracker(model,imageSize,Threads=obj.CpuThreads,Workers=min(obj.CpuThreads,64),TemplateCount=options.TemplateCount);
                catch failure
                    obj.CpuOptimizationReason="Updated CPU path unavailable; retained legacy CPU workflow: "+string(failure.message);
                end
            else
                obj.CpuOptimizationReason="Updated CPU module folder unavailable; retained legacy CPU workflow.";
            end
            if options.Backend=="cpu",return,end
            t=model.training;p=model.processing;
            if t.median_filter~=3||t.gauss_filter>1||t.imresize~=1||p.n_iteration~=1||p.walking_step==0||p.bkg_image_mean~=0||~isempty(p.roi)
                obj.FallbackReason="Experimental fast path requires median3, no Gaussian/resize/ROI/background subtraction, and one 3D iteration.";return
            end
            if any(mod(size(model.parameter.images{1}),2)==0)
                obj.FallbackReason="Resident uint16 median selection requires odd template dimensions; using CPU.";return
            end
            gpuFolder=fullfile(root,'bin','cuda');
            if ~isfolder(gpuFolder)
                obj.FallbackReason="CUDA modules are not installed; using CPU.";return
            end
            addpath(gpuFolder);
            if options.PersistentCpuThreads>0
                obj.CpuModule='workflow_pool_mex';
            end
            additions=string(gpuFolder);cudaRoot=string(getenv('CUDA_PATH'));
            if strlength(cudaRoot)>0
                for suffix=["bin","bin/x64"]
                    candidate=fullfile(cudaRoot,suffix);
                    if isfolder(candidate),additions(end+1)=candidate;end
                end
            end
            setenv('PATH',char(strjoin([additions(:);string(getenv('PATH'))],';')));
            try
                addpath(root);prepare_defocustracker_runtime;
                obj.DepthHandle=feval(obj.CpuModule,'init',model,imageSize,obj.SelectedIndices,obj.CpuThreads);
                theta=linspace(0,2*pi,16).';r=p.no_overlap_radius;
                feval(obj.CpuModule,'prepare_depth',obj.DepthHandle,[r*cos(theta),r*sin(theta)]);
                if options.PersistentCpuThreads>0,feval(obj.CpuModule,'worker_pool',obj.DepthHandle,options.PersistentCpuThreads);end
                obj.CandidateHandle=candidates_uint16_mex('init',model,imageSize,obj.SelectedIndices);
                if options.Lookahead,obj.SecondCandidateHandle=candidates_uint16_mex('init',model,imageSize,obj.SelectedIndices);end
                if options.FusedDepth,obj.ResidentHandle=resident_mex('init',model,imageSize,obj.SelectedIndices,options.Workers,options.BoundedXY,false,true);
                elseif options.MixedDepth,obj.ResidentHandle=resident_mex('init',model,imageSize,obj.SelectedIndices,options.Workers,options.BoundedXY,true);
                else,obj.ResidentHandle=resident_mex('init',model,imageSize,obj.SelectedIndices,options.Workers,options.BoundedXY);end
                obj.WarmupInfo=feval(obj.CpuModule,'warmup',obj.DepthHandle,100);
                if options.GpuWarmupMilliseconds>0,obj.GpuWarmupSeconds=resident_mex('warmup',obj.ResidentHandle,options.GpuWarmupMilliseconds);end
                obj.GpuReady=true;
            catch failure
                obj.releaseFast();obj.FallbackReason="Fast initialization failed; using CPU: "+string(failure.message);
            end
        end
        function [values,timing]=localize(obj,raw,options)
            arguments
                obj
                raw
                options.Diagnostics (1,1) logical=false
            end
            assert(~obj.InCallback,'DefocusRealtime:ReentrantCall','Frame callbacks cannot process another frame on the same tracker.');
            started=tic;obj.LastCoarse=[];obj.LastCpuOptimized=false;frame_reason="";
            if obj.GpuReady&&(~isa(raw,'uint16')||~isequal(size(raw),obj.ImageSize)),frame_reason="Fast input mode unavailable; used CPU for this frame.";end
            if obj.GpuReady&&isa(raw,'uint16')&&isequal(size(raw),obj.ImageSize)
                try
                    part=tic;image=feval(obj.CpuModule,'preprocess',raw,obj.CpuThreads);preprocessing_s=toc(part);
                    part=tic;[candidates,scores,candidate_stats]=candidates_uint16_mex('run',obj.CandidateHandle,image);candidate_s=toc(part);
                    part=tic;[coarse,depth_stats]=feval(obj.CpuModule,'depth',obj.DepthHandle,image,candidates,scores);depth_guess_s=toc(part);
                    part=tic;descriptor=candidates_uint16_mex('device_image',obj.CandidateHandle);
                    [values,refinement_stats]=resident_mex('refine_device',obj.ResidentHandle,descriptor,coarse,options.Diagnostics);refinement_s=toc(part);
                    if obj.FailNext,obj.FailNext=false;error('DefocusRealtime:CUDA','Injected failure after GPU refinement; discard GPU result.');end
                    obj.LastCoarse=coarse;obj.LastBackend="cuda";
                    timing=struct('total_s',toc(started),'preprocessing_s',preprocessing_s,'candidate_s',candidate_s, ...
                        'depth_guess_s',depth_guess_s,'refinement_s',refinement_s,'backend',"cuda",'fallback',false, ...
                        'candidate_stats',candidate_stats,'depth_stats',depth_stats,'refinement_stats',refinement_stats);
                    return
                catch failure
                    if ~contains(failure.identifier,':CUDA')&&~strcmp(failure.identifier,'MATLAB:invalidMEXFile'),rethrow(failure),end
                    obj.releaseFast();obj.FallbackReason="Fast processing failed; restarted original frame on CPU: "+string(failure.message);
                end
            end
            if ~isempty(obj.OptimizedCpu)&&isa(raw,'uint16')&&isequal(size(raw),obj.ImageSize)
                [values,timing]=obj.OptimizedCpu.localize(raw,Diagnostics=options.Diagnostics);obj.LastCpuOptimized=true;
            else
                [data,timing]=native_localize(obj.CpuTracker,obj.Model,raw);values=data{:,{'X','Y','Z','cm'}};
            end
            obj.LastBackend="cpu";timing.total_s=toc(started);timing.backend="cpu";timing.fallback=strlength(obj.FallbackReason)>0||strlength(frame_reason)>0;timing.fallback_reason=obj.FallbackReason+frame_reason;
        end
        function [values,sequence,decisions]=localizeSequence(obj,images,options)
            % One frame in flight; next-frame preparation overlaps resident refinement.
            arguments
                obj
                images cell
                options.PaceFPS (1,1) double {mustBeFinite,mustBeNonnegative}=0
                options.CaptureDecisions (1,1) logical=false
                options.OnFrame=[]
            end
            assert(~obj.InCallback,'DefocusRealtime:ReentrantCall','Frame callbacks cannot process another sequence on the same tracker.');
            assert(isempty(options.OnFrame)||isa(options.OnFrame,'function_handle'),'DefocusRealtime:InvalidInput','OnFrame must be a function handle or empty.');
            count=numel(images);assert(count>0,'DefocusRealtime:InvalidInput','Provide RAM-loaded images.');
            values=cell(count,1);decisions=cell(count,1);
            rows=repmat(struct('input_release_s',0,'preparation_started_s',0,'submission_s',NaN, ...
                'output_s',0,'latency_s',0,'preprocessing_s',NaN,'candidate_s',NaN,'depth_guess_s',NaN, ...
                'submit_call_s',NaN,'complete_call_s',NaN,'depth_native_s',NaN,'depth_nms_s',NaN,'depth_matching_s',NaN, ...
                'callback_invoked_s',NaN,'callback_latency_s',NaN,'backend',"",'fallback',false),count,1);
            started=tic;completed=0;callback_failure=false;
            use_async=obj.GpuReady&&obj.Workers>=2&&all(cellfun(@(x) isa(x,'uint16')&&isequal(size(x),obj.ImageSize),images));
            if use_async
                try
                    prepared=prepare(1);next_image=[];
                    for k=1:count
                        current=prepared;
                        if obj.Lookahead&&k<count
                            if isempty(next_image),next_image=preprocess(k+1);end
                            prepared=search(k+1,next_image);next_image=[];
                        end
                        descriptor=current.descriptor;
                        rows(k).submission_s=toc(started);part=tic;
                        resident_mex('submit_device',obj.ResidentHandle,descriptor,current.coarse,false);
                        rows(k).submit_call_s=toc(part);
                        % submit returns only once the borrowed image has been copied.
                        % The producer can now overwrite its own buffer for k+1.
                        if k<count
                            if obj.Lookahead
                                prepared.coarse=estimate(k+1,prepared);
                                % Only preprocess an input that has actually arrived.
                                if k+2<=count&&(options.PaceFPS==0||toc(started)>=(k+1)/options.PaceFPS),next_image=preprocess(k+2);end
                            else
                                prepared=prepare(k+1);
                            end
                        end
                        part=tic;values{k}=resident_mex('complete',obj.ResidentHandle);rows(k).complete_call_s=toc(part);
                        if obj.FailNext,obj.FailNext=false;error('DefocusRealtime:CUDA','Injected sequence failure; discard pending GPU result.');end
                        obj.LastCoarse=current.coarse;obj.LastBackend="cuda";
                        rows(k).output_s=toc(started);rows(k).latency_s=rows(k).output_s-rows(k).input_release_s;
                        rows(k).backend="cuda";
                        if options.CaptureDecisions,decisions{k}=obj.diagnostics();end
                        completed=k;emit(k);
                    end
                catch failure
                    % Releasing drains pending work before freeing borrowed buffers.
                    obj.releaseFast();
                    if callback_failure||(~contains(failure.identifier,':CUDA')&&~strcmp(failure.identifier,'MATLAB:invalidMEXFile')),rethrow(failure),end
                    obj.FallbackReason="Sequence GPU failure; restarted unreturned original frames on CPU: "+string(failure.message);
                end
            end
            % Also supports machines without CUDA, and restarts unreturned frames on error.
            for k=completed+1:count
                wait_release(k);rows(k).preparation_started_s=toc(started);
                [values{k},frame]=obj.localize(images{k});
                if isfield(frame,'preprocessing_s'),rows(k).preprocessing_s=frame.preprocessing_s;end
                if isfield(frame,'candidate_s'),rows(k).candidate_s=frame.candidate_s;end
                if isfield(frame,'depth_guess_s'),rows(k).depth_guess_s=frame.depth_guess_s;end
                rows(k).backend=frame.backend;rows(k).fallback=frame.fallback;
                rows(k).output_s=toc(started);rows(k).latency_s=rows(k).output_s-rows(k).input_release_s;
                if options.CaptureDecisions,decisions{k}=obj.diagnostics();end
                emit(k);
            end
            elapsed=toc(started);outputs=[rows.output_s].';intervals=diff(outputs);
            steady_fps=NaN;if count>1,steady_fps=(count-1)/(outputs(end)-outputs(1));end
            sequence=struct('frames',struct2table(rows),'wall_s',elapsed,'fps',count/elapsed, ...
                'steady_output_fps',steady_fps,'output_intervals_s',intervals,'pace_fps',options.PaceFPS, ...
                'all_cuda',all(string({rows.backend})=="cuda"),'fallback_frames',nnz([rows.fallback]), ...
                'max_latency_s',max([rows.latency_s]),'mean_latency_s',mean([rows.latency_s]), ...
                'timing_scope','RAM input to returned XYZ/score, including pipeline fill/drain and optional decision capture; initialization/loading excluded; overlapping stage times must not be added');
            function wait_release(index)
                if options.PaceFPS>0
                    rows(index).input_release_s=(index-1)/options.PaceFPS;
                    remaining=rows(index).input_release_s-toc(started);if remaining>0,pause(remaining);end
                end
            end
            function item=prepare(index)
                image=preprocess(index);item=search(index,image);item.coarse=estimate(index,item);
            end
            function image=preprocess(index)
                wait_release(index);rows(index).preparation_started_s=toc(started);
                part=tic;image=feval(obj.CpuModule,'preprocess',images{index},obj.CpuThreads);rows(index).preprocessing_s=toc(part);
            end
            function item=search(index,image)
                producer=obj.CandidateHandle;if obj.Lookahead&&mod(index,2)==0,producer=obj.SecondCandidateHandle;end
                part=tic;[item.candidates,item.scores]=candidates_uint16_mex('run',producer,image);rows(index).candidate_s=toc(part);
                item.image=image;item.descriptor=candidates_uint16_mex('device_image',producer);
            end
            function coarse=estimate(index,item)
                part=tic;[coarse,depth_stats]=feval(obj.CpuModule,'depth',obj.DepthHandle,item.image,item.candidates,item.scores);rows(index).depth_guess_s=toc(part);
                rows(index).depth_native_s=depth_stats.native_s;rows(index).depth_nms_s=depth_stats.nms_s;rows(index).depth_matching_s=depth_stats.matching_s;
            end
            function emit(index)
                if isempty(options.OnFrame),return,end
                obj.InCallback=true;guard=onCleanup(@finished_callback);
                rows(index).callback_invoked_s=toc(started);
                rows(index).callback_latency_s=rows(index).callback_invoked_s-rows(index).input_release_s;
                try
                    options.OnFrame(index,values{index},rows(index));
                catch failure
                    callback_failure=true;rethrow(failure)
                end
                clear guard
            end
            function finished_callback()
                obj.InCallback=false;
            end
        end
        function value=diagnostics(obj)
            if obj.LastBackend=="cuda",value=resident_mex('diagnostics',obj.ResidentHandle);value.coarse=obj.LastCoarse;
            elseif obj.LastBackend=="cpu"
                if obj.LastCpuOptimized,value=obj.OptimizedCpu.diagnostics();else,value=obj.CpuTracker.diagnostics();value.coarse=[];end
            else,error('DefocusRealtime:NoResult','No completed frame'),end
        end
        function value=lastImage(obj)
            if obj.LastBackend=="cuda",value=resident_mex('image',obj.ResidentHandle);
            elseif obj.LastBackend=="cpu"
                if obj.LastCpuOptimized,value=obj.OptimizedCpu.lastImage();else,value=obj.CpuTracker.lastImage();end
            else,error('DefocusRealtime:NoResult','No completed frame'),end
        end
        function value=info(obj)
            value=struct('gpu_ready',obj.GpuReady,'workers',obj.Workers,'cpu_threads',obj.CpuThreads,'cpu_optimized_ready',~isempty(obj.OptimizedCpu),'last_cpu_optimized',obj.LastCpuOptimized,'cpu_optimization_reason',obj.CpuOptimizationReason,'persistent_cpu_threads',obj.PersistentCpuThreads,'lookahead',obj.Lookahead,'selected_indices',obj.SelectedIndices, ...
                'fallback_reason',obj.FallbackReason,'last_backend',obj.LastBackend,'bounded_xy',obj.BoundedXY,'mixed_depth',obj.MixedDepth,'fused_depth',obj.FusedDepth,'cpu_warmup',obj.WarmupInfo,'gpu_warmup_s',obj.GpuWarmupSeconds);
        end
        function delete(obj)
            obj.releaseFast();if ~isempty(obj.OptimizedCpu),delete(obj.OptimizedCpu);obj.OptimizedCpu=[];end
            if ~isempty(obj.CpuTracker),delete(obj.CpuTracker);obj.CpuTracker=[];end
        end
    end
    methods (Hidden)
        function injectCudaFailureNextFrame(obj)
            assert(obj.GpuReady,'DefocusRealtime:NoCUDA','No GPU path available.');obj.FailNext=true;
        end
    end
    methods (Access=private)
        function releaseFast(obj)
            obj.GpuReady=false;
            if ~isempty(obj.ResidentHandle),try,resident_mex('release',obj.ResidentHandle);catch,end,obj.ResidentHandle=[];end
            if ~isempty(obj.CandidateHandle),try,candidates_uint16_mex('release',obj.CandidateHandle);catch,end,obj.CandidateHandle=[];end
            if ~isempty(obj.SecondCandidateHandle),try,candidates_uint16_mex('release',obj.SecondCandidateHandle);catch,end,obj.SecondCandidateHandle=[];end
            if ~isempty(obj.DepthHandle),try,feval(obj.CpuModule,'release',obj.DepthHandle);catch,end,obj.DepthHandle=[];end
        end
    end
end
