classdef DefocusTrackerNative < handle
    % Shared native tracker with an always-ready CPU context and optional CUDA.
    properties (SetAccess=private)
        SelectedIndices
        RequestedBackend
        CandidateBackend="cpu"
        RefinementBackend="cpu"
        FallbackReason=""
    end
    properties (Access=private)
        CpuHandle=[]
        CudaHandle=[]
        LastRefinement=""
    end
    methods
        function obj=DefocusTrackerNative(model,imageSize,options)
            arguments
                model (1,1) struct
                imageSize (1,2) double {mustBeInteger,mustBePositive}
                options.Backend (1,1) string {mustBeMember(options.Backend,["auto","cpu","cuda"])}="auto"
                options.TemplateCount (1,1) double {mustBeInteger,mustBePositive}=9
                options.Threads (1,1) double {mustBeInteger,mustBeNonnegative}=0
                options.DepthStrategy (1,1) string {mustBeMember(options.DepthStrategy,["adaptive","full_stack"])}="adaptive"
            end
            root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
            if isfolder(fullfile(root,'bin','cpu')),addpath(fullfile(root,'bin','cpu'));end
            available=numel(model.parameter.images);
            assert(options.TemplateCount<=available,'DefocusNative:TemplateCount','Requested more templates than calibration contains.');
            positions=round(linspace(1,available,2*options.TemplateCount+1));
            obj.SelectedIndices=double(positions(2:2:end-1));
            obj.RequestedBackend=options.Backend;
            % Validate all shared model inputs and initialize CPU before probing CUDA.
            obj.CpuHandle=defocus_cpu_mex('init',model,imageSize,obj.SelectedIndices,options.Threads);
            if options.Backend=="cpu", return, end
            gpuFolder=fullfile(root,'bin','cuda');
            if isfolder(gpuFolder), addpath(gpuFolder); end
            if exist('defocus_cuda_mex','file')~=3
                obj.FallbackReason="CUDA MEX is absent; using CPU.";
                return
            end
            cudaRoot=string(getenv('CUDA_PATH'));
            opencvBin=fullfile(root,'bin','cuda');
            additions=strings(0,1);
            if isfolder(opencvBin), additions(end+1)=opencvBin; end
            if strlength(cudaRoot)>0
                for suffix=["bin","bin/x64"]
                    candidate=fullfile(cudaRoot,suffix);
                    if isfolder(candidate), additions(end+1)=candidate; end
                end
            end
            if ~isempty(additions), setenv('PATH',char(strjoin([additions(:);string(getenv('PATH'))],';'))); end
            try
                obj.CudaHandle=defocus_cuda_mex('init',model,imageSize,obj.SelectedIndices,options.Threads);
                defocus_cuda_mex('depth_strategy',obj.CudaHandle,char(options.DepthStrategy));
                obj.CandidateBackend="cuda";
                % Both native stages were verified on the current CUDA backend.
                obj.RefinementBackend="cuda";
            catch error
                obj.FallbackReason="CUDA initialization failed: "+string(error.message);
            end
        end
        function [candidates,scores,seconds]=candidates(obj,image)
            if obj.CandidateBackend=="cuda"
                try
                    [candidates,scores,seconds]=defocus_cuda_mex('candidates',obj.CudaHandle,image);
                    return
                catch error
                    if ~obj.operationalCudaFailure(error), rethrow(error), end
                    obj.disableCuda(error.message);
                end
            end
            [candidates,scores,seconds]=defocus_cpu_mex('candidates',obj.CpuHandle,image);
        end
        function [data,stats]=refine(obj,image,coarse,options)
            arguments
                obj
                image
                coarse
                options.Diagnostics (1,1) logical=false
            end
            if obj.RefinementBackend=="cuda"
                try
                    [data,stats]=defocus_cuda_mex('refine',obj.CudaHandle,image,coarse,options.Diagnostics);
                    obj.LastRefinement="cuda";
                    stats.fallback=false;
                    return
                catch error
                    if ~obj.operationalCudaFailure(error), rethrow(error), end
                    obj.disableCuda(error.message);
                end
            end
            % Recompute from the original MATLAB inputs after any CUDA failure.
            [data,stats]=defocus_cpu_mex('refine',obj.CpuHandle,image,coarse,options.Diagnostics);
            obj.LastRefinement="cpu"; stats.fallback=strlength(obj.FallbackReason)>0;
        end
        function value=diagnostics(obj)
            if obj.LastRefinement=="cuda", value=defocus_cuda_mex('diagnostics',obj.CudaHandle);
            elseif obj.LastRefinement=="cpu", value=defocus_cpu_mex('diagnostics',obj.CpuHandle);
            else, error('DefocusNative:NoResult','No refined frame is available.'); end
        end
        function value=localCorrelation(obj,image,index,legacy)
            % Stage-2 compatibility service; refinement itself is one native call.
            value=defocus_cpu_mex('correlate',obj.CpuHandle,image,index,legacy);
        end
        function value=lastImage(obj)
            if obj.LastRefinement=="cuda", value=defocus_cuda_mex('image',obj.CudaHandle);
            elseif obj.LastRefinement=="cpu", value=defocus_cpu_mex('image',obj.CpuHandle);
            else, error('DefocusNative:NoResult','No refined frame is available.'); end
        end
        function value=info(obj)
            value=defocus_cpu_mex('info',obj.CpuHandle);
            if obj.RefinementBackend=="cuda"
                cuda_info=defocus_cuda_mex('info',obj.CudaHandle);
                value.depth_walk=cuda_info.depth_walk;
                value.depth_walk_setup_s=cuda_info.depth_walk_setup_s;
            end
            if obj.CandidateBackend==obj.RefinementBackend, value.backend=char(obj.RefinementBackend);
            else,value.backend='mixed';end
            value.requested_backend=obj.RequestedBackend;
            value.candidates_backend=obj.CandidateBackend;
            value.refinement_backend=obj.RefinementBackend;
            value.fallback_reason=obj.FallbackReason;
            value.selected_indices=obj.SelectedIndices;
        end
        function delete(obj)
            if ~isempty(obj.CudaHandle)
                try, defocus_cuda_mex('release',obj.CudaHandle); catch, end
                obj.CudaHandle=[];
            end
            if ~isempty(obj.CpuHandle)
                try, defocus_cpu_mex('release',obj.CpuHandle); catch, end
                obj.CpuHandle=[];
            end
        end
    end
    methods (Hidden)
        function injectCudaFailureAfter(obj,count)
            assert(~isempty(obj.CudaHandle),'DefocusNative:NoCUDA','CUDA is not active.');
            defocus_cuda_mex('test_fail_after',obj.CudaHandle,count);
        end
    end
    methods (Access=private)
        function disableCuda(obj,reason)
            if ~isempty(obj.CudaHandle)
                try, defocus_cuda_mex('release',obj.CudaHandle); catch, end
            end
            obj.CudaHandle=[]; obj.CandidateBackend="cpu"; obj.RefinementBackend="cpu";
            obj.FallbackReason="CUDA processing failed; restarted on CPU: "+string(reason);
        end
    end
    methods (Static,Access=private)
        function yes=operationalCudaFailure(error)
            yes=strcmp(error.identifier,'DefocusNative:CUDA') || strcmp(error.identifier,'MATLAB:invalidMEXFile');
        end
    end
end
