classdef CpuRefreshTracker < handle
    properties (SetAccess=private)
        SelectedIndices
    end
    properties (Access=private)
        Handle=[]
        Workers
        Threads
        WarmupSeconds
    end
    methods
        function obj=CpuRefreshTracker(model,imageSize,options)
            arguments
                model struct
                imageSize (1,2) double
                options.Threads (1,1) double {mustBeInteger,mustBePositive}=24
                options.Workers (1,1) double {mustBeInteger,mustBePositive}=24
                options.TemplateCount (1,1) double {mustBeInteger,mustBePositive}=9
            end
            t=model.training;p=model.processing;
            assert(t.median_filter==3&&t.gauss_filter<=1&&t.imresize==1&&p.n_iteration==1&&p.walking_step~=0&&isempty(p.roi)&&p.bkg_image_mean==0, ...
                'DefocusCpuRefresh:UnsupportedMode','Refreshed path requires full-frame median3, no Gaussian/resize/background, and one3D iteration.');
            root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
            if isfolder(fullfile(root,'bin','cpu')),addpath(fullfile(root,'bin','cpu'));end
            available=numel(model.parameter.images);assert(options.TemplateCount<=available);
            indices=round(linspace(1,available,2*options.TemplateCount+1));obj.SelectedIndices=double(indices(2:2:end-1));
            theta=linspace(0,2*pi,16).';radius=p.no_overlap_radius;
            obj.Handle=cpu_refresh_mex('init',model,imageSize,obj.SelectedIndices,options.Threads,options.Workers,[radius*cos(theta),radius*sin(theta)]);
            obj.Threads=options.Threads;obj.Workers=options.Workers;
            obj.WarmupSeconds=cpu_refresh_mex('warmup',obj.Handle);
        end
        function [values,timing]=localize(obj,raw,options)
            arguments
                obj
                raw
                options.Diagnostics logical=false
            end
            started=tic;[values,timing]=cpu_refresh_mex('localize',obj.Handle,raw,options.Diagnostics);timing.native_total_s=timing.total_s;timing.total_s=toc(started);
        end
        function value=diagnostics(obj),value=cpu_refresh_mex('diagnostics',obj.Handle);end
        function value=lastImage(obj),value=cpu_refresh_mex('image',obj.Handle);end
        function value=info(obj),value=struct('backend',"cpu_refreshed",'threads',obj.Threads,'workers',obj.Workers,'selected_indices',obj.SelectedIndices,'synthetic_warmup_s',obj.WarmupSeconds);end
        function delete(obj),if ~isempty(obj.Handle),cpu_refresh_mex('release',obj.Handle);obj.Handle=[];end,end
    end
end
