function varargout = dtracker_postprocess(what_to_do,varargin)
% Functions:
%   dtracker_postprocess(tracking,dataset,frame_index)
%   dtracker_postprocess('compare_true_values',dataset_meas,dataset_true,par)
%   dtracker_postprocess('min_cm',dataset,min_cm)
%   dtracker_postprocess('min_n_id',dataset,min_n_id)
%   dtracker_postprocess('scale',dataset)
%   dtracker_postprocess('smooth_tracks',dataset,[span_x span_y span_z])
%   dtracker_postprocess('smooth_piv_timeseries',dataset,[span_X span_Y])
%   dtracker_postprocess('unscale',dataset)
%   dtracker_postprocess('update_id',dataset)

% to be checked/improved:
%   dtracker_postprocess('nexscale',dataset,P)

if isstruct(what_to_do)
    tracking = what_to_do;
    what_to_do = ['tracking_',tracking.type];
end

switch what_to_do  
    
    case 'compare_true_values'
        dat_meas = varargin{1};
        dat_true = varargin{2};
        
        if nargin==3
            par.outliers_x = 2;
            par.outliers_y = 2;
            par.outliers_z = 0.1;
            par.delta_points = 20;
        elseif nargin==4
            par = varargin{3};
        end
        if nargout==1
            errors = compare_true(dat_meas,dat_true,par);
            varargout{1} = errors;
        end
        if nargout==2
            [errors, errors_delta] = compare_true(dat_meas,dat_true,par);
            varargout{1} = errors;
            varargout{2} = errors_delta;
        end
           
        
    case 'min_cm'
        this = varargin{1};
        min_cm = varargin{2};
        flag = this.cm>=min_cm;
        this = this(flag,:);
        varargout{1} = this;
        
    case 'min_n_id'
        this = varargin{1};
        min_id = varargin{2};
        uid = nonzeros(unique(this.id));
        flag = false(size(this.id));
        for ii = 1:length(uid)
            flag_min = this.id==uid(ii);
            if sum(flag_min)>=min_id
                flag(flag_min) = 1;
            end
        end
        this = this(flag,:);
        this = update_id(this);
        varargout{1} = this;
        
    case 'tracking_nearest_neighbor'
        this = varargin{1};
        if nargin==2
            frame_index = unique(this.fr);
        else
            frame_index = varargin{2};
        end
        this = ptv_nearest(tracking,this,frame_index);
        this.Properties.UserData.TrackingStep = tracking.tracking_step;
        varargout{1} = this;
        
    case 'scale'
        name_var = inputname(2);
        this = varargin{1};
        if nargin==3
            scal = varargin{2};
        else
            if ~isfield(this.Properties.UserData,'Scaling')
                fprintf(2, 'Scaling metadata missing. \n')
                return
            end
            scal = this.Properties.UserData.Scaling;
        end
        switch scal.type
            case 'linear'
                this = scaling_linear_apply(scal,this);
        end
        
        if nargout==1, varargout{1} = this;
        else, assignin('base',name_var,this)
        end
        
    case 'smooth_piv_timeseries'
        name_var = inputname(2);
        this = varargin{1};
        spans = varargin{2};
        this = smooth_piv(this,spans);
        if nargout==1, varargout{1} = this;
        else, assignin('base',name_var,this)
        end

    case 'smooth_tracks'
        name_var = inputname(2);
        this = varargin{1};
        spans = varargin{2};
        this = smooth_tracks(this,spans);
        if nargout==1, varargout{1} = this;
        else, assignin('base',name_var,this)
        end
        
    
    case 'unscale'
        name_var = inputname(2);
        this = varargin{1};
        if ~isfield(this.Properties.UserData,'Scaling')
            fprintf(2, 'Scaling metadata missing \n')
            return
        end
        
        scal = this.Properties.UserData.Scaling;
        switch scal.type
            case 'linear'
                this = scaling_linear_unscale(this);
        end
        
        if nargout==1, varargout{1} = this;
        else, assignin('base',name_var,this)
        end
        
    case 'update_id'
        name_var = inputname(2);
        this = varargin{1};
        this = update_id(this);
        if nargout==1, varargout{1} = this;
        else, assignin('base',name_var,this)
        end
        
    otherwise 
        disp('Postprocessing not existing!')
end



function this = update_id(this)
iduni = nonzeros(unique(this.id));
Ntrk = length(iduni);
for kk = 1:Ntrk
    this.id(this.id==iduni(kk)) = kk;
end

function this = smooth_tracks(this,spans)
ids = unique(this.id(this.id~=0));
A = table2array(this(:,2:7));
for ii = 1:length(ids)
    
    flag = this.id==ids(ii);
    dumt = this.fr(flag);
    for jj = 1:3
        if spans(jj)>2 && sum(flag)>=spans(jj)

            dumf = smooth(dumt,A(flag,jj),spans(jj),'rlowess');
            A(flag,jj) = dumf;
            A(flag,3+jj) = [diff(dumf); dumf(end)-dumf(end-1)]; 
        end
    end

end

if ismember('vx',this.Properties.VariableNames)
    A(:,4:6) = A(:,4:6)/this.Properties.UserData.Scaling.dt;
end  
this(:,2:7) = array2table(A);

function dat = smooth_piv(dat,spans)
ids = unique(dat.id);
for ii = 1:length(ids)
    flag = dat.id==ids(ii);
    frtemp = dat.fr(flag);
    dat.DX(flag) = smooth(frtemp,dat.DX(flag),spans(1),'rlowess');
    dat.DY(flag) = smooth(frtemp,dat.DY(flag),spans(2),'rlowess');
end
