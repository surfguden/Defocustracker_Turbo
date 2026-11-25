function dat = scaling_linear_unscale(dat)


if ismember('X',dat.Properties.VariableNames) 
    fprintf(2, 'Data already unscaled. \n')
    return
end

try 
    scaling = dat.Properties.UserData.Scaling;
catch
    fprintf(2, 'Scaling not available. \n')
    return
end

dat.x = dat.x/scaling.X_to_x;
dat.y = dat.y/scaling.Y_to_y;
dat.z = dat.z/scaling.Z_to_z;

%%% This options was discontinued since not clear enough. If modified, check
%%% also the function scaling_linear_apply.
% if scaling.Z_to_z<0
%     dat.z = 1+dat.z/scaling.Z_to_z;
% end

%%% This correction is needed to account for different tracking steps in the
%%% tracking. If modified, check also the function scaling_linear_apply.
if isfield(dat.Properties.UserData,'TrackingStep')
    tracking_step = dat.Properties.UserData.TrackingStep;
elseif sum(dat.id)==0
    tracking_step = 1;
else
    tracking_step = 1;
    fprintf(2, 'Tracking step not available. Set to 1. \n')
end
    
dat.vx = dat.vx/scaling.X_to_x*scaling.dt*tracking_step;
dat.vy = dat.vy/scaling.Y_to_y*scaling.dt*tracking_step;
dat.vz = dat.vz/scaling.Z_to_z*scaling.dt*tracking_step;

[~, ind] = ismember('x',dat.Properties.VariableNames);
dat.Properties.VariableNames{ind} = 'X';
[~, ind] = ismember('y',dat.Properties.VariableNames);
dat.Properties.VariableNames{ind} = 'Y';
[~, ind] = ismember('z',dat.Properties.VariableNames);
dat.Properties.VariableNames{ind} = 'Z';

[~, ind] = ismember('vx',dat.Properties.VariableNames);
dat.Properties.VariableNames{ind} = 'DX';
[~, ind] = ismember('vy',dat.Properties.VariableNames);
dat.Properties.VariableNames{ind} = 'DY';
[~, ind] = ismember('vz',dat.Properties.VariableNames);
dat.Properties.VariableNames{ind} = 'DZ';

dat.Properties.VariableUnits = {};