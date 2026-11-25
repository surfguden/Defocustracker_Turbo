function dat = scaling_linear_apply(scaling,dat)

if ~strcmp(scaling.type,'linear')
    fprintf(2, 'Scaling not compatible \n')
    return
end

if ismember('x',dat.Properties.VariableNames)
     %fprintf(2, 'Data already scaled \n')
    return
end

%%
dat.X = dat.X*scaling.X_to_x;
dat.Y = dat.Y*scaling.Y_to_y;
dat.Z = dat.Z*scaling.Z_to_z;

%%% This options was discontinued since not clear enough. If activated check
%%% also the function scaling_linear_unscale
% if scaling.Z_to_z<0
%     dat.Z = -scaling.Z_to_z+dat.Z;
% end

%%% this correction is needed to account for different tracking steps in the
%%% tracking. If modified, check also the function scaling_linear_unscale.
if isfield(dat.Properties.UserData,'TrackingStep')
    tracking_step = dat.Properties.UserData.TrackingStep;
elseif sum(dat.id)==0
    tracking_step = 1;
else
    tracking_step = 1;
    fprintf(2, 'Tracking step not available. Set to 1. \n')
end
    
dat.DX = dat.DX*scaling.X_to_x/(scaling.dt*tracking_step);
dat.DY = dat.DY*scaling.Y_to_y/(scaling.dt*tracking_step);
dat.DZ = dat.DZ*scaling.Z_to_z/(scaling.dt*tracking_step);

%%
[~, ind] = ismember('X',dat.Properties.VariableNames);
dat.Properties.VariableNames{ind} = 'x';
dat.Properties.VariableUnits{ind} = scaling.unit;
[~, ind] = ismember('Y',dat.Properties.VariableNames);
dat.Properties.VariableNames{ind} = 'y';
dat.Properties.VariableUnits{ind} = scaling.unit;
[~, ind] = ismember('Z',dat.Properties.VariableNames);
dat.Properties.VariableNames{ind} = 'z';
dat.Properties.VariableUnits{ind} = scaling.unit;

[~, ind] = ismember('DX',dat.Properties.VariableNames);
dat.Properties.VariableNames{ind} = 'vx';
dat.Properties.VariableUnits{ind} = [scaling.unit,'/s'];
[~, ind] = ismember('DY',dat.Properties.VariableNames);
dat.Properties.VariableNames{ind} = 'vy';
dat.Properties.VariableUnits{ind} = [scaling.unit,'/s'];
[~, ind] = ismember('DZ',dat.Properties.VariableNames);
dat.Properties.VariableNames{ind} = 'vz';
dat.Properties.VariableUnits{ind} = [scaling.unit,'/s'];

dat.Properties.UserData.Scaling = scaling;

