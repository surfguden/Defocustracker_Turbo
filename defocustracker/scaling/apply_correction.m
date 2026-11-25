function dat = apply_correction(dat, par)

X0 = par.X0;
Y0 = par.Y0;
pcurv = par.pcurv;
Rot = par.Rot;
scal = par.scal;

if ~ismember('X',dat.Properties.VariableNames)
    disp('Correction not possible. Unscale data')
    return
end

% find the curvature correction
X_curv = dat.X-X0;
Y_curv = dat.Y-Y0;
Z_curv = pcurv.c20*X_curv.^2 + pcurv.c02*Y_curv.^2+...
         pcurv.c11*X_curv.*Y_curv + pcurv.c00;
% subtract the curvature correction
dat.Z = dat.Z-Z_curv;

% center the data in X-Y
dat.X = dat.X-par.Xc;
dat.Y = dat.Y-par.Yc;

% scale the data
dat = scaling_linear_apply(scal,dat);

% create the data arrays to be rotated
xyz = [dat.x(:), dat.y(:), dat.z(:)]';
vxyz = [dat.vx(:), dat.vy(:), dat.vz(:)]';

% rotate the data
xyz = Rot*xyz;
vxyz = Rot*vxyz;

% apply post-correction (if present)
if isfield(par,'xyzc')
    xyz = xyz+par.xyzc;
end


% create the new dat
dat.x(:) = xyz(1,:);
dat.y(:) = xyz(2,:);
dat.z(:) = xyz(3,:);
dat.vx(:) = vxyz(1,:);
dat.vy(:) = vxyz(2,:);
dat.vz(:) = vxyz(3,:);


