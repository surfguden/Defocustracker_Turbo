function corr_data = find_correction(XYZ,DX,DY,XY0,XYc,scal)


% center the data on X0 and Y0
XYZr = XYZ-[XY0(1); XY0(2); 0];

% fit the parabolic surface
[xData, yData, zData] = prepareSurfaceData( XYZr(1,:), XYZr(2,:), XYZr(3,:) );
ft = fittype( 'poly22' );
opts = fitoptions( 'Method', 'LinearLeastSquares' );
opts.Robust = 'Bisquare';
f = fit( [xData, yData], zData, ft, opts );

% extract the field curvature coefficients
pcurv.c20 = f.p20;
pcurv.c02 = f.p02;
pcurv.c11 = f.p11;
pcurv.c00 = f.p00;

% extract the plane coefficients
a10_f = f.p10;
a01_f = f.p01;

% find the final axis orientation
ix = [DX*scal.X_to_x; DY*scal.Y_to_y; (DX*a10_f+DY*a01_f)*scal.Z_to_z]; 
jx = [-DY*scal.X_to_x; DX*scal.Y_to_y; (-DY*a10_f+DX*a01_f)*scal.Z_to_z];
kx = cross(ix,jx);

k = kx/norm(kx);
i = ix/norm(ix);
j = cross(k,i);

% find the inverse rotation
Rot = [i';j';k'];


%%

corr_data.X0 = XY0(1);
corr_data.Y0 = XY0(2);
corr_data.pcurv = pcurv;
corr_data.Rot = Rot;
corr_data.Xc = XYc(1);
corr_data.Yc = XYc(2);
corr_data.scal = scal;
