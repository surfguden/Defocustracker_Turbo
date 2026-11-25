function [XYZ,DX,DY,XYc] = find_midplane(dat,par)

if ~ismember('X',dat.Properties.VariableNames)
    disp('Correction not possible. Unscale data')
    return
end

%% find and apply XY angle
% no setting parameters here

DM = sqrt(dat.DX.^2+dat.DY.^2);
flag = DM>max(DM)*.7; % consider only "fast" displacements

DX = median(dat.DX(flag));
DY = median(dat.DY(flag));
ang_XY = angle(DX+DY*1i);

XYZ = [dat.X, dat.Y, dat.Z]';
DXYZ = [dat.DX, dat.DY, dat.DZ]';

R_XY = [cos(-ang_XY) -sin(-ang_XY) 0;...
    sin(-ang_XY) cos(-ang_XY) 0; 0 0 1];
XYZ = R_XY*XYZ;
DXYZ = R_XY*DXYZ;

XYcr = median(XYZ(1:2,:),2); % estimate center 
XYZ(1:2,:) = XYZ(1:2,:)-XYcr;

% find the center of the data
XYc = R_XY(1:2,1:2)^(-1)*XYcr;

%%
% ids = unique(dat.id);
% close all
% for ii = 2:50:length(ids)
%     flag = dat.id==ids(ii);
%     plot(dat.X(flag),dat.Y(flag)), hold on
%     plot(XYZ(1,flag),XYZ(2,flag)), hold on
% end

%% YZfit: find profile for central cross-section

DXmin = par.YZfit_DXmin*max(DM); % consider only tracks larger thatn a threshold
flag_id = dat.id~=0;
ids = unique(dat.id(flag_id));

DX_ids = ids*0;
Y_ids = ids*0; % Y positions of the tracks in the central plane (perpendicular to the flow)
Z_ids = ids*0; % Z positions of the tracks in the central plane (perpendicular to the flow)
id_ids = ids*0;

flag_valid = false(size(ids));
for ii = 1:length(ids)
    flag = dat.id==ids(ii);    
    if sum(flag)>=par.YZfit_min_n_id % track must be larger than a given value
        X = XYZ(1,flag);
        DX_ids(ii) = median(DXYZ(1,flag));
        if min(X)<0 && max(X)>0 && DX_ids(ii)>DXmin
            flag_valid(ii) = 1; % the tracks must cross X=0
            id_ids(ii) = ids(ii);
            Y_ids(ii) = interp1(XYZ(1,flag),XYZ(2,flag),0);
            Z_ids(ii) = interp1(XYZ(1,flag),XYZ(3,flag),0);
        end
    end
end
DX_ids = DX_ids(flag_valid);
Y_ids = Y_ids(flag_valid);
Z_ids = Z_ids(flag_valid);
id_ids = id_ids(flag_valid);

% close all
% subplot(1,2,1)
% plot3(DX_ids, Y_ids, Z_ids,'.')
% subplot(1,2,2)
% plot3(XYZ(1,:),XYZ(2,:),XYZ(3,:),'.',...
%     Y_ids*0, Y_ids,Z_ids,'.')

minY = min(XYZ(2,:));
maxY = max(XYZ(2,:));

Ystart = minY+(maxY-minY)*par.YZfit_Yminmax(1);
Yend = minY+(maxY-minY)*par.YZfit_Yminmax(2);
Ys = linspace(Ystart,Yend,par.YZfit_bin_number);
DYs = (Ys(2)-Ys(1))*par.YZfit_bin_size;
Zs = Ys*0;
DXs = Ys*0;

flag_valid = false(size(Ys));
for ii = 1:length(Ys)
    flag = abs(Y_ids-Ys(ii))<DYs/2;
    if sum(flag)>10
        p = polyfit(Z_ids(flag),DX_ids(flag),2);
        Zs(ii) = -p(2)/(2*p(1));
        DXs(ii) = polyval(p,Zs(ii));
        flag_valid(ii) = 1;
    end
end
Ys = Ys(flag_valid);
Zs = Zs(flag_valid);
DXs = DXs(flag_valid);

% Set up fittype and options.
[xData, yData] = prepareCurveData( Ys, Zs );
ft = fittype( 'poly2' );
opts = fitoptions( 'Method', 'LinearLeastSquares' );
opts.Robust = 'Bisquare';
fZ_Y = fit( xData, yData, ft, opts );


%% Find final surface

Xsurf = [];
Ysurf = [];
Zsurf = [];

% OPTION1: only tracks that cross 0
Z_ids_fit = Y_ids*0;
Z_ids_fit(:) = feval(fZ_Y,Y_ids(:));
for ii = 1:length(id_ids)
    DZ_dum = Z_ids_fit(ii)-Z_ids(ii);
    flag = dat.id==id_ids(ii);
    if abs(DZ_dum)<par.XYZfit_DZmin
        Xsurf = [Xsurf; dat.X(flag)];
        Ysurf = [Ysurf; dat.Y(flag)];
        Zsurf = [Zsurf; dat.Z(flag)+DZ_dum];
    end
end

XYZ = [Xsurf, Ysurf, Zsurf]';

    
% OPTION2: Tracks longer than XYZfit_min_track_Xlength (not working..)
% Xlength_all = max(XYZ(1,:))-min(XYZ(1,:));
% 
% for ii = 1:length(ids)
%     flag = dat.id==ids(ii);
%     Ymed = (median(XYZ(2,flag))-minY)/(maxY-minY);
%     Xlength = max(XYZ(1,flag))-min(XYZ(1,flag));
%     if Xlength>par.XYZfit_min_track_Xlength*Xlength_all &&...
%             Ymed>par.YZfit_Yminmax(1) &&...
%             Ymed<par.YZfit_Yminmax(2)
%         p = polyfit(XYZ(1,flag),XYZ(3,flag),2);
%         Z_dum = polyval(p,0);
%         Z_dum_fit = feval(fZ_Y,Ymed);
%         DZ_dum = Z_dum_fit-Z_dum;
%         if abs(Z_dum)<par.XYZfit_DZmin
%             Xsurf = [Xsurf, XYZ(1,flag)];
%             Ysurf = [Ysurf, XYZ(2,flag)];
%             Zsurf = [Zsurf, XYZ(3,flag)+DZ_dum];
%         end
%     end
% end
% XYZ = [Xsurf; Ysurf; Zsurf];
% XYZ(1:2,:) = XYZ(1:2,:)+XYc;
% XYZ = R_XY*XYZ;




%% make final plot

if par.show_plot==1
    figure, set(gcf,'position',[200 100 900 400])
    subplot(1,2,1)
    flag = Y_ids>Ys(1)-DYs & Y_ids<Ys(end)+DYs;  
    flag2 = Y_ids>Ys(1)-DYs & Y_ids<Ys(1)+DYs;  
    Ys_fit = linspace(min(Y_ids),max(Y_ids),100);
    Zs_fit = Ys_fit*0;
    Zs_fit(:) = feval(fZ_Y,Ys_fit(:));  
    plot3(DX_ids(flag),Y_ids(flag),Z_ids(flag),'.',...
        DXs,Ys,Zs,'go-',DXs*0,Ys,Zs,'go-',Ys_fit*0,Ys_fit,Zs_fit,...
        DX_ids(~flag),Y_ids(~flag),Z_ids(~flag),'rx',...
        DX_ids(flag2),Y_ids(flag2),Z_ids(flag2),'.')
    grid on
    xlabel('DX'), ylabel('Y'), zlabel('Z')
    
    
    subplot(1,2,2)
    plot3(XYZ(1,:),XYZ(2,:),XYZ(3,:),'.'), grid on
    
    [xData, yData, zData] = prepareSurfaceData( XYZ(1,:), XYZ(2,:), XYZ(3,:) );
    ft = fittype( 'poly22' );
    opts = fitoptions( 'Method', 'LinearLeastSquares' );
    opts.Robust = 'Bisquare';
    f = fit( [xData, yData], zData, ft, opts );
    [Xg, Yg] = meshgrid(linspace(min(XYZ(1,:)),max(XYZ(1,:)),20),...
        linspace(min(XYZ(2,:)),max(XYZ(2,:)),20));
    Zg = feval(f,Xg,Yg);
    hold on
    surf(Xg,Yg,Zg)
    
    xlabel('X'), ylabel('Y'), zlabel('Z')
    
    
    
end



return

%% Fit field curvature and plane vectors

% Set up fittype and options.
[xData, yData, zData] = prepareSurfaceData( Xsurf-XY0(1), Ysurf-XY0(2), Zsurf );
ft = fittype( 'poly22' );
opts = fitoptions( 'Method', 'LinearLeastSquares' );
opts.Robust = 'Bisquare';
f = fit( [xData, yData], zData, ft, opts );

% find central position
XYZ_c = [XY_c(1), XY_c(2), feval(f,XY_c(1)-XY0(1),XY_c(2)-XY0(2))];

% find field curvautre function
warning('off','all')
fZ_XY = f;
fZ_XY.p00 = 0;
fZ_XY.p10 = -2*f.p20*XY0(1)-f.p11*XY0(2);
fZ_XY.p01 = -2*f.p02*XY0(2)-f.p11*XY0(1);
fZ_dum = feval(fZ_XY,XY_c(1),XY_c(2));
fZ_XY.p00 = -fZ_dum;
warning('on','all')

% find plane vectors
V_flow = [cos(ang_XY); sin(ang_XY); f.p10*cos(ang_XY)+f.p01*sin(ang_XY)];
V_perp = [-sin(ang_XY); cos(ang_XY); -f.p10*sin(ang_XY)+f.p01*cos(ang_XY)];


if show_plot==1
    subplot(1,2,2)
    col = get(gca,'colororder');
    [X, Y] = meshgrid(linspace(min(dat.X),max(dat.X),20),...
        linspace(min(dat.Y),max(dat.Y),20));
    Ztot = feval(f,X-XY0(1),Y-XY0(2));
    
    surf(X,Y,Ztot), hold on, grid on
    ids = unique(dat.id(dat.id>0));
    for ii = 1:length(ids)
        flag = dat.id==ids(ii);
        plot3(dat.X(flag),dat.Y(flag),dat.Z(flag),'.-','color',col(1,:))
    end
    M = max([max(dat.X)-min(dat.X) max(dat.Y)-min(dat.Y)]);
    quiver3(XYZ_c(1),XYZ_c(2),XYZ_c(3),V_flow(1)*M,V_flow(2)*M,V_flow(3)*M,0)
    quiver3(XYZ_c(1),XYZ_c(2),XYZ_c(3),V_perp(1)*M,V_perp(2)*M,V_perp(3)*M,0)
    xlabel('X'), ylabel('Y'), zlabel('Z')
end




%% %%%%%%%%%%%%%%%
return
%%%%%%%%%%%%%%%%%%
% run plot checks
fZ_XY_plane = f;
fZ_XY_plane.p00 = 0;
fZ_XY_plane.p20 = 0;
fZ_XY_plane.p11 = 0;
fZ_XY_plane.p02 = 0;
fZ_dum = feval(fZ_XY_plane,XY_c(1)-XY0(1),XY_c(2)-XY0(2));
fZ_XY_plane.p00 = -fZ_dum+XYZ_c(3)-f.p10*XY0(1)-f.p01*XY0(2);

[X, Y] = meshgrid(linspace(0,2560,20),linspace(0,2160,20));
Zcorr = feval(fZ_XY,X,Y);
Zplane = feval(fZ_XY_plane,X,Y);

v_x = [1; 0; f.p10];
v_y = [0; 1; f.p01];
M = 1000;


% REFERENCE FIT
[xData, yData, zData] = prepareSurfaceData( Xsurf, Ysurf, Zsurf );
% Set up fittype and options.
ft = fittype( 'poly22' );
opts = fitoptions( 'Method', 'LinearLeastSquares' );
opts.Robust = 'Bisquare';
% Fit model to data.
fZ_XY_ref = fit( [xData, yData], zData, ft, opts );

Ztot = feval(fZ_XY_ref,X,Y);


% plot3(dat.X,dat.Y,dat.Z,'.'), hold on
surf(X,Y,Ztot), hold on
surf(X,Y,Zcorr+Zplane)
quiver3(XYZ_c(1),XYZ_c(2),XYZ_c(3),v_x(1)*M,v_x(2)*M,v_x(3)*M,0)
quiver3(XYZ_c(1),XYZ_c(2),XYZ_c(3),v_y(1)*M,v_y(2)*M,v_y(3)*M,0)
quiver3(XYZ_c(1),XYZ_c(2),XYZ_c(3),V_flow(1)*M,V_flow(2)*M,V_flow(3)*M,0)
quiver3(XYZ_c(1),XYZ_c(2),XYZ_c(3),V_perp(1)*M,V_perp(2)*M,V_perp(3)*M,0)



% fZ_XY_plane2 = fZ_XY_ref;
% fZ_XY_plane2.p20 = 0;
% fZ_XY_plane2.p02 = 0;
% fZ_XY_plane2.p11 = 0;
% fZ_XY_plane2.p10 = 2*fZ_XY_ref.p20;
% fZ_XY_plane2.p01 = 2*fZ_XY_ref.p02;
% fZ_XY_plane2.p00 = fZ_XY_ref.p11*(XY0(1)+XY0(2))+fZ_XY_ref.p10+fZ_XY_ref.p01;
% Zplane2 = feval(fZ_XY_plane2,X,Y);
% 
% 
% % plot3(dat.X,dat.Y,dat.Z,'.'), hold on
% surf(X,Y,Zplane), hold on, drawnow, pause(1)
% surf(X,Y,Zplane2)


%%

return
