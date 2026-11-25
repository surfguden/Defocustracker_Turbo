function varargout = scaling_linear_create(varargin)

if nargout==1 && nargin==0
    Scaling.type = 'linear';
    Scaling.X_to_x = 1;
    Scaling.Y_to_y = 1;
    Scaling.Z_to_z = 1;
    Scaling.dt = 1;
    Scaling.unit = '';
    Scaling.correction = [];
    varargout{1} = Scaling;   
    return
end

w_commands = -.68;
w_axes = .30;

this.im = [];
this.dx = [];
this.dy = [];
this.th = [];
this.xg = [];
this.yg = [];
this.tm = [];
this.i0 = [];

f = figure;
set(f,'Unit','Normalized','position',[.22 .26 .5 .5],'NumberTitle','Off',...
    'Color',[0.94 0.94 0.94])%,'IntegerHandle','Off')


% first row
annotation(f,'textbox',[.80+w_commands .91 .08 .05],'HorizontalAlignment','center',...
    'String','Min','FitBoxToText','Off','EdgeColor','none');
annotation(f,'textbox',[.89+w_commands .91 .08 .05],'HorizontalAlignment','center',...
    'String','Max','FitBoxToText','Off','EdgeColor','none');
this.handles.load = uicontrol('Style','pushbutton','String', 'Load image',...
    'Units','normalized','Position',[.70+w_commands .87 .09 .08]);
this.handles.bmin = uicontrol('Parent',f,'Style','edit','units','normalized',...
    'position',[.80+w_commands .87 .08 .05],'String','');
this.handles.bmax = uicontrol('Parent',f,'Style','edit','units','normalized',...
    'position',[.89+w_commands .87 .08 .05],'String','');

% second row
annotation(f,'textbox',[.68+w_commands .77 .16 .05],'HorizontalAlignment','center',...
    'String','Unit','FitBoxToText','Off','EdgeColor','none');
annotation(f,'textbox',[.83+w_commands .77 .16 .05],'HorizontalAlignment','center',...
    'String','Pitch [unit]','FitBoxToText','Off','EdgeColor','none');
% annotation(f,'textbox',[.88+w_commands .78 .12 .05],'HorizontalAlignment','center',...
%     'String','Pitch Y [unit]','FitBoxToText','Off','EdgeColor','none');
this.handles.unit = uicontrol('Style','edit','String', 'µm',...
    'Units','normalized','Position',[.70+w_commands .72 .12 .05]);
this.handles.pitch = uicontrol('Style','edit','String', '1',...
    'Units','normalized','Position',[.85+w_commands .72 .12 .05]);
% this.handles.pitchY = uicontrol('Style','edit','String', '1',...
%     'Units','normalized','Position',[.90+w_commands .73 .08 .05]);

% third row
this.handles.fitGrid = uicontrol('Style','pushbutton','String', 'Start',...
    'Units','normalized','Position',[.70+w_commands .58 .27 .10]);


% fourth row
annotation(f,'textbox',[.68+w_commands .49 .16 .05],'HorizontalAlignment','center',...
    'String','Grid orientation [deg]','FitBoxToText','Off','EdgeColor','none');
annotation(f,'textbox',[.83+w_commands .49 .16 .05],'HorizontalAlignment','center',...
    'String','Lens orientation [deg]','FitBoxToText','Off','EdgeColor','none');
this.handles.alpha_grid = uicontrol('Style','edit','String', '',...
    'Units','normalized','Position',[.70+w_commands .44 .12 .05],...
    'Enable','Off');
this.handles.alpha_lens = uicontrol('Style','edit','String', '',...
    'Units','normalized','Position',[.85+w_commands .44 .12 .05],...
    'Enable','Off');


% fifth row
annotation(f,'textbox',[.68+w_commands .35 .16 .05],'HorizontalAlignment','center',...
    'String','Scaling X [unit/px]','FitBoxToText','Off','EdgeColor','none');
annotation(f,'textbox',[.83+w_commands .35 .16 .05],'HorizontalAlignment','center',...
    'String','Scaling Y [unit/px]','FitBoxToText','Off','EdgeColor','none');
this.handles.dx = uicontrol('Style','edit','String', '',...
    'Units','normalized','Position',[.70+w_commands .30 .12 .05],...
    'Enable','On');
this.handles.dy = uicontrol('Style','edit','String', '',...
    'Units','normalized','Position',[.85+w_commands .30 .12 .05],...
    'Enable','On');


% sixth row
annotation(f,'textbox',[.68+w_commands .21 .16 .05],'HorizontalAlignment','center',...
    'String','Z Depth [unit]','FitBoxToText','Off','EdgeColor','none');
annotation(f,'textbox',[.83+w_commands .21 .16 .05],'HorizontalAlignment','center',...
    'String','\Deltat [s]','FitBoxToText','Off','EdgeColor','none');
this.handles.pitchZ = uicontrol('Style','edit','String', '1',...
    'Units','normalized','Position',[.70+w_commands .16 .12 .05]);
this.handles.dt = uicontrol('Style','edit','String', '1',...
    'Units','normalized','Position',[.85+w_commands .16 .12 .05]);


% seventh row
annotation(f,'textbox',[.68+w_commands .07 .16 .05],'HorizontalAlignment','center',...
    'String','Scaling name','FitBoxToText','Off','EdgeColor','none');
this.handles.takeVal = uicontrol('Style','pushbutton','String', 'Save',...
    'Units','normalized','Position',[.85+w_commands .02 .12 .08]);
this.handles.scalName = uicontrol('Style','edit','String', 'scaling_linear',...
    'Units','normalized','Position',[.70+w_commands .02 .12 .05]);


ax = axes('position',[.04+w_axes .1 .63 .86]);
imshow(this.im,[],'Initialmagnification','fit')

this.handles.load.Callback = {@load_img,f,ax};
this.handles.pitchX.Callback = {@calcScaling,f};
this.handles.pitchY.Callback = {@calcScaling,f};
this.handles.fitGrid.Callback = {@fitGrid,f,ax};
this.handles.takeVal.Callback = {@takeVal,f};
this.handles.bmin.Callback = {@update_clim,f,ax};
this.handles.bmax.Callback = {@update_clim,f,ax};

setappdata(f,'this',this);


% this.handles.selectCross.Callback = {@selectCross,f,ax};
% this.handles.findPoints.Callback = {@findPoints,f,ax};


% annotation(f,'textbox',[.70+w_commands .68 .27 .03],'HorizontalAlignment','center',...
%     'String','Acceptance valid points','FitBoxToText','Off','EdgeColor','none');
set(f,'Toolbar','figure','Menubar','None')%,'HandleVisibility','Off')
end

function load_img(~,~,f,ax)
this = getappdata(f,'this');
[filename, pathname, ok] = uigetfile({'*.tif';...
    '*.png';'*.jpg'},'Select set of images','MultiSelect','on');
if ok>0
    axes(ax), cla(ax)
    imd = imread(fullfile(pathname,filename));
    if size(imd,3)==3
        this.im = medfilt2(double(rgb2gray(imd)),[3 3]);
    else
        this.im = medfilt2(double(imd),[3 3]);
    end
    axes(ax);
    imshow(this.im,[],'Initialmagnification','fit'), hold on
    clims = [0 round(median(this.im(:))*2)];% get(ax,'Clim');
    set(ax,'Clim',clims);
    set(this.handles.bmin,'string',num2str(round(clims(1))))
    set(this.handles.bmax,'string',num2str(round(clims(2))))
    setappdata(f,'this',this)
end

end

function update_clim(~,~,f,ax)
this = getappdata(f,'this');
clim1 = get(this.handles.bmin,'String');
clim2 = get(this.handles.bmax,'String');
set(ax,'Clim',[str2double(clim1) str2double(clim2)])
end


function fitGrid(~,~,f,ax)
this = getappdata(f,'this');

grid_pitch = str2double(get(this.handles.pitch,'String'));
scaling = scaling_linear_gridfind(f,ax,this.im,grid_pitch);
set(this.handles.alpha_grid,'string',num2str(scaling.alpha_grid))
set(this.handles.alpha_lens,'string',num2str(scaling.alpha_lens))
set(this.handles.dx,'string',num2str(scaling.scal_x))
set(this.handles.dy,'string',num2str(scaling.scal_y))
setappdata(f,'this',this);

end


function takeVal(~,~,f)
this = getappdata(f,'this');

% if ~isempty(this.dx) && ~isempty(this.dy) 
% else
%     disp('Missing data')
%     return
% end

Scaling.type = 'linear';
Scaling.X_to_x = str2double(get(this.handles.dx,'String'));
Scaling.Y_to_y = str2double(get(this.handles.dy,'String'));
Scaling.Z_to_z = eval((get(this.handles.pitchZ,'String')));
Scaling.dt = eval((get(this.handles.dt,'String')));
Scaling.unit = get(this.handles.unit,'String');
Scaling.Z_correction = [];

assignin('base',get(this.handles.scalName,'String'),Scaling)



end

function [scaling, varargout] = scaling_linear_gridfind(f,ax,im,grid_pitch)

figure(gcf)
axes(ax)
this = getappdata(f,'this');

limx = get(gca,'xlim');
limy = get(gca,'ylim');
limc = get(gca,'clim');

cla reset

im = double(im);
imshow(im,limc)
xlim(limx)
ylim(limy)

Xi = zeros(1,3);
Yi = zeros(1,3);

set(this.handles.fitGrid,'String', 'Select origin point')
[Xi(1), Yi(1)] = ginput(1);
set(this.handles.fitGrid,'String', 'Select first point to the right of the origin')
hold on
plot(Xi(1),Yi(1),'gx')
[Xi(2), Yi(2)] = ginput(1);
set(this.handles.fitGrid,'String', 'Select first point above the origin')
quiver(Xi(1),Yi(1),Xi(2)-Xi(1),Yi(2)-Yi(1),0,'g')
[Xi(3), Yi(3)] = ginput(1);
set(this.handles.fitGrid,'String', 'Start')
quiver(Xi(1),Yi(1),Xi(3)-Xi(1),Yi(3)-Yi(1),0,'g')
drawnow

% initialize

DX_right = Xi(2)-Xi(1);
DY_right = Yi(2)-Yi(1);
DX_up = Xi(3)-Xi(1);
DY_up = Yi(3)-Yi(1);
Xc = round(Xi(1));
Yc = round(Yi(1));

W = min([round(abs(DX_right/2))*2 128]);
H = min([round(abs(DY_up/2))*2 128]);

imtmp = im(Yc-H/2:Yc+H/2,Xc-W/2:Xc+W/2);
imcc = normxcorr2(imtmp,im);
imcc = imcc(H/2+1:end-H/2,W/2+1:end-W/2);
[Xp, Yp, cc] = find_peaks_2d_subpx(imcc);

IMsize = size(im);
XYlim = [W/2-Xc, (IMsize(2)-W/2)-Xc, H/2-Yc, (IMsize(1)-H/2)-Yc];

Xp = Xp-Xc;
Yp = Yp-Yc;


hold on
plot(Xc,Yc,'*g')
quiver(Xc,Yc,DX_up,DY_up,0,'g')
quiver(Xc,Yc,DX_right,DY_right,0,'g')
drawnow

% calculate the grid
XY = [];
xy = [];

Xcurr = 0;
Ycurr = 0;
niter = 0;

% kkk.pp = plot(Xp+Xc,Yp+Yc,'y+');
% kkk.ppc = plot(Xc,Yc,'m');
% kkk.Xc = Xc;
% kkk.Yc = Yc;
% return
while Xcurr>XYlim(1) && Xcurr<XYlim(2) && Ycurr>XYlim(3) && Ycurr<XYlim(4)
    [XY, xy_y] = update_XY_up(XY, Xp, Yp, cc, Xcurr, Ycurr, -DX_up, -DY_up, XYlim, W, H);%, kkk);
    Xcurr = Xcurr+DX_right;
    Ycurr = Ycurr+DY_right;
    xy = [xy [xy_y*0+niter; xy_y]];
    niter = niter+1;
end
Xcurr = -DX_right;
Ycurr = -DY_right;
niter = 0;

while Xcurr>XYlim(1) && Xcurr<XYlim(2) && Ycurr>XYlim(3) && Ycurr<XYlim(4)
    [XY, xy_y] = update_XY_up(XY, Xp, Yp, cc, Xcurr, Ycurr, -DX_up, -DY_up, XYlim, W, H);%, kkk);
    Xcurr = Xcurr-DX_right;
    Ycurr = Ycurr-DY_right;
    niter = niter-1;
    xy = [xy [xy_y*0+niter; xy_y]];
end


xy = xy*grid_pitch;

[alpha, scal_x, scal_y, scal_xy] = find_magnification(xy,XY);
M = [scal_x, scal_xy; scal_xy scal_y];
[V, ~] = eig(M);
beta_raw = atan2(V(1,1),V(2,1));
beta = rad2deg(mod(abs(beta_raw),pi/2));
if beta>45, beta = beta-90; end

scaling.alpha_grid = alpha;
scaling.alpha_lens = beta;
scaling.scal_x = scal_x;
scaling.scal_y = scal_y;
scaling.scal_xy = scal_xy;

xy_r = [cosd(-alpha) -sind(-alpha); sind(-alpha) cosd(-alpha)]*xy;
XY_est = [scal_x scal_xy; scal_xy scal_y]^-1*xy_r;

plot(XY(1,:)+Xc,XY(2,:)+Yc,'or'); hold on
plot(XY_est(1,:)+Xc,XY_est(2,:)+Yc,'+y'); hold on
daspect([1 1 1])
figure(gcf)

if nargout==3
    varargout{1} = XY;
    varargout{2} = xy;
end
end


function [XY, xy_y] = update_XY_up(XY, Xp, Yp, cc, Xcurr0, Ycurr0, DX_temp, DY_temp, XYlim, W, H)%, kkk)
Xcurr = Xcurr0;
Ycurr = Ycurr0;
xy_y = [];
niter = 0;
while Xcurr>XYlim(1) && Xcurr<XYlim(2) && Ycurr>XYlim(3) && Ycurr<XYlim(4)
    XV = [-.5 -.5 .5 .5 -.5]*W+Xcurr;
    YV = [-.5 .5 .5 -.5 -.5]*H+Ycurr;
    flag = inpolygon(Xp,Yp,XV,YV);
    if sum(flag)>0 
        Xtemp = Xp(flag);
        Ytemp = Yp(flag);
        cctemp = cc(flag);
        [~, ind] = max(cctemp);
        xy_y = [xy_y, sign(DY_temp)*niter];    
        XY = [XY, [Xtemp(ind(1)); Ytemp(ind(1))]];
    end
    
    Xcurr = Xcurr+DX_temp;
    Ycurr = Ycurr+DY_temp;
    niter = niter+1;
end
Xcurr = Xcurr0-DX_temp;
Ycurr = Ycurr0-DY_temp;
niter = 0;
while Xcurr>XYlim(1) && Xcurr<XYlim(2) && Ycurr>XYlim(3) && Ycurr<XYlim(4)
    niter = niter+1;
    XV = [-.5 -.5 .5 .5 -.5]*W+Xcurr;
    YV = [-.5 .5 .5 -.5 -.5]*H+Ycurr;
    flag = inpolygon(Xp,Yp,XV,YV);
    if sum(flag)>0
        Xtemp = Xp(flag);
        Ytemp = Yp(flag);
        cctemp = cc(flag);
        [~, ind] = max(cctemp);
        xy_y = [xy_y, -sign(DY_temp)*niter];
        XY = [XY, [Xtemp(ind(1)); Ytemp(ind(1))]];
    end
    Xcurr = Xcurr-DX_temp;
    Ycurr = Ycurr-DY_temp;
end
end

function [alpha, scal_x, scal_y, scal_xy] = find_magnification(xy,XY)

X0 = zeros(1,4);
X0(2:3) = abs(nanmedian(xy./XY,2));

LB = [-90 0   0   -inf];
UB = [ 90 inf inf inf];

options = optimset('Display','off');
X = fmincon(@(X) give_cost(X,xy,XY),X0,[],[],[],[],LB,UB,[],options);
alpha = X(1);
scal_x = X(2);
scal_y = X(3);
scal_xy = X(4);
end

function out = give_cost(X,xy,XY)

alpha = X(1);
scal_x = X(2);
scal_y = X(3);
scal_xy = X(4);

xy_r = [cosd(-alpha) -sind(-alpha); sind(-alpha) cosd(-alpha)]*xy;
XY_est = [scal_x scal_xy; scal_xy scal_y]^-1*xy_r;

out = mean(sqrt(sum((XY-XY_est).^2)));
end

