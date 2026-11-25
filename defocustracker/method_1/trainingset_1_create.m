function trainingset_1_create(model,img,varargin)


if nargin==4
    model_name = varargin{1};
    img_name = varargin{2};
elseif nargin==2
    model_name = inputname(1);
    img_name = inputname(2);
end
    
stretch_factor = 5/4;

dat = dtracker_create('dataset',img.n_frames);
[f, ax] = dataset_show(dat,img);
f.Position(3) = f.Position(3)*stretch_factor;
ax.Position([1 3]) = ax.Position([1 3])/stretch_factor;
set(f,'name',['training set on: ',img_name,';  model: ',model_name],'numbertitle','off')

this = getappdata(f,'this');
this.model = model;

hfields = fields(this.handles);
for ii = 1:length(hfields)
    eval(['this.handles.',hfields{ii},'.Position([1 3]) = ',...
        'this.handles.',hfields{ii},'.Position([1 3])/stretch_factor;'])
end
 
tfields = fields(this.text);
for ii = 1:length(tfields)
    eval(['this.text.',tfields{ii},'.Position([1 3]) = ',...
        'this.text.',tfields{ii},'.Position([1 3])/stretch_factor;'])   
end

edit_width = this.handles.bmi.Position(3);

block_heights = linspace(.94,.0450,11);

block_height = block_heights(1);
uicontrol('Parent',f,'Style','text','units','normalized',...
    'position',[.8 block_height .09 .04],'String','Initial X',...
    'horizontalalignment','center')
uicontrol('Parent',f,'Style','text','units','normalized',...
    'position',[.88 block_height .11 .04],'String','Initial Y',...
    'horizontalalignment','center')
this.handles.xpos = uicontrol('Parent',f,'Style','edit','units','normalized',...
    'position',[.8 block_height-.03 edit_width .04],'String',num2str(0));
this.handles.ypos = uicontrol('Parent',f,'Style','edit','units','normalized',...
    'position',[.98-edit_width block_height-.03 edit_width .04],'String',num2str(0));

block_height = block_heights(2);
uicontrol('Parent',f,'Style','text','units','normalized',...
    'position',[.8 block_height .09 .04],'String','Box width',...
    'horizontalalignment','center')
uicontrol('Parent',f,'Style','text','units','normalized',...
    'position',[.88 block_height .11 .04],'String','Box height',...
    'horizontalalignment','center')
this.handles.imwidth = uicontrol('Parent',f,'Style','edit','units','normalized',...
    'position',[.8 block_height-.03 edit_width .04],'String',num2str(0));
this.handles.imheight = uicontrol('Parent',f,'Style','edit','units','normalized',...
    'position',[.98-edit_width block_height-.03 edit_width .04],'String',num2str(0));


block_height = block_heights(3);
uicontrol('Parent',f,'Style','text','units','normalized',...
    'position',[.8 block_height .09 .04],'String','Median filter',...
    'horizontalalignment','center')
uicontrol('Parent',f,'Style','text','units','normalized',...
    'position',[.88 block_height .11 .04],'String','Bounding threshold',...
    'horizontalalignment','center')
this.handles.median_filter = uicontrol('Parent',f,'Style','edit','units','normalized',...
    'position',[.8 block_height-.03 edit_width .04],'String',...
    num2str(model.training.median_filter));
this.handles.boundary_threshold = uicontrol('Parent',f,'Style','edit','units','normalized',...
    'position',[.98-edit_width block_height-.03 edit_width .04],'String',...
    num2str(model.training.boundary_threshold));

block_height = block_heights(4);
uicontrol('Parent',f,'Style','text','units','normalized',...
    'position',[.8 block_height .09 .04],'String','First frame',...
    'horizontalalignment','center')
this.handles.fr_st = uicontrol('Parent',f,'Style','edit','units','normalized',...
    'position',[.8 block_height-.03 edit_width .04],'String','1');


this.handles.apply = uicontrol('Parent',f,'Style','pushbutton','units','normalized',...
    'position',[.8 block_heights(5)-.03 .18 .06],'String','1. Initialize');


block_height = block_heights(6);
uicontrol('Parent',f,'Style','text','units','normalized',...
    'position',[.8 block_height .09 .04],'String','Last frame',...
    'horizontalalignment','center')
this.handles.fr_end = uicontrol('Parent',f,'Style','edit','units','normalized',...
    'position',[.8 block_height-.03 edit_width .04],'String',num2str(img.n_frames));
uicontrol('Parent',f,'Style','text','units','normalized',...
    'position',[.89 block_height .09 .04],'String','Stride',...
    'horizontalalignment','center')
this.handles.fr_stride = uicontrol('Parent',f,'Style','edit','units','normalized',...
    'position',[.98-edit_width block_height-.03 edit_width .04],'String',num2str(1));

block_height = block_heights(7);
this.handles.follow_part = uicontrol('Parent',f,'Style','checkbox','units','normalized',...
    'position',[.8 block_height+0.01 0.1 .04],'String','Follow particle',...
    'Value',0);

this.handles.apply2 = uicontrol('Parent',f,'Style','pushbutton','units','normalized',...
    'position',[.8 block_heights(8)+0.03 .18 .06],'String','2. Find stack');



block_height = block_heights(9);
uicontrol('Parent',f,'Style','text','units','normalized',...
    'position',[.8 block_height .09 .04],'String','Smoothing',...
    'horizontalalignment','center')
uicontrol('Parent',f,'Style','text','units','normalized',...
    'position',[.89 block_height .09 .04],'String','No. interp images',...
    'horizontalalignment','center')
this.handles.smoothing = uicontrol('Parent',f,'Style','edit','units','normalized',...
    'position',[.8 block_height-.03 edit_width .04],'String',...
    num2str(model.training.smoothing));
this.handles.n_interp_images = uicontrol('Parent',f,'Style','edit','units','normalized',...
    'position',[.98-edit_width block_height-.03 edit_width .04],'String',...
    num2str(model.training.n_interp_images));

block_height = block_heights(10);
uicontrol('Parent',f,'Style','text','units','normalized',...
    'position',[.8 block_height .09 .04],'String','Imresize',...
    'horizontalalignment','center')
this.handles.imresize = uicontrol('Parent',f,'Style','edit','units','normalized',...
    'position',[.8 block_height-.03 edit_width .04],'String',...
    num2str(model.training.imresize));

this.handles.train = uicontrol('Parent',f,'Style','pushbutton','units','normalized',...
    'position',[.8 block_heights(11)-.03 .18 .06],'String','3. Train');


this.handles.apply.Callback = {@apply,f};
this.handles.apply2.Callback = {@apply2,f};
this.handles.train.Callback = {@train,f,model_name};

setappdata(f,'this',this)


function apply(~,~,f)
this = getappdata(f,'this');
fr_st = str2double(get(this.handles.fr_st,'String'));
this.dat = dtracker_create('dataset',1);
this.dat.fr(1) = fr_st;
this.dat.X(1) = str2double(get(this.handles.xpos,'String'));
this.dat.Y(1) = str2double(get(this.handles.ypos,'String'));
this.dat.Z(1) = 0;

imw = str2double(get(this.handles.imwidth,'String'));
imh = str2double(get(this.handles.imheight,'String'));
median_filter = str2double(get(...
    this.handles.median_filter,'String'));
boundary_threshold = str2double(get(...
    this.handles.boundary_threshold,'String'));
im = dtracker_show(this.img,fr_st);
image_boundary = find_single_boundary(im,this.dat.X(1),this.dat.Y(1),...
    imw,imh,median_filter,boundary_threshold);

this.images_boundary{1} = [-imw/2 -imw/2 imw/2 imw/2 -imw/2 NaN image_boundary(1,:);...
     -imh/2 imh/2 imh/2 -imh/2 -imh/2 NaN image_boundary(2,:)];

set(this.handles.bnr,'String',num2str(fr_st)); 
set(this.handles.apply2,'String','2. Find stack')
set(this.handles.train,'String','3. Train')

setappdata(f,'this',this)
this.handles.bnr.Callback{1}([],[],f)


function apply2(~,~,f)
this = getappdata(f,'this');
fr_st = str2double(get(this.handles.fr_st,'String'));
fr_end = str2double(get(this.handles.fr_end,'String'));
fr_stride = str2double(get(this.handles.fr_stride,'String'));
frs = fr_st:fr_stride:fr_end;
imw = str2double(get(this.handles.imwidth,'String'));
imh = str2double(get(this.handles.imheight,'String'));
median_filter = str2double(get(...
    this.handles.median_filter,'String'));
boundary_threshold = str2double(get(...
    this.handles.boundary_threshold,'String'));
this.dat = dtracker_create('dataset',length(frs));
this.images_boundary = cell(size(frs));

set(this.handles.apply2,'String','Thinking...')
set(this.handles.bnr,'String',num2str(fr_st)); 
setappdata(f,'this',this)
this.handles.bnr.Callback{1}([],[],f)
drawnow

this.dat.fr(:) = frs;
this.dat.X(:) = str2double(get(this.handles.xpos,'String'));
this.dat.Y(:) = str2double(get(this.handles.ypos,'String'));
this.dat.Z(:) = linspace(0,1,length(frs));

if this.handles.follow_part.Value==1
%     Xdum = this.dat.X*0;
%     Ydum = this.dat.X*0;
    for ii = 1:length(frs) % for ii = 1:length(frs)
        im = dtracker_show(this.img,frs(ii));
        image_boundary = find_single_boundary(im,this.dat.X(ii),this.dat.Y(ii),...
            imw,imh,median_filter,boundary_threshold); % .X(ii) ... .Y(ii)
%         Xdum(ii) = this.dat.X(ii);
%         Ydum(ii) = this.dat.Y(ii);
        this.dat.X(ii) = this.dat.X(ii)+(max(image_boundary(1,:))+min(image_boundary(1,:)))/2; % .X(ii)
        this.dat.Y(ii) = this.dat.Y(ii)+(max(image_boundary(2,:))+min(image_boundary(2,:)))/2; % .Y(ii)
        if ii<length(frs)
            this.dat.X(ii+1) = this.dat.X(ii); 
            this.dat.Y(ii+1) = this.dat.Y(ii); 
%             im2 = dtracker_show(this.img,frs(ii+1));
%             [this.dat.X(ii+1), this.dat.Y(ii+1)] = find_follow_boundary(...
%                 im,im2,this.dat.X(ii),this.dat.Y(ii),imw,imh);
        end
    end
%     Xdum2 = Xdum(1) - [0; cumsum(Xdum(2:end)-this.dat.X(1:end-1))];
%     Ydum2 = Ydum(1) - [0; cumsum(Ydum(2:end)-this.dat.Y(1:end-1))];
%     Xcorr2 = linspace(0,this.dat.X(end)-Xdum2(end),length(Xdum))';
%     Ycorr2 = linspace(0,this.dat.Y(end)-Ydum2(end),length(Ydum))';
%     this.dat.X(:) = Xdum2+Xcorr2;
%     this.dat.Y(:) = Ydum2+Ycorr2;
end


for ii = 1:length(frs)
    im = dtracker_show(this.img,frs(ii));
    image_boundary = find_single_boundary(im,this.dat.X(ii),this.dat.Y(ii),...
        imw,imh,median_filter,boundary_threshold);
    this.images_boundary{ii} = [-imw/2 -imw/2 imw/2 imw/2 -imw/2 NaN image_boundary(1,:);...
        -imh/2 imh/2 imh/2 -imh/2 -imh/2 NaN image_boundary(2,:)];
end



%%


set(this.handles.apply2,'String','2. Find stack')
set(this.handles.bnr,'String',num2str(fr_st)); 
set(this.handles.bsk,'String',this.handles.fr_stride.String)
setappdata(f,'this',this)
this.handles.bnr.Callback{1}([],[],f)


function train(~,~,f,model_name)
this = getappdata(f,'this');
%%
model = this.model;
model.training.imwidth = str2double(get(this.handles.imwidth,'String'));
model.training.imheight = str2double(get(this.handles.imheight,'String'));
model.training.median_filter = str2double(get(...
    this.handles.median_filter,'String'));
model.training.boundary_threshold = str2double(get(...
    this.handles.boundary_threshold,'String'));
model.training.smoothing = str2double(get(...
    this.handles.smoothing,'String'));
model.training.n_interp_images = str2double(get(...
    this.handles.n_interp_images,'String'));
model.training.imresize = str2double(get(...
    this.handles.imresize,'String'));

dat = this.dat;
img = this.img;

set(this.handles.train,'String','Thinking...')
drawnow
model = dtracker_train(model,img,dat);
set(this.handles.train,'String','3. Train')

eval([model_name,' = model;'])
eval(['dtracker_show(',model_name,')'])


assignin('base',model_name,model)
setappdata(f,'this',this)


function image_boundary = find_single_boundary(im,X,Y,width,height,...
    median_filter,boundary_threshold)


im = imageset_crop(im,X,Y,[width, height]);
if median_filter>2
    im =  medfilt2(im,[median_filter median_filter],'symmetric');
end
background = median([im(1,:), im(end,:), im(:,1)', im(:,end)']);
noise = std([im(1,:), im(end,:), im(:,1)', im(:,end)']);

image_boundary = [];
boundary_threshold = max([noise 1])*boundary_threshold;

[~,im_segm] = bwboundaries(abs(im-background)>boundary_threshold,8);
%%%%%%%%%%%% Taking the larges area %%%%%%%%%%%%%%%%%
segm_data = regionprops(im_segm,'PixelIdxList','FilledArea');
[~, jj] = max([segm_data(:).FilledArea]);
if isempty(jj)
    id_y = []; id_x = [];
else
    [id_y, id_x] = ind2sub(size(im_segm),segm_data(jj).PixelIdxList);
end
%%%%%%%%%%%% Adding up the areas inside the box %%%%%%%%%%%%%%%%%
% segm_data = regionprops(im_segm,'PixelIdxList');
% id_x = []; id_y = [];
% if ~isempty(segm_data)
%     for jj = 1:length(segm_data)
%         if length(segm_data(jj).PixelIdxList)>15
%             [id_y_dum, id_x_dum] = ind2sub(size(im_segm),segm_data(jj).PixelIdxList);
%             id_x = [id_x; id_x_dum]; id_y = [id_y; id_y_dum];
%         end
%     end
% end
    
if ~isempty(id_x)
    ind_convex_hull = boundary(id_x,id_y,0);
    image_boundary(1,:) = id_x(ind_convex_hull)-(width+1)/2;
    image_boundary(2,:) = id_y(ind_convex_hull)-(height+1)/2;
else
    image_boundary(1,:) = [-.5 -.5 .5 .5 -.5]*width;
    image_boundary(2,:) = [-.5 .5 .5 -.5 -.5]*height;
end

function  [X, Y] = find_follow_boundary(im0,im,X0,Y0,imw,imh)

im0c = imageset_crop(im0,X0,Y0,[imw,imh]);
imc = imageset_crop(im,X0,Y0,[imw*2,imh*2]);
cc = normxcorr2(im0c,imc);
x0 = round((size(cc,2)+1)/2);
y0 = round((size(cc,1)+1)/2);
[x1,y1,c1] = find_peaks_2d(cc,0.5);
dx1 = x0-x1(c1==max(c1));
dy1 = y0-y1(c1==max(c1));
if isempty(dx1)
    dx1 = 0;
    dy1 = 0;
end

X = X0+dx1;
Y = Y0+dy1;
 


