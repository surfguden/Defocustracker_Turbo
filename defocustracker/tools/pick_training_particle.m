function [selection,fig] = pick_training_particle(files,initial,waitForSelection)
% Select a fixed particle crop and an inclusive calibration-frame range.
% Drag inside the box to move it; drag any edge or corner to resize it.
% X/Y are integer pixel centres; crop dimensions remain odd pixel counts.
% Auto-once/manual grayscale LUT changes display limits only, never pixels.
% files can be a calibration folder or a cell array of image filenames.
% Optional initial fields override a centered 151 x 151 box and full range.
% Third argument false creates a nonblocking UI for callback verification.
if nargin<3,waitForSelection=true;end
if ischar(files)||(isstring(files)&&isscalar(files))
    folder=char(files);listing=dir(fullfile(folder,'*.tif'));
    [~,order]=sort({listing.name});listing=listing(order);
    files=fullfile(folder,{listing.name});
end
assert(~isempty(files),'No calibration images supplied.');
im=imread(files{1});assert(ismatrix(im),'Calibration images must be grayscale.');
[rows,cols]=size(im);assert(min(rows,cols)>=3,'Images must be at least 3 x 3 pixels.');
if nargin<2||isempty(initial),initial=struct();end
defaults=struct('X',round((cols+1)/2),'Y',round((rows+1)/2), ...
    'width',min(151,2*floor((cols-1)/2)+1), ...
    'height',min(151,2*floor((rows-1)/2)+1), ...
    'start_frame',1,'stop_frame',numel(files));
fields=fieldnames(defaults);
for k=1:numel(fields)
    if ~isfield(initial,fields{k}),initial.(fields{k})=defaults.(fields{k});end
end
selection=[];movingROI=false;lastPosition=[];
lutLimits=imageLimits(im);appliedAlgorithm='Initial range';
paper=[.96 .97 .98];ink=[.16 .20 .25];accent=[.05 .55 .85];
fig=figure('Name','Training particle: check every height','NumberTitle','off', ...
    'Position',[100 100 1280 820],'Color',paper,'MenuBar','none', ...
    'ToolBar','none','CloseRequestFcn',@cancel, ...
    'DefaultUicontrolFontName','Segoe UI','DefaultUicontrolFontSize',11, ...
    'DefaultUicontrolBackgroundColor',paper,'DefaultUicontrolForegroundColor',ink);
uicontrol(fig,'Style','text','Units','normalized','Position',[.05 .945 .59 .035], ...
    'String','Choose a training particle','FontSize',16,'FontWeight','bold','HorizontalAlignment','left');
ax=axes('Parent',fig,'Position',[.05 .32 .58 .57], ...
    'FontName','Segoe UI','FontSize',10,'TitleFontWeight','normal');
imageHandle=imagesc(ax,im);axis(ax,'image');colormap(fig,gray);
set(ax,'FontName','Segoe UI','FontSize',10,'TitleFontWeight','normal');
set(imageHandle,'ButtonDownFcn',@clickParticle);
preview=axes('Parent',fig,'Position',[.705 .44 .255 .40], ...
    'FontName','Segoe UI','FontSize',11,'TitleFontWeight','normal');
cropHandle=imagesc(preview,im(1:3,1:3));axis(preview,'image');axis(preview,'off');
set(preview,'FontName','Segoe UI','FontSize',11,'TitleFontWeight','normal');

uicontrol(fig,'Style','text','Units','normalized','Position',[.705 .956 .255 .026], ...
    'String','Image LUT','HorizontalAlignment','left');
lutAlgorithms={'Rossi-Barnkob','Percentile (1-99%)','Min-max','Manual Linear'};
lutMode=uicontrol(fig,'Style','popupmenu','Units','normalized','Position',[.705 .921 .17 .035], ...
    'String',lutAlgorithms,'Value',1,'BackgroundColor','white', ...
    'Tag','lutMode','Callback',@changeLutMode, ...
    'TooltipString','Choose an algorithm, then press Auto once. Limits stay fixed while browsing frames.');
lutAuto=uicontrol(fig,'Style','pushbutton','Units','normalized','Position',[.884 .921 .08 .035], ...
    'String','Auto once','Tag','lutAutoOnce','FontSize',10,'Callback',@autoOnce, ...
    'TooltipString','Calculate black/white limits from the current full image once.');
uicontrol(fig,'Style','text','Units','normalized','Position',[.705 .886 .05 .026], ...
    'String','Black','HorizontalAlignment','left');
lutBlack=uicontrol(fig,'Style','edit','Units','normalized','Position',[.755 .881 .073 .039], ...
    'String',num2str(lutLimits(1)),'BackgroundColor','white','HorizontalAlignment','center', ...
    'Tag','lutBlack','Callback',@changeManualLut, ...
    'TooltipString','Image intensity displayed as black. White must be greater than black.');
uicontrol(fig,'Style','text','Units','normalized','Position',[.84 .886 .05 .026], ...
    'String','White','HorizontalAlignment','left');
lutWhite=uicontrol(fig,'Style','edit','Units','normalized','Position',[.89 .881 .073 .039], ...
    'String',num2str(lutLimits(2)),'BackgroundColor','white','HorizontalAlignment','center', ...
    'Tag','lutWhite','Callback',@changeManualLut, ...
    'TooltipString','Image intensity displayed as white. Invalid limits retain the previous values.');

keys={'X','Y','width','height'};
captions={'Centre X','Centre Y','Width (pixels)','Height (pixels)'};
edits=gobjects(1,4);
for k=1:4
    x=.05+(k-1)*.15;
    uicontrol(fig,'Style','text','Units','normalized','Position',[x .257 .135 .027], ...
        'String',captions{k},'HorizontalAlignment','left');
    edits(k)=uicontrol(fig,'Style','edit','Units','normalized','Position',[x .213 .125 .041], ...
        'BackgroundColor','white','HorizontalAlignment','center', ...
        'String',num2str(initial.(keys{k})),'Tag',keys{k},'Callback',@refresh);
end
rangeDefaults=[1 numel(files)];
if isfield(initial,'start_frame'),rangeDefaults(1)=initial.start_frame;end
if isfield(initial,'stop_frame'),rangeDefaults(2)=initial.stop_frame;end
rangeEdits=gobjects(1,2);rangeKeys={'start_frame','stop_frame'};rangeCaptions={'Start frame','Stop frame'};
for k=1:2
    x=.05+(k-1)*.15;
    uicontrol(fig,'Style','text','Units','normalized','Position',[x .143 .135 .027], ...
        'String',rangeCaptions{k},'HorizontalAlignment','left');
    rangeEdits(k)=uicontrol(fig,'Style','edit','Units','normalized','Position',[x .099 .125 .041], ...
        'BackgroundColor','white','HorizontalAlignment','center', ...
        'String',num2str(rangeDefaults(k)),'Tag',rangeKeys{k},'Callback',@changeRange);
end
rangeLabel=uicontrol(fig,'Style','text','Units','normalized','Position',[.35 .143 .285 .027], ...
    'String','Calibration frame','HorizontalAlignment','left');
slider=uicontrol(fig,'Style','slider','Units','normalized', ...
    'Position',[.35 .104 .285 .029],'Min',1,'Max',max(2,numel(files)), ...
    'Value',1,'SliderStep',[1/max(1,numel(files)-1) min(1,10/max(1,numel(files)-1))], ...
    'Tag','heightSlider','Callback',@changeHeight);
if numel(files)==1,set(slider,'Enable','off');end
listener=addlistener(slider,'ContinuousValueChange',@changeHeight);
setappdata(fig,'sliderListener',listener);
status=uicontrol(fig,'Style','text','Units','normalized','Position',[.05 .035 .59 .037], ...
    'FontSize',10,'HorizontalAlignment','left');
uicontrol(fig,'Style','text','Units','normalized','Position',[.70 .255 .27 .145], ...
    'HorizontalAlignment','left','String', ...
    {'Drag the box to move it.';'Drag an edge or corner to resize.';'Click the image to recenter.'; ...
    'Choose the first and last training frames.';'Slide through the range to check the crop.'});
uicontrol(fig,'Style','pushbutton','Units','normalized','Position',[.705 .150 .255 .052], ...
    'String','Use this training particle','FontWeight','bold','Tag','acceptSelection','Callback',@accept);
uicontrol(fig,'Style','pushbutton','Units','normalized','Position',[.705 .080 .255 .047], ...
    'String','Cancel','Callback',@cancel);

box=drawrectangle(ax,'Position',[.5 .5 3 3],'Color',accent,'LineWidth',1.5, ...
    'FaceAlpha',.03,'Rotatable',false,'DrawingArea',[.5 .5 cols rows]);
box.Deletable=false;box.Tag='trainingCrop';
movingListener=addlistener(box,'MovingROI',@(source,event) changeBox(source,event,false));
movedListener=addlistener(box,'ROIMoved',@(source,event) changeBox(source,event,true));
setappdata(fig,'roiListeners',{movingListener,movedListener});
changeRange(rangeEdits(1),[]);
autoOnce(); % Initial useful contrast on the selected start frame, then hold.
if waitForSelection
    uiwait(fig);
    if isgraphics(fig),selection=getappdata(fig,'selection');delete(fig);end
end

    function values=refresh(varargin)
        values=cellfun(@str2double,get(edits,'String'))';
        defaults=[initial.X initial.Y initial.width initial.height];
        invalid=~isfinite(values);values(invalid)=defaults(invalid);values=round(values);
        values(3:4)=max(3,2*floor(values(3:4)/2)+1);
        values(3)=min(values(3),2*floor((cols-1)/2)+1);
        values(4)=min(values(4),2*floor((rows-1)/2)+1);
        half=(values(3:4)-1)/2;
        values(1:2)=max(half+1,min([cols rows]-half,values(1:2)));
        for j=1:4,set(edits(j),'String',num2str(values(j)));end
        left=values(1)-half(1);top=values(2)-half(2);
        % Rectangle edges surround exactly the pixels used by imageset_crop.
        if ~movingROI
            lastPosition=[left-.5 top-.5 values(3:4)];box.Position=lastPosition;
        end
        set(cropHandle,'CData',im(top:top+values(4)-1,left:left+values(3)-1), ...
            'XData',[1 values(3)],'YData',[1 values(4)]);
        axis(preview,'image');axis(preview,'tight');axis(preview,'off');
        applyLut();
        range=readRange();plane=min(numel(files),round(get(slider,'Value')));
        [~,name,ext]=fileparts(files{plane});
        if plane>=range(1)&&plane<=range(2),note='included';else,note='outside training range';end
        title(ax,sprintf('Frame %d / %d  -  %s',plane,numel(files),note),'Interpreter','none');
        title(preview,sprintf('Crop preview  -  %d x %d pixels',values(3:4)),'Interpreter','none');
        set(rangeLabel,'String',sprintf('Training frames %d - %d (%d planes)',range(1),range(2),diff(range)+1));
        set(status,'String',[name ext]);
        setappdata(fig,'selection',struct('X',values(1),'Y',values(2),'width',values(3),'height',values(4), ...
            'start_frame',range(1),'stop_frame',range(2)));
        drawnow limitrate;
    end
    function range=readRange(source)
        range=cellfun(@str2double,get(rangeEdits,'String'))';
        defaults=[1 numel(files)];invalid=~isfinite(range);range(invalid)=defaults(invalid);
        range=max(1,min(numel(files),round(range)));minimum=min(3,numel(files));
        if nargin>0&&isequal(source,rangeEdits(2))
            range(2)=max(minimum,range(2));range(1)=min(range(1),range(2)-minimum+1);
        else
            range(1)=min(numel(files)-minimum+1,range(1));range(2)=max(range(2),range(1)+minimum-1);
        end
        for j=1:2,set(rangeEdits(j),'String',num2str(range(j)));end
    end
    function changeRange(source,~)
        range=readRange(source);plane=max(range(1),min(range(2),round(get(slider,'Value'))));
        set(slider,'Value',plane);changeHeight();
    end
    function changeHeight(varargin)
        plane=max(1,min(numel(files),round(get(slider,'Value'))));set(slider,'Value',plane);
        im=imread(files{plane});assert(isequal(size(im),[rows cols]),'Calibration image dimensions differ.');
        set(imageHandle,'CData',im);refresh();
    end
    function limits=imageLimits(image)
        limits=double([min(image(:)) max(image(:))]);
        if limits(2)<=limits(1),limits(2)=limits(1)+max(1,eps(limits(1)));end
    end
    function changeLutMode(varargin)
        % Algorithm selection does not recalculate contrast. Auto once does.
        applyLut();drawnow limitrate;
    end
    function autoOnce(varargin)
        algorithm=get(lutMode,'Value');
        switch algorithm
            case 1
                % Rossi-Barnkob: rule from DefocusTracker's imageset_show.m.
                % Filtering is only for statistics; display/training use raw im.
                filtered=medfilt2(im,[3 3]);
                limits=[0 median(filtered(:))*3];
                if limits(2)<=10,limits=[0 max(filtered(:))/2+1];end
                limits=double(limits);
            case 2
                % Robust to isolated bright/dark outliers, in raw intensity units.
                limits=prctile(double(im(:)),[1 99]);
            case 3
                limits=imageLimits(im);
            otherwise
                return % Manual Linear: edit Black and White directly.
        end
        if any(~isfinite(limits))||limits(2)<=limits(1),limits=imageLimits(im);end
        lutLimits=limits;appliedAlgorithm=lutAlgorithms{algorithm};
        applyLut();drawnow limitrate;
    end
    function changeManualLut(varargin)
        requested=[str2double(get(lutBlack,'String')) str2double(get(lutWhite,'String'))];
        if all(isfinite(requested))&&requested(2)>requested(1)
            lutLimits=requested;appliedAlgorithm='Manual Linear';set(lutMode,'Value',4);
        end
        applyLut();drawnow limitrate;
    end
    function applyLut()
        algorithm=get(lutMode,'Value');
        if algorithm==4,enabled='off';else,enabled='on';end
        set(lutAuto,'Enable',enabled);
        set(lutBlack,'Enable','on','String',sprintf('%.9g',lutLimits(1)));
        set(lutWhite,'Enable','on','String',sprintf('%.9g',lutLimits(2)));
        clim(ax,lutLimits);clim(preview,lutLimits);
        setappdata(fig,'lut',struct('selected_algorithm',lutAlgorithms{algorithm}, ...
            'applied_algorithm',appliedAlgorithm,'limits',lutLimits));
    end
    function changeBox(source,~,finished)
        if movingROI,return,end
        position=source.Position;values=zeros(1,4);
        for dimension=1:2
            limit=[cols rows];maximum=limit(dimension);
            span=max(3,min(2*floor((maximum-1)/2)+1,2*floor(round(position(dimension+2))/2)+1));
            leftChange=abs(position(dimension)-lastPosition(dimension));
            rightChange=abs(sum(position([dimension dimension+2]))-sum(lastPosition([dimension dimension+2])));
            resizing=abs(position(dimension+2)-lastPosition(dimension+2))>1e-6;
            if ~resizing
                first=max(1,min(maximum-span+1,round(position(dimension)+.5)));
            elseif leftChange>rightChange
                % Preserve the opposite border when dragging the left/top.
                right=max(3,min(maximum,round(sum(position([dimension dimension+2]))-.5)));
                span=min(span,2*floor((right-1)/2)+1);first=right-span+1;
            else
                first=max(1,min(maximum-2,round(position(dimension)+.5)));
                span=min(span,2*floor((maximum-first)/2)+1);
            end
            values(dimension)=first+(span-1)/2;values(dimension+2)=span;
        end
        for j=1:4,set(edits(j),'String',num2str(values(j)));end
        movingROI=~finished;refresh();movingROI=false;
    end
    function clickParticle(varargin)
        point=get(ax,'CurrentPoint');
        set(edits(1),'String',num2str(round(point(1,1))));set(edits(2),'String',num2str(round(point(1,2))));refresh();
    end
    function accept(varargin)
        refresh();setappdata(fig,'accepted',true);uiresume(fig);
    end
    function cancel(varargin)
        setappdata(fig,'selection',[]);uiresume(fig);
        if ~waitForSelection,delete(fig);end
    end
end
