function [dat, process_time] = process_piv2d(model,img,fr1)

disp('Evaluation started ...')


step = model.processing.step;
IW1 = model.processing.IW1;
pitch = model.processing.pitch;
DX_minmax = model.processing.DX_minmax;
DY_minmax = model.processing.DY_minmax;
medfilt_kernel = model.processing.medfilt_kernel;
medfilt_replace_pxdiff = model.processing.medfilt_replace_pxdiff;
bkg_image_mean = model.processing.bkg_image_mean;

%%%%%%%%%%%%%%%
fr1 = fr1(:)';
fr2 = fr1+step;
flag = fr2<=img.n_frames;
fr1 = fr1(flag);
fr2 = fr2(flag);
IW1 = IW1+mod(IW1/2,1)*2;
IW2 = IW1+2;%IW1+[diff(DX_minmax), diff(DY_minmax)];
% IW2 = IW2-mod(IW2/2,1)*2;

cc_XY = [(IW1(1)+IW2(1))/2, (IW1(2)+IW2(2))/2];

%%

im = dtracker_show(img,1);
[H,W] = size(im);
[Xg, Yg] = meshgrid(IW2(1)/2:pitch(1):W-IW2(1)/2,IW2(2)/2:pitch(2):H-IW2(2)/2);
X = Xg(:);
Y = Yg(:);
X = X+round((W-max(X)-min(X))/2);
Y = Y+round((H-max(Y)-min(Y))/2);

if ~isempty(model.processing.roi)
    flag = X>=model.processing.roi(1) &...
        X<=model.processing.roi(1)+model.processing.roi(3) &...
        Y>=model.processing.roi(2) &...
        Y<=model.processing.roi(2)+model.processing.roi(4); 
    X = X(flag);
    Y = Y(flag);
end
id = (1:length(X))';


%%
n_frames = length(fr1);
dat = dtracker_create('dataset',length(X)*n_frames);
fr_del = repmat(fr1,length(X),1);
dat.fr(:) = reshape(fr_del,length(fr1)*length(X),1);
dat.id(:) = repmat(id,n_frames,1);
dat.X(:) = repmat(X,n_frames,1);
dat.Y(:) = repmat(Y,n_frames,1);



[DX_mask, DY_mask] = meshgrid(-cc_XY(1)+1:cc_XY(1)-1,...
    -cc_XY(2)+1:cc_XY(2)-1);
mask = DX_mask*0;
mask(DX_mask>=DX_minmax(1)-2 & DX_mask<=DX_minmax(2)+2 &...
    DY_mask>=DY_minmax(1)-2 & DY_mask<=DY_minmax(2)+2) = .01;
mask(DX_mask>=DX_minmax(1)-1 & DX_mask<=DX_minmax(2)+1 &...
    DY_mask>=DY_minmax(1)-1 & DY_mask<=DY_minmax(2)+1) = .5;
mask(DX_mask>=DX_minmax(1) & DX_mask<=DX_minmax(2) &...
    DY_mask>=DY_minmax(1) & DY_mask<=DY_minmax(2)) = 1;

if size(bkg_image_mean,1)==H && size(bkg_image_mean,2)==W
    imbkg = bkg_image_mean;
elseif bkg_image_mean>1
    n_mean = min([bkg_image_mean,img.n_frames]); 
    fr_scan = round(linspace(1,img.n_frames,n_mean));
    imbkg = dtracker_show(img,1);
    imbkg = imbkg(:);
    for ii = 2:n_mean
        im = dtracker_show(img,fr_scan(ii));
        imbkg = min([imbkg(:), im(:)],[],2);
    end
    imbkg = reshape(imbkg,H,W);
else
    imbkg = 0;
end

eval_times = fr1*0;
for n = 1:length(fr1)
    tic
    
    DX = zeros(size(X));
    DY = zeros(size(Y));

    im1 = dtracker_show(img,fr1(n));
    im2 = dtracker_show(img,fr2(n));
    im1 = im1-imbkg;
    im2 = im2-imbkg;
    
    for ii = 1:length(X)
        try
            iw1 = im1((-IW1(2)/2+1:IW1(2)/2)+Y(ii),(-IW1(1)/2+1:IW1(1)/2)+X(ii));
            iw2 = im2((-IW2(2)/2+1:IW2(2)/2)+Y(ii),(-IW2(1)/2+1:IW2(1)/2)+X(ii));
            field = normxcorr2(iw1,iw2);
            field = field-min(field(:));
            
            field = mask.*field;
            [iy, ix] = find(field==max(field(:)),1);
            DX(ii) = ix - cc_XY(1) + (log(field(iy,ix-1))-log(field(iy,ix+1))) /...
                ( 2*log(field(iy,ix-1)) - 4* log(field(iy,ix)) + 2* log(field(iy,ix+1)));
            DY(ii) = iy - cc_XY(2) + (log(field(iy-1,ix))-log(field(iy+1,ix))) /...
                ( 2*log(field(iy-1,ix)) - 4* log(field(iy,ix)) + 2* log(field(iy+1,ix)));
        catch            
        end
        
    end
    
    if medfilt_kernel>2
        
        DXg = Xg*0;
        DXg(:) = DX;
        DXgm = medfilt2(DXg,[medfilt_kernel,medfilt_kernel],...
            'symmetric');
        
        DYg = Xg*0;
        DYg(:) = DY;
        DYgm = medfilt2(DYg,[medfilt_kernel,medfilt_kernel],...
            'symmetric');
        errX = abs(DXg-DXgm);
        errY = abs(DYg-DYgm);
        
        flagm = errX>medfilt_replace_pxdiff  |...
            errY>medfilt_replace_pxdiff ;
        
        DXn = DXg;
        DXn(flagm) = DXgm(flagm);
        DYn = DYg;
        DYn(flagm) = DYgm(flagm);
        
        DX = DXn(:);
        DY = DYn(:);
        
    end

    flagf = dat.fr==fr1(n);
    dat.DX(flagf) = DX;
    dat.DY(flagf) = DY;

    eval_times(n) = toc;
    timestr = secs2hms(eval_times(n)*(n_frames-n));
    if n==length(fr1), timestr = '0 secs '; end
    note_eval = [num2str(n),'/',num2str(n_frames),' - ',...
        num2str(round(eval_times(n))),' s - time left: ',...
        timestr];  
    disp(note_eval)
end

dat.Properties.UserData.TrackingStep = step;

process_time = sum(eval_times);
timestr = secs2hms(process_time);
disp('----------------------------')
disp(['Evaluation done! Total Time: ',timestr])

