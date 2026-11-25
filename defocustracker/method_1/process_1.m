function [dat, process_time] = process_1(model,imageset,frame_index)

disp('Evaluation started ...')

cal = model.parameter;
cal.imwidth = model.training.imwidth;
cal.gauss_imheight = model.training.imheight;
cal.gauss_filter = model.training.gauss_filter;
cal.median_filter = model.training.median_filter;
cal.imresize = model.training.imresize;
cal.training_imageset.preloaded_images=[];

par = model.processing;
use_gpu=model.processing.use_gpu;
use_parfor=model.processing.use_parfor;

dat = dtracker_create('dataset');
eval_times = frame_index*0;

ind = find(frame_index<=imageset.n_frames,1,'last');
frame_index = frame_index(1:ind);

n_frames = length(frame_index);

% to be improved
try 
    bkg_image_mean = model.processing.bkg_image_mean;
catch
    bkg_image_mean = 0;
end
% if size(bkg_image_mean,1)==H && size(bkg_image_mean,2)==W
%     imbkg = bkg_image_mean;
if bkg_image_mean>1
    n_mean = min([bkg_image_mean,imageset.n_frames]); 
    fr_scan = round(linspace(1,imageset.n_frames,n_mean));
    imbkg = dtracker_show(imageset,1);
    [H,W] = size(imbkg);
    imbkg = imbkg(:);
    for ii = 2:n_mean
        im = dtracker_show(imageset,fr_scan(ii));
        imbkg = min([imbkg(:), im(:)],[],2);
    end
    imbkg = reshape(imbkg,H,W);
else
    imbkg = 0;
end
%use parfor loop if flag is toggled on in model.processing
if use_parfor
    disp("processing in paralell")
    if strcmp(imageset.im_type, 'preloaded_imageset')
        disp("special case with preloaded imageset")
        images=imageset.preloaded_images;
        parfor n = 1:n_frames
        im = images{n};
        im = im-imbkg;
        
        if ~isempty(par.roi)
            im = imcrop(im, par.roi);
        end
            
        [datd, eval_times(n)] = singleprocess_1(im,cal,par,use_gpu);
        datd.fr(:) = frame_index(n);
        dat = [dat; datd];
        timestr = secs2hms(eval_times(n)*(n_frames-n));
        if n==n_frames, timestr = ' '; end
        
        note_eval = [num2str(n),'/',num2str(n_frames),' - ',...
            num2str(round(eval_times(n))),' s - ',...
            num2str(length(datd.X)),' particles - time left: ',...
            timestr];
        
        disp(note_eval)
        end
    else
        parfor n = 1:n_frames
            im = imageset_read(imageset,(frame_index(n)));
            im = im-imbkg;
            
            if ~isempty(par.roi)
                im = imcrop(im, par.roi);
            end
                
            [datd, eval_times(n)] = singleprocess_1(im,cal,par,use_gpu);
            datd.fr(:) = frame_index(n);
            dat = [dat; datd];
            timestr = secs2hms(eval_times(n)*(n_frames-n));
            if n==n_frames, timestr = ' '; end
            
            note_eval = [num2str(n),'/',num2str(n_frames),' - ',...
                num2str(round(eval_times(n))),' s - ',...
                num2str(length(datd.X)),' particles - time left: ',...
                timestr];
            
            disp(note_eval)
        end
    end
%otherwise use normal "for loop"
else
        for n = 1:n_frames
        im = imageset_read(imageset,(frame_index(n)));
        im = im-imbkg;
        
        if ~isempty(par.roi)
            im = imcrop(im, par.roi);
        end
            
        [datd, eval_times(n)] = singleprocess_1(im,cal,par,use_gpu);
        datd.fr(:) = frame_index(n);
        dat = [dat; datd];
        timestr = secs2hms(eval_times(n)*(n_frames-n));
        if n==n_frames, timestr = ' '; end
        
        note_eval = [num2str(n),'/',num2str(n_frames),' - ',...
            num2str(round(eval_times(n))),' s - ',...
            num2str(length(datd.X)),' particles - time left: ',...
            timestr];
        
        disp(note_eval)
    end
end

if ~isempty(par.roi)
    dat.X = dat.X+par.roi(1);
    dat.Y = dat.Y+par.roi(2);
end

dat.Properties.UserData.ParticleBoundaries = model.parameter.images_boundary;


process_time = sum(eval_times);
timestr = secs2hms(process_time);
disp('----------------------------')
disp(['Evaluation done! Total Time: ',timestr])




