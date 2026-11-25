function [data,timings]=native_localize(tracker,model,raw_image)
% RAM input -> MATLAB preprocessing/depth guess -> native candidate/refinement.
% Initialize DefocusTrackerNative once, outside the frame loop.
assert(model.processing.walking_step~=0,'DefocusNative:Mode','This entry point implements 3D walking refinement.');
assert(model.processing.bkg_image_mean==0 && isempty(model.processing.roi), ...
    'DefocusNative:Preprocessing','This entry point currently expects full-frame data without background-image subtraction.');
started=tic; part=tic;
image=double(raw_image); training=model.training;
if training.median_filter>2
    image=double(medfilt2(uint16(image),[training.median_filter training.median_filter],'symmetric'));
end
if training.gauss_filter>1
    psf=fspecial('gaussian',training.gauss_filter,training.gauss_filter);
    image=imfilter(image,psf,'symmetric','conv');
end
if training.imresize~=1, image=imresize(image,training.imresize); end
preprocessing_s=toc(part);
candidate_s=0; depth_guess_s=0; refinement_s=0; native_candidate_s=0; native_refinement_s=0;
all=zeros(0,4);
for iteration=1:model.processing.n_iteration
    part=tic;
    [candidates,scores,native_time]=tracker.candidates(image);
    candidate_s=candidate_s+toc(part); native_candidate_s=native_candidate_s+native_time;
    part=tic;
    coarse=initial_depth(tracker,image,candidates,scores,model);
    depth_guess_s=depth_guess_s+toc(part);
    part=tic;
    [values,stats]=tracker.refine(image,coarse);
    refinement_s=refinement_s+toc(part); native_refinement_s=native_refinement_s+stats.seconds;
    all=[all;values]; %#ok<AGROW>
    if iteration<model.processing.n_iteration, image=tracker.lastImage(); end
end
if training.imresize~=1, all(:,1:2)=all(:,1:2)/training.imresize; end
n=size(all,1); zero=zeros(n,1);
data=table(zero,all(:,1),all(:,2),all(:,3),zero,zero,zero,zero,all(:,4), ...
    'VariableNames',{'fr','X','Y','Z','DX','DY','DZ','id','cm'});
timings=struct('preprocessing_s',preprocessing_s,'candidate_s',candidate_s, ...
    'depth_guess_s',depth_guess_s,'refinement_s',refinement_s, ...
    'native_candidate_s',native_candidate_s,'native_refinement_s',native_refinement_s, ...
    'total_s',toc(started),'backend',tracker.info());
end

function coarse=initial_depth(tracker,image,candidates,scores,model)
cal=model.parameter; par=model.processing; indices=tracker.SelectedIndices;
size_t=size(cal.images{1})+8;
ii=round(-size_t(1)/2)+1:round(size_t(1)/2)-1;
jj=round(-size_t(2)/2)+1:round(size_t(2)/2)-1;
circx=par.no_overlap_radius*cos(linspace(0,2*pi,16));
circy=par.no_overlap_radius*sin(linspace(0,2*pi,16));
x=candidates(:,2); y=candidates(:,3); c=candidates(:,4);
sim_map0=cal.similarity_map(indices,:);
coarse=zeros(numel(c),5); count=0;
while ~isempty(c)
    selected=find(c==max(c),1);
    sim=scores(selected,:).';
    [depth,similarity]=guess_stack(sim_map0,sim);
    used=indices;
    crop=image(ii+round(y(selected)),jj+round(x(selected)));
    while ~any(used==depth)
        used=[used,depth]; %#ok<AGROW>
        map=tracker.localCorrelation(crop,depth,true);
        sim=[sim;max(map(:))]; %#ok<AGROW>
        [depth,similarity]=guess_stack(cal.similarity_map(used,:),sim);
    end
    count=count+1; coarse(count,:)=[x(selected),y(selected),depth,c(selected),similarity];
    outside=inpolygon(x,y,x(selected)+circx,y(selected)+circy);
    x(outside)=[];y(outside)=[];c(outside)=[];scores(outside,:)=[];
end
coarse=coarse(1:count,:);
end
function [depth,similarity]=guess_stack(map,values)
% Preserve the existing calculation; vectorization/cache is a separate TODO.
match=zeros(size(map(1,:))); mean_values=mean(values);
for k=1:numel(match)
    column=map(:,k); average=mean(column);
    match(k)=sum((column-average).*(values-mean_values))/ ...
        sqrt(sum((column-average).^2)*sum((values-mean_values).^2));
end
[similarity,depth]=max(match);
end
