function mydat = ptv_nearest(tracking,mydat,frame_index,varargin)
%
% tracking.tracking_step is the frame distance to the second frame
% tracking.bounding_box is [minDX maxDX minDY maxDY minDZ maxDZ]
% dat is a DefocusTracker dataset
% frame_index is a vector with frames index
%
% The ptv analysis is performed with the following scheme:
% (frame_index(1)+tracking_step) - (frame_index(1))
% (frame_index(2)+tracking_step) - (frame_index(2))
% (frame_index(3)+tracking_step) - (frame_index(3))
% .....


% dat = dtracker_postprocess('unscale',dat);
if ismember('x',mydat.Properties.VariableNames)
    disp('Warning: The data are scaled! Check the bounding box')
    tracking.bounding_box = tracking.bounding_box*mydat.Properties.UserData.Scaling.dt*tracking.tracking_step;
end

if ~isfield(tracking,'fitting')
    tracking.fitting = 'off'; ff_analysis = 0;
elseif strcmp(tracking.fitting,'4frame') == 1
    ff_analysis = 1;par.poly=2;
elseif strcmp(tracking.fitting,'Gaussian') == 1
    ff_analysis = 1;par.poly=1;
elseif strcmp(tracking.fitting,'off') == 1
    ff_analysis = 0;
end
    
fr_1 = frame_index;
fr_2 = fr_1+tracking.tracking_step;
indgood = fr_2<=max(mydat.fr);
fr_1 = fr_1(indgood);
fr_2 = fr_2(indgood);

par.minmax_DX = tracking.bounding_box(1:2);
par.minmax_DY = tracking.bounding_box(3:4);
par.minmax_DZ = tracking.bounding_box(5:6);

% if length(fr_1)==1
%     fr_step = tracking.tracking_step;
% else
%     fr_step = fr_1(2)-fr_1(1);
% end

if nargin==3
    disp('Tracking - nearest neighbor - started...')
end
tic

dat.fr = mydat.fr';
xyz = table2array(mydat(:,2:4));
dat.x = xyz(:,1)';
dat.y = xyz(:,2)';
dat.z = xyz(:,3)';
dat.dx = dat.x*0;
dat.dy = dat.x*0;
dat.dz = dat.x*0;
dat.id = dat.x*0;

if strcmp(tracking.predictor,'on') % && fr_step==tracking.tracking_step
    % time resolved with predictor
    disp('Warning: predictor not extensive tested yet')
    count_id = dat.id*0;
    lastID = 0;
    dxp1 = zeros(1,sum(dat.fr==fr_1(1)));
    dyp1 = dxp1;
    for n = 1:length(fr_1)
        
        [dat, id1, id2]  = ptv_predictor(par,dat,fr_1(n),fr_2(n),dxp1,dyp1);
        id10 = find(dat.id(id1)==0);
        newID = lastID+(1:length(id10));
        dat.id(id1(id10)) = newID;
        dat.id(id2) = dat.id(id1);
        count_id(id2) = count_id(id1)+1;
        lastID = max(dat.id);
        
        dxp1 = dat.dx*0;
        dyp1 = dat.dy*0;
        flag = count_id(id2)>2;
        dxp1(id2(flag)) = dat.dx(id2(flag));
        dyp1(id2(flag)) = dat.dy(id2(flag));
        dxp1 = dxp1(dat.fr==fr_2(n));
        dyp1 = dyp1(dat.fr==fr_2(n));
    end
% elseif strcmp(tracking.predictor,'off')%% && fr_step==tracking.tracking_step
else
    % time resolved without predictor
    lastID = 0;
    for n = 1:length(fr_1)
        [dat, id1, id2]  = ptv(par,dat,fr_1(n),fr_2(n));
        id10 = find(dat.id(id1)==0);
        newID = lastID+(1:length(id10));
        dat.id(id1(id10)) = newID;
        dat.id(id2) = dat.id(id1);
        lastID = max(dat.id);
    end
    
    % else
    % Double-frame with ID = 1 in first frame and ID = 2 in second frame.
    % Not-implemented now!
    %     for n = 1:length(fr_1)
    %         % double frame ptv
    %         [dat, id1, id2] = ptv(par,dat,fr_1(n),fr_2(n));
    %         dat.id(id1) = 1;
    %         dat.id(id2) = 2;
    %     end
end

if ff_analysis == 1
    [dat] = fourframe(par,dat);
end

mydat(:,2:8) = array2table([dat.x; dat.y; dat.z; dat.dx; dat.dy; dat.dz; dat.id]');

mydat.Properties.UserData.TrackingStep = tracking.tracking_step;
totTime = toc;
minTime = floor(totTime/60);
secTime = round(totTime-minTime*60);

if ismember('x',mydat.Properties.VariableNames)
    mydat.vx = mydat.vx/(mydat.Properties.UserData.Scaling.dt*tracking.tracking_step);
    mydat.vy = mydat.vy/(mydat.Properties.UserData.Scaling.dt*tracking.tracking_step);
    mydat.vz = mydat.vz/(mydat.Properties.UserData.Scaling.dt*tracking.tracking_step);
end


if nargin==3
    disp(['Tracking done! Total time: ',num2str(minTime),' min ',num2str(secTime),' sec'])
    if strcmp(tracking.fitting,'4frame') == 1
        disp(['Four frame fitting used.'])
    elseif strcmp(tracking.fitting,'Gaussian') == 1
        disp(['Gaussian smoothing used.'])
    end;    
end
end

function [ell1, varargout] = ptv(par,ell1,n_1,n_2)

ind01 = find(ell1.fr==n_1);
ind02 = find(ell1.fr==n_2);

if ~isempty(ind01) && ~isempty(ind02)
    
    x1 = ell1.x(ind01); y1 = ell1.y(ind01); z1 = ell1.z(ind01);
    x2 = ell1.x(ind02); y2 = ell1.y(ind02); z2 = ell1.z(ind02);
    ind1 = []; ind2 = [];
    
    d1 = length(x1);
    d2 = length(x2);
    ind_d1 = 1:d1;
    ind_d2 = 1:d2;
    
    diff_x = repmat(x2, [d1 1]) - repmat(x1', [1 d2]);
    diff_y = repmat(y2, [d1 1]) - repmat(y1', [1 d2]);
    diff_z = repmat(z2, [d1 1]) - repmat(z1', [1 d2]);
    
    mask = false(size(diff_x));
    mask(diff_x<par.minmax_DX(1) | diff_x>par.minmax_DX(2))=1;
    mask(diff_y<par.minmax_DY(1) | diff_y>par.minmax_DY(2))=1;
    mask(diff_z<par.minmax_DZ(1) | diff_z>par.minmax_DZ(2))=1;
    
    diff_xyz = sqrt(diff_x.^2+diff_y.^2);%+diff_z.^2);
    diff_xyz(mask) = NaN;
    
    [~, jj] = find(diff_xyz==min(diff_xyz(:)),1);
    while ~isempty(jj) && ~isempty(diff_xyz)
        [ii, jj] = find(diff_xyz==min(diff_xyz(:)),1);
        ind1 = [ind1 ind_d1(ii)];
        ind2 = [ind2 ind_d2(jj)];
        diff_xyz(ii,:) = []; diff_xyz(:,jj) = [];
        ind_d1(ii) = []; ind_d2(jj) = [];
    end
    
    if ~isempty(ind2)
        ell1.dx(ind01(ind1)) = x2(ind2)-x1(ind1);
        ell1.dy(ind01(ind1)) = y2(ind2)-y1(ind1);
        ell1.dz(ind01(ind1)) = z2(ind2)-z1(ind1);
        ell1.dx(ind02(ind2)) = x2(ind2)-x1(ind1);
        ell1.dy(ind02(ind2)) = y2(ind2)-y1(ind1);
        ell1.dz(ind02(ind2)) = z2(ind2)-z1(ind1);
    end
    
    if nargout == 3
        varargout{1} = ind01(ind1);
        varargout{2} = ind02(ind2);
    end
    
elseif nargout == 3
    varargout{1} = [];
    varargout{2} = [];
end
end

function [ell1, varargout] = ptv_predictor(par,ell1,n_1,n_2,dxp1,dyp1)
    ind01 = find(ell1.fr==n_1);
    ind02 = find(ell1.fr==n_2);
    
    if ~isempty(ind01) && ~isempty(ind02)
        
        x1 = ell1.x(ind01); y1 = ell1.y(ind01); z1 = ell1.z(ind01);
        x2 = ell1.x(ind02); y2 = ell1.y(ind02); z2 = ell1.z(ind02);
        ind1 = []; ind2 = [];
        
        d1 = length(x1);
        d2 = length(x2);
        ind_d1 = 1:d1;
        ind_d2 = 1:d2;
        
        diff_x = repmat(x2, [d1 1]) - repmat(x1', [1 d2]);
        diff_y = repmat(y2, [d1 1]) - repmat(y1', [1 d2]);
        diff_z = repmat(z2, [d1 1]) - repmat(z1', [1 d2]);
        
        mask = false(size(diff_x));
        mask(diff_x<par.minmax_DX(1) | diff_x>par.minmax_DX(2))=1;
        mask(diff_y<par.minmax_DY(1) | diff_y>par.minmax_DY(2))=1;
        mask(diff_z<par.minmax_DZ(1) | diff_z>par.minmax_DZ(2))=1;
    
        diff_x = diff_x - repmat(dxp1', [1 d2]);
        diff_y = diff_y - repmat(dyp1', [1 d2]);
    
        diff_xyz = sqrt(diff_x.^2+diff_y.^2);%+diff_z.^2);
        diff_xyz(mask) = NaN;
        
        [~, jj] = find(diff_xyz==min(diff_xyz(:)),1);
        while ~isempty(jj) && ~isempty(diff_xyz)
            [ii, jj] = find(diff_xyz==min(diff_xyz(:)),1);
            ind1 = [ind1 ind_d1(ii)];
            ind2 = [ind2 ind_d2(jj)];
            diff_xyz(ii,:) = []; diff_xyz(:,jj) = [];
            ind_d1(ii) = []; ind_d2(jj) = [];
        end
        
        if ~isempty(ind2)
            ell1.dx(ind01(ind1)) = x2(ind2)-x1(ind1);
            ell1.dy(ind01(ind1)) = y2(ind2)-y1(ind1);
            ell1.dz(ind01(ind1)) = z2(ind2)-z1(ind1);
            ell1.dx(ind02(ind2)) = x2(ind2)-x1(ind1);
            ell1.dy(ind02(ind2)) = y2(ind2)-y1(ind1);
            ell1.dz(ind02(ind2)) = z2(ind2)-z1(ind1);
        end
        
        if nargout == 3
            varargout{1} = ind01(ind1);
            varargout{2} = ind02(ind2);
        end
        
    elseif nargout == 3
        varargout{1} = [];
        varargout{2} = [];
    end
end

% added by Christian Cierpka 28.04.2023
% adds the velocity determination by fitting of four position with a polynomial 
% for details see https://link.springer.com/article/10.1007/s00348-013-1533-3

function [ell1] = fourframe(par,ell1) 

        %funtion to interpolate the position of a vector and determine the
        %displacement from the fit or integration
        
        x=ell1.x; y = ell1.y; z = ell1.z; id = ell1.id; fr = ell1.fr;

        p.weight_ipl=[1 1 1 1]; %all positions are euqally good 
        p.dt = [1 1 1]; %all frames have the same distance (for time resolved data)
        p.window = 4; %for Gaussian smoothing
 
        
        p.poly = par.poly;
        if p.poly==2; disp(['Four frame polyfit started ...']);end;
        if p.poly==1; disp(['Gaussian trajectory smoothing started ...']) ;end;
        
        
        t=[1 1+p.dt(1) 1+p.dt(1)+p.dt(2) 1+p.dt(1)+p.dt(2)+p.dt(3)]';
        t_data=(t(3)-0.5*(t(3)-t(2))); %point in time for the new vector
       
        x_ipl=[];
        y_ipl=[];
        z_ipl=[];
        dx_ipl=[];
        dy_ipl=[];
        dz_ipl=[];
        fr_ipl=[];
        id_ipl=[];
        
        kk=1;

        for ind=0:max(id)
            
        index=find(id==ind);
              
        if max(size(index))>3 & ind>0


        if p.poly == 1 % Gaussian smoothing of the position        
         
        x_ipl(index) = smoothdata(x(index)',"gaussian",p.window);
        y_ipl(index) = smoothdata(y(index)',"gaussian",p.window);
        z_ipl(index) = smoothdata(z(index)',"gaussian",p.window);
        id_ipl(index) = id(index);
        fr_ipl(index) = fr(index);
        
        
        dx_ipl(index(1:end-1)) = x_ipl(index(2:end))-x_ipl(index(1:end-1));
        dy_ipl(index(1:end-1)) = y_ipl(index(2:end))-y_ipl(index(1:end-1));
        dz_ipl(index(1:end-1)) = z_ipl(index(2:end))-z_ipl(index(1:end-1));

        x_ipl(index(1:end-1)) = (x_ipl(index(1:end-1))+x_ipl(index(2:end)))./2;
        y_ipl(index(1:end-1)) = (y_ipl(index(1:end-1))+y_ipl(index(2:end)))./2;
        z_ipl(index(1:end-1)) = (z_ipl(index(1:end-1))+z_ipl(index(2:end)))./2;

        dx_ipl(index(end)) = nan;dy_ipl(index(end)) = nan;dz_ipl(index(end)) = nan;
        x_ipl(index(end)) = nan;y_ipl(index(end)) = nan;z_ipl(index(end)) = nan;

        mask = false(size(dx_ipl));
        mask(dx_ipl<par.minmax_DX(1) | dx_ipl>par.minmax_DX(2))=1;
        mask(dy_ipl<par.minmax_DY(1) | dy_ipl>par.minmax_DY(2))=1;
        mask(dz_ipl<par.minmax_DZ(1) | dz_ipl>par.minmax_DZ(2))=1;
    
        dx_ipl(mask) = NaN;dy_ipl(mask) = NaN;dz_ipl(mask) = NaN;

         elseif p.poly==2 % piecewise polynomical interpolation (here
%         second order because it is the most physical and working
%         approach)
        id_ipl(index) = id(index);
        fr_ipl(index) = fr(index);
        % start of the trajectory

        xfit=fit(t,x(index(1:4))','poly2','Weight',p.weight_ipl);
        yfit=fit(t,y(index(1:4))','poly2','Weight',p.weight_ipl);
        zfit=fit(t,z(index(1:4))','poly2','Weight',p.weight_ipl);
        %Assigning positions 
        x_ipl(index(1))=xfit(1.5);
        y_ipl(index(1))=yfit(1.5);
        z_ipl(index(1))=zfit(1.5);
        %Assigning velocities 
%         xo= xfit.p1*t(2)^2 + xfit.p2*t(2); 
%         xu= xfit.p1*t(1)^2 + xfit.p2*t(1); 
%         yo= yfit.p1*t(2)^2 + yfit.p2*t(2); 
%         yu= yfit.p1*t(1)^2 + yfit.p2*t(1); 
%         zo= zfit.p1*t(2)^2 + zfit.p2*t(2); 
%         zu= zfit.p1*t(1)^2 + zfit.p2*t(1); 
%         dx_ipl(index(1))=(xo-xu)/(t(2)-t(1));  
%         dy_ipl(index(1))=(yo-yu)/(t(2)-t(1)); 
%         dz_ipl(index(1))=(zo-zu)/(t(2)-t(1));  

        dx_ipl(index(1))=xfit.p2 + 2*xfit.p1.*1.5;
        dy_ipl(index(1))=yfit.p2 + 2*yfit.p1.*1.5;
        dz_ipl(index(1))=zfit.p2 + 2*zfit.p1.*1.5;

        for kk = 1:size(index,2)-3    
        xfit=fit(t,x(index(kk:kk+3))','poly2','Weight',p.weight_ipl);
        yfit=fit(t,y(index(kk:kk+3))','poly2','Weight',p.weight_ipl);
        zfit=fit(t,z(index(kk:kk+3))','poly2','Weight',p.weight_ipl);

        %Assigning positions 
        x_ipl(index(kk+1))=xfit(2.5);
        y_ipl(index(kk+1))=yfit(2.5);
        z_ipl(index(kk+1))=zfit(2.5);
        %Assigning velocities 
        dx_ipl(index(kk+1))=xfit.p2 + 2*xfit.p1.*2.5;
        dy_ipl(index(kk+1))=yfit.p2 + 2*yfit.p1.*2.5;
        dz_ipl(index(kk+1))=zfit.p2 + 2*zfit.p1.*2.5;

        end; %for kk along the trajectory

        % end of the trajectory

        xfit=fit(t,x(index(end-3:end))','poly2','Weight',p.weight_ipl);
        yfit=fit(t,y(index(end-3:end))','poly2','Weight',p.weight_ipl);
        zfit=fit(t,z(index(end-3:end))','poly2','Weight',p.weight_ipl);
        %Assigning positions 
        x_ipl(index(end-1))=xfit(3.5);
        y_ipl(index(end-1))=yfit(3.5);
        z_ipl(index(end-1))=zfit(3.5);
        %Assigning velocities  

        dx_ipl(index(end-1))=xfit.p2 + 2*xfit.p1.*3.5;
        dy_ipl(index(end-1))=yfit.p2 + 2*yfit.p1.*3.5;
        dz_ipl(index(end-1))=zfit.p2 + 2*zfit.p1.*3.5;

        dx_ipl(index(end)) = nan;dy_ipl(index(end)) = nan;dz_ipl(index(end)) = nan;
        x_ipl(index(end)) = nan;y_ipl(index(end)) = nan;z_ipl(index(end)) = nan;
        
%         figure(11), plot(x(index),y(index),'-r*'), hold on;
%         figure(11), plot(x_ipl(index),y_ipl(index),'-g*'), hold on;

        end; %type of trajectoriy fitting/smoothing
        else 
            dx_ipl(index)=nan;

        end;%max size
        end;%for trajectories
        
        mask = false(size(dx_ipl));
        mask(dx_ipl<par.minmax_DX(1) | dx_ipl>par.minmax_DX(2))=1;
        mask(dy_ipl<par.minmax_DY(1) | dy_ipl>par.minmax_DY(2))=1;
        mask(dz_ipl<par.minmax_DZ(1) | dz_ipl>par.minmax_DZ(2))=1;
    
        dx_ipl(mask) = NaN;dy_ipl(mask) = NaN;dz_ipl(mask) = NaN;


        ell1.x=x_ipl;
        ell1.y=y_ipl;
        ell1.z=z_ipl;
        ell1.dx=dx_ipl;
        ell1.dy=dy_ipl;
        ell1.dz=dz_ipl;
        ell1.id=id_ipl;
        ell1.fr=fr_ipl;

        index=find(isnan(ell1.dx));
        ell1.x(index)=nan;
        ell1.y(index)=nan;
        ell1.z(index)=nan;
        ell1.dx(index)=nan;

        ell1.dy(index)=nan;
        ell1.dz(index)=nan;
        ell1.id(index)=nan;
        ell1.fr(index)=nan;
end
