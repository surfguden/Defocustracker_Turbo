function result=normxcorr2_mex(template,image,shape)
% NORMXCORR2_MEX
% RESULT = normxcorr2_mex(TEMPLATE, IMAGE, SHAPE)
%
%       TEMPLATE - type double, size <= size of image
%       IMAGE    - type double
%       SHAPE    - one of: 'valid', 'same', 'full'. same as conv2 shape parameter
%
%       RESULT   - type double, values in [-1,1]. size depends on SHAPE
%
% the syntax of this function is identical to Matlab's
% normxcorr2, except for the output size. the formula for the output size
% is:
%
%                  size(template) = [tp_H, tp_W]
%                  size(image)    = [im_H, im_W]
%
%   SHAPE='valid'  size(result)   = [im_H-tp_H+1, im_W-tp_W+1] 
%   SHAPE='same'   size(result)   = [im_H im_W] 
%   SHAPE='full'   size(result)   = [im_H+tp_H-1, im_W+tp_W-1] 
%
% note: 
% all choices of the SHAPE parameter yield the same output. by this I mean
% that the 'same' and 'full' cases just zero-pad the result so that they
% are the correct size (for your convenience). this implementation cannot return partial matching
% results near the boundary.
%
% the following are equivalent:
%       result = normxcorr2_mex(template, image, 'full');
% AND
%       result = normxcorr2(template, image);
% except that normxcorr2_mex has 0's in the 'invalid' area along the boundary
%
% SEE ALSO CONV2 (for an explanation of SHAPE), or normxcorr2_demo.m (for a demo)
%
% Major NB I want made clear:
% The core code uses the cvMatchTemplate from the OpenCV. I don't know who
% was responsible for this routine, but it is extremely well coded and I want
% to give them credit. All I've done is write the MEX interface. 
% For more on this, see the readme.
%
% Daniel Eaton, danieljameseaton@gmail.com, 2005

% MATLAB compatibility implementation for source-only installations. A
% separately installed legacy MEX takes precedence over this .m file.
% Preserve the Windows interface's valid-only map with zero padded borders.
if nargin<3,shape='full';end
fullMap=normxcorr2(template,image);
templateSize=size(template);imageSize=size(image);
result=zeros(size(fullMap),'like',fullMap);
validRows=templateSize(1):imageSize(1);validCols=templateSize(2):imageSize(2);
result(validRows,validCols)=fullMap(validRows,validCols);
switch shape
    case 'full'
        return
    case 'valid'
        result=result(validRows,validCols);
    case 'same'
        first=floor(templateSize/2)+1;
        result=result(first(1)+(0:imageSize(1)-1),first(2)+(0:imageSize(2)-1));
    otherwise
        error('DefocusTracker:CorrelationShape','Shape must be full, same or valid.');
end

