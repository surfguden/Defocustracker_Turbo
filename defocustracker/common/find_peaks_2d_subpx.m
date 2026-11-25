function [xjs, yis, cc] = find_peaks_2d_subpx(im,varargin)

if nargin==2
    thrs = varargin{1};
else
    thrs = (min(im(:))+max(im(:)))/2;
end

ims = im(2:end-1,2:end-1);
left = im(2:end-1,1:end-2)<ims;
right = im(2:end-1,3:end)<ims;
top = im(1:end-2,2:end-1)<ims;
bottom = im(3:end,2:end-1)<ims;
[II, JJ] = size(left);
kk = find(left & right & top & bottom & ims>thrs);
cc = ims(kk);
[ii, jj] = ind2sub([II,JJ],kk); yi = ii+1; xj = jj+1;

xjs = xj*0;
yis = yi*0;

for ii = 1:length(xjs)
    xjs(ii) = xj(ii) + (log(im(yi(ii),xj(ii)-1))-log(im(yi(ii),xj(ii)+1))) /...
        ( 2*log(im(yi(ii),xj(ii)-1)) - 4* log(im(yi(ii),xj(ii))) + 2* log(im(yi(ii),xj(ii)+1)));
    yis(ii) = yi(ii) + (log(im(yi(ii)-1,xj(ii)))-log(im(yi(ii)+1,xj(ii)))) /...
        ( 2*log(im(yi(ii)-1,xj(ii))) - 4* log(im(yi(ii),xj(ii))) + 2* log(im(yi(ii)+1,xj(ii))));
end

