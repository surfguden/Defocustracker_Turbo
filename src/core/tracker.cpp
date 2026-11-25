#include "defocus/tracker.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>

namespace defocus {
namespace {
struct PreparedCrop {
    CorrelationBackend& backend;
    PreparedCrop(CorrelationBackend& b,const Image& image):backend(b){backend.begin_crop(image);}
    ~PreparedCrop(){backend.end_crop();}
    void reset(const Image& image){backend.end_crop();backend.begin_crop(image);}
};
constexpr double nan=std::numeric_limits<double>::quiet_NaN();
double seconds(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
}
Image crop_integer(const Image& im,double x,double y,int w,int h) {
    int left=int(std::round(x))-1-w/2,top=int(std::round(y))-1-h/2;
    if (left<0 || top<0 || left+w>im.width || top+h>im.height)
        throw std::invalid_argument("Particle crop crosses the image boundary");
    Image result(w,h);
    for (int j=0;j<h;++j) std::copy_n(im.pixels.data()+size_t(top+j)*im.width+left,w,
                                    result.pixels.data()+size_t(j)*w);
    return result;
}
bool crop_subpixel(const Image& im,double x,double y,int w,int h,Image& result) {
    double left=x-(w-1)/2.0,top=y-(h-1)/2.0;
    if (!std::isfinite(left) || !std::isfinite(top) || std::floor(left)<1 || std::floor(top)<1 ||
        std::ceil(left+w-1)>im.width || std::ceil(top+h-1)>im.height) return false;
    double fx=left-std::floor(left),fy=top-std::floor(top);
    std::array<double,4> dx={fx,fx,fx-1,fx-1},dy={fy,fy-1,fy,fy-1};
    std::array<int,4> order={0,1,2,3};
    std::stable_sort(order.begin(),order.end(),[&](int a,int b) {
        return dx[a]*dx[a]+dy[a]*dy[a]<dx[b]*dx[b]+dy[b]*dy[b];
    });
    double b=std::abs(dx[order[0]]),c=std::abs(dy[order[0]]),a=1-b-c;
    std::array<double,3> weights={a,b,c};
    result=Image(w,h);
    for (int k=0;k<3;++k) {
        int corner=order[k];
        int ox=int(corner>=2 ? std::ceil(left) : std::floor(left))-1;
        int oy=int(corner==1 || corner==3 ? std::ceil(top) : std::floor(top))-1;
        for (int j=0;j<h;++j) for (int i=0;i<w;++i)
            result.at(i,j)+=im.at(ox+i,oy+j)*weights[k];
    }
    return true;
}
double median(Image image) {
    size_t middle=image.pixels.size()/2;
    std::nth_element(image.pixels.begin(),image.pixels.begin()+middle,image.pixels.end());
    double value=image.pixels[middle];
    if (!(image.pixels.size()%2)) value=(value+*std::max_element(image.pixels.begin(),image.pixels.begin()+middle))/2;
    return value;
}
void mask_crop(Image& image,const Model& model,double background) {
    for (int y=0;y<image.height;++y) for (int x=0;x<image.width;++x) {
        double mask=(x>=4 && y>=4 && x<image.width-4 && y<image.height-4) ? model.global_mask.at(x-4,y-4) : 0;
        image.at(x,y)=image.at(x,y)*mask+(mask==0 ? background : 0);
    }
}
std::complex<double> gaussian(double a,double b,double c) {
    using Complex=std::complex<double>;
    // MATLAB keeps log(0) and its resulting NaN arithmetic real. A complex
    // division of infinite values can manufacture a NaN imaginary component.
    if (a>=0 && b>=0 && c>=0) {
        double la=std::log(a),lb=std::log(b),lc=std::log(c);
        return Complex((la-lc)/(2*la-4*lb+2*lc),0);
    }
    Complex la=std::log(Complex(a,0)),lb=std::log(Complex(b,0)),lc=std::log(Complex(c,0));
    return (la-lc)/(2.0*la-4.0*lb+2.0*lc);
}
int first_max(const std::array<double,3>& values) {
    return int(std::max_element(values.begin(),values.end())-values.begin());
}
int guess_depth(const Model& model,const std::vector<double>& values) {
    double mean=std::accumulate(values.begin(),values.end(),0.0)/values.size();
    double input_norm=0;
    for (double v:values) input_norm+=(v-mean)*(v-mean);
    double best=-std::numeric_limits<double>::infinity(); int index=1;
    for (int j=0;j<int(model.templates.size());++j) {
        double column_mean=0;
        for (int k:model.selected) column_mean+=model.similarity.at(j,k-1);
        column_mean/=values.size();
        double numerator=0,column_norm=0;
        for (size_t k=0;k<values.size();++k) {
            double v=model.similarity.at(j,model.selected[k]-1)-column_mean;
            numerator+=v*(values[k]-mean); column_norm+=v*v;
        }
        double score=numerator/std::sqrt(column_norm*input_norm);
        if (score>best) { best=score; index=j+1; }
    }
    return index;
}
}

Image::Image(int w,int h,double value):width(w),height(h) {
    if (w<1 || h<1 || size_t(w)>size_t(1)<<28 || size_t(w)*h>size_t(1)<<30)
        throw std::invalid_argument("Invalid image dimensions");
    pixels.assign(size_t(w)*h,value);
}
void Model::validate() const {
    if (templates.size()<3 || frame_width<1 || frame_height<1 || threads<1)
        throw std::invalid_argument("Refinement needs at least three calibration templates and valid image dimensions");
    const auto& first=templates.front();
    if (first.width<3 || first.height<3 || !(first.width%2) || !(first.height%2))
        throw std::invalid_argument("The current refinement port requires odd template dimensions of at least three");
    if (frame_width<first.width+10 || frame_height<first.height+10 || selected.empty())
        throw std::invalid_argument("Invalid image size or empty selected template list");
    if (areas.size()!=templates.size() || similarity.width!=int(templates.size()) ||
        similarity.height!=int(templates.size()) || global_mask.width!=first.width || global_mask.height!=first.height)
        throw std::invalid_argument("Calibration dimensions disagree");
    for (const auto& image:templates) {
        if (image.width!=first.width || image.height!=first.height)
            throw std::invalid_argument("All templates must have the same dimensions");
        auto [lo,hi]=std::minmax_element(image.pixels.begin(),image.pixels.end());
        if (*lo==*hi || !std::all_of(image.pixels.begin(),image.pixels.end(),[](double x){return std::isfinite(x);}))
            throw std::invalid_argument("Calibration templates must be finite and nonconstant");
    }
    for (int k:selected) if (k<1 || k>int(templates.size())) throw std::invalid_argument("Invalid selected template index");
    auto copy=selected; std::sort(copy.begin(),copy.end());
    if (std::adjacent_find(copy.begin(),copy.end())!=copy.end()) throw std::invalid_argument("Duplicate selected template index");
    if (!std::isfinite(cm_guess) || !std::isfinite(cm_final)) throw std::invalid_argument("Thresholds must be finite");
}
Peak summarize(const Image& map) {
    Peak result; result.score=-std::numeric_limits<double>::infinity();
    // MATLAB find visits columns first and resolves equal maxima in that order.
    for (int x=0;x<map.width;++x) for (int y=0;y<map.height;++y) {
        double value=map.at(x,y);
        if (value>result.score) { result.score=value; result.x=x+1; result.y=y+1; result.ties=0; result.sum_x=0; result.sum_y=0; }
        if (value==result.score) { ++result.ties; result.sum_x+=x+1; result.sum_y+=y+1; }
    }
    int x=result.x-1,y=result.y-1;
    result.has_neighbors=x>0 && y>0 && x+1<map.width && y+1<map.height;
    if (result.has_neighbors) {
        result.left=map.at(x-1,y); result.right=map.at(x+1,y);
        result.above=map.at(x,y-1); result.below=map.at(x,y+1);
    }
    return result;
}
DepthWalk CorrelationBackend::depth_walk(const Image& patch,int depth,int depths) {
    if(depths<3 || depth<1 || depth>depths)throw std::invalid_argument("Invalid depth walk bounds");
    std::vector<DepthSample> samples;
    auto correlate=[&](const Image& crop,int index,CorrelationMode mode){
        auto p=peak(crop,index,mode);samples.push_back({index,p});return p;
    };
    auto correlate_batch=[&](const Image& crop,const std::vector<int>& indices,CorrelationMode mode){
        auto group=peaks_batch(crop,indices,mode);
        if(group.size()!=indices.size())throw std::runtime_error("Depth batch result size mismatch");
        for(size_t k=0;k<indices.size();++k)samples.push_back({indices[k],group[k]});
        return group;
    };
    DepthWalk result;
    auto& indices=result.indices;auto& peaks=result.peaks;
    if (depth==1) {
        indices={1,2,3};auto group=correlate_batch(patch,{1,2,3},CorrelationMode::legacy_padded_valid);
        peaks={group[0],group[1],group[2]};
    } else if (depth==depths) {
        indices={depth-2,depth-1,depth};auto group=correlate_batch(patch,{depth,depth-2,depth-1},CorrelationMode::legacy_padded_valid);
        peaks={group[1],group[2],group[0]};
    } else {
        indices={depth-1,depth,depth+1};auto group=correlate_batch(patch,{depth,depth-1,depth+1},CorrelationMode::legacy_padded_valid);
        peaks={group[1],group[0],group[2]};
    }
    auto best_index=[&](){return first_max({peaks[0].score,peaks[1].score,peaks[2].score});};
    int& best=result.best;bool& boundary=result.boundary;best=best_index();
    while (best!=1) {
        if (best==0 && indices[0]>1) {
        indices={indices[0]-1,indices[0],indices[1]};
        peaks={correlate(patch,indices[0],CorrelationMode::legacy_padded_valid),peaks[0],peaks[1]};
        } else if (best==2 && indices[2]<depths) {
        indices={indices[1],indices[2],indices[2]+1};
        peaks={peaks[1],peaks[2],correlate(patch,indices[2],CorrelationMode::legacy_padded_valid)};
        } else { boundary=true; break; }
        best=best_index();
    }
    result.samples=std::move(samples);return result;
}
RefineResult refine_xyz(const Model& model,CorrelationBackend& backend,const Image& image,
                        const std::vector<CoarseCandidate>& candidates,bool diagnostics) {
    model.validate();
    if (image.width!=model.frame_width || image.height!=model.frame_height ||
        !std::all_of(image.pixels.begin(),image.pixels.end(),[](double v){return std::isfinite(v);}))
        throw std::invalid_argument("Refinement image dimensions or pixels are invalid");
    auto started=std::chrono::steady_clock::now();
    backend.reset_metrics();
    RefineResult result; result.image=image;
    std::vector<uint8_t> ghost(image.pixels.size(),0);
    int tw=model.templates[0].width,th=model.templates[0].height,cw=tw+8,ch=th+8;
    int depths=int(model.templates.size());
    for (size_t ii=0;ii<candidates.size();++ii) {
        const auto& candidate=candidates[ii];
        if (candidate.depth<1 || candidate.depth>depths || !std::isfinite(candidate.x) || !std::isfinite(candidate.y))
            throw std::invalid_argument("Coarse candidate has invalid position or depth");
        ParticleTrace trace; trace.source_index=int(ii)+1; trace.initial_depth=candidate.depth;
        trace.x=trace.y=trace.z=trace.score=nan;
        auto finish=[&](int rejection){trace.rejection=rejection; result.particles.push_back(trace);};
        auto correlate=[&](const Image& crop,int depth,CorrelationMode mode) {
            if (mode==CorrelationMode::matlab_full) ++trace.full_calls; else ++trace.legacy_calls;
            Peak peak=backend.peak(crop,depth,mode);
            if (diagnostics) result.correlations.push_back({int(ii)+1,depth,int(mode),peak});
            return peak;
        };
        auto correlate_batch=[&](const Image& crop,const std::vector<int>& depths,CorrelationMode mode) {
            auto peaks=backend.peaks_batch(crop,depths,mode);
            if(peaks.size()!=depths.size())throw std::runtime_error("Correlation batch result size mismatch");
            for(size_t k=0;k<depths.size();++k) {
                if(mode==CorrelationMode::matlab_full)++trace.full_calls;else ++trace.legacy_calls;
                if(diagnostics)result.correlations.push_back({int(ii)+1,depths[k],int(mode),peaks[k]});
            }
            return peaks;
        };
        double xx=candidate.x,yy=candidate.y; int depth=candidate.depth;
        Image patch=crop_integer(result.image,xx,yy,cw,ch);
        double background=median(patch); mask_crop(patch,model,background);
        PreparedCrop prepared(backend,patch);
        int ox=int(std::round(xx))-1-cw/2,oy=int(std::round(yy))-1-ch/2,ghost_count=0;
        for (int y=0;y<ch;++y) for (int x=0;x<cw;++x) ghost_count+=ghost[size_t(oy+y)*image.width+ox+x];
        if (ghost_count>10 && candidate.score<model.cm_final) {
            trace.overlap_restarted=1;
            std::vector<double> similarities(model.selected.size(),0);
            auto comparisons=correlate_batch(patch,model.selected,CorrelationMode::matlab_full);
            // Preserve the original intermediate guesses until the separate TODO is implemented.
            for (size_t k=0;k<model.selected.size();++k) {
                similarities[k]=comparisons[k].score;
                depth=guess_depth(model,similarities);
            }
        }
        Peak initial=correlate(patch,depth,CorrelationMode::matlab_full);
        if (initial.score<model.cm_guess) { finish(1); continue; }
        if (!initial.has_neighbors) { finish(2); continue; }
        auto dx=gaussian(initial.left,initial.score,initial.right);
        auto dy=gaussian(initial.above,initial.score,initial.below);
        if (dx.imag()!=0 || dy.imag()!=0) { finish(3); continue; }
        xx=xx-(cw+tw)/2.0+initial.x+dx.real();
        yy=yy-(ch+th)/2.0+initial.y+dy.real();
        if (!crop_subpixel(result.image,xx,yy,cw,ch,patch)) { finish(4); continue; }
        mask_crop(patch,model,background);
        prepared.reset(patch);
        auto walk=backend.depth_walk(patch,depth,depths);
        trace.legacy_calls+=int(walk.samples.size());
        if(diagnostics)for(const auto& sample:walk.samples)
            result.correlations.push_back({int(ii)+1,sample.template_index,int(CorrelationMode::legacy_padded_valid),sample.peak});
        const auto& indices=walk.indices;const auto& peaks=walk.peaks;
        int best=walk.best;bool boundary=walk.boundary;
        Peak final=peaks[best]; double zz;
        if (!boundary && model.interpolate) {
            depth=indices[1];
            zz=depth+(peaks[0].score-peaks[2].score)/(2*peaks[0].score-4*peaks[1].score+2*peaks[2].score);
            final=peaks[1];
        } else { depth=indices[best]; zz=depth; }
        double xf,yf;
        if (final.ties==1) {
            if (!final.has_neighbors) { finish(2); continue; }
            dx=gaussian(final.left,final.score,final.right); dy=gaussian(final.above,final.score,final.below);
            // MATLAB applies its NaN fallback before filtering the final candidates.
            // Complex NaN fits must reach that fallback rather than throw on NaN != 0.
            bool undefined=std::isnan(dx.real())||std::isnan(dx.imag())||std::isnan(dy.real())||std::isnan(dy.imag());
            if(undefined){
                xf=xx-(cw+tw)/2.0+final.sum_x/2.0;yf=yy-(ch+th)/2.0+final.sum_y/2.0;
            }else if(dx.imag()!=0||dy.imag()!=0){
                if(final.score<model.cm_final||std::isnan(zz)){
                    trace.z=zz;trace.score=final.score;trace.final_depth=depth;finish(5);continue;
                }
                throw std::runtime_error("Non-real accepted final subpixel coordinate at source "+std::to_string(ii+1)+" score "+std::to_string(final.score));
            }else{
                xf=xx-(cw+tw)/2.0+final.x+dx.real();yf=yy-(ch+th)/2.0+final.y+dy.real();
            }
        } else if (final.ties==2) {
            xf=xx-(cw+tw)/2.0+final.sum_x/2.0; yf=yy-(ch+th)/2.0+final.sum_y/2.0;
        } else { xf=xx-(cw+tw)/2.0+final.x; yf=yy-(ch+th)/2.0+final.y; }
        if (std::isfinite(zz) && final.score>model.cm_final) {
            int mask_depth=int(std::round(zz));
            if (mask_depth<1 || mask_depth>depths) throw std::runtime_error("Interpolated depth outside calibration range");
            for (const auto& point:model.areas[mask_depth-1]) {
                int x=int(std::round(xx+point.x))-1,y=int(std::round(yy+point.y))-1;
                if (x<0 || y<0 || x>=image.width || y>=image.height) throw std::runtime_error("Particle suppression crosses image boundary");
                result.image.at(x,y)=background; ghost[size_t(y)*image.width+x]=1;
            }
        }
        trace.x=xf; trace.y=yf; trace.z=zz; trace.score=final.score; trace.final_depth=depth;
        if (std::isnan(xf) || std::isnan(yf) || std::isnan(zz) || final.score<model.cm_final) { finish(5); continue; }
        if (model.interpolate && (zz==1 || zz==depths)) { finish(6); continue; }
        result.localizations.push_back({xf,yf,model.interpolate ? (zz-1)/(depths-1) : zz,final.score,int(ii)+1});
        finish(0);
    }
    result.seconds=seconds(started);
    result.metrics=backend.metrics();
    return result;
}
} // namespace defocus
