#include "cpu_refresh.hpp"
#include "cpu_execution.hpp"
// Reuse the original math and validation helpers; original sources are not edited.
#define refine_xyz refine_xyz_reference_core
#include "../core/tracker.cpp"
#undef refine_xyz
namespace defocus {
void reused_integer(const Image& im,double x,double y,int w,int h,Image& out){
    int left=int(std::round(x))-1-w/2,top=int(std::round(y))-1-h/2;
    if(left<0||top<0||left+w>im.width||top+h>im.height)throw std::invalid_argument("Particle crop crosses image boundary");
    if(out.width!=w||out.height!=h)out=Image(w,h);
    for(int j=0;j<h;++j)std::copy_n(im.pixels.data()+size_t(top+j)*im.width+left,w,out.pixels.data()+size_t(j)*w);
}
double reused_median(const Image& image,std::vector<double>& work){
    work.assign(image.pixels.begin(),image.pixels.end());size_t mid=work.size()/2;
    std::nth_element(work.begin(),work.begin()+mid,work.end());double v=work[mid];
    if(!(work.size()%2))v=(v+*std::max_element(work.begin(),work.begin()+mid))/2;return v;
}
bool reused_subpixel(const Image& im,double x,double y,int w,int h,Image& result) {
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
    if(result.width!=w||result.height!=h)result=Image(w,h);else std::fill(result.pixels.begin(),result.pixels.end(),0);
    for (int k=0;k<3;++k) {
        int corner=order[k];
        int ox=int(corner>=2 ? std::ceil(left) : std::floor(left))-1;
        int oy=int(corner==1 || corner==3 ? std::ceil(top) : std::floor(top))-1;
        for (int j=0;j<h;++j) for (int i=0;i<w;++i)
            result.at(i,j)+=im.at(ox+i,oy+j)*weights[k];
    }
    return true;
}
RefineResult run_refreshed(const Model& model,std::vector<CorrelationBackend*>& backends,cpu_refresh::Workers* pool,
                          Image&& image,const std::vector<CoarseCandidate>& candidates,bool diagnostics){
    if(backends.empty()||image.width!=model.frame_width||image.height!=model.frame_height)throw std::invalid_argument("Invalid CPU frame context");
    auto started=std::chrono::steady_clock::now();RefineResult result;result.image=std::move(image);
    std::vector<uint8_t> ghost(result.image.pixels.size(),0);
    int tw=model.templates[0].width,th=model.templates[0].height,cw=tw+8,ch=th+8,depths=int(model.templates.size());
    double ax=0,ay=0;for(const auto& area:model.areas)for(auto p:area){ax=std::max(ax,std::abs(p.x));ay=std::max(ay,std::abs(p.y));}
    const double rx=std::max((cw+tw)/2.0+cw/2.0+2,(cw+tw)/2.0+ax+2);
    const double ry=std::max((ch+th)/2.0+ch/2.0+2,(ch+th)/2.0+ay+2);
    std::vector<int> level(candidates.size(),0);int levels=0;
    for(size_t i=0;i<candidates.size();++i){
        for(size_t j=0;j<i;++j)if(std::abs(candidates[i].x-candidates[j].x)<=2*rx&&std::abs(candidates[i].y-candidates[j].y)<=2*ry)
            level[i]=std::max(level[i],level[j]+1);
        levels=std::max(levels,level[i]+1);
    }
    std::vector<std::vector<int>> waves(levels);for(int i=0;i<int(level.size());++i)waves[level[i]].push_back(i);
    std::vector<RefineResult> slots(candidates.size());
    for(auto* backend:backends)backend->reset_metrics();
    auto process=[&](size_t ii,CorrelationBackend& backend){
        const auto& candidate=candidates[ii];
        if (candidate.depth<1 || candidate.depth>depths || !std::isfinite(candidate.x) || !std::isfinite(candidate.y))
            throw std::invalid_argument("Coarse candidate has invalid position or depth");
        ParticleTrace trace; trace.source_index=int(ii)+1; trace.initial_depth=candidate.depth;
        trace.x=trace.y=trace.z=trace.score=nan;
        auto finish=[&](int rejection){trace.rejection=rejection; slots[ii].particles.push_back(trace);};
        auto correlate=[&](const Image& crop,int depth,CorrelationMode mode) {
            if (mode==CorrelationMode::matlab_full) ++trace.full_calls; else ++trace.legacy_calls;
            Peak peak=backend.peak(crop,depth,mode);
            if (diagnostics) slots[ii].correlations.push_back({int(ii)+1,depth,int(mode),peak});
            return peak;
        };
        auto correlate_batch=[&](const Image& crop,const std::vector<int>& depths,CorrelationMode mode) {
            auto peaks=backend.peaks_batch(crop,depths,mode);
            if(peaks.size()!=depths.size())throw std::runtime_error("Correlation batch result size mismatch");
            for(size_t k=0;k<depths.size();++k) {
                if(mode==CorrelationMode::matlab_full)++trace.full_calls;else ++trace.legacy_calls;
                if(diagnostics)slots[ii].correlations.push_back({int(ii)+1,depths[k],int(mode),peaks[k]});
            }
            return peaks;
        };
        double xx=candidate.x,yy=candidate.y; int depth=candidate.depth;
        auto& memory=cpu_refresh::scratch(backend);auto& patch=memory.crop;reused_integer(result.image,xx,yy,cw,ch,patch);
        double background=reused_median(patch,memory.median); mask_crop(patch,model,background);
        PreparedCrop prepared(backend,patch);
        int ox=int(std::round(xx))-1-cw/2,oy=int(std::round(yy))-1-ch/2,ghost_count=0;
        for (int y=0;y<ch;++y) for (int x=0;x<cw;++x) ghost_count+=ghost[size_t(oy+y)*image.width+ox+x];
        if (ghost_count>10 && candidate.score<model.cm_final) {
            trace.overlap_restarted=1;
            std::vector<double> similarities(model.selected.size(),0);
            auto comparisons=correlate_batch(patch,model.selected,CorrelationMode::matlab_full);
            for (size_t k=0;k<model.selected.size();++k) similarities[k]=comparisons[k].score;
            depth=guess_depth(model,similarities);
        }
        Peak initial=correlate(patch,depth,CorrelationMode::matlab_full);
        if (initial.score<model.cm_guess) { finish(1); return; }
        if (!initial.has_neighbors) { finish(2); return; }
        auto dx=gaussian(initial.left,initial.score,initial.right);
        auto dy=gaussian(initial.above,initial.score,initial.below);
        if (dx.imag()!=0 || dy.imag()!=0) { finish(3); return; }
        xx=xx-(cw+tw)/2.0+initial.x+dx.real();
        yy=yy-(ch+th)/2.0+initial.y+dy.real();
        if (!reused_subpixel(result.image,xx,yy,cw,ch,patch)) { finish(4); return; }
        mask_crop(patch,model,background);
        prepared.reset(patch);
        auto walk=backend.depth_walk(patch,depth,depths);
        trace.legacy_calls+=int(walk.samples.size());
        if(diagnostics)for(const auto& sample:walk.samples)
            slots[ii].correlations.push_back({int(ii)+1,sample.template_index,int(CorrelationMode::legacy_padded_valid),sample.peak});
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
            if (!final.has_neighbors) { finish(2); return; }
            dx=gaussian(final.left,final.score,final.right); dy=gaussian(final.above,final.score,final.below);
            // MATLAB applies its NaN fallback before filtering the final candidates.
            // Complex NaN fits must reach that fallback rather than throw on NaN != 0.
            bool undefined=std::isnan(dx.real())||std::isnan(dx.imag())||std::isnan(dy.real())||std::isnan(dy.imag());
            if(undefined){
                xf=xx-(cw+tw)/2.0+final.sum_x/2.0;yf=yy-(ch+th)/2.0+final.sum_y/2.0;
            }else if(dx.imag()!=0||dy.imag()!=0){
                if(final.score<model.cm_final||std::isnan(zz)){
                    trace.z=zz;trace.score=final.score;trace.final_depth=depth;finish(5);return;
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
        if (std::isnan(xf) || std::isnan(yf) || std::isnan(zz) || final.score<model.cm_final) { finish(5); return; }
        if (model.interpolate && (zz==1 || zz==depths)) { finish(6); return; }
        slots[ii].localizations.push_back({xf,yf,model.interpolate ? (zz-1)/(depths-1) : zz,final.score,int(ii)+1});
        finish(0);
    };
    // One level of parallelism: small crop FFTs run with OpenCV threads=1.
    {CpuExecution execution(1);
        for(const auto& wave:waves){
            if(pool)pool->run(int(wave.size()),[&](int worker,int item){process(wave[item],*backends[worker]);});
            else for(int i:wave)process(i,*backends[0]);
        }
    }
    for(auto& slot:slots){
        result.localizations.insert(result.localizations.end(),slot.localizations.begin(),slot.localizations.end());
        result.particles.insert(result.particles.end(),slot.particles.begin(),slot.particles.end());
        result.correlations.insert(result.correlations.end(),slot.correlations.begin(),slot.correlations.end());
    }
    for(auto* backend:backends){auto m=backend->metrics();result.metrics.logical_requests+=m.logical_requests;
        result.metrics.prefix_builds+=m.prefix_builds;result.metrics.image_ffts+=m.image_ffts;result.metrics.inverse_ffts+=m.inverse_ffts;
        result.metrics.template_evaluations+=m.template_evaluations;result.metrics.cache_hits+=m.cache_hits;}
    result.metrics.batch_submissions=levels;result.seconds=seconds(started);return result;
}
RefineResult refine_xyz(const Model& model,CorrelationBackend& backend,const Image& image,const std::vector<CoarseCandidate>& candidates,bool diagnostics){
    model.validate();std::vector<CorrelationBackend*> backends{&backend};return run_refreshed(model,backends,nullptr,Image(image),candidates,diagnostics);
}
}
namespace cpu_refresh {
defocus::RefineResult refine(const defocus::Model& m,std::vector<defocus::CorrelationBackend*>& b,Workers* p,
                            defocus::Image&& i,const std::vector<defocus::CoarseCandidate>& c,bool d){return defocus::run_refreshed(m,b,p,std::move(i),c,d);}
}
