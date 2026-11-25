#define mexFunction shared_bridge_mexFunction
#include "mex_bridge.cpp"
#undef mexFunction
#include "cpu_refresh.hpp"
#include "initial_depth.hpp"
#include "cpu_execution.hpp"
#include <opencv2/imgproc.hpp>
#include <numeric>
namespace {
struct FrameState {
    workflow::InitialDepthMatcher depth;std::vector<std::unique_ptr<defocus::CorrelationBackend>> owned;
    std::vector<defocus::CorrelationBackend*> backends;cpu_refresh::Workers workers;
    defocus::Image frame;cv::Mat median;defocus::SearchResult search;std::vector<defocus::CoarseCandidate> coarse;
    FrameState(Context& c,std::vector<defocus::Point> polygon,int count):depth(c.model,std::move(polygon)),workers(count),
        frame(c.model.frame_width,c.model.frame_height),median(c.model.frame_width,c.model.frame_height,CV_16UC1){
        depth.set_worker_pool(count,1);backends.push_back(c.correlation.get());
        for(int i=1;i<count;++i){owned.push_back(cpu_refresh::clone_backend(*c.correlation));backends.push_back(owned.back().get());}
    }
};
std::unordered_map<uint64_t,std::unique_ptr<FrameState>> states;
uint64_t id(const mxArray* value){if(!mxIsUint64(value)||mxGetNumberOfElements(value)!=1)throw std::invalid_argument("Expected uint64 CPU handle");return *static_cast<const uint64_t*>(mxGetData(value));}
void release_states(){states.clear();release_all();}
mxArray* coarse_array(const std::vector<defocus::CoarseCandidate>& values){
    size_t n=values.size();auto* a=mxCreateDoubleMatrix(n,5,mxREAL);auto* p=mxGetPr(a);
    for(size_t i=0;i<n;++i){auto c=values[i];p[i]=c.x;p[n+i]=c.y;p[2*n+i]=c.depth;p[3*n+i]=c.score;p[4*n+i]=c.similarity;}return a;
}
}
extern "C" void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]){try{
    if(nrhs<1)throw std::invalid_argument("CPU command required");auto command=text(prhs[0]);
    if(command=="init"){
        if(nrhs!=7||nlhs!=1)throw std::invalid_argument("init(model,size,indices,threads,workers,polygon)");
        double workers=scalar(prhs[5]);if(workers<1||workers>64||workers!=std::floor(workers))throw std::invalid_argument("Workers must be1..64");
        auto* a=prhs[6];if(!mxIsDouble(a)||mxIsComplex(a)||mxIsSparse(a)||mxGetN(a)!=2||mxGetM(a)<3)throw std::invalid_argument("Invalid suppression polygon");
        std::vector<defocus::Point> polygon;size_t n=mxGetM(a);auto* p=mxGetPr(a);for(size_t i=0;i<n;++i){if(!std::isfinite(p[i])||!std::isfinite(p[n+i]))throw std::invalid_argument("Nonfinite polygon");polygon.push_back({p[i],p[n+i]});}
        shared_bridge_mexFunction(1,plhs,5,prhs);auto key=id(plhs[0]);
        try{states[key]=std::make_unique<FrameState>(context(plhs[0]),std::move(polygon),int(workers));mexAtExit(release_states);}
        catch(...){contexts.erase(key);if(contexts.empty())mexUnlock();throw;}return;
    }
    if(nrhs<2)throw std::invalid_argument("CPU handle required");auto key=id(prhs[1]);auto& c=context(prhs[1]);
    auto found=states.find(key);if(found==states.end())throw std::invalid_argument("Unknown refreshed CPU state");auto& s=*found->second;
    if(command=="release"){if(nrhs!=2||nlhs)throw std::invalid_argument("release(handle)");states.erase(key);shared_bridge_mexFunction(0,plhs,2,prhs);return;}
    if(command=="warmup"){
        if(nrhs!=2||nlhs!=1)throw std::invalid_argument("warmup(handle)");auto w=s.depth.warmup(100);
        defocus::Image synthetic(c.model.templates[0].width+8,c.model.templates[0].height+8);
        for(int y=0;y<synthetic.height;++y)for(int x=0;x<synthetic.width;++x)synthetic.at(x,y)=1+(x*17+y*7+x*y)%65521;
        {defocus::CpuExecution execution(1);s.workers.run(s.workers.size()*4,[&](int worker,int){auto& b=*s.backends[worker];b.begin_crop(synthetic);
            b.peak(synthetic,1,defocus::CorrelationMode::matlab_full);b.peak(synthetic,1,defocus::CorrelationMode::legacy_padded_valid);b.end_crop();});}
        plhs[0]=mxCreateDoubleScalar(w.seconds);return;
    }
    if(command=="localize"){
        if(nrhs<3||nrhs>4||nlhs<1||nlhs>2)throw std::invalid_argument("localize(handle,raw_uint16[,diagnostics])");
        auto* raw=prhs[2];if(!mxIsUint16(raw)||mxIsComplex(raw)||mxIsSparse(raw)||mxGetNumberOfDimensions(raw)!=2||mxGetM(raw)!=c.model.frame_height||mxGetN(raw)!=c.model.frame_width)throw std::invalid_argument("Expected matching uint16 RAM frame");
        bool diagnostics=nrhs==4&&scalar(prhs[3])!=0;auto started=std::chrono::steady_clock::now();auto part=started;
        if(c.last){s.frame=std::move(c.last->image);c.last.reset();}
        if(s.frame.width!=c.model.frame_width||s.frame.height!=c.model.frame_height||s.frame.pixels.empty())s.frame=defocus::Image(c.model.frame_width,c.model.frame_height);
        {defocus::CpuExecution execution(c.model.threads);
            cv::Mat input(c.model.frame_width,c.model.frame_height,CV_16UC1,mxGetData(raw));cv::medianBlur(input,s.median,3);
            auto* pixels=s.median.ptr<uint16_t>();int width=c.model.frame_width,height=c.model.frame_height;
            cv::parallel_for_(cv::Range(0,(height+31)/32),[&](const cv::Range& r){for(int block=r.start;block<r.end;++block)
                for(int bx=0;bx<width;bx+=32)for(int x=bx;x<std::min(bx+32,width);++x)for(int y=block*32;y<std::min((block+1)*32,height);++y)
                    s.frame.pixels[size_t(y)*width+x]=pixels[size_t(x)*height+y];});
        }
        auto elapsed=[&](auto t){return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();};
        double prep=elapsed(part);part=std::chrono::steady_clock::now();s.search=c.candidates->search(s.frame);double candidate=elapsed(part);
        std::vector<workflow::Seed> seeds;seeds.reserve(s.search.candidates.size());for(auto p:s.search.candidates)seeds.push_back({p.x,p.y,p.score});
        part=std::chrono::steady_clock::now();workflow::ImageView view{s.median.ptr<uint16_t>(),c.model.frame_width,c.model.frame_height,workflow::PixelType::u16};
        auto depth=s.depth.run(view,seeds,s.search.similarities);s.coarse=std::move(depth.coarse);double guessing=elapsed(part);
        part=std::chrono::steady_clock::now();c.last=std::make_unique<defocus::RefineResult>(cpu_refresh::refine(c.model,s.backends,&s.workers,std::move(s.frame),s.coarse,diagnostics));double refine=elapsed(part);
        const auto& values=c.last->localizations;size_t n=values.size();plhs[0]=mxCreateDoubleMatrix(n,4,mxREAL);auto* out=mxGetPr(plhs[0]);
        for(size_t i=0;i<n;++i){out[i]=values[i].x;out[n+i]=values[i].y;out[2*n+i]=values[i].z;out[3*n+i]=values[i].score;}
        if(nlhs==2){const char* names[]={"total_s","preprocessing_s","candidate_s","depth_guess_s","refinement_s","backend","fallback","workers","threads","prefix_builds","image_ffts","inverse_ffts","cache_hits","particle_waves"};
            plhs[1]=mxCreateStructMatrix(1,1,14,names);double numbers[]={elapsed(started),prep,candidate,guessing,refine};for(int i=0;i<5;++i)mxSetFieldByNumber(plhs[1],0,i,mxCreateDoubleScalar(numbers[i]));
            mxSetField(plhs[1],0,"backend",mxCreateString("cpu_refreshed"));mxSetField(plhs[1],0,"fallback",mxCreateLogicalScalar(false));
            mxSetField(plhs[1],0,"workers",mxCreateDoubleScalar(s.workers.size()));mxSetField(plhs[1],0,"threads",mxCreateDoubleScalar(c.model.threads));
            const auto& m=c.last->metrics;uint64_t counts[]={m.prefix_builds,m.image_ffts,m.inverse_ffts,m.cache_hits,m.batch_submissions};for(int i=0;i<5;++i)mxSetFieldByNumber(plhs[1],0,9+i,mxCreateDoubleScalar(double(counts[i])));
        }return;
    }
    if(command=="diagnostics"){
        shared_bridge_mexFunction(nlhs,plhs,nrhs,prhs);if(nlhs==1){mxAddField(plhs[0],"coarse");mxSetField(plhs[0],0,"coarse",coarse_array(s.coarse));
            mxAddField(plhs[0],"candidates");size_t n=s.search.candidates.size();auto* a=mxCreateDoubleMatrix(n,4,mxREAL);auto* p=mxGetPr(a);
            for(size_t i=0;i<n;++i){auto v=s.search.candidates[i];p[i]=v.template_index;p[n+i]=v.x;p[2*n+i]=v.y;p[3*n+i]=v.score;}mxSetField(plhs[0],0,"candidates",a);
            size_t k=c.model.selected.size();a=mxCreateDoubleMatrix(n,k,mxREAL);p=mxGetPr(a);for(size_t j=0;j<k;++j)for(size_t i=0;i<n;++i)p[j*n+i]=s.search.similarities[i*k+j];mxAddField(plhs[0],"scores");mxSetField(plhs[0],0,"scores",a);
        }return;
    }
    shared_bridge_mexFunction(nlhs,plhs,nrhs,prhs);
}catch(const std::exception& e){mexErrMsgIdAndTxt("DefocusCpuRefresh:InvalidInput","%s",e.what());}}
