#include "defocus/tracker.hpp"
#include "mex.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <thread>
#ifdef DEFOCUS_CUDA_BACKEND
#include "gpu/mxGPUArray.h"
#endif

namespace {
struct Context {
    defocus::Model model;
    std::unique_ptr<defocus::CorrelationBackend> correlation;
    std::unique_ptr<defocus::CandidateBackend> candidates;
    std::unique_ptr<defocus::RefineResult> last;
};
std::unordered_map<uint64_t,std::unique_ptr<Context>> contexts;
uint64_t next_id=1;
void release_all() { contexts.clear(); }
const mxArray* field(const mxArray* a,const char* name) {
    if (!mxIsStruct(a)) throw std::invalid_argument("Expected a model structure");
    auto* value=mxGetField(a,0,name);
    if (!value) throw std::invalid_argument(std::string("Missing model field: ")+name);
    return value;
}
double scalar(const mxArray* a) {
    if (mxGetNumberOfElements(a)!=1 || mxIsComplex(a) || mxIsSparse(a) ||
        (!mxIsNumeric(a) && !mxIsLogical(a))) throw std::invalid_argument("Expected a real scalar");
    double value=mxGetScalar(a);
    if (!std::isfinite(value)) throw std::invalid_argument("Scalar must be finite");
    return value;
}
double numeric(const mxArray* a,size_t i) {
    switch(mxGetClassID(a)) {
        case mxDOUBLE_CLASS:return static_cast<const double*>(mxGetData(a))[i];
        case mxSINGLE_CLASS:return static_cast<const float*>(mxGetData(a))[i];
        case mxUINT16_CLASS:return static_cast<const uint16_t*>(mxGetData(a))[i];
        case mxUINT8_CLASS:return static_cast<const uint8_t*>(mxGetData(a))[i];
        case mxLOGICAL_CLASS:return static_cast<const mxLogical*>(mxGetData(a))[i];
        default:throw std::invalid_argument("Unsupported numeric class");
    }
}
std::string text(const mxArray* a) {
    if (!mxIsChar(a)) throw std::invalid_argument("Command must be a character vector");
    char* p=mxArrayToString(a); if (!p) throw std::invalid_argument("Invalid command");
    std::string value(p); mxFree(p); return value;
}
template<class T> void import_pixels(const T* in,defocus::Image& result) {
    // Tile the MATLAB column-major to native row-major conversion for cache locality.
    for (int by=0;by<result.height;by+=32) for (int bx=0;bx<result.width;bx+=32)
        for (int x=bx;x<std::min(bx+32,result.width);++x)
            for (int y=by;y<std::min(by+32,result.height);++y) {
                double value=in[size_t(x)*result.height+y];
                if (!std::isfinite(value)) throw std::invalid_argument("Image values must be finite");
                result.pixels[size_t(y)*result.width+x]=value;
            }
}
defocus::Image image(const mxArray* a) {
    if (mxIsComplex(a) || mxIsSparse(a) || mxGetNumberOfDimensions(a)!=2 ||
        mxGetM(a)<1 || mxGetN(a)<1) throw std::invalid_argument("Expected a real, dense two-dimensional image");
    defocus::Image result(int(mxGetN(a)),int(mxGetM(a)));
    switch(mxGetClassID(a)) {
        case mxDOUBLE_CLASS:import_pixels(static_cast<const double*>(mxGetData(a)),result);break;
        case mxSINGLE_CLASS:import_pixels(static_cast<const float*>(mxGetData(a)),result);break;
        case mxUINT16_CLASS:import_pixels(static_cast<const uint16_t*>(mxGetData(a)),result);break;
        case mxUINT8_CLASS:import_pixels(static_cast<const uint8_t*>(mxGetData(a)),result);break;
        case mxLOGICAL_CLASS:import_pixels(static_cast<const mxLogical*>(mxGetData(a)),result);break;
        default:throw std::invalid_argument("Unsupported image class");
    }
    return result;
}
mxArray* image_array(const defocus::Image& value) {
    mxArray* result=mxCreateDoubleMatrix(value.height,value.width,mxREAL);
    double* out=mxGetPr(result);
    for (int x=0;x<value.width;++x) for (int y=0;y<value.height;++y) out[size_t(x)*value.height+y]=value.at(x,y);
    return result;
}
std::vector<int> indices(const mxArray* a) {
    if (!mxIsDouble(a) || mxIsComplex(a) || mxIsSparse(a) ||
        (mxGetM(a)!=1 && mxGetN(a)!=1)) throw std::invalid_argument("Selected indices must be a real double vector");
    std::vector<int> result;
    for (size_t i=0;i<mxGetNumberOfElements(a);++i) {
        double value=numeric(a,i);
        if (!std::isfinite(value) || value<1 || value>1000000 || value!=std::floor(value))
            throw std::invalid_argument("Template indices must be positive integers");
        result.push_back(int(value));
    }
    return result;
}
defocus::Model read_model(const mxArray* a,const mxArray* size,const mxArray* selected,const mxArray* threads) {
    defocus::Model result;
    const mxArray* cal=field(a,"parameter"),*par=field(a,"processing");
    const mxArray* templates=field(cal,"images");
    if (!mxIsCell(templates)) throw std::invalid_argument("Calibration images must be a cell array");
    for (size_t k=0;k<mxGetNumberOfElements(templates);++k) {
        const auto* value=mxGetCell(templates,k);
        if (!value) throw std::invalid_argument("Empty calibration image");
        result.templates.push_back(image(value));
    }
    result.global_mask=image(field(cal,"global_area"));
    result.similarity=image(field(cal,"similarity_map"));
    const mxArray* areas=field(cal,"images_area");
    if (!mxIsCell(areas)) throw std::invalid_argument("Calibration areas must be a cell array");
    for (size_t k=0;k<mxGetNumberOfElements(areas);++k) {
        const mxArray* points=mxGetCell(areas,k);
        if (!points || mxIsComplex(points) || mxIsSparse(points) || mxGetM(points)!=2)
            throw std::invalid_argument("Calibration area must be a two-row coordinate array");
        std::vector<defocus::Point> values;
        for (size_t i=0;i<mxGetN(points);++i) {
            double x=numeric(points,2*i),y=numeric(points,2*i+1);
            if (!std::isfinite(x) || !std::isfinite(y)) throw std::invalid_argument("Nonfinite calibration area");
            values.push_back({x,y});
        }
        result.areas.push_back(std::move(values));
    }
    if (!mxIsDouble(size) || mxIsComplex(size) || mxGetNumberOfElements(size)!=2)
        throw std::invalid_argument("Image size must be [height width]");
    double h=numeric(size,0),w=numeric(size,1),nthreads=scalar(threads);
    if (!std::isfinite(w) || !std::isfinite(h) || w<1 || h<1 || w>100000 || h>100000 ||
        w!=std::floor(w) || h!=std::floor(h) || nthreads<0 || nthreads>1024 || nthreads!=std::floor(nthreads))
        throw std::invalid_argument("Invalid frame size or CPU thread count");
    result.frame_width=int(w); result.frame_height=int(h);
    result.threads=nthreads>0 ? int(nthreads) : int(std::max(1u,std::thread::hardware_concurrency()));
    result.selected=indices(selected);
    result.cm_guess=scalar(field(par,"cm_guess")); result.cm_final=scalar(field(par,"cm_final"));
    result.interpolate=scalar(field(par,"subimage_interpolation"))!=0;
    result.validate(); return result;
}
Context& context(const mxArray* a) {
    if (mxGetClassID(a)!=mxUINT64_CLASS || mxGetNumberOfElements(a)!=1) throw std::invalid_argument("Invalid native context handle");
    uint64_t key=*static_cast<const uint64_t*>(mxGetData(a));
    auto found=contexts.find(key);
    if (found==contexts.end()) throw std::invalid_argument("Native context has been released or is invalid");
    return *found->second;
}
mxArray* metadata(const Context& ctx) {
    const char* names[]={"backend","template_count","calibration_count","frame_size","threads","depth_walk","depth_walk_setup_s"};
    mxArray* out=mxCreateStructMatrix(1,1,7,names);
    mxSetField(out,0,"backend",mxCreateString(ctx.correlation->name().c_str()));
    mxSetField(out,0,"template_count",mxCreateDoubleScalar(double(ctx.model.selected.size())));
    mxSetField(out,0,"calibration_count",mxCreateDoubleScalar(double(ctx.model.templates.size())));
    mxArray* size=mxCreateDoubleMatrix(1,2,mxREAL);
    mxGetPr(size)[0]=ctx.model.frame_height; mxGetPr(size)[1]=ctx.model.frame_width;
    mxSetField(out,0,"frame_size",size); mxSetField(out,0,"threads",mxCreateDoubleScalar(ctx.model.threads));
    mxSetField(out,0,"depth_walk",mxCreateString(ctx.correlation->depth_walk_implementation().c_str()));
    mxSetField(out,0,"depth_walk_setup_s",mxCreateDoubleScalar(ctx.correlation->depth_walk_setup_seconds()));
    return out;
}
}

extern "C" void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) {
    try {
        if (nrhs<1) throw std::invalid_argument("Use init, candidates, refine, correlate, diagnostics, image, info, or release");
        std::string command=text(prhs[0]);
        if (command=="init") {
            if (nrhs!=5 || nlhs!=1) throw std::invalid_argument("init(model,[height width],selected_indices,threads) returns a handle");
            auto ctx=std::make_unique<Context>();
            ctx->model=read_model(prhs[1],prhs[2],prhs[3],prhs[4]);
#ifdef DEFOCUS_CUDA_BACKEND
            if (mxInitGPU()!=MX_GPU_SUCCESS) throw defocus::CudaFailure("MATLAB GPU initialization failed");
            ctx->correlation=defocus::make_cuda_correlation(ctx->model);
            ctx->candidates=defocus::make_cuda_candidates(ctx->model);
#else
            ctx->correlation=defocus::make_cpu_correlation(ctx->model);
            ctx->candidates=defocus::make_cpu_candidates(ctx->model);
#endif
            uint64_t id=next_id++;
            if (contexts.empty()) {mexAtExit(release_all); mexLock();}
            contexts.emplace(id,std::move(ctx));
            plhs[0]=mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);
            *static_cast<uint64_t*>(mxGetData(plhs[0]))=id; return;
        }
        if (nrhs<2) throw std::invalid_argument("A native context handle is required");
        auto& ctx=context(prhs[1]);
        if (command=="release") {
            if (nrhs!=2 || nlhs!=0) throw std::invalid_argument("release expects a handle and no outputs");
            uint64_t id=*static_cast<const uint64_t*>(mxGetData(prhs[1])); contexts.erase(id);
            if (contexts.empty()) mexUnlock(); return;
        }
        if (command=="info") {
            if (nrhs!=2 || nlhs!=1) throw std::invalid_argument("info returns one metadata structure");
            plhs[0]=metadata(ctx); return;
        }
        if(command=="depth_strategy") {
            if(nrhs!=3||nlhs!=0)throw std::invalid_argument("depth_strategy(handle, 'adaptive' or 'full_stack')");
            auto strategy=text(prhs[2]);
            if(strategy!="adaptive"&&strategy!="full_stack")throw std::invalid_argument("Unknown depth strategy");
            ctx.correlation->set_depth_strategy(strategy=="adaptive"?defocus::DepthStrategy::adaptive:defocus::DepthStrategy::full_stack);
            return;
        }
#ifdef DEFOCUS_TEST_HOOKS
        if (command=="test_fail_after") {
            if (nrhs!=3 || nlhs!=0) throw std::invalid_argument("Failure injection takes a context and a nonnegative call count");
            double count=scalar(prhs[2]);
            if (count<0 || count>1000000 || count!=std::floor(count)) throw std::invalid_argument("Invalid injected failure count");
            ctx.correlation->test_failure_after(int(count)); return;
        }
#endif
        if (command=="correlate") {
            if (nrhs!=5 || nlhs!=1) throw std::invalid_argument("correlate(handle,crop,template_index,legacy_mode) returns a map");
            double index=scalar(prhs[3]);
            if (index!=std::floor(index)) throw std::invalid_argument("Template index must be integral");
            plhs[0]=image_array(ctx.correlation->correlate(image(prhs[2]),int(index),
                scalar(prhs[4])!=0 ? defocus::CorrelationMode::legacy_padded_valid : defocus::CorrelationMode::matlab_full));
            return;
        }
        if (command=="candidates") {
            if (nrhs!=3 || nlhs<1 || nlhs>3) throw std::invalid_argument("candidates(handle,image) returns candidates, scores, and seconds");
            auto input=image(prhs[2]);
            if (input.width!=ctx.model.frame_width || input.height!=ctx.model.frame_height) throw std::invalid_argument("Wrong frame dimensions");
            auto result=ctx.candidates->search(input); size_t n=result.candidates.size(),k=ctx.model.selected.size();
            plhs[0]=mxCreateDoubleMatrix(n,4,mxREAL); double* out=mxGetPr(plhs[0]);
            for (size_t i=0;i<n;++i) {
                const auto& c=result.candidates[i]; out[i]=c.template_index; out[n+i]=c.x; out[2*n+i]=c.y; out[3*n+i]=c.score;
            }
            if (nlhs>=2) {
                plhs[1]=mxCreateDoubleMatrix(n,k,mxREAL); out=mxGetPr(plhs[1]);
                for (size_t j=0;j<k;++j) for (size_t i=0;i<n;++i) out[j*n+i]=result.similarities[i*k+j];
            }
            if (nlhs==3) plhs[2]=mxCreateDoubleScalar(result.seconds); return;
        }
        if (command=="refine") {
            if ((nrhs!=4 && nrhs!=5) || nlhs<1 || nlhs>2) throw std::invalid_argument("refine(handle,image,coarse[,diagnostics]) returns XYZ/score and metadata");
            const auto* a=prhs[3];
            if (!mxIsDouble(a) || mxIsComplex(a) || mxIsSparse(a) || mxGetN(a)<4 || mxGetN(a)>5)
                throw std::invalid_argument("Coarse candidates must have X, Y, depth index, score, and optional similarity columns");
            std::vector<defocus::CoarseCandidate> coarse; size_t n=mxGetM(a); const double* values=mxGetPr(a);
            for (size_t i=0;i<n;++i) {
                double z=values[2*n+i];
                if (!std::isfinite(z) || z!=std::floor(z) || z<1 || z>ctx.model.templates.size() || !std::isfinite(values[3*n+i]))
                    throw std::invalid_argument("Invalid coarse depth or correlation score");
                coarse.push_back({values[i],values[n+i],int(z),values[3*n+i],mxGetN(a)==5 ? values[4*n+i] : 0});
            }
            auto input=image(prhs[2]);
            ctx.last=std::make_unique<defocus::RefineResult>(defocus::refine_xyz(ctx.model,*ctx.correlation,input,coarse,nrhs==5 && scalar(prhs[4])!=0));
            const auto& result=*ctx.last; n=result.localizations.size();
            plhs[0]=mxCreateDoubleMatrix(n,4,mxREAL); double* out=mxGetPr(plhs[0]);
            for (size_t i=0;i<n;++i) {
                const auto& p=result.localizations[i]; out[i]=p.x; out[n+i]=p.y; out[2*n+i]=p.z; out[3*n+i]=p.score;
            }
            if (nlhs==2) {
                const char* names[]={"backend","seconds","particles","full_correlations","legacy_correlations","overlap_restarts","metrics"};
                plhs[1]=mxCreateStructMatrix(1,1,7,names);
                mxSetField(plhs[1],0,"backend",mxCreateString(ctx.correlation->name().c_str()));
                mxSetField(plhs[1],0,"seconds",mxCreateDoubleScalar(result.seconds));
                mxSetField(plhs[1],0,"particles",mxCreateDoubleScalar(double(n)));
                double full=0,legacy=0,overlap=0;
                for (const auto& p:result.particles) {full+=p.full_calls; legacy+=p.legacy_calls; overlap+=p.overlap_restarted;}
                mxSetField(plhs[1],0,"full_correlations",mxCreateDoubleScalar(full));
                mxSetField(plhs[1],0,"legacy_correlations",mxCreateDoubleScalar(legacy));
                mxSetField(plhs[1],0,"overlap_restarts",mxCreateDoubleScalar(overlap));
                const char* metric_names[]={"logical_requests","crop_uploads","prefix_builds","image_ffts","inverse_ffts",
                    "template_evaluations","result_downloads","synchronizations","cache_hits","batch_submissions","gpu_depth_walks","host_depth_walks","full_stack_walks","depth_walk_s"};
                mxArray* metrics=mxCreateStructMatrix(1,1,14,metric_names);
                const auto& m=result.metrics;
                uint64_t values[]={m.logical_requests,m.crop_uploads,m.prefix_builds,m.image_ffts,m.inverse_ffts,
                    m.template_evaluations,m.result_downloads,m.synchronizations,m.cache_hits,m.batch_submissions,m.gpu_depth_walks,m.host_depth_walks,m.full_stack_walks};
                for(int j=0;j<13;++j)mxSetFieldByNumber(metrics,0,j,mxCreateDoubleScalar(double(values[j])));
                mxSetField(metrics,0,"depth_walk_s",mxCreateDoubleScalar(m.depth_walk_s));
                mxSetField(plhs[1],0,"metrics",metrics);
            }
            return;
        }
        if (command=="image") {
            if (nrhs!=2 || nlhs!=1 || !ctx.last) throw std::invalid_argument("No refined frame is available");
            plhs[0]=image_array(ctx.last->image); return;
        }
        if (command=="diagnostics") {
            if (nrhs!=2 || nlhs!=1 || !ctx.last) throw std::invalid_argument("No refinement diagnostics are available");
            const char* names[]={"particles","correlations"}; plhs[0]=mxCreateStructMatrix(1,1,2,names);
            size_t n=ctx.last->particles.size(); mxArray* a=mxCreateDoubleMatrix(n,11,mxREAL); double* out=mxGetPr(a);
            for (size_t i=0;i<n;++i) {
                const auto& p=ctx.last->particles[i];
                double row[]={double(p.source_index),double(p.initial_depth),double(p.final_depth),double(p.rejection),
                    double(p.full_calls),double(p.legacy_calls),double(p.overlap_restarted),p.x,p.y,p.z,p.score};
                for (size_t j=0;j<11;++j) out[j*n+i]=row[j];
            }
            mxSetField(plhs[0],0,"particles",a);
            n=ctx.last->correlations.size(); a=mxCreateDoubleMatrix(n,12,mxREAL); out=mxGetPr(a);
            for (size_t i=0;i<n;++i) {
                const auto& c=ctx.last->correlations[i]; const auto& p=c.peak;
                double row[]={double(c.source_index),double(c.template_index),double(c.mode),p.score,double(p.x),double(p.y),
                    double(p.ties),p.left,p.right,p.above,p.below,double(p.has_neighbors)};
                for (size_t j=0;j<12;++j) out[j*n+i]=row[j];
            }
            mxSetField(plhs[0],0,"correlations",a); return;
        }
        throw std::invalid_argument("Unknown tracker command");
    } catch (const defocus::CudaFailure& error) {
        mexErrMsgIdAndTxt("DefocusNative:CUDA","%s",error.what());
    } catch (const std::exception& error) {
        mexErrMsgIdAndTxt("DefocusNative:InvalidInput","%s",error.what());
    }
}
