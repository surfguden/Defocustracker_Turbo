#include "resident.hpp"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <thread>
namespace {
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

struct ResidentContext {defocus::Model model;std::unique_ptr<defocus::ResidentEngine> engine;std::unique_ptr<defocus::RefineResult> last;bool pending=false;};
std::unordered_map<uint64_t,std::unique_ptr<ResidentContext>> contexts;uint64_t next_id=1;
void clear_contexts(){contexts.clear();}
ResidentContext& get(const mxArray* a){
    if(!mxIsUint64(a)||mxGetNumberOfElements(a)!=1)throw std::invalid_argument("Invalid resident handle");
    auto it=contexts.find(*static_cast<const uint64_t*>(mxGetData(a)));if(it==contexts.end())throw std::invalid_argument("Unknown resident handle");return *it->second;
}
mxArray* diagnostics(const defocus::RefineResult& r){
    const char* names[]={"particles","correlations"};auto* result=mxCreateStructMatrix(1,1,2,names);
    size_t n=r.particles.size();auto* a=mxCreateDoubleMatrix(n,11,mxREAL);auto* out=mxGetPr(a);
    for(size_t i=0;i<n;++i){const auto& p=r.particles[i];double row[]={double(p.source_index),double(p.initial_depth),double(p.final_depth),double(p.rejection),double(p.full_calls),double(p.legacy_calls),double(p.overlap_restarted),p.x,p.y,p.z,p.score};for(int k=0;k<11;++k)out[size_t(k)*n+i]=row[k];}
    mxSetField(result,0,"particles",a);n=r.correlations.size();a=mxCreateDoubleMatrix(n,12,mxREAL);out=mxGetPr(a);
    for(size_t i=0;i<n;++i){const auto& c=r.correlations[i];const auto& p=c.peak;double row[]={double(c.source_index),double(c.template_index),double(c.mode),p.score,double(p.x),double(p.y),double(p.ties),p.left,p.right,p.above,p.below,double(p.has_neighbors)};for(int k=0;k<12;++k)out[size_t(k)*n+i]=row[k];}
    mxSetField(result,0,"correlations",a);return result;
}
void emit_result(const ResidentContext& c,int nlhs,mxArray* plhs[],std::chrono::steady_clock::time_point started,bool asynchronous){
    const auto& r=*c.last;size_t n=r.localizations.size();plhs[0]=mxCreateDoubleMatrix(n,4,mxREAL);auto* out=mxGetPr(plhs[0]);
    for(size_t i=0;i<n;++i){const auto& p=r.localizations[i];out[i]=p.x;out[n+i]=p.y;out[2*n+i]=p.z;out[3*n+i]=p.score;}
    if(nlhs==2){
        const char* names[]={"native_s","mex_s","detections","particles","frame_result_waits","particle_layers","clamped_xy","timing_scope","submit_to_complete_s"};
        auto* stats=mxCreateStructMatrix(1,1,asynchronous?9:7,names);
        double vals[]={r.seconds,std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count(),double(n),double(r.particles.size()),double(r.metrics.synchronizations?r.metrics.synchronizations:1),double(r.metrics.batch_submissions),double(c.engine->clamped_xy())};
        for(int k=0;k<7;++k)mxSetFieldByNumber(stats,0,k,mxCreateDoubleScalar(vals[k]));
        if(asynchronous){mxSetField(stats,0,"timing_scope",mxCreateString("host submission-to-completion latency; not GPU active time"));mxSetField(stats,0,"submit_to_complete_s",mxCreateDoubleScalar(r.seconds));}
        plhs[1]=stats;
    }
}
}
extern "C" void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]){try{
    if(nrhs<1)throw std::invalid_argument("resident_mex command required");auto cmd=text(prhs[0]);
    if(cmd=="init"){
        if((nrhs!=4&&nrhs!=5&&nrhs!=6&&nrhs!=7&&nrhs!=8)||nlhs!=1)throw std::invalid_argument("init(model,imageSize,indices)");
        auto c=std::make_unique<ResidentContext>();auto* threads=mxCreateDoubleScalar(1);c->model=read_model(prhs[1],prhs[2],prhs[3],threads);mxDestroyArray(threads);
        if(mxInitGPU()!=MX_GPU_SUCCESS)throw defocus::CudaFailure("MATLAB GPU initialization failed");c->engine=defocus::make_resident(c->model,nrhs>=5?int(scalar(prhs[4])):1,nrhs>=6&&scalar(prhs[5])!=0,nrhs>=7&&scalar(prhs[6])!=0,nrhs==8&&scalar(prhs[7])!=0);
        auto id=next_id++;if(contexts.empty()){mexAtExit(clear_contexts);mexLock();}contexts.emplace(id,std::move(c));
        plhs[0]=mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);*static_cast<uint64_t*>(mxGetData(plhs[0]))=id;return;
    }
    if(nrhs<2)throw std::invalid_argument("Resident handle required");auto& c=get(prhs[1]);
    if(cmd=="release"){if(nlhs||nrhs!=2)throw std::invalid_argument("release(handle)");contexts.erase(*static_cast<const uint64_t*>(mxGetData(prhs[1])));if(contexts.empty())mexUnlock();return;}
    if(cmd=="async_info"){
        if(nrhs!=2||nlhs!=1)throw std::invalid_argument("async_info(handle)");
        const char* names[]={"supported","pending","timing_scope"};plhs[0]=mxCreateStructMatrix(1,1,3,names);
        mxSetField(plhs[0],0,"supported",mxCreateLogicalScalar(c.engine->supports_async()));mxSetField(plhs[0],0,"pending",mxCreateLogicalScalar(c.pending));
        mxSetField(plhs[0],0,"timing_scope",mxCreateString("complete.native_s is host latency since submit, not GPU active time"));return;
    }
    if(cmd=="complete"){
        if(nrhs!=2||nlhs<1||nlhs>2||!c.pending)throw std::invalid_argument("complete(handle) requires one pending frame and returns values[,stats]");
        auto started=std::chrono::steady_clock::now();
        try{c.last=std::make_unique<defocus::RefineResult>(c.engine->complete());c.pending=false;}
        catch(...){c.pending=false;throw;}
        emit_result(c,nlhs,plhs,started,true);return;
    }
    if(c.pending)throw std::invalid_argument("Complete or release the pending resident frame before this command");
    if(cmd=="warmup"){if(nrhs!=3||nlhs!=1)throw std::invalid_argument("warmup(handle,milliseconds) returns elapsed seconds");double ms=scalar(prhs[2]);if(ms<0||ms>1000||ms!=std::floor(ms))throw std::invalid_argument("Warmup duration must be integer0..1000ms");plhs[0]=mxCreateDoubleScalar(c.engine->warmup(int(ms)));return;}
    if(cmd=="image"){if(nrhs!=2||nlhs!=1)throw std::invalid_argument("image(handle) returns one image");if(!c.last)throw std::invalid_argument("No diagnostic image");if(c.last->image.pixels.empty())c.last->image=c.engine->last_image();plhs[0]=image_array(c.last->image);return;}
    if(cmd=="diagnostics"){if(nrhs!=2||nlhs!=1)throw std::invalid_argument("diagnostics(handle) returns one structure");if(!c.last)throw std::invalid_argument("No result");plhs[0]=diagnostics(*c.last);return;}
    bool submitting=cmd=="submit_device";
    if((cmd!="refine"&&cmd!="refine_device"&&!submitting)||nrhs<4||nrhs>5||(submitting?nlhs!=0:(nlhs<1||nlhs>2)))throw std::invalid_argument("refine/refine_device return values[,stats]; submit_device(handle,descriptor,coarse[,diagnostics]) returns no outputs");
    auto* a=prhs[2];const float* device_image=nullptr;size_t device_step=0;
    if(cmd=="refine_device"||submitting){
        auto* data=field(a,"data");auto* pitch=field(a,"step_bytes");
        if(!mxIsUint64(data)||mxGetNumberOfElements(data)!=1||!mxIsUint64(pitch)||mxGetNumberOfElements(pitch)!=1||
            scalar(field(a,"width"))!=c.model.frame_width||scalar(field(a,"height"))!=c.model.frame_height||text(field(a,"layout"))!="row_major_float32")throw std::invalid_argument("Invalid borrowed GPU image descriptor");
        device_image=reinterpret_cast<const float*>(*static_cast<const uint64_t*>(mxGetData(data)));device_step=size_t(*static_cast<const uint64_t*>(mxGetData(pitch)));
        cudaPointerAttributes attributes{};auto status=cudaPointerGetAttributes(&attributes,device_image);int device=-1;cudaGetDevice(&device);
        if(status!=cudaSuccess||attributes.type!=cudaMemoryTypeDevice||attributes.device!=device||scalar(field(a,"device"))!=device||device_step<size_t(c.model.frame_width)*sizeof(float))throw defocus::CudaFailure("Borrowed GPU image is not usable on the active device");
    }else if(!mxIsDouble(a)||mxIsComplex(a)||mxIsSparse(a)||mxGetM(a)!=c.model.frame_height||mxGetN(a)!=c.model.frame_width)throw std::invalid_argument("Expected full-size preprocessed double image");
    auto* seeds=prhs[3];if(!mxIsDouble(seeds)||mxIsComplex(seeds)||mxIsSparse(seeds)||mxGetN(seeds)<4||mxGetN(seeds)>5)throw std::invalid_argument("Invalid coarse candidates");
    size_t n=mxGetM(seeds);auto* values=mxGetPr(seeds);std::vector<defocus::CoarseCandidate> coarse;coarse.reserve(n);
    for(size_t i=0;i<n;++i){double x=values[i],y=values[n+i],z=values[2*n+i],score=values[3*n+i];if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z)||z!=std::floor(z)||z<1||z>c.model.templates.size()||!std::isfinite(score))throw std::invalid_argument("Invalid seed values");coarse.push_back({x,y,int(z),score,mxGetN(seeds)==5?values[4*n+i]:0});}
    bool diagnostic=nrhs==5&&scalar(prhs[4])!=0;auto started=std::chrono::steady_clock::now();
    if(submitting){
        if(!c.engine->supports_async())throw std::invalid_argument("Async resident submission requires at least two workers");
        c.last.reset();c.engine->submit_device(device_image,device_step,coarse,diagnostic);c.pending=true;return;
    }
    c.last=std::make_unique<defocus::RefineResult>(device_image?c.engine->refine_device(device_image,device_step,coarse,diagnostic):c.engine->refine(mxGetPr(a),coarse,diagnostic));
    emit_result(c,nlhs,plhs,started,false);
}catch(const defocus::CudaFailure& e){mexErrMsgIdAndTxt("DefocusResident:CUDA","%s",e.what());}catch(const std::exception& e){mexErrMsgIdAndTxt("DefocusResident:InvalidInput","%s",e.what());}}
