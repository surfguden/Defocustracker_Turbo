#include "candidates_api.hpp"
#include "defocus/tracker.hpp"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>

namespace {
struct Context {candidate_direct::Config config;std::unique_ptr<candidate_direct::Search> search;};
std::unordered_map<uint64_t,std::unique_ptr<Context>> contexts;uint64_t next_id=1;
void release_all(){contexts.clear();}
std::string text(const mxArray* value){
    if(!mxIsChar(value))throw std::invalid_argument("Expected character command");
    char* p=mxArrayToString(value);if(!p)throw std::invalid_argument("Invalid command");std::string result(p);mxFree(p);return result;
}
const mxArray* field(const mxArray* value,const char* name){
    if(!mxIsStruct(value))throw std::invalid_argument("Expected model structure");auto* result=mxGetField(value,0,name);
    if(!result)throw std::invalid_argument(std::string("Missing model field ")+name);return result;
}
double number(const mxArray* value,size_t index){
    if(mxIsComplex(value)||mxIsSparse(value))throw std::invalid_argument("Expected dense real numeric input");
    switch(mxGetClassID(value)){
        case mxDOUBLE_CLASS:return static_cast<const double*>(mxGetData(value))[index];
        case mxSINGLE_CLASS:return static_cast<const float*>(mxGetData(value))[index];
        case mxUINT16_CLASS:return static_cast<const uint16_t*>(mxGetData(value))[index];
        case mxUINT8_CLASS:return static_cast<const uint8_t*>(mxGetData(value))[index];
        default:throw std::invalid_argument("Unsupported numeric type");
    }
}
uint64_t id(const mxArray* value){
    if(mxGetClassID(value)!=mxUINT64_CLASS||mxGetNumberOfElements(value)!=1)throw std::invalid_argument("Expected native handle");
    return *static_cast<const uint64_t*>(mxGetData(value));
}
Context& context(const mxArray* value){auto it=contexts.find(id(value));if(it==contexts.end())throw std::invalid_argument("Invalid or released handle");return *it->second;}
}
extern "C" void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]){
    try{
        if(nrhs<1)throw std::invalid_argument("Expected init, run, info or release");auto command=text(prhs[0]);
        if(command=="init"){
            if((nrhs!=4&&nrhs!=5)||nlhs!=1)throw std::invalid_argument("init(model, imageSize, selected_indices[,coalesced_statistics])");
            if(mxInitGPU()!=MX_GPU_SUCCESS)throw defocus::CudaFailure("MATLAB GPU initialization failed");
            auto ctx=std::make_unique<Context>();auto& cfg=ctx->config;
            if(nrhs==5){if(mxGetNumberOfElements(prhs[4])!=1)throw std::invalid_argument("coalesced_statistics must be a scalar logical");double value=mxIsLogical(prhs[4])?mxGetScalar(prhs[4]):number(prhs[4],0);if(value!=0&&value!=1)throw std::invalid_argument("coalesced_statistics must be false or true");cfg.coalesced_statistics=value!=0;}
            if(mxGetNumberOfElements(prhs[2])!=2)throw std::invalid_argument("imageSize must contain height,width");
            double h=number(prhs[2],0),w=number(prhs[2],1);
            if(!std::isfinite(h)||!std::isfinite(w)||h<1||w<1||h!=std::floor(h)||w!=std::floor(w)||h*w>INT32_MAX)
                throw std::invalid_argument("Invalid frame dimensions");cfg.height=int(h);cfg.width=int(w);
            const auto* images=field(field(prhs[1],"parameter"),"images");
            if(!mxIsCell(images)||mxGetNumberOfElements(images)<3)throw std::invalid_argument("Expected calibration image cells");
            cfg.calibration_count=int(mxGetNumberOfElements(images));size_t count=mxGetNumberOfElements(prhs[3]);
            if(count<1||count>65535)throw std::invalid_argument("Invalid selected template count");
            for(size_t k=0;k<count;++k){double index=number(prhs[3],k);if(!std::isfinite(index)||index!=std::floor(index)||index<1||index>cfg.calibration_count)throw std::invalid_argument("Invalid calibration index");cfg.indices.push_back(int(index));}
            auto sorted=cfg.indices;std::sort(sorted.begin(),sorted.end());if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())throw std::invalid_argument("Duplicate calibration indices");
            auto* threshold=field(field(prhs[1],"processing"),"cm_guess");if(mxGetNumberOfElements(threshold)!=1)throw std::invalid_argument("Invalid candidate threshold");
            cfg.threshold=number(threshold,0);if(!std::isfinite(cfg.threshold))throw std::invalid_argument("Invalid candidate threshold");
            std::vector<candidate_direct::Template> templates(count);
            for(size_t k=0;k<count;++k){
                const auto* source=mxGetCell(images,cfg.indices[k]-1);
                if(!source||mxGetNumberOfDimensions(source)!=2||mxIsEmpty(source))throw std::invalid_argument("Invalid calibration image");
                int th=int(mxGetM(source)),tw=int(mxGetN(source));
                if(k==0){cfg.template_width=tw;cfg.template_height=th;}
                if(tw<3||th<3||tw!=cfg.template_width||th!=cfg.template_height||!(tw%2)||!(th%2))throw std::invalid_argument("Calibration dimensions disagree");
                auto& pixels=templates[k].pixels;pixels.resize(size_t(tw)*th);
                for(int y=0;y<th;++y)for(int x=0;x<tw;++x){float p=float(number(source,size_t(x)*th+y));if(!std::isfinite(p))throw std::invalid_argument("Nonfinite calibration pixel");pixels[size_t(y)*tw+x]=p;}
                auto bounds=std::minmax_element(pixels.begin(),pixels.end());if(*bounds.first==*bounds.second)throw std::invalid_argument("Constant calibration template");
            }
            if(cfg.width<cfg.template_width+10||cfg.height<cfg.template_height+10)throw std::invalid_argument("Frame is too small for templates");
            ctx->search=candidate_direct::make(cfg,templates);uint64_t handle=next_id++;
            if(contexts.empty()){mexAtExit(release_all);mexLock();}contexts.emplace(handle,std::move(ctx));
            plhs[0]=mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);*static_cast<uint64_t*>(mxGetData(plhs[0]))=handle;return;
        }
        if(nrhs<2)throw std::invalid_argument("Expected native handle");auto& ctx=context(prhs[1]);
        if(command=="release"){
            if(nrhs!=2||nlhs!=0)throw std::invalid_argument("release(handle)");contexts.erase(id(prhs[1]));if(contexts.empty())mexUnlock();return;
        }
        if(command=="info"){
            if(nrhs!=2||nlhs!=1)throw std::invalid_argument("info(handle)");
            const char* names[]={"backend","input_type","frame_size","template_count","calibration_count","coalesced_statistics","method"};plhs[0]=mxCreateStructMatrix(1,1,7,names);
            mxSetField(plhs[0],0,"backend",mxCreateString("cuda_direct_uint16"));mxSetField(plhs[0],0,"input_type",mxCreateString("MATLAB column-major uint16"));
            auto* shape=mxCreateDoubleMatrix(1,2,mxREAL);mxGetPr(shape)[0]=ctx.config.height;mxGetPr(shape)[1]=ctx.config.width;mxSetField(plhs[0],0,"frame_size",shape);
            mxSetField(plhs[0],0,"template_count",mxCreateDoubleScalar(ctx.config.indices.size()));mxSetField(plhs[0],0,"calibration_count",mxCreateDoubleScalar(ctx.config.calibration_count));
            mxSetField(plhs[0],0,"coalesced_statistics",mxCreateLogicalScalar(ctx.config.coalesced_statistics));
            mxSetField(plhs[0],0,"method",mxCreateString(ctx.config.coalesced_statistics?"C02_uint16_coalesced_statistics":"C01_uint16_tiled_statistics"));return;
        }
        if(command=="device_image"){
            if(nrhs!=2||nlhs!=1)throw std::invalid_argument("device_image(handle) returns a borrowed GPU image descriptor");
            auto image=ctx.search->device_image();
            const char* names[]={"data","step_bytes","width","height","epoch","device","layout"};plhs[0]=mxCreateStructMatrix(1,1,7,names);
            auto u64=[](uint64_t value){auto* out=mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);*static_cast<uint64_t*>(mxGetData(out))=value;return out;};
            mxSetField(plhs[0],0,"data",u64(uint64_t(image.data)));mxSetField(plhs[0],0,"step_bytes",u64(uint64_t(image.step_bytes)));
            mxSetField(plhs[0],0,"width",mxCreateDoubleScalar(image.width));mxSetField(plhs[0],0,"height",mxCreateDoubleScalar(image.height));
            mxSetField(plhs[0],0,"epoch",u64(image.epoch));mxSetField(plhs[0],0,"device",mxCreateDoubleScalar(image.device));
            mxSetField(plhs[0],0,"layout",mxCreateString("row_major_float32"));return;
        }
        if(command=="run"){
            if(nrhs!=3||nlhs<1||nlhs>3)throw std::invalid_argument("run(handle, preprocessed_uint16) returns candidates,scores,stats");
            const auto* image=prhs[2];if(mxGetClassID(image)!=mxUINT16_CLASS||mxIsSparse(image)||mxIsComplex(image)||mxGetNumberOfDimensions(image)!=2||mxGetM(image)!=ctx.config.height||mxGetN(image)!=ctx.config.width)
                throw std::invalid_argument("Expected preprocessed uint16 frame of the initialized dimensions");
            auto result=ctx.search->run(static_cast<const uint16_t*>(mxGetData(image)));size_t n=result.candidates.size(),k=ctx.config.indices.size();
            plhs[0]=mxCreateDoubleMatrix(n,4,mxREAL);double* out=mxGetPr(plhs[0]);
            for(size_t i=0;i<n;++i){const auto& c=result.candidates[i];out[i]=c.template_index;out[n+i]=c.x;out[2*n+i]=c.y;out[3*n+i]=c.score;}
            if(nlhs>=2){plhs[1]=mxCreateDoubleMatrix(n,k,mxREAL);out=mxGetPr(plhs[1]);for(size_t i=0;i<n*k;++i)out[i]=result.similarities[i];}
            if(nlhs==3){
                const char* names[]={"native_s","upload_enqueue_s","engine_s","engine_download_s","sampling_s","uploaded_bytes","coalesced_statistics"};plhs[2]=mxCreateStructMatrix(1,1,7,names);
                double values[]={result.seconds,result.upload_enqueue_s,result.engine_s,result.engine_download_s,result.sampling_s,double(ctx.config.width)*ctx.config.height*sizeof(uint16_t)};
                for(int j=0;j<6;++j)mxSetFieldByNumber(plhs[2],0,j,mxCreateDoubleScalar(values[j]));
                mxSetField(plhs[2],0,"coalesced_statistics",mxCreateLogicalScalar(ctx.config.coalesced_statistics));
            }return;
        }
        throw std::invalid_argument("Unknown direct candidate command");
    }catch(const defocus::CudaFailure& error){mexErrMsgIdAndTxt("DefocusNative:CUDA","%s",error.what());}
    catch(const std::exception& error){mexErrMsgIdAndTxt("DefocusCandidates:InvalidInput","%s",error.what());}
}
