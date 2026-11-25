// Reuse the read-only shared CPU bridge for model validation and init/release.
// The experimental target supplies a distinct exported entry point and binary.
#define mexFunction shared_bridge_mexFunction
#include "mex_bridge.cpp"
#undef mexFunction
#include "initial_depth.hpp"
#include "cpu_execution.hpp"
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <variant>

namespace {
using DepthNumericView=std::variant<const double*,const float*,const uint16_t*,const uint8_t*>;
DepthNumericView depth_numeric_view(const mxArray* value){
    // Validate/dispatch once per matrix, never call mx APIs inside the per-element loop.
    const void* pointer=mxGetData(value);
    switch(mxGetClassID(value)){
        case mxDOUBLE_CLASS:return static_cast<const double*>(pointer);
        case mxSINGLE_CLASS:return static_cast<const float*>(pointer);
        case mxUINT16_CLASS:return static_cast<const uint16_t*>(pointer);
        case mxUINT8_CLASS:return static_cast<const uint8_t*>(pointer);
        default:throw std::invalid_argument("Unsupported candidate/similarity numeric class");
    }
}
std::unordered_map<uint64_t,std::unique_ptr<workflow::InitialDepthMatcher>> depth_matchers;
void workflow_release_all(){depth_matchers.clear();release_all();}
uint64_t handle_id(const mxArray* value){
    if(mxGetClassID(value)!=mxUINT64_CLASS||mxGetNumberOfElements(value)!=1)throw std::invalid_argument("Expected native handle");
    return *static_cast<const uint64_t*>(mxGetData(value));
}
workflow::ImageView image_view(const mxArray* value){
    if(mxIsSparse(value)||mxIsComplex(value)||mxGetNumberOfDimensions(value)!=2||mxIsEmpty(value))throw std::invalid_argument("Expected dense real image");
    workflow::ImageView result;result.data=mxGetData(value);result.width=int(mxGetN(value));result.height=int(mxGetM(value));
    switch(mxGetClassID(value)){
        case mxDOUBLE_CLASS:result.type=workflow::PixelType::f64;break;
        case mxSINGLE_CLASS:result.type=workflow::PixelType::f32;break;
        case mxUINT16_CLASS:result.type=workflow::PixelType::u16;break;
        case mxUINT8_CLASS:result.type=workflow::PixelType::u8;break;
        default:throw std::invalid_argument("Supported image types: double, single, uint16, uint8");
    }
    return result;
}
}
extern "C" void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]){
    try{
        if(nrhs<1)throw std::invalid_argument("Expected workflow command");std::string command=text(prhs[0]);
        if(command=="prepare_depth"){
            if(nrhs!=3||nlhs!=0)throw std::invalid_argument("prepare_depth(handle, polygon_offsets_Nx2)");
            auto& ctx=context(prhs[1]);const auto* polygon=prhs[2];
            if(!mxIsDouble(polygon)||mxIsComplex(polygon)||mxIsSparse(polygon)||mxGetN(polygon)!=2||mxGetM(polygon)<3)
                throw std::invalid_argument("Polygon offsets must be real double Nx2");
            std::vector<defocus::Point> points;size_t n=mxGetM(polygon);const double* p=mxGetPr(polygon);
            for(size_t k=0;k<n;++k){if(!std::isfinite(p[k])||!std::isfinite(p[n+k]))throw std::invalid_argument("Nonfinite polygon");points.push_back({p[k],p[n+k]});}
            depth_matchers[handle_id(prhs[1])]=std::make_unique<workflow::InitialDepthMatcher>(ctx.model,std::move(points));return;
        }
        if(command=="worker_pool"){
            if((nrhs!=3&&nrhs!=4)||nlhs!=0)throw std::invalid_argument("worker_pool(handle,worker_count[,grain]) creates parked CPU workers;0 restores OpenCV");
            auto& ctx=context(prhs[1]);auto found=depth_matchers.find(handle_id(prhs[1]));
            if(found==depth_matchers.end())throw std::invalid_argument("Call prepare_depth before worker_pool");
            double count=scalar(prhs[2]),grain=nrhs==4?scalar(prhs[3]):1;
            if(count!=std::floor(count)||grain!=std::floor(grain))throw std::invalid_argument("Worker count and grain must be integers");
            found->second->set_worker_pool(int(count),int(grain));return;
        }
        if(command=="depth"){
            if(nrhs!=5||nlhs<1||nlhs>2)throw std::invalid_argument("depth(handle, image, candidates_Nx4, scores_NxK)");
            auto& ctx=context(prhs[1]);auto found=depth_matchers.find(handle_id(prhs[1]));
            if(found==depth_matchers.end())throw std::invalid_argument("Call prepare_depth once before processing");
            const auto* candidates=prhs[3];const auto* values=prhs[4];size_t n=mxGetM(candidates),k=ctx.model.selected.size();
            if(mxIsSparse(candidates)||mxIsComplex(candidates)||!mxIsNumeric(candidates)||mxGetN(candidates)!=4||
               mxIsSparse(values)||mxIsComplex(values)||!mxIsNumeric(values)||mxGetM(values)!=n||mxGetN(values)!=k)
                throw std::invalid_argument("Candidate and score dimensions disagree");
            std::vector<workflow::Seed> seeds;seeds.reserve(n);std::vector<double> similarities(n*k);
            std::visit([&](auto candidate_pixels,auto similarity_pixels){
                for(size_t i=0;i<n;++i){
                    double x=double(candidate_pixels[n+i]),y=double(candidate_pixels[2*n+i]),score=double(candidate_pixels[3*n+i]);
                    if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(score))throw std::invalid_argument("Nonfinite candidate");
                    seeds.push_back({x,y,score});
                    for(size_t j=0;j<k;++j){double value=double(similarity_pixels[j*n+i]);if(!std::isfinite(value))throw std::invalid_argument("Nonfinite similarity");similarities[i*k+j]=value;}
                }
            },depth_numeric_view(candidates),depth_numeric_view(values));
            auto result=found->second->run(image_view(prhs[2]),seeds,similarities);n=result.coarse.size();
            plhs[0]=mxCreateDoubleMatrix(n,5,mxREAL);double* out=mxGetPr(plhs[0]);
            for(size_t i=0;i<n;++i){const auto& c=result.coarse[i];out[i]=c.x;out[n+i]=c.y;out[2*n+i]=c.depth;out[3*n+i]=c.score;out[4*n+i]=c.similarity;}
            if(nlhs==2){
                const char* names[]={"native_s","nms_s","matching_s","extra_correlations","source_indices","threads","worker_mode"};
                auto* stats=mxCreateStructMatrix(1,1,7,names);plhs[1]=stats;
                mxSetField(stats,0,"native_s",mxCreateDoubleScalar(result.seconds));mxSetField(stats,0,"nms_s",mxCreateDoubleScalar(result.nms_seconds));
                mxSetField(stats,0,"matching_s",mxCreateDoubleScalar(result.matching_seconds));
                mxSetField(stats,0,"extra_correlations",mxCreateDoubleScalar(std::accumulate(result.correlation_counts.begin(),result.correlation_counts.end(),0)));
                mxSetField(stats,0,"threads",mxCreateDoubleScalar(found->second->worker_threads()));mxSetField(stats,0,"worker_mode",mxCreateString(found->second->worker_mode()));auto* sources=mxCreateDoubleMatrix(n,1,mxREAL);
                for(size_t i=0;i<n;++i)mxGetPr(sources)[i]=result.source_indices[i];mxSetField(stats,0,"source_indices",sources);
            }
            return;
        }
        if(command=="warmup"){
            if((nrhs!=2&&nrhs!=3)||nlhs!=1)throw std::invalid_argument("warmup(handle[,duration_ms]) returns synthetic CPU initialization statistics");
            auto& ctx=context(prhs[1]);auto found=depth_matchers.find(handle_id(prhs[1]));
            if(found==depth_matchers.end())throw std::invalid_argument("Call prepare_depth before warmup");
            double duration=nrhs==3?scalar(prhs[2]):100;
            if(duration!=std::floor(duration)||duration<1||duration>2000)throw std::invalid_argument("Warmup duration must be an integer from1to2000 milliseconds");
            auto result=found->second->warmup(int(duration));
            const char* names[]={"seconds","worker_count","max_active","threads","wave_seconds","checksum","real_frames_processed","worker_mode"};
            plhs[0]=mxCreateStructMatrix(1,1,8,names);
            mxSetField(plhs[0],0,"seconds",mxCreateDoubleScalar(result.seconds));mxSetField(plhs[0],0,"worker_count",mxCreateDoubleScalar(result.worker_count));
            mxSetField(plhs[0],0,"max_active",mxCreateDoubleScalar(result.max_active));mxSetField(plhs[0],0,"threads",mxCreateDoubleScalar(found->second->worker_threads()));mxSetField(plhs[0],0,"worker_mode",mxCreateString(found->second->worker_mode()));
            auto* waves=mxCreateDoubleMatrix(result.wave_seconds.size(),1,mxREAL);std::copy(result.wave_seconds.begin(),result.wave_seconds.end(),mxGetPr(waves));
            mxSetField(plhs[0],0,"wave_seconds",waves);mxSetField(plhs[0],0,"checksum",mxCreateDoubleScalar(result.checksum));
            mxSetField(plhs[0],0,"real_frames_processed",mxCreateDoubleScalar(0));return;
        }
        if(command=="synthetic_check"){
            if(nrhs!=2||nlhs!=1)throw std::invalid_argument("synthetic_check(handle) returns peak-only versus shared CPU checks");
            auto& ctx=context(prhs[1]);auto found=depth_matchers.find(handle_id(prhs[1]));
            if(found==depth_matchers.end())throw std::invalid_argument("Call prepare_depth before synthetic_check");
            auto result=found->second->synthetic_peak_check(*ctx.correlation);
            const char* names[]={"max_peak_error","cases","real_frames_processed"};plhs[0]=mxCreateStructMatrix(1,1,3,names);
            mxSetField(plhs[0],0,"max_peak_error",mxCreateDoubleScalar(result.first));mxSetField(plhs[0],0,"cases",mxCreateDoubleScalar(result.second));
            mxSetField(plhs[0],0,"real_frames_processed",mxCreateDoubleScalar(0));return;
        }
        if(command=="preprocess"){
            if(nrhs<2||nrhs>3||nlhs<1||nlhs>2)throw std::invalid_argument("preprocess(uint16_image[,threads]) performs median3 only");
            auto view=image_view(prhs[1]);if(view.type!=workflow::PixelType::u16)throw std::invalid_argument("Median3 input must be uint16");
            int threads=nrhs==3?int(scalar(prhs[2])):int(std::max(1u,std::thread::hardware_concurrency()));
            if(threads<1||threads>1024)throw std::invalid_argument("Invalid thread count");
            defocus::CpuExecution execution(threads);auto start=std::chrono::steady_clock::now();
            plhs[0]=mxCreateNumericMatrix(view.height,view.width,mxUINT16_CLASS,mxREAL);
            // MATLAB's column-major array is a contiguous transposed image. A square median commutes with transpose.
            // For radius-one filtering, OpenCV's replicated border equals MATLAB's symmetric border.
            cv::Mat source(view.width,view.height,CV_16UC1,const_cast<void*>(view.data));
            cv::Mat destination(view.width,view.height,CV_16UC1,mxGetData(plhs[0]));cv::medianBlur(source,destination,3);
            if(nlhs==2)plhs[1]=mxCreateDoubleScalar(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());return;
        }
        if(command=="release"&&nrhs==2)depth_matchers.erase(handle_id(prhs[1]));
        shared_bridge_mexFunction(nlhs,plhs,nrhs,prhs);
        if(command=="init")mexAtExit(workflow_release_all);
    }catch(const std::exception& error){mexErrMsgIdAndTxt("DefocusWorkflow:InvalidInput","%s",error.what());}
}
