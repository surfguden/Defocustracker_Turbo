#include "candidates_api.hpp"
#include "candidate_engine.cuh"
#include <opencv2/cudaarithm.hpp>
#include <cstring>

namespace candidate_direct { namespace {
namespace cd=defocus::cuda_detail;
template<class T> struct Buffer {
    T* data=nullptr;size_t capacity=0;
    ~Buffer(){cudaFree(data);}
    void reserve(size_t n){if(n<=capacity)return;if(data)cd::check_cuda(cudaFree(data));data=nullptr;capacity=0;cd::check_cuda(cudaMalloc(reinterpret_cast<void**>(&data),n*sizeof(T)));capacity=n;}
};
struct MapView {const float* data;size_t step;};
// Source is a contiguous row-major H-column by W-row view of the MATLAB array.
// Shared memory transposes 32x32 tiles, with coalesced reads and coalesced float writes.
__global__ void upload_layout(const uint16_t* source,float* output,size_t output_step,int width,int height){
    __shared__ uint16_t tile[32][33];
    int sy=blockIdx.x*32+threadIdx.x,sx=blockIdx.y*32+threadIdx.y;
    for(int j=0;j<32;j+=8)if(sy<height&&sx+j<width)tile[threadIdx.y+j][threadIdx.x]=source[size_t(sx+j)*height+sy];
    __syncthreads();
    int dx=blockIdx.y*32+threadIdx.x,dy=blockIdx.x*32+threadIdx.y;
    for(int j=0;j<32;j+=8)if(dx<width&&dy+j<height){
        float* row=reinterpret_cast<float*>(reinterpret_cast<char*>(output)+size_t(dy+j)*output_step);
        row[dx]=float(tile[threadIdx.x][threadIdx.y+j]);
    }
}
__global__ void sample_candidates(const cd::Candidate* candidates,int count,const MapView* maps,
                                   float* output,int offset_x,int offset_y,int templates){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count*templates)return;
    int k=i/count;auto c=candidates[i%count];
    const float* row=reinterpret_cast<const float*>(reinterpret_cast<const char*>(maps[k].data)+size_t(c.y-offset_y)*maps[k].step);
    output[i]=row[c.x-offset_x];
}
class DirectSearch final:public Search {
    Config config_;
    cd::Config cfg_{};
    cv::cuda::Stream stream_;
    cv::cuda::GpuMat image_;
    Buffer<uint16_t> raw_;
    Buffer<MapView> views_;
    Buffer<cd::Candidate> candidates_;
    Buffer<float> values_;
    std::unique_ptr<cd::CachedGpuFftSearch> engine_;
    int device_=0;uint64_t epoch_=0;
public:
    DirectSearch(const Config& config,const std::vector<Template>& input):config_(config){
        cd::check_cuda(cudaGetDevice(&device_));
        cfg_.width=config.width;cfg_.height=config.height;cfg_.template_width=config.template_width;cfg_.template_height=config.template_height;
        cfg_.count=int(config.indices.size());cfg_.threshold=config.threshold;cfg_.indices=config.indices;
        cfg_.min_x=int(std::ceil(2+(cfg_.template_width+8)/2.0));cfg_.max_x=int(std::floor(cfg_.width-(cfg_.template_width+8)/2.0-1));
        cfg_.min_y=int(std::ceil(2+(cfg_.template_height+8)/2.0));cfg_.max_y=int(std::floor(cfg_.height-(cfg_.template_height+8)/2.0-1));
        std::vector<cd::GpuTemplate> templates(cfg_.count);
        for(int k=0;k<cfg_.count;++k){
            cv::Mat original(cfg_.template_height,cfg_.template_width,CV_32FC1,const_cast<float*>(input[k].pixels.data()));
            cv::Mat centered(cfg_.template_height,cfg_.template_width,CV_32FC1);double mean=cv::mean(original)[0];
            for(int y=0;y<original.rows;++y)for(int x=0;x<original.cols;++x)centered.at<float>(y,x)=float(original.at<float>(y,x)-mean);
            cv::Scalar residual,stddev;cv::meanStdDev(centered,residual,stddev);
            templates[k].residual_mean=residual[0];templates[k].norm=stddev[0]*std::sqrt(double(centered.total()));
            templates[k].pixels.upload(centered,stream_);
        }
        image_.create(cfg_.height,cfg_.width,CV_32FC1);raw_.reserve(size_t(cfg_.width)*cfg_.height);
        views_.reserve(cfg_.count);candidates_.reserve(65536);values_.reserve(size_t(65536)*cfg_.count);
        stream_.waitForCompletion();
        engine_=std::make_unique<cd::CachedGpuFftSearch>(cfg_,templates,stream_,true,true,true,config_.coalesced_statistics);
    }
    Result run(const uint16_t* input)override{
        try{
            auto started=cd::Clock::now();Result result;auto native_stream=cv::cuda::StreamAccessor::getStream(stream_);
            auto part=cd::Clock::now();
            cd::check_cuda(cudaMemcpyAsync(raw_.data,input,size_t(cfg_.width)*cfg_.height*sizeof(uint16_t),cudaMemcpyHostToDevice,native_stream));
            upload_layout<<<dim3((cfg_.height+31)/32,(cfg_.width+31)/32),dim3(32,8),0,native_stream>>>(raw_.data,image_.ptr<float>(),image_.step,cfg_.width,cfg_.height);
            cd::check_cuda(cudaGetLastError());result.upload_enqueue_s=cd::elapsed(part);
            // No new wait here: the reused engine consumes this same stream in order.
            auto native=engine_->run(image_);result.engine_s=native.stage_s;result.engine_download_s=native.download_s;
            std::vector<cd::Candidate> candidates;
            for(auto& group:native.candidates){
                std::sort(group.begin(),group.end(),[](const auto& a,const auto& b){return a.x==b.x?a.y<b.y:a.x<b.x;});
                candidates.insert(candidates.end(),group.begin(),group.end());
            }
            result.candidates.reserve(candidates.size());
            for(const auto& c:candidates)result.candidates.push_back({c.template_index,c.x,c.y,c.score});
            int n=int(candidates.size());part=cd::Clock::now();
            if(n){
                candidates_.reserve(std::max(size_t(n),candidates_.capacity));values_.reserve(size_t(n)*cfg_.count);
                std::vector<MapView> views(cfg_.count);
                for(int k=0;k<cfg_.count;++k)views[k]={native.device_maps[k].ptr<float>(),native.device_maps[k].step};
                cd::check_cuda(cudaMemcpyAsync(views_.data,views.data(),views.size()*sizeof(MapView),cudaMemcpyHostToDevice,native_stream));
                cd::check_cuda(cudaMemcpyAsync(candidates_.data,candidates.data(),candidates.size()*sizeof(cd::Candidate),cudaMemcpyHostToDevice,native_stream));
                sample_candidates<<<(n*cfg_.count+255)/256,256,0,native_stream>>>(candidates_.data,n,views_.data,values_.data,
                    (cfg_.template_width+1)/2,(cfg_.template_height+1)/2,cfg_.count);
                cd::check_cuda(cudaGetLastError());result.similarities.resize(size_t(n)*cfg_.count);
                cd::check_cuda(cudaMemcpyAsync(result.similarities.data(),values_.data,result.similarities.size()*sizeof(float),cudaMemcpyDeviceToHost,native_stream));
                stream_.waitForCompletion();
            }
            result.sampling_s=cd::elapsed(part);result.seconds=cd::elapsed(started);++epoch_;return result;
        }catch(const cv::Exception& error){throw defocus::CudaFailure(error.what());}
    }
    ImageDescriptor device_image()const override{
        if(!epoch_)throw std::invalid_argument("Run candidate search successfully before borrowing its device image");
        return {reinterpret_cast<uintptr_t>(image_.ptr<float>()),image_.step,cfg_.width,cfg_.height,device_,epoch_};
    }
};
}
std::unique_ptr<Search> make(const Config& cfg,const std::vector<Template>& templates){
    try{return std::make_unique<DirectSearch>(cfg,templates);}catch(const cv::Exception& error){throw defocus::CudaFailure(error.what());}
}
}
