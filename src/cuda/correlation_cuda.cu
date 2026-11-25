#include "defocus/tracker.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cub/cub.cuh>
#include <opencv2/core.hpp>
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <chrono>

namespace defocus { namespace {
void check(cudaError_t status) {if(status!=cudaSuccess) throw CudaFailure(cudaGetErrorString(status));}
void fftcheck(cufftResult status) {if(status!=CUFFT_SUCCESS) throw CudaFailure("Local cuFFT error "+std::to_string(status));}
template<class T> struct Buffer {
    T* p=nullptr;
    Buffer()=default; Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){cudaFree(p);}
    void allocate(size_t n){check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
};
template<class T> struct Pinned {
    T* p=nullptr;
    ~Pinned(){cudaFreeHost(p);}
    void allocate(size_t n){check(cudaHostAlloc(reinterpret_cast<void**>(&p),n*sizeof(T),cudaHostAllocDefault));}
};
struct Stream {cudaStream_t value=nullptr; Stream(){check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));} ~Stream(){cudaStreamDestroy(value);}};
struct Plan {cufftHandle value=0; ~Plan(){if(value)cufftDestroy(value);}};
struct Moment {double sum,square;};
struct Aggregate {double score; int first,ties,sum_x,sum_y;};
struct PeakWire {double score,left,right,above,below; int x,y,ties,sum_x,sum_y,neighbors;};
struct WalkState {int indices[3];PeakWire peaks[3];int initial,best,boundary,count,error;};
struct WalkSample {int index;PeakWire peak;};
struct Graph {
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    void clear()noexcept{if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);executable=nullptr;graph=nullptr;}
    ~Graph(){clear();}
};
Peak unpack(const PeakWire& p){return {p.score,p.left,p.right,p.above,p.below,p.x,p.y,p.ties,p.sum_x,p.sum_y,bool(p.neighbors)};}
struct ReducePeak {
    __host__ __device__ Aggregate operator()(Aggregate a,Aggregate b) const {
        if(a.score>b.score)return a; if(b.score>a.score)return b;
        return {a.score,a.first<b.first?a.first:b.first,a.ties+b.ties,a.sum_x+b.sum_x,a.sum_y+b.sum_y};
    }
};
__global__ void pad(const double* image,double* padded,int width,int height,int fw,int fh,double shift) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=fw*fh)return;
    int x=i%fw,y=i/fw; padded[i]=x<width&&y<height ? image[y*width+x]+shift : 0;
}
__global__ void prefix_rows(const double* image,Moment* prefix,int width,int height,double shift) {
    int y=blockIdx.x*blockDim.x+threadIdx.x; if(y>=height)return;
    double sum=0,square=0;
    for(int x=0;x<width;++x) {
        double v=image[y*width+x]+shift; sum+=v; square+=v*v;
        prefix[(y+1)*(width+1)+x+1]={sum,square};
    }
}
__global__ void prefix_columns(Moment* prefix,int width,int height) {
    int x=blockIdx.x*blockDim.x+threadIdx.x+1; if(x>width)return;
    double sum=0,square=0;
    for(int y=1;y<=height;++y) {
        Moment& v=prefix[y*(width+1)+x]; sum+=v.sum; square+=v.square; v={sum,square};
    }
}
__device__ Moment window(const Moment* prefix,int stride,int left,int top,int right,int bottom) {
    Moment a=prefix[bottom*stride+right],b=prefix[top*stride+right],c=prefix[bottom*stride+left],d=prefix[top*stride+left];
    return {a.sum-b.sum-c.sum+d.sum,a.square-b.square-c.square+d.square};
}
__global__ void full_statistics(const Moment* prefix,double* sums,double* norms,int width,int height,
                                 int tw,int th,double template_norm) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,fw=width+tw-1,fh=height+th-1; if(i>=fw*fh)return;
    int x=i%fw,y=i/fw;
    Moment v=window(prefix,width+1,max(0,x-tw+1),max(0,y-th+1),min(width,x+1),min(height,y+1));
    sums[i]=v.sum; norms[i]=template_norm*sqrt(fmax(v.square-v.sum*v.sum/(tw*th),0.0));
}
__global__ void multiply(const cufftDoubleComplex* image,const cufftDoubleComplex* templ,
                          cufftDoubleComplex* product,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count)return;
    auto a=image[i],b=templ[i]; product[i]={a.x*b.x+a.y*b.y,a.y*b.x-a.x*b.y};
}
__global__ void normalize_full(const double* inverse,double* output,const double* sums,const double* norms,
                               const double* maximum,int fw,int fh,int tw,int th,int fftw,int ffth,double residual,double template_norm) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=fw*fh)return;
    int x=i%fw,y=i/fw,ix=(x-tw+1+fftw)%fftw,iy=(y-th+1+ffth)%ffth;
    double m=*maximum*template_norm,tolerance=m>0 ? sqrt(ldexp(1.0,ilogb(m)-52)) : 0;
    double numerator=inverse[iy*fftw+ix]/(fftw*ffth)-sums[i]*residual;
    double denominator=norms[i]*template_norm;
    double value=denominator>tolerance ? numerator/denominator : 0;
    output[i]=fabs(value)-1>sqrt(0x1p-52) ? 0 : value;
}
__global__ void spatial_valid(const double* image,const double* templ,const Moment* prefix,double* output,
                              int width,int height,int tw,int th,double norm,double residual) {
    int columns=width-tw+1,position=blockIdx.x,x=position%columns,y=position/columns;
    if(y>=height-th+1)return;
    double dot=0;
    for(int i=threadIdx.x;i<tw*th;i+=blockDim.x) dot+=image[(y+i/tw)*width+x+i%tw]*templ[i];
    using Reduction=cub::BlockReduce<double,256>;
    __shared__ Reduction::TempStorage temporary;
    double sum=Reduction(temporary).Sum(dot);
    if(threadIdx.x==0) {
        Moment v=window(prefix,width+1,x,y,x+tw,y+th);
        double denominator=norm*sqrt(fmax(v.square-v.sum*v.sum/(tw*th),0.0));
        double value=denominator>0 ? (sum-v.sum*residual)/denominator : 0;
        output[(y+th-1)*(width+tw-1)+x+tw-1]=fabs(value)-1>sqrt(0x1p-52) ? 0 : value;
    }
}
__global__ void multiply_batch(const cufftDoubleComplex* image,const cufftDoubleComplex* templates,
                                cufftDoubleComplex* products,const int* indices,int spectrum_count,int batches) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=spectrum_count*batches)return;
    int k=i/spectrum_count,j=i%spectrum_count;
    auto a=image[j],b=templates[size_t(indices[k]-1)*spectrum_count+j];
    products[i]={a.x*b.x+a.y*b.y,a.y*b.x-a.x*b.y};
}
__global__ void normalize_batch(const double* inverse,double* maps,const double* sums,const double* norms,
                                 const double* maximum,const int* indices,const double* template_norms,
                                 const double* residuals,int width,int height,int tw,int th,int fftw,int ffth,int batches) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,count=width*height;if(i>=count*batches)return;
    int k=i/count,j=i%count,x=j%width,y=j/width,index=indices[k]-1;
    int ix=(x-tw+1+fftw)%fftw,iy=(y-th+1+ffth)%ffth;
    double norm=template_norms[index],maximum_denominator=*maximum*norm;
    double tolerance=maximum_denominator>0?sqrt(ldexp(1.0,ilogb(maximum_denominator)-52)):0;
    double denominator=norms[j]*norm;
    double numerator=inverse[size_t(k)*fftw*ffth+iy*fftw+ix]/(fftw*ffth)-sums[j]*residuals[index];
    double value=denominator>tolerance?numerator/denominator:0;
    maps[i]=fabs(value)-1>sqrt(0x1p-52)?0:value;
}
__global__ void spatial_batch(const double* image,const double* templates,const Moment* prefix,double* maps,
                               const int* indices,const double* norms,const double* residuals,
                               int width,int height,int tw,int th) {
    int columns=width-tw+1,rows=height-th+1,position=blockIdx.x,k=blockIdx.y;
    int x=position%columns,y=position/columns,index=indices[k]-1;
    const double* templ=templates+size_t(index)*tw*th;
    double dot=0;
    for(int i=threadIdx.x;i<tw*th;i+=blockDim.x)dot+=image[(y+i/tw)*width+x+i%tw]*templ[i];
    using Reduction=cub::BlockReduce<double,256>;
    __shared__ Reduction::TempStorage temporary;
    double sum=Reduction(temporary).Sum(dot);
    if(threadIdx.x==0) {
        Moment v=window(prefix,width+1,x,y,x+tw,y+th);
        double denominator=norms[index]*sqrt(fmax(v.square-v.sum*v.sum/(tw*th),0.0));
        double value=denominator>0?(sum-v.sum*residuals[index])/denominator:0;
        maps[k*columns*rows+position]=fabs(value)-1>sqrt(0x1p-52)?0:value;
    }
}
__global__ void reduce_full_batch(const double* maps,PeakWire* wires,int width,int height) {
    int k=blockIdx.x,count=width*height;const double* map=maps+size_t(k)*count;
    Aggregate partial{-INFINITY,INT_MAX,0,0,0};ReducePeak combine;
    for(int i=threadIdx.x;i<count;i+=blockDim.x) {
        int x=i%width,y=i/width;
        partial=combine(partial,Aggregate{map[i],x*height+y,1,x+1,y+1});
    }
    using Reduction=cub::BlockReduce<Aggregate,256>;
    __shared__ Reduction::TempStorage temporary;
    Aggregate a=Reduction(temporary).Reduce(partial,combine);
    if(threadIdx.x==0) {
        int x=a.first/height,y=a.first%height;
        PeakWire p{};p.score=a.score;p.x=x+1;p.y=y+1;p.ties=a.ties;p.sum_x=a.sum_x;p.sum_y=a.sum_y;
        p.neighbors=x>0&&y>0&&x+1<width&&y+1<height;
        if(p.neighbors){p.left=map[y*width+x-1];p.right=map[y*width+x+1];p.above=map[(y-1)*width+x];p.below=map[(y+1)*width+x];}
        wires[k]=p;
    }
}
__global__ void reduce_compact_batch(const double* maps,PeakWire* wires,int columns,int rows,
                                      int tw,int th,int full_width,int full_height) {
    int k=blockIdx.x,count=columns*rows;const double* map=maps+size_t(k)*count;
    Aggregate partial{-INFINITY,INT_MAX,0,0,0};ReducePeak combine;
    for(int i=threadIdx.x;i<count;i+=blockDim.x) {
        int x=i%columns+tw-1,y=i/columns+th-1;
        partial=combine(partial,Aggregate{map[i],x*full_height+y,1,x+1,y+1});
    }
    if(threadIdx.x==0) {
        int sum_x=rows*columns*(2*tw+columns-1)/2,sum_y=columns*rows*(2*th+rows-1)/2;
        Aggregate padding{0,0,full_width*full_height-count,
            full_height*full_width*(full_width+1)/2-sum_x,full_width*full_height*(full_height+1)/2-sum_y};
        partial=combine(partial,padding);
    }
    using Reduction=cub::BlockReduce<Aggregate,128>;
    __shared__ Reduction::TempStorage temporary;
    Aggregate a=Reduction(temporary).Reduce(partial,combine);
    if(threadIdx.x==0) {
        int x=a.first/full_height,y=a.first%full_height,vx=x-tw+1,vy=y-th+1;
        PeakWire p{};p.score=a.score;p.x=x+1;p.y=y+1;p.ties=a.ties;p.sum_x=a.sum_x;p.sum_y=a.sum_y;
        if(vx>=0&&vy>=0&&vx<columns&&vy<rows) {
            p.neighbors=1;p.left=vx>0?map[vy*columns+vx-1]:0;p.right=vx+1<columns?map[vy*columns+vx+1]:0;
            p.above=vy>0?map[(vy-1)*columns+vx]:0;p.below=vy+1<rows?map[(vy+1)*columns+vx]:0;
        }
        wires[k]=p;
    }
}
__global__ void walk_initialize(WalkState* state,const int* start,int* requests,int depths) {
    *state=WalkState{};state->initial=*start;int d=*start;
    if(d==1){requests[0]=1;requests[1]=2;requests[2]=3;}
    else if(d==depths){requests[0]=d;requests[1]=d-2;requests[2]=d-1;}
    else {requests[0]=d;requests[1]=d-1;requests[2]=d+1;}
}
__device__ bool walk_decide(WalkState* s,int* requests,int depths) {
    s->best=0;
    for(int k=1;k<3;++k)if(s->peaks[s->best].score<s->peaks[k].score)s->best=k;
    bool again=false;
    if(s->best==0&&s->indices[0]>1){requests[0]=s->indices[0]-1;again=true;}
    else if(s->best==2&&s->indices[2]<depths){requests[0]=s->indices[2]+1;again=true;}
    else if(s->best!=1)s->boundary=1;
    return again;
}
__device__ void walk_choose(WalkState* s,int* requests,int depths,cudaGraphConditionalHandle condition) {
    cudaGraphSetConditional(condition,walk_decide(s,requests,depths)?1:0);
}
__device__ void walk_seed(WalkState* s,const PeakWire* wires,int* requests,WalkSample* samples,
                                    int depths) {
    for(int k=0;k<3;++k)samples[k]={requests[k],wires[k]};
    s->count=3;int d=s->initial;
    if(d==1){for(int k=0;k<3;++k){s->indices[k]=k+1;s->peaks[k]=wires[k];}}
    else {
        s->indices[0]=d==depths?d-2:d-1;s->indices[1]=d==depths?d-1:d;s->indices[2]=d==depths?d:d+1;
        s->peaks[0]=wires[1];s->peaks[1]=wires[d==depths?2:0];s->peaks[2]=wires[d==depths?0:2];
    }
}
__global__ void walk_initial_peaks(WalkState* s,const PeakWire* wires,int* requests,WalkSample* samples,
                                    int depths,cudaGraphConditionalHandle condition) {
    walk_seed(s,wires,requests,samples,depths);walk_choose(s,requests,depths,condition);
}
__device__ bool walk_record(WalkState* s,const PeakWire* wires,int* requests,WalkSample* samples,
                              int depths) {
    if(s->count>=depths+3){s->error=1;return false;}
    int index=requests[0];samples[s->count++]={index,wires[0]};
    if(s->best==0){
        s->indices[2]=s->indices[1];s->indices[1]=s->indices[0];s->indices[0]=index;
        s->peaks[2]=s->peaks[1];s->peaks[1]=s->peaks[0];s->peaks[0]=wires[0];
    }else{
        s->indices[0]=s->indices[1];s->indices[1]=s->indices[2];s->indices[2]=index;
        s->peaks[0]=s->peaks[1];s->peaks[1]=s->peaks[2];s->peaks[2]=wires[0];
    }
    return true;
}
__global__ void walk_advance(WalkState* s,const PeakWire* wires,int* requests,WalkSample* samples,
                              int depths,cudaGraphConditionalHandle condition) {
    if(!walk_record(s,wires,requests,samples,depths)){cudaGraphSetConditional(condition,0);return;}
    walk_choose(s,requests,depths,condition);
}
__global__ void select_full_stack(WalkState* s,const PeakWire* all,int* requests,WalkSample* samples,int depths) {
    PeakWire initial[3]={all[requests[0]-1],all[requests[1]-1],all[requests[2]-1]};
    walk_seed(s,initial,requests,samples,depths);
    // Select exactly the same local peak as the adaptive walk, including plateaus and boundary stops.
    while(walk_decide(s,requests,depths))
        if(!walk_record(s,all+requests[0]-1,requests,samples,depths))break;
}
struct DepthTimer {
    double& target;std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
    ~DepthTimer(){target+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();}
};
class CudaCorrelation final:public CorrelationBackend {
    const Model& model_;
    int tw_,th_,cw_,ch_,fw_,fh_,fft_width_,fft_height_,real_count_,spectrum_count_;
    Stream stream_; Plan forward_,inverse_;
    Buffer<double> input_,padded_,inverse_data_,template_pixels_,map_,sums_,norms_,maximum_;
    Buffer<cufftDoubleComplex> spectra_,image_spectrum_,product_;
    Buffer<Moment> prefix_;
    Buffer<uint8_t> workspace_;
    size_t workspace_bytes_=0;
    std::vector<double> norms_host_,residuals_;
    Pinned<double> host_input_;
    bool crop_active_=false,prefix_ready_=false,full_stats_ready_=false,spectrum_ready_=false;
    double full_shift_=0,prefix_shift_=0;
    std::vector<Peak> full_cache_,valid_cache_;
    std::vector<uint8_t> full_cached_,valid_cached_;
    CorrelationMetrics metrics_;
    int batch_capacity_=1;
    Buffer<double> batch_inverse_,batch_maps_,norms_device_,residuals_device_;
    Buffer<cufftDoubleComplex> batch_products_;
    Buffer<int> batch_indices_;
    Buffer<PeakWire> batch_wires_;
    Pinned<int> host_indices_;
    Pinned<PeakWire> host_batch_wires_;
    std::vector<std::unique_ptr<Plan>> batch_plans_;
    Graph walk_graph_;
    Graph full_stack_graph_;
    DepthStrategy depth_strategy_=DepthStrategy::adaptive;
    Buffer<double> full_stack_maps_;Buffer<PeakWire> full_stack_peaks_;Buffer<int> full_stack_indices_;
    double full_stack_setup_seconds_=0;
    double walk_setup_seconds_=0;
    Buffer<WalkState> walk_state_;Buffer<WalkSample> walk_samples_;Buffer<int> walk_start_;
    Pinned<WalkState> host_walk_state_;Pinned<WalkSample> host_walk_samples_;
    void build_full_stack_graph() {
        if(full_stack_graph_.executable)return;
        auto started=std::chrono::steady_clock::now();
        int depths=int(model_.templates.size()),columns=cw_-tw_+1,rows=ch_-th_+1;
        full_stack_maps_.allocate(size_t(depths)*columns*rows);full_stack_peaks_.allocate(depths);full_stack_indices_.allocate(depths);
        std::vector<int> indices(depths);for(int k=0;k<depths;++k)indices[k]=k+1;
        check(cudaMemcpy(full_stack_indices_.p,indices.data(),size_t(depths)*sizeof(int),cudaMemcpyHostToDevice));
        check(cudaGraphCreate(&full_stack_graph_.graph,0));
        auto add=[&](void* function,void** arguments,dim3 grid,dim3 block,cudaGraphNode_t parent){
            cudaKernelNodeParams p{};p.func=function;p.kernelParams=arguments;p.gridDim=grid;p.blockDim=block;
            cudaGraphNode_t node;check(cudaGraphAddKernelNode(&node,full_stack_graph_.graph,parent?&parent:nullptr,parent?1:0,&p));return node;
        };
        void* initialize_args[]={&walk_state_.p,&walk_start_.p,&batch_indices_.p,&depths};
        auto init=add((void*)walk_initialize,initialize_args,1,1,nullptr);
        void* spatial_args[]={&input_.p,&template_pixels_.p,&prefix_.p,&full_stack_maps_.p,&full_stack_indices_.p,
            &norms_device_.p,&residuals_device_.p,&cw_,&ch_,&tw_,&th_};
        auto spatial=add((void*)spatial_batch,spatial_args,dim3(columns*rows,depths),256,init);
        void* reduce_args[]={&full_stack_maps_.p,&full_stack_peaks_.p,&columns,&rows,&tw_,&th_,&fw_,&fh_};
        auto reduce=add((void*)reduce_compact_batch,reduce_args,depths,128,spatial);
        void* select_args[]={&walk_state_.p,&full_stack_peaks_.p,&batch_indices_.p,&walk_samples_.p,&depths};
        add((void*)select_full_stack,select_args,1,1,reduce);
        check(cudaGraphInstantiate(&full_stack_graph_.executable,full_stack_graph_.graph,0));
        check(cudaGraphUpload(full_stack_graph_.executable,stream_.value));check(cudaStreamSynchronize(stream_.value));
        full_stack_setup_seconds_=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    }
    void build_walk_graph() {
        int depths=int(model_.templates.size()),columns=cw_-tw_+1,rows=ch_-th_+1;
        walk_state_.allocate(1);walk_samples_.allocate(depths+3);walk_start_.allocate(1);
        host_walk_state_.allocate(1);host_walk_samples_.allocate(depths+3);
        check(cudaGraphCreate(&walk_graph_.graph,0));
        cudaGraphConditionalHandle condition;
        auto status=cudaGraphConditionalHandleCreate(&condition,walk_graph_.graph,0,cudaGraphCondAssignDefault);
        if(status==cudaErrorNotSupported){walk_graph_.clear();cudaGetLastError();return;}check(status);
        auto add=[&](cudaGraph_t graph,void* function,void** arguments,dim3 grid,dim3 block,cudaGraphNode_t parent){
            cudaKernelNodeParams p{};p.func=function;p.kernelParams=arguments;p.gridDim=grid;p.blockDim=block;
            cudaGraphNode_t node;check(cudaGraphAddKernelNode(&node,graph,parent?&parent:nullptr,parent?1:0,&p));return node;
        };
        void* initialize_args[]={&walk_state_.p,&walk_start_.p,&batch_indices_.p,&depths};
        auto init=add(walk_graph_.graph,(void*)walk_initialize,initialize_args,1,1,nullptr);
        void* spatial_args[]={&input_.p,&template_pixels_.p,&prefix_.p,&batch_maps_.p,&batch_indices_.p,
            &norms_device_.p,&residuals_device_.p,&cw_,&ch_,&tw_,&th_};
        auto spatial=add(walk_graph_.graph,(void*)spatial_batch,spatial_args,dim3(columns*rows,3),256,init);
        void* reduce_args[]={&batch_maps_.p,&batch_wires_.p,&columns,&rows,&tw_,&th_,&fw_,&fh_};
        auto reduce=add(walk_graph_.graph,(void*)reduce_compact_batch,reduce_args,3,128,spatial);
        void* decision_args[]={&walk_state_.p,&batch_wires_.p,&batch_indices_.p,&walk_samples_.p,&depths,&condition};
        auto decide=add(walk_graph_.graph,(void*)walk_initial_peaks,decision_args,1,1,reduce);
        cudaGraphNodeParams params{};params.type=cudaGraphNodeTypeConditional;
        params.conditional.handle=condition;params.conditional.type=cudaGraphCondTypeWhile;params.conditional.size=1;
        cudaGraphNode_t loop;check(cudaGraphAddNode(&loop,walk_graph_.graph,&decide,nullptr,1,&params));
        auto body=params.conditional.phGraph_out[0];
        auto spatial_one=add(body,(void*)spatial_batch,spatial_args,dim3(columns*rows,1),256,nullptr);
        auto reduce_one=add(body,(void*)reduce_compact_batch,reduce_args,1,128,spatial_one);
        add(body,(void*)walk_advance,decision_args,1,1,reduce_one);
        status=cudaGraphInstantiate(&walk_graph_.executable,walk_graph_.graph,0);
        if(status==cudaErrorNotSupported){walk_graph_.clear();cudaGetLastError();return;}check(status);
        // Upload executable at initialization; first-particle upload overhead stays outside steady-state timing.
        check(cudaGraphUpload(walk_graph_.executable,stream_.value));check(cudaStreamSynchronize(stream_.value));
    }
#ifdef DEFOCUS_TEST_HOOKS
    int fail_after_=-1,completed_calls_=0;
#endif
    void prepare_image(CorrelationMode mode) {
        double shift=mode==CorrelationMode::matlab_full ? full_shift_ : 0;
        if(!prefix_ready_ || prefix_shift_!=shift) {
            check(cudaMemsetAsync(prefix_.p,0,size_t(cw_+1)*(ch_+1)*sizeof(Moment),stream_.value));
            prefix_rows<<<(ch_+127)/128,128,0,stream_.value>>>(input_.p,prefix_.p,cw_,ch_,shift);
            prefix_columns<<<(cw_+127)/128,128,0,stream_.value>>>(prefix_.p,cw_,ch_);
            prefix_ready_=true;prefix_shift_=shift;full_stats_ready_=false;++metrics_.prefix_builds;
        }
        if(mode==CorrelationMode::matlab_full) {
            if(!full_stats_ready_) {
                full_statistics<<<(fw_*fh_+255)/256,256,0,stream_.value>>>(prefix_.p,sums_.p,norms_.p,cw_,ch_,tw_,th_,1.0);
                check(cub::DeviceReduce::Max(workspace_.p,workspace_bytes_,norms_.p,maximum_.p,fw_*fh_,stream_.value));
                full_stats_ready_=true;
            }
            if(!spectrum_ready_) {
                pad<<<(real_count_+255)/256,256,0,stream_.value>>>(input_.p,padded_.p,cw_,ch_,fft_width_,fft_height_,shift);
                fftcheck(cufftExecD2Z(forward_.value,padded_.p,image_spectrum_.p));
                spectrum_ready_=true;++metrics_.image_ffts;
            }
        }
    }
    void compute(const Image& image,int index,CorrelationMode mode) {
        if(index<1||index>int(model_.templates.size())||image.width!=cw_||image.height!=ch_)
            throw std::invalid_argument("CUDA local correlation requires the initialized template index and crop dimensions");
        prepare_image(mode);++metrics_.template_evaluations;
        if(mode==CorrelationMode::legacy_padded_valid) {
            check(cudaMemsetAsync(map_.p,0,size_t(fw_)*fh_*sizeof(double),stream_.value));
            spatial_valid<<<(cw_-tw_+1)*(ch_-th_+1),256,0,stream_.value>>>(input_.p,
                template_pixels_.p+size_t(index-1)*tw_*th_,prefix_.p,map_.p,cw_,ch_,tw_,th_,norms_host_[index-1],residuals_[index-1]);
        } else {
            multiply<<<(spectrum_count_+255)/256,256,0,stream_.value>>>(image_spectrum_.p,
                spectra_.p+size_t(index-1)*spectrum_count_,product_.p,spectrum_count_);
            fftcheck(cufftExecZ2D(inverse_.value,product_.p,inverse_data_.p));
            ++metrics_.inverse_ffts;
            normalize_full<<<(fw_*fh_+255)/256,256,0,stream_.value>>>(inverse_data_.p,map_.p,sums_.p,norms_.p,
                maximum_.p,fw_,fh_,tw_,th_,fft_width_,fft_height_,residuals_[index-1],norms_host_[index-1]);
        }
        check(cudaGetLastError());
    }
    void execute_batch(const int* indices,int n,CorrelationMode mode) {
        prepare_image(mode);++metrics_.batch_submissions;metrics_.template_evaluations+=n;
        std::copy_n(indices,n,host_indices_.p);
        check(cudaMemcpyAsync(batch_indices_.p,host_indices_.p,size_t(n)*sizeof(int),cudaMemcpyHostToDevice,stream_.value));
        int compact_count=(cw_-tw_+1)*(ch_-th_+1),map_count=fw_*fh_;
        if(mode==CorrelationMode::matlab_full) {
            multiply_batch<<<(spectrum_count_*n+255)/256,256,0,stream_.value>>>(image_spectrum_.p,spectra_.p,batch_products_.p,batch_indices_.p,spectrum_count_,n);
            fftcheck(cufftExecZ2D(batch_plans_[n]->value,batch_products_.p,batch_inverse_.p));metrics_.inverse_ffts+=n;
            normalize_batch<<<(map_count*n+255)/256,256,0,stream_.value>>>(batch_inverse_.p,batch_maps_.p,sums_.p,norms_.p,maximum_.p,
                batch_indices_.p,norms_device_.p,residuals_device_.p,fw_,fh_,tw_,th_,fft_width_,fft_height_,n);
        } else {
            spatial_batch<<<dim3(compact_count,n),256,0,stream_.value>>>(input_.p,template_pixels_.p,prefix_.p,batch_maps_.p,
                batch_indices_.p,norms_device_.p,residuals_device_.p,cw_,ch_,tw_,th_);
        }
        if(mode==CorrelationMode::matlab_full)
            reduce_full_batch<<<n,256,0,stream_.value>>>(batch_maps_.p,batch_wires_.p,fw_,fh_);
        else reduce_compact_batch<<<n,128,0,stream_.value>>>(batch_maps_.p,batch_wires_.p,cw_-tw_+1,ch_-th_+1,tw_,th_,fw_,fh_);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(host_batch_wires_.p,batch_wires_.p,size_t(n)*sizeof(PeakWire),cudaMemcpyDeviceToHost,stream_.value));
        check(cudaStreamSynchronize(stream_.value));++metrics_.result_downloads;++metrics_.synchronizations;
        auto& cache=mode==CorrelationMode::matlab_full?full_cache_:valid_cache_;
        auto& flags=mode==CorrelationMode::matlab_full?full_cached_:valid_cached_;
        for(int k=0;k<n;++k) {
            const auto& p=host_batch_wires_.p[k];
            cache[indices[k]-1]={p.score,p.left,p.right,p.above,p.below,p.x,p.y,p.ties,p.sum_x,p.sum_y,bool(p.neighbors)};
            flags[indices[k]-1]=1;
        }
    }
public:
    explicit CudaCorrelation(const Model& model):model_(model),tw_(model.templates[0].width),th_(model.templates[0].height),
        cw_(tw_+8),ch_(th_+8),fw_(cw_+tw_-1),fh_(ch_+th_-1),
        fft_width_(cv::getOptimalDFTSize(fw_)),fft_height_(cv::getOptimalDFTSize(fh_)),
        real_count_(fft_width_*fft_height_),spectrum_count_(fft_height_*(fft_width_/2+1)) {
        int n=int(model.templates.size());
        host_input_.allocate(cw_*ch_);
        full_cache_.resize(n);valid_cache_.resize(n);full_cached_.resize(n);valid_cached_.resize(n);
        input_.allocate(cw_*ch_);padded_.allocate(real_count_);inverse_data_.allocate(real_count_);
        image_spectrum_.allocate(spectrum_count_);product_.allocate(spectrum_count_);spectra_.allocate(size_t(n)*spectrum_count_);
        template_pixels_.allocate(size_t(n)*tw_*th_);prefix_.allocate((cw_+1)*(ch_+1));
        map_.allocate(fw_*fh_);sums_.allocate(fw_*fh_);norms_.allocate(fw_*fh_);maximum_.allocate(1);
        size_t a=0;
        check(cub::DeviceReduce::Max(nullptr,a,norms_.p,maximum_.p,fw_*fh_));
        workspace_bytes_=a;workspace_.allocate(workspace_bytes_);
        fftcheck(cufftPlan2d(&forward_.value,fft_height_,fft_width_,CUFFT_D2Z));
        fftcheck(cufftPlan2d(&inverse_.value,fft_height_,fft_width_,CUFFT_Z2D));
        fftcheck(cufftSetStream(forward_.value,stream_.value));fftcheck(cufftSetStream(inverse_.value,stream_.value));
        std::vector<double> packed(size_t(n)*real_count_,0),pixels(size_t(n)*tw_*th_);
        for(int k=0;k<n;++k) {
            double mean=0,sum=0,square=0;
            for(int x=0;x<tw_;++x)for(int y=0;y<th_;++y)mean+=model.templates[k].at(x,y);
            mean/=tw_*th_;
            for(int x=0;x<tw_;++x)for(int y=0;y<th_;++y) {
                double value=model.templates[k].at(x,y)-mean;
                packed[size_t(k)*real_count_+y*fft_width_+x]=value;
                pixels[size_t(k)*tw_*th_+y*tw_+x]=value;sum+=value;square+=value*value;
            }
            residuals_.push_back(sum/(tw_*th_));norms_host_.push_back(std::sqrt(square-sum*sum/(tw_*th_)));
        }
        check(cudaMemcpy(template_pixels_.p,pixels.data(),pixels.size()*sizeof(double),cudaMemcpyHostToDevice));
        Buffer<double> device_packed;device_packed.allocate(packed.size());
        check(cudaMemcpy(device_packed.p,packed.data(),packed.size()*sizeof(double),cudaMemcpyHostToDevice));
        Plan batch; int dims[2]={fft_height_,fft_width_};
        fftcheck(cufftPlanMany(&batch.value,2,dims,nullptr,1,real_count_,nullptr,1,spectrum_count_,CUFFT_D2Z,n));
        fftcheck(cufftSetStream(batch.value,stream_.value));
        fftcheck(cufftExecD2Z(batch.value,device_packed.p,spectra_.p));check(cudaStreamSynchronize(stream_.value));
        batch_capacity_=std::min(16,std::max(3,int(model.selected.size())));
        batch_inverse_.allocate(size_t(batch_capacity_)*real_count_);batch_maps_.allocate(size_t(batch_capacity_)*fw_*fh_);
        batch_products_.allocate(size_t(batch_capacity_)*spectrum_count_);batch_indices_.allocate(batch_capacity_);
        batch_wires_.allocate(batch_capacity_);host_indices_.allocate(batch_capacity_);host_batch_wires_.allocate(batch_capacity_);
        norms_device_.allocate(n);residuals_device_.allocate(n);
        check(cudaMemcpy(norms_device_.p,norms_host_.data(),size_t(n)*sizeof(double),cudaMemcpyHostToDevice));
        check(cudaMemcpy(residuals_device_.p,residuals_.data(),size_t(n)*sizeof(double),cudaMemcpyHostToDevice));
        batch_plans_.resize(batch_capacity_+1);
        for(int count=1;count<=batch_capacity_;++count) {
            batch_plans_[count]=std::make_unique<Plan>();
            fftcheck(cufftPlanMany(&batch_plans_[count]->value,2,dims,nullptr,1,spectrum_count_,nullptr,1,real_count_,CUFFT_Z2D,count));
            fftcheck(cufftSetStream(batch_plans_[count]->value,stream_.value));
        }
        auto setup_started=std::chrono::steady_clock::now();build_walk_graph();
        walk_setup_seconds_=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup_started).count();
    }
    std::string name()const override{return "cuda";}
    std::string depth_walk_implementation()const override{return depth_strategy_==DepthStrategy::full_stack?"cuda_full_stack_graph":walk_graph_.executable?"cuda_graph":"host";}
    double depth_walk_setup_seconds()const override{return depth_strategy_==DepthStrategy::full_stack?full_stack_setup_seconds_:walk_setup_seconds_;}
    void set_depth_strategy(DepthStrategy strategy)override {
        if(strategy==DepthStrategy::full_stack)build_full_stack_graph();
        depth_strategy_=strategy;
    }
    ~CudaCorrelation()override{cudaStreamSynchronize(stream_.value);}
    void reset_metrics()override{metrics_={};}
    CorrelationMetrics metrics()const override{return metrics_;}
    void begin_crop(const Image& image)override {
        end_crop();
        if(image.width!=cw_ || image.height!=ch_) throw std::invalid_argument("CUDA crop dimensions do not match initialization");
        std::copy(image.pixels.begin(),image.pixels.end(),host_input_.p);
        double minimum=*std::min_element(image.pixels.begin(),image.pixels.end());
        full_shift_=minimum<0 ? -minimum : 0;
        check(cudaMemcpyAsync(input_.p,host_input_.p,image.pixels.size()*sizeof(double),cudaMemcpyHostToDevice,stream_.value));
        crop_active_=true;prefix_ready_=full_stats_ready_=spectrum_ready_=false;
        std::fill(full_cached_.begin(),full_cached_.end(),0);std::fill(valid_cached_.begin(),valid_cached_.end(),0);
        ++metrics_.crop_uploads;
    }
    void end_crop()noexcept override{crop_active_=false;}
#ifdef DEFOCUS_TEST_HOOKS
    void test_failure_after(int count) override {fail_after_=count;completed_calls_=0;}
#endif
    Image correlate(const Image& image,int index,CorrelationMode mode) override {
        bool transient=!crop_active_;if(transient)begin_crop(image);
        try {
            compute(image,index,mode);Image result(fw_,fh_);
            check(cudaMemcpyAsync(result.pixels.data(),map_.p,result.pixels.size()*sizeof(double),cudaMemcpyDeviceToHost,stream_.value));
            check(cudaStreamSynchronize(stream_.value));++metrics_.result_downloads;++metrics_.synchronizations;
            if(transient)end_crop();return result;
        }catch(...){if(transient)end_crop();throw;}
    }
    Peak peak(const Image& image,int index,CorrelationMode mode) override {
        return peaks_batch(image,{index},mode).front();
    }
    DepthWalk depth_walk(const Image& image,int depth,int depths)override {
        if(depths!=int(model_.templates.size())||depth<1||depth>depths||image.width!=cw_||image.height!=ch_)
            throw std::invalid_argument("Invalid CUDA depth walk bounds or crop");
        DepthTimer timer{metrics_.depth_walk_s};
        bool full_stack=depth_strategy_==DepthStrategy::full_stack;
        if(!full_stack&&!walk_graph_.executable){++metrics_.host_depth_walks;return CorrelationBackend::depth_walk(image,depth,depths);}
        bool transient=!crop_active_;if(transient)begin_crop(image);
        try {
            prepare_image(CorrelationMode::legacy_padded_valid);
            host_indices_.p[0]=depth;
            check(cudaMemcpyAsync(walk_start_.p,host_indices_.p,sizeof(int),cudaMemcpyHostToDevice,stream_.value));
            check(cudaGraphLaunch(full_stack?full_stack_graph_.executable:walk_graph_.executable,stream_.value));
            check(cudaMemcpyAsync(host_walk_state_.p,walk_state_.p,sizeof(WalkState),cudaMemcpyDeviceToHost,stream_.value));
            check(cudaMemcpyAsync(host_walk_samples_.p,walk_samples_.p,size_t(depths+3)*sizeof(WalkSample),cudaMemcpyDeviceToHost,stream_.value));
            check(cudaStreamSynchronize(stream_.value));
            ++metrics_.synchronizations;++metrics_.result_downloads;++metrics_.batch_submissions;++metrics_.gpu_depth_walks;
            if(full_stack)++metrics_.full_stack_walks;
            const auto& s=*host_walk_state_.p;
            if(s.error||s.count<3||s.count>depths+3)throw CudaFailure("GPU depth walk exceeded its bounded trace capacity");
            DepthWalk result;result.best=s.best;result.boundary=s.boundary!=0;result.samples.reserve(s.count);
            for(int k=0;k<3;++k){result.indices[k]=s.indices[k];result.peaks[k]=unpack(s.peaks[k]);}
            for(int k=0;k<s.count;++k){
#ifdef DEFOCUS_TEST_HOOKS
                if(fail_after_>=0&&completed_calls_++>=fail_after_)throw CudaFailure("Injected CUDA failure during device depth walk");
#endif
                const auto& sample=host_walk_samples_.p[k];auto p=unpack(sample.peak);
                result.samples.push_back({sample.index,p});valid_cache_[sample.index-1]=p;valid_cached_[sample.index-1]=1;
            }
            metrics_.logical_requests+=s.count;metrics_.template_evaluations+=full_stack?depths:s.count;
            if(transient)end_crop();return result;
        }catch(...){if(transient)end_crop();throw;}
    }
    std::vector<Peak> peaks_batch(const Image& image,const std::vector<int>& indices,CorrelationMode mode)override {
        if(indices.empty())return {};
        for(int index:indices) {
            if(index<1||index>int(model_.templates.size()))throw std::invalid_argument("Invalid batch template index");
            ++metrics_.logical_requests;
#ifdef DEFOCUS_TEST_HOOKS
            if(fail_after_>=0 && completed_calls_++>=fail_after_)throw CudaFailure("Injected CUDA failure after completed correlation calls");
#endif
        }
        bool transient=!crop_active_;if(transient)begin_crop(image);
        try {
            auto& flags=mode==CorrelationMode::matlab_full ? full_cached_:valid_cached_;
            auto& cache=mode==CorrelationMode::matlab_full ? full_cache_:valid_cache_;
            std::vector<int> missing;std::vector<uint8_t> pending(flags.size(),0);
            for(int index:indices) {
                if(flags[index-1]||pending[index-1])++metrics_.cache_hits;
                else {pending[index-1]=1;missing.push_back(index);}
            }
            for(size_t start=0;start<missing.size();start+=batch_capacity_)
                execute_batch(missing.data()+start,std::min(batch_capacity_,int(missing.size()-start)),mode);
            std::vector<Peak> result;result.reserve(indices.size());
            for(int index:indices)result.push_back(cache[index-1]);
            if(transient)end_crop();return result;
        }catch(...){if(transient)end_crop();throw;}
    }
};
}
std::unique_ptr<CorrelationBackend> make_cuda_correlation(const Model& model) {return std::make_unique<CudaCorrelation>(model);}
} // namespace defocus
