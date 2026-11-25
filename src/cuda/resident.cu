#include "resident.hpp"
#include <future>
#include <cuda.h>
// Reuse the audited double-precision NCC and reduction kernels without modifying them.
#include "correlation_cuda.cu"

namespace defocus { namespace {
void check_background_driver(CUresult status,const char* operation){
    if(status==CUDA_SUCCESS)return;
    const char* text=nullptr;cuGetErrorString(status,&text);
    throw CudaFailure(std::string(operation)+": "+(text?text:"unknown CUDA driver error"));
}
struct BackgroundCudaContext {
    bool active=false;
    explicit BackgroundCudaContext(CUcontext context){
        if(!context)throw CudaFailure("Background submission has no captured CUDA context");
        check_background_driver(cuCtxPushCurrent(context),"cuCtxPushCurrent background submission");active=true;
    }
    void close(){
        if(!active)return;CUcontext previous=nullptr;
        auto status=cuCtxPopCurrent(&previous);active=false;
        check_background_driver(status,"cuCtxPopCurrent background submission");
    }
    ~BackgroundCudaContext(){if(active){CUcontext previous=nullptr;cuCtxPopCurrent(&previous);}}
};
#include "mixed_depth.cuh"
#include "fused_depth.cuh"
#include "fused_preparation.cuh"
// Immutable calibration storage: the primary worker owns allocations;
// clone graphs borrow the same device addresses, never writing or freeing them.
template<class T> struct SharedBuffer {
    T* p=nullptr;size_t count=0;bool owner=false;
    SharedBuffer()=default;
    SharedBuffer(const SharedBuffer&)=delete;
    SharedBuffer& operator=(const SharedBuffer&)=delete;
    ~SharedBuffer(){if(owner)cudaFree(p);}
    void allocate(size_t n){
        if(p||!n||n>std::numeric_limits<size_t>::max()/sizeof(T))throw std::invalid_argument("Invalid shared calibration allocation");
        T* allocation=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&allocation),n*sizeof(T)));
        p=allocation;count=n;owner=true;
    }
    void borrow(const SharedBuffer& source){
        if(p||!source.p||!source.owner)throw std::invalid_argument("Calibration must be borrowed from an initialized primary owner");
        p=source.p;count=source.count;owner=false;
    }
};
__global__ void warmup_math(double* data,int count){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;double v=.125+double(i%31)*.001;
    for(int k=0;k<2048;++k)v=fma(v,.999999,.000001);data[i]=v;
}
struct ResidentState {
    int source,depth,ghost_count,overlap,active,suppress,mask_depth,trace_count;
    double x,y,background;
    int origin_x[3],origin_y[3];double weights[3];
};
struct ResidentOutput {
    int source,initial_depth,final_depth,rejection,full_calls,legacy_calls,overlap,clamped_xy;
    double x,y,z,score;
};
struct ResidentTrace {int index,mode;PeakWire peak;};
__global__ void transpose_integral(const double* input,double* image,int width,int height,int* error) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=width*height)return;
    int x=i%width,y=i/width;double v=input[size_t(x)*height+y];
    if(!isfinite(v)||v<0||v>65535||v!=floor(v)){atomicExch(error,1);v=0;}
    image[i]=v;
}
__global__ void copy_device_float(const float* input,size_t step,double* image,int width,int height,int* error) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=width*height)return;int x=i%width,y=i/width;
    double v=reinterpret_cast<const float*>(reinterpret_cast<const char*>(input)+size_t(y)*step)[x];
    if(!isfinite(v)||v<0||v>65535||v!=floor(v)){atomicExch(error,1);v=0;}image[i]=v;
}
__global__ void initial_spatial_bounded(const double* image,const double* templates,const Moment* prefix,double* maps,
    const int* index,const double* norms,const double* residuals,int width,int height,int tw,int th){
    int columns=width-tw+3,rows=height-th+3,position=blockIdx.x,x=position%columns-1,y=position/columns-1;
    if(y>=rows-1)return;int depth=*index-1;const double* templ=templates+size_t(depth)*tw*th;double dot=0;
    for(int i=threadIdx.x;i<tw*th;i+=256){int xx=x+i%tw,yy=y+i/tw;if(xx>=0&&yy>=0&&xx<width&&yy<height)dot+=image[yy*width+xx]*templ[i];}
    using Reduction=cub::BlockReduce<double,256>;__shared__ Reduction::TempStorage temp;double sum=Reduction(temp).Sum(dot);
    if(!threadIdx.x){Moment v=window(prefix,width+1,max(0,x),max(0,y),min(width,x+tw),min(height,y+th));
        double denominator=norms[depth]*sqrt(fmax(v.square-v.sum*v.sum/(tw*th),0.0));
        double value=denominator>0?(sum-v.sum*residuals[depth])/denominator:0;maps[position]=fabs(value)-1>sqrt(0x1p-52)?0:value;}
}
__global__ void reduce_initial_bounded(const double* map,PeakWire* wires,int columns,int rows,int tw,int th,int full_height){
    Aggregate partial{-INFINITY,INT_MAX,0,0,0};ReducePeak combine;
    int search_columns=columns-2,search_rows=rows-2;
    for(int i=threadIdx.x;i<search_columns*search_rows;i+=128){int mx=i%search_columns+1,my=i/search_columns+1;
        int fx=mx-1+tw-1,fy=my-1+th-1;partial=combine(partial,Aggregate{map[my*columns+mx],fx*full_height+fy,1,fx+1,fy+1});}
    using Reduction=cub::BlockReduce<Aggregate,128>;__shared__ Reduction::TempStorage temporary;Aggregate a=Reduction(temporary).Reduce(partial,combine);
    if(!threadIdx.x){int fx=a.first/full_height,fy=a.first%full_height,mx=fx-tw+2,my=fy-th+2;PeakWire p{};
        p.score=a.score;p.x=fx+1;p.y=fy+1;p.ties=a.ties;p.sum_x=a.sum_x;p.sum_y=a.sum_y;p.neighbors=1;
        p.left=map[my*columns+mx-1];p.right=map[my*columns+mx+1];p.above=map[(my-1)*columns+mx];p.below=map[(my+1)*columns+mx];*wires=p;}
}
__global__ void resident_begin(ResidentState* s,ResidentOutput* outputs,int* current,const CoarseCandidate* coarse,
                                int* request,int* start,WalkState* walk,const int* schedule) {
    int i=schedule[*current];auto c=coarse[i];*s=ResidentState{};*walk=WalkState{};
    s->source=i;s->depth=c.depth;s->x=c.x;s->y=c.y;
    *start=c.depth;request[0]=c.depth;
    ResidentOutput p{};p.source=i+1;p.initial_depth=c.depth;p.rejection=1;
    p.x=p.y=p.z=p.score=NAN;outputs[i]=p;
}
__global__ void resident_crop(const double* image,const uint8_t* ghost,double* crop,const double* mask,
                               ResidentState* s,const CoarseCandidate* coarse,int width,int height,int cw,int ch,
                               int tw,int th,double final_threshold,int* error,cudaGraphConditionalHandle overlap) {
    __shared__ int histogram[256],fine[256],bucket,rank;
    int t=threadIdx.x;histogram[t]=fine[t]=0;__syncthreads();
    int ox=int(round(s->x))-1-cw/2,oy=int(round(s->y))-1-ch/2;
    if(ox<0||oy<0||ox+cw>width||oy+ch>height){if(!t){*error=2;cudaGraphSetConditional(overlap,0);}return;}
    for(int i=t;i<cw*ch;i+=256){
        int x=ox+i%cw,y=oy+i/cw;double v=image[size_t(y)*width+x];crop[i]=v;
        atomicAdd(histogram+(int(v)>>8),1);
        if(ghost[size_t(y)*width+x])atomicAdd(&s->ghost_count,1);
    }
    __syncthreads();
    if(!t){int target=cw*ch/2,sum=0;bucket=0;while(bucket<255&&sum+histogram[bucket]<=target)sum+=histogram[bucket++];rank=target-sum;}
    __syncthreads();
    for(int i=t;i<cw*ch;i+=256){int v=int(crop[i]);if((v>>8)==bucket)atomicAdd(fine+(v&255),1);}
    __syncthreads();
    if(!t){int lo=0,sum=0;while(lo<255&&sum+fine[lo]<=rank)sum+=fine[lo++];s->background=bucket*256+lo;
        s->overlap=s->ghost_count>10&&coarse[s->source].score<final_threshold;cudaGraphSetConditional(overlap,s->overlap);}
    __syncthreads();
    for(int i=t;i<cw*ch;i+=256){int x=i%cw,y=i/cw;
        double m=x>=4&&y>=4&&x<cw-4&&y<ch-4?mask[(y-4)*tw+x-4]:0;
        crop[i]=crop[i]*m+(m==0?s->background:0);
    }
}
__device__ void resident_record(ResidentState* s,ResidentTrace* trace,int stride,int index,int mode,PeakWire p){
    trace[size_t(s->source)*stride+s->trace_count++]={index,mode,p};
}
__global__ void resident_overlap(ResidentState* s,const PeakWire* peaks,const int* selected,int count,
                                   const double* similarity,int depths,int* request,ResidentTrace* trace,int stride){
    double mean=0;for(int k=0;k<count;++k){mean+=peaks[k].score;resident_record(s,trace,stride,selected[k],0,peaks[k]);}mean/=count;
    double input_norm=0;for(int k=0;k<count;++k){double v=peaks[k].score-mean;input_norm+=v*v;}
    double best=-INFINITY;int depth=1;
    for(int j=0;j<depths;++j){double average=0;for(int k=0;k<count;++k)average+=similarity[size_t(selected[k]-1)*depths+j];average/=count;
        double numerator=0,norm=0;for(int k=0;k<count;++k){double v=similarity[size_t(selected[k]-1)*depths+j]-average;
            numerator+=v*(peaks[k].score-mean);norm+=v*v;}
        double score=numerator/sqrt(norm*input_norm);if(score>best){best=score;depth=j+1;}
    }s->depth=depth;request[0]=depth;
}
__device__ double real_gaussian(double a,double b,double c,int* error){
    if(a==0||b==0||c==0)return NAN;
    if(a<0||b<0||c<0){
        if(b>=0&&a<0&&c<0&&log(-a)==log(-c))return 0;
        *error=3;return NAN;
    }
    double la=log(a),lb=log(b),lc=log(c);return (la-lc)/(2*la-4*lb+2*lc);
}
__device__ bool resident_initial_fit(double a,double b,double c,double& value){
    if(a<0||b<0||c<0){if(b>=0&&a<0&&c<0&&log(-a)==log(-c)){value=0;return true;}return false;}
    double la=log(a),lb=log(b),lc=log(c);value=(la-lc)/(2*la-4*lb+2*lc);return true;
}
__global__ void resident_initial(ResidentState* s,ResidentOutput* outputs,const PeakWire* wire,int* start,
                                   int width,int height,int cw,int ch,int tw,int th,double threshold,bool bounded,int* error,
                                   ResidentTrace* trace,int stride,cudaGraphConditionalHandle active){
    PeakWire p=*wire;resident_record(s,trace,stride,s->depth,0,p);auto& out=outputs[s->source];
    cudaGraphSetConditional(active,0);
    if(p.score<threshold)return;if(!p.neighbors){out.rejection=2;return;}
    // The original initial Gaussian rejects nonreal fits per particle, not per frame.
    double dx,dy;if(!resident_initial_fit(p.left,p.score,p.right,dx)||!resident_initial_fit(p.above,p.score,p.below,dy)){out.rejection=3;return;}
    if(bounded){
        if(isfinite(dx)&&fabs(dx)>.5){dx=fmin(.5,fmax(-.5,dx));out.clamped_xy=1;}
        if(isfinite(dy)&&fabs(dy)>.5){dy=fmin(.5,fmax(-.5,dy));out.clamped_xy=1;}
    }
    double x=s->x-(cw+tw)/2.0+p.x+dx,y=s->y-(ch+th)/2.0+p.y+dy;
    double left=x-(cw-1)/2.0,top=y-(ch-1)/2.0;
    if(!isfinite(left)||!isfinite(top)||floor(left)<1||floor(top)<1||ceil(left+cw-1)>width||ceil(top+ch-1)>height){out.rejection=4;return;}
    s->x=x;s->y=y;double fx=left-floor(left),fy=top-floor(top);
    double ax[4]={fx,fx,fx-1,fx-1},ay[4]={fy,fy-1,fy,fy-1};int order[4]={0,1,2,3};
    for(int k=1;k<4;++k){int value=order[k],j=k;while(j>0&&ax[value]*ax[value]+ay[value]*ay[value]<ax[order[j-1]]*ax[order[j-1]]+ay[order[j-1]]*ay[order[j-1]]){order[j]=order[j-1];--j;}order[j]=value;}
    s->weights[1]=fabs(ax[order[0]]);s->weights[2]=fabs(ay[order[0]]);s->weights[0]=1-s->weights[1]-s->weights[2];
    for(int k=0;k<3;++k){int corner=order[k];s->origin_x[k]=int(corner>=2?ceil(left):floor(left))-1;s->origin_y[k]=int(corner==1||corner==3?ceil(top):floor(top))-1;}
    *start=s->depth;s->active=1;cudaGraphSetConditional(active,1);
}
__global__ void resident_recenter(const double* image,double* crop,const double* mask,const ResidentState* s,
                                    int width,int cw,int ch,int tw){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=cw*ch)return;int x=i%cw,y=i/cw;double v=0;
    for(int k=0;k<3;++k)v+=image[size_t(s->origin_y[k]+y)*width+s->origin_x[k]+x]*s->weights[k];
    double m=x>=4&&y>=4&&x<cw-4&&y<ch-4?mask[(y-4)*tw+x-4]:0;
    crop[i]=v*m+(m==0?s->background:0);
}
__global__ void resident_final(ResidentState* s,ResidentOutput* outputs,const WalkState* walk,const WalkSample* samples,
                                 int depths,int cw,int ch,int tw,int th,bool interpolate,double threshold,int* error,
                                 ResidentTrace* trace,int stride){
    auto& out=outputs[s->source];out.legacy_calls=walk->count;
    for(int k=0;k<walk->count;++k)resident_record(s,trace,stride,samples[k].index,1,samples[k].peak);
    if(walk->error){*error=4;return;}
    PeakWire final=walk->peaks[walk->best];int depth;double zz;
    if(!walk->boundary&&interpolate){depth=walk->indices[1];const auto* p=walk->peaks;
        zz=depth+(p[0].score-p[2].score)/(2*p[0].score-4*p[1].score+2*p[2].score);final=p[1];}
    else{depth=walk->indices[walk->best];zz=depth;}
    if(final.score<threshold||isnan(zz)){out.z=zz;out.score=final.score;out.final_depth=depth;out.rejection=5;return;}
    double xf,yf;
    if(final.ties==1){if(!final.neighbors){out.rejection=2;return;}
        xf=s->x-(cw+tw)/2.0+final.x+real_gaussian(final.left,final.score,final.right,error);
        yf=s->y-(ch+th)/2.0+final.y+real_gaussian(final.above,final.score,final.below,error);
        if(isnan(xf)||isnan(yf)){xf=s->x-(cw+tw)/2.0+final.sum_x/2.0;yf=s->y-(ch+th)/2.0+final.sum_y/2.0;}}
    else if(final.ties==2){xf=s->x-(cw+tw)/2.0+final.sum_x/2.0;yf=s->y-(ch+th)/2.0+final.sum_y/2.0;}
    else{xf=s->x-(cw+tw)/2.0+final.x;yf=s->y-(ch+th)/2.0+final.y;}
    if(isfinite(zz)&&final.score>threshold){s->mask_depth=int(round(zz));s->suppress=1;if(s->mask_depth<1||s->mask_depth>depths){*error=5;s->suppress=0;}}
    out.x=xf;out.y=yf;out.z=zz;out.score=final.score;out.final_depth=depth;
    if(isnan(xf)||isnan(yf)||isnan(zz)||final.score<threshold){out.rejection=5;return;}
    if(interpolate&&(zz==1||zz==depths)){out.rejection=6;return;}out.rejection=0;
}
__global__ void resident_remove(double* image,uint8_t* ghost,const ResidentState* s,const Point* areas,const int* offsets,
                                  int width,int height,int* error){
    if(!s->suppress)return;int i=blockIdx.x*blockDim.x+threadIdx.x,depth=s->mask_depth;
    int begin=offsets[depth-1],end=offsets[depth];if(begin+i>=end)return;auto p=areas[begin+i];
    int x=int(round(s->x+p.x))-1,y=int(round(s->y+p.y))-1;
    if(x<0||y<0||x>=width||y>=height){atomicExch(error,6);return;}
    image[size_t(y)*width+x]=s->background;ghost[size_t(y)*width+x]=1;
}
__global__ void resident_finish(const ResidentState* s,ResidentOutput* outputs,int* current,int selected){
    auto& out=outputs[s->source];out.overlap=s->overlap;out.full_calls=s->overlap?selected+1:1;++*current;
}

class Resident final:public ResidentEngine {
    friend class ConcurrentResident;
    friend void check_mixed_private(const Model&,const Image&,int);
    const Model& model_;int tw_,th_,cw_,ch_,fw_,fh_,fftw_,ffth_,real_,spectrum_,depths_,selected_,stride_,max_area_=0,capacity_=0;
    int width_,height_;double guess_,final_;bool interpolate_,bounded_,mixed_,fused_;uint64_t clamped_count_=0;
    double* frame_ptr_=nullptr;uint8_t* ghost_ptr_=nullptr;
    Stream stream_;Plan forward_,inverse_one_,inverse_selected_;
    Buffer<double> input_,frame_,column_,padded_,inverse_,maps_,sums_,norms_,maximum_;
    SharedBuffer<double> mask_,pixels_,norms_t_,residuals_t_,similarity_;
    Buffer<cufftDoubleComplex> image_fft_,products_;SharedBuffer<cufftDoubleComplex> spectra_;
    Buffer<Moment> prefix_;Buffer<uint8_t> workspace_,ghost_;size_t workspace_bytes_=0;
    Buffer<unsigned> fused_counter_;
    Buffer<int> indices_,start_,current_,error_,schedule_;SharedBuffer<int> selected_indices_,area_offsets_;
    SharedBuffer<Point> areas_;Buffer<PeakWire> wires_;Buffer<WalkState> walk_;Buffer<WalkSample> samples_;
    Buffer<ResidentState> state_;Buffer<CoarseCandidate> coarse_;Buffer<ResidentOutput> outputs_;Buffer<ResidentTrace> traces_;
    Pinned<ResidentOutput> host_outputs_;Pinned<ResidentTrace> host_traces_;Graph graph_,exact_graph_;
    cudaGraphNode_t add(cudaGraph_t graph,void* fn,void** args,dim3 grid,dim3 block,cudaGraphNode_t parent){
        cudaKernelNodeParams p{};p.func=fn;p.kernelParams=args;p.gridDim=grid;p.blockDim=block;
        cudaGraphNode_t n;check(cudaGraphAddKernelNode(&n,graph,parent?&parent:nullptr,parent?1:0,&p));return n;
    }
    template<class F> cudaGraphNode_t capture(cudaGraph_t graph,cudaGraphNode_t parent,F callback){
        cudaGraph_t child=nullptr;check(cudaStreamBeginCapture(stream_.value,cudaStreamCaptureModeGlobal));
        callback();check(cudaStreamEndCapture(stream_.value,&child));cudaGraphNode_t n;
        auto status=cudaGraphAddChildGraphNode(&n,graph,parent?&parent:nullptr,parent?1:0,child);cudaGraphDestroy(child);check(status);return n;
    }
    void full(int count,const int* indices){
        multiply_batch<<<(spectrum_*count+255)/256,256,0,stream_.value>>>(image_fft_.p,spectra_.p,products_.p,indices,spectrum_,count);
        fftcheck(cufftExecZ2D(count==1?inverse_one_.value:inverse_selected_.value,products_.p,inverse_.p));
        normalize_batch<<<(fw_*fh_*count+255)/256,256,0,stream_.value>>>(inverse_.p,maps_.p,sums_.p,norms_.p,maximum_.p,indices,
            norms_t_.p,residuals_t_.p,fw_,fh_,tw_,th_,fftw_,ffth_,count);
        reduce_full_batch<<<count,256,0,stream_.value>>>(maps_.p,wires_.p,fw_,fh_);
    }
    void prefix(){fused_prefix<<<1,256,0,stream_.value>>>(input_.p,prefix_.p,cw_,ch_,maximum_.p);}
    void build_graph(Graph& graph,bool bounded){
        cudaGraphConditionalHandle overlap,active,walking=0;
        check(cudaGraphCreate(&graph.graph,0));
        for(auto* h:{&overlap,&active})check(cudaGraphConditionalHandleCreate(h,graph.graph,0,cudaGraphCondAssignDefault));
        if(!mixed_)check(cudaGraphConditionalHandleCreate(&walking,graph.graph,0,cudaGraphCondAssignDefault));
        void* begin_args[]={&state_.p,&outputs_.p,&current_.p,&coarse_.p,&indices_.p,&start_.p,&walk_.p,&schedule_.p};
        auto begin=add(graph.graph,(void*)resident_begin,begin_args,1,1,nullptr);
        void* crop_args[]={&frame_ptr_,&ghost_ptr_,&input_.p,&mask_.p,&state_.p,&coarse_.p,&width_,&height_,&cw_,&ch_,&tw_,&th_,&final_,&error_.p,&overlap};
        auto crop=add(graph.graph,(void*)resident_crop,crop_args,1,256,begin);
        auto prepare_full=[&]{fused_full_prepare<<<(std::max(fw_*fh_,real_)+255)/256,256,0,stream_.value>>>(
                input_.p,prefix_.p,padded_.p,sums_.p,norms_.p,maximum_.p,cw_,ch_,tw_,th_,fftw_,ffth_,error_.p);
            fftcheck(cufftExecD2Z(forward_.value,padded_.p,image_fft_.p));};
        auto preparation=capture(graph.graph,crop,[&]{prefix();if(!bounded)prepare_full();});
        auto conditional=[&](cudaGraph_t graph,cudaGraphNode_t parent,cudaGraphConditionalHandle handle,cudaGraphConditionalNodeType type){
            cudaGraphNodeParams p{};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=handle;p.conditional.type=type;p.conditional.size=1;
            cudaGraphNode_t n;check(cudaGraphAddNode(&n,graph,parent?&parent:nullptr,nullptr,parent?1:0,&p));return std::make_pair(n,p.conditional.phGraph_out[0]);};
        auto overlap_node=conditional(graph.graph,preparation,overlap,cudaGraphCondTypeIf);
        auto overlap_corr=capture(overlap_node.second,nullptr,[&]{if(bounded)prepare_full();full(selected_,selected_indices_.p);});
        void* overlap_args[]={&state_.p,&wires_.p,&selected_indices_.p,&selected_,&similarity_.p,&depths_,&indices_.p,&traces_.p,&stride_};
        add(overlap_node.second,(void*)resident_overlap,overlap_args,1,1,overlap_corr);
        auto first=capture(graph.graph,overlap_node.first,[&]{
            if(bounded){int columns=cw_-tw_+3,rows=ch_-th_+3;
                initial_spatial_bounded<<<columns*rows,256,0,stream_.value>>>(input_.p,pixels_.p,prefix_.p,maps_.p,indices_.p,norms_t_.p,residuals_t_.p,cw_,ch_,tw_,th_);
                reduce_initial_bounded<<<1,128,0,stream_.value>>>(maps_.p,wires_.p,columns,rows,tw_,th_,fh_);
            }else full(1,indices_.p);});
        void* initial_args[]={&state_.p,&outputs_.p,&wires_.p,&start_.p,&width_,&height_,&cw_,&ch_,&tw_,&th_,&guess_,&bounded,&error_.p,&traces_.p,&stride_,&active};
        auto initial=add(graph.graph,(void*)resident_initial,initial_args,1,1,first);
        auto refine=conditional(graph.graph,initial,active,cudaGraphCondTypeIf);
        void* recenter_args[]={&frame_ptr_,&input_.p,&mask_.p,&state_.p,&width_,&cw_,&ch_,&tw_};
        auto recenter=add(refine.second,(void*)resident_recenter,recenter_args,(cw_*ch_+255)/256,256,nullptr);
        auto statistics=capture(refine.second,recenter,[&]{prefix();});
        cudaGraphNode_t depth_complete;
        if(mixed_){depth_complete=capture(refine.second,statistics,[&]{
            mixed_depth_walk<<<1,256,0,stream_.value>>>(input_.p,pixels_.p,prefix_.p,maps_.p,indices_.p,norms_t_.p,residuals_t_.p,
                cw_,ch_,tw_,th_,walk_.p,samples_.p,start_.p,depths_,nullptr);
        });}else{
        void* walk_args[]={&walk_.p,&start_.p,&indices_.p,&depths_};
        auto walk_init=add(refine.second,(void*)walk_initialize,walk_args,1,1,statistics);
        int columns=cw_-tw_+1,rows=ch_-th_+1;
        void* spatial_args[]={&input_.p,&pixels_.p,&prefix_.p,&maps_.p,&indices_.p,&norms_t_.p,&residuals_t_.p,&cw_,&ch_,&tw_,&th_};
        auto spatial=add(refine.second,(void*)spatial_batch,spatial_args,dim3(columns*rows,3),256,walk_init);
        void* reduce_args[]={&maps_.p,&wires_.p,&columns,&rows,&tw_,&th_,&fw_,&fh_};
        auto reduction=add(refine.second,(void*)reduce_compact_batch,reduce_args,3,128,spatial);
        void* decision_args[]={&walk_.p,&wires_.p,&indices_.p,&samples_.p,&depths_,&walking};
        void* fused_seed_args[]={&walk_.p,&wires_.p,&indices_.p,&samples_.p,&depths_,&fused_counter_.p,&walking};
        auto decision=add(refine.second,fused_?(void*)fused_depth_seed:(void*)walk_initial_peaks,fused_?fused_seed_args:decision_args,1,1,reduction);
        auto loop=conditional(refine.second,decision,walking,cudaGraphCondTypeWhile);
        if(fused_){void* step_args[]={&input_.p,&pixels_.p,&prefix_.p,&maps_.p,&indices_.p,&norms_t_.p,&residuals_t_.p,&cw_,&ch_,&tw_,&th_,&walk_.p,&samples_.p,&depths_,&fused_counter_.p,&walking};
            add(loop.second,(void*)fused_depth_step,step_args,columns*rows,256,nullptr);
        }else{auto step=add(loop.second,(void*)spatial_batch,spatial_args,dim3(columns*rows,1),256,nullptr);
            auto peak=add(loop.second,(void*)reduce_compact_batch,reduce_args,1,128,step);
            add(loop.second,(void*)walk_advance,decision_args,1,1,peak);}
        depth_complete=loop.first;
        }
        void* final_args[]={&state_.p,&outputs_.p,&walk_.p,&samples_.p,&depths_,&cw_,&ch_,&tw_,&th_,&interpolate_,&final_,&error_.p,&traces_.p,&stride_};
        auto final=add(refine.second,(void*)resident_final,final_args,1,1,depth_complete);
        void* remove_args[]={&frame_ptr_,&ghost_ptr_,&state_.p,&areas_.p,&area_offsets_.p,&width_,&height_,&error_.p};
        add(refine.second,(void*)resident_remove,remove_args,(max_area_+255)/256,256,final);
        void* finish_args[]={&state_.p,&outputs_.p,&current_.p,&selected_};
        add(graph.graph,(void*)resident_finish,finish_args,1,1,refine.first);
        check(cudaGraphInstantiate(&graph.executable,graph.graph,0));check(cudaGraphUpload(graph.executable,stream_.value));check(cudaStreamSynchronize(stream_.value));
    }
public:
    explicit Resident(const Model& m,double* shared_frame=nullptr,uint8_t* shared_ghost=nullptr,bool bounded=false,bool mixed=false,bool fused=false,const Resident* calibration=nullptr):model_(m),tw_(m.templates[0].width),th_(m.templates[0].height),cw_(tw_+8),ch_(th_+8),
        fw_(cw_+tw_-1),fh_(ch_+th_-1),fftw_(cv::getOptimalDFTSize(fw_)),ffth_(cv::getOptimalDFTSize(fh_)),
        real_(fftw_*ffth_),spectrum_(ffth_*(fftw_/2+1)),depths_(int(m.templates.size())),selected_(int(m.selected.size())),stride_(depths_+selected_+5),width_(m.frame_width),height_(m.frame_height),guess_(m.cm_guess),final_(m.cm_final),interpolate_(m.interpolate),bounded_(bounded),mixed_(mixed),fused_(fused){
        if(std::any_of(m.global_mask.pixels.begin(),m.global_mask.pixels.end(),[](double v){return v<0||!std::isfinite(v);}))throw CudaFailure("Resident prototype requires a nonnegative mask");
        size_t frame_count=size_t(m.frame_width)*m.frame_height;
        if(!shared_frame){frame_.allocate(frame_count);column_.allocate(frame_count);ghost_.allocate(frame_count);}
        frame_ptr_=shared_frame?shared_frame:frame_.p;ghost_ptr_=shared_ghost?shared_ghost:ghost_.p;input_.allocate(cw_*ch_);
        image_fft_.allocate(spectrum_);
        products_.allocate(size_t(std::max(3,selected_))*spectrum_);inverse_.allocate(size_t(selected_)*real_);maps_.allocate(size_t(std::max(3,selected_))*fw_*fh_);
        padded_.allocate(real_);sums_.allocate(fw_*fh_);norms_.allocate(fw_*fh_);maximum_.allocate(1);prefix_.allocate((cw_+1)*(ch_+1));
        
        indices_.allocate(std::max(3,selected_));start_.allocate(1);current_.allocate(1);error_.allocate(1);
        fused_counter_.allocate(1);state_.allocate(1);walk_.allocate(1);samples_.allocate(depths_+3);wires_.allocate(std::max(3,selected_));
        capacity_=4096;schedule_.allocate(capacity_);std::vector<int> schedule(capacity_);for(int i=0;i<capacity_;++i)schedule[i]=i;check(cudaMemcpy(schedule_.p,schedule.data(),size_t(capacity_)*sizeof(int),cudaMemcpyHostToDevice));coarse_.allocate(capacity_);outputs_.allocate(capacity_);traces_.allocate(size_t(capacity_)*stride_);
        host_outputs_.allocate(capacity_);host_traces_.allocate(size_t(capacity_)*stride_);
        check(cub::DeviceReduce::Max(nullptr,workspace_bytes_,norms_.p,maximum_.p,fw_*fh_,stream_.value));workspace_.allocate(workspace_bytes_);
        std::vector<double> packed; // Empty for clones: no duplicate CPU packing or template FFT.
        if(calibration){
            if(&m!=&calibration->model_||tw_!=calibration->tw_||th_!=calibration->th_||
               depths_!=calibration->depths_||selected_!=calibration->selected_||spectrum_!=calibration->spectrum_)
                throw std::invalid_argument("Resident calibration owner/model dimensions disagree");
            mask_.borrow(calibration->mask_);pixels_.borrow(calibration->pixels_);spectra_.borrow(calibration->spectra_);
            norms_t_.borrow(calibration->norms_t_);residuals_t_.borrow(calibration->residuals_t_);similarity_.borrow(calibration->similarity_);
            selected_indices_.borrow(calibration->selected_indices_);areas_.borrow(calibration->areas_);area_offsets_.borrow(calibration->area_offsets_);
            max_area_=calibration->max_area_;
        }else{
            mask_.allocate(tw_*th_);pixels_.allocate(size_t(depths_)*tw_*th_);spectra_.allocate(size_t(depths_)*spectrum_);
            norms_t_.allocate(depths_);residuals_t_.allocate(depths_);similarity_.allocate(size_t(depths_)*depths_);
            selected_indices_.allocate(selected_);
            packed.assign(size_t(depths_)*real_,0);
            std::vector<double> pixels(size_t(depths_)*tw_*th_),norms,residuals;
            for(int k=0;k<depths_;++k){double mean=0,sum=0,square=0;
                for(int x=0;x<tw_;++x)for(int y=0;y<th_;++y)mean+=m.templates[k].at(x,y);mean/=tw_*th_;
                for(int x=0;x<tw_;++x)for(int y=0;y<th_;++y){double v=m.templates[k].at(x,y)-mean;packed[size_t(k)*real_+y*fftw_+x]=v;pixels[size_t(k)*tw_*th_+y*tw_+x]=v;sum+=v;square+=v*v;}
                residuals.push_back(sum/(tw_*th_));norms.push_back(std::sqrt(square-sum*sum/(tw_*th_)));}
            check(cudaMemcpy(pixels_.p,pixels.data(),pixels.size()*8,cudaMemcpyHostToDevice));
            check(cudaMemcpy(mask_.p,m.global_mask.pixels.data(),size_t(tw_)*th_*8,cudaMemcpyHostToDevice));
            check(cudaMemcpy(norms_t_.p,norms.data(),size_t(depths_)*8,cudaMemcpyHostToDevice));check(cudaMemcpy(residuals_t_.p,residuals.data(),size_t(depths_)*8,cudaMemcpyHostToDevice));
            check(cudaMemcpy(similarity_.p,m.similarity.pixels.data(),size_t(depths_)*depths_*8,cudaMemcpyHostToDevice));
            check(cudaMemcpy(selected_indices_.p,m.selected.data(),size_t(selected_)*sizeof(int),cudaMemcpyHostToDevice));
            std::vector<Point> points;std::vector<int> offsets{0};for(const auto& area:m.areas){points.insert(points.end(),area.begin(),area.end());offsets.push_back(int(points.size()));max_area_=std::max(max_area_,int(area.size()));}
            max_area_=std::max(1,max_area_);areas_.allocate(std::max(size_t(1),points.size()));area_offsets_.allocate(offsets.size());
            if(!points.empty())check(cudaMemcpy(areas_.p,points.data(),points.size()*sizeof(Point),cudaMemcpyHostToDevice));check(cudaMemcpy(area_offsets_.p,offsets.data(),offsets.size()*sizeof(int),cudaMemcpyHostToDevice));
        }
        fftcheck(cufftPlan2d(&forward_.value,ffth_,fftw_,CUFFT_D2Z));int dims[2]={ffth_,fftw_};
        fftcheck(cufftPlanMany(&inverse_one_.value,2,dims,nullptr,1,spectrum_,nullptr,1,real_,CUFFT_Z2D,1));
        fftcheck(cufftPlanMany(&inverse_selected_.value,2,dims,nullptr,1,spectrum_,nullptr,1,real_,CUFFT_Z2D,selected_));
        for(auto p:{forward_.value,inverse_one_.value,inverse_selected_.value})fftcheck(cufftSetStream(p,stream_.value));
        if(!calibration){
            Buffer<double> temp;temp.allocate(packed.size());check(cudaMemcpy(temp.p,packed.data(),packed.size()*8,cudaMemcpyHostToDevice));Plan templates;
            fftcheck(cufftPlanMany(&templates.value,2,dims,nullptr,1,real_,nullptr,1,spectrum_,CUFFT_D2Z,depths_));fftcheck(cufftSetStream(templates.value,stream_.value));
            fftcheck(cufftExecD2Z(templates.value,temp.p,spectra_.p));
            check(cudaStreamSynchronize(stream_.value)); // Publish complete calibration before clones borrow it.
        }
        check(cudaMemsetAsync(padded_.p,0,size_t(real_)*8,stream_.value));check(cudaMemsetAsync(products_.p,0,size_t(std::max(3,selected_))*spectrum_*sizeof(cufftDoubleComplex),stream_.value));
        fftcheck(cufftExecD2Z(forward_.value,padded_.p,image_fft_.p));fftcheck(cufftExecZ2D(inverse_one_.value,products_.p,inverse_.p));fftcheck(cufftExecZ2D(inverse_selected_.value,products_.p,inverse_.p));check(cudaStreamSynchronize(stream_.value));
        build_graph(graph_,bounded_);if(bounded_)build_graph(exact_graph_,false);
    }
    ~Resident()override{cudaStreamSynchronize(stream_.value);}
    RefineResult process(const double* image,const float* device_image,size_t step,const std::vector<CoarseCandidate>& coarse,bool diagnostics){
        if(coarse.size()>size_t(capacity_))throw CudaFailure("Resident prototype candidate capacity exceeded");
        auto begun=std::chrono::steady_clock::now();size_t count=size_t(model_.frame_width)*model_.frame_height;
        check(cudaMemsetAsync(error_.p,0,sizeof(int),stream_.value));check(cudaMemsetAsync(current_.p,0,sizeof(int),stream_.value));check(cudaMemsetAsync(ghost_ptr_,0,count,stream_.value));
        if(device_image)copy_device_float<<<(count+255)/256,256,0,stream_.value>>>(device_image,step,frame_ptr_,model_.frame_width,model_.frame_height,error_.p);
        else{check(cudaMemcpyAsync(column_.p,image,count*8,cudaMemcpyHostToDevice,stream_.value));
            transpose_integral<<<(count+255)/256,256,0,stream_.value>>>(column_.p,frame_ptr_,model_.frame_width,model_.frame_height,error_.p);}
        if(!coarse.empty())check(cudaMemcpyAsync(coarse_.p,coarse.data(),coarse.size()*sizeof(CoarseCandidate),cudaMemcpyHostToDevice,stream_.value));
        for(size_t i=0;i<coarse.size();++i)check(cudaGraphLaunch(bounded_&&coarse[i].score<final_?exact_graph_.executable:graph_.executable,stream_.value));
        if(!coarse.empty())check(cudaMemcpyAsync(host_outputs_.p,outputs_.p,coarse.size()*sizeof(ResidentOutput),cudaMemcpyDeviceToHost,stream_.value));
        if(diagnostics&&!coarse.empty())check(cudaMemcpyAsync(host_traces_.p,traces_.p,coarse.size()*stride_*sizeof(ResidentTrace),cudaMemcpyDeviceToHost,stream_.value));
        int error=0;check(cudaMemcpyAsync(&error,error_.p,sizeof(int),cudaMemcpyDeviceToHost,stream_.value));check(cudaStreamSynchronize(stream_.value));
        if(error)throw CudaFailure("Resident unsupported input or device validation failure "+std::to_string(error));
        RefineResult result;clamped_count_=0;
        for(size_t i=0;i<coarse.size();++i){const auto& p=host_outputs_.p[i];clamped_count_+=p.clamped_xy;
            result.particles.push_back({p.source,p.initial_depth,p.final_depth,p.rejection,p.full_calls,p.legacy_calls,p.overlap,p.x,p.y,p.z,p.score});
            if(!p.rejection)result.localizations.push_back({p.x,p.y,model_.interpolate?(p.z-1)/(depths_-1):p.z,p.score,p.source});
            if(diagnostics)for(int k=0;k<p.full_calls+p.legacy_calls;++k){const auto& t=host_traces_.p[i*stride_+k];result.correlations.push_back({p.source,t.index,t.mode,unpack(t.peak)});}
        }
        result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begun).count();
        // Suppressed image is diagnostic output and is excluded from core timing, as with lastImage().
        if(diagnostics){result.image=Image(model_.frame_width,model_.frame_height);check(cudaMemcpy(result.image.pixels.data(),frame_ptr_,count*8,cudaMemcpyDeviceToHost));}
        return result;
    }
    RefineResult refine(const double* image,const std::vector<CoarseCandidate>& coarse,bool d)override{return process(image,nullptr,0,coarse,d);}
    RefineResult refine_device(const float* image,size_t step,const std::vector<CoarseCandidate>& coarse,bool d)override{return process(nullptr,image,step,coarse,d);}
    uint64_t clamped_xy()const override{return clamped_count_;}
    double warmup(int milliseconds)override{
        auto started=std::chrono::steady_clock::now();double elapsed=0;
        do{warmup_math<<<(real_+255)/256,256,0,stream_.value>>>(padded_.p,real_);check(cudaGetLastError());check(cudaStreamSynchronize(stream_.value));
            elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();}while(elapsed<milliseconds*.001);
        return elapsed;
    }
    Image last_image()override{Image result(width_,height_);check(cudaMemcpy(result.pixels.data(),frame_ptr_,size_t(width_)*height_*8,cudaMemcpyDeviceToHost));return result;}
};
void check_mixed_private(const Model& m,const Image& image,int depth){
    auto reference_backend=make_cuda_correlation(m);reference_backend->begin_crop(image);
    auto reference=reference_backend->CorrelationBackend::depth_walk(image,depth,int(m.templates.size()));reference_backend->end_crop();
    Resident gpu(m,nullptr,nullptr,false,true);check(cudaMemcpyAsync(gpu.input_.p,image.pixels.data(),image.pixels.size()*8,cudaMemcpyHostToDevice,gpu.stream_.value));
    check(cudaMemcpyAsync(gpu.start_.p,&depth,sizeof(int),cudaMemcpyHostToDevice,gpu.stream_.value));gpu.prefix();
    Buffer<uint64_t> dots;dots.allocate(1);check(cudaMemsetAsync(dots.p,0,8,gpu.stream_.value));
    mixed_depth_walk<<<1,256,0,gpu.stream_.value>>>(gpu.input_.p,gpu.pixels_.p,gpu.prefix_.p,gpu.maps_.p,gpu.indices_.p,gpu.norms_t_.p,gpu.residuals_t_.p,
        gpu.cw_,gpu.ch_,gpu.tw_,gpu.th_,gpu.walk_.p,gpu.samples_.p,gpu.start_.p,gpu.depths_,dots.p);
    WalkState actual{};std::vector<WalkSample> samples(m.templates.size()+3);uint64_t count=0;
    check(cudaMemcpyAsync(&actual,gpu.walk_.p,sizeof(actual),cudaMemcpyDeviceToHost,gpu.stream_.value));
    check(cudaMemcpyAsync(samples.data(),gpu.samples_.p,samples.size()*sizeof(WalkSample),cudaMemcpyDeviceToHost,gpu.stream_.value));
    check(cudaMemcpyAsync(&count,dots.p,8,cudaMemcpyDeviceToHost,gpu.stream_.value));check(cudaStreamSynchronize(gpu.stream_.value));
    if(actual.error||actual.count!=int(reference.samples.size())||actual.best!=reference.best||bool(actual.boundary)!=reference.boundary)
        throw std::runtime_error("Mixed GPU depth path or stopping decision changed");
    auto equal=[](const PeakWire& a,const Peak& b){return a.x==b.x&&a.y==b.y&&a.ties==b.ties&&a.sum_x==b.sum_x&&a.sum_y==b.sum_y&&bool(a.neighbors)==b.has_neighbors&&
        std::abs(a.score-b.score)<1e-12&&std::abs(a.left-b.left)<1e-12&&std::abs(a.right-b.right)<1e-12&&std::abs(a.above-b.above)<1e-12&&std::abs(a.below-b.below)<1e-12;};
    for(int k=0;k<3;++k)if(actual.indices[k]!=reference.indices[k]||!equal(actual.peaks[k],reference.peaks[k]))throw std::runtime_error("Mixed depth final window/peak/tie changed");
    for(int k=0;k<actual.count;++k)if(samples[k].index!=reference.samples[k].template_index||!equal(samples[k].peak,reference.samples[k].peak))throw std::runtime_error("Mixed depth trace/neighbor changed");
    if(count>uint64_t(actual.count)*85)throw std::runtime_error("Mixed depth dot count exceeded full-grid bound");
}
struct Fence {
    cudaEvent_t value=nullptr;Fence(){check(cudaEventCreateWithFlags(&value,cudaEventDisableTiming));}
    ~Fence(){cudaEventDestroy(value);}
};
class ConcurrentResident final:public ResidentEngine {
    const Model& model_;std::vector<std::unique_ptr<Resident>> workers_;
    std::vector<std::unique_ptr<Fence>> fences_;Fence ready_;Pinned<int> host_indices_;
    double rx_,ry_,wide_rx_,wide_ry_;bool bounded_;uint64_t clamped_count_=0;
    bool pending_=false,pending_diagnostics_=false;
    int pending_layers_=0;
    std::chrono::steady_clock::time_point pending_started_{};
    std::vector<CoarseCandidate> pending_coarse_;
    std::vector<int> pending_owner_;
    std::vector<std::vector<int>> pending_schedules_;
    Pinned<int> pending_errors_,pending_schedule_staging_;
    Pinned<CoarseCandidate> pending_coarse_staging_;
    CUcontext cuda_context_=nullptr;
    std::future<void> pending_enqueue_;
    std::vector<std::vector<int>> pending_parents_;
    std::vector<uint8_t> pending_needs_event_;
    void join_enqueue(){
        if(!pending_enqueue_.valid())return;
        try{pending_enqueue_.get();}
        catch(const CudaFailure&){throw;}
        catch(const std::exception& e){throw CudaFailure(std::string("Background submission failed: ")+e.what());}
    }
    void join_enqueue_noexcept()noexcept{
        if(pending_enqueue_.valid())try{pending_enqueue_.get();}catch(...){}
    }
    void enqueue_particles(){
        const int n=int(pending_coarse_.size()),nw=int(workers_.size());
        const auto& coarse=pending_coarse_;const auto& owner=pending_owner_;
        const auto& parents=pending_parents_;const auto& needs_event=pending_needs_event_;
        const bool diagnostics=pending_diagnostics_;
        for(int i=0;i<n;++i){int k=owner[i];auto& w=*workers_[k];
            for(int j:parents[i])if(owner[j]!=k)check(cudaStreamWaitEvent(w.stream_.value,fences_[j]->value,0));
            check(cudaGraphLaunch(bounded_&&coarse[i].score<model_.cm_final?w.exact_graph_.executable:w.graph_.executable,w.stream_.value));
            if(needs_event[i])check(cudaEventRecord(fences_[i]->value,w.stream_.value));
        }
        for(int k=0;k<nw;++k){auto& w=*workers_[k];
            if(n)check(cudaMemcpyAsync(w.host_outputs_.p,w.outputs_.p,size_t(n)*sizeof(ResidentOutput),cudaMemcpyDeviceToHost,w.stream_.value));
            if(n&&diagnostics)check(cudaMemcpyAsync(w.host_traces_.p,w.traces_.p,size_t(n)*w.stride_*sizeof(ResidentTrace),cudaMemcpyDeviceToHost,w.stream_.value));
            check(cudaMemcpyAsync(pending_errors_.p+k,w.error_.p,sizeof(int),cudaMemcpyDeviceToHost,w.stream_.value));}
    }
    void drain_pending()noexcept{
        join_enqueue_noexcept();
        for(auto& worker:workers_)if(worker)cudaStreamSynchronize(worker->stream_.value);
        pending_=false;
    }

    void destroy_workers()noexcept{
        join_enqueue_noexcept();
        for(auto& worker:workers_)if(worker)cudaStreamSynchronize(worker->stream_.value);
        // Destroy all borrower graphs/plans before the primary calibration owner.
        while(!workers_.empty())workers_.pop_back();
    }
public:
    ConcurrentResident(const Model& model,int count,bool bounded,bool mixed,bool fused):model_(model),bounded_(bounded){
        if(count<1||count>16)throw std::invalid_argument("Resident workers must be1..16");
        try{
        workers_.push_back(std::make_unique<Resident>(model,nullptr,nullptr,bounded,mixed,fused));
        for(int k=1;k<count;++k)workers_.push_back(std::make_unique<Resident>(model,workers_[0]->frame_ptr_,workers_[0]->ghost_ptr_,bounded,mixed,fused,workers_[0].get()));
        for(int k=0;k<workers_[0]->capacity_;++k)fences_.push_back(std::make_unique<Fence>());
        host_indices_.allocate(workers_[0]->capacity_);for(int i=0;i<workers_[0]->capacity_;++i)host_indices_.p[i]=i;
        pending_errors_.allocate(count);
        pending_coarse_staging_.allocate(workers_[0]->capacity_);
        pending_schedule_staging_.allocate(size_t(count)*workers_[0]->capacity_);
        check_background_driver(cuCtxGetCurrent(&cuda_context_),"cuCtxGetCurrent resident initialization");
        if(!cuda_context_)throw CudaFailure("Resident initialization has no CUDA context");
        auto& w=*workers_[0];double ax=0,ay=0;for(const auto& a:model.areas)for(auto p:a){ax=std::max(ax,std::abs(p.x));ay=std::max(ay,std::abs(p.y));}
        // Positive-neighbor Gaussian interpolation stays within half a pixel of its maximum.
        // This envelope covers every full-map initial shift, recentered input crop, and removal area.
        double sx=bounded?4.5:(w.cw_+w.tw_)/2.0,sy=bounded?4.5:(w.ch_+w.th_)/2.0;
        rx_=std::max(sx+w.cw_/2.0+2,sx+ax+2);ry_=std::max(sy+w.ch_/2.0+2,sy+ay+2);
        wide_rx_=std::max((w.cw_+w.tw_)/2.0+w.cw_/2.0+2,(w.cw_+w.tw_)/2.0+ax+2);
        wide_ry_=std::max((w.ch_+w.th_)/2.0+w.ch_/2.0+2,(w.ch_+w.th_)/2.0+ay+2);
        }catch(...){destroy_workers();throw;}
    }
    ~ConcurrentResident()override{destroy_workers();}
    void enqueue_frame(const double* image,const float* device_image,size_t step,const std::vector<CoarseCandidate>& input_coarse,bool diagnostics){
        if(pending_)throw std::invalid_argument("Complete the pending resident frame before another submission");
        if(input_coarse.size()>size_t(workers_[0]->capacity_))throw CudaFailure("Resident candidate capacity exceeded");
        pending_started_=std::chrono::steady_clock::now();
        pending_coarse_=input_coarse;pending_diagnostics_=diagnostics;pending_=true;
        const auto& coarse=pending_coarse_;
        try{

        CUcontext current_context=nullptr;
        check_background_driver(cuCtxGetCurrent(&current_context),"cuCtxGetCurrent resident submission");
        if(current_context!=cuda_context_)throw CudaFailure("Resident CUDA context changed after initialization");
        int n=int(coarse.size()),nw=int(workers_.size());
        std::vector<int> levels(n,0);auto& owner=pending_owner_;owner.assign(n,0);std::vector<std::vector<int>> parents(n);std::vector<double> radius_x(n),radius_y(n);int layers=0;
        for(int i=0;i<n;++i){bool wide=bounded_&&coarse[i].score<model_.cm_final;radius_x[i]=wide?wide_rx_:rx_;radius_y[i]=wide?wide_ry_:ry_;
            for(int j=0;j<i;++j)if(std::abs(coarse[i].x-coarse[j].x)<=radius_x[i]+radius_x[j]&&std::abs(coarse[i].y-coarse[j].y)<=radius_y[i]+radius_y[j])
                {levels[i]=std::max(levels[i],levels[j]+1);parents[i].push_back(j);}layers=std::max(layers,levels[i]+1);}
        // Remove transitive dependencies: waiting for a descendant already waits for its ancestors.
        int words=(n+63)/64;std::vector<std::vector<uint64_t>> ancestors(n,std::vector<uint64_t>(words,0));
        for(int i=0;i<n;++i){std::vector<int> reduced;auto& covered=ancestors[i];
            for(auto it=parents[i].rbegin();it!=parents[i].rend();++it){int j=*it;if((covered[j/64]>>(j%64))&1)continue;
                reduced.push_back(j);covered[j/64]|=uint64_t(1)<<(j%64);for(int word=0;word<words;++word)covered[word]|=ancestors[j][word];}
            std::reverse(reduced.begin(),reduced.end());parents[i]=std::move(reduced);}
        auto& schedules=pending_schedules_;schedules.assign(nw,{});std::vector<double> available(nw,0),finish(n,0);
        for(int i=0;i<n;++i){double prerequisite=0;for(int j:parents[i])prerequisite=std::max(prerequisite,finish[j]);
            int critical=-1;for(int j:parents[i])if(critical<0||finish[j]>finish[critical])critical=j;
            int best=critical<0?i%nw:owner[critical];double ready=std::max(available[best],prerequisite);
            for(int offset=1;offset<nw;++offset){int k=(i+offset)%nw;double candidate=std::max(available[k],prerequisite);
                if(candidate<ready){best=k;ready=candidate;}}
            owner[i]=best;schedules[best].push_back(i);finish[i]=ready+(coarse[i].score<model_.cm_final?3:1);available[best]=finish[i];}
        std::vector<uint8_t> needs_event(n,0);for(int i=0;i<n;++i)for(int j:parents[i])if(owner[j]!=owner[i])needs_event[j]=1;
        pending_layers_=layers;
        size_t count=size_t(model_.frame_width)*model_.frame_height;auto& primary=*workers_[0];
        if(n)std::copy(coarse.begin(),coarse.end(),pending_coarse_staging_.p);
        for(int k=0;k<nw;++k){auto& w=*workers_[k];check(cudaMemsetAsync(w.error_.p,0,sizeof(int),w.stream_.value));
            check(cudaMemsetAsync(w.current_.p,0,sizeof(int),w.stream_.value));
            if(!schedules[k].empty()){
                int* staging=pending_schedule_staging_.p+size_t(k)*primary.capacity_;
                std::copy(schedules[k].begin(),schedules[k].end(),staging);
                check(cudaMemcpyAsync(w.schedule_.p,staging,schedules[k].size()*sizeof(int),cudaMemcpyHostToDevice,w.stream_.value));
            }
            if(n)check(cudaMemcpyAsync(w.coarse_.p,pending_coarse_staging_.p,size_t(n)*sizeof(CoarseCandidate),cudaMemcpyHostToDevice,w.stream_.value));}
        check(cudaMemsetAsync(primary.ghost_ptr_,0,count,primary.stream_.value));
        if(device_image)copy_device_float<<<(count+255)/256,256,0,primary.stream_.value>>>(device_image,step,primary.frame_ptr_,model_.frame_width,model_.frame_height,primary.error_.p);
        else{check(cudaMemcpyAsync(primary.column_.p,image,count*8,cudaMemcpyHostToDevice,primary.stream_.value));
            transpose_integral<<<(count+255)/256,256,0,primary.stream_.value>>>(primary.column_.p,primary.frame_ptr_,model_.frame_width,model_.frame_height,primary.error_.p);}
        check(cudaEventRecord(ready_.value,primary.stream_.value));
        for(int k=1;k<nw;++k)check(cudaStreamWaitEvent(workers_[k]->stream_.value,ready_.value,0));
        pending_parents_=std::move(parents);pending_needs_event_=std::move(needs_event);
        std::fill_n(pending_errors_.p,nw,0);
        try{
            pending_enqueue_=std::async(std::launch::async,[this]{
                BackgroundCudaContext context(cuda_context_);
                enqueue_particles();context.close();
            });
        }catch(const std::exception& e){throw CudaFailure(std::string("Cannot start background submission: ")+e.what());}
        // Only the input-copy fence is awaited on the MATLAB thread. The joined
        // native thread continues queueing graph/event/D2H commands concurrently
        // with next-frame preparation; it never calls a MATLAB API.
        check(cudaEventSynchronize(ready_.value));
        }catch(...){drain_pending();throw;}
    }
    bool supports_async()const override{return true;}
    bool pending()const override{return pending_;}
    void submit_device(const float* image,size_t step,const std::vector<CoarseCandidate>& coarse,bool diagnostics)override{
        if(!image)throw std::invalid_argument("Async submission needs a borrowed device image");
        enqueue_frame(nullptr,image,step,coarse,diagnostics);
    }
    RefineResult complete()override{
        if(!pending_)throw std::invalid_argument("No resident frame is pending");
        const auto& coarse=pending_coarse_;const auto& owner=pending_owner_;
        const int n=int(coarse.size()),nw=int(workers_.size()),layers=pending_layers_;
        const bool diagnostics=pending_diagnostics_;auto& primary=*workers_[0];
        const size_t count=size_t(model_.frame_width)*model_.frame_height;
        try{
        // The enqueue thread may still be using streams, fences and pinned buffers.
        // Join/rethrow it before reading any result or considering stream completion.
        join_enqueue();
        for(auto& w:workers_)check(cudaStreamSynchronize(w->stream_.value));
        for(int k=0;k<nw;++k)if(pending_errors_.p[k])throw CudaFailure("Concurrent resident unsupported input/device error "+std::to_string(pending_errors_.p[k]));
        RefineResult result;clamped_count_=0;
        for(int i=0;i<n;++i){const auto& w=*workers_[owner[i]];const auto& p=w.host_outputs_.p[i];clamped_count_+=p.clamped_xy;
            result.particles.push_back({p.source,p.initial_depth,p.final_depth,p.rejection,p.full_calls,p.legacy_calls,p.overlap,p.x,p.y,p.z,p.score});
            if(!p.rejection)result.localizations.push_back({p.x,p.y,model_.interpolate?(p.z-1)/(int(model_.templates.size())-1):p.z,p.score,p.source});
            if(diagnostics)for(int k=0;k<p.full_calls+p.legacy_calls;++k){const auto& t=w.host_traces_.p[size_t(i)*w.stride_+k];result.correlations.push_back({p.source,t.index,t.mode,unpack(t.peak)});}
        }
        // Host latency since submission, including any overlapped preparation before complete().
        result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-pending_started_).count();
        result.metrics.batch_submissions=layers;result.metrics.synchronizations=nw;
        if(diagnostics){result.image=Image(model_.frame_width,model_.frame_height);check(cudaMemcpy(result.image.pixels.data(),primary.frame_ptr_,count*8,cudaMemcpyDeviceToHost));}
        pending_=false;return result;
            }catch(...){drain_pending();throw;}
    }
    RefineResult process(const double* image,const float* device_image,size_t step,const std::vector<CoarseCandidate>& coarse,bool diagnostics){
        enqueue_frame(image,device_image,step,coarse,diagnostics);return complete();
    }
    RefineResult refine(const double* image,const std::vector<CoarseCandidate>& coarse,bool d)override{return process(image,nullptr,0,coarse,d);}
    RefineResult refine_device(const float* image,size_t step,const std::vector<CoarseCandidate>& coarse,bool d)override{return process(nullptr,image,step,coarse,d);}
    uint64_t clamped_xy()const override{return clamped_count_;}
    double warmup(int milliseconds)override{if(pending_)throw std::invalid_argument("Cannot warm up a pending resident frame");return workers_[0]->warmup(milliseconds);}
    Image last_image()override{if(pending_)throw std::invalid_argument("Complete the pending resident frame before reading its image");return workers_[0]->last_image();}
};
}
void verify_mixed_depth(const Model& model,const Image& image,int depth){check_mixed_private(model,image,depth);}
std::unique_ptr<ResidentEngine> make_resident(const Model& model,int workers,bool bounded,bool mixed,bool fused){if(workers==1)return std::make_unique<Resident>(model,nullptr,nullptr,bounded,mixed,fused);return std::make_unique<ConcurrentResident>(model,workers,bounded,mixed,fused);}
}
