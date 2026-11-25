#pragma once
// Include inside defocus's anonymous namespace AFTER correlation_cuda.cu.
// Experimental conservative FP32 shortlist + original FP64 spatial NCC arithmetic.
// Requires a single 256-thread CTA, template+8 crop (9x9 valid translations),
// finite nonnegative frame/crop data, and finite nonconstant calibration.
// No universal numerical-equivalence claim is made before synthetic/all-frame tests.

struct MixedDepthScratch {
    double lower[81],upper[81],exact[81];
    int known[81],force[81],selected[81];
    int all_double,error,again,selected_count;
    unsigned long long double_dots;
    Aggregate aggregate;
    PeakWire seed[3],peak;
    cub::BlockReduce<double,256>::TempStorage reduction;
};

// All 256 threads call this uniformly. Dot accumulation and reduction deliberately
// match spatial_batch: identical input/template values, thread stride, and CUB type.
__device__ void mixed_depth_exact_dot(const double* image,const double* templ,const Moment* prefix,
    const double* norms,const double* residuals,int width,int height,int tw,int th,int index,
    int position,MixedDepthScratch* s) {
    int columns=width-tw+1,x=position%columns,y=position/columns;
    double dot=0;
    for(int i=threadIdx.x;i<tw*th;i+=blockDim.x)
        dot+=image[(y+i/tw)*width+x+i%tw]*templ[i];
    double sum=cub::BlockReduce<double,256>(s->reduction).Sum(dot);
    if(threadIdx.x==0){
        Moment v=window(prefix,width+1,x,y,x+tw,y+th);
        double denominator=norms[index]*sqrt(fmax(v.square-v.sum*v.sum/(tw*th),0.0));
        double value=denominator>0?(sum-v.sum*residuals[index])/denominator:0;
        s->exact[position]=fabs(value)-1>sqrt(0x1p-52)?0:value;
        s->known[position]=1;++s->double_dots;
        if(!isfinite(s->exact[position]))s->error=1;
    }
    // CUB temporary storage and the cached score cannot be reused before this.
    __syncthreads();
}

__device__ void mixed_depth_peak(const double* image,const double* templates,const Moment* prefix,
    const double* norms,const double* residuals,int width,int height,int tw,int th,int index,
    MixedDepthScratch* s,PeakWire* output) {
    int t=threadIdx.x,warp=t/32,lane=t%32;
    int columns=width-tw+1,rows=height-th+1,count=columns*rows,n=tw*th;
    const double* templ=templates+size_t(index)*n;
    if(!t){s->all_double=0;s->error=0;}
    for(int p=t;p<count;p+=256){s->known[p]=s->force[p]=s->selected[p]=0;s->exact[p]=0;}
    __syncthreads();

    // Each warp evaluates one translation at a time. Centering before FP32
    // conversion suppresses raw-background cancellation in approximate dot sums.
    for(int position=warp;position<count;position+=8){
        int x=position%columns,y=position/columns;
        double mean=0,variance=0,raw_square=0,denominator=0;
        if(!lane){
            Moment v=window(prefix,width+1,x,y,x+tw,y+th);
            mean=v.sum/n;variance=fmax(v.square-v.sum*v.sum/n,0.0);
            raw_square=v.square;denominator=norms[index]*sqrt(variance);
        }
        mean=__shfl_sync(0xffffffff,mean,0);
        variance=__shfl_sync(0xffffffff,variance,0);
        raw_square=__shfl_sync(0xffffffff,raw_square,0);
        denominator=__shfl_sync(0xffffffff,denominator,0);
        if(!(denominator>0)){
            if(!lane){s->lower[position]=s->upper[position]=0;s->exact[position]=0;s->known[position]=1;}
            continue;
        }
        float dot=0,centered_square=0;
        for(int i=lane;i<n;i+=32){
            float a=float(image[(y+i/tw)*width+x+i%tw]-mean),b=float(templ[i]);
            dot+=a*b;centered_square+=a*a;
        }
        for(int shift=16;shift;shift/=2){
            dot+=__shfl_down_sync(0xffffffff,dot,shift);
            centered_square+=__shfl_down_sync(0xffffffff,centered_square,shift);
        }
        if(!lane){
            constexpr double epsilon=0x1p-23,tiny=0x1p-126;
            double eta=double((n+31)/32+8)*epsilon;
            double gamma=eta<1?eta/(1-eta):INFINITY;
            double root_variance=sqrt(variance);
            double raw_condition=sqrt(fmax(raw_square,0.0))/root_variance;
            double centered_upper=gamma<1?
                sqrt((double(centered_square)+n*tiny)/(1-gamma))/(1-epsilon):INFINITY;
            double template_raw_norm=hypot(norms[index],sqrt(double(n))*residuals[index]);
            double template_factor=template_raw_norm/norms[index];
            // At least twice the parent's suggested 4*eps*(ceil(n/32)+8)
            // bound. The 1e-10 term conservatively covers FP64 mean/prefix,
            // residual, quantization, and old raw-dot cancellation differences.
            double error=8*gamma*centered_upper/root_variance*template_factor+
                1e-10*(raw_condition+1)+
                8*n*tiny*(centered_upper+template_raw_norm+1)/denominator;
            double approximate=double(dot)/denominator;
            if(!isfinite(approximate)||!isfinite(error)||eta>=.125||
               !(centered_square>0)||error>=.25||!isfinite(denominator)){
                atomicExch(&s->all_double,1);s->lower[position]=-INFINITY;s->upper[position]=INFINITY;
            }else{
                // Extra outward slack avoids cancellation when interval endpoints
                // are formed in FP64. Never use an interval through the score
                // clipping discontinuity to eliminate a possible maximum.
                error+=1e-12*(fabs(approximate)+1);
                s->lower[position]=approximate-error;s->upper[position]=approximate+error;
                if(fabs(approximate)+error>=1+sqrt(0x1p-52)){
                    s->force[position]=1;s->lower[position]=-INFINITY;s->upper[position]=INFINITY;
                }
            }
        }
    }
    __syncthreads();
    if(!t){
        double lower=0; // The unchanged full-map zero padding is also a candidate.
        for(int p=0;p<count;++p)lower=fmax(lower,s->lower[p]);
        int fw=width+tw-1,fh=height+th-1;
        int sum_x=rows*columns*(2*tw+columns-1)/2;
        int sum_y=columns*rows*(2*th+rows-1)/2;
        s->aggregate={0,0,fw*fh-count,fh*fw*(fw+1)/2-sum_x,fw*fh*(fh+1)/2-sum_y};
        s->selected_count=0;
        for(int p=0;p<count;++p){
            bool eligible=s->all_double||s->force[p]||s->upper[p]>=lower;
            if(!eligible)continue;
            if(s->known[p]){
                // Zero-variance translations contribute once to padded-zero ties,
                // but need neither a dot product nor a block-wide loop iteration.
                int x=p%columns+tw-1,y=p/columns+th-1;
                s->aggregate=ReducePeak{}(s->aggregate,Aggregate{s->exact[p],x*fh+y,1,x+1,y+1});
            }else s->selected[s->selected_count++]=p;
        }
    }
    __syncthreads();
    for(int item=0;item<s->selected_count;++item){
        int position=s->selected[item];
        mixed_depth_exact_dot(image,templ,prefix,norms,residuals,width,height,tw,th,index,position,s);
        if(!t){
            int x=position%columns+tw-1,y=position/columns+th-1,fh=height+th-1;
            s->aggregate=ReducePeak{}(s->aggregate,Aggregate{s->exact[position],x*fh+y,1,x+1,y+1});
        }
    }
    if(!t){
        const auto& a=s->aggregate;int fh=height+th-1,x=a.first/fh,y=a.first%fh;
        s->peak=PeakWire{};s->peak.score=a.score;s->peak.x=x+1;s->peak.y=y+1;
        s->peak.ties=a.ties;s->peak.sum_x=a.sum_x;s->peak.sum_y=a.sum_y;
        int vx=x-tw+1,vy=y-th+1;
        s->peak.neighbors=vx>=0&&vy>=0&&vx<columns&&vy<rows;
    }
    __syncthreads();
    // Recompute all four interpolation neighbors in the original FP64 arithmetic.
    // Reuse an already-exact shortlisted value instead of changing its reduction.
    if(s->peak.neighbors){
        int vx=s->peak.x-tw,vy=s->peak.y-th;
        for(int direction=0;direction<4;++direction){
            int x=vx+(direction==0?-1:direction==1?1:0);
            int y=vy+(direction==2?-1:direction==3?1:0);
            int p=x+y*columns;bool valid=x>=0&&y>=0&&x<columns&&y<rows;
            if(valid&&!s->known[p])mixed_depth_exact_dot(image,templ,prefix,norms,residuals,width,height,tw,th,index,p,s);
            if(!t){double value=valid?s->exact[p]:0;
                if(direction==0)s->peak.left=value;else if(direction==1)s->peak.right=value;
                else if(direction==2)s->peak.above=value;else s->peak.below=value;}
            __syncthreads();
        }
    }
    if(!t)*output=s->peak;
    __syncthreads();
}

__global__ void mixed_depth_walk(const double* image,const double* templates,const Moment* prefix,
    double* maps_or_unused,int* indices,const double* norms,const double* residuals,
    int width,int height,int tw,int th,WalkState* walk,WalkSample* samples,const int* start_depth,
    int depths,uint64_t* actual_double_dots=nullptr) {
    (void)maps_or_unused;
    __shared__ MixedDepthScratch scratch;
    if(blockIdx.x)return;
    int t=threadIdx.x,columns=width-tw+1,rows=height-th+1;
    if(blockDim.x!=256||columns!=9||rows!=9||depths<3){
        if(!t){*walk=WalkState{};walk->error=1;}return;
    }
    if(!t){
        *walk=WalkState{};walk->initial=*start_depth;scratch.double_dots=0;
        int d=*start_depth;
        if(d<1||d>depths){walk->error=1;scratch.again=0;}
        else {scratch.again=1;
            if(d==1){indices[0]=1;indices[1]=2;indices[2]=3;}
            else if(d==depths){indices[0]=d;indices[1]=d-2;indices[2]=d-1;}
            else {indices[0]=d;indices[1]=d-1;indices[2]=d+1;}}
    }
    __syncthreads();if(!scratch.again)return;
    for(int k=0;k<3;++k){
        mixed_depth_peak(image,templates,prefix,norms,residuals,width,height,tw,th,indices[k]-1,&scratch,scratch.seed+k);
        if(scratch.error){if(!t)walk->error=1;return;}
    }
    if(!t){walk_seed(walk,scratch.seed,indices,samples,depths);scratch.again=walk_decide(walk,indices,depths);}
    __syncthreads();
    while(scratch.again){
        mixed_depth_peak(image,templates,prefix,norms,residuals,width,height,tw,th,indices[0]-1,&scratch,&scratch.peak);
        if(scratch.error){if(!t)walk->error=1;return;}
        if(!t){scratch.again=walk_record(walk,&scratch.peak,indices,samples,depths)&&walk_decide(walk,indices,depths);}
        __syncthreads();
    }
    if(!t&&actual_double_dots)atomicAdd(reinterpret_cast<unsigned long long*>(actual_double_dots),scratch.double_dots);
}
