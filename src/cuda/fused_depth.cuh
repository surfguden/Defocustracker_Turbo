#pragma once
// Include inside defocus's anonymous namespace after correlation_cuda.cu.
// Exact-arithmetic experiment: fuse one adaptive depth's spatial NCC, peak
// reduction, and walk update. Initial three-template evaluation is unchanged.
// Launch one 256-thread CTA for every valid XY position. Counter and maps must
// be private to a worker; graph ordering must serialize uses of that worker.

__global__ void fused_depth_seed(WalkState* walk,const PeakWire* wires,int* indices,
    WalkSample* samples,int depths,unsigned* counter,cudaGraphConditionalHandle condition) {
    if(blockIdx.x||blockIdx.y||blockIdx.z||threadIdx.x)return;
    atomicExch(counter,0u);
    walk_seed(walk,wires,indices,samples,depths);
    walk_choose(walk,indices,depths,condition);
}

__global__ void fused_depth_step(const double* image,const double* templates,const Moment* prefix,
    double* maps,int* indices,const double* norms,const double* residuals,
    int width,int height,int tw,int th,WalkState* walk,WalkSample* samples,
    int depths,unsigned* counter,cudaGraphConditionalHandle condition) {
    int columns=width-tw+1,rows=height-th+1,count=columns*rows;
    int position=blockIdx.x,t=threadIdx.x,index=indices[0]-1;
    if(blockIdx.y||blockIdx.z)return;
    if(blockDim.x!=256||columns<1||rows<1||index<0||index>=depths){
        if(!t&&!blockIdx.x){atomicExch(&walk->error,1);atomicExch(counter,0u);cudaGraphSetConditional(condition,0);}return;
    }
    if(position>=count)return;
    __shared__ cub::BlockReduce<double,256>::TempStorage dot_storage;
    __shared__ cub::BlockReduce<Aggregate,256>::TempStorage peak_storage;
    __shared__ int last_block;
    int x=position%columns,y=position/columns;
    const double* templ=templates+size_t(index)*tw*th;
    double dot=0;
    // Deliberately identical to spatial_batch: same values, thread stride,
    // double accumulation and cub::BlockReduce<double,256> sum.
    for(int i=t;i<tw*th;i+=blockDim.x)
        dot+=image[(y+i/tw)*width+x+i%tw]*templ[i];
    double sum=cub::BlockReduce<double,256>(dot_storage).Sum(dot);
    volatile double* visible_maps=maps;
    if(!t){
        Moment v=window(prefix,width+1,x,y,x+tw,y+th);
        double denominator=norms[index]*sqrt(fmax(v.square-v.sum*v.sum/(tw*th),0.0));
        double value=denominator>0?(sum-v.sum*residuals[index])/denominator:0;
        value=fabs(value)-1>sqrt(0x1p-52)?0:value;
        if(!isfinite(value)){atomicExch(&walk->error,1);value=0;}
        // Volatile forces a globally visible score store/load. The fence orders
        // that store before publishing this CTA's completion through the atomic.
        visible_maps[position]=value;
        __threadfence();
        unsigned completed=atomicAdd(counter,1u);
        last_block=completed==unsigned(count-1);
    }
    __syncthreads();
    if(!last_block)return; // Uniform across every thread of this CTA.

    int fw=width+tw-1,fh=height+th-1;
    Aggregate partial{-INFINITY,INT_MAX,0,0,0};ReducePeak combine;
    for(int i=t;i<count;i+=blockDim.x){
        int fx=i%columns+tw-1,fy=i/columns+th-1;
        partial=combine(partial,Aggregate{visible_maps[i],fx*fh+fy,1,fx+1,fy+1});
    }
    if(!t){
        int sum_x=rows*columns*(2*tw+columns-1)/2;
        int sum_y=columns*rows*(2*th+rows-1)/2;
        Aggregate padding{0,0,fw*fh-count,fh*fw*(fw+1)/2-sum_x,fw*fh*(fh+1)/2-sum_y};
        partial=combine(partial,padding);
    }
    Aggregate maximum=cub::BlockReduce<Aggregate,256>(peak_storage).Reduce(partial,combine);
    if(!t){
        PeakWire peak{};int fx=maximum.first/fh,fy=maximum.first%fh;
        peak.score=maximum.score;peak.x=fx+1;peak.y=fy+1;
        peak.ties=maximum.ties;peak.sum_x=maximum.sum_x;peak.sum_y=maximum.sum_y;
        int vx=fx-tw+1,vy=fy-th+1;
        if(vx>=0&&vy>=0&&vx<columns&&vy<rows){
            peak.neighbors=1;
            peak.left=vx>0?visible_maps[vy*columns+vx-1]:0;
            peak.right=vx+1<columns?visible_maps[vy*columns+vx+1]:0;
            peak.above=vy>0?visible_maps[(vy-1)*columns+vx]:0;
            peak.below=vy+1<rows?visible_maps[(vy+1)*columns+vx]:0;
        }
        // Every score has already been read by the final CTA. The next graph
        // iteration cannot begin before this whole kernel completes.
        atomicExch(counter,0u);
        if(walk->error||!walk_record(walk,&peak,indices,samples,depths))
            cudaGraphSetConditional(condition,0);
        else walk_choose(walk,indices,depths,condition);
    }
}
