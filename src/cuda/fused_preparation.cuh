#pragma once
// Include inside defocus's anonymous namespace after correlation_cuda.cu.
// Header-only exact-expression experiment. No build/accuracy/speed claim until
// the parent's synthetic and real-frame checks pass.

// One CTA computes the same row prefixes followed by the same column prefixes.
// __syncthreads makes all preceding global prefix writes visible to the threads
// of this CTA before they read those values during the column pass.
__global__ void fused_prefix(const double* image,Moment* prefix,int width,int height,double* maximum) {
    int t=threadIdx.x;
    if(blockIdx.x||blockIdx.y||blockIdx.z)return;
    if(!t)*maximum=0.0; // Positive-FP64 atomicMax identity for the following kernel.
    for(int x=t;x<=width;x+=blockDim.x)prefix[x]={0,0};
    for(int y=t+1;y<=height;y+=blockDim.x)prefix[size_t(y)*(width+1)]={0,0};
    for(int y=t;y<height;y+=blockDim.x){
        double sum=0,square=0;
        for(int x=0;x<width;++x){
            double v=image[size_t(y)*width+x]+0.0;
            sum+=v;square+=v*v;
            prefix[size_t(y+1)*(width+1)+x+1]={sum,square};
        }
    }
    __syncthreads();
    for(int x=t+1;x<=width;x+=blockDim.x){
        double sum=0,square=0;
        for(int y=1;y<=height;++y){
            Moment& value=prefix[size_t(y)*(width+1)+x];
            sum+=value.sum;square+=value.square;value={sum,square};
        }
    }
}

struct FusedPreparationMax {
    __host__ __device__ double operator()(double a,double b)const{return a>b?a:b;}
};

// Launch ceil(max((width+tw-1)*(height+th-1),fftw*ffth)/256) CTAs of256.
// maximum must have been reset to+0 by fused_prefix, and these kernels must
// be ordered in the same worker stream/graph. The bitwise maximum is valid
// only for finite nonnegative doubles; invalid values are flagged explicitly.
// Pass the resident worker's error buffer to preserve whole-frame CPU fallback.
__global__ void fused_full_prepare(const double* image,const Moment* prefix,double* padded,
    double* sums,double* norms,double* maximum,int width,int height,int tw,int th,
    int fftw,int ffth,int* error=nullptr) {
    if(blockDim.x!=256||blockDim.y!=1||blockDim.z!=1){
        if(!threadIdx.x&&!blockIdx.x&&error)atomicExch(error,8);return;
    }
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int fw=width+tw-1,fh=height+th-1;
    double norm=0;
    if(i<fw*fh){
        int x=i%fw,y=i/fw;
        Moment value=window(prefix,width+1,max(0,x-tw+1),max(0,y-th+1),min(width,x+1),min(height,y+1));
        // Same arithmetic as full_statistics(..., template_norm=1.0).
        norm=sqrt(fmax(value.square-value.sum*value.sum/(tw*th),0.0));
        if(!isfinite(value.sum)||!isfinite(value.square)||!isfinite(norm)||norm<0){
            if(error)atomicExch(error,8);norm=0;
        }
        norm=fabs(norm); // Canonicalize zero's sign before positive-bit atomicMax.
        sums[i]=value.sum;norms[i]=norm;
    }
    if(i<fftw*ffth){
        int x=i%fftw,y=i/fftw;
        // Same nonnegative, zero-shift padding as the resident full-map path.
        double value=x<width&&y<height?image[size_t(y)*width+x]+0.0:0.0;
        if(!isfinite(value)){if(error)atomicExch(error,8);value=0;}
        padded[i]=value;
    }
    __shared__ cub::BlockReduce<double,256>::TempStorage temporary;
    double block_max=cub::BlockReduce<double,256>(temporary).Reduce(norm,FusedPreparationMax{});
    if(!threadIdx.x){
        // Positive finite IEEE754 doubles have the same ordering as their bits.
        atomicMax(reinterpret_cast<unsigned long long*>(maximum),
            static_cast<unsigned long long>(__double_as_longlong(block_max)));
    }
}
