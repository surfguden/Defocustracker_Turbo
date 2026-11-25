// CUDA candidate engine; project/upstream attribution is in THIRD_PARTY_NOTICES.md.
#include "defocus/tracker.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <immintrin.h>
#include <map>
#include <climits>

namespace defocus { namespace cuda_detail {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw defocus::CudaFailure(cudaGetErrorString(status));
    }
}

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

template<class T> class DeviceBuffer {
public:
    T* data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { cudaFree(data); }
    void allocate(size_t count) {
        if (data) throw std::logic_error("Device buffer already allocated");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
};

struct Config {
    int width, height, template_width, template_height, count;
    double threshold;
    int min_x, max_x, min_y, max_y;
    std::vector<int> indices;
};

struct Candidate {
    int template_index, x, y;
    float score;
};

struct Statistics { double sum, square_sum; };

struct AddStatistics {
    __host__ __device__ Statistics operator()(Statistics a, Statistics b) const {
        return {a.sum + b.sum, a.square_sum + b.square_sum};
    }
};

struct GpuTemplate {
    cv::cuda::GpuMat pixels;
    double residual_mean, norm;
};

struct Result {
    std::vector<cv::Mat> host_maps;
    std::vector<cv::cuda::GpuMat> device_maps;
    std::vector<std::vector<Candidate>> candidates;
    double correlation_s = 0, normalization_s = 0, peaks_s = 0;
    double stage_s = 0, upload_s = 0, download_s = 0;
    explicit Result(int count) : host_maps(count),device_maps(count),candidates(count) {}
};

Config read_config(const fs::path& folder) {
    Config cfg{};
    std::ifstream input(folder/"input.txt");
    input>>cfg.width>>cfg.height>>cfg.template_width>>cfg.template_height>>cfg.count>>cfg.threshold;
    input>>cfg.min_x>>cfg.max_x>>cfg.min_y>>cfg.max_y;
    if (!input || cfg.count<1 || cfg.count>65535 || cfg.template_width<3 || cfg.template_height<3 ||
        cfg.width<cfg.template_width || cfg.height<cfg.template_height || !std::isfinite(cfg.threshold) ||
        1LL*cfg.width*cfg.height>INT_MAX)
        throw std::runtime_error("Invalid configuration dimensions or template count");
    cfg.indices.resize(cfg.count);
    for (int& index:cfg.indices) {
        input>>index;
        if (!input || index<1) throw std::runtime_error("Invalid template index list");
    }
    auto sorted=cfg.indices;
    std::sort(sorted.begin(),sorted.end());
    if (std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())
        throw std::runtime_error("Template indices must be unique");
    return cfg;
}

int checked_product(int a,int b) {
    long long result=1LL*a*b;
    if (result<1 || result>INT_MAX) throw std::runtime_error("Requested dimensions exceed the supported indexing range");
    return int(result);
}

cv::Mat read_image(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    uint32_t dimensions[2]{};
    stream.read(reinterpret_cast<char*>(dimensions), sizeof(dimensions));
    if (!stream || dimensions[0] == 0 || dimensions[1] == 0) {
        throw std::runtime_error("Invalid image header: " + path.string());
    }
    cv::Mat image(dimensions[0], dimensions[1], CV_32FC1);
    stream.read(reinterpret_cast<char*>(image.data), image.total() * sizeof(float));
    if (!stream) throw std::runtime_error("Truncated image: " + path.string());
    return image;
}

void write_image(const fs::path& path, const cv::Mat& image) {
    std::ofstream stream(path, std::ios::binary);
    uint32_t dimensions[2] = {static_cast<uint32_t>(image.rows), static_cast<uint32_t>(image.cols)};
    stream.write(reinterpret_cast<const char*>(dimensions), sizeof(dimensions));
    for (int row = 0; row < image.rows; ++row) {
        stream.write(reinterpret_cast<const char*>(image.ptr<float>(row)), image.cols * sizeof(float));
    }
    if (!stream) throw std::runtime_error("Could not write " + path.string());
}

std::string numbered(const std::string& prefix, int index, const std::string& suffix) {
    std::ostringstream name;
    name << prefix << std::setw(2) << std::setfill('0') << index << suffix;
    return name.str();
}

// All prefix sums use doubles, including the squared intensities.
__global__ void make_samples(const float* image, size_t step, Statistics* values,
                             int* keys, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * height) return;
    int row = index / width;
    int col = index % width;
    double value = reinterpret_cast<const float*>(reinterpret_cast<const char*>(image) + row * step)[col];
    values[index] = {value, value * value};
    keys[index] = row;
}

__global__ void transpose_prefix(const Statistics* rows, Statistics* columns,
                                int* keys, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * height) return;
    int row = index / width;
    int col = index % width;
    int destination = col * height + row;
    columns[destination] = rows[index];
    if (keys) keys[destination] = col;
}

__global__ void scan_image_rows(const float* image,size_t step,Statistics* output,int width) {
    using Scan=cub::BlockScan<Statistics,256>;
    __shared__ Scan::TempStorage temporary;
    __shared__ Statistics carry;
    int row=blockIdx.x;
    if (threadIdx.x==0) carry={0,0};
    __syncthreads();
    const float* pixels=reinterpret_cast<const float*>(reinterpret_cast<const char*>(image)+row*step);
    for (int start=0;start<width;start+=256) {
        int col=start+threadIdx.x;
        double pixel=col<width ? pixels[col] : 0;
        Statistics value{pixel,pixel*pixel};
        Statistics scanned,aggregate;
        Scan(temporary).InclusiveScan(value,scanned,AddStatistics{},aggregate);
        Statistics previous=carry;
        if (col<width) output[row*width+col]=AddStatistics{}(previous,scanned);
        __syncthreads();
        if (threadIdx.x==0) carry=AddStatistics{}(previous,aggregate);
        __syncthreads();
    }
}

__global__ void scan_transposed_columns(const Statistics* input,Statistics* output,int height) {
    using Scan=cub::BlockScan<Statistics,256>;
    __shared__ Scan::TempStorage temporary;
    __shared__ Statistics carry;
    int col=blockIdx.x;
    if (threadIdx.x==0) carry={0,0};
    __syncthreads();
    for (int start=0;start<height;start+=256) {
        int row=start+threadIdx.x;
        Statistics value=row<height ? input[col*height+row] : Statistics{0,0};
        Statistics scanned,aggregate;
        Scan(temporary).InclusiveScan(value,scanned,AddStatistics{},aggregate);
        Statistics previous=carry;
        if (row<height) output[col*height+row]=AddStatistics{}(previous,scanned);
        __syncthreads();
        if (threadIdx.x==0) carry=AddStatistics{}(previous,aggregate);
        __syncthreads();
    }
}

template<bool row_major=false>
__device__ Statistics integral_at(const Statistics* integral, int row, int col, int width,int height) {
    if (row < 0 || col < 0) return {0, 0};
    return integral[row_major ? row*width+col : col*height+row];
}

template<bool row_major=false>
__global__ void window_statistics(const Statistics* integral, double* sums, double* norms,
                                  int width, int height, int template_width, int template_height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int result_width = width - template_width + 1;
    int result_height = height - template_height + 1;
    if (index >= result_width * result_height) return;
    int row = index / result_width;
    int col = index % result_width;
    Statistics a = integral_at<row_major>(integral, row + template_height - 1, col + template_width - 1, width,height);
    Statistics b = integral_at<row_major>(integral, row - 1, col + template_width - 1, width,height);
    Statistics c = integral_at<row_major>(integral, row + template_height - 1, col - 1, width,height);
    Statistics d = integral_at<row_major>(integral, row - 1, col - 1, width,height);
    double sum = a.sum - b.sum - c.sum + d.sum;
    double square_sum = a.square_sum - b.square_sum - c.square_sum + d.square_sum;
    double count = template_width * template_height;
    sums[index] = sum;
    norms[index] = sqrt(fmax(square_sum - sum * sum / count, 0.0));
}

__global__ void normalize_map(float* map, size_t step, const double* sums, const double* norms,
                              const double* maximum_norm, int width, int height,
                              double residual_mean, double template_norm) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * height) return;
    int row = index / width;
    int col = index % width;
    float* value = &reinterpret_cast<float*>(reinterpret_cast<char*>(map) + row * step)[col];
    double denominator = norms[index] * template_norm;
    double max_denominator = *maximum_norm * template_norm;
    double tolerance = max_denominator > 0 ? sqrt(ldexp(1.0, ilogb(max_denominator) - 52)) : 0;
    double score = denominator > tolerance ? (double(*value) - sums[index] * residual_mean) / denominator : 0;
    *value = fabs(score) - 1 > sqrt(0x1p-52) ? 0.0f : static_cast<float>(score);
}

__global__ void flag_peaks(const float* map, size_t step, Candidate* values, uint8_t* flags,
                           int width, int height, int template_index, int offset_x, int offset_y,
                           int min_x, int max_x, int min_y, int max_y, double threshold) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * height) return;
    int row = index / width;
    int col = index % width;
    const float* current = reinterpret_cast<const float*>(reinterpret_cast<const char*>(map) + row * step);
    float score = current[col];
    int x = col + offset_x;
    int y = row + offset_y;
    bool peak = false;
    if (x >= min_x && x <= max_x && y >= min_y && y <= max_y &&
        row > 0 && row + 1 < height && col > 0 && col + 1 < width) {
        const float* above = reinterpret_cast<const float*>(reinterpret_cast<const char*>(map) + (row - 1) * step);
        const float* below = reinterpret_cast<const float*>(reinterpret_cast<const char*>(map) + (row + 1) * step);
        peak = score > threshold && score > current[col - 1] && score > current[col + 1] &&
               score > above[col] && score > below[col];
    }
    flags[index] = peak;
    if (peak) values[index] = {template_index, x, y, score};
}

__global__ void transpose_statistics_tiled(const Statistics* source,Statistics* target,int width,int height) {
    __shared__ Statistics tile[32][33];
    int x=blockIdx.x*32+threadIdx.x,y=blockIdx.y*32+threadIdx.y;
    for (int j=0;j<32;j+=8)
        if (x<width && y+j<height) tile[threadIdx.y+j][threadIdx.x]=source[(y+j)*width+x];
    __syncthreads();
    x=blockIdx.y*32+threadIdx.x;
    y=blockIdx.x*32+threadIdx.y;
    for (int j=0;j<32;j+=8)
        if (x<height && y+j<width) target[(y+j)*height+x]=tile[threadIdx.x][threadIdx.y+j];
}

class GpuWindowStatistics {
    Config cfg;
    DeviceBuffer<Statistics> samples, prefix;
    DeviceBuffer<int> keys;
    DeviceBuffer<uint8_t> workspace;
    size_t workspace_bytes = 0;
public:
    DeviceBuffer<double> sums, norms, maximum_norm;
    explicit GpuWindowStatistics(const Config& config) : cfg(config) {
        int count = cfg.width * cfg.height;
        int output_count = (cfg.width-cfg.template_width+1) * (cfg.height-cfg.template_height+1);
        samples.allocate(count);
        prefix.allocate(count);
        keys.allocate(count);
        sums.allocate(output_count);
        norms.allocate(output_count);
        maximum_norm.allocate(1);
        size_t scan_bytes = 0, reduce_bytes = 0;
        check_cuda(cub::DeviceScan::InclusiveScanByKey(nullptr, scan_bytes, keys.data,
            samples.data, prefix.data, AddStatistics{}, count));
        check_cuda(cub::DeviceReduce::Max(nullptr, reduce_bytes, norms.data,
            maximum_norm.data, output_count));
        workspace_bytes = std::max(scan_bytes, reduce_bytes);
        workspace.allocate(workspace_bytes);
    }
    void compute(const cv::cuda::GpuMat& image, cv::cuda::Stream& stream) {
        cudaStream_t native_stream = cv::cuda::StreamAccessor::getStream(stream);
        int count = cfg.width * cfg.height;
        int output_count = (cfg.width-cfg.template_width+1) * (cfg.height-cfg.template_height+1);
        make_samples<<<(count+255)/256,256,0,native_stream>>>(
            image.ptr<float>(),image.step,samples.data,keys.data,cfg.width,cfg.height);
        check_cuda(cub::DeviceScan::InclusiveScanByKey(workspace.data,workspace_bytes,keys.data,
            samples.data,prefix.data,AddStatistics{},count,cuda::std::equal_to<>{},native_stream));
        transpose_prefix<<<(count+255)/256,256,0,native_stream>>>(
            prefix.data,samples.data,keys.data,cfg.width,cfg.height);
        check_cuda(cub::DeviceScan::InclusiveScanByKey(workspace.data,workspace_bytes,keys.data,
            samples.data,prefix.data,AddStatistics{},count,cuda::std::equal_to<>{},native_stream));
        window_statistics<false><<<(output_count+255)/256,256,0,native_stream>>>(
            prefix.data,sums.data,norms.data,cfg.width,cfg.height,cfg.template_width,cfg.template_height);
        check_cuda(cub::DeviceReduce::Max(workspace.data,workspace_bytes,norms.data,
            maximum_norm.data,output_count,native_stream));
        stream.waitForCompletion();
    }
    void compute_tiled(const cv::cuda::GpuMat& image,cv::cuda::Stream& stream) {
        cudaStream_t native_stream=cv::cuda::StreamAccessor::getStream(stream);
        int count=cfg.width*cfg.height;
        int output_count=(cfg.width-cfg.template_width+1)*(cfg.height-cfg.template_height+1);
        scan_image_rows<<<cfg.height,256,0,native_stream>>>(
            image.ptr<float>(),image.step,prefix.data,cfg.width);
        transpose_prefix<<<(count+255)/256,256,0,native_stream>>>(
            prefix.data,samples.data,nullptr,cfg.width,cfg.height);
        scan_transposed_columns<<<cfg.width,256,0,native_stream>>>(samples.data,prefix.data,cfg.height);
        window_statistics<false><<<(output_count+255)/256,256,0,native_stream>>>(
            prefix.data,sums.data,norms.data,cfg.width,cfg.height,cfg.template_width,cfg.template_height);
        check_cuda(cub::DeviceReduce::Max(workspace.data,workspace_bytes,norms.data,
            maximum_norm.data,output_count,native_stream));
        stream.waitForCompletion();
    }
    void compute_coalesced(const cv::cuda::GpuMat& image,cv::cuda::Stream& stream) {
        cudaStream_t native_stream=cv::cuda::StreamAccessor::getStream(stream);
        int output_count=(cfg.width-cfg.template_width+1)*(cfg.height-cfg.template_height+1);
        scan_image_rows<<<cfg.height,256,0,native_stream>>>(image.ptr<float>(),image.step,prefix.data,cfg.width);
        transpose_statistics_tiled<<<dim3((cfg.width+31)/32,(cfg.height+31)/32),dim3(32,8),0,native_stream>>>(
            prefix.data,samples.data,cfg.width,cfg.height);
        scan_transposed_columns<<<cfg.width,256,0,native_stream>>>(samples.data,prefix.data,cfg.height);
        transpose_statistics_tiled<<<dim3((cfg.height+31)/32,(cfg.width+31)/32),dim3(32,8),0,native_stream>>>(
            prefix.data,samples.data,cfg.height,cfg.width);
        window_statistics<true><<<(output_count+255)/256,256,0,native_stream>>>(samples.data,sums.data,norms.data,
            cfg.width,cfg.height,cfg.template_width,cfg.template_height);
        check_cuda(cub::DeviceReduce::Max(workspace.data,workspace_bytes,norms.data,
            maximum_norm.data,output_count,native_stream));
        stream.waitForCompletion();
    }
};

void check_fft(cufftResult status) {
    if (status!=CUFFT_SUCCESS) throw defocus::CudaFailure("cuFFT error "+std::to_string(status));
}

class FftPlan {
public:
    cufftHandle handle = 0;
    FftPlan() = default;
    FftPlan(const FftPlan&) = delete;
    FftPlan& operator=(const FftPlan&) = delete;
    ~FftPlan() { if (handle) cufftDestroy(handle); }
};

int smooth_fft_size(int minimum) {
    for (int size=minimum;;++size) {
        int remainder = size;
        for (int prime : {2,3,5,7}) {
            while (remainder%prime==0) remainder/=prime;
        }
        if (remainder==1) return size;
    }
}

__global__ void pad_image(const float* image,size_t step,float* padded,
                         int width,int height,int fft_width,int fft_height) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=fft_width*fft_height) return;
    int row=index/fft_width;
    int col=index%fft_width;
    float value=0;
    if (row<height && col<width) {
        value=reinterpret_cast<const float*>(reinterpret_cast<const char*>(image)+row*step)[col];
    }
    padded[index]=value;
}

__global__ void pack_template(const float* image,size_t step,float* padded,
                             int width,int height,int fft_width) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=width*height) return;
    int row=index/width;
    int col=index%width;
    padded[row*fft_width+col]=reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(image)+row*step)[col];
}

__global__ void multiply_templates(const cufftComplex* image,const cufftComplex* templates,
                                   cufftComplex* products,int spectrum_count,int batches) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=spectrum_count*batches) return;
    cufftComplex a=image[index%spectrum_count];
    cufftComplex b=templates[index];
    products[index]={a.x*b.x+a.y*b.y,a.y*b.x-a.x*b.y};
}

__global__ void normalize_fft_map(const float* inverse,int inverse_stride,float scale,
                                  float* map,size_t step,const double* sums,const double* norms,
                                  const double* maximum_norm,int width,int height,
                                  double residual_mean,double template_norm) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=width*height) return;
    int row=index/width;
    int col=index%width;
    double denominator=norms[index]*template_norm;
    double maximum=*maximum_norm*template_norm;
    double tolerance=maximum>0 ? sqrt(ldexp(1.0,ilogb(maximum)-52)) : 0;
    double numerator=double(inverse[row*inverse_stride+col])*scale-sums[index]*residual_mean;
    double value=denominator>tolerance ? numerator/denominator : 0;
    float* output=reinterpret_cast<float*>(reinterpret_cast<char*>(map)+row*step);
    output[col]=fabs(value)-1>sqrt(0x1p-52) ? 0.0f : static_cast<float>(value);
}

struct FusedMapParameters {
    float* output;
    size_t step;
    double residual_mean,template_norm;
    float inverse_template_norm;
};

struct PeakMapParameters { const float* map; size_t step; Candidate* selected; int template_index; };

__global__ void collect_peaks_atomic(const PeakMapParameters* maps,int* counts,int width,int height,
                                     int offset_x,int offset_y,int min_x,int max_x,int min_y,int max_y,
                                     float threshold) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=width*height) return;
    int row=index/width,col=index%width,k=blockIdx.y;
    int x=col+offset_x,y=row+offset_y;
    if (x<min_x || x>max_x || y<min_y || y>max_y || row==0 || row+1==height || col==0 || col+1==width) return;
    PeakMapParameters p=maps[k];
    const float* current=reinterpret_cast<const float*>(reinterpret_cast<const char*>(p.map)+row*p.step);
    float score=current[col];
    if (!(score>threshold && score>current[col-1] && score>current[col+1])) return;
    const float* above=reinterpret_cast<const float*>(reinterpret_cast<const char*>(p.map)+(row-1)*p.step);
    const float* below=reinterpret_cast<const float*>(reinterpret_cast<const char*>(p.map)+(row+1)*p.step);
    if (score>above[col] && score>below[col]) {
        int target=atomicAdd(counts+k,1);
        p.selected[target]={p.template_index,x,y,score};
    }
}

__global__ void prepare_fused_tolerances(const FusedMapParameters* maps,const double* maximum_norm,
                                        double* tolerances,int count) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if (k>=count) return;
    double maximum=*maximum_norm*maps[k].template_norm;
    tolerances[k]=maximum>0 ? sqrt(ldexp(1.0,ilogb(maximum)-52))/maps[k].template_norm : 0;
}

__global__ void normalize_fft_maps_fused(const float* inverse,int inverse_stride,int inverse_count,
                                        float scale,const FusedMapParameters* maps,
                                        const double* sums,const double* norms,const double* tolerances,
                                        int width,int height,int template_count) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=width*height) return;
    int row=index/width;
    int col=index%width;
    double norm=norms[index],sum=sums[index];
    float inverse_norm=norm>0 ? float(1.0/norm) : 0;
    for (int k=0;k<template_count;++k) {
        FusedMapParameters p=maps[k];
        float centered=float(sum*p.residual_mean);
        float numerator=inverse[k*inverse_count+row*inverse_stride+col]*scale-centered;
        float value=norm>tolerances[k] ? numerator*inverse_norm*p.inverse_template_norm : 0;
        float* output=reinterpret_cast<float*>(reinterpret_cast<char*>(p.output)+row*p.step);
        output[col]=fabs(double(value))-1>sqrt(0x1p-52) ? 0 : value;
    }
}

class CachedGpuFftSearch {
    Config cfg;
    const int template_count;
    cv::cuda::Stream& stream;
    cudaStream_t native_stream;
    int fft_width,fft_height,real_count,spectrum_count,width,height,output_count;
    GpuWindowStatistics statistics;
    FftPlan forward,inverse;
    DeviceBuffer<float> padded_image,inverse_maps;
    DeviceBuffer<cufftComplex> image_spectrum,template_spectra,products;
    DeviceBuffer<uint8_t> flags,workspace;
    DeviceBuffer<Candidate> dense;
    std::vector<DeviceBuffer<Candidate>> selected;
    DeviceBuffer<int> counts;
    size_t workspace_bytes=0;
    std::vector<cv::cuda::GpuMat> maps;
    std::vector<double> residual_means,template_norms;
    bool use_tiled_statistics;
    bool use_fused_normalization;
    DeviceBuffer<FusedMapParameters> fused_parameters;
    DeviceBuffer<double> fused_tolerances;
    DeviceBuffer<PeakMapParameters> peak_parameters;
    bool use_atomic_peaks;
    bool use_coalesced_statistics;
public:
    CachedGpuFftSearch(const Config& config,const std::vector<GpuTemplate>& templates,
                      cv::cuda::Stream& cuda_stream,bool tiled_statistics=false,bool fused_normalization=false,
                      bool atomic_peaks=false,bool coalesced_statistics=false)
        : cfg(config),template_count(cfg.count),stream(cuda_stream),native_stream(cv::cuda::StreamAccessor::getStream(stream)),
          fft_width(smooth_fft_size(cfg.width+cfg.template_width-1)),
          fft_height(smooth_fft_size(cfg.height+cfg.template_height-1)),
          real_count(checked_product(fft_width,fft_height)),spectrum_count(checked_product(fft_height,fft_width/2+1)),
          width(cfg.width-cfg.template_width+1),height(cfg.height-cfg.template_height+1),
          output_count(width*height),statistics(config),selected(template_count),maps(template_count),
          residual_means(template_count),template_norms(template_count),use_tiled_statistics(tiled_statistics),
          use_fused_normalization(fused_normalization),use_atomic_peaks(atomic_peaks),
          use_coalesced_statistics(coalesced_statistics) {
        checked_product(template_count,real_count);
        checked_product(template_count,spectrum_count);
        padded_image.allocate(real_count);
        image_spectrum.allocate(spectrum_count);
        template_spectra.allocate(template_count*spectrum_count);
        products.allocate(template_count*spectrum_count);
        inverse_maps.allocate(template_count*real_count);
        flags.allocate(output_count);
        dense.allocate(output_count);
        counts.allocate(template_count);
        for (int k=0;k<template_count;++k) {
            selected[k].allocate(output_count);
            maps[k].create(height,width,CV_32FC1);
            residual_means[k]=templates[k].residual_mean;
            template_norms[k]=templates[k].norm;
        }
        if (use_fused_normalization) {
            std::vector<FusedMapParameters> parameters(template_count);
            for (int k=0;k<template_count;++k) parameters[k]={maps[k].ptr<float>(),maps[k].step,
                residual_means[k],template_norms[k],float(1.0/template_norms[k])};
            fused_parameters.allocate(template_count);
            fused_tolerances.allocate(template_count);
            check_cuda(cudaMemcpy(fused_parameters.data,parameters.data(),parameters.size()*sizeof(FusedMapParameters),cudaMemcpyHostToDevice));
        }
        if (use_atomic_peaks) {
            std::vector<PeakMapParameters> parameters(template_count);
            for (int k=0;k<template_count;++k) parameters[k]={maps[k].ptr<float>(),maps[k].step,
                selected[k].data,cfg.indices[k]};
            peak_parameters.allocate(template_count);
            check_cuda(cudaMemcpy(peak_parameters.data,parameters.data(),parameters.size()*sizeof(PeakMapParameters),cudaMemcpyHostToDevice));
        }
        check_cuda(cub::DeviceSelect::Flagged(nullptr,workspace_bytes,dense.data,flags.data,
            selected[0].data,counts.data,output_count,native_stream));
        workspace.allocate(workspace_bytes);
        int dimensions[2]={fft_height,fft_width};
        check_fft(cufftPlan2d(&forward.handle,fft_height,fft_width,CUFFT_R2C));
        check_fft(cufftPlanMany(&inverse.handle,2,dimensions,nullptr,1,spectrum_count,
            nullptr,1,real_count,CUFFT_C2R,template_count));
        check_fft(cufftSetStream(forward.handle,native_stream));
        check_fft(cufftSetStream(inverse.handle,native_stream));
        DeviceBuffer<float> packed;
        packed.allocate(template_count*real_count);
        check_cuda(cudaMemsetAsync(packed.data,0,template_count*real_count*sizeof(float),native_stream));
        for (int k=0;k<template_count;++k) {
            pack_template<<<(cfg.template_width*cfg.template_height+255)/256,256,0,native_stream>>>(
                templates[k].pixels.ptr<float>(),templates[k].pixels.step,packed.data+k*real_count,
                cfg.template_width,cfg.template_height,fft_width);
        }
        FftPlan template_forward;
        check_fft(cufftPlanMany(&template_forward.handle,2,dimensions,nullptr,1,real_count,
            nullptr,1,spectrum_count,CUFFT_R2C,template_count));
        check_fft(cufftSetStream(template_forward.handle,native_stream));
        check_fft(cufftExecR2C(template_forward.handle,packed.data,template_spectra.data));
        stream.waitForCompletion();
    }

    Result run(const cv::cuda::GpuMat& image) {
        Result result(template_count);
        auto started=Clock::now();
        auto part=Clock::now();
        if (use_coalesced_statistics) statistics.compute_coalesced(image,stream);
        else if (use_tiled_statistics) statistics.compute_tiled(image,stream);
        else statistics.compute(image,stream);
        result.normalization_s=elapsed(part);
        part=Clock::now();
        pad_image<<<(real_count+255)/256,256,0,native_stream>>>(
            image.ptr<float>(),image.step,padded_image.data,cfg.width,cfg.height,fft_width,fft_height);
        check_fft(cufftExecR2C(forward.handle,padded_image.data,image_spectrum.data));
        multiply_templates<<<(spectrum_count*template_count+255)/256,256,0,native_stream>>>(
            image_spectrum.data,template_spectra.data,products.data,spectrum_count,template_count);
        check_fft(cufftExecC2R(inverse.handle,products.data,inverse_maps.data));
        stream.waitForCompletion();
        result.correlation_s=elapsed(part);
        part=Clock::now();
        if (use_fused_normalization) {
            prepare_fused_tolerances<<<(template_count+255)/256,256,0,native_stream>>>(
                fused_parameters.data,statistics.maximum_norm.data,fused_tolerances.data,template_count);
            normalize_fft_maps_fused<<<(output_count+255)/256,256,0,native_stream>>>(inverse_maps.data,
                fft_width,real_count,1.0f/real_count,fused_parameters.data,statistics.sums.data,
                statistics.norms.data,fused_tolerances.data,width,height,template_count);
        } else for (int k=0;k<template_count;++k) {
            normalize_fft_map<<<(output_count+255)/256,256,0,native_stream>>>(
                inverse_maps.data+k*real_count,fft_width,1.0f/real_count,maps[k].ptr<float>(),maps[k].step,
                statistics.sums.data,statistics.norms.data,statistics.maximum_norm.data,width,height,
                residual_means[k],template_norms[k]);
        }
        stream.waitForCompletion();
        result.normalization_s+=elapsed(part);
        part=Clock::now();
        if (use_atomic_peaks) {
            check_cuda(cudaMemsetAsync(counts.data,0,template_count*sizeof(int),native_stream));
            collect_peaks_atomic<<<dim3((output_count+255)/256,template_count),256,0,native_stream>>>(
                peak_parameters.data,counts.data,width,height,(cfg.template_width+1)/2,
                (cfg.template_height+1)/2,cfg.min_x,cfg.max_x,cfg.min_y,cfg.max_y,float(cfg.threshold));
            for (int k=0;k<template_count;++k) result.device_maps[k]=maps[k];
        } else for (int k=0;k<template_count;++k) {
            flag_peaks<<<(output_count+255)/256,256,0,native_stream>>>(
                maps[k].ptr<float>(),maps[k].step,dense.data,flags.data,width,height,cfg.indices[k],
                (cfg.template_width+1)/2,(cfg.template_height+1)/2,
                cfg.min_x,cfg.max_x,cfg.min_y,cfg.max_y,cfg.threshold);
            check_cuda(cub::DeviceSelect::Flagged(workspace.data,workspace_bytes,dense.data,flags.data,
                selected[k].data,counts.data+k,output_count,native_stream));
            result.device_maps[k]=maps[k];
        }
        stream.waitForCompletion();
        check_cuda(cudaGetLastError());
        result.peaks_s=elapsed(part);
        result.stage_s=elapsed(started);
        auto download_started=Clock::now();
        std::vector<int> sizes(template_count);
        check_cuda(cudaMemcpy(sizes.data(),counts.data,sizes.size()*sizeof(int),cudaMemcpyDeviceToHost));
        for (int k=0;k<template_count;++k) {
            result.candidates[k].resize(sizes[k]);
            if (sizes[k]>0) {
                check_cuda(cudaMemcpy(result.candidates[k].data(),selected[k].data,
                    sizes[k]*sizeof(Candidate),cudaMemcpyDeviceToHost));
            }
        }
        result.download_s=elapsed(download_started);
        return result;
    }
};


}} // namespace defocus::cuda_detail
