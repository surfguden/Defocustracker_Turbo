// Tiled CPU candidate engine; project/upstream attribution is in THIRD_PARTY_NOTICES.md.
#include "defocus/tracker.hpp"
#include "cpu_execution.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <vector>
namespace defocus { namespace cpu_detail {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
struct Config {int width,height,template_width,template_height,count; double threshold;
    int min_x,max_x,min_y,max_y; std::vector<int> indices;};
struct Candidate {int template_index,x,y; float score;};
struct Result {
    std::vector<cv::Mat> host_maps;
    std::vector<std::vector<Candidate>> candidates;
    double correlation_s=0,normalization_s=0,peaks_s=0,stage_s=0;
    explicit Result(int n):host_maps(n),candidates(n){}
};
std::vector<Candidate> rowwise_cpu_peaks(const cv::Mat& map,const Config& cfg,int template_index) {
    std::vector<Candidate> candidates;
    int offset_x=(cfg.template_width+1)/2;
    int offset_y=(cfg.template_height+1)/2;
    int first_row=std::max(1,cfg.min_y-offset_y);
    int last_row=std::min(map.rows-2,cfg.max_y-offset_y);
    int first_col=std::max(1,cfg.min_x-offset_x);
    int last_col=std::min(map.cols-2,cfg.max_x-offset_x);
    for (int row=first_row;row<=last_row;++row) {
        const float* current=map.ptr<float>(row);
        const float* above=map.ptr<float>(row-1);
        const float* below=map.ptr<float>(row+1);
        for (int col=first_col;col<=last_col;++col) {
            float value=current[col];
            if (value>cfg.threshold && value>current[col-1] && value>current[col+1] &&
                value>above[col] && value>below[col]) {
                candidates.push_back({template_index,col+offset_x,row+offset_y,value});
            }
        }
    }
    return candidates;
}

class CachedCpuFftSearch {
    Config cfg;
    const int template_count;
    int fft_width,fft_height,width,height;
    cv::Mat padded_image,image_spectrum,integral_sum,integral_square,sums,inverse_norm;
    std::vector<cv::Mat> spectra,products,inverses,maps;
    std::vector<float> residual_means,inverse_template_norms;
    bool packed_transform,parallel_integrals;
    int tile_size;
    bool use_avx2;
    struct TileWorkspace { cv::Mat image,spectrum,product,inverse; };
    std::vector<TileWorkspace> tiles;
    void compute_parallel_integrals(const cv::Mat& image) {
        integral_sum.create(cfg.height+1,cfg.width+1,CV_64FC1);
        integral_square.create(cfg.height+1,cfg.width+1,CV_64FC1);
        integral_sum.row(0).setTo(0);
        integral_square.row(0).setTo(0);
        cv::parallel_for_(cv::Range(0,cfg.height),[&](const cv::Range& range) {
            for (int row=range.start;row<range.end;++row) {
                const float* input=image.ptr<float>(row);
                double* sums=integral_sum.ptr<double>(row+1);
                double* squares=integral_square.ptr<double>(row+1);
                sums[0]=0;
                squares[0]=0;
                double sum=0,square=0;
                for (int col=0;col<cfg.width;++col) {
                    double pixel=input[col];
                    sum+=pixel;
                    square+=pixel*pixel;
                    sums[col+1]=sum;
                    squares[col+1]=square;
                }
            }
        });
        if (use_avx2) {
            cv::parallel_for_(cv::Range(0,(cfg.width+31)/32),[&](const cv::Range& range) {
                for (int block=range.start;block<range.end;++block) {
                    int first=1+block*32,columns=std::min(32,cfg.width+1-first);
                    std::array<double,32> sums{},squares{};
                    for (int row=1;row<=cfg.height;++row) {
                        double* values=integral_sum.ptr<double>(row)+first;
                        double* squared=integral_square.ptr<double>(row)+first;
                        for (int c=0;c<columns;++c) {
                            sums[c]+=values[c]; squares[c]+=squared[c];
                            values[c]=sums[c]; squared[c]=squares[c];
                        }
                    }
                }
            });
        } else cv::parallel_for_(cv::Range(1,cfg.width+1),[&](const cv::Range& range) {
            for (int col=range.start;col<range.end;++col) {
                double sum=0,square=0;
                for (int row=1;row<=cfg.height;++row) {
                    double& value=integral_sum.at<double>(row,col);
                    double& squared=integral_square.at<double>(row,col);
                    sum+=value;
                    square+=squared;
                    value=sum;
                    squared=square;
                }
            }
        });
    }
public:
    CachedCpuFftSearch(const Config& config,const std::vector<cv::Mat>& templates,
                       bool packed=false,bool parallel_statistics=false,int tiled_fft_size=0,bool avx2=false)
        : cfg(config),template_count(cfg.count),fft_width(tiled_fft_size ? tiled_fft_size : cv::getOptimalDFTSize(cfg.width+cfg.template_width-1)),
          fft_height(tiled_fft_size ? tiled_fft_size : cv::getOptimalDFTSize(cfg.height+cfg.template_height-1)),
          width(cfg.width-cfg.template_width+1),height(cfg.height-cfg.template_height+1),
          spectra(template_count),products(template_count),inverses(template_count),maps(template_count),
          residual_means(template_count),inverse_template_norms(template_count),
          packed_transform(packed),parallel_integrals(parallel_statistics),tile_size(tiled_fft_size),use_avx2(avx2) {
        if (use_avx2 && !cv::checkHardwareSupport(CV_CPU_AVX2))
            throw std::runtime_error("This experimental CPU method requires AVX2");
        padded_image=cv::Mat::zeros(fft_height,fft_width,CV_32FC1);
        image_spectrum.create(fft_height,fft_width,packed_transform ? CV_32FC1 : CV_32FC2);
        sums.create(height,width,CV_32FC1);
        inverse_norm.create(height,width,CV_32FC1);
        if (tile_size) {
            int tile_width=fft_width-cfg.template_width+1,tile_height=fft_height-cfg.template_height+1;
            tiles.resize(((width+tile_width-1)/tile_width)*((height+tile_height-1)/tile_height));
            for (auto& tile:tiles) {
                tile.image=cv::Mat::zeros(fft_height,fft_width,CV_32FC1);
                tile.spectrum.create(fft_height,fft_width,CV_32FC1);
                tile.product.create(fft_height,fft_width,CV_32FC1);
                tile.inverse.create(fft_height,fft_width,CV_32FC1);
            }
        }
        cv::parallel_for_(cv::Range(0,template_count),[&](const cv::Range& range) {
            for (int k=range.start;k<range.end;++k) {
                cv::Mat padded=cv::Mat::zeros(fft_height,fft_width,CV_32FC1);
                templates[k].copyTo(padded(cv::Rect(0,0,cfg.template_width,cfg.template_height)));
                cv::dft(padded,spectra[k],packed_transform ? 0 : cv::DFT_COMPLEX_OUTPUT);
                products[k].create(fft_height,fft_width,packed_transform ? CV_32FC1 : CV_32FC2);
                inverses[k].create(fft_height,fft_width,CV_32FC1);
                maps[k].create(height,width,CV_32FC1);
                cv::Scalar mean,standard_deviation;
                cv::meanStdDev(templates[k],mean,standard_deviation);
                residual_means[k]=static_cast<float>(mean[0]);
                inverse_template_norms[k]=static_cast<float>(
                    1.0/(standard_deviation[0]*std::sqrt(double(templates[k].total()))));
            }
        });
    }
    Result run(const cv::Mat& image) {
        Result result(template_count);
        auto started=Clock::now();
        auto part=Clock::now();
        if (parallel_integrals) compute_parallel_integrals(image);
        else cv::integral(image,integral_sum,integral_square,CV_64F,CV_64F);
        cv::parallel_for_(cv::Range(0,height),[&](const cv::Range& range) {
            for (int row=range.start;row<range.end;++row) {
                const double* top=integral_sum.ptr<double>(row);
                const double* bottom=integral_sum.ptr<double>(row+cfg.template_height);
                const double* top_square=integral_square.ptr<double>(row);
                const double* bottom_square=integral_square.ptr<double>(row+cfg.template_height);
                float* sum_output=sums.ptr<float>(row);
                float* norm_output=inverse_norm.ptr<float>(row);
                int col=0;
                if (use_avx2) {
                    __m256d count=_mm256_set1_pd(double(cfg.template_width*cfg.template_height));
                    __m256d zero=_mm256_setzero_pd(),one=_mm256_set1_pd(1.0);
                    for (;col+3<width;col+=4) {
                        int end=col+cfg.template_width;
                        __m256d sum=_mm256_add_pd(_mm256_sub_pd(_mm256_sub_pd(
                            _mm256_loadu_pd(bottom+end),_mm256_loadu_pd(bottom+col)),
                            _mm256_loadu_pd(top+end)),_mm256_loadu_pd(top+col));
                        __m256d square=_mm256_add_pd(_mm256_sub_pd(_mm256_sub_pd(
                            _mm256_loadu_pd(bottom_square+end),_mm256_loadu_pd(bottom_square+col)),
                            _mm256_loadu_pd(top_square+end)),_mm256_loadu_pd(top_square+col));
                        __m256d variance=_mm256_max_pd(_mm256_sub_pd(square,
                            _mm256_div_pd(_mm256_mul_pd(sum,sum),count)),zero);
                        __m256d inverse=_mm256_and_pd(_mm256_div_pd(one,_mm256_sqrt_pd(variance)),
                            _mm256_cmp_pd(variance,zero,_CMP_GT_OQ));
                        _mm_storeu_ps(sum_output+col,_mm256_cvtpd_ps(sum));
                        _mm_storeu_ps(norm_output+col,_mm256_cvtpd_ps(inverse));
                    }
                }
                for (;col<width;++col) {
                    int end=col+cfg.template_width;
                    double sum=bottom[end]-bottom[col]-top[end]+top[col];
                    double square=bottom_square[end]-bottom_square[col]-top_square[end]+top_square[col];
                    double variance=std::max(square-sum*sum/(cfg.template_width*cfg.template_height),0.0);
                    sum_output[col]=static_cast<float>(sum);
                    norm_output[col]=variance>0 ? static_cast<float>(1.0/std::sqrt(variance)) : 0;
                }
            }
        });
        result.normalization_s=elapsed(part);
        part=Clock::now();
        if (tile_size) {
            int tile_width=fft_width-cfg.template_width+1,tile_height=fft_height-cfg.template_height+1;
            int columns=(width+tile_width-1)/tile_width;
            cv::parallel_for_(cv::Range(0,int(tiles.size())),[&](const cv::Range& range) {
                for (int tile_index=range.start;tile_index<range.end;++tile_index) {
                    auto& tile=tiles[tile_index];
                    int x=(tile_index%columns)*tile_width,y=(tile_index/columns)*tile_height;
                    int input_width=std::min(fft_width,cfg.width-x),input_height=std::min(fft_height,cfg.height-y);
                    image(cv::Rect(x,y,input_width,input_height)).copyTo(tile.image(cv::Rect(0,0,input_width,input_height)));
                    cv::dft(tile.image,tile.spectrum,0);
                    cv::Rect source(0,0,std::min(tile_width,width-x),std::min(tile_height,height-y));
                    cv::Rect target(x,y,source.width,source.height);
                    for (int k=0;k<template_count;++k) {
                        cv::mulSpectrums(tile.spectrum,spectra[k],tile.product,0,true);
                        cv::dft(tile.product,tile.inverse,cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT|cv::DFT_SCALE);
                        tile.inverse(source).copyTo(maps[k](target));
                    }
                }
            });
        } else {
            image.copyTo(padded_image(cv::Rect(0,0,cfg.width,cfg.height)));
            cv::dft(padded_image,image_spectrum,packed_transform ? 0 : cv::DFT_COMPLEX_OUTPUT);
            cv::parallel_for_(cv::Range(0,template_count),[&](const cv::Range& range) {
                for (int k=range.start;k<range.end;++k) {
                    cv::mulSpectrums(image_spectrum,spectra[k],products[k],0,true);
                    cv::dft(products[k],inverses[k],cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT|cv::DFT_SCALE);
                }
            });
        }
        result.correlation_s=elapsed(part);
        part=Clock::now();
        cv::parallel_for_(cv::Range(0,height),[&](const cv::Range& range) {
            for (int row=range.start;row<range.end;++row) {
                const float* sum=sums.ptr<float>(row);
                const float* inv_norm=inverse_norm.ptr<float>(row);
                for (int k=0;k<template_count;++k) {
                    const float* values=tile_size ? maps[k].ptr<float>(row) : inverses[k].ptr<float>(row);
                    float* output=maps[k].ptr<float>(row);
                    float mean=residual_means[k];
                    float inv_template_norm=inverse_template_norms[k];
                    int col=0;
                    if (use_avx2) {
                        __m256 average=_mm256_set1_ps(mean),scale=_mm256_set1_ps(inv_template_norm);
                        __m256 sign=_mm256_set1_ps(-0.0f),one=_mm256_set1_ps(1.0f);
                        for (;col+7<width;col+=8) {
                            __m256 numerator=_mm256_sub_ps(_mm256_loadu_ps(values+col),
                                _mm256_mul_ps(_mm256_loadu_ps(sum+col),average));
                            __m256 score=_mm256_mul_ps(_mm256_mul_ps(numerator,
                                _mm256_loadu_ps(inv_norm+col)),scale);
                            __m256 invalid=_mm256_cmp_ps(_mm256_andnot_ps(sign,score),one,_CMP_GT_OQ);
                            _mm256_storeu_ps(output+col,_mm256_andnot_ps(invalid,score));
                        }
                    }
                    for (;col<width;++col) {
                        float score=(values[col]-sum[col]*mean)*inv_norm[col]*inv_template_norm;
                        output[col]=std::abs(score)>1.000000015 ? 0.0f : score;
                    }
                }
            }
        });
        result.normalization_s+=elapsed(part);
        part=Clock::now();
        cv::parallel_for_(cv::Range(0,template_count),[&](const cv::Range& range) {
            for (int k=range.start;k<range.end;++k) {
                result.candidates[k]=rowwise_cpu_peaks(maps[k],cfg,cfg.indices[k]);
                result.host_maps[k]=maps[k];
            }
        });
        result.peaks_s=elapsed(part);
        result.stage_s=elapsed(started);
        return result;
    }
};


} // namespace cpu_detail
namespace {
class CpuCandidates final:public CandidateBackend {
    const Model& model_;
    cpu_detail::Config cfg_{};
    std::unique_ptr<cpu_detail::CachedCpuFftSearch> engine_;
public:
    explicit CpuCandidates(const Model& model):model_(model) {
        CpuExecution execution(model.threads);
        cfg_.width=model.frame_width; cfg_.height=model.frame_height;
        cfg_.template_width=model.templates[0].width; cfg_.template_height=model.templates[0].height;
        cfg_.count=int(model.selected.size()); cfg_.threshold=model.cm_guess; cfg_.indices=model.selected;
        cfg_.min_x=int(std::ceil(2+(cfg_.template_width+8)/2.0));
        cfg_.max_x=int(std::floor(cfg_.width-(cfg_.template_width+8)/2.0-1));
        cfg_.min_y=int(std::ceil(2+(cfg_.template_height+8)/2.0));
        cfg_.max_y=int(std::floor(cfg_.height-(cfg_.template_height+8)/2.0-1));
        std::vector<cv::Mat> templates(cfg_.count);
        for (int k=0;k<cfg_.count;++k) {
            const auto& raw=model.templates[cfg_.indices[k]-1];
            cv::Mat temp(raw.height,raw.width,CV_32FC1);
            for (int y=0;y<raw.height;++y) for(int x=0;x<raw.width;++x) temp.at<float>(y,x)=float(raw.at(x,y));
            double mean=cv::mean(temp)[0]; templates[k].create(raw.height,raw.width,CV_32FC1);
            for (int y=0;y<raw.height;++y) for(int x=0;x<raw.width;++x)
                templates[k].at<float>(y,x)=float(temp.at<float>(y,x)-mean);
        }
        int tile=cfg_.template_width<512 && cfg_.template_height<512 ? 512 : 0;
        engine_=std::make_unique<cpu_detail::CachedCpuFftSearch>(cfg_,templates,true,true,tile,cv::checkHardwareSupport(CV_CPU_AVX2));
    }
    SearchResult search(const Image& image) override {
        CpuExecution execution(model_.threads);
        auto started=cpu_detail::Clock::now();
        cv::Mat input(image.height,image.width,CV_32FC1);
        cv::Mat original(image.height,image.width,CV_64FC1,const_cast<double*>(image.pixels.data()));
        original.convertTo(input,CV_32FC1);
        auto native=engine_->run(input); SearchResult result;
        for (auto& group:native.candidates) {
            std::sort(group.begin(),group.end(),[](const auto& a,const auto& b){return a.x==b.x ? a.y<b.y : a.x<b.x;});
            for (const auto& c:group) {
                result.candidates.push_back({c.template_index,double(c.x),double(c.y),double(c.score)});
                for (const auto& map:native.host_maps)
                    result.similarities.push_back(map.at<float>(c.y-(cfg_.template_height+1)/2,c.x-(cfg_.template_width+1)/2));
            }
        }
        result.seconds=cpu_detail::elapsed(started); return result;
    }
};
}
std::unique_ptr<CandidateBackend> make_cpu_candidates(const Model& model) {return std::make_unique<CpuCandidates>(model);}
} // namespace defocus
