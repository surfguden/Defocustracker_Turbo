#include "defocus/tracker.hpp"
#include "cpu_execution.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <immintrin.h>

namespace defocus { namespace {
struct TemplateState { cv::Mat spectrum; std::vector<double> centered; double norm=0,residual=0; };
class CpuCorrelation final:public CorrelationBackend {
    const Model& model_;
    std::vector<TemplateState> templates_;
    int fft_width_,fft_height_;
public:
    explicit CpuCorrelation(const Model& model):model_(model),templates_(model.templates.size()) {
        CpuExecution execution(model.threads);
        int tw=model.templates[0].width,th=model.templates[0].height;
        fft_width_=cv::getOptimalDFTSize(2*tw+7); fft_height_=cv::getOptimalDFTSize(2*th+7);
        for (size_t k=0;k<templates_.size();++k) {
            auto& state=templates_[k]; const auto& image=model.templates[k];
            state.centered.resize(image.pixels.size());
            double mean=0;
            for (int x=0;x<tw;++x) for (int y=0;y<th;++y) mean+=image.at(x,y);
            mean/=double(tw*th);
            cv::Mat centered=cv::Mat::zeros(fft_height_,fft_width_,CV_64FC1);
            double sum=0,square=0;
            for (int x=0;x<tw;++x) for (int y=0;y<th;++y) {
                double value=image.at(x,y)-mean;
                centered.at<double>(y,x)=value; sum+=value; square+=value*value;
                state.centered[size_t(y)*tw+x]=value;
            }
            state.residual=sum/(tw*th);
            state.norm=std::sqrt(square-sum*sum/(tw*th));
            cv::dft(centered,state.spectrum,cv::DFT_COMPLEX_OUTPUT);
        }
    }
    std::string name() const override {return "cpu";}
    Image correlate(const Image& image,int index,CorrelationMode mode) override {
        CpuExecution execution(model_.threads);
        if (index<1 || index>int(templates_.size())) throw std::invalid_argument("Correlation template index outside calibration");
        int tw=model_.templates[0].width,th=model_.templates[0].height;
        if (image.width<tw || image.height<th) throw std::invalid_argument("Correlation crop is smaller than its template");
        int fw=image.width+tw-1,fh=image.height+th-1;
        auto& state=templates_[index-1];
        Image result(fw,fh);
        if (mode==CorrelationMode::legacy_padded_valid) {
                int stride=image.width+1;
                std::vector<double> sums(size_t(stride)*(image.height+1),0),squares(sums.size(),0);
                for (int y=0;y<image.height;++y) {
                    double sum=0,square=0;
                    for (int x=0;x<image.width;++x) {
                        double v=image.at(x,y); sum+=v; square+=v*v;
                        sums[size_t(y+1)*stride+x+1]=sums[size_t(y)*stride+x+1]+sum;
                        squares[size_t(y+1)*stride+x+1]=squares[size_t(y)*stride+x+1]+square;
                    }
                }
                int columns=image.width-tw+1,rows=image.height-th+1;
                bool avx=cv::checkHardwareSupport(CV_CPU_AVX2);
                auto compute=[&](const cv::Range& range) {
                    for (int i=range.start;i<range.end;++i) {
                        int x=i%columns,y=i/columns;
                        auto window=[&](const auto& values) {
                            return values[size_t(y+th)*stride+x+tw]-values[size_t(y)*stride+x+tw]
                                -values[size_t(y+th)*stride+x]+values[size_t(y)*stride+x];
                        };
                        double sum=window(sums),square=window(squares),dot=0;
                        if (avx) {
                            __m256d a=_mm256_setzero_pd(),b=_mm256_setzero_pd();
                            for (int yy=0;yy<th;++yy) {
                                const double* p=image.pixels.data()+size_t(y+yy)*image.width+x;
                                const double* t=state.centered.data()+size_t(yy)*tw;
                                int xx=0;
                                for (;xx+7<tw;xx+=8) {
                                    a=_mm256_add_pd(a,_mm256_mul_pd(_mm256_loadu_pd(p+xx),_mm256_loadu_pd(t+xx)));
                                    b=_mm256_add_pd(b,_mm256_mul_pd(_mm256_loadu_pd(p+xx+4),_mm256_loadu_pd(t+xx+4)));
                                }
                                for (;xx<tw;++xx) dot+=p[xx]*t[xx];
                            }
                            alignas(32) double totals[4]; _mm256_store_pd(totals,_mm256_add_pd(a,b));
                            dot+=totals[0]+totals[1]+totals[2]+totals[3];
                        } else {
                            for (int yy=0;yy<th;++yy) for (int xx=0;xx<tw;++xx)
                                dot+=image.at(x+xx,y+yy)*state.centered[size_t(yy)*tw+xx];
                        }
                        double denominator=state.norm*std::sqrt(std::max(square-sum*sum/(tw*th),0.0));
                        double value=denominator>0 ? (dot-sum*state.residual)/denominator : 0;
                        result.at(x+tw-1,y+th-1)=std::abs(value)-1>std::sqrt(std::numeric_limits<double>::epsilon()) ? 0 : value;
                    }
                };
                cv::parallel_for_(cv::Range(0,columns*rows),compute);
                return result;
        }
        if (fw>fft_width_ || fh>fft_height_)
            throw std::invalid_argument("Correlation crop exceeds the prepared local FFT dimensions");
        double minimum=*std::min_element(image.pixels.begin(),image.pixels.end());
        double shift=minimum<0 ? -minimum : 0;
        cv::Mat padded=cv::Mat::zeros(fft_height_,fft_width_,CV_64FC1),spectrum,product,inverse;
        int stride=image.width+1;
        std::vector<double> sums(size_t(stride)*(image.height+1),0),squares(sums.size(),0);
        for (int y=0;y<image.height;++y) {
            double sum=0,square=0;
            for (int x=0;x<image.width;++x) {
                double value=image.at(x,y)+shift;
                padded.at<double>(y,x)=value; sum+=value; square+=value*value;
                sums[size_t(y+1)*stride+x+1]=sums[size_t(y)*stride+x+1]+sum;
                squares[size_t(y+1)*stride+x+1]=squares[size_t(y)*stride+x+1]+square;
            }
        }
        cv::dft(padded,spectrum,cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(spectrum,state.spectrum,product,0,true);
        cv::dft(product,inverse,cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT|cv::DFT_SCALE);
        std::vector<double> denominators(result.pixels.size()),window_sums(result.pixels.size());
        double maximum=0,count=tw*th;
        for (int y=0;y<fh;++y) for (int x=0;x<fw;++x) {
            int left=std::max(0,x-tw+1),right=std::min(image.width,x+1);
            int top=std::max(0,y-th+1),bottom=std::min(image.height,y+1);
            auto window=[&](const std::vector<double>& p) {
                return p[size_t(bottom)*stride+right]-p[size_t(top)*stride+right]
                    -p[size_t(bottom)*stride+left]+p[size_t(top)*stride+left];
            };
            double sum=window(sums),square=window(squares);
            size_t i=size_t(y)*fw+x;
            window_sums[i]=sum;
            denominators[i]=state.norm*std::sqrt(std::max(square-sum*sum/count,0.0));
            maximum=std::max(maximum,denominators[i]);
        }
        double epsilon=maximum>0 ? std::ldexp(1.0,std::ilogb(maximum)-52) : std::numeric_limits<double>::denorm_min();
        double tolerance=std::sqrt(epsilon);
        for (int y=0;y<fh;++y) for (int x=0;x<fw;++x) {
            size_t i=size_t(y)*fw+x;
            int ix=(x-tw+1+fft_width_)%fft_width_,iy=(y-th+1+fft_height_)%fft_height_;
            double numerator=inverse.at<double>(iy,ix)-window_sums[i]*state.residual;
            double value=denominators[i]>tolerance ? numerator/denominators[i] : 0;
            result.pixels[i]=std::abs(value)-1>std::sqrt(std::numeric_limits<double>::epsilon()) ? 0 : value;
        }
        return result;
    }
};
} // namespace
std::unique_ptr<CorrelationBackend> make_cpu_correlation(const Model& model) {
    return std::make_unique<CpuCorrelation>(model);
}
} // namespace defocus
