#include "cpu_refresh.hpp"
#include "cpu_execution.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <immintrin.h>
namespace defocus {namespace {
struct Template {cv::Mat spectrum;std::vector<double> centered;double norm=0,residual=0;};
struct Bank {std::vector<Template> templates;int tw,th,fw,fh;bool avx;};
class Cached final:public CorrelationBackend {
    cpu_refresh::Scratch scratch_;
    const Model& model_;std::shared_ptr<const Bank> bank_;const Image* crop_=nullptr;
    bool prefix_ready_[2]{false,false},fft_ready_=false;double shift_=0,max_norm_=0,full_tolerance_=0;
    std::vector<double> sums_[2],squares_[2],window_sums_,window_norms_,compact_;
    cv::Mat padded_,spectrum_,product_,inverse_;
    std::vector<Peak> cached_[2];std::vector<uint8_t> valid_[2];CorrelationMetrics metrics_;
    void allocate(){
        scratch_.crop=Image(bank_->tw+8,bank_->th+8);scratch_.median.reserve(scratch_.crop.pixels.size());
        size_t prefix_count=size_t(bank_->tw+9)*(bank_->th+9);for(int m=0;m<2;++m){sums_[m].reserve(prefix_count);squares_[m].reserve(prefix_count);}
        window_sums_.reserve(size_t(2*bank_->tw+7)*(2*bank_->th+7));window_norms_.reserve(window_sums_.capacity());compact_.reserve(81);
        for(int k=0;k<2;++k){cached_[k].resize(bank_->templates.size());valid_[k].resize(bank_->templates.size());}
        padded_.create(bank_->fh,bank_->fw,CV_64FC1);spectrum_.create(bank_->fh,bank_->fw,CV_64FC2);
        product_.create(bank_->fh,bank_->fw,CV_64FC2);inverse_.create(bank_->fh,bank_->fw,CV_64FC1);
    }
    void prefix(int mode){
        if(prefix_ready_[mode])return;const auto& im=*crop_;int stride=im.width+1;
        auto& sum=sums_[mode];auto& square=squares_[mode];sum.assign(size_t(stride)*(im.height+1),0);square.assign(sum.size(),0);
        double shift=0;if(!mode){double low=*std::min_element(im.pixels.begin(),im.pixels.end());shift_=low<0?-low:0;shift=shift_;padded_.setTo(0);}
        for(int y=0;y<im.height;++y){double a=0,b=0;for(int x=0;x<im.width;++x){
            double v=im.at(x,y)+shift;a+=v;b+=v*v;
            sum[size_t(y+1)*stride+x+1]=sum[size_t(y)*stride+x+1]+a;
            square[size_t(y+1)*stride+x+1]=square[size_t(y)*stride+x+1]+b;
            if(!mode)padded_.at<double>(y,x)=v;
        }}
        prefix_ready_[mode]=true;++metrics_.prefix_builds;
        if(!mode){
            int fw=im.width+bank_->tw-1,fh=im.height+bank_->th-1;window_sums_.resize(size_t(fw)*fh);window_norms_.resize(window_sums_.size());max_norm_=0;
            for(int y=0;y<fh;++y)for(int x=0;x<fw;++x){
                int l=std::max(0,x-bank_->tw+1),r=std::min(im.width,x+1),t=std::max(0,y-bank_->th+1),b=std::min(im.height,y+1);
                auto window=[&](const auto& p){return p[size_t(b)*stride+r]-p[size_t(t)*stride+r]-p[size_t(b)*stride+l]+p[size_t(t)*stride+l];};
                double a=window(sum),v=std::sqrt(std::max(window(square)-a*a/(bank_->tw*bank_->th),0.0));size_t i=size_t(y)*fw+x;
                window_sums_[i]=a;window_norms_[i]=v;max_norm_=std::max(max_norm_,v);
            }
        }
    }
    void full(int index){
        prefix(0);if(!fft_ready_){cv::dft(padded_,spectrum_,cv::DFT_COMPLEX_OUTPUT);fft_ready_=true;++metrics_.image_ffts;}
        double maximum=bank_->templates[index-1].norm*max_norm_;
        double epsilon=maximum>0?std::ldexp(1.0,std::ilogb(maximum)-52):std::numeric_limits<double>::denorm_min();full_tolerance_=std::sqrt(epsilon);
        cv::mulSpectrums(spectrum_,bank_->templates[index-1].spectrum,product_,0,true);
        cv::dft(product_,inverse_,cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT|cv::DFT_SCALE);++metrics_.inverse_ffts;
    }
    double full_value(int x,int y,int index)const{
        const auto& s=bank_->templates[index-1];int fw=crop_->width+bank_->tw-1;
        size_t i=size_t(y)*fw+x;
        int ix=(x-bank_->tw+1+bank_->fw)%bank_->fw,iy=(y-bank_->th+1+bank_->fh)%bank_->fh;
        double denominator=s.norm*window_norms_[i];double numerator=inverse_.at<double>(iy,ix)-window_sums_[i]*s.residual;
        double v=denominator>full_tolerance_?numerator/denominator:0;return std::abs(v)-1>std::sqrt(std::numeric_limits<double>::epsilon())?0:v;
    }
    void compact(int index){
        prefix(1);const auto& im=*crop_;const auto& s=bank_->templates[index-1];int tw=bank_->tw,th=bank_->th,stride=im.width+1;
        int columns=im.width-tw+1,rows=im.height-th+1;compact_.resize(size_t(columns)*rows);
        for(int y=0;y<rows;++y)for(int x=0;x<columns;++x){
            auto window=[&](const auto& p){return p[size_t(y+th)*stride+x+tw]-p[size_t(y)*stride+x+tw]-p[size_t(y+th)*stride+x]+p[size_t(y)*stride+x];};
            double sum=window(sums_[1]),square=window(squares_[1]),dot=0;
            if(bank_->avx){__m256d a=_mm256_setzero_pd(),b=_mm256_setzero_pd();
                for(int yy=0;yy<th;++yy){const double* p=im.pixels.data()+size_t(y+yy)*im.width+x;const double* t=s.centered.data()+size_t(yy)*tw;int xx=0;
                    for(;xx+7<tw;xx+=8){a=_mm256_add_pd(a,_mm256_mul_pd(_mm256_loadu_pd(p+xx),_mm256_loadu_pd(t+xx)));b=_mm256_add_pd(b,_mm256_mul_pd(_mm256_loadu_pd(p+xx+4),_mm256_loadu_pd(t+xx+4)));}
                    for(;xx<tw;++xx)dot+=p[xx]*t[xx];}
                alignas(32) double totals[4];_mm256_store_pd(totals,_mm256_add_pd(a,b));dot+=totals[0]+totals[1]+totals[2]+totals[3];
            }else for(int yy=0;yy<th;++yy)for(int xx=0;xx<tw;++xx)dot+=im.at(x+xx,y+yy)*s.centered[size_t(yy)*tw+xx];
            double denominator=s.norm*std::sqrt(std::max(square-sum*sum/(tw*th),0.0));double value=denominator>0?(dot-sum*s.residual)/denominator:0;
            compact_[size_t(y)*columns+x]=std::abs(value)-1>std::sqrt(std::numeric_limits<double>::epsilon())?0:value;
        }
    }
    Peak summarize_current(int index,int mode){
        int fw=crop_->width+bank_->tw-1,fh=crop_->height+bank_->th-1;Peak p;p.score=-INFINITY;
        auto append=[&](int x,int y,double v){if(v>p.score){p.score=v;p.x=x+1;p.y=y+1;p.ties=p.sum_x=p.sum_y=0;}
            if(v==p.score){++p.ties;p.sum_x+=x+1;p.sum_y+=y+1;}};
        if(!mode){
            for(int x=0;x<fw;++x)for(int y=0;y<fh;++y)append(x,y,full_value(x,y,index));
            p.has_neighbors=p.x>1&&p.y>1&&p.x<fw&&p.y<fh;
            if(p.has_neighbors){p.left=full_value(p.x-2,p.y-1,index);p.right=full_value(p.x,p.y-1,index);p.above=full_value(p.x-1,p.y-2,index);p.below=full_value(p.x-1,p.y,index);}
        }else{
            int cols=crop_->width-bank_->tw+1,rows=crop_->height-bank_->th+1;
            for(int x=0;x<cols;++x)for(int y=0;y<rows;++y)append(x+bank_->tw-1,y+bank_->th-1,compact_[size_t(y)*cols+x]);
            if(p.score<=0){
                p={};p.score=0;p.x=p.y=1;p.ties=fw*fh;
                int64_t sx=int64_t(fh)*fw*(fw+1)/2,sy=int64_t(fw)*fh*(fh+1)/2;
                for(int x=0;x<cols;++x)for(int y=0;y<rows;++y)if(compact_[size_t(y)*cols+x]!=0){--p.ties;sx-=x+bank_->tw;sy-=y+bank_->th;}
                p.sum_x=int(sx);p.sum_y=int(sy);
            }else{
                int x=p.x-bank_->tw,y=p.y-bank_->th;
                auto v=[&](int a,int b){return a<0||b<0||a>=cols||b>=rows?0:compact_[size_t(b)*cols+a];};
                p.has_neighbors=true;p.left=v(x-1,y);p.right=v(x+1,y);p.above=v(x,y-1);p.below=v(x,y+1);
            }
        }
        return p;
    }
    void check_image(const Image& im){if(im.width<bank_->tw||im.height<bank_->th||im.width+bank_->tw-1>bank_->fw||im.height+bank_->th-1>bank_->fh)throw std::invalid_argument("Invalid CPU crop geometry");}
public:
    cpu_refresh::Scratch& scratch(){return scratch_;}
    explicit Cached(const Model& model):model_(model){
        CpuExecution execution(model.threads);auto b=std::make_shared<Bank>();b->tw=model.templates[0].width;b->th=model.templates[0].height;
        b->fw=cv::getOptimalDFTSize(2*b->tw+7);b->fh=cv::getOptimalDFTSize(2*b->th+7);b->avx=cv::checkHardwareSupport(CV_CPU_AVX2);b->templates.resize(model.templates.size());
        for(size_t k=0;k<b->templates.size();++k){auto& s=b->templates[k];const auto& image=model.templates[k];double mean=0;
            for(int x=0;x<b->tw;++x)for(int y=0;y<b->th;++y)mean+=image.at(x,y);mean/=b->tw*b->th;
            cv::Mat centered=cv::Mat::zeros(b->fh,b->fw,CV_64FC1);s.centered.resize(size_t(b->tw)*b->th);double sum=0,square=0;
            for(int x=0;x<b->tw;++x)for(int y=0;y<b->th;++y){double v=image.at(x,y)-mean;centered.at<double>(y,x)=v;s.centered[size_t(y)*b->tw+x]=v;sum+=v;square+=v*v;}
            s.residual=sum/(b->tw*b->th);s.norm=std::sqrt(square-sum*sum/(b->tw*b->th));cv::dft(centered,s.spectrum,cv::DFT_COMPLEX_OUTPUT);
        }bank_=std::move(b);allocate();
    }
    Cached(const Cached& other):model_(other.model_),bank_(other.bank_){allocate();}
    std::string name()const override{return "cpu_cached";}
    void begin_crop(const Image& im)override{check_image(im);crop_=&im;prefix_ready_[0]=prefix_ready_[1]=fft_ready_=false;for(auto& v:valid_)std::fill(v.begin(),v.end(),0);}
    void end_crop()noexcept override{crop_=nullptr;}
    void reset_metrics()override{metrics_={};}CorrelationMetrics metrics()const override{return metrics_;}
    Peak peak(const Image& im,int index,CorrelationMode mode)override{
        if(index<1||index>int(bank_->templates.size()))throw std::invalid_argument("Invalid template index");
        bool oneoff=!crop_;if(oneoff)begin_crop(im);else if(crop_!=&im)throw std::invalid_argument("CPU prepared crop changed without reset");
        int m=mode==CorrelationMode::legacy_padded_valid; ++metrics_.logical_requests;
        try{Peak p;if(valid_[m][index-1]){p=cached_[m][index-1];++metrics_.cache_hits;}else{
                if(m)compact(index);else full(index);++metrics_.template_evaluations;p=summarize_current(index,m);cached_[m][index-1]=p;valid_[m][index-1]=1;}
            if(oneoff)end_crop();return p;}catch(...){if(oneoff)end_crop();throw;}
    }
    Image correlate(const Image& im,int index,CorrelationMode mode)override{
        if(index<1||index>int(bank_->templates.size()))throw std::invalid_argument("Invalid template index");
        bool oneoff=!crop_;if(oneoff)begin_crop(im);else if(crop_!=&im)throw std::invalid_argument("CPU crop mismatch");
        int m=mode==CorrelationMode::legacy_padded_valid;try{
            if(m)compact(index);else full(index);Image map(im.width+bank_->tw-1,im.height+bank_->th-1);
            if(m){int cols=im.width-bank_->tw+1,rows=im.height-bank_->th+1;for(int y=0;y<rows;++y)for(int x=0;x<cols;++x)map.at(x+bank_->tw-1,y+bank_->th-1)=compact_[size_t(y)*cols+x];}
            else for(int y=0;y<map.height;++y)for(int x=0;x<map.width;++x)map.at(x,y)=full_value(x,y,index);
            if(oneoff)end_crop();return map;
        }catch(...){if(oneoff)end_crop();throw;}
    }
};
}
std::unique_ptr<CorrelationBackend> make_cpu_correlation(const Model& m){return std::make_unique<Cached>(m);}
}
namespace cpu_refresh {
Scratch& scratch(defocus::CorrelationBackend& other){auto* cached=dynamic_cast<defocus::Cached*>(&other);if(!cached)throw std::invalid_argument("CPU scratch requires cached backend");return cached->scratch();}
std::unique_ptr<defocus::CorrelationBackend> clone_backend(const defocus::CorrelationBackend& other){
    auto* cached=dynamic_cast<const defocus::Cached*>(&other);if(!cached)throw std::invalid_argument("Cannot clone non-cached CPU backend");return std::make_unique<defocus::Cached>(*cached);
}
}
