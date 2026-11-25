#include "initial_depth.hpp"
#include "persistent_pool.hpp"
#include <optional>
#include "cpu_execution.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <limits>
#include <numeric>
#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <unordered_map>

namespace workflow { namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
struct WorkerScratch {defocus::Image image;std::vector<double> sums,squares;};
// Own buffers in library-scoped storage, with only a trivial pointer in worker TLS.
// This avoids leaving nontrivial TLS destructors on external OpenCV worker threads when the MEX unloads.
std::mutex scratch_mutex;
std::unordered_map<std::thread::id,std::unique_ptr<WorkerScratch>> scratch_registry;
thread_local WorkerScratch* thread_scratch=nullptr;
WorkerScratch& scratch(){
    if(!thread_scratch){
        std::lock_guard<std::mutex> guard(scratch_mutex);auto& entry=scratch_registry[std::this_thread::get_id()];
        if(!entry)entry=std::make_unique<WorkerScratch>();thread_scratch=entry.get();
    }
    return *thread_scratch;
}
template<class T> void copy_crop(const T* source,int image_height,defocus::Image& out,int left,int top){
    for(int x=0;x<out.width;++x)for(int y=0;y<out.height;++y){
        double value=double(source[size_t(left+x)*image_height+top+y]);
        if(!std::isfinite(value))throw std::invalid_argument("Nonfinite candidate crop");
        out.at(x,y)=value;
    }
}
defocus::Image& crop(const ImageView& image,double x,double y,int width,int height){
    int left=int(std::round(x))-1-width/2,top=int(std::round(y))-1-height/2;
    if(left<0||top<0||left+width>image.width||top+height>image.height)
        throw std::invalid_argument("Candidate crop crosses image boundary");
    auto& out=scratch().image;
    if(out.width!=width||out.height!=height)out=defocus::Image(width,height);
    switch(image.type){
        case PixelType::f64:copy_crop(static_cast<const double*>(image.data),image.height,out,left,top);break;
        case PixelType::f32:copy_crop(static_cast<const float*>(image.data),image.height,out,left,top);break;
        case PixelType::u16:copy_crop(static_cast<const uint16_t*>(image.data),image.height,out,left,top);break;
        case PixelType::u8:copy_crop(static_cast<const uint8_t*>(image.data),image.height,out,left,top);break;
    }
    return out;
}
}
InitialDepthMatcher::~InitialDepthMatcher()=default;
void InitialDepthMatcher::set_worker_pool(int workers,int grain){
    if(workers<0||workers>64||grain<1||grain>64)throw std::invalid_argument("Pool workers must be0..64; grain1..64");
    std::unique_ptr<PersistentPool> replacement;
    if(workers)replacement=std::make_unique<PersistentPool>(workers,grain);
    fixed_pool_=std::move(replacement);
}
int InitialDepthMatcher::worker_threads()const{return fixed_pool_?fixed_pool_->size():model_.threads;}
const char* InitialDepthMatcher::worker_mode()const{return fixed_pool_?"persistent_parked":"opencv";}
void InitialDepthMatcher::parallel_range(int count,const std::function<void(const cv::Range&)>& body)const{
    if(fixed_pool_)fixed_pool_->run(count,[&](int begin,int end){body(cv::Range(begin,end));});
    else cv::parallel_for_(cv::Range(0,count),body);
}
InitialDepthMatcher::InitialDepthMatcher(const defocus::Model& model,std::vector<defocus::Point> polygon)
    :model_(model),polygon_(std::move(polygon)),templates_(model.templates.size()){
    if(polygon_.size()<3)throw std::invalid_argument("Suppression polygon needs at least three vertices");
    min_x_=max_x_=polygon_[0].x;min_y_=max_y_=polygon_[0].y;
    for(const auto& p:polygon_){min_x_=std::min(min_x_,p.x);max_x_=std::max(max_x_,p.x);min_y_=std::min(min_y_,p.y);max_y_=std::max(max_y_,p.y);}
    avx_=cv::checkHardwareSupport(CV_CPU_AVX2);
    for(size_t k=0;k<templates_.size();++k){
        const auto& image=model.templates[k];auto& state=templates_[k];
        double mean=0;for(int x=0;x<image.width;++x)for(int y=0;y<image.height;++y)mean+=image.at(x,y);
        mean/=image.pixels.size();double sum=0,square=0;state.centered.resize(image.pixels.size());
        for(int x=0;x<image.width;++x)for(int y=0;y<image.height;++y){
            double value=image.at(x,y)-mean;state.centered[size_t(y)*image.width+x]=value;sum+=value;square+=value*value;
        }
        state.residual=sum/image.pixels.size();state.norm=std::sqrt(square-sum*sum/image.pixels.size());
    }
    selected_similarity_=prepare_similarity(model.selected);
}
InitialDepthMatcher::SimilarityState InitialDepthMatcher::prepare_similarity(const std::vector<int>& used)const{
    SimilarityState out;out.columns=int(model_.templates.size());out.rows=int(used.size());
    out.centered.resize(size_t(out.rows)*out.columns);out.norms.resize(out.columns);
    for(int d=0;d<out.columns;++d){
        double mean=0;for(int k:used)mean+=model_.similarity.at(d,k-1);mean/=used.size();
        double norm=0;for(size_t k=0;k<used.size();++k){
            double value=model_.similarity.at(d,used[k]-1)-mean;
            out.centered[size_t(d)*out.rows+k]=value;norm+=value*value;
        }
        out.norms[d]=norm;
    }
    return out;
}
std::pair<int,double> InitialDepthMatcher::guess(const SimilarityState& map,const std::vector<double>& values)const{
    double mean=std::accumulate(values.begin(),values.end(),0.0)/values.size(),norm=0;
    std::vector<double> centered(values.size());
    for(size_t k=0;k<values.size();++k){centered[k]=values[k]-mean;norm+=centered[k]*centered[k];}
    int index=1;double best=std::numeric_limits<double>::quiet_NaN();
    for(int d=0;d<map.columns;++d){
        const double* column=map.centered.data()+size_t(d)*map.rows;double numerator=0;
        for(size_t k=0;k<values.size();++k)numerator+=column[k]*centered[k];
        double score=numerator/std::sqrt(norm*map.norms[d]);
        if((std::isnan(best)&&!std::isnan(score))||score>best){best=score;index=d+1;}
    }
    return {index,best};
}
bool InitialDepthMatcher::inside(double x,double y,double cx,double cy)const{
    if(x<cx+min_x_||x>cx+max_x_||y<cy+min_y_||y>cy+max_y_)return false;
    bool answer=false;
    for(size_t i=0,j=polygon_.size()-1;i<polygon_.size();j=i++){
        double ax=cx+polygon_[j].x,ay=cy+polygon_[j].y,bx=cx+polygon_[i].x,by=cy+polygon_[i].y;
        double dx=bx-ax,dy=by-ay,cross=(x-ax)*dy-(y-ay)*dx;
        double tolerance=32*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs((x-ax)*dy)+std::abs((y-ay)*dx));
        if(std::abs(cross)<=tolerance&&x>=std::min(ax,bx)&&x<=std::max(ax,bx)&&y>=std::min(ay,by)&&y<=std::max(ay,by))return true;
        if((ay>y)!=(by>y))if(x<(bx-ax)*(y-ay)/(by-ay)+ax)answer=!answer;
    }
    return answer;
}
double InitialDepthMatcher::peak(const defocus::Image& image,int index)const{
    const auto& state=templates_.at(index-1);int tw=model_.templates[0].width,th=model_.templates[0].height;
    int stride=image.width+1;auto& sums=scratch().sums;auto& squares=scratch().squares;
    sums.assign(size_t(stride)*(image.height+1),0);squares.assign(sums.size(),0);
    for(int y=0;y<image.height;++y){double sum=0,square=0;for(int x=0;x<image.width;++x){
        double value=image.at(x,y);sum+=value;square+=value*value;
        sums[size_t(y+1)*stride+x+1]=sums[size_t(y)*stride+x+1]+sum;
        squares[size_t(y+1)*stride+x+1]=squares[size_t(y)*stride+x+1]+square;
    }}
    double best=0; // Full legacy map has an implicit zero border.
    for(int y=0;y<=image.height-th;++y)for(int x=0;x<=image.width-tw;++x){
        auto window=[&](const auto& values){return values[size_t(y+th)*stride+x+tw]-values[size_t(y)*stride+x+tw]-values[size_t(y+th)*stride+x]+values[size_t(y)*stride+x];};
        double sum=window(sums),square=window(squares),dot=0;
        if(avx_){
            __m256d a=_mm256_setzero_pd(),b=_mm256_setzero_pd();
            for(int yy=0;yy<th;++yy){
                const double* p=image.pixels.data()+size_t(y+yy)*image.width+x;
                const double* t=state.centered.data()+size_t(yy)*tw;int xx=0;
                for(;xx+7<tw;xx+=8){a=_mm256_add_pd(a,_mm256_mul_pd(_mm256_loadu_pd(p+xx),_mm256_loadu_pd(t+xx)));b=_mm256_add_pd(b,_mm256_mul_pd(_mm256_loadu_pd(p+xx+4),_mm256_loadu_pd(t+xx+4)));}
                for(;xx<tw;++xx)dot+=p[xx]*t[xx];
            }
            alignas(32) double totals[4];_mm256_store_pd(totals,_mm256_add_pd(a,b));dot+=totals[0]+totals[1]+totals[2]+totals[3];
        }else for(int yy=0;yy<th;++yy)for(int xx=0;xx<tw;++xx)dot+=image.at(x+xx,y+yy)*state.centered[size_t(yy)*tw+xx];
        double denominator=state.norm*std::sqrt(std::max(square-sum*sum/(tw*th),0.0));
        double value=denominator>0?(dot-sum*state.residual)/denominator:0;
        if(std::abs(value)-1>std::sqrt(std::numeric_limits<double>::epsilon()))value=0;
        best=std::max(best,value);
    }
    return best;
}
DepthResult InitialDepthMatcher::run(const ImageView& image,const std::vector<Seed>& seeds,const std::vector<double>& scores)const{
    if(image.width!=model_.frame_width||image.height!=model_.frame_height||scores.size()!=seeds.size()*model_.selected.size())throw std::invalid_argument("Depth input dimensions disagree");
    std::optional<defocus::CpuExecution> execution;if(!fixed_pool_)execution.emplace(model_.threads);auto started=Clock::now();DepthResult result;
    std::vector<int> order(seeds.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return seeds[a].score>seeds[b].score;});
    std::vector<uint8_t> active(seeds.size(),1);
    for(int selected:order)if(active[selected]){
        result.source_indices.push_back(selected+1);const auto& center=seeds[selected];
        for(size_t k=0;k<seeds.size();++k)if(active[k]&&inside(seeds[k].x,seeds[k].y,center.x,center.y))active[k]=0;
    }
    result.nms_seconds=elapsed(started);auto matching_started=Clock::now();
    result.coarse.resize(result.source_indices.size());result.correlation_counts.assign(result.source_indices.size(),0);
    int selected_count=int(model_.selected.size()),cw=model_.templates[0].width+8,ch=model_.templates[0].height+8;
    parallel_range(int(result.coarse.size()),[&](const cv::Range& range){
        for(int i=range.start;i<range.end;++i){
            int source=result.source_indices[i]-1;const auto& seed=seeds[source];
            std::vector<double> values(scores.begin()+size_t(source)*selected_count,scores.begin()+size_t(source+1)*selected_count);
            auto estimate=guess(selected_similarity_,values);std::vector<int> used=model_.selected;
            auto& patch=crop(image,seed.x,seed.y,cw,ch);
            while(std::find(used.begin(),used.end(),estimate.first)==used.end()){
                used.push_back(estimate.first);values.push_back(peak(patch,estimate.first));++result.correlation_counts[i];
                estimate=guess(prepare_similarity(used),values);
            }
            result.coarse[i]={seed.x,seed.y,estimate.first,seed.score,estimate.second};
        }
    });
    result.matching_seconds=elapsed(matching_started);result.seconds=elapsed(started);return result;
}
WarmupResult InitialDepthMatcher::warmup(int duration_ms)const{
    if(duration_ms<1||duration_ms>2000)throw std::invalid_argument("Warmup duration must be 1..2000 milliseconds");
    std::optional<defocus::CpuExecution> execution;if(!fixed_pool_)execution.emplace(model_.threads);WarmupResult result;
    int cw=model_.templates[0].width+8,ch=model_.templates[0].height+8;
    std::vector<uint16_t> pixels(size_t(cw)*ch);
    for(int x=0;x<cw;++x)for(int y=0;y<ch;++y)pixels[size_t(x)*ch+y]=uint16_t(1+(x*17+y*7+x*y)%65521);
    ImageView synthetic{pixels.data(),cw,ch,PixelType::u16};
    int tasks=std::max(64,worker_threads()*8);std::vector<double> checksums(tasks);
    std::unordered_set<std::thread::id> workers;std::mutex workers_mutex;std::atomic<int> active{0},maximum{0};
    auto started=Clock::now();
    do{
        auto wave=Clock::now();
        parallel_range(tasks,[&](const cv::Range& range){
            {std::lock_guard<std::mutex> guard(workers_mutex);workers.insert(std::this_thread::get_id());}
            int count=active.fetch_add(1)+1,observed=maximum.load();
            while(count>observed&&!maximum.compare_exchange_weak(observed,count)){}
            for(int i=range.start;i<range.end;++i){
                auto& patch=crop(synthetic,(cw+1)/2.0,(ch+1)/2.0,cw,ch);
                double value=0;for(int k=0;k<2;++k)value+=peak(patch,model_.selected[(i+k)%model_.selected.size()]);
                checksums[i]=value;
            }
            active.fetch_sub(1);
        });
        result.wave_seconds.push_back(elapsed(wave));result.checksum+=std::accumulate(checksums.begin(),checksums.end(),0.0);
    }while(elapsed(started)<duration_ms/1000.0);
    result.seconds=elapsed(started);result.worker_count=int(workers.size());result.max_active=maximum.load();return result;
}
std::pair<double,int> InitialDepthMatcher::synthetic_peak_check(defocus::CorrelationBackend& reference)const{
    int cw=model_.templates[0].width+8,ch=model_.templates[0].height+8;std::vector<double> pixels(size_t(cw)*ch);
    ImageView view{pixels.data(),cw,ch,PixelType::f64};double error=0;int cases=0;
    for(int pattern=0;pattern<3;++pattern){
        for(int x=0;x<cw;++x)for(int y=0;y<ch;++y)pixels[size_t(x)*ch+y]=pattern==0?0:1+(x*17+y*7+x*y)%65521-(pattern==2?25000:0);
        auto& image=crop(view,(cw+1)/2.0,(ch+1)/2.0,cw,ch);
        for(int index:std::vector<int>{1,model_.selected.front(),model_.selected.back(),int(model_.templates.size())}){
            double actual=peak(image,index),expected=reference.peak(image,index,defocus::CorrelationMode::legacy_padded_valid).score;
            if(!std::isfinite(actual)||!std::isfinite(expected))throw std::runtime_error("Nonfinite synthetic peak result");
            error=std::max(error,std::abs(actual-expected));++cases;
        }
    }
    return {error,cases};
}
}
