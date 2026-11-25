#pragma once
#include "defocus/tracker.hpp"
#include <cstdint>
#include <vector>
#include <functional>

namespace cv {class Range;}
namespace workflow {
class PersistentPool;
enum class PixelType { f64,f32,u16,u8 };
struct ImageView {const void* data=nullptr;int width=0,height=0;PixelType type=PixelType::f64;};
struct Seed {double x=0,y=0,score=0;};
struct DepthResult {
    std::vector<defocus::CoarseCandidate> coarse;
    std::vector<int> source_indices,correlation_counts;
    double seconds=0,nms_seconds=0,matching_seconds=0;
};
struct WarmupResult {double seconds=0,checksum=0;int worker_count=0,max_active=0;std::vector<double> wave_seconds;};
class InitialDepthMatcher {
    struct TemplateState {std::vector<double> centered;double norm=0,residual=0;};
    struct SimilarityState {std::vector<double> centered,norms;int columns=0,rows=0;};
    const defocus::Model& model_;
    std::vector<defocus::Point> polygon_;
    double min_x_=0,max_x_=0,min_y_=0,max_y_=0;
    std::vector<TemplateState> templates_;
    SimilarityState selected_similarity_;
    bool avx_=false;
    std::unique_ptr<PersistentPool> fixed_pool_;
    void parallel_range(int count,const std::function<void(const cv::Range&)>& body)const;
    SimilarityState prepare_similarity(const std::vector<int>& used)const;
    std::pair<int,double> guess(const SimilarityState& map,const std::vector<double>& values)const;
    double peak(const defocus::Image& crop,int index)const;
    bool inside(double x,double y,double cx,double cy)const;
public:
    InitialDepthMatcher(const defocus::Model& model,std::vector<defocus::Point> polygon);
    ~InitialDepthMatcher();
    void set_worker_pool(int workers,int grain=1);
    int worker_threads()const;
    const char* worker_mode()const;
    DepthResult run(const ImageView& image,const std::vector<Seed>& seeds,const std::vector<double>& scores)const;
    WarmupResult warmup(int duration_ms=100)const;
    std::pair<double,int> synthetic_peak_check(defocus::CorrelationBackend& reference)const;
};
}
