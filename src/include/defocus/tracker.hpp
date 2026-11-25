#pragma once
#include <cstdint>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace defocus {
struct Image {
    int width=0,height=0;
    std::vector<double> pixels;
    Image()=default;
    Image(int w,int h,double value=0);
    double& at(int x,int y) { return pixels.at(size_t(y)*width+x); }
    double at(int x,int y) const { return pixels.at(size_t(y)*width+x); }
};
struct Point { double x=0,y=0; };
struct Model {
    std::vector<Image> templates;
    Image global_mask,similarity;
    std::vector<std::vector<Point>> areas;
    std::vector<int> selected; // MATLAB calibration indices, starting at one.
    int frame_width=0,frame_height=0;
    double cm_guess=.4,cm_final=.5;
    bool interpolate=true;
    int threads=1;
    void validate() const;
};
enum class CorrelationMode { matlab_full,legacy_padded_valid };
enum class DepthStrategy { adaptive,full_stack };
struct Peak {
    double score=0,left=0,right=0,above=0,below=0;
    int x=1,y=1,ties=0,sum_x=0,sum_y=0;
    bool has_neighbors=false;
};
struct DepthSample {int template_index=0;Peak peak;};
struct DepthWalk {
    std::array<int,3> indices{};
    std::array<Peak,3> peaks{};
    int best=0;
    bool boundary=false;
    std::vector<DepthSample> samples; // Original logical request order, including the initial three.
};
Peak summarize(const Image& map);
class CudaFailure:public std::runtime_error { public: using std::runtime_error::runtime_error; };
struct CorrelationMetrics {
    uint64_t logical_requests=0,crop_uploads=0,prefix_builds=0,image_ffts=0,inverse_ffts=0;
    uint64_t template_evaluations=0,result_downloads=0,synchronizations=0,cache_hits=0,batch_submissions=0;
    uint64_t gpu_depth_walks=0,host_depth_walks=0;
    uint64_t full_stack_walks=0;
    double depth_walk_s=0;
};
class CorrelationBackend {
public:
    virtual ~CorrelationBackend()=default;
    virtual void begin_crop(const Image&) {}
    virtual void end_crop() noexcept {}
    virtual void reset_metrics() {}
    virtual CorrelationMetrics metrics() const {return {};}
    virtual Image correlate(const Image& crop,int template_index,CorrelationMode mode)=0;
    virtual Peak peak(const Image& crop,int template_index,CorrelationMode mode) {
        return summarize(correlate(crop,template_index,mode));
    }
    virtual std::vector<Peak> peaks_batch(const Image& crop,const std::vector<int>& indices,CorrelationMode mode) {
        std::vector<Peak> result;result.reserve(indices.size());
        for(int index:indices) result.push_back(peak(crop,index,mode));
        return result;
    }
    virtual std::string name() const=0;
    virtual DepthWalk depth_walk(const Image& crop,int depth,int depths);
    virtual std::string depth_walk_implementation()const{return "host";}
    virtual double depth_walk_setup_seconds()const{return 0;}
    virtual void set_depth_strategy(DepthStrategy) {}
    virtual void test_failure_after(int) {throw std::invalid_argument("Failure injection is not enabled in this build");}
};
struct Candidate { int template_index=0; double x=0,y=0,score=0; };
struct SearchResult {
    std::vector<Candidate> candidates;
    std::vector<double> similarities; // Candidate-major, selected template columns.
    double seconds=0;
};
class CandidateBackend {
public:
    virtual ~CandidateBackend()=default;
    virtual SearchResult search(const Image& image)=0;
};
struct CoarseCandidate { double x=0,y=0; int depth=1; double score=0,similarity=0; };
struct Localization { double x=0,y=0,z=0,score=0; int source_index=0; };
struct CorrelationTrace { int source_index=0,template_index=0,mode=0; Peak peak; };
struct ParticleTrace {
    int source_index=0,initial_depth=0,final_depth=0,rejection=0;
    int full_calls=0,legacy_calls=0,overlap_restarted=0;
    double x=0,y=0,z=0,score=0;
};
struct RefineResult {
    std::vector<Localization> localizations;
    std::vector<ParticleTrace> particles;
    std::vector<CorrelationTrace> correlations;
    Image image;
    double seconds=0;
    CorrelationMetrics metrics;
};
RefineResult refine_xyz(const Model& model,CorrelationBackend& backend,const Image& image,
                        const std::vector<CoarseCandidate>& candidates,bool diagnostics=false);
std::unique_ptr<CorrelationBackend> make_cpu_correlation(const Model& model);
std::unique_ptr<CandidateBackend> make_cpu_candidates(const Model& model);
std::unique_ptr<CorrelationBackend> make_cuda_correlation(const Model& model);
std::unique_ptr<CandidateBackend> make_cuda_candidates(const Model& model);
} // namespace defocus
