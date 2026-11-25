#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace candidate_direct {
struct Config {int width=0,height=0,template_width=0,template_height=0,calibration_count=0;double threshold=0;bool coalesced_statistics=false;std::vector<int> indices;};
struct Template {std::vector<float> pixels;}; // Row-major raw calibration pixels, same float conversion as native CUDA.
struct Candidate {int template_index=0,x=0,y=0;float score=0;};
struct Result {
    std::vector<Candidate> candidates;
    std::vector<float> similarities; // MATLAB column-major N-by-K output layout.
    double seconds=0,upload_enqueue_s=0,engine_s=0,engine_download_s=0,sampling_s=0;
};
struct ImageDescriptor {uintptr_t data=0;size_t step_bytes=0;int width=0,height=0,device=0;uint64_t epoch=0;};
class Search {
public:
    virtual ~Search()=default;
    virtual Result run(const uint16_t* column_major_image)=0;
    virtual ImageDescriptor device_image()const=0;
};
std::unique_ptr<Search> make(const Config& config,const std::vector<Template>& templates);
}
