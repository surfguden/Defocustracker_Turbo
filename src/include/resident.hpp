#pragma once
#include "defocus/tracker.hpp"
namespace defocus {
class ResidentEngine {
public:
    virtual ~ResidentEngine()=default;
    virtual bool supports_async()const{return false;}
    virtual bool pending()const{return false;}
    virtual void submit_device(const float*,size_t,const std::vector<CoarseCandidate>&,bool){
        throw std::invalid_argument("Async resident submission requires at least two workers");
    }
    virtual RefineResult complete(){throw std::invalid_argument("No supported async resident submission");}
    virtual Image last_image()=0;
    virtual double warmup(int milliseconds)=0;
    virtual uint64_t clamped_xy()const=0;
    virtual RefineResult refine_device(const float* image,size_t step,const std::vector<CoarseCandidate>& coarse,bool diagnostics)=0;
    virtual RefineResult refine(const double* column_major_image,const std::vector<CoarseCandidate>& coarse,bool diagnostics)=0;
};
void verify_mixed_depth(const Model&,const Image&,int);
std::unique_ptr<ResidentEngine> make_resident(const Model&,int workers=1,bool bounded=false,bool mixed=false,bool fused=false);
}
