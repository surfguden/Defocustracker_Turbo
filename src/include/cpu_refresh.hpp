#pragma once
#include "defocus/tracker.hpp"
#include <functional>
#include <memory>
namespace cpu_refresh {
struct Scratch {defocus::Image crop;std::vector<double> median;};
Scratch& scratch(defocus::CorrelationBackend&);
class Workers {
    struct Impl;std::unique_ptr<Impl> impl_;
public:
    explicit Workers(int count);~Workers();
    int size()const;
    void run(int count,const std::function<void(int,int)>& body);
};
std::unique_ptr<defocus::CorrelationBackend> clone_backend(const defocus::CorrelationBackend&);
defocus::RefineResult refine(const defocus::Model&,std::vector<defocus::CorrelationBackend*>&,Workers*,
    defocus::Image&&,const std::vector<defocus::CoarseCandidate>&,bool);
}
