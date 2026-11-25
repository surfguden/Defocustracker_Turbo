#pragma once
#include <opencv2/core.hpp>
#include <mutex>
namespace defocus {
// OpenCV's thread setting is global to this CPU library. Serialize operations
// that change it, so a diagnostic context cannot change another context's limit.
inline std::mutex cpu_opencv_mutex;
struct CpuExecution {
    std::unique_lock<std::mutex> lock;
    explicit CpuExecution(int threads):lock(cpu_opencv_mutex) {
        if (cv::getNumThreads()!=threads) cv::setNumThreads(threads);
    }
};
}
