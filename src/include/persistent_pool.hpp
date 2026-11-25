#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace workflow {
// CPU-only parked worker pool. No MATLAB APIs, CUDA calls, priority/affinity changes,
// or idle spin loops. One synchronous dispatch at a time per owner context.
class PersistentPool {
    std::mutex mutex_;
    std::condition_variable changed_,completed_,started_condition_;
    std::vector<std::thread> workers_;
    bool stopping_=false;
    uint64_t generation_=0;
    int started_=0,remaining_=0,total_=0,grain_=1;
    std::atomic<int> next_{0};
    std::function<void(int,int)> job_;
    std::exception_ptr exception_;
    void worker(){
        uint64_t observed=0;
        std::unique_lock<std::mutex> lock(mutex_);
        ++started_;started_condition_.notify_one();
        for(;;){
            changed_.wait(lock,[&]{return stopping_||generation_!=observed;});
            if(stopping_)return;
            observed=generation_;int total=total_,grain=grain_;
            lock.unlock();
            try{
                for(;;){int begin=next_.fetch_add(grain,std::memory_order_relaxed);if(begin>=total)break;job_(begin,std::min(total,begin+grain));}
            }catch(...){
                std::lock_guard<std::mutex> guard(mutex_);
                if(!exception_)exception_=std::current_exception();next_.store(total,std::memory_order_relaxed);
            }
            lock.lock();
            if(--remaining_==0)completed_.notify_one();
        }
    }
    void shutdown()noexcept{
        {std::lock_guard<std::mutex> guard(mutex_);stopping_=true;}
        changed_.notify_all();
        for(auto& thread:workers_)if(thread.joinable())thread.join();
    }
public:
    explicit PersistentPool(int count,int grain=1):grain_(grain){
        if(count<1||count>64||grain<1||grain>64)throw std::invalid_argument("Persistent workers and grain must be1..64");
        try{
            workers_.reserve(count);
            for(int i=0;i<count;++i)workers_.emplace_back([this]{worker();});
            std::unique_lock<std::mutex> lock(mutex_);started_condition_.wait(lock,[&]{return started_==count;});
        }catch(...){shutdown();throw;}
    }
    PersistentPool(const PersistentPool&)=delete;
    PersistentPool& operator=(const PersistentPool&)=delete;
    ~PersistentPool(){shutdown();}
    int size()const{return int(workers_.size());}
    int grain()const{return grain_;}
    void run(int count,const std::function<void(int,int)>& body){
        if(count<0)throw std::invalid_argument("Negative persistent work count");
        if(!count)return;
        std::unique_lock<std::mutex> lock(mutex_);
        if(remaining_||stopping_)throw std::invalid_argument("Persistent CPU pool is busy or stopping");
        total_=count;next_.store(0,std::memory_order_relaxed);exception_=nullptr;job_=body;
        remaining_=int(workers_.size());++generation_;changed_.notify_all();
        completed_.wait(lock,[&]{return remaining_==0;});
        auto failure=exception_;job_={};lock.unlock();if(failure)std::rethrow_exception(failure);
    }
};
}
