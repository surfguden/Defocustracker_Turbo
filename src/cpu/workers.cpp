#include "cpu_refresh.hpp"
#include <condition_variable>
#include <thread>
#include <mutex>
#include <atomic>
#include <exception>
namespace cpu_refresh {
struct Workers::Impl {
    std::mutex mutex;std::condition_variable changed,done,started;
    std::vector<std::thread> threads;std::function<void(int,int)> job;
    std::atomic<int> next{0};int total=0,remaining=0,ready=0;unsigned generation=0;bool stop=false;
    std::exception_ptr failure;
    explicit Impl(int count){
        if(count<1||count>64)throw std::invalid_argument("CPU workers must be1..64");
        try{for(int k=0;k<count;++k)threads.emplace_back([this,k]{
            std::unique_lock<std::mutex> lock(mutex);unsigned seen=0;++ready;started.notify_one();
            for(;;){changed.wait(lock,[&]{return stop||generation!=seen;});if(stop)return;seen=generation;lock.unlock();
                try{for(;;){int i=next.fetch_add(1);if(i>=total)break;job(k,i);}}
                catch(...){std::lock_guard<std::mutex> guard(mutex);if(!failure)failure=std::current_exception();next.store(total);}
                lock.lock();if(--remaining==0)done.notify_one();
            }
        });std::unique_lock<std::mutex> lock(mutex);started.wait(lock,[&]{return ready==count;});}
        catch(...){shutdown();throw;}
    }
    void shutdown(){ {std::lock_guard<std::mutex> lock(mutex);stop=true;}changed.notify_all();for(auto& t:threads)if(t.joinable())t.join(); }
    ~Impl(){shutdown();}
};
Workers::Workers(int count):impl_(std::make_unique<Impl>(count)){}Workers::~Workers()=default;
int Workers::size()const{return int(impl_->threads.size());}
void Workers::run(int count,const std::function<void(int,int)>& body){
    if(!count)return;auto& p=*impl_;std::unique_lock<std::mutex> lock(p.mutex);
    if(p.remaining||p.stop)throw std::runtime_error("CPU worker pool busy");
    p.total=count;p.next=0;p.failure=nullptr;p.job=body;p.remaining=size();++p.generation;p.changed.notify_all();
    p.done.wait(lock,[&]{return !p.remaining;});auto failure=p.failure;p.job={};lock.unlock();if(failure)std::rethrow_exception(failure);
}
}
