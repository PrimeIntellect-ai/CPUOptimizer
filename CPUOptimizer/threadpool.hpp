#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <mutex>
#include <pthread.h>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#if __has_include(<numa.h>)
#define NUMA_ENABLED 1
#include <numa.h>
#else
#define NUMA_ENABLED 0
#endif

namespace cpuoptim {

class ThreadPool {
public:
  struct task_element_t {
    bool run_with_id;
    const std::function<void()> no_id;
    const std::function<void(std::size_t)> with_id;

    explicit task_element_t(std::function<void()> f)
        : run_with_id(false), no_id(std::move(f)), with_id(nullptr) {}
    explicit task_element_t(std::function<void(std::size_t)> f)
        : run_with_id(true), no_id(nullptr), with_id(std::move(f)) {}
  };

  explicit ThreadPool(int pool_size, int numa_node_id = -1,
                      const std::function<void()> &init_thread = nullptr)
      : threads_(pool_size < 0 ? defaultNumThreads() : pool_size),
        running_(true), complete_(true), available_(threads_.size()),
        total_(threads_.size()), numa_node_id_(numa_node_id) {
    for (std::size_t i = 0; i < threads_.size(); ++i) {
      threads_[i] = std::thread([this, i, init_thread]() {
        pthread_setname_np(pthread_self(), "cpuoptimizer_pool");
        if (init_thread) {
          init_thread();
        }
        this->main_loop(i);
      });
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      running_ = false;
      condition_.notify_all();
    }

    for (auto &t : threads_) {
      try {
        t.join();
      } catch (const std::exception &e) {
        std::fprintf(stderr, "Exception joining thread! %s\n", e.what());
      }
    }
  }

  size_t size() const { return threads_.size(); }

  size_t numAvailable() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return available_;
  }

  void run(std::function<void()> func) {
    if (threads_.empty()) {
      throw std::runtime_error("No threads to run a task");
    }
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.emplace(std::move(func));
    complete_ = false;
    condition_.notify_one();
  }

  template <typename Task>
  void runTaskWithID(Task task) {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.emplace(static_cast<std::function<void(std::size_t)>>(task));
    complete_ = false;
    condition_.notify_one();
  }

  void waitWorkComplete() {
    std::unique_lock<std::mutex> lock(mutex_);
    completed_.wait(lock, [&]() { return complete_; });
  }

  static size_t defaultNumThreads() {
    return std::thread::hardware_concurrency();
  }

private:
  std::queue<task_element_t> tasks_;
  std::vector<std::thread> threads_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::condition_variable completed_;
  std::atomic_bool running_;
  bool complete_;
  std::size_t available_;
  std::size_t total_;
  int numa_node_id_;

  void main_loop(std::size_t index) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (running_) {
      condition_.wait(lock, [&]() { return !tasks_.empty() || !running_; });
      if (!running_) break;

      task_element_t task = std::move(tasks_.front());
      tasks_.pop();
      --available_;

      lock.unlock();

      try {
        if (task.run_with_id) {
          task.with_id(index);
        } else {
          task.no_id();
        }
      } catch (const std::exception &e) {
        std::fprintf(stderr, "Exception in thread pool task: %s\n", e.what());
      } catch (...) {
        std::fprintf(stderr, "Exception in thread pool task: unknown\n");
      }

      lock.lock();
      ++available_;
      if (tasks_.empty() && available_ == total_) {
        complete_ = true;
        completed_.notify_one();
      }
    }
  }
};

static inline void NUMABind(int numa_node_id) {
  #if NUMA_ENABLED
    if (numa_node_id >= 0) {
      auto bm = numa_allocate_nodemask();
      numa_bitmask_setbit(bm, numa_node_id);
      numa_bind(bm);
      numa_bitmask_free(bm);
    }
  #else
    (void)numa_node_id;
  #endif
  }

class TaskThreadPool : public ThreadPool {
public:
  explicit TaskThreadPool(int pool_size, int numa_node_id = -1)
      : ThreadPool(pool_size, numa_node_id, [numa_node_id]() {
          NUMABind(numa_node_id);
        }) {}
};

} // namespace cpuoptim