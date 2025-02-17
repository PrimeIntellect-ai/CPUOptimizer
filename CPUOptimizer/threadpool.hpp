// Adapted from the pytorch threadpool implementation.
// Split out into its own file so that torch is not a dependency for the C/C++ interface.
// https://github.com/pytorch/pytorch/blob/main/c10/core/thread_pool.h
// https://github.com/pytorch/pytorch/blob/main/c10/core/thread_pool.cpp

// As such, refer to TORCH_LICENSE.txt in the root of this repo, or at the link below.
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
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

class TaskThreadPoolBase {
public:
  virtual void run(std::function<void()> func) = 0;

  virtual size_t size() const = 0;

  virtual size_t numAvailable() const = 0;

  virtual bool inThreadPool() const = 0;

  virtual ~TaskThreadPoolBase() noexcept = default;

  static size_t defaultNumThreads();
};

class ThreadPool : public TaskThreadPoolBase {
protected:
  struct task_element_t {
    bool run_with_id;
    const std::function<void()> no_id;
    const std::function<void(std::size_t)> with_id;

    explicit task_element_t(std::function<void()> f)
        : run_with_id(false), no_id(std::move(f)), with_id(nullptr) {}
    explicit task_element_t(std::function<void(std::size_t)> f)
        : run_with_id(true), no_id(nullptr), with_id(std::move(f)) {}
  };

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

public:
  ThreadPool() = delete;

  explicit ThreadPool(int pool_size, int numa_node_id = -1,
                      const std::function<void()> &init_thread = nullptr);

  ~ThreadPool() override;

  size_t size() const override;

  size_t numAvailable() const override;

  bool inThreadPool() const override;

  void run(std::function<void()> func) override;

  template <typename Task> void runTaskWithID(Task task) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Set task and signal condition variable so that a worker thread will
    // wake up and use the task.
    tasks_.emplace(static_cast<std::function<void(std::size_t)>>(task));
    complete_ = false;
    condition_.notify_one();
  }

  /// @brief Wait for queue to be empty
  void waitWorkComplete();

private:
  // @brief Entry point for pool threads.
  void main_loop(std::size_t index);
};

static inline void NUMABind(int numa_node_id) {
#if NUMA_ENABLED
  if (numa_node_id < 0) {
    return;
  }

  auto bm = numa_allocate_nodemask();
  numa_bitmask_setbit(bm, numa_node_id);
  numa_bind(bm);
  numa_bitmask_free(bm);
#else
  (void)numa_node_id;
#endif
}

class TaskThreadPool : public ThreadPool {
public:
  explicit TaskThreadPool(int pool_size, int numa_node_id = -1)
      : ThreadPool(pool_size, numa_node_id, [numa_node_id]() {
          // setThreadName("CaffeTaskThread");
          NUMABind(numa_node_id);
        }) {}
};

inline size_t TaskThreadPoolBase::defaultNumThreads() {
  return std::thread::hardware_concurrency();
}

inline ThreadPool::ThreadPool(int pool_size, int numa_node_id,
                              const std::function<void()> &init_thread)
    : threads_(pool_size < 0 ? defaultNumThreads() : pool_size), running_(true),
      complete_(true), available_(threads_.size()), total_(threads_.size()),
      numa_node_id_(numa_node_id) {
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

inline ThreadPool::~ThreadPool() {
  // Set running flag to false then notify all threads.
  {
    std::unique_lock<std::mutex> lock(mutex_);
    running_ = false;
    condition_.notify_all();
  }

  for (auto &t : threads_) {
    try {
      t.join();
    } catch (const std::exception &) {
    }
  }
}

inline size_t ThreadPool::size() const { return threads_.size(); }

inline size_t ThreadPool::numAvailable() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return available_;
}

inline bool ThreadPool::inThreadPool() const {
  for (auto &thread : threads_) {
    if (thread.get_id() == std::this_thread::get_id()) {
      return true;
    }
  }
  return false;
}

inline void ThreadPool::run(std::function<void()> func) {
  if (threads_.empty()) {
    throw std::runtime_error("No threads to run a task");
  }
  std::unique_lock<std::mutex> lock(mutex_);

  // Set task and signal condition variable so that a worker thread will
  // wake up and use the task.
  tasks_.emplace(std::move(func));
  complete_ = false;
  condition_.notify_one();
}

inline void ThreadPool::waitWorkComplete() {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_.wait(lock, [&]() { return complete_; });
}

inline void ThreadPool::main_loop(std::size_t index) {
  std::unique_lock<std::mutex> lock(mutex_);
  while (running_) {
    // Wait on condition variable while the task is empty and
    // the pool is still running.
    condition_.wait(lock, [&]() { return !tasks_.empty() || !running_; });
    // If pool is no longer running, break out of loop.
    if (!running_) {
      break;
    }

    // Copy task locally and remove from the queue.  This is
    // done within its own scope so that the task object is
    // destructed immediately after running the task.  This is
    // useful in the event that the function contains
    // shared_ptr arguments bound via bind.
    {
      task_element_t tasks = std::move(tasks_.front());
      tasks_.pop();
      // Decrement count, indicating thread is no longer available.
      --available_;

      lock.unlock();

      // Run the task.
      try {
        if (tasks.run_with_id) {
          tasks.with_id(index);
        } else {
          tasks.no_id();
        }
      } catch (const std::exception &e) {
        printf("Exception in thread pool task: %s", e.what());
      } catch (...) {
        printf("Exception in thread pool task: unknown");
      }

      // Destruct tasks before taking the lock.  As tasks
      // are user provided std::function, they can run
      // arbitrary code during destruction, including code
      // that can reentrantly call into ThreadPool (which would
      // cause a deadlock if we were holding the lock).
    }

    // Update status of empty, maybe
    // Need to recover the lock first
    lock.lock();

    // Increment count, indicating thread is available.
    ++available_;
    if (tasks_.empty() && available_ == total_) {
      complete_ = true;
      completed_.notify_one();
    }

    // Deliberately hold the lock on the backedge, so this thread has an
    // opportunity to acquire a new task before another thread acquires
    // the lock.
  } // while running_
}
