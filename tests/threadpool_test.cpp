#include "../CPUOptimizer/threadpool.hpp"
#include <cstddef>
#include <atomic>

void taskfib(ThreadPool* pool, std::atomic<size_t>* accum, int n) {
    if (n <= 1) {
        accum->fetch_add(n, std::memory_order_relaxed);
    } else {
        pool->run([=]() { taskfib(pool, accum, n - 1); });
        pool->run([=]() { taskfib(pool, accum, n - 2); });
    }
}

int main() {
    printf("Stress testing thread pool implementation.\n");

    size_t fibn = 35;
    size_t n_threads = 20;

    std::atomic<size_t> accum{0};
    std::atomic<size_t>* _accum = &accum;
    ThreadPool pool(20);
    ThreadPool* _pool = &pool;

    printf("Stress testing threadpool by computing fib(%zu) on %zu threads.\n", fibn, n_threads);
    pool.run([=]() { taskfib(_pool, _accum, fibn); });
    pool.waitWorkComplete();

    printf("Work completed.\n");
    printf("Final answer: %zu\n", accum.load());

    return 0;
}
