#include "../CPUOptimizer/threadpool.hpp"
#include <cstddef>
#include <pthread.h>

;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void taskfib(ThreadPool* pool, size_t* accum, int n) {
    if (n <= 1) {
        pthread_mutex_lock(&mutex);
        *accum += n;
        pthread_mutex_unlock(&mutex);
    } else {
        pool->run([=]() { taskfib(pool, accum, n - 1); });
        pool->run([=]() { taskfib(pool, accum, n - 2); });
    }
}

int main() {
    printf("Stress testing thread pool implementation.\n");

    size_t fibn = 35;
    size_t n_threads = 20;

    size_t accum = 0;
    size_t* _accum = &accum;
    ThreadPool pool(20);
    ThreadPool* _pool = &pool;

    printf("Stress testing threadpool by computing fib(%zu) on %zu threads.\n", fibn, n_threads);
    pool.run([=]() { taskfib(_pool, _accum, fibn); });
    pool.waitWorkComplete();

    printf("Work completed\n");

    printf("Final answer: %zu\n", accum);
    printf("Correct answer: 75025\n");

    return 0;
}