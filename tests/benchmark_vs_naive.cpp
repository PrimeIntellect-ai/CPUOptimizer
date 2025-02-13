#include "../CPUOptimizer/offload_adam.hpp"

// c++ benchmark_vs_naive.cpp -lm -O3 -march=native -fno-math-errno

#define PARAM_COUNT 10000000

static double test_impl(void step_fn(AdamOptimizer* optimizer, float* volatile params, float* volatile gradients), float** out_params) {
    float* params = (float*)malloc(PARAM_COUNT * sizeof(float));
    float* gradients = (float*)malloc(PARAM_COUNT * sizeof(float));
    if (params == NULL || gradients == NULL) {
        fprintf(stderr, "Failed to allocate %zu bytes of memory for params and gradients.\n", (size_t)PARAM_COUNT * 2 * sizeof(float));
        exit(1);
    }   

    // Create some data
    for (int i = 0; i < PARAM_COUNT; i++) {
        params[i] = (float)(i + 1);
        gradients[i] = (float)(i + 1) * 0.1f * (i % 2 == 0 ? 1 : -1);
    }

    float learning_rate = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float weight_decay = 0.0f;
    float clip_max_norm = 1.0f;
    AdamOptimizer* optimizer = adam_init(PARAM_COUNT, learning_rate, beta1, beta2, epsilon, weight_decay, clip_max_norm);
    
    // Time the optimization steps
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < 100; i++) {  // Increase iterations for better timing
        step_fn(optimizer, params, gradients);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    adam_free(optimizer);
    free(optimizer);
    free(gradients);
    
    *out_params = params;
    return time_taken;
}

void verify_results(float* baseline, float* test, const char* impl_name) {
    for (int i = 0; i < PARAM_COUNT; i++) {
        if (fabsf(baseline[i] - test[i]) > 1e-5f) {
            printf("Mismatch at index %d between naive and %s: %f != %f\n", 
                   i, impl_name, baseline[i], test[i]);
            exit(1);
        }
    }
    printf("Results match between naive and %s!\n", impl_name);
}

int main(void) {
    float *params_naive, *params_avx2, *params_avx512;
    double time_naive = test_impl(adam_step_naive<StepKind::ADAM_STEP>, &params_naive);

    printf("\n\033[35m🦋 Benchmarking vectorized implementations vs naive C implementation.\033[0m\n\n");
    printf("Naive Adam implementation: %.3f seconds\n\n", time_naive);

#if defined(__AVX512F__)
    double time_avx512 = test_impl(adam_step_avx512<StepKind::ADAM_STEP>, &params_avx512);
    verify_results(params_naive, params_avx512, "AVX-512");
    printf("AVX-512 Adam implementation: %.3f seconds \033[31m(%.2fx speedup)\033[0m\n\n", 
           time_avx512, time_naive/time_avx512);
    free(params_avx512);
#endif

    // And now for adamw
    time_naive = test_impl(adam_step_naive<StepKind::ADAMW_STEP>, &params_naive);
    printf("Naive AdamW implementation: %.3f seconds\n\n", time_naive);

#if defined(__AVX512F__)
    time_avx512 = test_impl(adam_step_avx512<StepKind::ADAMW_STEP>, &params_avx512);
    verify_results(params_naive, params_avx512, "AVX-512");
    printf("AVX-512 AdamW implementation: %.3f seconds \033[31m(%.2fx speedup)\033[0m\n\n", 
           time_avx512, time_naive/time_avx512);
    free(params_avx512);
#endif

    free(params_naive);
    return 0;
}
