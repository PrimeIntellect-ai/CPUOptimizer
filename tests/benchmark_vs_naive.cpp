#include "../CPUOptimizer/cpu_optimizer.hpp"
#include <algorithm>
#include <cmath>

// c++ benchmark_vs_naive.cpp -lm -O3 -march=native -fno-math-errno

#define PARAM_COUNT 10000000

typedef enum {
    NAIVE = 0,
    AVX512 = 1,
} OptLevel;


template<StepKind stepkind, OptLevel opt_level>
static double test_impl(float** out_params) {
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
    CPUOptimizer* optimizer = cpu_optimizer_init(PARAM_COUNT, learning_rate, beta1, beta2, epsilon, weight_decay, clip_max_norm);
    
    // Time the optimization steps
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    if (opt_level == NAIVE) {
        for (int i = 0; i < 100; i++) {  // Increase iterations for better timing
            optimizer->t += 1;
            double norm = sqrt(sum_squares_naive(gradients, 0, PARAM_COUNT));
            adam_step_naive<stepkind>(optimizer, params, gradients, 0, PARAM_COUNT, (float)norm);
        }
    } else if (opt_level == AVX512) {
        for (int i = 0; i < 100; i++) {  // Increase iterations for better timing
            optimizer->t += 1;
            double norm = sqrt(sum_squares(gradients, 0, PARAM_COUNT));
            adam_step<stepkind>(optimizer, params, gradients, 0, PARAM_COUNT, norm);
        }
    } else {
        fprintf(stderr, "Invalid opt_level: %d\n", opt_level);
        exit(1);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    cpu_optimizer_free(optimizer);
    free(optimizer);
    free(gradients);
    
    *out_params = params;
    return time_taken;
}

#include <float.h>
#ifdef __SIZEOF_FLOAT128__
    typedef __float128 ultra_float;
    #define ULTRA_FLOAT_AVAILABLE 1
#else
    typedef long double ultra_float;
    #define ULTRA_FLOAT_AVAILABLE 0
#endif

// Comparison function for sorting
static int compare_floats(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

static float ultra_precise_average(float* array, size_t length) {
    if (length == 0) return 0.0;
    if (length == 1) return (ultra_float)array[0];
    
    // Step 1: Sort the array to group similar magnitudes
    float* sorted = (float*)malloc(length * sizeof(float));
    memcpy(sorted, array, length * sizeof(float));
    qsort(sorted, length, sizeof(float), compare_floats);
    
    // Step 2: Pairwise summation with extended precision
    size_t remaining = length;
    ultra_float* temp = (ultra_float*)malloc(length * sizeof(ultra_float));
    
    // Convert to higher precision
    for (size_t i = 0; i < length; i++) {
        temp[i] = (ultra_float)sorted[i];
    }
    
    // Perform pairwise summation
    while (remaining > 1) {
        size_t i;
        for (i = 0; i < remaining / 2; i++) {
            temp[i] = temp[2*i] + temp[2*i + 1];
        }
        if (remaining % 2) {
            temp[i] = temp[remaining - 1];
            remaining = i + 1;
        } else {
            remaining = i;
        }
    }
    
    ultra_float result = temp[0] / (ultra_float)length;
    
    // Cleanup
    free(sorted);
    free(temp);
    
    return (float)result;
}

void verify_results(float* baseline, float* test, const char* impl_name) {
    // Get an average and max deviation
    float dev, max_dev = 0;
    float* deviations = (float*)malloc(PARAM_COUNT * sizeof(float));
    for (int i = 0; i < PARAM_COUNT; i++) {
        dev = deviations[i] = fabsf(baseline[i] - test[i]);
        if (dev > max_dev) max_dev = dev;        
    }

    float avg_dev = ultra_precise_average(deviations, PARAM_COUNT);

    printf("Max deviation: %f\n", max_dev);
    printf("Avg deviation: %f\n", avg_dev);
    printf("\n");
    free(deviations);
}

template<StepKind stepkind>
void run_test_for_kind() {
    float *params_naive, *params_avx2, *params_avx512;

    const char* stepkind_name;
    if (stepkind == StepKind::ADAM_STEP) {
        stepkind_name = "adam_step";
    } else if (stepkind == StepKind::ADAMW_STEP) {
        stepkind_name = "adamw_step";
    } else if (stepkind == StepKind::ADAMW_TORCH_STEP) {
        stepkind_name = "adamw_torch_step";
    } else {
        fprintf(stderr, "Invalid step kind: %d\n", stepkind);
        exit(1);
    }

    double time_naive = test_impl<stepkind, OptLevel::NAIVE>(&params_naive);
    printf("Naive %s implementation: %.3f seconds\n", stepkind_name, time_naive);

#if defined(__AVX512F__)
    double time_avx512 = test_impl<stepkind, OptLevel::AVX512>(&params_avx512);
    
    printf("AVX-512 %s implementation: %.3f seconds \033[31m(%.2fx speedup)\033[0m\n",
           stepkind_name, time_avx512, time_naive/time_avx512);
    verify_results(params_naive, params_avx512, "AVX-512");
    free(params_avx512);
#else
    printf("\033[31mThis CPU does not support AVX-512, so no speedup can be measured.\033[0m\n\n");
#endif

    free(params_naive);
}

int main(void) {
    
    printf("\n\033[35m🦋 Benchmarking vectorized implementations vs naive C implementation.\033[0m\n\n");
    
    run_test_for_kind<StepKind::ADAM_STEP>();
    run_test_for_kind<StepKind::ADAMW_STEP>();
    run_test_for_kind<StepKind::ADAMW_TORCH_STEP>();

    return 0;
}
