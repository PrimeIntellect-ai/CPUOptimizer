#ifndef CPU_OPTIMIZER_INCLUDE
#define CPU_OPTIMIZER_INCLUDE

#include <cstddef>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stddef.h>

// If we need to change the grad or optimizer state dtype, we shall rewrite.

// TODO: Do fmadd consistently everywhere possible
// TODO: Benchmark one_minus_beta2 * (g * g) vs (one_minus_beta2 * g) * g
#define fmadd(a, b, c) __builtin_fmaf((a), (b), (c))
#define fmaddsub(a, b, c) __builtin_fmaf((a), (b), -(c))

#define restrict __restrict__ // Restrict is a builtin in C, but not C++, so we define it to the compiler intrinsic

typedef enum {
    ADAM_STEP = 0,
    ADAMW_STEP = 1,
    ADAMW_TORCH_STEP = 2, // Torch implements AdamW incorrectly and it sometimes works better.
} StepKind;

typedef struct {
#define SER_SIZE (6 * sizeof(float) + 2 * sizeof(uint64_t))
    float beta1;
    float beta2;
    float lr;
    float eps;
    float weight_decay;
    float clip_max_norm;
    uint64_t param_count;
    uint64_t t;
    void* m_base;      // Pointer to free for m
    void* v_base;      // Pointer to free for v
    float* restrict m; // 64-byte aligned first moment
    float* restrict v; // 64-byte aligned second moment
} CPUOptimizer;

// Initialize the Adam optimizer
static CPUOptimizer* cpu_optimizer_init(int param_count, float learning_rate, float beta1, float beta2, float eps, float weight_decay, float clip_max_norm) {
    // Allocate pointer to return
    CPUOptimizer* optimizer = (CPUOptimizer*)malloc(sizeof(CPUOptimizer));
    if (optimizer == NULL) {
        fprintf(stderr, "Failed to allocate memory for AdamOptimizer\n");
        exit(1);
    }

    // Store args
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->lr = learning_rate;
    optimizer->eps = eps;
    optimizer->weight_decay = weight_decay;
    optimizer->param_count = param_count;
    optimizer->clip_max_norm = clip_max_norm;
    optimizer->t = 0;

    // Initialize the optimizer state.
    // Calloc and align to 64 bytes (The size of __m512).
    // This allows us to use aligned instructions, although parameters may not be aligned.
    size_t aligned_size = param_count * sizeof(float) + 63;
    optimizer->m_base = calloc(1, aligned_size);
    optimizer->v_base = calloc(1, aligned_size);
    if (optimizer->m_base == NULL || optimizer->v_base == NULL) {
        fprintf(stderr, "Failed to allocate %zu bytes of memory for optimizer state.\n", (size_t)2 * aligned_size);
        exit(1);
    }
    optimizer->m = (float*)(((uintptr_t)optimizer->m_base + 63) & ~63);
    optimizer->v = (float*)(((uintptr_t)optimizer->v_base + 63) & ~63);

    return optimizer;
}

static void cpu_optimizer_free(CPUOptimizer* optimizer) {
    free(optimizer->m_base);
    free(optimizer->v_base);
}

static char* cpu_optimizer_serialize(const CPUOptimizer* optimizer) {
    // Allocate all the memory we need
    size_t mv_size = optimizer->param_count * sizeof(float);
    char* buffer = (char*)malloc(SER_SIZE + (2 * mv_size));
    if (buffer == NULL) {
        fprintf(stderr, "Failed to allocate memory for serialization\n");
        exit(1);
    }

    // Copy everything in
    memcpy(buffer, optimizer, SER_SIZE);
    memcpy(buffer + SER_SIZE, optimizer->m, mv_size);
    memcpy(buffer + SER_SIZE + mv_size, optimizer->v, mv_size);
    return buffer;
}

static CPUOptimizer* cpu_optimizer_deserialize(const char* buffer) {
    // Restore the non-pointer members of the optimizer struct
    CPUOptimizer* optimizer = (CPUOptimizer*)malloc(sizeof(CPUOptimizer));
    if (optimizer == NULL) {
        fprintf(stderr, "Failed to allocate memory during deserialization\n");
        exit(1);
    }
    memcpy(optimizer, buffer, SER_SIZE);

    // Allocate aligned memory for m and v
    size_t mv_size = optimizer->param_count * sizeof(float);
    size_t aligned_size = mv_size + 63;
    optimizer->m_base = malloc(aligned_size);
    optimizer->v_base = malloc(aligned_size);
    if (optimizer->m_base == NULL || optimizer->v_base == NULL) {
        fprintf(stderr, "Failed to allocate memory during deserialization\n");
        exit(1);
    }
    optimizer->m = (float*)(((uintptr_t)optimizer->m_base + 63) & ~63);
    optimizer->v = (float*)(((uintptr_t)optimizer->v_base + 63) & ~63);

    // Copy the arrays into the newly allocated memory
    memcpy(optimizer->m, buffer + SER_SIZE, mv_size);
    memcpy(optimizer->v, buffer + SER_SIZE + mv_size, mv_size);
    return optimizer;
}


//////////
// Norm //
//////////

static double sum_squares_naive(float* restrict vec, size_t start_idx, size_t end_idx) {
    double sum_sq = 0.0; // Accumulate in double
    for (size_t i = start_idx; i < end_idx; ++i) {
        double val = (double)vec[i];
        sum_sq += val * val;
    }
    return sum_sq;
}

#if defined(__AVX512F__)
#include <immintrin.h>
static double sum_squares_avx512(float* restrict vec, size_t start_idx, size_t end_idx) {
    __m512d vsum = _mm512_setzero_pd();  // Accumulate in double

    size_t i = start_idx;
    for (; i + 15 < end_idx; i += 16) {
        // Load a vector of 16 floats, convert to 2 vectors of 8 doubles, square, sum, accumulate.
        __m512 v_float = _mm512_loadu_ps(vec + i);
        __m256 v_float_lo = _mm512_extractf32x8_ps(v_float, 0);
        __m256 v_float_hi = _mm512_extractf32x8_ps(v_float, 1);
        __m512d v_double_lo = _mm512_cvtps_pd(v_float_lo);
        __m512d v_double_hi = _mm512_cvtps_pd(v_float_hi);
        __m512d lo_squared = _mm512_mul_pd(v_double_lo, v_double_lo);
        __m512d hi_squared = _mm512_mul_pd(v_double_hi, v_double_hi);
        __m512d sum_16 = _mm512_add_pd(lo_squared, hi_squared);
        vsum = _mm512_add_pd(vsum, sum_16);
    }

    // Sum the 8 doubles of the accumulator
    double sum_sq = _mm512_reduce_add_pd(vsum);

    // Process any remaining elements
    for (; i < end_idx; i++) {
        double val = (double)vec[i];
        sum_sq += val * val;
    }

    return sqrt(sum_sq);
}
#endif

static double sum_squares(float* restrict vec, size_t start_idx, size_t end_idx) {
#if !defined(__AVX512F__)
    return sum_squares_naive(vec, start_idx, end_idx);
#else
    return sum_squares_avx512(vec, start_idx, end_idx);
#endif
}

//////////
// STEP //
//////////

template<StepKind stepkind>
static void adam_step_naive(CPUOptimizer* optimizer, float* restrict param, float* restrict grad, size_t start_idx, size_t end_idx, float grad_l2_norm) {
    optimizer->t += 1;
    size_t t = optimizer->t;
    float lr = optimizer->lr;
    float beta1 = optimizer->beta1;
    float beta2 = optimizer->beta2;
    float eps = optimizer->eps;
    float weight_decay = optimizer->weight_decay;
    float max_norm = optimizer->clip_max_norm;

    float beta1_t = powf(beta1, t);
    float beta2_t = powf(beta2, t);
    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;
    float one_minus_beta1_t = 1.0f - beta1_t;
    float one_minus_beta2_t = 1.0f - beta2_t;
    float inv_one_minus_beta1_t = 1.0f / one_minus_beta1_t;
    float inv_one_minus_beta2_t = 1.0f / one_minus_beta2_t;
    float inv_one_minus_beta_2t_sqrt = 1.0f / sqrtf(one_minus_beta2_t);
    float step_size = lr * inv_one_minus_beta1_t;
    float weight_decay_factor = (weight_decay != 0.0f) * weight_decay * (stepkind != StepKind::ADAM_STEP ? lr : 1);
    float clip_grad_norm_scale = (max_norm != 0.0f) ? (max_norm / (grad_l2_norm + 0.000001f)) : 1;

    for(uint64_t i = start_idx; i < end_idx; i++) {
        float g = grad[i];
        float p = param[i];
        float m_ = optimizer->m[i];
        float v_ = optimizer->v[i];

        // Apply grad clipping
        g *= clip_grad_norm_scale;

        // Apply weight decay
        if constexpr (stepkind == StepKind::ADAM_STEP) {
            g += weight_decay_factor * p;
        } else if constexpr (stepkind == StepKind::ADAMW_STEP) {
            p -= weight_decay_factor * p;
        } else if constexpr (stepkind == StepKind::ADAMW_TORCH_STEP) {
            p *= 1 - weight_decay_factor;
        }

        if constexpr (stepkind == StepKind::ADAM_STEP || stepkind == StepKind::ADAMW_STEP) {
            // Update first and second moments
            float m = beta1 * m_ + one_minus_beta1 * g;
            float v = beta2 * v_ + one_minus_beta2 * g * g;
            optimizer->m[i] = m;
            optimizer->v[i] = v;

            // Apply parameter update
            float m_hat = m * inv_one_minus_beta1_t;
            float v_hat = v * inv_one_minus_beta2_t;
            param[i] = p - (lr * m_hat / (sqrtf(v_hat) + eps));
        } else if constexpr (stepkind == StepKind::ADAMW_TORCH_STEP) {
            // Update first and second moments
            // exp_avg.lerp_(grad, 1 - beta1)
            float m = m_ + one_minus_beta1 * (g - m_);
            // exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            float v = beta2 * v_ + one_minus_beta2 * g * g;
            optimizer->m[i] = m;
            optimizer->v[i] = v;

            // Apply parameter update
            // denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            float denom = sqrtf(v) * inv_one_minus_beta_2t_sqrt + eps;
            // param.addcdiv_(exp_avg, denom, value=-step_size)
            param[i] = p - (m * step_size / denom);
        }
    }
}

#if defined(__AVX512F__)
#include <immintrin.h>
template<StepKind stepkind>
static void adam_step_avx512(CPUOptimizer* optimizer, float* restrict param, float* restrict grad, size_t start_idx, size_t end_idx, float grad_l2_norm) {
    // Update time step and extract parameters.
    optimizer->t += 1;
    size_t t      = optimizer->t;
    float lr      = optimizer->lr;
    float beta1   = optimizer->beta1;
    float beta2   = optimizer->beta2;
    float eps     = optimizer->eps;
    float weight_decay      = optimizer->weight_decay;
    float max_norm= optimizer->clip_max_norm;

    // Compute bias correction factors.
    float beta1_t  = powf(beta1, t);
    float beta2_t  = powf(beta2, t);
    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;
    float one_minus_beta1_t = 1.0f - beta1_t;
    float one_minus_beta2_t = 1.0f - beta2_t;
    float inv_one_minus_beta1_t = 1.0f / one_minus_beta1_t;
    float inv_one_minus_beta2_t = 1.0f / one_minus_beta2_t;
    float inv_one_minus_beta_2t_sqrt = 1.0f / sqrtf(one_minus_beta2_t);
    float step_size = lr * inv_one_minus_beta1_t;

    // Compute weight decay factor.
    float weight_decay_factor = (weight_decay != 0.0f) * weight_decay * (stepkind != StepKind::ADAM_STEP ? lr : 1);
    float one_minus_wdf = 1.0f - weight_decay_factor;

    // Compute gradient clipping scale.
    float clip_grad_norm_scale = (max_norm != 0.0f) ? (max_norm / (grad_l2_norm + 0.000001f)) : 1;

    // Broadcast constants.
    __m512 beta1_vec = _mm512_set1_ps(beta1);
    __m512 beta2_vec = _mm512_set1_ps(beta2);
    __m512 one_minus_beta1_vec = _mm512_set1_ps(one_minus_beta1);
    __m512 one_minus_beta2_vec = _mm512_set1_ps(one_minus_beta2);
    __m512 inv_one_minus_beta1_t_vec = _mm512_set1_ps(inv_one_minus_beta1_t);
    __m512 inv_one_minus_beta2_t_vec = _mm512_set1_ps(inv_one_minus_beta2_t);
    __m512 inv_one_minus_beta_2t_sqrt_vec = _mm512_set1_ps(inv_one_minus_beta_2t_sqrt);
    __m512 lr_vec = _mm512_set1_ps(lr);
    __m512 eps_vec = _mm512_set1_ps(eps);
    __m512 weight_decay_vec = _mm512_set1_ps(weight_decay_factor);
    __m512 one_minus_wdf_vec = _mm512_set1_ps(one_minus_wdf);
    __m512 clip_scale_vec = _mm512_set1_ps(clip_grad_norm_scale);
    __m512 step_size_vec = _mm512_set1_ps(step_size);

    size_t i = start_idx;
    for(; i + 15 < end_idx; i += 16) {
        // Load 16 elements.
        __m512 grad_vec = _mm512_loadu_ps(&grad[i]);
        __m512 param_vec = _mm512_loadu_ps(&param[i]);
        __m512 m_prev_vec = _mm512_load_ps(&optimizer->m[i]);
        __m512 v_prev_vec = _mm512_load_ps(&optimizer->v[i]);

        // Apply gradient clipping.
        grad_vec = _mm512_mul_ps(grad_vec, clip_scale_vec);

        // Apply weight decay.
        if constexpr (stepkind == StepKind::ADAM_STEP) {
            grad_vec = _mm512_fmadd_ps(weight_decay_vec, param_vec, grad_vec);
        } else if constexpr (stepkind == StepKind::ADAMW_STEP) {
            param_vec = _mm512_fmaddsub_ps(weight_decay_vec, param_vec, param_vec);
        } else if constexpr (stepkind == StepKind::ADAMW_TORCH_STEP) {
            param_vec = _mm512_mul_ps(param_vec, one_minus_wdf_vec);
        }

        if constexpr (stepkind == StepKind::ADAM_STEP || stepkind == StepKind::ADAMW_STEP) {
            // v = beta2 * v_prev + (1 - beta2) * (grad * grad).
            __m512 m_vec = _mm512_fmadd_ps(beta1_vec, m_prev_vec, _mm512_mul_ps(one_minus_beta1_vec, grad_vec));
            // m = beta1 * m_prev + (1 - beta1) * grad.
            __m512 grad_sq = _mm512_mul_ps(grad_vec, grad_vec);
            __m512 v_vec = _mm512_fmadd_ps(beta2_vec, v_prev_vec, _mm512_mul_ps(one_minus_beta2_vec, grad_sq));
            _mm512_store_ps(&optimizer->m[i], m_vec);
            _mm512_store_ps(&optimizer->v[i], v_vec);
            // Apply parameter update.
            __m512 m_hat = _mm512_mul_ps(m_vec, inv_one_minus_beta1_t_vec);
            __m512 v_hat = _mm512_mul_ps(v_vec, inv_one_minus_beta2_t_vec);
            __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), eps_vec);
            __m512 update = _mm512_div_ps(_mm512_mul_ps(lr_vec, m_hat), denom);
            _mm512_storeu_ps(&param[i], _mm512_sub_ps(param_vec, update));
        } else if constexpr (stepkind == StepKind::ADAMW_TORCH_STEP) {
            // m = m_prev + (1 - beta1) * (grad - m_prev).
            __m512 m_vec = _mm512_fmadd_ps(one_minus_beta1_vec, _mm512_sub_ps(grad_vec, m_prev_vec), m_prev_vec);
            // v = beta2 * v_prev + (1 - beta2) * (grad * grad).
            __m512 v_vec = _mm512_fmadd_ps(beta2_vec, v_prev_vec, _mm512_mul_ps(one_minus_beta2_vec, _mm512_mul_ps(grad_vec, grad_vec)));
            _mm512_store_ps(&optimizer->m[i], m_vec);
            _mm512_store_ps(&optimizer->v[i], v_vec);
            // Parameter update.
            __m512 denom = _mm512_fmadd_ps(_mm512_sqrt_ps(v_vec), inv_one_minus_beta_2t_sqrt_vec, eps_vec);
            __m512 update = _mm512_div_ps(_mm512_mul_ps(m_vec, step_size_vec), denom);
            _mm512_storeu_ps(&param[i], _mm512_sub_ps(param_vec, update));
        }
    }

    // Scalar tail for remaining elements.
    for(; i < end_idx; i++) {
        float g = grad[i];
        float p = param[i];
        float m_ = optimizer->m[i];
        float v_ = optimizer->v[i];

        g *= clip_grad_norm_scale;

        if constexpr (stepkind == StepKind::ADAM_STEP) {
            g = fmadd(weight_decay_factor, p, g);
        } else if constexpr (stepkind == StepKind::ADAMW_STEP) {
            p = fmaddsub(weight_decay_factor, p, p);
        } else if constexpr (stepkind == StepKind::ADAMW_TORCH_STEP) {
            p *= one_minus_wdf;
        }

        if constexpr (stepkind == StepKind::ADAM_STEP || stepkind == StepKind::ADAMW_STEP) {
            // float m = beta1 * m_ + one_minus_beta1 * g;
            float m = fmadd(beta1, m_, (one_minus_beta1 * g));
            // float v = beta2 * v_ + one_minus_beta2 * g * g; 
            float v = fmadd(beta2, v_, (one_minus_beta2 * (g * g)));
            optimizer->m[i] = m;
            optimizer->v[i] = v;

            float m_hat = m * inv_one_minus_beta1_t;
            float v_hat = v * inv_one_minus_beta2_t;
            param[i] = p - (lr * m_hat / (sqrtf(v_hat) + eps));
        } else if constexpr (stepkind == StepKind::ADAMW_TORCH_STEP) {
            // float m = m_ + one_minus_beta1 * (g - m_);
            float m = fmadd(one_minus_beta1, (g - m_), m_);
            // float v = beta2 * v_ + one_minus_beta2 * g * g;
            float v = fmadd(beta2, v_, (one_minus_beta2 * (g * g)));
            optimizer->m[i] = m;
            optimizer->v[i] = v;

            float denom = fmadd(sqrtf(v), inv_one_minus_beta_2t_sqrt, eps);
            param[i] = p - ((m * step_size) / denom);
        }
    }
}
#endif

template<StepKind stepkind>
static void adam_step(CPUOptimizer* optimizer, float* restrict param, float* restrict grad, size_t start_idx, size_t end_idx, float grad_l2_norm) {
#if !defined(__AVX512F__)
    adam_step_naive<stepkind>(optimizer, param, grad, start_idx, end_idx, grad_l2_norm);
#else
    adam_step_avx512<stepkind>(optimizer, param, grad, start_idx, end_idx, grad_l2_norm);
#endif
}

#endif /* CPU_OPTIMIZER_INCLUDE */
