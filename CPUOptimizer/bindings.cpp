// Must install ninja to build this extension
// Also install torch and numpy

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <sched.h>
#include <thread>
#include <vector>

#include <torch/extension.h>
#include "cpu_optimizer.hpp"
#include "threadpool.hpp"

#ifdef __SIZEOF_FLOAT128__
#include <quadmath.h>
typedef __float128 ultra_float;
#define SQRTQ(x) sqrtq(x)
#else
typedef long double ultra_float;
#define SQRTQ(x) sqrtl(x)
#endif

#define POOL_NUM_THREADS 16

class StepContext {
  public:
    std::mutex running_lock;
    size_t running_param_count;
    std::vector<long double> running_sum_squares;

    cpuoptim::ThreadPool work_pool;
    cpuoptim::ThreadPool sched_pool;

    StepContext()
        : running_lock(),
          running_param_count(0),
          running_sum_squares((size_t)512),
          work_pool(POOL_NUM_THREADS), // Ordering does not matter here.
          sched_pool(1) // Enforce strict ordering with the queue.
          {}
    ~StepContext() {
        // We have two tiers of threadpools. The sched pool puts work onto the work pool.
        // The order these pools are destroyed is important, because sched_pool puts work into the work_pool.
        sched_pool.waitWorkComplete();
        work_pool.waitWorkComplete();
    }
};

static inline void sanity_check(CPUOptimizer* optimizer, torch::Tensor& param, torch::Tensor& grad) {
    TORCH_CHECK(optimizer != NULL, "optimizer must not be null");
    TORCH_CHECK(param.is_contiguous(), "param must be contiguous");
    TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous");
    TORCH_CHECK(param.dtype() == torch::kFloat32, "param must be float32");
    TORCH_CHECK(grad.dtype() == torch::kFloat32, "grad must be float32");
    TORCH_CHECK(param.sizes() == grad.sizes(), "param and grad must have same shape");
    TORCH_CHECK((uint64_t)param.numel() == optimizer->param_count, "parameter count mismatch");
    TORCH_CHECK((uint64_t)grad.numel() == optimizer->param_count, "gradient count mismatch");
}

template<StepKind stepkind>
static void step_binding(
    CPUOptimizer* optimizer,
    torch::Tensor& param,
    torch::Tensor& grad
) {
    sanity_check(optimizer, param, grad);
    float* grad_ptr = grad.data_ptr<float>();
    float* param_ptr = param.data_ptr<float>();

    long double grad_l2_norm = 0.0;
    if (optimizer->clip_max_norm != 0.0f)
        grad_l2_norm = sqrtl(sum_squares(grad_ptr, 0, optimizer->param_count));

    optimizer->t += 1;
    adam_step<stepkind>(optimizer, param_ptr, grad_ptr, 0, optimizer->param_count, grad_l2_norm);
}

template<StepKind stepkind>
static void step_binding_async(
    CPUOptimizer* optimizer,
    torch::Tensor& param,
    torch::Tensor& grad,
    StepContext* step_context
) {
    sanity_check(optimizer, param, grad);

    cpuoptim::ThreadPool* work_pool = &step_context->work_pool;
    float* grad_ptr = grad.data_ptr<float>();
    float* param_ptr = param.data_ptr<float>();
    uint64_t local_params = optimizer->param_count;

    // Throw onto sched pool and immediately return.
    step_context->sched_pool.run([=]() {
        // Slice the work into parts.
        size_t slice_size = local_params / POOL_NUM_THREADS;
        size_t remainder = local_params % POOL_NUM_THREADS;

        // Launch the grad norms sharded across threads.
        long double global_norm = 0.0f;
        if (optimizer->clip_max_norm != 0.0f) {
            std::mutex join_lock;
            size_t threads_launched = 0;
            size_t threads_finished = 0;

            long double shard_sum_squares[POOL_NUM_THREADS];
            size_t work_idx = 0;
            size_t thread_idx = 0;
            while (work_idx < local_params) {
                // Get the start and end indexes. Make sure the start is 64-byte aligned (16 floats).
                size_t start_idx = work_idx;
                work_idx = (work_idx + slice_size + 15) & ~15; // Round up to next alignment boundary
                size_t end_idx = work_idx;
                if (end_idx > local_params) end_idx = local_params;
                if (start_idx >= end_idx) break;

                threads_launched++;
                work_pool->run([grad_ptr, param_ptr, start_idx, end_idx, thread_idx, &shard_sum_squares, &join_lock, &threads_finished]() {
                    shard_sum_squares[thread_idx] = sum_squares(grad_ptr, start_idx, end_idx);
                    join_lock.lock();
                    threads_finished++;
                    join_lock.unlock();
                });
                thread_idx++;
            }

            // Wait for the completion of the local norm shards.
            while (1) {
                join_lock.lock();
                size_t finished = threads_finished;
                join_lock.unlock();
                if (finished == threads_launched)
                    break;
                std::this_thread::yield();
            }

            // Combine the shard square sums to get the local norm.
            long double local_sum_sq = neumaier_sum(shard_sum_squares, threads_finished);
            long double local_norm = sqrtl(local_sum_sq);

            // Calculate the global norm in maximal precision.
            step_context->running_lock.lock();
            step_context->running_param_count += local_params;
            step_context->running_sum_squares.push_back(local_sum_sq);
            size_t vec_sz = step_context->running_sum_squares.size();
            long double global_norm = neumaier_sum(step_context->running_sum_squares.data(), vec_sz);
            global_norm = (long double)SQRTQ((ultra_float)global_norm); // Clangd lies :)
            step_context->running_lock.unlock();
        }

        // Launch the optimizer step, sharded across threads.
        optimizer->t += 1; // Increment the step counter before launching. No sync here b/c pool of size 1.

        size_t work_idx = 0;
        while (work_idx < local_params) {
            // Get the start and end indexes. Make sure the start is 64-byte aligned (16 floats).
            size_t start_idx = work_idx;
            work_idx = (work_idx + slice_size + 15) & ~15; // Round up to next alignment boundary
            size_t end_idx = work_idx;
            if (end_idx > local_params) end_idx = local_params;
            if (start_idx >= end_idx) break;

            work_pool->run([=]() {
                adam_step<stepkind>(optimizer, param_ptr, grad_ptr, start_idx, end_idx, global_norm);
            });
        }

        // No need to explicitly join here, we shall let the StepContext destructor wait on all the parameter updates.
    });

    return;
}

template<StepKind stepkind>
static void step_sync_or_async(
    CPUOptimizer* optimizer,
    torch::Tensor& param,
    torch::Tensor& grad,
    StepContext* step_context
) {
    if (step_context != nullptr) {
        step_binding_async<stepkind>(optimizer, param, grad, step_context);
    } else {
        step_binding<stepkind>(optimizer, param, grad);
    }
}

//////////////
// Bindings //
//////////////


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    py::class_<CPUOptimizer>(m, "OptimizerBinding", py::module_local())
        .def_readonly("grad", &CPUOptimizer::param_count)
        .def_readwrite("lr", &CPUOptimizer::lr)
        .def_readwrite("beta1", &CPUOptimizer::beta1)
        .def_readwrite("beta2", &CPUOptimizer::beta2)
        .def_readwrite("eps", &CPUOptimizer::eps)
        .def_readwrite("weight_decay", &CPUOptimizer::weight_decay)
        .def_readwrite("clip_max_norm", &CPUOptimizer::clip_max_norm)
        .def_readwrite("t", &CPUOptimizer::t);

    py::class_<StepContext>(m, "StepContext", py::module_local());

    m.def("create_optimizer", [](torch::Tensor& grad, float lr, float beta1, float beta2, float epsilon, float weight_decay, float clip_max_norm) {
        TORCH_CHECK(grad.defined(), "grad tensor must not be null");
        TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous");
        TORCH_CHECK(grad.dtype() == torch::kFloat32, "grad must be float32");
        TORCH_CHECK(grad.numel() > 0, "grad tensor must not be empty");
        int64_t param_count = grad.numel();
        CPUOptimizer* opt = cpu_optimizer_init(param_count, lr, beta1, beta2, epsilon, weight_decay, clip_max_norm);
        TORCH_CHECK(opt != NULL, "Failed to allocate optimizer");
        return opt;
    }, "Create Adam optimizer",
        py::arg("grad"),
        py::arg("lr") = 0.001f,
        py::arg("beta1") = 0.9f,
        py::arg("beta2") = 0.999f,
        py::arg("epsilon") = 1e-8f,
        py::arg("weight_decay") = 0.0f,
        py::arg("clip_max_norm") = 0.0f);

    m.def("destroy_optimizer", [](CPUOptimizer* optimizer) {
        cpu_optimizer_free(optimizer);
    }, "Free Adam optimizer memory");

    m.def("create_step_context", []() {
        return new StepContext();
    }, "Initialize the threadpools for this optimizer context.");

    // Steps

    m.def("step_adam", &step_sync_or_async<StepKind::ADAM_STEP>, "",
        py::arg("optimizer"),
        py::arg("param"),
        py::arg("grad"),
        py::arg("step_context") = nullptr);

    m.def("step_adamw", &step_sync_or_async<StepKind::ADAMW_STEP>, "",
        py::arg("optimizer"),
        py::arg("param"),
        py::arg("grad"),
        py::arg("step_context") = nullptr);

    m.def("step_adamw_torch", &step_sync_or_async<StepKind::ADAMW_TORCH_STEP>, "",
        py::arg("optimizer"),
        py::arg("param"),
        py::arg("grad"),
        py::arg("step_context") = nullptr);

    // Other stuff

    m.def("vector_width", []() {
#if defined(__AVX512F__)
        return 512;
#else
        return 1;
#endif
    }, "Get simd vector width (1=Scalar, 512=AVX512)");

    m.def("serialize", [](CPUOptimizer* optimizer) {
        char* buffer = cpu_optimizer_serialize(optimizer);
        size_t size = SER_SIZE + (optimizer->param_count * sizeof(float));
        py::bytes result(buffer, size);
        free(buffer);
        return result;
    }, "Serialize optimizer to bytes");

    m.def("deserialize", [](py::bytes data) {
        char* buffer = PyBytes_AS_STRING(data.ptr());
        CPUOptimizer* opt = cpu_optimizer_deserialize(buffer);
        return opt;
    }, "Deserialize optimizer from bytes");
}
