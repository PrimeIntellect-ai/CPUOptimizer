// Must install ninja to build this extension
// Also install torch and numpy

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <optional>
#include <sched.h>
#include <thread>
#include <torch/extension.h>
#include "cpu_optimizer.hpp"
#include "threadpool.hpp"

#define POOL_NUM_THREADS 16


class StepContext {
  public:
    std::mutex running_lock;
    size_t running_l2_norm;
    size_t running_param_count;

    cpuoptim::ThreadPool work_pool;
    cpuoptim::ThreadPool sched_pool;

    StepContext()
        : running_lock(),
          running_l2_norm(0),
          running_param_count(0),
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

    float grad_l2_norm = 0.0f;
    if (optimizer->clip_max_norm != 0.0f)
        grad_l2_norm = l2_norm(grad_ptr, 0, optimizer->param_count);

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
        float global_norm = 0.0f;
        if (optimizer->clip_max_norm != 0.0f) {
            std::mutex join_lock;
            size_t threads_launched = 0;
            size_t threads_finished = 0;
            float shard_l2_norm_grads[POOL_NUM_THREADS];
            for (size_t i = 0; i < POOL_NUM_THREADS; i++) {
                size_t start_idx = i * slice_size;
                size_t end_idx = start_idx + slice_size;
                if (i == POOL_NUM_THREADS - 1) end_idx += remainder;
                if (start_idx >= end_idx) continue;
                else threads_launched++;

                work_pool->run([grad_ptr, param_ptr, start_idx, end_idx, i, &shard_l2_norm_grads, &join_lock, &threads_finished]() {
                    shard_l2_norm_grads[i] = l2_norm(grad_ptr, start_idx, end_idx);
                    join_lock.lock();
                    threads_finished++;
                    join_lock.unlock();
                });
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

            // Average the norms to get the local norm.
            float local_norm = 0.0f;
            for (size_t i = 0; i < threads_finished ; i++)
                local_norm += shard_l2_norm_grads[i];
            local_norm /= threads_finished;

            // Calculate the global norm.
            step_context->running_lock.lock();
            size_t total_params_processed = step_context->running_param_count + local_params;
            float local_proportion = (float)local_params / (float)total_params_processed;
            float global_proportion = 1.0f - local_proportion;
            global_norm = local_norm * local_proportion + step_context->running_l2_norm * global_proportion;
            step_context->running_l2_norm = global_norm;
            step_context->running_param_count = total_params_processed;
            step_context->running_lock.unlock();

            printf("Local norm: %f\n", local_norm);
            printf("Global norm: %f\n", global_norm);
            printf("Processed %zu params\n", total_params_processed);
        }

        // Launch the optimizer step, sharded across threads.
        for (size_t i = 0; i < POOL_NUM_THREADS; i++) {
            size_t start_idx = i * slice_size;
            size_t end_idx = start_idx + slice_size;
            if (i == POOL_NUM_THREADS - 1) end_idx += remainder;
            if (start_idx >= end_idx) continue;

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
