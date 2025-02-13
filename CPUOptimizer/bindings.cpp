#include <torch/extension.h>
#include "offload_adam.hpp"

// Must install ninja to build this extension
// Also install torch and numpy

//////////////////////
// Create / Destroy //
//////////////////////

AdamOptimizer* create_optimizer(torch::Tensor& grad, float lr, float beta1, float beta2, float epsilon, float weight_decay, float clip_max_norm) {
    TORCH_CHECK(grad.defined(), "grad tensor must not be null");
    TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous");
    TORCH_CHECK(grad.dtype() == torch::kFloat32, "grad must be float32");
    TORCH_CHECK(grad.numel() > 0, "grad tensor must not be empty");
    int64_t param_count = grad.numel();
    AdamOptimizer* opt = adam_init(param_count, lr, beta1, beta2, epsilon, weight_decay, clip_max_norm);
    TORCH_CHECK(opt != nullptr, "Failed to allocate optimizer");
    return opt;
}

void destroy_optimizer(AdamOptimizer* optimizer) {
    adam_free(optimizer);
}

#define STEP_CHECKS() \
    TORCH_CHECK(optimizer != nullptr, "optimizer must not be null"); \
    TORCH_CHECK(param.is_contiguous(), "param must be contiguous"); \
    TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous"); \
    TORCH_CHECK(param.dtype() == torch::kFloat32, "param must be float32"); \
    TORCH_CHECK(grad.dtype() == torch::kFloat32, "grad must be float32"); \
    TORCH_CHECK(param.sizes() == grad.sizes(), "param and grad must have same shape"); \
    TORCH_CHECK((uint64_t)param.numel() == optimizer->param_count, "parameter count mismatch");


//////////
// Adam //
//////////

torch::Tensor step_adam_naive(
    AdamOptimizer* optimizer,
    torch::Tensor& param,
    torch::Tensor& grad
) {
    STEP_CHECKS()
    adam_step_naive<StepKind::ADAM_STEP>(optimizer, param.data_ptr<float>(),  grad.data_ptr<float>());
    return param;
}

#if defined(__AVX512F__)
torch::Tensor step_adam_avx512(
    AdamOptimizer* optimizer,
    torch::Tensor& param,
    torch::Tensor& grad
) {
    STEP_CHECKS()
    adam_step_avx512<StepKind::ADAM_STEP>(optimizer, param.data_ptr<float>(),  grad.data_ptr<float>());
    return param;
}
#endif

torch::Tensor step_adam(
    AdamOptimizer* optimizer,
    torch::Tensor& param,
    torch::Tensor& grad
) {
#if defined(__AVX512F__)
    return step_adam_avx512(optimizer, param, grad);
#else
    return step_adam_naive(optimizer, param, grad);
#endif
    return param;
}


///////////
// AdamW //
///////////

torch::Tensor step_adamw_naive(
    AdamOptimizer* optimizer,
    torch::Tensor& param,
    torch::Tensor& grad
) {
    STEP_CHECKS()
    adam_step_naive<StepKind::ADAMW_STEP>(optimizer, param.data_ptr<float>(),  grad.data_ptr<float>());
    return param;
}

#if defined(__AVX512F__)
torch::Tensor step_adamw_avx512(
    AdamOptimizer* optimizer,
    torch::Tensor& param,
    torch::Tensor& grad
) {
    STEP_CHECKS()
    adam_step_avx512<StepKind::ADAMW_STEP>(optimizer, param.data_ptr<float>(),  grad.data_ptr<float>());
    return param;
}
#endif

torch::Tensor step_adamw(
    AdamOptimizer* optimizer,
    torch::Tensor& param,
    torch::Tensor& grad
) {
#if defined(__AVX512F__)
    return step_adam_avx512(optimizer, param, grad);
#else
    return step_adam_naive(optimizer, param, grad);
#endif
    return param;
}


//////////////
// Bindings //
//////////////

int vector_width(void) {
#if defined(__AVX512F__)
    return 512;
#else
    return 1;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Create/destroy
    py::class_<AdamOptimizer>(m, "AdamOptimizer", py::module_local())
        .def_readonly("grad", &AdamOptimizer::param_count)
        .def_readwrite("lr", &AdamOptimizer::lr)
        .def_readwrite("beta1", &AdamOptimizer::beta1)
        .def_readwrite("beta2", &AdamOptimizer::beta2)
        .def_readwrite("eps", &AdamOptimizer::eps)
        .def_readwrite("weight_decay", &AdamOptimizer::weight_decay)
        .def_readwrite("clip_max_norm", &AdamOptimizer::clip_max_norm)
        .def_readwrite("t", &AdamOptimizer::t);

    m.def("create_optimizer", &create_optimizer, "Create Adam optimizer",
          py::arg("grad"),
          py::arg("lr") = 0.001f,
          py::arg("beta1") = 0.9f,
          py::arg("beta2") = 0.999f,
          py::arg("epsilon") = 1e-8f,
          py::arg("weight_decay") = 0.0f,
          py::arg("clip_max_norm") = 0.0f);

    m.def("destroy_optimizer", &destroy_optimizer, "Free Adam optimizer memory");

    // Adam
    m.def("step_adam_naive", &step_adam_naive, "Naive Adam optimizer step",
          py::arg("optimizer"),
          py::arg("param"),
          py::arg("grad"));

#if defined(__AVX512F__)
    m.def("step_adam_avx512", &step_adam_avx512, "AVX512 optimized Adam optimizer step",
          py::arg("optimizer"),
          py::arg("param"),
          py::arg("grad"));
#endif

    m.def("step_adam", &step_adam, "The most optimized Adam optimizer step available.",
          py::arg("optimizer"),
          py::arg("param"),
          py::arg("grad"));

    // AdamW
    m.def("step_adamw_naive", &step_adamw_naive, "Naive Adam optimizer step",
        py::arg("optimizer"),
        py::arg("param"),
        py::arg("grad"));

#if defined(__AVX512F__)
    m.def("step_adamw_avx512", &step_adamw_avx512, "AVX512 optimized AdamW optimizer step",
        py::arg("optimizer"),
        py::arg("param"),
        py::arg("grad"));
#endif

    m.def("step_adamw", &step_adamw, "The most optimized AdamW optimizer step available.",
          py::arg("optimizer"),
          py::arg("param"),
          py::arg("grad"));

    // Other stuff

    m.def("vector_width", &vector_width, "Get simd vector width (1=Scalar, 512=AVX512)");

    m.def("serialize", [](AdamOptimizer* optimizer) {
        char* buffer = adam_serialize(optimizer);
        size_t size = SER_SIZE + (optimizer->param_count * sizeof(float));
        py::bytes result(buffer, size);
        free(buffer);
        return result;
    }, "Serialize optimizer to bytes");

    m.def("deserialize", [](py::bytes data) {
        char* buffer = PyBytes_AS_STRING(data.ptr());
        AdamOptimizer* opt = adam_deserialize(buffer);
        return opt;
    }, "Deserialize optimizer from bytes");

}
