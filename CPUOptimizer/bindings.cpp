#include <torch/extension.h>
#include "cpu_optimizer.hpp"

// Must install ninja to build this extension
// Also install torch and numpy

//////////////////////
// Create / Destroy //
//////////////////////

#define STEP_CHECKS() \
    TORCH_CHECK(optimizer != NULL, "optimizer must not be null"); \
    TORCH_CHECK(param.is_contiguous(), "param must be contiguous"); \
    TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous"); \
    TORCH_CHECK(param.dtype() == torch::kFloat32, "param must be float32"); \
    TORCH_CHECK(grad.dtype() == torch::kFloat32, "grad must be float32"); \
    TORCH_CHECK(param.sizes() == grad.sizes(), "param and grad must have same shape"); \
    TORCH_CHECK((uint64_t)param.numel() == optimizer->param_count, "parameter count mismatch");


//////////
// Adam //
//////////

template<StepKind stepkind>
torch::Tensor step_binding(
    CPUOptimizer* optimizer,
    torch::Tensor& param,
    torch::Tensor& grad
) {
    STEP_CHECKS()
    float* grad_ptr = grad.data_ptr<float>();
    float* param_ptr = param.data_ptr<float>();
    float grad_l2_norm = l2_norm(grad_ptr, 0, optimizer->param_count);
    adam_step<stepkind>(optimizer, param_ptr, grad_ptr, 0, optimizer->param_count, grad_l2_norm);
    return param;
}

//////////////
// Bindings //
//////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    // Create/destroy
    py::class_<CPUOptimizer>(m, "OptimizerBinding", py::module_local())
        .def_readonly("grad", &CPUOptimizer::param_count)
        .def_readwrite("lr", &CPUOptimizer::lr)
        .def_readwrite("beta1", &CPUOptimizer::beta1)
        .def_readwrite("beta2", &CPUOptimizer::beta2)
        .def_readwrite("eps", &CPUOptimizer::eps)
        .def_readwrite("weight_decay", &CPUOptimizer::weight_decay)
        .def_readwrite("clip_max_norm", &CPUOptimizer::clip_max_norm)
        .def_readwrite("t", &CPUOptimizer::t);

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

    // Steps

    m.def("step_adam", &step_binding<StepKind::ADAM_STEP>, "",
        py::arg("optimizer"),
        py::arg("param"),
        py::arg("grad"));

    m.def("step_adamw", &step_binding<StepKind::ADAMW_STEP>, "",
        py::arg("optimizer"),
        py::arg("param"),
        py::arg("grad"));

    m.def("step_adamw_torch", &step_binding<StepKind::ADAMW_TORCH_STEP>, "",
        py::arg("optimizer"),
        py::arg("param"),
        py::arg("grad"));

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
