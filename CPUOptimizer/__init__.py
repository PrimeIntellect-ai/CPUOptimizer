from typing import Callable
from enum import IntEnum

import torch
from torch.optim.optimizer import ParamsT, StateDict
from torch.distributed.tensor import DTensor

from . import bindings


class StepKind(IntEnum):
    ADAM = 0
    ADAMW = 1
    TORCH_ADAMW = 2

kind_name_map = {
    "adam": StepKind.ADAM,
    "adamw": StepKind.ADAMW,
    "torch_adamw": StepKind.TORCH_ADAMW,
}

_step_binding = {
    StepKind.ADAM: bindings.step_adam,
    StepKind.ADAMW: bindings.step_adamw,
    StepKind.TORCH_ADAMW: bindings.step_adamw_torch,
}


class CPUOptimizer(torch.optim.Optimizer):
    """
    A CPU-optimized implementation of the Adam and AdamW optimizers.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_max_norm: float = 0.0,
        pipeline_hook: Callable | None = None,
        step_kind: StepKind = StepKind.TORCH_ADAMW,
    ):
        super().__init__(
            params,
            defaults=dict(
                lr=lr,
                beta1=betas[0],
                beta2=betas[1],
                eps=eps,
                weight_decay=weight_decay,
                clip_max_norm=clip_max_norm,
                pipeline_hook=pipeline_hook,
                step_kind=step_kind,
            )
        )

        # For each param, create an optimizer object for that param.
        for group in self.param_groups:
            for param in group["params"]:
                _param = param._local_tensor if isinstance(param, DTensor) else param
                self.state[param] = bindings.create_optimizer(
                    _param, lr, betas[0], betas[1], eps, weight_decay, clip_max_norm,
                )
                if pipeline_hook is not None:
                    param.register_post_accumulate_grad_hook(pipeline_hook)

    def begin_step(self) -> None:
        """
        Create a step context for use with the pipeline hook. If you define a pipeline hook, call this before backward().

        This creates a backing threadpool to update each parameter as it becomes available with `register_post_accumulate_grad_hook()`.
        The step() function then just waits for that threadpool to complete, meaning the optimizer step can be overlapped with backwards().

        See: https://pytorch.org/docs/stable/generated/torch.Tensor.register_post_accumulate_grad_hook.html
        """
        if self.defaults["pipeline_hook"] is not None:
            self.step_ctx = bindings.create_step_context()

    def step(self) -> None:
        """Perform an optimizer step on all parameters."""
        if self.defaults["pipeline_hook"] is not None:
            # If a pipeline hook was registered, the grads are already being computed by the backwards hooks.
            # We only have to wait for the backing threadpool to complete by destroying the step_ctx.
            if hasattr(self, "step_ctx"):
                del self.step_ctx
            else:
                raise RuntimeError("Pipeline hook was used, but could not find a step context. Call optimizer.begin_step() before backwards().")
            return

        for group in self.param_groups:
            for p in group["params"]:
                self.step_param(p)

    def step_param(self, param: torch.Tensor) -> None:
        """
        Perform an optimizer step on one parameter. This is done on CPU with the fastest SIMD instructions that are available.

        If you register a pipeline hook, this function is to be called in the pipeline hook.
        """

        step_ctx = getattr(self, "step_ctx", None)
        if self.defaults["pipeline_hook"] is not None and step_ctx is None:
            raise RuntimeError("Pipeline hook was used, but step context was not found. Call self.begin_step() before backwards().")

        param_opt = self.state.get(param)
        if type(param_opt) is not bindings.OptimizerBinding:
            raise ValueError(f"Parameter is not registered with this optimizer: {param}")


        # Adam and AdamW are parameterwise optimizers, so we do not have to gather dtensors.
        # Each process can launch the update on its own shard.
        local_param = param._local_tensor if isinstance(param, DTensor) else param
        local_grad = param.grad._local_tensor if isinstance(param, DTensor) else param.grad # type: ignore (grad is a dtensor also)
        step_fn = _step_binding[self.defaults["step_kind"]]
        step_fn(param_opt, local_param, local_grad, step_ctx)

    def __del__(self):
        """Free the optimizer state memory held by C++."""
        for opt in self.state.values():
            if isinstance(opt, bindings.OptimizerBinding):
                bindings.destroy_optimizer(opt)

    def load_state_dict(self, state_dict: StateDict) -> None:
        super().load_state_dict(state_dict)

        # Restore optimizer state bindings
        for param, _bytes in self.state.items():
            opt = bindings.deserialize(_bytes)
            self.state[param] = opt
            [setattr(opt, k, v) for k, v in self.defaults.items() if k in opt.__dir__()]

    def state_dict(self) -> StateDict:
        if hasattr(self, "step_ctx"):
            raise RuntimeError("Cannot save optimizer state while a step is in progress.")

        state = super().state_dict()

        # Convert optimizer state bindings to bytes objects
        for param, opt in state["state"].items():
            state["state"][param] = bindings.serialize(opt)

        return state

    @classmethod
    def vector_width(cls) -> int:
        """Returns 1 if using the naive scalar implementation, 512 for avx512."""
        return bindings.vector_width()
