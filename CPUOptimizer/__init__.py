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

        for group in self.param_groups:
            for param in group["params"]:
                _param = param._local_tensor if isinstance(param, DTensor) else param
                self.state[param] = bindings.create_optimizer(
                    _param, lr, betas[0], betas[1], eps, weight_decay, clip_max_norm,
                )
                if pipeline_hook is not None:
                    param.register_post_accumulate_grad_hook(pipeline_hook)

    def step(self) -> None:
        """Perform an optimizer step on all parameters."""

        if self.defaults["pipeline_hook"] is not None:
            return

        step_ctx = self.begin_step()

        for group in self.param_groups:
            for p in group["params"]:
                self.step_param(p, step_ctx=step_ctx)

        del step_ctx # Gets GC'd, calling the destructor and joining the backing threadpool.

    def begin_step(self) -> bindings.StepContext | None:
        """Create a step context for use with the pipeline hook. Delete the context to join the threadpool and complete the step."""
        return bindings.create_step_context() if self.defaults["pipeline_hook"] is not None else None

    def step_param(self, param: torch.Tensor, step_ctx: bindings.StepContext | None = None) -> None:
        """Perform an optimizer step on one parameter. This is done with whatever SIMD is available."""

        param_opt = self.state.get(param)
        if type(param_opt) is not bindings.OptimizerBinding:
            raise ValueError(f"Parameter is not registered with this optimizer: {param}")

        local_param = param._local_tensor if isinstance(param, DTensor) else param
        local_grad = param.grad._local_tensor if isinstance(param, DTensor) else param.grad # type: ignore (grad is a dtensor also)
        step_fn = _step_binding[self.defaults["step_kind"]]
        step_fn(param_opt, local_param, local_grad, step_ctx)

    def __del__(self):
        """Free the memory held by C++. Otherwise we risk leaking unholy amounts of memory."""
        for opt in self.state.values():
            if isinstance(opt, bindings.OptimizerBinding):
                bindings.destroy_optimizer(opt)

    def load_state_dict(self, state_dict: StateDict) -> None:
        """Deserialize with torch.load()."""
        super().load_state_dict(state_dict)

        # Restore optimizer state bindings
        for param, _bytes in self.state.items():
            opt = bindings.deserialize(_bytes)
            self.state[param] = opt
            [setattr(opt, k, v) for k, v in self.defaults.items() if k in opt.__dir__()]

    def state_dict(self) -> StateDict:
        """Serialize with torch.save()."""
        state = super().state_dict()

        # Convert optimizer state bindings to bytes objects
        for param, opt in state["state"].items():
            state["state"][param] = bindings.serialize(opt)

        return state

    @classmethod
    def vector_width(cls) -> int:
        """Returns 1 if using the naive scalar implementation, 256 for avx2, 512 for avx512."""
        return bindings.vector_width()
