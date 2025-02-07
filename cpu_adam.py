from typing import Any, Callable, Iterable, Optional

import torch
import torch.optim.optimizer
from torch.optim.optimizer import ParamsT, StateDict

import cpu_optimizer_bindings


class CPUAdam(torch.optim.Optimizer):
    """
    A CPU-optimized implementation of the Adam optimizer.

    When 
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps=1e-8,
        pipeline_hook: Callable | None = None,
    ):
        super().__init__(
            params, defaults=dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, pipeline_hook=pipeline_hook)
        )

        for group in self.param_groups:
            for param in group["params"]:
                self.state[param] = cpu_optimizer_bindings.create_optimizer(
                    param, lr, betas[0], betas[1], eps
                )
                if pipeline_hook:
                    param.register_post_accumulate_grad_hook(pipeline_hook)

    def step(self) -> None:
        """Perform an optimizer step on all parameters."""
        if self.defaults["pipeline_hook"] is not None:
            return

        for group in self.param_groups:
            for p in group["params"]:
                self.step_param(p)

    def step_param(self, param: torch.Tensor) -> None:
        """Perform an optimizer step on one parameter. This is done with whatever SIMD is available."""
        param_opt = self.state.get(param)
        if type(param_opt) is not cpu_optimizer_bindings.AdamOptimizer:
            raise ValueError(
                f"Parameter is not registered with this optimizer: {param}"
            )
        cpu_optimizer_bindings.step(param_opt, param.data, param.grad)

    def __del__(self):
        """Free the memory held by C++. Otherwise we risk leaking unholy amounts of memory."""
        for opt in self.state.values():
            if isinstance(opt, cpu_optimizer_bindings.AdamOptimizer):
                cpu_optimizer_bindings.destroy_optimizer(opt)

    def load_state_dict(self, state_dict: StateDict) -> None:
        """Deserialize with torch.load()."""
        super().load_state_dict(state_dict)

        # Restore optimizer state bindings
        for param, _bytes in self.state.items():
            opt = cpu_optimizer_bindings.deserialize(_bytes)
            self.state[param] = opt
            [setattr(opt, k, v) for k, v in self.defaults.items() if k in opt.__dir__()]

    def state_dict(self) -> StateDict:
        """Serialize with torch.save()."""
        state = super().state_dict()

        # Convert optimizer state bindings to bytes objects
        for param, opt in state["state"].items():
            state["state"][param] = cpu_optimizer_bindings.serialize(opt)

        return state

    @classmethod
    def vector_width(cls) -> int:
        """Returns 1 if using the naive scalar implementation, 256 for avx2, 512 for avx512."""
        return cpu_optimizer_bindings.vector_width()
