# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

from torchspec.training.lr_scheduler import LRSchedulerWithWarmup
from torchspec.utils.logging import print_on_rank0


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        decay_style="cosine",
        min_lr=0.0,
        wsd_decay_steps=None,
        wsd_decay_style=None,
        flatten_buffers=False,
    ):
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self.flatten_buffers = flatten_buffers
        if flatten_buffers:
            if not self.model_params:
                raise ValueError("Cannot flatten optimizer buffers for a model with no parameters")
            devices = {p.device for p in self.model_params}
            if len(devices) != 1:
                raise ValueError(
                    "flatten_optimizer_buffers requires all trainable parameters on one device"
                )
            total_numel = sum(p.numel() for p in self.model_params)
            self._flat_fp32_param = torch.nn.Parameter(
                torch.empty(total_numel, device=self.model_params[0].device, dtype=torch.float32)
            )
            self._flat_fp32_grad = torch.empty_like(self._flat_fp32_param)
            self.fp32_params = []
            self.fp32_grads = []
            offset = 0
            for param in self.model_params:
                next_offset = offset + param.numel()
                self.fp32_params.append(self._flat_fp32_param[offset:next_offset].view(param.shape))
                self.fp32_grads.append(self._flat_fp32_grad[offset:next_offset].view(param.shape))
                offset = next_offset
            with torch.no_grad():
                torch._foreach_copy_(self.fp32_params, self.model_params)
            optimizer_params = [self._flat_fp32_param]
        else:
            self.fp32_params = [p.detach().clone().to(torch.float32) for p in self.model_params]
            self.fp32_grads = [torch.zeros_like(mp) for mp in self.fp32_params]
            for mp in self.fp32_params:
                mp.requires_grad = True
            optimizer_params = self.fp32_params
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=lr,
            weight_decay=weight_decay,
            fused=True,
        )
        self.scheduler = LRSchedulerWithWarmup(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
            decay_style=decay_style,
            min_lr=min_lr,
            wsd_decay_steps=wsd_decay_steps,
            wsd_decay_style=wsd_decay_style,
        )

    def step(self, closure=None):
        """Perform optimizer step with gradient clipping.

        Args:
            closure: Ignored, for compatibility with PyTorch optimizer interface.

        Returns:
            grad_norm: The gradient norm before clipping (for logging).
        """
        with torch.no_grad():
            grad_destinations = []
            grad_sources = []
            for p, mp, g in zip(self.model_params, self.fp32_params, self.fp32_grads):
                if p.grad is not None:
                    grad_destinations.append(g)
                    grad_sources.append(p.grad)
                    if not self.flatten_buffers:
                        mp.grad = g
                else:
                    if self.flatten_buffers:
                        raise RuntimeError(
                            "flatten_optimizer_buffers requires every trainable parameter "
                            "to receive a gradient"
                        )
                    else:
                        mp.grad = None
            if grad_destinations:
                torch._foreach_copy_(grad_destinations, grad_sources)

        if self.flatten_buffers:
            self._flat_fp32_param.grad = self._flat_fp32_grad
            params_to_clip = [self._flat_fp32_param]
        else:
            params_to_clip = self.fp32_params

        grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, self.max_grad_norm)
        self.optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            torch._foreach_copy_(self.model_params, self.fp32_params)
            for p in self.model_params:
                p.grad = None

        return grad_norm

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for p in self.model_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        print_on_rank0("Successfully loaded optimizer state_dict.")

    def sync_fp32_params_from_model(self):
        """Reinitialize fp32_params from model params. Call after loading model checkpoint."""
        with torch.no_grad():
            torch._foreach_copy_(self.fp32_params, self.model_params)

    def state_dict(self):
        return self.optimizer.state_dict()

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    @property
    def state(self):
        return self.optimizer.state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def lr_scheduler(self):
        return self.scheduler
