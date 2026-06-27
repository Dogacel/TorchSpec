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

"""DSpark trainer — DFlash trainer + Markov/confidence heads and L1 distillation.

Reuses the entire DFlash training pipeline (FSDP init, optimizer, checkpoint,
metric aggregation, hidden-state capture/transfer) via subclass hooks, and
additionally feeds the target ``last_hidden_states`` into the forward so the
L1 distribution-distillation and confidence-head losses can be computed.
"""

from argparse import Namespace
from typing import Tuple

import torch

from torchspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from torchspec.models.dspark import DSparkModel
from torchspec.training.dflash_trainer import DFlashTrainer


class DSparkTrainer(DFlashTrainer):
    """DSpark-specific trainer (DFlash backbone + EAGLE-style heads)."""

    _draft_config_class = DSparkConfig

    def __init__(self, args: Namespace):
        super().__init__(args)
        # DSpark uses its own knobs; override the dflash_* defaults read by the
        # parent so the shared init_model / wrapper builders pick them up.
        self.block_size = getattr(args, "dflash_block_size", 7)
        self.num_anchors = getattr(args, "dspark_num_anchors", 512)
        self.num_target_layers = getattr(args, "dspark_num_target_layers", 5)
        self.loss_decay_gamma = getattr(args, "dspark_loss_decay_gamma", 4.0)
        self.ce_loss_alpha = getattr(args, "dspark_ce_loss_alpha", 0.1)
        self.l1_loss_alpha = getattr(args, "dspark_l1_loss_alpha", 0.9)
        self.confidence_head_alpha = getattr(args, "dspark_confidence_head_alpha", 1.0)
        self._anchor_slot_offset = 0

    # ------------------------------------------------------------------
    # Build hooks (override DFlashTrainer's defaults)
    # ------------------------------------------------------------------

    def _build_draft_model(self, config):
        return DSparkDraftModel(config)

    def _build_training_wrapper(self, draft_model):
        return DSparkModel(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            loss_decay_gamma=self.loss_decay_gamma,
            ce_loss_alpha=self.ce_loss_alpha,
            l1_loss_alpha=self.l1_loss_alpha,
            confidence_head_alpha=self.confidence_head_alpha,
        )

    # ------------------------------------------------------------------
    # Forward — adds target last_hidden_states for L1 / confidence losses
    # ------------------------------------------------------------------

    def _forward(
        self, batch: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = torch.device("cuda")
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        hidden_states = batch["hidden_states"].to(device, non_blocking=True)

        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        loss_mask = loss_mask.to(device, non_blocking=True)

        last_hidden_states = batch.get("last_hidden_states", None)
        if last_hidden_states is not None:
            last_hidden_states = last_hidden_states.to(device, non_blocking=True)

        hidden_states_list = self._split_hidden_states(hidden_states)
        del hidden_states

        return self.model(
            input_ids=input_ids,
            hidden_states_list=hidden_states_list,
            loss_mask=loss_mask,
            lm_head_weight=self.target_lm_head_weight,
            last_hidden_states=last_hidden_states,
        )
