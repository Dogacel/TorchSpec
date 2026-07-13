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

"""FlowSpec trainer — DFlashTrainer with the flow-matching model/wrapper.

Reuses DFlash's init (FSDP, frozen embedding + external LM head, optimizer,
checkpointing), forward/backward, and metric aggregation. Only the build hooks
and the extra logged loss components differ. Shares the dflash_* args for
block size / anchors / decay so run configs stay uniform.
"""

from argparse import Namespace

import torch.distributed as dist

from torchspec.models.draft.flowspec import FlowSpecConfig, FlowSpecDraftModel
from torchspec.models.flowspec import FlowSpecModel
from torchspec.training.dflash_trainer import DFlashTrainer
from torchspec.utils.logging import logger


class FlowSpecTrainer(DFlashTrainer):
    _draft_config_class = FlowSpecConfig
    # block_acc_len is only emitted on eval forwards (the sampler is too
    # expensive to run every training step); _reduce_loss_components skips
    # keys absent from a step's metrics.
    _extra_loss_component_keys = ["fm_loss", "ce_loss", "l1_loss", "block_acc_len", "denoise_acc"]

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.ce_loss_alpha = getattr(args, "flowspec_ce_alpha", 0.2)

    def _build_draft_model(self, config):
        return FlowSpecDraftModel(config)

    def _build_training_wrapper(self, draft_model):
        # Warm-start + freeze the context fusion from a trained DFlash checkpoint
        # (stationary, pre-organized conditioning basis — the frozen-text-encoder
        # pattern from image models). Rank 0 loads real weights; FSDP's full-state
        # broadcast propagates them; the freeze applies on every rank.
        fc_path = getattr(self.args, "flowspec_fc_init_path", None)
        if fc_path:
            if dist.get_rank() == 0:
                draft_model.load_context_proj(fc_path)
                logger.info(f"[Rank 0] FlowSpec fc (context_proj/norm) loaded from {fc_path}")
            draft_model.freeze_context_proj()

        return FlowSpecModel(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            loss_decay_gamma=self.loss_decay_gamma,
            ce_loss_alpha=self.ce_loss_alpha,
            chunk_ae_path=getattr(self.args, "flowspec_chunk_ae_path", None),
            l1_loss_alpha=getattr(self.args, "flowspec_l1_alpha", 0.0),
        )
