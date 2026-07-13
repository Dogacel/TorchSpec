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

"""DFlash training model: wraps the DFlash draft model with training-specific logic.

Handles anchor sampling, block-causal mask generation, noise input construction,
and cross-entropy loss with exponential decay weighting.

Matches SpecForge's OnlineDFlashModel (specforge/core/dflash.py).
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchspec.models.ops.flex_attention import compile_friendly_create_block_mask
from torchspec.utils.logging import logger

_VALID_DFLASH_LOSS_OBJECTIVES = {"decay", "dpace"}


def _dpace_position_weights(confidences: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute detached D-PACE weights from per-position draft confidences."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {alpha}")

    with torch.no_grad():
        smoothed = (1.0 - alpha) * confidences.float() + alpha
        prefix_products = torch.cumprod(smoothed, dim=-1)
        weights = torch.flip(
            torch.cumsum(torch.flip(prefix_products, dims=[-1]), dim=-1),
            dims=[-1],
        )
        return weights.to(dtype=confidences.dtype)


def _create_dflash_mask_mod(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
):
    """Create a mask_mod function for DFlash block-causal attention.

    KV: [Context (ctx_len tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos)
      2. Intra-block attention is bidirectional (per SpecForge PR #427)
      3. Different blocks are invisible to each other
      4. Invalid blocks (block_keep_mask=False) see nothing
    """
    num_anchors = anchor_positions.shape[1]

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]

        is_context = kv_idx < ctx_len
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= ctx_len
        kv_block_id = (kv_idx - ctx_len) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    dflash_mask_mod.__name__ = f"dflash_mask_A{num_anchors}_B{block_size}_C{ctx_len}"
    return dflash_mask_mod


class DFlashModel(nn.Module):
    """DFlash training wrapper.

    Wraps the DFlash draft model with training-specific logic:
      - Random anchor sampling with block_keep_mask
      - Block-causal attention mask via FlexAttention
      - Noise input construction (anchor + MASK)
      - Cross-entropy loss with configurable position weighting
      - Per-position loss_mask application
    """

    def __init__(
        self,
        draft_model,
        block_size: int = 16,
        num_anchors: int = 512,
        loss_objective: str = "decay",
        dpace_alpha: float = 0.5,
        loss_decay_gamma: float = 7.0,
        ce_loss_alpha: float = 1.0,
        l1_loss_alpha: float = 0.0,
        self_cond_frac: float = 0.0,
        self_cond_keep_max: float = 0.75,
        remask_eval_keep: float = 0.0,
        self_cond_rounds: int = 1,
        self_cond_round2_keep_min: float = 0.4,
        self_cond_random_frac: float = 0.0,
    ):
        super().__init__()
        loss_objective = loss_objective.lower()
        if loss_objective not in _VALID_DFLASH_LOSS_OBJECTIVES:
            valid = ", ".join(sorted(_VALID_DFLASH_LOSS_OBJECTIVES))
            raise ValueError(
                f"Unknown DFlash loss objective {loss_objective!r}; expected one of {valid}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {dpace_alpha}")
        if not 0.0 <= self_cond_frac <= 1.0:
            raise ValueError(f"dflash_self_cond_frac must be in [0, 1], got {self_cond_frac}")
        if not 0.0 <= self_cond_keep_max < 1.0:
            raise ValueError(
                f"dflash_self_cond_keep_max must be in [0, 1), got {self_cond_keep_max}"
            )
        if not 0.0 <= remask_eval_keep < 1.0:
            raise ValueError(f"dflash_remask_eval_keep must be in [0, 1), got {remask_eval_keep}")

        self.draft_model = draft_model
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.loss_objective = loss_objective
        self.dpace_alpha = dpace_alpha
        self.loss_decay_gamma = loss_decay_gamma
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        self.self_cond_frac = float(self_cond_frac)
        self.self_cond_keep_max = float(self_cond_keep_max)
        self.remask_eval_keep = float(remask_eval_keep)
        if self_cond_rounds not in (1, 2):
            raise ValueError(f"dflash_self_cond_rounds must be 1 or 2, got {self_cond_rounds}")
        if not 0.0 <= self_cond_random_frac <= 1.0:
            raise ValueError(
                f"dflash_self_cond_random_frac must be in [0, 1], got {self_cond_random_frac}"
            )
        self.self_cond_rounds = int(self_cond_rounds)
        self.self_cond_round2_keep_min = float(self_cond_round2_keep_min)
        self.self_cond_random_frac = float(self_cond_random_frac)

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample anchor positions per sample; returns (anchors, keep_mask).

        Always returns exactly ``self.num_anchors`` anchor slots so that
        ``Q_LEN = num_anchors * block_size`` is constant across steps,
        preventing FlexAttention recompilation from shape changes.  Samples
        with fewer valid positions use ``block_keep_mask=False`` for the
        excess slots (those blocks are skipped by the block-sparse kernel).

        Args:
            seq_len: sequence length
            loss_mask: [B, seq_len] — 1 for valid positions, 0 for padding
            device: torch device

        Returns:
            anchors: [B, num_anchors] — sampled anchor positions (sorted)
            keep_mask: [B, num_anchors] — True for valid sampled anchors
        """
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)
        max_n = self.num_anchors

        if max_anchor == 0:
            logger.warning(
                f"Sequence too short for anchor sampling (seq_len={seq_len}, "
                f"block_size={bs}). Returning dummy anchors so loss is zero."
            )
            anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
            keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
            return anchors, keep_mask

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, seq_len + 1)

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, 2.0)

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)

        # Take up to num_anchors slots; pad with zeros if fewer valid positions
        take_n = min(max_n, gathered.shape[1])
        selected = gathered[:, :take_n].sort(dim=1).values
        if take_n < max_n:
            pad = torch.zeros(bsz, max_n - take_n, dtype=torch.long, device=device)
            selected = torch.cat([selected, pad], dim=1)
        anchors = selected

        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(
            1
        ).clamp(max=max_n)
        anchors = torch.where(keep_mask, anchors, 0)

        return anchors, keep_mask

    def _create_position_ids(
        self, anchor_positions: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create position IDs for context and draft tokens."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device

        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        draft_position_ids = anchor_positions.unsqueeze(-1) + offsets
        draft_position_ids = draft_position_ids.view(bsz, -1)

        return context_position_ids, draft_position_ids

    def _create_noise_ids(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create noise token ids: anchor token at block starts, MASK elsewhere."""
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.draft_model.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.draft_model.mask_token_id, dtype=torch.long, device=device),
        )

        return noise_ids

    def _create_noise_embed(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create noise embeddings: anchor token at block starts, MASK elsewhere.

        Matches SpecForge's OnlineDFlashModel._create_noise_embed().
        """
        noise_ids = self._create_noise_ids(input_ids, anchor_positions, block_keep_mask)
        return self.draft_model.embed_tokens(noise_ids)

    def _prepare_draft_inputs(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
    ) -> dict:
        """Build everything a draft forward needs except the block content.

        Steps 1-5 of the backbone (context features → anchor sampling → noise
        ids/embedding → position ids → block-causal mask). The attention mask
        depends only on anchors/validity/positions — never on block content —
        so one prepared bundle can serve several forward passes with different
        block fillings (all-MASK, self-conditioned, remasked).
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Extract context features from target hidden states
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)

        # 2. Sample anchor positions with validity mask
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.shape[1]

        # 3. Create noise token ids (anchor token + MASK tokens)
        noise_ids = self._create_noise_ids(input_ids, anchor_positions, block_keep_mask)

        # 4. Create position IDs
        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len
        )

        # 5. Create block-causal attention mask
        draft_len = n_blocks * self.block_size
        kv_len = seq_len + draft_len

        block_mask = None
        if device.type == "cuda":
            mask_mod = _create_dflash_mask_mod(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                ctx_len=seq_len,
                block_size=self.block_size,
            )
            block_mask = compile_friendly_create_block_mask(
                mask_mod=mask_mod,
                B=bsz,
                H=None,
                Q_LEN=draft_len,
                KV_LEN=kv_len,
                device=device,
            )

        return {
            "context_feature": context_feature,
            "anchor_positions": anchor_positions,
            "block_keep_mask": block_keep_mask,
            "noise_ids": noise_ids,
            "context_position_ids": context_position_ids,
            "draft_position_ids": draft_position_ids,
            "block_mask": block_mask,
            "n_blocks": n_blocks,
        }

    def _run_draft(self, prep: dict, noise_ids: torch.Tensor) -> torch.Tensor:
        """Step 6 of the backbone: one draft forward over prepared inputs."""
        noise_embedding = self.draft_model.embed_tokens(noise_ids)
        return self.draft_model(
            draft_input_ids=None,
            context_feature=prep["context_feature"],
            draft_position_ids=prep["draft_position_ids"],
            context_position_ids=prep["context_position_ids"],
            block_mask=prep["block_mask"],
            noise_embedding=noise_embedding,
        )

    def _draft_backbone(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Shared DFlash backbone (context features → anchor sampling → noise
        embedding → position ids → block-causal mask → draft model forward).

        Both ``DFlashModel.forward`` and ``DSparkModel.forward`` build the draft
        hidden states this exact way; only the label/loss tail differs. Keeping
        the attention/mask/anchor wiring here gives it a single source of truth.

        Returns:
            draft_hidden: [B, n_blocks*block_size, D] pre-loss draft hidden states
            anchor_positions: [B, n_blocks] sampled anchor positions
            block_keep_mask: [B, n_blocks] bool validity of each anchor slot
            n_blocks: number of anchor slots (== num_anchors)
        """
        prep = self._prepare_draft_inputs(input_ids, hidden_states_list, loss_mask)
        draft_hidden = self._run_draft(prep, prep["noise_ids"])
        return (
            draft_hidden,
            prep["anchor_positions"],
            prep["block_keep_mask"],
            prep["n_blocks"],
        )

    @torch.no_grad()
    def _predict_tokens(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        chunk_size: int = 2048,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunked argmax + log-confidence over the vocab.

        Returns (pred_ids [B, L], conf [B, L]) where conf is log P(argmax)
        (max logit minus logsumexp — monotone in the softmax probability
        without materializing a full-vocab softmax).
        """
        preds, confs = [], []
        for chunk in draft_hidden.split(chunk_size, dim=1):
            if hasattr(self.draft_model, "lm_head"):
                logits = self.draft_model.lm_head(chunk).float()
            else:
                logits = F.linear(chunk, lm_head_weight).float()
            mx, argmax = logits.max(dim=-1)
            confs.append(mx - torch.logsumexp(logits, dim=-1))
            preds.append(argmax)
        return torch.cat(preds, dim=1), torch.cat(confs, dim=1)

    @torch.no_grad()
    def _confidence_from_logits(self, logits: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
        """Log-confidence of the argmax from already-computed logits [B, L, V]."""
        confs = []
        for chunk in logits.split(chunk_size, dim=1):
            chunk = chunk.float()
            confs.append(chunk.max(dim=-1).values - torch.logsumexp(chunk, dim=-1))
        return torch.cat(confs, dim=1)

    @torch.no_grad()
    def _reveal_from_confidence(
        self,
        conf: torch.Tensor,
        block_keep_mask: torch.Tensor,
        reveal_block_mask: torch.Tensor,
        keep_frac: torch.Tensor,
    ) -> torch.Tensor:
        """Choose which slots to reveal: the top-``keep_frac`` most confident
        non-anchor slots of each selected block.

        Args:
            conf: [B, n_blocks*block_size] per-slot log-confidence
            block_keep_mask: [B, n_blocks] anchor-slot validity
            reveal_block_mask: [B, n_blocks] blocks eligible for revealing
            keep_frac: [B, n_blocks] fraction of the block_size-1 predictable
                slots to reveal, in [0, 1)

        Returns:
            revealed: [B, n_blocks, block_size] bool; slot 0 (anchor) never revealed
        """
        bsz, n = block_keep_mask.shape
        bs = self.block_size
        device = conf.device

        conf3 = conf.view(bsz, n, bs).clone()
        conf3[..., 0] = float("-inf")  # the anchor slot already holds a real token

        n_keep = (keep_frac.clamp(0.0, 1.0) * (bs - 1)).floor().long()  # [B, n]

        order = conf3.argsort(dim=-1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(-1, order, torch.arange(bs, device=device).view(1, 1, bs).expand_as(order))
        revealed = ranks < n_keep.unsqueeze(-1)
        revealed &= (reveal_block_mask & block_keep_mask).unsqueeze(-1)
        revealed[..., 0] = False
        return revealed

    @staticmethod
    def _logit_schedule(x: float, alpha: float) -> float:
        """Normalized logistic S-curve through (0,0) and (1,1).

        alpha=0 degenerates to linear; larger alpha concentrates commits in
        the middle of the schedule (slow start, fast middle, slow end) —
        the masked-diffusion analogue of the flow timestep shift.
        """
        if alpha == 0.0:
            return x
        g0 = 1.0 / (1.0 + math.exp(0.5 * alpha))
        g1 = 1.0 / (1.0 + math.exp(-0.5 * alpha))
        gx = 1.0 / (1.0 + math.exp(-alpha * (x - 0.5)))
        return (gx - g0) / (g1 - g0)

    @torch.no_grad()
    def sample_scheduled(
        self,
        prep: dict,
        lm_head_weight: torch.Tensor,
        num_passes: int,
        schedule_alpha: float = 4.0,
    ) -> torch.Tensor:
        """Multi-pass masked-diffusion sampling over prepared draft inputs.

        Logit-spaced cumulative commit schedule: entering pass p+1, the top
        ``_logit_schedule(p/num_passes)`` fraction of non-anchor slots (ranked
        by the model's current confidence) carry their current predicted
        tokens; the rest return to MASK. Every pass re-ranks ALL slots, so a
        committed slot may be revised or remasked until the final pass — the
        behaviour self-conditioned training licenses. Pass 1 always starts
        from the all-MASK block; ``num_passes=1`` is the standard DFlash
        one-jump. The final output is the last pass's argmax at every slot.

        Returns:
            final_preds: [B, n_blocks*block_size] predicted token ids
        """
        if num_passes < 1:
            raise ValueError(f"num_passes must be >= 1, got {num_passes}")
        bsz = prep["block_keep_mask"].shape[0]
        n_blocks = prep["n_blocks"]
        keep_mask = prep["block_keep_mask"]
        base_ids = prep["noise_ids"]

        ids = base_ids
        preds = None
        for p in range(1, num_passes + 1):
            hidden = self._run_draft(prep, ids)
            preds, conf = self._predict_tokens(hidden, lm_head_weight)
            if p == num_passes:
                break
            frac = self._logit_schedule(p / num_passes, schedule_alpha)
            keep_frac = torch.full((bsz, n_blocks), frac, device=base_ids.device)
            committed = self._reveal_from_confidence(conf, keep_mask, keep_mask, keep_frac)
            ids = torch.where(committed.view(bsz, -1), preds, base_ids)
        return preds

    @staticmethod
    def _consecutive_len(correct3: torch.Tensor, valid3: torch.Tensor) -> torch.Tensor:
        """Mean per-block consecutive-correct length starting at slot 1.

        correct3/valid3: [B, n_blocks, block_size]; blocks without a valid
        slot-1 label are excluded from the denominator.
        """
        if correct3.shape[-1] > 1:
            has_p1 = valid3[..., 1]
        else:
            has_p1 = valid3[..., 0]
        n_valid_blocks = has_p1.float().sum().clamp(min=1.0)
        run = (correct3 & valid3)[..., 1:].int().cumprod(dim=-1)
        return (run.sum(dim=-1).float() * has_p1.float()).sum() / n_valid_blocks

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Full DFlash training forward pass.

        Returns:
            loss: scalar training loss (objective-weighted)
            accuracy: scalar accuracy (binary mask, no decay)
            loss_per_position: [block_size] mean loss at each within-block position
                (index 0 is the anchor slot and always 0; indices 1..B-1 are the
                predicted tokens at 1..B-1 steps past the anchor)
            acc_per_position: [block_size] mean accuracy at each within-block position
            count_per_position: [block_size] valid label count at each within-block
                position before loss decay is applied
            loss_components: dict of extra per-component loss scalars for logging
                (empty for the base DFlash objective; populated by subclasses).
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 1-5. Prepare shared draft inputs. The attention mask is content-
        # independent (anchors/validity/positions only), so the same prepared
        # bundle legally serves every pass below.
        prep = self._prepare_draft_inputs(input_ids, hidden_states_list, loss_mask)
        anchor_positions = prep["anchor_positions"]
        block_keep_mask = prep["block_keep_mask"]
        n_blocks = prep["n_blocks"]
        noise_ids = prep["noise_ids"]

        # Self-conditioning (training only): a no-grad dry run on the all-MASK
        # block predicts every slot; for a random subset of blocks the most
        # confident predictions are revealed as inputs (the rest stay MASK) and
        # the gradient pass learns on blocks containing the model's own —
        # possibly wrong — tokens. This is the training-side twin of remask
        # sampling: inference pass 2 sees exactly this input distribution.
        # The dry run executes unconditionally on every rank whenever the
        # feature is on, keeping the FSDP collective sequence rank-uniform.
        self_cond_active = self.training and self.self_cond_frac > 0.0 and self.block_size > 1
        self_cond_blocks = None
        if self_cond_active:
            with torch.no_grad():
                dry_hidden = self._run_draft(prep, noise_ids)
                dry_pred_ids, dry_conf = self._predict_tokens(dry_hidden, lm_head_weight)
                self_cond_blocks = (
                    torch.rand(bsz, n_blocks, device=device) < self.self_cond_frac
                ) & block_keep_mask
                keep_frac = torch.rand(bsz, n_blocks, device=device) * self.self_cond_keep_max

                if self.self_cond_rounds > 1:
                    # Second dry round: run the model once more from an
                    # intermediate (early-schedule) revealed state, so half the
                    # self-conditioned blocks train on REFINED beliefs at HIGH
                    # reveal fractions — the late-sampler-pass states a
                    # single-round dry run can never produce. Both dry passes
                    # execute unconditionally on every rank (FSDP-safe).
                    mid_frac = torch.rand(bsz, n_blocks, device=device) * 0.6
                    mid_revealed = self._reveal_from_confidence(
                        dry_conf, block_keep_mask, self_cond_blocks, mid_frac
                    )
                    mid_ids = torch.where(mid_revealed.view(bsz, -1), dry_pred_ids, noise_ids)
                    dry2_hidden = self._run_draft(prep, mid_ids)
                    dry2_pred_ids, dry2_conf = self._predict_tokens(dry2_hidden, lm_head_weight)

                    round2 = (torch.rand(bsz, n_blocks, device=device) < 0.5) & self_cond_blocks
                    round2_flat = round2.repeat_interleave(self.block_size, dim=1)
                    dry_pred_ids = torch.where(round2_flat, dry2_pred_ids, dry_pred_ids)
                    dry_conf = torch.where(round2_flat, dry2_conf, dry_conf)
                    hi = self.self_cond_round2_keep_min + torch.rand(
                        bsz, n_blocks, device=device
                    ) * (self.self_cond_keep_max - self.self_cond_round2_keep_min)
                    keep_frac = torch.where(round2, hi, keep_frac)

                if self.self_cond_random_frac > 0.0:
                    # Exploration: some blocks reveal a random subset instead of
                    # the top-confidence one, for robustness to eval schedules
                    # whose keep rule differs from pure confidence ranking.
                    rand_blocks = (
                        torch.rand(bsz, n_blocks, device=device) < self.self_cond_random_frac
                    ) & self_cond_blocks
                    rand_flat = rand_blocks.repeat_interleave(self.block_size, dim=1)
                    dry_conf = torch.where(rand_flat, torch.rand_like(dry_conf), dry_conf)

                revealed = self._reveal_from_confidence(
                    dry_conf, block_keep_mask, self_cond_blocks, keep_frac
                )
                noise_ids = torch.where(revealed.view(bsz, -1), dry_pred_ids, noise_ids)

        # 6. Draft forward on the (possibly self-conditioned) block content.
        draft_hidden = self._run_draft(prep, noise_ids)

        # 7. Compute logits via frozen LM head
        logits = (
            self.draft_model.lm_head(draft_hidden)
            if hasattr(self.draft_model, "lm_head")
            else F.linear(draft_hidden, lm_head_weight)
        )

        # 8. Compute labels and weight mask (SpecForge pattern)
        # Labels: same-position prediction (position k predicts token at anchor+k)
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, n_blocks, block_size]
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )  # [B, n_blocks, block_size]

        # Weight mask: block validity × bounds × exclude anchor (pos 0) × loss_mask
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        # Gather original loss_mask at label positions
        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        # Capture binary mask BEFORE applying objective weights. Accuracy measures
        # "did we predict correctly?" uniformly across positions, while weighting
        # only shapes gradient contribution. SpecForge uses no decay at all;
        # our objective weighting is an addition to the training signal, not the metric.
        # Self-conditioned blocks are excluded from METRICS (never from the loss):
        # their inputs contain revealed tokens, so "accuracy" there is not a
        # prediction metric. The surviving all-MASK blocks are a uniform random
        # subset, keeping train metrics directly comparable to stock DFlash.
        metric_mask = weight_mask
        if self_cond_blocks is not None:
            metric_mask = weight_mask * (~self_cond_blocks).unsqueeze(-1).float()
        binary_eval_mask = metric_mask.view(-1)

        # 9. Per-token loss: ce_loss_alpha*CE + l1_loss_alpha*L1.
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = target_ids.view(-1)
        ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        loss_per_token = self.ce_loss_alpha * ce_per_token
        if self.l1_loss_alpha > 0:
            if last_hidden_states is None:
                raise ValueError(
                    "DFlash L1 distillation (l1_loss_alpha > 0) requires target "
                    "last_hidden_states; set inference.store_last_hidden_states=true in the "
                    "run config."
                )
            tgt_idx = (safe_label_indices - 1).clamp(min=0)  # [B, n_blocks, block_size]
            hdim = last_hidden_states.size(-1)
            gather_idx = tgt_idx.reshape(bsz, -1, 1).expand(-1, -1, hdim)
            aligned_hidden = torch.gather(last_hidden_states, 1, gather_idx)
            target_logits = F.linear(aligned_hidden, lm_head_weight).view(-1, vocab_size)
            target_probs = torch.softmax(target_logits.float(), dim=-1)
            draft_probs = torch.softmax(flat_logits.float(), dim=-1)
            l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1)
            loss_per_token = loss_per_token + self.l1_loss_alpha * l1_per_token

        loss_per_token_by_position = ce_per_token.view(bsz, n_blocks, self.block_size)

        objective_weights = weight_mask
        if (
            self.loss_objective == "decay"
            and self.loss_decay_gamma is not None
            and self.loss_decay_gamma > 0
        ):
            # Loss decay: exp(-(k-1)/γ) so k=1 (1st prediction) gets weight 1.0
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)
            objective_weights = weight_mask * decay_weights
        elif self.loss_objective == "dpace":
            dpace_weights = torch.ones_like(weight_mask)
            if self.block_size > 1:
                with torch.no_grad():
                    target_confidences = torch.exp(-loss_per_token_by_position[..., 1:].float())
                    dpace_pred_weights = _dpace_position_weights(
                        target_confidences,
                        self.dpace_alpha,
                    ).to(dtype=weight_mask.dtype)
                dpace_weights[..., 1:] = dpace_pred_weights
            objective_weights = weight_mask * dpace_weights

        flat_weights = objective_weights.view(-1)
        valid_token_count = flat_weights.sum().clamp(min=1e-6)
        loss = (loss_per_token * flat_weights).sum() / valid_token_count

        loss_components = {}

        # 10. Accuracy (using binary mask without decay)
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum().clamp(min=1e-6)
            accuracy = correct.sum().float() / actual_token_count

            # Per-position-within-block metrics (index 0 = anchor, masked out;
            # indices 1..block_size-1 correspond to 1..B-1 tokens past the anchor).
            # Matches Eagle3's per-TTT-position breakdown semantically.
            binary_weights = binary_eval_mask.view(bsz, n_blocks, self.block_size)
            count_per_position = binary_weights.sum(dim=(0, 1))
            count_per_pos = count_per_position.clamp(min=1.0)

            loss_per_position = (loss_per_token_by_position * binary_weights).sum(
                dim=(0, 1)
            ) / count_per_pos
            acc_per_position = (correct.view(bsz, n_blocks, self.block_size).float()).sum(
                dim=(0, 1)
            ) / count_per_pos

            # Per-block consecutive-correct length starting at position 1: a greedy
            # accept-length proxy that, unlike simulated_acc_len (a product of
            # per-position marginals), captures within-block correlation. Blocks
            # truncated by the sequence end under-count equally for all paths.
            correct3 = correct.view(bsz, n_blocks, self.block_size)
            valid3 = binary_weights > 0.5
            loss_components["block_acc_len_det"] = self._consecutive_len(correct3, valid3)

            # Two-pass remask sampling (eval only): keep the most confident
            # pass-1 tokens, re-mask the rest, run a second pass that predicts
            # the doubtful slots with the kept tokens visible in the block.
            # ReMDM-style inference-time refinement, measured two ways:
            #   frozen — kept tokens are final, pass 2 fills only re-masked slots
            #   revise — pass 2 re-predicts every slot (kept tokens steer attention)
            if (not self.training) and self.remask_eval_keep > 0.0 and self.block_size > 1:
                conf = self._confidence_from_logits(logits)
                keep_frac = torch.full((bsz, n_blocks), self.remask_eval_keep, device=device)
                revealed_eval = self._reveal_from_confidence(
                    conf, block_keep_mask, block_keep_mask, keep_frac
                )
                flat_revealed = revealed_eval.view(bsz, -1)
                pred_p1 = pred_ids.view(bsz, -1)
                remask_ids = torch.where(flat_revealed, pred_p1, prep["noise_ids"])
                hidden_p2 = self._run_draft(prep, remask_ids)
                pred_p2, _ = self._predict_tokens(hidden_p2, lm_head_weight)

                targets2 = flat_targets.view(bsz, -1)
                final_frozen = torch.where(flat_revealed, pred_p1, pred_p2)
                frozen_correct = (final_frozen == targets2).view(bsz, n_blocks, self.block_size)
                revise_correct = (pred_p2 == targets2).view(bsz, n_blocks, self.block_size)
                loss_components["block_acc_len_remask"] = self._consecutive_len(
                    frozen_correct, valid3
                )
                loss_components["block_acc_len_revise"] = self._consecutive_len(
                    revise_correct, valid3
                )
                kept = flat_revealed & (binary_weights.view(bsz, -1) > 0.5)
                kept_count = kept.float().sum().clamp(min=1.0)
                loss_components["remask_keep_acc"] = (
                    (pred_p1 == targets2) & kept
                ).float().sum() / kept_count

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
        )
