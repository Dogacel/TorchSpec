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

"""FlowSpec training wrapper: flow-matching objective over DFlash's block machinery.

Reuses DFlashModel's anchor sampling, block-causal FlexAttention mask, and
position-id construction verbatim; replaces the objective:

  - Per block, sample one flow time t (logit-normal, mid-noise heavy, optional
    shift), corrupt the block's clean target hiddens toward Gaussian noise,
    and train the backbone to predict the velocity (clean - noise).
  - Small token-space anchor: decode the one-jump estimate of the clean state
    through the frozen LM head and apply position-decayed CE, so training
    stays pointed at "tokens that verify" rather than pure hidden geometry.

Training never runs the sampler (each example is a single random point on the
noise-to-data path). Eval additionally integrates the flow with K Euler steps
on a shifted grid and reports sampled-token accuracy plus block_acc_len (the
per-block consecutive-correct accept-length proxy).
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torchspec.models.dflash import DFlashModel


def shift_time(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Timestep shift t' = s*t / (1 + (s-1)*t). shift=1 is the identity."""
    if shift == 1.0:
        return t
    return shift * t / (1.0 + (shift - 1.0) * t)


class FlowSpecModel(DFlashModel):
    """FlowSpec training wrapper (flow-matching objective + eval sampler)."""

    CHUNK = 4  # AE patch size (tokens per latent)

    def __init__(
        self,
        draft_model,
        block_size: int = 16,
        num_anchors: int = 512,
        loss_decay_gamma: float = 7.0,
        ce_loss_alpha: float = 0.2,
        chunk_ae_path: str = None,
        l1_loss_alpha: float = 0.0,
    ):
        # Reuse DFlash anchor/mask/position machinery; the "decay" objective
        # setting only feeds the CE anchor's per-position weights here.
        super().__init__(
            draft_model=draft_model,
            block_size=block_size,
            num_anchors=num_anchors,
            loss_objective="decay",
            dpace_alpha=0.5,
            loss_decay_gamma=loss_decay_gamma,
            ce_loss_alpha=ce_loss_alpha,
            l1_loss_alpha=float(l1_loss_alpha),
        )
        self._flow_generator: Optional[torch.Generator] = None

        # Chunk-latent mode: attach the frozen, gate-passed CALM-style chunk
        # autoencoder. It lives on the wrapper (not draft_model) so the
        # optimizer — built over draft_model params — never sees it.
        self.chunk_ae = None
        if str(getattr(draft_model.config, "flow_state", "hidden")) == "chunk_latent":
            if not chunk_ae_path:
                raise ValueError("flow_state=chunk_latent requires chunk_ae_path")
            from torchspec.models.calm_ae import Autoencoder, AutoencoderConfig

            ae_cfg = AutoencoderConfig.from_pretrained(chunk_ae_path)
            ae = Autoencoder(ae_cfg)
            import os

            ae.load_state_dict(
                torch.load(
                    os.path.join(chunk_ae_path, "autoencoder.pt"),
                    map_location="cpu",
                    weights_only=True,
                )
            )
            ae.requires_grad_(False)
            ae.eval()
            self.chunk_ae = ae.to(torch.bfloat16)
            if int(ae_cfg.patch_size) != self.CHUNK:
                raise ValueError(f"AE patch_size {ae_cfg.patch_size} != {self.CHUNK}")

    def _get_flow_generator(self, device: torch.device) -> torch.Generator:
        """Dedicated RNG for flow noise/time so the global stream (anchor
        sampling, data order) stays deterministic and comparable across runs."""
        if self._flow_generator is None:
            seed = torch.initial_seed() % (2**31)
            if dist.is_available() and dist.is_initialized():
                seed += dist.get_rank() * 1000003
            g = torch.Generator(device=device)
            g.manual_seed(seed + 7919)
            self._flow_generator = g
        return self._flow_generator

    def _sample_train_t(
        self, bsz: int, n_blocks: int, device: torch.device, gen: torch.Generator
    ) -> torch.Tensor:
        """Per-position flow times [B, nb, bs].

        Base time per block is logit-normal(+shift); with flow_t_per_position a
        depth-graded profile is added (diffusion-forcing style): nearer-future
        slots get HIGHER t (less noise), deeper slots lower t, matching how
        uncertainty grows with horizon. Slot 0 (the known anchor) is pinned to
        t=1 ("clean") so its input-additive time embedding announces certainty.
        """
        cfg = self.draft_model.config
        bs = self.block_size
        z = torch.randn(bsz, n_blocks, device=device, dtype=torch.float32, generator=gen)
        t_base = torch.sigmoid(
            z * float(getattr(cfg, "flow_t_std", 1.0)) + float(getattr(cfg, "flow_t_mean", 0.0))
        )
        t_base = shift_time(t_base, float(getattr(cfg, "flow_time_shift", 1.0)))
        t3 = t_base.unsqueeze(-1).expand(-1, -1, bs).clone()
        if bool(getattr(cfg, "flow_t_per_position", True)) and bs > 2:
            slope = float(getattr(cfg, "flow_t_slope", 0.35))
            jitter = float(getattr(cfg, "flow_t_jitter", 0.05))
            # Depth within the predicted region (slots 1..bs-1) in [0, 1].
            depth = (torch.arange(bs, device=device, dtype=torch.float32) - 1.0).clamp(min=0) / max(
                bs - 2, 1
            )
            profile = slope * (0.5 - depth)  # front +slope/2, deepest -slope/2
            jit = (torch.rand(bsz, n_blocks, bs, device=device, generator=gen) * 2.0 - 1.0) * jitter
            t3 = (t3 + profile.view(1, 1, -1) + jit).clamp(0.0, 0.98)
        t3[:, :, 0] = 1.0  # anchor slot: known/clean
        return t3

    @torch.no_grad()
    def _encode_chunks(self, token_windows: torch.Tensor) -> torch.Tensor:
        """Mean AE latent for token windows [B, nb, CHUNK] -> [B, nb, D_lat]."""
        bsz, n_blocks, k = token_windows.shape
        latent = self.chunk_ae.encoder(input_ids=token_windows.reshape(-1, k))
        mean, _ = torch.chunk(latent, 2, dim=-1)
        return mean.reshape(bsz, n_blocks, -1).float()

    def _chunk_tail(
        self,
        *,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        last_hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        context_feature: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        n_blocks: int,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask,
        seq_len: int,
        bsz: int,
        device: torch.device,
    ):
        """Chunk-latent flow objective (flow_state="chunk_latent").

        One AE latent per block: source = E(last CHUNK committed tokens) — pure
        tokens, available at serving by construction — target = E(next CHUNK
        tokens). Velocity + timestep scheduling in the AE's trained-smooth
        128-d space; decode through the frozen AE decoder. The draft block is
        [anchor embedding, latent slot] (block_size must be 2).

        Metric arrays are length 1+CHUNK (leading anchor slot dropped by the
        trainer): per-TOKEN accuracy of the decoded chunk, so sim/block accept
        metrics read exactly like the other drafters over a 4-token horizon.
        """
        draft = self.draft_model
        K = self.CHUNK
        bs = self.block_size
        if bs != 2:
            raise ValueError("chunk_latent mode expects dflash_block_size=2 (anchor + latent slot)")
        d_lat = draft.state_dim
        cfg = draft.config
        gen = self._get_flow_generator(device)

        # ---- Validity: need K committed tokens through the anchor and K labels after it ----
        keep = block_keep_mask & (anchor_positions >= K - 1) & (anchor_positions + K < seq_len)

        offs_tgt = torch.arange(1, K + 1, device=device).view(1, 1, -1)
        lbl_idx = (anchor_positions.unsqueeze(-1) + offs_tgt).clamp(max=seq_len - 1)
        target_ids = torch.gather(input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, lbl_idx)
        lm_tok = torch.gather(loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, lbl_idx)
        w_tok = keep.unsqueeze(-1).float() * lm_tok  # [B, nb, K]
        w_chunk = keep.float() * (lm_tok.amin(dim=-1) > 0.5).float()  # [B, nb]

        offs_src = torch.arange(-(K - 1), 1, device=device).view(1, 1, -1)
        src_idx = (anchor_positions.unsqueeze(-1) + offs_src).clamp(min=0)
        source_ids = torch.gather(input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, src_idx)

        z_src = self._encode_chunks(source_ids)
        z_tgt = self._encode_chunks(target_ids)

        if self.training:
            with torch.no_grad():
                cnt = w_chunk.sum().clamp(min=1.0)
                sel = w_chunk.unsqueeze(-1)
                mean_b = (z_tgt * sel).sum(dim=(0, 1)) / cnt
                var_b = ((z_tgt - mean_b).pow(2) * sel).sum(dim=(0, 1)) / cnt
                draft.update_stats(mean_b, var_b)

        x1 = draft.standardize(z_tgt.reshape(-1, d_lat)).view(bsz, n_blocks, d_lat)
        z0 = draft.standardize(z_src.reshape(-1, d_lat)).view(bsz, n_blocks, d_lat)

        # ---- t (slot 0 pinned clean; latent slot takes the block t) ----
        t3 = self._sample_train_t(bsz, n_blocks, device, gen)  # [B, nb, 2]
        t_lat = t3[..., 1]  # [B, nb]

        eps = torch.randn(x1.shape, device=device, dtype=torch.float32, generator=gen)
        sigma = float(getattr(cfg, "flow_source_sigma", 0.25))
        if str(getattr(cfg, "flow_source", "anchor")) == "noise":
            x0 = eps
        else:
            x0 = z0 + sigma * eps

        # Corrective states: corrupt the target with another block's latent.
        x1_build = x1
        corr_frac = float(getattr(cfg, "flow_corrective_frac", 0.0))
        if self.training and corr_frac > 0:
            with torch.no_grad():
                a_min = float(getattr(cfg, "flow_corrupt_alpha_min", 0.3))
                do_c = torch.rand(bsz, n_blocks, 1, device=device, generator=gen) < corr_frac
                alpha = (
                    torch.rand(bsz, n_blocks, 1, device=device, generator=gen) * (1.0 - a_min)
                    + a_min
                )
                x1_soft = alpha * x1 + (1.0 - alpha) * torch.roll(x1, 1, dims=1)
                x1_build = torch.where(do_c, x1_soft, x1)

        t_l1 = t_lat.unsqueeze(-1)
        x_t = (1.0 - t_l1) * x0 + t_l1 * x1_build

        state = torch.cat(
            [torch.zeros(bsz, n_blocks, 1, d_lat, device=device), x_t.unsqueeze(2)], dim=2
        )
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions.clamp(0, seq_len - 1))
        anchor_embeds = draft.embed_tokens(anchor_token_ids)
        stream = self._build_draft_stream(state, anchor_embeds)

        v = (
            self.draft_model(
                draft_inputs_embeds=stream,
                t=t3.reshape(bsz, -1),
                context_feature=context_feature,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
            )
            .float()
            .view(bsz, n_blocks, bs, d_lat)[:, :, 1]
        )  # latent slot

        one_minus = (1.0 - t_l1).clamp(min=0.05)
        v_target = (x1 - x_t) / one_minus
        fm_per_chunk = (v - v_target).pow(2).mean(dim=-1)
        fm_loss = (fm_per_chunk * w_chunk).sum() / w_chunk.sum().clamp(min=1e-6)

        # ---- CE anchor through the frozen AE decoder ----
        x1_hat = x_t + one_minus * v
        z_hat = draft.unstandardize(x1_hat.reshape(-1, d_lat))
        logits = (
            self.chunk_ae.decoder(
                latent_states=z_hat.view(-1, 1, d_lat).to(next(self.chunk_ae.parameters()).dtype)
            )
            .float()
            .view(bsz, n_blocks, K, -1)
        )
        vocab = logits.size(-1)
        ce_per_token = F.cross_entropy(
            logits.reshape(-1, vocab), target_ids.reshape(-1), reduction="none"
        ).view(bsz, n_blocks, K)
        k_pos = torch.arange(K, device=device).view(1, 1, -1)
        decay = torch.exp(-k_pos.float() / self.loss_decay_gamma)
        dw = w_tok * decay
        ce_loss = (ce_per_token * dw).sum() / dw.sum().clamp(min=1e-6)

        loss = fm_loss + self.ce_loss_alpha * ce_loss
        loss_components = {"fm_loss": fm_loss.detach(), "ce_loss": ce_loss.detach()}

        # L1 distribution distillation (DSpark-style): match the AE-decoded draft
        # distribution to the TARGET model's distribution at each position (from
        # its hidden at anchor+k-1 through the frozen LM head). The AE vocab has
        # one extra [PAD] row; compare over the shared vocab only.
        if self.l1_loss_alpha > 0:
            v_lm = lm_head_weight.size(0)
            hdim_t = last_hidden_states.size(-1)
            tgt_h_idx = (lbl_idx - 1).clamp(min=0)
            gi = tgt_h_idx.reshape(bsz, -1, 1).expand(-1, -1, hdim_t)
            aligned_h = torch.gather(last_hidden_states, 1, gi)
            tgt_logits = F.linear(aligned_h.to(lm_head_weight.dtype), lm_head_weight)
            # Shared vocab: the AE covers len(tokenizer)+PAD (151670 for Qwen3);
            # the LM head carries padded rows up to 151936 that are never
            # predicted. Softmax each over its own full vocab, compare the
            # shared prefix (the target's tail mass is ~0 by construction).
            shared = min(v_lm, logits.size(-1))
            tgt_probs = torch.softmax(tgt_logits.float().view(bsz, n_blocks, K, v_lm), dim=-1)[
                ..., :shared
            ]
            draft_probs = torch.softmax(logits, dim=-1)[..., :shared]
            l1_per_token = (draft_probs - tgt_probs).abs().sum(dim=-1)
            l1_loss = (l1_per_token * dw).sum() / dw.sum().clamp(min=1e-6)
            loss = loss + self.l1_loss_alpha * l1_loss
            loss_components["l1_loss"] = l1_loss.detach()

        # ---- Metrics: arrays of length 1+K (leading anchor slot, dropped) ----
        with torch.no_grad():
            zero = torch.zeros(1, device=device)
            if self.training:
                sel3 = w_tok * (t_lat.unsqueeze(-1) < 0.3).float()
                pred = logits.argmax(dim=-1)
                ok_all = (pred == target_ids) & (w_tok > 0.5)
                loss_components["denoise_acc"] = ok_all.sum().float() / w_tok.sum().clamp(min=1e-6)
                ok = ok_all & (sel3 > 0.5)
                count_pp = sel3.sum(dim=(0, 1))
                loss_pp = (ce_per_token * sel3).sum(dim=(0, 1)) / count_pp.clamp(min=1.0)
            else:
                s_logits = self._sample_chunk_latents(
                    z0=z0,
                    anchor_embeds=anchor_embeds,
                    context_feature=context_feature,
                    draft_position_ids=draft_position_ids,
                    context_position_ids=context_position_ids,
                    block_mask=block_mask,
                    gen=gen,
                    bsz=bsz,
                    n_blocks=n_blocks,
                )
                pred = s_logits.argmax(dim=-1)
                ok = (pred == target_ids) & (w_tok > 0.5)
                count_pp = w_tok.sum(dim=(0, 1))
                loss_pp = (ce_per_token * w_tok).sum(dim=(0, 1)) / count_pp.clamp(min=1.0)

                valid3 = w_tok > 0.5
                has_p1 = valid3[..., 0]
                nvb = has_p1.float().sum().clamp(min=1.0)
                run = (ok & valid3).int().cumprod(dim=-1)
                loss_components["block_acc_len"] = (
                    run.sum(dim=-1).float() * has_p1.float()
                ).sum() / nvb

            accuracy = ok.sum().float() / count_pp.sum().clamp(min=1e-6)
            acc_pp = ok.float().sum(dim=(0, 1)) / count_pp.clamp(min=1.0)
            loss_per_position = torch.cat([zero, loss_pp])
            acc_per_position = torch.cat([zero, acc_pp])
            count_per_position = torch.cat([zero, count_pp])

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
        )

    @torch.no_grad()
    def _sample_chunk_latents(
        self,
        *,
        z0: torch.Tensor,
        anchor_embeds: torch.Tensor,
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask,
        gen: torch.Generator,
        bsz: int,
        n_blocks: int,
    ) -> torch.Tensor:
        """Integrate one latent per block from the source and decode: returns
        token logits [B, nb, CHUNK, vocab] from the frozen AE decoder."""
        draft = self.draft_model
        cfg = draft.config
        d_lat = draft.state_dim
        device = z0.device
        num_steps = int(getattr(cfg, "flow_sample_steps", 8))
        sigma = float(getattr(cfg, "flow_source_sigma", 0.25))

        eps = torch.randn(z0.shape, device=device, dtype=torch.float32, generator=gen)
        x = eps if str(getattr(cfg, "flow_source", "anchor")) == "noise" else z0 + sigma * eps
        ts = self._sampler_timesteps(num_steps, device)
        for i in range(num_steps):
            state = torch.cat(
                [torch.zeros(bsz, n_blocks, 1, d_lat, device=device), x.unsqueeze(2)], dim=2
            )
            stream = self._build_draft_stream(state, anchor_embeds)
            t_tok = torch.stack(
                [
                    torch.ones(bsz, n_blocks, device=device),
                    torch.full((bsz, n_blocks), float(ts[i]), device=device),
                ],
                dim=-1,
            ).reshape(bsz, -1)
            v = (
                self.draft_model(
                    draft_inputs_embeds=stream,
                    t=t_tok,
                    context_feature=context_feature,
                    draft_position_ids=draft_position_ids,
                    context_position_ids=context_position_ids,
                    block_mask=block_mask,
                )
                .float()
                .view(bsz, n_blocks, self.block_size, d_lat)[:, :, 1]
            )
            x = x + (ts[i + 1] - ts[i]) * v
        z = draft.unstandardize(x.reshape(-1, d_lat))
        logits = self.chunk_ae.decoder(
            latent_states=z.view(-1, 1, d_lat).to(next(self.chunk_ae.parameters()).dtype)
        ).float()
        return logits.view(bsz, n_blocks, self.CHUNK, -1)

    def _flow_source_state(
        self,
        eps: torch.Tensor,
        last_hidden_states: torch.Tensor,
        anchor_positions: torch.Tensor,
        gen: torch.Generator,
    ) -> torch.Tensor:
        """x_0 for the flow: the newest AVAILABLE hidden state, standardized,
        broadcast to every slot plus sigma*noise — the path runs from where the
        sequence is now to where it is going. With flow_source="noise" this
        degrades to the pure-noise source (eps).

        "Available" means the hidden at anchor-1 (which generated the anchor
        token). The hidden AT the anchor position must not be used: at serving,
        the anchor is the bonus token whose own hidden has not been computed
        yet (the attention mask's strictly-before-anchor rule encodes the same
        timeline), and it would also hand slot 1 its own flow target — slot 1's
        target IS the anchor-position hidden.
        """
        cfg = self.draft_model.config
        if str(getattr(cfg, "flow_source", "anchor")) != "anchor":
            return eps
        sigma = float(getattr(cfg, "flow_source_sigma", 0.25))
        bsz, n_blocks, bs, hdim = eps.shape
        src_idx = (anchor_positions - 1).clamp(min=0)
        gather_idx = src_idx.reshape(bsz, -1, 1).expand(-1, -1, hdim)
        anchor_hidden = torch.gather(last_hidden_states, 1, gather_idx)  # [B, nb, D]
        anchor_std = self.draft_model.standardize(anchor_hidden.reshape(-1, hdim)).view(
            bsz, n_blocks, 1, hdim
        )
        return anchor_std + sigma * eps

    def _build_draft_stream(
        self,
        x_state: torch.Tensor,
        anchor_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Project the flow state into the model stream; slot 0 of every block
        carries the (known) anchor token embedding instead of a noisy state.

        Args:
            x_state: [B, nb, bs, D_target] standardized flow state (fp32)
            anchor_embeds: [B, nb, hidden]
        """
        bsz, n_blocks, bs, _ = x_state.shape
        dtype = self.draft_model.x_in.weight.dtype
        stream = self.draft_model.x_in(x_state.to(dtype))  # [B, nb, bs, hidden]
        stream = torch.cat([anchor_embeds.unsqueeze(2).to(dtype), stream[:, :, 1:]], dim=2)
        return stream.view(bsz, n_blocks * bs, -1)

    def _sampler_timesteps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Shifted Euler grid: uniform in u, mapped through the same shift used
        in training so step density concentrates where the trajectory bends."""
        cfg = self.draft_model.config
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=torch.float32)
        return shift_time(u, float(getattr(cfg, "flow_time_shift", 1.0)))

    @torch.no_grad()
    def _sample_block_hiddens(
        self,
        anchor_embeds: torch.Tensor,
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask,
        num_steps: int,
        gen: torch.Generator,
        bsz: int,
        n_blocks: int,
        last_hidden_states: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate the flow from the source to predicted target hiddens with a
        STAGGERED per-slot schedule: front slots reach t=1 up to flow_sample_lead
        steps early (then freeze) while deeper slots keep refining — mirroring
        the depth-graded training-time noise profile.

        Runs the full backbone once per step; context_feature is reused across
        steps. Returns un-standardized hiddens [B, nb, bs, D_target]; anchor
        slots (slot 0) are pinned at t=1 and never updated.
        """
        device = anchor_embeds.device
        d_t = self.draft_model.target_hidden_size
        bs = self.block_size
        cfg = self.draft_model.config

        eps = torch.randn(
            (bsz, n_blocks, bs, d_t), device=device, dtype=torch.float32, generator=gen
        )
        x = self._flow_source_state(eps, last_hidden_states, anchor_positions, gen)

        # Per-slot time grid [num_steps+1, bs]: slot k finishes at step F_k,
        # front slots earliest. t_k(i) = shift(clamp(i / F_k, 0, 1)).
        lead = min(int(getattr(cfg, "flow_sample_lead", 4)), max(num_steps - 1, 1))
        if bool(getattr(cfg, "flow_t_per_position", True)) and bs > 2:
            depth = (torch.arange(bs, dtype=torch.float32) - 1.0).clamp(min=0) / max(bs - 2, 1)
            finish = num_steps - (lead * (1.0 - depth)).round()
        else:
            finish = torch.full((bs,), float(num_steps))
        finish = finish.clamp(min=1.0).to(device)
        steps_i = torch.arange(num_steps + 1, device=device, dtype=torch.float32).unsqueeze(-1)
        grid = shift_time(
            (steps_i / finish.unsqueeze(0)).clamp(0.0, 1.0),
            float(getattr(cfg, "flow_time_shift", 1.0)),
        )  # [S+1, bs]
        grid[:, 0] = 1.0  # anchor slot: clean, frozen

        for i in range(num_steps):
            stream = self._build_draft_stream(x, anchor_embeds)
            t_tok = grid[i].view(1, 1, bs).expand(bsz, n_blocks, bs).reshape(bsz, -1)
            v = (
                self.draft_model(
                    draft_inputs_embeds=stream,
                    t=t_tok,
                    context_feature=context_feature,
                    draft_position_ids=draft_position_ids,
                    context_position_ids=context_position_ids,
                    block_mask=block_mask,
                )
                .float()
                .view(bsz, n_blocks, bs, d_t)
            )
            dt = (grid[i + 1] - grid[i]).view(1, 1, bs, 1)
            x = x + dt * v
        return self.draft_model.unstandardize(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """FlowSpec training forward.

        Returns DFlashModel.forward's contract: (loss, accuracy, loss_pp,
        acc_pp, count_pp, loss_components). In training mode the per-position
        metrics come from the cheap one-jump estimate (single forward); in eval
        mode acc metrics come from the true K-step sampler, and loss_components
        additionally carries block_acc_len (consecutive-correct accept proxy).
        """
        if last_hidden_states is None:
            raise ValueError(
                "FlowSpec requires target last_hidden_states (the flow targets); "
                "set inference.store_last_hidden_states=true in the run config."
            )

        bsz, seq_len = input_ids.shape
        device = input_ids.device
        draft = self.draft_model
        bs = self.block_size

        # ---- Context features (computed once; reused by every flow step) ----
        token_embeds = draft.embed_tokens(input_ids) if draft.fuse_embedding else None
        context_feature = draft.extract_context_feature(hidden_states_list, token_embeds)

        # ---- Anchors / positions / mask (DFlash machinery, reused) ----
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.shape[1]
        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len
        )

        block_mask = None
        if device.type == "cuda":
            from torchspec.models.dflash import _create_dflash_mask_mod
            from torchspec.models.ops.flex_attention import (
                compile_friendly_create_block_mask,
            )

            mask_mod = _create_dflash_mask_mod(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                ctx_len=seq_len,
                block_size=bs,
            )
            block_mask = compile_friendly_create_block_mask(
                mask_mod=mask_mod,
                B=bsz,
                H=None,
                Q_LEN=n_blocks * bs,
                KV_LEN=seq_len + n_blocks * bs,
                device=device,
            )

        if self.chunk_ae is not None:
            return self._chunk_tail(
                input_ids=input_ids,
                loss_mask=loss_mask,
                last_hidden_states=last_hidden_states,
                lm_head_weight=lm_head_weight,
                context_feature=context_feature,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                n_blocks=n_blocks,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
                seq_len=seq_len,
                bsz=bsz,
                device=device,
            )

        # ---- Labels + weights (same alignment as DFlash) ----
        label_offsets = torch.arange(0, bs, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, nb, bs]
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )

        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, bs).float()
        weight_mask = weight_mask * valid_label_mask.float()
        pos_in_block = torch.arange(bs, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()
        weight_mask = weight_mask * torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        flat_binary = weight_mask.view(-1)
        valid_count = flat_binary.sum().clamp(min=1e-6)

        # ---- Flow targets: hidden that generated the label token ----
        # Label at anchor+k was produced by the target's hidden at anchor+k-1.
        hdim = last_hidden_states.size(-1)
        tgt_idx = (safe_label_indices - 1).clamp(min=0)
        gather_idx = tgt_idx.reshape(bsz, -1, 1).expand(-1, -1, hdim)
        target_hidden = torch.gather(last_hidden_states, 1, gather_idx).view(
            bsz, n_blocks, bs, hdim
        )

        if self.training:
            with torch.no_grad():
                cnt = flat_binary.sum().clamp(min=1.0)
                sel = weight_mask.unsqueeze(-1)
                h_f32 = target_hidden.float()
                batch_mean = (h_f32 * sel).sum(dim=(0, 1, 2)) / cnt
                batch_var = ((h_f32 - batch_mean).pow(2) * sel).sum(dim=(0, 1, 2)) / cnt
                draft.update_stats(batch_mean, batch_var)

        x1 = draft.standardize(target_hidden.reshape(-1, hdim)).view(
            bsz, n_blocks, bs, hdim
        )  # fp32

        gen = self._get_flow_generator(device)
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions.clamp(0, seq_len - 1))
        anchor_embeds = draft.embed_tokens(anchor_token_ids)  # [B, nb, hidden]

        # ---- Corrupt: one t per block, linear path from noise ----
        t_tok4 = self._sample_train_t(bsz, n_blocks, device, gen)  # [B, nb, bs]
        eps = torch.randn(x1.shape, device=device, dtype=torch.float32, generator=gen)
        x0 = self._flow_source_state(eps, last_hidden_states, anchor_positions, gen)

        # Corrective-field corruption (training only): sometimes build x_t from a
        # blend of the true target and another position's target, so the model
        # learns to repair imperfect states like the ones its own sampler
        # produces mid-trajectory. See FlowSpecConfig.flow_corrective_frac.
        x1_build = x1
        corr_frac = float(getattr(draft.config, "flow_corrective_frac", 0.0))
        if self.training and corr_frac > 0:
            with torch.no_grad():
                a_min = float(getattr(draft.config, "flow_corrupt_alpha_min", 0.3))
                do_corrupt = (
                    torch.rand(bsz, n_blocks, 1, 1, device=device, generator=gen) < corr_frac
                )
                alpha = (
                    torch.rand(bsz, n_blocks, bs, 1, device=device, generator=gen) * (1.0 - a_min)
                    + a_min
                )
                slot_perm = torch.randperm(bs, device=device, generator=gen)
                x1_soft = alpha * x1 + (1.0 - alpha) * x1[:, :, slot_perm]
                x1_build = torch.where(do_corrupt, x1_soft, x1)

        x_t = (1.0 - t_tok4.unsqueeze(-1)) * x0 + t_tok4.unsqueeze(-1) * x1_build

        stream = self._build_draft_stream(x_t, anchor_embeds)
        v_pred = (
            self.draft_model(
                draft_inputs_embeds=stream,
                t=t_tok4.reshape(bsz, -1),
                context_feature=context_feature,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
            )
            .float()
            .view(bsz, n_blocks, bs, hdim)
        )

        # ---- Flow-matching loss (uniform over valid slots) ----
        # Corrective-field target: point from wherever x_t is toward the TRUE
        # clean target. On uncorrupted paths this reduces exactly to the
        # standard rectified-flow target x1 - eps; on corrupted paths it
        # teaches repair. Clamp (1-t) to keep the target bounded near t=1.
        one_minus_t = (1.0 - t_tok4.unsqueeze(-1)).clamp(min=0.05)
        v_target = (x1 - x_t) / one_minus_t
        fm_per_token = (v_pred - v_target).pow(2).mean(dim=-1)  # [B, nb, bs]
        fm_loss = (fm_per_token.view(-1) * flat_binary).sum() / valid_count

        # ---- CE anchor on the one-jump estimate (position-decayed) ----
        x1_hat = x_t + (1.0 - t_tok4.unsqueeze(-1)) * v_pred
        h_hat = draft.unstandardize(x1_hat.reshape(-1, hdim))
        logits = F.linear(h_hat.to(lm_head_weight.dtype), lm_head_weight)
        vocab_size = logits.size(-1)
        flat_targets = target_ids.view(-1)
        # Label smoothing blunts CE's exploding curvature on confident errors —
        # the mechanism behind the recurring loss spikes at high LR.
        ce_per_token = F.cross_entropy(logits, flat_targets, reduction="none", label_smoothing=0.1)

        k = torch.arange(bs, device=device).view(1, 1, -1)
        decay = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)
        decay_weights = (weight_mask * decay).view(-1)
        ce_loss = (ce_per_token * decay_weights).sum() / decay_weights.sum().clamp(min=1e-6)

        loss = fm_loss + self.ce_loss_alpha * ce_loss

        loss_components = {
            "fm_loss": fm_loss.detach(),
            "ce_loss": ce_loss.detach(),
        }

        # ---- Metrics ----
        with torch.no_grad():
            ce_by_pos = ce_per_token.view(bsz, n_blocks, bs)
            binary3 = weight_mask

            if self.training:
                # One-jump metrics LEAK the answer in proportion to t (x_t
                # contains t*x1), so per-position train metrics are computed
                # ONLY over low-noise blocks (t < 0.3), where leakage is minor.
                # The all-t decode accuracy is logged separately as denoise_acc
                # — a trend diagnostic, NOT an acceptance proxy.
                metric3 = binary3 * (t_tok4 < 0.3).float()
                pred_ids = logits.argmax(dim=-1)
                correct_all = (pred_ids == flat_targets) & (flat_binary > 0.5)
                loss_components["denoise_acc"] = correct_all.sum().float() / valid_count
                correct = correct_all & (metric3.view(-1) > 0.5)
                count_per_position = metric3.sum(dim=(0, 1))
                count_pp = count_per_position.clamp(min=1.0)
                loss_per_position = (ce_by_pos * metric3).sum(dim=(0, 1)) / count_pp
            else:
                count_per_position = binary3.sum(dim=(0, 1))
                count_pp = count_per_position.clamp(min=1.0)
                # Leak-mixed one-jump CE, kept for curve continuity only.
                loss_per_position = (ce_by_pos * binary3).sum(dim=(0, 1)) / count_pp
                # Real thing: run the K-step sampler and decode.
                num_steps = int(getattr(draft.config, "flow_sample_steps", 16))
                h_sampled = self._sample_block_hiddens(
                    anchor_embeds,
                    context_feature,
                    draft_position_ids,
                    context_position_ids,
                    block_mask,
                    num_steps,
                    gen,
                    bsz,
                    n_blocks,
                    last_hidden_states,
                    anchor_positions,
                )
                s_logits = F.linear(h_sampled.to(lm_head_weight.dtype), lm_head_weight)
                pred_ids = s_logits.argmax(dim=-1).view(-1)
                correct = (pred_ids == flat_targets) & (flat_binary > 0.5)

                correct3 = correct.view(bsz, n_blocks, bs)
                valid3 = binary3 > 0.5
                has_p1 = valid3[..., 1] if bs > 1 else valid3[..., 0]
                n_valid_blocks = has_p1.float().sum().clamp(min=1.0)
                run = (correct3 & valid3)[..., 1:].int().cumprod(dim=-1)
                loss_components["block_acc_len"] = (
                    run.sum(dim=-1).float() * has_p1.float()
                ).sum() / n_valid_blocks

            accuracy = correct.sum().float() / count_per_position.sum().clamp(min=1e-6)
            acc_per_position = correct.view(bsz, n_blocks, bs).float().sum(dim=(0, 1)) / count_pp

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
        )
