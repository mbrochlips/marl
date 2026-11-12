import copy
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import optim
from gymnasium.spaces import flatdim, Discrete, MultiDiscrete, Box, Tuple as GymTuple

from marlbase.utils.model_critics import FiLMBlock
from marlbase.utils.standardise_stream import RunningMeanStd
from marlbase.utils.detectors import CoordinationDetector, DummyDetector, RiskDetector


def compute_multichannel_gae(
    rewards,       # (T, B, N, M)
    dones,         # (T, B, N)
    values,        # (T, B, N, M)
    next_values,   # (T, B, N, M)
    gamma,
    gae_lambda,
):
    """Vectorized Multi-Channel Generalized Advantage Estimation."""
    T, B, N, M = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, N, M, device=rewards.device)

    non_terminals = 1.0 - dones.unsqueeze(-1).float()

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * non_terminals[t] - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminals[t] * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class MATDecomposedCriticHead(nn.Module):
    def __init__(self, d_model: int, n_agents: int, max_channels: int = 32, n_heads: int = 4, d_ffn: int = 256):
        super().__init__()
        self.d_model = d_model
        self.n_agents = n_agents
        self.max_channels = max_channels

        self.channel_embeddings = nn.Embedding(max_channels, d_model)
        nn.init.orthogonal_(self.channel_embeddings.weight)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.film_ffn = FiLMBlock(d_model, d_ffn)
        self.final_norm = nn.LayerNorm(d_model)

        self.w_team = nn.Parameter(torch.empty(max_channels, d_model))
        self.b_team = nn.Parameter(torch.zeros(max_channels))
        self.delta_w = nn.Parameter(torch.zeros(n_agents, max_channels, d_model))
        self.delta_b = nn.Parameter(torch.zeros(n_agents, max_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.w_team, gain=1.0)
        nn.init.zeros_(self.b_team)
        nn.init.zeros_(self.delta_w)
        nn.init.zeros_(self.delta_b)

    def forward(self, enc_tokens: torch.Tensor, active_indices: torch.Tensor) -> torch.Tensor:
        B, N, D = enc_tokens.shape
        ch_embs = self.channel_embeddings(active_indices)
        queries = ch_embs.unsqueeze(0).expand(B, -1, -1)

        q_norm = self.norm_q(queries)
        kv_norm = self.norm_kv(enc_tokens)
        attn_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm)
        h = queries + attn_out

        h = self.film_ffn(h, ch_embs)
        h = self.final_norm(h)

        w = self.w_team[active_indices]
        b = self.b_team[active_indices]
        dw = self.delta_w[:, active_indices]
        db = self.delta_b[:, active_indices]

        v_team = torch.einsum("bmd, md -> bm", h, w) + b.unsqueeze(0)
        v_offset = torch.einsum("bmd, nmd -> bnm", h, dw) + db.unsqueeze(0)
        return v_team.unsqueeze(1) + v_offset


# ==============================================================================
# 3. MAT BACKBONE
# ==============================================================================

class MATBackbone(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        max_channels: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 1,
        d_ffn: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_channels = max_channels

        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.agent_id_emb = nn.Embedding(n_agents, d_model)

        # Token action_dim is the <START> token
        self.start_token_id = action_dim
        self.act_emb = nn.Embedding(action_dim + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.action_head = nn.Linear(d_model, action_dim)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

        self.critic_head = MATDecomposedCriticHead(
            d_model=d_model, n_agents=n_agents, max_channels=max_channels, n_heads=n_heads, d_ffn=d_ffn
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        B, N, _ = obs.shape
        agent_ids = torch.arange(N, device=obs.device).unsqueeze(0).expand(B, -1)
        x = self.obs_proj(obs) + self.agent_id_emb(agent_ids)
        return self.encoder(x)

    def decode(
        self,
        enc_tokens: torch.Tensor,
        actions_in: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, K = actions_in.shape
        tgt = self.act_emb(actions_in)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(K, device=enc_tokens.device)

        dec_out = self.decoder(tgt=tgt, memory=enc_tokens[:, :K], tgt_mask=causal_mask)
        logits = self.action_head(dec_out)

        if action_masks is not None:
            logits = logits.masked_fill(action_masks[:, :K] == 0, -1e9)
        return logits

    def get_values(self, enc_tokens: torch.Tensor, active_indices: torch.Tensor) -> torch.Tensor:
        return self.critic_head(enc_tokens, active_indices)


# ==============================================================================
# 4. MAT-TO AGENT CLASS (Full train.py Compatibility)
# ==============================================================================

class MAT_TO(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()
        self.device = (
            torch.device(device) if device is not None
            else torch.device(getattr(cfg, "device", getattr(cfg.model, "device", "cuda" if torch.cuda.is_available() else "cpu")))
        )
        self.cfg = cfg

        # Parse spaces from train.py
        self.n_agents, self.obs_dim, self.action_dim = self._parse_spaces(obs_space, action_space)

        self.gamma = cfg.gamma
        self.gae_lambda = getattr(cfg, "gae_lambda", 0.95)
        self.clip_param = getattr(cfg, "clip_param", 0.2)
        self.ppo_epochs = getattr(cfg, "ppo_epochs", 10)
        self.num_mini_batches = getattr(cfg, "num_mini_batches", 1)
        self.value_loss_coef = cfg.value_loss_coef
        self.entropy_coef = cfg.entropy_coef
        self.grad_clip = cfg.grad_clip
        self.use_clipped_value_loss = getattr(cfg, "use_clipped_value_loss", True)
        self.optimism_alpha = getattr(cfg, "optimism_alpha", 0.1)
        self.max_channels = getattr(cfg, "max_channels", 32)
        self.standardise_returns = getattr(cfg, "standardise_returns", True)
        self.target_update_interval_or_tau = getattr(cfg, "target_update_interval_or_tau", 0.01)

        d_model = getattr(cfg, "d_model", getattr(cfg, "critic_d_model", 128))
        d_ffn = getattr(cfg, "d_ffn", getattr(cfg, "critic_d_ffn", 256))
        n_heads = getattr(cfg, "n_heads", getattr(cfg, "critic_n_heads", 4))

        self.mat = MATBackbone(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            n_agents=self.n_agents,
            max_channels=self.max_channels,
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=getattr(cfg, "num_encoder_layers", 2),
            num_decoder_layers=getattr(cfg, "num_decoder_layers", 1),
            d_ffn=d_ffn,
        )

        self.target_mat = copy.deepcopy(self.mat)
        for p in self.target_mat.parameters():
            p.requires_grad = False

        self.register_buffer("basis_coord_scores", torch.zeros(self.max_channels))
        self.register_buffer("active_channel_indices", torch.arange(4, dtype=torch.long))
        self.n_reward_channels = len(self.active_channel_indices)
        self.channel_names = [f"channel_{i}" for i in range(self.n_reward_channels)]

        self.coord_detector = CoordinationDetector(self.channel_names, self.n_agents)
        self.ret_ms = RunningMeanStd(shape=(self.n_agents, self.n_reward_channels), device=self.device)

        self.to(self.device)
        self.optimizer = optim.Adam(self.mat.parameters(), lr=cfg.lr, eps=1e-5)

    def _parse_spaces(self, obs_space, action_space):
        if isinstance(obs_space, (list, tuple, GymTuple)):
            n_agents = len(obs_space)
            obs_dim = flatdim(obs_space[0])
        elif isinstance(obs_space, Box) and len(obs_space.shape) >= 2:
            n_agents = obs_space.shape[0]
            obs_dim = obs_space.shape[1]
        else:
            n_agents = getattr(self.cfg, "n_agents", 2)
            obs_dim = flatdim(obs_space) // n_agents

        if isinstance(action_space, (list, tuple, GymTuple)):
            first_act = action_space[0]
            action_dim = first_act.n if hasattr(first_act, "n") else flatdim(first_act)
        elif isinstance(action_space, MultiDiscrete):
            action_dim = int(action_space.nvec[0])
        elif isinstance(action_space, Discrete):
            action_dim = action_space.n
        else:
            action_dim = flatdim(action_space)

        return n_agents, obs_dim, action_dim

    def set_active_channels(
        self,
        active_indices: torch.Tensor,
        channel_names: Optional[List[str]] = None,
        **kwargs,
    ):
        self.active_channel_indices = active_indices.to(self.device).long()
        self.n_reward_channels = len(active_indices)
        self.channel_names = list(channel_names) if channel_names else [f"channel_{i}" for i in range(self.n_reward_channels)]

        detector_cls = CoordinationDetector if getattr(self.cfg, "optimistic_targets", True) else DummyDetector
        self.coord_detector = detector_cls(
            channel_names=self.channel_names,
            n_agents=self.n_agents,
            confidence_lambda=getattr(self.cfg, "confidence_lambda", 0.35),
        ).to(self.device)

        if self.standardise_returns:
            self.ret_ms = RunningMeanStd(
                shape=(self.n_agents, self.n_reward_channels), device=self.device
            )

    def init_actor_hiddens(self, batch_size: int = 1):
        """MAT is non-recurrent across timesteps. Returns None for compatibility."""
        return None

    def _format_obs(self, obs) -> torch.Tensor:
        """Converts any obs structure from train.py into (B, N, obs_dim)."""
        if isinstance(obs, (list, tuple)):
            obs_t = [torch.as_tensor(o, device=self.device, dtype=torch.float32) for o in obs]
            if obs_t[0].ndim == 1:
                obs_t = [o.unsqueeze(0) for o in obs_t]
            return torch.stack(obs_t, dim=1)

        t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if t.ndim == 3:
            return t if t.shape[1] == self.n_agents else t.transpose(0, 1)
        if t.ndim == 2:
            if t.shape[0] == self.n_agents:
                return t.unsqueeze(0)
            if t.shape[1] == self.n_agents * self.obs_dim:
                return t.view(-1, self.n_agents, self.obs_dim)
        return t

    def _format_mask(self, mask, B: int) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        m = torch.as_tensor(mask, device=self.device, dtype=torch.float32)
        if m.ndim == 2 and m.shape[0] == self.n_agents:
            return m.unsqueeze(0)  # (1, N, act_dim)
        if m.ndim == 3 and m.shape[0] == self.n_agents:
            return m.transpose(0, 1)  # (B, N, act_dim)
        return m

    @torch.no_grad()
    def act(
        self,
        inputs,
        actor_hiddens=None,
        action_mask=None,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        """
        Rollout step. Strictly returns:
          - actions: (N, B, 1)
          - actor_hiddens: None
          - log_probs: (B, N)
        """
        obs_tensor = self._format_obs(inputs)  # (B, N, obs_dim)
        B, N, _ = obs_tensor.shape
        mask_tensor = self._format_mask(action_mask, B)

        enc_tokens = self.mat.encode(obs_tensor)

        actions = torch.zeros(B, N, dtype=torch.long, device=self.device)
        log_probs = torch.zeros(B, N, dtype=torch.float32, device=self.device)
        seq_inputs = torch.full((B, 1), self.mat.start_token_id, dtype=torch.long, device=self.device)

        for i in range(N):
            cur_mask = mask_tensor[:, :i+1] if mask_tensor is not None else None
            logits = self.mat.decode(enc_tokens[:, :i+1], seq_inputs, cur_mask)
            next_logits = logits[:, -1, :]

            dist = Categorical(logits=next_logits)
            a_i = dist.sample()

            actions[:, i] = a_i
            log_probs[:, i] = dist.log_prob(a_i)
            seq_inputs = torch.cat([seq_inputs, a_i.unsqueeze(1)], dim=1)

        # Reshape actions to (N, B, 1) for train.py
        actions_out = actions.transpose(0, 1).unsqueeze(-1)  # (N, B, 1)
        return actions_out, None, log_probs

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parallel causal trajectory evaluation."""
        B, N, _ = obs.shape
        enc_tokens = self.mat.encode(obs)
        values = self.mat.get_values(enc_tokens, self.active_channel_indices)

        start_tokens = torch.full((B, 1), self.mat.start_token_id, dtype=torch.long, device=obs.device)
        shifted_actions = torch.cat([start_tokens, actions[:, :-1]], dim=1)

        logits = self.mat.decode(enc_tokens, shifted_actions, action_masks)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().sum(dim=-1)

        return values, log_probs, entropy

    def soft_update(self, tau: float):
        with torch.no_grad():
            for target_p, p in zip(self.target_mat.parameters(), self.mat.parameters()):
                target_p.data.copy_((1.0 - tau) * target_p.data + tau * p.data)

    def update(self, batch, step: int) -> Dict[str, float]:
        """Executes MAT-TO updates on collected batch."""
        T, B, N = batch.actions.shape
        filled_float = batch.filled.float()
        
        # Align dones with step t: batch.dones has shape (T+1, B)
        batch_dones = batch.dones.float().unsqueeze(-1).repeat(1, 1, self.n_agents)
        dones = batch_dones[1:] if batch_dones.shape[0] == T + 1 else batch_dones

        # batch.obss is (T+1, B, total_obs_dim) -> reshape to (T+1, B, N, obs_dim)
        obss_formatted = batch.obss.view(T + 1, B, N, self.obs_dim)

        # 1. Target Values & Multi-Channel GAE
        with torch.no_grad():
            flat_obss = obss_formatted.view((T + 1) * B, N, self.obs_dim)
            flat_enc = self.mat.encode(flat_obss)
            all_values = self.mat.get_values(flat_enc, self.active_channel_indices).view(T + 1, B, N, -1)

            v_preds = all_values[:-1]
            next_v_preds = all_values[1:]

            if self.standardise_returns:
                std = torch.sqrt(self.ret_ms.var + 1e-8)
                v_preds_raw = v_preds * std + self.ret_ms.mean
                next_v_preds_raw = next_v_preds * std + self.ret_ms.mean
            else:
                v_preds_raw, next_v_preds_raw = v_preds, next_v_preds

            raw_advantages, raw_returns = compute_multichannel_gae(
                rewards=batch.rewards,
                dones=dones,
                values=v_preds_raw,
                next_values=next_v_preds_raw,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )

            if self.standardise_returns:
                self.ret_ms.update(raw_returns)
                target_returns = (raw_returns - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var + 1e-8)
            else:
                target_returns = raw_returns

            old_log_probs = batch.action_log_probs

        # 2. Targeted Optimism
        coord_scores = self.coord_detector.update(batch.rewards, filled_float)
        self.basis_coord_scores[self.active_channel_indices] = coord_scores.detach()

        alpha = 1.0 - coord_scores * (1.0 - self.optimism_alpha)
        alpha_broadcast = alpha.view(1, 1, 1, -1)

        optimistic_adv = torch.where(raw_advantages < 0.0, alpha_broadcast * raw_advantages, raw_advantages)
        total_adv = optimistic_adv.sum(dim=-1)

        active_mask = filled_float.unsqueeze(-1).expand_as(total_adv)
        adv_denom = active_mask.sum() + 1e-8
        adv_mean = (total_adv * active_mask).sum() / adv_denom
        adv_std = torch.sqrt((((total_adv - adv_mean) ** 2) * active_mask).sum() / adv_denom + 1e-8)
        norm_adv = (total_adv - adv_mean) / (adv_std + 1e-8)

        # 3. PPO Mini-Batch Training Loop
        mini_batch_size = max(1, B // self.num_mini_batches)
        epoch_actor_losses, epoch_value_losses = [], []
        epoch_entropies, epoch_approx_kls, epoch_clip_fracs = [], [], []

        for _ in range(self.ppo_epochs):
            env_perm = torch.randperm(B)
            for start_idx in range(0, B, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, B)
                b_idx = env_perm[start_idx:end_idx]

                mb_obss = obss_formatted[:-1, b_idx].reshape(-1, N, self.obs_dim)
                mb_actions = batch.actions[:, b_idx].reshape(-1, N)
                mb_masks = (
                    batch.action_masks[:-1, b_idx].reshape(-1, N, self.action_dim)
                    if batch.action_masks is not None else None
                )
                mb_filled = filled_float[:, b_idx].reshape(-1)
                mb_old_log_probs = old_log_probs[:, b_idx].reshape(-1, N)
                mb_adv = norm_adv[:, b_idx].reshape(-1, N)
                mb_target_ret = target_returns[:, b_idx].reshape(-1, N, self.n_reward_channels)
                mb_v_old = v_preds[:, b_idx].reshape(-1, N, self.n_reward_channels)

                new_values, new_log_probs, entropy = self.evaluate_actions(mb_obss, mb_actions, mb_masks)

                # Actor Loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs.detach())
                surr1 = ratio * mb_adv.detach()
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_adv.detach()
                actor_loss = -torch.min(surr1, surr2).sum(dim=-1)
                actor_loss = (actor_loss * mb_filled).sum() / (mb_filled.sum() * N + 1e-8)
                entropy_loss = (entropy * mb_filled).sum() / (mb_filled.sum() + 1e-8)

                # Critic Loss
                v_target = mb_target_ret.detach()
                if self.use_clipped_value_loss:
                    v_clipped = mb_v_old + torch.clamp(new_values - mb_v_old, -self.clip_param, self.clip_param)
                    vf_loss1 = (new_values - v_target).pow(2)
                    vf_loss2 = (v_clipped - v_target).pow(2)
                    channel_vf_loss = torch.max(vf_loss1, vf_loss2)
                else:
                    channel_vf_loss = (new_values - v_target).pow(2)

                mask_exp = mb_filled.unsqueeze(-1).unsqueeze(-1)
                total_value_loss = (channel_vf_loss * mask_exp).sum() / (
                    mask_exp.sum() * self.n_reward_channels + 1e-8
                )

                loss = (actor_loss - self.entropy_coef * entropy_loss) + self.value_loss_coef * total_value_loss

                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.mat.parameters(), self.grad_clip)
                self.optimizer.step()

                with torch.no_grad():
                    ratio_diff = ratio - 1.0
                    approx_kl = (ratio_diff - torch.log(ratio)).mean()
                    clip_frac = (ratio_diff.abs() > self.clip_param).float().mean()
                    epoch_actor_losses.append(actor_loss.detach())
                    epoch_value_losses.append(total_value_loss.detach())
                    epoch_entropies.append(entropy_loss.detach())
                    epoch_approx_kls.append(approx_kl.detach())
                    epoch_clip_fracs.append(clip_frac.detach())

        tau = self.target_update_interval_or_tau
        if tau >= 1.0:
            if step % int(tau) == 0:
                self.soft_update(1.0)
        else:
            self.soft_update(tau)

        mean_actor_loss = torch.stack(epoch_actor_losses).mean().item()
        mean_value_loss = torch.stack(epoch_value_losses).mean().item()
        coord_scores_list = coord_scores.detach().cpu().tolist()

        return {
            "loss": mean_actor_loss + self.value_loss_coef * mean_value_loss,
            "actor_loss": mean_actor_loss,
            "value_loss": mean_value_loss,
            "entropy": torch.stack(epoch_entropies).mean().item(),
            "approx_kl": torch.stack(epoch_approx_kls).mean().item(),
            "clip_frac": torch.stack(epoch_clip_fracs).mean().item(),
            **{
                f"coord_{self.channel_names[m]}": coord_scores_list[m]
                for m in range(self.n_reward_channels)
            },
        }