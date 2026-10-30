import copy
from einops import rearrange
from gymnasium.spaces import flatdim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import optim

from marlbase.utils.models import MultiAgentIndependentNetwork, MultiAgentSharedNetwork
from marlbase.utils.utils import MultiCategorical
from marlbase.utils.standardise_stream import RunningMeanStd


def _split_batch(splits):
    def thunk(batch):
        return torch.split(batch, splits, dim=-1)
    return thunk


def compute_multichannel_gae(
    rewards,       # (T, B, N, M)
    dones,         # (T, B, N)
    values,        # (T, B, N, M)
    next_values,   # (T, B, N, M)
    gamma,
    gae_lambda,
):
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


class FiLMBlock(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.film_proj = nn.Linear(d_model, d_ffn * 2)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.linear1.weight, gain=1.0)
        nn.init.zeros_(self.linear1.bias)
        nn.init.orthogonal_(self.linear2.weight, gain=1.0)
        nn.init.zeros_(self.linear2.bias)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, x, channel_emb):
        norm_x = self.norm(x)
        h = self.linear1(norm_x)
        film_params = self.film_proj(channel_emb)
        gamma, beta = film_params.chunk(2, dim=-1)
        h = (1.0 + gamma.unsqueeze(0)) * h + beta.unsqueeze(0)
        h = F.gelu(h)
        return x + self.linear2(h)


class CrossAgentAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_agents, identity_bias=1.5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_agents = n_agents

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Mild diagonal prior (1.5 instead of 4.0 so exploratory gradients can shift attention)
        eye_matrix = torch.eye(n_agents) * identity_bias
        self.register_buffer("eye_bias", eye_matrix.view(1, 1, 1, n_agents, n_agents))

    def forward(self, x):
        B, M, N, _ = x.shape
        Q = self.q_proj(x).view(B, M, N, self.n_heads, self.head_dim).transpose(2, 3)
        K = self.k_proj(x).view(B, M, N, self.n_heads, self.head_dim).transpose(2, 3)
        V = self.v_proj(x).view(B, M, N, self.n_heads, self.head_dim).transpose(2, 3)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores = scores + self.eye_bias
        attn_probs = F.softmax(scores, dim=-1)  # (B, M, H, N, N)

        attn_out = torch.matmul(attn_probs, V).transpose(2, 3).contiguous().view(B, M, N, self.d_model)
        return self.out_proj(attn_out), attn_probs


class ChannelQueryAgentAttentionCritic(nn.Module):
    def __init__(
        self,
        obs_dims,
        n_agents,
        n_reward_channels,
        d_model=128,
        n_heads=4,
        d_ffn=256,
        identity_bias=1.5,
        **kwargs,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_channels = n_reward_channels
        self.d_model = d_model

        self.agent_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            ) for dim in obs_dims
        ])

        self.channel_embeddings = nn.Embedding(n_reward_channels, d_model)
        nn.init.normal_(self.channel_embeddings.weight, std=0.02)

        self.norm_agent = nn.LayerNorm(d_model)
        self.cross_agent_attn = CrossAgentAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_agents=n_agents,
            identity_bias=identity_bias,
        )

        self.film_ffn = FiLMBlock(d_model, d_ffn)
        self.final_norm = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, 1)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs_list, return_attention=False):
        orig_shape = obs_list[0].shape[:-1]
        flat_obs = [o.reshape(-1, o.shape[-1]) for o in obs_list]
        B_flat = flat_obs[0].shape[0]

        agent_tokens = torch.stack(
            [enc(o) for enc, o in zip(self.agent_encoders, flat_obs)], dim=1
        )  # (B_flat, N, d_model)

        ch_indices = torch.arange(self.n_channels, device=agent_tokens.device)
        ch_embs = self.channel_embeddings(ch_indices)

        x = agent_tokens.unsqueeze(1) + ch_embs.unsqueeze(0).unsqueeze(2)  # (B_flat, M, N, d_model)
        norm_x = self.norm_agent(x)

        attn_out, attn_probs = self.cross_agent_attn(norm_x)
        x = x + attn_out

        x_reshaped = x.transpose(1, 2).reshape(B_flat * self.n_agents, self.n_channels, self.d_model)
        x_ffn = self.film_ffn(x_reshaped, ch_embs)
        x = x_ffn.view(B_flat, self.n_agents, self.n_channels, self.d_model).transpose(1, 2)

        values = self.head(self.final_norm(x)).squeeze(-1).transpose(1, 2)
        values = values.view(*orig_shape, self.n_agents, self.n_channels)

        if not return_attention:
            return values

        # Per-transition off-diagonal attention mass: (1 - Tr(A)/N)
        # attn_probs: (B_flat, M, H, N, N)
        trace = torch.diagonal(attn_probs, dim1=-2, dim2=-1).sum(dim=-1)  # (B_flat, M, H)
        off_diag_mass = (self.n_agents - trace) / self.n_agents           # (B_flat, M, H)
        sample_attn = off_diag_mass.mean(dim=-1)                          # (B_flat, M)
        sample_attn = sample_attn.view(*orig_shape, self.n_channels)      # (*orig_shape, M)

        return values, sample_attn


class NativeAttentionCoordinator:
    """
    Directional Attention-Return Covariance Coordinator.
    - Replaces R^2 with Pearson Covariance: Cov(CrossAttention, Return).
    - Shuts down channels where cross-attention correlates with penalties/negative surplus.
    - Explicitly masks out known penalty channel IDs.
    """
    def __init__(
        self,
        n_channels,
        penalty_channel_ids=None,
        ema_momentum=0.95,
        synergy_scale=2.0,
    ):
        self.n_channels = n_channels
        self.penalty_channel_ids = penalty_channel_ids or []
        self.momentum = ema_momentum
        self.synergy_scale = synergy_scale
        self.scores_ema = None

    def update(self, sample_attn, target_returns, mask):
        """
        Args:
            sample_attn: (T, B, M) - Instantaneous cross-agent attention mass
            target_returns: (T, B, N, M) - Decomposed returns
            mask: (T, B) - Environment active mask (batch.filled)
        """
        device = sample_attn.device
        if self.scores_ema is None:
            self.scores_ema = torch.zeros(self.n_channels, device=device)

        # Team return per channel: (T, B, M)
        g = target_returns.mean(dim=2)
        a = sample_attn

        expanded_mask = mask.unsqueeze(-1).float()  # (T, B, 1)
        valid_count = expanded_mask.sum(dim=(0, 1)) + 1e-8

        # Masked sample means
        mean_a = (a * expanded_mask).sum(dim=(0, 1)) / valid_count  # (M,)
        mean_g = (g * expanded_mask).sum(dim=(0, 1)) / valid_count  # (M,)

        # Centered deviations
        da = (a - mean_a.view(1, 1, -1)) * expanded_mask
        dg = (g - mean_g.view(1, 1, -1)) * expanded_mask

        # Masked variances and covariance
        var_a = (da ** 2).sum(dim=(0, 1)) / valid_count
        var_g = (dg ** 2).sum(dim=(0, 1)) / valid_count
        cov_ag = (da * dg).sum(dim=(0, 1)) / valid_count

        # Pearson correlation: Cov(A, G) / (std(A) * std(G))
        std_product = torch.sqrt(var_a * var_g + 1e-8)
        corr = cov_ag / std_product  # (M,) bounded in [-1.0, 1.0]

        # Only positive synergy qualifies as coordination
        synergy = F.relu(corr) * self.synergy_scale
        instant_scores = torch.clamp(synergy, 0.0, 1.0)

        # Rule out zero-variance channels (no exploration signal yet)
        instant_scores = torch.where(var_g > 1e-5, instant_scores, torch.zeros_like(instant_scores))

        # Hard zero for explicitly marked penalty channels
        for p_idx in self.penalty_channel_ids:
            if p_idx < self.n_channels:
                instant_scores[p_idx] = 0.0

        # EMA smoothing across updates
        self.scores_ema = self.momentum * self.scores_ema + (1.0 - self.momentum) * instant_scores
        return self.scores_ema.detach()


class TransformerCriticOTMAPPONetwork(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg,
        actor,
        critic=None,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        if "cuda" in str(device) and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.gamma = cfg.gamma
        self.gae_lambda = getattr(cfg, "gae_lambda", 0.95)
        self.clip_param = getattr(cfg, "clip_param", 0.2)
        self.ppo_epochs = getattr(cfg, "ppo_epochs", 10)
        self.num_mini_batches = getattr(cfg, "num_mini_batches", 1)
        self.value_loss_coef = cfg.value_loss_coef
        self.entropy_coef = cfg.entropy_coef
        self.grad_clip = cfg.grad_clip
        self.use_clipped_value_loss = getattr(cfg, "use_clipped_value_loss", True)

        self.n_agents = len(obs_space)
        self.obs_dims = [flatdim(o) for o in obs_space]
        act_dims = [flatdim(a) for a in action_space]

        self.n_reward_channels = getattr(cfg, "n_reward_channels", 4)
        self.optimism_alpha = getattr(cfg, "optimism_alpha", 0.1)

        weights = getattr(cfg, "channel_weights", [1.0] * self.n_reward_channels)
        self.register_buffer(
            "channel_weights", torch.tensor(weights, dtype=torch.float32, device=self.device)
        )

        # Decentralized Actors
        self.use_rnn = actor.use_rnn
        if not actor.parameter_sharing:
            self.actor = MultiAgentIndependentNetwork(
                self.obs_dims, list(actor.layers), act_dims, actor.use_rnn, actor.use_orthogonal_init
            )
        else:
            self.actor = MultiAgentSharedNetwork(
                self.obs_dims, list(actor.layers), act_dims, actor.parameter_sharing, actor.use_rnn, actor.use_orthogonal_init
            )

        # Transformer Critic
        critic_d_model = getattr(cfg, "critic_d_model", 128)
        critic_n_heads = getattr(cfg, "critic_n_heads", 4)
        critic_d_ffn = getattr(cfg, "critic_d_ffn", 256)
        identity_bias = getattr(cfg, "critic_identity_bias", 1.5)

        self.critic = ChannelQueryAgentAttentionCritic(
            obs_dims=self.obs_dims,
            n_agents=self.n_agents,
            n_reward_channels=self.n_reward_channels,
            d_model=critic_d_model,
            n_heads=critic_n_heads,
            d_ffn=critic_d_ffn,
            identity_bias=identity_bias,
        )
        self.target_critic = copy.deepcopy(self.critic)

        # Covariance Coordinator (R^2 removed, Directional Synergy added)
        penalty_channel_ids = getattr(cfg, "penalty_channel_ids", [3])
        ema_momentum = getattr(cfg, "coord_ema_momentum", 0.95)
        synergy_scale = getattr(cfg, "synergy_scale", 2.0)

        self.coordinator = NativeAttentionCoordinator(
            n_channels=self.n_reward_channels,
            penalty_channel_ids=penalty_channel_ids,
            ema_momentum=ema_momentum,
            synergy_scale=synergy_scale,
        )

        self.standardise_returns = cfg.standardise_returns
        if self.standardise_returns:
            self.ret_ms = RunningMeanStd(
                shape=(self.n_agents, self.n_reward_channels), device=self.device
            )

        self.soft_update(1.0)
        self.to(self.device)

        optimizer = getattr(optim, cfg.optimizer)
        if isinstance(optimizer, str):
            optimizer = getattr(optim, optimizer)
        self.optimizer = optimizer(self.parameters(), lr=cfg.lr)
        self.target_update_interval_or_tau = cfg.target_update_interval_or_tau

        self.split_obs = _split_batch(self.obs_dims)
        self.split_act = _split_batch(self.n_agents * [1])

    def init_actor_hiddens(self, batch_size):
        return self.actor.init_hiddens(batch_size, self.device)

    def get_dist(self, action_logits, action_mask=None):
        if action_mask is not None:
            masked_logits = []
            for logits, mask in zip(action_logits, action_mask):
                masked_logits.append(logits * mask + (1 - mask) * -1e8)
            action_logits = masked_logits
        return MultiCategorical([Categorical(logits=logits) for logits in action_logits])

    def act(self, inputs, actor_hiddens, action_mask=None):
        inputs = [i.unsqueeze(0) for i in inputs]
        actor_logits, actor_hiddens = self.actor(inputs, actor_hiddens)
        actor_logits = [logits.squeeze(0) for logits in actor_logits]
        dist = self.get_dist(actor_logits, action_mask)
        actions = dist.sample()
        return torch.stack(actions, dim=0), actor_hiddens

    def get_value(self, inputs, target=False, return_attention=False):
        critic_net = self.target_critic if target else self.critic
        return critic_net(inputs, return_attention=return_attention)

    def evaluate_actions(
        self,
        inputs,
        action,
        actor_hiddens=None,
        action_mask=None,
    ):
        values = self.get_value(inputs, target=False, return_attention=False)
        actor_features, actor_hiddens = self.actor(inputs, actor_hiddens)
        dist = self.get_dist(actor_features, action_mask)
        action_log_probs = torch.cat(dist.log_probs(action), dim=-1)
        dist_entropy = torch.stack(dist.entropy(), dim=-1).sum(dim=-1)
        return values, action_log_probs, dist_entropy, actor_hiddens

    def soft_update(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)

    def update(self, batch, step):
        T, B = batch.rewards.shape[:2]
        batch_dones = batch.dones.float().unsqueeze(-1).repeat(1, 1, self.n_agents)

        # > Phase 1: Values, GAE, and Per-Step Attention
        with torch.no_grad():
            split_obss = self.split_obs(batch.obss)
            all_values, sample_attn = self.get_value(
                split_obss, target=False, return_attention=True
            )

            v_preds = all_values[:-1]
            next_v_preds = all_values[1:]
            step_attn = sample_attn[:-1]  # (T, B, M)

            if self.standardise_returns:
                v_preds_raw = v_preds * torch.sqrt(self.ret_ms.var + 1e-8) + self.ret_ms.mean
                next_v_preds_raw = next_v_preds * torch.sqrt(self.ret_ms.var + 1e-8) + self.ret_ms.mean
            else:
                v_preds_raw = v_preds
                next_v_preds_raw = next_v_preds

            raw_advantages, raw_returns = compute_multichannel_gae(
                batch.rewards,
                batch_dones[:-1],
                v_preds_raw,
                next_v_preds_raw,
                self.gamma,
                self.gae_lambda,
            )

            if self.standardise_returns:
                self.ret_ms.update(raw_returns)
                target_returns = (raw_returns - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var + 1e-8)
            else:
                target_returns = raw_returns

            action_mask = (
                rearrange(batch.action_masks[:-1], "E B N A -> N E B A")
                if batch.action_masks is not None
                else None
            )
            if hasattr(batch, "action_log_probs") and batch.action_log_probs is not None:
                old_action_log_probs = (
                    batch.action_log_probs[:-1]
                    if batch.action_log_probs.shape[0] == batch.obss.shape[0]
                    else batch.action_log_probs
                )
            else:
                _, old_action_log_probs, _, _ = self.evaluate_actions(
                    self.split_obs(batch.obss[:-1]),
                    self.split_act(batch.actions),
                    action_mask=action_mask,
                )

        # > Phase 2: Directional Covariance Optimism (Fix 2)
        coord_scores = self.coordinator.update(
            sample_attn=step_attn,
            target_returns=target_returns,
            mask=batch.filled,
        )

        alpha = 1.0 - coord_scores * (1.0 - self.optimism_alpha)
        alpha_broadcast = alpha.view(1, 1, 1, -1)

        transformed_advantages = torch.where(
            raw_advantages < 0.0,
            alpha_broadcast * raw_advantages,
            raw_advantages,
        )

        total_advantage = (transformed_advantages * self.channel_weights).sum(dim=-1)
        active_mask = batch.filled.unsqueeze(-1).expand_as(total_advantage)
        adv_mean = (total_advantage * active_mask).sum() / (active_mask.sum() + 1e-8)
        adv_std = torch.sqrt(
            (((total_advantage - adv_mean) ** 2) * active_mask).sum() / (active_mask.sum() + 1e-8) + 1e-8
        )
        norm_total_advantage = (total_advantage - adv_mean) / (adv_std + 1e-8)

        # > Phase 3: PPO Mini-Batch Update Loop
        mini_batch_size = max(1, B // self.num_mini_batches)
        epoch_actor_losses, epoch_value_losses = [], []
        epoch_entropies, epoch_approx_kls, epoch_clip_fracs = [], [], []

        for _ in range(self.ppo_epochs):
            env_perm = torch.randperm(B)

            for start_idx in range(0, B, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, B)
                batch_indices = env_perm[start_idx:end_idx]

                mb_obss = batch.obss[:-1, batch_indices]
                mb_actions = batch.actions[:, batch_indices]
                mb_filled = batch.filled[:, batch_indices]
                mb_old_log_probs = old_action_log_probs[:, batch_indices]
                mb_adv = norm_total_advantage[:, batch_indices]
                mb_target_ret = target_returns[:, batch_indices]
                mb_v_preds_old = v_preds[:, batch_indices]

                mb_action_masks = (
                    rearrange(batch.action_masks[:-1, batch_indices], "E B N A -> N E B A")
                    if batch.action_masks is not None
                    else None
                )

                new_values, new_log_probs, entropy, _ = self.evaluate_actions(
                    self.split_obs(mb_obss),
                    self.split_act(mb_actions),
                    action_mask=mb_action_masks,
                )

                ratio = torch.exp(new_log_probs - mb_old_log_probs.detach())
                surr1 = ratio * mb_adv.detach()
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_adv.detach()
                actor_loss = -torch.min(surr1, surr2).sum(dim=-1) - self.entropy_coef * entropy
                actor_loss = (actor_loss * mb_filled).sum() / (mb_filled.sum() + 1e-8)

                v_target = mb_target_ret.detach()
                if self.use_clipped_value_loss:
                    v_old = mb_v_preds_old.detach()
                    v_pred_clipped = v_old + torch.clamp(
                        new_values - v_old, -self.clip_param, self.clip_param
                    )
                    vf_loss1 = (new_values - v_target).pow(2)
                    vf_loss2 = (v_pred_clipped - v_target).pow(2)
                    channel_vf_loss = torch.max(vf_loss1, vf_loss2)
                else:
                    channel_vf_loss = (new_values - v_target).pow(2)

                mb_mask = mb_filled.unsqueeze(-1).unsqueeze(-1)
                total_value_loss = (
                    (channel_vf_loss * mb_mask).sum(dim=(0, 1, 2)) / (mb_mask.sum() + 1e-8)
                ).sum()

                loss = actor_loss + self.value_loss_coef * total_value_loss
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
                    epoch_actor_losses.append(actor_loss.item())
                    epoch_value_losses.append(total_value_loss.item())
                    epoch_entropies.append(((entropy * mb_filled).sum() / (mb_filled.sum() + 1e-8)).item())
                    epoch_approx_kls.append(approx_kl.item())
                    epoch_clip_fracs.append(clip_frac.item())

        # > Phase 4: Soft Updates & Logging
        if (
            self.target_update_interval_or_tau > 1.0
            and step % self.target_update_interval_or_tau == 0
        ):
            self.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            self.soft_update(self.target_update_interval_or_tau)

        top_coop_idx = torch.argmax(coord_scores).item()
        coop_adv_raw = raw_advantages[..., top_coop_idx]
        neg_coop_mask = coop_adv_raw < 0.0
        active_agents = batch.filled.unsqueeze(-1).expand_as(coop_adv_raw)
        neg_coop_ratio = (neg_coop_mask.float() * active_agents).sum() / (
            active_agents.sum() + 1e-8
        )

        return {
            "loss": sum(epoch_actor_losses) / len(epoch_actor_losses)
            + self.value_loss_coef * (sum(epoch_value_losses) / len(epoch_value_losses)),
            "actor_loss": sum(epoch_actor_losses) / len(epoch_actor_losses),
            "value_loss": sum(epoch_value_losses) / len(epoch_value_losses),
            "entropy": sum(epoch_entropies) / len(epoch_entropies),
            "approx_kl": sum(epoch_approx_kls) / len(epoch_approx_kls),
            "clip_frac": sum(epoch_clip_fracs) / len(epoch_clip_fracs),
            "coop_neg_adv_ratio": float(neg_coop_ratio.item()),
            **{
                f"coordination_score_ch{m}": float(coord_scores[m].item())
                for m in range(self.n_reward_channels)
            },
        }