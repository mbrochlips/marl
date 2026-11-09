import copy
from einops import rearrange
from gymnasium.spaces import flatdim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import optim

from marlbase.utils.models import MultiAgentIndependentNetwork, MultiAgentSharedNetwork
from marlbase.utils.model_critics import ChannelQueryTransformerCritic, ChannelQueryTransformerCriticTeamResidualReadout
from marlbase.utils.utils import MultiCategorical
from marlbase.utils.standardise_stream import RunningMeanStd
from marlbase.utils.detectors import CoordinationDetector, DummyDetector, RiskDetector


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

class TransformerCriticOTMAPPONetwork(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg,
        actor,
        critic,
        device,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.cfg = cfg
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

        self.n_agents = len(obs_space)
        obs_dims = [flatdim(o) for o in obs_space]
        act_dims = [flatdim(a) for a in action_space]

        # Basis capacity
        self.max_channels = getattr(cfg, "max_channels", 32)

        # Placeholders for late binding in set_active_channels
        self.n_reward_channels = None
        self.channel_names = None
        self.active_channel_indices = None
        self.coord_detector = None
        self.ret_ms = None
        self.standardise_returns = cfg.standardise_returns

        # Persistent coordination scores across all basis slots
        self.register_buffer("basis_coord_scores", torch.zeros(self.max_channels))

        # Actor initialization
        self.use_rnn = actor.use_rnn
        if not actor.parameter_sharing:
            self.actor = MultiAgentIndependentNetwork(
                obs_dims, list(actor.layers), act_dims, actor.use_rnn, actor.use_orthogonal_init
            )
        else:
            self.actor = MultiAgentSharedNetwork(
                obs_dims, list(actor.layers), act_dims, actor.parameter_sharing, actor.use_rnn, actor.use_orthogonal_init
            )

        # Critic initialization (sized to max_channels)
        self.centralised_critic = critic.centralised

        self.detector_cls = (
            CoordinationDetector
            if critic.optimistic_targets
            else DummyDetector
        )

        critic_cls = (
            ChannelQueryTransformerCriticTeamResidualReadout
            if critic.team_residual_readout
            else ChannelQueryTransformerCritic
        )

        self.critic = critic_cls(
            n_agents=self.n_agents,
            max_channels=self.max_channels,
            max_agents=getattr(cfg, "max_agents", 3),
            max_entities=getattr(cfg, "max_entities", 11),
            d_model=getattr(cfg, "critic_d_model", 128),
            n_heads=getattr(cfg, "critic_n_heads", 4),
            d_ffn=getattr(cfg, "critic_d_ffn", 256),
            n_state_tokens=getattr(cfg, "n_state_tokens", 4),
        )

        self.target_critic = copy.deepcopy(self.critic)

        # Freeze target critic parameters so they are never updated by the optimizer
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.soft_update(1.0)
        self.to(device)

        # Optimize trainable actor and critic parameters (excludes target_critic)
        self.trainable_params = list(self.actor.parameters()) + list(self.critic.parameters())
        optimizer_cls = getattr(optim, cfg.optimizer) if isinstance(cfg.optimizer, str) else cfg.optimizer
        self.optimizer = optimizer_cls(self.trainable_params, lr=cfg.lr)
        self.target_update_interval_or_tau = cfg.target_update_interval_or_tau

        self.split_obs = _split_batch([flatdim(s) for s in obs_space])
        self.split_act = _split_batch(self.n_agents * [1])

    def set_active_channels(
        self,
        active_indices: torch.Tensor,
        channel_names=None,
        **kwargs,
    ):
        """Binds active basis slots, initializes the detector, and configures return standardisation."""
        self.active_channel_indices = active_indices.to(self.device)
        self.n_reward_channels = len(active_indices)

        if channel_names is not None:
            self.channel_names = list(channel_names)
        else:
            self.channel_names = [f"channel_{i}" for i in range(self.n_reward_channels)]

        # Bind basis queries in critics
        self.critic.set_active_channels(self.active_channel_indices)
        self.target_critic.set_active_channels(self.active_channel_indices)

        # Initialize detector for the active channels
        conf_lambda = getattr(self.cfg, "confidence_lambda", 0.35)

        self.coord_detector = self.detector_cls(
            channel_names=self.channel_names,
            n_agents=self.n_agents,
            confidence_lambda=conf_lambda,
        ).to(self.device)

        if self.standardise_returns:
            self.ret_ms = RunningMeanStd(
                shape=(self.n_agents, self.n_reward_channels), device=self.device
            )

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
        log_probs = torch.cat(dist.log_probs(actions), dim=-1)
        return torch.stack(actions, dim=0), actor_hiddens, log_probs

    def get_value(self, inputs, target=False):
        state = torch.cat(inputs, dim=-1) if self.centralised_critic else inputs[0]
        critic_net = self.target_critic if target else self.critic
        return critic_net(state)

    def evaluate_actions(self, inputs, action, actor_hiddens=None, action_mask=None, state=None):
        if state is None:
            state = inputs
        values = self.get_value(state)
        actor_features, actor_hiddens = self.actor(inputs, actor_hiddens)
        dist = self.get_dist(actor_features, action_mask)
        action_log_probs = torch.cat(dist.log_probs(action), dim=-1)
        dist_entropy = torch.stack(dist.entropy(), dim=-1).sum(dim=-1)
        return values, action_log_probs, dist_entropy, actor_hiddens

    def soft_update(self, tau):
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)

    def update(self, batch, step):
        T, B = batch.rewards.shape[:2]
        
        # Ensure masks and dones are floats to avoid int/bool dtype promotion
        filled_float = batch.filled.float()
        batch_dones = batch.dones.float().unsqueeze(-1).repeat(1, 1, self.n_agents)
        dones_slice = batch_dones[1:] if batch_dones.shape[0] == T + 1 else batch_dones

        # 1. Target Values & GAE
        with torch.no_grad():
            all_values = self.get_value(self.split_obs(batch.obss), target=False)
            v_preds = all_values[:-1]
            next_v_preds = all_values[1:]

            if self.standardise_returns:
                std = torch.sqrt(self.ret_ms.var + 1e-8)
                v_preds_raw = v_preds * std + self.ret_ms.mean
                next_v_preds_raw = next_v_preds * std + self.ret_ms.mean
            else:
                v_preds_raw, next_v_preds_raw = v_preds, next_v_preds

            raw_advantages, raw_returns = compute_multichannel_gae(
                batch.rewards,
                dones_slice,
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

        # 2. Targeted Coordination Optimism
        coord_scores = self.coord_detector.update(batch.rewards, filled_float)
        self.basis_coord_scores[self.active_channel_indices] = coord_scores.detach()

        alpha = 1.0 - coord_scores * (1.0 - self.optimism_alpha)
        alpha_broadcast = alpha.view(1, 1, 1, -1)

        transformed_advantages = torch.where(
            raw_advantages < 0.0,
            alpha_broadcast * raw_advantages,
            raw_advantages,
        )
        total_advantage = transformed_advantages.sum(dim=-1)

        active_mask = filled_float.unsqueeze(-1).expand_as(total_advantage)
        adv_denom = active_mask.sum() + 1e-8
        adv_mean = (total_advantage * active_mask).sum() / adv_denom
        adv_std = torch.sqrt(
            (((total_advantage - adv_mean) ** 2) * active_mask).sum() / adv_denom + 1e-8
        )
        norm_total_advantage = (total_advantage - adv_mean) / (adv_std + 1e-8)

        # 3. PPO Mini-Batch Update
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
                mb_filled = filled_float[:, batch_indices]
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

                # Actor Loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs.detach())
                surr1 = ratio * mb_adv.detach()
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_adv.detach()
                actor_loss = -torch.min(surr1, surr2).sum(dim=-1) - self.entropy_coef * entropy
                actor_loss = (actor_loss * mb_filled).sum() / (mb_filled.sum() + 1e-8)

                # Critic Loss
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

                # Normalize by total agent-steps (mb_mask.sum() * n_agents)
                mb_mask = mb_filled.unsqueeze(-1).unsqueeze(-1)
                total_agent_steps = mb_mask.sum() * self.n_agents + 1e-8
                per_channel_loss = (channel_vf_loss * mb_mask).sum(dim=(0, 1, 2)) / total_agent_steps
                total_value_loss = per_channel_loss.mean()

                # Optimization step
                loss = actor_loss + self.value_loss_coef * total_value_loss
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, self.grad_clip)
                self.optimizer.step()

                # Logging metrics
                with torch.no_grad():
                    ratio_diff = ratio - 1.0
                    approx_kl = (ratio_diff - torch.log(ratio)).mean()
                    clip_frac = (ratio_diff.abs() > self.clip_param).float().mean()
                    mb_entropy = (entropy * mb_filled).sum() / (mb_filled.sum() + 1e-8)

                    epoch_actor_losses.append(actor_loss.detach())
                    epoch_value_losses.append(total_value_loss.detach())
                    epoch_entropies.append(mb_entropy.detach())
                    epoch_approx_kls.append(approx_kl.detach())
                    epoch_clip_fracs.append(clip_frac.detach())

        # Target Critic updates
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
                f"coordination_score_{self.channel_names[m] if self.channel_names and m < len(self.channel_names) else f'ch{m}'}": coord_scores_list[m]
                for m in range(self.n_reward_channels)
            },
        }