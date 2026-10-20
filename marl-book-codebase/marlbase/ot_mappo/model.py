import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import optim
from einops import rearrange
from gymnasium.spaces import flatdim

from marlbase.utils.models import MultiAgentIndependentNetwork, MultiAgentSharedNetwork
from marlbase.utils.utils import MultiCategorical


def compute_multichannel_gae(rewards, dones, values, next_values, gamma, gae_lambda):
    """
    Vectorized Multi-Channel GAE across (T, B, N, M).
    """
    T, B, N, M = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, N, M, device=rewards.device)
    non_terminals = 1.0 - dones.unsqueeze(-1).float()  # (T, B, N, 1)

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * non_terminals[t] - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminals[t] * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class StreamlinedDecompOptiMAPPO(nn.Module):
    """
    Streamlined Decomposed OptiMAPPO:
      - Single unified critic outputting (N_agents, N_channels).
      - Dead target-critic networks removed.
      - Fully vectorized multi-channel PPO value loss.
      - Targeted Optimism on specified cooperative channels.
    """

    def __init__(self, obs_space, action_space, cfg, actor, critic, device):
        super().__init__()
        self.device = device
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
        obs_dims = [flatdim(o) for o in obs_space]
        act_dims = [flatdim(a) for a in action_space]

        # Multi-Channel & Optimism Config
        self.n_reward_channels = getattr(cfg, "n_reward_channels", 3)
        self.coop_channel_idx = getattr(cfg, "coop_channel_idx", 2)
        self.optimism_alpha = getattr(cfg, "optimism_alpha", 0.1)
        self.register_buffer(
            "channel_weights",
            torch.tensor(getattr(cfg, "channel_weights", [1.0] * self.n_reward_channels), dtype=torch.float32),
        )

        # 1. Decentralized Actors
        network_cls = MultiAgentSharedNetwork if actor.parameter_sharing else MultiAgentIndependentNetwork
        self.actor = network_cls(
            obs_dims, list(actor.layers), act_dims, actor.use_rnn, actor.use_orthogonal_init
        )

        # 2. Unified Centralized Multi-Channel Critic (Outputs M values per agent)
        critic_cls = MultiAgentSharedNetwork if critic.parameter_sharing else MultiAgentIndependentNetwork
        self.centralised_critic = critic.centralised
        critic_obs_shape = self.n_agents * [sum(obs_dims)] if critic.centralised else obs_dims

        # Each agent outputs `n_reward_channels` values directly
        self.critic = critic_cls(
            critic_obs_shape,
            list(critic.layers),
            [self.n_reward_channels] * self.n_agents,
            critic.use_rnn,
            critic.use_orthogonal_init,
        )

        self.to(device)
        optimizer_cls = getattr(optim, cfg.optimizer) if hasattr(optim, cfg.optimizer) else optim.Adam
        self.optimizer = optimizer_cls(self.parameters(), lr=cfg.lr)

        self._obs_splits = [flatdim(s) for s in obs_space]
        self._act_splits = self.n_agents * [1]

    def _split_obs(self, batch):
        return torch.split(batch, self._obs_splits, dim=-1)

    def _split_act(self, batch):
        return torch.split(batch, self._act_splits, dim=-1)

    def get_dist(self, action_logits, action_mask=None):
        if action_mask is not None:
            action_logits = [
                logits * mask + (1.0 - mask) * -1e8
                for logits, mask in zip(action_logits, action_mask)
            ]
        return MultiCategorical([Categorical(logits=l) for l in action_logits])

    def get_value(self, inputs, critic_hiddens=None):
        if self.centralised_critic:
            inputs = self.n_agents * [torch.cat(inputs, dim=-1)]
        values, next_hiddens = self.critic(inputs, critic_hiddens)
        # Reshape to (..., n_agents, n_reward_channels)
        values = torch.stack(values, dim=-2)
        return values, next_hiddens

    def evaluate_actions(self, inputs, action, critic_hiddens=None, actor_hiddens=None, action_mask=None, state=None):
        if state is None:
            state = inputs
        values, critic_hiddens = self.get_value(state, critic_hiddens)
        actor_features, actor_hiddens = self.actor(inputs, actor_hiddens)
        dist = self.get_dist(actor_features, action_mask)
        action_log_probs = torch.cat(dist.log_probs(action), dim=-1)
        dist_entropy = torch.stack(dist.entropy(), dim=-1).sum(dim=-1)
        return values, action_log_probs, dist_entropy, critic_hiddens, actor_hiddens

    def update(self, batch, step):
        T, B = batch.rewards.shape[:2]
        batch_dones = batch.dones.float().unsqueeze(-1).repeat(1, 1, self.n_agents)

        # -------------------------------------------------------------
        # Phase 1: Compute Values & Multi-Channel GAE
        # -------------------------------------------------------------
        with torch.no_grad():
            all_values, _ = self.get_value(self._split_obs(batch.obss))
            v_preds = all_values[:-1]       # (T, B, N, M)
            next_v_preds = all_values[1:]   # (T, B, N, M)

            raw_advantages, returns = compute_multichannel_gae(
                batch.rewards,
                batch_dones[:-1],
                v_preds,
                next_v_preds,
                self.gamma,
                self.gae_lambda,
            )

            # Retrieve old log probabilities
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
                _, old_action_log_probs, _, _, _ = self.evaluate_actions(
                    self._split_obs(batch.obss[:-1]),
                    self._split_act(batch.actions),
                    action_mask=action_mask,
                )

        # -------------------------------------------------------------
        # Phase 2: Targeted Optimism & Fusion
        # -------------------------------------------------------------
        transformed_advantages = raw_advantages.clone()
        if self.optimism_alpha < 1.0:
            transformed_advantages[..., self.coop_channel_idx] = F.leaky_relu(
                raw_advantages[..., self.coop_channel_idx],
                negative_slope=self.optimism_alpha,
            )

        # Weighted combination across channels -> (T, B, N)
        total_advantage = (transformed_advantages * self.channel_weights).sum(dim=-1)

        # Normalize advantage across batch
        active_mask = batch.filled.unsqueeze(-1).expand_as(total_advantage)
        adv_mean = (total_advantage * active_mask).sum() / (active_mask.sum() + 1e-8)
        adv_std = torch.sqrt(
            (((total_advantage - adv_mean) ** 2) * active_mask).sum() / (active_mask.sum() + 1e-8) + 1e-8
        )
        norm_total_advantage = (total_advantage - adv_mean) / (adv_std + 1e-8)

        # -------------------------------------------------------------
        # Phase 3: Mini-Batch PPO Updates
        # -------------------------------------------------------------
        mini_batch_size = max(1, B // self.num_mini_batches)
        epoch_actor_losses, epoch_value_losses = [], []

        for _ in range(self.ppo_epochs):
            env_perm = torch.randperm(B)
            for start_idx in range(0, B, mini_batch_size):
                batch_indices = env_perm[start_idx : start_idx + mini_batch_size]

                mb_filled = batch.filled[:, batch_indices]
                mb_adv = norm_total_advantage[:, batch_indices].detach()
                mb_target_ret = returns[:, batch_indices].detach()
                mb_v_preds_old = v_preds[:, batch_indices].detach()
                mb_old_log_probs = old_action_log_probs[:, batch_indices].detach()

                mb_action_masks = (
                    rearrange(batch.action_masks[:-1, batch_indices], "E B N A -> N E B A")
                    if batch.action_masks is not None
                    else None
                )

                new_values, new_log_probs, entropy, _, _ = self.evaluate_actions(
                    self._split_obs(batch.obss[:-1, batch_indices]),
                    self._split_act(batch.actions[:, batch_indices]),
                    action_mask=mb_action_masks,
                )

                # 1. Clipped Policy Objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_adv
                actor_loss = -torch.min(surr1, surr2).sum(dim=-1) - self.entropy_coef * entropy
                actor_loss = (actor_loss * mb_filled).sum() / (mb_filled.sum() + 1e-8)

                # 2. Vectorized Multi-Channel Critic Objective
                vf_loss1 = (new_values - mb_target_ret).pow(2)
                if self.use_clipped_value_loss:
                    v_pred_clipped = mb_v_preds_old + torch.clamp(
                        new_values - mb_v_preds_old, -self.clip_param, self.clip_param
                    )
                    vf_loss2 = (v_pred_clipped - mb_target_ret).pow(2)
                    vf_loss = torch.max(vf_loss1, vf_loss2)
                else:
                    vf_loss = vf_loss1

                total_value_loss = (
                    vf_loss.sum(dim=-1).sum(dim=-1) * mb_filled
                ).sum() / (mb_filled.sum() + 1e-8)

                # Joint Backward Step
                loss = actor_loss + self.value_loss_coef * total_value_loss
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                self.optimizer.step()

                epoch_actor_losses.append(actor_loss.item())
                epoch_value_losses.append(total_value_loss.item())

        return {
            "loss": sum(epoch_actor_losses) / len(epoch_actor_losses)
            + self.value_loss_coef * (sum(epoch_value_losses) / len(epoch_value_losses)),
            "actor_loss": sum(epoch_actor_losses) / len(epoch_actor_losses),
            "value_loss": sum(epoch_value_losses) / len(epoch_value_losses),
        }