# Citations & Foundations:
# - Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022
# - Zhao et al., "Optimistic Multi-Agent Policy Gradient", ICML 2024
# - https://github.com/marl-book/codebase

from collections import defaultdict
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
from marlbase.utils.interact_score import compute_coordination_score

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
    """
    Vectorized Multi-Channel Generalized Advantage Estimation (GAE).
    Computes advantages and targets across time, environments, agents, and reward channels.
    """
    T, B, N, M = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, N, M, device=rewards.device)

    # dones shape: (T, B, N) -> expand to (T, B, N, 1) for broadcasting across M channels
    non_terminals = 1.0 - dones.unsqueeze(-1).float()

    for t in reversed(range(T)):
        next_val = next_values[t]
        delta = rewards[t] + gamma * next_val * non_terminals[t] - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminals[t] * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class DecompOptiMAPPONetwork(nn.Module):
    """
    Decomposed Multi-Agent PPO with Targeted Optimism (Decomp-OptiMAPPO).
    
    Key Features:
      - N x M Decoupled Critics: M independent centralized multi-agent critics.
      - Multi-Channel GAE: Evaluates temporal returns independently per reward channel.
      - Targeted Optimism: Channel-selective LeakyReLU advantage softening for 
        cooperative interactions (OptiMAPPO), preserving risk-neutrality for 
        stochastic/solo/penalty channels (Standard MAPPO).
      - Multi-Epoch PPO Clipped Surrogate: Mini-batch updates with bounded ratio clipping.
      - Independent Channel RunningMeanStd Normalization.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        cfg,
        actor,
        critic,
        device,
    ):
        super(DecompOptiMAPPONetwork, self).__init__()
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

        # -------------------------------------------------------------
        # 1. Decomposed Channels & Coordination Latch Parameters
        # -------------------------------------------------------------
        self.n_reward_channels = getattr(cfg, "n_reward_channels", 3)
        self.optimism_alpha = getattr(cfg, "optimism_alpha", 0.1)
        self.channel_weights = getattr(cfg, "channel_weights", [1.0] * self.n_reward_channels)

        # Threshold to count as a genuine positive detection (filters epsilon float noise)
        self.coordination_threshold = getattr(cfg, "coordination_threshold", 0.01)
        self.required_hits = getattr(cfg, "coordination_required_hits", 2)

        # Coordination score EMA buffer (for logging)
        self.register_buffer(
            "coordination_ema", torch.zeros(self.n_reward_channels, device=device)
        )
        self.coordination_ema_decay = getattr(cfg, "coordination_ema_decay", 0.95)

        # Cumulative hit counter and permanent commitment latch
        self.register_buffer(
            "coordination_hits", torch.zeros(self.n_reward_channels, dtype=torch.int32, device=device)
        )
        self.register_buffer(
            "is_committed_optimism", torch.zeros(self.n_reward_channels, dtype=torch.bool, device=device)
        )

        # -------------------------------------------------------------
        # 2. Decentralized Actors
        # -------------------------------------------------------------
        self.use_rnn = actor.use_rnn
        if not actor.parameter_sharing:
            self.actor = MultiAgentIndependentNetwork(
                obs_dims,
                list(actor.layers),
                act_dims,
                actor.use_rnn,
                actor.use_orthogonal_init,
            )
        else:
            self.actor = MultiAgentSharedNetwork(
                obs_dims,
                list(actor.layers),
                act_dims,
                actor.parameter_sharing,
                actor.use_rnn,
                actor.use_orthogonal_init,
            )

        # -------------------------------------------------------------
        # 3. M x N Independent Critics
        # -------------------------------------------------------------
        self.centralised_critic = critic.centralised
        critic_obs_shape = (
            self.n_agents * [sum(obs_dims)] if critic.centralised else obs_dims
        )

        self.critics = nn.ModuleList()
        self.target_critics = nn.ModuleList()

        for _ in range(self.n_reward_channels):
            if not critic.parameter_sharing:
                c = MultiAgentIndependentNetwork(
                    critic_obs_shape,
                    list(critic.layers),
                    [1] * self.n_agents,
                    critic.use_rnn,
                    critic.use_orthogonal_init,
                )
                tc = MultiAgentIndependentNetwork(
                    critic_obs_shape,
                    list(critic.layers),
                    [1] * self.n_agents,
                    critic.use_rnn,
                    critic.use_orthogonal_init,
                )
            else:
                c = MultiAgentSharedNetwork(
                    critic_obs_shape,
                    list(critic.layers),
                    [1] * self.n_agents,
                    critic.parameter_sharing,
                    critic.use_rnn,
                    critic.use_orthogonal_init,
                )
                tc = MultiAgentSharedNetwork(
                    critic_obs_shape,
                    list(critic.layers),
                    [1] * self.n_agents,
                    critic.parameter_sharing,
                    critic.use_rnn,
                    critic.use_orthogonal_init,
                )
            self.critics.append(c)
            self.target_critics.append(tc)

        # -------------------------------------------------------------
        # 4. Independent Normalization & Optimizer
        # -------------------------------------------------------------
        self.standardise_returns = cfg.standardise_returns
        if self.standardise_returns:
            self.ret_ms = RunningMeanStd(
                shape=(self.n_agents, self.n_reward_channels), device=device
            )

        self.soft_update(1.0)
        self.to(device)

        optimizer = getattr(optim, cfg.optimizer)
        if isinstance(optimizer, str):
            optimizer = getattr(optim, optimizer)
        self.optimizer = optimizer(self.parameters(), lr=cfg.lr)
        self.target_update_interval_or_tau = cfg.target_update_interval_or_tau

        self.split_obs = _split_batch([flatdim(s) for s in obs_space])
        self.split_act = _split_batch(self.n_agents * [1])

    # -------------------------------------------------------------
    # State & Execution Handlers
    # -------------------------------------------------------------

    def init_critic_hiddens(self, batch_size, target=False):
        critic_list = self.target_critics if target else self.critics
        return [c.init_hiddens(batch_size, self.device) for c in critic_list]

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

    def get_value(self, inputs, critic_hiddens=None, target=False):
        if self.centralised_critic:
            inputs = self.n_agents * [torch.cat(inputs, dim=-1)]

        critic_list = self.target_critics if target else self.critics
        channel_values = []
        next_hiddens = []

        for m in range(self.n_reward_channels):
            h_m = critic_hiddens[m] if critic_hiddens is not None else None
            val_m, next_h_m = critic_list[m](inputs, h_m)
            val_m = torch.cat(val_m, dim=-1)
            channel_values.append(val_m)
            next_hiddens.append(next_h_m)

        values = torch.stack(channel_values, dim=-1)  # (..., n_agents, n_reward_channels)
        return values, next_hiddens

    def evaluate_actions(
        self,
        inputs,
        action,
        critic_hiddens=None,
        actor_hiddens=None,
        action_mask=None,
        state=None,
    ):
        if state is None:
            state = inputs

        values, critic_hiddens = self.get_value(state, critic_hiddens)
        actor_features, actor_hiddens = self.actor(inputs, actor_hiddens)
        dist = self.get_dist(actor_features, action_mask)
        action_log_probs = torch.cat(dist.log_probs(action), dim=-1)
        dist_entropy = torch.stack(dist.entropy(), dim=-1).sum(dim=-1)

        return values, action_log_probs, dist_entropy, critic_hiddens, actor_hiddens

    def soft_update(self, t):
        for source, target in zip(self.critics, self.target_critics):
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

    # -------------------------------------------------------------
    # The Decomp-OptiMAPPO Update Pipeline
    # -------------------------------------------------------------
    def update(self, batch, step):
        T, B = batch.rewards.shape[:2]
        batch_dones = batch.dones.float().unsqueeze(-1).repeat(1, 1, self.n_agents)

        # ---------------------------------------------------------
        # Phase 1: Compute Target Values & Multi-Channel GAE
        # ---------------------------------------------------------
        with torch.no_grad():
            # Query values for all states 0..T
            all_values, _ = self.get_value(
                self.split_obs(batch.obss), critic_hiddens=None, target=False
            )
            v_preds = all_values[:-1]       # (T, B, n_agents, n_reward_channels)
            next_v_preds = all_values[1:]   # (T, B, n_agents, n_reward_channels)

            # De-standardize for GAE computation in raw reward scale
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

            # Update RunningMeanStd with raw returns and normalize returns for critic targets
            if self.standardise_returns:
                self.ret_ms.update(raw_returns)
                target_returns = (raw_returns - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var + 1e-8)
            else:
                target_returns = raw_returns

            # Retrieve or compute old action log probabilities for PPO ratio
            action_mask = (
                rearrange(batch.action_masks[:-1], "E B N A -> N E B A")
                if batch.action_masks is not None
                else None
            )
            if hasattr(batch, "action_log_probs") and batch.action_log_probs is not None:
                if batch.action_log_probs.shape[0] == batch.obss.shape[0]:  # If length T + 1
                    old_action_log_probs = batch.action_log_probs[:-1]
                else:  # Already length T
                    old_action_log_probs = batch.action_log_probs
            else:
                _, old_action_log_probs, _, _, _ = self.evaluate_actions(
                    self.split_obs(batch.obss[:-1]),
                    self.split_act(batch.actions),
                    action_mask=action_mask,
                )
        # ---------------------------------------------------------
        # Phase 2: Coordination Detection & Permanent Optimism Latch
        # ---------------------------------------------------------
        # 1. Compute pessimistic coordination score on the fresh rollout batch
        current_scores = compute_coordination_score(
            raw_advantages=raw_advantages,
            rewards=batch.rewards,
            filled_mask=batch.filled,
        )

        # Update EMA for logging
        if step == 0:
            self.coordination_ema.copy_(current_scores)
        else:
            self.coordination_ema.copy_(
                self.coordination_ema_decay * self.coordination_ema
                + (1.0 - self.coordination_ema_decay) * current_scores
            )

        # 2. Check for positive coordination detections
        # Increment hits for channels exceeding the positive threshold
        positive_hits = current_scores > self.coordination_threshold
        self.coordination_hits += positive_hits.int()

        # 3. Permanent Commitment Latch: Lock to optimism if hit >= 2
        # The '|=' ensures once committed, it stays committed forever
        newly_committed = self.coordination_hits >= self.required_hits
        self.is_committed_optimism.copy_(self.is_committed_optimism | newly_committed)

        # 4. Apply Targeted Optimism based on commitment status
        transformed_advantages = raw_advantages.clone()
        for m in range(self.n_reward_channels):
            if self.is_committed_optimism[m]:
                # Committed channel: full optimism (alpha = 0.1)
                transformed_advantages[..., m] = F.leaky_relu(
                    raw_advantages[..., m], negative_slope=self.optimism_alpha
                )
            else:
                # Uncommitted channel (solo/penalty/splitting): risk-neutral MAPPO (alpha = 1.0)
                pass

        # 5. Weight and aggregate across channels
        channel_weights = torch.tensor(
            self.channel_weights, device=self.device, dtype=torch.float32
        )
        total_advantage = (transformed_advantages * channel_weights).sum(dim=-1)

        # 6. Standardize total advantage across batch dimension
        active_mask = batch.filled.unsqueeze(-1).expand_as(total_advantage)
        adv_mean = (total_advantage * active_mask).sum() / (active_mask.sum() + 1e-8)
        adv_std = torch.sqrt(
            (((total_advantage - adv_mean) ** 2) * active_mask).sum() / (active_mask.sum() + 1e-8) + 1e-8
        )
        norm_total_advantage = (total_advantage - adv_mean) / (adv_std + 1e-8)
        # ---------------------------------------------------------
        # Phase 3: PPO Mini-Batch Update Loop
        # ---------------------------------------------------------
        mini_batch_size = max(1, B // self.num_mini_batches)
        
        epoch_actor_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        epoch_approx_kls = []
        epoch_clip_fracs = []

        for _ in range(self.ppo_epochs):
            # Sample environment indices along batch dimension B to keep RNN time unrolls intact
            env_perm = torch.randperm(B)

            for start_idx in range(0, B, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, B)
                batch_indices = env_perm[start_idx:end_idx]

                # Slice mini-batch data
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

                # Re-evaluate policies and critics for current mini-batch parameters
                new_values, new_log_probs, entropy, _, _ = self.evaluate_actions(
                    self.split_obs(mb_obss),
                    self.split_act(mb_actions),
                    action_mask=mb_action_masks,
                )

                # 1. Actor PPO Clipped Loss
                # ratio = pi(a|s) / pi_old(a|s) -> exp(log_pi - log_pi_old)
                ratio = torch.exp(new_log_probs - mb_old_log_probs.detach())
                surr1 = ratio * mb_adv.detach()
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_adv.detach()
                
                actor_loss = -torch.min(surr1, surr2).sum(dim=-1) - self.entropy_coef * entropy
                actor_loss = (actor_loss * mb_filled).sum() / (mb_filled.sum() + 1e-8)

                # 2. Multi-Channel Critic Loss with Value Clipping
                val_losses = []
                for m in range(self.n_reward_channels):
                    v_pred_m = new_values[..., m]
                    v_target_m = mb_target_ret[..., m].detach()

                    if self.use_clipped_value_loss:
                        v_old_m = mb_v_preds_old[..., m].detach()
                        v_pred_clipped = v_old_m + torch.clamp(
                            v_pred_m - v_old_m, -self.clip_param, self.clip_param
                        )
                        vf_loss1 = (v_pred_m - v_target_m).pow(2)
                        vf_loss2 = (v_pred_clipped - v_target_m).pow(2)
                        channel_vf_loss = torch.max(vf_loss1, vf_loss2)
                    else:
                        channel_vf_loss = (v_pred_m - v_target_m).pow(2)

                    ch_loss = (channel_vf_loss.sum(dim=-1) * mb_filled).sum() / (mb_filled.sum() + 1e-8)
                    val_losses.append(ch_loss)

                total_value_loss = torch.stack(val_losses).sum()

                # 3. Total Loss & Optimization Step
                loss = actor_loss + self.value_loss_coef * total_value_loss
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                self.optimizer.step()

                # Diagnostics tracking
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
                    epoch_actor_losses.append(actor_loss.item())
                    epoch_value_losses.append(total_value_loss.item())
                    epoch_entropies.append(((entropy * mb_filled).sum() / (mb_filled.sum() + 1e-8)).item())
                    epoch_approx_kls.append(approx_kl.item())
                    epoch_clip_fracs.append(clip_frac.item())

        # ---------------------------------------------------------
        # Phase 4: Target Network Soft Updates
        # ---------------------------------------------------------
        if (
            self.target_update_interval_or_tau > 1.0
            and step % self.target_update_interval_or_tau == 0
        ):
            self.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            self.soft_update(self.target_update_interval_or_tau)

        # Diagnostic: negative advantage ratio on the first committed channel (or top score)
        if self.is_committed_optimism.any():
            top_coop_idx = torch.where(self.is_committed_optimism)[0][0].item()
        else:
            top_coop_idx = torch.argmax(self.coordination_ema).item()

        coop_adv_raw = raw_advantages[..., top_coop_idx]
        neg_coop_mask = coop_adv_raw < 0.0
        neg_coop_ratio = (neg_coop_mask.float() * batch.filled.unsqueeze(-1)).sum() / (
            batch.filled.unsqueeze(-1).sum() + 1e-8
        )

        return {
            "loss": sum(epoch_actor_losses) / len(epoch_actor_losses)
            + self.value_loss_coef * (sum(epoch_value_losses) / len(epoch_value_losses)),
            "actor_loss": sum(epoch_actor_losses) / len(epoch_actor_losses),
            "value_loss": sum(epoch_value_losses) / len(epoch_value_losses),
            "entropy": sum(epoch_entropies) / len(epoch_entropies),
            "approx_kl": sum(epoch_approx_kls) / len(epoch_approx_kls),
            "clip_frac": sum(epoch_clip_fracs) / len(epoch_clip_fracs),
            "coop_neg_adv_ratio": neg_coop_ratio.item(),
            # 1. Coordination Score EMA
            **{
                f"coordination_score_ch{m}": float(self.coordination_ema[m].item())
                for m in range(self.n_reward_channels)
            },
            # 2. Cumulative Positive Hits
            **{
                f"coordination_hits_ch{m}": int(self.coordination_hits[m].item())
                for m in range(self.n_reward_channels)
            },
            # 3. Latch Commitment Status (0 = standard MAPPO, 1 = locked to alpha=0.1)
            **{
                f"is_committed_ch{m}": int(self.is_committed_optimism[m].item())
                for m in range(self.n_reward_channels)
            },
        }