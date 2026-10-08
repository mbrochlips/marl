# Citations:
# https://github.com/marl-book/codebase
# https://github.com/HenriqueSabino/gym_soccer_env

from collections import defaultdict
from einops import rearrange
from gymnasium.spaces import flatdim
import torch
from torch.distributions import Categorical
import torch.nn as nn
from torch import optim

from marlbase.utils.models import MultiAgentIndependentNetwork, MultiAgentSharedNetwork
from marlbase.utils.utils import MultiCategorical, compute_nstep_returns
from marlbase.utils.standardise_stream import RunningMeanStd


def _split_batch(splits):
    def thunk(batch):
        return torch.split(batch, splits, dim=-1)

    return thunk


class DecompA2CNetwork(nn.Module):
    """
    Decomposed Reward Actor-Critic Network (Decomp-A2C).
    
    Features:
      - N x M independent critics: M separate critic networks (each evaluating N agents),
        yielding N x M completely decoupled networks.
      - Independent Channel Normalization via RunningMeanStd over (N, M).
      - Targeted Optimism: Asymmetric TD-loss on the cooperative channel.
      - Scalar-equivalent advantage aggregation for the decentralized actor.
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
        super(DecompA2CNetwork, self).__init__()
        self.gamma = cfg.gamma
        self.entropy_coef = cfg.entropy_coef
        self.n_steps = cfg.n_steps
        self.grad_clip = cfg.grad_clip
        self.value_loss_coef = cfg.value_loss_coef
        self.device = device

        self.n_agents = len(obs_space)
        obs_dims = [flatdim(o) for o in obs_space]
        act_dims = [flatdim(a) for a in action_space]

        # -------------------------------------------------------------
        # 1. Decomposed Reward & Targeted Optimism Settings
        # -------------------------------------------------------------
        self.n_reward_channels = getattr(cfg, "n_reward_channels", 3)
        self.coop_channel_idx = getattr(cfg, "coop_channel_idx", 2)  # Channel 2 = ID_3 (Coop Food)
        self.optimism_tau = getattr(cfg, "optimism_tau", 0.75)       # tau > 0.5 -> optimistic leniency
        self.channel_weights = getattr(cfg, "channel_weights", [1.0] * self.n_reward_channels)
        
        # Momentum
        self.momentum_beta1 = getattr(cfg, "momentum_beta1", 0.8)
        self.momentum_beta2 = getattr(cfg, "momentum_beta2", 0.95)
        self.use_adam_momentum = getattr(cfg, "use_adam_momentum", True)
        # -------------------------------------------------------------
        # 2. Decentralized Actors (Policy Networks)
        # -------------------------------------------------------------
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
        # 3. Exactly M x N Critics (M independent Multi-Agent Critics)
        # -------------------------------------------------------------
        self.centralised_critic = critic.centralised
        critic_obs_shape = (
            self.n_agents * [sum(obs_dims)] if critic.centralised else obs_dims
        )

        # We instantiate M separate networks.
        # If parameter_sharing is False, each MultiAgentIndependentNetwork contains
        # N independent networks -> exactly M x N = 6 independent neural networks!
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
        # 4. Independent Channel Normalization & Optimizer
        # -------------------------------------------------------------
        self.standardise_returns = cfg.standardise_returns
        if self.standardise_returns:
            # Independent running stats for each agent AND each reward channel
            self.ret_ms = RunningMeanStd(
                shape=(self.n_agents, self.n_reward_channels), device=device
            )

        self.soft_update(1.0)
        self.to(device)

        optimizer = getattr(optim, cfg.optimizer)
        if type(optimizer) is str:
            optimizer = getattr(optim, optimizer)
        self.optimizer_class = optimizer

        lr = cfg.lr
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.target_update_interval_or_tau = cfg.target_update_interval_or_tau

        self.split_obs = _split_batch([flatdim(s) for s in obs_space])
        self.split_act = _split_batch(self.n_agents * [1])

        print(self)

    # -------------------------------------------------------------
    # Hidden State Handlers
    # -------------------------------------------------------------
    def init_critic_hiddens(self, batch_size, target=False):
        critic_list = self.target_critics if target else self.critics
        return [c.init_hiddens(batch_size, self.device) for c in critic_list]

    def init_actor_hiddens(self, batch_size):
        return self.actor.init_hiddens(batch_size, self.device)

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError("Use act, get_value or evaluate_actions instead.")

    # -------------------------------------------------------------
    # Action Sampling
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Vector Value Evaluation (Querying all M x N critics)
    # -------------------------------------------------------------
    def get_value(self, inputs, critic_hiddens=None, target=False):
        """
        Queries all M critics.
        Returns:
            values: Tensor of shape (..., n_agents, n_reward_channels)
            next_hiddens: list of length M containing RNN hidden states.
        """
        if self.centralised_critic:
            inputs = self.n_agents * [torch.cat(inputs, dim=-1)]

        critic_list = self.target_critics if target else self.critics
        channel_values = []
        next_hiddens = []

        for m in range(self.n_reward_channels):
            h_m = critic_hiddens[m] if critic_hiddens is not None else None
            val_m, next_h_m = critic_list[m](inputs, h_m)
            # val_m is list of N tensors of shape (..., 1) -> cat to (..., n_agents)
            val_m = torch.cat(val_m, dim=-1)
            channel_values.append(val_m)
            next_hiddens.append(next_h_m)

        # Stack over channels -> shape: (..., n_agents, n_reward_channels)
        values = torch.stack(channel_values, dim=-1)
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

        return (values, action_log_probs, dist_entropy, critic_hiddens, actor_hiddens)

    # -------------------------------------------------------------
    # Target Network Soft Update
    # -------------------------------------------------------------
    def soft_update(self, t):
        for source, target in zip(self.critics, self.target_critics):
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

    # -------------------------------------------------------------
    # The Core Decomposed Update Method
    # -------------------------------------------------------------
    def update(self, batch, step):
        # 1. Target Value Predictions for next states: (T+1, B, n_agents, n_reward_channels)
        with torch.no_grad():
            next_values, _ = self.get_value(
                self.split_obs(batch.obss), critic_hiddens=None, target=True
            )

        # 2. De-standardize target values before Bellman return computation
        if self.standardise_returns:
            next_values = next_values * torch.sqrt(self.ret_ms.var + 1e-8) + self.ret_ms.mean

        # 3. Compute N-Step Returns independently for each reward channel
        batch_done = batch.dones.float().unsqueeze(-1).repeat(1, 1, self.n_agents)
        returns_list = []

        for m in range(self.n_reward_channels):
            # batch.rewards shape: (T, B, n_agents, n_reward_channels)
            ret_m = compute_nstep_returns(
                batch.rewards[..., m],
                batch_done,
                next_values[..., m],
                self.n_steps,
                self.gamma,
            )
            returns_list.append(ret_m)

        # Returns shape: (T, B, n_agents, n_reward_channels)
        returns = torch.stack(returns_list, dim=-1)

        # 4. Independent Channel Normalization
        if self.standardise_returns:
            self.ret_ms.update(returns)
            returns = (returns - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var + 1e-8)

        # 5. Evaluate Current Policy & Critic Predictions
        values, action_log_probs, entropy, _, _ = self.evaluate_actions(
            self.split_obs(batch.obss[:-1]),
            self.split_act(batch.actions),
            critic_hiddens=None,
            actor_hiddens=None,
            action_mask=rearrange(batch.action_masks[:-1], "E B N A -> N E B A")
            if batch.action_masks is not None
            else None,
        )

        # 6. Decomposed Advantages with Adam / Polyak Momentum Routing

        # advantages shape: (T, B, n_agents, n_reward_channels)
        advantages = returns - values

        channel_weights = torch.tensor(
            self.channel_weights, device=self.device, dtype=torch.float32
        )
        weighted_adv = advantages * channel_weights  # (T, B, n_agents, n_channels)

        T, B, N, M = weighted_adv.shape
        total_advantages = []

        # Hyperparameters (configurable via cfg)
        beta1 = getattr(self, "momentum_beta1", 0.8)   # 1st moment decay (Polyak velocity)
        beta2 = getattr(self, "momentum_beta2", 0.95)  # 2nd moment decay (Adam variance)
        use_adam = getattr(self, "use_adam_momentum", True)

        # Initialize momentum states across parallel environments and agents
        m_t = torch.zeros(B, N, M, device=self.device)  # 1st moment (velocity)
        s_t = torch.zeros(B, N, M, device=self.device)  # 2nd moment (variance)

        for t in range(T):
            # Check for episode boundaries to reset momentum
            # batch.dones has shape (T+1, B)
            if t > 0:
                done_mask = batch.dones[t].bool()  # (B,)
                if done_mask.any():
                    m_t[done_mask] = 0.0
                    s_t[done_mask] = 0.0

            curr_adv = weighted_adv[t]  # (B, N, M)

            # Update 1st moment (Directional Velocity)
            m_t = beta1 * m_t + (1.0 - beta1) * curr_adv

            if use_adam:
                # Update 2nd moment (Signal Volatility)
                s_t = beta2 * s_t + (1.0 - beta2) * (curr_adv ** 2)
                # Compute Signal-to-Noise Ratio (Adamized score)
                routing_scores = m_t / (torch.sqrt(s_t) + 1e-4)
            else:
                # Pure Polyak Momentum
                routing_scores = m_t

            # Route to the channel with the highest velocity / SNR
            chosen_channel = routing_scores.max(dim=-1).indices  # (B, N)

            # Extract the actual advantage of the winning channel for the policy gradient
            chosen_adv = curr_adv.gather(-1, chosen_channel.unsqueeze(-1)).squeeze(-1)  # (B, N)
            total_advantages.append(chosen_adv)

        total_advantage = torch.stack(total_advantages, dim=0)  # (T, B, n_agents)

        # 7. Actor Loss
        actor_loss = (
            -(action_log_probs * total_advantage.detach()).sum(dim=-1)
            - self.entropy_coef * entropy
        )
        actor_loss = (actor_loss * batch.filled).sum() / batch.filled.sum()

        # 8. Critic Losses with Targeted Optimism on Channel 2 (Coop Food)
        td_errors = returns - values
        value_losses = []
        metrics_dict = {}

        for m in range(self.n_reward_channels):
            delta_m = td_errors[..., m]

            if m == self.coop_channel_idx and self.optimism_tau != 0.5:
                # Targeted Asymmetric Loss for Cooperation:
                # When delta < 0 (failure due to teammate miscoordination), penalty is discounted by (1 - tau)
                # When delta >= 0 (successful coordination), weight is tau
                indicator_under = (delta_m < 0).float()
                weights = torch.abs(self.optimism_tau - indicator_under) * 2.0
                channel_loss = weights * delta_m.pow(2)
            else:
                # Standard risk-neutral MSE for solo channels
                channel_loss = delta_m.pow(2)

            ch_loss_mean = (channel_loss.sum(dim=-1) * batch.filled).sum() / batch.filled.sum()
            value_losses.append(ch_loss_mean)
            metrics_dict[f"v_loss_ch{m}"] = ch_loss_mean.item()

        total_value_loss = torch.stack(value_losses).sum()

        # 9. Joint Backward Pass & Gradient Clipping
        loss = actor_loss + self.value_loss_coef * total_value_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        # 10. Target Network Updates
        if (
            self.target_update_interval_or_tau > 1.0
            and step % self.target_update_interval_or_tau == 0
        ):
            self.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            self.soft_update(self.target_update_interval_or_tau)

        # Metrics for logging
        metrics_dict.update({
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "value_loss": total_value_loss.item(),
            "entropy": ((entropy * batch.filled).sum() / batch.filled.sum()).item(),
        })

        return metrics_dict