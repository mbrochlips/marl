import torch
import torch.nn as nn
import torch.nn.functional as F


class COMAInteractionCritic(nn.Module):
    """
    Centralized Multi-Channel Action-Value Critic for CTDE interaction estimation.
    For each agent i and reward channel m, estimates Q_i^m(s, (u^i, u^{-i})).
    Inspired by Foerster et al., AAAI 2018 (COMA).
    """
    def __init__(self, state_dim, act_dims, n_reward_channels, hidden_dim=128):
        super(COMAInteractionCritic, self).__init__()
        self.n_agents = len(act_dims)
        self.n_channels = n_reward_channels
        self.act_dims = act_dims

        # Input: Centralized state s + all other agents' one-hot actions u^{-i}
        # Output: Q-values for each discrete action u^i in act_dims[i], across all M channels
        self.q_networks = nn.ModuleList()
        for i in range(self.n_agents):
            other_act_dim = sum(act_dims[j] for j in range(self.n_agents) if j != i)
            in_dim = state_dim + other_act_dim
            out_dim = act_dims[i] * n_reward_channels
            self.q_networks.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dim),
                )
            )

    def forward(self, state, actions_one_hot):
        """
        Args:
            state: Tensor of shape (..., state_dim)
            actions_one_hot: List of length N with tensors of shape (..., act_dim[i])
        Returns:
            q_values: List of length N with tensors (..., act_dims[i], n_channels)
        """
        all_agent_q = []
        for i in range(self.n_agents):
            other_acts = [actions_one_hot[j] for j in range(self.n_agents) if j != i]
            critic_input = torch.cat([state] + other_acts, dim=-1)
            q_flat = self.q_networks[i](critic_input)
            # Reshape to (..., act_dim_i, n_channels)
            q_reshaped = q_flat.view(*q_flat.shape[:-1], self.act_dims[i], self.n_channels)
            all_agent_q.append(q_reshaped)
        return all_agent_q


def estimate_channel_interactions(
    batch,
    actor_network,
    coma_critic,
    n_agents,
    n_reward_channels,
    act_dims,
    device,
    sample_counterfactuals=8,
):
    """
    COMA-inspired Counterfactual Interaction Estimator.

    Quantifies how sensitive each reward channel m is to the coordinated actions
    of other agents (u^{-i}) versus independent execution.

    Returns:
        interaction_scores: Tensor of shape (n_reward_channels,), bounded in [0, 1].
                           Higher value indicates a channel driven by multi-agent coordination.
    """
    coma_critic.eval()
    with torch.no_grad():
        T, B = batch.rewards.shape[:2]
        # Global state representation in CTDE (flattened/concatenated observations)
        if hasattr(batch, "state") and batch.state is not None:
            states = batch.state[:-1]  # (T, B, state_dim)
        else:
            states = batch.obss[:-1].flatten(start_dim=-2)  # (T, B, n_agents * obs_dim)

        actions = batch.actions  # (T, B, n_agents)
        filled = batch.filled  # (T, B)

        # One-hot encode realized actions
        actions_one_hot = [
            F.one_hot(actions[..., i].long(), num_classes=act_dims[i]).float()
            for i in range(n_agents)
        ]

        # 1. Evaluate Q-values for the realized joint trajectories
        # List of length N, each: (T, B, act_dims[i], n_channels)
        q_realized = coma_critic(states, actions_one_hot)

        # 2. Extract policy distributions pi_i(a_i | o_i) for counterfactual expectation
        inputs = [batch.obss[:-1, :, i] for i in range(n_agents)]
        actor_logits, _ = actor_network(inputs, None)
        policies = [F.softmax(logits, dim=-1) for logits in actor_logits]  # (T, B, act_dim_i)

        # 3. Compute Counterfactual Cross-Agent Sensitivity (CCAS) per channel
        # How much does Agent i's Q-value vary when counterfactually sampling teammate actions u^{-i}?
        channel_interaction_variances = torch.zeros(n_reward_channels, device=device)

        for i in range(n_agents):
            actual_action_idx = actions[..., i].long().unsqueeze(-1).unsqueeze(-1)
            # Q-value for actual action under actual teammate actions: (T, B, 1, n_channels)
            q_actual_u = torch.gather(
                q_realized[i],
                dim=-2,
                index=actual_action_idx.expand(*actual_action_idx.shape[:-1], n_reward_channels),
            ).squeeze(-2)  # (T, B, n_channels)

            # Sample counterfactual teammate action profiles u^{-i} ~ pi^{-i}
            cf_q_samples = []
            for _ in range(sample_counterfactuals):
                cf_actions_one_hot = list(actions_one_hot)
                for j in range(n_agents):
                    if j != i:
                        dist_j = torch.distributions.Categorical(probs=policies[j])
                        cf_action_j = dist_j.sample()
                        cf_actions_one_hot[j] = F.one_hot(
                            cf_action_j, num_classes=act_dims[j]
                        ).float()

                # Recompute Q_i under counterfactual teammate actions
                q_cf = coma_critic(states, cf_actions_one_hot)[i]
                q_cf_u = torch.gather(
                    q_cf,
                    dim=-2,
                    index=actual_action_idx.expand(*actual_action_idx.shape[:-1], n_reward_channels),
                ).squeeze(-2)  # (T, B, n_channels)
                cf_q_samples.append(q_cf_u)

            # Variance across counterfactual teammate actions: (K, T, B, M) -> (T, B, M)
            cf_stack = torch.stack(cf_q_samples, dim=0)
            cf_var = torch.var(cf_stack, dim=0)  # (T, B, M)

            # Mask and accumulate across active timesteps
            valid_mask = filled.unsqueeze(-1)  # (T, B, 1)
            mean_agent_var = (cf_var * valid_mask).sum(dim=(0, 1)) / (valid_mask.sum() + 1e-8)
            channel_interaction_variances += mean_agent_var / n_agents

        # 4. Normalize interaction metric across channels into [0, 1]
        # High variance indicates strong coordination dependence.
        channel_scales = channel_interaction_variances + 1e-8
        interaction_scores = channel_scales / (torch.max(channel_scales) + 1e-8)

        # Smooth scaling using a sigmoid gate around median sensitivity
        median_sensitivity = torch.median(interaction_scores)
        interaction_scores = torch.sigmoid((interaction_scores - median_sensitivity) * 5.0)

    return interaction_scores



def fast_empirical_interaction_score(raw_advantages, filled_mask, eps=1e-8):
    """
    Computes cross-agent advantage coupling directly from multi-channel GAE.
    Evaluates cross-agent covariance relative to individual variance per channel.
    
    Args:
        raw_advantages: Tensor of shape (T, B, N, M)
        filled_mask:    Tensor of shape (T, B) or (T, B, 1)
        eps:            Numerical stability constant
    Returns:
        interaction_scores: Tensor of shape (M,), bounded in [0, 1]
    """
    T, B, N, M = raw_advantages.shape

    # 1. Flatten across time and batch for valid/filled steps
    mask = filled_mask.view(-1).bool()
    advs = raw_advantages.view(T * B, N, M)[mask]  # (V, N, M)

    if advs.shape[0] < 2 or N <= 1:
        # If single agent or not enough steps, zero multi-agent interaction
        return torch.zeros(M, device=raw_advantages.device)

    # 2. Mean-center advantages across the batch for each agent and channel
    centered = advs - advs.mean(dim=0, keepdim=True)  # (V, N, M)

    # 3. Vectorized covariance matrix per channel: shape (M, N, N)
    # v: batch/time steps, i/j: agents, m: reward channel
    cov_matrices = torch.einsum("v i m, v j m -> m i j", centered, centered) / (advs.shape[0] - 1)

    # 4. Separate individual agent variances (diagonal) from cross-agent covariances (off-diagonal)
    diag_mask = torch.eye(N, device=raw_advantages.device, dtype=torch.bool).unsqueeze(0)  # (1, N, N)
    
    total_agent_var = (cov_matrices * diag_mask).sum(dim=(-1, -2)) + eps  # (M,)
    cross_agent_cov = (cov_matrices.abs() * (~diag_mask)).sum(dim=(-1, -2))  # (M,)

    pair_count = max(1, N * (N - 1))
    mean_cross_cov = cross_agent_cov / pair_count
    mean_agent_var = total_agent_var / N

    coupling_ratios = mean_cross_cov / (mean_agent_var + eps)

    # 5. Normalize relative to the strongest coupled channel and apply sigmoid scaling
    max_coupling = torch.max(coupling_ratios)
    normalized_scores = coupling_ratios / (max_coupling + eps)
    interaction_scores = torch.sigmoid((normalized_scores - 0.5) * 6.0)

    return interaction_scores

import torch
import torch.nn.functional as F


def compute_coordination_score(raw_advantages, rewards, filled_mask, eps=1e-8):
    """
    Coordination Score with Solo Feasibility Penalty.
    
    Penalizes channels where agents can achieve positive rewards independently 
    (which eliminates solo foods with reward splitting).
    """
    T, B, N, M = raw_advantages.shape

    mask = filled_mask.view(-1).bool()
    valid_advs = raw_advantages.view(T * B, N, M)[mask]  # (V, N, M)
    valid_rews = rewards.view(T * B, N, M)[mask]         # (V, N, M)

    if valid_advs.shape[0] < 2 or N <= 1:
        return torch.zeros(M, device=raw_advantages.device)

    # ------------------------------------------------------------------
    # 1. Positive Upside Gate (Reject pure penalty channels)
    # ------------------------------------------------------------------
    max_reward = valid_rews.amax(dim=(0, 1))             # (M,)
    upside_gate = (max_reward > 0.0).float()

    # ------------------------------------------------------------------
    # 2. Solo Feasibility Penalty (The Fix for Reward Splitting)
    # ------------------------------------------------------------------
    # Detect timesteps where at least one agent got a reward > 0
    agent_has_rew = (valid_rews > 1e-4).float()          # (V, N, M)
    active_agent_count = agent_has_rew.sum(dim=1)        # (V, M): how many agents scored at timestep v

    # Event occurred on channel m
    any_scored = (active_agent_count > 0).float()        # (V, M)
    total_events = any_scored.sum(dim=0) + eps           # (M,)

    # Solo events: exactly 1 agent got a reward while others got 0
    solo_events = (active_agent_count == 1).float().sum(dim=0) # (M,)
    solo_ratio = solo_events / total_events               # (M,): ratio of solo collections

    # Joint events: all/multiple agents got a reward simultaneously
    joint_events = (active_agent_count >= 2).float().sum(dim=0) # (M,)
    joint_ratio = joint_events / total_events             # (M,)

    # Teammate Necessity Factor:
    # If a channel can be collected alone (high solo_ratio), it is NOT strictly cooperative.
    # For ch0: solo_ratio is 0.0, necessity is 1.0.
    # For ch1/ch2: solo_ratio is high (e.g. > 0.5), necessity -> 0.0.
    teammate_necessity = torch.clamp(1.0 - solo_ratio * 1.5, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 3. Positive Advantage Concordance (PAC)
    # ------------------------------------------------------------------
    pos_advs = F.relu(valid_advs)
    norms = torch.sqrt((pos_advs ** 2).sum(dim=0, keepdim=True)) + eps
    norm_pos_advs = pos_advs / norms
    pair_concordance = torch.einsum("v i m, v j m -> m i j", norm_pos_advs, norm_pos_advs)

    off_diag_mask = ~torch.eye(N, device=raw_advantages.device, dtype=torch.bool).unsqueeze(0)
    pair_count = max(1, N * (N - 1))
    mean_concordance = (pair_concordance * off_diag_mask).sum(dim=(-1, -2)) / pair_count

    # ------------------------------------------------------------------
    # 4. Final Coordination Score
    # ------------------------------------------------------------------
    # ch1/ch2: teammate_necessity drops them to ~0.0 because agents can collect them alone.
    # ch0: solo_ratio is 0.0, so teammate_necessity is 1.0.
    coordination_scores = upside_gate * teammate_necessity * (mean_concordance + 0.1 * joint_ratio)
    coordination_scores = torch.clamp(coordination_scores, 0.0, 1.0)

    return coordination_scores


class CoordinationDetector(nn.Module):
    """
    Algorithm-agnostic Multi-Agent Coordination Detector.
    Can be used by MAPPO, QMIX, MADDPG, or any decomposed MARL algorithm.
    """
    def __init__(self, n_reward_channels, n_agents, confidence_lambda=0.35):
        super().__init__()
        self.n_channels = n_reward_channels
        self.n_agents = n_agents
        self.confidence_lambda = confidence_lambda

        # Persistent GPU buffers (automatically saved/loaded with state_dict)
        self.register_buffer("all_time_max_solo", torch.zeros(n_reward_channels))
        self.register_buffer("all_time_max_joint", torch.zeros(n_reward_channels))
        self.register_buffer("cumulative_joint_events", torch.zeros(n_reward_channels))
        self.register_buffer("coordination_scores", torch.zeros(n_reward_channels))

    @torch.no_grad()
    def update(self, rewards, filled_mask):
        """
        Args:
            rewards: Tensor of shape (..., N_agents, N_channels) on CUDA
            filled_mask: Tensor of shape (...) indicating active steps
        Returns:
            scores: Tensor of shape (N_channels,) bounded in [0, 1]
        """
        # Vectorized zero-sync extraction
        mask = filled_mask.unsqueeze(-1).unsqueeze(-1)
        valid_rews = rewards * mask
        rews_flat = valid_rews.view(-1, self.n_agents, self.n_channels)

        has_rew = (rews_flat > 1e-4).float()
        active_counts = has_rew.sum(dim=1, keepdim=True)

        # 1. Solo detection (exactly 1 agent scored)
        is_solo = (active_counts == 1.0).float()
        batch_max_solo = (rews_flat * is_solo).amax(dim=(0, 1))

        # 2. Joint detection (>= 2 agents scored)
        is_joint = (active_counts >= 2.0).float()
        batch_max_joint = (rews_flat * is_joint).amax(dim=(0, 1))
        batch_joint_events = is_joint.amax(dim=1).sum(dim=0)

        # 3. Update all-time persistent buffers
        torch.maximum(self.all_time_max_solo, batch_max_solo, out=self.all_time_max_solo)
        torch.maximum(self.all_time_max_joint, batch_max_joint, out=self.all_time_max_joint)
        self.cumulative_joint_events.add_(batch_joint_events)

        # 4. Supermodular Surplus & Confidence
        surplus = F.relu(self.all_time_max_joint - self.all_time_max_solo)
        surplus_ratio = surplus / (self.all_time_max_joint + 1e-6)
        confidence = 1.0 - torch.exp(-self.confidence_lambda * self.cumulative_joint_events)

        self.coordination_scores.copy_(surplus_ratio * confidence)
        return self.coordination_scores

    def get_scores(self):
        return self.coordination_scores
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalCoordinationDetector(nn.Module):
    """
    Evaluates each reward channel on a single unified stance axis [-1.0, +1.0]:
      +1.0: Cooperative Coordination (Optimistic)
       0.0: Risk-Neutral Solo / Safe (Standard MAPPO)
      -1.0: Non-Cooperative Hazard (Pessimistic)
    """
    def __init__(self, n_reward_channels, n_agents, confidence_lambda=0.35):
        super().__init__()
        self.n_channels = n_reward_channels
        self.n_agents = n_agents
        self.confidence_lambda = confidence_lambda

        self.register_buffer("all_time_max_solo", torch.zeros(n_reward_channels))
        self.register_buffer("all_time_max_joint", torch.zeros(n_reward_channels))
        self.register_buffer("cumulative_joint_events", torch.zeros(n_reward_channels))
        self.register_buffer("stance_scores", torch.zeros(n_reward_channels))

    @torch.no_grad()
    def update(self, rewards, filled_mask):
        mask = filled_mask.unsqueeze(-1).unsqueeze(-1)
        valid_rews = rewards * mask
        rews_flat = valid_rews.view(-1, self.n_agents, self.n_channels)

        # -------------------------------------------------------------
        # 1. Cooperative Dimension (Supermodularity)
        # -------------------------------------------------------------
        has_pos_rew = (rews_flat > 1e-4).float()
        active_counts = has_pos_rew.sum(dim=1, keepdim=True)

        is_solo = (active_counts == 1.0).float()
        batch_max_solo = (rews_flat * is_solo).amax(dim=(0, 1))

        is_joint = (active_counts >= 2.0).float()
        batch_max_joint = (rews_flat * is_joint).amax(dim=(0, 1))
        batch_joint_events = is_joint.amax(dim=1).sum(dim=0)

        torch.maximum(self.all_time_max_solo, batch_max_solo, out=self.all_time_max_solo)
        torch.maximum(self.all_time_max_joint, batch_max_joint, out=self.all_time_max_joint)
        self.cumulative_joint_events.add_(batch_joint_events)

        surplus = F.relu(self.all_time_max_joint - self.all_time_max_solo)
        surplus_ratio = surplus / (self.all_time_max_joint + 1e-6)
        coop_confidence = 1.0 - torch.exp(-self.confidence_lambda * self.cumulative_joint_events)
        s_coop = surplus_ratio * coop_confidence # in [0.0, 1.0]

        # -------------------------------------------------------------
        # 2. Hazard Dimension (Negative Tail Risk)
        # -------------------------------------------------------------
        r_min = rews_flat.amin(dim=(0, 1))
        r_max = rews_flat.amax(dim=(0, 1))
        r_mean = rews_flat.mean(dim=(0, 1))

        neg_mag = F.relu(-r_min)
        pos_mag = F.relu(r_max)
        mag_ratio = neg_mag / (pos_mag + neg_mag + 1e-6)

        neg_shocks = F.relu(-(rews_flat - r_mean))
        pos_shocks = F.relu(rews_flat - r_mean)
        downside_std = torch.sqrt((neg_shocks ** 2).mean(dim=(0, 1)) + 1e-6)
        upside_std = torch.sqrt((pos_shocks ** 2).mean(dim=(0, 1)) + 1e-6)
        tail_asymmetry = downside_std / (upside_std + downside_std + 1e-6)

        s_risk = torch.clamp(2.0 * (mag_ratio * tail_asymmetry) - 0.4, 0.0, 1.0) # in [0.0, 1.0]

        # -------------------------------------------------------------
        # 3. Priority Gate: Lexicographic Single Score in [-1.0, +1.0]
        # -------------------------------------------------------------
        # Rule: If supermodular surplus exists, the score MUST be positive (+s_coop).
        #       Only if surplus is strictly 0.0 can the score be negative (-s_risk).
        is_cooperative = (surplus > 1e-4) & (self.cumulative_joint_events >= 1.0)

        unified_score = torch.where(
            is_cooperative,
            s_coop,        # Strictly in (0.0, +1.0]
            -s_risk        # Strictly in [-1.0, 0.0]
        )

        self.stance_scores.copy_(unified_score)
        return self.stance_scores

    # """
    # Evaluates reward channels on a spectrum of [-1.0, +1.0]:
    #   +1.0: Supermodular Cooperative Coordination (Enables Optimism)
    #    0.0: Independent / Risk-Neutral (Standard MAPPO)
    #   -1.0: Catastrophic Downside Hazard / Trap (Enables Pessimism)
    # """
    # def __init__(self, n_reward_channels, n_agents, confidence_lambda=0.35):
    #     super().__init__()
    #     self.n_channels = n_reward_channels
    #     self.n_agents = n_agents
    #     self.confidence_lambda = confidence_lambda

    #     # Persistent all-time buffers
    #     self.register_buffer("all_time_max_solo", torch.zeros(n_reward_channels))
    #     self.register_buffer("all_time_max_joint", torch.zeros(n_reward_channels))
    #     self.register_buffer("cumulative_joint_events", torch.zeros(n_reward_channels))
    #     self.register_buffer("spectrum_scores", torch.zeros(n_reward_channels))

    # @torch.no_grad()
    # def update(self, rewards, filled_mask):
    #     # Shape: (V, N, M)
    #     mask = filled_mask.unsqueeze(-1).unsqueeze(-1)
    #     valid_rews = rewards * mask
    #     rews_flat = valid_rews.view(-1, self.n_agents, self.n_channels)

    #     # =====================================================================
    #     # 1. Cooperative Dimension (Positive Supermodularity)
    #     # =====================================================================
    #     has_pos_rew = (rews_flat > 1e-4).float()
    #     active_counts = has_pos_rew.sum(dim=1, keepdim=True)

    #     is_solo = (active_counts == 1.0).float()
    #     batch_max_solo = (rews_flat * is_solo).amax(dim=(0, 1))

    #     is_joint = (active_counts >= 2.0).float()
    #     batch_max_joint = (rews_flat * is_joint).amax(dim=(0, 1))
    #     batch_joint_events = is_joint.amax(dim=1).sum(dim=0)

    #     torch.maximum(self.all_time_max_solo, batch_max_solo, out=self.all_time_max_solo)
    #     torch.maximum(self.all_time_max_joint, batch_max_joint, out=self.all_time_max_joint)
    #     self.cumulative_joint_events.add_(batch_joint_events)

    #     # Supermodular Surplus
    #     surplus = F.relu(self.all_time_max_joint - self.all_time_max_solo)
    #     surplus_ratio = surplus / (self.all_time_max_joint + 1e-6)
    #     coop_confidence = 1.0 - torch.exp(-self.confidence_lambda * self.cumulative_joint_events)
    #     s_coop = surplus_ratio * coop_confidence # [0.0, 1.0]

    #     # =====================================================================
    #     # 2. Hazard Dimension (Negative Tail Risk & Magnitude)
    #     # =====================================================================
    #     # Evaluate empirical distribution across valid steps
    #     r_min = rews_flat.amin(dim=(0, 1)) # (M,)
    #     r_max = rews_flat.amax(dim=(0, 1)) # (M,)

    #     # Catastrophic ratio: does negative magnitude dominate positive magnitude?
    #     neg_mag = F.relu(-r_min)
    #     pos_mag = F.relu(r_max)
    #     mag_ratio = neg_mag / (pos_mag + neg_mag + 1e-6) # [0, 1]

    #     # Downside semivariance: variance of negative shocks
    #     r_mean = rews_flat.mean(dim=(0, 1))
    #     neg_shocks = F.relu(-(rews_flat - r_mean)) # shocks below mean
    #     pos_shocks = F.relu(rews_flat - r_mean)    # shocks above mean

    #     downside_std = torch.sqrt((neg_shocks ** 2).mean(dim=(0, 1)) + 1e-6)
    #     upside_std = torch.sqrt((pos_shocks ** 2).mean(dim=(0, 1)) + 1e-6)

    #     tail_asymmetry = downside_std / (upside_std + downside_std + 1e-6) # [0, 1]

    #     # Risk score triggers when negative magnitude is severe AND asymmetric
    #     # (Blocks symmetric noise like zero-mean gaussian noise)
    #     s_risk = torch.clamp(2.0 * (mag_ratio * tail_asymmetry) - 0.4, 0.0, 1.0) # [0.0, 1.0]

    #     # =====================================================================
    #     # 3. Unified Spectrum Score: [-1.0, +1.0]
    #     # =====================================================================
    #     # A channel cannot be simultaneously a cooperative goal and a hazardous trap
    #     self.spectrum_scores.copy_(s_coop - s_risk)
    #     return self.spectrum_scores