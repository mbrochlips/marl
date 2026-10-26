import torch
import torch.nn as nn
import torch.nn.functional as F

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


        # 1. Cooperative Dimension (Supermodularity)

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


        # 2. Hazard Dimension (Negative Tail Risk)
    
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


        # 3. Priority Gate: Lexicographic Single Score in [-1.0, +1.0]

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