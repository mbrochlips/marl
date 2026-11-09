from pathlib import Path
from typing import Sequence, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinationDetector(nn.Module):
    """Ultra-fast, vectorized multi-agent coordination detector."""
    def __init__(self, channel_names: Sequence[str], n_agents: int, confidence_lambda: float = 0.35):
        super().__init__()
        self.channel_names = list(channel_names)
        self.n_channels, self.n_agents = len(self.channel_names), n_agents
        self.confidence_lambda = confidence_lambda
        self.channel_to_idx = {name: i for i, name in enumerate(self.channel_names)}

        self.register_buffer("all_time_max_solo", torch.zeros(self.n_channels))
        self.register_buffer("all_time_max_joint", torch.zeros(self.n_channels))
        self.register_buffer("cumulative_joint_events", torch.zeros(self.n_channels))
        self.register_buffer("coordination_scores", torch.zeros(self.n_channels))

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"channel_names": self.channel_names, "state": self.state_dict()}, path)

    @torch.no_grad()
    def load(self, path: str):
        data = torch.load(path, map_location=self.coordination_scores.device)
        for i, name in enumerate(data["channel_names"]):
            if name in self.channel_to_idx:
                idx = self.channel_to_idx[name]
                for k in ["coordination_scores", "all_time_max_joint", "all_time_max_solo", "cumulative_joint_events"]:
                    getattr(self, k)[idx] = data["state"][k][i]

    @torch.no_grad()
    def update(self, rewards: torch.Tensor, filled_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # [T * B, n_agents, n_channels]
        rews = rewards.reshape(-1, self.n_agents, self.n_channels)
        if filled_mask is not None:
            rews = rews * filled_mask.reshape(-1, 1, 1)

        # Collapse agents to 2D [M, C] immediately
        active = (rews > 1e-4).sum(dim=1)  # (M, C)
        step_max = rews.amax(dim=1)        # (M, C)

        max_solo = torch.where(active == 1, step_max, 0.0).amax(dim=0)
        max_joint = torch.where(active >= 2, step_max, 0.0).amax(dim=0)
        joint_events = (active >= 2).sum(dim=0)

        torch.maximum(self.all_time_max_solo, max_solo, out=self.all_time_max_solo)
        torch.maximum(self.all_time_max_joint, max_joint, out=self.all_time_max_joint)
        self.cumulative_joint_events.add_(joint_events)

        surplus = F.relu(self.all_time_max_joint - self.all_time_max_solo)
        confidence = 1.0 - torch.exp(-self.confidence_lambda * self.cumulative_joint_events)
        self.coordination_scores.copy_((surplus / (self.all_time_max_joint + 1e-6)) * confidence)
        return self.coordination_scores

    def get_scores_dict(self) -> Dict[str, float]:
        return dict(zip(self.channel_names, self.coordination_scores.tolist()))

    def get_scores(self) -> torch.Tensor:
        return self.coordination_scores

class DummyDetector(CoordinationDetector):
    def __init__(self, channel_names: Sequence[str], n_agents: int, confidence_lambda: float = 0.35):
        super().__init__(channel_names, n_agents, confidence_lambda)

        self.channel_names = list(channel_names)
        self.n_channels = len(self.channel_names)
        self.coordination_scores = torch.zeros(self.n_channels)

    def update(self, rewards: torch.Tensor, filled_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.coordination_scores
    
    def get_scores(self) -> torch.Tensor:
        return self.coordination_scores
    
    def get_scores_dict(self) -> Dict[str, float]:
        return dict(zip(self.channel_names, self.coordination_scores.tolist()))
    
    def load(self, path: str):
        return None
    
    def save(self, path: str):
        return None

class RiskDetector(nn.Module):
    """Conservative, slow-acting detector of potentially harmful reward channels."""

    def __init__(
        self,
        channel_names: Sequence[str],
        n_agents: int,
        ema_alpha: float = 0.02,
        trend_alpha: float = 0.005,
        min_observations: int = 500,
        confidence_threshold: float = 0.95,
        trend_tolerance: float = 0.01,
        negative_return_threshold: float = 1e-4,
        risk_scale: float = 0.5,
        risk_alpha: float = 0.002,
        max_risk: float = 0.5,
        reward_threshold: float = 1e-4,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.channel_names = list(channel_names)
        self.n_channels = len(channel_names)
        self.n_agents = n_agents

        self.ema_alpha = ema_alpha
        self.trend_alpha = trend_alpha
        self.min_observations = min_observations
        self.confidence_threshold = confidence_threshold
        self.trend_tolerance = trend_tolerance
        self.negative_return_threshold = negative_return_threshold
        self.risk_scale = risk_scale
        self.risk_alpha = risk_alpha
        self.max_risk = max_risk
        self.reward_threshold = reward_threshold
        self.eps = eps

        self.register_buffer("average_returns", torch.zeros(self.n_channels))
        self.register_buffer("average_nonzero_returns", torch.zeros(self.n_channels))
        self.register_buffer("recent_nonzero_returns", torch.zeros(self.n_channels))
        self.register_buffer("previous_trend", torch.zeros(self.n_channels))

        self.register_buffer("total_observations", torch.zeros(self.n_channels))
        self.register_buffer("nonzero_observations", torch.zeros(self.n_channels))

        self.register_buffer("risk_evidence", torch.zeros(self.n_channels))
        self.register_buffer("risk_confidence", torch.zeros(self.n_channels))
        self.register_buffer("risk_scores", torch.zeros(self.n_channels))

    @torch.no_grad()
    def update(
        self,
        rewards: torch.Tensor,
        coord_scores: torch.Tensor,
        filled_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        rews = rewards.reshape(-1, self.n_agents, self.n_channels)

        if filled_mask is not None:
            rews = rews[filled_mask.reshape(-1).bool()]

        if rews.shape[0] == 0:
            return -self.risk_scores

        # Average return per timestep
        step_returns = rews.mean(dim=1)
        batch_mean = step_returns.mean(dim=0)

        self.average_returns.lerp_(batch_mean, self.ema_alpha)

        # Average return conditional on non-zero rewards
        nonzero = rews.abs() > self.reward_threshold
        counts = nonzero.sum(dim=1)
        active = counts > 0

        event_returns = (
            (rews * nonzero).sum(dim=1)
            / counts.clamp(min=1)
        )

        event_counts = active.sum(dim=0).float()
        event_sums = event_returns.sum(dim=0)
        has_events = event_counts > 0

        batch_event_mean = event_sums / event_counts.clamp(min=1)

        initialized = self.nonzero_observations > 0

        self.average_nonzero_returns.copy_(
            torch.where(
                initialized,
                self.average_nonzero_returns.lerp(
                    batch_event_mean, self.ema_alpha
                ),
                torch.where(
                    has_events,
                    batch_event_mean,
                    self.average_nonzero_returns
                )
            )
        )

        self.recent_nonzero_returns.copy_(
            torch.where(
                initialized,
                self.recent_nonzero_returns.lerp(
                    batch_event_mean, self.trend_alpha
                ),
                torch.where(
                    has_events,
                    batch_event_mean,
                    self.recent_nonzero_returns
                )
            )
        )

        self.total_observations.add_(step_returns.shape[0])
        self.nonzero_observations.add_(event_counts)

        # Return trend
        trend = self.recent_nonzero_returns - self.previous_trend
        self.previous_trend.copy_(self.recent_nonzero_returns)

        stagnant = trend <= self.trend_tolerance

        # Observation confidence
        confidence = 1.0 - torch.exp(
            -self.nonzero_observations / self.min_observations
        )
        self.risk_confidence.copy_(confidence)

        sufficient_data = (
            (self.nonzero_observations >= self.min_observations)
            & (confidence >= self.confidence_threshold)
        )

        # Coordination protection
        protected = coord_scores.to(
            device=self.risk_scores.device,
            dtype=self.risk_scores.dtype
        ) > 0

        # Risk conditions
        negative_returns = (
            (self.average_returns < -self.negative_return_threshold)
            & (self.average_nonzero_returns < -self.negative_return_threshold)
        )

        risky = (
            negative_returns
            & stagnant
            & sufficient_data
            & ~protected
        )

        self.risk_evidence.copy_(
            torch.where(
                risky,
                self.risk_evidence + 1,
                torch.zeros_like(self.risk_evidence)
            )
        )

        # Slowly evolving risk penalty
        severity = (-self.average_returns).clamp(
            min=0, max=self.max_risk
        )

        target_risk = (
            self.risk_scale * severity * confidence
        ).clamp(max=self.max_risk)

        target_risk = torch.where(
            risky,
            target_risk,
            torch.zeros_like(target_risk)
        )

        self.risk_scores.lerp_(target_risk, self.risk_alpha)

        # Absolute coordination protection
        self.risk_scores.masked_fill_(protected, 0.0)

        return -self.risk_scores

    def get_risk_scores(self) -> torch.Tensor:
        return -self.risk_scores

    def get_scores_dict(self) -> Dict[str, float]:
        return dict(zip(self.channel_names, self.get_risk_scores().tolist()))

    def get_diagnostics(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "average_return": self.average_returns[i].item(),
                "average_nonzero_return": self.average_nonzero_returns[i].item(),
                "recent_nonzero_return": self.recent_nonzero_returns[i].item(),
                "observations": self.nonzero_observations[i].item(),
                "confidence": self.risk_confidence[i].item(),
                "risk_evidence": self.risk_evidence[i].item(),
                "risk_score": -self.risk_scores[i].item(),
            }
            for i, name in enumerate(self.channel_names)
        }

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


class CoordinationDetector_idx(nn.Module):
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