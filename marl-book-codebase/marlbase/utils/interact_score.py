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