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
from marlbase.utils.coor_score import CoordinationDetector


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
    Operates independently across time, batch, agents, and channels without loops over M.
    """
    T, B, N, M = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, N, M, device=rewards.device)

    # dones: (T, B, N) -> (T, B, N, 1) for broadcasting across M channels
    non_terminals = 1.0 - dones.unsqueeze(-1).float()

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * non_terminals[t] - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminals[t] * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class FiLMBlock(nn.Module):
    """
    Pre-LN Channel-conditioned Feed-Forward Block with Identity FiLM Initialization.
    """
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)

        # Modulation generator: predicts (gamma, beta)
        self.film_proj = nn.Linear(d_model, d_ffn * 2)

        self._reset_parameters()

    def _reset_parameters(self):
        # Orthogonal initialization for FFN layers
        nn.init.orthogonal_(self.linear1.weight, gain=1.0)
        nn.init.zeros_(self.linear1.bias)
        nn.init.orthogonal_(self.linear2.weight, gain=1.0)
        nn.init.zeros_(self.linear2.bias)

        # Zeroing weights and biases ensures gamma=0 and beta=0 at step 0
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, x, channel_emb):
        # x: (B, M, d_model), channel_emb: (M, d_model)

        norm_x = self.norm(x)
        h = self.linear1(norm_x)

        film_params = self.film_proj(channel_emb)   # (M, 2 * d_ffn)
        gamma, beta = film_params.chunk(2, dim=-1)  # (M, d_ffn) each

        # Identity-centered FiLM modulation: (1.0 + gamma) * h + beta
        # At init: (1.0 + 0.0) * h + 0.0 = h (perfect identity)
        h = (1.0 + gamma.unsqueeze(0)) * h + beta.unsqueeze(0)
        h = F.gelu(h)

        out = self.linear2(h)

        return x + out


import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityChannelQueryTransformerCritic(nn.Module):
    """
    Relational Entity-Centric Channel-Query Transformer Critic.
    
    Includes:
      1. Entity Self-Attention: allows agents and foods to compute relative
         distances and level differences before channel pooling.
      2. Contextualized Agent Readout: agent representations carry full scene
         awareness into the value head.
    """
    def __init__(
        self,
        n_agents,
        n_reward_channels,
        d_model=128,
        n_heads=4,
        d_ffn=256,
        state_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_channels = n_reward_channels
        self.d_model = d_model

        # 1. 5D Entity Projector with an extra hidden layer for rich initial features
        self.entity_proj = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.entity_norm1 = nn.LayerNorm(d_model)

        # 2. Entity-to-Entity Self-Attention (Crucial for relative distances!)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.entity_norm2 = nn.LayerNorm(d_model)

        # 3. Channel Query Embeddings
        self.channel_embeddings = nn.Embedding(n_reward_channels, d_model)
        nn.init.orthogonal_(self.channel_embeddings.weight)

        # 4. Channel Cross-Attention
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )

        # 5. Identity FiLM-FFN Subspaces
        self.film_ffn = FiLMBlock(d_model, d_ffn)
        self.final_channel_norm = nn.LayerNorm(d_model)
        self.final_agent_norm = nn.LayerNorm(d_model)

        # 6. Linear Pairwise Readout Head
        self.value_head = nn.Linear(d_model * 2, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _unpack_flat_state_to_tokens(self, state_flat):
        orig_shape = state_flat.shape[:-1]
        flat_dim = state_flat.shape[-1]
        assert flat_dim % 3 == 0, f"State dim {flat_dim} must be divisible by 3."
        num_entities = flat_dim // 3

        raw_entities = state_flat.view(*orig_shape, num_entities, 3)

        flags = torch.zeros(*orig_shape, num_entities, 2, device=state_flat.device)
        flags[..., :self.n_agents, 0] = 1.0  # Agents
        food_levels = raw_entities[..., self.n_agents:, 2]
        flags[..., self.n_agents:, 1] = (food_levels > 0.0).float()  # Active foods

        return torch.cat([raw_entities, flags], dim=-1)

    def forward(self, state):
        orig_shape = state.shape[:-1]
        state_flat = state.view(-1, state.shape[-1])
        B = state_flat.shape[0]

        # 1. Unpack into raw 5D tokens: (B, Total_Entities, 5)
        entity_tokens = self._unpack_flat_state_to_tokens(state_flat)

        # 2. Initial Entity Embeddings
        h_entities = self.entity_norm1(self.entity_proj(entity_tokens))

        # 3. Entity-to-Entity Self-Attention: Computes pairwise distances & relative levels!
        attn_entities, _ = self.self_attn(
            query=h_entities,
            key=h_entities,
            value=h_entities,
        )
        # Residual + Norm
        h_entities = self.entity_norm2(h_entities + attn_entities)

        # 4. Channel Cross-Attention across contextualized entities
        ch_indices = torch.arange(self.n_channels, device=state.device)
        ch_embs = self.channel_embeddings(ch_indices)
        ch_queries = ch_embs.unsqueeze(0).expand(B, -1, -1)

        norm_q = self.norm_query(ch_queries)
        norm_kv = self.norm_kv(h_entities)

        attn_ch, _ = self.cross_attn(
            query=norm_q,
            key=norm_kv,
            value=norm_kv,
        )
        x = ch_queries + attn_ch

        # 5. FiLM Subspaces
        channel_tokens = self.film_ffn(x, ch_embs)
        channel_tokens = self.final_channel_norm(channel_tokens)  # (B, M, d_model)

        # 6. Extract CONTEXTUALIZED Agent Tokens (They now know all food distances!)
        agent_tokens = self.final_agent_norm(h_entities[:, :self.n_agents, :])  # (B, N, d_model)

        # 7. Pairwise Readout
        agent_exp = agent_tokens.unsqueeze(2).expand(B, self.n_agents, self.n_channels, self.d_model)
        chan_exp = channel_tokens.unsqueeze(1).expand(B, self.n_agents, self.n_channels, self.d_model)
        joint_pairs = torch.cat([agent_exp, chan_exp], dim=-1)

        values = self.value_head(joint_pairs).squeeze(-1)  # (B, N, M)

        return values.view(*orig_shape, self.n_agents, self.n_channels)


class ExtensibleEntityCritic(EntityChannelQueryTransformerCritic):
    """
    Critic that dynamically grows its channel capacity on-the-fly
    as new reward IDs are encountered in the environment.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dynamic channel identifier registry: {name_or_id: channel_index}
        self.channel_registry = {}

    def register_new_channel(self, channel_id, optimizer=None):
        """
        Dynamically appends a strictly orthogonal query token to the critic.
        """
        device = self.channel_embeddings.weight.device
        current_weights = self.channel_embeddings.weight.data  # (M, d_model)
        M, d = current_weights.shape

        # 1. Gram-Schmidt Orthogonalization to find a perpendicular query vector
        v = torch.randn(d, device=device)
        for i in range(M):
            proj = torch.dot(v, current_weights[i]) * current_weights[i]
            v = v - proj
        new_query = F.normalize(v, dim=0).unsqueeze(0)  # (1, d_model)

        # 2. Expand embedding matrix: (M+1, d_model)
        expanded_weights = torch.cat([current_weights, new_query], dim=0)
        self.n_channels = M + 1
        self.channel_embeddings = nn.Embedding.from_pretrained(expanded_weights, freeze=False)

        # 3. Synchronize Target Critic
        if hasattr(self, "target_critic"):
            self.target_critic.channel_embeddings = nn.Embedding.from_pretrained(
                expanded_weights.clone(), freeze=False
            )

        # 4. Update Optimizer state for the new parameter tensor
        if optimizer is not None:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.shape == current_weights.shape:
                        # Replace parameter reference in optimizer
                        group["params"].remove(p)
                        group["params"].append(self.channel_embeddings.weight)
                        break

        # Register channel ID
        idx = len(self.channel_registry)
        self.channel_registry[channel_id] = idx
        print(f"[Dynamic Critic] Discovered reward '{channel_id}'. Allocated Channel {idx} (M={self.n_channels}).", flush=True)
        return idx

# =====================================================================
# Full Network Architecture (MAPPO-OT with CQ-Transformer Critic)
# =====================================================================

class TransformerCriticOTMAPPONetwork(nn.Module):
    """
    MAPPO-OT with Channel-Query Transformer Critic & Coordination Detector.
    
    Key Features:
      - Unified Transformer Critic: O(1) networks instead of M decoupled critics.
      - FiLM-modulated subspaces: Prevents gradient interference across channels.
      - Supermodular Coordination Detector: Selectively targets optimism to 
        cooperative channels while maintaining risk-neutrality on solo/penalty channels.
      - Completely vectorized advantage transformation and critic updates.
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

    
        # Channels, Weights & Autonomous Coordination Detector
        self.n_reward_channels = getattr(cfg, "n_reward_channels", 4)
        self.optimism_alpha = getattr(cfg, "optimism_alpha", 0.1)

        weights = getattr(cfg, "channel_weights", [1.0] * self.n_reward_channels)
        self.register_buffer(
            "channel_weights", torch.tensor(weights, dtype=torch.float32, device=device)
        )

        self.coord_detector = CoordinationDetector(
            n_reward_channels=self.n_reward_channels,
            n_agents=self.n_agents,
        ).to(device)


        # Decentralized Actors
        self.use_rnn = actor.use_rnn
        if not actor.parameter_sharing:
            self.actor = MultiAgentIndependentNetwork(
                obs_dims, list(actor.layers), act_dims, actor.use_rnn, actor.use_orthogonal_init
            )
        else:
            self.actor = MultiAgentSharedNetwork(
                obs_dims, list(actor.layers), act_dims, actor.parameter_sharing, actor.use_rnn, actor.use_orthogonal_init
            )

        # Unified Transformer Critic (Replaces M Independent Critics)
        self.centralised_critic = critic.centralised
        critic_state_dim = sum(obs_dims) if critic.centralised else obs_dims[0]

        critic_d_model = getattr(cfg, "critic_d_model", 128)
        critic_n_heads = getattr(cfg, "critic_n_heads", 4)
        critic_d_ffn = getattr(cfg, "critic_d_ffn", 256)
        n_state_tokens = getattr(cfg, "n_state_tokens", 4)

        self.critic = EntityChannelQueryTransformerCritic(
            state_dim=critic_state_dim,
            n_agents=self.n_agents,
            n_reward_channels=self.n_reward_channels,
            d_model=critic_d_model,
            n_heads=critic_n_heads,
            d_ffn=critic_d_ffn,
        )
        self.target_critic = copy.deepcopy(self.critic)

        # Return Standardization & Optimizer
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


    # State & Execution Handlers


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

    def get_value(self, inputs, target=False):
        """
        Unified forward pass through the Entity Transformer Critic.
        In LBF with full sight, inputs[0] canonically contains all agents 
        followed by all foods.
        """
        if isinstance(inputs, (list, tuple)):
            state = inputs[0]  # Shape: (..., (N + F) * 3)
        else:
            state = inputs

        critic_net = self.target_critic if target else self.critic
        return critic_net(state)

    def evaluate_actions(
        self,
        inputs,
        action,
        actor_hiddens=None,
        action_mask=None,
        state=None,
    ):
        if state is None:
            state = inputs

        values = self.get_value(state)
        actor_features, actor_hiddens = self.actor(inputs, actor_hiddens)
        dist = self.get_dist(actor_features, action_mask)
        action_log_probs = torch.cat(dist.log_probs(action), dim=-1)
        dist_entropy = torch.stack(dist.entropy(), dim=-1).sum(dim=-1)

        return values, action_log_probs, dist_entropy, actor_hiddens

    def soft_update(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)


    # The Update Pipeline
    def update(self, batch, step):
        T, B = batch.rewards.shape[:2]
        batch_dones = batch.dones.float().unsqueeze(-1).repeat(1, 1, self.n_agents)


        # > Phase 1: Compute Target Values & Multi-Channel GAE
        with torch.no_grad():
            all_values = self.get_value(self.split_obs(batch.obss), target=False)
            v_preds = all_values[:-1]       # (T, B, n_agents, n_channels)
            next_v_preds = all_values[1:]   # (T, B, n_agents, n_channels)

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


        # > Phase 2: Autonomous Coordination-Targeted Optimism

        coord_scores = self.coord_detector.update(batch.rewards, batch.filled)

        # Fully vectorized LeakyReLU softening across all M channels:
        # alpha_m: (M,) -> (1, 1, 1, M)
        alpha = 1.0 - coord_scores * (1.0 - self.optimism_alpha)
        alpha_broadcast = alpha.view(1, 1, 1, -1)

        transformed_advantages = torch.where(
            raw_advantages < 0.0,
            alpha_broadcast * raw_advantages,
            raw_advantages,
        )

        # Channel aggregation & standardization
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

                # Clipped Surrogate Policy Loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs.detach())
                surr1 = ratio * mb_adv.detach()
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_adv.detach()
                actor_loss = -torch.min(surr1, surr2).sum(dim=-1) - self.entropy_coef * entropy
                actor_loss = (actor_loss * mb_filled).sum() / (mb_filled.sum() + 1e-8)

                # Unified Vectorized Multi-Channel Critic Loss (No Python loops over M)
                v_target = mb_target_ret.detach()
                if self.use_clipped_value_loss:
                    v_old = mb_v_preds_old.detach()
                    v_pred_clipped = v_old + torch.clamp(
                        new_values - v_old, -self.clip_param, self.clip_param
                    )
                    vf_loss1 = (new_values - v_target).pow(2)
                    vf_loss2 = (v_pred_clipped - v_target).pow(2)
                    channel_vf_loss = torch.max(vf_loss1, vf_loss2)  # (T, mb_B, N, M)
                else:
                    channel_vf_loss = (new_values - v_target).pow(2)

                # Average across batch and agents, sum across channels
                mb_mask = mb_filled.unsqueeze(-1).unsqueeze(-1)  # (T, mb_B, 1, 1)
                total_value_loss = (
                    (channel_vf_loss * mb_mask).sum(dim=(0, 1, 2)) / (mb_mask.sum() + 1e-8)
                ).sum()

                # 3. Backward Pass
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


        # > Phase 4: Target Network Soft Updates & Diagnostics

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

        # Logging
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
