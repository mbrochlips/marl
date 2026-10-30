
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


class SingleEgoAttentionActor(nn.Module):
    """
    Ego-attention policy network for an individual agent.
    Flattens arbitrary leading dimensions (T, B) so it never crashes on shape unpacking.
    """
    def __init__(self, n_agents, act_dim=6, d_model=128):
        super().__init__()
        self.n_agents = n_agents
        self.d_model = d_model

        # Universal 5D Projector: [y, x, level, is_agent, is_food]
        self.entity_proj = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

        # Cross-Attention: Self attends to all surrounding entities
        self.ego_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.post_norm = nn.LayerNorm(d_model)

        # Action Readout Head (6 discrete actions in LBF)
        self.action_head = nn.Linear(d_model, act_dim)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    def _unpack_obs(self, obs_flat):
        # Flattens all leading dimensions into total_B for safe attention processing
        orig_shape = obs_flat.shape[:-1]
        flat_dim = obs_flat.shape[-1]
        K = flat_dim // 3

        # Reshape to (total_B, K, 3)
        raw = obs_flat.reshape(-1, K, 3)
        total_B = raw.shape[0]

        flags = torch.zeros(total_B, K, 2, device=obs_flat.device)
        flags[:, :self.n_agents, 0] = 1.0  # Agents
        food_levels = raw[:, self.n_agents:, 2]
        flags[:, self.n_agents:, 1] = (food_levels > 0.0).float()  # Active foods

        tokens = torch.cat([raw, flags], dim=-1)  # (total_B, K, 5)
        return tokens, orig_shape

    def forward(self, obs_flat):
        # 1. Unpack to (total_B, K, 5) and remember original shape
        tokens, orig_shape = self._unpack_obs(obs_flat)

        # 2. Project entities
        h = self.norm(self.entity_proj(tokens))  # (total_B, K, d_model)

        # 3. Ego-Token is always index 0 (self)
        ego_token = h[:, 0:1, :]                 # (total_B, 1, d_model)
        context_tokens = h                       # (total_B, K, d_model)

        # 4. Cross-Attention
        attn_out, _ = self.ego_attn(query=ego_token, key=context_tokens, value=context_tokens)
        ego_feat = self.post_norm(ego_token + attn_out).squeeze(1)  # (total_B, d_model)

        # 5. Action Logits
        logits = self.action_head(ego_feat)      # (total_B, act_dim)

        # 6. Reshape back to (*orig_shape, act_dim)
        return logits.view(*orig_shape, -1)


class MultiAgentEgoAttentionActor(nn.Module):
    """Multi-agent container handling shared or independent actors."""
    def __init__(self, n_agents, act_dims, d_model=128, parameter_sharing=True):
        super().__init__()
        self.n_agents = n_agents
        self.parameter_sharing = parameter_sharing

        if parameter_sharing:
            self.actor = SingleEgoAttentionActor(n_agents, act_dims[0], d_model)
        else:
            self.actors = nn.ModuleList([
                SingleEgoAttentionActor(n_agents, act_dims[i], d_model)
                for i in range(n_agents)
            ])

    def forward(self, inputs, hiddens=None):
        # inputs is a list of N tensors: [obs_agent_0, obs_agent_1, ...]
        logits = []
        for i in range(self.n_agents):
            actor_net = self.actor if self.parameter_sharing else self.actors[i]
            logits.append(actor_net(inputs[i]))
        return logits, hiddens

    def init_hiddens(self, batch_size, device):
        return None  # Feed-forward Transformer; no recurrent state required

def make_orthonormal_basis(dim: int, seed: int = 42) -> torch.Tensor:
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(dim, dim))
    torch.set_rng_state(rng_state)
    return Q


class ContinualEntityTransformerCritic(nn.Module):
    def __init__(self, n_agents, d_model=128, n_heads=4, d_ffn=256, **kwargs):
        super().__init__()
        self.n_agents = n_agents
        self.d_model = d_model

        # Universal fixed basis: 128 orthogonal queries
        basis = make_orthonormal_basis(d_model, seed=42)
        self.register_buffer("channel_basis", basis)

        # 1. 5D Entity Projector & Self-Attention
        self.entity_proj = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.entity_norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.entity_norm2 = nn.LayerNorm(d_model)

        # 2. Channel Cross-Attention
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        # 3. FiLM Subspaces
        self.film_ffn = FiLMBlock(d_model, d_ffn)
        self.final_channel_norm = nn.LayerNorm(d_model)
        self.final_agent_norm = nn.LayerNorm(d_model)

        # 4. Universal Readout Head
        self.value_head = nn.Linear(d_model * 2, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _unpack_flat_state_to_tokens(self, state_flat):
        orig_shape = state_flat.shape[:-1]
        flat_dim = state_flat.shape[-1]
        num_entities = flat_dim // 3
        raw_entities = state_flat.view(*orig_shape, num_entities, 3)

        flags = torch.zeros(*orig_shape, num_entities, 2, device=state_flat.device)
        flags[..., :self.n_agents, 0] = 1.0  # Agents
        food_levels = raw_entities[..., self.n_agents:, 2]
        flags[..., self.n_agents:, 1] = (food_levels > 0.0).float()  # Active foods
        return torch.cat([raw_entities, flags], dim=-1)

    def forward(self, state, channel_indices: torch.Tensor):
        orig_shape = state.shape[:-1]
        state_flat = state.view(-1, state.shape[-1])
        B = state_flat.shape[0]
        M = len(channel_indices)

        # 1. Self-Attention across entities
        entity_tokens = self._unpack_flat_state_to_tokens(state_flat)
        h_entities = self.entity_norm1(self.entity_proj(entity_tokens))
        attn_ent, _ = self.self_attn(h_entities, h_entities, h_entities)
        h_entities = self.entity_norm2(h_entities + attn_ent)

        # 2. Gather queries from basis for active environment channels
        ch_embs = self.channel_basis[channel_indices]  # (M, d_model)
        ch_queries = ch_embs.unsqueeze(0).expand(B, -1, -1)

        # 3. Cross-Attention
        norm_q = self.norm_query(ch_queries)
        norm_kv = self.norm_kv(h_entities)
        attn_ch, _ = self.cross_attn(query=norm_q, key=norm_kv, value=norm_kv)
        x = ch_queries + attn_ch

        # 4. Identity FiLM Subspaces
        channel_tokens = self.film_ffn(x, ch_embs)
        channel_tokens = self.final_channel_norm(channel_tokens)

        # 5. Extract Contextualized Agent Tokens
        agent_tokens = self.final_agent_norm(h_entities[:, :self.n_agents, :])

        # 6. Bilinear Readout
        agent_exp = agent_tokens.unsqueeze(2).expand(B, self.n_agents, M, self.d_model)
        chan_exp = channel_tokens.unsqueeze(1).expand(B, self.n_agents, M, self.d_model)
        joint_pairs = torch.cat([agent_exp, chan_exp], dim=-1)

        values = self.value_head(joint_pairs).squeeze(-1)  # (B, N, M)
        return values.view(*orig_shape, self.n_agents, M)

class TransformerCriticOTMAPPONetwork(nn.Module):
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
        self.optimism_alpha = getattr(cfg, "optimism_alpha", 0.1)

        self.n_agents = len(obs_space)
        act_dims = [flatdim(a) for a in action_space]
        d_model = getattr(cfg, "critic_d_model", 128)

        # ─────────────────────────────────────────────────────────────
        # STEP 3 INTEGRATION: Entity-Centric Ego-Attention Actor
        # ─────────────────────────────────────────────────────────────
        self.actor = MultiAgentEgoAttentionActor(
            n_agents=self.n_agents,
            act_dims=act_dims,
            d_model=d_model,
            parameter_sharing=actor.parameter_sharing,
        )

        # ─────────────────────────────────────────────────────────────
        # STEP 2 INTEGRATION: Continual Fixed-Basis Transformer Critic
        # ─────────────────────────────────────────────────────────────
        self.critic = ContinualEntityTransformerCritic(
            n_agents=self.n_agents,
            d_model=d_model,
            n_heads=getattr(cfg, "critic_n_heads", 4),
            d_ffn=getattr(cfg, "critic_d_ffn", 256),
        )
        self.target_critic = copy.deepcopy(self.critic)

        # Default active channel setup
        initial_channels = getattr(cfg, "n_reward_channels", 4)
        self.standardise_returns = cfg.standardise_returns
        self.set_active_channels(list(range(initial_channels)))

        # Optimizer & Updates
        self.to(device)
        self.soft_update(1.0)

        optimizer = getattr(optim, cfg.optimizer)
        if isinstance(optimizer, str):
            optimizer = getattr(optim, optimizer)
        self.optimizer = optimizer(self.parameters(), lr=cfg.lr)
        self.target_update_interval_or_tau = cfg.target_update_interval_or_tau

        self.split_obs = _split_batch([flatdim(s) for s in obs_space])
        self.split_act = _split_batch(self.n_agents * [1])

    def set_active_channels(self, channel_indices):
        """
        Dynamically binds the model to a set of active environment channels.
        Can be called whenever transferring to a new environment.
        """
        if not isinstance(channel_indices, torch.Tensor):
            channel_indices = torch.tensor(channel_indices, dtype=torch.long, device=self.device)
        self.active_channel_indices = channel_indices
        self.n_reward_channels = len(channel_indices)

        # Dynamically adapt return normalizer and coordination detector to M channels
        if self.standardise_returns:
            self.ret_ms = RunningMeanStd(
                shape=(self.n_agents, self.n_reward_channels), device=self.device
            )

        self.coord_detector = CoordinationDetector(
            n_reward_channels=self.n_reward_channels,
            n_agents=self.n_agents,
        ).to(self.device)

    def init_actor_hiddens(self, batch_size):
        return self.actor.init_hiddens(batch_size, self.device)

    def get_dist(self, action_logits, action_mask=None):
        if action_mask is not None:
            masked_logits = []
            for logits, mask in zip(action_logits, action_mask):
                masked_logits.append(logits * mask + (1 - mask) * -1e8)
            action_logits = masked_logits
        return MultiCategorical([Categorical(logits=logits) for logits in action_logits])

    def act(self, inputs, actor_hiddens=None, action_mask=None):
        actor_logits, actor_hiddens = self.actor(inputs, actor_hiddens)
        dist = self.get_dist(actor_logits, action_mask)
        actions = dist.sample()
        return torch.stack(actions, dim=0), actor_hiddens

    def get_value(self, inputs, target=False):
        """
        Evaluates the global state across active channel basis queries.
        """
        state = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        critic_net = self.target_critic if target else self.critic
        return critic_net(state, self.active_channel_indices)

    def evaluate_actions(self, inputs, action, actor_hiddens=None, action_mask=None, state=None):
        if state is None:
            state = inputs

        values = self.get_value(state)
        actor_logits, actor_hiddens = self.actor(inputs, actor_hiddens)
        dist = self.get_dist(actor_logits, action_mask)
        action_log_probs = torch.cat(dist.log_probs(action), dim=-1)
        dist_entropy = torch.stack(dist.entropy(), dim=-1).sum(dim=-1)

        return values, action_log_probs, dist_entropy, actor_hiddens

    def soft_update(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)

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

        # Vectorized LeakyReLU softening across active channels
        alpha = 1.0 - coord_scores * (1.0 - self.optimism_alpha)
        alpha_broadcast = alpha.view(1, 1, 1, -1)

        transformed_advantages = torch.where(
            raw_advantages < 0.0,
            alpha_broadcast * raw_advantages,
            raw_advantages,
        )

        # Pure decomposed sum across channels (zero hardcoded weights!)
        total_advantage = transformed_advantages.sum(dim=-1)

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

                # Clipped Policy Loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs.detach())
                surr1 = ratio * mb_adv.detach()
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_adv.detach()
                actor_loss = -torch.min(surr1, surr2).sum(dim=-1) - self.entropy_coef * entropy
                actor_loss = (actor_loss * mb_filled).sum() / (mb_filled.sum() + 1e-8)

                # Clipped Multi-Channel Critic Loss
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

                # Optimization step
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

        # > Phase 4: Target Critic Soft Update
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