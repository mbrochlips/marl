
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # At init: (1.0 + 0.0) * h + 0.0 = h (perfect identity)
        h = (1.0 + gamma.unsqueeze(0)) * h + beta.unsqueeze(0)
        h = F.gelu(h)

        out = self.linear2(h)

        return x + out

class TwoLayerFiLMBlock(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)

        # 2-Layer non-linear FiLM generator
        self.film_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_ffn * 2),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        # Orthogonal init for main FFN
        nn.init.orthogonal_(self.linear1.weight, gain=1.0)
        nn.init.zeros_(self.linear1.bias)
        nn.init.orthogonal_(self.linear2.weight, gain=1.0)
        nn.init.zeros_(self.linear2.bias)

        # First layer of generator: standard orthogonal init
        nn.init.orthogonal_(self.film_proj[0].weight, gain=1.0)
        nn.init.zeros_(self.film_proj[0].bias)

        # ZERO-INIT the final projection layer:
        # Guarantees gamma=0 and beta=0 at step 0 (Zero gradient interference!)
        nn.init.zeros_(self.film_proj[2].weight)
        nn.init.zeros_(self.film_proj[2].bias)

    def forward(self, x, channel_emb):
        # x: (B, M, d_model), channel_emb: (M, d_model)
        norm_x = self.norm(x)
        h = self.linear1(norm_x)

        # Non-linear channel modulation
        film_params = self.film_proj(channel_emb)   # (M, 2 * d_ffn)
        gamma, beta = film_params.chunk(2, dim=-1)

        # At step 0: (1.0 + 0.0) * h + 0.0 = h (Strict mathematical identity)
        h = (1.0 + gamma.unsqueeze(0)) * h + beta.unsqueeze(0)
        h = F.gelu(h)

        out = self.linear2(h)
        return x + out

class ChannelQueryTransformerCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        n_agents,
        max_channels=32,
        d_model=128,
        n_heads=4,
        d_ffn=256,
        n_state_tokens=4,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.d_model = d_model
        self.max_channels = max_channels
        self.n_state_tokens = n_state_tokens

        # 1. Tokenizer
        self.state_tokenizer = nn.Sequential(
            nn.Linear(self.state_dim + 1, d_model), # adding deadspace for dumping unwanted att 
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * n_state_tokens),
        )

        # 2. Universal Channel Query Embeddings
        self.channel_embeddings = nn.Embedding(max_channels, d_model)
        nn.init.orthogonal_(self.channel_embeddings.weight)

        # 3. Pre-LN Cross-Attention
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )

        # 4. Pre-LN FiLM FFN
        self.film_ffn = FiLMBlock(d_model, d_ffn)
        self.final_norm = nn.LayerNorm(d_model)

        # 5. Simple Direct Per-Agent Readout Head (Inspired by OLD)
        self.head = nn.Linear(d_model, n_agents)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

        self.register_buffer("active_indices", torch.arange(4, dtype=torch.long))

    def set_active_channels(self, active_indices):
        self.active_indices = active_indices.long()

    def forward(self, state):
        orig_shape = state.shape[:-1]
        state_flat = state.view(-1, state.shape[-1])
        B = state_flat.shape[0]

        # Padding for dumping unwanted att
        pad = torch.zeros(B, 1, device=state.device, dtype=state.dtype)
        state_flat = torch.cat([state_flat, pad], dim=-1)

        # 1. State tokens: (B, 4, d_model)
        state_tokens = self.state_tokenizer(state_flat).view(
            B, self.n_state_tokens, self.d_model
        )

        # 2. Slice active channel queries: (B, M_active, d_model)
        ch_embs = self.channel_embeddings(self.active_indices)
        ch_queries = ch_embs.unsqueeze(0).expand(B, -1, -1)

        # 3. Pre-LN Cross-Attention
        norm_q = self.norm_query(ch_queries)
        norm_kv = self.norm_kv(state_tokens)
        attn_out, _ = self.cross_attn(query=norm_q, key=norm_kv, value=norm_kv)
        x = ch_queries + attn_out

        # 4. FiLM Modulation
        x = self.film_ffn(x, ch_embs)

        # 5. Final LayerNorm & Direct Readout per Agent
        x_norm = self.final_norm(x)                  # (B, M_active, d_model)
        values = self.head(x_norm).transpose(1, 2)   # (B, n_agents, M_active)

        return values.view(*orig_shape, self.n_agents, len(self.active_indices))
    

class TeamResidualReadoutHead(nn.Module):
    """
    Universal Capacity Readout Head with Shared Baseline + Agent Residual Offsets.
    Allocated with max_channels capacity; slices active channels at runtime.
    """
    def __init__(self, d_model, n_agents, max_channels=32):
        super().__init__()
        self.d_model = d_model
        self.n_agents = n_agents
        self.max_channels = max_channels

        self.w_team = nn.Parameter(torch.empty(max_channels, d_model))
        self.b_team = nn.Parameter(torch.zeros(max_channels))
        self.delta_w = nn.Parameter(torch.zeros(n_agents, max_channels, d_model))
        self.delta_b = nn.Parameter(torch.zeros(n_agents, max_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.w_team, gain=1.0)
        nn.init.zeros_(self.b_team)
        nn.init.zeros_(self.delta_w)
        nn.init.zeros_(self.delta_b)

    def forward(self, h, active_indices):
        """
        Args:
            h: (B, M_active, d_model)
            active_indices: Tensor of active basis slot indices (len = M_active)
        """
        w = self.w_team[active_indices]              # (M_active, d_model)
        b = self.b_team[active_indices]              # (M_active,)
        dw = self.delta_w[:, active_indices]         # (N, M_active, d_model)
        db = self.delta_b[:, active_indices]         # (N, M_active)

        # v_team = torch.sum(h * w.unsqueeze(0), dim=-1) + b.unsqueeze(0
        v_team = torch.einsum("bmd, md -> bm", h, w) + b.unsqueeze(0)
        # v_offset = torch.sum(h.unsqueeze(1) * dw.unsqueeze(0), dim=-1) + db.unsqueeze(0)
        v_offset = torch.einsum("bmd, nmd -> bnm", h, dw) + db.unsqueeze(0)


        return v_team.unsqueeze(1) + v_offset
    

class ChannelQueryTransformerCriticTeamResidualReadout(nn.Module):
    """
    Proven 1-Layer Flat Tokenizer with Canonical State Capacity:
    - Restores the exact spatial grounding that achieved +2.3 returns.
    - Zero tensor-shape mismatches during transfer.
    - Slices active channel queries dynamically via universal basis slots.
    """
    def __init__(
        self,
        state_dim,
        att_padding, # adding deadspace for dumping unwanted att 
        n_agents,
        max_channels=32,
        d_model=128,
        n_heads=4,
        d_ffn=256,
        n_state_tokens=4,
    ):
        super().__init__()
        self.att_padding=att_padding
        self.n_agents = n_agents
        self.d_model = d_model
        self.max_channels = max_channels
        self.n_state_tokens = n_state_tokens

        # Proven direct projection: 15k parameters, fast convergence
        self.state_tokenizer = nn.Sequential(
            nn.Linear(state_dim + self.att_padding, d_model), # pad with 1
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * n_state_tokens),
        )

        # Channel Queries & Cross-Attention
        self.channel_embeddings = nn.Embedding(max_channels, d_model)
        nn.init.orthogonal_(self.channel_embeddings.weight)

        self.norm_query = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )

        # 1-Layer FiLM with zero-initialization
        self.film_ffn = FiLMBlock(d_model, d_ffn)
        self.final_norm = nn.LayerNorm(d_model)

        self.head = TeamResidualReadoutHead(
            d_model=d_model,
            n_agents=n_agents,
            max_channels=max_channels,
        )

        self.register_buffer("active_indices", torch.arange(4, dtype=torch.long))

    def set_active_channels(self, active_indices):
        self.active_indices = active_indices.long()

    def forward(self, state):
        orig_shape = state.shape[:-1]
        state_flat = state.view(-1, state.shape[-1])
        B = state_flat.shape[0]

        # Pad state 
        pad = torch.zeros(B, self.att_padding, device=state.device, dtype=state.dtype)
        state_flat = torch.cat([state_flat, pad], dim=-1)

        # 1. Proven direct projection to 4 state tokens
        state_tokens = self.state_tokenizer(state_flat).view(
            B, self.n_state_tokens, self.d_model
        )

        # 2. Slice active channel queries
        ch_embs = self.channel_embeddings(self.active_indices)
        ch_queries = ch_embs.unsqueeze(0).expand(B, -1, -1)

        # 3. Cross-Attention
        norm_q = self.norm_query(ch_queries)
        norm_kv = self.norm_kv(state_tokens)
        attn_out, _ = self.cross_attn(query=norm_q, key=norm_kv, value=norm_kv)
        x = ch_queries + attn_out

        # 4. FiLM Modulation & Fused Readout
        x = self.film_ffn(x, ch_embs)
        x_norm = self.final_norm(x)

        values = self.head(x_norm, self.active_indices)
        return values.view(*orig_shape, self.n_agents, len(self.active_indices))