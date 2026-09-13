"""SwarmStar policy network -- AlphaStar's architecture, re-targeted to one drone.

Layout follows the Nature paper's supplementary "Detailed Architecture":
3 encoders -> 1 recurrent core -> a chain of autoregressive heads.

    scalar encoder  ---+
    entity encoder ----+--> LSTM core --> action_type -> delay -> target
    spatial encoder ---+                      -> heading -> pitch -> speed -> comms
    (scatter connection)

Each head is conditioned on an *autoregressive embedding* threaded from the
previous head and gated by a GLU on the scalar context, exactly as in AlphaStar.
The pointer head selects a target from this drone's visible-entity list.

Two deliberate departures from AlphaStar:
  * the location head is a discrete heading x pitch x speed triple rather than a
    deconvolutional map distribution -- a drone commands a direction, not a pixel
  * unit selection picks one target, so the selected-units LSTM collapses to a
    single query projection

Sampling uses the Gumbel-max trick with noise supplied as an *input*. That keeps
the graph free of RNG so it exports to ONNX cleanly, and lets the browser choose
stochastic play (random noise) or deterministic flight (zeros -> argmax).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# head layout: 6 single-token heads, then n_latent comms tokens
H_ACTION, H_DELAY, H_TARGET, H_HEADING, H_PITCH, H_SPEED = range(6)
N_MAIN_HEADS = 6


class GLU(nn.Module):
    """Gated linear unit conditioned on context -- AlphaStar's gating primitive."""

    def __init__(self, d_in: int, d_ctx: int, d_out: int):
        super().__init__()
        self.gate = nn.Linear(d_ctx, d_in)
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        return self.proj(x * torch.sigmoid(self.gate(ctx)))


class ResBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.a, self.b = nn.Linear(d, d), nn.Linear(d, d)
        self.n = nn.LayerNorm(d)

    def forward(self, x: Tensor) -> Tensor:
        return self.n(x + self.b(F.relu(self.a(x))))


class SelfAttentionLayer(nn.Module):
    """Pre-norm transformer layer with an explicit key-padding mask.

    Written out rather than using nn.MultiheadAttention so the mask semantics are
    unambiguous under ONNX export.
    """

    def __init__(self, d: int, n_heads: int, d_ff: int):
        super().__init__()
        self.h, self.dk = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Linear(d_ff, d))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        M, K, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        shape = lambda t: t.view(M, K, self.h, self.dk).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)          # [M,h,K,K]
        att = att.masked_fill(~mask.view(M, 1, 1, K), float("-inf"))
        att = att.softmax(-1)
        # a drone with no contacts at all attends to nothing; zero it rather than
        # letting the all -inf row produce NaN
        att = torch.nan_to_num(att, 0.0)
        y = (att @ v).transpose(1, 2).reshape(M, K, d)
        x = x + self.out(y)
        return x + self.ff(self.n2(x))


class EntityEncoder(nn.Module):
    """Transformer over the <=K entities this drone can actually see."""

    def __init__(self, d_feat: int, d: int, n_layers: int, n_heads: int, d_ff: int):
        super().__init__()
        self.embed = nn.Linear(d_feat, d)
        self.layers = nn.ModuleList(
            [SelfAttentionLayer(d, n_heads, d_ff) for _ in range(n_layers)])
        self.pool = nn.Linear(d, d)

    def forward(self, ent: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        x = self.embed(ent)
        for l in self.layers:
            x = l(x, mask)
        x = x * mask.unsqueeze(-1)
        pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
        return x, F.relu(self.pool(pooled))


class SpatialEncoder(nn.Module):
    """AlphaStar's scatter connection, made egocentric.

    Entity embeddings are scattered into a body-frame bird's-eye grid, then read
    by a small conv stack. This is what lets the policy reason about the *shape*
    of the local formation -- gaps, flanks, clusters -- which pooled attention
    over an unordered set cannot express.
    """

    def __init__(self, d_ent: int, d_out: int, grid: int = 12, ch: int = 16):
        super().__init__()
        self.G, self.ch = grid, ch
        self.proj = nn.Linear(d_ent, ch)
        self.conv = nn.Sequential(
            nn.Conv2d(ch, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.head = nn.Linear(32 * (grid // 4) ** 2, d_out)

    def forward(self, ent_emb: Tensor, ent_feat: Tensor, mask: Tensor) -> Tensor:
        M, K, _ = ent_emb.shape
        G = self.G
        # entity features 0:2 are body-frame x,y already normalised by sensor range
        xy = ent_feat[..., :2].clamp(-1 + 1e-4, 1 - 1e-4)
        cell = ((xy + 1.0) * 0.5 * G).long().clamp(0, G - 1)           # [M,K,2]
        flat = (cell[..., 1] * G + cell[..., 0]) * mask.long()          # [M,K]

        vals = self.proj(ent_emb) * mask.unsqueeze(-1)                  # [M,K,ch]
        grid = torch.zeros(M, G * G, self.ch, device=ent_emb.device, dtype=vals.dtype)
        grid.scatter_add_(1, flat.unsqueeze(-1).expand(M, K, self.ch), vals)
        grid = grid.view(M, G, G, self.ch).permute(0, 3, 1, 2)
        return F.relu(self.head(self.conv(grid).flatten(1)))


class SwarmPolicy(nn.Module):
    """One drone's brain. Identical weights on every drone, in sim and on hardware."""

    def __init__(self, d_scalar: int = 28, d_entity: int = 13, n_entities: int = 16,
                 n_action_types: int = 7, n_heading: int = 16, n_pitch: int = 5,
                 n_speed: int = 4, n_delay: int = 4, n_comm_tokens: int = 8,
                 codebook: int = 64, d_model: int = 96, d_core: int = 160,
                 n_layers: int = 2, n_heads: int = 2, d_ff: int = 192):
        super().__init__()
        self.K, self.d_core = n_entities, d_core
        self.n_comm_tokens, self.codebook = n_comm_tokens, codebook
        self.head_sizes = [n_action_types, n_delay, n_entities,
                           n_heading, n_pitch, n_speed]
        self.max_logits = max(max(self.head_sizes), codebook)

        # ---- encoders
        self.scalar_enc = nn.Sequential(nn.Linear(d_scalar, 64), nn.ReLU())
        self.ctx = nn.Sequential(nn.Linear(d_scalar, 64), nn.ReLU())
        self.entity_enc = EntityEncoder(d_entity, d_model, n_layers, n_heads, d_ff)
        self.spatial_enc = SpatialEncoder(d_model, d_model)

        # ---- recurrent core: the memory that survives losing sight of a target
        self.core = nn.LSTMCell(64 + d_model + d_model, d_core)

        # ---- heads
        self.at_trunk = nn.Sequential(ResBlock(d_core), ResBlock(d_core))
        self.at_glu = GLU(d_core, 64, n_action_types)
        self.at_embed = nn.Linear(n_action_types, d_core)
        self.at_gate = GLU(d_core, 64, d_core)

        def mlp_head(n_out):
            return nn.ModuleDict({
                "trunk": nn.Sequential(nn.Linear(d_core, 128), nn.ReLU()),
                "out": nn.Linear(128, n_out),
                "back": nn.Linear(n_out, d_core),
            })
        self.delay_head = mlp_head(n_delay)
        self.heading_head = mlp_head(n_heading)
        self.pitch_head = mlp_head(n_pitch)
        self.speed_head = mlp_head(n_speed)

        # pointer head: dot-product attention from an autoregressive query onto keys
        self.ptr_key = nn.Linear(d_model, 32)
        self.ptr_query = nn.Sequential(nn.Linear(d_core, 128), nn.ReLU(),
                                       nn.Linear(128, 32))
        self.ptr_back = nn.Linear(32, d_core)

        # comms head: the mesh payload the swarm invents for itself
        self.comm_head = nn.Sequential(nn.Linear(d_core, 128), nn.ReLU(),
                                       nn.Linear(128, n_comm_tokens * codebook))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _sample(logits: Tensor, mask: Tensor | None,
                noise: Tensor) -> tuple[Tensor, Tensor]:
        """Gumbel-max over masked logits. noise=0 degenerates to argmax.

        Returns the MASKED logits alongside the sample: the loss must score the
        same distribution the action was drawn from, or illegal actions leak a
        gradient and the policy learns to want them.
        """
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return logits, (logits + noise[..., :logits.shape[-1]]).argmax(-1)

    def initial_state(self, M: int, device) -> tuple[Tensor, Tensor]:
        z = torch.zeros(M, self.d_core, device=device)
        return z, z.clone()

    # ------------------------------------------------------------------ forward
    def forward(self, scalars: Tensor, entities: Tensor, entity_mask: Tensor,
                action_mask: Tensor, hx: Tensor, cx: Tensor, noise: Tensor,
                forced: dict[str, Tensor] | None = None):
        """
        scalars      [M, d_scalar]
        entities     [M, K, d_entity]
        entity_mask  [M, K]  bool -- which slots hold a real contact
        action_mask  [M, n_action_types] bool -- which macro-actions are legal now
        noise        [M, 6 + n_comm_tokens, max_logits] -- Gumbel noise (0 = greedy)
        forced       optional teacher actions. When given, the autoregressive
                     chain conditions on THESE instead of its own samples --
                     teacher forcing, needed for supervised imitation and for
                     recomputing log-probs of actions taken by an older policy.
        """
        F_ = forced or {}
        ctx = self.ctx(scalars)
        emb_scalar = self.scalar_enc(scalars)
        ent_emb, emb_entity = self.entity_enc(entities, entity_mask)
        emb_spatial = self.spatial_enc(ent_emb, entities, entity_mask)

        hx, cx = self.core(torch.cat([emb_scalar, emb_entity, emb_spatial], -1), (hx, cx))

        logits, actions = {}, {}

        # --- 1. action_type
        z = self.at_trunk(hx)
        lg, a = self._sample(self.at_glu(z, ctx), action_mask, noise[:, H_ACTION])
        if "action_type" in F_: a = F_["action_type"]
        logits["action_type"], actions["action_type"] = lg, a
        ar = hx + self.at_gate(
            self.at_embed(F.one_hot(a, lg.shape[-1]).float()), ctx)

        # --- 2..6. the remaining single-token heads, each refining `ar`
        def run(head, name, idx, mask=None):
            nonlocal ar
            lg, s = self._sample(head["out"](head["trunk"](ar)), mask, noise[:, idx])
            if name in F_: s = F_[name]
            ar = ar + head["back"](F.one_hot(s, lg.shape[-1]).float())
            logits[name], actions[name] = lg, s

        run(self.delay_head, "delay", H_DELAY)

        # target: pointer network over visible entities only
        keys = self.ptr_key(ent_emb)                                    # [M,K,32]
        q = self.ptr_query(ar).unsqueeze(-1)                            # [M,32,1]
        lg, t = self._sample((keys @ q).squeeze(-1), entity_mask, noise[:, H_TARGET])
        if "target" in F_: t = F_["target"]
        picked = torch.gather(keys, 1, t.view(-1, 1, 1).expand(-1, 1, 32)).squeeze(1)
        ar = ar + self.ptr_back(picked)
        logits["target"], actions["target"] = lg, t

        run(self.heading_head, "heading", H_HEADING)
        run(self.pitch_head, "pitch", H_PITCH)
        run(self.speed_head, "speed", H_SPEED)

        # --- 7. comms: n_comm_tokens drawn jointly, the whole BLE payload
        cl = self.comm_head(ar).view(-1, self.n_comm_tokens, self.codebook)
        cn = noise[:, N_MAIN_HEADS:N_MAIN_HEADS + self.n_comm_tokens, :self.codebook]
        comms = F_["comms"] if "comms" in F_ else (cl + cn).argmax(-1)
        logits["comms"], actions["comms"] = cl, comms

        return logits, actions, hx, cx


class CentralizedCritic(nn.Module):
    """Value function over the TRUE state of both teams. Training only.

    AlphaStar: "during training only, the value function is estimated using
    information from the player's and the opponent's perspectives." This network
    is never exported and never runs on a drone.
    """

    def __init__(self, d_state: int, n_channels: int, d_hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        )
        self.heads = nn.Linear(d_hidden, 2 * n_channels)   # one value per team
        self.n_channels = n_channels

    def forward(self, state: Tensor) -> Tensor:
        """[B, d_state] -> [B, 2, n_channels]"""
        return self.heads(self.net(state)).view(-1, 2, self.n_channels)
