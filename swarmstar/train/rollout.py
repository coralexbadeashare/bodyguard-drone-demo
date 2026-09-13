"""Rollout collection and the actor-critic update.

Credit assignment is CTDE: the critic predicts one value per (team, reward
channel) from ground truth, so per-drone rewards are aggregated to their team
before returns are computed, and every drone on a team shares that team's
advantage. This is the multi-agent analogue of AlphaStar's setup, where the
value function sees both players but the policy does not.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..model.networks import SwarmPolicy, CentralizedCritic, N_MAIN_HEADS
from ..sim.env import SwarmEnv, REWARD_CHANNELS
from .evaluate import PolicyAgent
from .losses import (vtrace_returns, td_lambda_returns, upgo_returns,
                     multi_head_logp, multi_head_entropy)

HEADS = ("action_type", "delay", "target", "heading", "pitch", "speed", "comms")
COMMS_PG_COEF = 0.1


class RolloutBuffer:
    """Preallocated, time-major. Observations are stored fp16 to fit."""

    def __init__(self, T: int, B: int, N: int, K: int, d_scalar: int,
                 d_entity: int, d_state: int, n_ch: int, device):
        f16 = dict(dtype=torch.float16, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        i64 = dict(dtype=torch.long, device=device)
        self.T, self.B, self.N = T, B, N
        self.scalars = torch.zeros(T, B, N, d_scalar, **f16)
        self.entities = torch.zeros(T, B, N, K, d_entity, **f16)
        self.ent_mask = torch.zeros(T, B, N, K, dtype=torch.bool, device=device)
        self.act_mask = torch.zeros(T, B, N, 7, dtype=torch.bool, device=device)
        self.actions = {}
        self.behaviour_logp = torch.zeros(T, B, N, **f32)
        self.alive = torch.zeros(T, B, N, dtype=torch.bool, device=device)
        self.state = torch.zeros(T + 1, B, d_state, **f32)
        self.rew = torch.zeros(T, B, 2, n_ch, **f32)
        self.disc = torch.zeros(T, B, **f32)
        self.hx0 = None
        self.cx0 = None
        self._i64 = i64

    def alloc_actions(self, sample: dict[str, Tensor]):
        for k, v in sample.items():
            shape = (self.T, *v.shape)
            self.actions[k] = torch.zeros(shape, **self._i64)


def team_rewards(env: SwarmEnv, rew: dict[str, Tensor]) -> Tensor:
    """[B, 2, C] -- per-drone channels averaged over each team's living drones."""
    B, dev = env.B, env.device
    C = len(REWARD_CHANNELS)
    out = torch.zeros(B, 2, C, device=dev)
    alive = env.alive.float()
    for t in range(2):
        sel = (env.team == t).float() * alive
        denom = sel.sum(-1).clamp_min(1.0)
        for c, name in enumerate(REWARD_CHANNELS):
            out[:, t, c] = (rew[name] * sel).sum(-1) / denom
    return out


@torch.no_grad()
def collect(env: SwarmEnv, policy: SwarmPolicy, buf: RolloutBuffer,
            hx: Tensor, cx: Tensor, obs: dict, opponent=None,
            gen: torch.Generator | None = None, learner_team: int = 0):
    """Run T environment steps, filling `buf`. Returns (obs, hx, cx, stats)."""
    B, N, K = env.B, env.N, env.K
    M = B * N
    n_noise = N_MAIN_HEADS + policy.n_comm_tokens
    buf.hx0, buf.cx0 = hx.clone(), cx.clone()
    wins = torch.zeros(3, device=env.device)   # red, blue, draw
    n_done = 0

    for t in range(buf.T):
        am = PolicyAgent.action_mask(obs)
        u = torch.rand(M, n_noise, policy.max_logits, device=env.device,
                       generator=gen).clamp(1e-6, 1 - 1e-6)
        noise = -torch.log(-torch.log(u))
        lg, ac, hx, cx = policy(obs["scalars"].reshape(M, -1),
                                obs["entities"].reshape(M, K, -1),
                                obs["entity_mask"].reshape(M, K), am, hx, cx, noise)

        act = {k: v.view(B, N) for k, v in ac.items() if k != "comms"}
        act["comms"] = ac["comms"].view(B, N, -1)
        if not buf.actions:
            buf.alloc_actions(act)

        logp = multi_head_logp(lg, ac, HEADS, batch_ndim=1).view(B, N)

        buf.scalars[t] = obs["scalars"].half()
        buf.entities[t] = obs["entities"].half()
        buf.ent_mask[t] = obs["entity_mask"]
        buf.act_mask[t] = am.view(B, N, 7)
        for k, v in act.items():
            buf.actions[k][t] = v
        buf.behaviour_logp[t] = logp
        buf.alive[t] = env.alive
        buf.state[t] = env.global_state()

        if opponent is not None:
            # The learner may hold EITHER side. In attack/defend the two sides are
            # different jobs -- strike the base, or stop the strike -- so a policy
            # pinned to red would only ever learn to attack.
            from . import baselines as Bl
            oa = opponent.act(obs)
            act = (Bl.merge(act, oa, env.n) if learner_team == 0
                   else Bl.merge(oa, act, env.n))

        obs, rew, done, info = env.step(act)
        buf.rew[t] = team_rewards(env, rew)
        buf.disc[t] = torch.where(done, 0.0, 1.0)

        if done.any():
            w = info["winner"][done]
            wins[0] += (w == 0).sum(); wins[1] += (w == 1).sum()
            wins[2] += (w == -1).sum()
            n_done += int(done.sum())
            obs = env.reset(done)
            z = done.view(B, 1).expand(B, N).reshape(M, 1)
            hx = torch.where(z, 0.0, hx)
            cx = torch.where(z, 0.0, cx)
            if opponent is not None and hasattr(opponent, "reset"):
                pass                                  # scripted agents are stateless

    buf.state[buf.T] = env.global_state()
    stats = {"episodes": n_done,
             "red_win": (wins[0] / max(n_done, 1)).item(),
             "blue_win": (wins[1] / max(n_done, 1)).item(),
             "draw": (wins[2] / max(n_done, 1)).item()}
    return obs, hx, cx, stats


def learn(policy: SwarmPolicy, critic: CentralizedCritic, opt, buf: RolloutBuffer,
          weights: Tensor, learn_teams: tuple[int, ...], team_of: Tensor,
          gamma: float = 0.995, lam: float = 0.8, ent_coef: float = 3e-3,
          comms_ent_coef: float = 3e-4,
          vf_coef: float = 0.5, upgo_coef: float = 1.0,
          clip_grad: float = 10.0, max_seqs: int = 1024,
          anchor=None, kl_coef: float = 0.0) -> dict:
    """One synchronous actor-critic update over the collected rollout."""
    T, B, N = buf.T, buf.B, buf.N
    K = buf.entities.shape[3]
    dev = buf.scalars.device
    M = B * N

    # ---- critic over the whole trajectory (ground truth, both teams)
    vals = critic(buf.state.reshape((T + 1) * B, -1)).view(T + 1, B, 2, -1)
    disc = buf.disc.unsqueeze(-1).unsqueeze(-1) * gamma                # [T,B,1,1]

    # ---- subsample drone-sequences for the gradient pass.
    # Collection runs wide (many arenas) because the simulator is cheap; the
    # learner cannot hold that many sequences in one autograd graph. Sizing this
    # to the benchmarked figure keeps memory bounded and predictable.
    if max_seqs < M:
        idx = torch.randperm(M, device=dev)[:max_seqs]
    else:
        idx = torch.arange(M, device=dev)
    S = idx.numel()

    # ---- recompute target-policy logits with the CURRENT weights
    hx, cx = buf.hx0[idx], buf.cx0[idx]
    logps, logps_c, ents, ents_c, kls = [], [], [], [], []
    nz = torch.zeros(S, N_MAIN_HEADS + policy.n_comm_tokens,
                     policy.max_logits, device=dev)
    ah, ac_ = (anchor.initial_state(S, dev) if anchor is not None
               else (None, None))
    for t in range(T):
        lg, _, hx, cx = policy(
            buf.scalars[t].reshape(M, -1)[idx].float(),
            buf.entities[t].reshape(M, K, -1)[idx].float(),
            buf.ent_mask[t].reshape(M, K)[idx], buf.act_mask[t].reshape(M, 7)[idx],
            hx, cx, torch.zeros(S, N_MAIN_HEADS + policy.n_comm_tokens,
                                policy.max_logits, device=dev))
        a = {k: v[t].reshape(M, *v.shape[3:])[idx] for k, v in buf.actions.items()}
        if anchor is not None and kl_coef > 0:
            with torch.no_grad():
                alg, _, _, _ = anchor(
                    buf.scalars[t].reshape(M, -1)[idx].float(),
                    buf.entities[t].reshape(M, K, -1)[idx].float(),
                    buf.ent_mask[t].reshape(M, K)[idx],
                    buf.act_mask[t].reshape(M, 7)[idx],
                    ah, ac_, torch.zeros_like(nz), forced=a)
                ah, ac_ = _, _
            kls.append(sum(
                torch.nn.functional.kl_div(
                    torch.log_softmax(lg[h].float(), -1),
                    torch.log_softmax(alg[h].float(), -1),
                    log_target=True, reduction="none").sum(-1)
                for h in HEADS[:-1]))
        logps.append(multi_head_logp(lg, a, HEADS[:-1], batch_ndim=1))
        logps_c.append(multi_head_logp(lg, a, ("comms",), batch_ndim=1))
        ents.append(multi_head_entropy(lg, HEADS[:-1], batch_ndim=1))
        ents_c.append(multi_head_entropy(lg, ("comms",), batch_ndim=1))
        # detach the recurrent state periodically to bound BPTT memory
        if (t + 1) % 16 == 0:
            hx, cx = hx.detach(), cx.detach()
    logp = torch.stack(logps)                    # control heads only  [T,S]
    logp_comms = torch.stack(logps_c)            # the mesh payload    [T,S]
    # The comms head carries 8 tokens from a 64-codebook: 33 of the 45 nats in the
    # joint log-prob. Left at full weight it dominates both the policy gradient
    # and UPGO, so the swarm self-imitates a still-random protocol and the comms
    # entropy collapses before the control heads have learned anything. It stays
    # in the objective -- it must, or the protocol never becomes useful -- but at
    # a weight that keeps it from drowning out flying and shooting.
    logp_joint = logp + COMMS_PG_COEF * logp_comms
    ent = torch.stack(ents)
    ent_comms = torch.stack(ents_c)

    # ---- returns per (team, channel), then collapse channels by weight
    C = buf.rew.shape[-1]
    flat = lambda x: x.permute(1, 2, 3, 0).reshape(-1, T).t()          # [T, B*2*C]
    r_f = flat(buf.rew)
    v_f = vals.permute(1, 2, 3, 0).reshape(-1, T + 1).t()
    d_f = disc.expand(T, B, 2, C).permute(1, 2, 3, 0).reshape(-1, T).t()

    behaviour = buf.behaviour_logp.reshape(T, M)[:, idx]
    logp_full = (logp + logp_comms)              # true joint, for importance ratios
    # one importance ratio per drone, shared by its team's channels
    log_rho_full = (buf.behaviour_logp * 0)                            # [T,B,N]
    log_rho_full.reshape(T, M)[:, idx] = (logp_full.detach() - behaviour)
    rho_team = torch.zeros(T, B, 2, device=dev)
    for tm in range(2):
        sel = (team_of == tm).float().view(1, 1, N)
        rho_team[:, :, tm] = (log_rho_full * sel).sum(-1) / sel.sum().clamp_min(1)
    lr_f = rho_team.unsqueeze(-1).expand(T, B, 2, C).permute(1, 2, 3, 0)\
        .reshape(-1, T).t()

    vs_f, pg_f = vtrace_returns(lr_f, r_f, v_f, d_f)
    up_f = upgo_returns(r_f, v_f, d_f)
    td_f = td_lambda_returns(r_f, v_f, d_f, lam=lam)

    un = lambda x: x.t().reshape(B, 2, C, T).permute(3, 0, 1, 2)       # [T,B,2,C]
    pg_adv, up_ret, td_ret = un(pg_f), un(up_f), un(td_f)

    w = weights.view(1, 1, 1, C)
    adv_team = (pg_adv * w).sum(-1)                                    # [T,B,2]
    upgo_team = ((up_ret - vals[:-1].detach()) * w).sum(-1)

    # ---- broadcast the team advantage back to that team's drones
    onehot = torch.stack([(team_of == 0).float(), (team_of == 1).float()], 0)
    learn_mask = torch.zeros(N, device=dev)
    for tm in learn_teams:
        learn_mask += (team_of == tm).float()
    mask = buf.alive.float() * learn_mask.view(1, 1, N)

    adv_drone = torch.einsum("tbk,kn->tbn", adv_team, onehot).reshape(T, M)[:, idx]
    upgo_drone = torch.einsum("tbk,kn->tbn", upgo_team, onehot).reshape(T, M)[:, idx]
    mask = mask.reshape(T, M)[:, idx]

    denom = mask.sum().clamp_min(1.0)
    # Normalise over EXACTLY the set the loss sums over. Centring on all drones
    # and then masking leaves a residual mean, which multiplies against an
    # almost-constant initial logp (~-45, dominated by the 8-token comms head)
    # and shows up as a steadily diverging pg_loss that is pure offset.
    sel = mask > 0
    a_sel = adv_drone[sel]
    adv_n = (adv_drone - a_sel.mean()) / a_sel.std().clamp_min(1e-6)
    pg_loss = -(logp_joint * adv_n.detach() * mask).sum() / denom
    # Scale the UPGO advantage too. It is a raw discounted return -- O(30) here --
    # and multiplying it by logp (~-45 at init, dominated by the comms head) made
    # it swamp the policy gradient by three orders of magnitude. Divide by the
    # masked std WITHOUT centring: UPGO's meaning depends on the sign, since only
    # better-than-expected outcomes are imitated.
    u_sel = upgo_drone[sel]
    upgo_n = upgo_drone / u_sel.std().clamp_min(1e-6)
    # UPGO imitates the control heads only: self-imitating a random protocol is
    # noise, not signal
    upgo_loss = -(logp * upgo_n.detach().clamp(min=0) * mask).sum() / denom
    ent_loss = -(ent * mask).sum() / denom
    ent_c_loss = -(ent_comms * mask).sum() / denom
    v_loss = 0.5 * (vals[:-1] - td_ret.detach()).pow(2).mean()

    # KL toward the supervised policy. AlphaStar: "Agents also receive a penalty
    # whenever their action probabilities differ from the supervised policy."
    # It is what keeps RL exploring around competent play instead of wandering
    # off into the degenerate strategies this project collapsed into twice.
    kl_loss = torch.stack(kls).mul(mask).sum() / denom if kls else \
        torch.zeros((), device=dev)
    loss = (pg_loss + upgo_coef * upgo_loss + vf_coef * v_loss
            + ent_coef * ent_loss + comms_ent_coef * ent_c_loss
            + kl_coef * kl_loss)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(
        list(policy.parameters()) + list(critic.parameters()), clip_grad)
    opt.step()

    return {"loss": loss.item(), "pg_loss": pg_loss.item(),
            "upgo_loss": upgo_loss.item(), "v_loss": v_loss.item(),
            "upgo_raw_std": u_sel.std().item(),
            "entropy": ent[sel].mean().item(),
            "entropy_comms": ent_comms[sel].mean().item(),
            "kl_anchor": float(kl_loss), "grad_norm": gn.item(),
            "adv_std": a_sel.std().item(), "adv_mean": a_sel.mean().item(),
            "value_mean": vals.mean().item()}
