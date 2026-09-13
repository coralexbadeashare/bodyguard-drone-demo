"""Head-to-head evaluation. The only evidence that counts.

Win rates are computed on held-out seeds against opponents the policy never
trained on, because a training curve is not evidence of skill.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..model.networks import SwarmPolicy, N_MAIN_HEADS
from ..sim.env import SwarmEnv, MODE_WAR, ENGAGE, STRIKE_BASE
from . import baselines as B


class PolicyAgent:
    """Wraps a SwarmPolicy so it plugs into the same match loop as a baseline."""
    name = "policy"

    def __init__(self, env: SwarmEnv, team: int, policy: SwarmPolicy,
                 greedy: bool = False, seed: int = 0):
        self.env, self.team, self.pol = env, team, policy
        self.M = env.B * env.N
        self.dev = env.device
        self.greedy = greedy
        self.n_noise = N_MAIN_HEADS + policy.n_comm_tokens
        self.g = torch.Generator(device=self.dev).manual_seed(seed)
        self.reset()

    def reset(self):
        self.hx, self.cx = self.pol.initial_state(self.M, self.dev)

    def _noise(self):
        if self.greedy:
            return torch.zeros(self.M, self.n_noise, self.pol.max_logits,
                               device=self.dev)
        u = torch.rand(self.M, self.n_noise, self.pol.max_logits,
                       device=self.dev, generator=self.g).clamp(1e-6, 1 - 1e-6)
        return -torch.log(-torch.log(u))

    @staticmethod
    def action_mask(obs, n_action_types: int = 7) -> Tensor:
        """Macro-actions that make no sense right now are masked out.

        ENGAGE and EVADE need something to engage or evade. Masking is cheaper
        than hoping the policy learns not to pick them.
        """
        B_, N_ = obs["entity_mask"].shape[0], obs["entity_mask"].shape[1]
        M = B_ * N_
        has_contact = obs["entity_mask"].any(-1).reshape(M)
        m = torch.ones(M, n_action_types, dtype=torch.bool,
                       device=obs["entity_mask"].device)
        m[:, ENGAGE] = has_contact
        m[:, 4] = has_contact                      # EVADE
        # NOT masked: STRIKE_BASE when the enemy base is already gone.
        # It looks like a dead action -- 34-49% of defender choices aimed at a
        # base with 0 HP -- but the attacker's base is also the axis attackers
        # arrive along, so the policy repurposed it as "move toward the threat".
        # Masking it measured DEF greedy 0.42 -> 0.51 but DEF saturation
        # 0.95 -> 0.30 and DEF rush 0.30 -> 0.00. Net worse; left in place.
        return m

    def act(self, obs):
        env, K = self.env, self.env.K
        M = self.M
        with torch.no_grad():
            lg, ac, self.hx, self.cx = self.pol(
                obs["scalars"].reshape(M, -1), obs["entities"].reshape(M, K, -1),
                obs["entity_mask"].reshape(M, K), self.action_mask(obs),
                self.hx, self.cx, self._noise())
        out = {k: v.view(env.B, env.N) for k, v in ac.items() if k != "comms"}
        out["comms"] = ac["comms"].view(env.B, env.N, -1)
        return out


@torch.no_grad()
def play_match(env: SwarmEnv, red, blue, mode: int | None = MODE_WAR,
               max_steps: int | None = None) -> dict:
    """Run one batch of arenas to completion. Returns aggregate outcomes."""
    obs = env.reset()
    if mode is not None:
        from ..sim.env import MODE_ATTACK_DEFEND
        env.mode.fill_(mode)
        env.base_health.fill_(env.cfg.arena.base_health)
        if mode == MODE_ATTACK_DEFEND:
            # only blue holds a base: red attacks, blue survives
            env.base_health[:, 0] = 0.0
        obs = env.observe()
    for a in (red, blue):
        if hasattr(a, "reset"):
            a.reset()

    B_ = env.B
    dev = env.device
    settled = torch.zeros(B_, dtype=torch.bool, device=dev)
    winner = torch.full((B_,), -1, dtype=torch.long, device=dev)
    steps = max_steps or env.max_steps

    check_every = 8            # `settled.all()` forces a GPU sync; do it rarely
    for i in range(steps):
        act = B.merge(red.act(obs), blue.act(obs), env.n)
        obs, rew, done, info = env.step(act)
        fresh = done & ~settled
        winner = torch.where(fresh, info["winner"], winner)
        settled |= done
        if i % check_every == 0 and bool(settled.all()):
            break

    red_w = (winner == 0).float().mean().item()
    blue_w = (winner == 1).float().mean().item()
    return {"red_win": red_w, "blue_win": blue_w, "draw": 1.0 - red_w - blue_w,
            "decided": settled.float().mean().item(),
            "red_alive": (env.alive & (env.team == 0)).float().sum(-1).mean().item(),
            "blue_alive": (env.alive & (env.team == 1)).float().sum(-1).mean().item()}


@torch.no_grad()
def score_roles(env: SwarmEnv, policy: SwarmPolicy,
                names=("do_nothing", "random", "greedy", "rush", "perimeter",
                       "saturation", "interceptor", "obs_interceptor"),
                seed: int = 12345, greedy_policy: bool = False) -> dict:
    """Score the policy in BOTH attack/defend roles.

    Attacking and defending are different jobs, and one number cannot tell you
    whether the policy can do both. A policy that only ever plays red would post a
    fine average while being unable to defend at all.

    `interceptor` was added once the gun was removed: the pure-pursuit defenders
    score DEF 0.41 (perimeter) and 0.52 (greedy) against a rush, barely above
    do_nothing's 0.00, so `atk_*` against them was scored on opponents that do
    not meaningfully defend.

    Read `atk_obs_interceptor`, not `atk_interceptor`. `interceptor` engages
    attackers it has not seen, which no exported policy can do, so beating it is
    not a reachable target -- a 203k-step league sat at 0.00 against it. Both are
    kept: one is the realizable bar, the other the ceiling with perfect
    information. Both are DEFENCES, so neither is scored in the `def` role, where
    they would hunt the other swarm instead of striking the base.
    """
    from ..sim.env import MODE_ATTACK_DEFEND
    out = {}
    for role, team in (("atk", 0), ("def", 1)):
        worst = 1.0
        for nm in names:
            # `interceptor` is a DEFENCE. Cast as the attacker it hunts the other
            # swarm instead of striking the base, which no defender has to stop --
            # it scored def_interceptor = 1.00 on a policy simultaneously scoring
            # def_saturation = 0.00. Scoring it in that role measures nothing and
            # costs a match per eval.
            if nm in ("interceptor", "obs_interceptor") and team == 1:
                continue
            env.gen.manual_seed(seed)
            agent = PolicyAgent(env, team, policy, greedy=greedy_policy, seed=seed)
            opp = B.make(nm, env, 1 - team)
            r = play_match(env, agent if team == 0 else opp,
                           opp if team == 0 else agent, mode=MODE_ATTACK_DEFEND)
            sc = (r["red_win"] if team == 0 else r["blue_win"]) + 0.5 * r["draw"]
            out[f"{role}_{nm}"] = sc
            worst = min(worst, sc)
        out[f"worst_{role}"] = worst
    out["worst"] = min(out["worst_atk"], out["worst_def"])
    return out


@torch.no_grad()
def score_against_baselines(env: SwarmEnv, policy: SwarmPolicy,
                            names=("do_nothing", "random", "greedy", "rush",
                                   "perimeter"),
                            seed: int = 12345, greedy_policy: bool = False) -> dict:
    """Phase gate: the policy must beat every one of these on held-out seeds.

    Evaluated with SAMPLING, not argmax. A stochastic policy and its greedy
    projection are different policies: training reported red_win=1.00 while a
    greedy eval reported 0.50 on the same weights, because argmax over a
    high-entropy multi-head action collapses onto a degenerate joint action that
    was never actually played. Gate on the policy we are training.
    """
    out = {}
    for nm in names:
        env.gen.manual_seed(seed)                 # held-out, fixed across agents
        agent = PolicyAgent(env, 0, policy, greedy=greedy_policy, seed=seed)
        opp = B.make(nm, env, 1)
        r = play_match(env, agent, opp)
        out[f"vs_{nm}"] = r["red_win"] + 0.5 * r["draw"]
    out["worst"] = min(out.values())
    return out
