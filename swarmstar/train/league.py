"""League training -- AlphaStar's multi-agent mechanism, sized for one GPU.

Self-play alone chases cycles: A beats B beats C beats A, forever, with no
progress. This project has seen it twice. The league is the fix, and it has three
kinds of player:

  MAIN             the agent we actually want. Trains against a PFSP mixture of
                   every past player, plus some genuine self-play.
  MAIN_EXPLOITER   hunts the CURRENT main agent's specific weaknesses. Reset once
                   it has beaten the main agent, so it keeps finding new holes
                   rather than settling into one.
  LEAGUE_EXPLOITER hunts weaknesses common to the WHOLE league -- strategies
                   nobody covers.

Exploiters are never the deliverable. Their job is to keep the main agent honest,
and their discoveries reach it because their snapshots join the league the main
agent trains against.

PFSP (prioritised fictitious self-play) samples an opponent in proportion to how
hard it is to beat, so training time is spent where the agent is actually losing
instead of being wasted re-beating players it already dominates.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Literal

import torch

PlayerType = Literal["main", "main_exploiter", "league_exploiter"]


def pfsp_weights(win_rates: list[float], mode: str = "hard",
                 p: float = 2.0) -> list[float]:
    """Opponent-sampling weights from win rates.

    `hard`  (1-x)^p   -- concentrate on opponents we lose to
    `even`  x(1-x)    -- concentrate on close matchups, which teach the most
    """
    if not win_rates:
        return []
    if mode == "even":
        w = [max(x * (1.0 - x), 0.0) for x in win_rates]
    else:
        w = [max(1.0 - x, 0.0) ** p for x in win_rates]
    tot = sum(w)
    if tot <= 1e-9:                      # dominating everything: sample uniformly
        return [1.0 / len(w)] * len(w)
    return [x / tot for x in w]


@dataclass
class Player:
    """A frozen snapshot in the league, or a fixed scripted strategy."""
    pid: int
    ptype: PlayerType
    step: int
    state: dict
    parent: int | None = None
    scripted: str | None = None      # name of a baseline agent, if not a network

    def name(self) -> str:
        if self.scripted:
            return f"[{self.scripted}]"
        short = {"main": "MA", "main_exploiter": "ME", "league_exploiter": "LE"}
        return f"{short[self.ptype]}{self.pid}@{self.step//1000}k"


@dataclass
class Learner:
    """An agent currently being trained."""
    ptype: PlayerType
    policy: torch.nn.Module
    critic: torch.nn.Module
    opt: torch.optim.Optimizer
    steps: int = 0
    since_reset: int = 0
    # win rate against each frozen player, by pid
    wins: dict[int, float] = field(default_factory=dict)
    games: dict[int, float] = field(default_factory=dict)

    def win_rate(self, pid: int, prior: float = 0.5, prior_n: float = 4.0) -> float:
        """Smoothed win rate; an unplayed opponent starts at even odds."""
        w, n = self.wins.get(pid, 0.0), self.games.get(pid, 0.0)
        return (w + prior * prior_n) / (n + prior_n)

    def record(self, pid: int, score: float, n: float = 1.0):
        self.wins[pid] = self.wins.get(pid, 0.0) + score * n
        self.games[pid] = self.games.get(pid, 0.0) + n


class League:
    """Frozen player pool plus the matchmaking rules."""

    def __init__(self, bootstrap: dict, max_players: int = 40,
                 snapshot_every: int = 400, device="cuda"):
        self.device = device
        self.max_players = max_players
        self.snapshot_every = snapshot_every
        self.bootstrap = {k: v.detach().cpu().clone() for k, v in bootstrap.items()}
        self.players: list[Player] = []
        self._next = 0
        self.add(self.bootstrap, "main", 0)      # the league must start somewhere
        self._ensure_scripted()

    # -------------------------------------------------------------- players
    def add(self, state: dict, ptype: PlayerType, step: int,
            parent: int | None = None, scripted: str | None = None) -> Player:
        pl = Player(self._next, ptype, step,
                    {} if scripted else
                    {k: v.detach().cpu().clone() for k, v in state.items()},
                    parent, scripted)
        self.players.append(pl)
        self._next += 1
        if len(self.players) > self.max_players:
            # scripted strategies are PERMANENT: they encode real doctrine and a
            # main agent that quietly stops being able to beat them has
            # regressed, however good it looks against its own recent selves
            fixed = [p for p in self.players if p.scripted or p.pid == 0]
            learned = [p for p in self.players if not (p.scripted or p.pid == 0)]
            keep = self.max_players - len(fixed)
            self.players = fixed + learned[-max(keep, 4):]
        return pl

    #: Doctrine-realistic fixed strategies. `saturation` is a multi-axis,
    #: time-on-target massed attack -- the pattern real defences exist to stop --
    #: so the main agent must keep beating it, not merely its own past selves.
    #: `interceptor` is the only competent DEFENCE in the pool: measured on the
    #: guard scenario it keeps 0.372 of the base against a rush where `greedy`
    #: keeps 0.119 and `perimeter` 0.042 (do-nothing keeps 0.000, see
    #: docs/GUARD_REFERENCE.md). Without it an attacking learner faces nothing
    #: that can actually stop it and never has to learn to beat a real defence.
    SCRIPTED = ("saturation", "rush", "greedy", "perimeter", "interceptor")

    def _ensure_scripted(self):
        have = {p.scripted for p in self.players if p.scripted}
        for nm in self.SCRIPTED:
            if nm not in have:
                self.add({}, "main", 0, scripted=nm)

    def of_type(self, ptype: PlayerType) -> list[Player]:
        return [p for p in self.players if p.ptype == ptype]

    # ---------------------------------------------------------- matchmaking
    def sample_opponent(self, learner: Learner, rng: torch.Generator,
                        p_self: float = 0.30) -> tuple[Player | None, str]:
        """Pick an opponent per the learner's role. `None` means true self-play.

        `p_self` was hardcoded at 0.30 here while PHASES carried a `p_self` key
        that only the non-league training loop read. Phase 5 set it to 0.0 to stop
        the defence-only learner meeting itself, and self-play carried on at 30%
        regardless -- the setting was silently inert for a full training run.
        """
        u = torch.rand(1, generator=rng).item()

        if learner.ptype == "main":
            if u < p_self:
                return None, "self"
            # A GUARANTEED slice against fixed doctrine. Uniform PFSP gave this
            # only ~8% of updates, because the four scripted agents are a small
            # fraction of a league dominated by the agent's own snapshots -- and
            # every one of those snapshots inherits the same blind spot. A league
            # can only teach what is inside it, so the one strategy family we
            # measurably lose to has to be sampled deliberately, not incidentally.
            if u < 0.55:
                fixed = [p for p in self.players if p.scripted]
                if fixed:
                    return self._pfsp(learner, fixed, rng, "hard"), "doctrine"
            pool = [p for p in self.players if not p.scripted] or self.players
            if u > 0.90 and len(pool) > 4:
                pool = pool[: max(2, len(pool) // 2)]      # older half
            return self._pfsp(learner, pool, rng, "hard"), "pfsp"

        if learner.ptype == "main_exploiter":
            mains = self.of_type("main") or self.players
            # target the most recent main agents: their current holes, not old ones
            recent = mains[-3:]
            return self._pfsp(learner, recent, rng, "hard"), "vs_main"

        # league exploiter: anything the league as a whole struggles to cover
        return self._pfsp(learner, self.players, rng, "hard"), "pfsp_all"

    def _pfsp(self, learner: Learner, pool: list[Player],
              rng: torch.Generator, mode: str) -> Player:
        wr = [learner.win_rate(p.pid) for p in pool]
        w = pfsp_weights(wr, mode)
        idx = int(torch.multinomial(torch.tensor(w), 1, generator=rng).item())
        return pool[idx]

    # -------------------------------------------------------------- resets
    def should_reset(self, learner: Learner, threshold: float = 0.70,
                     max_steps: int = 3000) -> bool:
        """An exploiter that has done its job -- or stalled -- is recycled."""
        if learner.ptype == "main":
            return False
        if learner.since_reset >= max_steps:
            return True
        pool = (self.of_type("main")[-3:] if learner.ptype == "main_exploiter"
                else self.players)
        played = [p for p in pool if learner.games.get(p.pid, 0) >= 2]
        if len(played) < max(1, len(pool) // 2):
            return False
        return min(learner.win_rate(p.pid) for p in played) > threshold

    def reset_learner(self, learner: Learner, step: int) -> Player:
        """Snapshot what the exploiter learned, then send it back to the start."""
        snap = self.add(learner.policy.state_dict(), learner.ptype, step)
        learner.policy.load_state_dict(
            {k: v.to(self.device) for k, v in self.bootstrap.items()})
        for g in learner.opt.param_groups:          # fresh optimizer moments
            for prm in g["params"]:
                learner.opt.state.pop(prm, None)
        learner.wins.clear(); learner.games.clear()
        learner.since_reset = 0
        return snap

    # ------------------------------------------------------------ reporting
    def summary(self, learners: list[Learner]) -> str:
        counts = {t: len(self.of_type(t)) for t in
                  ("main", "main_exploiter", "league_exploiter")}
        parts = [f"players={len(self.players)} "
                 f"(MA {counts['main']} / ME {counts['main_exploiter']} "
                 f"/ LE {counts['league_exploiter']})"]
        for L in learners:
            played = [p for p in self.players if L.games.get(p.pid, 0) >= 1]
            if played:
                mean = sum(L.win_rate(p.pid) for p in played) / len(played)
                worst = min(L.win_rate(p.pid) for p in played)
                tag = {"main": "MA", "main_exploiter": "ME",
                       "league_exploiter": "LE"}[L.ptype]
                parts.append(f"{tag}: vs-league mean {mean:.2f} "
                             f"worst {worst:.2f} ({len(played)} opp)")
        return " | ".join(parts)

    def state_dict(self) -> dict:
        return {"players": [(p.pid, p.ptype, p.step, p.state, p.parent, p.scripted)
                            for p in self.players],
                "next": self._next, "bootstrap": self.bootstrap}

    def load_state_dict(self, d: dict):
        self.players = [Player(*t) for t in d["players"]]
        self._next = d["next"]
        self.bootstrap = d["bootstrap"]
        # A checkpoint written before the scripted strategies existed would drop
        # them on resume -- which is exactly what happened: the main agent played
        # them 0 times in 4,048 updates and its defence against `rush` collapsed.
        self._ensure_scripted()
