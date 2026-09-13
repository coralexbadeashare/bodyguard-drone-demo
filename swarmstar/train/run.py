"""SwarmStar training entry point.

One invocation = one night. It gates on GPU availability, resumes from the last
checkpoint, trains until the wall-clock budget expires, checkpoints, and exits.
Run it again the next night and it continues exactly where it stopped.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from swarmstar.sim.config import SimConfig
from swarmstar.sim.env import SwarmEnv, REWARD_CHANNELS, MODE_WAR
from swarmstar.model.networks import SwarmPolicy, CentralizedCritic
from swarmstar.train import baselines as B
from swarmstar.train.evaluate import PolicyAgent, score_roles, play_match
from swarmstar.train.rollout import RolloutBuffer, collect, learn
from swarmstar.train.session import SessionConfig, TrainingSession
from swarmstar.train.league import League, Learner
from swarmstar.train.bc import train_bc

# --------------------------------------------------------------- curriculum
PHASES = {
    0: dict(
        name="imitate",
        # AlphaStar's supervised stage. It trained on 971k human replays before
        # any RL; we have none, so the scripted agents are the demonstrators.
        # The resulting policy is both the RL starting point AND the KL anchor.
        weights={}, opponents=[None], learn_teams=(0, 1),
        gate={"atk_do_nothing": 0.70, "def_rush": 0.40},
        next=4,
    ),
    1: dict(
        name="nav",
        # Cross the arena and reach the enemy base without crashing or leaving
        # the box. No combat pressure yet -- learn to fly first.
        # `progress` is the dense driver; `base_damage` is scaled to be worth
        # having (it is normalised by 1000 base HP, so a weight of 5 was worth
        # 0.017/tick against a collision penalty of 1.0 -- the reward's optimum
        # was to sit still).
        # `energy` is deliberately ZERO here: it charges per unit speed, which is
        # a direct subsidy for dawdling. Time pressure replaces it.
        weights={"progress": 1.0, "base_damage": 400.0, "time": 0.1,
                 "collision": 0.2, "boundary": 0.3, "energy": 0.0},
        opponents=["do_nothing"],
        learn_teams=(0,),
        gate={"atk_do_nothing": 0.80},
        # Skip phase 2. Grinding against scripted opponents alternates the
        # learner between attacker and defender every iteration, so each gradient
        # step fights the previous one -- measured 2026-09-03: 183 evaluations,
        # DEF-vs-rush pinned at exactly 0.000 and ATK oscillating 0.00-0.88 with
        # no trend, over 9 hours. Self-play trains BOTH roles in one batch
        # (learn_teams=(0,1)), and it is what AlphaStar actually improves with;
        # its supervised stage only had to bootstrap. The scripted agents stay on
        # as evaluation, which is the job they are good at.
        next=3,
    ),
    2: dict(
        name="combat_scripted",
        # `win` is the actual objective, so nothing may dwarf it. At
        # base_damage=200 vs win=10 the policy optimised base chip damage and
        # base preservation instead of winning, learned to turtle, and then
        # collapsed to a degenerate deterministic strategy it could not escape.
        # `time` and `energy: 0` matter as much here as in phase 1: without them
        # the swarm is paid to fly slowly and re-learns to creep, which is exactly
        # what happened on 2026-08-29 (atk_do_nothing fell 1.00 -> 0.32 the moment
        # training crossed from phase 1 into phase 2).
        weights={"win": 20.0, "base_damage": 60.0, "base_preserved": 60.0,
                 "kills": 3.0, "losses": 3.0, "collision": 0.2,
                 "boundary": 0.3, "energy": 0.0, "progress": 0.2, "time": 0.05},
        opponents=["do_nothing", "random", "greedy", "rush", "perimeter"],
        learn_teams=(0,),
        # A BOOTSTRAP gate, not a mastery gate. The scripted ceiling for
        # attacking a competent defender is 0.18-0.67, so demanding 0.50 in every
        # cell was unreachable by construction and burned days of GPU time.
        # AlphaStar's supervised stage did not have to beat everything either --
        # it only had to produce a sane policy for the league to improve.
        gate={"atk_do_nothing": 0.60, "def_rush": 0.30},
    ),
    3: dict(
        name="self_play",
        # Opponents are drawn from a POOL of frozen past snapshots as well as the
        # live policy. Pure self-play chases cycles -- A beats B beats C beats A --
        # which is what the previous night did: it hit worst=0.664, switched to
        # self-play, and fell to 0.047 over the next 5,000 steps. This is
        # league training in miniature, and the reason AlphaStar needs one.
        pool=True, snapshot_every=250, p_self=0.6,
        # `time` and `energy: 0` matter as much here as in phase 1: without them
        # the swarm is paid to fly slowly and re-learns to creep, which is exactly
        # what happened on 2026-08-29 (atk_do_nothing fell 1.00 -> 0.32 the moment
        # training crossed from phase 1 into phase 2).
        weights={"win": 20.0, "base_damage": 60.0, "base_preserved": 60.0,
                 "kills": 3.0, "losses": 3.0, "collision": 0.2,
                 "boundary": 0.3, "energy": 0.0, "progress": 0.2, "time": 0.05},
        opponents=[None],                      # None = play against ourselves
        learn_teams=(0, 1),
        # same reasoning as phase 2: `worst` includes cells whose scripted
        # ceiling is 0.18, so gate on the two that measure real competence.
        gate={"atk_do_nothing": 0.80, "def_rush": 0.50},
    ),
}


PHASES[4] = dict(
    name="league",
    weights=PHASES[3]["weights"],
    opponents=[None],
    learn_teams=(0, 1),
    # No gate: the league is the end of the curriculum. It runs until the clock
    # stops, and progress is read from the main agent's score against the frozen
    # baselines and against its own past selves.
    gate={},
    snapshot_every=400,
)


class StallGuard:
    """Shout when a phase stops making progress.

    Phase 2 ran 9 hours and 183 evaluations with DEF-vs-rush pinned at exactly
    0.000 before anyone looked. On a week-long unattended run that is the single
    most expensive failure mode, so it now announces itself.
    """

    def __init__(self, patience: int = 8, min_delta: float = 0.02):
        self.best = -1e9
        self.since = 0
        self.patience = patience
        self.min_delta = min_delta

    def update(self, score: float, step: int, log) -> bool:
        if score > self.best + self.min_delta:
            self.best, self.since = score, 0
            return False
        self.since += 1
        if self.since >= self.patience:
            log(f"[STALL] step {step:,}: no improvement over {self.since} "
                f"evaluations (best {self.best:.3f}). This phase is not learning "
                f"-- change something rather than spending more of the week.")
            self.since = 0
            return True
        return False


def publish(policy, run_dir, log):
    """Export the current best policy to the browser demo.

    Makes progress watchable: reload the page and you are looking at the newest
    verified-best weights, not a stale snapshot.
    """
    try:
        from swarmstar.export.to_onnx import export
        import copy
        cpu = copy.deepcopy(policy).cpu().eval()
        export(cpu, "/workspace/DroneSwarmRL/browser/policy.onnx")
        log("[publish] browser/policy.onnx updated to the current best")
    except Exception as e:                     # never let publishing kill a run
        log(f"[publish] skipped: {e}")


PHASES[5] = dict(
    name="guard",
    # THE REAL TARGET. The base is a person walking at 1.4 m/s; 2 defenders must
    # stop 3 incoming drones reaching them. Asymmetric, small, and defence-first.
    weights={"win": 20.0, "base_preserved": 120.0, "kills": 6.0, "losses": 2.0,
             "base_damage": 10.0, "collision": 0.2, "boundary": 0.3,
             "energy": 0.0, "progress": 0.05, "time": 0.02},
    opponents=[None],
    # ONLY THE DEFENDER LEARNS. Measured 2026-09-11: with learn_teams=(0,1) the
    # shared policy went ATK do_nothing 0.97 -> 0.00 in 16k steps while DEF rush
    # stayed at 0.06. base_preserved (120) outweighs base_damage (10) 12:1, so a
    # policy that plays both teams maximises the objective by simply never
    # attacking -- and once the attacker is gone, self-play hands the defender a
    # harmless opponent and the defence stops improving. Freezing the attacker
    # removes the degenerate optimum: our gradient can no longer destroy the
    # thing that is supposed to be applying the pressure.
    learn_teams=(1,),
    defend_frac=1.0,                 # the learner always flies the defence
    # NO self-play and NO snapshots. With learn_teams=(1,) every snapshot is a
    # DEFENCE-only policy, and the league hands those back as ATTACKERS -- which
    # they are hopeless at, because this phase never trains attack. Measured
    # 2026-09-11: the learner met `rush` in 17 of 437 matchups (3.9%) and spent
    # 85% against self/snapshots that cannot attack, so it optimised for them.
    # def_rush peaked at 0.34 (step 1.5k) and decayed to 0.00 by step 8.5k while
    # def_greedy climbed to 1.00. Same degenerate loop as the role collapse, one
    # level up.
    #
    # The opponent pool is therefore fixed and competent: the scripted attackers
    # plus MA0, the phase-3/4 seed, which scores atk_do_nothing 0.97.
    pool=True, snapshot_every=10**9, p_self=0.0,
    # The deliverable of this phase is the DEFENCE, so select checkpoints on it.
    # The old key (atk_do_nothing + def_rush) froze best_policy.pt at step 250
    # once ATK hit 0: nothing was published for 16k steps.
    select=("def_rush", "def_saturation"),
    gate={},
    # base_health measured 2026-09-10: 3 attackers can deliver at most 330 dmg,
    # so the 900 HP fortress made defence FREE (DEF = 1.00 everywhere). At 240 the
    # attacker needs ~2.2 strikes of 3 and scores ~0.5 -- the defender has to work.
    # layout="azimuth": point the second camera at the HORIZON, not the ground.
    # The default front+down layout spends half a counter-UAS sentry's sensing on
    # the dirt. Measured 2026-09-11 on the scripted defence, +0.13 / +0.08 / +0.09
    # DEF at 2 / 3 / 4 defenders. Free -- no action-space or policy change.
    env_cfg=dict(n_active_red=3, n_active_blue=2, base_speed=1.4,
                 base_health=240.0, p_attack_defend=1.0, layout="azimuth"),
)


PHASES[6] = dict(
    name="imitate_guard",
    # AlphaStar's supervised stage, run IN the bodyguard scenario rather than the
    # 16v16 default. Phase 0 clones teachers in a symmetric arena; the thing we
    # actually ship defends one walking person against three attackers, and the
    # geometry of that is different enough that the phase-0 policy transfers
    # badly (guard phase 5 started from it at DEF rush 0.09 and never improved).
    #
    # Evidence for cloning rather than pushing RL harder: 203k league steps left
    # atk_interceptor at 0.00, and 16k guard steps moved def_rush 0.09 -> 0.06.
    # Lead pursuit is a precise geometric skill; AlphaStar's whole premise is
    # that you imitate competence first and let RL improve it, not discover it.
    weights={}, opponents=[None], learn_teams=(0, 1),
    gate={"def_rush": 0.70},
    env_cfg=dict(n_active_red=3, n_active_blue=2, base_speed=1.4,
                 base_health=240.0, p_attack_defend=1.0),
    next=5,
)


def weight_vector(w: dict[str, float], device) -> torch.Tensor:
    return torch.tensor([w.get(c, 0.0) for c in REWARD_CHANNELS], device=device)


def train_league(s, a, cfg, env, eval_env, dev, state, policy, critic, opt):
    """Phase 4. Three learners share one GPU by taking turns.

    AlphaStar ran these agents in parallel on TPU pods. With a single card they
    round-robin instead: one gradient step for one learner per iteration. The
    league mechanics -- PFSP matchmaking, exploiter resets, snapshotting -- are
    unchanged; only the scheduling differs.
    """
    from swarmstar.model.networks import SwarmPolicy, CentralizedCritic
    phase = PHASES[a.phase]
    w = weight_vector(phase["weights"], dev)
    team_of = env.team[0]
    d_state = env.global_state().shape[-1]
    n_ch = len(REWARD_CHANNELS)

    def fresh(seed_state):
        pol = SwarmPolicy().to(dev); pol.load_state_dict(seed_state)
        cri = CentralizedCritic(d_state, n_ch).to(dev)
        o = torch.optim.Adam(list(pol.parameters()) + list(cri.parameters()), lr=a.lr)
        return pol, cri, o

    seed = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    anchor = None
    apath = Path(a.run_dir or ".") / "phase0_passed.pt"
    if not apath.exists():                       # fall back to the best clone
        alt = Path(a.run_dir or ".") / "bc_policy.pt"
        if alt.exists():
            apath = alt
    if apath.exists() and a.kl_coef > 0:
        anchor = SwarmPolicy().to(dev)
        anchor.load_state_dict(torch.load(apath, map_location=dev))
        anchor.eval()
        for prm in anchor.parameters():
            prm.requires_grad_(False)
        s._log(f"[league] KL anchor = {apath.name} (coef {a.kl_coef})")
    league = League(seed, snapshot_every=phase["snapshot_every"], device=dev)
    learners = []
    for t in ("main", "main_exploiter", "league_exploiter"):
        pol, cri, o = fresh(seed)
        learners.append(Learner(t, pol, cri, o))
    if state.get("league"):
        league.load_state_dict(state["league"])
        if a.reseed_learners:
            # Keep the league's accumulated opponents, but restart the learners
            # from a known-good policy. Used to undo a regression without
            # throwing away the opponent diversity that took hours to build.
            w0 = torch.load(a.reseed_learners, map_location=dev)
            for L in learners:
                L.policy.load_state_dict(w0)
                L.wins.clear(); L.games.clear(); L.since_reset = 0
            s._log(f"[league] learners reseeded from "
                   f"{Path(a.reseed_learners).name}; league history kept "
                   f"({len(league.players)} players)")
            state.pop("learner_states", None)
        for L, sd in zip(learners, state.get("learner_states", [])):
            L.policy.load_state_dict(sd["policy"]); L.critic.load_state_dict(sd["critic"])
            L.opt.load_state_dict(sd["opt"]); L.wins = sd["wins"]; L.games = sd["games"]
            L.since_reset = sd["since_reset"]
        s._log(f"[league] resumed: {league.summary(learners)}")
    else:
        s._log(f"[league] seeded from the phase-3 policy; 3 learners "
               f"(main / main-exploiter / league-exploiter)")

    opp_policy = SwarmPolicy().to(dev)
    rng = torch.Generator(device="cpu").manual_seed(1234 + state["step"])
    gen = torch.Generator(device=dev).manual_seed(99 + state["step"])
    stall = StallGuard()
    buf = RolloutBuffer(a.unroll, env.B, env.N, env.K, 28, 13, d_state, n_ch, dev)
    obs = env.reset()
    dfrac = phase.get("defend_frac", a.defend_frac)
    sel_keys = phase.get("select", ("atk_do_nothing", "def_rush"))
    hxs = [policy.initial_state(env.B * env.N, dev) for _ in learners]
    step = state["step"]
    best = state.get("best", -1.0)
    t_last = time.time()

    while s.should_continue():
        L = learners[step % len(learners)]
        hx, cx = hxs[step % len(learners)]
        # Attack is AT CEILING on four of six opponents (1.00 vs do_nothing,
        # random, rush, saturation) while defence sits at 0.17 against a 0.99
        # ceiling (greedy) and 0.31 against 0.85 (rush). Alternating 50/50 was
        # right while both roles had headroom; it no longer is.
        lt = 1 if torch.rand(1, generator=rng).item() < dfrac else 0
        opp_player, kind = league.sample_opponent(L, rng, phase.get("p_self", 0.30))
        if opp_player is None:
            opp, teams, oname = None, phase["learn_teams"], "self"
        elif opp_player.scripted:
            opp = B.make(opp_player.scripted, env, 1 - lt)
            teams, oname = (lt,), opp_player.name()
        else:
            opp_policy.load_state_dict(
                {k: v.to(dev) for k, v in opp_player.state.items()})
            opp_policy.eval()
            opp = PolicyAgent(env, 1 - lt, opp_policy, greedy=False, seed=step)
            teams, oname = (lt,), opp_player.name()

        obs, hx, cx, rst = collect(env, L.policy, buf, hx, cx, obs,
                                   opponent=opp, gen=gen, learner_team=lt)
        hxs[step % len(learners)] = (hx.detach(), cx.detach())
        m = learn(L.policy, L.critic, L.opt, buf, w, teams, team_of,
                  ent_coef=a.ent_coef,
                  anchor=anchor if L.ptype == "main" else None,
                  kl_coef=a.kl_coef if L.ptype == "main" else 0.0)

        if opp_player is not None and rst["episodes"] > 0:
            score = rst["red_win"] if lt == 0 else rst["blue_win"]
            L.record(opp_player.pid, score + 0.5 * rst["draw"], rst["episodes"])
        L.steps += 1; L.since_reset += 1
        step += 1; state["step"] = step

        if step % phase["snapshot_every"] == 0:
            p = league.add(learners[0].policy.state_dict(), "main", step)
            s._log(f"[league] snapshot {p.name()} -- {league.summary(learners)}")
        for L2 in learners[1:]:
            if league.should_reset(L2):
                snap = league.reset_learner(L2, step)
                s._log(f"[league] {L2.ptype} did its job -> banked {snap.name()} "
                       f"and reset to bootstrap")

        dt = time.time() - t_last; t_last = time.time()
        row = {"step": step, "phase": 4, "learner": L.ptype, "opponent": oname,
               "match": kind, "side": lt,
               "ticks_per_s": round(a.unroll * env.B / max(dt, 1e-6), 1),
               "league_size": len(league.players),
               **{k: round(v, 5) for k, v in m.items()},
               **{k: round(v, 4) for k, v in rst.items()}}

        if step % a.eval_every == 0:
            sc = score_roles(eval_env, learners[0].policy)
            row.update({k: round(v, 4) for k, v in sc.items()})
            s._log(f"[eval] step {step:,} MAIN | ATK " +
                   " ".join(f"{k[4:]}={v:.2f}" for k, v in sc.items()
                            if k.startswith("atk_")) + " | DEF " +
                   " ".join(f"{k[4:]}={v:.2f}" for k, v in sc.items()
                            if k.startswith("def_")))
            s._log(f"[league] {league.summary(learners)}")
            key = sum(sc.get(k, 0.0) for k in sel_keys)
            stall.update(key, step, s._log)
            if key > best:
                best = state["best"] = key
                torch.save(learners[0].policy.state_dict(),
                           Path(a.run_dir or ".") / "best_policy.pt")
                s._log(f"[eval] new best ({'+'.join(sel_keys)} = {key:.3f}) "
                       f"-> best_policy.pt")
                publish(learners[0].policy, a.run_dir, s._log)

        state["league"] = league.state_dict()
        state["learner_states"] = [
            {"policy": L2.policy.state_dict(), "critic": L2.critic.state_dict(),
             "opt": L2.opt.state_dict(), "wins": L2.wins, "games": L2.games,
             "since_reset": L2.since_reset} for L2 in learners]
        state["policy"] = learners[0].policy.state_dict()
        state["critic"] = learners[0].critic.state_dict()
        state["opt"] = learners[0].opt.state_dict()
        s.tick(state, row)

        if step % 20 == 0:
            s._log(f"[train] step {step:,} {L.ptype[:2].upper()} vs {oname} "
                   f"loss {m['loss']:+.3f} ent {m['entropy']:.2f} "
                   f"({row['ticks_per_s']:,.0f} ticks/s)")
        if a.smoke and step - state.get("_smoke0", 0) >= a.smoke:
            s._log("[smoke] done"); break
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--arenas", type=int, default=512)
    ap.add_argument("--unroll", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--reseed-learners", default=None,
                    help="reset the league learners to these weights while "
                         "keeping the accumulated opponent pool")
    ap.add_argument("--max-bc-steps", type=int, default=6000,
                    help="imitation hands over to the league after this many "
                         "steps even if its gate is unmet -- it is a bootstrap, "
                         "not a wall")
    ap.add_argument("--defend-frac", type=float, default=0.72,
                    help="fraction of league updates spent holding the defending "
                         "side; attack is already at ceiling")
    ap.add_argument("--kl-coef", type=float, default=1e-1,
                    help="pull the main agent toward the imitation policy, as in "
                         "the AlphaStar paper; the exploiters run free")
    ap.add_argument("--ent-coef", type=float, default=1e-2,
                    help="raised from 3e-3: the policy collapsed to a degenerate "
                         "deterministic strategy at entropy ~4.5 and could not escape")
    ap.add_argument("--eval-every", type=int, default=250,
                    help="was 50; measured 2026-09-03 that evaluation was eating "
                         "48%% of wall time (86 s per eval vs 19 s per 10 steps). "
                         "At 250 it costs ~15%%.")
    ap.add_argument("--eval-arenas", type=int, default=32)
    ap.add_argument("--wait-hours", type=float, default=0.0)
    ap.add_argument("--init-policy", default=None,
                    help="seed policy weights from a file when starting fresh "
                         "(e.g. a phaseN_passed.pt snapshot); critic and optimizer "
                         "start clean so no collapsed Adam state is carried over")
    ap.add_argument("--smoke", type=int, default=0,
                    help="run N iterations with the GPU gate bypassed, then exit")
    a = ap.parse_args()

    phase = PHASES[a.phase]
    run_dir = a.run_dir or f"/workspace/DroneSwarmRL/runs/phase{a.phase}_{phase['name']}"
    dev = "cuda"

    sess_cfg = SessionConfig(run_dir=run_dir, gpu_index=a.gpu,
                             max_hours=a.max_hours, wait_hours=a.wait_hours)
    sess = TrainingSession(sess_cfg)
    if a.smoke:
        sess.guard.acquire = lambda *k, **kw: True

    with sess as s:
        cfg = SimConfig()
        ecfg = PHASES[a.phase].get("env_cfg", {})
        p_ad = ecfg.pop("p_attack_defend", 0.75) if ecfg else 0.75
        if ecfg:
            import dataclasses as _dc
            sw = {k: v for k, v in ecfg.items() if hasattr(cfg.swarm, k)}
            ar = {k: v for k, v in ecfg.items() if hasattr(cfg.arena, k)}
            sn = {k: v for k, v in ecfg.items() if hasattr(cfg.sensor, k)}
            if sw: cfg.swarm = _dc.replace(cfg.swarm, **sw)
            if ar: cfg.arena = _dc.replace(cfg.arena, **ar)
            if sn: cfg.sensor = _dc.replace(cfg.sensor, **sn)
            s._log(f"[env] phase override: {ecfg}  p_attack_defend={p_ad}")
        env = SwarmEnv(cfg, a.arenas, dev, seed=0, p_attack_defend=p_ad)
        policy = SwarmPolicy().to(dev)
        critic = CentralizedCritic(env.global_state().shape[-1],
                                   len(REWARD_CHANNELS)).to(dev)
        opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()),
                               lr=a.lr)

        # -- mandatory parameter audit before the first iteration
        np_pol = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        np_cri = sum(p.numel() for p in critic.parameters() if p.requires_grad)
        s._log(f"[audit] policy trainable {np_pol:,} | critic trainable {np_cri:,} "
               f"| total {np_pol + np_cri:,}")
        frozen = [n for n, p in policy.named_parameters() if not p.requires_grad]
        s._log(f"[audit] frozen policy tensors: {frozen or 'none'}")
        s._log(f"[audit] phase {a.phase} ({phase['name']}) weights={phase['weights']}")

        state = s.resume()
        step, best = 0, -1.0
        if state:
            # The imitation stage has no critic, so a BC checkpoint carries only
            # the policy. Resuming one must not require the RL-only entries.
            policy.load_state_dict(state["policy"])
            if "critic" in state:
                critic.load_state_dict(state["critic"])
            if "opt" in state:
                try:
                    opt.load_state_dict(state["opt"])
                except ValueError as e:
                    s._log(f"[resume] optimizer state incompatible ({e}); "
                           f"continuing with fresh moments")
            step, best = state["step"], state.get("best", -1.0)
            # The curriculum position lives in the checkpoint, not the CLI flag.
            # Without this, a night that advanced to phase 2 would silently drop
            # back to phase 1 the next night and undo its own progress.
            saved = state.get("phase", a.phase)
            if saved != a.phase:
                s._log(f"[phase] checkpoint is at phase {saved}; "
                       f"overriding --phase {a.phase}")
                a.phase = saved
            phase = PHASES[a.phase]
        else:
            if a.init_policy:
                policy.load_state_dict(torch.load(a.init_policy, map_location=dev))
                s._log(f"[init] policy seeded from {a.init_policy} "
                       f"(critic + optimizer fresh)")
            state = {"step": 0, "phase": a.phase, "nights": 0,
                     "cumulative_hours": 0.0, "best": -1.0}
        state["nights"] = state.get("nights", 0) + 1

        w = weight_vector(phase["weights"], dev)
        team_of = env.team[0]
        stall = StallGuard()
        s._log(f"[phase] running phase {a.phase} ({phase['name']})")
        gen = torch.Generator(device=dev).manual_seed(1234 + step)

        buf = RolloutBuffer(a.unroll, env.B, env.N, env.K,
                            obs_d := 28, 13, env.global_state().shape[-1],
                            len(REWARD_CHANNELS), dev)
        obs = env.reset()
        hx, cx = policy.initial_state(env.B * env.N, dev)

        eval_env = SwarmEnv(cfg, a.eval_arenas, dev, seed=999,
                            p_attack_defend=p_ad)
        a.run_dir = run_dir

        if a.phase in (0, 6):
            # Imitation runs pure attack/defend so "red teaches attacking, blue
            # teaches defending" is always true.
            env.p_attack_defend = 1.0
            state["_smoke0"] = state["step"]
            best = train_bc(s, a, env, policy, opt, state, eval_env,
                            score_roles, publish)
            # BC finished (budget or gate). Hand the league the imitation policy
            # as both its seed and its KL anchor.
            bc_path = Path(run_dir) / "bc_policy.pt"
            if bc_path.exists():
                policy.load_state_dict(torch.load(bc_path, map_location=dev))
            sc = score_roles(eval_env, policy, greedy_policy=True)
            passed = all(sc.get(k, -1) >= v for k, v in PHASES[0]["gate"].items())
            spent = state["step"] >= a.max_bc_steps

            # Imitation is a BOOTSTRAP, never a wall. A hard gate here would be
            # the fourth time this project stalled for days in front of a
            # threshold; AlphaStar's supervised agent was simply "good enough to
            # start the league", not required to master anything. So hand over on
            # the gate OR on a step budget, whichever comes first.
            if passed or spent:
                torch.save(policy.state_dict(), Path(run_dir) / "phase0_passed.pt")
                why = "gate met" if passed else f"budget ({a.max_bc_steps:,} steps)"
                s._log(f"[phase] imitation done ({why}): "
                       f"atk={sc.get('atk_do_nothing',0):.2f} "
                       f"def_rush={sc.get('def_rush',0):.2f} "
                       f"def_greedy={sc.get('def_greedy',0):.2f} -> LEAGUE")
                a.phase = 4
                state["phase"] = 4
                state["_smoke0"] = state["step"]
                s.save(state, force=True)
                return train_league(s, a, cfg, env, eval_env, dev, state,
                                    policy, critic, opt)
            s._log(f"[bc] session ended at step {state['step']:,} before gate or "
                   f"budget (atk={sc.get('atk_do_nothing',0):.2f} "
                   f"def={sc.get('def_rush',0):.2f}); resuming next session")
            return 0

        # Phase 4 has a different shape: three learners taking turns rather than
        # one policy against a fixed opponent list, so it runs its own loop.
        if a.phase in (4, 5):
            state["_smoke0"] = state["step"]
            return train_league(s, a, cfg, env, eval_env, dev, state,
                                policy, critic, opt)
        opp_names = phase["opponents"]
        t_last = time.time()

        # ---- opponent pool + a FROZEN yardstick.
        # Scripted scores alone cannot tell whether self-play is improving the
        # policy or merely rotating through strategies; a fixed reference can.
        pool: list = []
        opp_policy = SwarmPolicy().to(dev)
        ref_policy = None
        ref_path = Path(run_dir) / "phase2_passed.pt"
        if ref_path.exists():
            ref_policy = SwarmPolicy().to(dev)
            ref_policy.load_state_dict(torch.load(ref_path, map_location=dev))
            ref_policy.eval()
            pool.append({k: v.clone() for k, v in ref_policy.state_dict().items()})
            s._log(f"[pool] seeded with {ref_path.name}; it is also the frozen "
                   f"reference the policy is scored against")
        rng = torch.Generator(device="cpu").manual_seed(7)

        while s.should_continue():
            name = opp_names[step % len(opp_names)]
            # Sides were alternated 50/50, which was right while both roles had
            # headroom. Measured 2026-09-05: attack is AT CEILING on four of six
            # opponents (1.00 vs do_nothing, random, rush, saturation), while
            # defence sits at 0.17 against a 0.99 ceiling (greedy) and 0.31
            # against 0.85 (rush). Further attack updates buy nothing, so spend
            # the remaining budget where the gap is.
            lt = 1 if torch.rand(1, generator=rng).item() < a.defend_frac else 0
            teams = (lt,)
            if name is not None:
                opp = B.make(name, env, 1 - lt)
            elif phase.get("pool") and pool and \
                    torch.rand(1, generator=rng).item() > phase.get("p_self", 0.4):
                # frozen past opponent: only OUR side's experience is on-policy
                k = int(torch.randint(len(pool), (1,), generator=rng).item())
                opp_policy.load_state_dict(pool[k])
                opp_policy.eval()
                opp = PolicyAgent(env, 1 - lt, opp_policy, greedy=False, seed=step)
                name = f"pool[{k}/{len(pool)}]"
            else:
                opp = None
                name = "self"
                teams = phase["learn_teams"]        # true self-play: learn both

            obs, hx, cx, rstats = collect(env, policy, buf, hx, cx, obs,
                                          opponent=opp, gen=gen, learner_team=lt)
            hx, cx = hx.detach(), cx.detach()
            m = learn(policy, critic, opt, buf, w, teams, team_of,
                      ent_coef=a.ent_coef)
            step += 1
            state["step"] = step

            dt = time.time() - t_last
            t_last = time.time()
            row = {"step": step, "phase": a.phase, "opponent": name, "side": lt,
                   "ticks_per_s": round(a.unroll * env.B / max(dt, 1e-6), 1),
                   **{k: round(v, 5) for k, v in m.items()},
                   **{k: round(v, 4) for k, v in rstats.items()}}

            # grow the pool so the policy must keep beating its own past selves
            if phase.get("pool") and step % phase.get("snapshot_every", 250) == 0:
                pool.append({k: v.detach().clone()
                             for k, v in policy.state_dict().items()})
                if len(pool) > 12:
                    pool.pop(1)                    # keep the phase-2 reference at [0]
                s._log(f"[pool] snapshot added at step {step:,} (pool={len(pool)})")

            if step % a.eval_every == 0:
                sc = score_roles(eval_env, policy)
                if ref_policy is not None:
                    eval_env.gen.manual_seed(4242)
                    r = play_match(eval_env,
                                   PolicyAgent(eval_env, 0, policy, seed=1),
                                   PolicyAgent(eval_env, 1, ref_policy, seed=2))
                    sc["vs_frozen_ref"] = r["red_win"] + 0.5 * r["draw"]
                row.update({k: round(v, 4) for k, v in sc.items()})
                s._log(f"[eval] step {step:,} worst={sc['worst']:.3f} "
                       f"| ATK worst={sc['worst_atk']:.2f} " +
                       " ".join(f"{k[4:]}={v:.2f}" for k, v in sc.items()
                                if k.startswith("atk_")) +
                       f" | DEF worst={sc['worst_def']:.2f} " +
                       " ".join(f"{k[4:]}={v:.2f}" for k, v in sc.items()
                                if k.startswith("def_")))
                if sc["worst"] > best:
                    best = state["best"] = sc["worst"]
                    torch.save(policy.state_dict(), Path(run_dir) / "best_policy.pt")
                    s._log(f"[eval] new best worst-case score {best:.3f} -> best_policy.pt")
                    publish(policy, run_dir, s._log)

                # ---- automatic phase advance.
                # A 28-night unattended run cannot wait for a human to notice a
                # gate was met; phase 1's gate was cleared in 17 minutes.
                stall.update(sc.get("atk_do_nothing", 0) + sc.get("def_rush", 0),
                             step, s._log)
                if all(sc.get(k, -1) >= v for k, v in phase["gate"].items()):
                    nxt = phase.get("next", a.phase + 1)
                    torch.save(policy.state_dict(),
                               Path(run_dir) / f"phase{a.phase}_passed.pt")
                    s._log(f"[phase] GATE MET for phase {a.phase} ({phase['name']}): "
                           + ", ".join(f"{k}={sc.get(k):.3f}>={v}"
                                       for k, v in phase["gate"].items()))
                    if nxt not in PHASES:
                        s._log("[phase] no further phase defined -- holding here")
                    elif nxt == 4:
                        state["phase"] = 4
                        state["_smoke0"] = state["step"]
                        s._log("[phase] ADVANCING to phase 4 (league) -- "
                               "handing over to the league loop")
                        s.save(state, force=True)
                        return train_league(s, a, cfg, env, eval_env, dev, state,
                                            policy, critic, opt)
                    else:
                        a.phase = nxt
                        phase = PHASES[nxt]
                        w = weight_vector(phase["weights"], dev)
                        opp_names = phase["opponents"]
                        best = state["best"] = -1.0
                        state["phase"] = nxt
                        s._log(f"[phase] ADVANCING to phase {nxt} ({phase['name']}) "
                               f"weights={phase['weights']} "
                               f"opponents={opp_names} teams={phase['learn_teams']}")

            state.update({"policy": policy.state_dict(), "critic": critic.state_dict(),
                          "opt": opt.state_dict()})
            s.tick(state, row)

            if step % 10 == 0:
                s._log(f"[train] step {step:,} loss {m['loss']:+.4f} "
                       f"ent {m['entropy']:.3f} |g| {m['grad_norm']:.2f} "
                       f"red_win {rstats['red_win']:.2f} "
                       f"({row['ticks_per_s']:,.0f} ticks/s)")

            if a.smoke and step >= a.smoke:
                s._log(f"[smoke] completed {a.smoke} iterations -- exiting")
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
