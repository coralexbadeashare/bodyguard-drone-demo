"""Behaviour cloning -- AlphaStar's supervised stage, with scripted demonstrators.

AlphaStar trains on 971,000 human replays BEFORE any reinforcement learning, and
that supervised agent already plays at roughly top-16% human level. The league
then improves on a policy that can already play. Every failure on this project so
far came from skipping that step: RL was started from a policy that could barely
navigate, and it either collapsed, learned to creep, or oscillated between the
attacker and defender objectives.

We have no human replays, but we do have demonstrators: the scripted agents beat
each other in measurable, non-degenerate ways (`greedy` averages 0.845). Cloning
them is the same idea -- learn to play competently by imitation, then improve by
RL. The resulting policy is also the KL anchor the RL stage is pulled toward,
exactly as in the paper.

Data is generated on the fly: the simulator produces ~8,600 arena-ticks/s, so
there is no reason to store a dataset.
"""
from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from ..model.networks import SwarmPolicy, N_MAIN_HEADS
from ..sim.env import SwarmEnv, MODE_ATTACK_DEFEND, MODE_WAR
from . import baselines as B
from .evaluate import PolicyAgent

# Heads the scripted agents actually decide. `delay` and `comms` carry no teacher
# signal -- the demonstrators have no notion of either -- so they are left to RL.
# `delay` is cloned too: the demonstrators always act every tick (delay=0), and
# an untrained delay head makes the policy skip ~60% of its control ticks.
CLONE_HEADS = ("action_type", "target", "heading", "pitch", "speed", "delay")

# Teachers are chosen per ROLE from the measured payoff matrix, not by intuition.
# In attack/defend:
#   attacking  rush 1.00 vs do_nothing, greedy only 0.01 -- greedy duels instead
#              of striking, so it cannot break a base and must not teach attack
#   defending  interceptor 0.98 vs rush; greedy 0.52 and perimeter 0.41 are
#              barely above do_nothing's 0.00 (docs/GUARD_REFERENCE.md)
# Cloning greedy as the attacker was why the imitation policy scored 0.00 on
# attack while matching its teacher 98.7% of the time: it was a faithful copy of
# an agent that cannot attack.
#
# The same mistake was sitting in the DEFEND list until 2026-09-11. `greedy` and
# `perimeter` steer at where the target IS, which was correct when the drones had
# a gun; with kinetic interception a tail chase never closes, so both are
# non-defences and the student inherited that. `interceptor` solves the collision
# triangle and is the only teacher here that actually stops a rush.
# `obs_interceptor`, NOT `interceptor`. The full-observability version scores
# DEF 0.97 using contacts it has not seen, and a student cannot predict those
# turns: cloning it reached 97% action agreement and DEF 0.00, the do-nothing
# floor, and DAgger did not move it (0.00 at beta=0, accuracy 0.966). That is the
# signature of an unrealizable teacher, not covariate shift. `obs_interceptor`
# runs the same guidance law on the observation alone and scores 0.52, so every
# action it takes IS a function of what the student sees.
ATTACK_TEACHERS = ("rush", "rush", "greedy")
DEFEND_TEACHERS = ("obs_interceptor", "obs_interceptor", "greedy")


def bc_step(env: SwarmEnv, policy: SwarmPolicy, opt, hx, cx, obs,
            teacher_red: str, teacher_blue: str, gen, beta: float = 1.0) -> tuple:
    """One imitation update.

    `beta` is the DAgger mixing coefficient: the probability that an arena is
    advanced by the TEACHER's action rather than the student's own. At beta = 1
    this is plain behaviour cloning -- the student only ever sees states the
    teacher visits.

    Plain cloning is not enough here, and the failure is quantitative: at 97%
    action agreement the cloned interceptor scored def_rush 0.00, exactly the
    do-nothing floor, against its teacher's 0.97 (2026-09-11). Interception is
    closed-loop -- heading is 16 bins, the student picks the wrong bin ~11% of
    ticks, and at a ~30 m/s closing speed a couple of wrong bins in the endgame
    is a clean miss past a 2.0 m intercept radius. Those misses then take it to
    states the teacher never occupies, where it has no supervision at all.

    Annealing beta to 0 puts the student on its OWN state distribution while
    still labelling with the teacher, which is the standard remedy.
    """
    M, K = env.B * env.N, env.K
    # disperse=False: the demonstrator must pick the NEAREST contact, a decision
    # that is fully determined by the observation and therefore learnable.
    red = B.make(teacher_red, env, 0, disperse=False)
    blue = B.make(teacher_blue, env, 1, disperse=False)
    ra, ba = red.act(obs), blue.act(obs)
    act = B.merge(ra, ba, env.n)

    forced = {h: act[h].reshape(M) for h in CLONE_HEADS}
    # Let the student choose its own action (Gumbel-sampled, autoregressive) so
    # the logits are conditioned on the context it actually builds, and so we
    # have an action to drive the env with. The labels stay the teacher's.
    u = torch.rand(M, N_MAIN_HEADS + policy.n_comm_tokens, policy.max_logits,
                   device=env.device, generator=gen).clamp_min(1e-9)
    noise = -torch.log(-torch.log(u))
    lg, student, hx2, cx2 = policy(
        obs["scalars"].reshape(M, -1), obs["entities"].reshape(M, K, -1),
        obs["entity_mask"].reshape(M, K), PolicyAgent.action_mask(obs),
        hx, cx, noise)

    alive = env.alive.reshape(M).float()
    # A demonstrator with no contacts still emits target=0 as a placeholder, and
    # slot 0 may be masked. Cross-entropy on an impossible class costs ~1e9 (the
    # mask fill) and wrecks the policy, so score the pointer head only where the
    # demonstrated target is a real contact.
    ent_ok = obs["entity_mask"].reshape(M, K).float()
    tgt_valid = ent_ok.gather(1, forced["target"].clamp(0, K - 1)
                              .unsqueeze(1)).squeeze(1)
    weights = {h: alive for h in CLONE_HEADS}
    weights["target"] = alive * tgt_valid

    loss, correct = 0.0, {}
    for h in CLONE_HEADS:
        tgt, wgt = forced[h], weights[h]
        ce = F.cross_entropy(lg[h].float(), tgt, reduction="none")
        loss = loss + (ce * wgt).sum() / wgt.sum().clamp_min(1)
        hit = (lg[h].argmax(-1) == tgt).float()
        correct[h] = (hit * wgt).sum() / wgt.sum().clamp_min(1)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    opt.step()

    # DAgger mix, per ARENA -- a trajectory should be driven consistently by one
    # of the two, not alternate every tick.
    if beta >= 1.0:
        act_env = act
    else:
        use_t = (torch.rand(env.B, 1, device=env.device, generator=gen) < beta)
        act_env = {}
        for k, v in act.items():
            sv = student[k].view(env.B, env.N, -1) if v.dim() == 3 else \
                student[k].view(env.B, env.N)
            m = use_t if v.dim() == 2 else use_t.unsqueeze(-1)
            act_env[k] = torch.where(m, v, sv)
    obs, _, done, _ = env.step(act_env)
    any_done = bool(done.any())
    if any_done:
        obs = env.reset(done)
        z = done.view(env.B, 1).expand(env.B, env.N).reshape(M, 1)
        hx2 = torch.where(z, 0.0, hx2)
        cx2 = torch.where(z, 0.0, cx2)
    return obs, hx2.detach(), cx2.detach(), loss.item(), \
        {h: v.item() for h, v in correct.items()}, gn.item(), any_done


def train_bc(s, a, env, policy, opt, state, eval_env, score_fn, publish_fn):
    """Imitate the demonstrators until the policy plays like them."""
    gen = torch.Generator(device=env.device).manual_seed(7 + state["step"])
    obs = env.reset()
    hx, cx = policy.initial_state(env.B * env.N, env.device)
    step = state["step"]
    best = state.get("best", -1.0)
    t_last = time.time()
    s._log(f"[bc] attacker teachers {ATTACK_TEACHERS} | defender teachers "
           f"{DEFEND_TEACHERS} | heads {list(CLONE_HEADS)}")

    # Teachers are fixed for a whole EPISODE. Rotating them per step meant the
    # demonstrator changed every 0.1 s of simulated time, so the policy learned
    # the average of three incompatible strategies rather than any one of them.
    pair = [0]
    tr, tb = ATTACK_TEACHERS[0], DEFEND_TEACHERS[0]
    # DAgger schedule: pure cloning for the first `warm` steps so the student is
    # worth following at all, then anneal to fully self-driven over `anneal`.
    warm, anneal = 500, 4000
    while s.should_continue():
        beta = 1.0 if step < warm else max(0.0, 1.0 - (step - warm) / anneal)
        obs, hx, cx, loss, acc, gn, reset = bc_step(
            env, policy, opt, hx, cx, obs, tr, tb, gen, beta=beta)
        if reset:
            pair[0] += 1
            i = pair[0]
            tr = ATTACK_TEACHERS[i % len(ATTACK_TEACHERS)]   # red attacks
            tb = DEFEND_TEACHERS[i % len(DEFEND_TEACHERS)]   # blue defends
        step += 1
        state["step"] = step

        dt = time.time() - t_last; t_last = time.time()
        row = {"step": step, "phase": 0, "teachers": f"{tr}/{tb}",
               "dagger_beta": round(beta, 3), "bc_loss": round(loss, 4),
               "grad_norm": round(gn, 3),
               "ticks_per_s": round(env.B / max(dt, 1e-6), 1),
               **{f"acc_{h}": round(v, 4) for h, v in acc.items()}}

        if step % a.eval_every == 0:
            # The demonstrators are deterministic, so score the argmax policy.
            # Sampling seven heads every tick for 600 ticks compounds into
            # behaviour the teacher never showed.
            sc = score_fn(eval_env, policy, greedy_policy=True)
            row.update({k: round(v, 4) for k, v in sc.items()})
            mean_acc = sum(acc.values()) / len(acc)
            s._log(f"[bc] step {step:,} beta {beta:.2f} loss {loss:.3f} acc {mean_acc:.3f} "
                   f"| atk_do_nothing={sc.get('atk_do_nothing',0):.2f} "
                   f"def_rush={sc.get('def_rush',0):.2f} "
                   f"def_greedy={sc.get('def_greedy',0):.2f}")
            key = sc.get("atk_do_nothing", 0) + sc.get("def_rush", 0)
            if key > best:
                best = state["best"] = key
                torch.save(policy.state_dict(), f"{a.run_dir}/bc_policy.pt")
                publish_fn(policy, a.run_dir, s._log)
                s._log(f"[bc] new best (atk+def = {key:.3f}) -> bc_policy.pt")
        state["policy"] = policy.state_dict()
        state["opt"] = opt.state_dict()
        s.tick(state, row)

        if step % 100 == 0:
            s._log(f"[bc] step {step:,} loss {loss:.3f} "
                   f"acc " + " ".join(f"{h[:4]}={acc[h]:.2f}" for h in CLONE_HEADS))
        if a.smoke and step - state.get("_smoke0", 0) >= a.smoke:
            break
        # Hand back as soon as the budget is spent. Checking this only after the
        # loop meant imitation would consume the whole 11.5 h session before the
        # league ever started.
        if step >= getattr(a, "max_bc_steps", 6000):
            s._log(f"[bc] budget reached at step {step:,} -- returning to hand "
                   f"over to the league")
            break
    return best
