# The bodyguard drone: what the simulator actually established

**2026-09-11.** Scenario: a drone hovering over a walking person detects an
incoming drone with its camera and intercepts it. In the simulator that is
3 attackers vs 2 defenders, base health 240, base walking at 1.4 m/s, kinetic
interception only (no gun — both aircraft are destroyed on contact).

Scores are `DEF = blue_win + 0.5*draw` from `scripts/guard_reference.py`,
64 arenas, seed 12345. `do_nothing` = 0.00 by construction.

---

## 1. The hardware answer: detection range is the binding parameter

`scripts/sensor_sweep.py`, scripted lead-pursuit defence restricted to the
drone's own observation, vs a 3-drone rush:

| camera | FOV 90° | FOV 150° | FOV 360° |
|---:|---:|---:|---:|
| **70 m** (current sim default) | 0.61 | 0.78 | 0.78 |
| **120 m** | 0.95 | 0.97 | 0.98 |
| 180 m | 0.98 | 1.00 | 1.00 |
| 350 m | 0.98 | 1.00 | 1.00 |

**Range dominates. Field of view is second order and saturates by 150°.**
Going 70 → 120 m takes the defence from 0.61 to 0.95 — it closes almost the
entire gap to perfect information. Past 180 m nothing improves.

**For a real build: specify detection of a small drone at 120–180 m. Do not pay
for a fisheye beyond ~150°.** At 18 m/s closing, 120 m is ~6.7 s of warning,
which is what the interception geometry needs.

## 2. Control is a solved problem; a closed-form law beats everything learned

The interception law is the collision triangle — the earliest `t` with

    |p_t + v_t·t − p_o| = s·t

steering at the aim point `p_t + v_t·t`, so the line of sight stops rotating.
About 120 lines, microseconds per tick, trivially runnable on a Pi-class board.

| defence | DEF vs rush | note |
|---|---:|---|
| do_nothing | 0.00 | floor |
| perimeter | 0.44 | pre-existing, written for a gun |
| greedy | 0.56 | pre-existing, written for a gun |
| **obs_interceptor** | **0.53** | closed form, observation-limited — **the realizable bar** |
| interceptor | 0.98 | closed form, full observability — *not reachable by any exported policy* |
| best learned (RL) | 0.34 | peak, `runs/_archive/guard_rl_peak_def034.pt` |
| best learned (imitation) | 0.00 | see §3 |

**Nothing learned has beaten the closed-form law.** For an actual device, ship
the scripted interceptor.

## 3. Three things that looked like results and were not

Recorded because each cost real GPU time and each was an environment or
measurement defect, not an algorithmic one.

**Losing the base scored as a draw.** The attacker is destroyed by the strike
that destroys the base, so `blue_base_down` and `alive_red == 0` fired on the
same tick and hit the mutual-destruction branch. `do_nothing` conceded the base
in 100% of arenas and scored 0.50. `rew["win"]` is 0 on a draw, so attackers were
never rewarded for a successful strike and defenders never punished for losing —
with `win` weighted 20.0.

**A shared policy abandons the role its weights disfavour.** With
`learn_teams=(0,1)` and `base_preserved` (120) against `base_damage` (10), the
network maximised the objective by never attacking: `atk_do_nothing` 0.97 → 0.00
in 16k steps. The second-order effect did the damage — once the attacker was
gone, self-play handed the defender a harmless opponent and `def_rush` sat at
0.06.

**Freezing the attacker moved the loop up one level.** With `learn_teams=(1,)`
every league snapshot is a defence-only policy, and the league hands those back
as *attackers*. The learner met `rush` in 17 of 437 matchups (3.9%) and spent 85%
against opponents that cannot attack. `def_rush` peaked at 0.34 (step 1.5k) and
decayed to 0.00 by step 8.5k while `def_greedy` climbed to 1.00.

The common signature: **a role or opponent scoring exactly 0.00 against
`do_nothing`.** That is collapse, never difficulty.

## 4. Imitation failed twice, and the second failure is the interesting one

Cloning the full-observability interceptor reached **97% action agreement and
DEF 0.00** — the do-nothing floor. The obvious explanation was covariate shift,
so the demonstrator loop was converted to DAgger (student drives, teacher labels,
β annealed 1 → 0). **No change: 0.00 at β = 0 with accuracy recovered to 0.966.**

The next explanation was an unrealizable teacher — the full-obs interceptor turns
against contacts the student cannot see. So the teacher was replaced with
`obs_interceptor`, every one of whose decisions is by construction a function of
the student's observation. **Also 0.00, at 92% accuracy.**

So neither covariate shift nor realizability accounts for it. What remains is
that interception is a *precision closed-loop* task: heading is 16 bins (22.5°),
the student is wrong on ~8–13% of ticks, and at ~26 m/s closing a couple of wrong
bins inside the endgame is a clean miss past a 2.0 m radius. Per-tick action
accuracy is simply the wrong objective for it — the right one would penalise
miss distance, not label disagreement. **This is the open question worth taking
forward**, and it is a real one rather than a bug.

## 5. A physics defect that was silently discarding interceptions

`_resolve_combat()` ran once per 10 Hz policy tick, after 20 physics substeps. A
head-on pair closes ~2.6 m in that interval against a 2.0 m intercept radius, so
genuine interceptions passed through between samples — **12–15% of real contacts
were missed**. Now fixed with a swept closest-approach test (distance from the
origin to the relative-motion segment, exact for linear motion). The reference
table barely moved (0.52 → 0.53), so it was a correctness gain, not a rebalance.

## 6. What two defenders can and cannot do

Even with perfect information the scripted interceptor concedes **52% of the
base's health** (hp 0.48) and was never once left untouched. Two defenders do not
cleanly stop three attackers under these constants; they survive the clock.

The sweep in §1 says the cheaper fix is the sensor, not a third airframe: at
120 m detection the same two defenders reach 0.95.
