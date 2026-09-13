# The bar a bodyguard policy has to clear

**2026-09-11.** `scripts/guard_reference.py`, 64 arenas, full episodes, seed
12345. Bodyguard scenario: **3 attackers, 2 defenders, base health 240, base
walking at 1.4 m/s** — a person, not a building.

Value = **DEF score** (`blue_win + 0.5*draw`), the same number `score_roles`
reports. `[hp]` = mean fraction of base health left.

|        defence |         rush |   saturation |       greedy |
|---------------:|-------------:|-------------:|-------------:|
|     do_nothing | 0.00 [hp 0.00] | 0.14 [hp 0.02] | 0.95 [hp 0.54] |
|      perimeter | 0.41 [hp 0.06] | 0.20 [hp 0.03] | 0.89 [hp 0.51] |
|         greedy | 0.52 [hp 0.10] | 0.75 [hp 0.72] | 1.00 [hp 0.50] |
| **obs_interceptor** | **0.52 [hp 0.12]** | **0.27 [hp 0.04]** | 0.94 [hp 0.45] |
|    interceptor | 0.98 [hp 0.40] | 1.00 [hp 0.70] | 1.00 [hp 0.51] |

## Use `obs_interceptor` as the bar, not `interceptor`

`interceptor` reads `env` directly and engages every living attacker, visible or
not. Scripted agents are deliberately allowed that — it makes them stronger
opponents — but it means **0.98 is not a score any exported policy can reach**.
The policy sees camera cones plus the BLE mesh; it cannot turn against a contact
it has no knowledge of.

`obs_interceptor` runs the *identical* guidance law on the observation alone and
scores **0.52** against a rush. That is the realizable bar. Most of the 0.98 →
0.52 gap is simply that the full-observability version is always pursuing, while
the observation-limited one has a contact on roughly a third of ticks and escorts
the base the rest of the time.

Scoring against the wrong bar is not cosmetic: it also corrupts imitation. A BC
student cloned from `interceptor` matched it on 97% of actions and still scored
**0.00** — the do-nothing floor — because it was being asked to predict turns
caused by information it does not have. DAgger did not help (0.00 at β = 0 with
accuracy recovered to 0.966), which is the signature of an unrealizable teacher
rather than covariate shift.

## Detection range is the hardware lever, and the knee is ~120 m

`scripts/sensor_sweep.py`, `obs_interceptor` vs a 3-drone rush:

| camera | FOV 90° | FOV 150° | FOV 360° |
|---:|---:|---:|---:|
| **70 m** (default) | 0.61 | 0.78 | 0.78 |
| **120 m** | 0.95 | 0.97 | 0.98 |
| 180 m | 0.98 | 1.00 | 1.00 |
| 250 m | 0.98 | 1.00 | 1.00 |
| 350 m | 0.98 | 1.00 | 1.00 |

**Range dominates; field of view is second order and saturates by 150°.** Going
70 → 120 m takes a partially-observing defence from 0.61 to 0.95 — it closes
almost the entire gap to full observability. Past 180 m nothing improves, because
the attackers are then detected as early as the geometry allows.

For the real bodyguard build this is the specification that matters: **detect a
small drone at ~120–180 m**, and do not spend on a fisheye beyond ~150°.

## Three corrections this table forced

1. **`perimeter` and `greedy` are not defences.** Against a rush they are barely
   distinguishable from doing nothing. Both were written when the drones had a
   gun, where steering at the target's current position is correct; with kinetic
   interception a tail chase against an equally fast attacker never closes.
2. **Losing the base was being scored as a draw.** An attacker is destroyed by
   the strike that destroys the base, so `blue_base_down` and `alive_red == 0`
   fired on the same tick and landed in the mutual-destruction branch. That is
   why `do_nothing` used to score 0.50 here. `rew["win"]` is 0 on a draw, so the
   attacker was never rewarded for a successful strike and the defender never
   punished for losing.
3. **The bar itself was unreachable**, as above.

## What is still true

Even `interceptor`, with perfect information, concedes 60% of the base's health.
Two defenders do not cleanly stop three attackers under these constants — they
survive the clock. The honest reading is that the bodyguard scenario wants either
a third defender or a longer-range sensor, and the sweep says the sensor is the
cheaper fix.
