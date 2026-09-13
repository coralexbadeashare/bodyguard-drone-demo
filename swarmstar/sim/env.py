"""The drone-combat environment: batched, GPU-resident, partially observed.

Two modes
---------
WAR            both teams hold a base; destroy the enemy base or wipe their swarm
ATTACK_DEFEND  only blue holds a base; red must destroy it before the clock runs
               out, blue only has to survive

Both are driven by one network. Mode and role enter as observation scalars, so the
exported model stays byte-identical on every drone -- which is the whole point.

Observations are built strictly from `perception.py`, so a drone knows an enemy's
position only if it saw it through a camera cone or a teammate relayed it. Nothing
in this file leaks ground truth into a policy observation; the ground-truth tensors
are reserved for the centralized critic and for logging.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor

from .config import SimConfig
from .perception import camera_detections, mesh_hops, fuse_beliefs
from .quadrotor import (QuadrotorState, allocation_matrix, physics_step,
                        cascaded_controller, quat_to_rotmat)

# ---- action space -----------------------------------------------------------
HOLD, GOTO, ENGAGE, STRIKE_BASE, EVADE, REGROUP, SCOUT = range(7)

# Kinetic impact model. A strike drone is expendable ordnance: it flies INTO the
# target and is destroyed doing so. Damage scales with impact speed, so a fast
# committed run is worth far more than drifting in, and every base hit costs a
# drone -- which is the real tactical trade-off.
KAMIKAZE_BASE_DMG = 20.0      # detonation, independent of speed
KAMIKAZE_SPEED_DMG = 5.0      # per m/s of impact speed
GROUND_IMPACT_DMG = 6.0       # per m/s of descent rate into the floor
COLLISION_DMG = 4.0           # per m/s of closing speed, drone vs drone
N_ACTION_TYPES = 7
N_HEADING, N_PITCH, N_SPEED = 16, 5, 4
# Where the airframe POINTS, independent of where it flies. Optional: when an
# action dict carries no "look" key the nose follows the velocity vector exactly
# as before, so every existing policy and checkpoint is unaffected.
N_LOOK = 16
N_DELAY = 4                      # 0..3 extra ticks of silence
MODE_WAR, MODE_ATTACK_DEFEND = 0, 1

OBS_SCALARS = 28
OBS_ENTITY_FEATS = 13

REWARD_CHANNELS = ("win", "base_damage", "base_preserved", "kills",
                   "losses", "coverage", "collision", "energy", "boundary",
                   "progress", "time")


class SwarmEnv:
    def __init__(self, cfg: SimConfig, n_envs: int, device="cuda",
                 episode_seconds: float = 60.0, seed: int = 0,
                 p_attack_defend: float = 0.75):
        self.cfg, self.B, self.device = cfg, n_envs, device
        self.n = cfg.swarm.n_per_team
        self.N = 2 * self.n
        self.K = cfg.swarm.max_visible
        self.dt = cfg.drone.dt_physics * cfg.drone.policy_decim
        self.max_steps = int(episode_seconds / self.dt)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.p_attack_defend = p_attack_defend

        self.alloc = allocation_matrix(cfg.drone, device)
        self.alloc_inv = torch.linalg.inv(self.alloc)

        # team 0 = red (indices 0..n-1), team 1 = blue (n..2n-1). Static.
        t = torch.zeros(self.N, dtype=torch.long, device=device)
        t[self.n:] = 1
        self.team = t.unsqueeze(0).expand(n_envs, -1).contiguous()
        self.enemy_of = 1 - self.team

        a = cfg.arena
        self.base_pos = torch.tensor(
            [[-0.35 * a.size_x, 0.0, 5.0], [0.35 * a.size_x, 0.0, 5.0]],
            device=device).unsqueeze(0).expand(n_envs, -1, -1).contiguous()

        # Which drone slots are actually flying. Inactive slots spawn destroyed,
        # so tensor shapes (and the exported model) never change while the game
        # becomes asymmetric: 1-2 defenders guarding one person against several
        # incoming drones is the real target, not 16v16.
        idx_in_team = torch.arange(self.N, device=device) % self.n
        self.active = torch.where(t == 0,
                                  idx_in_team < cfg.swarm.n_active_red,
                                  idx_in_team < cfg.swarm.n_active_blue)
        self.active = self.active.unsqueeze(0).expand(n_envs, -1).contiguous()

        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, mask: Tensor | None = None):
        B, N, n, dev = self.B, self.N, self.n, self.device
        cfg = self.cfg
        if mask is None:
            mask = torch.ones(B, dtype=torch.bool, device=dev)

        # spawn each swarm in a loose cloud above its own base
        home = self.base_pos.gather(
            1, self.team.unsqueeze(-1).expand(B, N, 3))                # [B,N,3]
        jitter = torch.randn(B, N, 3, device=dev, generator=self.gen)
        jitter[..., :2] *= 12.0
        jitter[..., 2] = jitter[..., 2].abs() * 4.0 + 12.0
        pos = home + jitter

        st = QuadrotorState.hover(B, N, cfg.drone, dev, pos=pos)
        # face the enemy base
        enemy_base = self.base_pos.gather(
            1, self.enemy_of.unsqueeze(-1).expand(B, N, 3))
        yaw = torch.atan2(enemy_base[..., 1] - pos[..., 1],
                          enemy_base[..., 0] - pos[..., 0])
        st.quat[..., 0] = torch.cos(yaw * 0.5)
        st.quat[..., 3] = torch.sin(yaw * 0.5)

        if not hasattr(self, "state"):
            self.state = st
            self.health = torch.where(self.active, cfg.swarm.drone_health, 0.0)
            self.base_health = torch.full((B, 2), cfg.arena.base_health, device=dev)
            self.energy = torch.ones(B, N, device=dev)
            self.t = torch.zeros(B, dtype=torch.long, device=dev)
            self.mode = torch.zeros(B, dtype=torch.long, device=dev)
            self.delay_left = torch.zeros(B, N, dtype=torch.long, device=dev)
            self.prev_vel_sp = torch.zeros(B, N, 3, device=dev)
            self.prev_yaw_sp = torch.zeros(B, N, device=dev)
            self.comm_tokens = torch.zeros(
                B, N, cfg.comms.n_latent_tokens, dtype=torch.long, device=dev)
            self._prev_base_dist = torch.zeros(B, N, device=dev)
            self._pre_clamp_vz = torch.zeros(B, N, device=dev)
        else:
            m3 = mask.view(B, 1, 1)
            for dst, src in ((self.state.pos, st.pos), (self.state.vel, st.vel),
                             (self.state.quat, st.quat), (self.state.omega, st.omega),
                             (self.state.rotor_w, st.rotor_w)):
                dst.copy_(torch.where(m3, src, dst))
            m2 = mask.view(B, 1)
            self.health = torch.where(m2 & self.active, cfg.swarm.drone_health,
                                      torch.where(m2, torch.zeros_like(self.health),
                                                  self.health))
            self.base_health = torch.where(
                mask.view(B, 1), cfg.arena.base_health, self.base_health)
            self.energy = torch.where(m2, torch.ones_like(self.energy), self.energy)
            self.t = torch.where(mask, torch.zeros_like(self.t), self.t)
            self.delay_left = torch.where(m2, 0, self.delay_left)
            self.prev_vel_sp = torch.where(m3, 0.0, self.prev_vel_sp)
            self.comm_tokens = torch.where(m2.unsqueeze(-1), 0, self.comm_tokens)

        # give the walking base a fresh heading on reset
        if not hasattr(self, "_base_heading"):
            self._base_heading = torch.zeros(B, device=dev)
        newh = torch.rand(B, device=dev, generator=self.gen) * 6.2832
        self._base_heading = torch.where(mask, newh, self._base_heading)
        self._base_home = getattr(self, "_base_home", self.base_pos.clone())
        self.base_pos = torch.where(mask.view(B, 1, 1), self._base_home, self.base_pos)

        eb = self.base_pos.gather(1, self.enemy_of.unsqueeze(-1).expand(B, N, 3))
        d0 = (self.state.pos - eb).norm(dim=-1)
        self._prev_base_dist = torch.where(mask.view(B, 1), d0, self._prev_base_dist)

        # attack/defend is the primary game; war is kept in the mix so the policy
        # stays general and the role scalars keep meaning something
        coin = torch.rand(self.B, device=dev, generator=self.gen) < self.p_attack_defend
        self.mode = torch.where(mask, coin.long(), self.mode)
        # in attack/defend red has no base to lose: mark it already gone
        ad = mask & (self.mode == MODE_ATTACK_DEFEND)
        self.base_health[:, 0] = torch.where(ad, 0.0, self.base_health[:, 0])
        return self.observe()

    # ------------------------------------------------------------------- alive
    @property
    def alive(self) -> Tensor:
        return self.health > 0

    # ------------------------------------------------------- action -> setpoint
    def decode_actions(self, act: dict[str, Tensor], bel_pos: Tensor,
                       bel_age: Tensor, ent_gid: Tensor, ent_mask: Tensor):
        """Turn the autoregressive action tuple into a velocity + yaw setpoint.

        `action_type` picks a macro-behaviour; the waypoint head refines it. This
        mirrors AlphaStar's what/who/where split, and keeps the low-level control
        problem on the flight controller where it belongs.
        """
        B, N, dev = self.B, self.N, self.device
        cfg = self.cfg
        at = act["action_type"]                                        # [B,N]

        # --- where: heading x pitch x speed, decoded in world frame
        az = act["heading"].float() * (2 * math.pi / N_HEADING)
        el = (act["pitch"].float() - (N_PITCH - 1) / 2) * math.radians(40.0) / \
             ((N_PITCH - 1) / 2)
        sp = act["speed"].float() * (cfg.drone.max_speed / (N_SPEED - 1))
        way = torch.stack([torch.cos(el) * torch.cos(az),
                           torch.cos(el) * torch.sin(az),
                           torch.sin(el)], dim=-1) * sp.unsqueeze(-1)

        # --- who: the pointer head indexes this drone's own visible list
        tgt = act["target"].clamp(0, self.K - 1)
        gid = ent_gid.gather(2, tgt.unsqueeze(-1)).squeeze(-1)          # [B,N]
        valid = ent_mask.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        tpos = bel_pos.gather(2, gid.clamp_min(0).unsqueeze(-1).unsqueeze(-1)
                              .expand(B, N, 1, 3)).squeeze(2)           # [B,N,3]

        pos = self.state.pos
        to_target = tpos - pos
        dist_t = to_target.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        # Proportional pursuit: close fast when far, decelerate onto a standoff
        # ring at 60% of weapon range, back off if too close. Charging at full
        # speed overshoots the 12-degree cone every pass, and a swarm that all
        # converges on one contact simply collides with itself.
        # RAM the target. With no gun there is no standoff to hold: an intercept
        # is completed by making contact, so ENGAGE closes all the way.
        hold = 0.0
        closing = ((dist_t - hold) * 0.8).clamp(-0.4 * cfg.drone.max_speed,
                                                cfg.drone.max_speed)
        chase = to_target / dist_t * closing

        enemy_base = self.base_pos.gather(
            1, self.enemy_of.unsqueeze(-1).expand(B, N, 3))
        to_base = enemy_base - pos
        strike = to_base / to_base.norm(dim=-1, keepdim=True).clamp_min(1e-3) * \
            cfg.drone.max_speed

        # regroup: toward the centroid of known-alive teammates
        same = (self.team.unsqueeze(1) == self.team.unsqueeze(2)) & \
               self.alive.unsqueeze(1)
        w = same.float()
        centroid = (w.unsqueeze(-1) * pos.unsqueeze(1)).sum(2) / \
            w.sum(-1, keepdim=True).clamp_min(1.0)
        regroup = (centroid - pos)
        regroup = regroup / regroup.norm(dim=-1, keepdim=True).clamp_min(1e-3) * \
            (0.6 * cfg.drone.max_speed)

        zero = torch.zeros_like(way)
        sel = at.unsqueeze(-1)
        vel_sp = torch.where(sel == HOLD, zero,
                  torch.where(sel == ENGAGE, torch.where(valid.unsqueeze(-1), chase, way),
                   torch.where(sel == STRIKE_BASE, strike,
                    torch.where(sel == EVADE, torch.where(valid.unsqueeze(-1), -chase, way),
                     torch.where(sel == REGROUP, regroup, way)))))

        # --- nose. By default it follows the velocity vector, which is right for
        # a drone that is going somewhere and wrong for a sentry: a defender
        # holding station with no contact gets no yaw command at all and ends up
        # pointing wherever it drifted, so its camera covers an arc nobody chose.
        # An optional "look" head fixes that -- a world-frame bearing the drone
        # holds regardless of where it is flying.
        R = quat_to_rotmat(self.state.quat)
        cur_yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
        aim = torch.where(((at == ENGAGE) & valid).unsqueeze(-1), to_target, vel_sp)
        want_yaw = torch.atan2(aim[..., 1], aim[..., 0])
        moving = vel_sp.norm(dim=-1) > 0.1
        commanded = moving | (at == ENGAGE)
        if "look" in act:
            # -1 = "no command" -- agents that do not steer their nose leave the
            # head unset and merge() fills their half with -1, so the two teams
            # can disagree about whether they use it.
            lk = act["look"]
            has_look = lk >= 0
            want_yaw = torch.where(has_look,
                                   lk.clamp_min(0).float() * (2 * math.pi / N_LOOK),
                                   want_yaw)
            commanded = commanded | has_look
        err = torch.atan2(torch.sin(want_yaw - cur_yaw), torch.cos(want_yaw - cur_yaw))
        yaw_sp = torch.where(commanded, (2.5 * err).clamp(-4.0, 4.0),
                             torch.zeros_like(err))
        return vel_sp, yaw_sp, gid, valid

    # -------------------------------------------------------------------- step
    def step(self, act: dict[str, Tensor]):
        cfg, B, N = self.cfg, self.B, self.N
        bel_age, bel_pos, ent = self._last_belief

        vel_sp, yaw_sp, gid, tvalid = self.decode_actions(
            act, bel_pos, bel_age, ent["gid"], ent["mask"])

        # the delay head lets a drone stay quiet and keep its previous setpoint,
        # which is what saves radio traffic and battery on real hardware
        fresh = self.delay_left <= 0
        f3 = fresh.unsqueeze(-1)
        vel_sp = torch.where(f3, vel_sp, self.prev_vel_sp)
        yaw_sp = torch.where(fresh, yaw_sp, self.prev_yaw_sp)
        self.prev_vel_sp, self.prev_yaw_sp = vel_sp, yaw_sp
        self.delay_left = torch.where(fresh, act["delay"].clamp(0, N_DELAY - 1),
                                      self.delay_left - 1)
        self.comm_tokens = torch.where(fresh.unsqueeze(-1), act["comms"],
                                       self.comm_tokens)

        self._last_action_type = act["action_type"]
        dead3 = (~self.alive).unsqueeze(-1)
        vel_sp = torch.where(dead3, torch.zeros_like(vel_sp), vel_sp)

        # ---- integrate the twin
        pos_before = self.state.pos.clone()          # for swept collision below
        cmd = None
        for i in range(cfg.drone.policy_decim):
            if i % cfg.drone.ctrl_decim == 0:
                cmd = cascaded_controller(self.state, vel_sp, yaw_sp,
                                          cfg.drone, self.alloc_inv)
            self.state = physics_step(self.state, cmd, cfg.drone,
                                      self.alloc, cfg.drone.dt_physics)

        self._walk_base()
        self._enforce_bounds()
        # Closest approach DURING the tick, not just at its end. _resolve_combat
        # runs at the 10 Hz policy rate after 20 physics substeps, and a head-on
        # pair closes ~2.6 m in that time against a 2.0 m intercept radius, so a
        # real interception can pass straight through between samples. Measured
        # 2026-09-11: 12-15% of genuine contacts were missed this way.
        self._sweep_prev = pos_before
        rew = self._resolve_combat()
        self.t += 1

        obs = self.observe()
        done, winner = self._terminal()
        w = winner.unsqueeze(-1)                                   # [B,1]
        rew["win"] = torch.where(
            done.unsqueeze(-1),
            torch.where(w == self.team, 1.0, -1.0) * (w >= 0).float(),
            torch.zeros(B, N, device=self.device))
        return obs, rew, done, {"winner": winner}

    # ---------------------------------------------------------------- geometry
    def _walk_base(self):
        """The defended base is a PERSON: it walks, so a defender cannot park.

        A slow random-walk heading at `base_speed`, clamped well inside the arena.
        base_speed = 0 reproduces the old static-base game exactly.
        """
        v = self.cfg.arena.base_speed
        if v <= 0.0:
            return
        B, dev = self.B, self.device
        # heading drifts slowly; no sharp turns, a person does not strafe
        self._base_heading = self._base_heading + (
            torch.rand(B, device=dev, generator=self.gen) - 0.5) * 0.25
        step = v * self.dt
        d = torch.stack([torch.cos(self._base_heading),
                         torch.sin(self._base_heading)], dim=-1) * step   # [B,2]
        lim = 0.42 * self.cfg.arena.size_x
        # only team 1's base (the defended one) walks
        nb = self.base_pos.clone()
        nb[:, 1, :2] = (nb[:, 1, :2] + d).clamp(-lim, lim)
        self.base_pos = nb

    def _enforce_bounds(self):
        a, s = self.cfg.arena, self.state
        lo = torch.tensor([-a.size_x / 2, -a.size_y / 2, a.floor_z], device=self.device)
        hi = torch.tensor([a.size_x / 2, a.size_y / 2, a.size_z], device=self.device)
        clamped = s.pos.clamp(lo, hi)
        hit = (clamped != s.pos).any(-1)
        s.pos.copy_(clamped)
        # kill only the velocity component that pushes into the wall, so a drone
        # slides along the geofence instead of sticking to it
        # remember the descent rate before the geofence cancels it, so a drone
        # that flies into the ground is scored on the speed it actually carried
        self._pre_clamp_vz = s.vel[..., 2].clone()
        at_lo, at_hi = s.pos <= lo + 1e-4, s.pos >= hi - 1e-4
        s.vel.copy_(torch.where(at_lo, s.vel.clamp_min(0.0),
                                torch.where(at_hi, s.vel.clamp_max(0.0), s.vel)))
        self._wall_hit = hit

    def _resolve_combat(self):
        cfg, B, N = self.cfg, self.B, self.N
        sw = cfg.swarm
        pos, alive = self.state.pos, self.alive
        R = quat_to_rotmat(self.state.quat)
        fwd = R[..., :, 0]                                    # body +x in world

        rel = pos.unsqueeze(1) - pos.unsqueeze(2)             # [B,N,N,3]
        dist = rel.norm(dim=-1)
        # Swept closest approach over the tick. Relative motion is linear to good
        # accuracy across one policy tick, so the true minimum separation is the
        # distance from the origin to the segment [rel_before, rel_after]. Using
        # only the endpoint lets a fast head-on pair tunnel through each other.
        prev = getattr(self, "_sweep_prev", None)
        if prev is not None:
            a = prev.unsqueeze(1) - prev.unsqueeze(2)          # [B,N,N,3]
            ab = rel - a
            tt = (-(a * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12))
            tt = tt.clamp(0.0, 1.0).unsqueeze(-1)
            dist = torch.minimum(dist, (a + ab * tt).norm(dim=-1))
        d = rel / dist.unsqueeze(-1).clamp_min(1e-6)
        cos_off = (d * fwd.unsqueeze(2)).sum(-1)

        enemy = self.team.unsqueeze(1) != self.team.unsqueeze(2)
        pair_alive = alive.unsqueeze(1) & alive.unsqueeze(2)

        # per-cause attribution, so nothing downstream has to guess why a drone died
        by_intercept = torch.zeros(B, N, device=self.device)
        by_coll = torch.zeros(B, N, device=self.device)
        by_ground = torch.zeros(B, N, device=self.device)
        dmg = torch.zeros(B, N, device=self.device)

        not_self = ~torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0)
        contact = (dist < sw.collision_radius) & pair_alive & not_self
        # enemies are killed inside the (larger) intercept envelope
        in_envelope = (dist < sw.intercept_radius) & pair_alive & not_self

        # ---- INTERCEPTION. There is no gun. A defender kills by flying INTO the
        # attacker, and both aircraft are destroyed -- which is how real
        # counter-drone interceptors work and what a 250 g quadrotor can actually
        # do. Every kill therefore costs the killer, so defending is a 1:1 trade
        # and the whole problem becomes one of interception geometry rather than
        # of holding a firing cone.
        intercept = in_envelope & enemy
        struck_enemy = intercept.any(-1)                      # [B,N] both sides
        self.last_intercept = intercept

        # ---- accidental contact between TEAMMATES is damage, not annihilation:
        # scaled by closing speed so tight formation flying stays possible.
        friendly = contact & ~enemy
        rel_speed = (self.state.vel.unsqueeze(2) - self.state.vel.unsqueeze(1)) \
            .norm(dim=-1)
        n_coll = friendly.sum(-1).float()
        coll_dmg = (friendly.float() * rel_speed).sum(-1) * COLLISION_DMG
        dmg = dmg + coll_dmg
        by_coll = by_coll + coll_dmg

        # --- ground impact: a drone that flies into the floor is wrecked by it
        on_floor = self.state.pos[..., 2] <= cfg.arena.floor_z + 1e-4
        descent = (-self._pre_clamp_vz).clamp_min(0.0)
        gnd_dmg = (on_floor & alive).float() * descent * GROUND_IMPACT_DMG
        dmg = dmg + gnd_dmg
        by_ground = by_ground + gnd_dmg

        prev_alive = alive.clone()
        self.health = (self.health - dmg).clamp_min(0.0)
        # an intercept is unconditionally fatal to BOTH aircraft, regardless of
        # remaining health -- it is a mid-air collision, not attrition
        self.health = torch.where(struck_enemy,
                                  torch.zeros_like(self.health), self.health)
        by_intercept = by_intercept + struck_enemy.float() * 1e6   # dominates cause
        killed = prev_alive & ~self.alive                     # [B,N] died this tick

        # --- kamikaze: entering the enemy base volume is an IMPACT, not a loiter.
        # The attacking drone is destroyed; the base takes damage proportional to
        # the speed it was carrying.
        to_base = pos.unsqueeze(2) - self.base_pos.unsqueeze(1)   # [B,N,2,3]
        in_base = to_base.norm(dim=-1) < self.cfg.arena.base_radius
        # A base that is already destroyed -- or that never existed, as the
        # attacker's does in attack/defend -- is not a target. Without this a
        # defender that strays over the enemy's dead base detonates for nothing,
        # which reads as friendly drones suiciding at random.
        base_alive = (self.base_health > 0).unsqueeze(1)          # [B,1,2]
        hostile = (torch.arange(2, device=self.device).view(1, 1, 2) !=
                   self.team.unsqueeze(-1)) & prev_alive.unsqueeze(-1) & base_alive
        impact = in_base & hostile                             # [B,N,2]
        speed = self.state.vel.norm(dim=-1)                    # [B,N]
        per_hit = (KAMIKAZE_BASE_DMG + KAMIKAZE_SPEED_DMG * speed).unsqueeze(-1)
        base_dmg = (impact.float() * per_hit).sum(1)           # [B,2]
        self.base_health = (self.base_health - base_dmg).clamp_min(0.0)

        struck = impact.any(-1)                                # [B,N] detonated
        self.health = torch.where(struck, torch.zeros_like(self.health), self.health)
        killed = killed | (prev_alive & struck)

        # 0 intercept, 1 friendly collision, 2 ground, 3 base strike; -1 = alive
        cause = torch.stack([by_intercept, by_coll, by_ground], dim=-1).argmax(-1)
        cause = torch.where(struck, torch.full_like(cause, 3), cause)
        self.last_deaths = (prev_alive & ~self.alive)
        self.last_death_cause = torch.where(
            self.last_deaths, cause, torch.full_like(cause, -1))

        # energy: hovering is not free, and neither is sprinting
        self.energy = (self.energy - 1e-3 * (1.0 + self.state.vel.norm(dim=-1) /
                                             cfg.drone.max_speed)).clamp_min(0.0)

        # Dense navigation shaping: metres closed on the enemy base this tick,
        # normalised by how far a drone could possibly travel. Without a dense
        # positive term, phase 1's reward is all penalties and the optimal policy
        # is to sit still -- which is exactly what the first run learned.
        en_base = self.base_pos.gather(
            1, self.enemy_of.unsqueeze(-1).expand(self.B, N, 3))
        d_now = (pos - en_base).norm(dim=-1)
        progress = (self._prev_base_dist - d_now) / (cfg.drone.max_speed * self.dt)
        self._prev_base_dist = d_now
        progress = progress.clamp(-1.0, 1.0) * self.alive.float()

        own_base_dmg = base_dmg.gather(1, self.team)            # [B,N] mine lost
        enemy_base_dmg = base_dmg.gather(1, self.enemy_of)
        team_kills = self._team_sum(killed.float(), other=True)
        team_losses = self._team_sum(killed.float(), other=False)

        return {
            "base_damage": enemy_base_dmg / cfg.arena.base_health,
            "base_preserved": -own_base_dmg / cfg.arena.base_health,
            "kills": team_kills,
            "losses": -team_losses,
            "collision": -n_coll,
            "energy": -1e-3 * self.state.vel.norm(dim=-1) / cfg.drone.max_speed,
            "coverage": torch.zeros(B, N, device=self.device),
            # pressing against the geofence is a policy failure, so it is priced
            # into the reward rather than being lethal
            "boundary": -self._wall_hit.float(),
            "progress": progress,
            # Time pressure, charged ONLY to the side that has to make something
            # happen. Without it an attacker has no reason to hurry and learns to
            # creep at 1 m/s with an 18 m/s airframe. But billing the DEFENDER for
            # every tick it survives prices its own win at -10 against the
            # attacker's +75, and with shared weights the gradient simply
            # abandons defence -- which is exactly what happened.
            "time": -self._attacker_mask().float() * self.alive.float(),
        }

    def _attacker_mask(self) -> Tensor:
        """True for drones whose job is to force a result.

        In attack/defend only red attacks; blue wins by surviving. In war both
        sides must destroy the other's base, so both are on the clock.
        """
        ad = (self.mode == MODE_ATTACK_DEFEND).unsqueeze(-1)      # [B,1]
        return ~(ad & (self.team == 1))

    def _team_sum(self, per_drone: Tensor, other: bool) -> Tensor:
        """Broadcast a per-drone quantity to a per-team total (shared credit)."""
        same = self.team.unsqueeze(1) == self.team.unsqueeze(2)
        sel = ~same if other else same
        return (sel.float() * per_drone.unsqueeze(1)).sum(-1)

    def _terminal(self):
        cfg = self.cfg
        alive_red = (self.alive & (self.team == 0)).sum(-1)
        alive_blue = (self.alive & (self.team == 1)).sum(-1)
        blue_base_down = self.base_health[:, 1] <= 0
        red_base_down = self.base_health[:, 0] <= 0
        timeout = self.t >= self.max_steps

        war = self.mode == MODE_WAR
        red_wins = torch.where(war, blue_base_down | (alive_blue == 0), blue_base_down)
        # ATTACK/DEFEND: the defender wins only with the base still standing.
        # Without that guard the mode scores its own failure as a draw -- an
        # attacker is DESTROYED by the kamikaze strike that kills the base, so
        # `blue_base_down` and `alive_red == 0` fire on the same tick, land in
        # `both`, and resolve to -1. Measured: `do_nothing` conceded the base in
        # every arena (hp 0.00 vs a rush) and still scored DEF 0.50. Every DEF
        # number in the project was inflated by it, and phase 5 was paying a
        # `win` reward for being overrun.
        blue_wins = torch.where(war, red_base_down | (alive_red == 0),
                                (timeout | (alive_red == 0)) & ~blue_base_down)
        done = red_wins | blue_wins | timeout
        # Mutual destruction is a DRAW. Resolving it in red's favour puts a
        # systematic thumb on the scale: identical agents then score 0.65 as red,
        # and every self-play statistic inherits the bias. (WAR mode only -- in
        # attack/defend the guard above makes the two outcomes disjoint.)
        both = red_wins & blue_wins
        winner = torch.where(both, -1,
                             torch.where(red_wins, 0, torch.where(blue_wins, 1, -1)))
        return done, winner

    # ------------------------------------------------------------- observation
    def observe(self):
        """Build each drone's egocentric, partial view. No ground truth leaks here."""
        cfg, B, N, K = self.cfg, self.B, self.N, self.K
        pos, vel, quat = self.state.pos, self.state.vel, self.state.quat
        alive = self.alive

        seen, meas = camera_detections(pos, quat, alive, cfg.sensor, self.gen)
        hop = mesh_hops(pos, alive, self.team, cfg.comms, self.gen)
        dist = torch.cdist(pos, pos)
        age, bel_pos = fuse_beliefs(seen, meas, hop, dist, cfg.comms)

        # teammates inside the mesh always report themselves, at hop-count staleness
        same = self.team.unsqueeze(1) == self.team.unsqueeze(2)
        ally_known = same & (hop < 1e8)
        age = torch.where(ally_known, torch.minimum(age, hop), age)
        bel_pos = torch.where(ally_known.unsqueeze(-1), pos.unsqueeze(1), bel_pos)

        known = age < 1e8
        known &= ~torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0)
        known &= alive.unsqueeze(1)

        # rank contacts: fresh and close first, enemies ahead of allies
        is_enemy = self.team.unsqueeze(1) != self.team.unsqueeze(2)
        bd = (bel_pos - pos.unsqueeze(2)).norm(dim=-1)
        score = bd + 30.0 * age.clamp_max(10.0) - 60.0 * is_enemy.float()
        score = torch.where(known, score, torch.full_like(score, float("inf")))
        # K is fixed by the exported model, but a small swarm may hold fewer than K
        # other drones. Take what exists and pad the rest as empty slots, so the same
        # weights run an 8-drone team and a 32-drone team unchanged.
        k_eff = min(K, N)
        idx = score.topk(k_eff, dim=-1, largest=False).indices            # [B,N,k_eff]
        mask = torch.gather(known, 2, idx)
        if k_eff < K:
            pad = K - k_eff
            idx = torch.cat([idx, idx.new_zeros(B, N, pad)], dim=-1)
            mask = torch.cat([mask, mask.new_zeros(B, N, pad)], dim=-1)

        R = quat_to_rotmat(quat)
        gpos = torch.gather(bel_pos, 2, idx.unsqueeze(-1).expand(B, N, K, 3))
        rel = gpos - pos.unsqueeze(2)
        rel_body = torch.einsum("bnji,bnkj->bnki", R, rel)
        gvel = vel.unsqueeze(1).expand(B, N, N, 3)
        gvel = torch.gather(gvel, 2, idx.unsqueeze(-1).expand(B, N, K, 3))
        rvel_body = torch.einsum("bnji,bnkj->bnki", R, gvel - vel.unsqueeze(2))

        g_age = torch.gather(age, 2, idx).clamp_max(20.0)
        g_enemy = torch.gather(is_enemy, 2, idx).float()
        g_direct = torch.gather(seen, 2, idx).float()
        g_health = torch.gather(
            (self.health / cfg.swarm.drone_health).unsqueeze(1).expand(B, N, N), 2, idx)
        g_dist = rel.norm(dim=-1)

        scale = cfg.sensor.max_range
        ent_feat = torch.cat([
            rel_body / scale, (g_dist / scale).unsqueeze(-1),
            rvel_body / cfg.drone.max_speed,
            g_enemy.unsqueeze(-1), (1 - g_enemy).unsqueeze(-1),
            g_health.unsqueeze(-1), (g_age / 20.0).unsqueeze(-1),
            g_direct.unsqueeze(-1),
            (g_dist < cfg.swarm.intercept_range).float().unsqueeze(-1),
        ], dim=-1) * mask.unsqueeze(-1).float()

        # ---- scalars
        a = cfg.arena
        vel_body = torch.einsum("bnji,bnj->bni", R, vel)
        own_base = self.base_pos.gather(1, self.team.unsqueeze(-1).expand(B, N, 3))
        en_base = self.base_pos.gather(1, self.enemy_of.unsqueeze(-1).expand(B, N, 3))
        rb = torch.einsum("bnji,bnj->bni", R, own_base - pos)
        re = torch.einsum("bnji,bnj->bni", R, en_base - pos)
        n_ally = (self.alive.unsqueeze(1) & same).sum(-1).float() / self.n
        n_known_enemy = (known & is_enemy).sum(-1).float() / self.n
        mode1h = torch.nn.functional.one_hot(self.mode, 2).float().unsqueeze(1) \
            .expand(B, N, 2)
        # role: in attack/defend red attacks and blue defends; in war both do both
        role = torch.where((self.mode.unsqueeze(1) == MODE_ATTACK_DEFEND),
                           self.team, torch.full_like(self.team, 2))
        role1h = torch.nn.functional.one_hot(role.clamp(0, 2), 3).float()[..., :2]

        scal = torch.cat([
            vel_body / cfg.drone.max_speed, R[..., :, 2], self.state.omega / 10.0,
            (pos[..., 2:] / a.size_z),
            (self.health / cfg.swarm.drone_health).unsqueeze(-1),
            rb / a.size_x, rb.norm(dim=-1, keepdim=True) / a.size_x,
            re / a.size_x, re.norm(dim=-1, keepdim=True) / a.size_x,
            (self.base_health.gather(1, self.team) / a.base_health).unsqueeze(-1),
            (self.base_health.gather(1, self.enemy_of) / a.base_health).unsqueeze(-1),
            n_ally.unsqueeze(-1), n_known_enemy.unsqueeze(-1),
            (1.0 - self.t.float() / self.max_steps).view(B, 1, 1).expand(B, N, 1),
            mode1h, role1h,
        ], dim=-1)

        self._last_belief = (age, bel_pos, {"gid": idx, "mask": mask})
        return {"scalars": scal, "entities": ent_feat, "entity_mask": mask,
                "entity_gid": idx, "alive": alive}

    # --------------------------------------------------- centralized critic obs
    def global_state(self) -> Tensor:
        """Ground-truth state for the centralized critic. TRAINING ONLY.

        AlphaStar does exactly this: the value function sees both players'
        perspectives during training. It is never part of the exported policy.
        """
        B, N = self.B, self.N
        R = quat_to_rotmat(self.state.quat)
        per_drone = torch.cat([
            self.state.pos / self.cfg.arena.size_x,
            self.state.vel / self.cfg.drone.max_speed,
            R[..., :, 0], R[..., :, 2],
            (self.health / self.cfg.swarm.drone_health).unsqueeze(-1),
            self.alive.float().unsqueeze(-1),
            self.team.float().unsqueeze(-1),
        ], dim=-1)
        glob = torch.cat([
            self.base_health / self.cfg.arena.base_health,
            (1.0 - self.t.float() / self.max_steps).unsqueeze(-1),
            torch.nn.functional.one_hot(self.mode, 2).float(),
        ], dim=-1)
        return torch.cat([per_drone.reshape(B, -1), glob], dim=-1)
