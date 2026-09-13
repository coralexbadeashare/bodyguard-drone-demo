"""Scripted opponents.

These exist to answer one question the loss curve cannot: *is the learned policy
actually good?* The project rule is explicit -- a policy that cannot beat
do-nothing is a failed policy no matter how the training curve looks. Phase 2 is
gated on beating every agent in this file.

Scripted agents may read `env` directly. They are evaluation instruments, not
deployable policies, so the partial-observability discipline that binds the
learned policy does not apply to them -- and holding them to it would make them
weaker opponents, which is the wrong direction for a baseline.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor

from ..sim.quadrotor import quat_to_rotmat
from ..sim.env import (HOLD, GOTO, ENGAGE, STRIKE_BASE, EVADE, REGROUP, SCOUT,
                       N_LOOK,
                       N_ACTION_TYPES, N_HEADING, N_PITCH, N_SPEED, N_DELAY)

# entity feature layout (see SwarmEnv.observe)
F_DIST, F_ENEMY, F_HEALTH, F_INRANGE = 3, 7, 9, 12


class ScriptedAgent:
    name = "scripted"

    def __init__(self, env, team: int, disperse: bool = True):
        # `disperse` spreads fire across contacts so a scripted swarm does not
        # converge on one point and collide with itself. It keys on the drone's
        # own index, which is NOT in the observation -- so a policy imitating this
        # teacher cannot predict the choice. Imitation uses disperse=False.
        self.disperse = disperse
        self.env, self.team = env, team
        self.B, self.N, self.K = env.B, env.N, env.K
        self.dev = env.device
        self.mine = (env.team == team)                      # [B,N] bool

    # -- helpers -------------------------------------------------------------
    def _blank(self) -> dict[str, Tensor]:
        z = lambda: torch.zeros(self.B, self.N, dtype=torch.long, device=self.dev)
        a = self._blank_cache = {
            "action_type": z(), "target": z(), "heading": z(), "pitch": z(),
            "speed": z(), "delay": z(),
            "comms": torch.zeros(self.B, self.N, self.env.cfg.comms.n_latent_tokens,
                                 dtype=torch.long, device=self.dev)}
        a["pitch"][:] = (N_PITCH - 1) // 2                  # level flight
        a["speed"][:] = N_SPEED - 1                         # full speed
        return a

    def _nearest_enemy_slot(self, obs, disperse: bool = True) -> tuple[Tensor, Tensor]:
        """Pick a visible enemy, spreading the swarm's fire across contacts.

        Every drone locking the single nearest contact makes the whole swarm
        converge on one point in space, where it collides with itself. Any real
        scripted swarm does target assignment; drone i takes the (i mod n)-th
        closest contact.
        """
        ent, mask = obs["entities"], obs["entity_mask"]
        ok = (ent[..., F_ENEMY] > 0.5) & mask
        d = torch.where(ok, ent[..., F_DIST], torch.full_like(ent[..., F_DIST], 1e9))
        has = ok.any(-1)
        if not disperse:
            return d.argmin(-1), has
        order = d.argsort(-1)                                    # [B,N,K]
        n_vis = ok.sum(-1).clamp_min(1)                          # [B,N]
        # index WITHIN the team, not the global drone index: red holds 0..n-1 and
        # blue holds n..2n-1, so a global index gives the two teams different
        # target-assignment offsets and the mirror match stops being 0.5
        within = torch.arange(self.N, device=self.dev) % self.env.n
        rank = (within.view(1, -1) % n_vis)
        return order.gather(-1, rank.unsqueeze(-1)).squeeze(-1), has

    def _world_heading_bin(self, target_xy: Tensor) -> Tensor:
        """World-frame azimuth bin pointing from each drone at `target_xy`."""
        d = target_xy - self.env.state.pos[..., :2]
        az = torch.atan2(d[..., 1], d[..., 0]) % (2 * math.pi)
        return (az / (2 * math.pi / N_HEADING)).round().long() % N_HEADING

    def act(self, obs) -> dict[str, Tensor]:
        raise NotImplementedError


class DoNothing(ScriptedAgent):
    """The floor. Anything that cannot beat this has learned nothing."""
    name = "do_nothing"

    def act(self, obs):
        a = self._blank()
        a["action_type"][:] = HOLD
        a["speed"][:] = 0
        return a


class RandomAgent(ScriptedAgent):
    name = "random"

    def __init__(self, env, team, disperse: bool = True, seed=0):
        super().__init__(env, team, disperse)
        self.g = torch.Generator(device=self.dev).manual_seed(seed)

    def act(self, obs):
        r = lambda hi: torch.randint(0, hi, (self.B, self.N), device=self.dev,
                                     generator=self.g)
        a = self._blank()
        for k, hi in (("action_type", N_ACTION_TYPES), ("target", self.K),
                      ("heading", N_HEADING), ("pitch", N_PITCH),
                      ("speed", N_SPEED), ("delay", N_DELAY)):
            a[k] = r(hi)
        return a


class Greedy(ScriptedAgent):
    """Attack the nearest thing you can see; otherwise advance on the base."""
    name = "greedy"

    def act(self, obs):
        a = self._blank()
        slot, has = self._nearest_enemy_slot(obs, self.disperse)
        a["target"] = slot
        a["action_type"] = torch.where(has, torch.full_like(slot, ENGAGE),
                                       torch.full_like(slot, STRIKE_BASE))
        return a


class Rush(ScriptedAgent):
    """Ignore everything, drive at the enemy base. Punishes passive defenders."""
    name = "rush"

    def act(self, obs):
        a = self._blank()
        a["action_type"][:] = STRIKE_BASE
        return a


class Perimeter(ScriptedAgent):
    """Hold a ring around the home base and engage whatever enters it.

    The natural counter to `Rush`, and the agent a naive attacker loses to.
    """
    name = "perimeter"

    def __init__(self, env, team, disperse: bool = True, radius: float = 22.0):
        super().__init__(env, team, disperse)
        self.radius = radius

    def act(self, obs):
        a = self._blank()
        slot, has = self._nearest_enemy_slot(obs, self.disperse)
        home = self.env.base_pos.gather(
            1, self.env.team.unsqueeze(-1).expand(self.B, self.N, 3))
        d_home = (self.env.state.pos - home).norm(dim=-1)

        far = d_home > self.radius * 1.3
        a["heading"] = self._world_heading_bin(home[..., :2])
        a["target"] = slot
        # engage anything visible, otherwise return to the ring, otherwise loiter
        a["action_type"] = torch.where(
            has, torch.full_like(slot, ENGAGE),
            torch.where(far, torch.full_like(slot, GOTO),
                        torch.full_like(slot, HOLD)))
        a["speed"] = torch.where(far, torch.full_like(slot, N_SPEED - 1),
                                 torch.full_like(slot, 1))
        return a


class Saturation(ScriptedAgent):
    """Doctrine-realistic saturation attack.

    Built from how massed FPV/loitering-munition attacks are actually described
    in the open literature, rather than invented:

      * MULTI-AXIS -- the swarm splits into groups that converge on the target
        from different bearings and different altitudes, instead of arriving as
        one stream a defender can service in order
      * TIME ON TARGET -- groups hold at dispersed initial points and dive
        TOGETHER, so the defence faces every threat at once rather than
        sequentially. This is the whole point of saturation: defeat the
        defender's engagement *rate*, not its accuracy
      * TERMINAL DIVE -- once committed, maximum speed straight in, which also
        maximises kinetic damage on impact

    Sources: multi-vector convergence and altitude dispersion, and simultaneous
    massed waves overwhelming legacy air defence, are the two features every
    account of these attacks agrees on.
    """
    name = "saturation"

    def __init__(self, env, team, disperse: bool = True, n_groups: int = 4,
                 ip_radius: float = 96.0, sync_frac: float = 0.65):
        super().__init__(env, team, disperse)
        self.n_groups, self.ip_radius, self.sync_frac = n_groups, ip_radius, sync_frac
        self.committed = torch.zeros(self.B, dtype=torch.bool, device=self.dev)
        g = (torch.arange(self.N, device=self.dev) % n_groups).float()
        ang = g * (2 * math.pi / n_groups) + math.pi     # spread around the target; the IP ring sits OUTSIDE the
        # defender's 70 m detection range so the swarm assembles unseen
        # alternate low and high ingress so the defence cannot cover one band
        alt = torch.where((torch.arange(self.N, device=self.dev) // n_groups) % 2 == 0,
                          10.0, 34.0)
        tgt = self.env.base_pos.gather(
            1, self.env.enemy_of.unsqueeze(-1).expand(self.B, self.N, 3))
        self.ip = torch.stack([
            tgt[..., 0] + ip_radius * torch.cos(ang).unsqueeze(0),
            tgt[..., 1] + ip_radius * torch.sin(ang).unsqueeze(0),
            alt.unsqueeze(0).expand(self.B, self.N),
        ], dim=-1)

    def _bins_toward(self, dst, speed_bin):
        d = dst - self.env.state.pos
        az = torch.atan2(d[..., 1], d[..., 0]) % (2 * math.pi)
        head = (az / (2 * math.pi / N_HEADING)).round().long() % N_HEADING
        horiz = d[..., :2].norm(dim=-1).clamp_min(1e-3)
        el = torch.atan2(d[..., 2], horiz)
        pit = ((el / math.radians(20.0)).round().long() + (N_PITCH - 1) // 2)
        return head, pit.clamp(0, N_PITCH - 1), torch.full_like(head, speed_bin)

    def act(self, obs):
        a = self._blank()
        env = self.env
        tgt = env.base_pos.gather(
            1, self.env.enemy_of.unsqueeze(-1).expand(self.B, self.N, 3))

        at_ip = (env.state.pos - self.ip).norm(dim=-1) < 16.0
        mine = (env.team == self.team) & env.alive
        ready = ((at_ip & mine).sum(-1).float()
                 / mine.sum(-1).clamp_min(1).float()) >= self.sync_frac
        self.committed |= ready                     # time-on-target: dive together

        h_ip, p_ip, s_ip = self._bins_toward(self.ip, N_SPEED - 1)
        go = self.committed.view(self.B, 1).expand(self.B, self.N)

        # before commit: run to the initial point, or hold there and wait
        hold = at_ip & ~go
        a["heading"], a["pitch"] = h_ip, p_ip
        a["speed"] = torch.where(hold, torch.zeros_like(s_ip), s_ip)
        a["action_type"] = torch.where(
            go, torch.full_like(h_ip, STRIKE_BASE),      # terminal dive
            torch.where(hold, torch.full_like(h_ip, HOLD),
                        torch.full_like(h_ip, GOTO)))
        return a


class Interceptor(ScriptedAgent):
    """Lead-pursuit base defence -- the reference a guard policy has to beat.

    Every other defender here (`Greedy`, `Perimeter`) steers at where the target
    IS. With a gun that was right: you point the nose and fire. With interception
    it is wrong, and quietly so -- a tail-chase against an attacker with the same
    top speed never closes, so a pure-pursuit defender scores kills only when the
    geometry already favoured it. That makes it useless as a baseline: it flatters
    any learned policy that discovered lead.

    So this one solves the collision triangle. For target position p_t, target
    velocity v_t, own position p_o and own top speed s, find the earliest t with

        |p_t + v_t.t - p_o| = s.t
        (|v_t|^2 - s^2) t^2 + 2 (r.v_t) t + |r|^2 = 0,     r = p_t - p_o

    and steer at the aim point p_t + v_t.t. That is a constant-bearing course:
    the target's line of sight stops rotating and the two airframes converge.

    Three details that matter:

      * THREAT ORDER, not proximity. Targets are ranked by their distance to the
        base being defended -- which, in the bodyguard scenario, is walking. A
        defender that services the nearest contact instead of the most immediate
        threat lets the leader through. Defender i takes the i-th most
        threatening attacker, so two defenders never fly at the same one.
      * QUANTISATION IS SURVIVABLE. Heading is 16 bins (22.5 deg, so up to
        11.25 deg of aim error), but guidance re-solves every policy tick. At a
        ~30 m/s closing speed the final 0.1 s tick leaves ~3 m uncorrected, i.e.
        at most 3*sin(11.25 deg) = 0.58 m of lateral miss against a 2.0 m
        intercept radius. The loop closes faster than the error grows.
      * ESCORT WHEN BLIND. With nothing airborne to chase, the defenders hold a
        spread ring on the base and move with it, rather than parking on the spot
        the base has already left.
    """
    name = "interceptor"

    def __init__(self, env, team: int, disperse: bool = True,
                 station_r: float = 12.0, station_alt: float = 14.0,
                 t_max: float = 8.0):
        super().__init__(env, team, disperse)
        self.station_r, self.station_alt, self.t_max = station_r, station_alt, t_max
        within = torch.arange(self.N, device=self.dev) % env.n
        self.phase = within.float() * (2 * math.pi / max(1, env.n))   # ring offset

    def _lead_time(self, r: Tensor, v: Tensor, s: float) -> Tensor:
        """Earliest positive root of |r + v.t| = s.t, else the pure-pursuit time."""
        a = (v * v).sum(-1) - s * s
        b = 2.0 * (r * v).sum(-1)
        c = (r * r).sum(-1)
        fallback = c.sqrt() / s                                   # ignore target motion
        disc = b * b - 4.0 * a * c
        sq = disc.clamp_min(0.0).sqrt()
        # both roots; pick the smallest strictly positive one
        den = (2.0 * a)
        # |v_t| == s exactly: the quadratic degenerates to a line, t = -c/b, and
        # only a closing geometry (b < 0) has a solution at all.
        lin = torch.where(b < -1e-6, -c / b.clamp_max(-1e-6), fallback)
        t1 = (-b - sq) / den.where(den.abs() > 1e-6, torch.full_like(den, 1e-6))
        t2 = (-b + sq) / den.where(den.abs() > 1e-6, torch.full_like(den, 1e-6))
        big = torch.full_like(t1, 1e9)
        lo = torch.minimum(torch.where(t1 > 1e-4, t1, big),
                           torch.where(t2 > 1e-4, t2, big))
        t = torch.where(a.abs() < 1e-6, lin, lo)
        t = torch.where((t > 0) & (t < 1e8), t, fallback)
        return t.clamp(0.0, self.t_max)

    def _bins_toward(self, d: Tensor):
        az = torch.atan2(d[..., 1], d[..., 0]) % (2 * math.pi)
        head = (az / (2 * math.pi / N_HEADING)).round().long() % N_HEADING
        horiz = d[..., :2].norm(dim=-1).clamp_min(1e-3)
        el = torch.atan2(d[..., 2], horiz)
        pit = (el / math.radians(20.0)).round().long() + (N_PITCH - 1) // 2
        return head, pit.clamp(0, N_PITCH - 1)

    def act(self, obs):
        env, a = self.env, self._blank()
        B, N = self.B, self.N
        pos, vel = env.state.pos, env.state.vel
        smax = env.cfg.drone.max_speed

        home = env.base_pos.gather(
            1, env.team.unsqueeze(-1).expand(B, N, 3))                # [B,N,3]
        # --- threat ranking: enemies sorted by how close they are to MY base
        foe = (env.team.unsqueeze(1) != env.team.unsqueeze(2)) & env.alive.unsqueeze(1)
        base_me = env.base_pos.gather(1, env.team.unsqueeze(-1).expand(B, N, 3))
        d_base = (pos - base_me).norm(dim=-1)                         # [B,N] enemy->its foe's base
        thr = torch.where(foe, d_base.unsqueeze(1).expand(B, N, N),
                          torch.full((B, N, N), 1e9, device=self.dev))
        order = thr.argsort(-1)                                       # [B,N,N]
        n_foe = foe.sum(-1).clamp_min(1)                              # [B,N]
        if self.disperse:
            within = torch.arange(N, device=self.dev) % env.n
            rank = within.view(1, -1) % n_foe
        else:
            # As a BC TEACHER the choice has to be a function of the observation
            # alone. Threat rank keyed on the drone's own index is not -- the
            # index is not in the observation, so a student cannot predict which
            # attacker this defender was assigned and learns the average of
            # several targets. With disperse=False every defender commits to the
            # single most threatening attacker, which is fully determined by what
            # it can see.
            rank = torch.zeros_like(n_foe)
        sel = order.gather(-1, rank.unsqueeze(-1)).squeeze(-1)        # [B,N] enemy index
        has = foe.any(-1)

        tp = pos.gather(1, sel.unsqueeze(-1).expand(B, N, 3))
        tv = vel.gather(1, sel.unsqueeze(-1).expand(B, N, 3))
        r = tp - pos
        t = self._lead_time(r, tv, smax)
        aim = tp + tv * t.unsqueeze(-1) - pos                         # steer vector

        # --- escort: a spread ring that travels with the (walking) base
        ang = self.phase.view(1, N) + 0.0
        ring = torch.stack([home[..., 0] + self.station_r * torch.cos(ang),
                            home[..., 1] + self.station_r * torch.sin(ang),
                            torch.full_like(home[..., 2], self.station_alt)], dim=-1)
        to_ring = ring - pos
        near = to_ring.norm(dim=-1) < 3.0

        steer = torch.where(has.unsqueeze(-1), aim, to_ring)
        head, pit = self._bins_toward(steer)
        a["heading"], a["pitch"] = head, pit
        a["action_type"][:] = GOTO
        # full speed onto an intercept; loiter once on station with nothing to chase
        a["speed"] = torch.where(has | ~near, torch.full_like(head, N_SPEED - 1),
                                 torch.full_like(head, 1))
        return a


class ObsInterceptor(Interceptor):
    """The interceptor, restricted to what the drone can actually see.

    `Interceptor` reads `env` directly and engages every living attacker, visible
    or not. That is allowed for an evaluation baseline -- and it is why it scores
    DEF 0.97. But no exported policy can reproduce it: the policy sees camera
    cones plus the BLE mesh, so it cannot begin a turn against a contact it has
    no knowledge of. Using 0.97 as the target therefore measures a learned policy
    against something unreachable, and cloning it teaches turns the student has
    no way to predict.

    This variant runs the identical guidance law on the observation alone. On the
    default 70 m / 90 deg sensor it scores DEF 0.61 against a rush. **That, not
    0.97, is the bar a partially-observing policy can actually clear.**

    Two details the observation forces:

      * Entity feature [4:7] is velocity RELATIVE to this drone. The collision
        triangle needs the target's ABSOLUTE velocity, so own velocity is added
        back; feeding the relative one solves for the wrong intercept time.
      * With no contact there is nothing to intercept, so it escorts the base.
        The full-observability version is effectively always pursuing, which is
        most of the 0.97 vs 0.61 gap.
    """
    name = "obs_interceptor"

    def act(self, obs):
        env, a = self.env, self._blank()
        Bq, N = self.B, self.N
        pos, smax = env.state.pos, env.cfg.drone.max_speed
        ent, mask = obs["entities"], obs["entity_mask"]
        ok = (ent[..., F_ENEMY] > 0.5) & mask
        has = ok.any(-1)
        d = torch.where(ok, ent[..., F_DIST],
                        torch.full_like(ent[..., F_DIST], 1e9))
        slot = d.argmin(-1)
        # gather along the ENTITY axis (2). dim 1 is the drone axis and has size
        # N, so gather(1, .) silently reads another drone's contact list.
        rel = ent.gather(2, slot.view(Bq, N, 1, 1)
                         .expand(Bq, N, 1, ent.shape[-1])).squeeze(2)
        R = quat_to_rotmat(env.state.quat)                    # body -> world
        scale = env.cfg.sensor.max_range
        r = torch.einsum("bnij,bnj->bni", R, rel[..., 0:3] * scale)
        v = torch.einsum("bnij,bnj->bni", R, rel[..., 4:7] * smax) + env.state.vel
        t = self._lead_time(r, v, smax)
        aim = r + v * t.unsqueeze(-1)

        home = env.base_pos.gather(1, env.team.unsqueeze(-1).expand(Bq, N, 3))
        ang = self.phase.view(1, N)
        ring = torch.stack([home[..., 0] + self.station_r * torch.cos(ang),
                            home[..., 1] + self.station_r * torch.sin(ang),
                            torch.full_like(home[..., 2], self.station_alt)], dim=-1)
        to_ring = ring - pos
        near = to_ring.norm(dim=-1) < 3.0
        steer = torch.where(has.unsqueeze(-1), aim, to_ring)
        head, pit = self._bins_toward(steer)
        a["heading"], a["pitch"] = head, pit
        a["action_type"][:] = GOTO
        a["speed"] = torch.where(has | ~near, torch.full_like(head, N_SPEED - 1),
                                 torch.full_like(head, 1))
        return a


class SectorInterceptor(ObsInterceptor):
    """`obs_interceptor` that also decides where to LOOK.

    Each defender is assigned a bearing and holds it while it has no contact, so
    the swarm tiles the horizon between its members instead of each drone
    pointing wherever it happened to drift. With `sensor.layout="azimuth"` and
    n_cameras=C, D defenders cover D*C sectors, which means each camera can be
    C*D times narrower -- and detection range scales as 1/FOV.

    Once it HAS a contact it looks at the contact, not the sector: the point of
    scanning is to acquire, and after that keeping the target in frame matters
    more than covering the arc.
    """
    name = "sector_interceptor"

    def __init__(self, env, team: int, disperse: bool = True, **kw):
        super().__init__(env, team, disperse, **kw)
        n_cam = max(1, env.cfg.sensor.n_cameras)
        d = max(1, env.n)
        # front camera of drone i at i*360/(D*C) so the C cameras of D drones
        # tile the horizon exactly once
        within = torch.arange(self.N, device=self.dev) % d
        self.sector = within.float() * (2 * math.pi / (d * n_cam))

    def act(self, obs):
        a = super().act(obs)
        env = self.env
        ent, mask = obs["entities"], obs["entity_mask"]
        ok = (ent[..., F_ENEMY] > 0.5) & mask
        has = ok.any(-1)
        # bearing to the contact we are steering at, else the assigned sector
        R = quat_to_rotmat(env.state.quat)
        d = torch.where(ok, ent[..., F_DIST], torch.full_like(ent[..., F_DIST], 1e9))
        slot = d.argmin(-1)
        rel = ent.gather(2, slot.view(self.B, self.N, 1, 1)
                         .expand(self.B, self.N, 1, ent.shape[-1])).squeeze(2)
        r = torch.einsum("bnij,bnj->bni", R, rel[..., 0:3])
        tgt_yaw = torch.atan2(r[..., 1], r[..., 0]) % (2 * math.pi)
        yaw = torch.where(has, tgt_yaw, self.sector.view(1, -1).expand(self.B, self.N))
        a["look"] = (yaw / (2 * math.pi / N_LOOK)).round().long() % N_LOOK
        return a


class ProNavInterceptor(ObsInterceptor):
    """Proportional navigation: steer on the BEARING RATE alone. No range.

    `obs_interceptor` still solves the collision triangle, which needs the
    target's position AND velocity in 3-D. A monocular camera supplies neither --
    it measures direction well and range badly. This agent is the deployable
    formulation: it uses only the line-of-sight DIRECTION and how that direction
    is changing, both of which a camera measures directly.

    Pure PN commands lateral acceleration a = N * V_own * (omega_LOS x u_LOS).
    Writing omega_LOS = u x du/dt and expanding,

        (u x du/dt) x u  =  du/dt * (u.u)  -  u * (u . du/dt)  =  du/dt

    because u is a unit vector, so u . du/dt = 0. The whole guidance law collapses
    to *the change in the line-of-sight unit vector* -- which is exactly what a
    tracked blob's drift across the image plane is. The commanded direction is

        aim  =  normalise( u + G * delta_u )

    with G a single gain standing in for N * time-to-go. Range never appears.

    `strict` also removes range from TARGET SELECTION (picking the most
    boresight-centred contact instead of the nearest), so that the agent uses no
    range anywhere at all. With strict=False, selection matches `obs_interceptor`
    and the comparison isolates the guidance law by itself.
    """
    name = "pronav"
    strict = False

    def __init__(self, env, team: int, disperse: bool = True, gain: float = 6.0,
                 **kw):
        super().__init__(env, team, disperse, **kw)
        self.gain = gain
        self._prev_u = torch.zeros(self.B, self.N, 3, device=self.dev)
        self._prev_gid = torch.full((self.B, self.N), -1, dtype=torch.long,
                                    device=self.dev)

    def act(self, obs):
        env, a = self.env, self._blank()
        Bq, N = self.B, self.N
        pos, smax = env.state.pos, env.cfg.drone.max_speed
        ent, mask, gid = obs["entities"], obs["entity_mask"], obs["entity_gid"]
        ok = (ent[..., F_ENEMY] > 0.5) & mask
        has = ok.any(-1)

        R = quat_to_rotmat(env.state.quat)
        # unit line of sight, from the body-frame bearing. NOTE: only the
        # DIRECTION of feature [0:3] is used -- its magnitude (range) is not.
        los_all = torch.einsum("bnij,bnkj->bnki", R, ent[..., 0:3])
        los_all = los_all / los_all.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.strict:
            # most boresight-centred visible contact: a camera can rank by angle
            # off centre without knowing how far anything is.
            fwd = R[..., 0]                                   # body x in world
            cosang = (los_all * fwd.unsqueeze(2)).sum(-1)
            key = torch.where(ok, -cosang, torch.full_like(cosang, 1e9))
        else:
            key = torch.where(ok, ent[..., F_DIST],
                              torch.full_like(ent[..., F_DIST], 1e9))
        if self.disperse:
            # spread the defenders across contacts exactly as ObsInterceptor
            # does, so a comparison against it isolates the GUIDANCE LAW rather
            # than the target-assignment policy. Measured: dispersion is worth
            # ~0.20 DEF on its own, far more than the choice of guidance.
            order = key.argsort(-1)
            n_vis = ok.sum(-1).clamp_min(1)
            within = torch.arange(N, device=self.dev) % max(1, env.n)
            rank = within.view(1, -1) % n_vis
            slot = order.gather(-1, rank.unsqueeze(-1)).squeeze(-1)
        else:
            slot = key.argmin(-1)

        u = los_all.gather(2, slot.view(Bq, N, 1, 1).expand(Bq, N, 1, 3)).squeeze(2)
        tgt_gid = gid.gather(2, slot.unsqueeze(-1)).squeeze(-1)

        # delta_u, but only where we were already tracking THIS contact
        same = (tgt_gid == self._prev_gid) & has
        du = torch.where(same.unsqueeze(-1), u - self._prev_u,
                         torch.zeros_like(u))
        aim = u + self.gain * du
        aim = aim / aim.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        self._prev_u = torch.where(has.unsqueeze(-1), u, torch.zeros_like(u))
        self._prev_gid = torch.where(has, tgt_gid,
                                     torch.full_like(tgt_gid, -1))

        home = env.base_pos.gather(1, env.team.unsqueeze(-1).expand(Bq, N, 3))
        ang = self.phase.view(1, N)
        ring = torch.stack([home[..., 0] + self.station_r * torch.cos(ang),
                            home[..., 1] + self.station_r * torch.sin(ang),
                            torch.full_like(home[..., 2], self.station_alt)], -1)
        to_ring = ring - pos
        near = to_ring.norm(dim=-1) < 3.0
        steer = torch.where(has.unsqueeze(-1), aim, to_ring)
        head, pit = self._bins_toward(steer)
        a["heading"], a["pitch"] = head, pit
        a["action_type"][:] = GOTO
        a["speed"] = torch.where(has | ~near, torch.full_like(head, N_SPEED - 1),
                                 torch.full_like(head, 1))
        return a


class ProNavStrict(ProNavInterceptor):
    """ProNav with range removed from target selection as well."""
    name = "pronav_strict"
    strict = True


BASELINES = {c.name: c for c in
             (DoNothing, RandomAgent, Greedy, Rush, Perimeter, Saturation, Interceptor,
              ObsInterceptor, SectorInterceptor,
              ProNavInterceptor, ProNavStrict)}


def make(name: str, env, team: int, **kw) -> ScriptedAgent:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; have {sorted(BASELINES)}")
    return BASELINES[name](env, team, **kw)


def merge(red_act: dict[str, Tensor], blue_act: dict[str, Tensor],
          n_per_team: int) -> dict[str, Tensor]:
    """Splice two per-team action dicts into one env action dict.

    Iterate the UNION of keys, not just red's. Optional heads (`look`) are emitted
    by some agents and not others, and keying on red alone silently dropped
    whatever only blue produced -- the sector experiment ran three times with the
    look command discarded before a matched control caught it.

    A side that did not emit an optional head is filled with -1, which every
    consumer must read as "no command".
    """
    out = {}
    for k in set(red_act) | set(blue_act):
        r, b = red_act.get(k), blue_act.get(k)
        ref = r if r is not None else b
        if r is None:
            r = torch.full_like(ref, -1)
        if b is None:
            b = torch.full_like(ref, -1)
        out[k] = torch.cat([r[:, :n_per_team], b[:, n_per_team:]], dim=1)
    return out
