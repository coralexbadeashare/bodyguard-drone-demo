"""Render a match to an animated GIF.

Runs the trained policy against a scripted opponent in attack/defend and draws
every tick: drones, bases, vision cones, mesh links, and explosions coloured by
cause. Purely diagnostic -- it uses the same Python simulator the policy trains
in, so what you see is what the policy actually did.
"""
from __future__ import annotations
import argparse, math, sys, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio

sys.path.insert(0, "/workspace/DroneSwarmRL")
from swarmstar.sim.config import SimConfig
from swarmstar.sim.env import SwarmEnv, MODE_ATTACK_DEFEND, MODE_WAR
from swarmstar.model.networks import SwarmPolicy
from swarmstar.train import baselines as B
from swarmstar.train.evaluate import PolicyAgent

BLAST = {"kamikaze": ("#ffb43c", 260), "collision": ("#ff78c8", 150),
         "ground": ("#b48c5a", 110), "shot": ("#ffeb8c", 90)}


def _yaw_of(env):
    """World-frame heading of every drone, for drawing camera footprints."""
    from swarmstar.sim.quadrotor import quat_to_rotmat
    R = quat_to_rotmat(env.state.quat)[0]
    return torch.atan2(R[:, 1, 0], R[:, 0, 0]).cpu().numpy()


def rollout(env, red, blue, ticks):
    """Play one match, capturing what is needed to draw it."""
    frames = []
    obs = env.observe()
    for t in range(ticks):
        ra = red.act(obs) if hasattr(red, "act") else red
        ba = blue.act(obs)
        act = B.merge(ra, ba, env.n)
        prev_alive = env.alive[0].clone()
        prev_pos = env.state.pos[0].clone()
        obs, rew, done, info = env.step(act)
        # Ground truth from the simulator -- never inferred from geometry.
        events = []
        if hasattr(env, "last_death_cause"):
            names = {0: "shot", 1: "collision", 2: "ground", 3: "kamikaze"}
            for i in env.last_deaths[0].nonzero().flatten().tolist():
                events.append((prev_pos[i].cpu().numpy(),
                               names[int(env.last_death_cause[0, i])]))
        shots = []
        if hasattr(env, "last_firing"):
            for i in env.last_firing[0].nonzero().flatten().tolist():
                j = int(env.last_fire_target[0, i])
                shots.append((env.state.pos[0, i].cpu().numpy(),
                              env.state.pos[0, j].cpu().numpy(), i))
        frames.append({
            "shots": shots,
            "pos": env.state.pos[0].cpu().numpy(),
            "alive": env.alive[0].cpu().numpy(),
            "active": (env.active[0].cpu().numpy()
                       if hasattr(env, "active") else
                       np.ones(env.N, dtype=bool)),
            "hp": (env.health[0] / env.cfg.swarm.drone_health).cpu().numpy(),
            "yaw": _yaw_of(env),
            "base": env.base_health[0].cpu().numpy(),
            "events": events, "t": t,
            "done": bool(done[0]), "winner": int(info["winner"][0]),
        })
        if bool(done[0]):
            break
    return frames


SUBTITLE = ("ground shadows give exact position   |   thin lines = weapon fire, "
            "35 m range   |   * strike  * shot  * crash")


def draw(frames, out, cfg, n, title, every=2, show_cones=False, fps=16, elev=22, azim0=-58):
    """3D arena view.

    Two things make a 3D scatter readable rather than ambiguous:
      * every drone drops a SHADOW onto the ground plane, so its x,y is exact --
        without it you cannot tell whether a drone is above the base or behind it
      * the axes are stretched to fill the canvas; matplotlib's 3D default wastes
        most of the frame on margin
    """
    a = cfg.arena
    HX, HY, HZ = a.size_x / 2, a.size_y / 2, a.size_z
    RED, BLUE = "#ff5a5f", "#4da3ff"
    blasts, imgs, trails = [], [], {}
    sel = frames[::every]
    th = np.linspace(0, 2 * np.pi, 64)

    for k, f in enumerate(sel):
        pos, alive, hp = f["pos"], f["alive"], f["hp"]
        # Inactive tensor slots spawn destroyed and never fly. Drawing them
        # scatters 27 dead-drone crosses over a 3v2 scenario.
        active = f.get("active", np.ones(len(pos), dtype=bool))
        for i in range(len(pos)):
            trails.setdefault(i, []).append(pos[i, :3].copy())
            if len(trails[i]) > 16:
                trails[i].pop(0)

        fig = plt.figure(figsize=(9.2, 5.6), dpi=118)
        fig.patch.set_facecolor("#0b0e14")
        ax = fig.add_axes([-0.07, -0.10, 1.14, 1.10], projection="3d")
        ax.set_facecolor("#0b0e14")
        ax.set_xlim(-HX, HX); ax.set_ylim(-HY, HY); ax.set_zlim(0, HZ * 0.7)
        ax.set_box_aspect((1.0, 1.0, 0.34), zoom=1.42)
        ax.view_init(elev=elev, azim=azim0 + k * 0.30)
        ax.set_axis_off()

        # camera footprints -- what each defender can actually SEE. This is the
        # whole argument for the hardware spec, so it belongs in the picture:
        # a wedge per camera, spanning the real FOV out to the real range.
        yaw = f.get("yaw")
        if yaw is not None and show_cones:
            from swarmstar.sim.perception import camera_dirs
            sp = cfg.sensor
            cams = camera_dirs(sp.n_cameras, "cpu",
                               getattr(sp, "layout", "front_down")).numpy()
            half = sp.fov_h / 2.0
            for i in range(len(pos)):
                if not (alive[i] and active[i]):
                    continue
                col = RED if i < n else BLUE
                for c in cams:
                    if abs(c[2]) > 0.7:            # a down-facing camera has no
                        continue                   # horizon footprint to draw
                    b = math.atan2(c[1], c[0]) + yaw[i]
                    arc = np.linspace(b - half, b + half, 14)
                    xs = np.concatenate([[pos[i, 0]], pos[i, 0] + sp.max_range * np.cos(arc)])
                    ys = np.concatenate([[pos[i, 1]], pos[i, 1] + sp.max_range * np.sin(arc)])
                    zs = np.full_like(xs, pos[i, 2])
                    ax.add_collection3d(Poly3DCollection(
                        [list(zip(xs, ys, zs))], facecolor=col, alpha=0.055,
                        edgecolor=col, linewidths=0.45, zorder=1))

        # ground grid, drawn by hand so it reads as a floor
        for g in np.arange(-HX, HX + 1, 25):
            ax.plot([g, g], [-HY, HY], [0, 0], color="#18202e", lw=0.7, zorder=0)
            ax.plot([-HX, HX], [g, g], [0, 0], color="#18202e", lw=0.7, zorder=0)

        # bases: ground ring, glow, and a health arc that empties
        for b, col in ((0, RED), (1, BLUE)):
            frac = float(f["base"][b]) / a.base_health
            if frac <= 0.001 and b == 0:
                continue
            bx, by = env_bx[b], env_by[b]
            ax.plot(bx + a.base_radius * np.cos(th), by + a.base_radius * np.sin(th),
                    np.zeros_like(th), color=col, lw=2.4,
                    alpha=0.3 + 0.7 * max(frac, 0), zorder=2)
            if frac > 0:
                t2 = np.linspace(0, 2 * np.pi * frac, max(int(64 * frac), 2))
                ax.plot(bx + a.base_radius * 1.5 * np.cos(t2),
                        by + a.base_radius * 1.5 * np.sin(t2),
                        np.zeros_like(t2), color=col, lw=4, alpha=0.75, zorder=2)
            ax.scatter([bx], [by], [0], s=90, c=col, alpha=0.25 + 0.5 * max(frac, 0),
                       marker="s", edgecolors="none", zorder=2)

        # trails, plus their ground shadows
        for i, tr in trails.items():
            if not alive[i] or not active[i] or len(tr) < 2:
                continue
            t = np.array(tr); col = RED if i < n else BLUE
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=col, lw=1.1, alpha=0.35, zorder=4)
            ax.plot(t[:, 0], t[:, 1], np.zeros(len(t)), color=col, lw=0.7,
                    alpha=0.10, zorder=1)

        for team, col in ((0, RED), (1, BLUE)):
            live = [i for i in range(len(pos))
                    if (i < n) == (team == 0) and alive[i] and active[i]]
            dead = [i for i in range(len(pos))
                    if (i < n) == (team == 0) and not alive[i] and active[i]]
            if live:
                # SHADOW first: resolves the depth ambiguity entirely
                ax.scatter(pos[live, 0], pos[live, 1], np.zeros(len(live)), s=22,
                           c="#000000", alpha=0.40, edgecolors="none", zorder=3)
                ax.plot([], [])
                for i in live:                      # drop-line to the shadow
                    ax.plot([pos[i, 0], pos[i, 0]], [pos[i, 1], pos[i, 1]],
                            [0, pos[i, 2]], color=col, lw=0.5, alpha=0.16, zorder=3)
                ax.scatter(pos[live, 0], pos[live, 1], pos[live, 2], s=58, c=col,
                           depthshade=False, edgecolors="#0b0e14", linewidths=0.6,
                           zorder=6)
            if dead:
                ax.scatter(pos[dead, 0], pos[dead, 1], np.zeros(len(dead)), s=26,
                           c=col, alpha=0.18, marker="x", zorder=3)

        # weapon fire: a tracer from shooter to target. Without it drones appear
        # to explode spontaneously -- the gun reaches 35 m, so the killer is
        # typically ~20 m away and never touches its target.
        for a0, b0, i in f.get("shots", []):
            col = RED if i < n else BLUE
            ax.plot([a0[0], b0[0]], [a0[1], b0[1]], [a0[2], b0[2]],
                    color=col, lw=1.5, alpha=0.55, zorder=7)
            ax.scatter([b0[0]], [b0[1]], [b0[2]], s=40, c="#ffffff",
                       alpha=0.35, marker="x", zorder=7)

        for p, c in f["events"]:
            blasts.append([p, c, 0])
        keep = []
        for p, c, age in blasts:
            col, base = BLAST[c]
            al = max(0.0, 1 - age / 5.0)
            ax.scatter([p[0]], [p[1]], [p[2]], s=base * (0.6 + 1.8 * age), c=col,
                       alpha=al * 0.75, edgecolors="none", zorder=8)
            ax.scatter([p[0]], [p[1]], [p[2]], s=base * 0.55, c="#ffffff",
                       alpha=al * 0.85, marker="*", zorder=9)
            ax.scatter([p[0]], [p[1]], [0], s=base * 0.5, c=col, alpha=al * 0.30,
                       edgecolors="none", zorder=3)
            if age < 5:
                keep.append([p, c, age + 1])
        blasts = keep

        ra, ba = int((alive[:n] & active[:n]).sum()), int((alive[n:] & active[n:]).sum())
        na, nb = int(active[:n].sum()), int(active[n:].sum())
        res = ""
        if f["done"]:
            res = {0: "   >>> ATTACKER WINS", 1: "   >>> DEFENDER HOLDS"}.get(
                f["winner"], "   >>> DRAW")
        fig.text(0.035, 0.955, title, color="#e6e9ef", fontsize=13,
                 fontfamily="monospace", fontweight="bold")
        fig.text(0.035, 0.905,
                 f"t={f['t']*0.1:5.1f}s     attackers {ra:2d}/{na}     "
                 f"defenders {ba:2d}/{nb}     base {f['base'][1]:4.0f}/{a.base_health:.0f}{res}",
                 color="#9aa5b8", fontsize=10.5, fontfamily="monospace")
        fig.text(0.035, 0.028, SUBTITLE,
                 color="#3d4757", fontsize=8.5, fontfamily="monospace")

        fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    imgs += [imgs[-1]] * int(fps * 1.5)
    imageio.mimsave(out, imgs, fps=fps, loop=0)
    return out, len(imgs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="/workspace/DroneSwarmRL/runs/main/phase0_passed.pt")
    ap.add_argument("--role", default="attack", choices=("attack", "defend"))
    ap.add_argument("--opponent", default="perimeter")
    ap.add_argument("--out", default="/workspace/DroneSwarmRL/runs/match.gif")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--ticks", type=int, default=420)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--scripted-self", default=None,
                    help="render a SCRIPTED agent in --role instead of --policy, "
                         "e.g. obs_interceptor")
    ap.add_argument("--device", default="cuda",
                    help="cpu keeps a one-arena render off a busy GPU")
    ap.add_argument("--cams", type=int, default=None,
                    help="cameras per drone")
    ap.add_argument("--layout", default=None,
                    help="front_down (default) or azimuth (tile the horizon)")
    ap.add_argument("--fov", type=float, default=None, help="horizontal FOV, deg")
    ap.add_argument("--cones", action="store_true",
                    help="shade each camera's real footprint")
    ap.add_argument("--guard", action="store_true",
                    help="bodyguard scenario: 3 attackers, 2 defenders, base "
                         "health 240, base walking at 1.4 m/s")
    args = ap.parse_args()

    cfg = SimConfig()
    if args.guard:
        import dataclasses as _dc
        from swarmstar.sim.config import SwarmParams, ArenaParams
        cfg.swarm = _dc.replace(cfg.swarm, n_active_red=3, n_active_blue=2)
        cfg.arena = _dc.replace(cfg.arena, base_speed=1.4, base_health=240.0)
    if args.cams or args.layout or args.fov:
        import dataclasses as _dc2, math as _m
        from swarmstar.sim.config import SensorParams
        k = {}
        if args.cams:   k["n_cameras"] = args.cams
        if args.layout: k["layout"] = args.layout
        if args.fov:
            k["fov_h"] = _m.radians(args.fov)
            k["fov_v"] = _m.radians(min(args.fov, 90.0))
        cfg.sensor = _dc2.replace(cfg.sensor, **k)
    env = SwarmEnv(cfg, 1, args.device, seed=args.seed)
    env.gen.manual_seed(args.seed)
    env.reset()
    env.mode.fill_(MODE_ATTACK_DEFEND)
    env.base_health.fill_(cfg.arena.base_health)
    env.base_health[:, 0] = 0.0
    env.observe()
    env_bx = env.base_pos[0, :, 0].cpu().numpy()
    env_by = env.base_pos[0, :, 1].cpu().numpy()

    team = 0 if args.role == "attack" else 1
    if args.scripted_self:
        agent = B.make(args.scripted_self, env, team)
    else:
        pol = SwarmPolicy().to(args.device)
        sd = torch.load(args.policy, map_location=args.device, weights_only=False)
        pol.load_state_dict(sd["policy"] if isinstance(sd, dict) and "policy" in sd
                            else sd)
        pol.eval()
        agent = PolicyAgent(env, team, pol, greedy=True, seed=1)
    scripted = B.make(args.opponent, env, 1 - team)
    red, blue = (agent, scripted) if team == 0 else (scripted, agent)
    me = args.scripted_self or "policy"
    title = args.title or (
        f"SwarmStar  |  {me} ATTACKS  vs  {args.opponent}" if team == 0
        else f"SwarmStar  |  {me} DEFENDS  vs  {args.opponent}")

    if args.subtitle:
        globals()["SUBTITLE"] = args.subtitle
    frames = rollout(env, red, blue, args.ticks)
    out, nf = draw(frames, args.out, cfg, env.n, title, every=args.every,
                   show_cones=args.cones)
    print(f"wrote {out}  ({nf} frames, {os.path.getsize(out)/1e6:.1f} MB, "
          f"match ended t={frames[-1]['t']*0.1:.1f}s winner={frames[-1]['winner']})")
