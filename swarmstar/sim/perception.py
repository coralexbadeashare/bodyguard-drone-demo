"""Partial observability: camera cones and the Bluetooth-style mesh.

This module is what separates a swarm from 32 independent drones. Nothing here
gives a drone free information:

  * it sees an enemy only inside a real camera frustum, within range, and only
    if the shot is not occluded by the ground plane
  * measurements are noisy, with error growing linearly in range
  * teammates relay contacts over a range-limited, lossy, multi-hop radio, and
    every hop costs a tick of staleness
  * each packet carries at most `n_shared_detections` contacts, so the mesh has
    a hard bandwidth ceiling

Belief fusion is a min-plus (tropical) shortest-path over the radio graph: the
age at which drone i knows about enemy e is the cheapest path from i to anyone
who can actually see e.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor

from .config import CommsParams, SensorParams

INF = float("inf")


def camera_dirs(n_cameras: int, device, layout: str = "front_down") -> Tensor:
    """Body-frame boresight of each camera.

    `front_down` (default, unchanged): front, down, rear, left, right. The second
    camera points at the GROUND, which is right for optical flow and useless for
    spotting an incoming drone -- with n_cameras=2 half the sensing is wasted for
    a counter-UAS sentry.

    `azimuth`: n cameras spread evenly around the horizon, so a swarm can tile
    360 deg between its members and each can then afford a narrow, long-range
    lens. Detection range scales as 1/FOV, so this is the whole argument for
    having several drones rather than one.
    """
    if layout == "azimuth":
        k = torch.arange(n_cameras, device=device, dtype=torch.float32)
        a = k * (2 * math.pi / max(n_cameras, 1))
        return torch.stack([torch.cos(a), torch.sin(a), torch.zeros_like(a)], -1)
    presets = [
        (1.0, 0.0, 0.0),    # front
        (0.0, 0.0, -1.0),   # down
        (-1.0, 0.0, 0.0),   # rear
        (0.0, 1.0, 0.0),    # left
        (0.0, -1.0, 0.0),   # right
    ]
    return torch.tensor(presets[:n_cameras], device=device)


def camera_detections(pos: Tensor, quat: Tensor, alive: Tensor, sp: SensorParams,
                      gen: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """Who can see whom, and with what measurement error.

    Returns
    -------
    seen      : [B,N,N] bool   -- seen[b,i,j] = drone i currently observes drone j
    meas_pos  : [B,N,N,3]      -- i's noisy estimate of j's world position
    """
    from .quadrotor import quat_to_rotmat

    B, N, _ = pos.shape
    rel = pos.unsqueeze(1) - pos.unsqueeze(2)              # [B,N,N,3] rel[b,i,j]=p_j-p_i
    dist = rel.norm(dim=-1)                                # [B,N,N]

    # into the observer's body frame
    R = quat_to_rotmat(quat)                               # [B,N,3,3] body->world
    rel_body = torch.einsum("bnji,bnkj->bnki", R, rel)     # world->body via R^T

    d = rel_body / dist.unsqueeze(-1).clamp_min(1e-6)
    cams = camera_dirs(sp.n_cameras, pos.device,
                       getattr(sp, "layout", "front_down"))     # [C,3]

    # a rectangular frustum: azimuth about body z, elevation from the cam axis
    in_any_cone = torch.zeros(B, N, N, dtype=torch.bool, device=pos.device)
    for c in range(sp.n_cameras):
        axis = cams[c]
        # rotate so the camera axis becomes +x, then test az/el in that frame
        if abs(axis[2].item()) > 0.9:                      # downward/upward camera
            fwd = torch.tensor([0.0, 0.0, axis[2].item()], device=pos.device)
            az = torch.atan2(d[..., 0], d[..., 2] * axis[2].sign())
            el = torch.asin(d[..., 1].clamp(-1, 1))
        else:
            fwd = axis
            proj = d[..., 0] * fwd[0] + d[..., 1] * fwd[1]
            az = torch.atan2(d[..., 0] * fwd[1] - d[..., 1] * fwd[0], proj)
            el = torch.asin(d[..., 2].clamp(-1, 1))
        in_any_cone |= (az.abs() < sp.fov_h * 0.5) & (el.abs() < sp.fov_v * 0.5)

    alive_i = alive.unsqueeze(2)                           # observer must be alive
    alive_j = alive.unsqueeze(1)                           # target must be alive
    not_self = ~torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)

    seen = in_any_cone & (dist < sp.max_range) & alive_i & alive_j & not_self

    # stochastic miss even when nominally in view (motion blur, target aspect, ...)
    drop = torch.rand(seen.shape, device=pos.device, generator=gen) < sp.dropout_prob
    seen = seen & ~drop

    # measurement error grows with range
    sigma = (dist * sp.pos_noise_per_m).unsqueeze(-1)
    noise = torch.randn(rel.shape, device=pos.device, generator=gen) * sigma
    meas_pos = pos.unsqueeze(1) + noise                    # i's estimate of j
    return seen, meas_pos


def mesh_hops(pos: Tensor, alive: Tensor, team: Tensor, cp: CommsParams,
              gen: torch.Generator | None = None) -> Tensor:
    """Hop distance over the radio mesh. [B,N,N] float, INF where unreachable.

    Same-team only, range limited, lossy links, at most `max_hops` relays.
    Computed by min-plus matrix powers -- Bellman-Ford on the radio graph.
    """
    B, N, _ = pos.shape
    dist = torch.cdist(pos, pos)                                   # [B,N,N]
    same_team = team.unsqueeze(1) == team.unsqueeze(2)             # [B,N,N]
    alive_pair = alive.unsqueeze(1) & alive.unsqueeze(2)

    link = (dist < cp.radio_range) & same_team & alive_pair
    lost = torch.rand(link.shape, device=pos.device, generator=gen) < cp.packet_loss
    link = link & ~lost

    eye = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)
    step = torch.where(link | eye, torch.ones_like(dist), torch.full_like(dist, INF))
    step = torch.where(eye.expand_as(step), torch.zeros_like(step), step)

    hop = step.clone()
    for _ in range(cp.max_hops - 1):                               # min-plus power
        hop = torch.minimum(hop, (hop.unsqueeze(3) + step.unsqueeze(1)).amin(dim=2))
    return hop.clamp_max(1e9)


def fuse_beliefs(seen: Tensor, meas_pos: Tensor, hop: Tensor, dist: Tensor,
                 cp: CommsParams) -> tuple[Tensor, Tensor]:
    """Merge own sightings with relayed contacts under a per-packet bandwidth cap.

    Returns
    -------
    age       : [B,N,N] float -- ticks since the freshest info i holds about j
                                 (INF = i knows nothing about j)
    belief_pos: [B,N,N,3]     -- position i believes j is at
    """
    B, N, _ = seen.shape

    # own sightings are age 0
    direct = torch.where(seen, torch.zeros_like(dist), torch.full_like(dist, INF))

    # bandwidth cap: a drone relays only its k nearest contacts
    k = min(cp.n_shared_detections, N)
    rank_key = torch.where(seen, dist, torch.full_like(dist, INF))
    idx = rank_key.topk(k, dim=-1, largest=False).indices              # [B,N,k]
    shareable = torch.zeros_like(seen)
    shareable.scatter_(-1, idx, True)
    shareable &= seen
    shared = torch.where(shareable, torch.zeros_like(dist), torch.full_like(dist, INF))

    # relayed[i,e] = min_j ( hop[i,j] + shared[j,e] )   -- tropical matmul.
    # The [B,N,N,N] intermediate is the single largest allocation in the sim, so it
    # is built once and both the value and the argmin are taken from it.
    via = hop.unsqueeze(3) + shared.unsqueeze(1)                       # [B,N,N,N]
    relayed, best_j = via.min(dim=2)                                   # [B,N,N] each
    del via
    age = torch.minimum(direct, relayed)

    # position comes from whichever source was freshest: own sighting if it wins,
    # otherwise the measurement taken by the relay that provided the contact
    relay_pos = torch.gather(meas_pos, 1, best_j.unsqueeze(-1).expand(B, N, N, 3))
    belief_pos = torch.where((direct <= relayed).unsqueeze(-1), meas_pos, relay_pos)
    return age, belief_pos
