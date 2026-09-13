"""GPU-batched 6-DOF quadrotor digital twin.

Every tensor carries a leading [B, N] pair: B parallel arenas, N drones each.
Nothing here is drone-specific beyond `DroneParams`, so the same code integrates
131k airframes at once as easily as one.

Model
-----
  * rigid body: position, velocity, orientation (quaternion), body angular rate
  * 4 rotors in an X layout, each producing thrust k_f*w^2 along body +z and a
    reaction torque k_m*w^2 about body z with alternating sign
  * first-order motor lag (this is what stops the response feeling like a videogame)
  * quadratic body drag
  * semi-implicit Euler at 200 Hz

The policy never touches motors. It emits velocity setpoints, which the onboard
cascaded controller tracks -- the same interface a real offboard/flight-controller
split uses, so the trained policy transfers to hardware unchanged.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .config import DroneParams


# --------------------------------------------------------------------------
# quaternion helpers  (convention: q = [w, x, y, z], body -> world)
# --------------------------------------------------------------------------
def quat_normalize(q: Tensor) -> Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_to_rotmat(q: Tensor) -> Tensor:
    """[..., 4] -> [..., 3, 3] rotation matrix mapping body vectors to world."""
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    r = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ], dim=-1)
    return r.reshape(*q.shape[:-1], 3, 3)


def quat_integrate(q: Tensor, omega_body: Tensor, dt: float) -> Tensor:
    """Advance orientation by body angular rate. qdot = 0.5 * q (x) [0, omega]."""
    wx, wy, wz = omega_body.unbind(-1)
    qw, qx, qy, qz = q.unbind(-1)
    dq = 0.5 * torch.stack([
        -qx * wx - qy * wy - qz * wz,
        qw * wx + qy * wz - qz * wy,
        qw * wy - qx * wz + qz * wx,
        qw * wz + qx * wy - qy * wx,
    ], dim=-1)
    return quat_normalize(q + dq * dt)


def body_to_world(q: Tensor, v_body: Tensor) -> Tensor:
    return torch.einsum("...ij,...j->...i", quat_to_rotmat(q), v_body)


def world_to_body(q: Tensor, v_world: Tensor) -> Tensor:
    return torch.einsum("...ji,...j->...i", quat_to_rotmat(q), v_world)


# --------------------------------------------------------------------------
# rotor geometry / allocation
# --------------------------------------------------------------------------
def allocation_matrix(p: DroneParams, device, dtype=torch.float32) -> Tensor:
    """Map [w0^2..w3^2] -> [T, tau_x, tau_y, tau_z].  Shape [4, 4].

    X layout, rotor i at 45 deg offsets, spin directions alternating CCW/CW so
    that yaw torque cancels in hover.
    """
    a = p.arm_length / (2.0 ** 0.5)
    # (x, y) body position of each rotor and its spin sign (+1 = CCW)
    rotors = [(+a, +a, +1.0), (+a, -a, -1.0), (-a, -a, +1.0), (-a, +a, -1.0)]
    kf, km = p.k_thrust, p.k_torque
    rows = [
        [kf, kf, kf, kf],                              # total thrust
        [kf * ry for (_, ry, _) in rotors],            # roll  = sum r_y * f
        [-kf * rx for (rx, _, _) in rotors],           # pitch = sum -r_x * f
        [-km * s for (_, _, s) in rotors],             # yaw   = -sum sigma * k_m
    ]
    return torch.tensor(rows, device=device, dtype=dtype)


class QuadrotorState:
    """Batched airframe state, [B, N, ...]."""

    __slots__ = ("pos", "vel", "quat", "omega", "rotor_w")

    def __init__(self, pos, vel, quat, omega, rotor_w):
        self.pos = pos          # [B,N,3] world
        self.vel = vel          # [B,N,3] world
        self.quat = quat        # [B,N,4] body->world
        self.omega = omega      # [B,N,3] body angular rate
        self.rotor_w = rotor_w  # [B,N,4] rad/s

    @classmethod
    def hover(cls, B: int, N: int, p: DroneParams, device, pos: Tensor | None = None):
        z = torch.zeros(B, N, 3, device=device)
        q = torch.zeros(B, N, 4, device=device)
        q[..., 0] = 1.0
        w = torch.full((B, N, 4), p.hover_omega(), device=device)
        return cls(torch.zeros(B, N, 3, device=device) if pos is None else pos.clone(),
                   z.clone(), q, z.clone(), w)

    def detach(self):
        return QuadrotorState(*(t.detach() for t in
                                (self.pos, self.vel, self.quat, self.omega, self.rotor_w)))

    def index(self, b, n):
        return {k: getattr(self, k)[b, n] for k in self.__slots__}


# --------------------------------------------------------------------------
# dynamics
# --------------------------------------------------------------------------
def physics_step(s: QuadrotorState, rotor_cmd: Tensor, p: DroneParams,
                 alloc: Tensor, dt: float) -> QuadrotorState:
    """One semi-implicit Euler step. `rotor_cmd` is commanded rotor speed [B,N,4]."""
    # first-order motor lag: real rotors cannot change speed instantly
    alpha = dt / (p.motor_tau + dt)
    rotor_w = s.rotor_w + alpha * (rotor_cmd.clamp(p.w_min, p.w_max) - s.rotor_w)

    w_sq = rotor_w ** 2                                     # [B,N,4]
    wrench = torch.einsum("ij,bnj->bni", alloc, w_sq)       # [B,N,4] = T,tx,ty,tz
    thrust, torque = wrench[..., :1], wrench[..., 1:]

    R = quat_to_rotmat(s.quat)
    thrust_world = R[..., :, 2] * thrust                    # body +z scaled by T

    speed = s.vel.norm(dim=-1, keepdim=True)
    drag = -p.drag_coeff * speed * s.vel
    gravity = torch.zeros_like(s.vel)
    gravity[..., 2] = -p.gravity * p.mass

    acc = (thrust_world + drag + gravity) / p.mass
    vel = s.vel + acc * dt
    pos = s.pos + vel * dt

    # Euler's rigid-body equation: J w' = tau - w x (J w)
    J = torch.tensor([p.Ixx, p.Iyy, p.Izz], device=s.pos.device, dtype=s.pos.dtype)
    Jw = s.omega * J
    omega_dot = (torque - torch.cross(s.omega, Jw, dim=-1)) / J
    omega = s.omega + omega_dot * dt
    quat = quat_integrate(s.quat, omega, dt)

    return QuadrotorState(pos, vel, quat, omega, rotor_w)


def cascaded_controller(s: QuadrotorState, vel_sp: Tensor, yaw_rate_sp: Tensor,
                        p: DroneParams, alloc_inv: Tensor) -> Tensor:
    """Onboard flight controller: velocity setpoint -> rotor speed commands.

    velocity -> desired acceleration -> desired tilt + thrust -> body rate -> torque
    This is the standard cascade running on a real flight controller (PX4/Betaflight
    angle mode), so the policy's action interface matches hardware exactly.
    """
    R = quat_to_rotmat(s.quat)
    b3 = R[..., :, 2]                                       # current body z, world

    # --- velocity loop -> desired acceleration (incl. gravity compensation)
    acc_des = p.kp_vel * (vel_sp - s.vel)
    acc_des = acc_des + torch.tensor([0.0, 0.0, p.gravity], device=s.pos.device)

    # clamp commanded tilt so the controller never demands an impossible attitude
    horiz = acc_des[..., :2]
    vert = acc_des[..., 2:].clamp_min(0.5 * p.gravity)
    max_horiz = vert * torch.tan(torch.as_tensor(p.max_tilt, device=s.pos.device))
    hnorm = horiz.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    horiz = horiz * (hnorm.clamp(max=max_horiz.squeeze(-1).unsqueeze(-1)) / hnorm)
    acc_des = torch.cat([horiz, vert], dim=-1)

    b3_des = acc_des / acc_des.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # thrust along the axis we actually point at (not the one we wish we had)
    thrust = p.mass * (acc_des * b3).sum(-1, keepdim=True).clamp_min(0.0)

    # --- attitude loop: rotate b3 onto b3_des, expressed as a body-frame rate
    err_world = torch.cross(b3, b3_des, dim=-1)
    err_body = torch.einsum("...ji,...j->...i", R, err_world)
    omega_des = p.kp_att * err_body
    omega_des[..., 2] = yaw_rate_sp                          # yaw commanded directly
    # a real airframe cannot be asked for unbounded rate
    omega_des = omega_des.clamp(-p.max_body_rate, p.max_body_rate)

    # --- rate loop -> torque, with the gyroscopic term fed forward
    J = torch.tensor([p.Ixx, p.Iyy, p.Izz], device=s.pos.device, dtype=s.pos.dtype)
    rate_err = omega_des - s.omega
    torque = J * (p.kp_rate * rate_err) + torch.cross(s.omega, s.omega * J, dim=-1)

    return mix(thrust, torque, p, alloc_inv)


def mix(thrust: Tensor, torque: Tensor, p: DroneParams, alloc_inv: Tensor) -> Tensor:
    """Wrench -> rotor speeds, with thrust-prioritising desaturation.

    Four rotors cannot always deliver a requested (thrust, torque) pair. Naively
    clamping each rotor corrupts *both* components -- which makes a drone fall out
    of the sky while it tries to turn. Real flight controllers instead keep thrust
    intact and scale the torque back until the command is feasible; that is what
    the `k` factor below does.
    """
    lo, hi = p.w_min ** 2, p.w_max ** 2

    base = alloc_inv[:, 0].view(1, 1, 4) * thrust                    # [B,N,4]
    delta = torch.einsum("ij,bnj->bni", alloc_inv[:, 1:], torque)    # [B,N,4]

    base = base.clamp(lo, hi)                                        # thrust wins
    inf = torch.full_like(delta, float("inf"))
    eps = 1e-9
    k_hi = torch.where(delta > eps, (hi - base) / delta.clamp_min(eps), inf)
    k_lo = torch.where(delta < -eps, (lo - base) / delta.clamp_max(-eps), inf)
    k = torch.minimum(k_hi, k_lo).amin(dim=-1, keepdim=True).clamp(0.0, 1.0)

    w_sq = (base + k * delta).clamp(lo, hi)
    return w_sq.sqrt()
