"""Physical and simulation constants for SwarmStar.

Numbers describe a ~250 g class racing/FPV quadrotor (5" props), which is the
platform a hobby swarm would actually be built on. Values are drawn from the
standard quadrotor literature parameterisation (Mellinger & Kumar style).
"""
from dataclasses import dataclass, field
import math


@dataclass
class DroneParams:
    # --- airframe ---
    mass: float = 0.25             # kg
    arm_length: float = 0.09       # m, motor to centre
    Ixx: float = 1.4e-3            # kg m^2
    Iyy: float = 1.4e-3
    Izz: float = 2.6e-3            # yaw inertia is larger for a planar frame

    # --- rotors ---
    k_thrust: float = 2.9e-8       # N per (rad/s)^2  -> TWR 3.0, hover at ~57% throttle
    k_torque: float = 4.6e-10      # N m per (rad/s)^2  (k_m/k_f = 0.016 m, typical)
    w_min: float = 200.0           # rad/s, idle
    w_max: float = 8000.0          # rad/s, ~2.5:1 thrust-to-weight at max
    motor_tau: float = 0.025       # s, first-order motor spin-up lag

    # --- aerodynamics ---
    drag_coeff: float = 3.0e-3     # quadratic body drag -> terminal velocity ~29 m/s
    gravity: float = 9.81

    # --- integration rates ---
    dt_physics: float = 1.0 / 200.0    # 200 Hz rigid-body integration
    ctrl_decim: int = 2                # controller at 100 Hz
    policy_decim: int = 20             # policy at 10 Hz  (20 * dt_physics = 0.1 s)

    # --- cascaded controller gains ---
    kp_vel: float = 4.0            # velocity -> desired acceleration
    kp_att: float = 12.0           # attitude error -> desired body rate
    kp_rate: float = 12.0          # body rate error -> angular acceleration
    max_body_rate: float = 10.0     # rad/s, ~570 deg/s (racing-quad class)
    max_tilt: float = math.radians(35.0)   # safety clamp on commanded tilt
    max_speed: float = 18.0        # m/s, achievable top speed

    def thrust_to_weight(self) -> float:
        return 4.0 * self.k_thrust * self.w_max ** 2 / (self.mass * self.gravity)

    def hover_omega(self) -> float:
        """Rotor speed at which 4 rotors exactly cancel gravity."""
        return math.sqrt(self.mass * self.gravity / (4.0 * self.k_thrust))


@dataclass
class ArenaParams:
    size_x: float = 200.0
    size_y: float = 200.0
    size_z: float = 60.0
    floor_z: float = 0.5           # drones may not go below this
    base_radius: float = 8.0
    # 900 HP = ~8 full-speed strikes of 16 drones. Measured 2026-08-31: at 1200
    # the best scripted attacker scored 0.03 vs perimeter and 0.01 vs greedy --
    # attack was impossible, so only defence was learnable. At 900 the attacker
    # reaches 0.67 vs perimeter but still only 0.18 vs greedy, leaving real
    # headroom in BOTH roles.
    base_health: float = 900.0
    # The base WALKS. In the real scenario the base is a person, so a defender
    # cannot park on a fixed point -- it must stay with a moving charge.
    base_speed: float = 0.0            # m/s; 1.4 = walking pace


@dataclass
class SwarmParams:
    n_per_team: int = 16               # tensor width; both teams share it
    # ACTIVE counts. Drones beyond these spawn already destroyed, so the tensor
    # shapes stay fixed (and the exported model unchanged) while the GAME becomes
    # asymmetric. The real target is 1-2 defenders guarding one person against a
    # handful of incoming drones -- not a 16v16 battle.
    n_active_red: int = 16             # attackers
    n_active_blue: int = 16            # defenders
    max_visible: int = 16          # hard cap: browser budget + physical realism
    drone_health: float = 100.0
    # No gun. A 250 g quadrotor does not carry one, and real counter-drone
    # interceptors kill by flying INTO the target: both aircraft are destroyed.
    # Combat is therefore purely kinetic -- see COLLISION / KAMIKAZE in env.py.
    intercept_range: float = 35.0  # distance at which a closing intercept is cued
                                   # to the policy (an observation feature only)
    collision_radius: float = 0.6      # airframe-to-airframe, teammates
    # A real interceptor does not have to strike the target dead-on: it carries a
    # net or a proximity-fused charge, so the lethal envelope is metres, not
    # centimetres. With a 0.6 m contact requirement, catching an 18 m/s drone is
    # effectively impossible and mass attack wins every time.
    intercept_radius: float = 2.0   # measured 2026-09-10: at 0.6 m defence is
                                    # impossible (attacker 1.00); at 2.0 a naive
                                    # rush beats a passive ring (0.68) but loses
                                    # to active interception (0.12).


@dataclass
class SensorParams:
    n_cameras: int = 2
    fov_h: float = math.radians(90.0)
    fov_v: float = math.radians(60.0)
    max_range: float = 70.0
    layout: str = "front_down"         # or "azimuth": tile the horizon
    pos_noise_per_m: float = 0.012     # std of position error per metre of range
    dropout_prob: float = 0.05         # per-tick miss even when in view


@dataclass
class CommsParams:
    radio_range: float = 80.0
    max_hops: int = 3
    packet_loss: float = 0.08
    n_shared_detections: int = 2       # detections relayed per packet
    codebook_size: int = 64            # learned latent vocabulary
    n_latent_tokens: int = 8           # 8 tokens x 6 bits = 6 bytes/packet


@dataclass
class SimConfig:
    drone: DroneParams = field(default_factory=DroneParams)
    arena: ArenaParams = field(default_factory=ArenaParams)
    swarm: SwarmParams = field(default_factory=SwarmParams)
    sensor: SensorParams = field(default_factory=SensorParams)
    comms: CommsParams = field(default_factory=CommsParams)
