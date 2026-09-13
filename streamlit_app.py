"""Interactive demo: the bodyguard-drone interception ceiling.

Runs the REAL swarmstar simulator (the same physics/env used for training)
with two scripted defences against a chosen attacker, and renders what
happened. No neural policy is involved here -- both defences below are
closed-form pursuit-geometry controllers. See docs/GUARD_REFERENCE.md and
docs/BODYGUARD_FINDINGS.md for how these numbers were established.

Local launch:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deployed on Streamlit Community Cloud by pointing it at this repo's
streamlit_app.py.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import tempfile

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import streamlit as st

for p in ("/workspace/DroneSwarmRL", os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

from swarmstar.sim.config import SimConfig, SwarmParams, ArenaParams, SensorParams
from swarmstar.sim.env import SwarmEnv, MODE_ATTACK_DEFEND
from swarmstar.train import baselines as B
from scripts.make_gif import rollout, draw  # reuse the exact render pipeline

st.set_page_config(page_title="Bodyguard Drone — Interception Demo", layout="wide")

DEFENDERS = {
    "interceptor": (
        "Full-observability ceiling (DEF 0.98 vs rush)",
        "Reads the simulator's ground truth directly -- engages every attacker "
        "whether or not it has actually been detected. This is a theoretical "
        "upper bound used to size the problem, **not a deployable policy and "
        "not the trained SwarmStar model**. No exported policy can reach it: "
        "a real drone only knows what its own camera + BLE mesh has seen."
    ),
    "obs_interceptor": (
        "Sensor-limited, realizable bar (DEF 0.52-0.61 vs rush)",
        "The identical pursuit-geometry law, but run only on this drone's own "
        "observation (camera cone + relayed contacts). This is the honest bar: "
        "what a real, exportable defence could actually achieve."
    ),
}
ATTACKERS = ["rush", "saturation", "greedy"]

st.title("Bodyguard drone: interception demo")
st.caption(
    "A drone hovering over a walking person detects incoming drones and "
    "intercepts them. Simulator: 3 attackers vs 2 defenders, base health 240, "
    "base walking at 1.4 m/s, kinetic-only interception (no gun)."
)

with st.expander("Why two defenders are shown, and what '98%' actually means", expanded=True):
    st.markdown(
        "- **`interceptor` (0.98 DEF)** is a closed-form pursuit-geometry "
        "controller with *full observability* -- it is the ceiling used to "
        "bound the problem, not a result of the RL/imitation training effort.\n"
        "- **`obs_interceptor` (0.52-0.61 DEF)** runs the same guidance law "
        "restricted to the drone's real sensor observation. This is the "
        "realizable bar a deployable policy is judged against.\n"
        "- For comparison: the best *learned* RL policy peaks at **0.34 DEF**, "
        "and imitation learning from either scripted teacher collapses to "
        "**0.00 DEF** (a documented open problem, see "
        "`docs/BODYGUARD_FINDINGS.md` §4).\n\n"
        "Source: `docs/GUARD_REFERENCE.md`, `docs/BODYGUARD_FINDINGS.md` "
        "(2026-09-11, `scripts/guard_reference.py`, 64 arenas, seed 12345)."
    )

col_l, col_r = st.columns([1, 2])

with col_l:
    st.subheader("Scenario")
    defender = st.selectbox(
        "Defender", list(DEFENDERS.keys()),
        format_func=lambda k: DEFENDERS[k][0])
    st.caption(DEFENDERS[defender][1])

    attacker = st.selectbox("Attacker", ATTACKERS, index=0)

    seed = st.number_input("Seed", min_value=0, max_value=999999, value=3, step=1)
    ticks = st.slider("Episode length (ticks, 10/s)", 60, 420, 260, step=20)

    range_m, fov_deg = 70.0, 90.0
    if defender == "obs_interceptor":
        st.markdown("**Sensor spec** (`scripts/sensor_sweep.py` knee is ~120 m)")
        range_m = st.select_slider("Detection range (m)", [70, 120, 180, 250, 350], value=70)
        fov_deg = st.select_slider("Camera FOV (deg)", [90, 150, 360], value=90)

    run = st.button("Run scenario", type="primary")

with col_r:
    if run:
        with st.spinner("Running the real physics simulator and rendering..."):
            cfg = SimConfig()
            cfg.swarm = dataclasses.replace(SwarmParams(), n_active_red=3, n_active_blue=2)
            cfg.arena = dataclasses.replace(ArenaParams(), base_speed=1.4, base_health=240.0)
            if defender == "obs_interceptor":
                cfg.sensor = dataclasses.replace(
                    SensorParams(), max_range=float(range_m),
                    fov_h=math.radians(float(fov_deg)),
                    fov_v=math.radians(min(float(fov_deg), 90.0)))

            device = "cpu"  # scripted-vs-scripted, single arena: no GPU needed
            env = SwarmEnv(cfg, 1, device, seed=int(seed))
            env.gen.manual_seed(int(seed))
            env.reset()
            env.mode.fill_(MODE_ATTACK_DEFEND)
            env.base_health.fill_(cfg.arena.base_health)
            env.base_health[:, 0] = 0.0
            env.observe()

            import scripts.make_gif as mg
            mg.env_bx = env.base_pos[0, :, 0].cpu().numpy()
            mg.env_by = env.base_pos[0, :, 1].cpu().numpy()

            red = B.make(attacker, env, 0)
            blue = B.make(defender, env, 1)
            frames = rollout(env, red, blue, int(ticks))

            title = f"{DEFENDERS[defender][0].split(' (')[0]}  |  defends vs {attacker}"
            tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
            tmp.close()
            out, nframes = draw(frames, tmp.name, cfg, env.n, title, every=3,
                                 show_cones=(defender == "obs_interceptor"))

        last = frames[-1]
        winner = last["winner"]
        def_score = 1.0 if winner == 1 else (0.5 if winner == -1 else 0.0)
        base_hp_frac = float(last["base"][1]) / cfg.arena.base_health

        m1, m2, m3 = st.columns(3)
        m1.metric("DEF score (this run)", f"{def_score:.2f}")
        m2.metric("Base HP remaining", f"{base_hp_frac*100:.0f}%")
        m3.metric("Duration", f"{last['t']*0.1:.1f}s")
        if winner == -1 and last["t"] * 0.1 >= (int(ticks) - 1) * 0.1 and not last["done"]:
            st.caption("Episode ran to the tick limit without a decisive result.")

        st.image(out, caption=title)
        with open(out, "rb") as f:
            st.download_button("Download GIF", f, file_name="bodyguard_demo.gif")
        st.caption(
            "This is one seed, not the 64-arena average in `docs/GUARD_REFERENCE.md` "
            "-- re-run with a different seed to see the variance."
        )
    else:
        st.info("Pick a scenario on the left and press **Run scenario**.")

st.divider()
st.subheader("Reference table (64 arenas, seed 12345)")
st.caption("DEF score = blue_win + 0.5*draw, from `docs/GUARD_REFERENCE.md`.")
st.table({
    "defence": ["do_nothing", "perimeter", "greedy", "obs_interceptor", "interceptor"],
    "vs rush": [0.00, 0.41, 0.52, 0.52, 0.98],
    "vs saturation": [0.14, 0.20, 0.75, 0.27, 1.00],
    "vs greedy": [0.95, 0.89, 1.00, 0.94, 1.00],
})
