# Bodyguard drone — interception demo

A drone hovering over a walking person detects incoming drones and intercepts
them. This app runs the real `swarmstar` physics simulator live (3 attackers
vs 2 defenders, base health 240, base walking at 1.4 m/s, kinetic-only
interception) and renders the result.

Two scripted defences are available, not a trained neural policy:

- **`interceptor`** — full-observability closed-form pursuit law (DEF 0.98
  vs rush). A theoretical ceiling, not reachable by any real/exported policy.
- **`obs_interceptor`** — the same guidance law restricted to the drone's own
  sensor observation (DEF 0.52-0.61 vs rush). The realistic, deployable bar.

See `GUARD_REFERENCE.md` and `BODYGUARD_FINDINGS.md` for how these numbers
were measured (`scripts/guard_reference.py`, 64 arenas, seed 12345).

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
