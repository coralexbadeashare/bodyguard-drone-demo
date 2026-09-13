"""Export one drone's policy to ONNX -- the same file runs in the browser and,
later, on the aircraft. The centralized critic is deliberately not exported.

The graph is fully static: fixed entity count, fixed hidden size, no RNG (Gumbel
noise is an input). Feed zeros as noise for deterministic flight.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from swarmstar.model.networks import SwarmPolicy, N_MAIN_HEADS


class ExportWrapper(torch.nn.Module):
    """Flattens the dict outputs into fixed tensors, which ONNX requires."""

    def __init__(self, pol: SwarmPolicy):
        super().__init__()
        self.pol = pol

    def forward(self, scalars, entities, entity_mask, action_mask, hx, cx, noise):
        lg, ac, h, c = self.pol(scalars, entities, entity_mask.bool(),
                                action_mask.bool(), hx, cx, noise)
        return (ac["action_type"].int(), ac["delay"].int(), ac["target"].int(),
                ac["heading"].int(), ac["pitch"].int(), ac["speed"].int(),
                ac["comms"].int(), lg["action_type"], h, c)


def sample_inputs(pol: SwarmPolicy, batch: int = 1):
    n_noise = N_MAIN_HEADS + pol.n_comm_tokens
    hx, cx = pol.initial_state(batch, "cpu")
    return (torch.randn(batch, 28), torch.randn(batch, pol.K, 13),
            torch.ones(batch, pol.K, dtype=torch.bool),
            torch.ones(batch, 7, dtype=torch.bool),
            hx, cx, torch.zeros(batch, n_noise, pol.max_logits))


def export(pol: SwarmPolicy, path: str, opset: int = 17):
    pol = pol.eval().cpu()
    wrap = ExportWrapper(pol).eval()
    args = sample_inputs(pol)
    torch.onnx.export(
        wrap, args, path, opset_version=opset,
        input_names=["scalars", "entities", "entity_mask", "action_mask",
                     "hx", "cx", "noise"],
        output_names=["action_type", "delay", "target", "heading", "pitch",
                      "speed", "comms", "action_logits", "hx_out", "cx_out"],
        dynamo=False,
    )
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/DroneSwarmRL/browser/policy.onnx")
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    pol = SwarmPolicy()
    if a.ckpt:
        pol.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    export(pol, a.out)
    mb = Path(a.out).stat().st_size / 2**20
    print(f"exported {a.out}  ({mb:.2f} MB, "
          f"{sum(p.numel() for p in pol.parameters()):,} params)")
