"""Torch <-> ONNX parity and the browser latency budget.

Latency is measured single-threaded on purpose: in the browser each drone gets
its own sequential inference on one CPU core, which is also the honest stand-in
for the weak compute on the real aircraft.
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from swarmstar.model.networks import SwarmPolicy, N_MAIN_HEADS
from swarmstar.export.to_onnx import ExportWrapper, sample_inputs, export

OUT_NAMES = ["action_type", "delay", "target", "heading", "pitch", "speed",
             "comms", "action_logits", "hx_out", "cx_out"]


def make_session(path, threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def main(path="/workspace/DroneSwarmRL/browser/policy.onnx", n_drones=32):
    pol = SwarmPolicy().eval().cpu()
    export(pol, path)
    sess = make_session(path, threads=1)
    wrap = ExportWrapper(pol).eval()

    ok = True
    # ---- parity over several random states, greedy (zero noise)
    max_logit_err, mismatches, trials = 0.0, 0, 25
    for _ in range(trials):
        args = sample_inputs(pol)
        args = (args[0], args[1], torch.rand(1, pol.K) > 0.35,
                torch.ones(1, 7, dtype=torch.bool), args[4], args[5], args[6])
        with torch.no_grad():
            ref = wrap(*args)
        feed = {n: a.numpy() for n, a in zip(
            ["scalars", "entities", "entity_mask", "action_mask", "hx", "cx", "noise"],
            args)}
        got = sess.run(OUT_NAMES, feed)
        for i in range(7):
            if not np.array_equal(ref[i].numpy(), got[i]):
                mismatches += 1
        max_logit_err = max(max_logit_err,
                            float(np.abs(ref[7].numpy() - got[7]).max()),
                            float(np.abs(ref[8].numpy() - got[8]).max()))
    print(f"[{'PASS' if mismatches==0 else 'FAIL'}] action parity: "
          f"{mismatches} mismatched actions over {trials} states x 7 heads")
    print(f"[{'PASS' if max_logit_err<1e-4 else 'FAIL'}] logit/state parity: "
          f"max abs diff = {max_logit_err:.2e} (tol 1e-4)")
    ok &= mismatches == 0 and max_logit_err < 1e-4

    # ---- latency: one drone at a time, single thread
    args = sample_inputs(pol)
    feed = {n: a.numpy() for n, a in zip(
        ["scalars", "entities", "entity_mask", "action_mask", "hx", "cx", "noise"],
        args)}
    for _ in range(50):
        sess.run(OUT_NAMES, feed)
    t0 = time.perf_counter()
    iters = 500
    for _ in range(iters):
        sess.run(OUT_NAMES, feed)
    per = (time.perf_counter() - t0) / iters * 1000

    with torch.no_grad():
        for _ in range(20): wrap(*args)
        t0 = time.perf_counter()
        for _ in range(200): wrap(*args)
        torch_per = (time.perf_counter() - t0) / 200 * 1000

    tick = per * n_drones
    print(f"\n  PyTorch CPU, batch 1 : {torch_per:8.3f} ms/drone")
    print(f"  ONNX Runtime, 1 thread: {per:8.3f} ms/drone   "
          f"({torch_per/per:.1f}x faster)")
    print(f"  {n_drones} drones sequentially: {tick:.1f} ms/tick "
          f"-> {100.0/tick:.1f}x headroom at 10 Hz")
    good = tick < 100.0
    print(f"[{'PASS' if good else 'FAIL'}] browser tick budget "
          f"({n_drones} drones under 100 ms)")
    return ok and good


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
