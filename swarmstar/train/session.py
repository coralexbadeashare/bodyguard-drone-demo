"""Nightly training session control: GPU gating, an 11-hour wall clock, and
checkpoint/resume that is exact enough to stitch many nights into one run.

Design constraints this file exists to satisfy:
  * training may only run at night, in bounded sessions (default 11 h)
  * it must only start if the designated GPU is genuinely free
  * a session must be resumable, so night N+1 continues night N exactly

Everything needed to resume lives in one checkpoint: weights, optimizer state,
step counters, curriculum phase, league state, and every RNG stream. Resuming
without the RNG state silently changes the data distribution, which is the kind
of bug that only shows up as a mysteriously worse learning curve three nights in.
"""
from __future__ import annotations

import csv, json, os, signal, subprocess, threading, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import torch


# ---------------------------------------------------------------- GPU gating
@dataclass
class GPUStatus:
    index: int
    free_gb: float
    total_gb: float
    util: float
    n_foreign_procs: int

    def is_free(self, min_free_gb: float, max_util: float) -> bool:
        return (self.free_gb >= min_free_gb and self.util <= max_util
                and self.n_foreign_procs == 0)

    def __str__(self) -> str:
        return (f"GPU{self.index}: {self.free_gb:.1f}/{self.total_gb:.0f} GB free, "
                f"{self.util:.0f}% util, {self.n_foreign_procs} foreign proc(s)")


def _nvsmi(args: list[str]) -> list[str]:
    out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                         timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {out.stderr.strip()}")
    return [l for l in out.stdout.strip().splitlines() if l.strip()]


def gpu_status(index: int, own_pids: set[int] | None = None) -> GPUStatus:
    """Query one physical GPU. Requires the container to see all devices."""
    own_pids = own_pids or set()
    row = _nvsmi(["-i", str(index), "--query-gpu=memory.used,memory.total,"
                  "utilization.gpu", "--format=csv,noheader,nounits"])[0]
    used, total, util = [float(x) for x in row.split(",")]

    procs = []
    try:
        procs = _nvsmi(["-i", str(index), "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"])
    except RuntimeError:
        pass
    foreign = 0
    for p in procs:
        pid = int(p.split(",")[0])
        mem = float(p.split(",")[1])
        # the display/Isaac process holds a few hundred MB on every card; ignore it
        if pid not in own_pids and mem > 1024:
            foreign += 1
    return GPUStatus(index, (total - used) / 1024, total / 1024, util, foreign)


class GPUGuard:
    """Refuses to start unless the target GPU is free, and stays free."""

    def __init__(self, index: int = 1, min_free_gb: float = 30.0,
                 max_util: float = 25.0, settle_s: float = 60.0,
                 poll_s: float = 30.0, log: Callable[[str], None] = print):
        self.index, self.min_free_gb, self.max_util = index, min_free_gb, max_util
        self.settle_s, self.poll_s, self.log = settle_s, poll_s, log

    def check(self) -> GPUStatus:
        return gpu_status(self.index, own_pids={os.getpid()})

    def acquire(self, wait_hours: float = 0.0) -> bool:
        """Gate. Optionally wait up to `wait_hours` for the GPU to free up.

        Requires the GPU to look free twice, `settle_s` apart, so we do not start
        into the gap between two of someone else's jobs.
        """
        deadline = time.time() + wait_hours * 3600
        while True:
            st = self.check()
            if st.is_free(self.min_free_gb, self.max_util):
                self.log(f"[gpu] {st} -- looks free, confirming in {self.settle_s:.0f}s")
                time.sleep(self.settle_s)
                st2 = self.check()
                if st2.is_free(self.min_free_gb, self.max_util):
                    self.log(f"[gpu] {st2} -- ACQUIRED")
                    return True
                self.log(f"[gpu] {st2} -- became busy during confirmation, backing off")
            else:
                self.log(f"[gpu] {st} -- busy, waiting")
            if time.time() >= deadline:
                self.log(f"[gpu] giving up: GPU{self.index} not free")
                return False
            time.sleep(self.poll_s)


# ------------------------------------------------------------- checkpointing
class Checkpointer:
    """Atomic, resumable checkpoints. Never leaves a half-written file behind."""

    def __init__(self, run_dir: str | Path, keep_last: int = 3):
        self.dir = Path(run_dir) / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def _path(self, step: int) -> Path:
        return self.dir / f"ckpt_{step:012d}.pt"

    def save(self, state: dict[str, Any], step: int) -> Path:
        p = self._path(step)
        tmp = p.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, p)                      # atomic on POSIX
        (self.dir / "LATEST").write_text(p.name)
        self._prune()
        return p

    def _prune(self):
        cks = sorted(self.dir.glob("ckpt_*.pt"))
        for old in cks[:-self.keep_last]:
            old.unlink(missing_ok=True)

    def latest(self) -> Path | None:
        marker = self.dir / "LATEST"
        if marker.exists():
            p = self.dir / marker.read_text().strip()
            if p.exists():
                return p
        cks = sorted(self.dir.glob("ckpt_*.pt"))
        return cks[-1] if cks else None

    def load(self, map_location="cuda") -> dict[str, Any] | None:
        p = self.latest()
        if p is None:
            return None
        return torch.load(p, map_location=map_location, weights_only=False)


def rng_state(device: str = "cuda") -> dict[str, Any]:
    s = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def load_rng_state(s: dict[str, Any]):
    """Restore RNG. Tolerates a change in how many CUDA devices are visible.

    A checkpoint written with CUDA_VISIBLE_DEVICES unset carries one state per
    physical GPU; resuming pinned to a single device exposes only one generator,
    and set_rng_state_all would index past the end. Restore the states we can
    place and seed any extra generators from the first saved state -- the run is
    bit-exact whenever the device count is unchanged, which is the case resume
    actually promises.
    """
    torch.set_rng_state(s["cpu"].cpu() if hasattr(s["cpu"], "cpu") else s["cpu"])
    if "cuda" not in s or not torch.cuda.is_available():
        return
    saved = [t.cpu() for t in s["cuda"]]
    have = torch.cuda.device_count()
    if len(saved) != have:
        print(f"[resume] checkpoint holds {len(saved)} CUDA RNG state(s), "
              f"{have} device(s) visible -- restoring {min(len(saved), have)}")
        saved = (saved[:have] if len(saved) > have
                 else saved + [saved[0]] * (have - len(saved)))
    torch.cuda.set_rng_state_all(saved)


# ------------------------------------------------------------------- logging
class CSVLogger:
    """Appends to training_log.csv -- mandatory for convergence analysis.

    Survives resume: reopens in append mode and only writes the header once.
    """

    def __init__(self, run_dir: str | Path, name: str = "training_log.csv"):
        self.path = Path(run_dir) / name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] | None = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path) as f:
                head = f.readline().strip()
            if head:
                self._fields = head.split(",")

    def log(self, row: dict[str, Any]):
        row = {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}
        if self._fields is None:
            self._fields = list(row.keys())
            with open(self.path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fields).writeheader()
        elif not set(row).issubset(self._fields):
            # Columns can appear late -- evaluation metrics are only produced every
            # Nth step. Locking the header on the first row silently discarded them
            # for the whole run, so grow the schema and rewrite instead.
            self._grow([k for k in row if k not in self._fields])
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore")
            w.writerow({k: row.get(k, "") for k in self._fields})

    def _grow(self, new_cols: list[str]):
        with open(self.path) as f:
            old_rows = list(csv.DictReader(f))
        self._fields = self._fields + new_cols
        tmp = self.path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            w.writeheader()
            for r in old_rows:
                w.writerow({k: r.get(k, "") for k in self._fields})
        os.replace(tmp, self.path)


# ------------------------------------------------------------------- session
@dataclass
class SessionConfig:
    run_dir: str
    gpu_index: int = 1
    max_hours: float = 11.0
    min_free_gb: float = 30.0
    max_util: float = 25.0
    wait_hours: float = 0.0            # how long to wait for the GPU at startup
    checkpoint_every_min: float = 10.0   # a week-long run must lose little on a crash
    keep_last: int = 12
    yield_on_contention: bool = False  # exit if someone else takes the GPU
    contention_poll_min: float = 5.0
    # HARD deadline. The cooperative budget above can only fire if the training
    # loop is still calling should_continue(); a hung kernel or deadlock would
    # otherwise hold the GPU all night. This watchdog kills the process outright.
    hard_kill_grace_min: float = 15.0


class TrainingSession:
    """Context manager wrapping one night of training.

    Usage:
        with TrainingSession(cfg) as s:
            state = s.resume() or fresh_state()
            while s.should_continue():
                ... train one iteration, advance state["step"] ...
                s.tick(state, metrics)
    """

    def __init__(self, cfg: SessionConfig):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt = Checkpointer(self.run_dir, cfg.keep_last)
        self.logger = CSVLogger(self.run_dir)
        self.guard = GPUGuard(cfg.gpu_index, cfg.min_free_gb, cfg.max_util,
                              log=self._log)
        self._stop = False
        self._stop_reason = ""
        self._watchdog: threading.Thread | None = None
        self.t_start = 0.0
        self._last_ckpt = 0.0
        self._last_gpu_poll = 0.0
        self.state: dict[str, Any] | None = None

    # -- plumbing
    def _log(self, msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.run_dir / "session.log", "a") as f:
            f.write(line + "\n")

    def _on_signal(self, signum, frame):
        self._stop, self._stop_reason = True, f"signal {signal.Signals(signum).name}"
        self._log(f"[session] {self._stop_reason} -- will checkpoint and exit")

    def __enter__(self):
        self.t_start = time.time()
        self._last_ckpt = self.t_start
        self._last_gpu_poll = self.t_start
        (self.run_dir / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2))
        self._log(f"[session] run_dir={self.run_dir}")
        self._log(f"[session] budget {self.cfg.max_hours:.1f} h, "
                  f"target GPU {self.cfg.gpu_index}")
        if not self.guard.acquire(self.cfg.wait_hours):
            raise SystemExit(f"GPU {self.cfg.gpu_index} unavailable -- not training. "
                             f"(This is the intended behaviour, not an error.)")
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, self._on_signal)
        self._start_watchdog()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._exited = True
        if self.state is not None:
            self._log("[session] final checkpoint")
            self.save(self.state, force=True)
        h = (time.time() - self.t_start) / 3600
        self._log(f"[session] done after {h:.2f} h -- {self._stop_reason or exc_type}")
        return False

    def _start_watchdog(self):
        """Independent thread that enforces the deadline even if the loop hangs.

        Required by the project's hard GPU-time rule: the limit must be enforced
        by something that actually kills the process, not merely observed by the
        training loop. Two stages -- ask nicely, then terminate.
        """
        soft_s = self.cfg.max_hours * 3600
        hard_s = soft_s + self.cfg.hard_kill_grace_min * 60

        def watch():
            while not self._stop:
                if time.time() - self.t_start >= soft_s:
                    break
                time.sleep(5.0)
            if not self._stop:
                self._stop = True
                self._stop_reason = "watchdog: soft deadline"
                self._log("[watchdog] soft deadline reached -- asking loop to stop")
            while time.time() - self.t_start < hard_s:
                if not self._alive():
                    return
                time.sleep(5.0)
            self._log(f"[watchdog] HARD DEADLINE +{self.cfg.hard_kill_grace_min:.0f} min "
                      f"-- loop did not exit, killing process to free GPU "
                      f"{self.cfg.gpu_index}")
            os._exit(3)

        self._watchdog = threading.Thread(target=watch, name="deadline-watchdog",
                                          daemon=True)
        self._watchdog.start()
        self._log(f"[watchdog] armed: soft {self.cfg.max_hours:.1f} h, "
                  f"hard kill at +{self.cfg.hard_kill_grace_min:.0f} min")

    def _alive(self) -> bool:
        """False once the main loop has finished and released the session."""
        return not getattr(self, "_exited", False)

    # -- API
    @property
    def elapsed_hours(self) -> float:
        return (time.time() - self.t_start) / 3600

    def resume(self, map_location="cuda") -> dict[str, Any] | None:
        st = self.ckpt.load(map_location)
        if st is None:
            self._log("[resume] no checkpoint found -- starting fresh")
            return None
        if "rng" in st:
            load_rng_state(st["rng"])
        self._log(f"[resume] step {st.get('step', 0):,}  phase {st.get('phase')}  "
                  f"night {st.get('nights', 0)}  "
                  f"cumulative {st.get('cumulative_hours', 0):.1f} h")
        return st

    def should_continue(self) -> bool:
        if self._stop:
            return False
        if self.elapsed_hours >= self.cfg.max_hours:
            self._stop_reason = f"budget reached ({self.cfg.max_hours:.1f} h)"
            self._log(f"[session] {self._stop_reason}")
            return False
        now = time.time()
        if now - self._last_gpu_poll > self.cfg.contention_poll_min * 60:
            self._last_gpu_poll = now
            try:
                st = self.guard.check()
                if st.n_foreign_procs > 0:
                    self._log(f"[gpu] contention detected -- {st}")
                    if self.state is not None:
                        self.save(self.state, force=True)
                    if self.cfg.yield_on_contention:
                        self._stop_reason = "yielded to another job"
                        return False
            except Exception as e:
                self._log(f"[gpu] poll failed ({e}) -- continuing")
        return True

    def save(self, state: dict[str, Any], force: bool = False) -> bool:
        self.ckpt.dir.mkdir(parents=True, exist_ok=True)   # survive external rm
        now = time.time()
        if not force and now - self._last_ckpt < self.cfg.checkpoint_every_min * 60:
            return False
        state = dict(state)
        state["rng"] = rng_state()
        state["cumulative_hours"] = state.get("cumulative_hours", 0.0) + \
            (now - self._last_ckpt) / 3600
        self._last_ckpt = now
        p = self.ckpt.save(state, int(state.get("step", 0)))
        self._log(f"[ckpt] step {state.get('step', 0):,} -> {p.name}")
        return True

    def tick(self, state: dict[str, Any], metrics: dict[str, Any] | None = None):
        """Call once per training iteration."""
        self.state = state
        if metrics:
            metrics = dict(metrics)
            metrics.setdefault("step", state.get("step", 0))
            metrics.setdefault("phase", state.get("phase", ""))
            metrics.setdefault("wall_h", round(self.elapsed_hours, 4))
            self.logger.log(metrics)
        self.save(state)

    def steps_for_budget(self, measured_rate_per_s: float,
                         reserve_frac: float = 0.10) -> int:
        """How many steps fit in this session, from a MEASURED rate.

        The project rule requires sizing step counts from a measured per-step time
        rather than a guess. `reserve_frac` leaves margin for contention and for
        the final checkpoint.
        """
        usable = self.cfg.max_hours * 3600 * (1.0 - reserve_frac)
        return max(1, int(usable * measured_rate_per_s))
