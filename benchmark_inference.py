#!/usr/bin/env python3
"""Measure inference latency for the frozen 72-D seven-cell SAC policy.

Example (Windows PowerShell):
    python benchmark_inference.py --model "results\\primary\\tempered\\seed0\\crl_per_slice.zip"

The script reports the latency of Stable-Baselines3 ``model.predict`` on CPU,
which includes the normal Python/NumPy-to-Torch prediction path used by the
reproducibility code.  It also times the three-element projected dual-update
kernel separately.  Run on the machine/platform whose CPU you intend to name
with the results; do not reuse timings from another machine.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from multicell_env import MultiCellNGRANSlicingEnv


def cpu_name() -> str:
    name = platform.processor().strip()
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "name"], text=True,
                stderr=subprocess.DEVNULL
            )
            lines = [x.strip() for x in out.splitlines() if x.strip()]
            if len(lines) >= 2:
                return lines[1]
        except Exception:
            pass
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return name or platform.machine()


def pct(values_ns: np.ndarray, q: float) -> float:
    return float(np.percentile(values_ns, q) / 1e6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--runs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--threads", type=int, default=1,
                    help="PyTorch intra-op CPU threads")
    ap.add_argument("--output", type=Path, default=Path("inference_timing.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    env = MultiCellNGRANSlicingEnv({
        "stress": False,
        "stress_prob": 0.0,
        "stress_factor": 1.6,
        "constraint_mode": "per_slice",
        "control_mode": "joint",
    })
    obs, _ = env.reset(seed=args.seed)
    if tuple(obs.shape) != (72,):
        raise RuntimeError(f"Expected 72-D observation, got {obs.shape}")

    model = SAC.load(model_path, device="cpu")

    # Use a reproducible bank of valid observations from the actual environment
    # instead of repeatedly timing one identical vector.
    rng = np.random.default_rng(args.seed)
    obs_bank = [obs.copy()]
    for _ in range(255):
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        obs_bank.append(obs.copy())

    for i in range(args.warmup):
        model.predict(obs_bank[i % len(obs_bank)], deterministic=True)

    times = np.empty(args.runs, dtype=np.int64)
    for i in range(args.runs):
        x = obs_bank[i % len(obs_bank)]
        t0 = time.perf_counter_ns()
        model.predict(x, deterministic=True)
        times[i] = time.perf_counter_ns() - t0

    # Time the exact arithmetic kernel used by the projected dual update.
    eps = np.array([0.20, 0.05, 0.15], dtype=np.float64)
    lam = np.array([0.1, 0.3, 0.1], dtype=np.float64)
    residual = np.array([0.02, -0.01, 0.03], dtype=np.float64)
    eta = 0.02 * float(np.mean(eps)) ** 0.5
    dual_times = np.empty(args.runs, dtype=np.int64)
    for i in range(args.runs):
        t0 = time.perf_counter_ns()
        lam = np.maximum(0.0, lam + eta * residual / np.sqrt(eps))
        dual_times[i] = time.perf_counter_ns() - t0

    result = {
        "cpu": cpu_name(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "stable_baselines3": __import__("stable_baselines3").__version__,
        "torch_threads": args.threads,
        "model": str(model_path),
        "observation_dimension": int(obs_bank[0].size),
        "warmup_calls": args.warmup,
        "timed_calls": args.runs,
        "predict_latency_ms": {
            "mean": float(times.mean() / 1e6),
            "median": pct(times, 50),
            "p95": pct(times, 95),
            "p99": pct(times, 99),
            "max": float(times.max() / 1e6),
        },
        "dual_update_latency_us": {
            "mean": float(dual_times.mean() / 1e3),
            "median": float(np.percentile(dual_times, 50) / 1e3),
            "p99": float(np.percentile(dual_times, 99) / 1e3),
            "max": float(dual_times.max() / 1e3),
        },
        "dual_update_period_network_steps": 3000,
    }

    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
