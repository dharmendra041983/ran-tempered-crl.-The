#!/usr/bin/env python3
"""Train one controller variant for the matched multi-seed experiment.

Variants:
  raw       -> alpha=0
  tempered  -> alpha=0.5
  full      -> alpha=1

Everything except the dual weighting rule is frozen.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
from stable_baselines3 import SAC
from multicell_env import MultiCellNGRANSlicingEnv
from policies import DiagnosticLagrangianCallback

ROOT = Path(__file__).resolve().parent

SAC_KW = dict(
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=512,
    buffer_size=100_000,
    learning_starts=1000,
    verbose=0,
    device="cpu",
)

VARIANT_ALPHA = {"raw": 0.0, "tempered": 0.5, "full": 1.0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANT_ALPHA), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=90_000)
    ap.add_argument("--lambda-lr", type=float, default=0.02)
    ap.add_argument("--lambda-update", type=int, default=3000)
    ap.add_argument("--stress-prob", type=float, default=0.4)
    ap.add_argument("--stress-factor", type=float, default=1.6)
    args = ap.parse_args()

    alpha = VARIANT_ALPHA[args.variant]
    out = ROOT / "results" / "primary" / args.variant / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    env = MultiCellNGRANSlicingEnv({
        "stress": False,
        "stress_prob": args.stress_prob,
        "stress_factor": args.stress_factor,
        "constraint_mode": "per_slice",
        "control_mode": "joint",
    })
    model = SAC("MlpPolicy", env, seed=args.seed, **SAC_KW)

    if alpha == 0.0:
        mode = "raw"
    elif alpha == 1.0:
        mode = "budget_normalized"
    else:
        mode = "tempered"

    cb = DiagnosticLagrangianCallback(
        eps=env.get_eps(),
        lr_lambda=args.lambda_lr,
        update_every=args.lambda_update,
        constraint_mode="per_slice",
        dual_update_mode=mode,
        normalization_alpha=alpha,
        normalized_lr_scale="mean_budget",
        log_path=out / "dual_dynamics.csv",
        verbose=1,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)
    elapsed = time.perf_counter() - t0
    model.save(out / "crl_per_slice")

    eps = env.get_eps()
    eta_effective = (
        args.lambda_lr if alpha == 0.0
        else args.lambda_lr * float(np.mean(eps)) ** alpha
    )

    meta = {
        "experiment": "primary",
        "variant": args.variant,
        "alpha": alpha,
        "seed": args.seed,
        "train_steps": args.steps,
        "train_seconds": elapsed,
        "stress_prob": args.stress_prob,
        "stress_factor": args.stress_factor,
        "eps": eps.tolist(),
        "lambda_lr_raw_reference": args.lambda_lr,
        "eta_effective": eta_effective,
        "lambda_update": args.lambda_update,
        "final_lambda": cb.lam.tolist(),
        "num_dual_updates": len(cb.records),
        "environment": env.cfg,
    }
    with (out / "train_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"training complete: variant={args.variant}, seed={args.seed}")
    print(f"alpha={alpha}, eta_effective={eta_effective:.8f}")
    print(f"final_lambda={np.round(cb.lam,4)}")
    print(f"train_seconds={elapsed:.1f}")

if __name__ == "__main__":
    main()
