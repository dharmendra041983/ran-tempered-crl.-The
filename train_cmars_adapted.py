#!/usr/bin/env python3
"""Train the CMARS-inspired adapted baseline.

One network macro decision requires 3 sequential slice-agent PPO transitions.
Therefore --macro-steps 90000 corresponds to 270000 PPO transitions.
"""
from __future__ import annotations
import argparse, csv, json, os, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from cmars_adapted_env import CMARSSequentialWrapper

ROOT = Path(__file__).resolve().parent


class CMARSDiagnosticCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.rows = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if not info.get("macro_step_complete", False):
                continue
            dual = info.get("dual_update_record")
            if dual is None:
                continue
            lam = np.asarray(dual["lambda"])
            cost = np.asarray(dual["mean_cost"])
            residual = np.asarray(dual["residual"])
            row = {
                "macro_step": int(dual["macro_step"]),
                "lambda_embb": float(lam[0]),
                "lambda_urllc": float(lam[1]),
                "lambda_bursty": float(lam[2]),
                "cost_embb": float(cost[0]),
                "cost_urllc": float(cost[1]),
                "cost_bursty": float(cost[2]),
                "residual_embb": float(residual[0]),
                "residual_urllc": float(residual[1]),
                "residual_bursty": float(residual[2]),
                "throughput": float(info["throughput"]),
                "rejection": float(info["rejection"]),
                "sla_cost": float(info["sla_cost"]),
                "prb_embb": float(info["prb_share_per_slice"][0]),
                "prb_urllc": float(info["prb_share_per_slice"][1]),
                "prb_bursty": float(info["prb_share_per_slice"][2]),
                "admit_embb": float(info["admission"][0]),
                "admit_urllc": float(info["admission"][1]),
                "admit_bursty": float(info["admission"][2]),
                "mean_cell_load": float(np.mean(info["cell_load"])),
                "max_cell_load": float(np.max(info["cell_load"])),
            }
            self.rows.append(row)
            if self.verbose:
                print(
                    f"[CMARS dual] macro={row['macro_step']} "
                    f"lambda={np.round(lam,3)} cost={np.round(cost,3)}"
                )
        return True

    def _on_training_end(self):
        if not self.rows:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--macro-steps", type=int, default=90_000)
    ap.add_argument("--stress-prob", type=float, default=0.4)
    ap.add_argument("--stress-factor", type=float, default=1.6)
    args = ap.parse_args()

    out = ROOT / "results" / "cmars_adapted" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    env = CMARSSequentialWrapper(
        base_config={
            "stress": False,
            "stress_prob": args.stress_prob,
            "stress_factor": args.stress_factor,
            "control_mode": "joint",
        },
        raw_lambda_lr=0.02,
        lambda_update_macro=3000,
    )

    # PPO is chosen because the closest prior work uses constrained
    # multi-agent PPO-Lagrangian. Hyperparameters are conventional and frozen
    # before the CMARS baseline seed comparison; there is no CMARS baseline grid search.
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=0,
        device="cpu",
    )

    cb = CMARSDiagnosticCallback(out / "dual_dynamics.csv", verbose=1)
    ppo_steps = int(args.macro_steps * env.S)

    t0 = time.perf_counter()
    model.learn(total_timesteps=ppo_steps, callback=cb, progress_bar=False)
    elapsed = time.perf_counter() - t0
    model.save(out / "cmars_adapted")

    meta = {
        "experiment": "CMARS baseline_v2",
        "baseline": "CMARS-inspired adapted sequential cooperative PPO",
        "seed": args.seed,
        "macro_steps_target": args.macro_steps,
        "ppo_transitions_target": ppo_steps,
        "train_seconds": elapsed,
        "stress_prob": args.stress_prob,
        "stress_factor": args.stress_factor,
        "dual_update": "raw per-slice projected ascent",
        "lambda_lr": 0.02,
        "lambda_update_macro": 3000,
        "eps": env.base.get_eps().tolist(),
        "final_lambda": env.lam.tolist(),
        "num_dual_updates": len(cb.rows),
        "adaptation_note": (
            "Not an exact CMARS reproduction. Sequential shuffled slice-agent "
            "resource sharing, cooperative PPO, local/global state, and "
            "constraint-aware raw Lagrangian structure are adapted to the same "
            "7-cell joint admission/resource environment used by our method."
        ),
    }
    with (out / "train_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved CMARS baseline_v2 CMARS-adapted seed {args.seed} to {out}")
    print(f"Training time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
