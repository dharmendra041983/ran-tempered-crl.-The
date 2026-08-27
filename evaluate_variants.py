#!/usr/bin/env python3
"""Evaluate one trained controller on the paired episode set.

Default: 5 evaluation seeds x 10 episodes per condition = 50 nominal and
50 stress episodes per trained model. Episode IDs are identical across variants.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from multicell_env import MultiCellNGRANSlicingEnv

ROOT = Path(__file__).resolve().parent

def rollout(model, condition, eval_seeds, episodes, stress_factor):
    stress = condition == "stress"
    rows = []
    for eval_seed in eval_seeds:
        for ep in range(episodes):
            episode_id = eval_seed * 10_000 + ep
            env = MultiCellNGRANSlicingEnv({
                "stress": stress,
                "stress_prob": 0.0,
                "constraint_mode": "per_slice",
                "stress_factor": stress_factor,
            })
            obs, _ = env.reset(seed=episode_id)
            done = False
            buf = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
                if (not stress) or info["stress"]:
                    buf.append(info)

            rows.append({
                "eval_seed": eval_seed,
                "episode": ep,
                "episode_id": episode_id,
                "condition": condition,
                "throughput": np.mean([x["throughput"] for x in buf]),
                "rejection": np.mean([x["rejection"] for x in buf]),
                "sla_cost": np.mean([x["sla_cost"] for x in buf]),
                "fairness": np.mean([x["jain"] for x in buf]),
                "cost_embb": np.mean([x["cost_per_slice"][0] for x in buf]),
                "cost_urllc": np.mean([x["cost_per_slice"][1] for x in buf]),
                "cost_bursty": np.mean([x["cost_per_slice"][2] for x in buf]),
                "mean_cell_load": np.mean([np.mean(x["cell_load"]) for x in buf]),
                "max_cell_load": np.mean([np.max(x["cell_load"]) for x in buf]),
                "mean_interference": np.mean([np.mean(x["interference"]) for x in buf]),
                "prb_embb": np.mean([x["prb_share_per_slice"][0] for x in buf]),
                "prb_urllc": np.mean([x["prb_share_per_slice"][1] for x in buf]),
                "prb_bursty": np.mean([x["prb_share_per_slice"][2] for x in buf]),
                "admit_embb": np.mean([x["admission"][0] for x in buf]),
                "admit_urllc": np.mean([x["admission"][1] for x in buf]),
                "admit_bursty": np.mean([x["admission"][2] for x in buf]),
            })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["raw","tempered","full"], required=True)
    ap.add_argument("--train-seed", type=int, required=True)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--stress-factor", type=float, default=1.6)
    args = ap.parse_args()

    out = ROOT / "results" / "primary" / args.variant / f"seed{args.train_seed}"
    model_path = out / "crl_per_slice.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    model = SAC.load(model_path, device="cpu")
    seeds = list(range(args.eval_seeds))
    df = pd.concat([
        rollout(model, "nominal", seeds, args.episodes, args.stress_factor),
        rollout(model, "stress", seeds, args.episodes, args.stress_factor),
    ], ignore_index=True)

    df.insert(0, "variant", args.variant)
    df.insert(1, "train_seed", args.train_seed)
    df.to_csv(out / "evaluation_episodes.csv", index=False)

    summary = df.groupby("condition").agg(
        n=("sla_cost","size"),
        throughput_mean=("throughput","mean"),
        throughput_std=("throughput","std"),
        sla_cost_mean=("sla_cost","mean"),
        sla_cost_std=("sla_cost","std"),
        rejection_mean=("rejection","mean"),
        rejection_std=("rejection","std"),
        embb_cost_mean=("cost_embb","mean"),
        urllc_cost_mean=("cost_urllc","mean"),
        bursty_cost_mean=("cost_bursty","mean"),
        prb_embb_mean=("prb_embb","mean"),
        prb_urllc_mean=("prb_urllc","mean"),
        prb_bursty_mean=("prb_bursty","mean"),
        max_cell_load_mean=("max_cell_load","mean"),
        interference_mean=("mean_interference","mean"),
    )
    summary.to_csv(out / "evaluation_summary.csv")
    print(summary.round(4).to_string())

if __name__ == "__main__":
    main()
