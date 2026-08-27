#!/usr/bin/env python3
"""Evaluate one CMARS-inspired adapted baseline on matched episodes."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from cmars_adapted_env import CMARSSequentialWrapper

ROOT = Path(__file__).resolve().parent


def run_episode(model, stress, episode_id, stress_factor):
    env = CMARSSequentialWrapper(
        base_config={
            "stress": stress,
            "stress_prob": 0.0,
            "stress_factor": stress_factor,
            "control_mode": "joint",
        },
        raw_lambda_lr=0.02,
        lambda_update_macro=3000,
    )
    obs, _ = env.reset(seed=episode_id)
    done = False
    macro = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        if info.get("macro_step_complete", False):
            if (not stress) or info["stress"]:
                macro.append(info)

    return {
        "throughput": np.mean([x["throughput"] for x in macro]),
        "rejection": np.mean([x["rejection"] for x in macro]),
        "sla_cost": np.mean([x["sla_cost"] for x in macro]),
        "fairness": np.mean([x["jain"] for x in macro]),
        "cost_embb": np.mean([x["cost_per_slice"][0] for x in macro]),
        "cost_urllc": np.mean([x["cost_per_slice"][1] for x in macro]),
        "cost_bursty": np.mean([x["cost_per_slice"][2] for x in macro]),
        "prb_embb": np.mean([x["prb_share_per_slice"][0] for x in macro]),
        "prb_urllc": np.mean([x["prb_share_per_slice"][1] for x in macro]),
        "prb_bursty": np.mean([x["prb_share_per_slice"][2] for x in macro]),
        "admit_embb": np.mean([x["admission"][0] for x in macro]),
        "admit_urllc": np.mean([x["admission"][1] for x in macro]),
        "admit_bursty": np.mean([x["admission"][2] for x in macro]),
        "mean_cell_load": np.mean([np.mean(x["cell_load"]) for x in macro]),
        "max_cell_load": np.mean([np.max(x["cell_load"]) for x in macro]),
        "mean_interference": np.mean([np.mean(x["interference"]) for x in macro]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seed", type=int, required=True)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--stress-factor", type=float, default=1.6)
    args = ap.parse_args()

    out = ROOT / "results" / "cmars_adapted" / f"seed{args.train_seed}"
    model_path = out / "cmars_adapted.zip"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = PPO.load(model_path, device="cpu")
    rows = []
    for condition in ["nominal", "stress"]:
        stress = condition == "stress"
        for eval_seed in range(args.eval_seeds):
            for ep in range(args.episodes):
                episode_id = eval_seed * 10_000 + ep
                rec = run_episode(model, stress, episode_id, args.stress_factor)
                rec.update({
                    "baseline": "cmars_adapted",
                    "train_seed": args.train_seed,
                    "eval_seed": eval_seed,
                    "episode": ep,
                    "episode_id": episode_id,
                    "condition": condition,
                })
                rows.append(rec)

    df = pd.DataFrame(rows)
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
    )
    summary.to_csv(out / "evaluation_summary.csv")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
