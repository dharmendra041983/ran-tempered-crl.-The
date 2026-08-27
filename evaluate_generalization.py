#!/usr/bin/env python3
"""Evaluate trained controllers under unseen load, stress, and capacity shifts.

No retraining is performed. All controller variants and training seeds use the
same evaluation episode IDs for paired analysis.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from multicell_env import MultiCellNGRANSlicingEnv, DEFAULT_CFG

ROOT = Path(__file__).resolve().parent
VARIANTS = ["raw", "tempered", "full"]

# Operating shifts used for the unseen-condition evaluation.
SCENARIOS = {
    # Unseen nominal load levels
    "load_low_0p85": dict(load_scale=0.85, stress=False, stress_factor=1.6, capacity_scale=1.0),
    "load_high_1p15": dict(load_scale=1.15, stress=False, stress_factor=1.6, capacity_scale=1.0),

    # Unseen stress severities; training used 1.6
    "stress_mild_1p30": dict(load_scale=1.0, stress=True, stress_factor=1.30, capacity_scale=1.0),
    "stress_severe_1p90": dict(load_scale=1.0, stress=True, stress_factor=1.90, capacity_scale=1.0),

    # Unseen capacity perturbations, evaluated under the familiar 1.6 stress
    "capacity_down_0p90": dict(load_scale=1.0, stress=True, stress_factor=1.6, capacity_scale=0.90),
    "capacity_up_1p10": dict(load_scale=1.0, stress=True, stress_factor=1.6, capacity_scale=1.10),
}

def env_config(scenario):
    spec = SCENARIOS[scenario]
    base_arr = np.asarray(DEFAULT_CFG["aggregate_arrival"], dtype=float)
    cfg = {
        "aggregate_arrival": (base_arr * spec["load_scale"]).tolist(),
        "prbs_per_cell": float(DEFAULT_CFG["prbs_per_cell"]) * spec["capacity_scale"],
        "stress": bool(spec["stress"]),
        "stress_prob": 0.0,
        "stress_factor": float(spec["stress_factor"]),
        "constraint_mode": "per_slice",
        "control_mode": "joint",
    }
    return cfg

def rollout(model, scenario, eval_seeds, episodes):
    cfg = env_config(scenario)
    stress = bool(SCENARIOS[scenario]["stress"])
    rows = []
    for eval_seed in eval_seeds:
        for ep in range(episodes):
            episode_id = eval_seed * 10_000 + ep
            env = MultiCellNGRANSlicingEnv(cfg)
            obs, _ = env.reset(seed=episode_id)
            done = False
            buf = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
                # For stress scenarios evaluate only the stress window, matching
                # stress-window reporting. Nominal-load scenarios use the full episode.
                if (not stress) or info["stress"]:
                    buf.append(info)

            rows.append({
                "eval_seed": eval_seed,
                "episode": ep,
                "episode_id": episode_id,
                "scenario": scenario,
                "load_scale": SCENARIOS[scenario]["load_scale"],
                "stress_factor": SCENARIOS[scenario]["stress_factor"],
                "capacity_scale": SCENARIOS[scenario]["capacity_scale"],
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
    ap.add_argument("--primary-root", type=Path, required=True,
                    help="Path to trained primary controller results")
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    args = ap.parse_args()

    primary_root = args.primary_root.resolve()
    outroot = ROOT / "results" / "generalization"
    outroot.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for variant in args.variants:
        for train_seed in args.train_seeds:
            model_path = primary_root / variant / f"seed{train_seed}" / "crl_per_slice.zip"
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            print(f"\nLoading {variant} seed {train_seed}: {model_path}", flush=True)
            model = SAC.load(model_path, device="cpu")

            for scenario in args.scenarios:
                print(f"  evaluating {scenario}", flush=True)
                df = rollout(model, scenario, list(range(args.eval_seeds)), args.episodes)
                df.insert(0, "variant", variant)
                df.insert(1, "train_seed", train_seed)

                seedout = outroot / variant / f"seed{train_seed}"
                seedout.mkdir(parents=True, exist_ok=True)
                df.to_csv(seedout / f"{scenario}.csv", index=False)
                all_frames.append(df)

    full = pd.concat(all_frames, ignore_index=True)
    full.to_csv(outroot / "all_generalization_episodes.csv", index=False)

    with (outroot / "scenario_definition.json").open("w") as f:
        json.dump(SCENARIOS, f, indent=2)

    print(f"\nSaved generalization evaluation data to {outroot}")

if __name__ == "__main__":
    main()
