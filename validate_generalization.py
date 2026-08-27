#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
p=ROOT/"results"/"generalization"/"all_generalization_episodes.csv"
if not p.exists():
    raise SystemExit(f"Missing {p}")
df=pd.read_csv(p)
expected_variants={"raw","tempered","full"}
expected_seeds=set(range(10))
expected_scenarios={
    "load_low_0p85","load_high_1p15",
    "stress_mild_1p30","stress_severe_1p90",
    "capacity_down_0p90","capacity_up_1p10"
}
assert set(df.variant.unique())==expected_variants, set(df.variant.unique())
assert set(df.train_seed.unique())==expected_seeds, set(df.train_seed.unique())
assert set(df.scenario.unique())==expected_scenarios, set(df.scenario.unique())

# 3 variants x 10 train seeds x 6 scenarios x 5 eval seeds x 10 episodes
expected_rows=3*10*6*5*10
if len(df)!=expected_rows:
    raise SystemExit(f"Expected {expected_rows} rows, found {len(df)}")
print(f"Generalization validation passed: {len(df)} paired evaluation episodes.")
