#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
base=ROOT/"results"/"cmars_adapted"/"seed0"
required=["cmars_adapted.zip","train_metadata.json","evaluation_episodes.csv","dual_dynamics.csv"]
missing=[str(base/f) for f in required if not (base/f).exists()]
if missing:
    print("Missing required files:")
    print("\n".join(missing))
    raise SystemExit(1)
dual=pd.read_csv(base/"dual_dynamics.csv")
if len(dual)==0:
    raise SystemExit("dual_dynamics.csv is empty; constrained baseline invalid.")
print(f"CMARS baseline_v2 seed0 validation passed with {len(dual)} dual updates.")
