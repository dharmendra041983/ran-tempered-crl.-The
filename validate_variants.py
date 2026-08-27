#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parent
missing=[]
for v in ["raw","tempered","full"]:
    for s in range(10):
        base=ROOT/"results"/"primary"/v/f"seed{s}"
        for f in ["crl_per_slice.zip","dual_dynamics.csv","train_metadata.json","evaluation_episodes.csv"]:
            if not (base/f).exists(): missing.append(str(base/f))
if missing:
    print(f"Missing {len(missing)} required files:")
    print("\n".join(missing))
    raise SystemExit(1)
print("Result set complete: 3 variants x 10 training seeds.")
