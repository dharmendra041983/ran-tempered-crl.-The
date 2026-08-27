#!/usr/bin/env python3
"""Aggregate results for the CMARS-inspired adapted baseline.

Optionally compare it with the tempered controller using matched training seeds
and the same evaluation episodes.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

ROOT=Path(__file__).resolve().parent
METRICS=["throughput","sla_cost","cost_embb","cost_urllc","cost_bursty",
         "rejection","fairness","prb_embb","prb_urllc","prb_bursty"]

def rbc(x,y):
    d=np.asarray(x)-np.asarray(y)
    d=d[~np.isclose(d,0)]
    if len(d)==0:return 0.0
    r=rankdata(np.abs(d))
    rp=r[d>0].sum(); rn=r[d<0].sum()
    return float((rp-rn)/(rp+rn))

def load_cmars(seeds):
    frames=[]
    for s in seeds:
        p=ROOT/"results"/"cmars_adapted"/f"seed{s}"/"evaluation_episodes.csv"
        if not p.exists(): raise FileNotFoundError(p)
        frames.append(pd.read_csv(p))
    return pd.concat(frames,ignore_index=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,nargs="+",default=[0])
    ap.add_argument("--primary-root",type=Path,default=None,
                    help="Optional path to primary controller results")
    args=ap.parse_args()

    out=ROOT/"results"/"cmars_adapted"
    df=load_cmars(args.seeds)
    df.to_csv(out/"cmars_all_evaluation_episodes.csv",index=False)
    sm=df.groupby(["train_seed","condition"])[METRICS].mean().reset_index()
    sm.to_csv(out/"cmars_seed_level_means.csv",index=False)

    print("\nCMARS-inspired baseline seed-level summary:")
    print(sm.groupby("condition")[METRICS].agg(["mean","std"]).round(4).to_string())

    if args.primary_root is None:
        return

    primary_root=args.primary_root.resolve()
    frames=[]
    for s in args.seeds:
        p=primary_root/"tempered"/f"seed{s}"/"evaluation_episodes.csv"
        if not p.exists(): raise FileNotFoundError(p)
        frames.append(pd.read_csv(p))
    primary=pd.concat(frames,ignore_index=True)
    tsm=primary.groupby(["train_seed","condition"])[METRICS].mean().reset_index()

    tests=[]
    for cond in ["nominal","stress"]:
        a=tsm[tsm.condition==cond].sort_values("train_seed")
        b=sm[sm.condition==cond].sort_values("train_seed")
        for m in METRICS:
            x=a[m].to_numpy()
            y=b[m].to_numpy()
            d=x-y
            if np.allclose(d,0):
                stat,p=0.0,1.0
            else:
                stat,p=wilcoxon(x,y,alternative="two-sided",method="auto")
            tests.append({
                "condition":cond,"metric":m,
                "tempered_mean":x.mean(),"cmars_adapted_mean":y.mean(),
                "mean_diff_tempered_minus_cmars":d.mean(),
                "rank_biserial":rbc(x,y),
                "wilcoxon_stat":stat,"p_value":p,
                "n_train_seeds":len(x)
            })
    pd.DataFrame(tests).to_csv(out/"tempered_vs_cmars_adapted_paired.csv",index=False)
    print("\nSaved matched tempered vs CMARS-inspired baseline comparison.")


if __name__=="__main__":
    main()
