#!/usr/bin/env python3
"""Analyze unseen-condition evaluation with training seed as the independent unit."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

ROOT = Path(__file__).resolve().parent
METRICS = [
    "throughput","sla_cost","cost_embb","cost_urllc","cost_bursty",
    "rejection","fairness","mean_cell_load","max_cell_load",
    "prb_embb","prb_urllc","prb_bursty"
]
VARIANTS=["raw","tempered","full"]

def rbc(x,y):
    d=np.asarray(x)-np.asarray(y)
    d=d[~np.isclose(d,0)]
    if len(d)==0: return 0.0
    r=rankdata(np.abs(d))
    rp=r[d>0].sum(); rn=r[d<0].sum()
    return float((rp-rn)/(rp+rn))

def hier_ci(sub, metric, n_boot, rng):
    seeds=sorted(sub.train_seed.unique())
    vals=np.empty(n_boot)
    for i in range(n_boot):
        sampled=rng.choice(seeds,size=len(seeds),replace=True)
        means=[]
        for s in sampled:
            ss=sub[sub.train_seed==s]
            idx=rng.integers(0,len(ss),size=len(ss))
            means.append(ss.iloc[idx][metric].mean())
        vals[i]=np.mean(means)
    return np.quantile(vals,[.025,.975])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bootstrap",type=int,default=10000)
    args=ap.parse_args()

    p=ROOT/"results"/"generalization"/"all_generalization_episodes.csv"
    if not p.exists(): raise FileNotFoundError(p)
    df=pd.read_csv(p)
    out=ROOT/"results"/"generalization"

    seedmeans=(df.groupby(["variant","train_seed","scenario"])[METRICS]
                 .mean().reset_index())
    seedmeans.to_csv(out/"seed_level_means.csv",index=False)

    rows=[]
    for vi,v in enumerate(VARIANTS):
        for si,sc in enumerate(sorted(df.scenario.unique())):
            sub=df[(df.variant==v)&(df.scenario==sc)]
            for mi,m in enumerate(METRICS):
                sv=seedmeans[(seedmeans.variant==v)&(seedmeans.scenario==sc)][m]
                lo,hi=hier_ci(sub,m,args.bootstrap,
                              np.random.default_rng(20260823+1000*vi+100*si+mi))
                rows.append({
                    "variant":v,"scenario":sc,"metric":m,
                    "mean_across_train_seeds":sv.mean(),
                    "std_across_train_seeds":sv.std(ddof=1),
                    "median_across_train_seeds":sv.median(),
                    "ci95_lo_hier_boot":lo,"ci95_hi_hier_boot":hi,
                    "n_train_seeds":len(sv)
                })
    pd.DataFrame(rows).to_csv(out/"aggregate_hierarchical_ci.csv",index=False)

    tests=[]
    for sc in sorted(df.scenario.unique()):
        for a,b in [("tempered","raw"),("tempered","full"),("full","raw")]:
            for m in METRICS:
                aa=seedmeans[(seedmeans.variant==a)&(seedmeans.scenario==sc)].sort_values("train_seed")
                bb=seedmeans[(seedmeans.variant==b)&(seedmeans.scenario==sc)].sort_values("train_seed")
                x,y=aa[m].to_numpy(),bb[m].to_numpy()
                d=x-y
                if np.allclose(d,0):
                    stat,pv=0.0,1.0
                else:
                    stat,pv=wilcoxon(x,y,alternative="two-sided",method="auto")
                tests.append({
                    "scenario":sc,"metric":m,
                    "variant_a":a,"variant_b":b,
                    "mean_a":x.mean(),"mean_b":y.mean(),
                    "mean_diff_a_minus_b":d.mean(),
                    "median_diff_a_minus_b":np.median(d),
                    "rank_biserial":rbc(x,y),
                    "wilcoxon_stat":stat,"p_value":pv,
                    "n_train_seeds":len(x)
                })
    pd.DataFrame(tests).to_csv(out/"paired_tests_effect_sizes.csv",index=False)

    compact=(seedmeans.groupby(["scenario","variant"])[
        ["throughput","sla_cost","cost_embb","cost_urllc","cost_bursty","rejection"]
    ].agg(["mean","std"]))
    compact.to_csv(out/"generalization_summary.csv")
    print(compact.round(4).to_string())

if __name__=="__main__":
    main()
