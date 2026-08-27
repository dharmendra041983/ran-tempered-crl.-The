#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
ROOT=Path(__file__).resolve().parent
VARIANTS=["raw","tempered","full"]
METRICS=["throughput","sla_cost","cost_embb","cost_urllc","cost_bursty",
         "rejection","fairness","prb_embb","prb_urllc","prb_bursty"]

def load_all(seeds):
    fs=[]
    for v in VARIANTS:
        for s in seeds:
            p=ROOT/"results"/"primary"/v/f"seed{s}"/"evaluation_episodes.csv"
            if not p.exists(): raise FileNotFoundError(p)
            fs.append(pd.read_csv(p))
    return pd.concat(fs,ignore_index=True)

def hier_ci(sub,m,n_boot,rng):
    seeds=sorted(sub.train_seed.unique())
    vals=[]
    for _ in range(n_boot):
        sampled=rng.choice(seeds,size=len(seeds),replace=True)
        sm=[]
        for s in sampled:
            ss=sub[sub.train_seed==s]
            idx=rng.integers(0,len(ss),size=len(ss))
            sm.append(ss.iloc[idx][m].mean())
        vals.append(np.mean(sm))
    return np.quantile(vals,[.025,.975])

def rbc(x,y):
    d=np.asarray(x)-np.asarray(y)
    d=d[~np.isclose(d,0)]
    if len(d)==0:return 0.0
    r=rankdata(np.abs(d))
    rp=r[d>0].sum(); rn=r[d<0].sum()
    return float((rp-rn)/(rp+rn))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,nargs="+",default=list(range(10)))
    ap.add_argument("--bootstrap",type=int,default=10000)
    args=ap.parse_args()
    out=ROOT/"results"/"primary"
    df=load_all(args.seeds)
    df.to_csv(out/"all_evaluation_episodes.csv",index=False)
    seedmeans=df.groupby(["variant","train_seed","condition"])[METRICS].mean().reset_index()
    seedmeans.to_csv(out/"seed_level_means.csv",index=False)

    rows=[]
    for vi,v in enumerate(VARIANTS):
        for ci,c in enumerate(["nominal","stress"]):
            sub=df[(df.variant==v)&(df.condition==c)]
            for mi,m in enumerate(METRICS):
                lo,hi=hier_ci(sub,m,args.bootstrap,np.random.default_rng(20260822+1000*vi+100*ci+mi))
                sv=seedmeans[(seedmeans.variant==v)&(seedmeans.condition==c)][m]
                rows.append(dict(variant=v,condition=c,metric=m,
                    mean_across_train_seeds=sv.mean(),std_across_train_seeds=sv.std(ddof=1),
                    median_across_train_seeds=sv.median(),
                    ci95_lo_hier_boot=lo,ci95_hi_hier_boot=hi,n_train_seeds=len(sv)))
    pd.DataFrame(rows).to_csv(out/"aggregate_hierarchical_ci.csv",index=False)

    tests=[]
    for c in ["nominal","stress"]:
        for a,b in [("tempered","raw"),("tempered","full"),("full","raw")]:
            for m in METRICS:
                aa=seedmeans[(seedmeans.variant==a)&(seedmeans.condition==c)].sort_values("train_seed")
                bb=seedmeans[(seedmeans.variant==b)&(seedmeans.condition==c)].sort_values("train_seed")
                x,y=aa[m].to_numpy(),bb[m].to_numpy()
                d=x-y
                stat,p=(0.0,1.0) if np.allclose(d,0) else wilcoxon(x,y,alternative="two-sided",method="auto")
                tests.append(dict(condition=c,metric=m,variant_a=a,variant_b=b,
                    mean_a=x.mean(),mean_b=y.mean(),mean_diff_a_minus_b=d.mean(),
                    median_diff_a_minus_b=np.median(d),rank_biserial=rbc(x,y),
                    wilcoxon_stat=stat,p_value=p,n_train_seeds=len(x)))
    pd.DataFrame(tests).to_csv(out/"paired_tests_effect_sizes.csv",index=False)

    stress=seedmeans[seedmeans.condition=="stress"].groupby("variant")[[
        "throughput","sla_cost","cost_embb","cost_urllc","cost_bursty","rejection"
    ]].agg(["mean","std","median"])
    stress.to_csv(out/"stress_seed_summary.csv")
    print(stress.round(4).to_string())

if __name__=="__main__":
    main()
