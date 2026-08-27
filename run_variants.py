#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
VARIANTS = ["raw","tempered","full"]
def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["train","eval","all"],default="all")
    ap.add_argument("--seeds",type=int,nargs="+",default=[5,6,7,8,9])
    ap.add_argument("--steps",type=int,default=90000)
    ap.add_argument("--eval-seeds",type=int,default=5)
    ap.add_argument("--episodes",type=int,default=10)
    args=ap.parse_args()
    py=sys.executable
    if args.stage in {"train","all"}:
        for v in VARIANTS:
            for s in args.seeds:
                run([py,"train_primary.py","--variant",v,"--seed",str(s),"--steps",str(args.steps)])
    if args.stage in {"eval","all"}:
        for v in VARIANTS:
            for s in args.seeds:
                run([py,"evaluate_primary.py","--variant",v,"--train-seed",str(s),
                     "--eval-seeds",str(args.eval_seeds),"--episodes",str(args.episodes)])
if __name__=="__main__":
    main()
