#!/usr/bin/env python3
"""Run CMARS-inspired baseline training and evaluation."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["train","eval","all"],default="all")
    ap.add_argument("--seeds",type=int,nargs="+",default=[0])
    ap.add_argument("--macro-steps",type=int,default=90000)
    ap.add_argument("--eval-seeds",type=int,default=5)
    ap.add_argument("--episodes",type=int,default=10)
    args=ap.parse_args()
    py=sys.executable

    if args.stage in {"train","all"}:
        for s in args.seeds:
            run([py,"train_cmars_adapted.py","--seed",str(s),
                 "--macro-steps",str(args.macro_steps)])
    if args.stage in {"eval","all"}:
        for s in args.seeds:
            run([py,"evaluate_cmars_adapted.py","--train-seed",str(s),
                 "--eval-seeds",str(args.eval_seeds),
                 "--episodes",str(args.episodes)])

if __name__=="__main__":
    main()
