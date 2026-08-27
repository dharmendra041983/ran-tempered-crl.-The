#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--primary-root",required=True)
    ap.add_argument("--bootstrap",type=int,default=10000)
    args=ap.parse_args()
    py=sys.executable
    cmds=[
        [py,"evaluate_generalization.py","--primary-root",args.primary_root,
         "--train-seeds","0","1","2","3","4","5","6","7","8","9",
         "--eval-seeds","5","--episodes","10"],
        [py,"validate_generalization.py"],
        [py,"analyze_generalization.py","--bootstrap",str(args.bootstrap)],
    ]
    for cmd in cmds:
        print("\n>>>"," ".join(cmd),flush=True)
        subprocess.run(cmd,cwd=ROOT,check=True)
if __name__=="__main__":
    main()
