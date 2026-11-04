#!/usr/bin/env python3
"""Paired bootstrap comparison between two saved experiment runs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from topokpi.statistics import paired_bootstrap_detection,paired_bootstrap_hits


def main():
    p=argparse.ArgumentParser(); p.add_argument("--a",required=True); p.add_argument("--b",required=True); p.add_argument("--samples",type=int,default=5000); p.add_argument("--seed",type=int,default=17); p.add_argument("--output"); args=p.parse_args(); a=np.load(Path(args.a)/"test_outputs.npz"); b=np.load(Path(args.b)/"test_outputs.npz")
    if not np.array_equal(a["labels"],b["labels"]): raise SystemExit("runs are not paired on identical test labels")
    result={"detection":paired_bootstrap_detection(a["labels"],a["system_scores"],float(a["alarm_threshold"]),b["system_scores"],float(b["alarm_threshold"]),args.samples,args.seed)}
    if "event_rankings" in a.files and "event_rankings" in b.files and np.array_equal(a["event_roots"],b["event_roots"]):
        roots=a["event_roots"]; hit_a=np.array([root in row[:1] for root,row in zip(roots,a["event_rankings"])]); hit_b=np.array([root in row[:1] for root,row in zip(roots,b["event_rankings"])]); result["rca_a1"]=paired_bootstrap_hits(hit_a,hit_b,args.samples,args.seed)
    text=json.dumps(result,indent=2); print(text)
    if args.output: Path(args.output).write_text(text,encoding="utf-8")


if __name__=="__main__": main()

