#!/usr/bin/env python3
"""Measure parameter count, batch-1 detection latency and RCA latency."""
from __future__ import annotations
import argparse, json, time, resource
import numpy as np
import torch
from topokpi.model import TopologyAwareSSL
from topokpi.rca import localize


def sync(device):
    if str(device).startswith("cuda"): torch.cuda.synchronize()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--nodes",type=int,choices=[12,24,48],required=True); p.add_argument("--device"); p.add_argument("--warmup",type=int,default=100); p.add_argument("--iterations",type=int,default=1000); p.add_argument("--train-iterations",type=int,default=50); args=p.parse_args()
    device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"); model=TopologyAwareSSL().to(device).eval(); x=torch.randn(1,20,args.nodes,8,device=device); y=torch.randn(1,args.nodes,8,device=device); a=torch.eye(args.nodes,device=device)
    with torch.no_grad():
        for _ in range(args.warmup): model.scores(x,y,a)
        sync(device); start=time.perf_counter()
        for _ in range(args.iterations): model.scores(x,y,a)
        sync(device); detection=(time.perf_counter()-start)*1000/args.iterations
    scores=np.abs(np.random.default_rng(1).normal(size=args.nodes)); w=np.zeros((args.nodes,args.nodes)); w[np.arange(args.nodes-1),np.arange(1,args.nodes)]=1
    for _ in range(args.warmup): localize(scores,w,np.quantile(scores,.7))
    start=time.perf_counter()
    for _ in range(args.iterations): localize(scores,w,np.quantile(scores,.7))
    rca=(time.perf_counter()-start)*1000/args.iterations
    model.train(); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    if str(device).startswith("cuda"): torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for _ in range(args.train_iterations):
        optimizer.zero_grad(set_to_none=True); kpi,node,rec,pred,latent=model.scores(x,y,a); loss=rec.abs().mean()+pred.abs().mean()+latent.square().mean()*.01; loss.backward(); optimizer.step()
    sync(device); training_ms=(time.perf_counter()-start)*1000/max(args.train_iterations,1)
    peak_gpu=torch.cuda.max_memory_allocated()/1024**3 if str(device).startswith("cuda") else None; peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux KiB.
    peak_rss_gb=peak_rss/1024**3 if peak_rss>10**7 else peak_rss/1024**2
    result={"nodes":args.nodes,"parameters":sum(p.numel() for p in model.parameters()),"training_step_ms":training_ms,"detection_ms":detection,"throughput_windows_per_second":1000/detection,"rca_ms":rca,"combined_ms":detection+rca,"peak_gpu_gb":peak_gpu,"peak_process_rss_gb":peak_rss_gb,"device":device,"warmup":args.warmup,"timed_iterations":args.iterations}
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
