#!/usr/bin/env python3
"""Macro-average multiple topology runs and emit JSON, CSV and LaTeX tables."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from topokpi.metrics import average_precision,binary_metrics,rca_metrics,roc_auc


def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True,help='JSON list: [{"scale":12,"run":"..."}]'); p.add_argument("--output",required=True); args=p.parse_args(); entries=json.loads(Path(args.manifest).read_text(encoding="utf-8")); rows=[]
    for entry in entries:
        z=np.load(Path(entry["run"])/"test_outputs.npz"); threshold=float(z["alarm_threshold"]); scores=z["system_scores"]; labels=z["labels"]
        groups=["all"]+sorted(set(z["fault_types"].tolist())-{"normal"})
        for fault in groups:
            mask=np.ones(len(labels),bool) if fault=="all" else ((z["fault_types"]==fault)|(labels==0))
            detection=binary_metrics(labels[mask],scores[mask],threshold); detection.update({"auroc":roc_auc(labels[mask],scores[mask]),"auprc":average_precision(labels[mask],scores[mask])})
            event_mask=np.ones(len(z["event_roots"]),bool) if fault=="all" else z["event_fault_types"]==fault
            rankings=[[int(v) for v in row if v>=0] for row in z["event_rankings"][event_mask]]; roots=z["event_roots"][event_mask].tolist(); rca=rca_metrics(rankings,roots) if roots else {k:float("nan") for k in ("a@1","a@3","a@5","mrr","mean_rank")}
            rows.append({"scale":entry["scale"],"fault":fault,**{f"det_{k}":v for k,v in detection.items() if k in ("precision","recall","f1","auroc","auprc")},**rca})
    numeric=[k for k in rows[0] if k not in ("scale","fault")]; overall={k:float(np.nanmean([r[k] for r in rows if r["fault"]=="all"])) for k in numeric}; output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    (output/"aggregate.json").write_text(json.dumps({"rows":rows,"macro_across_scales":overall},indent=2),encoding="utf-8")
    with (output/"aggregate.csv").open("w",newline="",encoding="utf-8") as f: writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    columns=["det_f1","det_auroc","det_auprc","a@1","a@3","a@5","mrr","mean_rank"]; latex=["\\begin{tabular}{l"+"c"*len(columns)+"}","Scale & "+" & ".join(columns)+" \\\\ \\hline"]
    for row in [r for r in rows if r["fault"]=="all"]: latex.append(str(row["scale"])+" & "+" & ".join(f"{row[c]:.3f}" for c in columns)+" \\\\")
    latex.append("Macro & "+" & ".join(f"{overall[c]:.3f}" for c in columns)+" \\\\"); latex.append("\\end{tabular}"); (output/"aggregate.tex").write_text("\n".join(latex),encoding="utf-8"); print(json.dumps(overall,indent=2))


if __name__=="__main__": main()
