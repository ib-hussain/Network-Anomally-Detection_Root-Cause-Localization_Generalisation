"""Mininet campaign specifications and data-recording helpers.

The module itself does not import Mininet, so topology, schedules and labels can
be reviewed and tested on machines without root privileges or Mininet installed.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import numpy as np
from .data import FAULT_TYPES, RawDataset, save_npz


@dataclass(frozen=True)
class FaultEvent:
    event_id: int
    fault_type: str
    root: int
    start_step: int
    duration_steps: int
    magnitude: float

    @property
    def stop_step(self) -> int: return self.start_step + self.duration_steps


def topology_edges(nodes: int) -> list[tuple[int, int]]:
    """Deterministic connected hierarchical topology for 12/24/48 nodes."""
    if nodes not in (12, 24, 48): raise ValueError("paper topology size must be 12, 24 or 48")
    edges=set()
    # Backbone ring among one quarter of the nodes.
    backbone=max(3,nodes//4)
    for i in range(backbone): edges.add(tuple(sorted((i,(i+1)%backbone))))
    # Attach remaining nodes evenly, then add sparse lateral redundancy.
    for i in range(backbone,nodes):
        parent=(i-backbone)%backbone; edges.add(tuple(sorted((parent,i))))
        if (i-backbone)%3==0 and i+1<nodes: edges.add((i,i+1))
    return sorted(edges)


def adjacency(nodes: int, edges: list[tuple[int,int]] | None = None) -> np.ndarray:
    a=np.zeros((nodes,nodes),np.float32)
    for i,j in edges or topology_edges(nodes): a[i,j]=a[j,i]=1
    return a


def directed_forwarding(nodes: int, root: int, edges: list[tuple[int,int]] | None = None) -> np.ndarray:
    """Orient a BFS tree away from the injected root as an event forwarding snapshot."""
    graph=[[] for _ in range(nodes)]
    for i,j in edges or topology_edges(nodes): graph[i].append(j); graph[j].append(i)
    w=np.zeros((nodes,nodes),np.float32); seen={root}; queue=[root]
    while queue:
        i=queue.pop(0)
        for j in sorted(graph[i]):
            if j not in seen: seen.add(j); queue.append(j); w[i,j]=1
    return w


def default_schedule(nodes: int, steps: int, event_duration: int = 20, seed: int = 17) -> list[FaultEvent]:
    if steps < 10*event_duration: raise ValueError("campaign is too short for six separated events")
    rng=np.random.default_rng(seed); starts=np.linspace(int(.62*steps),steps-event_duration-2,len(FAULT_TYPES),dtype=int)
    roots=rng.choice(nodes,len(FAULT_TYPES),replace=nodes>=len(FAULT_TYPES))
    magnitudes=(.9,.5,.15,.12,1.,1.)
    return [FaultEvent(i,str(name),int(root),int(start),event_duration,float(mag)) for i,(name,root,start,mag) in enumerate(zip(FAULT_TYPES,roots,starts,magnitudes))]


def event_labels(nodes: int, steps: int, events: list[FaultEvent]):
    labels=np.zeros(steps,np.int8); roots=np.full(steps,-1,np.int32); faults=np.full(steps,"normal",dtype="<U32"); ids=np.full(steps,-1,np.int32)
    w=np.repeat(adjacency(nodes)[None],steps,axis=0)
    for event in events:
        sl=slice(event.start_step,min(event.stop_step,steps)); labels[sl]=1; roots[sl]=event.root; faults[sl]=event.fault_type; ids[sl]=event.event_id
        w[sl]=directed_forwarding(nodes,event.root)
    return labels,roots,faults,ids,w


def write_campaign_plan(path: str | Path, nodes: int, steps: int, interval: int, events: list[FaultEvent]) -> None:
    payload={"nodes":nodes,"steps":steps,"sampling_seconds":interval,"edges":topology_edges(nodes),"events":[asdict(e) for e in events]}
    Path(path).write_text(json.dumps(payload,indent=2),encoding="utf-8")


def save_campaign(path: str | Path, kpis: np.ndarray, nodes: int, interval: int,
                  events: list[FaultEvent], timestamps: np.ndarray | None = None) -> None:
    labels,roots,faults,_,w=event_labels(nodes,len(kpis),events)
    save_npz(path,RawDataset(np.asarray(kpis,np.float32),labels,roots,faults,adjacency(nodes),w,
                             np.arange(len(kpis))*interval if timestamps is None else timestamps))


def fault_commands(fault_type: str, interface: str, magnitude: float) -> tuple[list[str],list[str]]:
    """Return apply/cleanup commands for a Mininet host interface."""
    cleanup=[f"tc qdisc del dev {interface} root 2>/dev/null || true",f"ip link set dev {interface} up"]
    if fault_type=="cpu_overload": return ["sh -c 'yes >/dev/null & echo $!'"],["pkill -f '^yes$' 2>/dev/null || true"]
    if fault_type=="bandwidth_bottleneck": return [f"tc qdisc replace dev {interface} root tbf rate {max(magnitude,0.1):g}mbit burst 32kbit latency 400ms"],cleanup
    if fault_type=="packet_loss": return [f"tc qdisc replace dev {interface} root netem loss {100*magnitude:g}%"],cleanup
    if fault_type=="link_delay": return [f"tc qdisc replace dev {interface} root netem delay {1000*magnitude:g}ms"],cleanup
    if fault_type=="interface_failure": return [f"ip link set dev {interface} down"],cleanup
    if fault_type=="routing_instability": return [f"sh -c 'for i in 1 2 3; do ip link set dev {interface} down; sleep 1; ip link set dev {interface} up; sleep 1; done' &"],cleanup
    raise ValueError(f"unknown fault type {fault_type}")

