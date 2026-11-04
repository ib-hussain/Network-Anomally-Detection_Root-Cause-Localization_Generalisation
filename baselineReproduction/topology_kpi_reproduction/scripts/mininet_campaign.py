#!/usr/bin/env python3
"""Run a 12/24/48-node Mininet KPI/fault campaign or emit a dry-run plan.

Actual execution requires Linux, root, Mininet, iproute2, ping and /proc. KPI
definitions are operational proxies documented in the generated metadata.
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
import numpy as np
from topokpi.testbed import default_schedule, directed_forwarding, fault_commands, save_campaign, topology_edges, write_campaign_plan


def number(text, pattern, default=0.):
    match=re.search(pattern,text,re.M); return float(match.group(1)) if match else default


def collect(host, interface, peer_ip, previous):
    cpu=number(host.cmd(f"ps --ppid {host.pid} -o %cpu= 2>/dev/null | awk '{{s+=$1}} END{{print s+0}}'"),r"([\d.]+)")
    memory=number(host.cmd(f"ps --ppid {host.pid} -o rss= 2>/dev/null | awk '{{s+=$1}} END{{print s+0}}'"),r"([\d.]+)")
    rx=number(host.cmd(f"cat /sys/class/net/{interface}/statistics/rx_bytes 2>/dev/null"),r"(\d+)")
    tx=number(host.cmd(f"cat /sys/class/net/{interface}/statistics/tx_bytes 2>/dev/null"),r"(\d+)")
    errors=number(host.cmd(f"cat /sys/class/net/{interface}/statistics/rx_errors 2>/dev/null"),r"(\d+)")+number(host.cmd(f"cat /sys/class/net/{interface}/statistics/tx_errors 2>/dev/null"),r"(\d+)")
    ping=host.cmd(f"ping -c 3 -W 1 {peer_ip} 2>/dev/null"); rtt=number(ping,r"= [\d.]+/([\d.]+)/",1000.); transmitted=number(ping,r"(\d+) packets transmitted",3); received=number(ping,r"transmitted, (\d+) received",0); loss=1-received/max(transmitted,1)
    queue=number(host.cmd(f"tc -s qdisc show dev {interface}"),r"backlog\s+\S+\s+(\d+)p",0)
    if previous is None: inbound=outbound=0.
    else: inbound=max(rx-previous[0],0); outbound=max(tx-previous[1],0)
    return np.array([cpu,memory,inbound,outbound,rtt,loss,queue,errors],np.float32),(rx,tx)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--nodes",type=int,choices=[12,24,48],required=True); p.add_argument("--steps",type=int,default=1200); p.add_argument("--interval",type=int,default=30); p.add_argument("--output",required=True); p.add_argument("--seed",type=int,default=17); p.add_argument("--dry-run",action="store_true"); args=p.parse_args()
    events=default_schedule(args.nodes,args.steps,20,args.seed); output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); write_campaign_plan(output.with_suffix(".campaign.json"),args.nodes,args.steps,args.interval,events)
    if args.dry_run: print(output.with_suffix(".campaign.json")); return
    try:
        from mininet.net import Mininet
        from mininet.node import OVSController
    except ImportError as exc: raise SystemExit("Mininet is unavailable; use --dry-run or install Mininet on Linux") from exc
    net=Mininet(controller=OVSController); net.addController("c0"); hosts=[net.addHost(f"h{i+1}") for i in range(args.nodes)]
    links={}
    for i,j in topology_edges(args.nodes): links[(i,j)]=net.addLink(hosts[i],hosts[j])
    net.start(); values=np.zeros((args.steps,args.nodes,8),np.float32); previous=[None]*args.nodes; active={}
    dominant={"cpu_overload":[0],"bandwidth_bottleneck":[2,3,6],"packet_loss":[5],"link_delay":[4],"interface_failure":[7],"routing_instability":[2,3,4]}
    event_metadata=[]
    for event in events:
        w=directed_forwarding(args.nodes,event.root); item=dict(event.__dict__); item["dominant_kpis"]=dominant[event.fault_type]; item["true_path_edges"]=np.argwhere(w>0).tolist(); event_metadata.append(item)
    metadata={"kpis":["child_process_cpu_percent","child_process_rss_kb","rx_bytes_per_interval","tx_bytes_per_interval","rtt_ms","packet_loss_fraction","queue_packets","interface_errors"],"events":event_metadata}
    try:
        for step in range(args.steps):
            for event in events:
                root=event.root; incident=next(link for edge,link in links.items() if root in edge); interface=incident.intf1.name if incident.intf1.node==hosts[root] else incident.intf2.name
                if step==event.start_step:
                    apply,cleanup=fault_commands(event.fault_type,interface,event.magnitude); active[event.event_id]=(hosts[root],cleanup)
                    for command in apply: hosts[root].cmd(command)
                if step==event.stop_step and event.event_id in active:
                    host,cleanup=active.pop(event.event_id)
                    for command in cleanup: host.cmd(command)
            start=time.monotonic()
            for i,host in enumerate(hosts):
                edge,incident=next((edge,link) for edge,link in links.items() if i in edge); interface=incident.intf1.name if incident.intf1.node==host else incident.intf2.name; peer=hosts[edge[1] if edge[0]==i else edge[0]]
                values[step,i],previous[i]=collect(host,interface,peer.IP(),previous[i])
            time.sleep(max(0,args.interval-(time.monotonic()-start)))
    finally:
        for host,cleanup in active.values():
            for command in cleanup: host.cmd(command)
        net.stop()
    save_campaign(output,values,args.nodes,args.interval,events); output.with_suffix(".metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8"); print(output)


if __name__=="__main__": main()
