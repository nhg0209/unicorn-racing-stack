#!/usr/bin/env python3
"""What obs_margin buys and what it costs, for the closed-track QP.

The margin chain, line-centre to obstacle EDGE:

    SM free-check accept     gb_ego_width_m/2 + lateral_width_static_gb_m          = 0.25
    reactive CLEAR trigger   width_car/2 + clear_margin_m                          = 0.25
    reactive IDLE entry      width_car/2 + clear_margin_m + clear_hyst_m           = 0.28  <- binding

A box the global line CLAIMS must be published far enough out that the reactive layer will go
idle about it. Under 0.28 it avoids the box again on top of the global line -- the double
avoidance this subsystem exists to remove. closed_qp delivers obs_margin + w_veh/2; the hump
delivers its own obs_margin (0.35), which is why it never had to think about this.

Measurement only -- nothing here changes a default.

  ~/miniforge3/envs/unicorn/bin/python3 planner/gb_optimizer/scripts/sweep_obs_margin.py
"""
import sys  # noqa: E402
sys.path.insert(0,"planner/gb_optimizer"); sys.path.insert(0,"planner/gb_optimizer/scripts")
import numpy as np
from bench_closed_reopt import load_ifac, corridor_from_map, box
import compare_reopt as CMP
from gb_optimizer import closed_reopt as C
from gb_optimizer.closed_reopt import ReoptParams
ref=load_ifac(); cor=corridor_from_map(ref); n=len(ref)
kc=float(np.max(np.abs(C.menger_closed(ref[:,:2]))))
IDLE=0.28   # width_car/2 + clear_margin_m + clear_hyst_m, static_avoidance_params.yaml
cases=CMP.build_cases(n)
print("obs_margin | delivers | covered/38 | clearance min/med | headroom vs 0.28 | hold 3.0/6.0/8.5 | C5a | C5b max")
for om in (0.15,0.16,0.18,0.20):
    p=ReoptParams(obs_margin=om); need=om+0.5*p.w_veh
    cov=0; tot=0; cl=[]; a_ok=True; bmax=-9
    for name,st in cases:
        _l,d,r=C.reoptimize_closed(ref,[box(ref,i) for i in st],cor,p)
        tot+=len(st); cov+=len(st)-len(r.infeasible); cl+=list(r.clearances)
        if r.ok:
            a_ok &= r.peak_kappa_nodes<=kc+1e-9; bmax=max(bmax,r.peak_kappa-kc)
    holds=[]
    for dst in (30,60,85):
        _l,_d,r=C.reoptimize_closed(ref,[box(ref,273),box(ref,(273+dst)%n)],cor,p)
        holds.append(f"{r.hold:.3f}" if r.hold==r.hold else " none")
    mn=min(cl) if cl else float('nan'); md=float(np.median(cl)) if cl else float('nan')
    print(f"   {om:.2f}    |  {need:.3f}   |   {cov:2d}/{tot}    | {mn:.3f} / {md:.3f}     |"
          f" {mn-IDLE:+.3f} min      | {'/'.join(holds)} | {'ok' if a_ok else 'NO'} | {bmax:+.4f}")
