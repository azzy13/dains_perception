"""Experiment 7: relation accuracy given PERFECT detection.
Feeds GT boxes straight into SceneGraphBuilder._depth_relation, so any error is
the relation logic, not the detector.  Compares the object-centric branch (real
GT heading) against the viewer-centric fallback (heading_vec = [0,0])."""
import json, glob, os, sys, collections, numpy as np
R='/isis/home/hasana3/vlmtest/GroundingDINO'
sys.path.insert(0, f'{R}/eval')
from scene_graph import SceneGraphBuilder

WANT={'behind':'behind','in_front_of':'in-front-of'}
b=SceneGraphBuilder(text_prompt='x'); b._aspect=1080.0/1920.0

def node(a,W=1920.0,H=1080.0,heading=(0.0,0.0)):
    x,y,w,h=a['bbox_xywh']
    return {'cx_norm':(x+w/2)/W, 'cy_norm':(y+h/2)/H, 'heading_vec':list(heading)}

rows=[]
for seqdir in sorted(glob.glob(f'{R}/dataset/rmot3/cfg*')):
    gt=json.load(open(os.path.join(seqdir,'gt.json')))
    rel=gt['meta']['relation']; want=WANT[rel]
    byfr=collections.defaultdict(dict)
    for a in gt['annotations']:
        if a['role'] in ('target','confuser','bus'): byfr[a['image_id']][a['role']]=a
    # image-space heading of the bus from its own GT track
    cx={f:(d['bus']['bbox_xywh'][0]+d['bus']['bbox_xywh'][2]/2)/1920.0 for f,d in byfr.items() if 'bus' in d}
    cy={f:(d['bus']['bbox_xywh'][1]+d['bus']['bbox_xywh'][3]/2)/1080.0 for f,d in byfr.items() if 'bus' in d}
    res={'obj':collections.Counter(),'view':collections.Counter()}
    for f,d in sorted(byfr.items()):
        if not all(k in d for k in ('bus','target','confuser')): continue
        if any((d[k].get('visibility') or 0)<0.2 for k in ('bus','target','confuser')): continue
        hv=(0.0,0.0)
        if f-5 in cx: 
            v=(cx[f]-cx[f-5], cy[f]-cy[f-5]); n=np.hypot(*v)
            if n>1e-6: hv=(v[0]/n, v[1]/n)
        for mode,heading in (('obj',hv),('view',(0.0,0.0))):
            bus=node(d['bus'],heading=heading)
            for role in ('target','confuser'):
                n1=node(d[role])
                dx=n1['cx_norm']-bus['cx_norm']; dy=n1['cy_norm']-bus['cy_norm']
                got=b._depth_relation(n1,bus,dx,dy)
                truth = want if role=='target' else ('behind' if want=='in-front-of' else 'in-front-of')
                res[mode]['ok' if got==truth else ('none' if got is None else 'WRONG')]+=1
    rows.append((os.path.basename(seqdir), rel, res))

print(f"{'sequence':16} {'relation':12} | {'object-centric (GT heading)':>30} | {'viewer-centric fallback':>26}")
print(f"{'':16} {'':12} | {'correct':>9} {'wrong':>7} {'none':>6} | {'correct':>9} {'wrong':>7} {'none':>6}")
agg={'obj':collections.Counter(),'view':collections.Counter()}
for name,rel,res in rows:
    o,v=res['obj'],res['view']
    for m in ('obj','view'):
        for k in ('ok','WRONG','none'): agg[m][k]+=res[m][k]
    to=sum(o.values()) or 1; tv=sum(v.values()) or 1
    print(f"{name:16} {rel:12} | {o['ok']/to*100:8.0f}% {o['WRONG']/to*100:6.0f}% {o['none']/to*100:5.0f}% "
          f"| {v['ok']/tv*100:8.0f}% {v['WRONG']/tv*100:6.0f}% {v['none']/tv*100:5.0f}%")
for m,label in (('obj','OBJECT-CENTRIC'),('view','VIEWER-CENTRIC')):
    t=sum(agg[m].values()) or 1
    print(f"\n{label:16} correct={agg[m]['ok']/t*100:.1f}%  wrong={agg[m]['WRONG']/t*100:.1f}%  none={agg[m]['none']/t*100:.1f}%")
