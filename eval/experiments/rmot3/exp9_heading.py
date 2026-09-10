"""Experiment 9: is image-space heading a valid proxy for direction of travel
under a moving (drone) camera?

Reference "ahead" direction in image space is taken from GT: in a `behind` clip
the confuser is ahead of the bus and the target trails it, so
(confuser - target) points along travel.  Compare that against the bus's own
image-space displacement, which is what _motion_attrs recovers.
"""
import json, glob, os, math, collections, numpy as np
R='/isis/home/hasana3/vlmtest/GroundingDINO'
def ctr(a,W=1920.0,H=1080.0):
    x,y,w,h=a['bbox_xywh']; return np.array([(x+w/2)/W,(y+h/2)/H])

print(f"{'sequence':16} {'rel':12} {'n':>5} {'median angle':>13} {'aligned<60d':>12} {'opposed>120d':>13}")
allang=[]
for d in sorted(glob.glob(f'{R}/dataset/rmot3/cfg*')):
    gt=json.load(open(os.path.join(d,'gt.json'))); rel=gt['meta']['relation']
    byfr=collections.defaultdict(dict)
    for a in gt['annotations']:
        if a['role'] in ('target','confuser','bus'): byfr[a['image_id']][a['role']]=a
    C={f:ctr(v['bus']) for f,v in byfr.items() if 'bus' in v}
    angs=[]
    for f,v in sorted(byfr.items()):
        if not all(k in v for k in ('bus','target','confuser')): continue
        if any((v[k].get('visibility') or 0)<0.2 for k in ('bus','target','confuser')): continue
        if f-5 not in C: continue
        hv=C[f]-C[f-5]                       # bus image motion (what the tracker sees)
        if np.linalg.norm(hv)<1e-6: continue
        # GT "ahead" in image: from the trailing car toward the leading one
        if rel=='behind': ahead=ctr(v['confuser'])-ctr(v['target'])
        else:             ahead=ctr(v['target'])-ctr(v['confuser'])
        if np.linalg.norm(ahead)<1e-6: continue
        cos=np.dot(hv,ahead)/(np.linalg.norm(hv)*np.linalg.norm(ahead))
        angs.append(math.degrees(math.acos(max(-1,min(1,cos)))))
    if not angs: continue
    a=np.array(angs); allang+=angs
    print(f"{os.path.basename(d):16} {rel:12} {len(a):5} {np.median(a):12.0f}° "
          f"{(a<60).mean()*100:11.0f}% {(a>120).mean()*100:12.0f}%")
a=np.array(allang)
print(f"\nALL  n={len(a)}  median={np.median(a):.0f}°  aligned(<60°)={(a<60).mean()*100:.0f}%  "
      f"opposed(>120°)={(a>120).mean()*100:.0f}%  orthogonal(60-120°)={((a>=60)&(a<=120)).mean()*100:.0f}%")
