"""Experiment 10: ego-motion compensation for heading recovery.

Under a moving camera an object's image displacement is its own motion plus the
camera's.  The camera term is common to every object in the frame, so the median
displacement over all detections estimates it and subtracting recovers true
motion.  Compared against the same GT "ahead" reference as exp9.
"""
import json, glob, os, math, collections, numpy as np
R='/isis/home/hasana3/vlmtest/GroundingDINO'
def ctr(a,W=1920.0,H=1080.0):
    x,y,w,h=a['bbox_xywh']; return np.array([(x+w/2)/W,(y+h/2)/H])

def angles(compensate):
    out=collections.defaultdict(list)
    for d in sorted(glob.glob(f'{R}/dataset/rmot3/cfg*')):
        gt=json.load(open(os.path.join(d,'gt.json'))); rel=gt['meta']['relation']
        byfr=collections.defaultdict(dict); allfr=collections.defaultdict(dict)
        for a in gt['annotations']:
            allfr[a['image_id']][a['gt_id']]=a
            if a['role'] in ('target','confuser','bus'): byfr[a['image_id']][a['role']]=a
        for f,v in sorted(byfr.items()):
            if not all(k in v for k in ('bus','target','confuser')): continue
            if any((v[k].get('visibility') or 0)<0.2 for k in ('bus','target','confuser')): continue
            if f-5 not in byfr or 'bus' not in byfr[f-5]: continue
            hv=ctr(v['bus'])-ctr(byfr[f-5]['bus'])
            if compensate:
                # common motion across every actor visible in both frames
                deltas=[ctr(a)-ctr(allfr[f-5][gid]) for gid,a in allfr[f].items()
                        if gid in allfr[f-5] and (a.get('visibility') or 0)>=0.2]
                if len(deltas)>=3:
                    hv=hv-np.median(np.array(deltas),axis=0)
            if np.linalg.norm(hv)<1e-6: continue
            ahead=(ctr(v['confuser'])-ctr(v['target'])) if rel=='behind' else (ctr(v['target'])-ctr(v['confuser']))
            if np.linalg.norm(ahead)<1e-6: continue
            c=np.dot(hv,ahead)/(np.linalg.norm(hv)*np.linalg.norm(ahead))
            out[os.path.basename(d)].append(math.degrees(math.acos(max(-1,min(1,c)))))
    return out

for tag,comp in (("RAW image motion",False),("EGO-COMPENSATED",True)):
    o=angles(comp); a=np.array([x for v in o.values() for x in v])
    print(f"\n=== {tag} ===  n={len(a)}  median={np.median(a):.0f}°  "
          f"aligned(<60°)={(a<60).mean()*100:.0f}%  opposed(>120°)={(a>120).mean()*100:.0f}%")
    for k in sorted(o):
        v=np.array(o[k])
        print(f"   {k:16} median={np.median(v):4.0f}°  aligned={(v<60).mean()*100:3.0f}%  opposed={(v>120).mean()*100:3.0f}%")
