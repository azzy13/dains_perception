"""Experiment 8: is the viewer-centric branch simply inverted, and is the
object-centric relation recoverable at all?  GT boxes + GT world positions."""
import json, glob, os, sys, collections, numpy as np, math
R='/isis/home/hasana3/vlmtest/GroundingDINO'
sys.path.insert(0,f'{R}/eval')
ASPECT=1080.0/1920.0; TH=0.02
WANT={'behind':'behind','in_front_of':'in-front-of'}

def viewer(dy, flip=False):
    d=dy*ASPECT
    if flip: d=-d
    if d> TH: return 'in-front-of'
    if d<-TH: return 'behind'
    return None

def objc(dx,dy,hx,hy):
    hy_w=hy*ASPECT; s=math.hypot(hx,hy_w)
    if s==0: return None
    along=(dx*hx+(dy*ASPECT)*hy_w)/s
    if along> TH: return 'in-front-of'
    if along<-TH: return 'behind'
    return None

def world_rel(t,bus):
    """Truth from 3D: project (target-bus) onto the bus's own yaw direction."""
    yaw=math.radians(bus['yaw']); hx,hy=math.cos(yaw),math.sin(yaw)
    d=(t['x']-bus['x'])*hx+(t['y']-bus['y'])*hy
    if d> 0.5: return 'in-front-of'
    if d<-0.5: return 'behind'
    return None

agg=collections.Counter()
for seqdir in sorted(glob.glob(f'{R}/dataset/rmot3/cfg*')):
    gt=json.load(open(os.path.join(seqdir,'gt.json')))
    want=WANT[gt['meta']['relation']]
    byfr=collections.defaultdict(dict)
    for a in gt['annotations']:
        if a['role'] in ('target','confuser','bus'): byfr[a['image_id']][a['role']]=a
    C={f:((d['bus']['bbox_xywh'][0]+d['bus']['bbox_xywh'][2]/2)/1920.0,
          (d['bus']['bbox_xywh'][1]+d['bus']['bbox_xywh'][3]/2)/1080.0)
       for f,d in byfr.items() if 'bus' in d}
    for f,d in sorted(byfr.items()):
        if not all(k in d for k in ('bus','target','confuser')): continue
        if any((d[k].get('visibility') or 0)<0.2 for k in ('bus','target','confuser')): continue
        hv=(0.0,0.0)
        if f-5 in C:
            v=(C[f][0]-C[f-5][0], C[f][1]-C[f-5][1]); n=math.hypot(*v)
            if n>1e-6: hv=(v[0]/n, v[1]/n)
        bx,by=C[f]
        for role in ('target','confuser'):
            a=d[role]; x,y,w,h=a['bbox_xywh']
            n1=((x+w/2)/1920.0,(y+h/2)/1080.0)
            dx,dy=n1[0]-bx, n1[1]-by
            truth = want if role=='target' else ('behind' if want=='in-front-of' else 'in-front-of')
            wr = world_rel(a['world_position'], d['bus']['world_position'])
            agg['world_ok' if wr==truth else ('world_none' if wr is None else 'world_WRONG')]+=1
            for tag,got in (('view',viewer(dy)),('view_flip',viewer(dy,True)),
                            ('obj',objc(dx,dy,*hv))):
                agg[f'{tag}_ok' if got==truth else (f'{tag}_none' if got is None else f'{tag}_WRONG')]+=1
tot=agg['world_ok']+agg['world_WRONG']+agg['world_none']
print(f"pairs evaluated: {tot}\n")
for tag,label in (('world','GT 3D world yaw projection (upper bound)'),
                  ('view','viewer-centric  (current)'),
                  ('view_flip','viewer-centric  (SIGN FLIPPED)'),
                  ('obj','object-centric  (image heading)')):
    ok,wr,nn=agg[f'{tag}_ok'],agg[f'{tag}_WRONG'],agg[f'{tag}_none']
    print(f"{label:42} correct={ok/tot*100:5.1f}%  wrong={wr/tot*100:5.1f}%  abstain={nn/tot*100:5.1f}%")
