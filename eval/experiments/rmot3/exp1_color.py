"""Experiment 1: colour sampling methods across all rmot3 sequences.
Read-only: reads dataset + repo modules, writes nothing outside scratchpad."""
import json, cv2, numpy as np, collections, glob, os, sys
sys.path.insert(0, '/isis/home/hasana3/vlmtest/GroundingDINO/eval')
from scene_graph import _dominant_color_from_crop, _classify_lab_patch

ROOT='/isis/home/hasana3/vlmtest/GroundingDINO/dataset/rmot3'

def gt_color_name(rgbstr):
    r,g,b=[int(v) for v in rgbstr.split(',')]
    mx,mn=max(r,g,b),min(r,g,b)
    if mx-mn < 30:
        return 'white' if mx>180 else ('black' if mx<60 else 'gray')
    if r==mx: return 'red' if g<=b+40 else 'orange'
    if g==mx: return 'green'
    return 'blue'

def m_current(crop):
    return _dominant_color_from_crop(crop)[0]

def m_chroma(crop, pct):
    lab=cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1,3).astype(float)
    ch=np.hypot(lab[:,1]-128, lab[:,2]-128)
    thr=np.percentile(ch,pct)
    sel=lab[ch>=thr]
    if len(sel)==0: return 'none'
    return _classify_lab_patch(*sel.mean(axis=0))

PCTS=[50,60,70,75,80,85,90,95]
methods={'current':m_current}
for p in PCTS: methods[f'chroma_p{p}']=lambda c,p=p: m_chroma(c,p)

# confusion: per method, per GT-truth-colour, what did it say
stat={k:collections.defaultdict(collections.Counter) for k in methods}
nseq=0
for seqdir in sorted(glob.glob(os.path.join(ROOT,'cfg*'))):
    gtp=os.path.join(seqdir,'gt.json')
    if not os.path.isfile(gtp): continue
    gt=json.load(open(gtp)); nseq+=1
    byfr=collections.defaultdict(list)
    for a in gt['annotations']: byfr[a['image_id']].append(a)
    for fr in range(0,300,15):
        ip=os.path.join(seqdir,'images',f'{fr:06d}.png')
        img=cv2.imread(ip)
        if img is None: continue
        for a in byfr.get(fr,[]):
            if (a.get('visibility') or 0)<0.3: continue
            x1,y1,x2,y2=[int(v) for v in a['bbox_xyxy']]
            crop=img[max(0,y1):y2, max(0,x1):x2]
            if crop.size==0 or min(crop.shape[:2])<10: continue
            truth=gt_color_name(a['color'])
            for k,fn in methods.items():
                stat[k][truth][fn(crop)]+=1

print(f"sequences: {nseq}\n")
truths=sorted({t for k in stat for t in stat[k]})
print(f"{'method':14} " + " ".join(f"{t+'→'+t:>14}" for t in truths) + f" {'OVERALL':>9}")
for k in methods:
    cells=[]; tot=0; cor=0
    for t in truths:
        c=stat[k][t]; n=sum(c.values()); hit=c.get(t,0)
        tot+=n; cor+=hit
        cells.append(f"{hit:5}/{n:<5}({hit/n*100 if n else 0:3.0f}%)".rjust(14))
    print(f"{k:14} " + " ".join(cells) + f" {cor/tot*100:8.1f}%")

# The operational question: red-sedan recall vs bus false-positive
print(f"\n{'method':14} {'red→red':>10} {'white(bus)→red':>16}")
for k in methods:
    red=stat[k]['red']; white=stat[k]['white']
    rn=sum(red.values()); wn=sum(white.values())
    print(f"{k:14} {red.get('red',0)/rn*100 if rn else 0:9.1f}% {white.get('red',0)/wn*100 if wn else 0:15.1f}%")
