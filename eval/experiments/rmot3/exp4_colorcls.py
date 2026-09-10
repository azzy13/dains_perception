"""Experiment 4: color_classifier.peer_relative_scores vs the scene_graph LAB bins.
Scored as a discrimination task, which is how a soft scorer is actually used."""
import json, cv2, numpy as np, collections, glob, os, sys
R='/isis/home/hasana3/vlmtest/GroundingDINO'
sys.path.insert(0, f'{R}/eval')
from color_classifier import peer_relative_scores
from scene_graph import _dominant_color_from_crop

def crops_for_frame(seqdir, fr, byfr):
    img=cv2.imread(os.path.join(seqdir,'images',f'{fr:06d}.png'))
    if img is None: return [],[]
    out,roles=[],[]
    for a in byfr.get(fr,[]):
        if (a.get('visibility') or 0)<0.3: continue
        x1,y1,x2,y2=[int(v) for v in a['bbox_xyxy']]
        c=img[max(0,y1):y2, max(0,x1):x2]
        if c.size==0 or min(c.shape[:2])<10: continue
        out.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)); roles.append(a['role'])
    return out,roles

sc=collections.defaultdict(list)   # (target_colour, role) -> scores
lab=collections.defaultdict(collections.Counter)
nfr=0
for seqdir in sorted(glob.glob(f'{R}/dataset/rmot3/cfg*')):
    gt=json.load(open(os.path.join(seqdir,'gt.json')))
    byfr=collections.defaultdict(list)
    for a in gt['annotations']: byfr[a['image_id']].append(a)
    for fr in range(0,300,20):
        crops,roles=crops_for_frame(seqdir,fr,byfr)
        if len(crops)<2: continue
        nfr+=1
        for tgt in ('red','white'):
            for s,r in zip(peer_relative_scores(crops,tgt), roles):
                sc[(tgt,r)].append(s)
        for c,r in zip(crops,roles):
            lab[r][_dominant_color_from_crop(cv2.cvtColor(c,cv2.COLOR_RGB2BGR))[0]]+=1

print(f"frames sampled: {nfr}\n")
roles=('target','confuser','bus','distractor')
print("color_classifier.peer_relative_scores  — mean score by true role")
print(f"{'target colour':16}" + "".join(f"{r:>12}" for r in roles))
for tgt in ('red','white'):
    row=f"{tgt:16}"
    for r in roles:
        v=sc[(tgt,r)]
        row+=f"{np.mean(v):12.3f}" if v else f"{'-':>12}"
    print(row)

print("\nseparation (want red: sedans >> bus ; white: bus >> sedans)")
for tgt in ('red','white'):
    sed=np.mean(sc[(tgt,'target')]+sc[(tgt,'confuser')]); bus=np.mean(sc[(tgt,'bus')])
    print(f"  target={tgt:6} sedans={sed:.3f}  bus={bus:.3f}  margin={sed-bus:+.3f}")

print("\nscene_graph LAB (patched) — hard label distribution")
for r in roles:
    d=dict(lab[r]); n=sum(d.values())
    top=sorted(d.items(), key=lambda x:-x[1])[:4]
    print(f"  {r:11} n={n:4}  {top}")
