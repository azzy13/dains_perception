"""Experiment 6: measure crop chroma over the car's pixels, not the whole box.
L* stays a median (correct for the achromatic peer ranking); only chroma/hue
are taken from the most-chromatic pixels.  Monkeypatched; files untouched."""
import json, cv2, numpy as np, collections, glob, os, sys
R='/isis/home/hasana3/vlmtest/GroundingDINO'
sys.path.insert(0, f'{R}/eval')
import color_classifier as cc

_orig = cc.lab_stats

def lab_stats_chroma(crop_rgb, chroma_pct=85.0):
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).reshape(-1,3).astype(np.float32)
    med = np.median(lab, axis=0)
    L = float(med[0])*100.0/255.0            # lightness: unchanged, median
    ch = np.hypot(lab[:,1]-128.0, lab[:,2]-128.0)
    if lab.shape[0] >= 16:
        sel = lab[ch >= np.percentile(ch, chroma_pct)]
    else:
        sel = lab
    a = float(np.median(sel[:,1]))-128.0; b = float(np.median(sel[:,2]))-128.0
    return L, float(np.hypot(a,b)), float(np.degrees(np.arctan2(b,a))%360.0)

def run(tag):
    sc=collections.defaultdict(list); chstat=collections.defaultdict(list)
    for seqdir in sorted(glob.glob(f'{R}/dataset/rmot3/cfg*')):
        gt=json.load(open(os.path.join(seqdir,'gt.json')))
        byfr=collections.defaultdict(list)
        for a in gt['annotations']: byfr[a['image_id']].append(a)
        for fr in range(0,300,20):
            img=cv2.imread(os.path.join(seqdir,'images',f'{fr:06d}.png'))
            if img is None: continue
            crops,roles=[],[]
            for a in byfr.get(fr,[]):
                if (a.get('visibility') or 0)<0.3: continue
                x1,y1,x2,y2=[int(v) for v in a['bbox_xyxy']]
                c=img[max(0,y1):y2, max(0,x1):x2]
                if c.size==0 or min(c.shape[:2])<10: continue
                crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)); roles.append(a['role'])
            if len(crops)<2: continue
            for c,r in zip(crops,roles): chstat[r].append(cc.lab_stats(c)[1])
            for tgt in ('red','white'):
                for s,r in zip(cc.peer_relative_scores(crops,tgt), roles):
                    sc[(tgt,r)].append(s)
    print(f"\n=== {tag} ===")
    print(f"  median crop chroma by role (CHROMA_MIN={cc.CHROMA_MIN}):")
    for r in ('target','confuser','bus','distractor'):
        v=chstat[r]
        print(f"     {r:11} {np.median(v):6.1f}   (frac >= CHROMA_MIN: {np.mean(np.array(v)>=cc.CHROMA_MIN)*100:5.1f}%)")
    for tgt in ('red','white'):
        sed=np.mean(sc[(tgt,'target')]+sc[(tgt,'confuser')]); bus=np.mean(sc[(tgt,'bus')])
        dis=np.mean(sc[(tgt,'distractor')])
        print(f"  target={tgt:6} sedans={sed:.3f} bus={bus:.3f} distractor={dis:.3f}  margin(sedan-bus)={sed-bus:+.3f}")

run("BEFORE — whole-crop median chroma")
cc.lab_stats = lab_stats_chroma
run("AFTER  — chroma from the car's own pixels")
