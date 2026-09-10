"""Experiment 5: does a chroma floor inside color_classifier.patch_votes give
one scorer that handles BOTH chromatic (rmot3 'red') and achromatic
(Refer-KITTI 'black'/'silver'/'light') targets?  Monkeypatched, file untouched."""
import json, cv2, numpy as np, collections, glob, os, sys
R='/isis/home/hasana3/vlmtest/GroundingDINO'
sys.path.insert(0, f'{R}/eval')
import color_classifier as cc

_orig_patch_votes = cc.patch_votes

def patch_votes_chroma(crop_rgb, grid_size=4, chroma_pct=85.0):
    h,w = crop_rgb.shape[:2]
    if h==0 or w==0: return "unknown", {}
    if h<grid_size or w<grid_size: grid_size=max(1,min(h,w,2))
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ch  = np.hypot(lab[:,:,1]-128.0, lab[:,:,2]-128.0)
    floor = float(np.percentile(ch, chroma_pct))
    ph,pw = max(1,h//grid_size), max(1,w//grid_size)
    votes={}
    for i in range(grid_size):
        for j in range(grid_size):
            y0,y1 = i*ph, min((i+1)*ph,h); x0,x1 = j*pw, min((j+1)*pw,w)
            patch = crop_rgb[y0:y1, x0:x1]
            if patch.size==0: continue
            m = ch[y0:y1, x0:x1] >= floor
            if m.sum()>=4:
                sel = patch.reshape(-1,3)[m.reshape(-1)]
                label = cc.classify_patch(*cc.lab_stats(sel.reshape(1,-1,3)))
            else:
                label = cc.classify_patch(*cc.lab_stats(patch))
            votes[label]=votes.get(label,0)+1
    if not votes: return "unknown", {}
    return max(votes.items(), key=lambda kv: kv[1])[0], votes

def run(tag):
    sc=collections.defaultdict(list)
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
            for tgt in ('red','white'):
                for s,r in zip(cc.peer_relative_scores(crops,tgt), roles):
                    sc[(tgt,r)].append(s)
    print(f"\n=== {tag} ===")
    for tgt in ('red','white'):
        sed=np.mean(sc[(tgt,'target')]+sc[(tgt,'confuser')]); bus=np.mean(sc[(tgt,'bus')])
        dis=np.mean(sc[(tgt,'distractor')])
        print(f"  target={tgt:6} sedans={sed:.3f} bus={bus:.3f} distractor={dis:.3f}  "
              f"sedan-vs-bus margin={sed-bus:+.3f}")

run("BEFORE (original patch_votes)")
cc.patch_votes = patch_votes_chroma
run("AFTER  (chroma-floored patch_votes)")
