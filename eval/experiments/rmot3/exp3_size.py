"""Experiment 3: is the target physically detectable in each sequence?"""
import json, glob, os, collections, numpy as np
R='/isis/home/hasana3/vlmtest/GroundingDINO/dataset/rmot3'
print(f"{'sequence':16} {'tgt frames':>10} {'med area px':>12} {'med w':>6} {'med h':>6} {'med vis':>8} {'<32px wide':>11} {'vis<0.3':>8}")
for d in sorted(glob.glob(os.path.join(R,'cfg*'))):
    gt=json.load(open(os.path.join(d,'gt.json')))
    tg=[a for a in gt['annotations'] if a['role']=='target']
    if not tg: continue
    ws=np.array([a['bbox_xywh'][2] for a in tg]); hs=np.array([a['bbox_xywh'][3] for a in tg])
    vis=np.array([(a.get('visibility') or 0) for a in tg])
    ar=ws*hs
    print(f"{os.path.basename(d):16} {len(tg):10} {np.median(ar):12.0f} {np.median(ws):6.0f} {np.median(hs):6.0f} "
          f"{np.median(vis):8.2f} {(ws<32).mean()*100:10.0f}% {(vis<0.3).mean()*100:7.0f}%")
