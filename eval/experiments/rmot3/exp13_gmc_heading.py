"""Experiment 13: recover the anchor's true heading using the GMC affine.

exp9/exp10 showed the anchor's raw image displacement agrees with its true
direction of travel only 50% of the time (63% after subtracting the frame's
median displacement).  We now have a per-frame camera affine from tracker/gmc.py,
which is a far better ego-motion estimate than a median.

To compare positions across a k-frame window they must be in the same
coordinates, so the older position is warped forward through the composed
affines; the residual displacement is then the object's own motion.
"""
import json, glob, os, math, collections, sys
import numpy as np, cv2
R='/isis/home/hasana3/vlmtest/GroundingDINO'
sys.path.insert(0, R)
from tracker.gmc import GMC

K = 5   # frame gap for the heading estimate, matching _motion_attrs' horizon

def ctr(a, W=1920.0, H=1080.0):
    x, y, w, h = a['bbox_xywh']
    return np.array([(x + w / 2), (y + h / 2)])   # pixels

def compose(Hs):
    """Compose a list of 2x3 affines (oldest first) into one 2x3."""
    M = np.eye(3)
    for h in Hs:
        M3 = np.vstack([h, [0, 0, 1]])
        M = M3 @ M
    return M[:2, :]

def warp(pt, H):
    return H[:2, :2] @ pt + H[:2, 2]

rows = []
for d in sorted(glob.glob(f'{R}/dataset/rmot3/cfg*')):
    seq = os.path.basename(d)
    gt = json.load(open(os.path.join(d, 'gt.json'))); rel = gt['meta']['relation']
    byfr = collections.defaultdict(dict)
    for a in gt['annotations']:
        if a['role'] in ('target', 'confuser', 'bus'):
            byfr[a['image_id']][a['role']] = a

    g = GMC(downscale=2)
    Hs = {}                       # frame f -> affine mapping f-1 into f
    for f in range(300):
        img = cv2.imread(os.path.join(d, 'images', f'{f:06d}.png'))
        if img is None: break
        Hs[f] = g.apply(img)

    raw, comp = [], []
    for f in sorted(byfr):
        v = byfr[f]
        if not all(k in v for k in ('bus', 'target', 'confuser')): continue
        if any((v[k].get('visibility') or 0) < 0.2 for k in ('bus', 'target', 'confuser')): continue
        if f - K not in byfr or 'bus' not in byfr[f - K]: continue
        if any(x not in Hs for x in range(f - K + 1, f + 1)): continue

        now = ctr(v['bus']); old = ctr(byfr[f - K]['bus'])
        ahead = (ctr(v['confuser']) - ctr(v['target'])) if rel == 'behind' \
                else (ctr(v['target']) - ctr(v['confuser']))
        if np.linalg.norm(ahead) < 1e-6: continue

        M = compose([Hs[x] for x in range(f - K + 1, f + 1)])
        for tag, vec in (('raw', now - old), ('comp', now - warp(old, M))):
            if np.linalg.norm(vec) < 1e-6: continue
            c = float(np.dot(vec, ahead) / (np.linalg.norm(vec) * np.linalg.norm(ahead)))
            ang = math.degrees(math.acos(max(-1, min(1, c))))
            (raw if tag == 'raw' else comp).append(ang)
    if raw and comp:
        rows.append((seq, np.array(raw), np.array(comp)))
        print(f"  {seq:16} raw aligned={(rows[-1][1]<60).mean()*100:3.0f}%   "
              f"gmc aligned={(rows[-1][2]<60).mean()*100:3.0f}%")

allraw = np.concatenate([r[1] for r in rows]); allc = np.concatenate([r[2] for r in rows])
print(f"\n{'':16} {'aligned<60':>11} {'opposed>120':>12} {'median':>8}")
print(f"{'raw image':16} {(allraw<60).mean()*100:10.0f}% {(allraw>120).mean()*100:11.0f}% {np.median(allraw):7.0f}°")
print(f"{'GMC-compensated':16} {(allc<60).mean()*100:10.0f}% {(allc>120).mean()*100:11.0f}% {np.median(allc):7.0f}°")
