"""Experiment 2: why does the anchor go missing? Sweep text_threshold.
Read-only w.r.t. the repo; loads the detector and reports."""
import json, cv2, collections, sys, os
sys.path.insert(0,'/isis/home/hasana3/vlmtest/GroundingDINO/eval')
from PIL import Image
from groundingdino.util.inference import load_model, predict
from worker_clean import build_normalize_transform
from query_grounding import assign_detection_roles
import query_parser as qp

R='/isis/home/hasana3/vlmtest/GroundingDINO'
SEQ=sys.argv[1] if len(sys.argv)>1 else 'cfg04_behind'
PROMPT=json.load(open(f'{R}/dataset/rmot3/{SEQ}/gt.json'))['meta']['prompt']
q=qp.parse(PROMPT)
gt=json.load(open(f'{R}/dataset/rmot3/{SEQ}/gt.json'))
byfr=collections.defaultdict(list)
for a in gt['annotations']: byfr[a['image_id']].append(a)
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0

m=load_model(f'{R}/groundingdino/config/GroundingDINO_SwinB_cfg.py',
             f'{R}/weights/groundingdino_swinb_cogcoor.pth').to('cuda:0').eval()
t=build_normalize_transform()
FRAMES=list(range(0,300,20))
print(f"seq={SEQ}  prompt={PROMPT!r}\n")
print(f"{'box':>5} {'text':>5} | {'dets':>5} {'empty_phrase':>13} | {'bus found':>10} {'bus→anchor':>11} {'bus→TARGET(bug)':>16} | {'tgt found':>10}")
for box_t in (0.40, 0.25):
    for text_t in (0.80, 0.60, 0.40, 0.25, 0.15):
        nd=ne=busf=busA=busT=tgtf=nbus=ntgt=0
        for fr in FRAMES:
            img=cv2.imread(f'{R}/dataset/rmot3/{SEQ}/images/{fr:06d}.png')
            if img is None: continue
            H,W=img.shape[:2]
            ten=t(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
            b,l,ph=predict(model=m,image=ten,caption='red sedan . bus .',
                           box_threshold=box_t,text_threshold=text_t,device='cuda:0')
            roles=assign_detection_roles(ph,q)
            gts=[a for a in byfr.get(fr,[]) if (a.get('visibility') or 0)>=0.2]
            nbus+=sum(1 for a in gts if a['role']=='bus')
            ntgt+=sum(1 for a in gts if a['role']=='target')
            nd+=len(ph); ne+=sum(1 for p in ph if not p.strip())
            for box,p,r in zip(b,ph,roles):
                cx,cy,w,h=box.tolist()
                bx=[(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H]
                best,bi=None,0.0
                for a in gts:
                    v=iou(bx,a['bbox_xyxy'])
                    if v>bi: best,bi=a,v
                if best and bi>=0.5:
                    if best['role']=='bus':
                        busf+=1; busA+= (r=='anchor'); busT+= (r!='anchor')
                    if best['role']=='target': tgtf+=1
        print(f"{box_t:5.2f} {text_t:5.2f} | {nd:5} {ne/nd*100 if nd else 0:12.0f}% | "
              f"{busf:4}/{nbus:<5} {busA:11} {busT:16} | {tgtf:4}/{ntgt:<5}")
