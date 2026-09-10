"""Experiment 12: recovering the small `front` targets.
(a) sanity-check the resolution result by reporting raw detection counts
(b) sweep box_threshold for target recall"""
import json, cv2, sys, collections
R='/isis/home/hasana3/vlmtest/GroundingDINO'; sys.path.insert(0,f'{R}/eval')
from PIL import Image
from groundingdino.util.inference import load_model, predict
from worker_clean import build_normalize_transform
from torchvision import transforms as T

def tf(cap):
    def rs(img):
        w,h=img.size; ss=min(w,h)
        if ss>cap: s=cap/ss; return img.resize((int(w*s),int(h*s)))
        return img
    return T.Compose([T.Lambda(rs),T.ToTensor(),
                      T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); i=iw*ih
    u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u>0 else 0

m=load_model(f'{R}/groundingdino/config/GroundingDINO_SwinB_cfg.py',
             f'{R}/weights/groundingdino_swinb_cogcoor.pth').to('cuda:0').eval()

print("(a) raw detection counts by input cap  [cfg01_front, 15 frames]")
gt=json.load(open(f'{R}/dataset/rmot3/cfg01_front/gt.json'))
for cap in (800,1080):
    t=tf(cap); nd=0
    for fr in range(0,300,20):
        img=cv2.imread(f'{R}/dataset/rmot3/cfg01_front/images/{fr:06d}.png')
        if img is None: continue
        ten=t(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
        b,_,_=predict(model=m,image=ten,caption='red sedan . bus .',
                      box_threshold=0.40,text_threshold=0.40,device='cuda:0')
        nd+=len(b)
    print(f"    cap={cap}: {nd} detections total  (tensor short side "
          f"{tf(cap)(Image.fromarray(cv2.cvtColor(cv2.imread(f'{R}/dataset/rmot3/cfg01_front/images/000000.png'),cv2.COLOR_BGR2RGB))).shape})")

print("\n(b) target recall vs box_threshold (cap=800, text=0.40)")
t=build_normalize_transform()
SEQS=['cfg01_front','cfg03_front','cfg13_front','cfg05_front']
print(f"{'sequence':16}" + "".join(f"{('box'+str(b)):>12}" for b in (0.40,0.25,0.15,0.08)))
for seq in SEQS:
    gt=json.load(open(f'{R}/dataset/rmot3/{seq}/gt.json'))
    byfr=collections.defaultdict(list)
    for a in gt['annotations']: byfr[a['image_id']].append(a)
    row=f"{seq:16}"
    for bt in (0.40,0.25,0.15,0.08):
        found=tot=0
        for fr in range(0,300,20):
            img=cv2.imread(f'{R}/dataset/rmot3/{seq}/images/{fr:06d}.png')
            if img is None: continue
            H,W=img.shape[:2]
            tg=[a for a in byfr[fr] if a['role']=='target' and (a.get('visibility') or 0)>=0.2]
            if not tg: continue
            tot+=len(tg)
            ten=t(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
            b,_,_=predict(model=m,image=ten,caption='red sedan . bus .',
                          box_threshold=bt,text_threshold=0.40,device='cuda:0')
            bx=[]
            for bb in b:
                cx,cy,w,h=bb.tolist()
                bx.append([(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H])
            for a in tg:
                if any(iou(q,a['bbox_xyxy'])>=0.5 for q in bx): found+=1
        row+=f"{found:5}/{tot:<6}"
    print(row)
