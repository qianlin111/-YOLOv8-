
import os, json, time, argparse, platform
import numpy as np
import cv2
from ultralytics import YOLO

# ----------------- 常量 -----------------
RES = {"1080p": (1920,1080), "2k": (2560,1440), "4k": (3840,2160)}
PX_PER_MM   = 10
A4_WMM, A4_HMM = 210, 297
A4_BORDER_MM   = 20
CALIB_JSON = "calib_D.json"

# ----------------- 相机 -----------------
def open_camera(idx=0, res_key="2k"):
    W,H = RES[res_key]
    backend = cv2.CAP_DSHOW if platform.system()=="Windows" else 0
    cap = cv2.VideoCapture(idx, backend)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    ok, fr = cap.read()
    if not ok or fr is None:
        raise RuntimeError("相机打开失败")
    hh,ww = fr.shape[:2]
    print(f"[Cam] 要求 {W}x{H}，实际 {ww}x{hh}")
    return cap

def show_raw(win, frame, maxw=1600, maxh=900):
    h,w = frame.shape[:2]
    s = min(1.0, maxw/w, maxh/h)
    disp = frame if s>=1 else cv2.resize(frame,(int(w*s),int(h*s)),interpolation=cv2.INTER_NEAREST)
    cv2.putText(disp, f"VIEW {w}x{h}  scale={s:.2f}", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),2,cv2.LINE_AA)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.imshow(win, disp)

def show_fit(win_name, img, max_w=900, max_h=600):
    """把图像按给定上限等比缩放后显示（防撑屏）。"""
    h, w = img.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    disp = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, disp)

# ----------------- 标定 D(y) -----------------
class DCalib:
    def __init__(self): self.abcd=None
    def load(self):
        if not os.path.exists(CALIB_JSON):
            print("[Calib] 未找到 calib_D.json（仍可演示）"); return False
        with open(CALIB_JSON,"r",encoding="utf-8") as f: obj=json.load(f)
        if "abcd" in obj: self.abcd=obj["abcd"]
        elif all(k in obj for k in ("a","b","c","d")): self.abcd=[obj["a"],obj["b"],obj["c"],obj["d"]]
        else: print("[Calib] JSON 无 abcd"); return False
        return True
    def y_to_Dcm(self, y):
        if not self.abcd: return -1.0
        a,b,c,d = self.abcd
        return float((a*y + b) / (c*y + d))

# ----------------- A4 检出/透视 -----------------
def order_quad(pts):
    if pts is None: return None
    pts = np.asarray(pts, np.float32)
    if pts.ndim!=2 or pts.shape!=(4,2):
        pts2 = pts.reshape(-1,1,2).astype(np.float32)
        rect = cv2.minAreaRect(pts2)
        pts  = cv2.boxPoints(rect).astype(np.float32)
    s=pts.sum(1); d=np.diff(pts,axis=1).ravel()
    tl=pts[np.argmin(s)]; br=pts[np.argmax(s)]
    tr=pts[np.argmin(d)]; bl=pts[np.argmax(d)]
    return np.array([tl,tr,br,bl], np.float32)

def refine_quad_in_crop(crop_bgr):
    g  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g  = cv2.GaussianBlur(g,(5,5),0)
    thr= cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,10)
    thr= cv2.medianBlur(thr,5)
    thr= cv2.morphologyEx(thr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), 2)
    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best,score=None,-1
    for c in sorted(cnts,key=cv2.contourArea,reverse=True)[:8]:
        peri=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*peri,True)
        if len(approx)!=4: continue
        area=cv2.contourArea(approx)
        if area<2000: continue
        q=approx.reshape(-1,2).astype(np.float32)
        w=np.linalg.norm(q[0]-q[1]); h=np.linalg.norm(q[1]-q[2])
        r=max(w,h)/max(1.0,min(w,h)); sc=area/(abs(r-1.414)+0.01)
        if sc>score: best,score=q,sc
    return order_quad(best) if best is not None else None

def detect_a4_quad(frame, shape_model, imgsz=1792, conf=0.20, iou=0.50):
    res = shape_model.predict(source=frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    boxes=[]
    for b in res.boxes:
        if int(b.cls.item())==0:
            x1,y1,x2,y2=b.xyxy.cpu().numpy().ravel(); boxes.append((x1,y1,x2,y2,float(b.conf.item())))
    if not boxes: return None
    boxes.sort(key=lambda t:(t[2]-t[0])*(t[3]-t[1]), reverse=True)
    x1,y1,x2,y2,_ = boxes[0]
    H,W=frame.shape[:2]; pad=0.02
    rx1,ry1=max(0,int(x1-(x2-x1)*pad)), max(0,int(y1-(y2-y1)*pad))
    rx2,ry2=min(W-1,int(x2+(x2-x1)*pad)), min(H-1,int(y2+(y2-y1)*pad))
    crop = frame[ry1:ry2, rx1:rx2].copy()
    qrel = refine_quad_in_crop(crop)
    if qrel is None: quad=np.array([[rx1,ry1],[rx2,ry1],[rx2,ry2],[rx1,ry2]],np.float32)
    else: quad=qrel+np.array([[rx1,ry1]],np.float32)
    return order_quad(quad)

def warp_to_a4(bgr, quad):
    W,H=int(A4_WMM*PX_PER_MM), int(A4_HMM*PX_PER_MM)
    dst=np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],np.float32)
    q=order_quad(quad)
    if q is None: return None,None
    M=cv2.getPerspectiveTransform(q, dst); a4=cv2.warpPerspective(bgr,M,(W,H))
    return a4,M

def crop_inner(a4_bgr):
    off=int(A4_BORDER_MM*PX_PER_MM)
    return a4_bgr[off:-off,off:-off].copy(), (off,off)

# ----------------- 几何分类（护方） -----------------
def _poly_area(pts):
    x=pts[:,0]; y=pts[:,1]
    return 0.5*abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))

def _orth_score_from_lines(edge):
    if edge.ndim==3:
        g=cv2.cvtColor(edge,cv2.COLOR_BGR2GRAY); edge=cv2.Canny(g,60,180)
    h,w=edge.shape[:2]
    lines=cv2.HoughLinesP(edge,1,np.pi/180,threshold=30,
                          minLineLength=int(max(12,min(h,w)*0.45)),
                          maxLineGap=int(max(8,min(h,w)*0.20)))
    if lines is None or len(lines)<2: return 0.0
    ang=[]
    for x1,y1,x2,y2 in lines[:,0]:
        dx,dy=x2-x1,y2-y1
        if dx==0 and dy==0: continue
        a=abs(np.degrees(np.arctan2(dy,dx)))%180; ang.append(a)
    if not ang: return 0.0
    ang=np.array(ang); ang=np.minimum(ang,180-ang)
    return float(min(1.0,0.5*(np.mean(ang<10.0)+np.mean(ang>80.0))))

def classify_shape_v2(crop_bgr):
    h0,w0=crop_bgr.shape[:2]
    scale=2.0 if min(h0,w0)<120 else 1.0
    work=cv2.resize(crop_bgr,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    g=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
    g=cv2.GaussianBlur(g,(3,3),0)
    _,th=cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th=cv2.medianBlur(th,3)
    th=cv2.morphologyEx(th,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),1)
    cnts,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c=max(cnts,key=cv2.contourArea); area=cv2.contourArea(c)
    if area<180: return None
    peri=cv2.arcLength(c,True); x,y,w,h=cv2.boundingRect(c)
    ratio=min(w,h)/max(w,h); fill=area/(w*h+1e-6); circ=4*np.pi*area/(peri*peri+1e-6)
    hull=cv2.convexHull(c); hperi=cv2.arcLength(hull,True)
    approx2=cv2.approxPolyDP(hull,0.02*hperi,True)
    approx4=cv2.approxPolyDP(hull,0.04*hperi,True); k=min(len(approx2),len(approx4))
    tri=cv2.minEnclosingTriangle(hull)[1].reshape(-1,2).astype(np.float32)
    tri_fit=float(area/(_poly_area(tri)+1e-6))
    box=cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    def right_angle_score(P):
        if P.shape[0]<4: return 999
        c0=np.mean(P,axis=0); ang=np.arctan2(P[:,1]-c0[1],P[:,0]-c0[0]); P=P[np.argsort(ang)]
        def ang_err(a,b,c):
            v1,v2=a-b,c-b; cos=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
            return abs(np.degrees(np.arccos(np.clip(cos,-1,1)))-90.0)
        return float(np.mean([ang_err(P[(i-1)%4],P[i],P[(i+1)%4]) for i in range(4)]))
    ra_err=right_angle_score(box)
    edges=cv2.Canny(g,60,180); ortho=_orth_score_from_lines(edges); small=min(w,h)<110
    if (ortho>=(0.55 if small else 0.70) and ratio>=(0.80 if small else 0.86) and fill>=(0.74 if small else 0.82) and circ<0.92):
        return "square"
    if ((k<=5 and tri_fit>=(0.88 if small else 0.84) and circ<0.86 and fill<=0.72) or (k==3 and fill<=0.72)):
        return "triangle"
    if (circ>=(0.90 if small else 0.88)) or (circ>=0.86 and 0.65<=fill<=0.88 and k>=6):
        return "circle"
    if (ratio>=(0.82 if small else 0.84) and ra_err<18.0 and fill>=(0.78 if small else 0.80)):
        return "square"
    return None

# ----------------- YOLO 推理 -----------------
def nms_keep(dets, iou_th=0.5):
    kept=[]
    for d in sorted(dets,key=lambda x:x[-1],reverse=True):
        if all(iou(d[:4],k[:4])<iou_th for k in kept):
            kept.append(d)
    return kept

def iou(b1,b2):
    x1,y1,x2,y2=b1; X1,Y1,X2,Y2=b2
    ix1,iy1=max(x1,X1),max(y1,Y1); ix2,iy2=min(x2,X2),min(y2,Y2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    a1=(x2-x1)*(y2-y1); a2=(X2-X1)*(Y2-Y1)
    return inter/max(1.0,a1+a2-inter)

def detect_shapes_on_roi(roi, shape_model, imgsz=1280, conf=0.18, iou_th=0.50):
    res = shape_model.predict(source=roi, imgsz=imgsz, conf=conf, iou=iou_th, verbose=False)[0]
    raw=[]
    for b in res.boxes:
        cls=int(b.cls.item())
        if cls not in (1,2,3): continue
        x1,y1,x2,y2=b.xyxy.cpu().numpy().ravel(); raw.append([x1,y1,x2,y2,float(b.conf.item()),cls])
    dets=nms_keep(raw,0.5)
    out={"squares":[]}
    H,W=roi.shape[:2]; ppm=max(1.0,min(W,H)/210.0)
    for x1,y1,x2,y2,score,cls in dets:
        sub=roi[int(max(0,y1)):int(max(0,y2)), int(max(0,x1)):int(max(0,x2))]
        gcls=classify_shape_v2(sub)
        final=gcls if gcls else {1:"circle",2:"square",3:"triangle"}[cls]
        if final=="square":
            a_mm=0.5*((x2-x1)+(y2-y1))/ppm
            out["squares"].append({"bbox":(x1,y1,x2,y2), "a_mm":a_mm, "score":score})
    return out

def detect_digits_on_roi(roi, digit_model, imgsz=960, conf=0.18, iou_th=0.50):
    res = digit_model.predict(source=roi, imgsz=imgsz, conf=conf, iou=iou_th, verbose=False)[0]
    raw=[]
    for b in res.boxes:
        d=int(b.cls.item()); x1,y1,x2,y2=b.xyxy.cpu().numpy().ravel()
        raw.append([x1,y1,x2,y2,float(b.conf.item()),d])
    return nms_keep(raw,0.5)

def detect_digit_in_square(roi_bgr, square, digit_model, imgsz=384, conf=0.15):
    """只在某个方块内部识别数字；返回 dict 或 None"""
    x1, y1, x2, y2 = map(int, square["bbox"])
    dx = int(0.12 * (x2 - x1)); dy = int(0.12 * (y2 - y1))  # 内缩，避开白边/阴影
    x1i, y1i, x2i, y2i = x1 + dx, y1 + dy, x2 - dx, y2 - dy
    h, w = roi_bgr.shape[:2]
    x1i = max(0, x1i); y1i = max(0, y1i); x2i = min(w - 1, x2i); y2i = min(h - 1, y2i)
    if x2i <= x1i or y2i <= y1i: return None

    crop = roi_bgr[y1i:y2i, x1i:x2i]
    res = digit_model.predict(source=crop, imgsz=imgsz, conf=conf, iou=0.50, verbose=False)[0]
    best = None
    for b in res.boxes:
        d  = int(b.cls.item())
        sc = float(b.conf.item())
        bx1, by1, bx2, by2 = b.xyxy.cpu().numpy().ravel()
        # 还原到 ROI 坐标
        bx1, by1, bx2, by2 = x1i + bx1, y1i + by1, x1i + bx2, y1i + by2
        if (best is None) or (sc > best["score"]):
            best = {"digit": d, "score": sc, "bbox_in_roi": (float(bx1), float(by1), float(bx2), float(by2))}
    return best

# ----------------- 匹配（旧策略，作为回退） -----------------
def box_center(b): x1,y1,x2,y2=b; return ((x1+x2)/2.0,(y1+y2)/2.0)

def match_digit_to_square(dets_square, dets_digit, target_digit):
    if not dets_square or not dets_digit: return None, None
    cand=[d for d in dets_digit if d[-1]==target_digit]
    if not cand:
        return None, max(dets_digit,key=lambda d:d[-2])  # 用最高分数字做提示

    # 1) 中心点在“外扩12%”的方块框内
    for d in sorted(cand, key=lambda x:x[-2], reverse=True):
        cx,cy=box_center(d[:4])
        best=None; best_conf=-1
        for s in dets_square:
            x1,y1,x2,y2=s["bbox"]; dx=0.12*(x2-x1); dy=0.12*(y2-y1)
            if (x1-dx)<=cx<=(x2+dx) and (y1-dy)<=cy<=(y2+dy):
                if s["score"]>best_conf: best,best_conf=s,s["score"]
        if best is not None: return best, d

    # 2) IoU 兜底（≥0.12）
    for d in sorted(cand, key=lambda x:x[-2], reverse=True):
        best=None; best_iou=0
        for s in dets_square:
            if iou(d[:4], s["bbox"]) >= 0.12:
                if s["score"]>getattr(best,"score",0): best=s; best_iou=1
        if best_iou: return best, d

    # 3) 最近中心点兜底
    best=None; bestdist=1e9; bestd=None
    for d in cand:
        cx,cy=box_center(d[:4])
        for s in dets_square:
            sx,sy=box_center(s["bbox"])
            dist=(sx-cx)**2+(sy-cy)**2
            if dist<bestdist: best=s; bestdist=dist; bestd=d
    return best, bestd

# ----------------- ROI连拍挑优 -----------------
def enhance_roi(bgr, mode=None):
    if mode is None: return bgr
    if mode=="clahe":
        lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB); L,A,B=cv2.split(lab)
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)); L2=clahe.apply(L)
        return cv2.cvtColor(cv2.merge([L2,A,B]),cv2.COLOR_LAB2BGR)
    if mode=="gamma":
        g=1.5; table=(np.linspace(0,1,256)**(1.0/g)*255).astype(np.uint8); return cv2.LUT(bgr,table)
    return bgr

def best_roi_and_dets(cap, quad, shape_model, digit_model):
    best=None; best_score=(-1,-1)
    for _ in range(3):
        ok,fr=cap.read()
        if not ok: continue
        a4,_=warp_to_a4(fr,quad)
        if a4 is None: continue
        roi,_=crop_inner(a4)
        trials=[
            (roi,                 1280,0.18,0.50,  960,0.18,0.50, None),
            (roi,                 1536,0.15,0.50, 1280,0.16,0.50, None),
            (enhance_roi(roi,"clahe"),1280,0.15,0.45,1280,0.16,0.50,"clahe"),
            (enhance_roi(roi,"gamma"),1280,0.15,0.45,1280,0.16,0.50,"gamma"),
        ]
        for r,s_img,s_conf,s_iou,d_img,d_conf,d_iou,tag in trials:
            squares = detect_shapes_on_roi(r, shape_model, imgsz=s_img, conf=s_conf, iou_th=s_iou)["squares"]
            digits  = detect_digits_on_roi(r, digit_model, imgsz=d_img, conf=d_conf, iou_th=d_iou)
            sc=(len(squares)+len(digits), sum(s["score"] for s in squares)+sum(d[-2] for d in digits))
            if sc>best_score: best=(r,squares,digits); best_score=sc
            if sc[0]>=3 and sc[1]>1.8: return best
    return best

# ----------------- 主流程 -----------------
def run(args):
    print("按数字键 0~9 设定目标；按 S 测量；ESC 退出。")
    cap = open_camera(args.cam, args.res)
    shape_model = YOLO(args.shape_model)
    digit_model = YOLO(args.digit_model)
    calib = DCalib(); calib.load()

    quad=None; t_last=0
    target=args.target_digit

    while True:
        ok,frame=cap.read()
        if not ok: break

        # 周期检测 A4
        if time.time()-t_last>0.2 or quad is None:
            quad = detect_a4_quad(frame, shape_model, imgsz=args.imgsz_a4, conf=args.conf, iou=0.50)
            t_last=time.time()

        vis = frame.copy()
        if quad is not None: cv2.polylines(vis,[quad.astype(int)],True,(0,255,0),2)
        cv2.putText(vis, f"target={target if target is not None else '-'}", (12,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,255),2,cv2.LINE_AA)
        show_raw("measure", vis)

        k=cv2.waitKey(1)&0xFF
        if k==27: break
        if ord('0')<=k<=ord('9'):
            target=k-ord('0'); print(f"[Target] 目标数字 -> {target}")

        if k==ord('s'):
            if quad is None:
                print("未检测到 a4_board，重试"); continue
            if target is None:
                print("[Warn] 未设置目标数字：按 0~9 或用 --target_digit"); continue

            t0=time.time()
            bl,br=quad[3],quad[2]; D_cm = calib.y_to_Dcm(float((bl[1]+br[1])/2.0))
            best = best_roi_and_dets(cap, quad, shape_model, digit_model)
            if best is None:
                print("未找到正方形/数字"); continue
            roi,squares,digits = best

            # —— 给每个方块单独裁剪识别数字（优先用这个结果）——
            mapped = 0
            for s in squares:
                dd = detect_digit_in_square(roi, s, digit_model, imgsz=384, conf=0.15)
                if dd is not None:
                    s["digit"] = int(dd["digit"])
                    s["d_score"] = float(dd["score"])
                    s["digit_box"] = dd["bbox_in_roi"]
                    mapped += 1

            # 可视化 ROI
            roi_show=roi.copy()
            for s in squares:
                x1,y1,x2,y2=map(int,s["bbox"])
                cv2.rectangle(roi_show,(x1,y1),(x2,y2),(0,255,0),2)
                tag = f"S:{s.get('digit','-')}"
                cv2.putText(roi_show, tag, (x1,max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
                if "digit_box" in s:
                    bx1,by1,bx2,by2 = map(int, s["digit_box"])
                    cv2.rectangle(roi_show,(bx1,by1),(bx2,by2),(255,255,0),2)
            # 显示全局数字框（回退用）
            for d in digits:
                x1,y1,x2,y2=map(int,d[:4])
                cv2.rectangle(roi_show,(x1,y1),(x2,y2),(255,255,0),1)
                cv2.putText(roi_show, str(d[-1]), (x1,y2+16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,0),1)
            show_fit("ROI", roi_show, max_w=900, max_h=600)

            # —— 先用“方块内识别”的结果直接选 ——
            cands = [s for s in squares if s.get("digit", None) == target]
            if cands:
                # 选综合置信度高的；没有 d_score 就当 0 处理
                sq = max(cands, key=lambda s: min(s.get("score",0), s.get("d_score", s.get("score",0))))
                x_cm = sq["a_mm"]/10.0
                print(f"[digit {target}]  D = {D_cm:.1f} cm,  x = {x_cm:.1f} cm   (优先方块内识别，用时 {(time.time()-t0)*1000:.0f} ms)")
                continue

            # —— 回退：再用“全 ROI 检测的数字+匹配”策略 ——
            sq, dsel = match_digit_to_square(squares, digits, target)
            if sq is None:
                if dsel is None: print("未匹配到目标数字/方块")
                else: print(f"[Warn] 未匹配到数字 {target} 的方块（但检测到 {dsel[-1]}）")
                continue

            x_cm = sq["a_mm"]/10.0
            print(f"[digit {target}]  D = {D_cm:.1f} cm,  x = {x_cm:.1f} cm   (用时 {(time.time()-t0)*1000:.0f} ms)")

    cap.release(); cv2.destroyAllWindows()

# ----------------- CLI -----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--res", type=str, default="2k", choices=list(RES.keys()))
    ap.add_argument("--shape_model", type=str, required=True)
    ap.add_argument("--digit_model", type=str, required=True)
    ap.add_argument("--imgsz_a4", type=int, default=1792)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--target_digit", type=int, default=None)
    args=ap.parse_args()
    run(args)

if __name__=="__main__":
    main()
