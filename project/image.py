
import os, sys, json, time, argparse, platform
import numpy as np
import cv2
from ultralytics import YOLO

# ================= 常量 =================
NAMES = {0:"a4_board", 1:"circle", 2:"square", 3:"triangle"}
PX_PER_MM    = 10
A4_WMM, A4_HMM = 210, 297
A4_BORDER_MM = 20
CALIB_JSON   = "calib_D.json"

# ================= 相机分辨率档位 =================
RES_TABLE = {"1080p": (1920,1080), "2k": (2560,1440), "4k": (3840,2160)}

def open_camera(index=0, res_key="2k"):
    if res_key not in RES_TABLE:
        print(f"[WARN] 未知分辨率档位 {res_key}，回退 1080p"); res_key="1080p"
    W,H = RES_TABLE[res_key]
    backend = cv2.CAP_DSHOW if platform.system()=="Windows" else 0
    cap = cv2.VideoCapture(index, backend)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    ok, f = cap.read()
    if not ok or f is None:
        raise RuntimeError("相机打开失败")
    hh, ww = f.shape[:2]
    print(f"[Camera] 期望: {W}x{H} / 实际: {ww}x{hh}")
    if abs(ww-W)>16 or abs(hh-H)>16:
        print("[WARN] 实际与期望分辨率不一致，可能是驱动裁切/缩放。")
    return cap

# ================ 显示：只缩小不放大，主窗保持原视角 ================
def show_raw_nonzoom(win, frame, max_w=1600, max_h=900):
    h,w = frame.shape[:2]
    scale = min(1.0, max_w/w, max_h/h)
    disp = frame if scale>=1.0 else cv2.resize(frame,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_NEAREST)
    cv2.putText(disp, f"VIEW raw {w}x{h}  scale={scale:.2f}", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),2,cv2.LINE_AA)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.imshow(win, disp)

def show_fit(win_name, img, max_w=900, max_h=600):
    h, w = img.shape[:2]
    s = min(1.0, max_w/w, max_h/h)
    disp = img if s>=1 else cv2.resize(img,(int(w*s),int(h*s)),interpolation=cv2.INTER_AREA)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL); cv2.imshow(win_name, disp)

# ================= A4 检测/展开 =================
def order_quad(pts):
    pts = np.asarray(pts, np.float32)
    if pts.ndim!=2 or pts.shape!=(4,2):
        pts2 = pts.reshape(-1,1,2).astype(np.float32)
        rect  = cv2.minAreaRect(pts2)
        pts   = cv2.boxPoints(rect).astype(np.float32)
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl,tr,br,bl], np.float32)

def refine_quad_in_crop(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thr  = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21,10)
    thr  = cv2.medianBlur(thr,5)
    thr  = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), 2)
    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best,score=None,-1
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)!=4: continue
        area = cv2.contourArea(approx);
        if area<2000: continue
        q = approx.reshape(-1,2)
        w = np.linalg.norm(q[0]-q[1]); h = np.linalg.norm(q[1]-q[2])
        r = max(w,h)/max(1.0,min(w,h)); s = area/(abs(r-1.414)+0.01)
        if s>score: best,score=q,s
    return order_quad(best) if best is not None else None

def detect_a4_quad_yolo(frame_bgr, yolo_model, conf=0.20, imgsz=1280, iou=0.50):
    res = yolo_model.predict(source=frame_bgr, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    boxes=[]
    for b in res.boxes:
        if int(b.cls.item())==0:  # a4_board
            x1,y1,x2,y2 = b.xyxy.cpu().numpy().ravel()
            boxes.append((x1,y1,x2,y2))
    if not boxes: return None
    boxes.sort(key=lambda t:(t[2]-t[0])*(t[3]-t[1]), reverse=True)
    x1,y1,x2,y2 = boxes[0]
    H,W = frame_bgr.shape[:2]; pad=0.02  # 更贴近 A4
    rx1,ry1 = max(0,int(x1-(x2-x1)*pad)), max(0,int(y1-(y2-y1)*pad))
    rx2,ry2 = min(W-1,int(x2+(x2-x1)*pad)), min(H-1,int(y2+(y2-y1)*pad))
    crop = frame_bgr[ry1:ry2, rx1:rx2].copy()
    quad_rel = refine_quad_in_crop(crop)
    quad = np.array([[rx1,ry1],[rx2,ry1],[rx2,ry2],[rx1,ry2]], np.float32) if quad_rel is None \
           else quad_rel + np.array([[rx1,ry1]], np.float32)
    return order_quad(quad)

def warp_to_a4(bgr, quad):
    W, H = int(A4_WMM*PX_PER_MM), int(A4_HMM*PX_PER_MM)
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)
    M = cv2.getPerspectiveTransform(order_quad(quad).astype(np.float32), dst)
    a4 = cv2.warpPerspective(bgr, M, (W,H))
    return a4, M

def crop_inner(a4_bgr):
    off = int(A4_BORDER_MM*PX_PER_MM)
    return a4_bgr[off:-off, off:-off].copy(), (off,off)

# ================= 形状分类（鲁棒） =================
def _poly_area(pts):
    x = pts[:,0]; y = pts[:,1]
    return 0.5*abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _orth_score_from_lines(bin_or_edge):
    # 输入二值图或边缘图，输出 0~1 的“横竖直线”评分
    if bin_or_edge.ndim == 3:
        g = cv2.cvtColor(bin_or_edge, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(g, 60, 180)
    else:
        edges = bin_or_edge
    h, w = edges.shape[:2]
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                            minLineLength=int(max(12, min(h,w)*0.45)),
                            maxLineGap=int(max(8,  min(h,w)*0.20)))
    if lines is None or len(lines) < 2:
        return 0.0
    ang = []
    for x1,y1,x2,y2 in lines[:,0]:
        dx, dy = x2-x1, y2-y1
        if dx==0 and dy==0:
            continue
        a = abs(np.degrees(np.arctan2(dy, dx))) % 180.0
        ang.append(a)
    if not ang: return 0.0
    ang = np.array(ang)
    ang = np.minimum(ang, 180-ang)  # 折叠到 [0,90]
    near_h = np.mean(ang < 10.0)
    near_v = np.mean(ang > 80.0)
    score = min(1.0, 0.5*(near_h + near_v))
    return float(score)

def classify_shape_v2(crop_bgr):
    h0, w0 = crop_bgr.shape[:2]
    scale = 2.0 if min(h0, w0) < 120 else 1.0
    work = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    g  = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    g  = cv2.GaussianBlur(g, (3,3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th = cv2.medianBlur(th, 3)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)

    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 180: return None

    # 几何量
    peri  = cv2.arcLength(c, True)
    x,y,w,h = cv2.boundingRect(c)
    ratio = min(w,h)/max(w,h)                 # 近方性
    fill  = area/(w*h + 1e-6)                 # 填充率
    circ  = 4*np.pi*area/(peri*peri + 1e-6)   # 圆度

    hull  = cv2.convexHull(c)
    hperi = cv2.arcLength(hull, True)
    approx2 = cv2.approxPolyDP(hull, 0.02*hperi, True)
    approx4 = cv2.approxPolyDP(hull, 0.04*hperi, True)
    k = min(len(approx2), len(approx4))

    tri = cv2.minEnclosingTriangle(hull)[1].reshape(-1,2).astype(np.float32)
    tri_fit = float(area / (_poly_area(tri) + 1e-6))

    box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    def right_angle_score(P):
        if P.shape[0] < 4: return 999
        c0 = np.mean(P, axis=0)
        ang = np.arctan2(P[:,1]-c0[1], P[:,0]-c0[0]); P = P[np.argsort(ang)]
        def ang_err(a,b,c):
            v1, v2 = a-b, c-b
            cos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
            deg = np.degrees(np.arccos(np.clip(cos,-1,1)))
            return abs(deg-90.0)
        return float(np.mean([ang_err(P[(i-1)%4], P[i], P[(i+1)%4]) for i in range(4)]))
    ra_err = right_angle_score(box)

    edges = cv2.Canny(g, 60, 180)
    ortho = _orth_score_from_lines(edges)

    small = min(w,h) < 110  # 小目标=远距离

    # —— 判定
    if (ortho >= (0.55 if small else 0.70) and ratio >= (0.80 if small else 0.86)
        and fill >= (0.74 if small else 0.82) and circ < 0.92):
        return "square"
    if ((k <= 5 and tri_fit >= (0.88 if small else 0.84) and circ < 0.86 and fill <= 0.72)
        or (k==3 and fill <= 0.72)):
        return "triangle"
    if (circ >= (0.90 if small else 0.88)) or (circ >= 0.86 and 0.65 <= fill <= 0.88 and k >= 6):
        return "circle"
    if (ratio >= (0.82 if small else 0.84) and ra_err < 18.0 and fill >= (0.78 if small else 0.80)):
        return "square"
    return None

# ================ YOLO 推理 + 方形兜底 =================
def _fallback_square_by_contour(roi_bgr):
    g  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g  = cv2.GaussianBlur(g, (3,3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th = cv2.medianBlur(th, 3)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)

    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    H,W = roi_bgr.shape[:2]
    roi_area = H*W
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
        area = cv2.contourArea(c)
        if area < 0.002*roi_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        sub = roi_bgr[y:y+h, x:x+w]
        gcls = classify_shape_v2(sub)
        if gcls == "square":
            a_mm = np.mean([w,h]) / PX_PER_MM
            return {"bbox":(x,y,x+w,y+h), "a_mm":a_mm, "score":0.6}
    return None

def measure_shapes_on_roi_yolo(roi_bgr, yolo_model, imgsz=960, conf=0.20, iou=0.50):
    res = yolo_model.predict(source=roi_bgr, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    out = {"circles":[], "triangles":[], "squares":[]}
    name_map = {1:"circle", 2:"square", 3:"triangle"}
    for b in res.boxes:
        cls = int(b.cls.item())
        if cls == 0 or cls not in name_map:   # 跳过 a4_board
            continue
        x1,y1,x2,y2 = b.xyxy.cpu().numpy().ravel()
        sub = roi_bgr[int(max(0,y1)):int(max(0,y2)), int(max(0,x1)):int(max(0,x2))]

        label = name_map[cls]
        try:
            gcls = classify_shape_v2(sub)
            final = gcls if gcls in ("circle","square","triangle") else label
        except Exception:
            final = label

        w=(x2-x1); h=(y2-y1)
        a_mm = np.mean([w,h]) / PX_PER_MM
        item = {"bbox":(x1,y1,x2,y2), "score":float(b.conf.item())}

        if final=="circle":
            item.update({"d_mm":np.mean([w,h])/PX_PER_MM})
            out["circles"].append(item)
        elif final=="square":
            item.update({"a_mm":a_mm})
            out["squares"].append(item)
        elif final=="triangle":
            item.update({"a_mm":a_mm})
            out["triangles"].append(item)

    # YOLO 完全没检出时，几何兜底找方形
    if not (out["circles"] or out["triangles"] or out["squares"]):
        fb = _fallback_square_by_contour(roi_bgr)
        if fb is not None:
            out["squares"].append(fb)

    return out

# ================ 结果选择================
def _enhance_roi(bgr, mode=None):
    if mode is None:
        return bgr
    if mode == "clahe":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L2 = clahe.apply(L)
        lab2 = cv2.merge([L2,A,B])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    if mode == "gamma":
        g = 1.5
        table = (np.linspace(0,1,256)**(1.0/g)*255).astype(np.uint8)
        return cv2.LUT(bgr, table)
    return bgr

def _score_shapes(packs):
    cnt = len(packs["circles"]) + len(packs["squares"]) + len(packs["triangles"])
    conf_sum = sum(s["score"] for s in packs["circles"]) + \
               sum(s["score"] for s in packs["squares"]) + \
               sum(s["score"] for s in packs["triangles"])
    return (cnt, conf_sum)

# 连拍+多策略兜底：返回 (best_roi, best_shapes)
def find_shapes_best(cap, quad, model):
    best = None
    best_score = (-1, -1)
    for _ in range(3):  # 连拍3帧
        ok2, f2 = cap.read()
        if not ok2: continue
        a4_2, _ = warp_to_a4(f2, quad)
        if a4_2 is None:
            continue
        roi_2, _ = crop_inner(a4_2)

        trials = [
            (roi_2,                 960,  0.20, 0.50, None),
            (roi_2,                1280,  0.15, 0.45, None),
            (roi_2,                1536,  0.12, 0.45, None),
            (_enhance_roi(roi_2,"clahe"), 1280, 0.15, 0.45, "clahe"),
            (_enhance_roi(roi_2,"gamma"), 1280, 0.15, 0.45, "gamma"),
        ]

        for roi_try, imgsz, conf, iou, tag in trials:
            packs = measure_shapes_on_roi_yolo(roi_try, model, imgsz=imgsz, conf=conf, iou=iou)
            sc = _score_shapes(packs)
            if sc > best_score:
                best, best_score = (roi_try, packs), sc
            if sc[0] >= 1 and sc[1] > 1.2:
                return best
    return best

def pick_square(squares, ratio_th=0.80):
    filtered = []
    for s in squares:
        x1,y1,x2,y2 = s["bbox"]
        w = max(1.0, x2-x1); h = max(1.0, y2-y1)
        if min(w,h)/max(w,h) >= ratio_th:
            filtered.append(s)
    base = filtered if filtered else squares
    return max(base, key=lambda s: s["score"]) if base else None

def capture_y_at_D(cap, yolo_model, D_cm, a4_imgsz=1280, a4_conf=0.20, burst_frames=8, edge_samples=5, min_take=12):
    takes=[]
    print(f"\n把 A4 放到 D={D_cm}cm，按 SPACE 采样（≥{min_take} 次，中位数稳健估计）")
    while True:
        ok, frame = cap.read()
        if not ok: break
        disp = frame.copy()
        quad = detect_a4_quad_yolo(frame, yolo_model, conf=a4_conf, imgsz=a4_imgsz)
        if quad is not None:
            bl,br = quad[3], quad[2]
            mid = (int((bl[0]+br[0])/2), int((bl[1]+br[1])/2))
            cv2.circle(disp, mid, 6, (0,255,0), -1)
            cv2.polylines(disp, [quad.astype(int)], True, (0,0,255), 2)
        show_raw_nonzoom("Calibrate D", disp, 1600, 900)
        k = cv2.waitKey(10) & 0xFF
        if k == 27: break
        if k == 32:
            ys=[]
            for _ in range(burst_frames):
                ok2, f2 = cap.read()
                if not ok2: continue
                q2 = detect_a4_quad_yolo(f2, yolo_model, conf=a4_conf, imgsz=a4_imgsz)
                if q2 is None: continue
                bl,br = q2[3].astype(float), q2[2].astype(float)
                ys_i=[]
                for t in np.linspace(0.1,0.9,edge_samples):
                    y_t = (1-t)*bl[1] + t*br[1]
                    ys_i.append(y_t)
                ys.append(float(np.mean(ys_i))); time.sleep(0.01)
            if ys:
                y_med = float(np.median(ys)); takes.append(y_med)
                print(f"sample#{len(takes):02d}: y_med={y_med:.1f} (frames={len(ys)})")
            if len(takes) >= max(min_take, 25): break
    return float(np.median(takes)) if takes else None

# ----------------- 拟合/保存/加载/校验（保持不动） -----------------
def fit_projective(ys, Ds):
    A=[]
    for y,D in zip(ys,Ds): A.append([y,1.0,-D*y,-D])
    A=np.asarray(A,float)
    _,_,Vt=np.linalg.svd(A)
    a,b,c,d = Vt[-1,:]
    return dict(a=float(a),b=float(b),c=float(c),d=float(d))

def save_calib(json_path, params, frame_w, frame_h):
    data = dict(params); data["frame_w"]=int(frame_w); data["frame_h"]=int(frame_h)
    with open(json_path,"w",encoding="utf-8") as f: json.dump(data,f,indent=2,ensure_ascii=False)
    print(f"[Calib] 已保存到 {json_path}（{frame_w}x{frame_h}）")

def load_calib(json_path=CALIB_JSON):
    with open(json_path,"r",encoding="utf-8") as f: return json.load(f)

def check_calib_vs_camera(calib, w, h):
    cw=int(calib.get("frame_w",-1)); ch=int(calib.get("frame_h",-1))
    if cw<0 or ch<0:
        print("[WARN] 标定未记录分辨率，建议重做 --calib"); return
    if (cw,ch)!=(w,h):
        raise RuntimeError(f"标定分辨率 {cw}x{ch} 与当前相机 {w}x{h} 不一致，请在当前分辨率下重新标定。")

def y_to_Dcm(y, abcd):
    a,b,c,d = abcd["a"],abcd["b"],abcd["c"],abcd["d"];
    return (a*y + b)/(c*y + d)

# ================= 运行流程（融合版） =================
def run_measure(model_path, cam_id, res_key="2k", imgsz_a4=1536, conf=0.20):
    model = YOLO(model_path)
    cap = open_camera(cam_id, res_key=res_key)
    ok, frame = cap.read(); assert ok
    H,W = frame.shape[:2]
    if not os.path.exists(CALIB_JSON):
        raise FileNotFoundError("未找到 calib_D.json，请先 --calib")
    calib = load_calib(CALIB_JSON); check_calib_vs_camera(calib, W, H)

    print("按 S 一键测量；ESC 退出。")
    quad=None; t_last=0
    while True:
        ok,frame = cap.read()
        if not ok: break
        show = frame.copy()

        # 周期性刷新 A4
        if (time.time()-t_last) > 0.20 or quad is None:
            q = detect_a4_quad_yolo(frame, model, conf=conf, imgsz=imgsz_a4, iou=0.50)
            if q is not None: quad=q
            t_last=time.time()

        if quad is not None:
            cv2.polylines(show, [quad.astype(int)], True, (0,255,0), 2)
            bl,br = quad[3],quad[2]
            D_est = y_to_Dcm((bl[1]+br[1])/2.0, calib)
            cv2.putText(show, f"D~{D_est:.1f}cm", (30,70), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,255),2)

        show_raw_nonzoom("measure", show, 1600, 900)
        k=cv2.waitKey(10)&0xFF
        if k==27: break

        if k==ord('s'):
            if quad is None:
                print("未检测到 a4_board，重试"); continue
            t0=time.time()

            # 实际距离 D（严格用第一份标定函数）
            bl,br = quad[3],quad[2]
            D_cm = y_to_Dcm((bl[1]+br[1])/2.0, calib)

            # 透视展开 + ROI
            a4,_ = warp_to_a4(frame, quad); roi,_ = crop_inner(a4)
            show_fit("a4_warp", a4); show_fit("roi", roi)

            # 连拍挑优 + 多策略兜底（来自第二份代码）
            best = find_shapes_best(cap, quad, model)
            if best is None:
                # fallback 用当前帧的 ROI
                packs = measure_shapes_on_roi_yolo(roi, model, imgsz=960, conf=0.20, iou=0.50)
                best = (roi, packs)
            roi_best, shapes = best

            # 激进兜底一次
            if not (shapes["circles"] or shapes["squares"] or shapes["triangles"]):
                shapes = measure_shapes_on_roi_yolo(roi_best, model, imgsz=1536, conf=0.10, iou=0.40)
                if not (shapes["circles"] or shapes["squares"] or shapes["triangles"]):
                    print("未找到可测的图形"); continue

            # 结果优先级：三角 > 圆 > 方
            x_cm=None; mode="auto"
            if shapes["triangles"]:
                t0s = max(shapes["triangles"], key=lambda s:s["score"])
                x_cm = t0s["a_mm"]/10.0; mode="triangle"
            elif shapes["circles"]:
                c0 = max(shapes["circles"], key=lambda s:s["score"])
                x_cm = c0["d_mm"]/10.0; mode="circle"
            elif shapes["squares"]:
                sq = pick_square(shapes["squares"])
                x_cm = sq["a_mm"]/10.0; mode="square"
            else:
                print("未找到可测的图形"); continue

            dt=(time.time()-t0)*1000
            print(f"[{mode}]  D = {D_cm:.1f} cm,  x = {x_cm:.1f} cm   (用时 {dt:.0f} ms)")

    cap.release(); cv2.destroyAllWindows()

# ================= CLI =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--res", type=str, default="2k", choices=["1080p","2k","4k"])
    ap.add_argument("--imgsz_a4", type=int, default=1536)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--calib", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.model)

    if args.calib:
        cap = open_camera(args.cam, res_key=args.res)
        assert cap.isOpened(), "摄像头打开失败"
        print("开始三点标定（建议 120/160/200 cm）")
        Ds=[]; ys=[]
        try:
            for D in [120,160,200]:
                y = capture_y_at_D(cap, model, D, a4_imgsz=args.imgsz_a4, a4_conf=args.conf)
                if y is None:
                    print("该点未采到，退出"); cap.release(); sys.exit(1)
                Ds.append(D); ys.append(y)
        finally:
            ok, fr = cap.read(); h,w = fr.shape[:2] if ok else (0,0)
            cap.release(); cv2.destroyAllWindows()
        abcd = fit_projective(ys, Ds)
        save_calib(CALIB_JSON, abcd, w, h)
        return

    run_measure(args.model, args.cam, res_key=args.res, imgsz_a4=args.imgsz_a4, conf=args.conf)

if __name__ == "__main__":
    main()
