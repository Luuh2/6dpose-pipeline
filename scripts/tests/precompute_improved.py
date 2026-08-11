"""Improved M1-M7 precompute with fixes for K, mask, and depth."""
import sys,os,cv2,torch,numpy as np
sys.path.insert(0,'E:/zhijiyige')
O='E:/zhijiyige/output_improved'
os.makedirs(O, exist_ok=True); os.makedirs(f'{O}/meshes', exist_ok=True)

# === M1: Load all frames ===
from modules.video_decoder import VideoDecoder
d=VideoDecoder(360)
frames=[f for _,f in d.decode_stream('E:/zhijiyige/demo/test_mustard.mp4')]
N,h,w=len(frames),frames[0].shape[0],frames[0].shape[1]
print(f'M1: {N}f {w}x{h}')

# === M2: YOLO — scan multiple frames for best detection ===
from modules.yolo_world_detector import YOLOWorldDetector
y=YOLOWorldDetector(model_path='E:/zhijiyige/weights/yolo_world/yolov8s-worldv2.pt',conf_threshold=0.10)
best_f, best_bb, best_score = 0, None, 0
for fi in range(0, min(N, 700), 20):
    dets = y.detect(cv2.cvtColor(frames[fi],cv2.COLOR_BGR2RGB), 'mustard bottle,bottle,condiment')
    for d in dets:
        bw = d['bbox'][2]-d['bbox'][0]; bh = d['bbox'][3]-d['bbox'][1]
        s = d['score'] * min(bw*bh, 5000)
        if s > best_score: best_score = s; best_f = fi; best_bb = d['bbox']
bb = np.array(best_bb); y.unload(); del y; torch.cuda.empty_cache()
r1 = cv2.cvtColor(frames[best_f], cv2.COLOR_BGR2RGB)
bbox_w = bb[2]-bb[0]; bbox_h = bb[3]-bb[1]
print(f'M2: frame {best_f}, {best_bb} score={best_score:.0f} bbox={bb.astype(int)} size={bbox_w:.0f}x{bbox_h:.0f}px')

# === M3: GrabCut on best detection frame + aggressive expansion ===
print(f'[SAM] GrabCut on frame {best_f}...')
bgd = np.zeros((1,65), np.float64); fgd = np.zeros((1,65), np.float64)
rect = (int(bb[0]), int(bb[1]), int(bb[2]-bb[0]), int(bb[3]-bb[1]))
mask_gc = np.zeros((h,w), np.uint8)
cv2.grabCut(frames[best_f], mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
m0 = ((mask_gc==1) | (mask_gc==3)).astype(np.uint8)
kernel = np.ones((15,15), np.uint8)
m0 = cv2.dilate(m0, kernel, iterations=3)
# Also add rectangular ROI
x1,y1,x2,y2 = bb.astype(int); cx,cy = (x1+x2)//2, (y1+y2)//2
bw,bh = x2-x1, y2-y1
x1e = max(0, cx - int(bw*2.0)); y1e = max(0, cy - int(bh*1.2))
x2e = min(w, cx + int(bw*2.0)); y2e = min(h, cy + int(bh*1.2))
m_roi = np.zeros((h,w), dtype=np.uint8); m_roi[y1e:y2e, x1e:x2e] = 1
m0 = m0 | m_roi
print(f'[SAM] GrabCut+expand+ROI: area={m0.sum()}px ({m0.sum()/(h*w)*100:.1f}%)')

# === M4: Create masks for all frames (expanded) ===
masks = np.memmap(f'{O}/masks.dat', dtype=np.uint8, mode='w+', shape=(N,h,w))
# Further expand for robustness
kernel2 = np.ones((20,20), np.uint8)
m_expanded = cv2.dilate(m0, kernel2, iterations=2)
# Add rectangular ROI for safety
x1,y1,x2,y2 = bb.astype(int); cx,cy = (x1+x2)//2, (y1+y2)//2
bw,bh = x2-x1, y2-y1
x1e = max(0, cx - int(bw*1.5)); y1e = max(0, cy - int(bh*1.5))
x2e = min(w, cx + int(bw*1.5)); y2e = min(h, cy + int(bh*1.5))
m_roi = np.zeros((h,w), dtype=np.uint8); m_roi[y1e:y2e, x1e:x2e] = 1
m_final = m_expanded | m_roi
for i in range(N): masks[i] = m_final
print(f'M4: mask {m_final.sum()}px ({m_final.sum()/(h*w)*100:.1f}%)')

# === M5: Depth — every 10th frame for temporal diversity ===
from modules.depth_estimator import DepthAnythingV2Estimator
de=DepthAnythingV2Estimator(model_size='vits',model_path='E:/zhijiyige/weights/depth_anything_v2/depth_anything_v2_vits.pth')
rd = np.memmap(f'{O}/depths_rel.dat', dtype=np.float16, mode='w+', shape=(N,h,w))
step = 10
for i in range(0, N, step):
    rd[i] = de.estimate(frames[i])
    if i % 100 == 0: print(f'[Depth] Frame {i}/{N}')
# Fill gaps with nearest
for i in range(N):
    if i % step != 0:
        base = (i // step) * step
        next_f = min(base + step, N-1)
        rd[i] = rd[base] if next_f >= N else rd[base]  # nearest-neighbor
de.unload(); del de; torch.cuda.empty_cache()
print(f'M5: {rd.shape} (every {step}th frame)')

# === M6: TripoSR mesh ===
from modules.triposr_mesh_generator import TripoSRMeshGenerator
t=TripoSRMeshGenerator(mc_resolution=128,output_dir=f'{O}/meshes')
mp,mi=t.generate(cv2.cvtColor(frames[0],cv2.COLOR_BGR2RGB),masks[0])
t.unload(); del t; torch.cuda.empty_cache()
print(f'M6: {mi["source"]} {mi["vertices"]}v/{mi["faces"]}f')

# === M7: Scale recovery ===
from modules.depth_scale_recovery import DepthScaleRecovery
sr=DepthScaleRecovery(method='triposr_bbox',heuristic_depth_range_mm=500.0)
md=sr.recover_batch(rd,masks,f'{O}/depths_metric.dat')
print(f'M7: [{md[0].min():.0f},{md[0].max():.0f}]mm')

# === Heuristic K (improved) ===
# f = max(w,h) * 0.6 gives f=288 — reasonable for ~65° HFOV webcam
f_improved = max(w, h) * 0.6
K_improved = np.array([[f_improved, 0, w/2], [0, f_improved, h/2], [0, 0, 1]])
print(f'\n[K] Improved heuristic: f={f_improved:.0f} (old plan default was {max(w,h)*1.1:.0f})')
print(f'[K] Expected improvement: projection error {max(w,h)*1.1/f_improved:.1f}x → 1.0x (vs old {max(w,h)*1.1/f_improved:.1f}x over)')
np.savetxt(f'{O}/K_improved.txt', K_improved)
print(f'[K] Saved to {O}/K_improved.txt')

print(f'\nALL DONE. VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
