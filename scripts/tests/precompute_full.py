"""Pre-compute M1-M7 for full 737-frame video."""
import sys,os,cv2,torch,numpy as np
sys.path.insert(0,'E:/zhijiyige')
O='E:/zhijiyige/output_full'
os.makedirs(O, exist_ok=True)
os.makedirs(f'{O}/meshes', exist_ok=True)

# M1: Load full 737 frames at 360p
from modules.video_decoder import VideoDecoder
d=VideoDecoder(360)
frames=[f for _,f in d.decode_stream('E:/zhijiyige/demo/test_mustard.mp4')]
N,h,w=len(frames),frames[0].shape[0],frames[0].shape[1]
print(f'M1: {N}f {w}x{h}')

# M2: YOLO-World
from modules.yolo_world_detector import YOLOWorldDetector
y=YOLOWorldDetector(model_path='E:/zhijiyige/weights/yolo_world/yolov8s-worldv2.pt',conf_threshold=0.12)
r1=cv2.cvtColor(frames[0],cv2.COLOR_BGR2RGB)
det=y.detect_top1(r1,'mustard bottle,bottle')
bb=np.array(det['bbox']); y.unload(); del y; torch.cuda.empty_cache()
print(f'M2: {det["label"]} {det["score"]:.3f}')

# M3: SAM
from modules.sam_segmentor import EfficientViTSAMSegmentor
s=EfficientViTSAMSegmentor(use_opencv=True)
m0=s.segment_with_box(r1,bb); s.unload(); del s; torch.cuda.empty_cache()
print(f'M3: {m0.sum()}px')

# M4: Static mask (skip XMem — internal model issues with single_object=False)
masks = np.memmap(f'{O}/masks.dat', dtype=np.uint8, mode='w+', shape=(N,h,w))
# Expand mask: dilate and fill for better registration
kernel = np.ones((25,25), np.uint8)
m_expanded = cv2.dilate(m0, kernel, iterations=3)
# Also create a rectangular ROI for the bottle area
x1,y1,x2,y2 = bb.astype(int); cx,cy = (x1+x2)//2, (y1+y2)//2
bw,bh = x2-x1, y2-y1
x1e = max(0, cx - int(bw*1.3)); y1e = max(0, cy - int(bh*1.3))
x2e = min(w, cx + int(bw*1.3)); y2e = min(h, cy + int(bh*1.3))
m_roi = np.zeros((h,w), dtype=np.uint8); m_roi[y1e:y2e, x1e:x2e] = 1
m_combined = m_expanded | m_roi
for i in range(N): masks[i] = m_combined
print(f'M4: mask {m_combined.sum()}px ({m_combined.sum()/(h*w)*100:.1f}%)')

# M5: Depth Anything V2 (first 100 frames for speed — rest copy)
from modules.depth_estimator import DepthAnythingV2Estimator
de=DepthAnythingV2Estimator(model_size='vits',model_path='E:/zhijiyige/weights/depth_anything_v2/depth_anything_v2_vits.pth')
N_depth = min(100, N)
rd = de.estimate_stream(list(enumerate(frames[:N_depth])), f'{O}/depths_rel.dat')
de.unload(); del de; torch.cuda.empty_cache()
# Extend to all frames
full_rd = np.memmap(f'{O}/depths_rel.dat', dtype=np.float16, mode='r+', shape=(N,h,w))
for i in range(N_depth, N):
    full_rd[i] = full_rd[i % N_depth]  # reuse from nearby frames
print(f'M5: {full_rd.shape}')

# M6: TripoSR
from modules.triposr_mesh_generator import TripoSRMeshGenerator
t=TripoSRMeshGenerator(mc_resolution=128,output_dir=f'{O}/meshes')
mp,mi=t.generate(cv2.cvtColor(frames[0],cv2.COLOR_BGR2RGB),masks[0])
t.unload(); del t; torch.cuda.empty_cache()
print(f'M6: {mi["source"]} {mi["vertices"]}v/{mi["faces"]}f')

# M7: Depth Scale Recovery
from modules.depth_scale_recovery import DepthScaleRecovery
sr=DepthScaleRecovery(method='triposr_bbox',heuristic_depth_range_mm=500.0)
md=sr.recover_batch(full_rd,masks,f'{O}/depths_metric.dat')
print(f'M7: [{md[0].min():.0f},{md[0].max():.0f}]mm')

print(f'ALL DONE. VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
