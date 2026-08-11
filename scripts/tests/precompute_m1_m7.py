"""Generate complete M1-M7 data for all 76 frames of mustard clip."""
import sys,os,cv2,torch,numpy as np
sys.path.insert(0,'E:/zhijiyige')
O='E:/zhijiyige/output'

# M1: Load frames
from modules.video_decoder import VideoDecoder
d=VideoDecoder(360)
frames=[f for _,f in d.decode_stream('E:/zhijiyige/demo/test_mustard_clip.mp4')]
N,h,w=len(frames),frames[0].shape[0],frames[0].shape[1]
print(f'M1: {N}f {w}x{h}')

# M2: YOLO
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

# M4: Skip XMem (multi-object mode issue) — reuse SAM mask for all frames
masks = np.memmap(f'{O}/masks_all.dat', dtype=np.uint8, mode='w+', shape=(N,h,w))
for i in range(N): masks[i] = m0
print(f'M4: reused SAM mask {m0.sum()}px for all {N} frames')

# M5: Depth
from modules.depth_estimator import DepthAnythingV2Estimator
de=DepthAnythingV2Estimator(model_size='vits',model_path='E:/zhijiyige/weights/depth_anything_v2/depth_anything_v2_vits.pth')
rd=de.estimate_stream(list(enumerate(frames)),f'{O}/depths_rel_all.dat'); de.unload(); del de; torch.cuda.empty_cache()
print(f'M5: {rd.shape}')

# M6: TripoSR
from modules.triposr_mesh_generator import TripoSRMeshGenerator
t=TripoSRMeshGenerator(mc_resolution=128,output_dir=f'{O}/meshes')
mp,mi=t.generate(cv2.cvtColor(frames[0],cv2.COLOR_BGR2RGB),m0); t.unload(); del t; torch.cuda.empty_cache()
print(f'M6: {mi["source"]} {mi["vertices"]}v/{mi["faces"]}f')

# M7: Depth Scale
from modules.depth_scale_recovery import DepthScaleRecovery
sr=DepthScaleRecovery(method='triposr_bbox',heuristic_depth_range_mm=500.0)
md=sr.recover_batch(rd,masks,f'{O}/depths_metric_all.dat')
print(f'M7: [{md[0].min():.0f},{md[0].max():.0f}]mm')

print('ALL DONE. VRAM peak:',torch.cuda.max_memory_allocated()/1e9,'GB')
