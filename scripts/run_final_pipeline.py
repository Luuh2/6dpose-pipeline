"""
FINAL pipeline: EfficientViT-SAM + DA2 vits + proper K + GrabCut fallback
Pure RGB input, no GT. Generalized for any video.
Run: python run_final_pipeline.py
"""
import sys,os,cv2,torch,numpy as np,math
sys.path.insert(0,'E:/zhijiyige')
OUT='E:/zhijiyige/output_final'
os.makedirs(OUT,exist_ok=True);os.makedirs(f'{OUT}/meshes',exist_ok=True)

VIDEO='E:/zhijiyige/demo/test_mustard.mp4'
PROMPT='mustard bottle,bottle,condiment'  # text prompt for YOLO

# ====== M1: Video Decode ======
from modules.video_decoder import VideoDecoder
d=VideoDecoder(360)
frames=[f for _,f in d.decode_stream(VIDEO)]
N=len(frames);h_proc,w_proc=frames[0].shape[0],frames[0].shape[1]
print(f'M1: {N}f {w_proc}x{h_proc}')

# ====== M2: YOLO (scan for best detection) ======
from modules.yolo_world_detector import YOLOWorldDetector
y=YOLOWorldDetector(model_path='E:/zhijiyige/weights/yolo_world/yolov8s-worldv2.pt',conf_threshold=0.10)
best_f,best_bb,best_score=0,None,0
for fi in range(0,min(N,700),20):
    dets=y.detect(cv2.cvtColor(frames[fi],cv2.COLOR_BGR2RGB),PROMPT)
    for d in dets:
        bw,bh=d['bbox'][2]-d['bbox'][0],d['bbox'][3]-d['bbox'][1]
        s=d['score']*min(bw*bh,5000)
        if s>best_score:best_score=s;best_f=fi;best_bb=d['bbox']
bb=np.array(best_bb);y.unload();del y;torch.cuda.empty_cache()
print(f'M2: frame{best_f} bbox={bb.astype(int)} size={bb[2]-bb[0]:.0f}x{bb[3]-bb[1]:.0f}px')

# ====== M3: SAM (EfficientViT with triton shim) ======
from modules.sam_segmentor import EfficientViTSAMSegmentor
sam=EfficientViTSAMSegmentor()
rgb_best=cv2.cvtColor(frames[best_f],cv2.COLOR_BGR2RGB)
m0=sam.segment_with_box(rgb_best,bb);sam.unload();del sam;torch.cuda.empty_cache()
print(f'M3: SAM mask={m0.sum()}px ({m0.sum()/(h_proc*w_proc)*100:.1f}%)')

# ====== M4: Expand mask for all frames ======
kernel=np.ones((10,10),np.uint8)
m0=cv2.dilate(m0,kernel,iterations=2)
x1,y1,x2,y2=bb.astype(int);cx,cy=(x1+x2)//2,(y1+y2)//2
bw,bh=x2-x1,y2-y1
x1e=max(0,cx-int(bw*2.0));y1e=max(0,cy-int(bh*1.2))
x2e=min(w_proc,cx+int(bw*2.0));y2e=min(h_proc,cy+int(bh*1.2))
m_roi=np.zeros((h_proc,w_proc),dtype=np.uint8);m_roi[y1e:y2e,x1e:x2e]=1
m_final=m0|m_roi
masks=np.memmap(f'{OUT}/masks.dat',dtype=np.uint8,mode='w+',shape=(N,h_proc,w_proc))
for i in range(N):masks[i]=m_final
print(f'M4: mask {m_final.sum()}px ({m_final.sum()/(h_proc*w_proc)*100:.1f}%)')

# ====== M5: Depth Anything V2 vits ======
from modules.depth_estimator import DepthAnythingV2Estimator
de=DepthAnythingV2Estimator(model_size='vits',model_path='E:/zhijiyige/weights/depth_anything_v2/depth_anything_v2_vits.pth')
rd=np.memmap(f'{OUT}/depths_rel.dat',dtype=np.float16,mode='w+',shape=(N,h_proc,w_proc))
step=5  # every 5th frame
for i in range(0,N,step):rd[i]=de.estimate(frames[i])
for i in range(N):
    if i%step!=0:base=(i//step)*step;rd[i]=rd[min(base,N-1)]
de.unload();del de;torch.cuda.empty_cache()
print(f'M5: depth every {step}th frame')

# ====== M6: TripoSR mesh ======
from modules.triposr_mesh_generator import TripoSRMeshGenerator
t=TripoSRMeshGenerator(mc_resolution=128,output_dir=f'{OUT}/meshes')
mp,mi=t.generate(cv2.cvtColor(frames[0],cv2.COLOR_BGR2RGB),masks[0])
t.unload();del t;torch.cuda.empty_cache()
print(f'M6: {mi["source"]} {mi["vertices"]}v/{mi["faces"]}f')

# ====== M7: Scale recovery ======
from modules.depth_scale_recovery import DepthScaleRecovery
sr=DepthScaleRecovery(method='triposr_bbox',heuristic_depth_range_mm=500.0)
md=sr.recover_batch(rd,masks,f'{OUT}/depths_metric.dat')
print(f'M7: [{md[0].min():.0f},{md[0].max():.0f}]mm')

# ====== K estimation: f = w / (2*tan(30deg)) = w*0.866 for 60deg HFOV ======
f_est = max(w_proc,h_proc) * 0.866
K = np.array([[f_est,0,w_proc/2],[0,f_est,h_proc/2],[0,0,1]],dtype=np.float64)
np.savetxt(f'{OUT}/K.txt',K)
print(f'\n[K] f={f_est:.0f}px (60deg HFOV assumption)')
print(f'[K] {K}')

print(f'\nM1-M7 COMPLETE. VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
print(f'\n--- NEXT: Run in WSL2 ---')
print(f'wsl -d Ubuntu -- bash -c "export CUDA_HOME=/root/miniconda3/envs/rgb6d &&')
print(f'  /root/miniconda3/envs/rgb6d/bin/python /mnt/e/zhijiyige/run_final_wsl.py"')
