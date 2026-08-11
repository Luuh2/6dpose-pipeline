"""DA3 pipeline: metric depth + estimated K + SAM. Pure RGB, no GT."""
import sys,os,cv2,torch,numpy as np
sys.path.insert(0,'E:/zhijiyige')
OUT='E:/zhijiyige/output_da3'
os.makedirs(OUT,exist_ok=True);os.makedirs(f'{OUT}/meshes',exist_ok=True)
VIDEO='E:/zhijiyige/demo/test_mustard.mp4'
PROMPT='mustard bottle,bottle,condiment'

# === M1: Video ===
from modules.video_decoder import VideoDecoder
d=VideoDecoder(360)
frames=[f for _,f in d.decode_stream(VIDEO)]
N,h_proc,w_proc=len(frames),frames[0].shape[0],frames[0].shape[1]
print(f'M1: {N}f {w_proc}x{h_proc}')

# === M2: YOLO ===
from modules.yolo_world_detector import YOLOWorldDetector
y=YOLOWorldDetector(model_path='E:/zhijiyige/weights/yolo_world/yolov8s-worldv2.pt',conf_threshold=0.10)
best_f,best_score,best_bb=0,0,None
for fi in range(0,min(N,700),20):
    dets=y.detect(cv2.cvtColor(frames[fi],cv2.COLOR_BGR2RGB),PROMPT)
    for d_ in dets:
        bw,bh=d_['bbox'][2]-d_['bbox'][0],d_['bbox'][3]-d_['bbox'][1]
        s=d_['score']*min(bw*bh,5000)
        if s>best_score:best_score=s;best_f=fi;best_bb=d_['bbox']
bb=np.array(best_bb);y.unload();del y;torch.cuda.empty_cache()
print(f'M2: F{best_f} bbox={bb.astype(int)} sz={bb[2]-bb[0]:.0f}x{bb[3]-bb[1]:.0f}')

# === M3: SAM ===
from modules.sam_segmentor import EfficientViTSAMSegmentor
sam=EfficientViTSAMSegmentor()
m0=sam.segment_with_box(cv2.cvtColor(frames[best_f],cv2.COLOR_BGR2RGB),bb)
sam.unload();del sam;torch.cuda.empty_cache()
print(f'M3: SAM {m0.sum()}px ({m0.sum()/(h_proc*w_proc)*100:.1f}%)')

# === M4: Mask (expand SAM mask) ===
kernel=np.ones((10,10),np.uint8);m0=cv2.dilate(m0,kernel,iterations=2)
x1,y1,x2,y2=bb.astype(int);cx,cy=(x1+x2)//2,(y1+y2)//2
bw,bh=x2-x1,y2-y1
x1e=max(0,cx-int(bw*2));y1e=max(0,cy-int(bh*1.2))
x2e=min(w_proc,cx+int(bw*2));y2e=min(h_proc,cy+int(bh*1.2))
m_roi=np.zeros((h_proc,w_proc),np.uint8);m_roi[y1e:y2e,x1e:x2e]=1
m_final=m0|m_roi
masks=np.memmap(f'{OUT}/masks.dat',dtype=np.uint8,mode='w+',shape=(N,h_proc,w_proc))
for i in range(N):masks[i]=m_final
print(f'M4: mask {m_final.sum()}px ({m_final.sum()/(h_proc*w_proc)*100:.1f}%)')

# === M5: DA3 Metric Depth (replaces M5 old + M7 scale recovery) ===
from modules.depth_estimator import DepthEstimator
de=DepthEstimator()
depths,K_da3=de.estimate_da3_batch(frames,every_n=10)
de.unload();del de;torch.cuda.empty_cache()
# Save depths as metric (already in meters)
np.save(f'{OUT}/depths_metric.npy',depths)
print(f'M5: DA3 metric depth [{depths.min():.2f},{depths.max():.2f}]m')
print(f'     K_da3 estimated:\n{K_da3}')

# === M6: TripoSR ===
from modules.triposr_mesh_generator import TripoSRMeshGenerator
t=TripoSRMeshGenerator(mc_resolution=128,output_dir=f'{OUT}/meshes')
mp,mi=t.generate(cv2.cvtColor(frames[0],cv2.COLOR_BGR2RGB),masks[0])
t.unload();del t;torch.cuda.empty_cache()
print(f'M6: {mi["source"]} {mi["vertices"]}v/{mi["faces"]}f')

# === K: DA3 depth is metric, use heuristic K ===
K = np.array([[max(w_proc,h_proc)*0.866,0,w_proc/2],[0,max(w_proc,h_proc)*0.866,h_proc/2],[0,0,1]],dtype=np.float64)
print(f'[K] Heuristic K (DA3 depth is metric, no scale recovery needed)')
np.savetxt(f'{OUT}/K.txt',K)
print(f'[K]\n{K}')

print(f'\nM1-M7 DONE. VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB')
print(f'\n--- NEXT (WSL2) ---')
print(f'wsl -d Ubuntu -- bash -c \"export CUDA_HOME=/root/miniconda3/envs/rgb6d &&')
print(f'  /root/miniconda3/envs/rgb6d/bin/python /mnt/e/zhijiyige/run_da3_wsl.py\"')
