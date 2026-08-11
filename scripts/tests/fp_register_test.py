"""Quick FoundationPose registration test."""
import sys, cv2, numpy as np, torch, logging
logging.getLogger().setLevel(logging.WARNING)
sys.path.insert(0,'/mnt/e/zhijiyige/src/nvdiffrast/nvdiffrast-main')
sys.path.insert(0,'/mnt/e/zhijiyige/src/FoundationPose')

import nvdiffrast.torch as dr
glctx = dr.RasterizeCudaContext()

from estimater import FoundationPose
import trimesh
mesh = trimesh.load("/mnt/e/zhijiyige/output/meshes/proxy_mesh.glb", force="mesh")
mesh.vertices = mesh.vertices.astype(np.float32) * 100.0
pts, fidx = trimesh.sample.sample_surface(mesh, 3000)
normals = mesh.face_normals[fidx]
fp = FoundationPose(model_pts=pts, model_normals=normals, mesh=mesh, glctx=glctx, debug=0)
fp.make_rotation_grid(min_n_views=6, inplane_step=180)
print(f"[FP] {len(fp.rot_grid)} rots, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

cap = cv2.VideoCapture("/mnt/e/zhijiyige/demo/test_mustard_clip.mp4")
frames, fps = [], cap.get(cv2.CAP_PROP_FPS)
while True:
    ret, f = cap.read()
    if not ret: break
    h,w = f.shape[:2]; s = 360/min(h,w)
    frames.append(cv2.resize(f, (int(w*s), int(h*s))))
cap.release()

depths_raw = np.memmap("/mnt/e/zhijiyige/output/depths_metric_all.dat", dtype=np.float16, mode="r", shape=(len(frames),360,480))
masks = np.memmap("/mnt/e/zhijiyige/output/masks_all.dat", dtype=np.uint8, mode="r", shape=(len(frames),360,480))
K = np.array([[528,0,240],[0,396,180],[0,0,1]], dtype=np.float64)
rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
depth = depths_raw[0].astype(np.float32) / 1000.0

print("[FP] Registering...")
torch.cuda.reset_peak_memory_stats()
pose = fp.register(K=K, rgb=rgb, depth=depth, ob_mask=masks[0], ob_id=0, iteration=2)
print(f"*** REGISTER DONE! ***")
print(f"pose:\n{pose}")
print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
