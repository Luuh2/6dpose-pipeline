"""验证 scorer.predict 单 pose 评分可用性"""
import sys, numpy as np, torch, cv2, traceback
sys.path.insert(0, '/mnt/e/zhijiyige/src/nvdiffrast/nvdiffrast-main')
sys.path.insert(0, '/mnt/e/zhijiyige/src/FoundationPose')
sys.path.insert(0, '/mnt/e/zhijiyige')
import nvdiffrast.torch as dr
import trimesh
from estimater import FoundationPose
from Utils import set_logging_format, set_seed
set_logging_format(); set_seed(0)
glctx = dr.RasterizeCudaContext()
mesh = trimesh.load('/mnt/e/zhijiyige/output/meshes/proxy_mesh_aligned.glb', force='mesh')
mesh.vertices = mesh.vertices.astype(np.float32) / 1000.0
pts, fidx = trimesh.sample.sample_surface(mesh, 3000)
normals = mesh.face_normals[fidx].astype(np.float32)
fp = FoundationPose(model_pts=pts.astype(np.float32), model_normals=normals,
                    mesh=mesh, glctx=glctx, debug=0)
fp.make_rotation_grid(min_n_views=8, inplane_step=120)
print('fp.pose_last:', fp.pose_last, flush=True)

mask = np.zeros((360, 480), np.uint8); mask[200:260, 80:130] = 1
depth = np.zeros((360, 480), np.float32); depth[200:260, 80:130] = 0.8
rgb = np.zeros((360, 480, 3), np.uint8); rgb[200:260, 80:130] = [180, 180, 0]
K = np.array([[416, 0, 240], [0, 416, 180], [0, 0, 1]], np.float64)
pose_init = np.eye(4)
try:
    scores = fp.scorer.predict(
        rgb=rgb, depth=depth, K=K,
        ob_in_cams=pose_init[None],
        mesh=fp.mesh, mesh_tensors=None,
        glctx=glctx, mesh_diameter=fp.diameter)
    print('scorer.predict OK, scores:', np.asarray(scores.cpu()).reshape(-1), flush=True)
except Exception:
    traceback.print_exc()
    print('FAIL', flush=True)
