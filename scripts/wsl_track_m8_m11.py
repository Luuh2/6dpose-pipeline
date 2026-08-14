#!/usr/bin/env python3
"""
wsl_track_m8_m11.py — WSL2 端 FoundationPose 追踪 (M8-M11)

在 WSL2 内运行 (nvdiffrast + pytorch3d 需要 Linux CUDA):
  /root/miniconda3/envs/rgb6d/bin/python /mnt/e/zhijiyige/scripts/wsl_track_m8_m11.py

数据 (来自 Windows 端 M1-M7 预计算):
  /mnt/e/zhijiyige/output/
    intermediate/depths_metric.dat   # fp16 meters (每帧估计)
    intermediate/masks.dat           # uint8 (每30帧重检测的逐帧掩码)
    meshes/proxy_mesh_stream3d_mm.glb    # mm 单位 mesh (MeshDepthAligner 输出)
    K.npy                            # 相机内参

单位约定 (FoundationPose 要求一致):
  mesh:  mm → 加载后转 meters (/1000)
  depth: meters (DA3 输出, 直接用)
  K:     像素焦距

追踪策略 (修复跟丢):
  1. 每帧 track_one() 局部 RGB-D 优化 (实时深度跟随物体)
  2. 掩码引导重锚定: 每 REANCHOR_EVERY 帧, 用当前帧掩码质心+深度
     计算物体 3D 中心, 修正 pose 平移并重置 pose_last
  3. 避免 track_one 因遮挡/深度噪声累积漂移
"""
import sys, os, time, cv2, numpy as np, torch

# ── sys.path 设置 ──────────────────────────────────────────────────────
BASE = '/mnt/20T/xieyongling/zhijiyige'
sys.path.insert(0, f'{BASE}/src/nvdiffrast/nvdiffrast-main')
sys.path.insert(0, f'{BASE}/src/FoundationPose')
sys.path.insert(0, BASE)  # 以便 import modules/*

# ── 配置 ──────────────────────────────────────────────────────────────
OUT = f'{BASE}/output'
VIDEO = f'{BASE}/demo/test_mustard.mp4'
DEPTHS = f'{OUT}/intermediate/depths_metric.dat'
# 优先使用 XMem 时序掩码 (精确跟随物体), 失败时回退颜色检测
MASKS = f'{OUT}/intermediate/masks_xmem_full.dat'
MESH = f'{OUT}/meshes/proxy_mesh_stream3d_mm.glb'
K_PATH = f'{OUT}/K.npy'
# ── CLI 覆盖 (多场景测试) ──────────────────────────────────────────
import argparse as _argparse
_ap = _argparse.ArgumentParser()
_ap.add_argument('--out', default=None, help='输出目录 (相对 BASE, 如 output_driller)')
_ap.add_argument('--video', default=None, help='视频路径 (相对 BASE, 如 demo/driller.mp4)')
_ap.add_argument('--mesh', default=None, help='mesh 路径 (相对 BASE 或绝对)')
_ap.add_argument('--npts', type=int, default=None, help='覆盖 N_PTS (GPU 资源受限时降采样)')
_argv, _ = _ap.parse_known_args()
if _argv.out:
    OUT = f'{BASE}/{_argv.out}'
    DEPTHS = f'{OUT}/intermediate/depths_metric.dat'
    MASKS = f'{OUT}/intermediate/masks_xmem_full.dat'
    MESH = f'{OUT}/meshes/proxy_mesh_aligned.glb'
    K_PATH = f'{OUT}/K.npy'
if _argv.video:
    VIDEO = _argv.video if _argv.video.startswith('/') else f'{BASE}/{_argv.video}'
if _argv.mesh:
    MESH = _argv.mesh if _argv.mesh.startswith('/') else f'{BASE}/{_argv.mesh}'
N_PTS = 8000
if _argv.npts:
    N_PTS = _argv.npts
DEBUG = 0
REANCHOR_EVERY = 30     # 每 N 帧用掩码质心重锚定一次 (防漂移)
MIN_MASK_PX = 100       # 掩码质心有效的最小面积

# ── 颜色引导参数 (每帧修正, 无需逐帧 mask) ─────────────────────────
# 瓶子是低纹理黄色物体, track_one 的 RGB-D 优化无法积累正确平移.
# 用窄色相黄色检测瓶子 2D 质心, 每帧修正 pose 平移.
COLOR_GUIDE_ENABLED = True   # 启用颜色引导
HUE_LOW, HUE_HIGH = 17, 29   # 瓶子黄色窄色相范围 (已验证 H≈23)
SAT_MIN = 60
VAL_MIN = 60
COLOR_REF_AREA = 2000        # 首帧瓶子面积参考 (~2000px)
COLOR_MAX_SHIFT_PX = 15.0    # 单帧最大 2D 修正量 (px), 防跳变
COLOR_HISTORY = 5            # 质心平滑窗口
COLOR_FALLBACK_EVERY = 15    # 颜色检测失败时, 尝试一次大范围搜索的间隔

# ── 惯性主轴朝向参数 (修正翻滚角) ──────────────────────────────────
# track_one 无法分辨物体旋转 (单目深度噪声 + 低纹理).
# 用黄色连通域的惯性主轴角 (图像平面朝向) 每帧修正 pose 翻滚角 (roll).
AXIS_GUIDE_ENABLED = True    # 启用主轴朝向修正
AXIS_MIN_ELONGATION = 1.5    # mask 长宽比低于此值(端向/圆)跳过翻滚修正
AXIS_MAX_DELTA_DEG = 8.0     # 静止时单帧最大翻滚角修正 (度), 防跳变
AXIS_HISTORY = 5             # 主轴角平滑窗口 (帧)

# ── 主轴角预测 (消除转速上限瓶颈) ──────────────────────────────────
# 不要只依赖单帧 moments 主轴: 取最近 3 帧主轴角加权差分,
# 预测下一帧 roll 角, 给 refiner.predict 提供角度预判初值.
AXIS_PREDICT_ENABLED = True    # 启用主轴角预测
AXIS_PRED_HISTORY = 3          # 加权差分帧数 (3 帧)
AXIS_PRED_VEL_W0 = 1.0         # 速度外推系数 (1.0=完全外推无滞后)
AXIS_PRED_VEL_W1 = 0.4         # 加速度二阶修正权重 (预测转向提前量)
AXIS_PRED_MAX_DEG = 12.0       # 预测修正单帧上限 (度), 防止预测发散

# ── 初始化阶段优化 (首帧 register) ──────────────────────────────────
# 1) mask 屏蔽背景: 只把 XMem 掩码内部像素送入 refiner/scorer, 防背景干扰
INIT_MASK_BACKGROUND = True    # 首帧 register 前 mask 外像素置 0
# 2) 首帧惯性主轴 roll 约束: register 后立即用主轴角强制修正翻滚
INIT_AXIS_ROLL = True          # 从源头规避对称物体随机朝向
INIT_AXIS_MAX_DEG = 30.0       # 首帧 roll 修正限幅 (大, 因初始化可能有大幅偏差)

# ── 姿态置信度校验 ─────────────────────────────────────────────────
# 几何一致性置信度: 追踪 bbox 投影 vs XMem mask 的 IoU.
# ScoreNet 首帧已在 register 内打分; 单独调用 scorer 会额外加载 H5Dataset
# 导致 OOM/WSL 崩溃, 故用 IoU 反映追踪贴合度, 低于阈值触发重注册兜底.
CONFIDENCE_CHECK_ENABLED = True
CONFIDENCE_THRESHOLD = 0.22    # IoU 低于此值触发重注册 (追踪bbox是AABB含空隙, 正常IoU~0.3)
CONFIDENCE_CHECK_EVERY = 30    # 每 N 帧检查一次
CONFIDENCE_CONSEC = 2        # 连续 N 次低 IoU 才重注册 (防单帧抖动)

# ── 旋转一致性校验 (防止跳到对称歧义解) ────────────────────────────
# 物体旋转连续, 两帧间三轴方向不应突变. RefineNet 可能跳到对称歧义解
# (如近圆柱物体绕轴 180° 翻转外观几乎不变). 校验三轴夹角, 超限即舍弃.
ROT_CONSISTENCY_ENABLED = True
ROT_CONSISTENCY_MAX_DEG = 50.0    # 任一根本体轴与上帧夹角 > 此值 → 歧义解
ROT_CONSISTENCY_FALLBACK_ROT = True  # 舍弃时回退旋转, 保留当前平移

# ── ROI 深度降噪参数 ───────────────────────────────────────────────
# DA3 单目深度噪声大, 对瓶子 ROI 深度做中值+双边降噪
DEPTH_DENOISE_ENABLED = True
DEPTH_MEDIAN_KSIZE = 5       # 中值滤波核
DEPTH_BILATERAL_D = 5        # 双边滤波直径
DEPTH_BILATERAL_SIGMA = 3.0  # 双边滤波空间/色彩 sigma

# ── LIEKF 在线融合参数 ─────────────────────────────────────────────
# 将平移观测 (颜色质心) + 旋转观测 (主轴角) 接入 SE(3)-LIEKF 一同滤波
LIEKF_ENABLED = True
LIEKF_STATIC_V_THRESH = 0.2  # motion_level 低于此值=静止, 速度归零防漂移
LIEKF_PROCESS_POS = 0.005    # 位置过程噪声
LIEKF_PROCESS_ROT = 0.002    # 旋转过程噪声
LIEKF_MEAS_POS = 0.002       # 位置测量噪声 (颜色质心可靠, 较小)
LIEKF_MEAS_ROT = 0.01        # 旋转测量噪声 (主轴角有歧义, 较大)

# ── 自适应噪声参数 (解决卡尔曼滞后) ────────────────────────────────
# 不动就磨平噪声, 一动就放开枷锁
LIEKF_ADAPTIVE = True        # 启用自适应噪声
MOTION_AXIS_THRESH_DEG = 3.0    # 主轴角变化 > 此值 (度) 视为运动
MOTION_CENTER_THRESH_PX = 3.0   # 掩码质心位移 > 此值 (px) 视为运动
MOTION_SMOOTH_ALPHA = 0.3       # 运动级别 EMA 平滑系数 (防突变)
# 运动时动态调节:
AXIS_DELTA_DEG_MOTION = 18.0    # 运动时 roll 限幅放宽到 (度)
AXIS_DELTA_DEG_STATIC = 8.0     # 静止时 roll 限幅 (度)
EMA_MOTION_WEIGHT = 0.4         # 运动时主轴 EMA 权重 (调低 → 快响应)
EMA_STATIC_WEIGHT = 0.25        # 静止时主轴 EMA 权重 (调高 → 强平滑)

# ── 1. 加载视频帧 (按数据实际分辨率, 不再硬编码 360p) ──────────────
print('[1] Loading frames...')
cap = cv2.VideoCapture(VIDEO)
native = None
frames = []
fps = cap.get(cv2.CAP_PROP_FPS)
while True:
    ret, f = cap.read()
    if not ret:
        break
    if native is None:
        native = f.shape[:2]
    frames.append(f)
cap.release()
n = len(frames)
# 数据分辨率: 从 memmap 文件大小推断 (masks uint8, depths fp16)
import math
masks_px = os.path.getsize(MASKS) // n           # H*W
nh, nw = native
ratio = nw / nh
Wp = int(round(math.sqrt(masks_px * ratio)))     # 保持宽高比
Hp = masks_px // Wp
hp, wp = Hp, Wp
print(f'  {n}f, native={nw}x{nh}, data={wp}x{hp} @ {fps:.1f}fps')
# 帧缩放到数据分辨率
if (nh, nw) != (hp, wp):
    frames = [cv2.resize(f, (wp, hp)) for f in frames]

# ── 2. 加载预计算数据 ───────────────────────────────────────────────
print('[2] Loading precomputed data...')
depths_raw = np.memmap(DEPTHS, dtype=np.float16, mode='r', shape=(n, hp, wp))
masks = np.memmap(MASKS, dtype=np.uint8, mode='r', shape=(n, hp, wp))
# depth: meters → mm (与 mesh 单位一致)
# depth 直接用 meters (DA3 输出单位), 与 mesh 的 meters 一致
depths_m = depths_raw[:].astype(np.float32)
del depths_raw
K = np.load(K_PATH)
print(f'  depth range: {depths_m.min():.3f}-{depths_m.max():.3f}m')
print(f'  mask frame0: {masks[0].sum()}px')
print(f'  K:\n{K}')

# ── 3. 初始化 nvdiffrast + FoundationPose ───────────────────────────
print('[3] Initializing nvdiffrast + FoundationPose...')
import nvdiffrast.torch as dr
import trimesh
from estimater import FoundationPose
from Utils import set_logging_format, set_seed

set_logging_format()
set_seed(0)
glctx = dr.RasterizeCudaContext()

mesh = trimesh.load(MESH, force='mesh')
# FoundationPose 期望 mesh 和 depth 都用 meters 单位
# erode_depth 默认 zfar=100 会把 >100 的深度清 0, 所以必须用 meters
mesh.vertices = mesh.vertices.astype(np.float32) / 1000.0  # mm -> m
print(f'  mesh: {len(mesh.vertices)}v, bounds={mesh.bounds}')

# 顶点降采样 — FoundationPose 内部 compute_mesh_diameter 用 n_sample=10000
# 会生成 10000x10000 距离矩阵 (2.4GB), 顶点过多导致 OOM (SIGKILL)
if len(mesh.vertices) > 5000:
    target = 4000
    print(f'  [OOM-guard] simplifying mesh {len(mesh.vertices)}v -> {target}v...')
    import open3d as o3d
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(mesh.faces.astype(np.int32)))
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(target)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices).astype(np.float32),
        faces=np.asarray(o3d_mesh.triangles).astype(np.int64))
    print(f'  simplified to {len(mesh.vertices)}v')

# mesh 8 角点 (meters, 用于 bbox 轮廓对齐)
mn, mx = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
mesh_bbox_corners = np.array([
    [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
    [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
    [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
    [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
], dtype=np.float32)
print(f'  mesh bbox corners: {mesh_bbox_corners.shape}')
# ── 物体本体长轴 (bbox 最长边): 从固定网格一次性定义, 永久固定 ──
long_axis_idx = int(np.argmax(mx - mn))
_axis_names = ['X', 'Y', 'Z']
print(f'  [BodyFrame] 本体长轴 = {_axis_names[long_axis_idx]}'
      f' ({(mx - mn)[long_axis_idx] * 1000:.0f}mm), '
      f'轴修正将对齐此固定本体轴')

# 采样模型点+法线 (如官方 run_demo.py 用 mesh.vertices)
# 采样到 N_PTS 个点 (配准鲁棒性 + 内存平衡)
pts, face_idx = trimesh.sample.sample_surface(mesh, N_PTS)
normals = mesh.face_normals[face_idx]
pts = pts.astype(np.float32)
normals = normals.astype(np.float32)
print(f'  sampled {len(pts)} points')

# 使用默认 scorer/refiner (从 src/FoundationPose/weights 自动加载)
fp = FoundationPose(
    model_pts=pts, model_normals=normals, mesh=mesh,
    glctx=glctx, debug=DEBUG, debug_dir=os.path.join(OUT, "fp_debug"),
)
# 降低 rotation grid 以适配 6GB VRAM
fp.make_rotation_grid(min_n_views=8, inplane_step=120)
print(f'  rotation grid: {len(fp.rot_grid)} rots')
print(f'  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB')

# ── 4. 追踪 (track_one + 颜色质心 + 主轴朝向 + LIEKF 融合) ──────────
print(f'[4] Tracking {n} frames (color+axis guide, LIEKF fused)...')
poses, confs = [], []
t0 = time.time()
n_color_fix = 0
n_axis_fix = 0
n_fallback = 0

# ── ROI 深度降噪 (对瓶子区域深度做中值+双边滤波) ───────────────────
def denoise_depth_roi(depth, mask):
    """对深度图降噪: 只在 mask 膨胀区域内做中值滤波, 保留全局结构"""
    if not DEPTH_DENOISE_ENABLED:
        return depth
    d = depth.copy()
    # 中值滤波
    d_med = cv2.medianBlur(d, DEPTH_MEDIAN_KSIZE)
    # 只在 mask 附近混合 (保留非物体区域原始深度)
    kernel = np.ones((11, 11), np.uint8)
    roi = cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1)
    blend = np.where(roi > 0, d_med, d)
    return blend.astype(np.float32)

# ── 颜色引导: 检测瓶子 2D 质心 + 惯性主轴 (窄色相黄色 + 深度 + 面积) ─
def detect_bottle_2d(frame_bgr, depth, prev_center, ref_area, search_radius=120):
    """在 prev_center 附近搜索黄色瓶子, 返回 (质心, 面积, 主轴角) or None"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (HUE_LOW, SAT_MIN, VAL_MIN), (HUE_HIGH, 255, 255))
    d = depth
    dm = (d > 0.5) & (d < 1.3)
    mm = ((m > 0) & dm).astype(np.uint8)
    mm[:, 350:] = 0  # 排除右侧背景黄色
    mm = cv2.erode(mm, np.ones((3, 3), np.uint8))
    mm = cv2.dilate(mm, np.ones((7, 7), np.uint8))

    # 搜索窗口: 以 prev_center 为中心
    H, W = mm.shape
    if prev_center is not None:
        win = np.zeros_like(mm)
        x1 = max(0, int(prev_center[0] - search_radius))
        y1 = max(0, int(prev_center[1] - search_radius))
        x2 = min(W, int(prev_center[0] + search_radius))
        y2 = min(H, int(prev_center[1] + search_radius))
        win[y1:y2, x1:x2] = 255
        mm = mm & win

    n, labels, stats, cent = cv2.connectedComponentsWithStats(mm, connectivity=8)
    if n <= 1:
        return None
    cands = []
    for i in range(1, n):
        area = stats[i, 4]
        if 500 < area < 10000:
            cands.append((area, i))
    if not cands:
        return None
    # 选面积最接近参考面积 的连通域 (瓶子面积稳定)
    cands.sort(key=lambda t: abs(t[0] - ref_area))
    best = cands[0][1]
    bmask = (labels == best).astype(np.uint8)

    # 惯性主轴角 (图像平面朝向, cv2.moments 二阶中心矩)
    axis_angle = 0.0
    moments = cv2.moments(bmask)
    if moments['mu20'] + moments['mu02'] > 1e-6:
        axis_angle = 0.5 * np.arctan2(
            2 * moments['mu11'], moments['mu20'] - moments['mu02'])

    return cent[best], stats[best, 4], axis_angle

# 平滑历史
cent_hist = []
axis_hist = []

def smooth_center(c):
    cent_hist.append(c)
    if len(cent_hist) > COLOR_HISTORY:
        cent_hist.pop(0)
    return np.mean(cent_hist, axis=0)

def smooth_axis(a, motion_level=0.0):
    """平滑主轴角 (处理 ±π 环绕)

    Args:
        a: 当前主轴角
        motion_level: [0,1] 运动强度. 运动时降低历史权重(快响应), 静止时提高(强平滑).
    """
    axis_hist.append(a)
    if len(axis_hist) > AXIS_HISTORY:
        axis_hist.pop(0)
    angles = np.array(axis_hist)
    # 权重: 越新权重越高; 运动时加大最新权重(减弱惯性)
    w = EMA_STATIC_WEIGHT + (EMA_MOTION_WEIGHT - EMA_STATIC_WEIGHT) * motion_level
    weights = np.exp(w * np.arange(len(angles)))  # 指数权重, 最新帧权重最大
    weights = weights / weights.sum()
    return np.arctan2(np.sum(weights * np.sin(angles)),
                      np.sum(weights * np.cos(angles)))

# ── 主轴角预测 (最近 3 帧加权差分, 预测下一帧 roll) ────────────────
axis_pred_hist = []   # 主轴角历史 (用于差分预测)

def wrap_angle_diff(a, b):
    """环绕归一化角度差 a-b -> [-pi, pi]"""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi

def predict_next_axis():
    """用最近 3 帧主轴角加权差分预测下一帧主轴角

    方法: 速度 = 最近帧角差; 加速度 = 最近两段角差之差.
    预测角 = 当前角 + w0*速度 + w1*加速度 (即线性外推 + 二阶修正).
    """
    if len(axis_pred_hist) < 2:
        return axis_pred_hist[-1] if axis_pred_hist else None
    # 各段角速度 (环绕归一化)
    a0 = axis_pred_hist[-1]
    v_last = wrap_angle_diff(axis_pred_hist[-1], axis_pred_hist[-2])  # 最近速度
    if len(axis_pred_hist) >= 3:
        v_prev = wrap_angle_diff(axis_pred_hist[-2], axis_pred_hist[-3])  # 上一段速度
    else:
        v_prev = v_last
    # 二阶差分 (加速度项)
    accel = wrap_angle_diff(v_last, v_prev)
    # 加权差分预测: 速度外推 + 加速度二阶修正
    #   pred = a0 + W0*v_last + W1*accel
    # 匀速时 accel=0 → pred = a0 + W0*v (W0=1 完全外推无滞后)
    # 加速时 accel 补充二阶趋势, 提前预判转向
    pred = a0 + AXIS_PRED_VEL_W0 * v_last + AXIS_PRED_VEL_W1 * accel
    # 归一化到 [-pi, pi]
    pred = (pred + np.pi) % (2 * np.pi) - np.pi
    return pred

def update_axis_pred(axis_angle):
    """更新预测历史, 返回预测角"""
    axis_pred_hist.append(axis_angle)
    if len(axis_pred_hist) > AXIS_PRED_HISTORY:
        axis_pred_hist.pop(0)
    return predict_next_axis()

def apply_axis_pred_to_pose(pose, pred_axis, K, motion_level=0.0):
    """将预测的 roll 角应用到 pose (作为 refiner.predict 初值预判)

    用预测主轴角修正 pose 翻滚, 让 refiner 从更接近真实旋转的角度开始.
    """
    if pred_axis is None:
        return pose
    R = pose[:3, :3].copy()
    # 当前 mesh 本体长轴在图像平面的方向 (固定本体轴)
    x_axis_cam = R[:, long_axis_idx]
    cur_alpha = np.arctan2(x_axis_cam[1], x_axis_cam[0])
    # 预测修正量 (环绕归一化, 限幅防发散)
    delta = wrap_angle_diff(pred_axis, cur_alpha)
    max_pred = np.radians(AXIS_PRED_MAX_DEG)
    delta = np.clip(delta, -max_pred, max_pred)
    if abs(delta) < 1e-4:
        return pose
    cz, sz = np.cos(delta), np.sin(delta)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    pose[:3, :3] = Rz @ R
    return pose

# ── 旋转一致性校验 (三轴夹角, 检测对称歧义解) ─────────────────────
def compute_axis_angle_diff(pose_a, pose_b):
    """计算两个 pose 的物体三轴 (X/Y/Z) 在相机系的方向夹角 (度)

    Args:
        pose_a, pose_b: 4x4 [R|t], 物体在相机系

    Returns:
        (3,) 数组: 三轴方向夹角 [deg]
    """
    Ra, Rb = pose_a[:3, :3], pose_b[:3, :3]
    angles = []
    for i in range(3):
        # 物体 i 轴在相机系方向 (R 的列)
        va, vb = Ra[:, i], Rb[:, i]
        cos_ang = np.clip(np.dot(va, vb), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_ang)))
    return np.array(angles)

def detect_symmetry_ambiguity(pose_current, pose_prev, max_deg=50.0):
    """检测 RefineNet 输出是否跳到对称歧义解

    物体旋转连续, 两帧间任一轴夹角不应突变 > max_deg.
    若超限, 判定为对称歧义解 (如近圆柱物体绕轴翻转外观不变).
    Returns: (is_ambiguous, max_angle_deg)
    """
    ang = compute_axis_angle_diff(pose_current, pose_prev)
    max_ang = float(ang.max())
    return (max_ang > max_deg), max_ang

def fallback_rotation(pose_current, pose_prev):
    """舍弃 RefineNet 旋转跳变, 回退到上一帧稳定旋转

    保留当前平移 (平移来自 mask 引导, 可靠), 旋转回退到上一帧.
    这样避免对称歧义解, 同时不丢失平移跟踪.
    """
    pose_out = pose_current.copy()
    pose_out[:3, :3] = pose_prev[:3, :3]  # 旋转回退
    return pose_out

def apply_2d_pose_fix(pose, target_2d, depth, K):
    """将 pose 平移在 2D 投影上向 target_2d 移动 (限幅)"""
    cur_3d = pose[:3, 3].copy()
    p2d = (K @ cur_3d)
    if p2d[2] < 0.01:
        return pose
    cur_2d = p2d[:2] / p2d[2]
    du, dv = target_2d[0] - cur_2d[0], target_2d[1] - cur_2d[1]
    mag = np.hypot(du, dv)
    if mag > COLOR_MAX_SHIFT_PX:
        du, dv = du * COLOR_MAX_SHIFT_PX / mag, dv * COLOR_MAX_SHIFT_PX / mag
    z = cur_3d[2]
    target_2d_capped = cur_2d + np.array([du, dv])
    target_3d = np.array([
        (target_2d_capped[0] - K[0, 2]) * z / K[0, 0],
        (target_2d_capped[1] - K[1, 2]) * z / K[1, 1],
        z,
    ])
    pose[0, 3], pose[1, 3] = target_3d[0], target_3d[1]
    return pose

def apply_axis_pose_fix(pose, target_axis, K, motion_level=0.0):
    """将 pose 翻滚角 (roll, 绕相机 z 轴) 对齐到目标主轴角

    pose 是 [R|t], 物体在相机系.
    主轴角是图像平面朝向, 主要对应绕相机 z 轴的翻滚 (roll).
    将 R 绕相机 z 轴旋转 delta, 使 mesh 主轴投影与 mask 主轴对齐.
    运动时放宽限幅上限 (放开枷锁), 静止时收紧 (防跳变).
    """
    R = pose[:3, :3].copy()

    # 当前 mesh 本体长轴在图像平面的投影方向 (固定本体轴)
    # 用 pose 的本体长轴列 (固定网格定义) 的 2D 投影
    x_axis_cam = R[:, long_axis_idx]  # 物体本体长轴在相机系
    cur_alpha = np.arctan2(x_axis_cam[1], x_axis_cam[0])

    # 目标: 将 cur_alpha 旋转到 target_axis
    delta = target_axis - cur_alpha
    # 归一化到 [-pi, pi]
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    # 运动时放宽限幅: 静止 8°/帧 → 运动 18°/帧
    max_delta = AXIS_DELTA_DEG_STATIC + \
        (AXIS_DELTA_DEG_MOTION - AXIS_DELTA_DEG_STATIC) * motion_level
    delta = np.clip(delta, -np.radians(max_delta), np.radians(max_delta))

    if abs(delta) < 1e-4:
        return pose

    # 绕相机 z 轴旋转 delta (左乘 Rz)
    cz, sz = np.cos(delta), np.sin(delta)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    pose[:3, :3] = Rz @ R
    return pose

def sync_pose_last(fp_obj, pose):
    """修正 pose 后同步 FoundationPose 内部 pose_last"""
    centered = pose @ np.linalg.inv(fp_obj.get_tf_to_centered_mesh().cpu().numpy())
    fp_obj.pose_last = torch.as_tensor(centered, device='cuda', dtype=torch.float)

# ── XMem 掩码引导: 用掩码计算质心 + 惯性主轴 ───────────────────────
def mask_centroid_and_axis(mask):
    """从二值 mask 计算质心和惯性主轴角"""
    if mask.sum() < MIN_MASK_PX:
        return None, None
    ys, xs = np.where(mask > 0)
    c2d = np.array([xs.mean(), ys.mean()])
    moments = cv2.moments(mask.astype(np.uint8))
    axis_angle = 0.0
    if moments['mu20'] + moments['mu02'] > 1e-6:
        axis_angle = 0.5 * np.arctan2(
            2 * moments['mu11'], moments['mu20'] - moments['mu02'])
    return c2d, axis_angle

def mask_elongation(mask):
    """mask 长宽比 (主轴/垂直轴 std). 细长→可靠; 近圆→端向不可靠."""
    m = mask.astype(np.uint8)
    mom = cv2.moments(m)
    if mom['mu20'] + mom['mu02'] < 1e-6:
        return 1.0
    ang = 0.5 * np.arctan2(2 * mom['mu11'], mom['mu20'] - mom['mu02'])
    u = np.array([np.cos(ang), np.sin(ang)]); v = np.array([-u[1], u[0]])
    ys, xs = np.where(m > 0)
    if len(ys) < 50:
        return 1.0
    cx = mom['m10']/mom['m00']; cy = mom['m01']/mom['m00']
    pts = np.stack([xs-cx, ys-cy], 1)
    sm = float(np.std(pts @ u)); ss = float(np.std(pts @ v))
    return sm/ss if ss > 1e-6 else 1.0

def apply_bbox_align_pose_fix(pose, mask, depth, K, corners, max_shift_px=15.0):
    """将 pose 的 2D 投影 bbox 中心对齐到 mask 的 2D bbox 中心

    轮廓约束的轻量实现: 让 mesh 投影包围盒与 mask 包围盒在图像平面重合.
    """
    # mask bbox 中心
    ys, xs = np.where(mask > 0)
    if len(ys) < MIN_MASK_PX:
        return pose
    mask_center = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])

    # 当前 pose 投影 bbox 中心 (用 mesh 8 角点投影)
    if corners.shape != (8, 3):
        print(f'[WARN] corners bad shape: {corners.shape}')
        corners = np.asarray(corners, dtype=np.float32).reshape(8, 3)
    corners_h = np.hstack([corners, np.ones((8, 1))])
    cam = (pose @ corners_h.T).T[:, :3]
    img = (K @ cam.T).T
    img = img[:, :2] / img[:, 2:3]
    pose_center = np.array([img[:, 0].mean(), img[:, 1].mean()])

    # 2D 偏移 (限幅)
    du, dv = mask_center[0] - pose_center[0], mask_center[1] - pose_center[1]
    mag = np.hypot(du, dv)
    if mag > max_shift_px:
        du, dv = du * max_shift_px / mag, dv * max_shift_px / mag
    # 反投影到 3D 平移修正
    z = pose[2, 3]
    pose[0, 3] += du * z / K[0, 0]
    pose[1, 3] += dv * z / K[1, 1]
    return pose

# mesh 8 角点 (用于 bbox 对齐; 在 mesh 加载后计算, 见上方赋值)

# ── 初始化 LIEKF 在线滤波 ───────────────────────────────────────────
from modules.se3_kalman_filter import SE3LieKalmanFilter
kf = None
if LIEKF_ENABLED:
    kf = SE3LieKalmanFilter(
        dt=1.0 / fps,
        process_noise_pos=LIEKF_PROCESS_POS,
        process_noise_rot=LIEKF_PROCESS_ROT,
        measurement_noise_pos=LIEKF_MEAS_POS,
        measurement_noise_rot=LIEKF_MEAS_ROT,
    )

prev_bottle_center = None  # 上一帧瓶子 2D 质心
prev_xmem_axis = None      # 上一帧主轴角 (运动检测)
motion_level = 0.0         # 当前运动强度 [0,1]
prev_pose_stable = None    # 上一帧稳定姿态 (旋转一致性校验基准)
n_rot_reject = 0           # 对称歧义解被拒绝次数
n_flip_fix = 0              # 翻转守护修正次数
_low_conf_count = 0          # 连续低 IoU 计数

for i in range(n):
    rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
    depth = depths_m[i]
    mask = masks[i].astype(np.uint8)

    if i == 0:
        # ── ① mask 屏蔽背景: 只把 XMem 掩码内部像素送入 refiner/scorer ──
        # register 内部 refiner.predict/scorer.predict 接收整幅图像, 背景会干扰
        # 初始旋转求解. 将 mask 外像素置 0, 网络只看到物体内部.
        if INIT_MASK_BACKGROUND:
            mask_bool = (mask > 0)
            rgb_reg = np.where(mask_bool[..., None], rgb, 0).astype(np.uint8)
            depth_reg = np.where(mask_bool, depth, 0.0).astype(np.float32)
            pose = fp.register(K=K, rgb=rgb_reg, depth=depth_reg,
                               ob_mask=mask, ob_id=0, iteration=2)
            print('  [Init] mask 背景屏蔽: 仅物体内部像素入网络')
        else:
            pose = fp.register(K=K, rgb=rgb, depth=depth,
                               ob_mask=mask, ob_id=0, iteration=2)

        # ── ② 首帧惯性主轴 roll 约束: 从源头规避对称物体随机朝向 ──
        if INIT_AXIS_ROLL:
            _, xmem_axis0 = mask_centroid_and_axis(mask)
            if xmem_axis0 is not None:
                # 记录主轴历史 (供后续预测)
                if AXIS_PREDICT_ENABLED:
                    update_axis_pred(xmem_axis0)
                # 用较大限幅强制修正 roll (初始化可能有大幅偏差)
                pose_before = pose.copy()
                # 暂存 motion 级 = 1 (放宽限幅) 强制拉正
                axis_s0 = smooth_axis(xmem_axis0, 1.0)
                pose = apply_axis_pose_fix(pose, axis_s0, K, 1.0)
                print(f'  [Init] 主轴 roll 约束: 修正 '
                      f'{np.degrees(np.arccos(np.clip(np.trace(pose[:3,:3]@pose_before[:3,:3].T)-1, -1, 1)/2)):.1f}°')

        conf = 1.0
        # 固化: 初始化 register 阶段永久关闭掩码质心修正.
        # register 内部已用 mask 做全局搜索 (guess_translation), 质心修正是
        # 冗余约束, 真值深度下反而降低精度 (BOP 实验 D2 vs D4: 34.9→39.3mm).
        # 质心/bbox 修正仅在后续时序跟踪阶段 (i>0) 启用.
        # 用首帧 mask 质心初始化颜色追踪
        ys, xs = np.where(mask > 0)
        if len(ys) > MIN_MASK_PX:
            prev_bottle_center = np.array([xs.mean(), ys.mean()])
        # 初始化 LIEKF (用修正后的 pose)
        if kf is not None:
            kf.initialize(pose)
        sync_pose_last(fp, pose)
        # 首帧稳定姿态 (旋转一致性校验基准)
        prev_pose_stable = pose.copy()
    else:
        # 用降噪后的深度
        depth_den = denoise_depth_roi(depth, mask)
        # ── 主轴角预测: 预偏 pose_last, 给 refiner.predict 初值预判 ──
        # track_one 内部用 fp.pose_last 作为 refiner 的 ob_in_cams 初值.
        # 先用 3 帧加权差分预测的 roll 预旋转 pose_last, refiner 从接近真实的角度开始.
        if AXIS_PREDICT_ENABLED and len(axis_pred_hist) >= 2:
            pred_axis = predict_next_axis()
            if pred_axis is not None:
                # 从 pose_last (centered) 构造完整 pose, 预偏 roll
                pose_last_centered = fp.pose_last.detach().cpu().numpy()
                pose_for_pred = pose_last_centered @ np.linalg.inv(
                    fp.get_tf_to_centered_mesh().cpu().numpy())
                pose_for_pred = apply_axis_pred_to_pose(
                    pose_for_pred, pred_axis, K, motion_level)
                # 同步回 pose_last (centered)
                fp.pose_last = torch.as_tensor(
                    pose_for_pred @ fp.get_tf_to_centered_mesh().cpu().numpy(),
                    device='cuda', dtype=torch.float)
        pose = fp.track_one(rgb=rgb, depth=depth_den, K=K, iteration=1)
        conf = 0.8

        # ── 旋转一致性校验: 检测 RefineNet 是否跳到对称歧义解 ──
        if ROT_CONSISTENCY_ENABLED and prev_pose_stable is not None:
            is_ambig, max_ang = detect_symmetry_ambiguity(
                pose, prev_pose_stable, ROT_CONSISTENCY_MAX_DEG)
            if is_ambig:
                n_rot_reject += 1
                print(f'  [Rot@{i}] 轴夹角{max_ang:.0f}° > '
                      f'{ROT_CONSISTENCY_MAX_DEG:.0f}°, 对称歧义解被拒, 回退上帧旋转')
                # 回退旋转到上一帧稳定姿态 (保留 mask 引导的平移)
                pose = fallback_rotation(pose, prev_pose_stable)
                # 重置 pose_last: 让下一帧 refiner 从稳定旋转开始小范围搜索
                sync_pose_last(fp, pose)

    # ── 运动状态检测 (主轴角变化 + 掩码质心位移) ──
    if i > 0:
        xmem_c2d_now, xmem_axis_now = mask_centroid_and_axis(mask)
        # 主轴角变化 (环绕归一化)
        d_axis = 0.0
        if xmem_axis_now is not None and prev_xmem_axis is not None:
            diff = xmem_axis_now - prev_xmem_axis
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            d_axis = abs(np.degrees(diff))
        # 质心位移
        d_center = 0.0
        if xmem_c2d_now is not None and prev_bottle_center is not None:
            d_center = np.hypot(xmem_c2d_now[0] - prev_bottle_center[0],
                                xmem_c2d_now[1] - prev_bottle_center[1])
        # 归一化运动得分
        motion_score = max(
            d_axis / MOTION_AXIS_THRESH_DEG,
            d_center / MOTION_CENTER_THRESH_PX,
        )
        motion_score = float(np.clip(motion_score / 2.0, 0.0, 1.0))  # 超过阈值2倍才饱和
        # EMA 平滑运动级别 (防突变)
        motion_level = (1 - MOTION_SMOOTH_ALPHA) * motion_level + \
            MOTION_SMOOTH_ALPHA * motion_score
        # 自适应 LIEKF 噪声
        if kf is not None and LIEKF_ADAPTIVE:
            kf.set_motion(motion_level)
        # 更新上一帧状态
        if xmem_c2d_now is not None:
            prev_bottle_center = xmem_c2d_now
        if xmem_axis_now is not None:
            prev_xmem_axis = xmem_axis_now

    # ── 每帧引导修正 (XMem mask 优先, 颜色检测回退) ──
    if i > 0 and (COLOR_GUIDE_ENABLED or AXIS_GUIDE_ENABLED):
        # mask 长宽比: 端向(圆)时主轴不可靠, 用于门控翻滚修正 (防 180° 翻转)
        _mask_elong = mask_elongation(mask)
        # 1) 优先用 XMem 掩码 (精确轮廓) 计算质心 + 主轴
        xmem_c2d, xmem_axis = mask_centroid_and_axis(mask)
        if xmem_c2d is not None:
            # 平移: mask 质心 → pose 投影对齐 (轮廓约束)
            if COLOR_GUIDE_ENABLED:
                pose_before = pose.copy()
                pose = apply_bbox_align_pose_fix(pose, mask, depth, K, mesh_bbox_corners)
                prev_bottle_center = xmem_c2d
                if np.max(np.abs(pose[:3, 3] - pose_before[:3, 3])) > 1e-7:
                    n_color_fix += 1
            # 翻滚: mask 惯性主轴 → pose 翻滚角 (门控: 端向跳过)
            if (AXIS_GUIDE_ENABLED and xmem_axis is not None
                    and _mask_elong >= AXIS_MIN_ELONGATION):
                axis_s = smooth_axis(xmem_axis, motion_level)
                pose_before = pose.copy()
                pose = apply_axis_pose_fix(pose, axis_s, K, motion_level)
                if np.max(np.abs(pose[:3, :3] - pose_before[:3, :3])) > 1e-6:
                    n_axis_fix += 1
                # 更新主轴角预测历史 (供下一帧 track_one 预判)
                if AXIS_PREDICT_ENABLED:
                    update_axis_pred(axis_s)
        else:
            # 2) XMem mask 不可用 → 颜色检测回退
            anchor = prev_bottle_center
            if anchor is None:
                p2d = K @ pose[:3, 3]
                anchor = p2d[:2] / p2d[2] if p2d[2] > 0.01 else np.array([240.0, 180.0])
            res = detect_bottle_2d(frames[i], depth, anchor, COLOR_REF_AREA)
            if res is not None:
                c2d, area, axis_angle = res
                if COLOR_GUIDE_ENABLED:
                    c2d_s = smooth_center(c2d)
                    pose = apply_2d_pose_fix(pose, c2d_s, depth, K)
                    prev_bottle_center = c2d_s
                    n_color_fix += 1
                if AXIS_GUIDE_ENABLED and _mask_elong >= AXIS_MIN_ELONGATION:
                    axis_s = smooth_axis(axis_angle, motion_level)
                    pose_before = pose.copy()
                    pose = apply_axis_pose_fix(pose, axis_s, K, motion_level)
                    if np.max(np.abs(pose[:3, :3] - pose_before[:3, :3])) > 1e-6:
                        n_axis_fix += 1
                    if AXIS_PREDICT_ENABLED:
                        update_axis_pred(axis_s)
            else:
                # 颜色检测也失败: 大范围搜索
                if i % COLOR_FALLBACK_EVERY == 0:
                    res2 = detect_bottle_2d(frames[i], depth, None, COLOR_REF_AREA,
                                            search_radius=250)
                    if res2 is not None:
                        c2d, area, _ = res2
                        prev_bottle_center = c2d
                        n_fallback += 1

    # ── SE(3)-LIEKF 在线融合 (平移 + 旋转观测) ──
    if i > 0 and kf is not None:
        if motion_level < LIEKF_STATIC_V_THRESH:
            kf.v = np.zeros(6)  # 静止: 速度归零, 停止积分, 杜绝滚转漂移
        kf.predict()
        kf.update(pose)  # pose 已含颜色质心 + 主轴观测修正
        pose = kf.X.copy()
        sync_pose_last(fp, pose)
    elif i > 0:
        sync_pose_last(fp, pose)
    # 直接用 scorer.predict 单独评分会额外加载 H5Dataset 导致 OOM/WSL 崩溃.
    # 改用几何一致性置信度: 追踪 bbox 投影 vs XMem mask 的 IoU.
    # (ScoreNet 首帧已在 register 内打分, 后续用 IoU 反映追踪贴合度)
    if (i > 0 and CONFIDENCE_CHECK_ENABLED and
            i % CONFIDENCE_CHECK_EVERY == 0):
        try:
            # 追踪 bbox 投影 (mesh 8 角点)
            corners_h = np.hstack([mesh_bbox_corners, np.ones((8, 1))])
            cam = (pose @ corners_h.T).T[:, :3]
            img = (K @ cam.T).T
            img = img[:, :2] / img[:, 2:3]
            x1, y1 = int(img[:, 0].min()), int(img[:, 1].min())
            x2, y2 = int(img[:, 0].max()), int(img[:, 1].max())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(wp - 1, x2), min(hp - 1, y2)  # 用实际帧尺寸, 勿硬编码 360p
            track_bb = np.zeros_like(mask)
            track_bb[y1:y2, x1:x2] = 1
            inter = np.logical_and(track_bb > 0, mask > 0).sum()
            union = np.logical_or(track_bb > 0, mask > 0).sum()
            iou = inter / max(union, 1)
            # IoU 低 → 追踪偏移 → 连续多次才触发重注册兜底 (防单帧抖动)
            if iou < CONFIDENCE_THRESHOLD:
                _low_conf_count += 1
            else:
                _low_conf_count = 0
            if iou < CONFIDENCE_THRESHOLD and _low_conf_count >= CONFIDENCE_CONSEC:
                print(f'  [Conf@{i}] IoU={iou:.3f} < {CONFIDENCE_THRESHOLD} '
                      f'x{_low_conf_count}, re-register...')
                # 重新 register (带 mask 背景屏蔽) 兜底修正朝向
                if INIT_MASK_BACKGROUND:
                    mask_bool = (mask > 0)
                    rgb_rr = np.where(mask_bool[..., None], rgb, 0).astype(np.uint8)
                    depth_rr = np.where(mask_bool, depth, 0.0).astype(np.float32)
                    pose_new = fp.register(K=K, rgb=rgb_rr, depth=depth_rr,
                                           ob_mask=mask, ob_id=0, iteration=2)
                else:
                    pose_new = fp.register(K=K, rgb=rgb, depth=depth,
                                           ob_mask=mask, ob_id=0, iteration=2)
                # 平滑: 与当前 pose 插值, 避免跳变 (SE(3) 加权)
                from scipy.spatial.transform import Rotation as _Rot
                _Rc = _Rot.from_matrix(pose[:3, :3].copy())
                _Rn = _Rot.from_matrix(pose_new[:3, :3])
                _k = 0.2  # 朝向新 pose 20% (更平滑, 防跳变)
                _old_t = pose[:3, 3].copy()
                _R_mix = _Rc.inv() * _Rn
                _axis = _R_mix.as_rotvec() * _k
                pose = np.eye(4)
                pose[:3, :3] = (_Rot.from_rotvec(_axis) * _Rc).as_matrix()
                pose[:3, 3] = _old_t + _k * (pose_new[:3, 3] - _old_t)
                # 重新应用主轴 roll 约束
                _, xmem_axis_r = mask_centroid_and_axis(mask)
                if xmem_axis_r is not None:
                    pose = apply_axis_pose_fix(pose, xmem_axis_r, K, 1.0)
                # 重置 LIEKF (避免旧状态拖慢恢复)
                if kf is not None:
                    kf.initialize(pose)
                sync_pose_last(fp, pose)
                _low_conf_count = 0
                print(f'  [Conf@{i}] re-registered, IoU corrected')
        except Exception as e:
            # 校验失败不阻塞追踪
            if i % 300 == 0:
                print(f'  [Conf@{i}] confidence check skipped: {str(e)[:40]}')

    # 更新上一帧稳定姿态 (旋转一致性校验基准, 用最终融合后的 pose)
    if i > 0:
        prev_pose_stable = pose.copy()

    poses.append(pose)
    confs.append(conf)

    if i % 100 == 0:
        dt = time.time() - t0
        fps_est = i / dt if dt > 0 else 0
        print(f'  F{i}/{n} | {fps_est:.1f}fps')

dt = time.time() - t0
print(f'[4] Done: {n}f in {dt:.1f}s ({n/dt:.1f}fps), '
      f'{n_color_fix} color-fixes, {n_axis_fix} axis-fixes, {n_fallback} fallbacks, '
      f'{n_rot_reject} rot-rejects')
print(f'  max VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')

# ── 5. 输出 (LIEKF 已在线融合, 跳过重复 batch smooth) ─────────────
smoothed = poses

# ── 6. 输出 CSV + 可视化 ────────────────────────────────────────────
print('[6] Writing outputs...')
from modules.output_writer import PoseOutputWriter, VisualizationRenderer
ts = [i / fps for i in range(n)]

PoseOutputWriter.write_csv(smoothed, confs, ts, f'{OUT}/poses.csv')

# 可视化: 原始 mesh 是 mm, FoundationPose 用 meters (mesh/1000)
# 渲染器需要 mesh.vertices *= model_scale 转成 meters, 与 pose 一致
renderer = VisualizationRenderer(MESH, model_scale=0.001)
renderer.render_video(frames, smoothed, K, f'{OUT}/tracking_vis.mp4', fps=fps)

print(f'\nDONE!')
print(f'  CSV:  {OUT}/poses.csv')
print(f'  Video:{OUT}/tracking_vis.mp4')
print(f'  平均帧率: {n/dt:.1f}fps')
