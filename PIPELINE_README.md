# RGB 6D Pose Tracking Pipeline

> **纯 RGB 视频 → 物体 3D 检测 + 6D 位姿追踪**
>
> 无真值依赖：不需要深度相机、不需要 GT 掩码、不需要 CAD 模型。
> 核心算法：[FoundationPose](https://github.com/NVlabs/FoundationPose)，辅以单目深度估计 + 单视图 3D 重建。
>
> **运行环境：全部在服务器上执行（6×RTX 4090，本机无需 GPU）。**

---

## 目录

1. [管线总览](#1-管线总览)
2. [服务器环境](#2-服务器环境)
3. [核心设计思路](#3-核心设计思路)
4. [各阶段详解](#4-各阶段详解)
5. [环境要求](#5-环境要求)
6. [快速开始](#6-快速开始)
7. [输出格式](#7-输出格式)
8. [配置说明](#8-配置说明)
9. [故障排查](#9-故障排查)

---

## 1. 管线总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RGB 6D Pose Tracking Pipeline                   │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────────┤
│   M1     │   M2     │   M3     │   M5     │   M6     │     M7      │
│ 视频解码 │ 目标检测 │ 实例分割 │ 深度估计 │ 网格生成 │  尺度对齐   │
│ 720p RGB │ COCO+深度│ SAM      │ DA3逐帧  │ Stream3D │  ★核心改进  │
├──────────┴──────────┴──────────┴──────────┴──────────┴─────────────┤
│                 M4 XMem 时序掩码跟踪 (平滑连贯)                    │
├─────────────────────────────────────────────────────────────────────┤
│             M8 FoundationPose 注册 + 追踪 + 掩码重锚定              │
│                     ★ 需 nvdiffrast CUDA (服务器)                  │
├──────────────────────────────┬──────────────────────────────────────┤
│      M10 失败检测 + 自动恢复  │      M9 SE(3) Kalman 平滑            │
├──────────────────────────────┴──────────────────────────────────────┤
│                      M11 CSV 输出 + 可视化视频                        │
└─────────────────────────────────────────────────────────────────────┘
```

**输入**：一个 `.mp4` 文件（无需任何其他信息）

**可选输入**：一句文本描述（如 `"mustard bottle"`），不提供时自动检测画面中最显著的物体

**输出**：
- `poses.csv` — 每帧的 6D 位姿（四元数 + 平移）+ 置信度
- `tracking_vis.mp4` — 3D 包围盒 + 坐标轴叠加的可视化视频

---

## 2. 服务器环境

| 项目 | 值 |
|------|-----|
| GPU | 6 × NVIDIA RTX 4090 (48GB)，追踪只用 1 张 |
| 项目目录 | `/mnt/20T/xieyongling/zhijiyige` |
| 数据盘 | `/mnt/20T/xieyongling`（20TB，含 `stream3d_data/`、`stream3d_out/` 等） |
| 模型权重 | `/mnt/20T/xieyongling/zhijiyige/weights/` |
| Python 环境 | `~/miniconda3/envs/ego_env`（唯一完整环境，预计算 + 追踪共用） |

全部阶段（预计算 M1-M7、追踪 M8-M11）都在服务器上执行，本机不需要 GPU。

---

## 3. 核心设计思路

### 3.1 网格生成：为什么不用 SAM3D

| 问题 | 说明 |
|------|------|
| **掩码质量** | SAM3D 内置的分割不如 YOLO+SAM 两阶段鲁棒，尤其是复杂背景 |
| **深度缺失** | FoundationPose 需要深度图做 RGB-D 配准，纯 RGB 下无法工作 |
| **尺度模糊** | SAM3D 输出的是任意尺度 mesh，与真实世界尺度不对齐 |

### 3.2 5 个核心改进

#### 改进 1：COCO 检测 + 深度平面验证（★ 检测架构的核心）

检测策略：
```
COCO 80类全检测 (YOLOv8, 可靠)
  │
  ▼
过滤背景类 (person/car/table...)
  │
  ▼
深度平面验证 ★: 只保留在桌面平面上的候选
  │     - 估算桌面深度 (画面下2/5中央区域中位数)
  │     - 丢弃 物体深度 > 桌面深度 + 0.25m 的候选 (背景)
  ▼
score × √area × 桌面接近奖励 → 选最优
```

> YOLO-World `yolov8s-worldv2` 的 CLIP 文本编码器在真实场景会产生严重 **hallucination**（把桌腿、三脚架、纸板误判为 `tripod` / `mustard-container`，置信度虚高到 1.00），因此只作为 COCO 完全失效时的 fallback。

关键：**深度验证利用"物体在桌面上"的物理先验**。芥末瓶深度 ≈ 桌面深度，而 2.5m 外的背景噪点会被直接剔除。

#### 改进 2：Depth Anything V3 度量深度，每帧估计

```
RGB 图像 ──→ Depth Anything V3 Metric ──→ 度量深度 (米)
```

- DA3 从单张 RGB 直接估计**度量深度**（不是相对深度！）
- **每帧估计**（`every_n=1`）—— 追踪跟得上的前提
- 提供物理上合理的深度值，是 RGB-D 配准能工作的关键

> ⚠ `every_n` 必须为 1。若间隔多帧估计一次、中间帧用最近邻填充，物体移动后深度图陈旧，`track_one` 的 RGB-D refiner 会找不到物体新位置 → 追踪框钉在原地。

#### 改进 3：深度引导的 Mesh 尺度对齐（★ 最关键的改进）

```
DA3 度量深度 (m) + mask + K
        │
        ▼
  掩码区域像素 ──→ 相机内参反投影 ──→ 3D 点云 (米)
        │
        ▼
  物体 3D bbox 对角线长度 (mm)
        │
        ▼
  Mesh bbox 对角线 (任意单位)  ←→  缩放因子
        │
        ▼
  Mesh 顶点 × 缩放因子 → 对齐后的 mesh (mm 单位)
        │
        ▼
  FoundationPose: mesh 转 meters (/1000) + 度量深度 (m)
                → 物理一致的 6D 位姿
```

FoundationPose 拿到**物理上一致的 mesh 和深度**，无需启发式假设。

> ⚠ **单位约定**：FoundationPose 内部 `erode_depth` 默认 `zfar=100`，会把 >100 的深度像素清零。因此 **mesh 和 depth 必须统一用 meters**：
> - mesh：mm 单位加载后 `/1000` → meters
> - depth：DA3 输出即 meters，直接用

#### 改进 4：多观测引导追踪（★ 追踪跟得上的关键）

**关键认知**：`track_one(rgb, depth, K, iteration)` 签名**没有 mask 参数**（FoundationPose 官方设计）。它是纯 RGB-D 局部优化，在单目噪声深度 + 低纹理物体上**无法可靠跟随**——物体移动时 pose 会漂移钉死。

**追踪策略**（XMem 掩码引导 + LIEKF 融合）：
```
每帧:
  track_one() → 基础 pose (RGB-D 局部优化)
  ↓
  ① XMem 时序掩码观测:
      - 掩码质心 → 修正 pose 平移 (轮廓对齐, 限幅 15px/帧)
      - 掩码惯性主轴 (cv2.moments) → 修正 pose 翻滚角
      - 掩码包围盒 → 与 mesh 投影 bbox 对齐 (轮廓约束)
  ↓
  ② ROI 深度降噪: 对物体区域深度做中值+双边滤波
  ↓
  ③ 主轴角预测: 最近 3 帧加权差分外推 roll, 预偏 pose_last
  ↓
  ④ SE(3)-LIEKF 在线融合: 平移观测 + 旋转观测接入滤波器
      → 输出平滑位姿, 同步 fp.pose_last
```

> **为什么需要这些层**：`track_one` 单独无法解决两件事——(a) 纯色物体平移时 RGB-D 优化收敛不到正确位置（漂移），(b) 旋转时低纹理轮廓无法区分朝向（朝向错乱）。XMem 掩码提供精确的逐帧物体区域，质心修正平移、惯性主轴修正翻滚、bbox 对齐施加轮廓约束，LIEKF 融合平滑，深度降噪提升 refiner 输入质量。

> **★ 本体坐标系固定原则**：物体 XYZ 本体朝向从**固定网格的 bbox 一次性定义**（长轴 = 主朝向，`long_axis_idx = argmax(bbox)`），永久固定。所有轴修正（mask 惯性主轴、主轴预测）一律对齐该固定本体轴，**禁止由当前帧点云/网格重新生成坐标轴**。这对细长物体至关重要：若轴修正硬编码对齐 mesh X 轴，而本体长轴是 Y（如 Stream3D 细长网格），每帧都会把框往错误方向拧（实测朝向误差 41.9°）。修复后误差降至 5.9°，与 TripoSR 同级。

#### 改进 5：SE(3) 李代数 Kalman 滤波 + 失败自动恢复

- **LIEKF**：在 SE(3) 流形上进行前向卡尔曼滤波 + 后向 RTS 平滑
- **自适应噪声**：置信度低时自动增大观测噪声；检测到运动时放宽 roll 限幅、静止时收紧（防跳变 + 不滞后）
- **失败恢复**：掩码面积骤变 / 追踪 bbox 与掩码 IoU 过低 → 触发重注册兜底
- **旋转一致性校验**：帧间三轴夹角突变 → 判定对称歧义解并回退上帧旋转

---

## 4. 各阶段详解

### 阶段 1：视频解码 (M1)

| 项目 | 说明 |
|------|------|
| 模块 | `modules/video_decoder.py` |
| 输入 | `.mp4` / `.avi` 视频文件 |
| 输出 | 缩放后的帧列表 (720p BGR) |
| 策略 | 短边缩放至 720px，保持宽高比；源视频低于 720p 时保持原生分辨率 |

### 阶段 2：目标检测 (M2)

| 项目 | 说明 |
|------|------|
| 模块 | `modules/yolo_world_detector.py` |
| 模型 | YOLOv8s-worldv2 (26MB) |
| 输入 | 首帧 + DA3 度量深度 |
| 输出 | 深度验证通过的检测框 `[x1, y1, x2, y2]` |
| 自动检测 | **无需 prompt**：COCO 80类 → 过滤背景类 → 深度平面验证 → 选最优 |
| 精确检测 | 提供 `--prompt` 时：COCO 关键词匹配（中英文均支持） |
| Fallback | YOLO-World 零样本（COCO 完全无结果时） |

### 阶段 3：实例分割 (M3)

| 项目 | 说明 |
|------|------|
| 模块 | `modules/sam_segmentor.py` |
| 模型 | EfficientViT-SAM l0 (139MB) |
| 输入 | 首帧 RGB + 检测框 |
| 输出 | 二值掩码 (H×W uint8) |

### 阶段 4：XMem 时序掩码跟踪 (M4)

| 项目 | 说明 |
|------|------|
| 策略 | 首帧 SAM 掩码初始化 XMem，时序传播到所有帧 |
| 输出 | 帧间平滑连贯的掩码 `masks_xmem_full.dat` (N, H, W) uint8 |
| 优势 | 帧间平滑的掩码，质心位移连续、轮廓稳定 |
| 回退 | XMem 不可用时降级为每 30 帧 YOLO+SAM 重检测 |

### 阶段 5：单目深度估计 (M5) ★ 每帧

| 项目 | 说明 |
|------|------|
| 模块 | `modules/depth_estimator.py` |
| 模型 | Depth Anything V3 Metric Large (1.3GB) |
| 输入 | 全部帧 RGB |
| 输出 | 度量深度 memmap `(N, H, W) fp16` (米) |
| 策略 | **每帧估计**（`every_n=1`）— 追踪跟得上的前提 |

### 阶段 6：3D 网格生成 (M6) ★ Stream3D

| 项目 | 说明 |
|------|------|
| 模块 | `triposr_mesh_generator.py` 生成首帧网格 + Stream3D 流式多视图重建 |
| 流程 | TripoSR 首帧初始网格 → 预追踪出位姿 → Stream3D 流式重建细化网格 |
| 模型 | Stream3D (服务器 GPU, 输出高斯+网格) + TripoSR (1.6GB) |
| 输入 | 首帧 RGB + 掩码 → 多视图 (RGB+掩码+深度+位姿) |
| 输出 | 多视图融合网格 `.glb`（含背面几何, ~17万顶点） |
| 备选 | 失败时退化为 bbox 代理网格 |

> **Stream3D** 流式多视图重建能补全背面几何（背面覆盖 23%→36%），结合本体轴固定修复后，追踪质量全面优于 TripoSR（IoU 0.58 vs 0.47，朝向误差 5.9° vs 4.6°，细长形态更贴合物体）。配套「本体坐标系固定原则」见[改进 4](#改进-4多观测引导追踪★-追踪跟得上的关键)。

### 阶段 7：尺度对齐 (M7) ★

| 项目 | 说明 |
|------|------|
| 模块 | `modules/depth_scale_recovery.py` → `MeshDepthAligner` |
| 输入 | mesh + DA3 深度 + mask + K |
| 输出 | 对齐后的 mesh (mm 单位) + 缩放因子 |
| 方法 | 深度反投影 → 3D bbox 对角线 → mesh 缩放 |

### 阶段 8：FoundationPose 6D 追踪 (M8)

| 项目 | 说明 |
|------|------|
| 脚本 | `scripts/wsl_track_m8_m11.py` |
| 模型 | FoundationPose Scorer (190MB) + Refiner (68MB) |
| 输入 | 全部帧 + 每帧深度 + XMem 掩码 + mesh + K |
| 输出 | 每帧 4×4 位姿矩阵 + 置信度 |
| 首帧 | `register()` — 全局搜索初始化（mask 屏蔽背景 + 主轴 roll 约束） |
| 后续帧 | `track_one()` — 局部 RGB-D 优化 + 掩码质心/主轴/bbox 引导 + 主轴角预测 + LIEKF |
| 采样 | `N_PTS=8000` |
| 旋转网格 | `rot_grid 16×60` |
| 重锚定 | 每 30 帧用掩码质心修正 pose 平移 + 重置 `pose_last` |
| 单位 | mesh: mm→m (/1000)，depth: meters，K: 像素 |

### 阶段 9：失败检测与恢复 (M10)

- 掩码面积相对首帧变化超过 50% → 触发恢复
- 追踪 bbox 与掩码 IoU 低于 0.22 → 触发重注册兜底
- 恢复策略：回退几帧 → 重跑 YOLO+SAM+FP 注册 → 继续追踪

### 阶段 10：SE(3) Kalman 平滑 (M9)

- 前向 LIEKF 滤波 + 后向 RTS 平滑
- 置信度自适应观测噪声；运动自适应（动则放开、静则平滑）
- 输出平滑后的 6D 轨迹

### 阶段 11：输出 (M11)

- **CSV**：`frame, timestamp, qw, qx, qy, qz, tx, ty, tz, confidence`
- **可视化视频**：绿色 3D 包围盒 + RGB 坐标轴叠加 (X=红, Y=绿, Z=蓝)

---

## 5. 环境要求

### 硬件（服务器）

| 组件 | 配置 |
|------|------|
| GPU | 6 × NVIDIA RTX 4090 (48GB)，追踪只用 1 张 (cuda:0) |
| CPU/RAM | 服务器级（≥64GB RAM） |
| 存储 | `/mnt/20T/xieyongling` 数据盘（20TB） |

### 软件（环境 `~/miniconda3/envs/ego_env`）

```
Python 3.12
torch 2.7.1 + cu126
ultralytics (YOLO)
opencv-python
trimesh / scipy / pandas / pyyaml / open3d
nvdiffrast       # CUDA 光栅化 (FoundationPose 渲染, 源码构建)
pytorch3d        # SE(3) 运算 (FoundationPose 姿态)
depth_anything_3 # Depth Anything V3 (源路径加到 sys.path)
```

> 统一使用 `ego_env`。其他环境（`zhijiyige`、`stream3d`、`ego_moge_env`）依赖不全。

### 模型权重（服务器路径）

| 模型 | 路径 | 大小 |
|------|------|------|
| YOLOv8s-worldv2 | `weights/yolo_world/yolov8s-worldv2.pt` | 26MB |
| EfficientViT-SAM l0 | `weights/efficientvit_sam/efficientvit_sam_l0.pt` | 139MB |
| XMem | `weights/xmem/XMem-s012.pth` | 249MB |
| Depth Anything V3 | `weights/da3_metric/` | 1.3GB |
| TripoSR | `weights/triposr/` | 1.6GB |
| FoundationPose Scorer | `weights/foundationpose/FoundationPosescorer.pth` | 190MB |
| FoundationPose Refiner | `weights/foundationpose/FoundationPoserefiner.pth` | 68MB |

---

## 6. 快速开始

以下命令在服务器上执行。

### 6.1 预计算 (M1-M7)

```bash
cd /mnt/20T/xieyongling/zhijiyige

# 自动检测模式 (无需 --prompt)
~/miniconda3/envs/ego_env/bin/python test_precompute.py \
    --video demo/test_mustard.mp4 --output ./output

# 或指定文本描述以提高精度
~/miniconda3/envs/ego_env/bin/python test_precompute.py \
    --video demo/test_mustard.mp4 --prompt "mustard bottle" --output ./output
```

预计算完成后，中间结果已全部保存：
```
output/
├── intermediate/
│   ├── depths_metric.dat      # 全帧度量深度 (fp16 memmap, 米, 每帧)
│   ├── masks.dat              # XMem 时序掩码副本 (兼容读取)
│   └── masks_xmem_full.dat    # ★ XMem 逐帧时序掩码 (平滑连贯, 追踪脚本读取)
├── meshes/
│   ├── proxy_mesh.glb         # TripoSR 首帧初始网格
│   └── proxy_mesh_stream3d_mm.glb  # ★ Stream3D 多视图重建网格 (mm 单位, 追踪用)
├── debug/                     # 检测/分割/深度可视化
└── K.npy                      # 相机内参
```

### 6.2 追踪 + 输出 (M8-M11)

```bash
~/miniconda3/envs/ego_env/bin/python scripts/wsl_track_m8_m11.py
```

追踪脚本自动完成：加载数据 → FoundationPose 初始化 → 逐帧追踪（XMem 掩码质心+主轴+bbox 轮廓约束 + 主轴角预测）→ SE(3)-LIEKF 融合 → CSV + 可视化视频。

> ⚠ 追踪脚本顶部 `VIDEO = f'{BASE}/demo/test_mustard.mp4'` 是硬编码。换视频需修改该行。

### 6.3 查看结果

```bash
cat output/poses.csv | head -20
```

---

## 7. 输出格式

### 7.1 `poses.csv`

| 列名 | 说明 | 示例 |
|------|------|------|
| `frame` | 帧索引 (0-based) | `0` |
| `timestamp` | 时间戳 (秒) | `0.0` |
| `qw, qx, qy, qz` | 旋转四元数 (w,x,y,z) | `0.999, 0.001, -0.010, 0.005` |
| `tx, ty, tz` | 平移向量 (**米**) | `0.012, -0.046, 0.850` |
| `confidence` | 追踪置信度 [0, 1] | `0.923` |

### 7.2 `tracking_vis.mp4`

- 绿色线框 = 物体 3D 包围盒
- 红/绿/蓝轴 = 物体坐标系 X/Y/Z
- 分辨率 = 输入视频分辨率（720p / 原生）

---

## 8. 配置说明

`config.yaml` 作为参数文档与尺度校准配置源，实际生效值以 `test_precompute.py` / `scripts/wsl_track_m8_m11.py` 内的硬编码为准。关键参数：

```yaml
# 视频处理
video:
  target_short_edge: 720     # 720p (源视频低于 720p 保持原生)

# 检测
detection:
  conf_threshold: 0.20       # 放宽以增加召回

# 深度: 每帧估计 (代码内强制 every_n=1)
depth:
  method: "da3"
  every_n: 1                 # ★ 每帧估计, 否则追踪跟丢

# 尺度对齐
scale_alignment:
  method: "depth_guided"     # 深度引导 (核心改进)

# 尺度校准 (可选, 需独立已知尺寸)
depth_calibration:
  enabled: true
  known_object_size_mm: null   # 例: 160.0 (芥末瓶对角线). null=禁用

# FoundationPose (服务器参数, 脚本硬编码)
foundationpose:
  n_pts: 8000                # 采样点数
  rot_grid: 16x60            # 初始化旋转搜索网格

# XMem 时序掩码
mask_propagation:
  enabled: true              # false = 只用首帧掩码 (追踪会跟丢)
```

---

## 9. 故障排查

### FoundationPose 报 `PermissionError: '/home/bowen'`

**原因**：FoundationPose 源码默认 `debug_dir=/home/bowen`，服务器无该目录/无权限。

**解决**：调用 `FoundationPose(...)` 时传 `debug_dir=os.path.join(OUT, "fp_debug")`（追踪脚本已处理）。

### mesh 顶点过多 → OOM (SIGKILL, exit 9)

**原因**：FoundationPose `reset_object` 内部 `compute_mesh_diameter` 用 `n_sample=10000` 生成 10000×10000 距离矩阵（2.4GB），TripoSR 网格 15780 顶点会触发 OOM killer。

**解决**：追踪脚本自动用 open3d 将 mesh 降采样到 4000 顶点。

### 单位不匹配 → register 报 "valid is empty"

**原因**：`erode_depth` 默认 `zfar=100` 会把 >100 的深度像素清零。若 depth 误用 mm（值 358-3623）会被全部清除。

**解决**：mesh 转 meters（/1000），depth 用 meters，K 用像素。

### 追踪框钉在原地 / 跟丢物体

**原因**（按概率排序）：
1. **深度没有每帧估计**：确认 `test_precompute.py` 里 `every_n = 1`
2. **掩码全首帧副本**：M4 未运行 → 无重锚定引导
3. **物体移动过快**：超出 `track_one` 局部优化范围

**解决**：
- 确保 `every_n = 1`
- 确认 `masks_xmem_full.dat` 各帧不同
- 减小 `reanchor_every`（如 15 帧）

### 可视化里 3D 框朝向偏斜 / 与物体不贴合

追踪的翻滚角（roll）由掩码惯性主轴引导 + LIEKF 融合。若框朝向明显偏斜：
- 检查 `ROT_CONSISTENCY_ENABLED` / `AXIS_GUIDE_ENABLED` 是否开启
- 掩码质量差（XMem 未跟随）时主轴引导失效
- 排查方向：更强的主轴观测、颜色引导权重、逐帧 bbox 长边对齐

### 检测失败（首帧找不到物体）

**原因**：物体不在 COCO 80 类中，且深度验证过滤过严。

**解决**：
- 提供 `--prompt` 指定物体
- 调大 `max_table_offset`（config 中 depth 验证容差）
- 物体太远（>10m）时深度验证会丢弃，换更近的物体或视频

### "Should never be installed" → EfficientViT-SAM 导入失败

**原因**：安装了不兼容的 triton 包。

**解决**：管线内置 triton shim（`sam_segmentor.py` 中的 `_install_triton_shim()`），会在导入 efficientvit 之前自动应用。

### XMem 掩码传播报错

XMem 相关 3 处 bug 修复位于：
- `group_modules.py` `MainToGroupDistributor.forward`：x 为 5 维时 `x.expand(-1,num_objects,-1,-1,-1)`（广播 T 维），4 维时 `x.unsqueeze(1).expand(...)`
- `network.py` `segment()`：对 `f.dim()==5 and f.shape[1]==1` 的特征 `squeeze(1)`
- `xmem_propagator.py`：`torch.argmax(prob, dim=0)` 而非 `torch.argmax(prob[0], dim=0)`

> 若 XMem 仍不可用，可用掩码周期重检测（每30帧 YOLO+SAM）作为回退。

---

## 参考

- [FoundationPose](https://github.com/NVlabs/FoundationPose) — NVIDIA, CVPR 2024
- [Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-3) — 度量深度估计
- [TripoSR](https://github.com/VAST-AI-Research/TripoSR) — 单视图 3D 重建
- [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit) — 轻量级 SAM
- [XMem](https://github.com/hkchengrex/XMem) — 视频目标分割
- [Stream3D](https://github.com/wenqsun/Stream3D) — 流式多视图 3D 重建
- RGB-Track — 纯 RGB 追踪的深度先验校验思路

---

## 已知问题 / 待办

- **180° 翻转**：机械臂把物体从水平转到竖直方向时，3D 框可能翻转 180°（颠倒）。属特殊 case，待解决。
- **中心偏移 ~6px**：Stream3D 网格 bbox 不对称，追踪框中心相对瓶子偏移约 6px（网格重居中可改善）。
- **多场景测试**：当前仅在 test_mustard.mp4 验证，需在更多视频（driller 等）上跑通大多数场景。

---

*最后更新：2026-08-11（采用 Stream3D + 本体坐标系固定原则）*
