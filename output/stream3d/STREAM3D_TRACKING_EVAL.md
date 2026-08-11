# Stream3D vs TripoSR 网格追踪质量评估

> 日期: 2026-08-11 | 测试视频: demo/test_mustard.mp4 (737帧, 480×360)
> 结论先行: **Stream3D 网格追踪质量显著优于 TripoSR, 建议替换 M6。**

---

## 1. 实验设计 (公平对比)

| 运行 | 网格 | 顶点数 | 追踪参数 | 运行环境 |
|------|------|--------|----------|----------|
| TripoSR@3k | proxy_mesh_aligned.glb | 15,780 | N_PTS=3000, rot_grid 8×120 | 原 WSL2 (2026-08-07) |
| TripoSR@8k | proxy_mesh_aligned.glb | 15,780 | **N_PTS=8000, rot_grid 16×60** | 服务器重跑 (公平基线) |
| Stream3D@8k | proxy_mesh_stream3d_mm.glb | 4,008* | **N_PTS=8000, rot_grid 16×60** | 服务器 (2026-08-11) |

\* Stream3D 原始网格 173,770 顶点, 被追踪脚本的 OOM 防护降采样到 4000。
仅网格不同, 追踪代码/参数/深度/掩码/内参全部相同 → 差异纯粹来自网格几何。

**网格物理尺寸 (均经深度尺度对齐, 对角线统一为 179mm):**

| 网格 | W | H | T | 形态 |
|------|------|------|------|------|
| TripoSR | 121.7mm | 115.8mm | 62.4mm | 矮胖 (立方体状) |
| Stream3D | 72.8mm | 158.0mm | 43.0mm | 细长 (符合瓶子) |

---

## 2. 核心指标对比

| 指标 | TripoSR@3k | TripoSR@8k | Stream3D@8k | 判定 |
|------|-----------|-----------|-------------|------|
| 掩码贴合 mean (px) | 1.22 | 1.00 | 1.07 | ≈ |
| 掩码贴合 max (px) | 8.25 | 8.87 | **6.85** | ✅ Stream3D |
| 掩码质心 mean (px) | 3.60 | 3.37 | 3.39 | ≈ |
| **追踪 AABB IoU mean** | 0.428 | 0.453 | **0.590** | ✅✅ Stream3D (+30%) |
| **IoU < 0.3 占比** | 10.4% | 10.7% | **0.3%** | ✅✅ Stream3D (34×↓) |
| 旋转平滑 mean (°/帧) | 0.62 | 0.67 | 0.76 | ≈ (均平滑) |
| 旋转平滑 max (°/帧) | 11.6 | 6.0 | 8.2 | ≈ |
| 平移连续性 mean (mm) | 2.4 | 2.3 | 2.4 | ≈ |
| 平移连续性 max (mm) | 20.9 | 30.3 | **20.5** | ✅ Stream3D |
| 翻滚跟随范围 (°) | 180 | 120 | 170 | Stream3D 更广 |
| 置信度 mean | 0.800 | 0.800 | 0.800 | = |
| 对称歧义解拒绝次数 | 0 | 0 | 0 | = |

---

## 3. 关键发现

### 3.1 追踪贴合度大幅提升 (核心结论)

逐帧 AABB-IoU 分析:
- **71% 的帧** Stream3D 的 IoU 比 TripoSR 高 ≥ 5pp
- **仅 1% 的帧** TripoSR 更优
- TripoSR 在前 **30 帧系统性 IoU<0.3**(矮胖网格不贴合细长瓶子), Stream3D 仅帧 0/30 略低

原因: 两个网格对角线被强制对齐到 179mm, 但**宽高比差异巨大**。
瓶子实际是细长物体 (mask 宽高比 ≈0.82), Stream3D (73×158) 比 TripoSR (122×116) 的投影轮廓更贴合真实形状。

### 3.2 N_PTS 混淆已排除

N_PTS 3000→8000 只让 TripoSR 的 IoU 从 0.428→0.453 (微升),
而 Stream3D 直接到 0.590 → **改进归因于网格几何, 而非追踪参数**。

### 3.3 顶点数不是决定性因素

Stream3D 网格被降采样到 4,008 顶点 (去掉 97%), 仍显著优于 TripoSR 的 15,780 顶点。
说明决定质量的是**几何形态**, 不是顶点密度。

### 3.4 旋转平滑度

Stream3D mean 0.76°/帧 vs TripoSR 0.67°/帧 (0.09°差, 均远低于感知阈值);
翻滚跟随范围 Stream3D 170° vs TripoSR 120°, 跟随旋转更充分。

---

## 4. 结论与建议

**结论: 用 Stream3D 替换 TripoSR 作为 M6 网格生成器。**

理由:
1. 追踪贴合 (IoU) 显著提升, 失效帧从 10.7% 降到 0.3%
2. 网格形态更符合物体物理 (细长 vs 矮胖)
3. 相同参数下的公平对比, 结论稳健
4. 追踪逻辑无需任何改动, 只换网格文件

**注意事项 (替换前需要解决):**

| 问题 | 说明 |
|------|------|
| **两遍流程** | Stream3D 需要 位姿+掩码+深度 作为输入 → 需先用 TripoSR 跑一遍预追踪, 再生成 Stream3D 网格, 再最终追踪 |
| **服务器依赖** | Stream3D 需 RTX 4090 级 GPU (本地 3060 6GB 不足以跑), 且需在服务器环境运行 |
| **顶点降采样** | 173k 顶点需降到 4000 才能进 FoundationPose (compute_mesh_diameter 2.4GB 距离矩阵), 可考虑更保形的降采样策略 |
| **时间成本** | 预计算阶段增加 Stream3D 推理 (~分钟级), 但追踪阶段帧率不变 (26.5fps) |

---

## 5. 产物

- 评估脚本: `scripts/eval_stream3d_tracking.py` (3 路对比)
- 可视化脚本: `scripts/viz_stream3d_compare.py`
- 曲线图: `output/stream3d/tracking_eval_compare.png`
- 关键帧叠加: `output/stream3d/tracking_keyframes.png`
- 数据: `output/stream3d/poses.csv` (Stream3D) / `poses_triposr_fair.csv` (公平基线)
- 网格: `output/stream3d/proxy_mesh_stream3d_mm.glb` (追踪用, mm) / `stream3d_bottle.glb` (原始)
