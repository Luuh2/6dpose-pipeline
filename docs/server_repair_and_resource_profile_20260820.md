# 服务器管线环境修复 + 资源画像报告（2026-08-20）

## 1. 背景

服务器主管线 conda 环境 `ego_env` 被移除，替换为**依赖不完整的 `ego_fa_env`**（及 `egogvae`），导致管线运行大量报错。**管线代码本身未受影响**（`modules/`、`scripts/`、`test_precompute.py` 等全部完好，仅服务器路径与本地不同）。本次在 `ego_fa_env` 上补齐依赖、修复编译问题，并完成管线端到端验证。

## 2. 环境修复记录

### 2.1 缺失依赖补装（ego_fa_env, Python 3.12 / torch 2.7.1+cu126）

| 依赖 | 用途 | 说明 |
|------|------|------|
| `ultralytics` | M2 检测 | 需先 `export PATH=~/miniconda3/envs/ego_fa_env/bin:$PATH` 再 pip（否则 pip 子进程找不到 python3.12） |
| `omegaconf` `timm` `evo` | M5 DA3 | DA3 源路径 `Depth-Anything-3/Depth-Anything-3-main/src` 需加 sys.path |
| `transformers==4.35.0` `huggingface-hub==0.23.5` | M6 TripoSR / DA3 | TripoSR 依赖此版本；`huggingface-hub` 版本对 DA3 的 `PyTorchModelHubMixin` safetensors 加载有影响 |
| `safetensors` | 模型加载 | transformers 加载 `model.safetensors` 必需 |
| `onnx` `onnxsim` | M3 EfficientViT-SAM | efficientvit 导入链拖带（训练/导出依赖） |
| `segment-anything` | M3 SAM | efficientvit 的 SAM 导入 |
| `PyMCubes` | M6 TripoSR mesh 提取 | **注意**：PyPI 上有同名错包 `mcubes 0.1.6`（API 为 `MarchingCubes/Mesh`），必须装 `PyMCubes`（提供 `mcubes.marching_cubes`）。装错会导致 TripoSR 网格生成退回 bbox。 |
| `rembg` `onnxruntime` | M6 TripoSR | TSR 包导入依赖 |
| `warp-lang` | M8 FoundationPose | `erode_depth` 内核（`wp.launch`）依赖，缺失会导致 `erode_depth is not defined` |
| `pytorch3d==0.7.9` | M8 FP | **不在 PyPI**，需源码编译：`pip install . --no-build-isolation`（构建隔离环境没有 torch）；FAIR 无 py312 预编译 wheel |
| `ninja` | 编译加速 | nvdiffrast/pytorch3d 编译 |

### 2.2 源码修复

| 问题 | 修复 |
|------|------|
| `src/nvdiffrast/nvdiffrast-main/setup.py` 硬编码 `os.environ["CUDA_HOME"] = "/root/miniconda3/envs/rgb6d"`（WSL2 残留） | 改为 `/usr/local/cuda`（服务器实际 CUDA 12.1）后 `setup.py build_ext --inplace` 编译成功 |
| TripoSR mesh 提取失败（`mcubes` 无 `marching_cubes`） | 卸载错包 `mcubes 0.1.6`，安装 `PyMCubes` |

### 2.3 端到端验证

完整预计算 + 追踪已在 `ego_fa_env` 跑通（user_video3，414 帧 720p）：
- 检测 mouse@f0 ✓ | TripoSR 真实网格 13574v ✓ | XMem 414 帧、0 次恢复 ✓ | DA3 深度 ✓
- 追踪 8.5fps ✓

## 3. 管线资源画像（720p · 414 帧 · ego_fa_env）

### 3.1 逐阶段显存 / 内存 / 时间

| 阶段 | 峰值显存 | 峰值 RSS | 时间 | 说明 |
|------|---------|---------|------|------|
| M1 视频解码 | 0 MB | 1706 MB | 1.0s | 帧数组驻留 RAM |
| M2 YOLO 检测 | 91 MB | ~3.4 GB | 1.3s | |
| M3 EfficientViT-SAM | 284 MB | ~3.7 GB | 0.7s | |
| M5 DepthAnythingV3 | 2127 MB | ~3.0 GB | 5.6s | 逐帧估计 |
| M6 TripoSR 网格 | 3846 MB | ~6.1 GB | 34.9s | 单帧重建 |
| M4 XMem 传播（全 414 帧）| 2947 MB | ~4.4 GB | 20s | 干净环境实测 |
| **M8 FoundationPose** | **8279 MB** | ~6.5 GB | ~48s | register+track，720p/8000 点，nvidia-smi 采样 |
| Stream3D 重建 | ~19.4 GB | — | ~5 min | 独立 stream3d env |

> 注：模型串行加载/卸载（任意时刻 GPU 只驻留一个大模型），管线峰值显存 ≈ 最大单阶段（M8 8.3GB）。

### 3.2 数组尺寸（414 帧 720p）

| 数组 | 形状 | 大小 |
|------|------|------|
| frames | 414×720×1280×3 uint8 | 1145 MB（RAM）|
| masks memmap | 414×720×1280 uint8 | 382 MB |
| depth memmap | 414×720×1280 fp16 | 763 MB |
| Stream3D mesh glb | ~17 万顶点 | ~10 MB |
| FP 采样点 | 8000 × 3 float32 | 96 KB |

### 3.3 关键结论

1. **管线峰值显存 = M8 FP 8279MB**（720p、N_PTS=8000）。
2. **对 3060 6GB 目标不达标**：需降到 360p + 减小 N_PTS（显存随分辨率平方下降），预计可压到 ~2-3GB。
3. **RAM 峰值 ~6.5GB**，`frames` 列表（1145MB）为最大单一内存块；流式/逐帧读取可省。
4. TripoSR（3.8GB）、XMem（2.9GB）为中段大户，均被 M8 盖过。

## 4. 遗留问题与建议

1. **依赖固化**：`ego_fa_env` 依赖链脆弱（hf 版本反复、transformers 固定、pytorch3d 源码编译），建议固化 requirements 文件，环境出问题可一键重建。
2. **可复现性**（来自前一轮实验）：追踪对 DA3 深度 4-7mm 的 fp16 噪声极敏感，相同输入可能落入不同位姿解（好的/坏的差 180°）。建议 DA3 推理固定 seed + `torch.use_deterministic_algorithms`。
3. **真实相机内参**：DA3 估计的 K（fx=554）与视频真值（fx=319.58）差 1.7 倍，已知晓真实内参时直接用标定值。
4. **TripoSR mcubes 已修**：注意不要重装回错包。

## 5. 相关文件

- 画像脚本：`scripts/profile_pipeline.py`（分模块微基准，输出峰值显存/内存/时间/数组尺寸）
