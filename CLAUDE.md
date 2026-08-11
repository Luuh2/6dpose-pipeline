# CLAUDE.md — RGB 6D Pose Tracking Pipeline

纯RGB视频端到端6D物体姿态追踪系统。
目标硬件: RTX 3060 Laptop (6GB VRAM) + 16GB RAM (全链路低配适配)。

## 项目结构

```
main_pipeline.py              # 顶层入口, 串联11个模块
config.yaml                   # 低配默认配置文件
modules/
├── video_decoder.py          # M1: 流式视频解码 (360p)
├── yolo_world_detector.py    # M2: YOLO-World v2_s 零样本检测
├── sam_segmentor.py          # M3: EfficientViT-SAM l0 分割
├── xmem_propagator.py        # M4: XMem 分段mask传播
├── depth_estimator.py        # M5: Depth Anything V2 vits
├── triposr_mesh_generator.py # M6: TripoSR 网格 (mc=128)
├── depth_scale_recovery.py   # M7: 深度尺度恢复
├── foundationpose_runner.py  # M8: FoundationPose 核心 (n_pts=3000)
├── se3_kalman_filter.py      # M9: SE(3) LIEKF 平滑
├── failure_detector.py       # M10: 失败检测+自动恢复
└── output_writer.py          # M11: CSV输出+可视化
```

## 关键约束

- 全局分辨率: 360p (target_short_edge=360)
- 全局精度: FP16 (所有模型 model.half())
- 模型驻留: 串行动态加载/卸载 (任意时刻 GPU 最多 1 个大模型)
- 中间结果: memmap 落盘 (mask, depth), 不常驻内存
- 峰值 VRAM: ≤ 5.5GB | 峰值 RAM: ≤ 12GB

## 运行方式

```bash
python main_pipeline.py --config config.yaml --video demo/test.mp4 --prompt "object name"
```

## 编码规范

Behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
