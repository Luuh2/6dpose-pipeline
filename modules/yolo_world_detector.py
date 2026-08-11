"""
yolo_world_detector.py — Module 2
功能: 目标检测 — YOLO-World 零样本 (默认) / COCO降级 / 自动检测
低配适配: yolov8s-worldv2 (25MB) + FP16 + unload()
"""

import numpy as np
import torch
import os
from typing import List, Dict, Optional

# COCO 80 类映射 (YOLOv8 预训练)
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

# 自动检测时过滤的背景/非物体类别
AUTO_IGNORE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "potted plant", "bed", "dining table", "toilet", "couch", "tv", "remote",
    "sink", "refrigerator", "oven", "microwave", "toaster",
}

# 自动检测时偏好的桌面/手持物体 (高权重)
AUTO_PREFER_CLASSES = {
    "bottle", "cup", "wine glass", "bowl", "book", "laptop", "cell phone",
    "keyboard", "mouse", "backpack", "scissors", "knife", "fork", "spoon",
    "sports ball", "banana", "apple", "orange", "cake", "donut", "vase",
    "teddy bear", "clock", "umbrella", "handbag", "suitcase", "frisbee",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "kite", "skis", "snowboard", "broccoli", "carrot", "hot dog", "pizza",
    "sandwich", "toothbrush", "hair drier",
}

# COCO 关键词→类名映射 (中文+英文)
COCO_KEYWORDS = {
    # 英文
    "bottle": "bottle", "cup": "cup", "mug": "cup", "bowl": "bowl",
    "laptop": "laptop", "book": "book", "cell phone": "cell phone", "phone": "cell phone",
    "chair": "chair", "backpack": "backpack", "knife": "knife", "fork": "fork",
    "spoon": "spoon", "scissors": "scissors", "sports ball": "sports ball", "ball": "sports ball",
    "banana": "banana", "apple": "apple", "orange": "orange", "cake": "cake",
    "vase": "vase", "clock": "clock", "keyboard": "keyboard", "mouse": "mouse",
    "remote": "remote", "tv": "tv", "toothbrush": "toothbrush",
    # 中文
    "瓶子": "bottle", "杯子": "cup", "碗": "bowl",
    "笔记本": "laptop", "书": "book", "手机": "cell phone",
    "剪刀": "scissors", "刀": "knife", "叉": "fork", "勺": "spoon",
    "香蕉": "banana", "苹果": "apple", "橙子": "orange", "蛋糕": "cake",
    "球": "sports ball", "包": "backpack", "瓶子": "bottle",
    "键盘": "keyboard", "鼠标": "mouse", "遥控器": "remote",
}

# YOLO-World 零样本默认词汇表 (自动检测时使用)
# 关键: 必须包含足够多的负类/背景类, 否则 YOLO-World 的对比头会 hallucinate
DEFAULT_WORLD_VOCABULARY = [
    # 前景物体 (我们想检测的)
    "a bottle", "a cup", "a bowl", "a book", "a laptop",
    "a cell phone", "a keyboard", "a mouse", "scissors",
    "a knife", "a fork", "a spoon", "a banana", "an apple",
    "an orange", "a cake", "a donut", "a vase", "a clock",
    "a backpack", "a handbag", "a ball", "a toy",
    "a remote control", "a toothbrush", "a pair of glasses",
    "a box", "a can", "a jar", "a tool",
    # 背景/非目标 (负类, 用于校准 — YOLO-World 必须有负类才不会 hallucinate)
    "a table", "a desk", "a wall", "a chair", "a floor",
    "a piece of paper", "a whiteboard", "a door", "a window",
    "a tripod", "a cable", "a wire", "a metal rod", "a pipe",
    "a person", "a hand", "a finger", "an arm",
    "a shadow", "a reflection", "a light spot", "a speck of dust",
    "a piece of tape", "a sticker", "a mark", "a scratch",
]


class YOLOWorldDetector:
    """检测器 — YOLO-World 零样本优先, COCO降级, 自动检测"""

    def __init__(
        self,
        model_path: str = "E:/zhijiyige/weights/yolo_world/yolov8s-worldv2.pt",
        device: str = "cuda:0",
        conf_threshold: float = 0.20,
        use_world: bool = True,  # 默认启用 YOLO-World 零样本
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.use_world = use_world
        self._clip_ok = False
        self.device = device

        if use_world:
            try:
                import sys as _sys
                import importlib.util
                # scripts/utils/ 缺少 __init__.py, 用直接路径加载
                shim_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "scripts", "utils", "clip_shim.py")
                spec = importlib.util.spec_from_file_location(
                    "clip_shim", shim_path)
                clip_shim = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(clip_shim)
                _sys.modules['clip'] = clip_shim
                self._clip_ok = True
                print("[YOLO-World] CLIP shim loaded. Zero-shot mode ENABLED.")
            except Exception as e:
                print(f"[YOLO-World] CLIP unavailable ({e}), falling back to COCO classes.")

    # ── 公开 API ─────────────────────────────────────────────────────

    def detect(self, image: np.ndarray, text_prompts: str) -> List[Dict]:
        """文本引导检测

        Args:
            image: BGR ndarray (H, W, 3) uint8
            text_prompts: 逗号分隔类别名 (中/英文均可)

        Returns:
            list of dict: [{"bbox": [x1,y1,x2,y2], "score": 0.85, "label": "..."}, ...]
        """
        text_list = [t.strip() for t in text_prompts.split(",") if t.strip()]

        # YOLO-World 零样本
        if self.use_world and self._clip_ok:
            return self._detect_world(image, text_list)

        # COCO 降级 (支持中英文关键词)
        return self._detect_coco(image, text_list)

    def detect_top1(self, image: np.ndarray, text_prompts: str) -> Optional[Dict]:
        detections = self.detect(image, text_prompts)
        return detections[0] if detections else None

    def auto_detect(self, image: np.ndarray, depth_m: np.ndarray = None) -> Optional[Dict]:
        """自动检测 — 无需文本 prompt, 选择画面中最显著的前景物体

        策略 (改进版):
          1. COCO 检测优先 (YOLOv8 80类, 可靠) → 深度过滤桌面物体
          2. 若 COCO 无结果 → YOLO-World 零样本 (平衡词汇表+深度落差)
          3. 按综合得分 (深度验证 + 尺寸偏好) 排序返回最优

        Args:
            image: BGR ndarray
            depth_m: DA3 度量深度图 (可选, 用于过滤背景候选)

        Returns:
            best detection dict or None
        """
        # ═══ 策略 1: COCO 检测 + 深度验证 (主力, 可靠) ═══
        coco_dets = self._auto_detect_coco(image)
        if coco_dets and depth_m is not None:
            coco_dets = self._filter_by_table_depth(coco_dets, depth_m)

        if coco_dets:
            best = coco_dets[0]
            x1,y1,x2,y2 = [int(v) for v in best['bbox']]
            depth_info = ""
            if 'depth_median' in best:
                depth_info = f" depth={best['depth_median']:.2f}m"
            print(f"[AutoDetect-COCO] Selected: {best['label']} "
                  f"score={best['score']:.3f}{depth_info} "
                  f"bbox=[{x1},{y1},{x2},{y2}]")
            return best
        elif coco_dets:
            # 无深度图时用尺寸偏好选最优
            best = self._select_best(coco_dets)
            print(f"[AutoDetect-COCO] Selected (no depth): {best['label']} "
                  f"score={best['score']:.3f}")
            return best

        # ═══ 策略 2: YOLO-World 零样本 (fallback) ═══
        if self.use_world and self._clip_ok:
            print("[AutoDetect] COCO found nothing, trying YOLO-World zero-shot...")
            world_dets = self._detect_world(image, DEFAULT_WORLD_VOCABULARY)
            if world_dets:
                if depth_m is not None:
                    world_dets = self.filter_by_depth(world_dets, depth_m)
                if world_dets:
                    if "depth_score" in (world_dets[0] if world_dets else {}):
                        best = world_dets[0]
                    else:
                        best = self._select_best(world_dets)
                    print(f"[YOLO-World Auto] Selected: {best['label']} "
                          f"score={best['score']:.3f}")
                    return best

        print("[AutoDetect] No object found in any mode.")
        return None

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()
        print("[YOLO] Unloaded.")

    # ── 内部实现 ─────────────────────────────────────────────────────

    def _detect_world(self, image: np.ndarray, text_list: List[str]) -> List[Dict]:
        """YOLO-World 零样本检测"""
        self.model.set_classes(text_list)
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_idx = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else 0
                label = text_list[min(cls_idx, len(text_list) - 1)]
                detections.append({
                    "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                    "score": float(boxes.conf[i].cpu().numpy()),
                    "label": label.replace("a ", "").replace("an ", "").replace("A ", ""),
                })
        detections.sort(key=lambda x: x["score"], reverse=True)
        if detections:
            print(f"[YOLO-World] {len(detections)} detections: "
                  f"{[(d['label'], round(d['score'],3)) for d in detections[:5]]}")
        return detections

    def _detect_coco(self, image: np.ndarray, text_list: List[str]) -> List[Dict]:
        """YOLOv8 COCO 检测, 匹配文本prompt中的关键词 (中/英文)"""
        target_class_ids = set()
        for prompt in text_list:
            prompt_lower = prompt.lower()
            for keyword, coco_name in COCO_KEYWORDS.items():
                if keyword in prompt_lower:
                    cls_id = COCO_CLASSES.index(coco_name)
                    target_class_ids.add(cls_id)

        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].cpu().numpy())
                score = float(boxes.conf[i].cpu().numpy())
                label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                # 有关键词匹配则筛选, 否则返回所有
                if not target_class_ids or cls_id in target_class_ids:
                    detections.append({
                        "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                        "score": score,
                        "label": label,
                    })

        detections.sort(key=lambda x: x["score"], reverse=True)
        print(f"[YOLOv8-COCO] Detected {len(detections)} objects matching {text_list}: "
              f"{[(d['label'], round(d['score'],3)) for d in detections[:5]]}")
        return detections

    def _auto_detect_coco(self, image: np.ndarray) -> List[Dict]:
        """COCO 自动检测: 全类检测 → 过滤背景 → 返回前景候选列表"""
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        all_dets = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].cpu().numpy())
                score = float(boxes.conf[i].cpu().numpy())
                label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                all_dets.append({"bbox": bbox, "score": score, "label": label,
                                 "cls_id": cls_id})

        if not all_dets:
            print("[AutoDetect-COCO] No objects found.")
            return []

        # 过滤背景类别
        fg_dets = [d for d in all_dets if d["label"] not in AUTO_IGNORE_CLASSES]
        if not fg_dets:
            fg_dets = all_dets

        # 偏好评分: score × sqrt(area) × prefer_bonus
        scored = []
        for d in fg_dets:
            x1, y1, x2, y2 = d["bbox"]
            area = max(1, (x2 - x1) * (y2 - y1))
            prefer_bonus = 1.5 if d["label"] in AUTO_PREFER_CLASSES else 1.0
            saliency = d["score"] * np.sqrt(area) * prefer_bonus
            scored.append((saliency, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [d for _, d in scored]
        print(f"[AutoDetect-COCO] {len(all_dets)} total → {len(result)} foreground candidates")
        return result

    @staticmethod
    def _filter_by_table_depth(detections: List[Dict], depth_m: np.ndarray,
                                max_table_offset: float = 0.25) -> List[Dict]:
        """深度验证: 只保留在桌面平面上的物体, 丢弃远处背景

        原理: 目标物体 (芥末瓶) 在桌面上 → 深度 close to 桌面深度
              背景物体 (2.5m远) → 深度 >> 桌面深度 → 丢弃

        Args:
            detections: COCO/YOLO-World 检测候选
            depth_m: DA3 度量深度图 (meters)
            max_table_offset: 允许的最大深度偏移 (物体深度 − 桌面深度)

        Returns:
            过滤后的候选列表, 附带 depth_median, depth_offset 字段
        """
        if depth_m is None or not detections:
            return detections

        h, w = depth_m.shape

        # 估算桌面平面深度 (画面下2/5中央区域的中位深度)
        y0, y1 = 2 * h // 5, h
        x0, x1 = w // 6, 5 * w // 6
        table_region = depth_m[y0:y1, x0:x1]
        table_vals = table_region[(table_region > 0.1) & (table_region < 10.0) & np.isfinite(table_region)]
        if len(table_vals) < 100:
            return detections  # 深度图无效, 不过滤
        table_depth = float(np.median(table_vals))

        filtered = []
        rejected_bg = 0
        for d in detections:
            x1, y1_b, x2, y2_b = [int(v) for v in d["bbox"]]
            x1, y1_b = max(0, x1), max(0, y1_b)
            x2, y2_b = min(w - 1, x2), min(h - 1, y2_b)
            if x2 <= x1 + 5 or y2_b <= y1_b + 5:
                continue

            roi = depth_m[y1_b:y2_b, x1:x2]
            valid = roi[(roi > 0.1) & (roi < 10.0) & np.isfinite(roi)]
            if len(valid) < 15:
                continue

            d_med = float(np.median(valid))
            depth_offset = d_med - table_depth

            # 核心过滤: 物体必须在桌面深度附近 (不能远在背景)
            if depth_offset > max_table_offset:
                rejected_bg += 1
                continue  # 背景 → 丢弃

            d_out = dict(d)
            d_out["depth_median"] = d_med
            d_out["depth_offset"] = depth_offset
            d_out["table_depth"] = table_depth
            # 综合得分: YOLO score × (1 + 桌面接近奖励)
            d_out["depth_score"] = d["score"] * (1.0 + max(0, 0.5 - abs(depth_offset) * 2.0))
            filtered.append(d_out)

        if rejected_bg > 0:
            print(f"[DepthVerify] {rejected_bg} background detections rejected "
                  f"(table={table_depth:.2f}m, max_offset={max_table_offset:.2f}m)")

        # 按 depth_score 排序
        filtered.sort(key=lambda x: x.get("depth_score", 0), reverse=True)
        return filtered if filtered else detections  # 如果全过滤了则不过滤

    # ── 深度落差过滤 (RGB-Track 校验思路) ────────────────────────────

    @staticmethod
    def filter_by_depth(
        detections: List[Dict],
        depth_m: np.ndarray,       # (H, W) DA3 度量深度 (meters)
        min_depth_variance: float = 0.003,   # 最小深度方差 (m²) — 滤除平面
        max_table_offset: float = 0.15,      # 物体必须比桌面近至少 15cm
    ) -> List[Dict]:
        """深度落差过滤: 抛弃平坦无高度差的候选

        原理 (借鉴 RGB-Track):
          - 芥末瓶高出白色板子 → 物体区域存在明显深度落差 (前后景深不同)
          - 桌面、白纸、三脚架根部 → 基本处在同一平面, 深度数值平缓
          - 物体顶部应明显比桌面更靠近相机

        Args:
            detections: 检测候选列表
            depth_m: DA3 度量深度图 (H, W) meters

        Returns:
            过滤后的候选列表, 附加 depth_variance, depth_median, depth_relief 字段
        """
        if depth_m is None or len(detections) == 0:
            return detections

        # 估算桌面平面深度 (画面下半部分的中位深度)
        h, w = depth_m.shape
        table_region = depth_m[h // 2:, :]
        table_vals = table_region[(table_region > 0.1) & (table_region < 10.0)]
        table_depth = np.median(table_vals) if len(table_vals) > 100 else 1.5

        filtered = []
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 <= x1 + 5 or y2 <= y1 + 5:
                continue  # bbox 太小, 无法可靠判断

            roi_depth = depth_m[y1:y2, x1:x2]
            valid = roi_depth[(roi_depth > 0.1) & (roi_depth < 10.0) & np.isfinite(roi_depth)]

            if len(valid) < 20:
                continue  # 有效深度点太少

            depth_var = float(np.var(valid))
            depth_med = float(np.median(valid))

            # 深度落差 = 物体 bbox 上半部分的深度 减 下半部分的深度
            mid_y = (y1 + y2) // 2
            top_half = roi_depth[:mid_y - y1, :]
            bot_half = roi_depth[mid_y - y1:, :]
            top_valid = top_half[(top_half > 0.1) & (top_half < 10.0)]
            bot_valid = bot_half[(bot_half > 0.1) & (bot_half < 10.0)]

            if len(top_valid) >= 10 and len(bot_valid) >= 10:
                relief = float(np.median(bot_valid) - np.median(top_valid))
                # relief > 0: 物体底部比顶部远 → 物体向前倾斜/竖立 (好)
                # relief ~ 0: 物体是平的 (纸、桌面)
            else:
                relief = 0.0

            # 过滤条件:
            # 1. 深度方差足够大 (不是平面)
            # 2. 物体比桌面近 (高出桌面的物体更靠近相机)
            # 3. 或者有明显的深度落差
            passes_var = depth_var >= min_depth_variance
            passes_height = (table_depth - depth_med) >= max_table_offset
            passes_relief = abs(relief) >= 0.02

            if passes_var or passes_height or passes_relief:
                d_out = dict(d)
                d_out["depth_variance"] = depth_var
                d_out["depth_median"] = depth_med
                d_out["depth_relief"] = relief
                d_out["table_depth"] = table_depth
                filtered.append(d_out)

        if not filtered:
            # 如果全被过滤了, 放回原始候选 (不过滤)
            return detections

        # 按深度落差得分重排: 综合 depth_relief + height_above_table + variance
        for d in filtered:
            height_score = max(0, (d["table_depth"] - d["depth_median"]) / 0.3)
            relief_score = max(0, abs(d["depth_relief"]) / 0.05)
            var_score = min(d["depth_variance"] / 0.01, 1.0)
            d["depth_score"] = height_score + relief_score + var_score

        filtered.sort(key=lambda x: x.get("depth_score", 0), reverse=True)
        return filtered

    @staticmethod
    def _select_best(detections: List[Dict], img_area: int = 480 * 360) -> Optional[Dict]:
        """从检测列表中选最优: 偏好可追踪尺寸的物体

        评分策略 (trackability score):
          - score × size_bonus: 偏好中等大小物体 (2%~30% 画面)
          - 惩罚过小 (<1%) 和过大 (>50%) 的检测
        """
        if not detections:
            return None
        if img_area <= 0:
            img_area = 480 * 360

        scored = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            w, h = x2 - x1, y2 - y1
            area = max(1, w * h)
            area_ratio = area / img_area

            # 尺寸偏好: 高斯型, 峰值在 5% 画面占比 (可追踪的理想尺寸)
            # 1-2%: 可接受但偏小; 5-30%: 理想; >50%: 可能是背景
            if area_ratio < 0.005:    # <0.5%: 太小
                size_bonus = area_ratio / 0.005 * 0.3
            elif area_ratio < 0.02:   # 0.5-2%: 偏小但可用
                size_bonus = 0.3 + (area_ratio - 0.005) / 0.015 * 0.5
            elif area_ratio < 0.30:   # 2-30%: 理想
                size_bonus = 0.8 + (area_ratio - 0.02) / 0.28 * 0.2
            elif area_ratio < 0.50:   # 30-50%: 偏大
                size_bonus = 1.0 - (area_ratio - 0.30) / 0.20 * 0.6
            else:                      # >50%: 可能是背景/桌面
                size_bonus = max(0.05, 0.4 - (area_ratio - 0.50) * 2.0)

            # 宽高比惩罚 (极端细长大概率是误检)
            aspect = max(w, h) / max(min(w, h), 1)
            aspect_penalty = 1.0 if aspect < 5 else max(0.3, 1.0 - (aspect - 5) * 0.1)

            saliency = d["score"] * size_bonus * aspect_penalty
            scored.append((saliency, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        return best
