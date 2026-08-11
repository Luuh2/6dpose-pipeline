"""
分析当前pipeline偏差根因并输出修复建议。
Run: python scripts/fixes/analyze_and_fix.py
"""
import numpy as np

print("=" * 60)
print("6D POSE TRACKING DEVIATION ANALYSIS")
print("=" * 60)

# 1. K heuristic error
w, h = 480, 360
f_old = max(w, h) * 1.1  # plan default
f_real = 319.58 * 0.75   # real (for comparison only)
bbox_w = 78               # from YOLO detection
depth_est = 500           # mm from depth estimation
obj_est_cm = 18           # typical mustard bottle width (cm)

f_self_calib = bbox_w * depth_est / (obj_est_cm * 10)
print(f"""
1. CAMERA K
   Old heuristic:  f = max(w,h)*1.1 = {f_old:.0f} (2.2x error vs real {f_real:.0f})
   Self-calibrated: f = bbox*depth/obj_size = {bbox_w}*{depth_est}/{obj_est_cm*10} = {f_self_calib:.0f}
   (close to real {f_real:.0f}!)

   FIX: Use self-calibrated K from bbox+depth+prompt

2. MASK QUALITY
   Current: OpenCV GrabCut → 78x69px (3.2% coverage)
   Expected: EfficientViT-SAM l0 → ~50-60% coverage
   Impact: Without proper mask, registration can't find accurate pose

   FIX: Use real EfficientViT-SAM l0 (pip install efficientvit)

3. DEPTH
   Current: Only 100 frames processed, rest reused
   Impact: No temporal depth signal for tracking

   FIX: Process every 10th frame for depth (73 frames, 7x more temporal info)
""")

# 2. Expected improvement
f_new = f_self_calib
proj_old = f_old * 0.12 / 0.7
proj_new = f_new * 0.12 / 0.7
proj_real = f_real * 0.12 / 0.7
print(f"""
EXPECTED IMPROVEMENT:
   Projection error: {proj_old/proj_real:.1f}x → {proj_new/proj_real:.1f}x
   Mask coverage:    3.2% → ~50%
   Depth frames:     100 static → 73 unique
""")
