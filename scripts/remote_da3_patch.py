#!/usr/bin/env python3
"""
remote_da3_patch.py — 远程 DA3 导入补丁
=========================================
DA3 的 export 模块 (gs/glb/colmap 可视化) 拖出大量依赖 (evo, pycolmap, moviepy...).
评估/推理不需要 export, 用桩模块屏蔽, 避免装多余依赖.

用法: 在 import DepthAnything3 前执行:
  import sys; sys.path.insert(0, "脚本目录")
  from remote_da3_patch import patch_da3_export
  patch_da3_export()
"""
import sys
import types


def patch_da3_export():
    """用桩模块屏蔽 DA3 的 export 依赖链"""
    # 屏蔽 depth_anything_3.utils.export (含 gs/glb/colmap/ply 等)
    export_stub = types.ModuleType('depth_anything_3.utils.export')
    export_stub.export = lambda *a, **kw: None
    sys.modules['depth_anything_3.utils.export'] = export_stub
    sys.modules['depth_anything_3.utils.export.gs'] = types.ModuleType(
        'depth_anything_3.utils.export.gs')
    sys.modules['depth_anything_3.utils.export.gs_video'] = types.ModuleType(
        'depth_anything_3.utils.export.gs_video')
    sys.modules['depth_anything_3.utils.export.colmap'] = types.ModuleType(
        'depth_anything_3.utils.export.colmap')
    sys.modules['depth_anything_3.utils.export.ply'] = types.ModuleType(
        'depth_anything_3.utils.export.ply')
    sys.modules['depth_anything_3.utils.export.glb'] = types.ModuleType(
        'depth_anything_3.utils.export.glb')
    sys.modules['depth_anything_3.utils.export.npz'] = types.ModuleType(
        'depth_anything_3.utils.export.npz')
    sys.modules['depth_anything_3.utils.export.feat_vis'] = types.ModuleType(
        'depth_anything_3.utils.export.feat_vis')
    sys.modules['depth_anything_3.utils.export.__init__'] = export_stub
    return export_stub


def patch_export_submodules():
    """更彻底: 屏蔽 export 包及其子模块的 __init__"""
    from importlib import import_module
    # 强制替换 export 包的 __path__ 为空, 阻止子模块加载
    try:
        mod = import_module('depth_anything_3.utils.export')
    except Exception:
        pass
    return patch_da3_export()
