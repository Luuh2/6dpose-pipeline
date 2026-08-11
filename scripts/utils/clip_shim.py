"""
clip_shim.py — 用 open_clip_torch 替代 openai/CLIP, 提供 ultralytics 需要的接口
避免因 GitHub 不可达导致 CLIP 安装失败
"""

import open_clip
import torch

# 映射 open_clip → clip API
_tokenizer = None
_model_cache = {}
_preprocess_cache = {}

def tokenize(texts, context_length=77, truncate=False):
    """ultralytics 需要的 tokenize 接口 (兼容 truncate 参数)"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = open_clip.get_tokenizer('ViT-B-32')
    if isinstance(texts, str):
        texts = [texts]
    tokens = _tokenizer(texts)
    # open_clip 的 tokenizer 自动截断到 77, 不需要额外处理
    return tokens

def load(name, device='cpu', jit=False, download_root=None):
    """ultralytics 需要的 load 接口

    YOLO-World 使用 CLIP 的 ViT-B/32 文本编码器
    """
    if name not in _model_cache:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k')
        model = model.to(device)
        model.eval()
        _model_cache[name] = model
        _preprocess_cache[name] = preprocess
    return _model_cache[name], _preprocess_cache.get(name, None)

def available_models():
    return ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50']

# 暴露 encode_text 供 ultralytics 内部调用
def encode_text(model, text_tokens, **kwargs):
    """兼容 openai/CLIP 的 encode_text 接口"""
    if hasattr(model, 'encode_text'):
        return model.encode_text(text_tokens, **kwargs)
    # open_clip 模型使用不同的 API
    return model.encode_text(text_tokens)
