#!/bin/bash
set -e
export PATH=/root/miniconda3/bin:$PATH

echo "=== 1. Remove old CUDA 11.8 packages ==="
conda remove -n rgb6d cuda-nvcc cuda-cudart-dev cuda-nvrtc-dev cuda-profiler-api libcusparse -y 2>/dev/null || true

echo "=== 2. Install CUDA 12.1 nvcc + dev packages ==="
conda install -n rgb6d -c nvidia/label/cuda-12.1.0 -c nvidia -c conda-forge \
  cuda-nvcc cuda-cudart-dev cuda-nvrtc-dev cuda-profiler-api -y 2>&1 | tail -5

echo "=== 3. Install PyTorch 2.1.0+cu121 ==="
/root/miniconda3/envs/rgb6d/bin/pip install --force-reinstall \
  /mnt/e/zhijiyige/wheels/torch-2.1.0+cu121-cp310-cp310-linux_x86_64.whl 2>&1 | tail -5

echo "=== 4. Install torchvision 2.1 ==="
/root/miniconda3/envs/rgb6d/bin/pip install torchvision==0.16.0+cu121 \
  -f https://download.pytorch.org/whl/cu121/torch_stable.html 2>&1 | tail -3

echo "=== 5. Reinstall pytorch3d for CUDA 12.1 ==="
/root/miniconda3/envs/rgb6d/bin/pip install --force-reinstall pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html 2>&1 | tail -5 || \
  echo "pytorch3d wheel not found for cu121, trying conda..." && \
  conda install -n rgb6d -c conda-forge pytorch3d -y 2>&1 | tail -5

echo "=== 6. Verify ==="
/root/miniconda3/envs/rgb6d/bin/python -c "
import torch; print('torch:', torch.__version__, '| CUDA avail:', torch.cuda.is_available())
import torch.version; print('cuda version:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
import nvidia; print('cuda available:', torch.cuda._is_compiled())
" 2>&1 || echo "Verification had issues but continuing..."

echo "=== DONE ==="
