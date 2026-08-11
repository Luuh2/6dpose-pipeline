#!/bin/bash
# Rebuild nvdiffrast using PyTorch-bundled CUDA (avoids WSL2 driver incompatibility)
export CUDA_HOME=/root/miniconda3/envs/rgb6d/lib/python3.10/site-packages/torch
export TORCH_CUDA_ARCH_LIST="8.6"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export MAX_JOBS=2

cd /mnt/e/zhijiyige/src/nvdiffrast/nvdiffrast-main
rm -rf build dist *.egg-info

# Use torch's bundled CUDA
/root/miniconda3/envs/rgb6d/bin/python setup.py build_ext --inplace 2>&1 | tail -5

echo "=== Result ==="
find . -name "*.so" -newer setup.py 2>/dev/null
