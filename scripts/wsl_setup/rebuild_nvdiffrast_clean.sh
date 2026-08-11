#!/bin/bash
set -e
# Rebuild nvdiffrast linking against PyTorch's CUDA 12 directly
TP=/root/miniconda3/envs/rgb6d/lib/python3.10/site-packages/torch
ENV=/root/miniconda3/envs/rgb6d

export CUDA_HOME=$ENV  # nvcc + headers from conda
export TORCH_CUDA_ARCH_LIST="8.6"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export MAX_JOBS=2

# Override library paths to use torch CUDA 12
export LIBRARY_PATH="$TP/lib:$TP/lib/stubs:$ENV/lib"
export LD_LIBRARY_PATH="$TP/lib:$TP/lib/stubs:$ENV/lib"

cd /mnt/e/zhijiyige/src/nvdiffrast/nvdiffrast-main
rm -rf build dist *.egg-info

# Link only against torch's libcudart
python=$ENV/bin/python

# Build
$python setup.py build_ext --inplace 2>&1 | tail -5

echo "=== Done, checking .so ==="
ls -la _nvdiffrast_c*.so
readelf -d _nvdiffrast_c*.so | grep NEEDED | grep cuda

# Test
$python -c "
import torch; torch.cuda.set_device(0)
import nvdiffrast.torch as dr
ctx=dr.RasterizeCudaContext()
print('nvdiffrast OK!')
"
