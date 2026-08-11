#!/bin/bash
set -e

# Environment
export CUDA_HOME=/root/miniconda3/envs/rgb6d
export CUDA_TOOLKIT_ROOT_DIR=/root/miniconda3/envs/rgb6d
export TORCH_CUDA_ARCH_LIST="8.6"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export MAX_JOBS=1

echo "=== nvcc test ==="
$CUDA_HOME/bin/nvcc --version | head -1

echo "=== Building nvdiffrast ==="
cd /mnt/e/zhijiyige/src/nvdiffrast/nvdiffrast-main
rm -rf build dist *.egg-info 2>/dev/null

export LIBRARY_PATH="/root/miniconda3/envs/rgb6d/targets/x86_64-linux/lib:/root/miniconda3/envs/rgb6d/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/rgb6d/targets/x86_64-linux/lib:/root/miniconda3/envs/rgb6d/lib:$LD_LIBRARY_PATH"
/root/miniconda3/envs/rgb6d/bin/python setup.py build_ext --inplace 2>&1 | tail -5

echo "=== Checking result ==="
find . -name "*.so" 2>/dev/null
if [ $? -eq 0 ] && [ -n "$(find . -name '*.so' 2>/dev/null)" ]; then
    echo "BUILD SUCCESS"
else
    echo "BUILD FAILED"
fi
