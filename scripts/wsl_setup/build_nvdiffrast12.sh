#!/bin/bash
export CUDA_HOME=/root/miniconda3/envs/rgb6d
export CUDA_TOOLKIT_ROOT_DIR=/root/miniconda3/envs/rgb6d
export TORCH_CUDA_ARCH_LIST="8.6"

cd /mnt/e/zhijiyige/src/nvdiffrast/nvdiffrast-main
rm -rf build dist *.egg-info 2>/dev/null

CUDA_HOME=/root/miniconda3/envs/rgb6d \
/root/miniconda3/envs/rgb6d/bin/pip install --no-build-isolation -e . 2>&1 | tail -15
