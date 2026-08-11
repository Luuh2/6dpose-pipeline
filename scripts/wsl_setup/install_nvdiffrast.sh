#!/bin/bash
export CUDA_HOME=/root/miniconda3/envs/rgb6d
export CUDA_TOOLKIT_ROOT_DIR=/root/miniconda3/envs/rgb6d
export TORCH_CUDA_ARCH_LIST="8.6"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11

cd /mnt/e/zhijiyige/src/nvdiffrast/nvdiffrast-main
CUDA_HOME=/root/miniconda3/envs/rgb6d \
CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11 \
/root/miniconda3/envs/rgb6d/bin/pip install --no-build-isolation -e . 2>&1 | tail -15
