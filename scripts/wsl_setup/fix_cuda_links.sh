#!/bin/bash
TP=/root/miniconda3/envs/rgb6d/lib/python3.10/site-packages/torch/lib
ENV=/root/miniconda3/envs/rgb6d/lib

# Remove old CUDA 13 symlinks that point to wrong libs
rm -f $ENV/libcudart.so
rm -f $ENV/libcudart.so.13
rm -f $ENV/libcudart.so.13.3.29

# Create proper CUDA 12 symlinks
ln -sf $TP/libcudart-9335f6a2.so.12 $ENV/libcudart.so
ln -sf $TP/libcudart-9335f6a2.so.12 $ENV/libcudart.so.13
ln -sf $TP/libcudart-9335f6a2.so.12 $ENV/libcudart.so.13.3.29

echo "=== Fixed symlinks ==="
ls -la $ENV/libcudart*
