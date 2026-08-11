#!/bin/bash
SO=/mnt/e/zhijiyige/src/nvdiffrast/nvdiffrast-main/_nvdiffrast_c.cpython-310-x86_64-linux-gnu.so
TP_LIB=/root/miniconda3/envs/rgb6d/lib/python3.10/site-packages/torch/lib
ENV_LIB=/root/miniconda3/envs/rgb6d/lib

patchelf --set-rpath "$TP_LIB:$ENV_LIB" "$SO"
echo "RPATH set to:"
patchelf --print-rpath "$SO"
