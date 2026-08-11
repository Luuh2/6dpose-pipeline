#!/bin/bash
SO=/mnt/e/zhijiyige/src/nvdiffrast/nvdiffrast-main/_nvdiffrast_c.cpython-310-x86_64-linux-gnu.so
TP=/root/miniconda3/envs/rgb6d/lib/python3.10/site-packages/torch/lib

echo "=== Current NEEDED ==="
readelf -d $SO | grep NEEDED

# Replace libcudart.so.13 with torch's libcudart
patchelf --replace-needed libcudart.so.13 libcudart-9335f6a2.so.12 $SO
echo "=== After patch ==="
readelf -d $SO | grep NEEDED | grep cuda
