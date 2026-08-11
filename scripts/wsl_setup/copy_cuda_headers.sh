#!/bin/bash
# Copy Windows CUDA 12.1 headers to WSL conda env
SRC="/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include"
DST=/root/miniconda3/envs/rgb6d/include

cp -r "$SRC"/* "$DST"/ 2>/dev/null
echo "Headers copied to $DST"

for h in cuda_runtime.h cusparse.h cublas_v2.h cusolver_common.h; do
  if [ -f "$DST/$h" ]; then
    echo "  OK: $h"
  else
    echo "  MISSING: $h"
  fi
done
