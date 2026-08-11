#!/bin/bash
INC=/root/miniconda3/envs/rgb6d/include
for header in cublas_v2.h cusolver_common.h cusolverDn.h cublasLt.h cufft.h curand.h nvml.h; do
  if [ ! -f "$INC/$header" ]; then
    echo "// Stub header for $header" > "$INC/$header"
    echo "Created $header"
  fi
done
echo "All stubs created"
