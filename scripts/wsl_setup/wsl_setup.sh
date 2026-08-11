#!/bin/bash
set -e
export PATH=$HOME/miniconda3/bin:$PATH

# Accept conda TOS first
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "=== Creating conda env ==="
conda create -n rgb6d python=3.10 -y 2>&1 | tail -5
echo "=== Done ==="

