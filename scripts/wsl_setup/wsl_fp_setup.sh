#!/bin/bash
# FoundationPose setup script — run from inside WSL2
# Usage: bash /mnt/e/zhijiyige/wsl_fp_setup.sh
set -e

echo "=== 1. Downloading FoundationPose core files via ghproxy ==="
mkdir -p /mnt/e/zhijiyige/src/FoundationPose
cd /mnt/e/zhijiyige/src/FoundationPose

FILES=(
  "https://ghproxy.net/https://raw.githubusercontent.com/NVlabs/FoundationPose/main/estimater.py"
  "https://ghproxy.net/https://raw.githubusercontent.com/NVlabs/FoundationPose/main/README.md"
)

for url in "${FILES[@]}"; do
  fname=$(basename "$url")
  echo "  Downloading $fname..."
  curl -sL "$url" -o "$fname"
  if [ -s "$fname" ] && [ "$(wc -c < "$fname")" -gt 100 ]; then
    echo "    OK ($(wc -c < "$fname") bytes)"
  else
    echo "    FAILED"
  fi
done

# 2. Install PyTorch (try conda first, then pip)
echo "=== 2. Installing PyTorch ==="
export PATH=$HOME/miniconda3/bin:$PATH

# Try pip with direct wheel URL (faster than full index scan)
/root/miniconda3/envs/rgb6d/bin/pip install --no-deps \
  https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --trusted-host download.pytorch.org \
  --timeout 600 2>&1 | tail -5 || echo "Torch install had issues, checking..."

# Check
/root/miniconda3/envs/rgb6d/bin/python -c "import torch; print('PyTorch OK:', torch.__version__)" 2>/dev/null || echo "PyTorch NOT installed yet"

echo "=== FoundationPose setup script complete ==="
