#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p models
if [ ! -f models/depth-large.onnx ]; then
  curl -L --fail --retry 3 'https://huggingface.co/onnx-community/depth-anything-v2-large/resolve/main/onnx/model.onnx' -o models/depth-large.onnx.part
  mv models/depth-large.onnx.part models/depth-large.onnx
fi
printf '\n安装完成。运行 ./start.command 启动高精度版本。\n'
