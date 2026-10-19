#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ] || [ ! -f models/depth-large.onnx ]; then
  echo '请先在终端运行 bash setup.sh 完成安装。'
  exit 1
fi
(sleep 2; open 'http://127.0.0.1:8765') &
exec .venv/bin/python app.py
