#!/usr/bin/env bash
# Downloads the ResNet-10 SSD face detector weights (OpenCV's sample model, ~5 MB).
set -euo pipefail
cd "$(dirname "$0")/../models"
f=res10_300x300_ssd_iter_140000_fp16.caffemodel
[ -f "$f" ] || curl -fsSL -o "$f" \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/$f
[ -f deploy.prototxt ] || curl -fsSL -o deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt
echo "models ready: $(pwd)"
