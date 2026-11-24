#!/bin/sh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"

sudo apt-get update
sudo apt-get install -y python3-usb python3-numpy python3-pil python3-opencv libusb-1.0-0

MODEL="$DIR/models/yolox_s.onnx"
MODEL_SHA="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
mkdir -p "$DIR/models"
if ! echo "$MODEL_SHA  $MODEL" | sha256sum -c - >/dev/null 2>&1; then
  curl -L --retry 3 -o "$MODEL" https://huggingface.co/opencv/object_detection_yolox/resolve/main/object_detection_yolox_2022nov.onnx
  echo "$MODEL_SHA  $MODEL" | sha256sum -c - || { echo "Model download failed. Run this script again."; rm -f "$MODEL"; }
fi

echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="09cb", ATTR{idProduct}=="1996", MODE="0666"' |
  sudo tee /etc/udev/rules.d/99-flirone.rules >/dev/null
sudo udevadm control --reload-rules
sudo udevadm trigger

sudo tee /etc/systemd/system/flird.service >/dev/null <<UNIT
[Unit]
Description=FLIR ONE multi-camera server
After=network.target

[Service]
User=$RUN_USER
WorkingDirectory=$DIR
ExecStart=/usr/bin/python3 $DIR/flird.py --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable flird.service
sudo systemctl restart flird.service

echo "Running. Dashboard: http://$(hostname -I | awk '{print $1}'):8000/"
echo "Logs: journalctl -u flird -f"
