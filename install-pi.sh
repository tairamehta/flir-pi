#!/bin/sh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"

sudo apt-get update
sudo apt-get install -y python3-usb python3-numpy python3-pil libusb-1.0-0

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
sudo systemctl enable --now flird.service

echo "Running. Dashboard: http://$(hostname -I | awk '{print $1}'):8000/"
echo "Logs: journalctl -u flird -f"
