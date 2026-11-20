# flir-pi

Run one or more FLIR ONE (USB-C) thermal cameras from a Raspberry Pi and watch them live in a web browser.

## Status

Read this first.

- The web page and server work. They were tested with simulated cameras.
- Reading from a real camera has **not** been tested on a Pi yet. That is the next job.
- This cannot be tested on a Mac. macOS blocks the camera's video stream. Linux (the Pi) does not.
- Temperature readings are rough until checked against something of known temperature.

## What you need

- Raspberry Pi 3, 4, 5 or Zero 2 W, with a power supply and microSD card
- FLIR ONE or FLIR ONE Pro, USB-C version (not Lightning)
- A USB-C female to USB-A male adapter for each camera
- A computer on the same Wi-Fi as the Pi

## 1. Prepare the SD card

1. Install [Raspberry Pi Imager](https://www.raspberrypi.com/software/) on your computer.
2. Choose your Pi model, then **Raspberry Pi OS Lite (64-bit)**, then your SD card.
3. When asked to customise settings, set:
   - Hostname: `flirpi`
   - A username and password (the examples below use `pi`)
   - Your Wi-Fi name and password
   - Enable SSH, with password login
4. Write the sd card (using the usb looking dongle, you push the micro sd card in the bottom open area of the usb stick before plugging it into your laptop), put it in the Pi and power it on. Wait two minutes.

## 2. Connect to the Pi

From a terminal on your computer:

    ssh pi@flirpi.local

Type `yes` if asked, then your password. If the name is not found, look up the Pi's IP address in your router and use that instead of `flirpi.local`.

## 3. Download and install

Run these on the Pi:

    curl -L https://github.com/tairamehta/flir-pi/archive/refs/heads/main.tar.gz | tar xz
    cd flir-pi-main
    sh install-pi.sh

The installer adds what the code needs, lets it use the cameras, and sets it to start every time the Pi boots.

## 4. Plug in the camera

1. Plug the camera's **phone connector** into the Pi using the adapter. The port on the bottom of the camera is only for charging.
2. Press the camera's power button and wait for a steady light.
3. On your computer, open `http://flirpi.local:8000/`.

Each camera you plug in gets its own tile with a live picture and its minimum, centre and maximum temperature.

The cameras run on their own batteries, which last about an hour. For longer runs, keep a charger plugged into the bottom port of each camera.

## If it does not work

See what the server is doing:

    journalctl -u flird -f

Check that the Pi can see the camera (look for `09cb:1996`):

    lsusb

Restart the server:

    sudo systemctl restart flird

## Try it without a camera

On any computer with Python 3:

    pip install pyusb numpy pillow
    python3 flird.py --fake 3

Then open `http://localhost:8000/`.

## Getting the data

| Address | What you get |
|---|---|
| `/api/cameras` | List of cameras with temperatures |
| `/cam/SERIAL/thermal.jpg` | Thermal picture |
| `/cam/SERIAL/visible.jpg` | Normal camera picture |
| `/cam/SERIAL/thermal.mjpg` | Live thermal video |
| `/cam/SERIAL/raw.npy` | Raw sensor values (NumPy file) |
| `/cam/SERIAL/celsius.npy` | Temperatures in Celsius (NumPy file) |

Replace `SERIAL` with the camera's serial number, shown on its tile.

To log temperatures to a file, add `--csv temps.csv` to the `ExecStart` line in `/etc/systemd/system/flird.service`, then run `sudo systemctl daemon-reload` and `sudo systemctl restart flird`.

## Files

- `flirone.py` talks to the camera over USB.
- `flird.py` is the web server and dashboard.
- `install-pi.sh` sets everything up on the Pi.

## Credit and licence

The camera's USB protocol was worked out by tomas123, cynfab and others on the EEVblog forum, published at [fnoop/flirone-v4l2](https://github.com/fnoop/flirone-v4l2). Licence: GPL-2.0.
