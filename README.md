# flir-pi

Run one or more FLIR ONE (USB-C) thermal cameras from a Raspberry Pi and watch them live in a web browser.

## Status

- Tested on a Raspberry Pi 4 with one FLIR ONE Pro (USB-C). Live thermal and normal video both work, at about 10 frames per second.
- Running several cameras at once has only been tested with simulated cameras.
- Temperature readings are rough. They read about 10 degrees too high in testing, so check them against something of known temperature before relying on them.
- This does not work on a Mac. macOS blocks the camera's video stream. Linux (the Pi) does not.

## What you need

- Raspberry Pi 3, 4, 5 or Zero 2 W, with a power supply and microSD card
- FLIR ONE or FLIR ONE Pro, USB-C version (not Lightning)
- A USB-C female to USB-A male adapter for each camera
- A computer on the same Wi-Fi as the Pi, IDEALLY NOT EDUROAM- use your hotspot!!

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

## Hot spots and people

The Pi does more than show video. It returns results as data.

**People.** A pretrained person detector (YOLOX) looks at the thermal picture about every two seconds and reports each person it finds, with a box and a confidence score. It was trained on ordinary photos, not thermal ones, so treat it as a good first pass. In testing it found people in 12 out of 12 public thermal photos shrunk to this camera's size. People are outlined with a thick green box.

**Hot spots.** A few times a second the Pi also:

1. Works out the room temperature (the middle value of the whole picture).
2. Finds every patch that is at least 3 degrees warmer than the room.
3. Remembers what the scene normally looks like, so fixed warm things (lights, vents, screens) are learned as background.
4. Labels each patch as `hot_object` (25 degrees or more above the room), `person_like` (new to the scene and 2 to 18 degrees above the room), or `warm_area` (everything else).

Limits to know about:

- The person detector has not been tuned on this camera. It can miss people who are far away, partly hidden or lying down.
- With several cameras, each one is checked in turn, so updates get slower.
- If the detector model is missing, the Pi falls back to the `person_like` rule. The `people_source` field says which one is in use: `model` or `rule`.

## Getting the data

| Address | What you get |
|---|---|
| `/api/cameras` | List of cameras with temperatures |
| `/api/analysis` | Hot spots and people for every camera |
| `/cam/SERIAL/analysis` | People, hot spots and an 8 by 6 grid of zone temperatures |
| `/cam/SERIAL/celsius.json` | Temperature of every pixel, as rows of numbers |
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
- `detector.py` finds people in the thermal picture.
- `install-pi.sh` sets everything up on the Pi.

## Credit and licence

The camera's USB protocol was worked out by tomas123, cynfab and others on the EEVblog forum, published at [fnoop/flirone-v4l2](https://github.com/fnoop/flirone-v4l2). Licence: GPL-2.0.
