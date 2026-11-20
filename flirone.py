from __future__ import annotations

import json
import math
import os
import struct
import time
from dataclasses import dataclass, field

import numpy as np
import usb.backend.libusb1
import usb.core
import usb.util

VENDOR_ID = 0x09CB
PRODUCT_ID = 0x1996
MAGIC = b"\xef\xbe\x00\x00"
HEADER_LEN = 28
EP_FRAME = 0x85
EP_FILEIO = 0x83
EP_IAP = 0x81
MAX_FRAME = 1 << 20
PACKET_WORDS = 82


@dataclass
class Calibration:
    R1: float = 16528.178
    R2: float = 0.012258549
    B: float = 1427.5
    F: float = 1.0
    O: float = -1307.0
    emissivity: float = 0.95
    reflected_c: float = 20.0
    raw_scale: float = 4.0


def counts_to_celsius(counts, cal: Calibration = Calibration()):
    raw = np.asarray(counts, dtype=np.float64) * cal.raw_scale
    refl = cal.R1 / (cal.R2 * (math.exp(cal.B / (cal.reflected_c + 273.15)) - cal.F)) - cal.O
    obj = (raw - (1 - cal.emissivity) * refl) / cal.emissivity
    with np.errstate(divide="ignore", invalid="ignore"):
        return cal.B / np.log(cal.R1 / (cal.R2 * (obj + cal.O)) + cal.F) - 273.15


@dataclass
class Frame:
    serial: str
    timestamp: float
    thermal: np.ndarray
    jpeg: bytes = b""
    status: dict = field(default_factory=dict)

    @property
    def ffc(self) -> bool:
        return self.status.get("shutterState") == "FFC"


def decode_thermal(payload: bytes) -> np.ndarray:
    words = np.frombuffer(payload[: len(payload) // 2 * 2], dtype="<u2")
    packets = len(words) // PACKET_WORDS
    shape = {240: (120, 160), 60: (60, 80)}.get(packets)
    if shape is None:
        raise ValueError(f"unrecognised thermal payload of {len(payload)} bytes")
    pixels = words[: packets * PACKET_WORDS].reshape(packets, PACKET_WORDS)[:, 2:]
    return pixels.reshape(shape).copy()


def parse_frame(buf: bytes, serial: str = "") -> Frame:
    frame_size, thermal_size, jpg_size, status_size = struct.unpack_from("<4I", buf, 8)
    body = buf[HEADER_LEN:HEADER_LEN + frame_size]
    status_raw = body[thermal_size + jpg_size:thermal_size + jpg_size + status_size]
    try:
        status = json.loads(status_raw.split(b"\x00")[0].decode("utf-8", "replace"))
    except ValueError:
        status = {"_raw": status_raw.decode("latin-1")}
    return Frame(serial, time.time(), decode_thermal(body[:thermal_size]),
                 body[thermal_size:thermal_size + jpg_size], status)


def _backend():
    for path in ("/opt/homebrew/lib/libusb-1.0.dylib", "/usr/local/lib/libusb-1.0.dylib"):
        if os.path.exists(path):
            return usb.backend.libusb1.get_backend(find_library=lambda _: path)
    return usb.backend.libusb1.get_backend()


def list_devices() -> list[usb.core.Device]:
    return list(usb.core.find(find_all=True, backend=_backend(),
                              idVendor=VENDOR_ID, idProduct=PRODUCT_ID))


class FlirOne:
    def __init__(self, device: usb.core.Device | None = None, serial: str | None = None):
        if device is None:
            devices = list_devices()
            if serial is not None:
                devices = [d for d in devices if d.serial_number == serial]
            if not devices:
                raise RuntimeError("FLIR ONE not found" + (f" (serial {serial})" if serial else ""))
            device = devices[0]
        self.dev = device
        self.serial = device.serial_number
        self._buf = bytearray()
        self._claimed = False

    def start(self) -> None:
        dev = self.dev
        try:
            active = dev.get_active_configuration().bConfigurationValue
        except usb.core.USBError:
            active = None
        if active != 3:
            dev.set_configuration(3)
        for i in (0, 1, 2):
            usb.util.claim_interface(dev, i)
        self._claimed = True
        self._set_interface(2, 0)
        self._set_interface(1, 0)
        self._set_interface(1, 1)
        self._set_interface(2, 1, b"\x00\x00")
        self._buf.clear()

    def stop(self) -> None:
        if self._claimed:
            for intf in (2, 1):
                try:
                    self._set_interface(intf, 0)
                except usb.core.USBError:
                    pass
            for i in (0, 1, 2):
                try:
                    usb.util.release_interface(self.dev, i)
                except usb.core.USBError:
                    pass
            self._claimed = False
        usb.util.dispose_resources(self.dev)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def _set_interface(self, interface: int, alt: int, data: bytes | None = None) -> None:
        self.dev.ctrl_transfer(0x01, 0x0B, alt, interface, data, timeout=500)

    def _bulk(self, ep: int, timeout: int) -> bytes:
        try:
            return bytes(self.dev.read(ep, MAX_FRAME, timeout=timeout))
        except usb.core.USBTimeoutError:
            return b""

    def read_frame(self, timeout: float = 5.0) -> Frame:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            chunk = self._bulk(EP_FRAME, 100)
            self._bulk(EP_IAP, 10)
            self._bulk(EP_FILEIO, 10)
            if not chunk:
                continue
            if chunk.startswith(MAGIC) or len(self._buf) + len(chunk) > MAX_FRAME:
                self._buf.clear()
            self._buf += chunk
            if not self._buf.startswith(MAGIC):
                self._buf.clear()
                continue
            if len(self._buf) < HEADER_LEN:
                continue
            frame_size = struct.unpack_from("<I", self._buf, 8)[0]
            if len(self._buf) < HEADER_LEN + frame_size:
                continue
            data = bytes(self._buf)
            self._buf.clear()
            try:
                return parse_frame(data, self.serial)
            except (ValueError, struct.error):
                continue
        raise TimeoutError(f"no frame from {self.serial} within {timeout}s")


class FakeCamera:

    def __init__(self, serial: str, shape: tuple[int, int] = (120, 160)):
        self.serial = serial
        self.shape = shape
        self._t0 = time.time()
        self._phase = sum(serial.encode()) % 100 / 10

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read_frame(self, timeout: float = 5.0) -> Frame:
        time.sleep(1 / 8.7)
        h, w = self.shape
        t = time.time() - self._t0 + self._phase
        y, x = np.mgrid[0:h, 0:w]
        cx, cy = w / 2 + w / 3 * math.cos(t / 2), h / 2 + h / 3 * math.sin(t / 3)
        blob = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (h / 6) ** 2))
        counts = (3100 + 650 * blob + np.random.normal(0, 3, (h, w))).astype("<u2")
        packets = np.zeros((h * w // 80, PACKET_WORDS), dtype="<u2")
        packets[:, 2:] = counts.reshape(-1, 80)
        thermal = packets.tobytes()
        status = json.dumps({"shutterState": "ON", "simulated": True}).encode() + b"\x00"
        header = MAGIC + struct.pack("<I4I", 1, len(thermal) + len(status), len(thermal), 0, len(status)) + b"\x00" * 4
        return parse_frame(header + thermal + status, self.serial)
