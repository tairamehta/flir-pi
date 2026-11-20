#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

import flirone

log = logging.getLogger("flird")


def _palette() -> np.ndarray:
    anchors = [(0, (0, 0, 0)), (0.2, (60, 0, 110)), (0.45, (190, 20, 90)),
               (0.7, (250, 130, 10)), (0.9, (255, 225, 70)), (1, (255, 255, 255))]
    xs = [a[0] for a in anchors]
    grid = np.linspace(0, 1, 256)
    return np.stack([np.interp(grid, xs, [a[1][c] for a in anchors]) for c in range(3)], 1).astype(np.uint8)


PALETTE = _palette()


def render(thermal: np.ndarray, width: int = 640) -> bytes:
    lo, hi = np.percentile(thermal, (1, 99))
    norm = np.clip((thermal.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    img = Image.fromarray(PALETTE[(norm * 255).astype(np.uint8)], "RGB")
    scale = width // thermal.shape[1]
    img = img.resize((thermal.shape[1] * scale, thermal.shape[0] * scale), Image.BICUBIC)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue()


class CameraWorker(threading.Thread):
    def __init__(self, camera, key=None):
        super().__init__(daemon=True, name=f"cam-{camera.serial}")
        self.camera = camera
        self.serial = camera.serial
        self.key = key
        self.cond = threading.Condition()
        self.seq = 0
        self.frame: flirone.Frame | None = None
        self.thermal_jpg = b""
        self.visible_jpg = b""
        self.stats: dict = {}
        self.fps = 0.0
        self.alive = True

    def run(self) -> None:
        try:
            self.camera.start()
            log.info("%s: streaming", self.serial)
            last = None
            while self.alive:
                frame = self.camera.read_frame()
                now = time.monotonic()
                if last is not None:
                    self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3) if self.fps else 1 / max(now - last, 1e-3)
                last = now
                if frame.ffc:
                    continue
                celsius = flirone.counts_to_celsius(frame.thermal)
                h, w = frame.thermal.shape
                stats = {
                    "min_c": round(float(np.nanmin(celsius)), 1),
                    "max_c": round(float(np.nanmax(celsius)), 1),
                    "center_c": round(float(np.nanmean(celsius[h // 2 - 1:h // 2 + 1, w // 2 - 1:w // 2 + 1])), 1),
                    "resolution": [w, h],
                    "timestamp": frame.timestamp,
                }
                jpg = render(frame.thermal)
                with self.cond:
                    self.frame, self.stats, self.thermal_jpg = frame, stats, jpg
                    if frame.jpeg:
                        self.visible_jpg = frame.jpeg
                    self.seq += 1
                    self.cond.notify_all()
        except Exception as e:
            log.warning("%s: stopped: %s", self.serial, e)
        finally:
            self.alive = False
            try:
                self.camera.stop()
            except Exception:
                pass
            with self.cond:
                self.cond.notify_all()

    def info(self) -> dict:
        return {"serial": self.serial, "fps": round(self.fps, 1), **self.stats}


class Manager:
    def __init__(self, fake: int = 0, csv_path: str | None = None):
        self.workers: dict[str, CameraWorker] = {}
        self.lock = threading.Lock()
        self.fake = fake
        self.csv_path = csv_path

    def _add(self, worker: CameraWorker) -> None:
        with self.lock:
            self.workers[worker.serial] = worker
        worker.start()

    def scan(self) -> None:
        with self.lock:
            for serial in [s for s, w in self.workers.items() if not w.alive]:
                del self.workers[serial]
            known = {w.key for w in self.workers.values()}
            have = set(self.workers)
        if self.fake:
            for i in range(self.fake):
                serial = f"FAKE{i + 1:03d}"
                if serial not in have:
                    self._add(CameraWorker(flirone.FakeCamera(serial, (60, 80) if i % 2 else (120, 160))))
            return
        for dev in flirone.list_devices():
            key = (dev.bus, dev.address)
            if key in known:
                continue
            try:
                self._add(CameraWorker(flirone.FlirOne(dev), key))
            except Exception as e:
                log.warning("device %s: cannot open: %s", key, e)

    def run(self) -> None:
        writer = None
        if self.csv_path:
            f = open(self.csv_path, "a", newline="")
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["time", "serial", "min_c", "center_c", "max_c"])
        while True:
            try:
                self.scan()
            except Exception as e:
                log.warning("scan failed: %s", e)
            if writer:
                for info in self.list():
                    if "min_c" in info:
                        writer.writerow([f"{time.time():.1f}", info["serial"], info["min_c"], info["center_c"], info["max_c"]])
                f.flush()
            time.sleep(2)

    def list(self) -> list[dict]:
        with self.lock:
            return [w.info() for w in sorted(self.workers.values(), key=lambda w: w.serial)]

    def get(self, serial: str) -> CameraWorker | None:
        with self.lock:
            return self.workers.get(serial)


PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>FLIR ONE cameras</title><style>
body{margin:0;padding:16px;background:#111;color:#eee;font:15px system-ui,sans-serif}
h1{font-size:18px;margin:0 0 12px}#grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}
.card{background:#1c1c1c;border-radius:8px;overflow:hidden}.card img{width:100%;display:block;image-rendering:auto;background:#000}
.views{display:grid;grid-template-columns:1fr 1fr;gap:2px}.meta{padding:10px 12px;display:flex;flex-wrap:wrap;gap:4px 14px;align-items:baseline}
.meta b{font-size:16px}.dim{color:#999;font-size:13px}.t{font-variant-numeric:tabular-nums}#empty{color:#999}
</style></head><body><h1>FLIR ONE cameras <span class=dim id=count></span></h1><div id=grid></div><p id=empty hidden>No cameras connected.</p>
<script>
const grid=document.getElementById('grid'),cards={};
function card(s){const d=document.createElement('div');d.className='card';
d.innerHTML=`<div class=views><img src="/cam/${s}/thermal.mjpg" alt="thermal"><img src="/cam/${s}/visible.mjpg" alt="visible"></div>
<div class=meta><b>${s}</b><span class="t" data-k=temps></span><span class="dim t" data-k=info></span></div>`;grid.append(d);return d}
async function tick(){try{const cams=await (await fetch('/api/cameras')).json();const seen=new Set();
for(const c of cams){seen.add(c.serial);const d=cards[c.serial]||(cards[c.serial]=card(c.serial));
d.querySelector('[data-k=temps]').textContent=c.min_c===undefined?'starting…':`min ${c.min_c}° · centre ${c.center_c}° · max ${c.max_c}°C`;
d.querySelector('[data-k=info]').textContent=c.resolution?`${c.resolution[0]}×${c.resolution[1]} · ${c.fps} fps`:''}
for(const s in cards)if(!seen.has(s)){cards[s].remove();delete cards[s]}
document.getElementById('count').textContent=cams.length?`(${cams.length})`:'';document.getElementById('empty').hidden=cams.length>0}catch(e){}}
tick();setInterval(tick,1000);
</script></body></html>"""


def make_handler(manager: Manager):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _npy(self, arr: np.ndarray) -> None:
            out = io.BytesIO()
            np.save(out, arr)
            self._send(out.getvalue(), "application/octet-stream")

        def _mjpeg(self, worker: CameraWorker, attr: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            seq = -1
            while worker.alive:
                with worker.cond:
                    worker.cond.wait_for(lambda: worker.seq != seq or not worker.alive, timeout=5)
                    seq, jpg = worker.seq, getattr(worker, attr)
                if jpg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n")

        def do_GET(self):
            try:
                path = self.path.split("?")[0]
                if path == "/":
                    return self._send(PAGE.encode(), "text/html; charset=utf-8")
                if path == "/api/cameras":
                    return self._send(json.dumps(manager.list()).encode(), "application/json")
                parts = path.strip("/").split("/")
                worker = manager.get(parts[1]) if len(parts) == 3 and parts[0] == "cam" else None
                if worker is None:
                    return self._send(b"not found\n", "text/plain", 404)
                what = parts[2]
                if what in ("thermal.mjpg", "visible.mjpg"):
                    return self._mjpeg(worker, what.split(".")[0] + "_jpg")
                if worker.frame is None:
                    return self._send(b"no frame yet\n", "text/plain", 503)
                if what == "thermal.jpg":
                    return self._send(worker.thermal_jpg, "image/jpeg")
                if what == "visible.jpg" and worker.visible_jpg:
                    return self._send(worker.visible_jpg, "image/jpeg")
                if what == "raw.npy":
                    return self._npy(worker.frame.thermal)
                if what == "celsius.npy":
                    return self._npy(flirone.counts_to_celsius(worker.frame.thermal).astype(np.float32))
                self._send(b"not found\n", "text/plain", 404)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-camera FLIR ONE server")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--fake", type=int, default=0, metavar="N", help="serve N simulated cameras instead of USB ones")
    ap.add_argument("--csv", metavar="FILE", help="append min/centre/max per camera to FILE every 2 s")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    manager = Manager(args.fake, args.csv)
    threading.Thread(target=manager.run, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(manager))
    server.daemon_threads = True
    log.info("dashboard on http://%s:%d/", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
