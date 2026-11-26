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
from PIL import Image, ImageDraw

import detector
import flirone

log = logging.getLogger("flird")


def _palette() -> np.ndarray:
    anchors = [(0, (0, 0, 0)), (0.2, (60, 0, 110)), (0.45, (190, 20, 90)),
               (0.7, (250, 130, 10)), (0.9, (255, 225, 70)), (1, (255, 255, 255))]
    xs = [a[0] for a in anchors]
    grid = np.linspace(0, 1, 256)
    return np.stack([np.interp(grid, xs, [a[1][c] for a in anchors]) for c in range(3)], 1).astype(np.uint8)


PALETTE = _palette()


OFFSET_C = 0.0
DELTA_C = 3.0
MIN_AREA_PX = 20
PERSON_ABOVE_ROOM_C = (2.0, 18.0)
PERSON_MIN_FRACTION = 0.01
HOT_OBJECT_ABOVE_ROOM_C = 25.0
NEW_C = 1.5
BACKGROUND_SECONDS = 120.0
BOX_COLOURS = {"person_like": (0, 255, 120), "hot_object": (255, 255, 255), "warm_area": (90, 200, 255)}


def to_celsius(thermal: np.ndarray, offset: float = 0.0) -> np.ndarray:
    return flirone.counts_to_celsius(thermal) + OFFSET_C + offset


def label_blobs(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    h, w = mask.shape
    grid = mask.tolist()
    blobs = []
    for sy in range(h):
        row = grid[sy]
        for sx in range(w):
            if not row[sx]:
                continue
            row[sx] = False
            stack = [(sy, sx)]
            ys, xs = [], []
            while stack:
                y, x = stack.pop()
                ys.append(y)
                xs.append(x)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and grid[ny][nx]:
                        grid[ny][nx] = False
                        stack.append((ny, nx))
            blobs.append((np.array(ys), np.array(xs)))
    return blobs


def analyze(celsius: np.ndarray, background: np.ndarray | None = None) -> dict:
    h, w = celsius.shape
    ambient = float(np.nanmedian(celsius))
    hotspots = []
    for ys, xs in label_blobs(np.nan_to_num(celsius, nan=ambient) > ambient + DELTA_C):
        if len(ys) < MIN_AREA_PX:
            continue
        values = celsius[ys, xs]
        mean, peak = float(values.mean()), float(values.max())
        fraction = len(ys) / (h * w)
        change = float((values - background[ys, xs]).mean()) if background is not None else 0.0
        is_new = background is None or change >= NEW_C
        if peak - ambient >= HOT_OBJECT_ABOVE_ROOM_C:
            kind = "hot_object"
        elif is_new and PERSON_ABOVE_ROOM_C[0] <= mean - ambient <= PERSON_ABOVE_ROOM_C[1] and fraction >= PERSON_MIN_FRACTION:
            kind = "person_like"
        else:
            kind = "warm_area"
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        hotspots.append({
            "kind": kind,
            "x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1,
            "center": [round(float(xs.mean()) / w, 3), round(float(ys.mean()) / h, 3)],
            "area_px": len(ys),
            "area_fraction": round(fraction, 4),
            "mean_c": round(mean, 1),
            "max_c": round(peak, 1),
            "above_ambient_c": round(mean - ambient, 1),
            "above_background_c": round(change, 1),
            "new": bool(background is not None and change >= NEW_C),
        })
    hotspots.sort(key=lambda b: -b["area_px"])
    zones = celsius.reshape(6, h // 6, 8, w // 8).mean(axis=(1, 3))
    return {
        "ambient_c": round(ambient, 1),
        "person_like": sum(b["kind"] == "person_like" for b in hotspots),
        "hotspots": hotspots,
        "zones_c": [[round(float(v), 1) for v in row] for row in zones],
    }


def render(thermal: np.ndarray, hotspots: list[dict] = (), people: list[dict] = (), width: int = 640) -> bytes:
    img = Image.fromarray(PALETTE[(flirone.normalise(thermal) * 255).astype(np.uint8)], "RGB")
    scale = width // thermal.shape[1]
    img = img.resize((thermal.shape[1] * scale, thermal.shape[0] * scale), Image.BICUBIC)
    draw = ImageDraw.Draw(img)
    for b in hotspots:
        box = [b["x"] * scale, b["y"] * scale, (b["x"] + b["width"]) * scale - 1, (b["y"] + b["height"]) * scale - 1]
        draw.rectangle(box, outline=BOX_COLOURS[b["kind"]], width=2)
    for p in people:
        box = [p["x"] * scale, p["y"] * scale, (p["x"] + p["width"]) * scale - 1, (p["y"] + p["height"]) * scale - 1]
        draw.rectangle(box, outline=(0, 255, 0), width=4)
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
        self.analysis: dict = {}
        self.shutter_check: dict = {}
        self.background: np.ndarray | None = None
        self.background_time = 0.0
        self.people: list[dict] | None = None
        self.offset = 0.0
        self.calibrated = False
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
                    shutter_k = frame.status.get("shutterTemperature")
                    if shutter_k:
                        measured = float(np.nanmedian(flirone.counts_to_celsius(frame.thermal)))
                        target = shutter_k - 273.15 - measured
                        if abs(target) < 25:
                            previous = self.offset
                            self.offset = target if not self.calibrated else 0.8 * self.offset + 0.2 * target
                            self.calibrated = True
                            if self.background is not None:
                                self.background += self.offset - previous
                        self.shutter_check = {
                            "measured_c": round(measured, 1),
                            "shutter_c": round(shutter_k - 273.15, 1),
                            "offset_c": round(self.offset, 1),
                        }
                    continue
                celsius = to_celsius(frame.thermal, self.offset)
                h, w = frame.thermal.shape
                stats = {
                    "min_c": round(float(np.nanmin(celsius)), 1),
                    "max_c": round(float(np.nanmax(celsius)), 1),
                    "center_c": round(float(np.nanmean(celsius[h // 2 - 1:h // 2 + 1, w // 2 - 1:w // 2 + 1])), 1),
                    "resolution": [w, h],
                    "timestamp": frame.timestamp,
                }
                if self.seq % 3 == 0 or not self.analysis:
                    if self.background is None or self.background.shape != celsius.shape:
                        self.background = celsius.copy()
                    else:
                        rate = min((now - self.background_time) / BACKGROUND_SECONDS, 1.0)
                        self.background += rate * (celsius - self.background)
                    self.background_time = now
                    analysis = {"serial": self.serial, "timestamp": frame.timestamp, **analyze(celsius, self.background)}
                    if self.shutter_check:
                        analysis["shutter_check"] = self.shutter_check
                else:
                    analysis = self.analysis
                people = self.people
                if people is None:
                    analysis = {**analysis, "people_source": "rule", "people_count": analysis["person_like"], "people": []}
                else:
                    analysis = {**analysis, "people_source": "model", "people_count": len(people), "people": people}
                stats["hotspots"] = len(analysis["hotspots"])
                stats["person_like"] = analysis["person_like"]
                analysis["presence"] = analysis["people_count"] > 0 or analysis["person_like"] > 0
                analysis["calibration"] = "shutter" if self.calibrated else "uncalibrated"
                stats["people"] = analysis["people_count"]
                stats["people_source"] = analysis["people_source"]
                stats["presence"] = analysis["presence"]
                stats["calibration"] = analysis["calibration"]
                jpg = render(frame.thermal, analysis["hotspots"], analysis["people"])
                with self.cond:
                    self.frame, self.stats, self.thermal_jpg, self.analysis = frame, stats, jpg, analysis
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
        self.detector = detector.load()

    def detect_loop(self) -> None:
        while True:
            with self.lock:
                workers = list(self.workers.values())
            for worker in workers:
                frame = worker.frame
                if frame is None or not worker.alive:
                    continue
                try:
                    worker.people = self.detector.detect(frame.thermal)
                except Exception as e:
                    log.warning("%s: detector failed: %s", worker.serial, e)
            time.sleep(1.0)

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

    def analyses(self) -> list[dict]:
        with self.lock:
            return [w.analysis for w in sorted(self.workers.values(), key=lambda w: w.serial) if w.analysis]

    def get(self, serial: str) -> CameraWorker | None:
        with self.lock:
            return self.workers.get(serial)


PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>FLIR ONE cameras</title><style>
body{margin:0;padding:16px;background:#111;color:#eee;font:15px system-ui,sans-serif}
h1{font-size:18px;margin:0 0 12px}h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:#999;margin:0 0 6px;font-weight:600}
#grid{display:grid;gap:16px}.card{background:#1c1c1c;border-radius:8px;overflow:hidden;display:grid;grid-template-columns:minmax(0,1fr) 340px}
.views{display:grid;grid-template-columns:1fr 1fr;gap:2px;align-content:start}.views img{width:100%;display:block;background:#000;aspect-ratio:4/3;object-fit:cover}
.side{padding:14px 16px;display:grid;gap:16px;align-content:start;border-left:1px solid #2a2a2a;min-width:0}
.head{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.head b{font-size:17px}
.flag{padding:2px 10px;border-radius:999px;font-weight:600;font-size:13px;background:#333;color:#bbb}.flag.person{background:#0c6;color:#031}.flag.warm{background:#fa3;color:#320}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 14px}.kv span:nth-child(odd){color:#999}.t,.kv,.zones,table{font-variant-numeric:tabular-nums}
.zones{display:grid;grid-template-columns:repeat(8,1fr);gap:2px}.zones div{aspect-ratio:1;display:grid;place-items:center;font-size:11px;border-radius:2px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:2px 8px 2px 0;font-weight:400}th{color:#999}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}.dim{color:#999;font-size:13px}
.links{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px}a{color:#6cf;text-decoration:none}a:hover{text-decoration:underline}
#empty{color:#999}@media(max-width:900px){.card{grid-template-columns:1fr}.side{border-left:0;border-top:1px solid #2a2a2a}}
</style></head><body><h1>FLIR ONE cameras <span class=dim id=count></span></h1><div id=grid></div><p id=empty hidden>No cameras connected.</p>
<script>
const grid=document.getElementById('grid'),cards={};
const KIND={person_like:['#00ff78','Person-like'],hot_object:['#ffffff','Hot object'],warm_area:['#5ac8ff','Warm area']};
const IRON=[[0,[0,0,0]],[.2,[60,0,110]],[.45,[190,20,90]],[.7,[250,130,10]],[.9,[255,225,70]],[1,[255,255,255]]];
function iron(t){t=Math.max(0,Math.min(1,t));for(let i=1;i<IRON.length;i++){if(t<=IRON[i][0]){const a=IRON[i-1],b=IRON[i],k=(t-a[0])/(b[0]-a[0]);return a[1].map((v,j)=>Math.round(v+(b[1][j]-v)*k))}}return[255,255,255]}
function where(c){return(c[1]<.34?'top':c[1]>.66?'bottom':'middle')+' '+(c[0]<.34?'left':c[0]>.66?'right':'centre')}
function live(img,url,gap){const next=()=>{if(!img.isConnected)return;img.src=url+'?t='+Date.now()};img.onload=()=>setTimeout(next,gap);img.onerror=()=>setTimeout(next,1000);next()}
function card(s){const d=document.createElement('div');d.className='card';
d.innerHTML=`<div class=views><img data-k=thermal alt="thermal"><img data-k=visible alt="visible"></div>
<aside class=side><div class=head><b>${s}</b><span class=flag data-k=flag></span></div>
<section><h2>Temperature</h2><div class=kv data-k=temps></div></section>
<section><h2>Heat map by zone (°C)</h2><div class=zones data-k=zones></div></section>
<section><h2>Hot spots</h2><div data-k=spots></div></section>
<section><h2>People</h2><div data-k=people></div></section>
<section><h2>Data</h2><div class=links><a href="/cam/${s}/analysis">Analysis (JSON)</a><a href="/cam/${s}/celsius.json">Full heat map (JSON)</a><a href="/cam/${s}/celsius.npy">Full heat map (NumPy)</a><a href="/cam/${s}/raw.npy">Raw sensor values</a><a href="/cam/${s}/thermal.jpg">Thermal picture</a><a href="/cam/${s}/visible.jpg">Normal picture</a></div></section>
<div class="dim t" data-k=info></div></aside>`;
grid.append(d);live(d.querySelector('[data-k=thermal]'),`/cam/${s}/thermal.jpg`,120);live(d.querySelector('[data-k=visible]'),`/cam/${s}/visible.jpg`,400);return d}
function fill(d,c,a){const q=k=>d.querySelector(`[data-k=${k}]`);
const f=q('flag'),person=c.people>0&&c.people_source=='model';
f.className='flag'+(person?' person':c.person_like>0?' warm':'');f.textContent=person?`PERSON DETECTED (${c.people})`:c.person_like>0?'WARM BODY IN VIEW':'Nothing detected';
if(c.min_c===undefined){q('temps').textContent='starting…';return}
q('temps').innerHTML=`<span>Room</span><span>${a?a.ambient_c:'–'}°</span><span>Coolest</span><span>${c.min_c}°</span><span>Centre</span><span>${c.center_c}°</span><span>Hottest</span><span>${c.max_c}°</span><span>Accuracy</span><span>${c.calibration=='shutter'?'corrected':'reads high until first self-calibration'}</span>`;
q('info').textContent=`${c.resolution[0]}×${c.resolution[1]} · ${c.fps} frames/s`;
if(!a)return;
const z=a.zones_c.flat(),lo=Math.min(...z),hi=Math.max(...z),span=Math.max(hi-lo,2),base=(lo+hi)/2-span/2;
q('zones').innerHTML=z.map(v=>{const t=(v-base)/span,rgb=iron(.15+.75*t);return`<div style="background:rgb(${rgb});color:${t>.6?'#000':'#fff'}">${v.toFixed(1)}</div>`}).join('');
q('spots').innerHTML=a.hotspots.length?`<table><tr><th>Type</th><th>Where</th><th>Size</th><th>Avg</th><th>Peak</th></tr>${a.hotspots.map(b=>`<tr><td><span class=dot style="background:${KIND[b.kind][0]}"></span>${KIND[b.kind][1]}${b.new?'':' (fixed)'}</td><td>${where(b.center)}</td><td>${(b.area_fraction*100).toFixed(1)}%</td><td>${b.mean_c}°</td><td>${b.max_c}°</td></tr>`).join('')}</table>`:'<span class=dim>None</span>';
q('people').innerHTML=a.people_source!='model'?'<span class=dim>Detector model not installed. Using the warm-body rule.</span>':a.people.length?`<table><tr><th>#</th><th>Where</th><th>Confidence</th></tr>${a.people.map((p,i)=>`<tr><td>${i+1}</td><td>${where([(p.x+p.width/2)/c.resolution[0],(p.y+p.height/2)/c.resolution[1]])}</td><td>${Math.round(p.score*100)}%</td></tr>`).join('')}</table>`:'<span class=dim>None</span>'}
async function tick(){try{const[cams,an]=await Promise.all([fetch('/api/cameras').then(r=>r.json()),fetch('/api/analysis').then(r=>r.json())]);
const by={};for(const a of an)by[a.serial]=a;const seen=new Set();
for(const c of cams){seen.add(c.serial);fill(cards[c.serial]||(cards[c.serial]=card(c.serial)),c,by[c.serial])}
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
                if path == "/api/analysis":
                    return self._send(json.dumps(manager.analyses()).encode(), "application/json")
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
                    return self._npy(to_celsius(worker.frame.thermal, worker.offset).astype(np.float32))
                if what == "celsius.json":
                    rows = np.round(to_celsius(worker.frame.thermal, worker.offset), 1).tolist()
                    return self._send(json.dumps(rows).encode(), "application/json")
                if what == "analysis":
                    return self._send(json.dumps(worker.analysis).encode(), "application/json")
                self._send(b"not found\n", "text/plain", 404)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    return Handler


def main() -> None:
    global OFFSET_C, DELTA_C
    ap = argparse.ArgumentParser(description="Multi-camera FLIR ONE server")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--fake", type=int, default=0, metavar="N", help="serve N simulated cameras instead of USB ones")
    ap.add_argument("--csv", metavar="FILE", help="append min/centre/max per camera to FILE every 2 s")
    ap.add_argument("--offset", type=float, default=0.0, metavar="C", help="degrees C added to every temperature")
    ap.add_argument("--delta", type=float, default=DELTA_C, metavar="C", help="degrees C above the room for a hot spot")
    args = ap.parse_args()
    OFFSET_C, DELTA_C = args.offset, args.delta
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    manager = Manager(args.fake, args.csv)
    threading.Thread(target=manager.run, daemon=True).start()
    if manager.detector:
        threading.Thread(target=manager.detect_loop, daemon=True).start()
    log.info("person detection: %s", "model" if manager.detector else "rule (no model installed)")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(manager))
    server.daemon_threads = True
    log.info("dashboard on http://%s:%d/", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
