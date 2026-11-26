from __future__ import annotations

import os
import threading

import numpy as np

import flirone

try:
    import cv2
except ImportError:
    cv2 = None

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolox_s.onnx")
SIZE = 640
PERSON_CLASS = 0


class PersonDetector:
    def __init__(self, path: str = MODEL_PATH, confidence: float = 0.4):
        self.net = cv2.dnn.readNet(path)
        self.confidence = confidence
        self.lock = threading.Lock()
        grids = []
        for stride in (8, 16, 32):
            n = SIZE // stride
            ys, xs = np.mgrid[0:n, 0:n]
            grids.append(np.stack([xs.ravel(), ys.ravel(), np.full(n * n, stride)], 1))
        self.grid = np.concatenate(grids).astype(np.float32)

    def detect(self, thermal: np.ndarray) -> list[dict]:
        h, w = thermal.shape
        grey = (flirone.normalise(thermal) * 255).astype(np.uint8)
        scale = min(SIZE / h, SIZE / w)
        resized = cv2.resize(grey, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((SIZE, SIZE), 114, np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR).astype(np.float32))
        with self.lock:
            self.net.setInput(blob)
            out = self.net.forward()[0]
        score = out[:, 4] * out[:, 5 + PERSON_CLASS]
        keep = score > self.confidence
        if not keep.any():
            return []
        centre = (out[keep, :2] + self.grid[keep, :2]) * self.grid[keep, 2:3]
        size = np.exp(out[keep, 2:4]) * self.grid[keep, 2:3]
        boxes = np.concatenate([centre - size / 2, size], 1) / scale
        scores = score[keep]
        people = []
        for i in np.array(cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.confidence, 0.45)).ravel():
            x, y, bw, bh = boxes[i]
            x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
            x1, y1 = min(int(round(x + bw)), w), min(int(round(y + bh)), h)
            if x1 > x0 and y1 > y0:
                people.append({"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0, "score": round(float(scores[i]), 2)})
        return people


def load() -> PersonDetector | None:
    if cv2 is None or not os.path.exists(MODEL_PATH):
        return None
    return PersonDetector()
