"""facesentry core: face detection, enrolling and recognition with OpenCV only.

Finds faces with a pretrained YOLOv8n-face model (Haar cascade as fallback) and tells whose face
it is with OpenCV's SFace recognizer, which compares 128-number face "fingerprints".
Everything runs on your own machine. Nothing is uploaded.
"""
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np


# Run from source: files live in the current folder.
# Run as the packaged Mac app: private data goes to Application Support, snapshots to Pictures.
if getattr(sys, "frozen", False):
    _DATA = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "FaceSentry")
    INTRUDER_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "FaceSentry Intruders")
else:
    _DATA = "."
    INTRUDER_DIR = "intruders"
KNOWN_DIR = os.path.join(_DATA, "known_faces")
EMBED_PATH = os.path.join(_DATA, "face_prints.npz")  # enrolled face fingerprints (SFace)
LOG_PATH = os.path.join(_DATA, "facesentry.log")
# YOLOv8n-face (pretrained, ONNX). Downloaded at build time into models/ and bundled in the app.
YOLO_FILE = "yolov8n-face.onnx"
YOLO_CONF = 0.60  # minimum YOLO confidence for a face
MIN_FACE = 60  # smallest face side in pixels that counts
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
# Cosine similarity between two face fingerprints: HIGHER means more alike. At or above this = a known
# face. 0.363 is the value recommended for SFace. Tune it live with - (looser) and + (stricter).
THRESHOLD = 0.363
COOLDOWN = 10  # seconds between intruder snapshots


def log(message):
    """Append a timestamped line to the log file, so you can always see what the app did."""
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
    try:
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def _model_dir():
    if getattr(sys, "frozen", False):
        return os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)), "models")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class HaarDetector:
    """Classic OpenCV Haar cascade. Fallback when the YOLO model is missing or fails to load."""
    name = "haar"

    def __init__(self):
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError("Could not load the OpenCV face detector.")

    def detect_full(self, frame):
        return [(box, None) for box in self.detect(frame)]

    def detect(self, frame):
        # minNeighbors=7 is strict: the default of 5 lets random objects pass as faces.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(90, 90))
        return [tuple(int(v) for v in b) for b in boxes]


class YoloDetector:
    """YOLOv8n-face (pretrained) run through OpenCV's dnn module. No torch needed."""
    name = "yolo"
    SIZE = 640

    def __init__(self, path):
        self.net = cv2.dnn.readNetFromONNX(path)

    def detect_full(self, frame):
        """Return [(box, landmarks)] where landmarks is 5 (x, y) points: eyes, nose, mouth corners."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.SIZE, self.SIZE), swapRB=True)
        self.net.setInput(blob)
        return parse_yolo_full(self.net.forward(), w, h)

    def detect(self, frame):
        return [box for box, _ in self.detect_full(frame)]


def parse_yolo_full(output, width, height, conf=YOLO_CONF, min_size=MIN_FACE):
    """Turn the model output into [(box, landmarks)] in frame pixels.

    Rows are [x1, y1, x2, y2, confidence, class, 5 x (x, y, visibility)] in 640x640 space,
    already de-duplicated. Weak, tiny or oddly shaped boxes are dropped.
    """
    arr = np.asarray(output)
    rows = arr.reshape(-1, arr.shape[-1])
    sx, sy = width / YoloDetector.SIZE, height / YoloDetector.SIZE
    found = []
    for row in rows:
        if row[4] < conf:
            continue
        x1, x2 = max(0, int(row[0] * sx)), min(width, int(row[2] * sx))
        y1, y2 = max(0, int(row[1] * sy)), min(height, int(row[3] * sy))
        bw, bh = x2 - x1, y2 - y1
        if bw < min_size or bh < min_size or not 0.5 <= bw / bh <= 1.6:
            continue
        lm = None
        if row.shape[0] >= 21:
            lm = row[6:21].reshape(5, 3)[:, :2] * np.array([sx, sy], dtype="float32")
        found.append(((x1, y1, bw, bh), lm))
    return found


def parse_yolo(output, width, height, conf=YOLO_CONF, min_size=MIN_FACE):
    """Same as parse_yolo_full but only the (x, y, w, h) boxes."""
    return [box for box, _ in parse_yolo_full(output, width, height, conf, min_size)]


_DETECTOR = None


def load_detector():
    """Best detector available: YOLO if its model loads, otherwise Haar. Cached."""
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    path = os.path.join(_model_dir(), YOLO_FILE)
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        _DETECTOR = YoloDetector(path)
        log("Face detector: YOLOv8n-face.")
    except Exception as err:
        log(f"YOLO model not available ({err}); using the Haar fallback detector.")
        _DETECTOR = HaarDetector()
    return _DETECTOR


def detect_faces(frame, detector):
    """Return a list of (x, y, w, h) face boxes in a BGR camera frame."""
    return detector.detect(frame)


def detect_faces_full(frame, detector):
    """Return [(box, landmarks or None)] for every face in a BGR camera frame."""
    return detector.detect_full(frame)


def faces_match(box_a, box_b, tolerance=0.5):
    """True when two face boxes are roughly the same face in consecutive frames."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    return (abs((ax + aw / 2) - (bx + bw / 2)) <= aw * tolerance and
            abs((ay + ah / 2) - (by + bh / 2)) <= ah * tolerance)


class FaceEmbedder:
    """OpenCV SFace: turns a face into a 128-number fingerprint that stays similar across head
    angles, lighting and expressions far better than the old LBPH recognizer."""

    def __init__(self, path):
        self.net = cv2.FaceRecognizerSF.create(path, "")

    def embed(self, frame, box, landmarks=None):
        """Fingerprint (unit length) for one face. With landmarks the face is straightened first."""
        if landmarks is not None:
            row = np.concatenate([np.array(box, dtype="float32"),
                                  np.asarray(landmarks, dtype="float32").reshape(-1),
                                  [1.0]]).astype("float32").reshape(1, -1)
            aligned = self.net.alignCrop(frame, row)
        else:
            x, y, w, h = box
            aligned = cv2.resize(frame[max(0, y):y + h, max(0, x):x + w], (112, 112))
        feat = self.net.feature(aligned).flatten().astype("float32")
        return feat / (np.linalg.norm(feat) + 1e-9)


_EMBEDDER = None


def load_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        path = os.path.join(_model_dir(), SFACE_FILE)
        if not os.path.exists(path):
            raise RuntimeError(f"The face recognition model is missing from the app ({SFACE_FILE}).")
        _EMBEDDER = FaceEmbedder(path)
        log("Face recognizer: SFace.")
    return _EMBEDDER


def has_enrollment(path=EMBED_PATH):
    return os.path.exists(path)


def save_enrollment(prints, labels, path=EMBED_PATH):
    """prints: list of unit-length fingerprints; labels: the person's name for each one."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, prints=np.array(prints, dtype="float32"), labels=np.array(labels))


def load_enrollment(path=EMBED_PATH):
    data = np.load(path)
    return data["prints"].astype("float32"), [str(x) for x in data["labels"]]


def identify(prints, labels, fingerprint, threshold=THRESHOLD):
    """Return (name or None, similarity). Similarity is the best cosine match against enrolled prints."""
    sims = prints @ fingerprint
    best = int(np.argmax(sims))
    score = float(sims[best])
    if score >= threshold:
        return labels[best], score
    return None, score


def snapshot_name(now=None, prefix="intruder"):
    now = now or datetime.now()
    return f"{prefix}_{now:%Y%m%d_%H%M%S}.jpg"


def should_save(last_saved, now, cooldown=COOLDOWN):
    """Avoid saving a photo every frame: wait `cooldown` seconds between saves."""
    return last_saved is None or (now - last_saved) >= cooldown


def open_camera(index=0):
    log(f"Opening camera (index {index})...")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        log("The camera would not open.")
        if getattr(sys, "frozen", False):
            raise RuntimeError(
                "macOS is blocking the camera. Go to System Settings > Privacy & Security > "
                "Camera and turn FaceSentry ON, then open FaceSentry again."
            )
        raise RuntimeError(
            "Could not open the camera. On a Mac: System Settings > Privacy & Security > "
            "Camera, allow Terminal (or your Python app), then start again."
        )
    log("Camera opened.")
    return cap
