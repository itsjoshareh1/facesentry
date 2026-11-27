"""facesentry core: face detection, enrolling and recognition with OpenCV only.

Uses OpenCV's Haar cascade to FIND faces and the LBPH recognizer to tell whose face it is.
Everything runs on your own machine. Nothing is uploaded.
"""
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

FACE_SIZE = (200, 200)

# Run from source: files live in the current folder.
# Run as the packaged Mac app: private data goes to Application Support, snapshots to Pictures.
if getattr(sys, "frozen", False):
    _DATA = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "FaceSentry")
    INTRUDER_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "FaceSentry Intruders")
else:
    _DATA = "."
    INTRUDER_DIR = "intruders"
KNOWN_DIR = os.path.join(_DATA, "known_faces")
MODEL_PATH = os.path.join(_DATA, "model.yml")
NAMES_PATH = os.path.join(_DATA, "model_names.json")
LOG_PATH = os.path.join(_DATA, "facesentry.log")
# LBPH distance: LOWER means more similar. Above this = unknown face.
# 55 errs on the side of calling a stranger UNKNOWN (you can tune it live with - and +).
THRESHOLD = 55.0
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


def load_detector():
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError("Could not load the OpenCV face detector.")
    return detector


def detect_faces(gray, detector):
    """Return a list of (x, y, w, h) boxes for faces in a grayscale image.

    minNeighbors=7 is deliberately strict: the default of 5 lets random objects
    (chairs, hands, patterns) pass as faces. Watch mode also requires a face to
    stick around for several frames before it counts, see CONFIRM_FRAMES there.
    """
    return detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(90, 90))


def faces_match(box_a, box_b, tolerance=0.5):
    """True when two face boxes are roughly the same face in consecutive frames."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    return (abs((ax + aw / 2) - (bx + bw / 2)) <= aw * tolerance and
            abs((ay + ah / 2) - (by + bh / 2)) <= ah * tolerance)


def prepare_face(gray, box):
    """Crop a face, resize to a fixed size and even out the lighting."""
    x, y, w, h = box
    face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
    return cv2.equalizeHist(face)


def load_dataset(known_dir=KNOWN_DIR):
    """Read known_faces/<name>/*.jpg into (faces, labels, names)."""
    faces, labels, names = [], [], []
    for name in sorted(os.listdir(known_dir)):
        folder = os.path.join(known_dir, name)
        if not os.path.isdir(folder):
            continue
        label = len(names)
        count = 0
        for file in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(cv2.resize(img, FACE_SIZE))
            labels.append(label)
            count += 1
        if count:
            names.append(name)
    return faces, labels, names


def train(known_dir=KNOWN_DIR, model_path=MODEL_PATH, names_path=NAMES_PATH):
    """Train the recognizer on everything in known_dir and save it."""
    faces, labels, names = load_dataset(known_dir)
    if not faces:
        raise RuntimeError("No face photos found. Run the enroll step first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    recognizer.write(model_path)
    with open(names_path, "w") as f:
        json.dump(names, f)
    return len(faces), names


def load_model(model_path=MODEL_PATH, names_path=NAMES_PATH):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    with open(names_path) as f:
        names = json.load(f)
    return recognizer, names


def identify(recognizer, names, face, threshold=THRESHOLD):
    """Return (name or None, distance). None means an unknown face."""
    label, distance = recognizer.predict(face)
    if distance < threshold and 0 <= label < len(names):
        return names[label], distance
    return None, distance


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
