"""facesentry core: face detection, enrolling and recognition with OpenCV only.

Uses OpenCV's Haar cascade to FIND faces and the LBPH recognizer to tell whose face it is.
Everything runs on your own machine. Nothing is uploaded.
"""
import json
import os
from datetime import datetime

import cv2
import numpy as np

FACE_SIZE = (200, 200)
KNOWN_DIR = "known_faces"
MODEL_PATH = "model.yml"
NAMES_PATH = "model_names.json"
INTRUDER_DIR = "intruders"
THRESHOLD = 70.0  # LBPH distance: LOWER means more similar. Above this = unknown face.
COOLDOWN = 10  # seconds between intruder snapshots


def load_detector():
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError("Could not load the OpenCV face detector.")
    return detector


def detect_faces(gray, detector):
    """Return a list of (x, y, w, h) boxes for faces in a grayscale image."""
    return detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))


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
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the camera. On a Mac: System Settings > Privacy & Security > "
            "Camera, allow Terminal, then start again."
        )
    return cap
