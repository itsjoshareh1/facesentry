"""Watch mode: live camera window. Known faces get a green label, unknown faces get red
and a snapshot is saved with a timestamp, with a beep and a banner so you notice.

A face only counts after it has been seen for several frames in a row, so objects the
detector mistakes for a face for a split second never trigger a photo.

Keys in the window:
  q     quit
  t     save a TEST photo right now (proves the camera and saving both work)
  o     open the folder where photos are saved
  - / + stricter / looser recognition (watch the distance numbers while you tune)

Only watch your own machine. Run:  python watch.py   (add --enroll to retrain your face)
"""
import argparse
import os
import subprocess
import sys
import time

import cv2

import facesentry as fs
from enroll import enroll

BANNER_SECONDS = 3.0  # how long the "PHOTO SAVED" banner stays on screen
UNKNOWN_LOG_GAP = 5  # seconds between "unknown face" log lines, so the log stays readable
CONFIRM_FRAMES = 4  # a face must be tracked this many frames before it can trigger a photo
GRACE_MISSES = 1  # frames a tracked face may vanish (blink, angle) before it is forgotten

CHECKING_COLOR = (0, 165, 255)  # orange: possible face, not confirmed yet
UNKNOWN_COLOR = (0, 0, 255)  # red: confirmed unknown face
KNOWN_COLOR = (0, 200, 0)  # green: recognized face


def beep():
    """Make a noise when a photo is saved, so a caught face is impossible to miss."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["osascript", "-e", "beep"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


def open_photos_folder():
    os.makedirs(fs.INTRUDER_DIR, exist_ok=True)
    fs.log(f"Photos folder: {fs.INTRUDER_DIR}")
    if sys.platform == "darwin":
        subprocess.run(["open", fs.INTRUDER_DIR], check=False)
    else:
        print(f"Photos are saved in: {fs.INTRUDER_DIR}")


def save_snapshot(frame, prefix="intruder"):
    os.makedirs(fs.INTRUDER_DIR, exist_ok=True)
    path = os.path.join(fs.INTRUDER_DIR, fs.snapshot_name(prefix=prefix))
    cv2.imwrite(path, frame)
    fs.log(f"Saved {prefix} photo: {path}")
    beep()
    return path


def draw_box(frame, box, color, label):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, max(48, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_hud(frame, threshold, face_count):
    """Top bar with the live state, so you can see the app is working."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    status = f"WATCHING | faces: {face_count} | strictness: {threshold:.0f} (-/+) | t=test photo | q=quit"
    cv2.putText(frame, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def draw_banner(frame, text):
    """Big red banner at the bottom after a photo is saved."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 200), -1)
    cv2.putText(frame, text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def track_unknowns(candidates, unknown_boxes):
    """Update per-face tracking with this frame's unknown faces.

    Each candidate is a possible unknown face. Its count grows while the same face keeps
    showing up; only a candidate seen CONFIRM_FRAMES times is treated as real. This stops
    objects the detector mistakes for a face in a single frame from triggering photos.
    Returns (candidates, confirmed) where confirmed are the candidates that count now.
    """
    seen = set()
    for box, distance in unknown_boxes:
        match = None
        for i, cand in enumerate(candidates):
            if i not in seen and fs.faces_match(cand["box"], box):
                match = i
                break
        if match is None:
            candidates.append({"box": box, "distance": distance, "count": 1, "misses": 0})
            seen.add(len(candidates) - 1)
        else:
            cand = candidates[match]
            cand.update(box=box, distance=distance, count=cand["count"] + 1, misses=0)
            seen.add(match)
    kept = []
    for i, cand in enumerate(candidates):
        if i not in seen:
            cand["misses"] += 1
        if cand["misses"] <= GRACE_MISSES:
            kept.append(cand)
    confirmed = [c for c in kept if c["count"] >= CONFIRM_FRAMES]
    return kept, confirmed


def watch(threshold=fs.THRESHOLD):
    detector = fs.load_detector()
    recognizer, names = fs.load_model()
    fs.log(f"Watch started. Detector: {detector.name}. Model knows: {', '.join(names)}. Strictness {threshold:.0f}. "
           f"Photos save to: {fs.INTRUDER_DIR}")
    cap = fs.open_camera()
    os.makedirs(fs.INTRUDER_DIR, exist_ok=True)
    candidates = []
    last_saved = None
    last_unknown_log = 0.0
    banner_until = 0.0
    banner_text = ""
    print("Watching. Keys: q quit, t test photo, o open photos folder, -/+ tune strictness.")
    while True:
        ok, frame = cap.read()
        if not ok:
            fs.log("Camera stopped returning frames; ending watch.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = fs.detect_faces(frame, detector)
        unknown_boxes = []
        for box in boxes:
            name, distance = fs.identify(recognizer, names, fs.prepare_face(gray, box), threshold)
            if name is not None:
                draw_box(frame, box, KNOWN_COLOR, f"{name} ({distance:.0f})")
            else:
                unknown_boxes.append((box, distance))
        candidates, confirmed = track_unknowns(candidates, unknown_boxes)
        for cand in candidates:
            if cand["count"] >= CONFIRM_FRAMES:
                draw_box(frame, cand["box"], UNKNOWN_COLOR, f"UNKNOWN ({cand['distance']:.0f})")
            else:
                draw_box(frame, cand["box"], CHECKING_COLOR, f"checking ({cand['distance']:.0f})")
        now = time.time()
        if confirmed:
            if now - last_unknown_log >= UNKNOWN_LOG_GAP:
                worst = max(c["distance"] for c in confirmed)
                fs.log(f"UNKNOWN face confirmed (distance {worst:.0f}, strictness {threshold:.0f}).")
                last_unknown_log = now
            if fs.should_save(last_saved, now):
                path = save_snapshot(frame)
                last_saved = now
                candidates = []  # the next photo needs a fresh confirmation too
                banner_until = now + BANNER_SECONDS
                banner_text = f"PHOTO SAVED: {os.path.basename(path)} - press o to see it"
        draw_hud(frame, threshold, len(boxes))
        if now < banner_until:
            draw_banner(frame, banner_text)
        cv2.imshow("FaceSentry watching", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t"):
            path = save_snapshot(frame, prefix="test")
            banner_until = time.time() + BANNER_SECONDS
            banner_text = f"TEST PHOTO SAVED: {os.path.basename(path)} - press o to see it"
        elif key == ord("o"):
            open_photos_folder()
        elif key in (ord("-"), ord("_")):
            threshold = max(10.0, threshold - 5)
            fs.log(f"Strictness changed to {threshold:.0f} (lower = stricter).")
        elif key in (ord("+"), ord("=")):
            threshold = min(150.0, threshold + 5)
            fs.log(f"Strictness changed to {threshold:.0f} (higher = looser).")
    cap.release()
    cv2.destroyAllWindows()
    fs.log("Watch stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll", action="store_true", help="capture your face again")
    parser.add_argument("--threshold", type=float, default=fs.THRESHOLD,
                        help="lower = stricter. Default %(default)s")
    args = parser.parse_args()
    if (args.enroll or not os.path.exists(fs.MODEL_PATH)
            or not fs.model_matches_detector(fs.load_detector())):
        enroll()
    watch(args.threshold)
