"""Watch mode: live camera window. Known faces get a green label, unknown faces get red
and a snapshot is saved to intruders/ with a timestamp.

Only watch your own machine. Run:  python watch.py   (add --enroll to retrain your face)
"""
import argparse
import os
import time

import cv2

import facesentry as fs
from enroll import enroll


def watch(threshold=fs.THRESHOLD):
    detector = fs.load_detector()
    recognizer, names = fs.load_model()
    cap = fs.open_camera()
    os.makedirs(fs.INTRUDER_DIR, exist_ok=True)
    last_saved = None
    print("Watching. Press q in the window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        unknown_seen = False
        for box in fs.detect_faces(gray, detector):
            x, y, w, h = box
            name, distance = fs.identify(recognizer, names, fs.prepare_face(gray, box), threshold)
            known = name is not None
            color = (0, 200, 0) if known else (0, 0, 255)
            label = f"{name} ({distance:.0f})" if known else f"UNKNOWN ({distance:.0f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            unknown_seen = unknown_seen or not known
        now = time.time()
        if unknown_seen and fs.should_save(last_saved, now):
            path = os.path.join(fs.INTRUDER_DIR, fs.snapshot_name())
            cv2.imwrite(path, frame)
            last_saved = now
            print(f"Unknown face! Saved {path}")
            # TODO (you): play a sound or show a macOS notification here
            # TODO (you): send the photo to Claude and ask it to describe the person
            #             (needs your own API key, never commit the key)
        cv2.imshow("facesentry (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll", action="store_true", help="capture your face again")
    parser.add_argument("--threshold", type=float, default=fs.THRESHOLD,
                        help="lower = stricter. Default %(default)s")
    args = parser.parse_args()
    if args.enroll or not os.path.exists(fs.MODEL_PATH):
        enroll()
    watch(args.threshold)
