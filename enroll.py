"""Enroll step: capture photos of your face with the webcam, then train the recognizer.

Only enroll people who said yes. Run:  python enroll.py   (or python enroll.py --name sam)
"""
import argparse
import os
import time

import cv2

import facesentry as fs

TARGET = 20  # photos to capture
GAP = 0.5  # seconds between photos so they are not all identical


def enroll(name="me", target=TARGET):
    folder = os.path.join(fs.KNOWN_DIR, name)
    os.makedirs(folder, exist_ok=True)
    detector = fs.load_detector()
    fs.log(f"Enroll started for '{name}': taking {target} photos.")
    cap = fs.open_camera()
    saved, last = 0, 0.0
    print(f"Enrolling '{name}'. Look at the camera, turn your head a little, change expression.")
    print("Press q to stop early.")
    while saved < target:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = fs.detect_faces(gray, detector)
        if len(boxes) == 1 and time.time() - last >= GAP:
            face = fs.prepare_face(gray, boxes[0])
            cv2.imwrite(os.path.join(folder, f"{saved:02d}.jpg"), face)
            saved += 1
            last = time.time()
            fs.log(f"Enroll photo {saved}/{target} captured.")
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        msg = f"Captured {saved}/{target}" if len(boxes) == 1 else "Need exactly one face in view"
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("FaceSentry enroll", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if saved == 0:
        fs.log("Enroll failed: no photos captured.")
        raise RuntimeError(
            "No photos captured. Check that the camera shows your face, the room is bright, "
            "and only one face is in view."
        )
    count, names = fs.train()
    fs.log(f"Trained on {count} photos for: {', '.join(names)}")
    print(f"Trained on {count} photos for: {', '.join(names)}")
    # TODO (you): enroll several people (it already supports more than one name), and let the
    #             user choose a name in the window instead of the command line.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="me")
    enroll(parser.parse_args().name)
