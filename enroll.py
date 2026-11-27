"""Enroll step: capture your face from several angles with the webcam.

You are guided through poses (straight, left, right, up, down). For each pose the app stores
face "fingerprints" (SFace), including mirrored ones, so it still knows you when you turn your head.
Only enroll people who said yes. Run:  python enroll.py   (or python enroll.py --name sam)
"""
import argparse
import time

import cv2

import facesentry as fs

# (instruction on screen, photos to take). Photos are taken only after a short settle time.
POSES = [
    ("Look straight at the camera", 6),
    ("Slowly turn your head LEFT", 6),
    ("Slowly turn your head RIGHT", 6),
    ("Tilt your head UP a little", 4),
    ("Tilt your head DOWN a little", 4),
    ("Smile, then look straight again", 4),
]
SETTLE = 2.5  # seconds to get into each pose before photos start
GAP = 0.45  # seconds between photos
TARGET = sum(n for _, n in POSES)


def fingerprints_for(embedder, detector, frame, face_box_landmarks):
    """Fingerprint of the face plus one of its mirror image (helps with left/right turns)."""
    box, lm = face_box_landmarks
    prints = [embedder.embed(frame, box, lm)]
    flipped = cv2.flip(frame, 1)
    again = fs.detect_faces_full(flipped, detector)
    if len(again) == 1:
        prints.append(embedder.embed(flipped, again[0][0], again[0][1]))
    return prints


def enroll(name="me", poses=POSES):
    detector = fs.load_detector()
    embedder = fs.load_embedder()
    total = sum(n for _, n in poses)
    fs.log(f"Enroll started for '{name}': {total} photos over {len(poses)} poses.")
    cap = fs.open_camera()
    prints, labels = [], []
    taken = 0
    aborted = False
    for pose_index, (instruction, count) in enumerate(poses):
        pose_start = time.time()
        got, last = 0, 0.0
        while got < count:
            ok, frame = cap.read()
            if not ok:
                aborted = True
                break
            found = fs.detect_faces_full(frame, detector)
            now = time.time()
            ready = now - pose_start >= SETTLE
            if len(found) == 1 and ready and now - last >= GAP:
                for p in fingerprints_for(embedder, detector, frame, found[0]):
                    prints.append(p)
                    labels.append(name)
                got += 1
                taken += 1
                last = now
                fs.log(f"Enroll photo {taken}/{total} ({instruction}).")
            for (box, _) in found:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            if not ready:
                msg = f"Get ready: {instruction}  ({max(0, SETTLE - (now - pose_start)):.0f}s)"
            elif len(found) != 1:
                msg = "Need exactly one face in view"
            else:
                msg = f"{instruction}  [{taken}/{total}]"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (30, 30, 30), -1)
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("FaceSentry enroll", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                aborted = True
                break
        if aborted:
            break
    cap.release()
    cv2.destroyAllWindows()
    if not prints:
        fs.log("Enroll failed: no photos captured.")
        raise RuntimeError(
            "No photos captured. Check that the camera shows your face, the room is bright, "
            "and only one face is in view."
        )
    fs.save_enrollment(prints, labels)
    fs.log(f"Enrolled '{name}' with {len(prints)} face fingerprints ({taken} photos).")
    print(f"Enrolled '{name}' with {len(prints)} face fingerprints.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="me")
    enroll(parser.parse_args().name)
