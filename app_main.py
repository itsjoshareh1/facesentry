"""Entry point for the packaged Mac app (FaceSentry.app).

First launch: enroll your face, then watch. Later launches: watch right away.
Anything that goes wrong is written to the log file and shown in a dialog.
"""
import os
import subprocess
import sys

import facesentry as fs

CAMERA_SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"


def _clean(message):
    return message.replace("\\", " ").replace('"', "'")


def alert(message):
    """Show a plain dialog, since the app has no terminal."""
    subprocess.run(["osascript", "-e",
                    f'display dialog "{_clean(message)}" with title "FaceSentry" buttons {{"OK"}}'])


def ask_open_camera_settings(message):
    """Explain the camera problem and offer to jump straight to the right System Settings pane."""
    result = subprocess.run(
        ["osascript", "-e",
         f'display dialog "{_clean(message)}" with title "FaceSentry" '
         'buttons {"Open Camera Settings", "Quit"} default button "Open Camera Settings"'],
        capture_output=True, text=True)
    if "Open Camera Settings" in result.stdout:
        subprocess.run(["open", CAMERA_SETTINGS_URL])


def selftest():
    """Camera-free check used by the build: detector loads, training and matching work."""
    import cv2
    import numpy as np

    detector = fs.load_detector()
    assert detector.name == "yolo", "YOLO model was not bundled into the app"
    blank = np.zeros((480, 640, 3), dtype="uint8")
    assert detector.detect(blank) == [], "detector found a face in a blank frame"
    embedder = fs.load_embedder()
    rng = np.random.default_rng(1)
    crop = (rng.random((200, 200, 3)) * 255).astype("uint8")
    p1 = embedder.embed(crop, (0, 0, 200, 200))
    p2 = np.roll(p1, 1) * -1  # a clearly different fingerprint
    assert p1.shape == (128,) and abs(float(p1 @ p1) - 1.0) < 1e-3, "recognizer self-test failed"
    prints = np.array([p1])
    assert fs.identify(prints, ["me"], p1)[0] == "me", "recognizer should know the enrolled print"
    assert fs.identify(prints, ["me"], p2)[0] is None, "recognizer accepted a different image"
    print("SELFTEST OK")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    fs.log("FaceSentry app started.")
    try:
        from enroll import enroll
        from watch import watch

        if not fs.has_enrollment():
            fs.log("No enrolled face found; starting enrollment.")
            alert("FaceSentry will learn your face from several angles. After you click OK, "
                  "follow the on-screen prompts: look straight, then slowly turn your head left, "
                  "right, up and down. It takes about 40 seconds.")
            enroll()
        watch()
    except Exception as err:  # show it instead of dying silently
        fs.log(f"FaceSentry stopped with an error: {err}")
        message = str(err)
        if "camera" in message.lower():
            ask_open_camera_settings(f"FaceSentry could not start: {message}")
        else:
            alert(f"FaceSentry could not start: {message}")


if __name__ == "__main__":
    main()
