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
    import tempfile

    import cv2
    import numpy as np

    detector = fs.load_detector()
    assert detector.name == "yolo", "YOLO model was not bundled into the app"
    blank = np.zeros((480, 640, 3), dtype="uint8")
    assert detector.detect(blank) == [], "detector found a face in a blank frame"
    rng = np.random.default_rng(1)
    face = (rng.random(fs.FACE_SIZE) * 255).astype("uint8")
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "k", "me"))
    for i in range(3):
        cv2.imwrite(os.path.join(tmp, "k", "me", f"{i}.jpg"), face)
    fs.train(os.path.join(tmp, "k"), os.path.join(tmp, "m.yml"), os.path.join(tmp, "n.json"))
    recognizer, names = fs.load_model(os.path.join(tmp, "m.yml"), os.path.join(tmp, "n.json"))
    name, _ = fs.identify(recognizer, names, face)
    assert name == "me", "recognizer self-test failed"
    print("SELFTEST OK")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    fs.log("FaceSentry app started.")
    try:
        from enroll import enroll
        from watch import watch

        if not os.path.exists(fs.MODEL_PATH):
            fs.log("No trained model found; starting enrollment.")
            enroll()
        elif not fs.model_matches_detector(fs.load_detector()):
            fs.log("Detector changed since enrollment; enrolling again.")
            alert("FaceSentry got a better face detector. It needs 20 new photos of your face, "
                  "so look at the camera after you click OK.")
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
