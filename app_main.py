"""Entry point for the packaged Mac app (FaceSentry.app).

First launch: enroll your face, then watch. Later launches: watch right away.
"""
import os
import subprocess
import sys

import facesentry as fs


def alert(message):
    """Show a plain dialog, since the app has no terminal."""
    text = message.replace("\\", " ").replace('"', "'")
    subprocess.run(["osascript", "-e", f'display dialog "{text}" with title "FaceSentry" buttons {{"OK"}}'])


def selftest():
    """Camera-free check used by the build: detector loads, training and matching work."""
    import tempfile

    import cv2
    import numpy as np

    fs.load_detector()
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
    try:
        from enroll import enroll
        from watch import watch

        if not os.path.exists(fs.MODEL_PATH):
            enroll()
        watch()
    except Exception as err:  # show it instead of dying silently
        alert(f"FaceSentry could not start: {err}")


if __name__ == "__main__":
    main()
