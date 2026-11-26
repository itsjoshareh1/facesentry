# facesentry

A webcam watchman for your own laptop. It learns your face, then watches. If someone else's face shows up, it saves a photo with a timestamp.

**Status:** work in progress. Enroll, watch and snapshot work. Alerts and AI descriptions are next.

## Ethics, please read

- Only enroll people who said yes.
- Only watch your own machine, in your own space.
- Your face photos, the trained model and intruder snapshots stay on your computer. They are in `.gitignore` so they are never uploaded to GitHub. Keep it that way.
- This is a learning project, not a lock. Do not rely on it to protect anything important.

## Run it (Mac, no Terminal needed)

1. On this page click **Code > Download ZIP**.
2. Double-click the ZIP to unzip it.
3. Open the folder and double-click **Start FaceSentry.command**.
   - First time only: if macOS says it cannot verify the file, right-click it, choose **Open**, then **Open** again.
   - The first launch takes a minute or two while it installs OpenCV.
4. macOS asks to let Terminal use the camera. Click **OK**. (If you missed it: System Settings > Privacy & Security > Camera > turn on Terminal, then double-click the file again.)
5. First run: look at the camera. It takes 20 photos of your face (turn your head a little, change expression), trains, then starts watching.
6. Press **q** in the camera window to quit. Unknown faces are saved in the `intruders` folder.

To retrain your face later (new glasses, new lighting), delete `model.yml` and launch again.

### For developers

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python enroll.py            # capture and train
python watch.py             # live watch (enrolls first if there is no model)
python watch.py --threshold 60   # stricter
pytest
```

## How it works

1. **Detect:** OpenCV's Haar cascade finds face boxes in each frame.
2. **Prepare:** crop, resize to 200x200, even out lighting.
3. **Recognize:** an LBPH recognizer compares the face to your photos and returns a distance. Lower means more similar. Under 70 is "known", over is "unknown".
4. **React:** unknown faces get a red box and a snapshot, at most one every 10 seconds.

Pure OpenCV, no `dlib`, no cloud, installs with plain `pip`.

## Roadmap

- [x] Enroll with the webcam
- [x] Live watch with known/unknown labels
- [x] Timestamped intruder snapshots
- [ ] Multiple known faces from the window (the code already supports several names)
- [ ] Alert sound or macOS notification
- [ ] Claude describes the intruder photo (needs your own API key, never commit it)
- [ ] Sensitivity tuning: find the best threshold for your lighting

## Limits

LBPH is a classic method. Strong light changes, glasses, or a photo of your face held up to the camera can fool it. Treat the distance number on screen as the thing to learn from.

## License

MIT
