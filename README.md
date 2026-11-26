# facesentry

A webcam watchman for your own laptop. It learns your face, then watches. If someone else's face shows up, it saves a photo with a timestamp.

**Status:** work in progress. Enroll, watch and snapshot work. Alerts and AI descriptions are next.

## Ethics, please read

- Only enroll people who said yes.
- Only watch your own machine, in your own space.
- Your face photos, the trained model and intruder snapshots stay on your computer. They are in `.gitignore` so they are never uploaded to GitHub. Keep it that way.
- This is a learning project, not a lock. Do not rely on it to protect anything important.

## Download (Mac app, no Terminal)

For Macs with Apple silicon (M1 and newer).

1. Go to the [latest release](https://github.com/itsjoshareh1/facesentry/releases/latest) and download **FaceSentry.dmg**.
2. Open the DMG and drag **FaceSentry** onto **Applications**.
3. First launch only: the app is not signed by Apple, so macOS blocks it at first.
   - Right-click **FaceSentry** in Applications and choose **Open**, then **Open** again.
   - If there is no Open button (newer macOS): try to open it once, then go to **System Settings > Privacy & Security**, scroll down and click **Open Anyway**.
4. macOS asks to let FaceSentry use the camera. Click **OK**. (Missed it? System Settings > Privacy & Security > Camera > turn on FaceSentry.)
5. First run: it takes 20 photos of your face (turn your head a little, change expression), then starts watching.
6. Press **q** in the camera window to quit. Unknown faces are saved in **Pictures > FaceSentry Intruders**.

To retrain your face, delete the folder `~/Library/Application Support/FaceSentry` and open the app again.

Your face photos and the trained model stay in that folder on your Mac. Nothing is uploaded.

### Run from source (no build)

Download the ZIP of this repo, unzip, and double-click **Start FaceSentry.command** (right-click > Open the first time).

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

## How the app is built

`.github/workflows/build-mac-app.yml` runs on a macOS Apple-silicon runner: installs the requirements, runs the tests, builds `FaceSentry.app` with PyInstaller (`FaceSentry.spec`), self-tests it, packs `FaceSentry.dmg` and attaches it to a GitHub Release. Run it from the Actions tab with a tag like `v1.0.0`.

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
