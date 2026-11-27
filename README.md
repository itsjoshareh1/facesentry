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
6. While it watches, the top bar shows what it sees. Keys: **q** quit, **t** save a test photo (proves it works), **o** open the photos folder, **-** / **+** tune how strict it is.
7. Unknown faces are saved in **Pictures > FaceSentry Intruders**, with a beep and a red PHOTO SAVED banner when it happens.

Everything the app does is written to a log you can check: **~/Library/Application Support/FaceSentry/facesentry.log** (in Finder press Cmd-Shift-G and paste that path).

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

1. **Detect:** A small pretrained YOLOv8n-face model (run through OpenCV, bundled in the app) finds faces in each frame and ignores weak, tiny or oddly shaped matches. If the model cannot load, FaceSentry falls back to the strict OpenCV Haar detector. The first time you run v1.2 it asks for new photos of you, because recognition is retrained with the new detector. A face must then be tracked for several frames in a row before it counts (you see an orange "checking" box first), so a chair or a one-frame glitch never triggers a photo.
2. **Prepare:** crop, resize to 200x200, even out lighting.
3. **Recognize:** an LBPH recognizer compares the face to your photos and returns a distance. Lower means more similar. Under 55 is "known", over is "unknown". You see the number live and tune it with **-** and **+** while it runs.
4. **React:** confirmed unknown faces get a red box and a snapshot, at most one every 10 seconds, and the next photo needs a fresh confirmation too.

Pure OpenCV, no `dlib`, no cloud, installs with plain `pip`.

## How the app is built

`.github/workflows/build-mac-app.yml` runs on a macOS Apple-silicon runner: installs the requirements, runs the tests, builds `FaceSentry.app` with PyInstaller (`FaceSentry.spec`), self-tests it, packs `FaceSentry.dmg` and attaches it to a GitHub Release. Run it from the Actions tab with a tag like `v1.0.0`.

## Roadmap

- [x] Enroll with the webcam
- [x] Live watch with known/unknown labels
- [x] Timestamped intruder snapshots
- [x] Beep and on-screen banner when a photo is saved
- [x] Live sensitivity tuning and a test-photo key
- [ ] Multiple known faces from the window (the code already supports several names)
- [ ] macOS notification when someone is caught
- [ ] Claude describes the intruder photo (needs your own API key, never commit it)

## If it is not catching people

1. **Press t** in the watch window. A test photo with a beep should land in **Pictures > FaceSentry Intruders**. If it does, saving works fine.
2. Have someone else sit down and watch the number next to their face. If they show a green box with a number under the strictness value, press **-** a few times to make it stricter. Your own face should still show green.
3. Nothing happens at all, or an error about the camera: **System Settings > Privacy & Security > Camera** and make sure FaceSentry is turned on, then reopen the app.
4. Still stuck: open the log at **~/Library/Application Support/FaceSentry/facesentry.log**. Every start, enrollment, camera opening and saved photo is listed there with a time.

## Limits

LBPH is a classic method. Strong light changes, glasses, or a photo of your face held up to the camera can fool it. Treat the distance number on screen as the thing to learn from.

## License

MIT


## Model credit
Face detection uses the pretrained YOLOv8n-face weights from https://github.com/akanametov/yolo-face (GPL-3.0 licensed weights, downloaded at build time and not stored in this repo). If you redistribute the built app, check that license.
