import numpy as np
import pytest

import facesentry as fs
from datetime import datetime


def test_snapshot_name_has_timestamp():
    assert fs.snapshot_name(datetime(2026, 11, 26, 14, 5, 9)) == "intruder_20261126_140509.jpg"


def test_snapshot_name_test_prefix():
    assert fs.snapshot_name(datetime(2026, 11, 27, 8, 0, 0), prefix="test") == "test_20261127_080000.jpg"


def test_faces_match_same_face():
    assert fs.faces_match((100, 100, 80, 80), (110, 105, 80, 80))
    assert not fs.faces_match((100, 100, 80, 80), (300, 100, 80, 80))


def test_track_unknowns_needs_confirm_frames():
    from watch import track_unknowns, CONFIRM_FRAMES
    box = (100, 100, 80, 80)
    cands, confirmed = track_unknowns([], [(box, 80.0)])
    assert confirmed == []  # one sighting is not enough: objects never trigger photos
    for _ in range(CONFIRM_FRAMES - 1):
        cands, confirmed = track_unknowns(cands, [(box, 80.0)])
    assert len(confirmed) == 1
    # a face that vanishes right away is forgotten and never confirmed
    cands, confirmed = track_unknowns([], [(box, 80.0)])
    for _ in range(3):
        cands, confirmed = track_unknowns(cands, [])
    assert confirmed == [] and cands == []





def test_should_save_respects_cooldown():
    assert fs.should_save(None, 100.0)
    assert not fs.should_save(100.0, 105.0, cooldown=10)
    assert fs.should_save(100.0, 111.0, cooldown=10)












def test_parse_yolo_filters_weak_small_and_odd_boxes():
    import numpy as np
    rows = np.zeros((1, 6, 21), dtype="float32")
    rows[0, 0, :6] = [100, 100, 220, 240, 0.90, 0]   # good face (scaled 1:1 below)
    rows[0, 1, :6] = [300, 100, 420, 240, 0.30, 0]   # too weak
    rows[0, 2, :6] = [10, 10, 40, 40, 0.95, 0]       # too small
    rows[0, 3, :6] = [400, 100, 630, 160, 0.95, 0]   # far too wide for a face
    boxes = fs.parse_yolo(rows, 640, 640)
    assert boxes == [(100, 100, 120, 140)]


def test_parse_yolo_scales_to_frame():
    import numpy as np
    rows = np.zeros((1, 2, 21), dtype="float32")
    rows[0, 0, :6] = [200, 200, 400, 400, 0.9, 0]
    assert fs.parse_yolo(rows, 1280, 960) == [(400, 300, 400, 300)]


def test_detector_falls_back_to_haar_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(fs, "_model_dir", lambda: str(tmp_path))
    monkeypatch.setattr(fs, "_DETECTOR", None)
    assert fs.load_detector().name == "haar"
    monkeypatch.setattr(fs, "_DETECTOR", None)


def test_yolo_model_finds_no_face_in_noise():
    import os
    import numpy as np
    path = os.path.join(fs._model_dir(), fs.YOLO_FILE)
    if not os.path.exists(path):
        pytest.skip("YOLO model not downloaded")
    frame = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype("uint8")
    assert fs.YoloDetector(path).detect(frame) == []


def unit(seed):
    v = np.random.default_rng(seed).normal(size=128).astype("float32")
    return v / np.linalg.norm(v)


def test_identify_known_and_unknown():
    me, other = unit(1), unit(2)
    prints = np.array([me, unit(3)])
    assert fs.identify(prints, ["me", "me"], me) == ("me", pytest.approx(1.0, abs=1e-4))
    assert fs.identify(prints, ["me", "me"], other)[0] is None


def test_identify_uses_best_angle():
    front, side = unit(1), unit(2)
    probe = 0.5 * side + 0.5 * unit(9)
    probe /= np.linalg.norm(probe)
    prints = np.array([front, side])  # side view enrolled, so the side-on probe is recognized
    assert fs.identify(prints, ["me", "me"], probe, threshold=0.5)[0] == "me"
    assert fs.identify(prints[:1], ["me"], probe, threshold=0.5)[0] is None


def test_enrollment_round_trip(tmp_path):
    path = str(tmp_path / "p.npz")
    fs.save_enrollment([unit(1), unit(2)], ["me", "me"], path)
    assert fs.has_enrollment(path)
    prints, labels = fs.load_enrollment(path)
    assert prints.shape == (2, 128) and labels == ["me", "me"]


def test_known_face_resets_unknown_tracking():
    from watch import track_unknowns, CONFIRM_FRAMES
    box = (100, 100, 80, 80)
    cands = []
    for _ in range(CONFIRM_FRAMES - 1):
        cands, confirmed = track_unknowns(cands, [(box, 0.1)])
    cands, confirmed = track_unknowns(cands, [], [box])  # recognized as a known person
    assert cands == [] and confirmed == []


def test_unknown_must_persist_for_seconds():
    from watch import track_unknowns, CONFIRM_FRAMES, CONFIRM_SECONDS
    box = (100, 100, 80, 80)
    cands = []
    for i in range(CONFIRM_FRAMES + 3):  # many frames but only a fraction of a second
        cands, confirmed = track_unknowns(cands, [(box, 0.1)], now=100.0 + i * 0.05)
    assert confirmed == []
    cands, confirmed = track_unknowns(cands, [(box, 0.1)], now=100.0 + CONFIRM_SECONDS + 0.1)
    assert len(confirmed) == 1


def test_parse_yolo_full_returns_landmarks():
    rows = np.zeros((1, 2, 21), dtype="float32")
    rows[0, 0, :6] = [100, 100, 220, 240, 0.9, 0]
    rows[0, 0, 6:21] = list(range(15))
    (box, lm), = fs.parse_yolo_full(rows, 640, 640)
    assert box == (100, 100, 120, 140)
    assert lm.shape == (5, 2) and lm[0][0] == 0 and lm[1][0] == 3


def test_embedder_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(fs, "_model_dir", lambda: str(tmp_path))
    monkeypatch.setattr(fs, "_EMBEDDER", None)
    with pytest.raises(RuntimeError):
        fs.load_embedder()


def test_sface_model_embeds_a_face():
    import os
    path = os.path.join(fs._model_dir(), fs.SFACE_FILE)
    if not os.path.exists(path):
        pytest.skip("SFace model not downloaded")
    img = (np.random.default_rng(0).random((160, 160, 3)) * 255).astype("uint8")
    e = fs.FaceEmbedder(path).embed(img, (0, 0, 160, 160))
    assert e.shape == (128,) and abs(float(e @ e) - 1.0) < 1e-3
