import numpy as np
import pytest

import facesentry as fs
from datetime import datetime


def fake_face(seed):
    rng = np.random.default_rng(seed)
    return (rng.random(fs.FACE_SIZE) * 255).astype("uint8")


def test_snapshot_name_has_timestamp():
    assert fs.snapshot_name(datetime(2026, 11, 26, 14, 5, 9)) == "intruder_20261126_140509.jpg"


def test_should_save_respects_cooldown():
    assert fs.should_save(None, 100.0)
    assert not fs.should_save(100.0, 105.0, cooldown=10)
    assert fs.should_save(100.0, 111.0, cooldown=10)


def test_prepare_face_size():
    gray = np.zeros((300, 300), dtype="uint8")
    assert fs.prepare_face(gray, (50, 50, 120, 120)).shape == (200, 200)


def test_train_and_identify(tmp_path):
    import cv2
    folder = tmp_path / "known" / "me"
    folder.mkdir(parents=True)
    base = fake_face(1)
    for i in range(5):
        cv2.imwrite(str(folder / f"{i}.jpg"), base)
    count, names = fs.train(str(tmp_path / "known"), str(tmp_path / "m.yml"), str(tmp_path / "n.json"))
    assert count == 5 and names == ["me"]
    recognizer, names = fs.load_model(str(tmp_path / "m.yml"), str(tmp_path / "n.json"))
    name, _ = fs.identify(recognizer, names, base)
    assert name == "me"
    other, _ = fs.identify(recognizer, names, fake_face(99), threshold=1.0)
    assert other is None


def test_train_without_photos_fails(tmp_path):
    (tmp_path / "known").mkdir()
    with pytest.raises(RuntimeError):
        fs.train(str(tmp_path / "known"), str(tmp_path / "m.yml"), str(tmp_path / "n.json"))

# TODO: add a test for two enrolled people once you support multiple known faces.
