import io

import numpy as np
import pytest
from PIL import Image

import inference


# ---------- preprocessing (real code) ----------
def test_preprocess_shape_and_range(png):
    x = inference.preprocess(png, 160).numpy()
    assert x.shape == (1, 160, 160, 3)
    assert x.min() >= 0 and x.max() <= 255


def test_preprocess_keeps_aspect_ratio(png):
    x = inference.preprocess(png, 160).numpy()[0]
    assert x[0].sum() == 0          # top row = black padding (64x48 is wider than tall)
    assert x[80].sum() > 0          # middle row = image content


def test_preprocess_rejects_non_image():
    with pytest.raises(ValueError):
        inference.preprocess(b"definitely not an image", 160)


# ---------- web app (fake model) ----------
def post(client, data, name):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    j = r.get_json()
    assert r.status_code == 200 and j["status"] == "ok" and j["database"] == "ok"
    assert j["debug"] is False


def test_upload_page(client):
    assert client.get("/").status_code == 200


def test_wrong_file_type(client):
    assert post(client, b"hello", "notes.txt").status_code == 415


def test_file_too_large(client):
    assert post(client, b"\xff\xd8\xff" + bytes(2 * 1024 * 1024), "big.jpg").status_code == 413


def test_corrupt_image(client):
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8))
    buf = io.BytesIO()
    noise.save(buf, "PNG")
    data = buf.getvalue()
    assert post(client, data[: len(data) // 2], "broken.png").status_code == 400


def test_valid_upload_is_logged_and_name_sanitized(client, png):
    r = post(client, png, "../../evil name.png")
    assert r.status_code == 302                                   # Post/Redirect/Get
    assert b"AI-generated" in client.get(r.headers["Location"]).data
    assert b"evil_name.png" in client.get("/history").data        # secure_filename worked