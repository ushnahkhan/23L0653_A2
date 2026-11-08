"""Test setup: temporary SQLite DB, small upload limit, and a fake model (no real weights needed)."""
import io
import os
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"   # set BEFORE config is imported
os.environ["MAX_UPLOAD_MB"] = "1"


class FakeModel:
    meta = {"version": "test", "model_name": "VisionGuard", "run_id": "ci", "source_run": "fake"}
    img_size, threshold, model_path = 160, 0.53, "fake.keras"

    def predict(self, raw):
        return {"label": "AI-generated", "confidence": 0.91, "p_ai": 0.91,
                "threshold": 0.53, "model_version": "test", "latency_ms": 1.0}


@pytest.fixture(scope="session")
def client():
    import inference
    inference.get_model = lambda: FakeModel()       # replace the real model before app is imported
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


@pytest.fixture
def png():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 30, 200)).save(buf, "PNG")   # wide image -> padding expected
    return buf.getvalue()