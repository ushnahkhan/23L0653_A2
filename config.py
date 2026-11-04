"""
config.py - every VisionGuard setting comes from environment variables (.env for local use).
Real environment variables (shell, Docker, cloud) override values in .env.
"""
import os

from dotenv import load_dotenv

load_dotenv()                                         # does NOT override variables already set


def _opt_float(name):
    v = os.getenv(name)
    return float(v) if v not in (None, "") else None


def _opt_int(name):
    v = os.getenv(name)
    return int(v) if v not in (None, "") else None


# ---- model ----
MODEL_PATH = os.getenv("VISIONGUARD_MODEL_PATH")              # explicit .keras file (optional)
MODEL_DIR = os.getenv("VISIONGUARD_MODEL_DIR", "model_store")  # registry download cache
THRESHOLD = _opt_float("VISIONGUARD_THRESHOLD")               # overrides registry threshold
IMG_SIZE = _opt_int("VISIONGUARD_IMG_SIZE")                   # overrides registry img_size

# ---- database ----
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")

# ---- web app ----
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "8"))
LOW_CONFIDENCE = float(os.getenv("LOW_CONFIDENCE", "0.65"))
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"                  # OFF unless explicitly enabled