"""
app.py - VisionGuard Flask web app (Task 4.2)
Run: python app.py   ->  http://127.0.0.1:8000
"""
#for rebuild purposes
import io
import os
import time
import hashlib
import math

from flask import Flask, Request, jsonify, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

import db
from inference import get_model

import config
MAX_MB = config.MAX_UPLOAD_MB
ALLOWED_EXT = {"jpg", "jpeg", "png", "gif", "bmp"}
MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"BM")  # real file signatures
Image.MAX_IMAGE_PIXELS = 50_000_000            # block "decompression bomb" images
LOW_CONFIDENCE = config.LOW_CONFIDENCE

class InMemoryRequest(Request):
    """Keep every upload in RAM - Werkzeug would otherwise spill files >500 KB to a temp file."""
    def _get_file_stream(self, total_content_length, content_type, filename=None,
                            content_length=None):
        return io.BytesIO()


app = Flask(__name__)
app.request_class = InMemoryRequest
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024   # bigger bodies -> 413 before reading
db.init_db()
model = get_model()                                        # loaded ONCE at start-up


class UploadError(Exception):
    def __init__(self, title, message, status=400, filename=None, size=None):
        super().__init__(title)
        self.title, self.message, self.status = title, message, status
        self.filename, self.size = filename, size


def validate(file):
    """Return (safe_name, bytes) or raise UploadError. Nothing is written to disk."""
    if file is None or file.filename == "":
        raise UploadError("No file selected", "Please choose an image before pressing Analyse.")

    safe_name = (secure_filename(file.filename) or "unnamed")[:100]
    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    if ext not in ALLOWED_EXT:
        raise UploadError("Unsupported file type",
                            f"'.{ext or '?'}' files are not accepted. Please upload a JPG, PNG, GIF or BMP image.",
                            415, safe_name)

    data = file.read()
    if not data:
        raise UploadError("Empty file", "The uploaded file contains no data.", 400, safe_name, 0)
    if not data.startswith(MAGIC):
        raise UploadError("Not a real image",
                            "The file name says it is an image, but its content is not a JPG, PNG, GIF or BMP.",
                            400, safe_name, len(data))
    try:
        with Image.open(io.BytesIO(data)) as im:   # structure check
            im.verify()
        with Image.open(io.BytesIO(data)) as im:   # full decode - catches truncated files
            im.load()
            info = {"image_format": im.format, "image_width": im.width, "image_height": im.height}
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise UploadError("Image could not be read",
                            "The file looks like an image but is damaged, incomplete or too large in pixels. "
                            "Try re-saving it or upload a different image.", 400, safe_name, len(data))
    return safe_name, data, info


# ----------------------------- pages -----------------------------
@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_MB, allowed=", ".join(sorted(ALLOWED_EXT)))


@app.post("/predict")
def predict():
    t0 = time.perf_counter()
    try:
        name, data, info = validate(request.files.get("image"))
        r = model.predict(data)
    except UploadError as e:
        db.log_request(filename=e.filename, file_size=e.size, status="rejected", error=e.title,
                       latency_ms=(time.perf_counter() - t0) * 1000)
        return render_template("error.html", title=e.title, message=e.message), e.status
    except ValueError as e:                               # TensorFlow could not decode
        db.log_request(filename=name, file_size=len(data), status="rejected", error=str(e)[:200])
        return render_template("error.html", title="Image could not be read", message=str(e)), 400

    row_id = db.log_request(
        filename=name, file_size=len(data), image_sha256=hashlib.sha256(data).hexdigest(), **info,
        status="ok", label=r["label"], p_ai=r["p_ai"], threshold=r["threshold"],
        confidence=r["confidence"],
        model_name=model.meta.get("model_name", "VisionGuard"),
        model_version=str(r["model_version"]), model_run_id=model.meta.get("run_id"),
        model_source=model.meta.get("source_run"), model_input=model.img_size,
        latency_ms=(time.perf_counter() - t0) * 1000)
    del data                                              # image bytes dropped after the request
    return redirect(url_for("result", row_id=row_id))     # Post/Redirect/Get

@app.get("/result/<int:row_id>")
def result(row_id):
    row = db.get(row_id)
    if row is None or row["status"] != "ok":
        return render_template("error.html", title="Result not found",
                                message="This result does not exist."), 404
    return render_template("result.html", r=row, low=row["confidence"] < LOW_CONFIDENCE)

@app.get("/history")
def history():
    status = request.args.get("status") if request.args.get("status") in ("ok", "rejected") else None
    label = request.args.get("label") if request.args.get("label") in ("AI-generated", "Real") else None
    try:
        page = max(int(request.args.get("page", 1)), 1)
    except ValueError:
        page = 1
    per_page = 25
    rows, total = db.query(status=status, label=label, page=page, per_page=per_page)
    return render_template("history.html", rows=rows, total=total, page=page,
                            pages=max(math.ceil(total / per_page), 1),
                            stats=db.stats(), status=status, label=label)


@app.get("/health")
def health():
    try:
        db.ping()
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {type(e).__name__}"
    ok = db_status == "ok" and model is not None
    return jsonify(
        status="ok" if ok else "degraded",
        model_loaded=model is not None,
        model_version=model.meta.get("version"),
        model_path=model.model_path,
        model_input=model.img_size,
        threshold=model.threshold,
        database=db_status,
        database_backend=db.engine.url.get_backend_name(),   # e.g. sqlite / postgresql - no password
        debug=app.debug,
        server=request.environ.get("SERVER_SOFTWARE", "unknown"),
    ), (200 if ok else 503)


# ----------------------------- friendly errors -----------------------------
@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    db.log_request(status="rejected", error=f"file larger than {MAX_MB} MB")
    return render_template("error.html", title="File too large",
                            message=f"The maximum upload size is {MAX_MB} MB. "
                                    "Please resize or compress the image and try again."), 413


@app.errorhandler(HTTPException)
def http_error(e):
    return render_template("error.html", title=f"{e.code} - {e.name}",
                            message=e.description), e.code


@app.errorhandler(Exception)
def server_error(e):
    app.logger.exception(e)
    return render_template("error.html", title="Something went wrong",
                            message="An unexpected error occurred. Please try again later."), 500

if __name__ == "__main__":                       # development only - production uses waitress/gunicorn
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG)