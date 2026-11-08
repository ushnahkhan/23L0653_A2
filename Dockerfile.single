# ---------- 1. slim base: small Debian + Python, no compilers or docs ----------
FROM python:3.12-slim

# ---------- 2. runtime settings (no secrets here) ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    VISIONGUARD_MODEL_DIR=/app/model_store \
    DATABASE_URL=sqlite:////app/data/visionguard.db \
    MAX_UPLOAD_MB=8 \
    FLASK_DEBUG=0 \
    PORT=8000

WORKDIR /app

# ---------- 3. non-root user (created early, rarely changes -> cached) ----------
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data && chown appuser:appuser /app/data

# ---------- 4. dependencies BEFORE source code (layer cache) ----------
COPY requirements-app.txt .
RUN --mount=type=bind,source=wheels,target=/wheels \
    pip install --no-index --find-links=/wheels -r requirements-app.txt

# ---------- 5. model (changes rarely) ----------
COPY --chown=appuser:appuser model_store/ /app/model_store/

# ---------- 6. source code (changes most often -> last) ----------
COPY --chown=appuser:appuser config.py db.py inference.py app.py ./
COPY --chown=appuser:appuser templates/ templates/

# ---------- 7. drop root ----------
USER appuser

EXPOSE 8000

# ---------- 8. health check: Docker marks the container healthy/unhealthy ----------
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4).status == 200 else 1)"

# ---------- 9. production server: 1 worker (one model copy), 4 threads ----------
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 --access-logfile - app:app"]