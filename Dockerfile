# ================= stage 1: builder (thrown away after the build) =================
FROM python:3.12-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-app.txt .
RUN --mount=type=bind,source=wheels,target=/wheels \
    pip install --no-index --find-links=/wheels --no-compile -r requirements-app.txt \
    && find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && rm -rf /opt/venv/lib/python3.12/site-packages/tensorflow/include

# ================= stage 2: runtime (what is shipped) =================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PATH="/opt/venv/bin:$PATH" \
    VISIONGUARD_MODEL_DIR=/app/model_store \
    DATABASE_URL=sqlite:////app/data/visionguard.db \
    MAX_UPLOAD_MB=8 \
    FLASK_DEBUG=0 \
    PORT=8000

WORKDIR /app
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data && chown appuser:appuser /app/data

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser model_store/ /app/model_store/
COPY --chown=appuser:appuser config.py db.py inference.py app.py ./
COPY --chown=appuser:appuser templates/ templates/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4).status == 200 else 1)"

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 --access-logfile - app:app"]