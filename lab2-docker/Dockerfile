# Lab 2 (Docker) - modified Dockerfile
# Original (Docker_Labs/Lab1): FROM python:3.10 (full image), COPY src/ .
# before installing, unpinned requirements, ran as root, model saved
# inside the container and lost when it exited.
#
# Core Docker changes in this file (see README.md for the full list):
#   1. Slim, pinned base image: python:3.10-slim instead of python:3.10
#   2. Cache-friendly layer order: requirements are copied and installed
#      BEFORE the source, so code changes do not re-install dependencies
#   3. --no-cache-dir + pinned requirements: smaller image, reproducible
#   4. The container runs as a non-root user (appuser), not root
#   5. /app/output is a declared VOLUME: model + metrics persist on the
#      host with `docker run -v ...:/app/output`
#   6. ENTRYPOINT (not just CMD): the training script is the container's
#      fixed job, and run-time arguments pass straight to it, e.g.
#      `docker run lab2-docker --n-estimators 200` - no rebuild needed

FROM python:3.10-slim

LABEL description="MLOps Lab 2 - Docker: containerised RandomForest training (Breast Cancer Wisconsin)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OUTPUT_DIR=/app/output

WORKDIR /app

# Dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source second
COPY src/ ./src/

# Non-root user that owns the app and the output volume directory
RUN useradd --create-home appuser \
    && mkdir -p /app/output \
    && chown -R appuser:appuser /app
USER appuser

VOLUME ["/app/output"]

ENTRYPOINT ["python", "src/main.py"]
CMD ["--output-dir", "/app/output"]
