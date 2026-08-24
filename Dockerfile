# MFIE — production image.
#
# Two stages so the compiler toolchain needed to build numpy/scipy wheels does
# not ship in the runtime image. The result is roughly half the size and has a
# much smaller attack surface than a single-stage build.
#
#   docker build -t mfie .
#   docker run -p 8000:8000 --env-file .env -v mfie-data:/app/data mfie
#
# The frontend is plain files served by the same process — there is no node
# stage here because there is nothing to compile.

# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is cached and only invalidated when the
# requirements change, not on every source edit.
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt \
    && /opt/venv/bin/pip install "fastapi>=0.110" "uvicorn[standard]>=0.27"

# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MFIE_HOST=0.0.0.0 \
    MFIE_PORT=8000

# curl is here for the healthcheck below and nothing else.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 mfie

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY --chown=mfie:mfie mfie ./mfie
COPY --chown=mfie:mfie config ./config
COPY --chown=mfie:mfie pyproject.toml README.md ./

# The default SQLite database lives here. Mount a volume over it to keep
# ingested history and realised trades across container restarts — without
# that, calibration starts from the prior again on every deploy.
RUN mkdir -p /app/data && chown mfie:mfie /app/data
VOLUME ["/app/data"]

USER mfie
EXPOSE 8000

# Hits the liveness endpoint, which deliberately does not run the engine — a
# healthcheck that triggered a full macro rebuild would fail its own timeout.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${MFIE_PORT}/api/health" || exit 1

CMD ["sh", "-c", "exec uvicorn mfie.api:app --host ${MFIE_HOST} --port ${MFIE_PORT} --proxy-headers --forwarded-allow-ips='*'"]
