# LiteLLM proxy image with the session router hook installed.
FROM ghcr.io/berriai/litellm:main-stable

COPY packages/router-common /build/router-common
COPY packages/session-router /build/session-router
# The wolfi-based image ships a venv at /app/.venv without pip.
RUN /app/.venv/bin/python -m ensurepip && \
    /app/.venv/bin/python -m pip install --no-cache-dir \
    /build/router-common /build/session-router

# Config, callbacks module, and policies are mounted (or COPYed in prod) at /app.
WORKDIR /app
