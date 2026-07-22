# Build stage: resolve and install dependencies into a self-contained venv.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# Dependencies first so this layer caches across code changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# Runtime stage: no uv, no build caches, non-root user.
FROM python:3.14-slim-bookworm

RUN groupadd -r app && useradd -r -g app app

COPY --from=builder --chown=app:app /app /app

USER app
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD ["python", "-c", \
    "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]

CMD ["streamlit", "run", "streamlit_app.py", \
    "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
