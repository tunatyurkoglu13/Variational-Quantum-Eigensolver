FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies first (separate layer, cache-friendly on source changes).
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# Now add the actual source and install the project itself.
COPY src/ ./src/
COPY conf/ ./conf/
COPY scripts/ ./scripts/
RUN uv sync --locked --no-dev

ENTRYPOINT ["python"]
