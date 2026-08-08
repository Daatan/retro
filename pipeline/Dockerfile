FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements/project files and install pinned deps from the lockfile
# (docker-compose mounts .:/app at runtime, so deps must live in system
# site-packages, not a /app/.venv that the bind mount would shadow).
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv export --frozen --no-dev --no-emit-project -o /tmp/requirements.txt && \
    uv pip install --system --no-cache -r /tmp/requirements.txt && \
    pip install --no-cache-dir --no-deps -e .

# Create data directories
RUN mkdir -p data/vault data/atlas data/events data/sources data/pages

# Command to keep the container running
CMD ["tail", "-f", "/dev/null"]
