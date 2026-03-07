# Dockerfile for Local Transcription API
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies + curl for uv installer
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies from lockfile (exact same versions as local)
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 3030

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3030"]