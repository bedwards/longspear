FROM python:3.12-slim-bookworm

WORKDIR /app

# System deps for psycopg (PostgreSQL client) and yt-dlp
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq-dev gcc curl ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ src/
COPY config/ config/
COPY pyproject.toml .

# Create data directories
RUN mkdir -p data/transcripts/heather_cox_richardson \
             data/transcripts/nate_b_jones \
             data/processed/heather_cox_richardson \
             data/processed/nate_b_jones \
             data/vectordb/lancedb

EXPOSE 8000

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
