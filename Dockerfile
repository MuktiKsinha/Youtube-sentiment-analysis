# ==========================
# Stage 1: Builder
# ==========================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages in a separate folder to copy later
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# ==========================
# Stage 2: Final Runtime Image
# ==========================
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies needed for LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy Flask application
COPY flask_app/ /app/

# Copy trained vectorizer
COPY data/processed/bow_vectorizer.pkl /app/data/processed/bow_vectorizer.pkl

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Download NLTK data only
RUN python -m nltk.downloader stopwords wordnet

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
