# Use Python 3.11 to match your model's environment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgomp1 build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy Flask app and model/vectorizer
COPY flask_app/ /app/
RUN mkdir -p /app/data/processed
COPY data/processed/bow_vectorizer.pkl /app/data/processed/bow_vectorizer.pkl

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download NLTK corpora
RUN python -m nltk.downloader stopwords wordnet

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
