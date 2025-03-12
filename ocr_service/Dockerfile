# Use official Python image
FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["sh", "-c", "\
    if [ -f /app/.env ]; then \
        echo 'Running locally: Loading .env file'; \
        export $(grep -v '^#' /app/.env | xargs); \
    else \
        echo 'Running on AWS: Using IAM roles'; \
    fi && \
    uvicorn main:app --host 0.0.0.0 --port 8001"]
