# Use official lightweight Python image
FROM python:3.9-slim

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
