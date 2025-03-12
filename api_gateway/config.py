import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Service URLs
OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://ocr_service:8001/extract-text")
FAISS_SERVICE_URL = os.getenv("FAISS_SERVICE_URL", "http://faiss_service:8002/store-text")

# AWS Configurations (if used)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
