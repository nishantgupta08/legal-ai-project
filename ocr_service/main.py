from fastapi import FastAPI, UploadFile, File, HTTPException
import boto3
import hashlib
import pytesseract
from pdf2image import convert_from_bytes
import os
import json
from dotenv import load_dotenv  

# Load environment variables
load_dotenv()

app = FastAPI()

# AWS Configuration from Environment Variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

S3_BUCKET = os.getenv("S3_BUCKET", "ocr-service-legal")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "OCRResults")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")

# Initialize AWS Clients
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    dynamodb = boto3.resource(
        "dynamodb",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    ).Table(DYNAMODB_TABLE)
    sqs_client = boto3.client(
        "sqs",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
else:
    # Use IAM Role (for AWS Fargate, ECS, Lambda)
    s3_client = boto3.client("s3", region_name=AWS_REGION)
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION).Table(DYNAMODB_TABLE)
    sqs_client = boto3.client("sqs", region_name=AWS_REGION)

def generate_pdf_hash(file_bytes):
    """Generate SHA-256 hash of the PDF."""
    hasher = hashlib.sha256()
    hasher.update(file_bytes)
    return hasher.hexdigest()

def check_existing_ocr(hash_value):
    """Check if OCR results already exist in DynamoDB."""
    response = dynamodb.get_item(Key={"pdf_hash ": hash_value})
    return response.get("Item")

def store_ocr_result(hash_value, text):
    """Store OCR result in DynamoDB & S3."""
    dynamodb.put_item(Item={"pdf_hash ": hash_value, "ocr_text": text})
    s3_client.put_object(Bucket=S3_BUCKET, Key=f"ocr_results/{hash_value}.txt", Body=text)

def publish_to_queue(hash_value):
    """Publish message to SQS for FAISS indexing."""
    message = json.dumps({"pdf_hash ": hash_value})
    sqs_client.send_message(QueueUrl=SQS_QUEUE_URL, MessageBody=message)

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """API to process PDF and run OCR."""
    pdf_bytes = await file.read()
    file_hash = generate_pdf_hash(pdf_bytes)

    existing_result = check_existing_ocr(file_hash)
    if existing_result:
        return {"message": "OCR result exists", "ocr_text": existing_result["ocr_text"]}

    images = convert_from_bytes(pdf_bytes)
    extracted_text = "\n".join([pytesseract.image_to_string(img) for img in images])

    store_ocr_result(file_hash, extracted_text)
    publish_to_queue(file_hash)

    return {"message": "OCR completed", "ocr_text": extracted_text}
