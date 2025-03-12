from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict
import requests
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Microservice URLs
OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://ocr_service:8001/extract-text")
CHROMA_MANAGER_URL = os.getenv("CHROMA_MANAGER_URL", "http://chroma_manager:8002/store-text")
CHROMA_RETRIEVAL_URL = os.getenv("CHROMA_RETRIEVAL_URL", "http://chroma_manager:8002/retrieve-text")
CLAUSE_VALIDATOR_URL = os.getenv("CLAUSE_VALIDATOR_URL", "http://clause_validator:8003/validate_clauses")

# Request Model
class ClauseRequest(BaseModel):
    queries: List[str]

# Helper function to parse JSON from a string
async def parse_clause_request(request: str = None):
    if request:
        return ClauseRequest(**json.loads(request))
    return ClauseRequest(queries=[])

@app.post("/process_document")
async def process_document(
    file: UploadFile = File(...),
    request: ClauseRequest = Depends(parse_clause_request)
):
    """
    1️⃣ Extract text using OCR
    2️⃣ Store extracted text in ChromaDB
    3️⃣ Retrieve stored text from ChromaDB
    4️⃣ Validate retrieved text using Legal Clause Validator
    """
    try:
        file_bytes = await file.read()
        pdf_name = file.filename

        # Step 1: OCR Service Call
        logger.info(f"📄 Processing OCR for file: {pdf_name}")
        files = {"file": (pdf_name, file_bytes, file.content_type)}
        ocr_response = requests.post(OCR_SERVICE_URL, files=files)

        if ocr_response.status_code != 200:
            raise HTTPException(status_code=ocr_response.status_code, detail="OCR Service Error")

        ocr_data = ocr_response.json()
        ocr_text = ocr_data.get("ocr_text", "")

        # Step 2: Store extracted text in ChromaDB
        logger.info(f"🗄️ Storing extracted text in ChromaDB for file: {pdf_name}")
        chroma_payload = {"text": ocr_text, "metadata": {"filename": pdf_name}}
        chroma_response = requests.post(CHROMA_MANAGER_URL, json=chroma_payload)

        if chroma_response.status_code != 200:
            raise HTTPException(status_code=chroma_response.status_code, detail="ChromaDB Service Error")

        # Step 3: Retrieve text from ChromaDB
        logger.info(f"🔍 Retrieving stored text from ChromaDB for file: {pdf_name}")
        chroma_retrieval_payload = {"query": "", "top_k": 10, "metadata_filter": {"file_name": pdf_name}}
        retrieval_response = requests.post(CHROMA_RETRIEVAL_URL, json=chroma_retrieval_payload)

        if retrieval_response.status_code != 200:
            raise HTTPException(status_code=retrieval_response.status_code, detail="ChromaDB Retrieval Failed")

        retrieved_paragraphs = retrieval_response.json().get("documents", [])
        if not retrieved_paragraphs:
            return {"message": "No relevant clauses found in the document."}

        # Step 4: Organize retrieved text for validation
        queries = request.queries if request else []
        query_paragraph_map: Dict[str, List[str]] = {query: [] for query in queries}

        for para in retrieved_paragraphs:
            for query in queries:
                query_paragraph_map[query].append(para["text"])

        # Step 5: Validate clauses using Legal Clause Validator
        logger.info(f"✅ Validating clauses for queries: {queries}")
        clause_response = requests.post(CLAUSE_VALIDATOR_URL, json=query_paragraph_map)

        if clause_response.status_code != 200:
            raise HTTPException(status_code=clause_response.status_code, detail="Clause validation failed")

        validated_clauses = clause_response.json()

        return {
            "message": "📜 Document processed successfully.",
            "ocr_text": ocr_text,
            "validated_clauses": validated_clauses
        }

    except Exception as e:
        logger.error(f"❌ Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/")
def health_check():
    """Simple health check endpoint"""
    return {"message": "API Gateway is running"}
