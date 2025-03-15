from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from pydantic import BaseModel
import requests
import logging
import os
import json
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Microservice URLs
OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://ocr_service:8001/extract-text")
CHROMA_STORE_URL = os.getenv("CHROMA_STORE_URL", "http://vector_database_service:8002/store-text")
CHROMA_RETRIEVAL_URL = os.getenv("CHROMA_RETRIEVAL_URL", "http://vector_database_service:8002/retrieve-text")
CLAUSE_VALIDATOR_URL = os.getenv("CLAUSE_VALIDATOR_URL", "http://clause_validator:8003/validate")

# Define a Pydantic model for clauses
class ClauseRequest(BaseModel):
    clauses: List[str]

@app.post("/extract-clauses")
async def extract_clauses(file: UploadFile = File(...),   clauses_list: List[str] = Query(...)
):
    """
    1️⃣ Extract text using OCR (with page numbers)
    2️⃣ Store extracted text in ChromaDB
    3️⃣ Retrieve stored text from ChromaDB
    4️⃣ Validate extracted text against given clauses
    """
    try:
        # clauses_list = clause_data.clauses
        if not clauses_list:
            raise HTTPException(status_code=400, detail="Clauses list cannot be empty")

        file_bytes = await file.read()
        pdf_name = file.filename
        
        # Step 1: OCR Service Call
        logger.info(f"📄 Processing OCR for file: {pdf_name}")
        ocr_response = requests.post(OCR_SERVICE_URL, files={"file": (pdf_name, file_bytes, file.content_type)})

        if ocr_response.status_code != 200:
            raise HTTPException(status_code=ocr_response.status_code, detail="OCR Service Error")

        page_para_text = ocr_response.json().get("ocr_text", [])
        if not page_para_text:
            raise HTTPException(status_code=400, detail="OCR service returned no text.")

        # Step 2: Store extracted text in ChromaDB (Batch Insert)
        documents = [
            {"text": text, "filename": pdf_name, "page_number": page, "para_number": para}
            for page, para, text in page_para_text
        ]
        requests.post(CHROMA_STORE_URL, json={"documents": documents})

        # Step 3: Retrieve text from ChromaDB
        logger.info(f"🔍 Retrieving stored text from ChromaDB for file: {pdf_name}")
        clause_paragraph_map = {}
        print(clauses_list)

        for clause in clauses_list:
            chroma_retrieval_payload = {"query": clause, "top_k": 10, "metadata_filter": {"filename": pdf_name}}
            retrieval_response = requests.post(CHROMA_RETRIEVAL_URL, json=chroma_retrieval_payload)

            if retrieval_response.status_code != 200:
                raise HTTPException(status_code=retrieval_response.status_code, detail="ChromaDB Retrieval Failed")

            retrieved_paragraphs = retrieval_response.json().get("documents", [])
            if retrieved_paragraphs:
                clause_paragraph_map[clause] = retrieved_paragraphs

        if not clause_paragraph_map:
            return {"message": "No relevant clauses found in the document."}
        
        return clause_paragraph_map

        # Step 4: Validate extracted clauses
        # logger.info(f"✅ Validating extracted clauses against: {clauses_list}")
        # clause_response = requests.post(CLAUSE_VALIDATOR_URL, json=clause_paragraph_map)

        # if clause_response.status_code != 200:
        #     raise HTTPException(status_code=clause_response.status_code, detail="Clause validation failed")

        # validated_clauses = clause_response.json()
        # return {"message": "📜 Document processed successfully.", "validated_clauses": validated_clauses}

    except Exception as e:
        logger.error(f"❌ Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/")
def health_check():
    """Simple health check endpoint"""
    return {"message": "OCR Processing Service is running"}
