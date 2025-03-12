from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import logging
import os

from legal_clause_validator import LegalClauseValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load Model Name from Env (Defaults to FLAN-T5)
MODEL_NAME = os.getenv("LLM_MODEL", "google/flan-t5-large")

# Initialize Legal Clause Validator
clause_validator = LegalClauseValidator(llm_model=MODEL_NAME)

# Request Model
class ClauseValidationRequest(BaseModel):
    clauses: Dict[str, List[str]]  # Mapping: Clause Type → List of Retrieved Paragraphs

@app.post("/validate_clauses")
def validate_clauses(request: ClauseValidationRequest):
    """
    API endpoint to validate clauses in legal documents.
    """
    try:
        logger.info(f"🔍 Validating clauses for: {list(request.clauses.keys())}")
        validation_results = clause_validator.validate_clauses(request.clauses)

        return {"validated_clauses": validation_results}

    except Exception as e:
        logger.error(f"❌ Error during clause validation: {e}")
        raise HTTPException(status_code=500, detail="Internal validation error")


@app.get("/")
def health_check():
    """Simple health check endpoint"""
    return {"message": "Legal Clause Validator Service is running"}
