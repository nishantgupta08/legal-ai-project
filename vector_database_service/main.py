from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from chroma_service import ChromaService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize ChromaDB Manager
chroma_service = ChromaService()

# Request models
class DocumentItem(BaseModel):
    text: str
    filename: str
    page_number: int
    para_number: int

class MultiDocumentRequest(BaseModel):
    documents: List[DocumentItem]  # Simplified structure

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    metadata_filter: Dict[str, Any] = None

@app.post("/store-text")
def store_text(request: MultiDocumentRequest):
    """
    Stores multiple extracted OCR texts in ChromaDB with metadata.
    """
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        chroma_service.add_documents(request.documents)

        return {"message": f"✅ Stored {len(request.documents)} documents successfully"}
    except Exception as e:
        logger.error(f"Error storing text: {str(e)}")
        raise HTTPException(status_code=500, detail="Error storing text")

@app.post("/retrieve-text")
def retrieve_text(request: QueryRequest):
    """
    Retrieves relevant documents based on query.
    """
    try:
        results = chroma_service.retrieve_documents(request.query, request.top_k, request.metadata_filter)
        return {"documents": results}
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving documents")

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {"message": "FAISS/Chroma Service is running"}
