from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from chroma_manager import ChromaManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize ChromaDB Manager
chroma_manager = ChromaManager()

# Request models
class DocumentRequest(BaseModel):
    text: str
    metadata: dict

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    metadata_filter: dict = None

@app.post("/store-text")
def store_text(request: DocumentRequest):
    """
    Stores extracted OCR text in ChromaDB with metadata.
    """
    try:
        chroma_manager.add_documents([request.text], [(request.metadata["filename"], 1)])
        return {"message": "Text stored successfully"}
    except Exception as e:
        logger.error(f"Error storing text: {str(e)}")
        raise HTTPException(status_code=500, detail="Error storing text")

@app.post("/retrieve-text")
def retrieve_text(request: QueryRequest):
    """
    Retrieves relevant documents based on query.
    """
    try:
        results = chroma_manager.retrieve_documents(request.query, request.top_k, request.metadata_filter)
        return {"documents": results}
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving documents")

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {"message": "FAISS/Chroma Service is running"}
