import tempfile
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import logging
from ocr import OCRProcessor  # Import OCRProcessor class

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
ocr_processor = OCRProcessor()  # Initialize OCR Processor

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """Process PDF and return extracted text with page numbers (No Storage)."""
    try:
        # Use a temporary directory (Cross-platform compatibility)
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, file.filename)

            # Save the uploaded file to the temporary path
            with open(pdf_path, "wb") as f:
                f.write(await file.read())

            # Extract text from the PDF
            ocr_text = ocr_processor.process_pdf(pdf_path)

        if not ocr_text:
            raise HTTPException(status_code=400, detail="No text extracted from PDF.")        
        
        return {"ocr_text": ocr_text}
    
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"message": "OCR Service is running"}
