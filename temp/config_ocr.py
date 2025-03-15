import os
from dotenv import load_dotenv

load_dotenv()

TESSERACT_PATH = os.getenv("TESSERACT_PATH", "/usr/bin/tesseract")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
