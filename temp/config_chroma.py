import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
