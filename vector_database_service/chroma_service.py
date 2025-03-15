import chromadb
from chromadb.utils import embedding_functions
import logging
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ChromaService:
    """
    Manages ChromaDB for vector storage and retrieval.
    """
    def __init__(self, collection_name="legal_docs"):
        """
        Initializes ChromaDB with a given collection name.
        """
        self.client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    

    def add_documents(self, documents):
        """
        Adds texts to ChromaDB with metadata (filename, page number, para number).
        
        Args:
            documents (List[dict{text, filename, page_number, para_number}]): List of document texts.
        """
        
        texts, metadatas = [], []

        for doc in documents:
            texts.append(doc.text)
            metadatas.append({"filename": doc.filename, "page_number": doc.page_number, "para_number": doc.para_number})

        
        ids = [
            hashlib.md5(f"{meta['filename']}_{meta['page_number']}_{meta['para_number']}".encode()).hexdigest()
            for meta in metadatas
        ]

        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
        
        logger.info(f"✅ Added {len(texts)} paras to ChromaDB.")


    def retrieve_documents(self, query, top_k=5, metadata_filter=None):
        """
        Retrieves relevant documents based on a query and optional metadata filter.

        Args:
            query (str): Query text.
            top_k (int): Number of documents to retrieve.
            metadata_filter (Dict[str, Any], optional): Filter for metadata (e.g., {"filename": "contract1.pdf"}).

        Returns:
            List[str]: Retrieved document texts.
        """
        query_params = {
            "query_texts": [query],
            "n_results": top_k
        }

        if metadata_filter:
            query_params["where"] = metadata_filter  # Apply metadata filtering

        results = self.collection.query(**query_params)
        retrieved_docs = []

        if results and "documents" in results and results["documents"][0]:
            retrieved_docs = results["documents"][0]  # Extract only text content

        return retrieved_docs
