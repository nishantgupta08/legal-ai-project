import chromadb
from chromadb.utils import embedding_functions
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ChromaManager:
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

    def add_documents(self, docs, metadata):
        """
        Adds documents to ChromaDB with metadata (filename, page number).
        
        Args:
            docs (List[str]): List of document texts.
            metadata (List[Tuple[str, int]]): List of (filename, page_number) tuples.
        """
        ids = [str(i) for i in range(len(docs))]
        metadatas = [{"filename": meta[0], "page_number": meta[1]} for meta in metadata]
        self.collection.add(ids=ids, documents=docs, metadatas=metadatas)
        
        logger.info(f"✅ Added {len(docs)} documents to ChromaDB.")

    def retrieve_documents(self, query, top_k=5, metadata_filter=None):
        """
        Retrieves relevant documents based on a query and optional metadata filter.

        Args:
            query (str): Query text.
            top_k (int): Number of documents to retrieve.
            metadata_filter (Dict[str, Any], optional): Filter for metadata (e.g., {"filename": "contract1.pdf"}).

        Returns:
            List[Dict[str, Any]]: Retrieved documents along with file name & page number metadata.
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
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]  # Extract filename & page number
                retrieved_docs.append({
                    "text": doc,
                    "filename": metadata.get("filename", "Unknown"),
                    "page_number": metadata.get("page_number", "Unknown")
                })

        return retrieved_docs
