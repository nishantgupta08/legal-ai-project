from diagrams import Diagram, Cluster
from diagrams.aws.database import Dynamodb
from diagrams.onprem.compute import Server
from diagrams.onprem.client import User
from diagrams.onprem.database import Mongodb
from diagrams.onprem.inmemory import Redis
from diagrams.custom import Custom
from diagrams.programming.language import Python

# Define Diagram
with Diagram("Legal AI Microservices Architecture", show=True, direction="TB"):

    user = User("User")

    with Cluster("API Gateway"):
        api_gateway = Python("FastAPI Server")
    
    with Cluster("Storage"):
        dynamodb = Dynamodb("DynamoDB (Metadata)")
        chromadb = Mongodb("ChromaDB (Vector Storage)")
    
    with Cluster("Microservices"):
        ocr_service = Server("OCR Service")
        clause_retrieval = Redis("Clause Retrieval")
        
        # Use a custom LLM image (Download an LLM icon and save it in the same directory)
        clause_validation = Custom("LLM Clause Validator", "llm_image.jpeg")

    # User uploads file to API Gateway
    user >> api_gateway

    # API Gateway checks DynamoDB
    api_gateway >> dynamodb

    # If not found, OCR Service is called
    dynamodb >> ocr_service >> dynamodb

    # OCR Text stored in ChromaDB
    ocr_service >> chromadb

    # Clause Retrieval fetches relevant clauses
    chromadb >> clause_retrieval

    # Clause Validation using LLM
    clause_retrieval >> clause_validation

    # Send response back to user
    clause_validation >> api_gateway >> user
