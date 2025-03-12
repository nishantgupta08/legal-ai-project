import logging
from typing import Any, Dict, List, Tuple
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from langgraph.graph import StateGraph, START, END

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LegalClauseValidator:
    """
    A class for validating legal clauses using an LLM and LangGraph.
    """

    def __init__(self, llm_model: str) -> None:
        """
        Initializes the LegalClauseValidator with:
        - The pre-trained LLM model.
        """
        self.tokenizer, self.llm_model = self._load_llm_model(llm_model)
        self.graph = self._initialize_graph()

    def validate_clauses(self, clause_queries: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Validates legal clauses using LangGraph workflow.

        Args:
            clause_queries (Dict[str, List[str]]): 
                - Key = Clause Type (e.g., "Indemnification Clause")
                - Value = List of retrieved paragraphs from ChromaDB

        Returns:
            Dict[str, Any]: Dictionary containing validated clauses.
        """
        results = {}
        for clause_type, paragraphs in clause_queries.items():
            try:
                response = self.graph.invoke({"query": clause_type, "retrieved_paragraphs": paragraphs})
                results[clause_type] = response["validated_paragraphs"]
            except Exception as e:
                logger.error(f"Error processing clause '{clause_type}': {e}")
                results[clause_type] = {"error": "Processing failed."}
        return results

    def _initialize_graph(self) -> StateGraph:
        """
        Defines the LangGraph workflow for validation only.
        """
        workflow = StateGraph(Dict[str, Any])
        workflow.add_node("validate", self._validate_clauses)

        workflow.add_edge(START, "validate")
        workflow.add_edge("validate", END)

        logger.info("✅ LangGraph validation workflow initialized.")
        return workflow.compile()

    def _validate_clauses(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the retrieved paragraphs using the LLM model.
        """
        retrieved_paragraphs = state.get("retrieved_paragraphs", [])
        clause_type = state["query"]

        if not retrieved_paragraphs:
            logger.warning(f"⚠️ No paragraphs retrieved for validation of '{clause_type}'.")
            return {**state, "validated_paragraphs": []}

        logger.info(f"🔍 Validating {len(retrieved_paragraphs)} paragraphs for '{clause_type}'.")

        # Batch process all paragraphs for better performance
        batch_inputs = [
            f"Does this paragraph correspond to a '{clause_type}'?\n\nParagraph:\n{p}\n\nAnswer with 'Yes' or 'No'."
            for p in retrieved_paragraphs
        ]

        inputs = self.tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.llm_model.generate(inputs["input_ids"], max_new_tokens=3)

        validated_paragraphs = []
        rejected_paragraphs = []

        for para, output in zip(retrieved_paragraphs, outputs):
            answer = self.tokenizer.decode(output, skip_special_tokens=True).lower()
            if answer == "yes":
                validated_paragraphs.append(para)
            else:
                rejected_paragraphs.append(para)

        logger.info(f"✅ {len(validated_paragraphs)} paragraphs validated for '{clause_type}'.")
        logger.info(f"❌ {len(rejected_paragraphs)} paragraphs rejected for '{clause_type}'.")

        return {**state, "validated_paragraphs": validated_paragraphs}

    def _load_llm_model(self, model_name: str) -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
        """
        Loads the pre-trained LLM model and tokenizer.
        """
        logger.info(f"🔄 Loading LLM model: {model_name}")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.eval()
        logger.info(f"✅ Model '{model_name}' loaded successfully.")
        return tokenizer, model
