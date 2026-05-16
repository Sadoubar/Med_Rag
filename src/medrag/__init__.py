"""medrag - Pipeline RAG medical adaptatif."""

__version__ = "0.1.0"
__author__ = "Sadou BARRY"

from medrag.client import MedRAG
from medrag.search import SearchResult
from medrag.exceptions import MedRAGError

__all__ = ["MedRAG", "SearchResult", "MedRAGError", "__version__"]
