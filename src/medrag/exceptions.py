"""medrag.exceptions - Exceptions pour le pipeline medrag."""


class MedRAGError(Exception):
    """Base exception pour medrag."""


class ExtractionError(MedRAGError):
    """Erreur lors de l'extraction PDF."""


class IndexingError(MedRAGError):
    """Erreur lors de l'indexation."""


class SearchError(MedRAGError):
    """Erreur lors de la recherche."""
