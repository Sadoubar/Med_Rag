"""
medrag.client

Classe orchestratrice MedRAG : pipeline complet extract -> hierarchy -> chunk -> index -> search.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Optional

from medrag.exceptions import MedRAGError
from medrag.search import SearchResult


class MedRAG:
    """Pipeline RAG medical complet : extract -> hierarchy -> chunk -> index -> search."""

    def __init__(
        self,
        chroma_path: str = "./chromadb",
        collection_name: str = "medrag_corpus",
        embedding_model: str = "intfloat/multilingual-e5-base",
        extract_api_url: Optional[str] = None,
        extract_api_key: Optional[str] = None,
    ):
        """
        Args:
            chroma_path: dossier ChromaDB persistant
            collection_name: nom de la collection
            embedding_model: modele sentence-transformers
            extract_api_url: URL de l'API d'extraction Modal (optionnel)
            extract_api_key: cle API pour l'API d'extraction (optionnel)

        Si extract_api_url est fourni, utilise l'API distante.
        Sinon, tente docling en local (necessite pip install medrag[docling]).
        """
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.extract_api_url = extract_api_url
        self.extract_api_key = extract_api_key
        # Initialisation paresseuse des composants
        self._searcher = None
        self._all_chunks: list[dict] = []

    def add_pdf(self, pdf_path: str | Path, source_name: Optional[str] = None) -> dict:
        """Pipeline complet pour un PDF : extract -> hierarchy -> chunk -> index."""
        from medrag.extract import extract_pdf
        from medrag.hierarchy import reconstruct_text
        from medrag.chunk import chunk_markdown_text
        from medrag.index import index_chunks

        pdf_path = Path(pdf_path)
        if source_name is None:
            source_name = pdf_path.stem

        # 1. Extract
        markdown_text = extract_pdf(
            pdf_path,
            api_url=self.extract_api_url,
            api_key=self.extract_api_key,
        )

        # 2. Reconstruire la hierarchie
        reconstructed = reconstruct_text(markdown_text)

        # 3. Chunker
        chunks, stats = chunk_markdown_text(reconstructed, source_name)

        # 4. Indexer (no_reset=True par defaut, on ajoute au corpus existant)
        chunks_dicts = [asdict(c) for c in chunks]
        index_result = index_chunks(
            chunks_dicts,
            chroma_path=self.chroma_path,
            collection_name=self.collection_name,
            embed_model=self.embedding_model,
            no_reset=True,
            verbose=False,
        )

        # Invalider le searcher cache
        self._searcher = None
        self._all_chunks.extend(chunks_dicts)

        return {
            "source": source_name,
            "chunks_created": stats.total_chunks,
            "total_indexed": index_result["documents"],
        }

    def add_markdown(self, markdown_text: str, source_name: str) -> dict:
        """Pipeline pour du markdown deja extrait : hierarchy -> chunk -> index."""
        from medrag.hierarchy import reconstruct_text
        from medrag.chunk import chunk_markdown_text
        from medrag.index import index_chunks

        # 1. Reconstruire la hierarchie
        reconstructed = reconstruct_text(markdown_text)

        # 2. Chunker
        chunks, stats = chunk_markdown_text(reconstructed, source_name)

        # 3. Indexer
        chunks_dicts = [asdict(c) for c in chunks]
        index_result = index_chunks(
            chunks_dicts,
            chroma_path=self.chroma_path,
            collection_name=self.collection_name,
            embed_model=self.embedding_model,
            no_reset=True,
            verbose=False,
        )

        self._searcher = None
        self._all_chunks.extend(chunks_dicts)

        return {
            "source": source_name,
            "chunks_created": stats.total_chunks,
            "total_indexed": index_result["documents"],
        }

    def _ensure_searcher(self):
        """Initialise le searcher si pas encore fait."""
        if self._searcher is None:
            from medrag.search import HybridSearcher
            if not self._all_chunks:
                raise MedRAGError(
                    "Aucun chunk en memoire. Appelez add_pdf() ou add_markdown() d'abord, "
                    "ou utilisez HybridSearcher.from_jsonl() directement."
                )
            self._searcher = HybridSearcher.from_chunks(
                self._all_chunks,
                chroma_path=self.chroma_path,
                collection_name=self.collection_name,
                embed_model=self.embedding_model,
            )

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        """Recherche hybride BM25 + Dense + RRF."""
        self._ensure_searcher()
        return self._searcher.search(query, top_k=top_k)

    def benchmark(self, queries: list[str], top_k: int = 4) -> dict:
        """Lance un benchmark sur une liste de requetes."""
        from medrag.benchmark import run_benchmark
        self._ensure_searcher()
        report = run_benchmark(self._searcher, queries, top_k=top_k)
        return {
            "total_queries": report.total_queries,
            "queries_with_both": report.queries_with_both,
            "avg_results_per_query": report.avg_results_per_query,
            "source_distribution": report.source_distribution,
        }

    def count(self) -> int:
        """Nombre de chunks indexes."""
        from chromadb import PersistentClient
        client = PersistentClient(path=self.chroma_path)
        try:
            collection = client.get_collection(self.collection_name)
            return collection.count()
        except Exception:
            return 0
