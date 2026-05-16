"""
medrag.benchmark

Benchmark integre pour evaluer la qualite de la recherche hybride.
"""

from dataclasses import dataclass, field
from typing import Optional

from medrag.search import HybridSearcher, SearchResult


@dataclass
class BenchmarkResult:
    """Resultat d'un benchmark sur une requete."""
    query: str
    results: list[SearchResult]
    top_sources: list[str]  # "both", "bm25", "dense"
    has_both: bool          # au moins un resultat vient des 2 sources


@dataclass
class BenchmarkReport:
    """Rapport de benchmark sur l'ensemble des requetes."""
    total_queries: int = 0
    queries_with_both: int = 0         # requetes ou au moins 1 resultat = "both"
    avg_results_per_query: float = 0.0
    source_distribution: dict = field(default_factory=dict)  # both/bm25/dense counts
    results: list[BenchmarkResult] = field(default_factory=list)


def run_benchmark(
    searcher: HybridSearcher,
    queries: list[str],
    top_k: int = 4,
) -> BenchmarkReport:
    """
    Lance un benchmark sur une liste de requetes.

    Args:
        searcher: instance HybridSearcher deja initialisee
        queries: liste de requetes medicales
        top_k: nombre de resultats par requete

    Returns:
        BenchmarkReport avec les stats agregees
    """
    report = BenchmarkReport()
    report.total_queries = len(queries)
    source_counts = {"both": 0, "bm25": 0, "dense": 0}
    total_results = 0

    for query in queries:
        results = searcher.search(query, top_k=top_k)
        top_sources = [r.source for r in results]
        has_both = "both" in top_sources

        if has_both:
            report.queries_with_both += 1

        for r in results:
            source_counts[r.source] = source_counts.get(r.source, 0) + 1

        total_results += len(results)

        report.results.append(BenchmarkResult(
            query=query,
            results=results,
            top_sources=top_sources,
            has_both=has_both,
        ))

    report.avg_results_per_query = total_results / len(queries) if queries else 0
    report.source_distribution = source_counts

    return report
