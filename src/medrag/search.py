"""
medrag.search

Recherche hybride BM25 + Dense avec fusion RRF dans le corpus medical.

Usage CLI:
    medrag-search "traitement paludisme grave"
    medrag-search "cefotaxime nourrisson" --debug
    medrag-search "anemie" --top-k 5
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Recherche hybride BM25 + Dense dans le corpus medical")
console = Console()

# === Configuration (defaults) ===
DEFAULT_CHROMA_PATH = "./data/chromadb"
DEFAULT_COLLECTION_NAME = "medrag_corpus"
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_TOP_K_PER_SOURCE = 20    # top 20 BM25 + top 20 Dense
DEFAULT_RRF_K = 60               # constante standard pour RRF
DEFAULT_FINAL_TOP_K = 4          # nombre de resultats finaux apres fusion

# === Stop words francais courants ===
FRENCH_STOPWORDS = {
    "le", "la", "les", "des", "du", "de", "un", "une", "et", "ou",
    "a", "au", "aux", "ce", "cet", "cette", "ces", "il", "elle",
    "ils", "elles", "on", "se", "son", "sa", "ses", "leur", "leurs",
    "pour", "par", "avec", "sans", "dans", "sur", "sous", "vers",
    "chez", "comment", "quand", "que", "qui", "quoi", "ou",
    "est", "sont", "ai", "ais", "etait", "ete", "etre", "avoir",
    "faut", "doit", "peut", "fait", "en", "ne", "pas", "plus",
    "tout", "tous", "toute", "toutes", "tres", "meme",
    "si", "alors", "donc", "car", "aussi", "puis",
}


@dataclass
class SearchResult:
    """Resultat de recherche hybride."""
    id: str
    text: str
    section_path: list[str]
    score: float          # RRF score
    cosine_sim: float | None
    source: str           # "both", "bm25", "dense"
    chapter: str | None
    pathology: str | None


def tokenize_fr(text: str) -> list[str]:
    """
    Tokenization adaptee au francais medical :
    - Lowercasing
    - Conserve les traits d'union (pour les DCI : artemether-lumefantrine)
    - Supprime ponctuation
    - Supprime stop words
    - Pas de stemming agressif (preserve les DCI exacts)
    """
    text = text.lower()
    # Match les mots (avec lettres accentuees, chiffres, traits d'union)
    tokens = re.findall(r"\b[\w\-]+\b", text, flags=re.UNICODE)
    # Filtre stop words et tokens trop courts
    return [t for t in tokens if t not in FRENCH_STOPWORDS and len(t) >= 2]


# === Fusion RRF ===

def rrf_fusion(
    bm25_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float, str]]:
    """
    Fusion par Reciprocal Rank Fusion.

    Returns:
        Liste de (chunk_id, score_rrf, source) triee par score decroissant.
        source = "both" si present dans BM25 ET Dense, sinon "bm25" ou "dense".
    """
    rrf_scores = defaultdict(float)
    bm25_ids = set()
    dense_ids = set()

    for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
        rrf_scores[chunk_id] += 1.0 / (k + rank)
        bm25_ids.add(chunk_id)

    for rank, (chunk_id, _) in enumerate(dense_results, start=1):
        rrf_scores[chunk_id] += 1.0 / (k + rank)
        dense_ids.add(chunk_id)

    # Trier et marquer la source
    results = []
    for chunk_id, score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
        if chunk_id in bm25_ids and chunk_id in dense_ids:
            source = "both"
        elif chunk_id in bm25_ids:
            source = "bm25"
        else:
            source = "dense"
        results.append((chunk_id, score, source))

    return results


# === HybridSearcher ===

class HybridSearcher:
    """Recherche hybride BM25 + Dense avec persistance en memoire."""

    def __init__(
        self,
        chunks: list[dict],
        bm25,
        model,
        collection,
        rrf_k: int = DEFAULT_RRF_K,
        top_k_per_source: int = DEFAULT_TOP_K_PER_SOURCE,
    ):
        self.chunks = chunks
        self.chunks_by_id = {c["id"]: c for c in chunks}
        self.bm25 = bm25
        self.model = model
        self.collection = collection
        self.rrf_k = rrf_k
        self.top_k_per_source = top_k_per_source

    @classmethod
    def from_chunks(
        cls,
        chunks: list[dict],
        chroma_path: str = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embed_model: str = DEFAULT_EMBED_MODEL,
        rrf_k: int = DEFAULT_RRF_K,
        top_k_per_source: int = DEFAULT_TOP_K_PER_SOURCE,
    ):
        """Cree un searcher a partir d'une liste de chunks deja charges."""
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        from chromadb import PersistentClient

        tokenized = [tokenize_fr(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)

        model = SentenceTransformer(embed_model)

        client = PersistentClient(path=chroma_path)
        collection = client.get_collection(collection_name)

        return cls(chunks, bm25, model, collection, rrf_k, top_k_per_source)

    @classmethod
    def from_jsonl(
        cls,
        jsonl_paths: list[str | Path],
        chroma_path: str = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embed_model: str = DEFAULT_EMBED_MODEL,
        rrf_k: int = DEFAULT_RRF_K,
        top_k_per_source: int = DEFAULT_TOP_K_PER_SOURCE,
    ):
        """Cree un searcher a partir de fichiers JSONL."""
        chunks = []
        for jf in jsonl_paths:
            with open(jf, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))

        return cls.from_chunks(
            chunks, chroma_path, collection_name, embed_model,
            rrf_k, top_k_per_source,
        )

    def search_bm25(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Retourne les top_k chunks par BM25 (id, score)."""
        top_k = top_k or self.top_k_per_source
        query_tokens = tokenize_fr(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.chunks[i]["id"], scores[i]) for i in top_indices if scores[i] > 0]

    def search_dense(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Retourne les top_k chunks par dense (id, similarite cosine)."""
        top_k = top_k or self.top_k_per_source
        # IMPORTANT : prefixe "query: " pour e5
        query_emb = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
        )[0].tolist()

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
        )

        ids = results["ids"][0]
        distances = results["distances"][0]
        # cosine similarity = 1 - distance
        return [(id_, 1 - dist) for id_, dist in zip(ids, distances)]

    def search(self, query: str, top_k: int = DEFAULT_FINAL_TOP_K, debug: bool = False) -> list[SearchResult]:
        """
        Recherche hybride BM25 + Dense avec fusion RRF.

        Returns:
            Liste de SearchResult triee par score RRF decroissant.
        """
        bm25_results = self.search_bm25(query)
        dense_results = self.search_dense(query)
        fused = rrf_fusion(bm25_results, dense_results, k=self.rrf_k)

        dense_sims = {id_: sim for id_, sim in dense_results}

        if debug:
            self._debug_print(query, bm25_results, dense_results, fused, top_k)

        results = []
        for chunk_id, rrf_score, source in fused[:top_k]:
            chunk = self.chunks_by_id[chunk_id]
            results.append(SearchResult(
                id=chunk_id,
                text=chunk["text"],
                section_path=chunk["section_path"],
                score=rrf_score,
                cosine_sim=dense_sims.get(chunk_id),
                source=source,
                chapter=chunk.get("chapter"),
                pathology=chunk.get("pathology"),
            ))
        return results

    def _debug_print(self, query, bm25_results, dense_results, fused, top_k):
        """Affiche en parallele les top 5 BM25 / Dense / RRF."""
        table = Table(title=f"Debug : {query}", show_header=True)
        table.add_column("Rang", justify="right", width=4)
        table.add_column("BM25 (top 5)", style="blue")
        table.add_column("Dense (top 5)", style="magenta")
        table.add_column("RRF final (top 5)", style="green")

        for i in range(5):
            bm25_str = ""
            dense_str = ""
            rrf_str = ""
            if i < len(bm25_results):
                chunk = self.chunks_by_id[bm25_results[i][0]]
                bm25_str = f"{chunk.get('pathology') or 'N/A'} ({bm25_results[i][1]:.2f})"
            if i < len(dense_results):
                chunk = self.chunks_by_id[dense_results[i][0]]
                dense_str = f"{chunk.get('pathology') or 'N/A'} ({dense_results[i][1]:.3f})"
            if i < len(fused):
                chunk = self.chunks_by_id[fused[i][0]]
                rrf_str = f"{chunk.get('pathology') or 'N/A'} [{fused[i][2]}]"
            table.add_row(str(i + 1), bm25_str, dense_str, rrf_str)

        console.print(table)


# === Affichage rich ===

def display_results(results: list[SearchResult], query: str):
    """Affichage des resultats hybrides."""
    console.print(f"\n[bold yellow]Question : {query}[/]\n")

    for rank, r in enumerate(results, start=1):
        if r.source == "both":
            badge = "[bold green]* BOTH[/]"
        elif r.source == "bm25":
            badge = "[blue]BM25 only[/]"
        else:
            badge = "[magenta]DENSE only[/]"

        sim_str = f"sim={r.cosine_sim:.3f}" if r.cosine_sim else "sim=N/A"

        title = f"#{rank} -- {r.id} [{badge}]"

        section_str = " > ".join(r.section_path)
        preview = r.text[:400]
        if len(r.text) > 400:
            preview += "..."

        body = f"""[bold]Section[/]    : {section_str}
[bold]RRF score[/]  : {r.score:.4f}
[bold]Cosine[/]     : {sim_str}

[dim]{preview}[/]"""

        console.print(Panel(body, title=title, border_style="cyan", padding=(1, 2)))


# === CLI typer ===

@app.command()
def main(
    query: str = typer.Argument(..., help="Question medicale a rechercher"),
    top_k: int = typer.Option(DEFAULT_FINAL_TOP_K, "--top-k", "-k"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Afficher BM25/Dense/RRF en parallele"),
    chroma_path: str = typer.Option(DEFAULT_CHROMA_PATH, "--chroma-path"),
    collection_name: str = typer.Option(DEFAULT_COLLECTION_NAME, "--collection"),
):
    """Recherche hybride BM25 + Dense dans le corpus medical."""
    import glob

    console.print("[dim]-> Initialisation de la recherche hybride...[/]")

    # Charger les chunks depuis tous les JSONL dans le CWD
    jsonl_files = sorted(glob.glob("chunks*.jsonl"))
    if not jsonl_files:
        console.print(f"[red]X Aucun fichier chunks*.jsonl trouve[/]")
        raise typer.Exit(1)

    searcher = HybridSearcher.from_jsonl(
        jsonl_files,
        chroma_path=chroma_path,
        collection_name=collection_name,
    )
    console.print("[green]OK Pret[/]\n")

    results = searcher.search(query, top_k=top_k, debug=debug)
    display_results(results, query)


if __name__ == "__main__":
    app()
