"""
medrag.index

Indexe les chunks JSONL dans ChromaDB avec des embeddings semantiques
via intfloat/multilingual-e5-base.

Usage CLI:
    medrag-index chunks.jsonl
    medrag-index chunks.jsonl --no-reset
"""

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)
import os

app = typer.Typer(help="Indexe les chunks dans ChromaDB avec multilingual-e5-base")
console = Console()

# === Configuration (defaults, overridable via class) ===
DEFAULT_CHROMA_PATH = "./data/chromadb"
DEFAULT_COLLECTION_NAME = "medrag_corpus"
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_BATCH_SIZE = 32


def load_chunks(jsonl_path: Path) -> list[dict]:
    """Charge les chunks depuis un fichier JSONL."""
    chunks = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def prepare_metadata(chunk: dict) -> dict:
    """Prepare les metadonnees pour ChromaDB (strings/int/float uniquement)."""
    return {
        "source": chunk["source"],
        "chapter": chunk.get("chapter") or "",
        "pathology": chunk.get("pathology") or "",
        "section_path": " > ".join(chunk["section_path"]),
        "depth": chunk["depth"],
        "char_count": chunk["char_count"],
        "chunk_index": chunk["chunk_index"],
    }


def get_db_size(path: str) -> str:
    """Calcule la taille du dossier ChromaDB."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    if total > 1024 * 1024:
        return f"{total / (1024 * 1024):.1f} MB"
    return f"{total / 1024:.1f} KB"


def index_chunks(
    chunks: list[dict],
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embed_model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    no_reset: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Indexe une liste de chunks dans ChromaDB.

    Args:
        chunks: liste de dicts avec au minimum "id", "text", "source", "section_path", etc.
        chroma_path: chemin du dossier ChromaDB persistant
        collection_name: nom de la collection
        embed_model: modele sentence-transformers
        batch_size: taille des batches
        no_reset: si True, garde la collection existante
        verbose: affichage rich

    Returns:
        dict avec les stats d'indexation
    """
    from sentence_transformers import SentenceTransformer
    from chromadb import PersistentClient

    if verbose:
        console.print(f"[bold]-> Chargement du modele {embed_model}...[/]")

    model = SentenceTransformer(embed_model)
    embed_dim = model.get_sentence_embedding_dimension()

    if verbose:
        console.print(f"[green]OK[/] Modele charge (dim={embed_dim})\n")

    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=chroma_path)

    if not no_reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    else:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    start_time = time.time()

    if verbose:
        console.print("[bold]-> Encoding + indexation...[/]\n")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # Prefixe e5 OBLIGATOIRE pour les documents
        texts_to_encode = [f"passage: {c['text']}" for c in batch]

        embeddings = model.encode(
            texts_to_encode,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        ).tolist()

        metadatas = [prepare_metadata(c) for c in batch]

        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=metadatas,
        )

    elapsed = time.time() - start_time
    final_count = collection.count()

    return {
        "collection_name": collection_name,
        "chroma_path": chroma_path,
        "documents": final_count,
        "embed_dim": embed_dim,
        "embed_model": embed_model,
        "elapsed_seconds": elapsed,
        "speed": len(chunks) / elapsed if elapsed > 0 else 0,
    }


@app.command()
def main(
    jsonl_path: str = typer.Argument(..., help="Chemin du chunks.jsonl"),
    no_reset: bool = typer.Option(
        False, "--no-reset",
        help="Garder la collection existante au lieu de la reset",
    ),
    chroma_path: str = typer.Option(DEFAULT_CHROMA_PATH, "--chroma-path"),
    collection_name: str = typer.Option(DEFAULT_COLLECTION_NAME, "--collection"),
):
    """Indexe les chunks dans ChromaDB avec multilingual-e5-base."""
    src = Path(jsonl_path)
    if not src.exists():
        console.print(f"[red]X Fichier introuvable : {src}[/]")
        raise typer.Exit(1)

    console.print("[bold]-> Chargement des chunks...[/]")
    chunks = load_chunks(src)
    console.print(f"[green]OK[/] {len(chunks)} chunks charges\n")

    result = index_chunks(
        chunks,
        chroma_path=chroma_path,
        collection_name=collection_name,
        no_reset=no_reset,
    )

    db_size = get_db_size(chroma_path)
    minutes = int(result["elapsed_seconds"] // 60)
    seconds = int(result["elapsed_seconds"] % 60)

    body = f"""[bold]Collection[/]      : {result['collection_name']}
[bold]Path[/]            : {result['chroma_path']}
[bold]Documents[/]       : [cyan]{result['documents']}[/]
[bold]Embeddings dim[/]  : [cyan]{result['embed_dim']}[/]
[bold]Modele[/]          : {result['embed_model']}
[bold]Duree totale[/]    : [cyan]{minutes}m {seconds:02d}s[/]
[bold]Vitesse moyenne[/] : [cyan]{result['speed']:.1f}[/] chunks/sec
[bold]Taille DB[/]       : [cyan]{db_size}[/]"""

    console.print(Panel(
        body,
        title="[Indexation terminee]",
        border_style="green",
        padding=(1, 2),
    ))


if __name__ == "__main__":
    app()
