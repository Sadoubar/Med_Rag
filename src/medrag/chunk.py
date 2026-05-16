"""
medrag.chunk

Decoupe un markdown reconstruit (avec hierarchie H1/H2/H3/H4) en chunks
RAG-ready au format JSONL, en exploitant la hierarchie pour enrichir
chaque chunk avec son contexte complet (section_path).

Usage CLI:
    medrag-chunk source_reconstructed.md chunks.jsonl
"""

from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import re

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Decoupe un markdown reconstruit en chunks RAG-ready")
console = Console()


# === Configuration ===
DEFAULT_MAX_CHUNK_CHARS = 1500
DEFAULT_SUB_CHUNK_TARGET = 1000
DEFAULT_OVERLAP = 150
DEFAULT_MIN_CHUNK_CHARS = 200

META_TITLES = {
    "Guide clinique et th\u00e9rapeutique",
    "Table des mati\u00e8res",
    "Auteurs/Contributeurs",
    "Auteurs / Contributeurs",
    "Avant-propos",
    "Abr\u00e9viations, sigles et acronymes",
    "Pr\u00e9face",
    "Index",
    "Bibliographie",
    "Remerciements",
}

META_COMMENT_RE = re.compile(r"<!--\s*META[:\s][^>]*-->")
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$")


# === Dataclasses ===

@dataclass
class Chunk:
    """Un chunk RAG-ready."""
    id: str
    source: str
    chapter: str | None
    pathology: str | None
    section_path: list[str]
    text: str
    char_count: int
    chunk_index: int
    depth: int


@dataclass
class ChunkingStats:
    """Statistiques du chunking."""
    total_chunks: int = 0
    avg_chars: float = 0.0
    median_chars: float = 0.0
    min_chars: int = 0
    max_chars: int = 0
    filtered_meta_sections: int = 0
    filtered_orphan_chunks: int = 0
    chunks_per_chapter: Counter = field(default_factory=Counter)


# === Helpers ===

def clean_html_entities(text: str) -> str:
    """Nettoie les entites HTML residuelles."""
    return (
        text.replace("&gt;", ">")
            .replace("&lt;", "<")
            .replace("&amp;", "&")
            .replace("&quot;", '"')
            .replace("&apos;", "'")
            .replace("&nbsp;", " ")
    )


def is_meta_section(section_path: list[str]) -> bool:
    """Determine si un chunk est dans une meta-section a filtrer."""
    return any(title in META_TITLES for title in section_path)


def make_prefixed_text(section_path: list[str], content: str) -> str:
    """Prefixe le contenu avec la hierarchie pour enrichir l'embedding."""
    section_str = " > ".join(section_path)
    return f"[Section] {section_str}\n\n{content.strip()}"


def split_long_content(
    content: str,
    target_chars: int,
    overlap: int,
) -> list[str]:
    """Decoupe un contenu trop long en sous-chunks avec overlap."""
    if len(content) <= target_chars:
        return [content]

    sub_chunks = []
    start = 0
    while start < len(content):
        end = start + target_chars
        if end >= len(content):
            sub_chunks.append(content[start:])
            break
        # Essayer de couper sur un saut de ligne ou un point
        cut = content.rfind("\n\n", start, end)
        if cut == -1 or cut < start + target_chars // 2:
            cut = content.rfind(". ", start, end)
        if cut == -1 or cut < start + target_chars // 2:
            cut = end
        else:
            cut += 1  # garder le point dans le chunk precedent
        sub_chunks.append(content[start:cut])
        start = max(cut - overlap, start + 1)

    return sub_chunks


# === Parsing principal ===

def parse_markdown_into_sections(md_path: Path) -> list[dict]:
    """
    Parse le markdown en liste de sections avec leur section_path.

    Retourne une liste de dicts :
        {
            "section_path": [...],
            "content": "...",
            "depth": int,
        }
    """
    lines = md_path.read_text(encoding="utf-8").splitlines()
    return parse_markdown_text_into_sections(lines)


def parse_markdown_text_into_sections(lines: list[str]) -> list[dict]:
    """Parse des lignes markdown en liste de sections avec leur section_path."""
    sections = []
    section_stack = []  # liste de (level, title)
    current_content = []

    def flush_current(stack_snapshot):
        """Sauvegarde la section courante si elle a du contenu."""
        if current_content and stack_snapshot:
            content = "\n".join(current_content).strip()
            content = clean_html_entities(content)
            if content:  # ignorer si vide apres nettoyage
                sections.append({
                    "section_path": [t for _, t in stack_snapshot],
                    "content": content,
                    "depth": stack_snapshot[-1][0] if stack_snapshot else 0,
                })

    for line in lines:
        # Ignorer les commentaires META
        if META_COMMENT_RE.search(line):
            continue

        h_match = HEADING_RE.match(line)
        if h_match:
            # Flush la section precedente
            flush_current(list(section_stack))
            current_content = []

            # Mettre a jour la stack
            level = len(h_match.group(1))
            title = h_match.group(2).strip()
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, title))
        else:
            current_content.append(line)

    # Flush la derniere section
    flush_current(list(section_stack))

    return sections


# === Chunking ===

def chunk_sections(
    sections: list[dict],
    source_name: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    sub_chunk_target: int = DEFAULT_SUB_CHUNK_TARGET,
    overlap: int = DEFAULT_OVERLAP,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> tuple[list[Chunk], ChunkingStats]:
    """
    Decoupe les sections en chunks RAG-ready.

    Returns:
        (chunks, stats)
    """
    chunks: list[Chunk] = []
    stats = ChunkingStats()
    chunk_index = 0

    for section in sections:
        section_path = section["section_path"]
        content = section["content"]
        depth = section["depth"]

        # Filtre 1 : meta-sections
        if is_meta_section(section_path):
            stats.filtered_meta_sections += 1
            continue

        # Filtre 2 : contenu trop court
        if len(content) < min_chunk_chars:
            stats.filtered_orphan_chunks += 1
            continue

        # Recuperer chapter et pathology
        chapter = section_path[0] if len(section_path) >= 1 else None
        pathology = section_path[1] if len(section_path) >= 2 else None

        # Si trop long, sous-chunker
        sub_contents = (
            split_long_content(content, sub_chunk_target, overlap)
            if len(content) > max_chunk_chars
            else [content]
        )

        for sub_content in sub_contents:
            prefixed = make_prefixed_text(section_path, sub_content)

            # Filtre final : skip si toujours trop court apres prefixe
            if len(prefixed) < min_chunk_chars:
                stats.filtered_orphan_chunks += 1
                continue

            chunk = Chunk(
                id=f"{source_name}_chunk_{chunk_index:05d}",
                source=source_name,
                chapter=chapter,
                pathology=pathology,
                section_path=section_path,
                text=prefixed,
                char_count=len(prefixed),
                chunk_index=chunk_index,
                depth=depth,
            )
            chunks.append(chunk)
            chunk_index += 1

            if chapter:
                stats.chunks_per_chapter[chapter] += 1

    # Stats finales
    char_counts = [c.char_count for c in chunks]
    if char_counts:
        stats.total_chunks = len(chunks)
        stats.avg_chars = sum(char_counts) / len(char_counts)
        stats.median_chars = sorted(char_counts)[len(char_counts) // 2]
        stats.min_chars = min(char_counts)
        stats.max_chars = max(char_counts)

    return chunks, stats


def chunk_markdown_text(
    text: str,
    source_name: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    sub_chunk_target: int = DEFAULT_SUB_CHUNK_TARGET,
    overlap: int = DEFAULT_OVERLAP,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> tuple[list[Chunk], ChunkingStats]:
    """Chunk du texte markdown directement (sans fichier)."""
    lines = text.splitlines()
    sections = parse_markdown_text_into_sections(lines)
    return chunk_sections(
        sections, source_name,
        max_chunk_chars=max_chunk_chars,
        sub_chunk_target=sub_chunk_target,
        overlap=overlap,
        min_chunk_chars=min_chunk_chars,
    )


# === Sauvegarde JSONL ===

def save_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    """Sauvegarde les chunks au format JSONL (un chunk par ligne)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


# === Affichage rich ===

def display_report(stats: ChunkingStats, source: str, output: str):
    """Affiche le rapport de chunking."""
    body = f"""[bold]Source[/]        : {source}
[bold]Output[/]        : {output}

[bold]Total chunks[/]  : [cyan]{stats.total_chunks}[/]
[bold]Caracteres[/]    : avg=[cyan]{stats.avg_chars:.0f}[/] / median=[cyan]{stats.median_chars:.0f}[/]
[bold]Longueur[/]      : min=[cyan]{stats.min_chars}[/] / max=[cyan]{stats.max_chars}[/]

[bold]Filtres[/]       : [yellow]{stats.filtered_meta_sections}[/] meta-sections + [yellow]{stats.filtered_orphan_chunks}[/] orphelins"""

    console.print(Panel(body, title="[Chunking termine]",
                        border_style="green", padding=(1, 2)))


def display_chapter_distribution(stats: ChunkingStats):
    """Affiche la distribution par chapitre."""
    table = Table(title="Distribution par chapitre", show_header=True,
                  header_style="bold cyan")
    table.add_column("Chapitre", style="white")
    table.add_column("Chunks", justify="right", style="cyan")

    for chapter, count in sorted(stats.chunks_per_chapter.items()):
        table.add_row(chapter, str(count))

    console.print(table)


# === CLI principal ===

@app.command()
def main(
    input_path: str = typer.Argument(..., help="Markdown reconstruit en entree"),
    output_path: str = typer.Argument("chunks.jsonl", help="Fichier JSONL de sortie"),
    max_chars: int = typer.Option(DEFAULT_MAX_CHUNK_CHARS, "--max-chars"),
    overlap: int = typer.Option(DEFAULT_OVERLAP, "--overlap"),
    min_chars: int = typer.Option(DEFAULT_MIN_CHUNK_CHARS, "--min-chars"),
    show_samples: bool = typer.Option(True, "--show-samples/--no-samples"),
):
    """Decoupe un markdown reconstruit en chunks RAG-ready."""
    src = Path(input_path)
    out = Path(output_path)

    if not src.exists():
        console.print(f"[red]X Fichier introuvable : {src}[/]")
        raise typer.Exit(1)

    source_name = src.stem.replace("_reconstructed", "")

    console.print(f"[bold]-> Parsing du markdown...[/]")
    sections = parse_markdown_into_sections(src)
    console.print(f"[green]OK[/] {len(sections)} sections detectees")

    console.print(f"[bold]-> Generation des chunks...[/]")
    chunks, stats = chunk_sections(
        sections, source_name,
        max_chunk_chars=max_chars,
        overlap=overlap,
        min_chunk_chars=min_chars,
    )

    console.print(f"[bold]-> Sauvegarde JSONL...[/]")
    save_jsonl(chunks, out)
    console.print(f"[green]OK[/] Sauvegarde : {out}\n")

    display_report(stats, src.name, out.name)
    console.print()
    display_chapter_distribution(stats)


if __name__ == "__main__":
    app()
