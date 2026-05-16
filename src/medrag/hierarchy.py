"""
medrag.hierarchy

Reconstruit la hierarchie d'un markdown extrait par docling.
Adaptatif : detecte automatiquement la qualite de la hierarchie existante
et applique la strategie appropriee.

Usage CLI:
    medrag-hierarchy source.md
    medrag-hierarchy source.md output.md
    medrag-hierarchy source.md --force-strategy FLAT
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

app = typer.Typer(help="Reconstruct hierarchy in markdown files")
console = Console()

# === Regex de detection ===
CHAPTER_RE = re.compile(r"^(Chapitre|Annexe|Partie|Section)\s+[\dIVXivx]+", re.IGNORECASE)
SECTION_NUM_RE = re.compile(r"^(\d+)((?:\.\d+)+)\s+")  # 1.1, 1.1.1, 1.1.1.1
STEP_RE = re.compile(r"^(Étape|Phase|Stade)\s+\d", re.IGNORECASE)
NUMBERED_RE = re.compile(r"^\d+[\.\)]\s")
ROMAN_NUM_RE = re.compile(r"^[IVX]+[\.\)]\s")

# === Seuils statistiques (ajustables) ===
RECURRENT_THRESHOLD = 5         # un titre qui revient >=5 fois = sous-section clinique standard
EMPTY_TITLE_THRESHOLD = 80      # chars : titre quasi vide = titre de regroupement
META_POSITION_RATIO = 0.02      # premier 2% du doc = meta-pages probable
META_CONTENT_THRESHOLD = 500    # meta-page = peu de contenu propre


# === Dataclasses ===

@dataclass
class HierarchyStats:
    """Statistiques de hierarchie d'un document markdown."""
    h1: int = 0
    h2: int = 0
    h3: int = 0
    h4: int = 0
    total_titles: int = 0
    total_lines: int = 0
    diagnosis: str = "UNKNOWN"
    diagnosis_message: str = ""


@dataclass
class TitleStats:
    """Statistiques detaillees d'un titre individuel."""
    title: str
    line_num: int
    position_index: int      # 0, 1, 2... ordre d'apparition
    count: int               # nb occurrences dans le doc
    content_size: int        # chars jusqu'au prochain titre
    position_ratio: float    # 0.0 = debut, 1.0 = fin
    target_level: int = 0    # rempli apres classification (None si meta)


# === Phase 1 -- Diagnostic ===

def count_titles_by_level(lines: list[str]) -> dict:
    """Compte les titres par niveau dans le markdown."""
    return {
        'h1': sum(1 for l in lines if l.startswith("# ") and not l.startswith("## ")),
        'h2': sum(1 for l in lines if l.startswith("## ") and not l.startswith("### ")),
        'h3': sum(1 for l in lines if l.startswith("### ") and not l.startswith("#### ")),
        'h4': sum(1 for l in lines if l.startswith("#### ") and not l.startswith("##### ")),
    }


def diagnose(md_path: Path) -> HierarchyStats:
    """Diagnostique la qualite de la hierarchie d'un markdown."""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    counts = count_titles_by_level(lines)
    total = sum(counts.values())

    # Cas ideal : hierarchie multi-niveaux deja presente
    if counts['h1'] >= 5 and counts['h2'] >= 20 and counts['h3'] >= 50:
        dx, msg = "OK", "Hierarchie multi-niveaux deja bien detectee"
    # Hierarchie totalement plate
    elif counts['h1'] == 0 and counts['h3'] == 0 and counts['h2'] > 100:
        dx = "FLAT"
        msg = f"Hierarchie plate ({counts['h2']} titres au meme niveau)"
    # Pas de H1 mais H2/H3 presents
    elif counts['h1'] == 0 and counts['h2'] >= 10 and counts['h3'] >= 30:
        dx, msg = "MISSING_H1", "Chapitres manquants, inference H1 necessaire"
    # Tres peu de titres
    elif total < 20:
        dx, msg = "SPARSE", f"Document peu structure ({total} titres)"
    # Cas atypique
    else:
        dx = "UNKNOWN"
        msg = f"Distribution inattendue : H1={counts['h1']} H2={counts['h2']} H3={counts['h3']}"

    return HierarchyStats(
        h1=counts['h1'], h2=counts['h2'],
        h3=counts['h3'], h4=counts['h4'],
        total_titles=total, total_lines=len(lines),
        diagnosis=dx, diagnosis_message=msg,
    )


def diagnose_text(text: str) -> HierarchyStats:
    """Diagnostique la qualite de la hierarchie depuis du texte brut."""
    lines = text.splitlines()
    counts = count_titles_by_level(lines)
    total = sum(counts.values())

    if counts['h1'] >= 5 and counts['h2'] >= 20 and counts['h3'] >= 50:
        dx, msg = "OK", "Hierarchie multi-niveaux deja bien detectee"
    elif counts['h1'] == 0 and counts['h3'] == 0 and counts['h2'] > 100:
        dx = "FLAT"
        msg = f"Hierarchie plate ({counts['h2']} titres au meme niveau)"
    elif counts['h1'] == 0 and counts['h2'] >= 10 and counts['h3'] >= 30:
        dx, msg = "MISSING_H1", "Chapitres manquants, inference H1 necessaire"
    elif total < 20:
        dx, msg = "SPARSE", f"Document peu structure ({total} titres)"
    else:
        dx = "UNKNOWN"
        msg = f"Distribution inattendue : H1={counts['h1']} H2={counts['h2']} H3={counts['h3']}"

    return HierarchyStats(
        h1=counts['h1'], h2=counts['h2'],
        h3=counts['h3'], h4=counts['h4'],
        total_titles=total, total_lines=len(lines),
        diagnosis=dx, diagnosis_message=msg,
    )


# === Phase 2 -- Calcul des statistiques par titre ===

def parse_h2_titles(lines: list[str]) -> list[tuple[int, str]]:
    """Extrait tous les titres ## avec leur numero de ligne."""
    titles = []
    for i, line in enumerate(lines):
        if line.startswith("## ") and not line.startswith("### "):
            titles.append((i, line[3:].strip()))
    return titles


def compute_title_stats(
    lines: list[str],
    titles: list[tuple[int, str]],
) -> list[TitleStats]:
    """Calcule les statistiques (occurrences, content_size, position) pour chaque titre."""
    title_counts = Counter([t for _, t in titles])
    total_titles = len(titles)

    stats = []
    for idx, (line_num, title) in enumerate(titles):
        next_line = titles[idx + 1][0] if idx + 1 < len(titles) else len(lines)
        content = "\n".join(lines[line_num + 1:next_line])

        stats.append(TitleStats(
            title=title,
            line_num=line_num,
            position_index=idx,
            count=title_counts[title],
            content_size=len(content),
            position_ratio=idx / total_titles if total_titles > 0 else 0,
        ))
    return stats


# === Phase 3 -- Classification des titres ===

def classify_title(stat: TitleStats) -> int | None:
    """
    Determine le niveau cible d'un titre :
        - None : meta-page a exclure du RAG
        - 1    : chapitre / partie / annexe
        - 2    : pathologie / syndrome / medicament
        - 3    : sous-section clinique standardisee (recurrente)
        - 4    : sous-detail (etape, numerote)
    """
    title = stat.title

    # Regle 1 : Chapitre explicite (priorite max)
    if CHAPTER_RE.match(title):
        return 1

    # Regle 2 : Meta-page (position debut + peu de contenu)
    if (stat.position_ratio < META_POSITION_RATIO
            and stat.content_size < META_CONTENT_THRESHOLD):
        return None

    # Regle 3 : Sections numerotees hierarchiques (1.1, 1.1.1, 1.1.1.1)
    section_match = SECTION_NUM_RE.match(title)
    if section_match:
        dots = section_match.group(2).count(".")
        # "1.1" = 1 point -> H2
        # "1.1.1" = 2 points -> H3
        # "1.1.1.1" = 3+ points -> H4
        if dots <= 1:
            return 2
        elif dots == 2:
            return 3
        else:
            return 4

    # Regle 4 : Etape numerotee ou romain
    if STEP_RE.match(title) or NUMBERED_RE.match(title) or ROMAN_NUM_RE.match(title):
        return 4

    # Regle 5 (statistique) : titre recurrent = sous-section clinique
    if stat.count >= RECURRENT_THRESHOLD:
        return 3

    # Regle 6 : titre quasi vide = regroupement (devient H2)
    if stat.content_size < EMPTY_TITLE_THRESHOLD:
        return 2

    # Defaut : pathologie/syndrome
    return 2


# === Phase 4 -- Strategies de reconstruction ===

def strategy_copy_as_is(md_path: Path, output_path: Path) -> dict:
    """Strategie OK : copie sans modification."""
    output_path.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "strategy": "copy_as_is",
        "message": "Markdown utilise tel quel (hierarchie deja bonne)",
    }


def strategy_reconstruct_flat(md_path: Path, output_path: Path) -> dict:
    """Strategie FLAT : reconstruction complete par analyse statistique."""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    titles = parse_h2_titles(lines)
    stats = compute_title_stats(lines, titles)

    # Classifier chaque titre
    classifications = [classify_title(s) for s in stats]
    for s, level in zip(stats, classifications):
        s.target_level = level

    # Construire le nouveau markdown
    line_to_level = {s.line_num: s.target_level for s in stats}

    output_lines = []
    for i, line in enumerate(lines):
        if i in line_to_level:
            level = line_to_level[i]
            title_text = line[3:].strip()
            if level is None:
                # Meta-page : commentaire HTML pour filtrage ulterieur
                output_lines.append(f"<!-- META: {title_text} -->")
                output_lines.append(f"## {title_text}")
            else:
                prefix = "#" * level
                output_lines.append(f"{prefix} {title_text}")
        else:
            output_lines.append(line)

    output_path.write_text("\n".join(output_lines), encoding="utf-8")

    # Distribution finale
    final_dist = Counter(s.target_level for s in stats)

    return {
        "strategy": "reconstruct_flat",
        "stats": stats,
        "final_distribution": dict(final_dist),
    }


def reconstruct_text(text: str) -> str:
    """Reconstruit la hierarchie d'un texte markdown et retourne le resultat.

    Pratique pour l'usage en tant que bibliotheque (pas besoin de fichiers).
    """
    diag = diagnose_text(text)
    if diag.diagnosis == "OK":
        return text

    lines = text.splitlines()

    if diag.diagnosis == "FLAT":
        titles = parse_h2_titles(lines)
        stats = compute_title_stats(lines, titles)
        classifications = [classify_title(s) for s in stats]
        for s, level in zip(stats, classifications):
            s.target_level = level

        line_to_level = {s.line_num: s.target_level for s in stats}
        output_lines = []
        for i, line in enumerate(lines):
            if i in line_to_level:
                level = line_to_level[i]
                title_text = line[3:].strip()
                if level is None:
                    output_lines.append(f"<!-- META: {title_text} -->")
                    output_lines.append(f"## {title_text}")
                else:
                    prefix = "#" * level
                    output_lines.append(f"{prefix} {title_text}")
            else:
                output_lines.append(line)
        return "\n".join(output_lines)

    elif diag.diagnosis == "MISSING_H1":
        output_lines = []
        for line in lines:
            if line.startswith("## ") and CHAPTER_RE.match(line[3:].strip()):
                output_lines.append("#" + line[1:])
            else:
                output_lines.append(line)
        return "\n".join(output_lines)

    # SPARSE / UNKNOWN: return as-is
    return text


def strategy_infer_h1(md_path: Path, output_path: Path) -> dict:
    """Strategie MISSING_H1 : inferer les chapitres manquants."""
    lines = md_path.read_text(encoding="utf-8").splitlines()

    # Detecter les H2 qui matchent "Chapitre X :" -> les promouvoir en H1
    output_lines = []
    promoted = 0
    for line in lines:
        if line.startswith("## ") and CHAPTER_RE.match(line[3:].strip()):
            output_lines.append("#" + line[1:])  # ## -> #
            promoted += 1
        else:
            output_lines.append(line)

    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    return {
        "strategy": "infer_h1",
        "promoted_to_h1": promoted,
    }


STRATEGY_MAP = {
    "OK": strategy_copy_as_is,
    "FLAT": strategy_reconstruct_flat,
    "MISSING_H1": strategy_infer_h1,
    "SPARSE": strategy_copy_as_is,
    "UNKNOWN": strategy_copy_as_is,
}


# === Phase 5 -- Affichage rich ===

def display_diagnosis(stats: HierarchyStats, md_path: Path):
    """Affiche le diagnostic avec rich."""
    color_map = {
        "OK": "green",
        "FLAT": "yellow",
        "MISSING_H1": "yellow",
        "SPARSE": "red",
        "UNKNOWN": "red",
    }
    color = color_map.get(stats.diagnosis, "white")

    max_count = max(stats.h1, stats.h2, stats.h3, stats.h4, 1)
    def bar(n):
        width = 20
        filled = int(n / max_count * width)
        return "#" * filled + "." * (width - filled)

    body = f"""[bold]Fichier[/]  : {md_path.name}
[bold]Lignes[/]   : {stats.total_lines:,}
[bold]Titres[/]   : {stats.total_titles}

H1 : [cyan]{stats.h1:>4}[/]  {bar(stats.h1)}
H2 : [cyan]{stats.h2:>4}[/]  {bar(stats.h2)}
H3 : [cyan]{stats.h3:>4}[/]  {bar(stats.h3)}
H4 : [cyan]{stats.h4:>4}[/]  {bar(stats.h4)}

[bold]Diagnostic[/] : [{color}]{stats.diagnosis}[/]
[dim]-> {stats.diagnosis_message}[/]"""

    console.print(Panel(body, title="[Diagnostic hierarchie]",
                        border_style=color, padding=(1, 2)))


def display_comparison(before: HierarchyStats, after: HierarchyStats, strategy_name: str):
    """Affiche le comparatif avant/apres."""
    table = Table(title=f"[Resultat final] (strategie : {strategy_name})",
                  show_header=True, header_style="bold cyan")
    table.add_column("Niveau", style="cyan")
    table.add_column("Avant", justify="right")
    table.add_column("Apres", justify="right", style="bold")
    table.add_column("Delta", justify="right")

    for level in ['h1', 'h2', 'h3', 'h4']:
        b = getattr(before, level)
        a = getattr(after, level)
        delta = a - b
        delta_str = f"[green]+{delta}[/]" if delta > 0 else (
            f"[red]{delta}[/]" if delta < 0 else "[dim]0[/]"
        )
        table.add_row(level.upper(), str(b), str(a), delta_str)

    table.add_row("Total",
                  str(before.total_titles),
                  str(after.total_titles),
                  "")
    console.print(table)


def display_samples(output_path: Path, num: int = 5):
    """Affiche un echantillon de titres par niveau."""
    lines = output_path.read_text(encoding="utf-8").splitlines()

    samples = {1: [], 2: [], 3: [], 4: []}
    for line in lines:
        for level in [1, 2, 3, 4]:
            prefix = "#" * level + " "
            if line.startswith(prefix) and not line.startswith("#" * (level + 1) + " "):
                if len(samples[level]) < num:
                    samples[level].append(line[len(prefix):].strip())
                break

    for level in [1, 2, 3, 4]:
        if samples[level]:
            console.print(f"\n[bold]H{level} (premiers {len(samples[level])}):[/]")
            for s in samples[level]:
                console.print(f"  - {s}")


# === Phase 6 -- CLI typer ===

@app.command()
def main(
    input_path: str = typer.Argument(..., help="Markdown source"),
    output_path: str = typer.Argument(None, help="Markdown reconstruit (defaut: {input}_reconstructed.md)"),
    force_strategy: str = typer.Option(None, "--force-strategy",
                                        help="Forcer une strategie : OK | FLAT | MISSING_H1"),
    show_samples: bool = typer.Option(True, "--show-samples/--no-samples",
                                       help="Afficher des echantillons par niveau"),
):
    """Reconstruit la hierarchie d'un markdown extrait par docling, adaptatif."""
    src = Path(input_path)
    if not src.exists():
        console.print(f"[red]X Fichier introuvable : {src}[/]")
        raise typer.Exit(1)

    if output_path is None:
        out = src.parent / f"{src.stem}_reconstructed{src.suffix}"
    else:
        out = Path(output_path)

    # Phase 1 -- Diagnostic
    before_stats = diagnose(src)
    display_diagnosis(before_stats, src)

    # Phase 2 -- Selection de strategie
    diagnosis = force_strategy if force_strategy else before_stats.diagnosis
    if diagnosis not in STRATEGY_MAP:
        console.print(f"[red]X Strategie inconnue : {diagnosis}[/]")
        raise typer.Exit(1)

    strategy = STRATEGY_MAP[diagnosis]
    console.print(f"\n[bold cyan]-> Strategie : {strategy.__name__}[/]")

    if before_stats.diagnosis in ("SPARSE", "UNKNOWN") and not force_strategy:
        console.print(f"[yellow]! {before_stats.diagnosis_message}[/]")
        console.print("[yellow]  Inspection manuelle recommandee[/]\n")

    # Phase 3 -- Application
    with console.status("[bold green]Reconstruction en cours..."):
        result = strategy(src, out)

    console.print(f"[green]OK[/] Sauvegarde : [bold]{out}[/]")

    # Phase 4 -- Comparaison apres
    after_stats = diagnose(out)
    display_comparison(before_stats, after_stats, strategy.__name__)

    if show_samples:
        display_samples(out)


if __name__ == "__main__":
    app()
