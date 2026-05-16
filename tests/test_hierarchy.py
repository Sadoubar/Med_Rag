"""Tests pour medrag.hierarchy"""

import pytest
from pathlib import Path

from medrag.hierarchy import (
    HierarchyStats, TitleStats,
    classify_title, diagnose, count_titles_by_level,
    parse_h2_titles, compute_title_stats,
)


# === Tests classify_title ===

def make_stat(title, count=1, content_size=500, position_ratio=0.1):
    return TitleStats(
        title=title, line_num=0, position_index=0,
        count=count, content_size=content_size,
        position_ratio=position_ratio,
    )


def test_chapter_detection():
    assert classify_title(make_stat("Chapitre 1 : Quelques symptomes")) == 1
    assert classify_title(make_stat("Annexe II : Tableaux")) == 1
    assert classify_title(make_stat("Partie III")) == 1


def test_clinical_section_by_recurrence():
    """Un titre qui revient >=5 fois doit etre en H3."""
    assert classify_title(make_stat("Traitement", count=76)) == 3
    assert classify_title(make_stat("Signes cliniques", count=87)) == 3


def test_pathology_default():
    """Un titre unique avec contenu = pathologie (H2)."""
    assert classify_title(make_stat("Anaphylaxie", count=1, content_size=2000)) == 2
    assert classify_title(make_stat("Paludisme grave", count=1, content_size=1500)) == 2


def test_step_or_numbered():
    """Etapes et numerotes = H4."""
    assert classify_title(make_stat("\u00c9tape 1 - Traitement", count=1)) == 4
    assert classify_title(make_stat("Phase 2 - Maintien", count=1)) == 4
    assert classify_title(make_stat("1) Gestion des voies", count=1)) == 4
    assert classify_title(make_stat("2. Maintien circulation", count=1)) == 4


def test_meta_page():
    """Titre en debut de doc + peu de contenu = meta-page."""
    s = make_stat("Table des matieres", count=1, content_size=50, position_ratio=0.001)
    assert classify_title(s) is None

    s = make_stat("Avant-propos", count=1, content_size=100, position_ratio=0.005)
    assert classify_title(s) is None


def test_empty_title_grouping():
    """Titre quasi vide = regroupement H2."""
    s = make_stat("Section avec peu de contenu", count=1, content_size=30)
    assert classify_title(s) == 2


def test_numbered_section_l2():
    """Section numerotee 1.1 -> H2."""
    s = make_stat("1.1 Caracteristiques de M. tuberculosis", count=1, content_size=500)
    assert classify_title(s) == 2


def test_numbered_section_l3():
    """Sous-section numerotee 1.1.1 -> H3."""
    s = make_stat("1.1.1 Primo-infection", count=1, content_size=300)
    assert classify_title(s) == 3


def test_numbered_section_l4():
    """Sous-sous-section numerotee 1.1.1.1 -> H4."""
    s = make_stat("1.1.1.1 Detail specifique", count=1, content_size=200)
    assert classify_title(s) == 4


# === Tests diagnose ===

def test_diagnose_flat(tmp_path):
    """Tout en H2 -> FLAT."""
    md = tmp_path / "flat.md"
    md.write_text("\n".join([f"## Titre {i}\nContenu" for i in range(150)]),
                  encoding="utf-8")
    stats = diagnose(md)
    assert stats.diagnosis == "FLAT"


def test_diagnose_ok(tmp_path):
    """Hierarchie multi-niveaux -> OK."""
    content = []
    for i in range(10):
        content.append(f"# Chapitre {i}")
        for j in range(25):
            content.append(f"## Section {i}.{j}")
            for k in range(6):
                content.append(f"### Sous-section {i}.{j}.{k}")
    md = tmp_path / "ok.md"
    md.write_text("\n".join(content), encoding="utf-8")
    stats = diagnose(md)
    assert stats.diagnosis == "OK"


def test_diagnose_sparse(tmp_path):
    """Tres peu de titres -> SPARSE."""
    md = tmp_path / "sparse.md"
    md.write_text("# Titre 1\n## A\n## B", encoding="utf-8")
    stats = diagnose(md)
    assert stats.diagnosis == "SPARSE"


def test_diagnose_missing_h1(tmp_path):
    """H2 et H3 mais pas de H1 -> MISSING_H1."""
    content = []
    for i in range(15):
        content.append(f"## Section {i}")
        for j in range(5):
            content.append(f"### Detail {i}.{j}")
    md = tmp_path / "missing_h1.md"
    md.write_text("\n".join(content), encoding="utf-8")
    stats = diagnose(md)
    assert stats.diagnosis == "MISSING_H1"


# === Tests fonctionnels ===

def test_count_titles_by_level():
    lines = [
        "# Niveau 1",
        "## Niveau 2",
        "### Niveau 3",
        "#### Niveau 4",
        "## Niveau 2 bis",
        "Contenu normal",
    ]
    counts = count_titles_by_level(lines)
    assert counts == {'h1': 1, 'h2': 2, 'h3': 1, 'h4': 1}


def test_parse_h2_titles():
    lines = [
        "# Pas H2",
        "## Premier H2",
        "Contenu",
        "## Deuxieme H2",
        "### Pas H2 non plus",
    ]
    titles = parse_h2_titles(lines)
    assert titles == [(1, "Premier H2"), (3, "Deuxieme H2")]


def test_compute_title_stats():
    lines = [
        "## A", "Contenu A long",
        "## B", "Contenu B",
        "## A", "Contenu A bis",
    ]
    titles = parse_h2_titles(lines)
    stats = compute_title_stats(lines, titles)
    assert len(stats) == 3
    assert stats[0].count == 2  # "A" apparait 2 fois
    assert stats[1].count == 1  # "B" unique
    assert stats[2].count == 2  # "A" encore
