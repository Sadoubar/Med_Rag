"""Tests pour medrag.chunk"""

import pytest
from pathlib import Path

from medrag.chunk import (
    clean_html_entities,
    is_meta_section,
    make_prefixed_text,
    split_long_content,
    parse_markdown_into_sections,
    chunk_sections,
    META_TITLES,
)


# === Tests helpers ===

def test_clean_html_entities():
    assert clean_html_entities("a &gt; b") == "a > b"
    assert clean_html_entities("&amp;") == "&"
    assert clean_html_entities("&lt; &gt; &nbsp;") == "< >  "


def test_is_meta_section():
    assert is_meta_section(["Table des mati\u00e8res"]) is True
    assert is_meta_section(["Chapitre 1", "Anaphylaxie"]) is False
    assert is_meta_section(["Avant-propos", "Section X"]) is True


def test_make_prefixed_text():
    result = make_prefixed_text(["Ch1", "Palu", "Traitement"], "Contenu...")
    assert result.startswith("[Section] Ch1 > Palu > Traitement\n\n")
    assert "Contenu..." in result


def test_split_long_content_short():
    """Contenu sous le seuil = pas de decoupage."""
    content = "a" * 500
    result = split_long_content(content, 1000, 100)
    assert len(result) == 1
    assert result[0] == content


def test_split_long_content_long():
    """Contenu long = decoupage en sous-chunks."""
    content = "a" * 3000
    result = split_long_content(content, 1000, 100)
    assert len(result) >= 3


# === Tests parsing ===

def test_parse_markdown_basic(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("""# Chapitre 1

## Section A

Contenu de A.

### Sous-section A.1

Detail A.1.

## Section B

Contenu B.
""", encoding="utf-8")

    sections = parse_markdown_into_sections(md)
    assert len(sections) >= 3  # au minimum les 3 sections avec contenu


def test_parse_meta_filter(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("""## Table des mati\u00e8res

Liste des chapitres...

# Chapitre 1

## Anaphylaxie

Traitement urgent.
""", encoding="utf-8")

    sections = parse_markdown_into_sections(md)
    # La section "Table des matieres" est parsee mais sera filtree au chunking


# === Tests chunking ===

def test_chunk_basic(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("""# Chapitre 1 : Test

## Anaphylaxie

### Traitement

""" + ("Adrenaline IM 0,3 ml. " * 50), encoding="utf-8")

    sections = parse_markdown_into_sections(md)
    chunks, stats = chunk_sections(sections, source_name="test", min_chunk_chars=50)

    assert stats.total_chunks >= 1
    assert chunks[0].source == "test"
    assert chunks[0].chapter == "Chapitre 1 : Test"
    assert chunks[0].pathology == "Anaphylaxie"
    assert "[Section]" in chunks[0].text


def test_chunk_filters_meta(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("""## Table des mati\u00e8res

""" + ("Ligne de TOC. " * 100), encoding="utf-8")

    sections = parse_markdown_into_sections(md)
    chunks, stats = chunk_sections(sections, source_name="test")

    assert stats.total_chunks == 0
    assert stats.filtered_meta_sections >= 1


def test_chunk_filters_short(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("""# Ch1

## P1

abc

## P2

def
""", encoding="utf-8")

    sections = parse_markdown_into_sections(md)
    chunks, stats = chunk_sections(sections, source_name="test", min_chunk_chars=200)

    assert stats.total_chunks == 0
    assert stats.filtered_orphan_chunks >= 2
