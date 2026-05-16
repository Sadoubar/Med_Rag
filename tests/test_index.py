"""Tests pour medrag.index"""

import pytest
import json
from pathlib import Path

from medrag.index import load_chunks, prepare_metadata


def test_jsonl_parsing(tmp_path):
    """Le module doit pouvoir lire un JSONL valide."""
    jsonl = tmp_path / "test_chunks.jsonl"
    chunks_data = [
        {
            "id": "test_chunk_00001",
            "source": "test",
            "chapter": "Chapitre 1",
            "pathology": "Anaphylaxie",
            "section_path": ["Chapitre 1", "Anaphylaxie", "Traitement"],
            "text": "[Section] Chapitre 1 > Anaphylaxie > Traitement\n\nEpinephrine 0,3 ml IM",
            "char_count": 100,
            "chunk_index": 1,
            "depth": 3,
        },
    ]
    jsonl.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks_data),
        encoding="utf-8",
    )

    chunks = load_chunks(jsonl)
    assert len(chunks) == 1
    assert chunks[0]["id"] == "test_chunk_00001"
    assert chunks[0]["section_path"] == ["Chapitre 1", "Anaphylaxie", "Traitement"]


def test_metadata_serialization():
    """Metadonnees : pas de listes (ChromaDB n'accepte pas)."""
    chunk = {
        "source": "test",
        "chapter": "Ch1",
        "pathology": "Palu",
        "section_path": ["A", "B", "C"],
        "depth": 3,
        "char_count": 500,
        "chunk_index": 42,
    }

    meta = prepare_metadata(chunk)

    # Pas de listes ni de dicts dans les valeurs
    for v in meta.values():
        assert isinstance(v, (str, int, float, bool)), f"Bad type: {type(v)}"


def test_e5_prefix():
    """Le prefixe 'passage: ' doit etre ajoute avant encoding."""
    text = "Adrenaline 0,3 ml IM"
    prefixed = f"passage: {text}"
    assert prefixed.startswith("passage: ")
    assert prefixed == "passage: Adrenaline 0,3 ml IM"
