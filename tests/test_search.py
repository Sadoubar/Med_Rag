"""Tests pour medrag.search"""

import pytest

from medrag.search import tokenize_fr, rrf_fusion, DEFAULT_RRF_K


def test_tokenize_fr_basic():
    tokens = tokenize_fr("Le traitement du paludisme grave")
    assert "traitement" in tokens
    assert "paludisme" in tokens
    assert "grave" in tokens
    assert "le" not in tokens  # stop word
    assert "du" not in tokens  # stop word


def test_tokenize_fr_accents():
    tokens = tokenize_fr("L'epinephrine est administree")
    assert "epinephrine" in tokens
    assert "administree" in tokens


def test_tokenize_fr_dci_with_hyphen():
    """Les DCI composes avec trait d'union doivent etre preserves."""
    tokens = tokenize_fr("artemether-lumefantrine pour enfant")
    # Soit comme un seul token, soit deux tokens
    assert "artemether" in tokens or "artemether-lumefantrine" in tokens


def test_tokenize_fr_lowercase():
    tokens = tokenize_fr("PALUDISME Grave Anaphylaxie")
    assert "paludisme" in tokens
    assert "grave" in tokens
    assert "anaphylaxie" in tokens


def test_rrf_fusion_basic():
    """Test de base : un doc dans BM25 et Dense doit gagner."""
    bm25 = [("doc_A", 0.9), ("doc_B", 0.5), ("doc_C", 0.3)]
    dense = [("doc_A", 0.85), ("doc_D", 0.80), ("doc_B", 0.70)]

    result = rrf_fusion(bm25, dense, k=60)

    # doc_A est top 1 dans les deux -> doit etre premier
    assert result[0][0] == "doc_A"
    assert result[0][2] == "both"


def test_rrf_fusion_source_markers():
    """Verifie les marqueurs de source (both/bm25/dense)."""
    bm25 = [("A", 1.0), ("B", 0.5)]
    dense = [("A", 0.9), ("C", 0.8)]

    result = rrf_fusion(bm25, dense)
    sources = {chunk_id: source for chunk_id, _, source in result}

    assert sources["A"] == "both"
    assert sources["B"] == "bm25"
    assert sources["C"] == "dense"


def test_rrf_fusion_ranks_matter():
    """Plus le rang est haut, plus le score RRF est grand."""
    # Doc en position 1 vs doc en position 10
    bm25 = [("first", 1.0)] + [(f"d{i}", 0.5) for i in range(9)]
    dense = []

    result = rrf_fusion(bm25, dense, k=DEFAULT_RRF_K)
    first_score = next(s for id_, s, _ in result if id_ == "first")
    d8_score = next(s for id_, s, _ in result if id_ == "d8")
    assert first_score > d8_score
