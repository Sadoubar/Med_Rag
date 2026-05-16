"""Tests pour medrag.client"""

from medrag import MedRAG, __version__, MedRAGError


def test_version():
    assert __version__ == "0.1.0"


def test_medrag_instantiation():
    rag = MedRAG(chroma_path="./test_chroma")
    assert rag is not None
    assert rag.chroma_path == "./test_chroma"


def test_medrag_default_config():
    rag = MedRAG()
    assert rag.collection_name == "medrag_corpus"
    assert rag.embedding_model == "intfloat/multilingual-e5-base"
    assert rag.extract_api_url is None


def test_medrag_custom_config():
    rag = MedRAG(
        chroma_path="/tmp/test",
        collection_name="test_collection",
        extract_api_url="https://example.com/api",
        extract_api_key="test_key",
    )
    assert rag.chroma_path == "/tmp/test"
    assert rag.collection_name == "test_collection"
    assert rag.extract_api_url == "https://example.com/api"
    assert rag.extract_api_key == "test_key"
