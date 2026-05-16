"""
medrag.extract

Extraction PDF vers Markdown via :
- API Modal distante (docling GPU) si extract_api_url fourni
- docling local si installe (pip install medrag[docling])

Usage:
    from medrag.extract import extract_pdf
    markdown_text = extract_pdf("guide.pdf", api_url="https://...", api_key="sk_...")
"""

from pathlib import Path
from typing import Optional

from medrag.exceptions import ExtractionError


def extract_pdf(
    pdf_path: str | Path,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Extrait un PDF en markdown.

    Args:
        pdf_path: chemin du fichier PDF
        api_url: URL de l'API d'extraction Modal (optionnel)
        api_key: cle API pour l'API d'extraction (optionnel)

    Returns:
        Texte markdown extrait

    Si api_url est fourni, utilise l'API distante.
    Sinon, tente docling en local (necessite pip install medrag[docling]).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise ExtractionError(f"Fichier PDF introuvable : {pdf_path}")

    if api_url:
        return _extract_via_api(pdf_path, api_url, api_key)
    else:
        return _extract_via_docling(pdf_path)


def _extract_via_api(pdf_path: Path, api_url: str, api_key: Optional[str]) -> str:
    """Extraction via API Modal distante."""
    try:
        import requests
    except ImportError:
        raise ExtractionError(
            "Le module 'requests' est requis pour l'extraction via API. "
            "Installez-le : pip install requests"
        )

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        response = requests.post(api_url, files=files, headers=headers, timeout=300)

    if response.status_code != 200:
        raise ExtractionError(
            f"API extraction a retourne {response.status_code}: {response.text}"
        )

    data = response.json()
    if "markdown" in data:
        return data["markdown"]
    elif "result" in data:
        return data["result"]
    else:
        raise ExtractionError(f"Reponse API inattendue : {list(data.keys())}")


def _extract_via_docling(pdf_path: Path) -> str:
    """Extraction via docling local."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ExtractionError(
            "docling n'est pas installe. Deux options :\n"
            "  1. pip install medrag[docling]  (extraction locale GPU/CPU)\n"
            "  2. Fournir api_url pour utiliser l'API Modal distante"
        )

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()
