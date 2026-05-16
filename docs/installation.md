# Installation

## Depuis les sources (mode developpement)

```bash
git clone https://github.com/Sadoubar/Med_Rag.git
cd Med_Rag
pip install -e ".[dev]"
```

## Dependances principales

- **chromadb** : base vectorielle persistante
- **sentence-transformers** : embeddings multilingual-e5-base (768 dim)
- **rank-bm25** : recherche lexicale BM25
- **rich** : affichage terminal
- **typer** : CLI

## Option : extraction PDF locale

Pour extraire des PDFs en local avec docling (GPU recommande) :

```bash
pip install -e ".[docling]"
```

Sans docling, vous pouvez utiliser l'API Modal distante en fournissant `extract_api_url`.
