# medrag

Pipeline RAG medical adaptatif pour guides cliniques francophones (MSF, OMS).

## Features

- **Extraction PDF** : via API Modal (docling GPU) ou docling local
- **Reconstruction hierarchique** : classification statistique adaptative des titres plats en H1-H4
- **Chunking enrichi** : decoupage section-based avec prefixe `[Section]` pour contexte d'embedding
- **Indexation ChromaDB** : embeddings multilingual-e5-base (768 dim, cosine)
- **Recherche hybride** : BM25 + Dense + fusion RRF

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from medrag import MedRAG

rag = MedRAG(chroma_path="./chromadb")

# Ajouter un PDF au corpus
# rag.add_pdf("guide_clinique.pdf")

# Rechercher
results = rag.search("posologie paludisme enfant")
for r in results:
    print(f"[{r.source}] {' > '.join(r.section_path)}")
    print(f"  Score: {r.score:.3f}")
```

## CLI

```bash
# Reconstruire la hierarchie d'un markdown
medrag-hierarchy source.md

# Chunker en JSONL
medrag-chunk source_reconstructed.md chunks.jsonl

# Indexer dans ChromaDB
medrag-index chunks.jsonl

# Recherche hybride
medrag-search "traitement paludisme grave"
```

## License

MIT
