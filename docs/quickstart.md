# Quick Start

## 1. En tant que bibliotheque Python

```python
from medrag import MedRAG

# Initialiser le pipeline
rag = MedRAG(
    chroma_path="./chromadb",
    extract_api_url="https://your-modal-api.modal.run",  # optionnel
    extract_api_key="sk_...",                              # optionnel
)

# Ajouter un PDF (extract -> hierarchy -> chunk -> index)
result = rag.add_pdf("guide_clinique.pdf")
print(f"Chunks crees : {result['chunks_created']}")

# Rechercher
results = rag.search("posologie paludisme enfant")
for r in results:
    print(f"[{r.source}] {' > '.join(r.section_path)}")
    print(f"  Score: {r.score:.3f}")
```

## 2. En ligne de commande (CLI)

### Reconstruire la hierarchie d'un markdown

```bash
medrag-hierarchy source.md output.md
```

### Chunker en JSONL

```bash
medrag-chunk output.md chunks.jsonl
```

### Indexer dans ChromaDB

```bash
medrag-index chunks.jsonl
medrag-index chunks_extra.jsonl --no-reset  # ajouter au corpus existant
```

### Recherche hybride

```bash
medrag-search "traitement paludisme grave"
medrag-search "cefotaxime nourrisson" --debug --top-k 5
```

## 3. Pipeline etape par etape

Si vous avez deja du markdown (pas besoin d'extraction PDF) :

```python
from medrag import MedRAG

rag = MedRAG(chroma_path="./chromadb")

# Ajouter du markdown directement
with open("guide.md", encoding="utf-8") as f:
    markdown_text = f.read()

result = rag.add_markdown(markdown_text, source_name="mon_guide")
```
