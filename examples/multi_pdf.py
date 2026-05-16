"""Exemple : indexer plusieurs PDFs et chercher dans le corpus unifie."""
import os
from dotenv import load_dotenv
from medrag import MedRAG

load_dotenv()

rag = MedRAG(
    chroma_path="./chromadb",
    collection_name="multi_guides",
    extract_api_url=os.getenv("EXTRACT_API_URL"),
    extract_api_key=os.getenv("EXTRACT_API_KEY"),
)

# Ajouter plusieurs guides
# rag.add_pdf("data/msf_guide_clinique.pdf", source_name="msf_clinique")
# rag.add_pdf("data/msf_tuberculose_2025.pdf", source_name="msf_tuberculose")

# Recherche dans le corpus complet
# results = rag.search("traitement tuberculose multirresistante", top_k=5)
# for r in results:
#     print(f"[{r.source}] {r.chapter} > {r.pathology}")
#     print(f"  RRF: {r.score:.4f}  Cosine: {r.cosine_sim or 'N/A'}")
#     print()
