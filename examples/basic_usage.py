"""Exemple basique : ajouter un PDF et chercher."""
import os
from dotenv import load_dotenv
from medrag import MedRAG

load_dotenv()

rag = MedRAG(
    chroma_path="./chromadb",
    extract_api_url=os.getenv("EXTRACT_API_URL"),  # optionnel
    extract_api_key=os.getenv("EXTRACT_API_KEY"),  # optionnel
)

# Decommente pour ajouter un PDF au corpus :
# rag.add_pdf("data/guide_clinique.pdf")

# Recherche hybride
# results = rag.search("posologie paludisme enfant")
# for r in results:
#     print(f"[{r.source}] {' > '.join(r.section_path)}")
#     print(f"  Score: {r.score:.3f}")
