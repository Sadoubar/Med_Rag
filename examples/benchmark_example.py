"""Exemple : lancer un benchmark sur des requetes medicales."""
from medrag import MedRAG

rag = MedRAG(chroma_path="./chromadb")

# Requetes de benchmark
queries = [
    "posologie paludisme enfant",
    "traitement tuberculose multiresistante",
    "cefotaxime nourrisson meningite",
    "anemie ferriprive grossesse",
    "artesunate IV paludisme grave",
    "fievre enfant moins de 5 ans",
]

# Lancer le benchmark (necessite un corpus deja indexe)
# report = rag.benchmark(queries, top_k=4)
# print(f"Requetes testees   : {report['total_queries']}")
# print(f"Avec 'both' source : {report['queries_with_both']}")
# print(f"Distribution       : {report['source_distribution']}")
