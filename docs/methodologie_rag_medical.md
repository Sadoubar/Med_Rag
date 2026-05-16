# Methodologie RAG pour corpus medical francophone

> Document pedagogique -- Pipeline medrag v0.1.0
> Auteur : Sadou BARRY

---

## Sommaire

1. [Pourquoi le RAG ?](#1-pourquoi-le-rag)
2. [Vue d'ensemble du pipeline](#2-vue-densemble-du-pipeline)
3. [Etape 1 : Extraction PDF vers Markdown](#3-etape-1--extraction-pdf-vers-markdown)
4. [Etape 2 : Reconstruction de la hierarchie](#4-etape-2--reconstruction-de-la-hierarchie)
5. [Etape 3 : Chunking enrichi](#5-etape-3--chunking-enrichi)
6. [Etape 4 : Embeddings et indexation vectorielle](#6-etape-4--embeddings-et-indexation-vectorielle)
7. [Etape 5 : Recherche hybride BM25 + Dense + RRF](#7-etape-5--recherche-hybride-bm25--dense--rrf)
8. [Choix techniques et justifications](#8-choix-techniques-et-justifications)
9. [Limites et perspectives](#9-limites-et-perspectives)

---

## 1. Pourquoi le RAG ?

### Le probleme

Les grands modeles de langage (LLM) comme GPT-4, Claude ou Mistral sont puissants, mais :

- **Ils hallucinent** : ils inventent des posologies, des contre-indications, des protocoles qui n'existent pas.
- **Leurs connaissances sont figees** : un LLM entraine en 2024 ne connait pas les recommandations OMS/MSF 2025.
- **Ils ne citent pas leurs sources** : impossible de verifier d'ou vient une information.

En medecine rurale, une hallucination peut tuer. Un medecin isole au Sahel qui demande "posologie artesunate IV enfant 12 kg" a besoin d'une reponse **exacte, sourcee, verifiable**.

### La solution : Retrieval-Augmented Generation (RAG)

Le RAG inverse le paradigme :

```
Approche classique :  Question --> LLM --> Reponse (potentiellement hallucinee)

Approche RAG :        Question --> Recherche dans les guides --> Passages pertinents
                                                                        |
                                                                        v
                                                              LLM + contexte --> Reponse sourcee
```

Au lieu de faire confiance a la memoire du LLM, on va **chercher les passages pertinents** dans les guides medicaux officiels (MSF, OMS), puis on les donne au LLM comme contexte. Le LLM synthetise, mais la source reste tracable.

### Pourquoi c'est critique en contexte medical

| Critere | Sans RAG | Avec RAG |
|---------|----------|----------|
| Exactitude posologie | Approximative | Copie du guide officiel |
| Source verifiable | Non | Oui (chapitre, section, page) |
| Mise a jour | Re-entrainer le LLM | Ajouter le nouveau PDF |
| Cout | Enorme (fine-tuning) | Faible (indexation locale) |
| Offline possible | Non | Oui (ChromaDB local) |

---

## 2. Vue d'ensemble du pipeline

```
    PDF medical (MSF, OMS, etc.)
           |
           v
  [1. EXTRACTION]  docling GPU  ------>  Markdown brut
           |
           v
  [2. HIERARCHIE]  analyse statistique -->  Markdown structure (H1/H2/H3/H4)
           |
           v
  [3. CHUNKING]   decoupage intelligent -->  chunks JSONL avec contexte
           |
           v
  [4. INDEXATION]  embeddings e5-base  ---->  ChromaDB (vecteurs 768-dim)
           |                                      |
           v                                      v
  [5. RECHERCHE]  BM25 (lexical) + Dense (semantique) + RRF (fusion)
           |
           v
      Top-K passages pertinents + scores + sections
```

Chaque etape resout un probleme specifique. Supprimer une etape degrade la qualite finale.

---

## 3. Etape 1 : Extraction PDF vers Markdown

### Le probleme

Les guides medicaux sont distribues en PDF. Un PDF est un format de **presentation visuelle**, pas un format de **donnees structurees**. Il contient :

- Des tableaux de posologie avec des colonnes fusionnees
- Des encadres "Attention" avec du texte dans des boites
- Des figures, diagrammes, schemas
- Une mise en page multi-colonne
- Des en-tetes/pieds de page repetitifs

Un simple copier-coller donne un texte inutilisable :

```
Paludisme graveChapitre 7Traitement    Artesunate IV/IM :
• Enfant < 20 kg : 3 mg/kg/dose• Adulte : 2,4 mg/kg/dose
```

### La solution : docling sur GPU

**docling** (IBM) est un convertisseur PDF vers Markdown qui utilise des modeles de deep learning pour comprendre la structure du document :

- Detection de la mise en page (layout analysis)
- OCR si necessaire
- Reconstruction des tableaux
- Identification des titres et hierarchie

On le deploie sur une **GPU T4** via Modal pour avoir la vitesse necessaire (un guide MSF de 400 pages en ~3 min au lieu de 30 min sur CPU).

### Resultat

```markdown
## Paludisme grave

### Traitement

Artesunate IV/IM :
- Enfant < 20 kg : 3 mg/kg/dose
- Adulte : 2,4 mg/kg/dose
```

Le texte est propre, structure, et les tableaux sont reconvertis en Markdown.

### Architecture de l'API

```
Colab/Client  ----HTTP POST----> Modal (GPU T4)
   |                                  |
   | fichier PDF                      | docling.convert()
   |                                  |
   v                                  v
  Reponse JSON <----HTTP 200---- { markdown, pages, hierarchy }
```

---

## 4. Etape 2 : Reconstruction de la hierarchie

### Le probleme

Meme avec docling, la hierarchie est souvent **plate**. Un guide MSF de 1153 titres peut avoir **tous ses titres au niveau `##`** (H2) :

```markdown
## Guide clinique et therapeutique      <-- meta-page
## Table des matieres                   <-- meta-page
## Chapitre 1 : Quelques symptomes      <-- devrait etre H1
## Anemie                                <-- devrait etre H2 (pathologie)
## Signes cliniques                      <-- devrait etre H3 (sous-section)
## Traitement                            <-- devrait etre H3 (sous-section)
## Paludisme                             <-- devrait etre H2 (pathologie)
## Signes cliniques                      <-- devrait etre H3 (recurrent !)
## Etape 1 - Traitement initial          <-- devrait etre H4
```

Sans hierarchie, le chunking produit des blocs sans contexte. Le RAG ne sait pas que "Traitement" appartient a "Anemie" du "Chapitre 1".

### La solution : classification statistique adaptative

On analyse **les statistiques du document** pour deviner le bon niveau de chaque titre :

#### Diagnostic automatique

| Diagnostic | Condition | Action |
|------------|-----------|--------|
| `OK` | H1 >= 5, H2 >= 20, H3 >= 50 | Garder tel quel |
| `FLAT` | H1 = 0, H3 = 0, H2 > 100 | Reconstruction complete |
| `MISSING_H1` | H1 = 0, H2 >= 10, H3 >= 30 | Inferer les H1 |
| `SPARSE` | Total < 20 titres | Inspection manuelle |

#### Regles de classification (mode FLAT)

Pour chaque titre, on applique des regles dans l'ordre de priorite :

```
Regle 1 : "Chapitre X", "Annexe II", "Partie III"  -->  H1
Regle 2 : Position < 2% du doc + peu de contenu     -->  Meta-page (exclue)
Regle 3 : "1.1 Titre", "1.1.1 Sous-titre"           -->  H2/H3/H4 (par profondeur)
Regle 4 : "Etape 1", "Phase 2", "1) ..."            -->  H4
Regle 5 : Titre qui revient >= 5 fois                -->  H3 (sous-section clinique)
Regle 6 : Contenu < 80 chars                         -->  H2 (regroupement)
Defaut  : Titre unique avec contenu                  -->  H2 (pathologie)
```

La **Regle 5** est la plus puissante : dans un guide clinique, les titres comme "Signes cliniques", "Traitement", "Diagnostic", "Etiologie" reviennent dizaines de fois. Ce sont des sous-sections standardisees (H3), pas des pathologies (H2).

### Resultat

Avant :
```
H1=0, H2=1153, H3=0, H4=0  (FLAT)
```

Apres :
```
H1=13, H2=712, H3=414, H4=14  (hierarchie riche)
```

---

## 5. Etape 3 : Chunking enrichi

### Le probleme

Un LLM a une fenetre de contexte limitee (4K-128K tokens). On ne peut pas lui donner un guide de 400 pages. Il faut decouper en **chunks** (morceaux) pertinents.

Mais un decoupage naif (tous les 1000 caracteres) detruit le sens :

```
Chunk 42: "...0,3 ml IM dans la face anterolaterale de la cuisse.
Pour les enfants de plus de"

Chunk 43: "30 kg : 0,5 ml IM. Renouveler apres 5 minutes
si pas d'amelioration. CORTICOIDES"
```

On a coupe en plein milieu d'une posologie. Et on ne sait pas que ca parle d'adrenaline pour l'anaphylaxie.

### La solution : chunking section-based avec contexte hierarchique

On decoupe **par section** du markdown reconstruit, et on enrichit chaque chunk avec son chemin hierarchique :

```
[Section] Chapitre 5 : Maladies infectieuses > Paludisme > Traitement > Paludisme grave

Artesunate IV/IM :
- H0, H12, H24, puis toutes les 24 heures
- Enfant < 20 kg : 3 mg/kg/dose
- Adulte et enfant >= 20 kg : 2,4 mg/kg/dose
Relais oral par ACT des que le patient peut avaler.
```

#### Parametres du chunking

| Parametre | Valeur | Raison |
|-----------|--------|--------|
| Max chunk | 1500 chars | Tient dans la fenetre du LLM |
| Target sub-chunk | 1000 chars | Equilibre contexte/precision |
| Overlap | 150 chars | Evite de couper une phrase en deux |
| Min chunk | 200 chars | Filtre les orphelins sans contenu utile |

#### Filtres appliques

1. **Meta-sections** : "Table des matieres", "Avant-propos", "Bibliographie" sont exclues -- elles n'apportent rien au RAG.
2. **Orphelins** : les chunks de moins de 200 chars sont elimines (titres seuls, pages vides).

#### Le prefixe `[Section]`

Chaque chunk commence par `[Section] Chapitre > Pathologie > Sous-section`. Ce prefixe est **encode avec le texte** dans l'embedding. Resultat : le vecteur sait que ce chunk parle de "paludisme" meme si le mot n'apparait pas dans le contenu (qui parle d'"artesunate").

### Resultat

```
Guide MSF clinique : 1350 chunks
Guide MSF TB 2025  : 1189 chunks
Total corpus       : 2539 chunks
```

---

## 6. Etape 4 : Embeddings et indexation vectorielle

### Le probleme

Quand un medecin cherche "fievre enfant moins de 5 ans", il faut trouver les chunks pertinents parmi 2539. Une recherche par mots-cles ne suffit pas : le guide dit peut-etre "hyperthermie chez le nourrisson" (meme sens, mots differents).

### La solution : embeddings semantiques

Un **embedding** est un vecteur numerique (768 nombres) qui represente le **sens** d'un texte :

```
"paludisme grave enfant"     --> [0.12, -0.45, 0.87, ..., 0.33]  (768 dim)
"malaria severe pediatrique"  --> [0.11, -0.44, 0.86, ..., 0.34]  (768 dim, tres proche !)
"recette de gateau"           --> [-0.78, 0.23, -0.11, ..., 0.92] (768 dim, tres loin)
```

Deux textes avec le meme sens ont des vecteurs proches (cosine similarity ~ 0.9). Deux textes sans rapport sont loin (cosine ~ 0.1).

### Le modele : intfloat/multilingual-e5-base

| Critere | Choix | Alternative envisagee |
|---------|-------|----------------------|
| Modele | multilingual-e5-base | all-MiniLM-L6-v2 |
| Dimension | 768 | 384 |
| Multilingual | Oui (francais natif) | Anglais seulement |
| Taille | ~500 MB | ~80 MB |
| Performance | Top-5 MTEB multilingual | Correct en anglais |

**Pourquoi e5 ?** Les guides sont en francais medical. Un modele anglais perd en precision sur "artesunate", "cefotaxime", "anemie ferriprive". Le modele e5 de Microsoft/intfloat est entraine sur 100+ langues et comprend le vocabulaire medical francophone.

#### Le prefixe obligatoire e5

Le modele e5 exige un prefixe :
- `"passage: "` pour les documents (chunks indexes)
- `"query: "` pour les requetes (questions de l'utilisateur)

C'est une exigence du pre-entrainement. Sans ce prefixe, les embeddings perdent ~15% de precision.

### ChromaDB : la base vectorielle

**ChromaDB** stocke les vecteurs et permet de chercher les plus proches :

```
Requete : "traitement paludisme enfant"
    |
    v
Encode en vecteur (768 dim)
    |
    v
ChromaDB.query(vecteur, n_results=20)
    |
    v
Top 20 chunks les plus proches (par cosine similarity)
```

Configuration :
- **Persistant** : les vecteurs sont sauvegardes sur disque, pas besoin de re-indexer
- **HNSW** : algorithme de recherche approximative (O(log n) au lieu de O(n))
- **Cosine** : distance cosine pour la similarite (adaptee aux embeddings normalises)

### Resultat

```
Collection    : medrag_corpus
Documents     : 2539
Embedding dim : 768
Taille DB     : ~50 MB
```

---

## 7. Etape 5 : Recherche hybride BM25 + Dense + RRF

### Le probleme

La recherche dense (embeddings) est forte pour le sens, mais faible pour les mots exacts. Si le medecin cherche "cefotaxime 200mg/kg", la recherche dense peut retourner un chunk sur "amoxicilline 100mg/kg" (semantiquement proche, mais cliniquement different).

A l'inverse, une recherche lexicale (BM25) trouve exactement "cefotaxime" mais rate "cephalosporine de 3eme generation" (synonyme).

### La solution : fusion hybride

On combine les deux approches :

```
Question : "cefotaxime nourrisson meningite"
                    |
        +-----------+-----------+
        |                       |
        v                       v
   BM25 (lexical)         Dense (semantique)
   Top 20 resultats       Top 20 resultats
        |                       |
        v                       v
     Rang BM25              Rang Dense
        |                       |
        +--------> RRF <--------+
                    |
                    v
            Top 4 resultats fusionnes
            (avec source: both/bm25/dense)
```

#### BM25 : recherche lexicale

BM25 (Best Matching 25) est un algorithme classique de recherche textuelle :
- Compte les occurrences des mots de la requete dans chaque chunk
- Penalise les mots trop frequents (comme "le", "de")
- Favorise les documents courts (densite de mots pertinents)

Notre tokenizer est adapte au francais medical :
- **Conserve les accents** : "anemie" != "anémie" pour BM25
- **Conserve les traits d'union** : "artemether-lumefantrine" reste un token (nom DCI compose)
- **Stop words francais** : "le", "la", "des", "avec", "dans"... sont filtres

#### Dense : recherche semantique

Utilise les embeddings e5 et ChromaDB (voir etape 4).

#### RRF : Reciprocal Rank Fusion

RRF combine les deux classements avec une formule simple et robuste :

```
score_RRF(doc) = sum( 1 / (k + rang_i) )  pour chaque source i ou doc apparait
```

Avec `k = 60` (constante standard). Exemple :

| Document | Rang BM25 | Rang Dense | Score RRF |
|----------|-----------|------------|-----------|
| Chunk A | #1 | #1 | 1/61 + 1/61 = **0.0328** |
| Chunk B | #3 | #2 | 1/63 + 1/62 = **0.0320** |
| Chunk C | #2 | - | 1/62 = **0.0161** |
| Chunk D | - | #3 | 1/63 = **0.0159** |

Chunk A gagne car il est bien classe dans les **deux** sources. Un document present dans les deux sources ("both") est presque toujours pertinent.

### Pourquoi l'hybride est superieur

| Type de requete | BM25 seul | Dense seul | Hybride |
|-----------------|-----------|------------|---------|
| Nom exact de medicament ("cefotaxime") | Excellent | Moyen | Excellent |
| Question semantique ("fievre enfant") | Moyen | Excellent | Excellent |
| DCI + contexte ("artesunate IV grave") | Bon | Bon | Excellent |
| Synonymes ("cephalosporine C3G") | Faible | Excellent | Tres bon |

### Resultat

```
Question : "cefotaxime nourrisson meningite"

#1 [BOTH]  RRF=0.0328  Chapitre 7 > Meningite bacterienne > Traitement
#2 [BOTH]  RRF=0.0320  Chapitre 7 > Meningite > Posologies
#3 [BM25]  RRF=0.0161  Chapitre 3 > Antibiotiques > Cefotaxime
#4 [DENSE] RRF=0.0159  Chapitre 7 > Infections neonatales > Traitement
```

---

## 8. Choix techniques et justifications

### Pourquoi Markdown et pas du texte brut ?

Le Markdown preserve la **structure** : titres, listes, tableaux, gras/italique. Cette structure est essentielle pour :
- Reconstruire la hierarchie (etape 2)
- Identifier les sections pour le chunking (etape 3)
- Donner du contexte au LLM ("cette posologie est dans un tableau")

### Pourquoi pas de fine-tuning du LLM ?

| Critere | Fine-tuning | RAG |
|---------|-------------|-----|
| Cout | $1000+ par entrainement | $0 (modeles open source) |
| Mise a jour | Re-entrainer | Ajouter un PDF |
| Tracabilite | Impossible | Section + chapitre cites |
| Hallucination | Reduite mais presente | Eliminee (source directe) |
| Offline | Possible | Possible (ChromaDB local) |

### Pourquoi ChromaDB et pas FAISS / Pinecone / Qdrant ?

- **ChromaDB** est le plus simple a installer (`pip install chromadb`)
- Persistance sur disque sans serveur externe
- Suffisant pour < 100K chunks (nos guides = 2539 chunks)
- Pour un corpus plus gros (> 500K), migrer vers Qdrant ou Weaviate

### Pourquoi sentence-transformers et pas OpenAI embeddings ?

- **Gratuit** : pas de cout par requete
- **Offline** : fonctionne sans internet apres le premier telechargement
- **Francais natif** : multilingual-e5-base est meilleur en francais que text-embedding-ada-002
- **Reproductible** : memes embeddings a chaque execution

---

## 9. Limites et perspectives

### Limites actuelles

1. **Pas de generation** : medrag retourne les passages pertinents, mais ne genere pas de reponse synthetisee. L'integration avec un LLM (Claude, GPT, Mistral) est l'etape suivante.

2. **Qualite dependante de l'extraction** : si docling rate un tableau de posologie, le RAG ne le retrouvera pas.

3. **Pas de multi-modal** : les schemas, diagrammes et images du PDF sont ignores.

4. **Pas de versioning** : si un guide est mis a jour, il faut re-indexer manuellement.

### Perspectives

- **Integration LLM** : ajouter une couche de generation avec prompt template medical
- **Evaluation automatique** : benchmark sur des questions/reponses annotees par des medecins
- **Multi-langue** : le modele e5 supporte deja l'arabe, l'anglais, le portugais
- **Mode conversationnel** : historique de chat avec contexte medical cumule
- **Deploiement mobile** : version allegee pour smartphones en zone rurale (ONNX + SQLite)

---

## Annexe : Schema recapitulatif

```
+------------------+     +-------------------+     +------------------+
|                  |     |                   |     |                  |
|   PDF Medical    |---->|   docling (GPU)   |---->|   Markdown brut  |
|   (MSF, OMS)    |     |                   |     |   "## Titre..."  |
|                  |     +-------------------+     |                  |
+------------------+                               +--------+---------+
                                                            |
                                                            v
                                                   +------------------+
                                                   |  Reconstruction  |
                                                   |  hierarchique    |
                                                   |  (H1/H2/H3/H4)  |
                                                   +--------+---------+
                                                            |
                                                            v
                                                   +------------------+
                                                   |   Chunking       |
                                                   |   enrichi        |
                                                   |   [Section] ...  |
                                                   +--------+---------+
                                                            |
                              +-----------------------------+
                              |                             |
                              v                             v
                     +------------------+          +------------------+
                     |   BM25 index     |          |   ChromaDB       |
                     |   (tokenize_fr)  |          |   (e5-base 768d) |
                     +--------+---------+          +--------+---------+
                              |                             |
                              v                             v
                     +------------------+          +------------------+
                     |  Rang BM25       |          |  Rang Dense      |
                     +--------+---------+          +--------+---------+
                              |                             |
                              +-------------+---------------+
                                            |
                                            v
                                   +------------------+
                                   |   RRF Fusion     |
                                   |   (k=60)         |
                                   +--------+---------+
                                            |
                                            v
                                   +------------------+
                                   |  Top-K passages  |
                                   |  + scores        |
                                   |  + sources       |
                                   +------------------+
```

---

*Document genere pour le projet medrag v0.1.0 -- Pipeline RAG medical adaptatif pour guides cliniques francophones.*
