# Assistant RAG sur articles scientifiques

Assistant de recherche sur des articles consacrés aux ondes gravitationnelles. Le projet utilise des briques Python explicites : PyMuPDF pour extraire le texte, Sentence Transformers pour les embeddings locaux, ChromaDB pour la recherche vectorielle, Mistral pour la génération et FastAPI pour l'API.

## Environnement

Python 3.12 ou plus récent est recommandé. Depuis la racine du projet, crée et active un environnement virtuel :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dans VS Code, sélectionne `.venv\Scripts\python.exe` comme interpréteur. Le premier lancement de l'ingestion télécharge le modèle d'embeddings `BAAI/bge-small-en-v1.5`.

## Corpus et indexation

Pour télécharger jusqu'à 25 articles arXiv correspondant à la requête par défaut puis indexer leurs pages :

```powershell
python -m src.ingest --download --max-results 25
```

L'ingestion enregistre les PDF dans `data/pdfs/`, conserve le titre et l'identifiant arXiv dans `data/papers.json`, puis crée l'index persistant dans `chroma_db/`. Les PDF téléchargés et l'index sont ignorés par Git. Pour indexer tes propres PDF, dépose-les dans `data/pdfs/` et lance `python -m src.ingest`. Les noms de fichiers servent d'identifiants de source si aucune métadonnée arXiv n'est disponible.

Le découpage par défaut est de 1 000 caractères avec 200 caractères de recouvrement. Les options de découpage sont aussi disponibles dans `ingest_papers()` pour mener des comparaisons contrôlées.

## Questions et API

La recherche vectorielle seule est disponible avec `retrieve(question, k=5)` dans `src.retrieve`. Pour générer une réponse sourcée, configure la clé d'API sans l'inscrire dans le dépôt :

```powershell
$env:MISTRAL_API_KEY = "..."
```

Le modèle de génération utilisé par défaut est `mistral-small-latest`.

Lance ensuite l'API :

```powershell
uvicorn src.api:app --reload
```

Teste `http://127.0.0.1:8000/docs` ou `http://127.0.0.1:8000/ask?q=...`. La réponse contient le texte généré et les métadonnées de ses extraits sources. Le modèle doit suivre la consigne de citer ses affirmations avec `[n]` et de signaler explicitement les informations manquantes.

## Évaluation du retrieval

Ajoute au moins 20 questions dont tu as vérifié la réponse, avec l'identifiant arXiv attendu dans `eval/questions.jsonl` (une entrée JSON par ligne) :

```json
{"question":"Quelle est la masse finale du trou noir issu de GW150914 ?","source":"1602.03837"}
```

Lance `python -m eval.run_eval` pour afficher hit@k, son intervalle de confiance de Wilson à 95 %, et le MRR pour k = 1, 3, 5 et 10. Le fichier fourni est un exemple de départ, pas un jeu d'évaluation suffisant pour tirer des conclusions. Évalue séparément plusieurs tailles de chunks et modèles d'embedding ; note les résultats et les échecs après vérification manuelle.

## Docker

Construis d'abord l'index localement, puis crée et lance l'image depuis la racine du projet :

```powershell
docker build -t rag-app .
docker run --rm -e MISTRAL_API_KEY -p 8000:8000 rag-app
```

Ne place jamais la clé Mistral dans l'image ou dans le dépôt.
