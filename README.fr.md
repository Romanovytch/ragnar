[🇫🇷 Français](README.fr.md) | [🇬🇧 English](README.md)

# Comment utiliser RAGnaR ?

## Initialisation de l'environnement virtuel

(Optionnel) C'est une bonne pratique, en python, d'utiliser un environnement virtuel :

```shell
cd RAGnaR
python -m venv .venv
source .venv/bin/activate
```

## Installer les dépendances

RAGnaR a un `requirements.txt` listant toutes les dépendances et leur version minimale ainsi qu'un `pyproject.toml` :

```shell
pip install -U pip
pip install -e .
```

Selon votre machine et votre réseau, cette opération peut prendre quelques minutes.

## Variables d'environnement

Vous pouvez spécifier les variables d'environnement dans un fichier `.env` à la racine du projet.
Le fichier `.env.example` vous donne un exemple de ce à quoi il devrait ressembler :

```
# Embeddings
EMBED_API_BASE="https://my-embeddings-model-url/v1"
EMBED_API_KEY="api-key"
EMBED_MODEL="model-name"

# Qdrant
QDRANT_API_KEY="api-key"
QDRANT_URL="http://qdrant:6333"
```

## Données

En l'état, RAGnaR ne supporte que les documents markdown dans un dossier local. Il faut donc cloner le projet à côté :

```shell
cd ..
git clone https://github.com/InseeFrLab/utilitR.git
```

## Utilisation

Pour lancer l'ingestion des données, vous pouvez utiliser les paramètres listés par `ragnar-ingest --help` :

```
usage: Ingest utilitR into Qdrant (remote embeddings) [-h] --repo-path REPO_PATH --collection COLLECTION [--qdrant-url QDRANT_URL] [--qdrant-api-key QDRANT_API_KEY] [--drop-collection]
                                                      [--target-tokens TARGET_TOKENS] [--overlap-tokens OVERLAP_TOKENS] [--max-tokens MAX_TOKENS] [--batch-size BATCH_SIZE] [--embed-api-base EMBED_API_BASE]
                                                      [--embed-api-key EMBED_API_KEY] [--embed-model EMBED_MODEL] [--embed-timeout EMBED_TIMEOUT] [--embed-ca-bundle EMBED_CA_BUNDLE] [--embed-insecure]
                                                      [--no-progress] [--log-level LOG_LEVEL]

options:
  -h, --help            show this help message and exit
  --repo-path REPO_PATH
  --collection COLLECTION
  --qdrant-url QDRANT_URL
  --qdrant-api-key QDRANT_API_KEY
  --drop-collection
  --target-tokens TARGET_TOKENS
  --overlap-tokens OVERLAP_TOKENS
  --max-tokens MAX_TOKENS
  --batch-size BATCH_SIZE
  --embed-api-base EMBED_API_BASE
  --embed-api-key EMBED_API_KEY
  --embed-model EMBED_MODEL
  --embed-timeout EMBED_TIMEOUT
  --embed-ca-bundle EMBED_CA_BUNDLE
  --embed-insecure
  --no-progress         Disable progress bars
  --log-level LOG_LEVEL
                        DEBUG, INFO, WARNING, ERROR
```

`--embed-api-base`, `--embed-api-key`, `--embed-model`, `--qdrant-url`, `--qdrant-api-key` n'ont pas besoin d'être spécifiés s'ils sont définis dans le `.env`.

Par défaut :
* `--target-tokens` = 800
* `--max-tokens` = 1200
* `--overlap-tokens` = 120

Exemple d'utilisation :
```shell
# Lit tous les fichiers markdown (.md) du repo et crée une collection Qdrant "utilitr_v1"
ragnar-ingest --repo-path ../utilitR --collection utilitr_v1
```

