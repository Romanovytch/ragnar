# How to use RAGnaR

RAGnaR is a data ingestion pipeline used to prepare data for Retrieval Augmented Generation for CanaR.

## Virtual env setup

(Optional) First you should use a virtual environment it's best practice:

```shell
cd RAGnaR
python -m venv .venv
source .venv/bin/activate
```

## Download dependencies

RAGnaR has a `requirements.txt` listing all the dependencies and their minimal versions and a `pyproject.toml` as well:

```shell
pip install -U pip
pip install -e .
```
Depending on your machine and network, it can take some time.

## Environement variables

You can either add the environment variables manually or add them in a `.env` file at the root of `ragnar/`.

```
# Embedding model env vars
EMBED_API_BASE="https://my-embeddings-model-url/v1"
EMBED_API_KEY="api-key"
EMBED_MODEL="model-name"

# Qdrand env vars
QDRANT_API_KEY="api-key"
QDRANT_URL="http://qdrant:6333"
```

## Data

For now, RAGnaR is only stable for markdown documents ingestion. You will need to clone the repository containing the markdown documents locally.

```shell
cd ..
git clone https://github.com/InseeFrLab/utilitR.git
``` 

## Usage

To launch data ingestion, you can use multiple parameters that are listed by `ragnar-ingest --help`

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

`--embed-api-base`, `--embed-api-key`, `--embed-model`, `--qdrant-url`, `--qdrant-api-key` don't need to be specified if already set in `.env`.

By default :
* `--target-tokens` = 800
* `--max-tokens` = 1200
* `--overlap-tokens` = 120

Example of usage :
```shell
# Read all .md from the utilitR project and create a Qdrant collection named "utilitr_v1"
ragnar-ingest --repo-path ../utilitR --collection utilitr_v1
```