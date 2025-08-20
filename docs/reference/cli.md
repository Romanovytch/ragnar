# CLI Reference — `ragnar-ingest`

```bash
ragnar-ingest --help

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
  --no-progress
  --log-level LOG_LEVEL
```

Key options:

- --target-tokens, --overlap-tokens, --max-tokens — chunk sizing
- --embed-api-base, --embed-model — remote embedding endpoint/model
- --batch-size — embedding/upsert batch
- --drop-collection — recreate Qdrant collection
- --no-progress — hide progress bars
- --log-level — DEBUG/INFO/…

## Ingest utilitR

```
ragnar-ingest \
  --repo-path ../utilitR \
  --collection utilitr_v1 \
  --qdrant-url http://localhost:6333 \
  --embed-api-base https://api.openai.com/v1 \
  --embed-model BAAI/bge-multilingual-gemma2
```
