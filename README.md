[🇫🇷 Français](README.fr.md) | [🇬🇧 English](README.md)

# AgoRa

**AgoRa** is a document ingestion pipeline for **Retrieval-Augmented Generation (RAG)**.

It ships command-line tools to:

- 🧩 Configure **what to ingest** (sources, paths, filters)
- ✂️ Control **how to chunk** documents (target / overlap / max tokens)
- 🔤 **Embed** chunks via an **OpenAI-compatible embeddings endpoint**
- 📦 **Store** vectors + payload in **Qdrant**

Pair it with **[CanaR](https://github.com/Romanovytch/canar)** :duck: , a chat UI that queries Qdrant to deliver RAG answers with citations.

AgoRa is designed to be run as a **CLI** so ingestion can be automated: for example, trigger an ingestion job when a repo/site changes (CI pipeline, cron, Kubernetes Job) to **upsert** or **replace** a Qdrant collection.

> :construction: AgoRa is currently in 0.x and supports Markdown files only *(most online `books` are actually markdown)*.
> Until PDF/HTML/DOCX support lands for 1.0, we recommend converting documents to Markdown (e.g., with an online converter or GenAI) before ingestion.

---

## Ecosystem (AgoRa + CanaR)

- **AgoRa**: ingestion (sources → chunking → embeddings → Qdrant payload)
- **CanaR**: chat UI + assistants + retrieval + citations + conversation history

If you’re looking for the “entrypoint” of the ecosystem (installation guide, contribution workflow, templates), have a look at **CanaR** repository first:
- Repo: https://github.com/Romanovytch/canar
- Contributing: https://github.com/Romanovytch/canar/blob/main/CONTRIBUTING.md

CanaR repository has a docker compose to quickly deploy the required Qdrant instance you need for AgoRa.

> :warning: You don't HAVE to pair AgoRa with CanaR (or the other way around). Both can be suitable for other products if you already have an ingestion tool or a chatbot UI.

---

## Ingestion payload contract

AgoRa writes points to Qdrant with a **stable payload schema** consumed by CanaR.

**This contract is a core compatibility surface.** Changes to payload keys should be treated as breaking changes and must be coordinated with CanaR.

The payload includes (at least):

- `source` (source identifier e.g document name)
- `doc_id` (stable document identifier)
- `path` (relative path inside repo / source)
- `url` (citation URL (if any), built from `base_url` + breadcrumbs/anchors when possible)
- `breadcrumbs` (chapter/section hierarchy)
- `content` (chunk text)
- `token_count` (or similar, depending on tokenizer)
- `chunk_id`, `chunk_index` (stable within a document)
- optional: `repo_url` (link to file at commit), language metadata, etc.

> The exact schema should live in a dedicated doc file soon (e.g. `docs/payload-contract.md`) and be covered by golden tests.

---

## Features

- **Config-driven sources** via `sources.yaml`
- **Code-aware chunker** 
- OpenAI-compatible **remote embeddings** (e.g., API endpoint, vLLM)
- **Qdrant** vector store
- Rich payload: breadcrumbs, chapter/section, token counts, URLs
- **Validation** for configs with helpful errors

## Requirements

- **Python ≥ 3.10**
- A running **Qdrant** instance
- A running **embeddings** endpoint (OpenAI-compatible), e.g. OpenAI's `text-embedding-3-large/small`, vLLM-served `bge-multilingual-gemma2`, etc...

---

## Quickstart (local)

### Makefile (optional but recommended)

A minimal `Makefile` is available (aligned with CanaR):

```text
Targets:
  make venv          - Create venv (.venv)
  make install       - Install AgoRa (editable) + dev tools
  make test          - Run tests (pytest)
  make lint          - Run ruff lint (check)
  make format        - Auto-format with ruff
  make format-check  - Check formatting with ruff
  make ci            - Run lint + format-check + tests
  ```

### 1) Install (editable)

```shell
# optional but recommended: use a virtualenv
python -m venv .venv
source .venv/bin/activate

# from repo root
pip install -U pip
pip install -e .
```
or `make venv` & `make install`

This installs the CLI binaries: `agora-config` and `agora-ingest`.

### 2) Configure environment variables

You can pass credentials/URLs either via **CLI flags**, **environment variables**, or a **`.env` file**.

Example `.env`:
```dotenv
# Embeddings
EMBED_API_BASE=https://my-embeddings-model-url/v1
EMBED_API_KEY=api-key
EMBED_MODEL=model-name

# Qdrant
QDRANT_API_KEY=api-key
QDRANT_URL=http://qdrant:6333

# Optional synthetic LLM summaries
LLM_API_BASE=https://my-chat-model-url/v1
LLM_API_KEY=api-key
LLM_MODEL=model-name
# Optional fallback for API bases that are not recognized as OpenAI or Mistral
LLM_PROVIDER_NAME=ollama
```
> The ingest CLI uses CLI args first, then env vars, and can also load a `.env` file via `--dotenv-path` (absolute or relative).

---

## Data compatibility

Stable today:
- Markdown: `*.md`
- R Markdown: `*.Rmd`
- Quarto: `*.qmd` (including Quarto books)

Planned (roadmap):
- HTML ingestion (local files or URL scrap)
- PDF / DOCX / XLSX ingestion

Have your docs available locally (except for URL scrap), example:
```shell
cd ..
git clone https://github.com/InseeFrLab/utilitR.git
```

---

## Configure sources (`sources.yaml`)

Create a starter file:
```shell
agora-config init
```

This command creates a pre-configured `sources.yaml`. Edit the `sources:` section and add a source.
> For Markdown/Quarto repositories, use `kind: markdown_repo`.

### Source fields

- `kind` **(required)**: `markdown_repo` for `.md`, `.Rmd` or `.qmd` documents
- `repo_path` **(required)**: path to the folder with your docs (absolute or relative to this YAML file)
- `base_url` **(required)**: public docs base URL used to build citation links
- `repo_url_template` *(optional)*: template for repo links, e.g. `https://github.com/org/repo/blob/{commit}/{path}`
- `default_lang` *(optional)*: main language if different than default English
- `exclude_dirs`, `include_globs` *(optional)*
- `synthetic_llm_summary` *(optional, default `false`)*: generate LLM summaries for grouped raw chunks and store them in a separate summary collection
- `synthetic_llm_summary_group_max_tokens` *(optional, default `3000`)*: token budget used to group raw chunks before each LLM call; it must be larger than `--max-tokens` when summaries are enabled
- `synthetic_llm_summary_max_sentences` *(optional, default `3`)*: maximum number of sentences kept in each generated summary

Example config for [utilitR](https://book.utilitr.org/):
```yaml
version: 1

defaults:
  include_globs: ["**/*.md", "**/*.qmd", "**/*.Rmd"]
  exclude_dirs: [".git", "_book", "docs", ".quarto", "renv", ".github", "node_modules", "build", "dist", "site"]
  default_lang: "en"
  follow_symlinks: false

sources:
  utilitr:
    kind: markdown_repo
    repo_path: "../utilitR"
    base_url: "https://book.utilitr.org/"
    repo_url_template: "https://github.com/InseeFrLab/utilitR/blob/{commit}/{path}"
    default_lang: "fr"
    exclude_dirs: [".git", "_book", "docs", ".quarto", "renv", ".github"]
    synthetic_llm_summary: false
    # To enable summary retrieval:
    # synthetic_llm_summary: true
    # synthetic_llm_summary_group_max_tokens: 3000
    # synthetic_llm_summary_max_sentences: 3
```

When `synthetic_llm_summary` is `false`, ingestion keeps the existing raw-chunk behavior.
When it is `true`, Agora groups adjacent raw chunks from the same document, asks the
configured OpenAI-compatible chat model for a JSON `{summary, keywords}` result, embeds
the generated summaries, and writes them to `<collection>_summaries`. Summary payloads
keep the raw chunk metadata shape, add `summary` and `keywords`, and preserve
`source_chunk_ids`/`source_chunk_indexes` so retrieval can later display the original
source chunks instead of only the generated summary. If your `vector_index` includes
sparse vectors, summary points are stored with those sparse vectors too.

Validate your config YAML at any time:
```shell
agora-config validate --file config/sources.yaml
```

---

## Usage

Once your config validates, ingest one source into Qdrant:
```shell
# using .env, single source in YAML:
agora-ingest \
  --sources-config-path config/sources.yaml \
  --collection utilitr_v1 \
  --dotenv-path config/.env
```

You can also pass everything explicitly:
```shell
agora-ingest \
  --sources-config-path config/sources.yaml \
  --source utilitr \
  --collection utilitr_v1 \
  --embed-api-base https://my-embeddings.example.com/v1 \
  --embed-model BAAI/bge-multilingual-gemma2 \
  --qdrant-url http://localhost:6333 \
  --drop-collection
```

## CLI help (abridged)
```
usage: agora-ingest [-h]
                    [--sources-config-path PATH]
                    [--source NAME]
                    --collection NAME
                    [--dotenv-path PATH]
                    [--embed-api-base URL] [--embed-model ID] [--embed-api-key KEY]
                    [--llm-api-base URL] [--llm-model ID] [--llm-api-key KEY]
                    [--qdrant-url URL] [--qdrant-api-key KEY]
                    [--insecure]
                    [--target-tokens N] [--overlap-tokens N] [--max-tokens N]
                    [--batch-size N] [--drop-collection]

Ingest one source from sources.yaml into Qdrant.

If a required value is missing, the CLI will suggest:
- passing a CLI flag,
- setting an env var,
- or supplying a .env file via --dotenv-path.
```

Defaults:
- `--target-tokens` = `800`
- `--max-tokens` = `1200`
- `--overlap-tokens` = `120`
- `--batch-size` = `64`

---

## Contributing

This repository is part of the `AgoRa + CanaR` ecosystem.
- For global contribution guidelines, start here: https://github.com/Romanovytch/canar/blob/main/CONTRIBUTING.md
- Issues and PRs specific to ingestion (sources, chunkers, parsers, payload contract) are welcome here.

## License

TBD (to be confirmed by maintainers)
