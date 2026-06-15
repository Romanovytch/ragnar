# Copilot Instructions

This repository contains Agora, a Python package for a RAG ingestion CLI. Treat
`ARCHITECTURE.md` as the source of truth for the markdown ingestion pipeline.

## Scope

- Keep ingestion changes aligned with `ARCHITECTURE.md`.
- The documented ingestion flow starts at the `agora-ingest` CLI, loads
  `sources.yaml`, reads markdown-like files, chunks them, embeds chunks through a
  remote OpenAI-compatible endpoint, and upserts dense vectors into Qdrant.
- Chat, retrieval, and query behavior are outside the architecture document's

## Repository Conventions

- Main package code lives under `agora/`.
- CLI entry points live in `agora/cli/`.
- Source loading lives in `agora/sources/`.
- Markdown parsing and token-budget chunking live in `agora/chunking.py`.
- Remote embedding code lives in `agora/embeddings/remote.py`.
- Qdrant integration lives in `agora/vectorstores/qdrant_store.py`.
- Shared helpers such as token counting, frontmatter parsing, and chunk IDs live
  in `agora/util.py`.
- Tests live in `agora/tests/`.

## Ingestion Design Rules

- Preserve deterministic chunk IDs. They are UUIDv5 values derived from
  `file_path`, `chunk_index`, and `git_commit`.
- Preserve the payload contract documented in `ARCHITECTURE.md`: source
  metadata, heading metadata, token counts, URLs, chunk index, and raw text.
- Keep code fences intact during chunking. Do not split them, and do not include
  them in overlap windows.
- Keep overlap paragraph-based and bounded by the configured overlap token
  budget.
- Keep token budgeting based on `target_tokens`, `max_tokens`, and the tokenizer
  helper rather than character counts.
- Keep embeddings L2-normalized before Qdrant upsert.
- Keep Qdrant collection creation compatible with the embedding dimension and
  cosine distance unless a task explicitly changes the documented architecture.
- Avoid silently adding incremental indexing, stale-point cleanup, retry logic,
  sparse vectors, local embedding fallback, quantisation, or dry-run behavior
  without also updating `ARCHITECTURE.md`.

## Configuration

- Environment and CLI inputs include `EMBED_API_BASE`, `EMBED_MODEL`,
  `EMBED_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY`.
- Source configuration is loaded from `sources.yaml` through the source loader
  and registry.
- Prefer extending the existing Pydantic source models and registry pattern when
  adding source configuration.

## Development Practices

- Follow the existing Python style: Python 3.10+, type hints where useful, and
  focused functions with minimal new abstraction.
- Activate and use locally venv located at `.venv/` (dependencies already installed)
- Keep line length compatible with the Ruff configuration in `pyproject.toml`.
- Use structured parsers already in the project for markdown, YAML, and tokens.
- Add or update focused pytest coverage when changing parsing, chunking,
  source-loading, CLI argument handling, embedding payloads, or Qdrant point
  shape.
- When changing architecture-relevant behavior, update `ARCHITECTURE.md` in the
  same change.
