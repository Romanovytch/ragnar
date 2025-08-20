"""
RAGnaR — utilitR ingestion CLI.

This script builds a dense retrieval index from the utilitR repository:
1) loads Markdown/Quarto docs (frontmatter removed) with rich metadata,
2) chunks them into model-friendly pieces (paragraphs + fenced code),
3) embeds chunks via a remote OpenAI-compatible /v1/embeddings endpoint,
4) upserts vectors + payloads into a Qdrant collection.

Typical use:
    ragnar-ingest \
        --repo-path ../utilitR \
        --collection utilitr_v1 \
        --qdrant-url http://localhost:6333 \
        --embed-api-base https://api.openai.com/v1 \
        --embed-model BAAI/bge-multilingual-gemma2 \
        --drop-collection

Environment:
- EMBED_API_BASE / EMBED_API_KEY / EMBED_MODEL (optional CLI overrides)
- QDRANT_URL / QDRANT_API_KEY (optional CLI overrides)

Outputs:
- A Qdrant collection with one point per chunk (cosine distance),
  payload includes: source, chapter/section, breadcrumbs, urls, token_count, text.
- Console stats (docs, chunks, token distribution, throughput).
"""
from __future__ import annotations
import os
import argparse
import time
import logging
from pathlib import Path
from typing import List

from tqdm import tqdm
from qdrant_client import QdrantClient
from dotenv import load_dotenv

from ragnar.chunking import MarkdownChunker, Chunk
from ragnar.sources.utilitr import UtilitrSource
from ragnar.embeddings.remote import RemoteOpenAIEncoder
from ragnar.vectorstores.qdrant_store import ensure_collection_dense, upsert_dense
from ragnar.util import slugify, count_tokens, make_chunk_id


def _setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main():
    load_dotenv()
    ap = argparse.ArgumentParser("Ingest utilitR into Qdrant (remote embeddings)")
    ap.add_argument("--repo-path", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", ""))
    ap.add_argument("--drop-collection", action="store_true")

    ap.add_argument("--target-tokens", type=int, default=800)
    ap.add_argument("--overlap-tokens", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=64)

    ap.add_argument("--embed-api-base", default=os.getenv("EMBED_API_BASE"))
    ap.add_argument("--embed-api-key", default=os.getenv("EMBED_API_KEY", ""))
    ap.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", ""))
    ap.add_argument("--embed-timeout", type=float, default=60.0)
    ap.add_argument("--embed-ca-bundle", default=os.getenv("REQUESTS_CA_BUNDLE", None))
    ap.add_argument("--embed-insecure", action="store_true")

    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    ap.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")

    args = ap.parse_args()
    _setup_logging(args.log_level)

    if not args.embed_api_base:
        raise SystemExit("Set --embed-api-base or EMBED_API_BASE")

    repo = Path(args.repo_path).expanduser().resolve()
    chunker = MarkdownChunker(args.target_tokens, args.overlap_tokens, args.max_tokens)
    source = UtilitrSource(repo)

    # -------- Stage 1: load docs & chunk with progress --------
    t0 = time.perf_counter()
    docs = list(source.iter_docs())
    logging.info("Git commit: %s", source.commit or "main")
    logging.info("Discovered %d documents — starting chunking…", len(docs))

    all_chunks: List[Chunk] = []
    total_tokens = 0
    max_tokens = 0

    for doc in tqdm(docs, disable=args.no_progress, ncols=100, desc="Chunking"):
        title = doc.metadata.get("doc_title", "")
        tqdm.write(f"⮞ {title}")
        units = chunker.parse_units(doc.text)
        raw = chunker.chunk(units)
        for idx, (body, heading_path) in enumerate(raw):
            headings = [h for _, h in heading_path if h.strip()]
            chapter = headings[0] if headings else doc.metadata.get("doc_title")
            section = headings[-1] if headings else doc.metadata.get("doc_title")
            anchor = slugify(section) if section else None
            source_url = doc.metadata.get("source_url")
            url = f"{source_url}#{anchor}" if source_url and anchor else source_url

            meta = doc.metadata | {
                "chapter": chapter,
                "section": section,
                "breadcrumbs": headings,
                "token_count": count_tokens(body),
                "url": url,
                "chunk_index": idx,
            }
            cid = make_chunk_id(doc.metadata.get("file_path", ""), idx,
                                doc.metadata.get("git_commit"))
            all_chunks.append(Chunk(id=cid, text=body, metadata=meta))

            # stats
            tk = meta["token_count"]
            total_tokens += tk
            if tk > max_tokens:
                max_tokens = tk

    t1 = time.perf_counter()
    if all_chunks:
        avg_tokens = total_tokens / len(all_chunks)
    else:
        avg_tokens = 0.0
    logging.info(
        "Chunking done: %d chunks (avg tokens/chunk=%.1f, max=%d) in %.2fs",
        len(all_chunks), avg_tokens, max_tokens, t1 - t0
    )

    # -------- Stage 2: embeddings + upsert with progress --------
    enc = RemoteOpenAIEncoder(
        api_base=args.embed_api_base, model=args.embed_model,
        api_key=args.embed_api_key, timeout=args.embed_timeout,
        ca_bundle=args.embed_ca_bundle, insecure=args.embed_insecure
    )

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key or None)

    if args.drop_collection:
        logging.info("Recreating collection %r (dim=%d)…", args.collection, enc.dim)
    else:
        logging.info("Ensuring collection %r exists (dim=%d)…", args.collection, enc.dim)
    ensure_collection_dense(client, args.collection, dim=enc.dim, drop=args.drop_collection)

    logging.info("Embedding + upserting %d chunks (batch=%d)…", len(all_chunks), args.batch_size)
    t2 = time.perf_counter()
    done = 0
    pbar = tqdm(total=len(all_chunks), disable=args.no_progress, ncols=100, desc="Embed+Upsert")

    for i in range(0, len(all_chunks), args.batch_size):
        batch = all_chunks[i: i + args.batch_size]
        # show the current doc title in the bar postfix
        first_meta = batch[0].metadata if batch else {}
        pbar.set_postfix_str((first_meta.get("doc_title") or first_meta.get("section") or "")[:48])

        vecs = enc.encode([c.text for c in batch], batch_size=args.batch_size)
        upsert_dense(client, args.collection, batch, vecs)

        done += len(batch)
        pbar.update(len(batch))

    pbar.close()
    t3 = time.perf_counter()
    logging.info(
        "Ingestion complete: %d chunks in %.2fs (%.1f chunks/s)",
        done, t3 - t2, done / max(1e-9, (t3 - t2))
    )
    logging.info("Total wall time: %.2fs", t3 - t0)


if __name__ == "__main__":
    main()
