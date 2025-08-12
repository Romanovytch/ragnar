from __future__ import annotations
import os
import argparse
from pathlib import Path
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from ragnar.chunking import MarkdownChunker, Chunk
from ragnar.sources.utilitr import UtilitrSource
from ragnar.embeddings.remote import RemoteOpenAIEncoder
from ragnar.vectorstores.qdrant_store import ensure_collection_dense, upsert_dense
from ragnar.util import slugify, count_tokens, make_chunk_id


def build_chunks_for_doc(text: str, base_meta: dict, chunker: MarkdownChunker) -> list[Chunk]:
    units = chunker.parse_units(text)
    raw_chunks = chunker.chunk(units)
    out: list[Chunk] = []
    for idx, (body, heading_path) in enumerate(raw_chunks):
        headings_only = [h for _, h in heading_path if h.strip()]
        chapter = headings_only[0] if headings_only else base_meta.get("doc_title")
        section = headings_only[-1] if headings_only else base_meta.get("doc_title")
        anchor = slugify(section) if section else None
        source_url = base_meta.get("source_url")
        url_with_anchor = f"{source_url}#{anchor}" if source_url and anchor else source_url

        meta = base_meta | {
            "chapter": chapter,
            "section": section,
            "breadcrumbs": headings_only,
            "token_count": count_tokens(body),
            "url": url_with_anchor,
            "chunk_index": idx,
        }
        chunk_id = make_chunk_id(base_meta.get("file_path", ""), idx, base_meta.get("git_commit"))
        out.append(Chunk(id=chunk_id, text=body, metadata=meta))
    return out


def main():
    load_dotenv()
    ap = argparse.ArgumentParser("Ingest documents into Qdrant (remote embeddings)")
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
    ap.add_argument("--embed-model", default=os.getenv("EMBED_MODEL",
                                                       "BAAI/bge-multilingual-gemma2"))
    ap.add_argument("--embed-timeout", type=float, default=60.0)
    ap.add_argument("--embed-ca-bundle", default=os.getenv("REQUESTS_CA_BUNDLE", None))
    ap.add_argument("--embed-insecure", action="store_true")

    args = ap.parse_args()
    if not args.embed_api_base:
        raise SystemExit("Set --embed-api-base or EMBED_API_BASE")

    repo = Path(args.repo_path).expanduser().resolve()
    chunker = MarkdownChunker(args.target_tokens, args.overlap_tokens, args.max_tokens)
    source = UtilitrSource(repo)

    docs = 0
    all_chunks: list[Chunk] = []
    for doc in source.iter_docs():
        all_chunks.extend(build_chunks_for_doc(doc.text, doc.metadata, chunker))
        docs += 1
    print(f"Discovered {docs} documents, produced {len(all_chunks)} chunks.")

    encoder = RemoteOpenAIEncoder(
        api_base=args.embed_api_base, model=args.embed_model,
        api_key=args.embed_api_key, timeout=args.embed_timeout,
        ca_bundle=args.embed_ca_bundle, insecure=args.embed_insecure
    )

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key or None)
    ensure_collection_dense(client, args.collection, dim=encoder.dim, drop=args.drop_collection)

    # batch in slices to avoid huge payloads
    for i in range(0, len(all_chunks), args.batch_size):
        batch = all_chunks[i:i+args.batch_size]
        vecs = encoder.encode([c.text for c in batch], batch_size=args.batch_size)
        upsert_dense(client, args.collection, batch, vecs)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
