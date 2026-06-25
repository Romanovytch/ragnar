from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from tqdm import tqdm

from agora.chunking import Chunk, MarkdownChunker
from agora.embeddings.named import ConfiguredNamedVectorEncoder
from agora.sources.loader import load_sources_config, resolve_sources_config
from agora.sources.registry import build_source
from agora.util import count_tokens, make_chunk_id, make_parent_chunk_id, slugify
from agora.vectorstores.qdrant_store import (
    ensure_collection,
    ensure_payload_collection,
    upsert_named,
    upsert_payload_points,
    validate_named_vectors,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agora-ingest",
        description="Ingest one source from sources.yaml into Qdrant",
    )
    p.add_argument(
        "--sources-config-path",
        default="sources.yaml",
        help="Path to sources.yaml (default: ./sources.yaml)",
    )
    p.add_argument(
        "--source",
        help="Source name under `sources:`. "
        "If omitted and YAML has exactly one source, that one is used.",
    )
    p.add_argument("--collection", required=True, help="Qdrant collection name")
    p.add_argument("--qdrant-url", help="Qdrant endpoint, e.g. http://qdrant:6333")
    p.add_argument("--qdrant-api-key", help="Qdrant API key (optional)")
    p.add_argument("--drop-collection", action="store_true", help="Drop & recreate the collection")
    p.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    p.add_argument(
        "--qdrant-batch-size",
        type=int,
        default=16,
        help="Qdrant point upload batch size",
    )
    p.add_argument("--dotenv-path", help="Path to a .env file to load before resolving env vars")

    # Embeddings (remote OpenAI-compatible)
    p.add_argument("--embed-api-base", help="Embeddings API base, e.g. https://vllm.example/v1")
    p.add_argument("--embed-model", help="Embedding model id")
    p.add_argument("--embed-api-key", default="", help="Embeddings API key (optional)")
    p.add_argument(
        "--insecure", action="store_true", help="Skip TLS verify for embeddings (dev only)"
    )

    # Chunking knobs (can later be driven by YAML policy)
    p.add_argument("--target-tokens", type=int, default=800)
    p.add_argument("--overlap-tokens", type=int, default=120)
    p.add_argument("--max-tokens", type=int, default=1200)

    return p


def _preflight_qdrant(url: str, api_key: str) -> QdrantClient:
    client = QdrantClient(url=url, api_key=api_key)
    try:
        # Light call to verify connectivity & auth
        client.get_collections()
        return client
    except Exception as e:
        hint = (
            f"[!] Could not connect to Qdrant at {url}\n"
            f"    Error: {e.__class__.__name__}: {e}\n\n"
            "    Checks:\n"
            "      • Is Qdrant running & reachable from this machine?\n"
            "      • Does the hostname resolve here?\n"
            "      • URL scheme/port correct? (http://host:6333)\n"
            "      • If using HTTP, consider omitting QDRANT_API_KEY\n"
            "    Quick test:\n"
            f"      curl -s {url}/collections | head -c 200\n"
        )
        raise SystemExit(hint) from e


def _resolve_required(name: str, cli_val: str | None, env_var: str) -> str:
    """Return CLI value or env var; if both empty, exit with a clear message."""
    val = cli_val or os.getenv(env_var)
    if not val:
        dash = name.replace("_", "-")
        raise SystemExit(
            f"[!] Missing required '{name}'. Provide --{dash}, or set {env_var}, "
            f"or pass a .env file via --dotenv-path."
        )
    return val


def _resolve_optional(cli_val: str | None, env_var: str, default: str = "") -> str:
    """Return CLI value or env var; if both empty, return default."""
    return cli_val or os.getenv(env_var) or default


def _pick_single_source(resolved: dict, requested: str | None) -> tuple[str, object]:
    """Return (name, typed_config) for exactly one source.

    If `requested` is provided, ensure it exists.
    If not provided:
      - if YAML contains exactly one source, pick it
      - else, raise with a helpful list
    """
    if requested:
        if requested not in resolved:
            names = ", ".join(sorted(resolved.keys()))
            raise SystemExit(f"[!] Unknown source '{requested}'. Available: {names}")
        return requested, resolved[requested]

    if len(resolved) == 1:
        name = next(iter(resolved.keys()))
        return name, resolved[name]

    names = ", ".join(sorted(resolved.keys()))
    raise SystemExit(f"[!] Multiple sources in YAML; please specify --source. Available: {names}")


def _nice_path(p: Path) -> str:
    try:
        return str(p.relative_to(Path.cwd()))
    except Exception:
        return str(p)


def _ensure_config_exists(cfg_path: Path) -> None:
    if not cfg_path.exists():
        rel = _nice_path(cfg_path)
        raise SystemExit(
            f"\n[!] Config file not found: {rel}\n"
            "    To create a starter file:\n"
            f"      agora-config init\n"
            "    Then validate it:\n"
            f"      agora-config validate --file {rel}\n"
            "    Finally, run ingestion with:\n"
            f"      agora-ingest --sources-config-path {rel} --collection <name> ...\n"
        )


def _chunk_metadata(
    rec_metadata: dict,
    text: str,
    heading_path: list[tuple[int, str]],
    idx: int,
    index_key: str = "chunk_index",
) -> dict:
    headings = [h for _, h in heading_path if h and h.strip()]
    chapter = headings[0] if headings else rec_metadata.get("doc_title")
    section = headings[-1] if headings else rec_metadata.get("doc_title")
    anchor = slugify(section) if section else None
    base_url = rec_metadata.get("source_url")
    url = f"{base_url}#{anchor}" if base_url and anchor else base_url

    return rec_metadata | {
        "chapter": chapter,
        "section": section,
        "breadcrumbs": headings,
        "token_count": count_tokens(text),
        "url": url,
        index_key: idx,
    }


def _parent_index_for_child(
    child_start: int,
    child_end: int,
    parent_ranges: list[tuple[int, int]],
) -> int:
    best_index = 0
    best_overlap = -1
    for idx, (parent_start, parent_end) in enumerate(parent_ranges):
        overlap = max(0, min(child_end, parent_end) - max(child_start, parent_start))
        if overlap > best_overlap:
            best_index = idx
            best_overlap = overlap
    return best_index


def _validate_parent_storage_config(cfg: object, args: argparse.Namespace) -> None:
    if cfg.parent_storage_mode == "none":
        return
    if cfg.parent_target_tokens <= args.target_tokens:
        raise SystemExit(
            "[!] parent_target_tokens must be greater than --target-tokens "
            "when parent_storage_mode is enabled."
        )
    if cfg.parent_max_tokens <= args.max_tokens:
        raise SystemExit(
            "[!] parent_max_tokens must be greater than --max-tokens "
            "when parent_storage_mode is enabled."
        )


def _parent_collection_name(collection: str) -> str:
    return f"{collection}_parents"


def _drop_parent_collection_if_requested(
    client: QdrantClient,
    collection: str,
    drop: bool,
) -> None:
    if not drop:
        return
    parent_collection = _parent_collection_name(collection)
    if client.collection_exists(parent_collection):
        client.delete_collection(parent_collection)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Load users .env if provided
    if args.dotenv_path:
        env_path = Path(args.dotenv_path)
        if not env_path.exists():
            raise SystemExit(f"[!] --dotenv-path not found: {env_path}")
        load_dotenv(env_path, override=False)

    # Resolve required/optional params from CLI or env
    embed_api_base = _resolve_required("embed_api_base", args.embed_api_base, "EMBED_API_BASE")
    embed_model = _resolve_required("embed_model", args.embed_model, "EMBED_MODEL")
    qdrant_url = _resolve_required("qdrant_url", args.qdrant_url, "QDRANT_URL")

    embed_api_key = _resolve_optional(args.embed_api_key, "EMBED_API_KEY", default="")
    qdrant_api_key = _resolve_optional(args.qdrant_api_key, "QDRANT_API_KEY", default="")

    # 1) Load & validate config
    cfg_path = Path(args.sources_config_path or "sources.yaml").resolve()
    _ensure_config_exists(cfg_path)
    ingestion_config = load_sources_config(cfg_path)
    resolved = resolve_sources_config(ingestion_config)

    # 2) Choose exactly one source
    src_name, cfg = _pick_single_source(resolved, args.source)

    # 3) Instantiate adapter from registry
    adapter = build_source(src_name, cfg)

    # 4) Collect docs
    docs = list(adapter.iter_docs())
    print(f"[info] Source='{src_name}': discovered {len(docs)} documents")

    # 5) Chunk
    chunker = MarkdownChunker(args.target_tokens, args.overlap_tokens, args.max_tokens)
    _validate_parent_storage_config(cfg, args)
    parent_chunker = MarkdownChunker(
        cfg.parent_target_tokens,
        cfg.parent_overlap_tokens,
        cfg.parent_max_tokens,
    )
    parent_storage_enabled = cfg.parent_storage_mode != "none"
    chunks: list[Chunk] = []
    parent_chunks: list[Chunk] = []
    for rec in tqdm(docs, desc="Chunking", unit="doc"):
        units = chunker.parse_units(rec.text)
        parent_ids: list[str] = []
        parent_ranges: list[tuple[int, int]] = []
        if parent_storage_enabled:
            parent_spans = parent_chunker.chunk_spans(units)
            for parent_idx, parent_span in enumerate(parent_spans):
                parent_id = make_parent_chunk_id(
                    rec.metadata.get("file_path", ""),
                    parent_idx,
                    rec.metadata.get("git_commit"),
                )
                parent_meta = _chunk_metadata(
                    rec.metadata,
                    parent_span.text,
                    parent_span.heading_path,
                    parent_idx,
                    index_key="parent_chunk_index",
                )
                parent_chunks.append(
                    Chunk(id=parent_id, text=parent_span.text, metadata=parent_meta)
                )
                parent_ids.append(parent_id)
                parent_ranges.append((parent_span.start_unit, parent_span.end_unit))

        raw_chunks = chunker.chunk_spans(units)
        for idx, span in enumerate(raw_chunks):
            meta = _chunk_metadata(rec.metadata, span.text, span.heading_path, idx)
            if parent_storage_enabled and parent_ids:
                parent_idx = _parent_index_for_child(
                    span.start_unit,
                    span.end_unit,
                    parent_ranges,
                )
                meta["parent_id"] = parent_ids[parent_idx]

            cid = make_chunk_id(
                rec.metadata.get("file_path", ""),
                idx,
                rec.metadata.get("git_commit"),
            )

            chunks.append(Chunk(id=str(cid), text=span.text, metadata=meta))

    print(f"[info] Produced {len(chunks)} chunks")
    if parent_storage_enabled:
        print(f"[info] Produced {len(parent_chunks)} parent chunks")

    # 6) Encode
    enc = ConfiguredNamedVectorEncoder(
        config=ingestion_config.vector_index,
        api_base=embed_api_base,
        model=embed_model,
        api_key=embed_api_key,
        insecure=bool(args.insecure),
        default_lang=getattr(cfg, "default_lang", None),
    )
    texts = [c.text for c in chunks]
    vector_modes = ingestion_config.vector_index.vectors
    with tqdm(total=len(vector_modes), desc="Embedding", unit="mode") as pbar:
        named_vectors = enc.encode(texts, batch_size=args.batch_size)
        pbar.update(len(vector_modes))
    dimensions = enc.resolved_dimensions()
    validate_named_vectors(chunks, ingestion_config.vector_index, named_vectors, dimensions)

    # 7) Upsert
    client = _preflight_qdrant(qdrant_url, qdrant_api_key)
    ensure_collection(
        client,
        args.collection,
        ingestion_config.vector_index,
        dimensions=dimensions,
        drop=args.drop_collection,
    )
    _drop_parent_collection_if_requested(client, args.collection, args.drop_collection)
    if parent_storage_enabled:
        parent_collection = _parent_collection_name(args.collection)
        ensure_payload_collection(client, parent_collection)
        upsert_payload_points(
            client,
            parent_collection,
            parent_chunks,
            batch_size=args.qdrant_batch_size,
        )
    upsert_named(
        client,
        args.collection,
        chunks,
        ingestion_config.vector_index,
        named_vectors,
        dimensions,
        batch_size=args.qdrant_batch_size,
    )

    print(f"[ok] Ingested {len(chunks)} chunks into '{args.collection}' (source={src_name})")
    if parent_storage_enabled:
        print(f"[ok] Ingested {len(parent_chunks)} parent chunks into '{parent_collection}'")


if __name__ == "__main__":
    main()
