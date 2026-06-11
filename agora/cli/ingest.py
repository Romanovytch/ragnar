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
from agora.util import count_tokens, make_chunk_id, slugify
from agora.vectorstores.qdrant_store import ensure_collection, upsert_named


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
    chunks: list[Chunk] = []
    for rec in tqdm(docs, desc="Chunking", unit="doc"):
        units = chunker.parse_units(rec.text)
        raw_chunks = chunker.chunk(units)  # list[(text, heading_path)]
        for idx, (text, heading_path) in enumerate(raw_chunks):
            headings = [h for _, h in heading_path if h and h.strip()]
            chapter = headings[0] if headings else rec.metadata.get("doc_title")
            section = headings[-1] if headings else rec.metadata.get("doc_title")
            anchor = slugify(section) if section else None
            base_url = rec.metadata.get("source_url")
            url = f"{base_url}#{anchor}" if base_url and anchor else base_url

            meta = rec.metadata | {
                "chapter": chapter,
                "section": section,
                "breadcrumbs": headings,
                "token_count": count_tokens(text),
                "url": url,
                "chunk_index": idx,
            }

            cid = make_chunk_id(
                rec.metadata.get("file_path", ""),
                idx,
                rec.metadata.get("git_commit"),
            )

            chunks.append(Chunk(id=str(cid), text=text, metadata=meta))

    print(f"[info] Produced {len(chunks)} chunks")

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

    # 7) Upsert
    client = _preflight_qdrant(qdrant_url, qdrant_api_key)
    ensure_collection(
        client,
        args.collection,
        ingestion_config.vector_index,
        dimensions=dimensions,
        drop=args.drop_collection,
    )
    upsert_named(
        client,
        args.collection,
        chunks,
        ingestion_config.vector_index,
        named_vectors,
        dimensions,
    )

    print(f"[ok] Ingested {len(chunks)} chunks into '{args.collection}' (source={src_name})")


if __name__ == "__main__":
    main()
