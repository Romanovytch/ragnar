#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG ingestion for utilitR → Qdrant (remote embeddings only)

- Scans a local clone of https://github.com/InseeFrLab/utilitR for .md/.qmd
- Parses markdown into units (paragraphs + fenced code), heading-aware
- Chunks without splitting code blocks; paragraph-only overlap
- Calls a remote OpenAI-compatible /v1/embeddings endpoint (e.g. vLLM)
- Stores normalized dense vectors + rich metadata in Qdrant

CLI example:
  export EMBED_API_BASE="https://projet-llm-vllm.user.lab.sspcloud.fr/v1"
  export EMBED_API_KEY="your-token-if-any"
  export EMBED_MODEL="BAAI/bge-multilingual-gemma2"

  python rag.py \
    --repo-path ../utilitR \
    --collection utilitr_v1 \
    --qdrant-url http://localhost:6333 \
    --drop-collection \
    --embed-api-base "$EMBED_API_BASE" \
    --embed-api-key "$EMBED_API_KEY" \
    --embed-model "$EMBED_MODEL"

Requirements:
  pip install qdrant-client markdown-it-py pyyaml numpy requests tiktoken
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
import yaml
from markdown_it import MarkdownIt
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from dotenv import load_dotenv

# -------------------------- token counting (best-effort) -------------------- #
def _get_tokenizer():
    try:
        import tiktoken  # optional, just for sizing
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        class _Fallback:
            def encode(self, text):  # crude heuristic
                return text.split()
        return _Fallback()

ENC = _get_tokenizer()

def count_tokens(text: str) -> int:
    try:
        return len(ENC.encode(text))
    except Exception:
        return max(1, len(text) // 4)

# ------------------------------ utilities ---------------------------------- #
def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_-]+", "-", s)
    s = re.sub(r"^-+|-+$", "", s)
    return s or "section"

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def extract_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            fm = text[4:end]
            body = text[end + 4 :]
            try:
                data = yaml.safe_load(fm) or {}
                return (data if isinstance(data, dict) else {}), body.lstrip()
            except Exception:
                return {}, text
    return {}, text

def git_head_commit(repo_path: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return None

# -------------------------- markdown → units → chunks ---------------------- #
@dataclass
class Unit:
    kind: str             # "para" | "code"
    text: str
    lang: Optional[str]
    heading_path: List[Tuple[int, str]]  # [(level, title), ...]

@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]

class MarkdownChunker:
    """
    Heading-aware chunker that does NOT split fenced code blocks.
    Prefers to break on paragraph boundaries; paragraph-only overlap.
    """
    def __init__(self, target_tokens=800, overlap_tokens=120, max_tokens=1200):
        self.target = target_tokens
        self.overlap = overlap_tokens
        self.max_tokens = max_tokens
        self.md = MarkdownIt("commonmark").enable("table").enable("strikethrough")

    def parse_units(self, text: str) -> List[Unit]:
        tokens = self.md.parse(text)
        units: List[Unit] = []
        heading_stack: List[Tuple[int, str]] = []
        buffer_lines: List[str] = []

        def flush_para():
            if buffer_lines:
                para_text = "\n".join(buffer_lines).strip()
                if para_text:
                    units.append(Unit(kind="para", text=para_text, lang=None, heading_path=heading_stack.copy()))
                buffer_lines.clear()

        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.type == "heading_open":
                flush_para()
                level = int(t.tag[1]) if t.tag and t.tag.startswith("h") else 1
                title = ""
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    title = tokens[i + 1].content.strip()
                # pop deeper/equal headings then push
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                i += 3  # skip inline + heading_close
                continue

            if t.type == "fence":
                flush_para()
                lang = (t.info or "").strip() or None
                code = t.content.rstrip("\n")
                fenced = f"```{lang or ''}\n{code}\n```"
                units.append(Unit(kind="code", text=fenced, lang=lang, heading_path=heading_stack.copy()))
                i += 1
                continue

            if t.type == "paragraph_open":
                j = i + 1
                lines: List[str] = []
                while j < len(tokens) and tokens[j].type != "paragraph_close":
                    if tokens[j].type == "inline":
                        lines.append(tokens[j].content)
                    j += 1
                para = "\n".join(lines).strip()
                if para:
                    buffer_lines.append(para)
                i = j + 1
                flush_para()
                continue

            # flatten lists/tables/quotes as paragraphs of text
            if t.type in {"bullet_list_open", "ordered_list_open", "blockquote_open", "table_open"}:
                depth = 1
                j = i + 1
                lines: List[str] = []
                while j < len(tokens) and depth > 0:
                    if tokens[j].type.endswith("_open"):
                        depth += 1
                    elif tokens[j].type.endswith("_close"):
                        depth -= 1
                    elif tokens[j].type == "inline":
                        lines.append(tokens[j].content)
                    j += 1
                flush_para()
                para = "\n".join(lines).strip()
                if para:
                    units.append(Unit(kind="para", text=para, lang=None, heading_path=heading_stack.copy()))
                i = j
                continue

            i += 1

        flush_para()
        return units

    def chunk(self, units: List[Unit]) -> List[Tuple[str, List[Tuple[int, str]]]]:
        chunks: List[Tuple[str, List[Tuple[int, str]]]] = []
        buf: List[str] = []
        buf_units: List[Unit] = []
        buf_tokens = 0
        last_para_for_overlap: Optional[str] = None

        def close_chunk():
            nonlocal buf, buf_units, buf_tokens, last_para_for_overlap
            if not buf:
                return
            heading_path = buf_units[-1].heading_path if buf_units else []
            body = "\n\n".join(buf).strip()
            # remember last paragraph for overlap
            last_para_for_overlap = None
            for u in reversed(buf_units):
                if u.kind == "para":
                    last_para_for_overlap = u.text
                    break
            chunks.append((body, heading_path))
            buf, buf_units, buf_tokens = [], [], 0

        for u in units:
            u_tokens = count_tokens(u.text)

            if not buf:
                # optional paragraph-only overlap at the beginning of a new chunk
                if last_para_for_overlap and u.kind == "para":
                    ov_tokens = count_tokens(last_para_for_overlap)
                    if ov_tokens <= self.overlap:
                        buf.append(last_para_for_overlap)
                        buf_units.append(Unit(kind="para", text=last_para_for_overlap, lang=None, heading_path=u.heading_path))
                        buf_tokens += ov_tokens
                # start with current unit
                buf.append(u.text)
                buf_units.append(u)
                buf_tokens += u_tokens
                continue

            # if adding u would exceed hard max, close and start fresh (overlap applied on next loop)
            if buf_tokens + u_tokens > self.max_tokens:
                close_chunk()
                # start fresh in next iteration
                # prime the new buffer with overlap when next unit comes
                # so continue to re-handle the same unit
                # but we need to reprocess current unit
                # so push back logic: append after closing by resetting iteration
                # easiest approach: add to new buf immediately (with overlap handled at next loop start)
                # Instead, we "rewind" by handling u again:
                # append u at start of new buffer (overlap will be applied if para)
                if last_para_for_overlap and u.kind == "para":
                    ov_tokens = count_tokens(last_para_for_overlap)
                    if ov_tokens <= self.overlap:
                        buf.append(last_para_for_overlap)
                        buf_units.append(Unit(kind="para", text=last_para_for_overlap, lang=None, heading_path=u.heading_path))
                        buf_tokens += ov_tokens
                buf.append(u.text)
                buf_units.append(u)
                buf_tokens += u_tokens
                continue

            # if we'd cross target and current is paragraph, break here to keep coherence
            if buf_tokens + u_tokens > self.target and u.kind == "para":
                close_chunk()
                # start new buffer, add overlap if fits, then current unit
                if last_para_for_overlap:
                    ov_tokens = count_tokens(last_para_for_overlap)
                    if ov_tokens <= self.overlap:
                        buf.append(last_para_for_overlap)
                        buf_units.append(Unit(kind="para", text=last_para_for_overlap, lang=None, heading_path=u.heading_path))
                        buf_tokens += ov_tokens
                buf.append(u.text)
                buf_units.append(u)
                buf_tokens += u_tokens
                continue

            # otherwise, keep filling (code may overflow beyond target up to max)
            buf.append(u.text)
            buf_units.append(u)
            buf_tokens += u_tokens

        close_chunk()
        return chunks

# --------------------------------- source ---------------------------------- #
@dataclass
class DocRecord:
    text: str
    metadata: Dict[str, Any]

class UtilitrSource:
    """Load .md/.qmd from a local utilitR clone and derive URLs & titles."""
    def __init__(self, repo_path: Path, base_url: str = "https://book.utilitr.org/"):
        self.repo_path = Path(repo_path)
        self.base_url = base_url.rstrip("/") + "/"
        self.commit = git_head_commit(self.repo_path)

    def iter_docs(self) -> Iterable[DocRecord]:
        patterns = ["**/*.md", "**/*.qmd"]
        ignore_dirs = {".git", "_book", "docs", ".quarto", "renv", ".github"}
        files: List[Path] = []
        for pat in patterns:
            files.extend(self.repo_path.glob(pat))

        for p in sorted(files):
            if any(part in ignore_dirs for part in p.parts):
                continue
            rel = p.relative_to(self.repo_path).as_posix()
            raw = read_text(p)
            fm, body = extract_frontmatter(raw)

            title = fm.get("title")
            if not title:
                m = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
                title = m.group(1).strip() if m else p.stem

            html_rel = rel.rsplit(".", 1)[0] + ".html"
            source_url = self.base_url + html_rel
            repo_url = f"https://github.com/InseeFrLab/utilitR/blob/{self.commit or 'main'}/{rel}"

            meta = {
                "source": "utilitr",
                "source_type": "quarto_book",
                "file_path": rel,
                "doc_title": title,
                "source_url": source_url,
                "repo_url": repo_url,
                "git_commit": self.commit,
                "lang": fm.get("lang") or "fr",
            }
            yield DocRecord(text=body, metadata=meta)

# -------------------------- remote embeddings client ----------------------- #
class RemoteOpenAIEncoder:
    """
    Minimal client for OpenAI-compatible /v1/embeddings.
    - Sends {"model": <name>, "input": [texts]}
    - Expects {"data": [{"embedding": [...]}, ...]}
    - Normalizes vectors for cosine similarity
    """
    def __init__(self, api_base: str, model: str, api_key: str = "", timeout: float = 60.0, ca_bundle: Optional[str] = None, insecure: bool = False):
        if not api_base:
            raise ValueError("embed_api_base is required")
        self.url = api_base.rstrip("/") + "/embeddings"
        self.model = model
        self.api_key = api_key or ""
        self.timeout = timeout
        self.verify = True
        if insecure:
            self.verify = False
        elif ca_bundle:
            self.verify = ca_bundle  # path to custom CA bundle

        self._dim: Optional[int] = None
        self._session = requests.Session()

    @property
    def dim(self) -> int:
        if self._dim is None:
            arr = self.encode(["dimension probe"], batch_size=1)
            self._dim = int(arr.shape[1])
        return self._dim

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        out_vecs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            payload = {
                "model": self.model,
                "input": texts[i : i + batch_size],
            }
            resp = self._session.post(self.url, json=payload, headers=headers, timeout=self.timeout, verify=self.verify)
            if resp.status_code != 200:
                raise RuntimeError(f"Embeddings API error {resp.status_code}: {resp.text[:500]}")
            js = resp.json()
            data = js.get("data") or []
            if not data:
                raise RuntimeError(f"Embeddings API returned no data: {js}")
            # Some servers may return items unsorted; we assume order aligns with inputs by index
            for d in data:
                emb = d.get("embedding")
                if emb is None:
                    raise RuntimeError(f"Embeddings API item missing 'embedding': {d}")
                out_vecs.append(emb)

        arr = np.array(out_vecs, dtype="float32")
        # L2-normalize for cosine
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms)

# ------------------------------ Qdrant helpers ----------------------------- #
def ensure_collection_dense(client, name: str, dim: int, drop: bool):
    if drop and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

# ----------------------------- pipeline glue ------------------------------- #
def build_chunks_for_doc(text: str, base_meta: Dict[str, Any], chunker: MarkdownChunker) -> List[Chunk]:
    units = chunker.parse_units(text)
    raw_chunks = chunker.chunk(units)

    chunks: List[Chunk] = []
    for idx, (body, heading_path) in enumerate(raw_chunks):
        headings_only = [h for _, h in heading_path if h.strip()]
        chapter = headings_only[0] if headings_only else base_meta.get("doc_title")
        section = headings_only[-1] if headings_only else base_meta.get("doc_title")
        breadcrumbs = headings_only
        anchor = slugify(section) if section else None
        source_url = base_meta.get("source_url")
        url_with_anchor = f"{source_url}#{anchor}" if source_url and anchor else source_url

        meta = {
            **base_meta,
            "chapter": chapter,
            "section": section,
            "breadcrumbs": breadcrumbs,
            "token_count": count_tokens(body),
            "url": url_with_anchor,
            "chunk_index": idx,
        }
        h = hashlib.sha256()
        h.update((base_meta.get("file_path", "") + f"#{idx}" + (base_meta.get("git_commit") or "")).encode("utf-8"))
        name = f"{base_meta.get('file_path', '')}#{idx}@{base_meta.get('git_commit') or 'main'}"
        ns = uuid.uuid5(uuid.NAMESPACE_URL, "https://book.utilitr.org")
        chunk_id = str(uuid.uuid5(ns, name)) # e.g. "550e8400-e29b-41d4-a716-446655440000"
        chunks.append(Chunk(id=chunk_id, text=body, metadata=meta))
    return chunks

def upsert_dense(client: QdrantClient, collection: str, chunks: List[Chunk], encoder: RemoteOpenAIEncoder, batch_size: int = 64):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        vecs = encoder.encode(texts, batch_size=batch_size)
        points = [PointStruct(id=c.id, vector=vecs[j].tolist(), payload={"text": c.text, **c.metadata}) for j, c in enumerate(batch)]
        client.upsert(collection_name=collection, points=points)

# ----------------------------------- CLI ----------------------------------- #
def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Ingest utilitR into Qdrant using a remote OpenAI-compatible embeddings API")
    ap.add_argument("--repo-path", required=True, help="Path to local clone of utilitR")
    ap.add_argument("--collection", required=True, help="Qdrant collection name")
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", None))
    ap.add_argument("--drop-collection", action="store_true", help="Recreate collection before ingest")

    # chunking
    ap.add_argument("--target-tokens", type=int, default=800)
    ap.add_argument("--overlap-tokens", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=64)

    # remote embeddings
    ap.add_argument("--embed-api-base", default=os.getenv("EMBED_API_BASE"), help="Base URL, e.g. https://.../v1")
    ap.add_argument("--embed-api-key", default=os.getenv("EMBED_API_KEY", ""), help="Bearer token if required")
    ap.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "BAAI/bge-multilingual-gemma2"))
    ap.add_argument("--embed-timeout", type=float, default=60.0)
    ap.add_argument("--embed-ca-bundle", default=os.getenv("REQUESTS_CA_BUNDLE", None), help="Path to custom CA bundle (PEM)")
    ap.add_argument("--embed-insecure", action="store_true", help="Disable TLS verification (NOT recommended)")

    args = ap.parse_args()

    if not args.embed_api_base:
        print("ERROR: --embed-api-base (or EMBED_API_BASE) is required.", file=sys.stderr)
        sys.exit(2)

    repo_path = Path(args.repo_path).expanduser().resolve()
    if not repo_path.exists():
        print(f"Repo path not found: {repo_path}", file=sys.stderr)
        sys.exit(1)

    # build chunker & source
    chunker = MarkdownChunker(target_tokens=args.target_tokens, overlap_tokens=args.overlap_tokens, max_tokens=args.max_tokens)
    source = UtilitrSource(repo_path=repo_path)

    # collect chunks
    all_chunks: List[Chunk] = []
    n_docs = 0
    for doc in source.iter_docs():
        all_chunks.extend(build_chunks_for_doc(doc.text, doc.metadata, chunker))
        n_docs += 1
    print(f"Discovered {n_docs} documents, produced {len(all_chunks)} chunks.")

    # remote encoder
    encoder = RemoteOpenAIEncoder(
        api_base=args.embed_api_base,
        api_key=args.embed_api_key or "",
        model=args.embed_model,
        timeout=args.embed_timeout,
        ca_bundle=args.embed_ca_bundle,
        insecure=args.embed_insecure,
    )

    # Qdrant
    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    if args.drop_collection:
        ensure_collection_dense(client, args.collection, dim=encoder.dim, drop=args.drop_collection)

    # ingest
    upsert_dense(client, args.collection, all_chunks, encoder, batch_size=args.batch_size)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()