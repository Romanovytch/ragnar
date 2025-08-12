from __future__ import annotations
import re
import yaml
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple


def _get_tokenizer():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        class _Fallback:
            def encode(self, text): return text.split()
        return _Fallback()


ENC = _get_tokenizer()


def count_tokens(text: str) -> int:
    try:
        return len(ENC.encode(text))
    except Exception:
        return max(1, len(text) // 4)


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
            body = text[end + 4:]
            try:
                data = yaml.safe_load(fm) or {}
                if not isinstance(data, dict):
                    data = {}
                return data, body.lstrip()
            except Exception:
                return {}, text
    return {}, text


def git_head_commit(repo_path: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out
    except Exception:
        return None


# Deterministic UUIDv5 IDs (Qdrant-friendly)
def make_chunk_id(file_path: str, idx: int, commit: str | None) -> str:
    ns = uuid.uuid5(uuid.NAMESPACE_URL, "https://book.utilitr.org")
    name = f"{file_path}#{idx}@{commit or 'main'}"
    return str(uuid.uuid5(ns, name))
