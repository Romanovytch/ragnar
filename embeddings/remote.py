from __future__ import annotations
import requests
import numpy as np
from typing import List, Optional


class RemoteOpenAIEncoder:
    """
    Minimal OpenAI-compatible /v1/embeddings client.
    Normalizes vectors for cosine similarity.
    """
    def __init__(self, api_base: str, model: str, api_key: str = "", timeout: float = 60.0,
                 ca_bundle: Optional[str] = None, insecure: bool = False):
        if not api_base or not api_base.startswith(("http://", "https://")):
            raise ValueError("api_base must start with http(s)://")
        self.url = api_base.rstrip("/") + "/embeddings"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.verify = False if insecure else (ca_bundle or True)
        self._session = requests.Session()
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            v = self.encode(["dimension probe"], batch_size=1)
            self._dim = int(v.shape[1])
        return self._dim

    def encode(self, texts: List[str], batch_size: int = 64):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        out = []
        for i in range(0, len(texts), batch_size):
            payload = {"model": self.model, "input": texts[i:i+batch_size]}
            r = self._session.post(self.url, json=payload, headers=headers,
                                   timeout=self.timeout, verify=self.verify)
            if r.status_code != 200:
                raise RuntimeError(f"Embeddings API {r.status_code}: {r.text[:500]}")
            data = r.json().get("data") or []
            out.extend([d["embedding"] for d in data])

        arr = np.array(out, dtype="float32")
        arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        return arr
