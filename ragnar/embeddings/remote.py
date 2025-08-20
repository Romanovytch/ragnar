from __future__ import annotations
import requests
import numpy as np
from typing import List, Optional


class RemoteOpenAIEncoder:
    """Minimal client for OpenAI-compatible `/v1/embeddings`.

    Sends batched POST requests to `{api_base}/embeddings`, returns a
    NumPy array of L2-normalized embeddings suitable for cosine search
    in vector databases (e.g., Qdrant).

    Normalization ensures dot product ≡ cosine similarity.

    Args:
        api_base: Base URL of the OpenAI-compatible API
            (e.g., "https://api.openai.com/v1").
        model: Embedding model identifier understood by the server
            (e.g., "BAAI/bge-multilingual-gemma2").
        api_key: Optional bearer token ("Authorization: Bearer ...").
        timeout: Per-request timeout in seconds.
        ca_bundle: Path to a custom CA bundle (PEM) for TLS verification.
        insecure: If True, disables TLS verification (use only for local testing).

    Note:
        Uses a persistent :class:`requests.Session` for connection pooling.
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
        """Return the embedding dimensionality (D), probing once if unknown.

        On first access, performs a minimal embedding request
        (:meth:`encode` on a single short string) to discover D and caches it.

        Returns:
            The embedding dimensionality as an integer.

        Raises:
            RuntimeError: If the embeddings API returns a non-200 response
                during the probe request.
        """
        if self._dim is None:
            v = self.encode(["dimension probe"], batch_size=1)
            self._dim = int(v.shape[1])
        return self._dim

    def encode(self, texts: List[str], batch_size: int = 64):
        """Embed a list of strings and return L2-normalized vectors.

        Batches the input to reduce request size, performs POST requests to
        `/embeddings`, collects the `embedding` fields, converts to float32,
        and normalizes each vector to unit length.

        Args:
            texts: List of input strings to embed (order preserved).
            batch_size: Number of texts per request batch.

        Returns:
            A NumPy array of shape `(len(texts), D)` with dtype `float32`,
            where each row is L2-normalized.

        Raises:
            RuntimeError: If the embeddings API returns a non-200 response.

        Notes:
            - The server must return a JSON object with a `data` list of
              `{ "embedding": [...] }` items, one per input string.
            - Normalization uses a small epsilon (1e-12) to avoid division by zero.
        """
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
