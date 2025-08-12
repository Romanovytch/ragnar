# test_retrieval.py
import os, sys, argparse, requests, numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

def embed_query_openai(base, model, text, api_key="", timeout=30, verify=True):
    url = base.rstrip("/") + "/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    r = requests.post(url, json={"model": model, "input": [text]},
                      headers=headers, timeout=timeout, verify=verify)
    r.raise_for_status()
    v = np.array(r.json()["data"][0]["embedding"], dtype="float32")
    v /= (np.linalg.norm(v) + 1e-12)
    return v.tolist()

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default="http://qdrant:6333")
    ap.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY",""))
    ap.add_argument("--collection", default="utilitr_v1")
    ap.add_argument("--embed-api-base", default=os.getenv("EMBED_API_BASE",""))
    ap.add_argument("--embed-model", default="BAAI/bge-multilingual-gemma2")
    ap.add_argument("--embed-api-key", default=os.getenv("EMBED_API_KEY",""))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--insecure", action="store_true")
    ap.add_argument("query", nargs="*")
    args = ap.parse_args()

    query = " ".join(args.query) or "Comment utiliser renv dans un projet R ?"
    vec = embed_query_openai(args.embed_api_base, args.embed_model, query,
                             api_key=args.embed_api_key, verify=not args.insecure)

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key or None)
    flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value="utilitr"))])

    hits = client.search(collection_name=args.collection, query_vector=vec,
                         query_filter=flt, limit=args.top_k, with_payload=True)

    print(f"\nQuery: {query}\nTop {len(hits)} results:\n")
    for i, h in enumerate(hits, 1):
        p = h.payload or {}
        url = p.get("url") or p.get("source_url")
        section = p.get("section"); chapter = p.get("chapter")
        text = (p.get("text") or "").strip()
        snippet = (text[:800] + "…") if len(text) > 800 else text
        print(f"[{i}] score={h.score:.4f}  {chapter or ''} — {section or ''}")
        if url: print("    ", url)
        print("    file:", p.get("file_path"))
        print("    --- snippet ---")
        print("    " + snippet.replace("\n", "\n    "))
        print()

if __name__ == "__main__":
    main()