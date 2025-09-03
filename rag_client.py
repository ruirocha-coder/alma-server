import os
import re
import uuid
import httpx
from typing import List, Dict, Any
from bs4 import BeautifulSoup

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from openai import OpenAI

# ========================= Config =========================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

QDRANT_COLLECTION = "alma_docs"
EMBED_MODEL = "text-embedding-3-small"  # 1536 dims
UPSERT_BATCH = 32

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
openai_client = OpenAI(api_key=OPENAI_KEY)

# ========================= Utils =========================
def _uuid_for_chunk(namespace: str, url: str, idx: int) -> str:
    base = f"{namespace}:{url}:{idx}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks

def _embed_texts(chunks: List[str]) -> List[List[float]]:
    if not chunks:
        return []
    resp = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=chunks
    )
    return [d.embedding for d in resp.data]

# ========================= Fetch HTML =========================
def fetch_url(url: str, timeout: int = 20) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0 Safari/537.36"
        )
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        raise RuntimeError(f"fetch_failed: {e}")

# ========================= Ingest =========================
def ingest(namespace: str, url: str, title: str, full_text: str) -> int:
    chunks = _chunk_text(full_text)
    if not chunks:
        return 0

    vecs = _embed_texts(chunks)
    if not vecs:
        return 0

    # garantir que a collection tem a dimensão correta
    embed_dim = len(vecs[0])
    try:
        info = qdrant.get_collection(QDRANT_COLLECTION)
        existing_dim = None
        if info and getattr(info, "config", None):
            vc = info.config.vectors_config
            if hasattr(vc, "config") and getattr(vc.config, "size", None):
                existing_dim = vc.config.size
            elif getattr(vc, "size", None):
                existing_dim = vc.size
        if existing_dim and existing_dim != embed_dim:
            qdrant.recreate_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
            )
    except Exception:
        # se não existir, cria
        qdrant.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )

    points: List[PointStruct] = []
    for idx, (c, v) in enumerate(zip(chunks, vecs)):
        points.append(PointStruct(
            id=_uuid_for_chunk(namespace, url, idx),
            vector=v,
            payload={"url": url, "title": title, "text": c, "namespace": namespace}
        ))

    total = 0
    for i in range(0, len(points), UPSERT_BATCH):
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i+UPSERT_BATCH])
        total += len(points[i:i+UPSERT_BATCH])
    return total

# ========================= Sitemap ingest =========================
def ingest_sitemap(namespace: str, sitemap_url: str, max_pages: int = 200) -> Dict[str, Any]:
    from urllib.parse import urljoin
    collected, ingested, failed = [], [], []

    try:
        xml = fetch_url(sitemap_url)
        soup = BeautifulSoup(xml, "xml")
        locs = [loc.text for loc in soup.find_all("loc")]
    except Exception as e:
        return {"ok": False, "error": "sitemap_failed", "detail": str(e)}

    for url in locs[:max_pages]:
        try:
            html = fetch_url(url)
            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.string.strip() if soup.title else url
            text = soup.get_text(" ", strip=True)
            if text:
                n = ingest(namespace, url, title, text)
                if n > 0:
                    ingested.append(url)
        except Exception as e:
            failed.append((url, str(e)))
        collected.append(url)

    return {
        "ok": True,
        "sitemap": sitemap_url,
        "pages_ingested": len(ingested),
        "pages_failed": len(failed),
        "namespace": namespace,
        "ingested_urls": ingested,
        "failed_urls": failed,
    }

# ========================= Search =========================
def search(namespace: str, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    emb = _embed_texts([query])[0]
    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=emb,
        limit=top_k,
        query_filter={"must": [{"key": "namespace", "match": {"value": namespace}}]} if namespace else None
    )
    return [{"url": h.payload.get("url"), "title": h.payload.get("title"), "text": h.payload.get("text")} for h in hits]
