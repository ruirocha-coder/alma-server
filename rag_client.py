# rag_client.py
import os, re, math, time, hashlib, io
import requests
from typing import List, Dict, Iterable, Tuple, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from pypdf import PdfReader
import tiktoken

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
QDRANT_URL        = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "alma_docs").strip()

EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
EMBED_DIM         = 1536  # text-embedding-3-* -> 1536

# chunking
MAX_TOKENS      = int(os.getenv("RAG_MAX_TOKENS_PER_CHUNK", "800"))
OVERLAP_TOKENS  = int(os.getenv("RAG_OVERLAP_TOKENS", "80"))

# crawling
CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "30"))
CRAWL_MAX_DEPTH = int(os.getenv("CRAWL_MAX_DEPTH", "2"))

# retrieval
RAG_TOP_K              = int(os.getenv("RAG_TOP_K", "6"))
RAG_CONTEXT_TOKEN_BUDGET = int(os.getenv("RAG_CONTEXT_TOKEN_BUDGET", "1600"))
RAG_DEFAULT_NAMESPACE  = os.getenv("RAG_DEFAULT_NAMESPACE", "default").strip() or "default"

REQUEST_TIMEOUT = 30

# ─────────────────────────────────────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────────────────────────────────────
oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=REQUEST_TIMEOUT)

# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────
def ensure_collection() -> None:
    """Garante que a coleção existe com 1536 dims + cosine (não recria se já existir)."""
    try:
        qdr.get_collection(QDRANT_COLLECTION)
        return
    except Exception:
        pass
    qdr.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

def tokenize_len(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))

def chunk_text(text: str, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(text or "")
    if not ids:
        return []
    chunks = []
    step = max(1, max_tokens - overlap)
    for start in range(0, len(ids), step):
        end = min(start + max_tokens, len(ids))
        chunk_ids = ids[start:end]
        chunks.append(enc.decode(chunk_ids))
        if end >= len(ids):
            break
    return chunks

def embed_texts(texts: List[str], batch_size: int = 96) -> List[List[float]]:
    """OpenAI embeddings com batching seguro."""
    out: List[List[float]] = []
    if not texts:
        return out
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = oai.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out

def _point_id(namespace: str, src_id: str, chunk_ix: int) -> str:
    raw = f"{namespace}:{src_id}:{chunk_ix}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def upsert_chunks(namespace: str, source_meta: Dict, chunks: List[str]) -> None:
    """Insere/atualiza os chunks na coleção."""
    if not chunks:
        return
    ensure_collection()
    vectors = embed_texts(chunks)
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        pid = _point_id(namespace, source_meta.get("source_id", ""), i)
        payload = {
            "text": chunk,
            "namespace": namespace,
            **source_meta
        }
        points.append(PointStruct(id=pid, vector=vec, payload=payload))
    if points:
        qdr.upsert(collection_name=QDRANT_COLLECTION, points=points)

def delete_by_source_id(source_id: str, namespace: Optional[str] = None) -> int:
    """Remove todos os pontos de um dado source_id (+ opcionalmente namespace)."""
    ensure_collection()
    must = [FieldCondition(key="source_id", match=MatchValue(value=source_id))]
    if namespace:
        must.append(FieldCondition(key="namespace", match=MatchValue(value=namespace)))
    res = qdr.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(must=must),
    )
    return int(getattr(res, "status", 0) == "acknowledged")

# ─────────────────────────────────────────────────────────────────────────────
# Ingest: TEXT
# ─────────────────────────────────────────────────────────────────────────────
def ingest_text(title: str, text: str, namespace: str = RAG_DEFAULT_NAMESPACE, source_id: str = None) -> Dict:
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "Texto vazio"}
    chunks = chunk_text(text)
    src_id = source_id or hashlib.sha1((title + text[:200]).encode("utf-8")).hexdigest()
    upsert_chunks(namespace, {"type": "text", "title": title, "source_id": src_id}, chunks)
    return {"ok": True, "chunks": len(chunks), "source_id": src_id}

# ─────────────────────────────────────────────────────────────────────────────
# Ingest: PDF por URL / bytes (uploads)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_pdf_text(pdf_url: str) -> Tuple[str, int]:
    r = requests.get(pdf_url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "alma-rag/1.0"})
    r.raise_for_status()
    reader = PdfReader(io.BytesIO(r.content))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\n".join(pages)
    return text, len(reader.pages)

def ingest_pdf_url(pdf_url: str, title: str = None, namespace: str = RAG_DEFAULT_NAMESPACE) -> Dict:
    text, n_pages = fetch_pdf_text(pdf_url)
    title = title or pdf_url.split("/")[-1]
    src_id = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()
    chunks = chunk_text(text)
    upsert_chunks(namespace, {"type": "pdf", "title": title, "url": pdf_url, "pages": n_pages, "source_id": src_id}, chunks)
    return {"ok": True, "chunks": len(chunks), "pages": n_pages, "source_id": src_id}

def ingest_pdf_bytes(data: bytes, title: str, namespace: str = RAG_DEFAULT_NAMESPACE) -> Dict:
    reader = PdfReader(io.BytesIO(data))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\n".join(pages)
    src_id = hashlib.sha1((title + str(len(data))).encode("utf-8")).hexdigest()
    chunks = chunk_text(text)
    upsert_chunks(namespace, {"type": "pdf", "title": title, "pages": len(pages), "source_id": src_id}, chunks)
    return {"ok": True, "chunks": len(chunks), "pages": len(pages), "source_id": src_id}

# ─────────────────────────────────────────────────────────────────────────────
# Ingest: Crawl website (mesmo domínio)
# ─────────────────────────────────────────────────────────────────────────────
def _is_same_domain(seed: str, url: str) -> bool:
    a = urlparse(seed)
    b = urlparse(url)
    return a.netloc == b.netloc

def _clean_url(u: str) -> str:
    return u.split("#")[0]

def _extract_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        absu = urljoin(base_url, href)
        out.append(_clean_url(absu))
    return out

def _extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text

def crawl_and_ingest(seed_url: str, namespace: str = RAG_DEFAULT_NAMESPACE,
                     max_pages: int = CRAWL_MAX_PAGES, max_depth: int = CRAWL_MAX_DEPTH) -> Dict:
    seen = set()
    queue = [(seed_url, 0)]
    n_ingested = 0
    src_host = urlparse(seed_url).netloc
    while queue and n_ingested < max_pages:
        url, depth = queue.pop(0)
        url = _clean_url(url)
        if url in seen:
            continue
        seen.add(url)
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "alma-rag/1.0"})
            if "text/html" not in r.headers.get("Content-Type", ""):
                continue
            html = r.text
            text = _extract_main_text(html)
            if tokenize_len(text) >= 50:
                title = url
                src_id = hashlib.sha1(url.encode("utf-8")).hexdigest()
                chunks = chunk_text(text)
                upsert_chunks(
                    namespace,
                    {"type": "web", "title": title, "url": url, "domain": src_host, "source_id": src_id},
                    chunks,
                )
                n_ingested += 1
            if depth < max_depth:
                for link in _extract_links(url, html):
                    if _is_same_domain(seed_url, link):
                        queue.append((link, depth + 1))
        except Exception:
            continue
    return {"ok": True, "pages_ingested": n_ingested, "visited": len(seen), "domain": src_host}

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval (search) no Qdrant
# ─────────────────────────────────────────────────────────────────────────────
def search_chunks(query: str, namespace: Optional[str] = None, top_k: int = None) -> List[Dict]:
    query = (query or "").strip()
    if not query:
        return []
    ensure_collection()
    qvec = embed_texts([query])[0]
    flt = None
    if namespace:
        flt = Filter(must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))])
    hits = qdr.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qvec,
        limit=top_k or RAG_TOP_K,
        query_filter=flt,
        with_payload=True,
        with_vectors=False,
        score_threshold=None,
    )
    out: List[Dict] = []
    for h in hits:
        pl = h.payload or {}
        out.append({
            "text": pl.get("text", ""),
            "score": float(h.score),
            "title": pl.get("title", ""),
            "url": pl.get("url"),
            "type": pl.get("type"),
            "namespace": pl.get("namespace"),
            "source_id": pl.get("source_id"),
        })
    return out

def build_context_block(matches: List[Dict], token_budget: int = None) -> str:
    """Empilha os melhores trechos até ao orçamento de tokens."""
    budget = token_budget or RAG_CONTEXT_TOKEN_BUDGET
    enc = tiktoken.get_encoding("cl100k_base")
    chosen = []
    total = 0
    for m in matches:
        head = m.get("title") or m.get("url") or m.get("type") or "doc"
        chunk = f"[{head}] {m.get('text', '')}".strip()
        cost = len(enc.encode(chunk))
        if total + cost > budget:
            break
        chosen.append(chunk)
        total += cost
    if not chosen:
        return ""
    return "Contexto de conhecimento interno (RAG):\n" + "\n\n".join(chosen)
