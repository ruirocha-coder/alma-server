# rag_client.py
import os, re, math, time, hashlib, io
import requests
from typing import List, Dict, Iterable, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from pypdf import PdfReader
import tiktoken

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "alma_docs").strip()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
EMBED_DIM = 1536  # text-embedding-3-* -> 1536

MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS_PER_CHUNK", "800"))
OVERLAP_TOKENS = int(os.getenv("RAG_OVERLAP_TOKENS", "80"))

CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "30"))
CRAWL_MAX_DEPTH = int(os.getenv("CRAWL_MAX_DEPTH", "2"))

# ─────────────────────────────────────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────────────────────────────────────
oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────
def ensure_collection():
    """Garante que a coleção existe com 1536 dims + cosine."""
    exist = False
    try:
        info = qdr.get_collection(QDRANT_COLLECTION)
        exist = True if info else False
    except Exception:
        exist = False

    if not exist:
        qdr.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
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
    step = max_tokens - overlap
    for start in range(0, len(ids), step):
        end = min(start + max_tokens, len(ids))
        chunk_ids = ids[start:end]
        chunks.append(enc.decode(chunk_ids))
        if end == len(ids):
            break
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    """OpenAI embeddings batched."""
    if not texts:
        return []
    # OpenAI SDK já faz batching; aqui chamamos de forma simples:
    resp = oai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def _point_id(namespace: str, src_id: str, chunk_ix: int) -> str:
    raw = f"{namespace}:{src_id}:{chunk_ix}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def upsert_chunks(namespace: str, source_meta: Dict, chunks: List[str]):
    """Insere/atualiza os chunks na coleção."""
    ensure_collection()
    vectors = embed_texts(chunks)
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        pid = _point_id(namespace, source_meta.get("source_id",""), i)
        payload = {
            "text": chunk,
            "namespace": namespace,
            **source_meta
        }
        points.append(PointStruct(id=pid, vector=vec, payload=payload))
    if points:
        qdr.upsert(collection_name=QDRANT_COLLECTION, points=points)

# ─────────────────────────────────────────────────────────────────────────────
# Ingest: TEXT
# ─────────────────────────────────────────────────────────────────────────────
def ingest_text(title: str, text: str, namespace: str="default", source_id: str=None) -> Dict:
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "Texto vazio"}
    chunks = chunk_text(text)
    src_id = source_id or hashlib.sha1((title + text[:200]).encode("utf-8")).hexdigest()
    upsert_chunks(namespace, {"type": "text", "title": title, "source_id": src_id}, chunks)
    return {"ok": True, "chunks": len(chunks), "source_id": src_id}

# ─────────────────────────────────────────────────────────────────────────────
# Ingest: PDF por URL
# ─────────────────────────────────────────────────────────────────────────────
def fetch_pdf_text(pdf_url: str) -> Tuple[str, int]:
    r = requests.get(pdf_url, timeout=30)
    r.raise_for_status()
    reader = PdfReader(io.BytesIO(r.content))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    text = "\n\n".join(pages)
    return text, len(reader.pages)

def ingest_pdf_url(pdf_url: str, title: str=None, namespace: str="default") -> Dict:
    text, n_pages = fetch_pdf_text(pdf_url)
    title = title or pdf_url.split("/")[-1]
    src_id = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()
    chunks = chunk_text(text)
    upsert_chunks(namespace, {"type": "pdf", "title": title, "url": pdf_url, "pages": n_pages, "source_id": src_id}, chunks)
    return {"ok": True, "chunks": len(chunks), "pages": n_pages, "source_id": src_id}

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
    # remove script/style/nav/footer
    for tag in soup(["script","style","nav","footer","header","form"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text

def crawl_and_ingest(seed_url: str, namespace: str="default", max_pages: int=CRAWL_MAX_PAGES, max_depth: int=CRAWL_MAX_DEPTH) -> Dict:
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
            if tokenize_len(text) < 50:
                pass  # ignora páginas quase vazias
            else:
                title = url
                src_id = hashlib.sha1(url.encode("utf-8")).hexdigest()
                chunks = chunk_text(text)
                upsert_chunks(namespace, {"type": "web", "title": title, "url": url, "domain": src_host, "source_id": src_id}, chunks)
                n_ingested += 1
            if depth < max_depth:
                for link in _extract_links(url, html):
                    if _is_same_domain(seed_url, link):
                        queue.append((link, depth+1))
        except Exception:
            continue
    return {"ok": True, "pages_ingested": n_ingested, "visited": len(seen), "domain": src_host}
