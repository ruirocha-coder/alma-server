# rag_client.py
import os, re, math, time, hashlib, io, random
import requests
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin, urlparse, quote_plus

from bs4 import BeautifulSoup
from pypdf import PdfReader
import tiktoken

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
def _getint(env: str, default: str) -> int:
    try:
        return int((os.getenv(env, default) or default).strip())
    except Exception:
        return int(default)

QDRANT_URL         = (os.getenv("QDRANT_URL", "") or "").strip()
QDRANT_API_KEY     = (os.getenv("QDRANT_API_KEY", "") or "").strip()
QDRANT_COLLECTION  = (os.getenv("QDRANT_COLLECTION", "alma_docs") or "alma_docs").strip()

EMBEDDING_MODEL    = (os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small").strip()
EMBED_DIM          = _getint("EMBED_DIM", "1536")  # text-embedding-3-* -> 1536

# chunking
MAX_TOKENS         = _getint("RAG_MAX_TOKENS_PER_CHUNK", "800")
OVERLAP_TOKENS     = _getint("RAG_OVERLAP_TOKENS", "80")
MIN_TOKENS_PAGE    = _getint("RAG_MIN_TOKENS_PER_PAGE", "50")

# crawling
CRAWL_MAX_PAGES    = _getint("CRAWL_MAX_PAGES", "30")
CRAWL_MAX_DEPTH    = _getint("CRAWL_MAX_DEPTH", "2")
SAME_DOMAIN_ONLY   = (os.getenv("RAG_SAME_DOMAIN_ONLY", "true").strip().lower() in ("1","true","yes"))

# retrieval
RAG_TOP_K                 = _getint("RAG_TOP_K", "6")
RAG_CONTEXT_TOKEN_BUDGET  = _getint("RAG_CONTEXT_TOKEN_BUDGET", "1600")
RAG_DEFAULT_NAMESPACE     = (os.getenv("RAG_DEFAULT_NAMESPACE", "default") or "default").strip()

REQUEST_TIMEOUT     = _getint("RAG_REQUEST_TIMEOUT", "30")

# HTTP proxy opcional (ex.: "https://app.scrapingbee.com/api/v1?api_key=XXX&url=")
HTTP_PROXY_BASE     = (os.getenv("HTTP_PROXY_BASE", "") or "").strip()

# ─────────────────────────────────────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────────────────────────────────────
oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=REQUEST_TIMEOUT)

# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────
def ensure_collection() -> None:
    """Garante que a coleção existe (não recria se já existir)."""
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

def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(text or "")
    if not ids:
        return []
    chunks: List[str] = []
    step = max(1, max_tokens - overlap)
    for start in range(0, len(ids), step):
        end = min(start + max_tokens, len(ids))
        chunk_ids = ids[start:end]
        chunks.append(enc.decode(chunk_ids))
        if end >= len(ids):
            break
    return chunks

def embed_texts(texts: List[str], batch_size: int = 96) -> List[List[float]]:
    """OpenAI embeddings com batching."""
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
# HTTP fetcher (anti-403 básico + proxy opcional)
# ─────────────────────────────────────────────────────────────────────────────
_UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36",
]

def _proxied_url(url: str) -> str:
    if not HTTP_PROXY_BASE:
        return url
    # Se a base já incluir "url=" no fim, apenas anexamos a URL codificada
    if HTTP_PROXY_BASE.endswith("="):
        return HTTP_PROXY_BASE + quote_plus(url)
    # Caso contrário, tentamos o padrão ?url=
    joiner = "&" if "?" in HTTP_PROXY_BASE else "?"
    return f"{HTTP_PROXY_BASE}{joiner}url={quote_plus(url)}"

def http_get(url: str, timeout: int = REQUEST_TIMEOUT, expect: str = "html") -> requests.Response:
    """
    GET com headers de browser, retries e proxy opcional.
    expect: "html" | "pdf" (apenas para checks simples de Content-Type)
    """
    headers = {
        "User-Agent": random.choice(_UA_LIST),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8" if expect=="html" else "*/*",
        "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
    }
    last_exc = None
    for i in range(4):
        try:
            tgt = _proxied_url(url)
            r = requests.get(tgt, headers=headers, timeout=timeout, allow_redirects=True)
            # Checks leves por tipo
            ctype = r.headers.get("Content-Type", "").lower()
            if r.status_code in (403, 429, 503):
                time.sleep(1.0 + i * 0.8)
                continue
            if expect == "html" and ("text/html" not in ctype and "application/xhtml+xml" not in ctype):
                # alguns CDNs omitem; deixamos passar se body tem tags básicas
                if "<html" not in r.text.lower():
                    r.raise_for_status()
            if expect == "pdf" and ("application/pdf" not in ctype) and not r.url.lower().endswith(".pdf"):
                r.raise_for_status()
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(0.8 + i * 0.7)
    if last_exc:
        raise last_exc
    raise RuntimeError("Falha HTTP sem exceção explícita")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers de parsing HTML
# ─────────────────────────────────────────────────────────────────────────────
def _is_same_domain(seed: str, url: str) -> bool:
    a = urlparse(seed)
    b = urlparse(url)
    return a.netloc == b.netloc

def _clean_url(u: str) -> str:
    # remove fragmentos e normaliza espaços
    return (u or "").split("#")[0].strip()

def _extract_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        absu = urljoin(base_url, href)
        out.append(_clean_url(absu))
    # dedup simples
    seen = set()
    uniq = []
    for x in out:
        if x and x not in seen:
            uniq.append(x); seen.add(x)
    return uniq

def _guess_title(url: str, html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    t = (soup.title.string if soup.title else "") or ""
    t = " ".join((t or "").split())
    return t or url

def _extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript"]):
        tag.decompose()
    # remove menus repetidos por classes comuns
    for sel in [".navbar", ".menu", ".breadcrumbs", ".cookie", ".cookies", ".newsletter"]:
        for el in soup.select(sel):
            el.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text

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
# Ingest: PDF (URL / bytes)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_pdf_text(pdf_url: str) -> Tuple[str, int]:
    r = http_get(pdf_url, timeout=REQUEST_TIMEOUT, expect="pdf")
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
# Ingest: Single URL (HTML)
# ─────────────────────────────────────────────────────────────────────────────
def ingest_url(page_url: str, namespace: str = RAG_DEFAULT_NAMESPACE) -> Dict:
    u = _clean_url(page_url)
    r = http_get(u, timeout=REQUEST_TIMEOUT, expect="html")
    html = r.text
    text = _extract_main_text(html)
    if tokenize_len(text) < MIN_TOKENS_PAGE:
        return {"ok": False, "error": "Página sem conteúdo útil suficiente", "url": u}
    title = _guess_title(u, html)
    src_id = hashlib.sha1(u.encode("utf-8")).hexdigest()
    chunks = chunk_text(text)
    upsert_chunks(namespace, {"type": "web", "title": title, "url": u, "domain": urlparse(u).netloc, "source_id": src_id}, chunks)
    return {"ok": True, "chunks": len(chunks), "source_id": src_id, "url": u, "title": title}

# ─────────────────────────────────────────────────────────────────────────────
# Ingest: Crawl website (BFS, mesmo domínio opcional)
# ─────────────────────────────────────────────────────────────────────────────
def crawl_and_ingest(seed_url: str, namespace: str = RAG_DEFAULT_NAMESPACE,
                     max_pages: int = CRAWL_MAX_PAGES, max_depth: int = CRAWL_MAX_DEPTH) -> Dict:
    seed_url = _clean_url(seed_url)
    seen = set()
    queue: List[Tuple[str, int]] = [(seed_url, 0)]
    n_ingested = 0
    src_host = urlparse(seed_url).netloc

    while queue and n_ingested < max_pages:
        url, depth = queue.pop(0)
        url = _clean_url(url)
        if url in seen:
            continue
        seen.add(url)
        try:
            r = http_get(url, timeout=REQUEST_TIMEOUT, expect="html")
            html = r.text
            ctype = r.headers.get("Content-Type", "").lower()
            if "text/html" not in ctype and "<html" not in html.lower():
                # ignora tipos não HTML
                pass
            else:
                text = _extract_main_text(html)
                if tokenize_len(text) >= MIN_TOKENS_PAGE:
                    title = _guess_title(url, html)
                    src_id = hashlib.sha1(url.encode("utf-8")).hexdigest()
                    chunks = chunk_text(text)
                    upsert_chunks(
                        namespace,
                        {"type": "web", "title": title, "url": url, "domain": src_host, "source_id": src_id},
                        chunks,
                    )
                    n_ingested += 1
                # expande links
                if depth < max_depth:
                    for link in _extract_links(url, html):
                        if not link or link in seen:
                            continue
                        if SAME_DOMAIN_ONLY and not _is_same_domain(seed_url, link):
                            continue
                        queue.append((link, depth + 1))
        except Exception:
            # falhas de rede/bloqueios são normais — continuamos
            continue

    return {"ok": True, "pages_ingested": n_ingested, "visited": len(seen), "domain": src_host}

# ─────────────────────────────────────────────────────────────────────────────
# Ingest: Sitemap (urlset e sitemapindex)
# ─────────────────────────────────────────────────────────────────────────────
def ingest_sitemap(sitemap_or_site_url: str, namespace: str = RAG_DEFAULT_NAMESPACE,
                   max_urls: int = 200) -> Dict:
    """
    Aceita:
      - URL do sitemap (ex.: https://site.tld/sitemap.xml)
      - OU a homepage (tentamos /sitemap.xml).
    Percorre sitemapindex/urlset e chama ingest_url por cada URL (até max_urls).
    """
    start = sitemap_or_site_url.strip()
    # tenta autodetectar /sitemap.xml se parecer homepage
    if not start.endswith(".xml"):
        base = start.rstrip("/")
        start = base + "/sitemap.xml"

    r = http_get(start, timeout=REQUEST_TIMEOUT, expect="html")  # XML vem como text/xml
    soup = BeautifulSoup(r.text, "xml")

    urls: List[str] = []
    # sitemapindex de sub-sitemaps
    for loc in soup.find_all("loc"):
        val = (loc.text or "").strip()
        if not val:
            continue
        urls.append(val)

    collected: List[str] = []
    def _harvest(url: str):
        try:
            rr = http_get(url, timeout=REQUEST_TIMEOUT, expect="html")
            ss = BeautifulSoup(rr.text, "xml")
            # se for um sitemap de URLs finais
            for loc2 in ss.find_all("loc"):
                u = (loc2.text or "").strip()
                if u:
                    collected.append(u)
                    if len(collected) >= max_urls:
                        return True
        except Exception:
            pass
        return False

    # se o primeiro já era um urlset, collected enche direto
    if collected or soup.find("urlset"):
        for loc in soup.find_all("loc"):
            u = (loc.text or "").strip()
            if u:
                collected.append(u)
                if len(collected) >= max_urls:
                    break
    else:
        # era um sitemapindex -> abre cada sub-sitemap
        for sm in urls:
            if _harvest(sm):
                break

    # dedup e ingest
    seen = set()
    ok_count = 0
    for u in collected:
        if u in seen:
            continue
        seen.add(u)
        try:
            res = ingest_url(u, namespace=namespace)
            if res.get("ok"):
                ok_count += 1
        except Exception:
            continue

    return {"ok": True, "found": len(collected), "ingested": ok_count}

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
    chosen: List[str] = []
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
