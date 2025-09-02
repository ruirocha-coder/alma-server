# rag_client.py — RAG client com deadlines + sitemap fix + verbose progress_cb
# -----------------------------------------------------------------------------
import os, io, re, time, gzip, hashlib
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import tiktoken

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from openai import OpenAI

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
QDRANT_URL        = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "alma_docs").strip()

EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
EMBED_DIM         = int(os.getenv("EMBED_DIM", "1536"))  # text-embedding-3-* = 1536

# chunking
MAX_TOKENS       = int(os.getenv("RAG_MAX_TOKENS_PER_CHUNK", "800"))
OVERLAP_TOKENS   = int(os.getenv("RAG_OVERLAP_TOKENS", "80"))
MIN_TOKENS       = int(os.getenv("RAG_MIN_TOKENS_TO_INGEST", "30"))

# crawling
CRAWL_MAX_PAGES  = int(os.getenv("CRAWL_MAX_PAGES", "30"))
CRAWL_MAX_DEPTH  = int(os.getenv("CRAWL_MAX_DEPTH", "2"))
LANG_PATH_ONLY   = os.getenv("RAG_LANG_PATH_ONLY", "").strip()  # ex: "/pt-pt/" ou "/pt/"

# deadlines
DEFAULT_DEADLINE_S = int(os.getenv("RAG_DEADLINE_S", "55"))

# retrieval
RAG_TOP_K                 = int(os.getenv("RAG_TOP_K", "6"))
RAG_CONTEXT_TOKEN_BUDGET  = int(os.getenv("RAG_CONTEXT_TOKEN_BUDGET", "1600"))
RAG_DEFAULT_NAMESPACE     = os.getenv("RAG_DEFAULT_NAMESPACE", "default").strip() or "default"

REQUEST_TIMEOUT = int(os.getenv("RAG_REQUEST_TIMEOUT", "30"))
HTTP_PROXY      = os.getenv("HTTP_PROXY", "").strip() or None

# -----------------------------------------------------------------------------
# Clients
# -----------------------------------------------------------------------------
oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=REQUEST_TIMEOUT)

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 AlmaRAG/1.2"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.6",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    })
    if HTTP_PROXY:
        s.proxies.update({"http": HTTP_PROXY, "https": HTTP_PROXY})
    s.max_redirects = 5
    return s

def _nop(*args, **kwargs):
    """Callback de progresso no-op (quando não é passado do servidor)."""
    pass

# -----------------------------------------------------------------------------
# Qdrant helpers
# -----------------------------------------------------------------------------
def ensure_collection() -> None:
    try:
        qdr.get_collection(QDRANT_COLLECTION)
        return
    except Exception:
        pass
    qdr.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

def _point_id(namespace: str, src_id: str, chunk_ix: int) -> str:
    raw = f"{namespace}:{src_id}:{chunk_ix}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def upsert_chunks(namespace: str, source_meta: Dict, chunks: List[str]) -> None:
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
    ensure_collection()
    must = [FieldCondition(key="source_id", match=MatchValue(value=source_id))]
    if namespace:
        must.append(FieldCondition(key="namespace", match=MatchValue(value=namespace)))
    res = qdr.delete(collection_name=QDRANT_COLLECTION, points_selector=Filter(must=must))
    return int(getattr(res, "status", 0) == "acknowledged")

# -----------------------------------------------------------------------------
# Tokenize & embed
# -----------------------------------------------------------------------------
def tokenize_len(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))

def chunk_text(text: str, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(text or "")
    if not ids:
        return []
    chunks, step = [], max(1, max_tokens - overlap)
    for start in range(0, len(ids), step):
        end = min(start + max_tokens, len(ids))
        chunks.append(enc.decode(ids[start:end]))
        if end >= len(ids):
            break
    return chunks

def embed_texts(texts: List[str], batch_size: int = 48) -> List[List[float]]:
    """
    OpenAI embeddings com batching + timeout defensivo por batch.
    batch_size 48 reduz risco de timeouts em redes mais lentas.
    """
    out: List[List[float]] = []
    if not texts:
        return out
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # timeout curto por batch para não estourar o limite do proxy
        resp = oai.embeddings.create(model=EMBEDDING_MODEL, input=batch, timeout=20)
        out.extend([d.embedding for d in resp.data])
    return out

# -----------------------------------------------------------------------------
# HTML helpers
# -----------------------------------------------------------------------------
def _clean_url(u: str) -> str:
    return (u or "").split("#")[0].strip()

def _is_same_domain(seed: str, url: str) -> bool:
    a, b = urlparse(seed), urlparse(url)
    return a.netloc == b.netloc

def _extract_title(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        t = soup.title.get_text(strip=True) if soup.title else ""
        return t or ""
    except Exception:
        return ""

def _extract_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        absu = urljoin(base_url, href)
        out.append(_clean_url(absu))
    # dedup básico
    seen = set(); uniq = []
    for u in out:
        if u and u not in seen:
            uniq.append(u); seen.add(u)
    return uniq

def _extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "form", "noscript"]):
        tag.decompose()
    main = soup.find(["main"]) or soup.find(attrs={"role": "main"}) or soup.find("article")
    text = main.get_text(separator=" ") if main else soup.get_text(separator=" ")
    return " ".join(text.split())

def _get(session: requests.Session, url: str) -> Tuple[Optional[str], Optional[str], int]:
    """Devolve (text, content_type, status). Faz pequeno backoff a 403/429."""
    for attempt in range(3):
        r = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        ctype = r.headers.get("Content-Type", "")
        if r.status_code in (403, 429):
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code >= 400:
            return None, ctype, r.status_code
        # gzip manual (alguns CDNs mandam gz sem header adequado)
        if r.content and (ctype.endswith("gzip") or url.endswith(".gz")):
            try:
                data = gzip.decompress(r.content).decode("utf-8", errors="ignore")
                return data, ctype, r.status_code
            except Exception:
                pass
        text = r.text
        return text, ctype, r.status_code
    return None, None, 599

# -----------------------------------------------------------------------------
# Ingest: TEXT
# -----------------------------------------------------------------------------
def ingest_text(title: str, text: str, namespace: str = RAG_DEFAULT_NAMESPACE, source_id: str = None) -> Dict:
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "Texto vazio"}
    chunks = chunk_text(text)
    src_id = source_id or hashlib.sha1((title + text[:200]).encode("utf-8")).hexdigest()
    upsert_chunks(namespace, {"type": "text", "title": title, "source_id": src_id}, chunks)
    return {"ok": True, "chunks": len(chunks), "source_id": src_id}

# -----------------------------------------------------------------------------
# Ingest: PDF URL / bytes  (com deadline opcional)
# -----------------------------------------------------------------------------
def fetch_pdf_text(pdf_url: str) -> Tuple[str, int]:
    s = _session()
    r = s.get(pdf_url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    reader = PdfReader(io.BytesIO(r.content))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\n".join(pages)
    return text, len(reader.pages)

def ingest_pdf_url(pdf_url: str, title: str = None, namespace: str = RAG_DEFAULT_NAMESPACE,
                   deadline_s: Optional[int] = None, progress_cb=None) -> Dict:
    progress = progress_cb or _nop
    t0 = time.time()
    text, n_pages = fetch_pdf_text(pdf_url)
    if deadline_s and (time.time() - t0) > deadline_s:
        return {"ok": True, "pages": n_pages, "timed_out": True}
    title = title or pdf_url.split("/")[-1]
    src_id = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()
    chunks = chunk_text(text)
    upsert_chunks(namespace, {"type": "pdf", "title": title, "url": pdf_url, "pages": n_pages, "source_id": src_id}, chunks)
    progress({"event": "ingest_pdf", "url": pdf_url, "title": title})
    return {"ok": True, "chunks": len(chunks), "pages": n_pages, "source_id": src_id}

def ingest_pdf_bytes(data: bytes, title: str, namespace: str = RAG_DEFAULT_NAMESPACE) -> Dict:
    reader = PdfReader(io.BytesIO(data))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\n".join(pages)
    src_id = hashlib.sha1((title + str(len(data))).encode("utf-8")).hexdigest()
    chunks = chunk_text(text)
    upsert_chunks(namespace, {"type": "pdf", "title": title, "pages": len(pages), "source_id": src_id}, chunks)
    return {"ok": True, "chunks": len(chunks), "pages": len(pages), "source_id": src_id}

# -----------------------------------------------------------------------------
# Ingest: Single URL (HTML) — com deadline e verbose progress_cb
# -----------------------------------------------------------------------------
def ingest_url(page_url: str, namespace: str = RAG_DEFAULT_NAMESPACE,
               deadline_s: Optional[int] = None, progress_cb=None) -> Dict:
    progress = progress_cb or _nop
    t0 = time.time()
    u = _clean_url(page_url)
    progress({"event": "fetch", "url": u})

    s = _session()
    html, ctype, status = _get(s, u)
    if not html or "text/html" not in (ctype or ""):
        return {"ok": False, "error": "Página não-HTML", "url": u, "status": status}

    if deadline_s and (time.time() - t0) > deadline_s:
        return {"ok": True, "timed_out": True, "url": u}

    text = _extract_main_text(html)
    if tokenize_len(text) < MIN_TOKENS:
        return {"ok": False, "error": "Página sem conteúdo útil suficiente", "url": u}

    title = _extract_title(html) or u
    src_id = hashlib.sha1(u.encode("utf-8")).hexdigest()
    chunks = chunk_text(text)
    upsert_chunks(namespace, {"type": "web", "title": title, "url": u,
                              "domain": urlparse(u).netloc, "source_id": src_id}, chunks)
    progress({"event": "ingest", "url": u, "title": title})
    return {"ok": True, "chunks": len(chunks), "source_id": src_id, "url": u, "title": title}

# -----------------------------------------------------------------------------
# Ingest: Crawl website (BFS) — com deadline e progress_cb
# -----------------------------------------------------------------------------
def crawl_and_ingest(seed_url: str,
                     namespace: str = RAG_DEFAULT_NAMESPACE,
                     max_pages: int = CRAWL_MAX_PAGES,
                     max_depth: int = CRAWL_MAX_DEPTH,
                     deadline_s: Optional[int] = None,
                     progress_cb=None) -> Dict:
    """
    Crawl BFS com hard deadline: devolve parcial com {"timed_out": True} se exceder.
    """
    progress = progress_cb or _nop
    t0 = time.time()

    s = _session()
    seen = set()
    queue = [(seed_url, 0)]
    n_ingested = 0
    src_host = urlparse(seed_url).netloc

    progress({"event": "start_crawl", "seed": seed_url, "namespace": namespace})

    while queue and n_ingested < max_pages:
        if deadline_s and (time.time() - t0) > deadline_s:
            return {"ok": True, "pages_ingested": n_ingested, "visited": len(seen),
                    "domain": src_host, "timed_out": True}

        url, depth = queue.pop(0)
        url = _clean_url(url)
        if url in seen:
            continue
        seen.add(url)

        # filtro por língua (ex.: apenas /pt-pt/)
        if LANG_PATH_ONLY and LANG_PATH_ONLY not in url:
            continue

        progress({"event": "visit", "url": url, "depth": depth})
        html, ctype, status = _get(s, url)
        if not html or "text/html" not in (ctype or ""):
            continue

        title = _extract_title(html) or url
        text = _extract_main_text(html)
        if tokenize_len(text) >= MIN_TOKENS:
            src_id = hashlib.sha1(url.encode("utf-8")).hexdigest()
            chunks = chunk_text(text)
            upsert_chunks(
                namespace,
                {"type": "web", "title": title, "url": url, "domain": src_host, "source_id": src_id},
                chunks,
            )
            n_ingested += 1
            progress({"event": "ingest", "url": url, "title": title})

        if depth < max_depth:
            for link in _extract_links(url, html):
                if _is_same_domain(seed_url, link):
                    if LANG_PATH_ONLY and LANG_PATH_ONLY not in link:
                        continue
                    queue.append((link, depth + 1))
                    progress({"event": "queue", "url": link, "depth": depth + 1})

    return {"ok": True, "pages_ingested": n_ingested, "visited": len(seen), "domain": src_host}

# -----------------------------------------------------------------------------
# Ingest: Sitemap (recursivo) — FIX: filtrar idioma só nas páginas (não nos childs)
# -----------------------------------------------------------------------------
def _parse_sitemap_xml(xml_text: str) -> Tuple[List[str], List[str]]:
    """Devolve (child_sitemaps, page_urls)."""
    soup = BeautifulSoup(xml_text, "xml")
    child_maps = [loc.get_text(strip=True) for sm in soup.find_all("sitemap") for loc in sm.find_all("loc")]
    page_urls = [loc.get_text(strip=True) for url in soup.find_all("url") for loc in url.find_all("loc")]
    return child_maps, page_urls

def ingest_sitemap(sitemap_url: str,
                   namespace: str = RAG_DEFAULT_NAMESPACE,
                   max_pages: int = CRAWL_MAX_PAGES,
                   deadline_s: Optional[int] = None,
                   progress_cb=None) -> Dict:
    """
    Aceita URL do sitemap (ou homepage). Segue sitemapindex → urlset.
    Corrige filtro de idioma: aplica-se APENAS às páginas, NUNCA aos child sitemaps.
    Hard deadline e devolve parcial com {"timed_out": True}.
    """
    progress = progress_cb or _nop
    t0 = time.time()
    s = _session()

    # autodetect se vier homepage
    start = sitemap_url.strip()
    if not start.endswith(".xml"):
        base = start.rstrip("/")
        # tenta /sitemap_index.xml (Yoast) e depois /sitemap.xml
        for cand in (base + "/sitemap_index.xml", base + "/sitemap.xml"):
            progress({"event": "open_sitemap", "url": cand})
            xml, ctype, status = _get(s, cand)
            if xml and any(t in (ctype or "") for t in ("xml", "text/xml", "application/xml", "gzip")):
                start = cand
                break

    progress({"event": "open_sitemap", "url": start})
    xml, ctype, status = _get(s, start)
    if not xml:
        return {"ok": False, "error": "Falha a abrir sitemap", "url": start, "status": status}

    to_visit = [start]
    page_list: List[str] = []
    seen_maps = set()

    # 1) recolhe URLs reais (seguindo índices)
    while to_visit and len(page_list) < max_pages * 5:
        if deadline_s and (time.time() - t0) > deadline_s:
            break
        sm = to_visit.pop(0)
        if sm in seen_maps:
            continue
        seen_maps.add(sm)

        xml, ctype, status = _get(s, sm)
        if not xml or not any(t in (ctype or "") for t in ("xml", "text/xml", "application/xml", "gzip")):
            continue

        childs, pages = _parse_sitemap_xml(xml)

        # ⚠️ filtro de língua SÓ nas páginas, nunca nos child sitemaps
        if LANG_PATH_ONLY:
            pages = [u for u in pages if LANG_PATH_ONLY in u]

        page_list.extend(pages)
        # não filtra childs — precisamos de os abrir para chegar às páginas PT
        to_visit.extend(childs)
        progress({"event": "collected", "sitemap": sm, "pages": len(pages), "childs": len(childs)})

    # 2) visita páginas e ingere
    n_ingested = 0
    s_host = urlparse(start).netloc
    unique_pages = []
    seen_p = set()
    for u in page_list:
        u = _clean_url(u)
        if u in seen_p:
            continue
        seen_p.add(u)
        unique_pages.append(u)

    for url in unique_pages:
        if n_ingested >= max_pages:
            break
        if deadline_s and (time.time() - t0) > deadline_s:
            return {"ok": True, "pages_ingested": n_ingested, "collected": len(unique_pages),
                    "domain": s_host, "timed_out": True}

        html, ctype, status = _get(s, url)
        if not html or "text/html" not in (ctype or ""):
            continue
        title = _extract_title(html) or url
        text = _extract_main_text(html)
        if tokenize_len(text) < MIN_TOKENS:
            continue
        src_id = hashlib.sha1(url.encode("utf-8")).hexdigest()
        chunks = chunk_text(text)
        upsert_chunks(namespace, {"type": "web", "title": title, "url": url,
                                  "domain": s_host, "source_id": src_id}, chunks)
        n_ingested += 1
        progress({"event": "ingest", "url": url, "title": title})

    return {"ok": True, "pages_ingested": n_ingested, "collected": len(unique_pages), "domain": s_host}

# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------
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
    budget = token_budget or RAG_CONTEXT_TOKEN_BUDGET
    enc = tiktoken.get_encoding("cl100k_base")
    chosen, total = [], 0
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
