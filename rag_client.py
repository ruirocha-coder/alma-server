# rag_client.py — RAG client com deadline + verbose + fallbacks
# -------------------------------------------------------------
import os, io, re, time, hashlib, gzip, math
from typing import List, Dict, Tuple, Optional, Callable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# pypdf é opcional (não quebra se faltar)
try:
    from pypdf import PdfReader
    _PDF_OK = True
except Exception:
    _PDF_OK = False

# tiktoken é opcional; se faltar, usa contador simples
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def _tok_count(s: str) -> int: return len(_ENC.encode(s or ""))
except Exception:
    def _tok_count(s: str) -> int: return max(1, len((s or "").split()))

# -----------------------------
# Config
# -----------------------------
QDRANT_URL        = (os.getenv("QDRANT_URL") or "").strip()
QDRANT_API_KEY    = (os.getenv("QDRANT_API_KEY") or "").strip()
QDRANT_COLLECTION = (os.getenv("QDRANT_COLLECTION") or "alma_docs").strip()

EMBEDDING_MODEL   = (os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small").strip()
EMBED_DIM         = int(os.getenv("EMBED_DIM") or "1536")

MAX_TOKENS        = int(os.getenv("RAG_MAX_TOKENS_PER_CHUNK") or "800")
OVERLAP_TOKENS    = int(os.getenv("RAG_OVERLAP_TOKENS") or "80")
MIN_TOKENS        = int(os.getenv("RAG_MIN_TOKENS_TO_INGEST") or "30")

CRAWL_MAX_PAGES   = int(os.getenv("CRAWL_MAX_PAGES") or "30")
CRAWL_MAX_DEPTH   = int(os.getenv("CRAWL_MAX_DEPTH") or "2")
LANG_PATH_ONLY    = (os.getenv("RAG_LANG_PATH_ONLY") or "").strip()  # ex.: "/pt-pt/"

RAG_TOP_K                = int(os.getenv("RAG_TOP_K") or "6")
RAG_CONTEXT_TOKEN_BUDGET = int(os.getenv("RAG_CONTEXT_TOKEN_BUDGET") or "1600")
RAG_DEFAULT_NAMESPACE    = (os.getenv("RAG_DEFAULT_NAMESPACE") or "default").strip()

REQUEST_TIMEOUT   = int(os.getenv("RAG_REQUEST_TIMEOUT") or "8")      # por URL
HTTP_PROXY        = (os.getenv("HTTP_PROXY") or "").strip() or None

# -----------------------------
# Clientes (com fallback)
# -----------------------------
_qdrant_ready = False
_inmem_points: List[Dict] = []   # fallback em RAM

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    if QDRANT_URL:
        qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=REQUEST_TIMEOUT)
        _qdrant_ready = True
    else:
        qdr = None
except Exception:
    qdr = None
    _qdrant_ready = False

# OpenAI embeddings (com fallback determinístico)
try:
    from openai import OpenAI
    _oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    _oai_ok = True
except Exception:
    _oai = None
    _oai_ok = False

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 AlmaRAG/1.2",
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

# -----------------------------
# Qdrant helpers (ou RAM)
# -----------------------------
def ensure_collection() -> None:
    if not _qdrant_ready:  # fallback RAM
        return
    try:
        qdr.get_collection(QDRANT_COLLECTION)
    except Exception:
        qdr.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

def _point_id(namespace: str, src_id: str, chunk_ix: int) -> str:
    return hashlib.sha1(f"{namespace}:{src_id}:{chunk_ix}".encode("utf-8")).hexdigest()

def _cosine(a: List[float], b: List[float]) -> float:
    # segura e leve (sem numpy)
    s = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(y*y for y in b)) or 1.0
    return s / (na*nb)

# -----------------------------
# Tokenize & embed
# -----------------------------
def chunk_text(text: str, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS) -> List[str]:
    if not text:
        return []
    # com tiktoken fica melhor; aqui usamos _tok_count para decidir os “cortes”
    words = text.split()
    if not words:
        return []
    chunks, cur, cur_len = [], [], 0
    for w in words:
        cur.append(w)
        cur_len += 1
        if cur_len >= max_tokens:
            chunks.append(" ".join(cur))
            # overlap
            cur = cur[-overlap:] if overlap>0 else []
            cur_len = len(cur)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def embed_texts(texts: List[str], batch_size: int = 96) -> List[List[float]]:
    if not texts:
        return []
    # OpenAI se disponível
    if _oai_ok:
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = _oai.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out
    # Fallback: embedding determinístico por SHA1 (mesmo dim, números pequenos)
    out = []
    for t in texts:
        h = hashlib.sha1((t or "").encode("utf-8")).digest()
        # repete hash para preencher EMBED_DIM
        vec = []
        while len(vec) < EMBED_DIM:
            for b in h:
                vec.append(((b / 255.0) - 0.5) * 2.0)
                if len(vec) >= EMBED_DIM:
                    break
        out.append(vec)
    return out

def upsert_chunks(namespace: str, source_meta: Dict, chunks: List[str]) -> None:
    if not chunks:
        return
    vectors = embed_texts(chunks)
    if _qdrant_ready:
        ensure_collection()
        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            pid = _point_id(namespace, source_meta.get("source_id", ""), i)
            payload = {"text": chunk, "namespace": namespace, **source_meta}
            points.append(PointStruct(id=pid, vector=vec, payload=payload))
        if points:
            qdr.upsert(collection_name=QDRANT_COLLECTION, points=points)
    else:
        # RAM fallback
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            pid = _point_id(namespace, source_meta.get("source_id", ""), i)
            payload = {"text": chunk, "namespace": namespace, **source_meta}
            _inmem_points.append({"id": pid, "vector": vec, "payload": payload})

# -----------------------------
# HTTP helpers
# -----------------------------
def _clean_url(u: str) -> str: return (u or "").split("#")[0]

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
        absu = urljoin(base_url, a["href"].strip())
        out.append(_clean_url(absu))
    # dedup
    seen, uniq = set(), []
    for u in out:
        if u and u not in seen:
            uniq.append(u); seen.add(u)
    return uniq

def _extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "form"]):
        tag.decompose()
    main = soup.find(["main"]) or soup.find(attrs={"role": "main"}) or soup.find("article")
    text = main.get_text(separator=" ") if main else soup.get_text(separator=" ")
    return " ".join(text.split())

def _get(session: requests.Session, url: str, timeout: Optional[int] = None) -> Tuple[Optional[str], Optional[str], int]:
    per_url_timeout = timeout or REQUEST_TIMEOUT
    for attempt in range(3):
        r = session.get(url, timeout=per_url_timeout, allow_redirects=True)
        ctype = r.headers.get("Content-Type", "") or ""
        if r.status_code in (403, 429):
            time.sleep(1.2 * (attempt + 1)); continue
        if r.status_code >= 400:
            return None, ctype, r.status_code
        if r.content and (ctype.endswith("gzip") or url.endswith(".gz")):
            try:
                data = gzip.decompress(r.content).decode("utf-8", errors="ignore")
                return data, ctype, r.status_code
            except Exception:
                pass
        return r.text, ctype, r.status_code
    return None, None, 599

def _emit(cb: Optional[Callable], kind: str, **fields):
    if not callable(cb): return
    ev = {"t": time.time(), "kind": kind}
    ev.update({k:v for k,v in fields.items() if v is not None})
    try: cb(ev)
    except Exception: pass

# -----------------------------
# Ingest: TEXT / URL / PDF
# -----------------------------
def ingest_text(title: str, text: str, namespace: str = RAG_DEFAULT_NAMESPACE, source_id: str = None) -> Dict:
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "Texto vazio"}
    chunks = chunk_text(text)
    src_id = source_id or hashlib.sha1((title + text[:200]).encode("utf-8")).hexdigest()
    upsert_chunks(namespace, {"type": "text", "title": title, "source_id": src_id}, chunks)
    return {"ok": True, "chunks": len(chunks), "source_id": src_id}

def ingest_url(page_url: str, namespace: str = RAG_DEFAULT_NAMESPACE, deadline_s: int = 55) -> Dict:
    s = _session()
    t0 = time.time()
    try:
        html, ctype, _ = _get(s, _clean_url(page_url), timeout=REQUEST_TIMEOUT)
    except Exception as e:
        return {"ok": False, "error": "http_error", "detail": str(e)}
    if not html or "text/html" not in (ctype or ""):
        return {"ok": False, "error": "not_html", "content_type": ctype}
    title = _extract_title(html) or page_url
    text = _extract_main_text(html)
    if _tok_count(text) < MIN_TOKENS:
        return {"ok": False, "error": "short_page"}
    src_id = hashlib.sha1(page_url.encode("utf-8")).hexdigest()
    chunks = chunk_text(text)
    upsert_chunks(namespace, {"type": "web", "title": title, "url": page_url, "domain": urlparse(page_url).netloc, "source_id": src_id}, chunks)
    return {"ok": True, "chunks": len(chunks), "elapsed_s": round(time.time()-t0, 3), "source_id": src_id}

def fetch_pdf_text(pdf_url: str) -> Tuple[str, int]:
    if not _PDF_OK:
        raise RuntimeError("pypdf não instalado")
    s = _session()
    r = s.get(pdf_url, timeout=max(REQUEST_TIMEOUT, 15))
    r.raise_for_status()
    reader = PdfReader(io.BytesIO(r.content))
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n\n".join(pages), len(reader.pages)

def ingest_pdf_url(pdf_url: str, title: str = None, namespace: str = RAG_DEFAULT_NAMESPACE) -> Dict:
    try:
        text, n_pages = fetch_pdf_text(pdf_url)
    except Exception as e:
        return {"ok": False, "error": "pdf_error", "detail": str(e)}
    title = title or pdf_url.split("/")[-1]
    src_id = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()
    chunks = chunk_text(text)
    upsert_chunks(namespace, {"type": "pdf", "title": title, "url": pdf_url, "pages": n_pages, "source_id": src_id}, chunks)
    return {"ok": True, "chunks": len(chunks), "pages": n_pages, "source_id": src_id}

# -----------------------------
# Crawl BFS (deadline + verbose)
# -----------------------------
def crawl_and_ingest(seed_url: str,
                     namespace: str = RAG_DEFAULT_NAMESPACE,
                     max_pages: int = CRAWL_MAX_PAGES,
                     max_depth: int = CRAWL_MAX_DEPTH,
                     deadline_s: int = 55,
                     progress_cb: Optional[Callable] = None) -> Dict:
    s = _session()
    started = time.time()
    deadline_at = started + max(5, int(deadline_s))
    seen = set()
    queue: List[Tuple[str, int]] = [(_clean_url(seed_url), 0)]
    n_ingested = 0
    src_host = urlparse(seed_url).netloc

    _emit(progress_cb, "start", url=seed_url, namespace=namespace, max_pages=max_pages, max_depth=max_depth, deadline_s=deadline_s)

    while queue and n_ingested < max_pages:
        if time.time() >= deadline_at:
            _emit(progress_cb, "deadline", pages_ingested=n_ingested, seen=len(seen))
            return {"ok": True, "pages_ingested": n_ingested, "visited": len(seen), "domain": src_host, "timed_out": True}

        url, depth = queue.pop(0)
        url = _clean_url(url)
        if url in seen: continue
        seen.add(url)

        if LANG_PATH_ONLY and LANG_PATH_ONLY not in url:
            _emit(progress_cb, "skip_lang", url=url); continue

        _emit(progress_cb, "visit", url=url, depth=depth)

        try:
            html, ctype, status = _get(s, url, timeout=REQUEST_TIMEOUT)
        except Exception as e:
            _emit(progress_cb, "error_get", url=url, msg=str(e)); continue
        if not html or "text/html" not in (ctype or ""):
            _emit(progress_cb, "skip_non_html", url=url, ctype=ctype, status=status); continue

        title = _extract_title(html) or url
        text = _extract_main_text(html)
        if _tok_count(text) >= MIN_TOKENS:
            try:
                src_id = hashlib.sha1(url.encode("utf-8")).hexdigest()
                chunks = chunk_text(text)
                upsert_chunks(namespace, {"type": "web", "title": title, "url": url, "domain": src_host, "source_id": src_id}, chunks)
                n_ingested += 1
                _emit(progress_cb, "ingest", url=url, chunks=len(chunks), title=title)
            except Exception as e:
                _emit(progress_cb, "error_upsert", url=url, msg=str(e))

        if depth < max_depth:
            try:
                for link in _extract_links(url, html):
                    if _is_same_domain(seed_url, link):
                        if LANG_PATH_ONLY and LANG_PATH_ONLY not in link: continue
                        queue.append((link, depth + 1))
                _emit(progress_cb, "queue_size", size=len(queue))
            except Exception as e:
                _emit(progress_cb, "error_links", url=url, msg=str(e))

        if (n_ingested % 5) == 0: time.sleep(0.01)

    return {"ok": True, "pages_ingested": n_ingested, "visited": len(seen), "domain": src_host, "timed_out": False}

# -----------------------------
# Sitemap (deadline + verbose)
# -----------------------------
def _parse_sitemap_xml(xml_text: str) -> Tuple[List[str], List[str]]:
    soup = BeautifulSoup(xml_text, "xml")
    # urlset
    url_locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
    # podemos ter duplicados; o “child maps” distinguimos por heurística
    child_maps, page_urls = [], []
    for u in url_locs:
        if u.endswith(".xml"):
            child_maps.append(u)
        else:
            page_urls.append(u)
    return child_maps, page_urls

def ingest_sitemap(sitemap_url: str,
                   namespace: str = RAG_DEFAULT_NAMESPACE,
                   max_pages: int = CRAWL_MAX_PAGES,
                   deadline_s: int = 55,
                   progress_cb: Optional[Callable] = None) -> Dict:
    s = _session()
    started = time.time()
    deadline_at = started + max(5, int(deadline_s))
    start = sitemap_url.strip()
    if not start.endswith(".xml"):
        base = start.rstrip("/")
        start = base + "/sitemap.xml"

    _emit(progress_cb, "smap_start", url=start, namespace=namespace, max_pages=max_pages)

    to_visit = [start]
    page_list: List[str] = []
    seen_maps = set()

    while to_visit and len(page_list) < max_pages * 5:
        if time.time() >= deadline_at:
            _emit(progress_cb, "deadline_urls", collected=len(page_list))
            break
        sm = to_visit.pop(0)
        if sm in seen_maps: continue
        seen_maps.add(sm)

        try:
            xml, ctype, status = _get(s, sm, timeout=REQUEST_TIMEOUT)
        except Exception as e:
            _emit(progress_cb, "error_smap_get", url=sm, msg=str(e))
            continue
        if not xml or not any(t in (ctype or "") for t in ("xml", "text/xml", "application/xml", "gzip")):
            _emit(progress_cb, "skip_not_xml", url=sm, ctype=ctype)
            continue

        childs, pages = _parse_sitemap_xml(xml)
        if LANG_PATH_ONLY:
            pages  = [u for u in pages if LANG_PATH_ONLY in u]
            childs = [u for u in childs if LANG_PATH_ONLY in u]
        to_visit.extend(childs)
        for u in pages:
            uu = _clean_url(u)
            if uu not in page_list:
                page_list.append(uu)

        _emit(progress_cb, "smap_collect", url=sm, pages=len(pages), childs=len(childs), total=len(page_list))

    # visitar páginas
    n_ingested = 0
    s_host = urlparse(start).netloc
    for url in page_list:
        if n_ingested >= max_pages:
            break
        if time.time() >= deadline_at:
            _emit(progress_cb, "deadline_ingest", done=n_ingested, total=len(page_list))
            return {"ok": True, "pages_ingested": n_ingested, "collected": len(page_list), "domain": s_host, "timed_out": True}

        try:
            html, ctype, status = _get(s, url, timeout=REQUEST_TIMEOUT)
        except Exception as e:
            _emit(progress_cb, "error_get_page", url=url, msg=str(e)); continue
        if not html or "text/html" not in (ctype or ""):
            _emit(progress_cb, "skip_non_html", url=url, ctype=ctype); continue

        title = _extract_title(html) or url
        text = _extract_main_text(html)
        if _tok_count(text) < MIN_TOKENS:
            _emit(progress_cb, "skip_short", url=url); continue

        try:
            src_id = hashlib.sha1(url.encode("utf-8")).hexdigest()
            chunks = chunk_text(text)
            upsert_chunks(namespace, {"type": "web", "title": title, "url": url, "domain": s_host, "source_id": src_id}, chunks)
            n_ingested += 1
            _emit(progress_cb, "ingest", url=url, chunks=len(chunks), title=title)
        except Exception as e:
            _emit(progress_cb, "error_upsert", url=url, msg=str(e))

    return {"ok": True, "pages_ingested": n_ingested, "collected": len(page_list), "domain": s_host, "timed_out": False}

# -----------------------------
# Search + contexto
# -----------------------------
def search_chunks(query: str, namespace: Optional[str] = None, top_k: int = None) -> List[Dict]:
    query = (query or "").strip()
    if not query:
        return []
    qvec = embed_texts([query])[0]
    limit = top_k or RAG_TOP_K

    # Qdrant
    if _qdrant_ready:
        flt = None
        if namespace:
            flt = Filter(must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))])
        hits = qdr.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=qvec,
            limit=limit,
            query_filter=flt,
            with_payload=True,
            with_vectors=False,
            score_threshold=None,
        )
        out = []
        for h in hits:
            pl = h.payload or {}
            out.append({
                "text": pl.get("text",""),
                "score": float(h.score),
                "title": pl.get("title",""),
                "url": pl.get("url"),
                "type": pl.get("type"),
                "namespace": pl.get("namespace"),
                "source_id": pl.get("source_id"),
            })
        return out

    # Fallback RAM
    cand = []
    for p in _inmem_points:
        pl = p["payload"]
        if namespace and pl.get("namespace") != namespace:
            continue
        cand.append({
            "score": _cosine(qvec, p["vector"]),
            "payload": pl
        })
    cand.sort(key=lambda x: x["score"], reverse=True)
    out = []
    for c in cand[:limit]:
        pl = c["payload"]
        out.append({
            "text": pl.get("text",""),
            "score": float(c["score"]),
            "title": pl.get("title",""),
            "url": pl.get("url"),
            "type": pl.get("type"),
            "namespace": pl.get("namespace"),
            "source_id": pl.get("source_id"),
        })
    return out

def build_context_block(matches: List[Dict], token_budget: int = None) -> str:
    budget = token_budget or RAG_CONTEXT_TOKEN_BUDGET
    chosen, total = [], 0
    for m in matches:
        head = m.get("title") or m.get("url") or m.get("type") or "doc"
        chunk = f"[{head}] {m.get('text','')}".strip()
        cost = _tok_count(chunk)
        if total + cost > budget: break
        chosen.append(chunk); total += cost
    if not chosen: return ""
    return "Contexto de conhecimento interno (RAG):\n" + "\n\n".join(chosen)
