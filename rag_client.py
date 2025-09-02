# rag_client.py
import os, io, re, time, gzip, hashlib
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin, urlparse, quote_plus

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import tiktoken

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from openai import OpenAI

# ───────────────────────────────────────── Config
QDRANT_URL        = (os.getenv("QDRANT_URL", "") or "").strip()
QDRANT_API_KEY    = (os.getenv("QDRANT_API_KEY", "") or "").strip()
QDRANT_COLLECTION = (os.getenv("QDRANT_COLLECTION", "alma_docs") or "alma_docs").strip()

EMBEDDING_MODEL   = (os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small").strip()
EMBED_DIM         = int(os.getenv("EMBED_DIM", "1536"))

MAX_TOKENS        = int(os.getenv("RAG_MAX_TOKENS_PER_CHUNK", "800"))
OVERLAP_TOKENS    = int(os.getenv("RAG_OVERLAP_TOKENS", "80"))
MIN_TOKENS        = int(os.getenv("RAG_MIN_TOKENS_TO_INGEST", "10"))

CRAWL_MAX_PAGES   = int(os.getenv("CRAWL_MAX_PAGES", "40"))
CRAWL_MAX_DEPTH   = int(os.getenv("CRAWL_MAX_DEPTH", "2"))

# Ex.: "/pt-pt/" (Boa Safra), "/pt/" (Interior Guider). Opcional.
LANG_PATH_ONLY    = (os.getenv("RAG_LANG_PATH_ONLY", "") or "").strip()

# Listas de paths (regex) opcionais por tipo de site
ALLOW_PATHS       = (os.getenv("RAG_ALLOW_PATHS", "") or "").split("|") if os.getenv("RAG_ALLOW_PATHS") else []
DENY_PATHS        = (os.getenv("RAG_DENY_PATHS", "") or "(/cart|/checkout|/account|/wp-json|/feed|/tag/|/author/|\\?add-to-cart=)").split("|")

RAG_TOP_K                 = int(os.getenv("RAG_TOP_K", "6"))
RAG_CONTEXT_TOKEN_BUDGET  = int(os.getenv("RAG_CONTEXT_TOKEN_BUDGET", "1600"))
RAG_DEFAULT_NAMESPACE     = (os.getenv("RAG_DEFAULT_NAMESPACE", "default") or "default").strip()

REQUEST_TIMEOUT   = int(os.getenv("RAG_REQUEST_TIMEOUT", "30"))

# Proxy: usa UM dos dois formatos
HTTP_PROXY_BASE   = (os.getenv("HTTP_PROXY_BASE", "") or "").strip()  # ex.: https://.../api?api_key=XXX&url=
HTTP_PROXY        = (os.getenv("HTTP_PROXY", "") or "").strip()       # ex.: http://user:pass@host:port

# Headless render (opcional) para páginas JS: ex.: http://render:3000/snapshot?url=
RENDER_SERVICE_URL = (os.getenv("RENDER_SERVICE_URL", "") or "").strip()

oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=REQUEST_TIMEOUT)

# ───────────────────────────────────────── Utils Qdrant/Embeddings
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

def embed_texts(texts: List[str], batch_size: int = 96) -> List[List[float]]:
    out: List[List[float]] = []
    if not texts: return out
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = oai.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out

def _point_id(namespace: str, src_id: str, chunk_ix: int) -> str:
    return hashlib.sha1(f"{namespace}:{src_id}:{chunk_ix}".encode("utf-8")).hexdigest()

def upsert_chunks(namespace: str, source_meta: Dict, chunks: List[str]) -> None:
    if not chunks: return
    ensure_collection()
    vectors = embed_texts(chunks)
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        pid = _point_id(namespace, source_meta.get("source_id", ""), i)
        payload = {"text": chunk, "namespace": namespace, **source_meta}
        points.append(PointStruct(id=pid, vector=vec, payload=payload))
    if points:
        qdr.upsert(collection_name=QDRANT_COLLECTION, points=points)

# ───────────────────────────────────────── HTTP / Fetch
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36 AlmaRAG/1.2",
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

def _proxied(url: str) -> str:
    if not HTTP_PROXY_BASE: return url
    if HTTP_PROXY_BASE.endswith("="):
        return HTTP_PROXY_BASE + quote_plus(url)
    joiner = "&" if "?" in HTTP_PROXY_BASE else "?"
    return f"{HTTP_PROXY_BASE}{joiner}url={quote_plus(url)}"

def _clean_url(u: str) -> str:
    return (u or "").split("#")[0].strip()

def _is_same_domain(seed: str, url: str) -> bool:
    a, b = urlparse(seed), urlparse(url)
    return a.netloc == b.netloc

def _get(session: requests.Session, url: str) -> Tuple[Optional[str], Optional[str], int]:
    try:
        r = session.get(_proxied(url), timeout=REQUEST_TIMEOUT, allow_redirects=True)
    except Exception:
        return None, None, 599
    ctype = r.headers.get("Content-Type", "")
    if r.content and (ctype.endswith("gzip") or url.endswith(".gz")):
        try:
            data = gzip.decompress(r.content).decode("utf-8", errors="ignore")
            return data, ctype, r.status_code
        except Exception:
            pass
    return (r.text, ctype, r.status_code)

def _rendered_html(url: str) -> Optional[str]:
    if not RENDER_SERVICE_URL: return None
    s = _session()
    try:
        # assume RENDER_SERVICE_URL já inclui '?url=' ou que aceitamos anexar
        endpoint = RENDER_SERVICE_URL
        if "{url}" in endpoint:
            endpoint = endpoint.replace("{url}", quote_plus(url))
        elif endpoint.endswith("="):
            endpoint = endpoint + quote_plus(url)
        elif "url=" not in endpoint:
            joiner = "&" if "?" in endpoint else "?"
            endpoint = f"{endpoint}{joiner}url={quote_plus(url)}"
        r = s.get(endpoint, timeout=min(REQUEST_TIMEOUT, 25))
        if r.status_code == 200 and r.text and "<html" in r.text.lower():
            return r.text
    except Exception:
        return None
    return None

# ───────────────────────────────────────── Parse helpers
def _extract_title(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        t = soup.title.get_text(strip=True) if soup.title else ""
        return (t or "").strip()
    except Exception:
        return ""

def _extract_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: List[str] = []
    # links normais
    for a in soup.find_all("a", href=True):
        absu = urljoin(base_url, a["href"].strip())
        out.append(_clean_url(absu))
    # hreflang alternates (útil para Weglot/WordPress)
    for link in soup.find_all("link", rel=lambda v: v and "alternate" in v):
        href = link.get("href")
        if href:
            absu = urljoin(base_url, href.strip())
            out.append(_clean_url(absu))
    # dedup
    seen, uniq = set(), []
    for u in out:
        if u and u not in seen:
            uniq.append(u); seen.add(u)
    return uniq

def _match_any(patterns: List[str], path: str) -> bool:
    if not patterns: return False
    for p in patterns:
        try:
            if re.search(p, path):
                return True
        except Exception:
            continue
    return False

def _extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "form", "noscript"]):
        tag.decompose()

    candidates = []

    # Landmarks
    main = soup.find("main")
    if main: candidates.append(main)
    rmain = soup.find(attrs={"role": "main"})
    if rmain: candidates.append(rmain)
    article = soup.find("article")
    if article: candidates.append(article)

    # WordPress
    wp_sels = [".entry-content", ".wp-block-post-content", ".site-content", ".page-content", ".hentry", ".post-content"]
    for sel in wp_sels:
        candidates.extend(soup.select(sel))

    # BigCommerce comuns (product/category/cms)
    bc_sels = [".productView", ".productView-description", ".page", ".page-content", ".container", ".content"]
    for sel in bc_sels:
        candidates.extend(soup.select(sel))

    # Escolher o maior bloco
    best_txt, best_len = None, 0
    for el in (candidates or [soup]):
        txt = el.get_text(separator=" ", strip=True)
        L = len(txt)
        if L > best_len:
            best_txt, best_len = txt, L

    text = best_txt or soup.get_text(separator=" ", strip=True)
    return " ".join((text or "").split())

# ───────────────────────────────────────── Ingest: TEXT / PDF
def ingest_text(title: str, text: str, namespace: str = RAG_DEFAULT_NAMESPACE, source_id: str = None) -> Dict:
    text = (text or "").strip()
    if not text: return {"ok": False, "error": "Texto vazio"}
    chunks = chunk_text(text)
    src_id = source_id or hashlib.sha1((title + text[:200]).encode("utf-8")).hexdigest()
    upsert_chunks(namespace, {"type": "text", "title": title, "source_id": src_id}, chunks)
    return {"ok": True, "chunks": len(chunks), "source_id": src_id}

def fetch_pdf_text(pdf_url: str) -> Tuple[str, int]:
    s = _session()
    r = s.get(_proxied(pdf_url), timeout=REQUEST_TIMEOUT)
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

# ───────────────────────────────────────── Ingest: single URL
def ingest_url(page_url: str, namespace: str = RAG_DEFAULT_NAMESPACE) -> Dict:
    s = _session()
    u = _clean_url(page_url)
    html, ctype, status = _get(s, u)
    if not html: return {"ok": False, "error": "Falha HTTP", "url": u, "status": status}
    if ("text/html" not in (ctype or "")) and ("<html" not in html.lower()):
        return {"ok": False, "error": "Não é HTML", "url": u, "status": status}
    text = _extract_main_text(html)
    if tokenize_len(text) < MIN_TOKENS:
        # tenta headless se disponível
        rend = _rendered_html(u)
        if rend:
            text = _extract_main_text(rend)
    if tokenize_len(text) < MIN_TOKENS:
        return {"ok": False, "error": "Página sem conteúdo útil suficiente", "url": u}
    title = _extract_title(html) or u
    src_id = hashlib.sha1(u.encode("utf-8")).hexdigest()
    chunks = chunk_text(text)
    upsert_chunks(namespace, {"type": "web", "title": title, "url": u, "domain": urlparse(u).netloc, "source_id": src_id}, chunks)
    return {"ok": True, "chunks": len(chunks), "source_id": src_id, "url": u, "title": title}

# ───────────────────────────────────────── Ingest: Crawl (BFS)
def crawl_and_ingest(seed_url: str, namespace: str = RAG_DEFAULT_NAMESPACE,
                     max_pages: int = CRAWL_MAX_PAGES, max_depth: int = CRAWL_MAX_DEPTH) -> Dict:
    s = _session()
    seed_url = _clean_url(seed_url)
    seen, queue, errors = set(), [(seed_url, 0)], []
    n_ingested = 0
    src_host = urlparse(seed_url).netloc

    while queue and n_ingested < max_pages:
        url, depth = queue.pop(0)
        url = _clean_url(url)
        if url in seen: continue
        seen.add(url)

        # NÃO filtrar o seed; aplicar filtro de língua só aos links descobertos
        if LANG_PATH_ONLY and (url != seed_url) and (LANG_PATH_ONLY not in url):
            continue

        try:
            html, ctype, status = _get(s, url)
            if not html: continue
            if ("text/html" not in (ctype or "")) and ("<html" not in html.lower()):
                continue

            title = _extract_title(html) or url
            text = _extract_main_text(html)

            if tokenize_len(text) < MIN_TOKENS:
                # tentar render headless (JS) apenas se configurado
                rend = _rendered_html(url)
                if rend:
                    text = _extract_main_text(rend)

            if tokenize_len(text) >= MIN_TOKENS:
                # Respeitar allow/deny por path
                path = urlparse(url).path or ""
                if ALLOW_PATHS and not _match_any(ALLOW_PATHS, path):
                    pass  # fora da allowlist -> não ingere, mas segue links
                elif DENY_PATHS and _match_any(DENY_PATHS, path):
                    pass  # em denylist -> não ingere, mas segue links
                else:
                    try:
                        src_id = hashlib.sha1(url.encode("utf-8")).hexdigest()
                        chunks = chunk_text(text)
                        if chunks:
                            upsert_chunks(
                                namespace,
                                {"type": "web", "title": title, "url": url, "domain": src_host, "source_id": src_id},
                                chunks,
                            )
                            n_ingested += 1
                    except Exception as e:
                        errors.append({"url": url, "where": "upsert", "error": str(e)})

            if depth < max_depth:
                for link in _extract_links(url, html):
                    if not link or link in seen: continue
                    if not _is_same_domain(seed_url, link): continue
                    # aplicar filtro de língua aqui
                    if LANG_PATH_ONLY and (LANG_PATH_ONLY not in link):
                        continue
                    # apply allow/deny antes de enfileirar (menos ruído)
                    path = urlparse(link).path or ""
                    if DENY_PATHS and _match_any(DENY_PATHS, path):
                        continue
                    queue.append((link, depth + 1))

        except Exception as e:
            errors.append({"url": url, "where": "fetch|parse", "error": str(e)})
            continue

    return {"ok": True, "pages_ingested": n_ingested, "visited": len(seen), "domain": src_host, "errors": errors}

# ───────────────────────────────────────── Ingest: Sitemap
def _parse_sitemap_xml(xml_text: str) -> Tuple[List[str], List[str]]:
    soup = BeautifulSoup(xml_text, "xml")
    child_maps = [loc.get_text(strip=True) for sm in soup.find_all("sitemap") for loc in sm.find_all("loc")]
    page_urls = [loc.get_text(strip=True) for url in soup.find_all("url") for loc in url.find_all("loc")]
    # fallback geral (alguns WP expõem <loc> direto)
    if (not child_maps) and (not page_urls):
        page_urls = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
    return child_maps, page_urls

def ingest_sitemap(sitemap_or_site_url: str, namespace: str = RAG_DEFAULT_NAMESPACE,
                   max_pages: int = CRAWL_MAX_PAGES) -> Dict:
    s = _session()
    start = sitemap_or_site_url.strip()
    if not start.endswith(".xml"):
        base = start.rstrip("/")
        start = base + "/sitemap_index.xml"  # WP default
    collected: List[str] = []
    to_visit = [start]
    seen_maps, errors = set(), []

    # 1) Recolha recursiva
    while to_visit and len(collected) < max_pages * 5:
        sm = to_visit.pop(0)
        if sm in seen_maps: continue
        seen_maps.add(sm)
        try:
            xml, ctype, status = _get(s, sm)
            if not xml:
                continue
            childs, pages = _parse_sitemap_xml(xml)
            # filtro de língua
            if LANG_PATH_ONLY:
                pages  = [u for u in pages  if LANG_PATH_ONLY in u]
                childs = [u for u in childs if LANG_PATH_ONLY in u]
            collected.extend(pages)
            to_visit.extend(childs)
        except Exception as e:
            errors.append({"sitemap": sm, "error": str(e)})

    # dedup
    uniq, seen_u = [], set()
    for u in collected:
        u = _clean_url(u)
        if u not in seen_u:
            uniq.append(u); seen_u.add(u)

    # 2) Ingest real
    n_ingested = 0
    host = urlparse(start).netloc
    for u in uniq:
        if n_ingested >= max_pages: break
        try:
            res = ingest_url(u, namespace=namespace)
            if res.get("ok"):
                n_ingested += 1
        except Exception as e:
            errors.append({"url": u, "where": "ingest_url", "error": str(e)})

    return {"ok": True, "pages_ingested": n_ingested, "collected": len(uniq), "domain": host, "errors": errors}

# ───────────────────────────────────────── Retrieval
def search_chunks(query: str, namespace: Optional[str] = None, top_k: int = None) -> List[Dict]:
    query = (query or "").strip()
    if not query: return []
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
        chosen.append(chunk); total += cost
    return "" if not chosen else "Contexto de conhecimento interno (RAG):\n" + "\n\n".join(chosen)
