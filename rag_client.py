# rag_client.py — OpenAI embeddings (1536D) + Qdrant + crawler/sitemap com fetch "browser-like"
# --------------------------------------------------------------------------------------------
import os
import time
import uuid
import random
import requests
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, urljoin

from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from openai import OpenAI


# =========================== Config / ENV ===========================

QDRANT_URL         = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION  = os.getenv("QDRANT_COLLECTION", "alma_docs")

# Modelo e dimensão (text-embedding-3-* = 1536)
OPENAI_MODEL       = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
MODEL_DIMS         = {
    "text-embedding-3-large": 1536,
    "text-embedding-3-small": 1536,
}
VECTOR_SIZE        = MODEL_DIMS.get(OPENAI_MODEL, 1536)

# Ingestão
UPSERT_BATCH       = int(os.getenv("UPSERT_BATCH", "64"))
TIMEOUT_FETCH_S    = int(os.getenv("FETCH_TIMEOUT_S", "20"))

# Crawler limites (podes sobrepor via endpoints)
DEFAULT_MAX_PAGES  = int(os.getenv("CRAWL_MAX_PAGES", "500"))
DEFAULT_MAX_DEPTH  = int(os.getenv("CRAWL_MAX_DEPTH", "4"))
DEFAULT_DEADLINE_S = int(os.getenv("RAG_DEADLINE_S", "110"))

# Sitemap limites
SITEMAP_MAX_PAGES  = int(os.getenv("SITEMAP_MAX_PAGES", "2000"))

# Fetch anti-403
CRAWLER_DELAY_S        = float(os.getenv("CRAWLER_DELAY_S", "0.6"))
MAX_RETRIES_PER_URL    = int(os.getenv("MAX_RETRIES_PER_URL", "3"))
SCRAPING_HTTP_PROXY    = os.getenv("SCRAPING_HTTP_PROXY", "")
SCRAPING_HTTPS_PROXY   = os.getenv("SCRAPING_HTTPS_PROXY", "")
WP_CONSENT_COOKIE      = os.getenv("WP_CONSENT_COOKIE", "euconsent-v2=1; cookie_consent=true;")

# Se verdadeiro, e a dimensão da coleção não bater com o modelo, recria (DANGER: apaga dados)
AUTO_RECREATE_COLLECTION = os.getenv("AUTO_RECREATE_COLLECTION", "false").lower() == "true"


# ============================== Clients =============================

qdrant = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()


# ==================== Qdrant collection: ensure =====================

def ensure_collection(dim: int = VECTOR_SIZE):
    """
    Cria a collection se não existir. Se existir com dimensão diferente:
      - se AUTO_RECREATE_COLLECTION=true → recria com a dimensão certa (⚠️ apaga dados)
      - caso contrário → levanta erro com explicação
    """
    try:
        info = qdrant.get_collection(QDRANT_COLLECTION)
        existing_dim = None
        if info and getattr(info, "config", None) and getattr(info.config, "vectors_config", None):
            vc = info.config.vectors_config
            if hasattr(vc, "config") and getattr(vc.config, "size", None):
                existing_dim = vc.config.size
            elif getattr(vc, "size", None):
                existing_dim = vc.size

        if existing_dim is None:
            # coleção existe mas não conseguimos ler o size → assume OK e segue
            return

        if existing_dim != dim:
            if AUTO_RECREATE_COLLECTION:
                qdrant.recreate_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                )
            else:
                raise RuntimeError(
                    f"A collection '{QDRANT_COLLECTION}' tem dimensão {existing_dim}, "
                    f"mas o modelo '{OPENAI_MODEL}' produz {dim}. "
                    f"Muda QDRANT_COLLECTION, apaga a existente ou ativa AUTO_RECREATE_COLLECTION=true."
                )
        return
    except Exception:
        # não existe → cria
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

ensure_collection(VECTOR_SIZE)


# ==================== URL helpers, denylist, IDs ====================

DENY_PATTERNS = [
    "/carrinho", "/checkout", "/minha-conta", "/wp-login",
    "/feed", "/tag/", "/categoria/", "/author/",
    "/cart/", "/my-account/", "/wp-admin",
]
DENY_CONTAINS = [
    "add-to-cart=", "orderby=", "wc-ajax", "utm_", "replytocom=", "sessionid=",
]

def _clean_url(u: str) -> str:
    if not u:
        return ""
    p = urlsplit(u)
    path = p.path or "/"
    if not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        path += "/"
    return urlunsplit((p.scheme, p.netloc, path, "", ""))

def _url_allowed(u: str) -> bool:
    low = u.lower()
    for pat in DENY_PATTERNS:
        if pat in low:
            return False
    for pat in DENY_CONTAINS:
        if pat in low:
            return False
    return True

def _uuid_for_chunk(namespace: str, url: str, idx: int) -> str:
    base = f"{namespace}|{url}|{idx}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


# ========================= Fetch anti-403 ===========================

BROWSER_UAS = [
    # Chrome Win
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36",
    # Chrome Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36",
    # Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
]

def _make_session() -> requests.Session:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    retry = Retry(
        total=MAX_RETRIES_PER_URL,
        backoff_factor=0.6,
        status_forcelist=(403, 408, 429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))

    if SCRAPING_HTTP_PROXY or SCRAPING_HTTPS_PROXY:
        proxies = {}
        if SCRAPING_HTTP_PROXY:
            proxies["http"] = SCRAPING_HTTP_PROXY
        if SCRAPING_HTTPS_PROXY:
            proxies["https"] = SCRAPING_HTTPS_PROXY
        s.proxies.update(proxies)

    s.headers.update({
        "User-Agent": random.choice(BROWSER_UAS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "close",
    })
    return s

def _browser_headers(host_url: str) -> dict:
    return {
        "Referer": host_url,
        "Cookie": WP_CONSENT_COOKIE,
    }

def _fetch(url: str, timeout: int = TIMEOUT_FETCH_S) -> requests.Response:
    """
    Faz GET com comportamento semelhante a um browser:
      - headers realistas (UA/Referer/Cookies)
      - retries + backoff
      - pequena pausa (CRAWLER_DELAY_S)
      - roda o UA e altera Accept quando apanha 403/429
    """
    base = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}/"
    s = _make_session()
    last_err = None

    for _ in range(MAX_RETRIES_PER_URL):
        try:
            if CRAWLER_DELAY_S > 0:
                time.sleep(CRAWLER_DELAY_S)

            h = dict(s.headers)
            h.update(_browser_headers(base))
            r = s.get(url, timeout=timeout, headers=h)
            if r.status_code in (403, 429):
                # muda UA e aceita XML
                s.headers["User-Agent"] = random.choice(BROWSER_UAS)
                h = dict(s.headers)
                h.update(_browser_headers(base))
                h["Accept"] = "text/html,application/xml;q=0.9,*/*;q=0.8"
                if CRAWLER_DELAY_S > 0:
                    time.sleep(CRAWLER_DELAY_S + 0.4)
                r = s.get(url, timeout=timeout, headers=h)

            if r.status_code in (403, 429):
                # terceira tentativa com referer direto
                h = dict(s.headers)
                h.update(_browser_headers(url))
                if CRAWLER_DELAY_S > 0:
                    time.sleep(CRAWLER_DELAY_S + 0.6)
                r = s.get(url, timeout=timeout, headers=h)

            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            continue

    raise last_err if last_err else requests.HTTPError("fetch_failed")


# ========================== Texto / Embeds ==========================

def _chunk_text(text: str, max_tokens: int = 450) -> List[str]:
    """
    Split simples por frases (aproximado). Mantém ~450 palavras por chunk,
    o que ajuda a controlar custos de embeddings e relevância.
    """
    parts, buf, count = [], [], 0
    for sent in text.split(". "):
        t = sent.strip()
        if not t:
            continue
        toks = len(t.split())
        if count + toks > max_tokens and buf:
            parts.append(". ".join(buf))
            buf, count = [], 0
        buf.append(t)
        count += toks
    if buf:
        parts.append(". ".join(buf))
    return parts

def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    # OpenAI embeddings (ordem preservada)
    resp = openai_client.embeddings.create(model=OPENAI_MODEL, input=texts)
    return [d.embedding for d in resp.data]


# ========================= Núcleo de ingest =========================

def _ingest(namespace: str, url: str, title: str, full_text: str) -> int:
    chunks = _chunk_text(full_text)
    if not chunks:
        return 0

    vecs = _embed_texts(chunks)
    # sanity check de dimensão
    if vecs and len(vecs[0]) != VECTOR_SIZE:
        raise RuntimeError(
            f"Dimensão de embedding ({len(vecs[0])}) != dimensão da coleção ({VECTOR_SIZE}). "
            f"Modelo={OPENAI_MODEL} Coleção={QDRANT_COLLECTION}"
        )

    points: List[qm.PointStruct] = []
    for idx, (c, v) in enumerate(zip(chunks, vecs)):
        points.append(qm.PointStruct(
            id=_uuid_for_chunk(namespace, url, idx),
            vector=v,
            payload={"url": url, "title": title, "text": c, "namespace": namespace}
        ))

    total = 0
    for i in range(0, len(points), UPSERT_BATCH):
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i+UPSERT_BATCH])
        total += len(points[i:i+UPSERT_BATCH])
    return total


# ========================= Ingest público ===========================

def ingest_text(title: str, text: str, namespace: str = "default") -> Dict:
    count = _ingest(namespace, f"text://{title}", title, text)
    return {"ok": True, "count": count}

def ingest_url(page_url: str, namespace: str = "default", deadline_s: int = DEFAULT_DEADLINE_S) -> Dict:
    u = _clean_url(page_url)
    if not _url_allowed(u):
        return {"ok": False, "error": "url_blocked", "url": u}
    try:
        r = _fetch(u, timeout=TIMEOUT_FETCH_S)
    except Exception as e:
        return {"ok": False, "error": f"fetch_failed: {e}", "url": u}

    soup = BeautifulSoup(r.text, "html.parser")
    title = (soup.title.string if soup.title else u).strip()
    text = soup.get_text(" ", strip=True)
    count = _ingest(namespace, u, title, text)
    return {"ok": True, "url": u, "count": count}

def ingest_pdf_url(pdf_url: str, title: Optional[str] = None, namespace: str = "default") -> Dict:
    import fitz  # PyMuPDF
    try:
        r = _fetch(pdf_url, timeout=TIMEOUT_FETCH_S + 10)
    except Exception as e:
        return {"ok": False, "error": f"fetch_pdf_failed: {e}", "url": pdf_url}

    doc = fitz.open("pdf", r.content)
    full = " ".join(page.get_text() for page in doc)
    count = _ingest(namespace, pdf_url, title or pdf_url, full)
    return {"ok": True, "url": pdf_url, "count": count}

def ingest_sitemap(sitemap_url: str, namespace: str = "default",
                   max_pages: int = SITEMAP_MAX_PAGES, deadline_s: int = DEFAULT_DEADLINE_S) -> Dict:
    """
    Lê <loc> de um sitemap XML (ou índice). Para cada URL chama ingest_url (com _fetch anti-403).
    Devolve listas de sucesso/fracasso para poderes mostrar na consola.
    """
    t0 = time.time()
    try:
        r = requests.get(sitemap_url, timeout=TIMEOUT_FETCH_S)
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_sitemap_failed: {e}"}

    # usar parser XML quando disponível (BeautifulSoup com 'xml' é fiável aqui)
    soup = BeautifulSoup(r.text, "xml")
    locs = [loc.get_text().strip() for loc in soup.find_all("loc")]

    ingested_urls: List[str] = []
    failed_urls: List[Tuple[str, str]] = []

    pages_ok, pages_fail = 0, 0
    for loc in locs[:max_pages]:
        if time.time() - t0 > deadline_s:
            break
        res = ingest_url(loc, namespace=namespace, deadline_s=deadline_s)
        if res.get("ok"):
            ingested_urls.append(res.get("url", loc))
            pages_ok += res.get("count", 0)
        else:
            failed_urls.append((loc, res.get("error", "unknown_error")))
            pages_fail += 1

    return {
        "ok": True,
        "sitemap": sitemap_url,
        "pages_ingested": pages_ok,
        "pages_failed": pages_fail,
        "namespace": namespace,
        "ingested_urls": ingested_urls,
        "failed_urls": failed_urls,
    }


# ============================= Crawler ==============================

def crawl_and_ingest(seed_url: str, namespace: str = "default",
                     max_pages: int = DEFAULT_MAX_PAGES,
                     max_depth: int = DEFAULT_MAX_DEPTH,
                     deadline_s: int = DEFAULT_DEADLINE_S) -> Dict:
    """
    BFS simples, mesmo domínio, com filtros de e-commerce e _fetch anti-403.
    """
    start = _clean_url(seed_url)
    start_host = urlsplit(start).netloc

    seen: set[str] = set()
    queue: List[Tuple[str, int]] = [(start, 0)]
    ok_chunks, fail = 0, 0
    t0 = time.time()

    while queue and len(seen) < max_pages and time.time() - t0 < deadline_s:
        url, depth = queue.pop(0)
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        if not _url_allowed(url):
            continue

        try:
            r = _fetch(url, timeout=TIMEOUT_FETCH_S)
        except Exception:
            fail += 1
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.string if soup.title else url).strip()
        text = soup.get_text(" ", strip=True)
        ok_chunks += _ingest(namespace, url, title, text)

        # próximos links no mesmo domínio
        for a in soup.find_all("a", href=True):
            nxt = _clean_url(urljoin(url, a["href"]))
            if nxt in seen or not _url_allowed(nxt):
                continue
            if urlsplit(nxt).netloc != start_host:
                continue
            queue.append((nxt, depth + 1))

    return {
        "ok": True,
        "visited": len(seen),
        "ok_chunks": ok_chunks,
        "fail": fail,
        "namespace": namespace,
    }


# ============================== Search ==============================

def search_chunks(query: str, namespace: Optional[str] = None, top_k: int = 6) -> List[Dict]:
    vec = _embed_texts([query])[0]
    flt = qm.Filter(must=[qm.FieldCondition(
        key="namespace", match=qm.MatchValue(value=namespace or "default")
    )])
    res = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vec,
        limit=top_k,
        query_filter=flt,
    )
    out: List[Dict] = []
    for m in res:
        p = dict(m.payload or {})
        p["score"] = float(getattr(m, "score", 0.0))
        out.append(p)
    return out

def build_context_block(matches: List[Dict], token_budget: int = 1600) -> str:
    lines, used = [], 0
    for m in matches:
        t = (m.get("text") or "").strip()
        toks = len(t.split())
        if used + toks > token_budget:
            break
        lines.append(f"[{m.get('title')}] {t}")
        used += toks
    return "\n".join(lines)
