# rag_client.py — RAG + Qdrant + crawler robusto
# ---------------------------------------------------------------
import os, time, uuid, requests, hashlib
from urllib.parse import urlsplit, urlunsplit, urljoin
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION = os.getenv("QDRANT_COLLECTION", "alma_rag")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

qdrant = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer(EMBED_MODEL)

# ---------------------------------------------------------------
# Helpers: URL normalização, denylist, ids únicos
# ---------------------------------------------------------------
DENY_PATTERNS = [
    "/carrinho", "/checkout", "/minha-conta", "/wp-login",
    "/feed", "/tag/", "/categoria/", "/author/",
]
DENY_CONTAINS = [
    "add-to-cart=", "orderby=", "wc-ajax", "utm_", "replytocom=",
]

def _clean_url(u: str) -> str:
    """Normaliza URL removendo query/fragmento e forçando slash final."""
    if not u:
        return ""
    p = urlsplit(u)
    path = p.path or "/"
    # Se for diretório sem extensão → força slash
    if not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        path += "/"
    return urlunsplit((p.scheme, p.netloc, path, "", ""))

def _url_allowed(u: str) -> bool:
    """Filtro básico contra loops/lixo de e-commerce."""
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

# ---------------------------------------------------------------
# Helpers: Texto e embeddings
# ---------------------------------------------------------------
def _chunk_text(text: str, max_tokens: int = 400) -> list[str]:
    """Divide texto em chunks aproximados (por frases)."""
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

def _embed_texts(texts: list[str]):
    return embedder.encode(texts).tolist()

# ---------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------
def _ingest(namespace: str, url: str, title: str, text: str):
    chunks = _chunk_text(text)
    if not chunks:
        return 0
    vecs = _embed_texts(chunks)
    points = []
    for idx, (c, v) in enumerate(zip(chunks, vecs)):
        pid = _uuid_for_chunk(namespace, url, idx)
        points.append(qm.PointStruct(
            id=pid,
            vector=v,
            payload={"url": url, "title": title, "text": c, "namespace": namespace}
        ))
    qdrant.upsert(collection_name=COLLECTION, points=points)
    return len(points)

# ---------------------------------------------------------------
# Public ingestion funcs
# ---------------------------------------------------------------
def ingest_text(title: str, text: str, namespace: str = "default"):
    return {"ok": True, "count": _ingest(namespace, f"text://{title}", title, text)}

def ingest_url(url: str, namespace: str = "default", deadline_s: int = 55):
    u = _clean_url(url)
    if not _url_allowed(u):
        return {"ok": False, "error": "url_blocked"}
    try:
        r = requests.get(u, timeout=20)
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_failed: {e}"}
    soup = BeautifulSoup(r.text, "html.parser")
    title = (soup.title.string if soup.title else u).strip()
    text = soup.get_text(" ", strip=True)
    count = _ingest(namespace, u, title, text)
    return {"ok": True, "url": u, "count": count}

def ingest_pdf_url(pdf_url: str, title: str = None, namespace: str = "default"):
    import fitz
    try:
        r = requests.get(pdf_url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_pdf_failed: {e}"}
    doc = fitz.open("pdf", r.content)
    full = " ".join(page.get_text() for page in doc)
    count = _ingest(namespace, pdf_url, title or pdf_url, full)
    return {"ok": True, "url": pdf_url, "count": count}

def ingest_sitemap(sitemap_url: str, namespace: str = "default", max_pages: int = 100, deadline_s: int = 55):
    try:
        r = requests.get(sitemap_url, timeout=20)
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_sitemap_failed: {e}"}
    soup = BeautifulSoup(r.text, "xml")
    locs = [loc.get_text() for loc in soup.find_all("loc")]
    ok, fail = 0, 0
    t0 = time.time()
    for loc in locs[:max_pages]:
        if time.time() - t0 > deadline_s:
            break
        res = ingest_url(loc, namespace=namespace, deadline_s=deadline_s)
        if res.get("ok"):
            ok += res["count"]
        else:
            fail += 1
    return {"ok": True, "sitemap": sitemap_url, "pages_ok": ok, "pages_failed": fail}

# ---------------------------------------------------------------
# Crawl (multi-página)
# ---------------------------------------------------------------
def crawl_and_ingest(seed_url: str, namespace: str = "default", max_pages: int = 200, max_depth: int = 3, deadline_s: int = 55):
    start = _clean_url(seed_url)
    seen, queue = set(), [(start, 0)]
    ok, fail = 0, 0
    t0 = time.time()
    while queue and len(seen) < max_pages and time.time() - t0 < deadline_s:
        url, depth = queue.pop(0)
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        if not _url_allowed(url):
            continue
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
        except Exception:
            fail += 1
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.string if soup.title else url).strip()
        text = soup.get_text(" ", strip=True)
        ok += _ingest(namespace, url, title, text)
        # próximos links
        for a in soup.find_all("a", href=True):
            nxt = urljoin(url, a["href"])
            nxt = _clean_url(nxt)
            if nxt not in seen and _url_allowed(nxt):
                queue.append((nxt, depth + 1))
    return {"ok": True, "visited": len(seen), "ok_chunks": ok, "fail": fail, "namespace": namespace}
    
# ---------------------------------------------------------------
# Search
# ---------------------------------------------------------------
def search_chunks(query: str, namespace: str = None, top_k: int = 6):
    vec = _embed_texts([query])[0]
    res = qdrant.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=top_k,
        query_filter=qm.Filter(
            must=[qm.FieldCondition(key="namespace", match=qm.MatchValue(value=namespace or "default"))]
        )
    )
    return [m.payload for m in res]

def build_context_block(matches, token_budget: int = 1600):
    lines, used = [], 0
    for m in matches:
        t = m.get("text", "")
        toks = len(t.split())
        if used + toks > token_budget:
            break
        lines.append(f"[{m.get('title')}] {t}")
        used += toks
    return "\n".join(lines)
