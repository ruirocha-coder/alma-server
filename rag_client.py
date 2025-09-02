# rag_client.py — RAG + Qdrant + crawler robusto (UUID, ensure_collection, deadline)
# ---------------------------------------------------------------------------------
import os, time, uuid, requests, hashlib
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlsplit, urlunsplit, urljoin

from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

# -------------------- Config --------------------
QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_API_KEY    = (os.getenv("QDRANT_API_KEY") or "").strip() or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "alma_rag").strip()
EMBED_MODEL       = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2").strip()

HTTP_TIMEOUT      = int(os.getenv("RAG_REQUEST_TIMEOUT", "20"))
CRAWL_MAX_PAGES   = int(os.getenv("CRAWL_MAX_PAGES", "100"))
CRAWL_MAX_DEPTH   = int(os.getenv("CRAWL_MAX_DEPTH", "3"))
DEFAULT_NAMESPACE = os.getenv("RAG_DEFAULT_NAMESPACE", "default").strip()
TOP_K             = int(os.getenv("RAG_TOP_K", "6"))

# ----------------- Clients / init ----------------
qdrant  = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY, timeout=HTTP_TIMEOUT)
embedder = SentenceTransformer(EMBED_MODEL)
EMBED_DIM = embedder.get_sentence_embedding_dimension()

# ----------------- URL helpers -------------------
DENY_PATTERNS = [
    "/carrinho", "/checkout", "/minha-conta", "/wp-login",
    "/feed", "/tag/", "/categoria/", "/author/", "/cart", "/account",
    "/wp-json", "/xmlrpc.php"
]
DENY_CONTAINS = [
    "add-to-cart=", "orderby=", "wc-ajax", "utm_", "replytocom=", "fbclid=",
    "sessionid=", "phpsessid=", "sid="
]

def _clean_url(u: str) -> str:
    if not u:
        return ""
    p = urlsplit(u)
    path = p.path or "/"
    # diretório sem extensão => força slash final
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

def _same_host(a: str, b: str) -> bool:
    return urlsplit(a).netloc == urlsplit(b).netloc

# ----------------- Qdrant helpers ----------------
def ensure_collection() -> None:
    try:
        qdrant.get_collection(QDRANT_COLLECTION)
        return
    except Exception:
        pass
    qdrant.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE)
    )

def _uuid_for_chunk(namespace: str, url: str, idx: int) -> str:
    base = f"{namespace}|{url}|{idx}"
    # UUID v5 é aceite pelo Qdrant (string no formato UUID)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

# ----------------- Text & embed ------------------
def _chunk_text(text: str, max_words: int = 220) -> List[str]:
    # chunking aproximado por frases (rápido e robusto)
    sents = [s.strip() for s in text.split(".") if s.strip()]
    parts, buf, cnt = [], [], 0
    for s in sents:
        w = len(s.split())
        if cnt + w > max_words and buf:
            parts.append(". ".join(buf) + ".")
            buf, cnt = [], 0
        buf.append(s)
        cnt += w
    if buf:
        parts.append(". ".join(buf) + ".")
    return parts

def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    return embedder.encode(texts, convert_to_numpy=False).tolist()

def _ingest(namespace: str, url: str, title: str, text: str) -> int:
    ensure_collection()
    chunks = _chunk_text(text)
    if not chunks:
        return 0
    vecs = _embed_texts(chunks)
    points: List[qm.PointStruct] = []
    for idx, (c, v) in enumerate(zip(chunks, vecs)):
        pid = _uuid_for_chunk(namespace, url, idx)
        points.append(qm.PointStruct(
            id=pid,
            vector=v,
            payload={"url": url, "title": title, "text": c, "namespace": namespace}
        ))
    # upsert em lotes (evita payloads muito grandes)
    for i in range(0, len(points), 64):
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i+64])
    return len(points)

# ----------------- Public ingest -----------------
def ingest_text(title: str, text: str, namespace: str = DEFAULT_NAMESPACE) -> Dict:
    count = _ingest(namespace, f"text://{hashlib.sha1(title.encode()).hexdigest()}/", title, text)
    return {"ok": True, "count": count}

def ingest_url(url: str, namespace: str = DEFAULT_NAMESPACE, deadline_s: int = 55) -> Dict:
    u = _clean_url(url)
    if not _url_allowed(u):
        return {"ok": False, "error": "url_blocked", "url": u}
    try:
        r = requests.get(u, timeout=min(HTTP_TIMEOUT, deadline_s))
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_failed: {e}", "url": u}
    soup = BeautifulSoup(r.text, "html.parser")
    # remove ruído
    for tag in soup(["script", "style", "nav", "footer", "form", "noscript"]):
        tag.decompose()
    title = (soup.title.get_text(strip=True) if soup.title else u)
    text  = soup.get_text(" ", strip=True)
    count = _ingest(namespace, u, title, text)
    return {"ok": True, "url": u, "count": count}

def ingest_pdf_url(pdf_url: str, title: Optional[str] = None, namespace: str = DEFAULT_NAMESPACE) -> Dict:
    import fitz  # PyMuPDF
    try:
        r = requests.get(pdf_url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_pdf_failed: {e}", "url": pdf_url}
    doc = fitz.open("pdf", r.content)
    full = " ".join((page.get_text() or "") for page in doc)
    count = _ingest(namespace, pdf_url, title or pdf_url, full)
    return {"ok": True, "url": pdf_url, "count": count}

def ingest_sitemap(sitemap_url: str, namespace: str = DEFAULT_NAMESPACE,
                   max_pages: int = 100, deadline_s: int = 55) -> Dict:
    t0 = time.time()
    try:
        r = requests.get(sitemap_url, timeout=min(HTTP_TIMEOUT, deadline_s))
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_sitemap_failed: {e}", "sitemap_url": sitemap_url}
    soup = BeautifulSoup(r.text, "xml")
    locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
    ok_pages, fail_pages = 0, 0
    for loc in locs[:max_pages]:
        if time.time() - t0 > deadline_s:
            break
        res = ingest_url(loc, namespace=namespace, deadline_s=deadline_s)
        if res.get("ok"):
            ok_pages += 1
        else:
            fail_pages += 1
    return {"ok": True, "sitemap": sitemap_url, "pages_ok": ok_pages, "pages_failed": fail_pages}

def crawl_and_ingest(seed_url: str, namespace: str = DEFAULT_NAMESPACE,
                     max_pages: int = CRAWL_MAX_PAGES, max_depth: int = CRAWL_MAX_DEPTH,
                     deadline_s: int = 55) -> Dict:
    start = _clean_url(seed_url)
    seen, queue = set(), [(start, 0)]
    ok_chunks, fail_fetch = 0, 0
    t0 = time.time()
    host = urlsplit(start).netloc
    while queue and len(seen) < max_pages and (time.time() - t0) < deadline_s:
        url, depth = queue.pop(0)
        if url in seen or depth > max_depth or not _url_allowed(url):
            continue
        seen.add(url)
        try:
            r = requests.get(url, timeout=min(HTTP_TIMEOUT, deadline_s))
            r.raise_for_status()
        except Exception:
            fail_fetch += 1
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "form", "noscript"]):
            tag.decompose()
        title = (soup.title.get_text(strip=True) if soup.title else url)
        text  = soup.get_text(" ", strip=True)
        ok_chunks += _ingest(namespace, url, title, text)
        # expandir links (mesmo domínio)
        if depth < max_depth:
            for a in soup.find_all("a", href=True):
                nxt = _clean_url(urljoin(url, a["href"]))
                if nxt not in seen and _same_host(start, nxt) and _url_allowed(nxt):
                    queue.append((nxt, depth + 1))
    timed_out = (time.time() - t0) >= deadline_s
    return {
        "ok": True,
        "visited": len(seen),
        "ok_chunks": ok_chunks,
        "fail_fetch": fail_fetch,
        "domain": host,
        "timed_out": timed_out
    }

# ----------------- Search / context --------------
def search_chunks(query: str, namespace: Optional[str] = None, top_k: int = TOP_K) -> List[Dict]:
    vec = _embed_texts([query])[0]
    must = []
    if (namespace or DEFAULT_NAMESPACE):
        must.append(qm.FieldCondition(key="namespace", match=qm.MatchValue(value=namespace or DEFAULT_NAMESPACE)))
    res = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vec,
        limit=top_k,
        query_filter=qm.Filter(must=must) if must else None
    )
    out = []
    for m in res:
        pl = m.payload or {}
        pl["score"] = float(m.score)
        out.append(pl)
    return out

def build_context_block(matches: List[Dict], token_budget: int = 1600) -> str:
    lines, used = [], 0
    for m in matches:
        t = m.get("text", "") or ""
        wc = len(t.split())
        if used + wc > token_budget:
            break
        head = m.get("title") or m.get("url") or "doc"
        lines.append(f"[{head}] {t}")
        used += wc
    return "Contexto de conhecimento interno (RAG):\n" + "\n\n".join(lines) if lines else ""
