# rag_client.py — OpenAI embeddings (1536D) + Qdrant + crawler/sitemap (parallel, retries, cursor)
# -------------------------------------------------------------------
import os, time, uuid, math, random, requests
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, urljoin
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# =========================== Config ================================
QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "alma_docs")   # default consistente
OPENAI_MODEL      = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536D
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
UPSERT_BATCH      = int(os.getenv("UPSERT_BATCH", "64"))
FETCH_TIMEOUT_S   = int(os.getenv("FETCH_TIMEOUT_S", "20"))
QDRANT_AUTO_MIGRATE = os.getenv("QDRANT_AUTO_MIGRATE", "1") == "1"  # recria coleção se dim não bater

# =========================== Crawling defaults =====================
CRAWL_MAX_PAGES   = int(os.getenv("CRAWL_MAX_PAGES", "150"))
CRAWL_DEADLINE_S  = int(os.getenv("CRAWL_DEADLINE_S", "300"))
CRAWL_MAX_DEPTH   = int(os.getenv("CRAWL_MAX_DEPTH", "3"))

# ---- Ritmo/robustez para sitemaps (anti-WAF/CDN) ----
SITEMAP_SLEEP_MS      = int(os.getenv("SITEMAP_SLEEP_MS", "300"))    # pausa base entre tarefas
SITEMAP_MAX_RETRIES   = int(os.getenv("SITEMAP_MAX_RETRIES", "4"))   # tentativas por URL
SITEMAP_BACKOFF_MS    = int(os.getenv("SITEMAP_BACKOFF_MS", "500"))  # backoff base por tentativa
SITEMAP_JITTER_MS     = int(os.getenv("SITEMAP_JITTER_MS", "200"))   # jitter aleatório
SITEMAP_CONCURRENCY   = int(os.getenv("SITEMAP_CONCURRENCY", "6"))   # trabalhadores paralelos
SITEMAP_LIMIT_PER_CALL= int(os.getenv("SITEMAP_LIMIT_PER_CALL", "60")) # paginação por chamada

# Dimensões por modelo (ambos 1536 nas versões atuais)
MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 1536,
}
VECTOR_SIZE = MODEL_DIMS.get(OPENAI_MODEL, 1536)

print(f"[rag_client] defaults → max_pages={CRAWL_MAX_PAGES}, deadline_s={CRAWL_DEADLINE_S}, "
      f"max_depth={CRAWL_MAX_DEPTH}, embed_model={OPENAI_MODEL}, vector_size={VECTOR_SIZE}, "
      f"collection={QDRANT_COLLECTION}, site_concurrency={SITEMAP_CONCURRENCY}, "
      f"limit_per_call={SITEMAP_LIMIT_PER_CALL}")

# ========================= Clients ================================
qdrant = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()

# =================== Qdrant collection ensure =====================
def _get_existing_dim(info) -> Optional[int]:
    try:
        cfg = info.config
        if not cfg:
            return None
        vc = cfg.vectors_config
        if not vc:
            return None
        if hasattr(vc, "config") and vc.config and hasattr(vc.config, "size"):
            return vc.config.size
        if hasattr(vc, "size"):
            return vc.size
    except Exception:
        return None
    return None

def ensure_collection(dim: int = VECTOR_SIZE):
    try:
        info = qdrant.get_collection(QDRANT_COLLECTION)
        existing_dim = _get_existing_dim(info)
        if existing_dim and existing_dim != dim:
            if QDRANT_AUTO_MIGRATE:
                qdrant.recreate_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                )
            else:
                raise RuntimeError(
                    f"Qdrant collection '{QDRANT_COLLECTION}' tem dim={existing_dim}, "
                    f"mas o modelo '{OPENAI_MODEL}' produz dim={dim}. "
                    f"Ativa QDRANT_AUTO_MIGRATE=1 ou muda QDRANT_COLLECTION."
                )
        return
    except Exception:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

ensure_collection(VECTOR_SIZE)

# =================== URL helpers / filtros ========================
DENY_PATTERNS = [
    "/carrinho", "/checkout", "/minha-conta", "/wp-login",
    "/feed", "/tag/", "/categoria/", "/author/", "/cart/", "/my-account/",
]
DENY_CONTAINS = [
    "add-to-cart=", "orderby=", "wc-ajax", "utm_", "replytocom=", "sessionid=",
]

DEFAULT_HEADERS = {
    "User-Agent": "alma-bot/1.0 (+https://example.com/bot) Python-requests",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

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

# =================== Texto / embeddings ===========================
def _chunk_text(text: str, max_tokens: int = 450) -> List[str]:
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
    resp = openai_client.embeddings.create(model=OPENAI_MODEL, input=texts)
    return [d.embedding for d in resp.data]  # 1536D

# ====================== Núcleo de ingest ==========================
def _ingest(namespace: str, url: str, title: str, full_text: str) -> int:
    chunks = _chunk_text(full_text)
    if not chunks:
        return 0

    vecs = _embed_texts(chunks)
    if vecs:
        embed_dim = len(vecs[0])
        if embed_dim != VECTOR_SIZE:
            if QDRANT_AUTO_MIGRATE:
                qdrant.recreate_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=qm.VectorParams(size=embed_dim, distance=qm.Distance.COSINE),
                )
            else:
                raise RuntimeError(
                    f"Dimensão do embedding={embed_dim} difere da coleção={VECTOR_SIZE}. "
                    f"Ativa QDRANT_AUTO_MIGRATE=1 ou recria coleção manualmente."
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

# ====================== Ingest público ============================
def ingest_text(title: str, text: str, namespace: str = "default") -> Dict:
    count = _ingest(namespace, f"text://{title}", title, text)
    return {"ok": True, "count": count}

def ingest_url(page_url: str, namespace: str = "default", deadline_s: int = 55) -> Dict:
    u = _clean_url(page_url)
    if not _url_allowed(u):
        return {"ok": False, "error": "url_blocked", "url": u}
    try:
        r = requests.get(u, timeout=FETCH_TIMEOUT_S, headers=DEFAULT_HEADERS)
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_failed: {e}", "url": u}
    soup = BeautifulSoup(r.text, "html.parser")
    title = (soup.title.string if soup.title else u).strip()
    text = soup.get_text(" ", strip=True)
    count = _ingest(namespace, u, title, text)
    return {"ok": True, "url": u, "count": count}

def ingest_pdf_url(pdf_url: str, title: Optional[str] = None, namespace: str = "default") -> Dict:
    import fitz
    try:
        r = requests.get(pdf_url, timeout=FETCH_TIMEOUT_S + 10, headers=DEFAULT_HEADERS)
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_pdf_failed: {e}", "url": pdf_url}
    doc = fitz.open("pdf", r.content)
    full = " ".join(page.get_text() for page in doc)
    count = _ingest(namespace, pdf_url, title or pdf_url, full)
    return {"ok": True, "url": pdf_url, "count": count}

# ---- helpers de ritmo para sitemaps ----
def _sleep_ms(ms: int):
    if ms > 0:
        time.sleep(ms / 1000.0)

def _backoff_sleep(attempt: int):
    # attempt: 1,2,3,... → (attempt * base) + jitter
    base = max(SITEMAP_BACKOFF_MS, 0) * max(attempt, 1)
    jitter = random.randint(0, max(SITEMAP_JITTER_MS, 0))
    _sleep_ms(base + jitter)

def _ingest_url_with_retry(url: str, namespace: str, deadline_s: int) -> Tuple[str, bool, str]:
    """Tenta ingerir uma URL com retries + backoff. Retorna (url, ok, err_msg)."""
    for attempt in range(1, SITEMAP_MAX_RETRIES + 1):
        res = ingest_url(url, namespace=namespace, deadline_s=deadline_s)
        if res.get("ok"):
            return (url, True, "")
        err = (res.get("error") or "unknown").lower()
        # Bloqueios lógicos não valem a pena re-tentar
        if "url_blocked" in err:
            return (url, False, "url_blocked")
        # Em 403/429/5xx/timeout → backoff e nova tentativa
        if any(code in err for code in ("403", "429", "5", "timeout", "temporarily")) and attempt < SITEMAP_MAX_RETRIES:
            _backoff_sleep(attempt)
            continue
        return (url, False, res.get("error", "unknown"))
    return (url, False, "max_retries_exceeded")

def ingest_sitemap(
    sitemap_url: str,
    namespace: str = "default",
    max_pages: Optional[int] = None,
    deadline_s: Optional[int] = None,
    start_index: Optional[int] = None,
    limit_per_call: Optional[int] = None,
) -> Dict:
    """Ingestão do sitemap com:
       - paginação por cursor (start_index/limit_per_call)
       - paralelismo controlado (thread pool)
       - retries + backoff
       - filtro de bloqueados/duplicados
    """
    max_pages = max_pages or CRAWL_MAX_PAGES
    deadline_s = deadline_s or CRAWL_DEADLINE_S
    limit_per_call = limit_per_call or SITEMAP_LIMIT_PER_CALL
    start_index = int(start_index or 0)

    try:
        r = requests.get(sitemap_url, timeout=FETCH_TIMEOUT_S, headers=DEFAULT_HEADERS)
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_sitemap_failed: {e}"}

    soup = BeautifulSoup(r.text, "xml")
    locs = [loc.get_text().strip() for loc in soup.find_all("loc")]
    total_locs = len(locs)

    # Cursor window
    end_index = min(total_locs, start_index + limit_per_call)
    window = locs[start_index:end_index]

    # Pré-filtrar e normalizar
    seen: set[str] = set()
    candidates: List[str] = []
    skipped_blocked: List[str] = []
    skipped_dupe: List[str] = []
    for loc in window:
        url = _clean_url(loc)
        if url in seen:
            skipped_dupe.append(url)
            continue
        seen.add(url)
        if not _url_allowed(url):
            skipped_blocked.append(url)
            continue
        candidates.append(url)

    ingested_urls: List[str] = []
    failed_urls: List[Tuple[str, str]] = []

    t0 = time.time()
    # Paraleliza em lotes com semáforo via ThreadPoolExecutor
    max_workers = max(1, SITEMAP_CONCURRENCY)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for i, url in enumerate(candidates):
            # pequeno espaçamento entre submissões para não fazer burst
            _sleep_ms(SITEMAP_SLEEP_MS)
            futures[pool.submit(_ingest_url_with_retry, url, namespace, deadline_s)] = url

        for fut in as_completed(futures):
            url = futures[fut]
            try:
                u, ok, err = fut.result()
                if ok:
                    ingested_urls.append(u)
                else:
                    failed_urls.append((u, err))
            except Exception as e:
                failed_urls.append((url, f"exception: {e}"))

    elapsed_s = round(time.time() - t0, 2)

    # Prepara cursor para próxima chamada
    next_cursor = end_index if end_index < total_locs else None

    return {
        "ok": True,
        "sitemap": sitemap_url,
        "namespace": namespace,
        "pages_ingested": len(ingested_urls),
        "pages_failed": len(failed_urls),
        "total_locs": total_locs,
        "start_index": start_index,
        "end_index": end_index,
        "next_cursor": next_cursor,
        "limits": {
            "max_pages": max_pages,
            "deadline_s": deadline_s,
            "limit_per_call": limit_per_call,
            "concurrency": max_workers,
            "retries": SITEMAP_MAX_RETRIES
        },
        "skipped_blocked": skipped_blocked[:200],
        "skipped_dupe": skipped_dupe[:200],
        "failed_urls": failed_urls[:200],
        "ingested_urls": ingested_urls[:200],
        "elapsed_s": elapsed_s,
    }

# ========================= Crawler =================================
def crawl_and_ingest(
    seed_url: str,
    namespace: str = "default",
    max_pages: Optional[int] = None,
    max_depth: Optional[int] = None,
    deadline_s: Optional[int] = None,
) -> Dict:
    max_pages = max_pages or CRAWL_MAX_PAGES
    max_depth = max_depth or CRAWL_MAX_DEPTH
    deadline_s = deadline_s or CRAWL_DEADLINE_S

    start = _clean_url(seed_url)
    seen, queue = set(), [(start, 0)]
    ok_chunks, fail = 0, 0
    t0 = time.time()
    start_host = urlsplit(start).netloc

    while queue and len(seen) < max_pages and (time.time() - t0) < deadline_s:
        url, depth = queue.pop(0)
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        if not _url_allowed(url):
            continue
        try:
            r = requests.get(url, timeout=FETCH_TIMEOUT_S, headers=DEFAULT_HEADERS)
            r.raise_for_status()
        except Exception:
            fail += 1
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.string if soup.title else url).strip()
        text = soup.get_text(" ", strip=True)
        ok_chunks += _ingest(namespace, url, title, text)

        for a in soup.find_all("a", href=True):
            nxt = urljoin(url, a["href"])
            nxt = _clean_url(nxt)
            if urlsplit(nxt).netloc != start_host:
                continue
            if nxt not in seen and _url_allowed(nxt):
                queue.append((nxt, depth + 1))

    return {"ok": True, "visited": len(seen), "ok_chunks": ok_chunks, "fail": fail, "namespace": namespace}

# ========================= Search ==================================
def search_chunks(query: str, namespace: Optional[str] = None, top_k: int = 6):
    vec = _embed_texts([query])[0]
    flt = qm.Filter(must=[qm.FieldCondition(
        key="namespace", match=qm.MatchValue(value=namespace or "default")
    )])
    res = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vec,
        limit=top_k,
        query_filter=flt
    )
    out = []
    for m in res:
        p = dict(m.payload or {})
        p["score"] = float(getattr(m, "score", 0.0))
        out.append(p)
    return out

def build_context_block(matches, token_budget: int = 1600):
    lines, used = [], 0
    for m in matches:
        t = m.get("text", "") or ""
        toks = len(t.split())
        if used + toks > token_budget:
            break
        lines.append(f"[{m.get('title')}] {t}")
        used += toks
    return "\n".join(lines)

# ========================= Public API ===============================
__all__ = [
    "ensure_collection",
    "ingest_text",
    "ingest_url",
    "ingest_pdf_url",
    "ingest_sitemap",
    "crawl_and_ingest",
    "search_chunks",
    "build_context_block",
    "QDRANT_COLLECTION",
]
