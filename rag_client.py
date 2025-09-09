# rag_client.py — OpenAI embeddings (1536D) + Qdrant + crawler/sitemap (com chunking por tokens)
# ----------------------------------------------------------------------------------------------
import os, time, uuid, requests
from typing import List, Dict, Optional, Tuple, Iterable
from urllib.parse import urlsplit, urlunsplit, urljoin
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from openai import OpenAI

# =========================== Config ================================
QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "alma_docs")  # coleção única
OPENAI_MODEL      = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536D
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
UPSERT_BATCH      = int(os.getenv("UPSERT_BATCH", "64"))
FETCH_TIMEOUT_S   = int(os.getenv("FETCH_TIMEOUT_S", "20"))
QDRANT_AUTO_MIGRATE = os.getenv("QDRANT_AUTO_MIGRATE", "1") == "1"

# Chunking seguro para embeddings
RAG_CHUNK_TOKENS  = int(os.getenv("RAG_CHUNK_TOKENS", "7500"))  # < 8192
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# =========================== Crawling defaults =====================
CRAWL_MAX_PAGES   = int(os.getenv("CRAWL_MAX_PAGES", "500"))
CRAWL_DEADLINE_S  = int(os.getenv("CRAWL_DEADLINE_S", "500"))
CRAWL_MAX_DEPTH   = int(os.getenv("CRAWL_MAX_DEPTH", "4"))

# Sitemap “chunking” (para UI com cursor/limite por chamada)
LIMIT_PER_CALL    = int(os.getenv("LIMIT_PER_CALL", "60"))          # quantos URLs por clique
SITE_CONCURRENCY  = int(os.getenv("SITE_CONCURRENCY", "6"))         # informativo
SITEMAP_MAX_RETRIES = int(os.getenv("SITEMAP_MAX_RETRIES", "2"))    # por URL (leve)
SITEMAP_SLEEP_MS  = int(os.getenv("SITEMAP_SLEEP_MS", "0"))         # pausa entre URLs (ms)

# Dimensão do embedding
MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,  # oficial
}
VECTOR_SIZE = MODEL_DIMS.get(OPENAI_MODEL, 1536)

print(
    "[rag_client] defaults → "
    f"max_pages={CRAWL_MAX_PAGES}, deadline_s={CRAWL_DEADLINE_S}, max_depth={CRAWL_MAX_DEPTH}, "
    f"embed_model={OPENAI_MODEL}, vector_size={VECTOR_SIZE}, collection={QDRANT_COLLECTION}, "
    f"site_concurrency={SITE_CONCURRENCY}, limit_per_call={LIMIT_PER_CALL}, "
    f"chunk_tokens={RAG_CHUNK_TOKENS}, overlap={RAG_CHUNK_OVERLAP}"
)

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

# ---- payload index p/ namespace (filtro rápido) ----
def ensure_payload_indexes():
    try:
        qdrant.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="namespace",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        print("[rag_client] payload index 'namespace' criado (KEYWORD)")
    except Exception as e:
        print(f"[rag_client] payload index 'namespace' pode já existir: {e}")

ensure_collection(VECTOR_SIZE)
ensure_payload_indexes()

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

# =================== Chunking / embeddings ========================
def _chunks_for_embedding(text: str, max_tokens: int = RAG_CHUNK_TOKENS, overlap: int = RAG_CHUNK_OVERLAP) -> Iterable[str]:
    """
    Divide 'text' em segmentos <= max_tokens com sobreposição 'overlap'.
    Usa tiktoken se disponível; caso contrário, fallback aproximado por caracteres (~4 chars/token).
    """
    text = (text or "").strip()
    if not text:
        return []

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        step = max(1, max_tokens - overlap)
        for i in range(0, len(toks), step):
            yield enc.decode(toks[i:i + max_tokens])
    except Exception:
        # Fallback grosseiro: ~4 chars ≈ 1 token
        max_chars = max_tokens * 4
        step = max(1, (max_tokens - overlap) * 4)
        for i in range(0, len(text), step):
            yield text[i:i + max_chars]

def _embed_texts(texts: List[str], batch_size: int = 128) -> List[List[float]]:
    """
    Cria embeddings em batches. Protege contra inputs vazios.
    """
    out: List[List[float]] = []
    buf = [t if isinstance(t, str) else "" for t in (texts or [])]
    buf = [t for t in buf if t.strip()]
    if not buf:
        return out

    for i in range(0, len(buf), batch_size):
        batch = buf[i:i + batch_size]
        # Retry leve
        last_err = None
        for attempt in range(2):
            try:
                resp = openai_client.embeddings.create(model=OPENAI_MODEL, input=batch)
                out.extend([d.embedding for d in resp.data])
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.6)
        if last_err:
            raise last_err
    return out

# ====================== Núcleo de ingest ==========================
def _ingest(namespace: str, url: str, title: str, full_text: str) -> int:
    # 1) chunking por tokens (seguro para modelos 8k)
    chunks = list(_chunks_for_embedding(full_text, RAG_CHUNK_TOKENS, RAG_CHUNK_OVERLAP))
    if not chunks:
        return 0

    # 2) embeddings em batch
    vecs = _embed_texts(chunks)

    # 3) sanity-check dimensão da coleção
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

    # 4) upsert em batches
    points: List[qm.PointStruct] = []
    for idx, (c, v) in enumerate(zip(chunks, vecs)):
        points.append(qm.PointStruct(
            id=_uuid_for_chunk(namespace, url, idx),
            vector=v,
            payload={"url": url, "title": title, "text": c, "namespace": namespace, "chunk_idx": idx}
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
    err = None
    for att in range(1, SITEMAP_MAX_RETRIES + 1):
        try:
            r = requests.get(u, timeout=FETCH_TIMEOUT_S, headers=DEFAULT_HEADERS)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            title = (soup.title.string if soup.title else u).strip()
            text = soup.get_text(" ", strip=True)
            count = _ingest(namespace, u, title, text)
            return {"ok": True, "url": u, "count": count}
        except Exception as e:
            err = e
            time.sleep(max(SITEMAP_SLEEP_MS, 0) / 1000.0)
    return {"ok": False, "error": f"fetch_failed: {err}", "url": u}


# --- tuning opcional por env ---
PDF_PAGES_PER_BATCH = int(os.getenv("PDF_PAGES_PER_BATCH", "200"))  # nº de páginas por lote (200)
PDF_MAX_PAGES       = int(os.getenv("PDF_MAX_PAGES", "3000"))       # corta PDFs absurdos
PDF_SKIP_EMPTY      = os.getenv("PDF_SKIP_EMPTY", "1") in ("1","true","yes")

def ingest_pdf_url(pdf_url: str, title: Optional[str] = None, namespace: str = "default") -> Dict:
    """
    Ingest de PDF remoto em LOTES de páginas (streaming), para evitar limites de tokens/memória.
    Mantém a mesma API externa. IDs não colidem porque o URL leva sufixo com o intervalo de páginas.
    """
    import re
    import fitz  # PyMuPDF

    if not pdf_url:
        return {"ok": False, "error": "missing_pdf_url"}

    u = pdf_url.strip()

    # Normalizar Google Drive (view/open -> uc?export=download&id=...)
    if "drive.google.com" in u:
        if "/uc?" not in u or "id=" not in u:
            m = re.search(r"/d/([A-Za-z0-9_-]{20,})", u) or re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", u)
            if m:
                file_id = m.group(1)
                u = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Download
    try:
        r = requests.get(u, timeout=FETCH_TIMEOUT_S + 60, headers=DEFAULT_HEADERS, allow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_failed: {e}", "url": u}

    # Abrir PDF
    try:
        doc = fitz.open("pdf", r.content)
    except Exception as e:
        return {"ok": False, "error": f"pdf_open_failed: {e}", "url": u}

    total_pages = min(getattr(doc, "page_count", len(doc) if hasattr(doc, "__len__") else 0), PDF_MAX_PAGES)
    if total_pages == 0:
        return {"ok": True, "url": u, "count": 0, "warning": "pdf_has_no_pages"}

    # Processar por lotes de 200
    per = max(1, PDF_PAGES_PER_BATCH)
    grand_total_chunks = 0
    batches_info: List[Dict] = []

    for start in range(0, total_pages, per):
        end = min(start + per, total_pages)
        texts: List[str] = []

        for p in range(start, end):
            try:
                t = doc[p].get_text() or ""
            except Exception:
                t = ""
            if PDF_SKIP_EMPTY and not t.strip():
                continue
            if t.strip():
                texts.append(f"[Página {p+1}] {t.strip()}")

        if not texts:
            batches_info.append({"range": [start+1, end], "chunks": 0, "skipped": True})
            continue

        url_with_range = f"{pdf_url}#p{start+1}-{end}"
        chunk_title = (title or pdf_url)

        try:
            count = _ingest(namespace, url_with_range, chunk_title, "\n\n".join(texts))
            grand_total_chunks += count
            batches_info.append({"range": [start+1, end], "chunks": count})
        except Exception as e:
            batches_info.append({"range": [start+1, end], "error": str(e)})
            # continua nos lotes seguintes

    return {
        "ok": True,
        "url": u,
        "title": title or None,
        "pages": total_pages,
        "pages_per_batch": per,
        "count": grand_total_chunks,
        "batches": batches_info,
    }


# ====================== Sitemap (com cursor/limite) ================
def ingest_sitemap(
    sitemap_url: str,
    namespace: str = "default",
    max_pages: Optional[int] = None,
    deadline_s: Optional[int] = None,
    cursor: Optional[int] = None,           # índice inicial (para “Próximo bloco”)
    limit_per_call: Optional[int] = None,   # override do limite por chamada
) -> Dict:
    max_pages = max_pages or CRAWL_MAX_PAGES
    deadline_s = deadline_s or CRAWL_DEADLINE_S
    limit = int(limit_per_call or LIMIT_PER_CALL)

    try:
        r = requests.get(sitemap_url, timeout=FETCH_TIMEOUT_S, headers=DEFAULT_HEADERS)
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_sitemap_failed: {e}"}

    soup = BeautifulSoup(r.text, "xml")
    locs = [loc.get_text().strip() for loc in soup.find_all("loc")]
    total_locs = len(locs)

    # janela desta chamada
    start_idx = int(cursor or 0)
    if start_idx < 0: start_idx = 0
    end_idx = min(start_idx + limit, total_locs)
    window = locs[start_idx:end_idx]

    seen: set[str] = set()
    ingested_urls: List[str] = []
    failed_urls: List[Tuple[str, str]] = []
    skipped_blocked: List[str] = []
    skipped_dupe: List[str] = []

    t0 = time.time()
    for raw in window:
        if len(ingested_urls) >= max_pages:
            break
        if (time.time() - t0) > deadline_s:
            break

        url = _clean_url(raw)

        if url in seen:
            skipped_dupe.append(url)
            continue
        seen.add(url)

        if not _url_allowed(url):
            skipped_blocked.append(url)
            continue

        res = ingest_url(url, namespace=namespace, deadline_s=deadline_s)
        if res.get("ok"):
            ingested_urls.append(url)
        else:
            failed_urls.append((url, res.get("error", "unknown")))

        if SITEMAP_SLEEP_MS > 0:
            time.sleep(SITEMAP_SLEEP_MS / 1000.0)

    next_cursor = end_idx if end_idx < total_locs else None
    elapsed_s = round(time.time() - t0, 2)
    return {
        "ok": True,
        "sitemap": sitemap_url,
        "namespace": namespace,
        "pages_ingested": len(ingested_urls),
        "pages_failed": len(failed_urls),
        "total_locs": total_locs,
        "start_index": start_idx,
        "end_index": end_idx if end_idx <= total_locs else total_locs,
        "next_cursor": next_cursor,
        "limits": {
            "max_pages": max_pages,
            "deadline_s": deadline_s,
            "limit_per_call": limit,
            "concurrency": SITE_CONCURRENCY,
            "retries": SITEMAP_MAX_RETRIES,
        },
        "skipped_blocked": skipped_blocked[:200],
        "skipped_dupe": skipped_dupe[:200],
        "failed_urls": failed_urls[:200],
        "ingested_urls": ingested_urls[:200],
        "elapsed_s": elapsed_s,
    }

# ========================= Crawler (mesmo domínio) =================
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
