# rag_client.py — OpenAI embeddings (1536D) + Qdrant + crawler/sitemap com relatório de URLs
# ------------------------------------------------------------------------------------------
import os, time, uuid, requests
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, urljoin
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from openai import OpenAI

# ============================= Config =========================================

# Qdrant
QDRANT_URL         = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION  = os.getenv("QDRANT_COLLECTION", "alma_docs")  # default consistente

# OpenAI embeddings (ambos 1536D)
OPENAI_MODEL       = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # ou "text-embedding-3-large"
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")

# Limites “generosos” (podes sobrepor via ENV ou payloads dos endpoints)
UPSERT_BATCH       = int(os.getenv("UPSERT_BATCH", "64"))
TIMEOUT_FETCH_S    = int(os.getenv("FETCH_TIMEOUT_S", "20"))
DEFAULT_MAX_PAGES  = int(os.getenv("CRAWL_MAX_PAGES", "500"))
DEFAULT_MAX_DEPTH  = int(os.getenv("CRAWL_MAX_DEPTH", "4"))
DEFAULT_DEADLINE_S = int(os.getenv("RAG_DEADLINE_S", "120"))
SITEMAP_MAX_PAGES  = int(os.getenv("SITEMAP_MAX_PAGES", "1000"))

# Dimensão esperada (OpenAI text-embedding-3-* => 1536)
MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 1536,
}
VECTOR_SIZE = MODEL_DIMS.get(OPENAI_MODEL, 1536)

# ============================ Clients =========================================

qdrant = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()

# ====================== Collection ensure (sem destruir) =======================

def ensure_collection(dim: int = VECTOR_SIZE):
    """
    Garante que a coleção existe e tem a dimensão correta.
    Se existir com dimensão diferente, lança erro claro para não destruir dados.
    """
    try:
        info = qdrant.get_collection(QDRANT_COLLECTION)
        cfg = getattr(info, "config", None)
        existing_dim = None
        if cfg and getattr(cfg, "vectors_config", None):
            vc = cfg.vectors_config
            # qdrant_client>=1.7
            if hasattr(vc, "config") and getattr(vc.config, "size", None):
                existing_dim = vc.config.size
            elif getattr(vc, "size", None):
                existing_dim = vc.size
        if existing_dim and existing_dim != dim:
            raise RuntimeError(
                f"Qdrant collection '{QDRANT_COLLECTION}' tem dim={existing_dim}, "
                f"mas o modelo '{OPENAI_MODEL}' produz dim={dim}. "
                f"Cria nova coleção (env QDRANT_COLLECTION) ou reconfigura."
            )
        # Se chegou aqui, existe e está ok.
        return
    except Exception:
        # Não existe → cria
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

ensure_collection(VECTOR_SIZE)

# ======================= URL helpers / filtros ================================

DENY_PATTERNS = [
    "/carrinho", "/checkout", "/minha-conta", "/wp-login",
    "/feed", "/tag/", "/categoria/", "/author/", "/cart/", "/my-account/",
]
DENY_CONTAINS = [
    "add-to-cart=", "orderby=", "wc-ajax", "utm_", "replytocom=", "sessionid=",
]

def _clean_url(u: str) -> str:
    if not u:
        return ""
    p = urlsplit(u)
    path = p.path or "/"
    # se for diretório sem extensão, força slash
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

# ========================= Texto / embeddings ================================

def _chunk_text(text: str, max_tokens: int = 450) -> List[str]:
    """Split simples por frases (aprox)."""
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
    """Embeddings com OpenAI — devolve lista de vetores 1536D."""
    if not texts:
        return []
    # OpenAI respeita a ordem; usamos a API batched
    resp = openai_client.embeddings.create(model=OPENAI_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# ========================= Núcleo de ingest ==================================

def _ingest(namespace: str, url: str, title: str, full_text: str) -> int:
    chunks = _chunk_text(full_text)
    if not chunks:
        return 0

    # Embeddings
    vecs = _embed_texts(chunks)  # -> List[List[float]]
    if vecs and len(vecs[0]) != VECTOR_SIZE:
        # Proteção extra caso mudem o modelo e não atualizem a coleção
        raise RuntimeError(
            f"Dimensão de embedding {len(vecs[0])} != dimensão da coleção {VECTOR_SIZE}."
        )

    # Construção dos pontos
    points: List[qm.PointStruct] = []
    for idx, (c, v) in enumerate(zip(chunks, vecs)):
        points.append(qm.PointStruct(
            id=_uuid_for_chunk(namespace, url, idx),
            vector=v,
            payload={"url": url, "title": title, "text": c, "namespace": namespace}
        ))

    # Upsert em batches
    total = 0
    for i in range(0, len(points), UPSERT_BATCH):
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i+UPSERT_BATCH])
        total += len(points[i:i+UPSERT_BATCH])
    return total

# ========================= Ingest público ====================================

def ingest_text(title: str, text: str, namespace: str = "default"):
    count = _ingest(namespace, f"text://{title}", title, text)
    return {"ok": True, "count": count}

def ingest_url(page_url: str, namespace: str = "default", deadline_s: int = DEFAULT_DEADLINE_S):
    u = _clean_url(page_url)
    if not _url_allowed(u):
        return {"ok": False, "error": "url_blocked", "url": u}
    try:
        r = requests.get(u, timeout=TIMEOUT_FETCH_S, headers={"User-Agent": "alma-bot/1.0"})
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_failed: {e}", "url": u}
    soup = BeautifulSoup(r.text, "html.parser")
    title = (soup.title.string if soup.title else u).strip()
    text = soup.get_text(" ", strip=True)
    count = _ingest(namespace, u, title, text)
    return {"ok": True, "url": u, "count": count}

def ingest_pdf_url(pdf_url: str, title: Optional[str] = None, namespace: str = "default"):
    import fitz
    try:
        r = requests.get(pdf_url, timeout=TIMEOUT_FETCH_S + 10)
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_pdf_failed: {e}", "url": pdf_url}
    doc = fitz.open("pdf", r.content)
    full = " ".join(page.get_text() for page in doc)
    count = _ingest(namespace, pdf_url, title or pdf_url, full)
    return {"ok": True, "url": pdf_url, "count": count}

def ingest_sitemap(
    sitemap_url: str,
    namespace: str = "default",
    max_pages: int = SITEMAP_MAX_PAGES,
    deadline_s: int = DEFAULT_DEADLINE_S
):
    """
    Lê <loc> do sitemap e ingere, devolvendo relatório de URLs.
    """
    try:
        r = requests.get(sitemap_url, timeout=TIMEOUT_FETCH_S)
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_sitemap_failed: {e}"}

    # Usar parser XML; BeautifulSoup sem lxml pode avisar — é ok
    soup = BeautifulSoup(r.text, "xml")
    locs = [loc.get_text().strip() for loc in soup.find_all("loc") if loc.get_text().strip()]

    ok_chunks_total, fail = 0, 0
    urls_ok: List[str] = []
    urls_fail: List[str] = []
    t0 = time.time()

    for loc in locs[:max_pages]:
        if time.time() - t0 > deadline_s:
            break
        res = ingest_url(loc, namespace=namespace, deadline_s=deadline_s)
        if res.get("ok"):
            ok_chunks_total += res.get("count", 0)
            urls_ok.append(loc)
        else:
            fail += 1
            urls_fail.append(loc)

    return {
        "ok": True,
        "sitemap": sitemap_url,
        "pages_ingested": len(urls_ok),
        "pages_failed": fail,
        "urls_ok": urls_ok,
        "urls_fail": urls_fail,
        "namespace": namespace,
    }

# ========================= Crawler (multi-página) ============================

def crawl_and_ingest(
    seed_url: str,
    namespace: str = "default",
    max_pages: int = DEFAULT_MAX_PAGES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    deadline_s: int = DEFAULT_DEADLINE_S
):
    start = _clean_url(seed_url)
    seen, queue = set(), [(start, 0)]
    ok_chunks, fail = 0, 0
    urls_ok: List[str] = []
    urls_fail: List[str] = []
    t0 = time.time()

    while queue and len(seen) < max_pages and (time.time() - t0) < deadline_s:
        url, depth = queue.pop(0)
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        if not _url_allowed(url):
            continue

        try:
            r = requests.get(url, timeout=TIMEOUT_FETCH_S, headers={"User-Agent": "alma-bot/1.0"})
            r.raise_for_status()
        except Exception:
            fail += 1
            urls_fail.append(url)
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.string if soup.title else url).strip()
        text = soup.get_text(" ", strip=True)

        try:
            added = _ingest(namespace, url, title, text)
            ok_chunks += added
            urls_ok.append(url)
        except Exception:
            fail += 1
            urls_fail.append(url)
            # não paramos o crawl — seguimos

        # próximos links (mesmo domínio)
        for a in soup.find_all("a", href=True):
            nxt = _clean_url(urljoin(url, a["href"]))
            if urlsplit(nxt).netloc != urlsplit(start).netloc:
                continue
            if nxt not in seen and _url_allowed(nxt):
                queue.append((nxt, depth + 1))

    return {
        "ok": True,
        "visited": len(seen),
        "ok_chunks": ok_chunks,
        "fail": fail,
        "namespace": namespace,
        "urls_ok": urls_ok,
        "urls_fail": urls_fail,
    }

# ========================= Search ============================================

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
