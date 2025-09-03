# rag_client.py — OpenAI embeddings + Qdrant + crawler/sitemap
# -------------------------------------------------------------------
import os
import time
import uuid
import requests
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, urljoin

from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from openai import OpenAI

# =========================== Config ================================

QDRANT_URL         = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION  = os.getenv("QDRANT_COLLECTION", "alma_docs")

# Modelo de embeddings (Escolhe um destes):
# - text-embedding-3-small  -> 1536 dims (barato/rápido)
# - text-embedding-3-large  -> 3072 dims (mais qualidade)
OPENAI_MODEL       = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")

# Se True, quando a dimensão da coleção não bater no arranque, recria a coleção
AUTO_MIGRATE_COLLECTION = os.getenv("AUTO_MIGRATE_COLLECTION", "true").lower() in ("1", "true", "yes")

UPSERT_BATCH       = int(os.getenv("UPSERT_BATCH", "64"))
TIMEOUT_FETCH_S    = int(os.getenv("FETCH_TIMEOUT_S", "20"))
CRAWL_UA           = os.getenv("CRAWL_UA", "alma-bot/1.0 (+https://example.invalid)")

# Dimensões conhecidas
MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
VECTOR_SIZE = MODEL_DIMS.get(OPENAI_MODEL)
if VECTOR_SIZE is None:
    # fallback seguro
    VECTOR_SIZE = 1536

# ======================= Clients / Setup ===========================

qdrant = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()


def _get_collection_dim() -> Optional[int]:
    """Tenta descobrir a dimensão atual da coleção no Qdrant."""
    try:
        info = qdrant.get_collection(QDRANT_COLLECTION)
        cfg = getattr(info, "config", None)
        vc = getattr(cfg, "vectors_config", None)
        # qdrant_client varia um pouco a API; cobrimos os casos comuns
        if vc is None:
            return None
        # Caso 1: vc.config.size
        if hasattr(vc, "config") and getattr(vc.config, "size", None):
            return int(vc.config.size)
        # Caso 2: vc.size diretamente
        if getattr(vc, "size", None):
            return int(vc.size)
        return None
    except Exception:
        return None


def _ensure_collection(dim: int):
    """Garante que a coleção existe com a dimensão pedida; recria se permitido e necessário."""
    current = _get_collection_dim()
    if current is None:
        # Não existe: cria
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        return

    if current != dim:
        if not AUTO_MIGRATE_COLLECTION:
            raise RuntimeError(
                f"A coleção '{QDRANT_COLLECTION}' tem dimensão {current}, "
                f"mas o modelo '{OPENAI_MODEL}' produz {dim}. "
                f"Defina AUTO_MIGRATE_COLLECTION=true para recriar automaticamente "
                f"ou mude QDRANT_COLLECTION."
            )
        # Recria (⚠️ apaga dados antigos dessa coleção)
        qdrant.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )


# chama no import
_ensure_collection(VECTOR_SIZE)

# ===================== URL helpers / filtros =======================

DENY_PATTERNS = [
    "/carrinho", "/checkout", "/minha-conta", "/wp-login",
    "/feed", "/tag/", "/categoria/", "/author/",
    "/cart/", "/my-account/", "/wp-json/", "/wp-admin/",
]
DENY_CONTAINS = [
    "add-to-cart=", "orderby=", "wc-ajax", "utm_", "replytocom=",
    "sessionid=", "sessid=", "phpsessid=", "fbclid=", "gclid=",
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

# ===================== Texto / embeddings ==========================

def _chunk_text(text: str, max_tokens: int = 450) -> List[str]:
    """Split simples por frases aproximadas."""
    parts: List[str] = []
    buf: List[str] = []
    count = 0
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
    """Embeddings com OpenAI, respeita a ordem do input."""
    if not texts:
        return []
    resp = openai_client.embeddings.create(model=OPENAI_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# ====================== Núcleo de ingest ===========================

def _ingest(namespace: str, url: str, title: str, full_text: str) -> int:
    chunks = _chunk_text(full_text)
    if not chunks:
        return 0

    vecs = _embed_texts(chunks)  # -> List[List[float]]

    # AUTO-FIX em runtime: se por algum motivo a coleção tiver dimensão trocada, realinha
    if vecs:
        embed_dim = len(vecs[0])
        current = _get_collection_dim()
        if current is None or current != embed_dim:
            if AUTO_MIGRATE_COLLECTION:
                qdrant.recreate_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=qm.VectorParams(size=embed_dim, distance=qm.Distance.COSINE),
                )
            else:
                raise RuntimeError(
                    f"Dimensão da coleção ({current}) != embeddings ({embed_dim}). "
                    f"Ativa AUTO_MIGRATE_COLLECTION ou ajusta a coleção."
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

# ==================== Ingest público ===============================

def ingest_text(title: str, text: str, namespace: str = "default"):
    count = _ingest(namespace, f"text://{title}", title, text)
    return {"ok": True, "count": count}

def ingest_url(page_url: str, namespace: str = "default", deadline_s: int = 55):
    u = _clean_url(page_url)
    if not _url_allowed(u):
        return {"ok": False, "error": "url_blocked", "url": u}
    try:
        r = requests.get(u, timeout=TIMEOUT_FETCH_S, headers={"User-Agent": CRAWL_UA})
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_failed: {e}", "url": u}
    soup = BeautifulSoup(r.text, "html.parser")
    title = (soup.title.string if soup.title else u).strip()
    text = soup.get_text(" ", strip=True)
    count = _ingest(namespace, u, title, text)
    return {"ok": True, "url": u, "count": count}

def ingest_pdf_url(pdf_url: str, title: Optional[str] = None, namespace: str = "default"):
    import fitz  # PyMuPDF
    try:
        r = requests.get(pdf_url, timeout=TIMEOUT_FETCH_S + 20, headers={"User-Agent": CRAWL_UA})
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_pdf_failed: {e}", "url": pdf_url}
    doc = fitz.open("pdf", r.content)
    full = " ".join(page.get_text() for page in doc)
    count = _ingest(namespace, pdf_url, title or pdf_url, full)
    return {"ok": True, "url": pdf_url, "count": count}

def ingest_sitemap(sitemap_url: str, namespace: str = "default", max_pages: int = 100, deadline_s: int = 55):
    try:
        r = requests.get(sitemap_url, timeout=TIMEOUT_FETCH_S, headers={"User-Agent": CRAWL_UA})
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

# ======================== Crawler ==================================

def crawl_and_ingest(seed_url: str, namespace: str = "default",
                     max_pages: int = 200, max_depth: int = 3, deadline_s: int = 55):
    start = _clean_url(seed_url)
    seen, queue = set(), [(start, 0)]
    ok_chunks, fail = 0, 0
    t0 = time.time()
    start_netloc = urlsplit(start).netloc

    while queue and len(seen) < max_pages and time.time() - t0 < deadline_s:
        url, depth = queue.pop(0)
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        if not _url_allowed(url):
            continue
        try:
            r = requests.get(url, timeout=TIMEOUT_FETCH_S, headers={"User-Agent": CRAWL_UA})
            r.raise_for_status()
        except Exception:
            fail += 1
            continue
        ctype = r.headers.get("Content-Type", "")
        if "text/html" not in ctype.lower():
            # ignora imagens/pdf/etc. (usa endpoint próprio p/ PDF)
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.string if soup.title else url).strip()
        text = soup.get_text(" ", strip=True)
        ok_chunks += _ingest(namespace, url, title, text)
        # próximos links (mesmo domínio)
        for a in soup.find_all("a", href=True):
            nxt = _clean_url(urljoin(url, a["href"]))
            if urlsplit(nxt).netloc != start_netloc:
                continue
            if nxt not in seen and _url_allowed(nxt):
                queue.append((nxt, depth + 1))

    return {
        "ok": True,
        "visited": len(seen),
        "ok_chunks": ok_chunks,
        "fail": fail,
        "namespace": namespace
    }

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
        title = m.get("title") or ""
        lines.append(f"[{title}] {t}")
        used += toks
    return "\n".join(lines)
