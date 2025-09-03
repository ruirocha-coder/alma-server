# rag_client.py — OpenAI Embeddings + Qdrant + Crawler/Sitemap robusto
# -----------------------------------------------------------------------------
import os
import time
import uuid
import requests
from typing import List, Dict, Optional, Callable
from urllib.parse import urlsplit, urlunsplit, urljoin

from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from openai import OpenAI

# ============================ Configuração ===================================

# --- Qdrant / OpenAI ---
QDRANT_URL         = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION  = os.getenv("QDRANT_COLLECTION", "alma_docs")

# Modelos embeddings OpenAI recomendados (2024+)
OPENAI_MODEL       = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")

# Dimensões por modelo (atenção: 3-large = 3072, 3-small = 1536)
MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
VECTOR_SIZE        = MODEL_DIMS.get(OPENAI_MODEL, 1536)

# --- Limites generosos por defeito (ajustáveis por env) ---
# comerciais grandes costumam ter 200–2.000 páginas relevantes; damos folga
CRAWL_MAX_PAGES    = int(os.getenv("CRAWL_MAX_PAGES", "1500"))
CRAWL_MAX_DEPTH    = int(os.getenv("CRAWL_MAX_DEPTH", "5"))
RAG_DEADLINE_S     = int(os.getenv("RAG_DEADLINE_S", "240"))

# Upserts maiores para acelerar (128 é seguro para Qdrant Cloud)
UPSERT_BATCH       = int(os.getenv("UPSERT_BATCH", "128"))

# Networking
TIMEOUT_FETCH_S    = int(os.getenv("FETCH_TIMEOUT_S", "20"))
USER_AGENT         = os.getenv("CRAWL_UA", "alma-bot/1.0 (+https://example.com)")

# Comportamento em caso de “dimensão errada” na coleção
# Se True, apaga e recria automaticamente com a dimensão correta
RAG_RECREATE_ON_DIM_MISMATCH = os.getenv("RAG_RECREATE_ON_DIM_MISMATCH", "false").lower() == "true"

# ============================ Clientes =======================================

qdrant = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()

# ============================ Helpers URL / Filtros ==========================

DENY_PATTERNS = [
    "/carrinho", "/checkout", "/minha-conta", "/wp-login",
    "/cart/", "/my-account/", "/account/", "/privacy-policy",
    "/feed", "/tag/", "/categoria/", "/author/", "/search/",
]
DENY_CONTAINS = [
    "add-to-cart=", "orderby=", "wc-ajax", "utm_", "replytocom=",
    "sessionid=", "fbclid=", "gclid=", "adid=", "variant=",
]

def _clean_url(u: str) -> str:
    if not u:
        return ""
    p = urlsplit(u)
    path = p.path or "/"
    # diretórios sem extensão recebem slash para evitar duplicados
    if not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        path += "/"
    return urlunsplit((p.scheme, p.netloc, path, "", ""))

def _url_allowed(u: str) -> bool:
    low = u.lower()
    if not (low.startswith("http://") or low.startswith("https://")):
        return False
    for pat in DENY_PATTERNS:
        if pat in low:
            return False
    for pat in DENY_CONTAINS:
        if pat in low:
            return False
    return True

def _same_host(a: str, b: str) -> bool:
    return urlsplit(a).netloc == urlsplit(b).netloc

def _uuid_for_chunk(namespace: str, url: str, idx: int) -> str:
    base = f"{namespace}|{url}|{idx}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

# ============================ Embeddings / Texto =============================

def _chunk_text(text: str, max_tokens: int = 450) -> List[str]:
    """
    Chunk simples por “frases”. 450 ~ seguro para embeddings e boa densidade.
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
    """Embeddings com OpenAI — list[list[float]] com a dimensão do modelo ativo."""
    if not texts:
        return []
    resp = openai_client.embeddings.create(model=OPENAI_MODEL, input=texts)
    # mantém ordem
    return [d.embedding for d in resp.data]

# ============================ Qdrant Collection ==============================

def _get_collection_dim(name: str) -> Optional[int]:
    try:
        info = qdrant.get_collection(name)
        cfg = getattr(info, "config", None)
        if not cfg:
            return None
        vc = getattr(cfg, "vectors_config", None)
        if not vc:
            return None
        # qdrant >=1.7 tem .config.size; fallback .size
        if hasattr(vc, "config") and getattr(vc.config, "size", None):
            return vc.config.size
        if getattr(vc, "size", None):
            return vc.size
    except Exception:
        return None
    return None

def ensure_collection(dim: int = VECTOR_SIZE):
    """
    Garante que a coleção existe e tem a dimensão igual ao modelo.
    - Se não existir: cria com 'dim'.
    - Se existir e a dimensão diferir:
        * se RAG_RECREATE_ON_DIM_MISMATCH=True → recria
        * senão → lança RuntimeError com mensagem clara.
    """
    try:
        existing_dim = _get_collection_dim(QDRANT_COLLECTION)
        if existing_dim is None:
            # cria
            qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )
            return
        if existing_dim != dim:
            msg = (
                f"Qdrant collection '{QDRANT_COLLECTION}' tem dim={existing_dim}, "
                f"mas o modelo '{OPENAI_MODEL}' produz dim={dim}."
            )
            if RAG_RECREATE_ON_DIM_MISMATCH:
                # ⚠️ recria (apaga dados)
                qdrant.recreate_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                )
            else:
                raise RuntimeError(msg + " Defina RAG_RECREATE_ON_DIM_MISMATCH=true para recriar automaticamente "
                                         "ou mude QDRANT_COLLECTION para uma coleção nova.")
    except Exception as e:
        # Se der “já existe” (409), tentamos seguir; mais erros sobem
        if "already exists" in str(e):
            return
        raise

# chama já ao importar
ensure_collection(VECTOR_SIZE)

# ============================ Núcleo de ingest ===============================

def _ingest(namespace: str, url: str, title: str, full_text: str) -> int:
    chunks = _chunk_text(full_text)
    if not chunks:
        return 0

    vecs = _embed_texts(chunks)
    # verificação de dimensão defensiva (não custa nada)
    if vecs:
        embed_dim = len(vecs[0])
        col_dim = _get_collection_dim(QDRANT_COLLECTION)
        if col_dim is not None and col_dim != embed_dim:
            if RAG_RECREATE_ON_DIM_MISMATCH:
                qdrant.recreate_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=qm.VectorParams(size=embed_dim, distance=qm.Distance.COSINE),
                )
            else:
                raise RuntimeError(
                    f"Dimensão da coleção ({col_dim}) != dimensão do embedding ({embed_dim}). "
                    f"Ajuste o modelo ou recrie a coleção."
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

# ============================ Ingest público =================================

def ingest_text(title: str, text: str, namespace: str = "default") -> Dict:
    count = _ingest(namespace, f"text://{title}", title, text)
    return {"ok": True, "count": count}

def ingest_url(page_url: str, namespace: str = "default", deadline_s: int = RAG_DEADLINE_S) -> Dict:
    u = _clean_url(page_url)
    if not _url_allowed(u):
        return {"ok": False, "error": "url_blocked", "url": u}
    try:
        r = requests.get(u, timeout=TIMEOUT_FETCH_S, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
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
        r = requests.get(pdf_url, timeout=TIMEOUT_FETCH_S + 10, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch_pdf_failed: {e}", "url": pdf_url}
    doc = fitz.open("pdf", r.content)
    full = " ".join(page.get_text() for page in doc)
    count = _ingest(namespace, pdf_url, title or pdf_url, full)
    return {"ok": True, "url": pdf_url, "count": count}

# ============================ Sitemap ========================================

def _extract_sitemap_locs(xml_text: str) -> List[str]:
    soup = BeautifulSoup(xml_text, "xml")
    locs = [loc.get_text().strip() for loc in soup.find_all("loc")]
    return [u for u in locs if u]

def ingest_sitemap(
    sitemap_url: str,
    namespace: str = "default",
    max_pages: int = 2000,
    deadline_s: int = RAG_DEADLINE_S
) -> Dict:
    """
    Processa sitemap(s) sem mandar XML ao LLM. Suporta sitemapindex → urlset.
    Limita o total de páginas ingeridas a max_pages.
    """
    t0 = time.time()
    pages_ok, pages_failed = 0, 0
    seen_urls = set()

    def _fetch(url: str) -> Optional[str]:
        try:
            resp = requests.get(url, timeout=TIMEOUT_FETCH_S, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            return resp.text
        except Exception:
            return None

    root_xml = _fetch(sitemap_url)
    if not root_xml:
        return {"ok": False, "error": f"fetch_sitemap_failed: could not fetch {sitemap_url}"}

    roots = _extract_sitemap_locs(root_xml)
    host = urlsplit(sitemap_url).netloc

    # Se o sitemap raiz já for urlset simples, roots conterá as páginas finais
    # Se for sitemapindex, roots conterá os sub-sitemaps
    to_visit = roots or [sitemap_url]

    for item in to_visit:
        if time.time() - t0 > deadline_s or pages_ok >= max_pages:
            break
        xml = _fetch(item) if item != sitemap_url else root_xml
        if not xml:
            continue
        locs = _extract_sitemap_locs(xml)
        if not locs:
            # pode ser o próprio urlset (root era urlset)
            locs = roots

        for loc in locs:
            if time.time() - t0 > deadline_s or pages_ok >= max_pages:
                break
            if urlsplit(loc).netloc != host:
                # mantém no domínio do sitemap
                continue
            if loc in seen_urls:
                continue
            seen_urls.add(loc)
            res = ingest_url(loc, namespace=namespace, deadline_s=deadline_s)
            if res.get("ok"):
                pages_ok += 1
            else:
                pages_failed += 1

    return {
        "ok": True,
        "sitemap": sitemap_url,
        "pages_ingested": pages_ok,
        "pages_failed": pages_failed,
        "namespace": namespace,
    }

# ============================ Crawler ========================================

def crawl_and_ingest(
    seed_url: str,
    namespace: str = "default",
    max_pages: int = CRAWL_MAX_PAGES,
    max_depth: int = CRAWL_MAX_DEPTH,
    deadline_s: int = RAG_DEADLINE_S,
    progress_cb: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """
    Crawler BFS simples, mesmo domínio, respeita limites. Em cada página:
    - fetch + parse
    - upsert dos chunks no Qdrant
    """
    start = _clean_url(seed_url)
    seen, queue = set(), [(start, 0)]
    ok_chunks, fail = 0, 0
    t0 = time.time()
    host = urlsplit(start).netloc

    if progress_cb:
        progress_cb({"kind": "start", "url": start, "namespace": namespace,
                     "max_pages": max_pages, "max_depth": max_depth, "deadline_s": deadline_s})

    while queue and len(seen) < max_pages and time.time() - t0 < deadline_s:
        url, depth = queue.pop(0)
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        if not _url_allowed(url) or urlsplit(url).netloc != host:
            continue

        if progress_cb:
            progress_cb({"kind": "visit", "url": url, "depth": depth})

        try:
            r = requests.get(url, timeout=TIMEOUT_FETCH_S, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
        except Exception as e:
            fail += 1
            if progress_cb:
                progress_cb({"kind": "error_fetch", "url": url, "msg": str(e)})
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.string if soup.title else url).strip()
        text = soup.get_text(" ", strip=True)
        try:
            ing = _ingest(namespace, url, title, text)
            ok_chunks += ing
        except Exception as e:
            fail += 1
            if progress_cb:
                progress_cb({"kind": "error_upsert", "url": url, "msg": str(e)})

        # próximos links
        for a in soup.find_all("a", href=True):
            nxt = _clean_url(urljoin(url, a["href"]))
            if nxt not in seen and _url_allowed(nxt) and _same_host(nxt, start):
                queue.append((nxt, depth + 1))

        if progress_cb and len(seen) % 20 == 0:
            progress_cb({"kind": "queue_size", "size": len(queue)})

    return {"ok": True, "visited": len(seen), "ok_chunks": ok_chunks, "fail": fail, "namespace": namespace}

# ============================ Search =========================================

def search_chunks(query: str, namespace: Optional[str] = None, top_k: int = 6) -> List[Dict]:
    ns = (namespace or "default").strip() or "default"  # vazio -> "default"
    vec = _embed_texts([query])[0]
    flt = qm.Filter(must=[qm.FieldCondition(
        key="namespace", match=qm.MatchValue(value=ns)
    )])
    try:
        res = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vec,
            limit=top_k,
            query_filter=flt
        )
    except Exception as e:
        # devolve erro claro à API de cima
        raise RuntimeError(f"qdrant_search_failed: {e}")

    out = []
    for m in res:
        p = dict(m.payload or {})
        p["score"] = float(getattr(m, "score", 0.0))
        out.append(p)
    return out

def build_context_block(matches: List[Dict], token_budget: int = 1600) -> str:
    lines, used = [], 0
    for m in matches:
        t = m.get("text", "") or ""
        toks = len(t.split())
        if used + toks > token_budget:
            break
        lines.append(f"[{m.get('title')}] {t}")
        used += toks
    return "\n".join(lines)
