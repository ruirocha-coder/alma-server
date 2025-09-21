# main.py — Alma Server (RAG + Memória; top-k dinâmico; mini-pesquisa; injeção e fallback de links; Consola RAG; utilitários)
# ---------------------------------------------------------------------------------------
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

import os
import requests
import logging
import uvicorn
import time
import re
import csv
from io import StringIO
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, urlunparse
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------------------
# FastAPI & CORS
# ---------------------------------------------------------------------------------------
app = FastAPI(title="Alma Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- servir a pasta static (onde fica o alma.png) ---
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------------------
# Logging / Versão
# ---------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("alma")

APP_VERSION = os.getenv("APP_VERSION", "alma-server/rag+mem+links-fallback-1")

# ---------------------------------------------------------------------------------------
# Prompt nuclear da Alma
# ---------------------------------------------------------------------------------------
ALMA_MISSION = """
És a Alma, inteligência da Boa Safra Lda (Boa Safra + Interior Guider).
A tua missão é apoiar a direção (Rui Rocha) e a equipa para que a empresa
sobreviva e prospere, com respostas úteis, objetivas e calmas.

Estilo (estrito)
- Clareza e concisão: vai direto ao ponto. Máximo 1 frase de abertura.
- Empatia sob medida: só comenta o estado emocional quando houver sinais de stress.
- Valores implícitos: mantém o alinhamento sem o declarar.
- Vocabulário disciplinado; evita entusiasmismos.
- Seguimento: termina com 1 próxima ação concreta.
- usar o Tom de Voz do RAG quando presente, adota-o rigorosamente nas respostas.

Proibido
- Small talk, emojis ou tom efusivo.
- Inventar links ou preços.

Funções
1) Estratégia e apoio comercial (produtos, prazos, preços).
2) Método e procedimentos (quando relevante).
3) RAG + Grok; se faltar evidência, diz o que falta e o próximo passo.

Fontes e prioridade
1) Catálogo interno (CSV → SQLite): usar SEMPRE que pedirem preços, orçamentos ou detalhes objetivos de produto.
   - Se o produto não existir no Catálogo → assinala “fora do catálogo” e informa que precisa de confirmação dos serviços.
   - Nunca inventes preços ou dimensões.
2) RAG corporativo (sites, PDFs e marcas): fonte principal para informação da empresa e marcas vendidas.
3) LLM base: livre para raciocínio, estratégia e contexto externo.

Regras de resposta sobre PRODUTOS
- Inclui SEMPRE links clicáveis dos produtos (URL do Catálogo ou, na falta, do RAG; se não houver, escreve literalmente “sem URL”).
- Resume SEMPRE em bullets claros (nome/ref, preço+moeda, dimensões/materiais, nota de disponibilidade/estado no site).
- Sê direto e rápido; evita floreados.

Formato
- 1 bloco curto; bullets só quando ajudam a agir.
"""

# ---------------------------------------------------------------------------------------
# Utilidades de URL e normalização
# ---------------------------------------------------------------------------------------
IG_HOST = os.getenv("IG_HOST", "interiorguider.com").lower()

def _canon_ig_url(u: str) -> str:
    try:
        p = urlparse((u or "").strip())
    except Exception:
        return u or ""
    if not p.netloc:
        return u or ""
    host = p.netloc.lower().replace("www.", "")
    if IG_HOST not in host:
        # só canonizamos IG; externos ficam como estão
        return u or ""
    path = re.sub(r"/(products?|produtos?)\/", "/", p.path, flags=re.I)
    path = re.sub(r"//+", "/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    p = p._replace(scheme="https", netloc=IG_HOST, path=path)
    return urlunparse(p)

# --- normaliza links IG existentes e converte URLs IG “nuas” em markdown (sem look-behind) ---
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", re.I)
_RAW_URL_RE = re.compile(r"https?://[^\s)>\]]+", re.I)

def _already_inside_md(text: str, start_idx: int) -> bool:
    open_br = text.rfind("[", 0, start_idx)
    close_br = text.rfind("]", 0, start_idx)
    paren_open = text.find("(", close_br if close_br != -1 else start_idx)
    return (open_br != -1) and (close_br != -1) and (close_br > open_br) and (paren_open != -1) and (paren_open <= start_idx)

def _fix_product_links_markdown(text: str) -> str:
    if not text:
        return text
    # corrige links markdown existentes (canon IG)
    def _md_repl(m):
        label, url = m.group(1), m.group(2)
        fixed = _canon_ig_url(url)
        return f"[{label}]({fixed})" if fixed else m.group(0)
    text = _MD_LINK_RE.sub(_md_repl, text)
    # converte URLs nuas IG em markdown, se não estiverem já dentro de []
    out = []
    last = 0
    for m in _RAW_URL_RE.finditer(text):
        url = m.group(0)
        host = (urlparse(url).netloc or "").lower()
        if IG_HOST in host and not _already_inside_md(text, m.start()):
            out.append(text[last:m.start()])
            fixed = _canon_ig_url(url)
            out.append(f"[ver produto]({fixed})" if fixed else url)
            last = m.end()
    out.append(text[last:])
    return "".join(out)

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = (s.replace("ã","a").replace("õ","o").replace("á","a").replace("à","a").replace("â","a")
           .replace("é","e").replace("ê","e").replace("í","i").replace("ó","o").replace("ô","o")
           .replace("ú","u").replace("ç","c"))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------------------------------------------------------------------------------------
# RAG (qdrant + openai embeddings) — usa rag_client.py
# ---------------------------------------------------------------------------------------
try:
    from rag_client import (
        ingest_text, ingest_pdf_url, ingest_url,
        crawl_and_ingest, ingest_sitemap,
        search_chunks, build_context_block
    )
    RAG_READY = True
    log.info("[rag] rag_client import OK")
except Exception as e:
    RAG_READY = False
    log.warning(f"[rag] a importar rag_client falhou: {e}")

# -------- Top-K dinâmico --------
RAG_TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K", "12"))
RAG_CONTEXT_TOKEN_BUDGET = int(os.getenv("RAG_CONTEXT_TOKEN_BUDGET", "1600"))
DEFAULT_NAMESPACE = os.getenv("RAG_DEFAULT_NAMESPACE", "").strip() or None

def _clamp_int(v, lo=1, hi=50, default=None):
    try:
        x = int(v)
        return max(lo, min(hi, x))
    except Exception:
        return default if default is not None else lo

# --- heurísticas de termos de produto + preço
_NAME_HINTS = [
    r"\bmesa\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bcadeira[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bbanco[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bcama[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bluminária[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
]
_PRICE_RE = re.compile(r"(?:€|\bEUR?\b)\s*([\d\.\s]{1,12}[,\.]\d{2}|\d{1,6})")

BUDGET_PATTERNS = [
    r"\borçament[oô]\s+(?:para|de)\s+(\d+)",
    r"(\d+)\s*(?:artigos?|itens?|produtos?)",
    r"(\d+)\s*(?:x|unid\.?|unidades?)\s*(?:de|da|do)?\s*([a-z0-9çáéíóúâêôãõ ]{3,})",
]

def _extract_name_terms(text:str) -> List[str]:
    t = " " + (text or "") + " "
    terms = []
    for pat in _NAME_HINTS:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            terms.append(m.group(0).strip())
    pieces = re.split(r"[,;\n]| e | com | para | de ", text or "", flags=re.I)
    for c in pieces:
        c = c.strip()
        if 3 <= len(c) <= 60 and any(w in c.lower() for w in ["mesa","cadeira","banco","cama","luminária","luminaria"]):
            terms.append(c)
    seen=set(); out=[]
    for s in map(_norm, terms):
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out[:8]

def _extract_budget_items(text: str) -> List[Tuple[int, str]]:
    t = " " + (text or "") + " "
    items = []
    for pat in BUDGET_PATTERNS:
        for m in re.finditer(pat, t, flags=re.I):
            if len(m.groups()) == 2:
                try:
                    qty = int(m.group(1))
                except Exception:
                    continue
                name = m.group(2).strip()
                if qty > 0 and len(name) > 2:
                    items.append((qty, name))
            elif len(m.groups()) == 1:
                try:
                    qty = int(m.group(1))
                except Exception:
                    continue
                if qty >= 2:
                    items.append((qty, "artigo genérico"))
    seen = set()
    out = []
    for qty, name in items:
        key = _norm(name)
        if key not in seen:
            seen.add(key)
            out.append((qty, name))
    return out[:5]

def _parse_money_eu(s: str) -> Optional[float]:
    try:
        if s is None:
            return None
        s = str(s).strip().replace("€", "").replace("\u00A0", " ").strip()
        s = re.sub(r"\s+", "", s)
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        else:
            if "," in s:
                s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None

def _price_from_text(txt: str) -> Optional[float]:
    if not txt: return None
    m = _PRICE_RE.search(txt)
    if not m: return None
    return None if m is None else _parse_money_eu(m.group(1))

def _expand_variants(term: str) -> List[str]:
    t = (term or "").strip()
    if not t:
        return []
    t = re.sub(r"^\s*\d+\s*x?\s*", "", t, flags=re.I)
    t = re.sub(r"^\s*\d+\s+", "", t, flags=re.I)
    toks = re.findall(r"[A-Za-z0-9ÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-]+", t)
    base = " ".join(toks[:6]).strip()
    if not base:
        return []
    variants = {base}
    sp_map = {
        "cadeiras": "cadeira",
        "bancos": "banco",
        "mesas": "mesa",
        "camas": "cama",
        "luminárias": "luminária",
        "luminarias": "luminaria",
    }
    words = base.split()
    if words:
        w0 = words[0].lower()
        if w0 in sp_map:
            variants.add(" ".join([sp_map[w0]] + words[1:]))
        inv = {v: k for k, v in sp_map.items()}
        if w0 in inv:
            variants.add(" ".join([inv[w0]] + words[1:]))
    if len(words) >= 2: variants.add(" ".join(words[:2]))
    if len(words) >= 3: variants.add(" ".join(words[:3]))
    if len(words) >= 4: variants.add(" ".join(words[:4]))
    out = []
    seen = set()
    for v in variants:
        vn = _norm(v)
        if vn and vn not in seen:
            seen.add(vn)
            out.append(v)
    return out

# --- Two-stage retrieval helpers (full-text + re-rank) -----------------------
# Nota: se rag_client expuser search_fulltext(...), usamos. Caso contrário,
# fazemos fallback para search_chunks(...) com variantes de query (embeddings).

NUMERAL_MAP = {
    # pt
    "um": "1", "uma": "1", "dois": "2", "duas": "2", "tres": "3", "três": "3",
    # en
    "one": "1", "two": "2", "three": "3",
    # es
    "uno": "1", "dos": "2", "tres": "3",
}

LANG_HINTS = {
    "pt": {"cadeira", "banco", "mesa", "cama", "luminaria", "luminária"},
    "en": {"chair", "stool", "table", "bed", "lamp"},
    "es": {"silla", "taburete", "mesa", "cama", "lampara", "lámpara"},
}

def _detect_lang_hint(qnorm: str) -> str:
    toks = set(qnorm.split())
    for lang, vocab in LANG_HINTS.items():
        if toks & vocab:
            return lang
    # fallback: se contiver acentos pt/es assume pt
    if any(ch in qnorm for ch in "áàâãéêíóôõúç"):
        return "pt"
    return ""

def _slugify_tokens(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _norm(s)).strip("-")

def _tokenize(s: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", _norm(s)) if t]

def _ngram_variants(tokens: List[str], min_n=1, max_n=4) -> List[str]:
    out = []
    n = len(tokens)
    for size in range(min_n, min(max_n, n) + 1):
        for i in range(0, n - size + 1):
            out.append(" ".join(tokens[i:i+size]))
    return out

def _word_number_swaps(tokens: List[str]) -> List[str]:
    out = set()
    for i, t in enumerate(tokens):
        repl = NUMERAL_MAP.get(t, "")
        if repl:
            alt = tokens[:]
            alt[i] = repl
            out.add(" ".join(alt))
        if t.isdigit():
            # trocar 3->three/ tres
            for w, d in NUMERAL_MAP.items():
                if d == t:
                    alt = tokens[:]
                    alt[i] = w
                    out.add(" ".join(alt))
    return list(out)

def _make_query_variants(term: str) -> List[str]:
    """
    Gera variantes robustas: base, hífen<->espaço, numerais, n-gramas encurtados.
    Mantém no máx. ~12 queries para custo controlado.
    """
    base = term.strip()
    if not base:
        return []
    toks = _tokenize(base)
    variants = set()
    # base
    variants.add(" ".join(toks))
    # hífenizações
    variants.add(_slugify_tokens(base).replace("-", " "))
    variants.add(_slugify_tokens(base))
    # numerais ↔ palavras
    variants.update(_word_number_swaps(toks))
    # n-gramas (encurtar cauda)
    ngs = _ngram_variants(toks, min_n=2, max_n=min(4, len(toks)))
    variants.update(ngs[-6:])  # últimas combinações mais específicas
    # dedup e limites
    out = []
    seen = set()
    for v in variants:
        vn = _norm(v)
        if vn and vn not in seen:
            seen.add(vn)
            out.append(v)
        if len(out) >= 12:
            break
    return out

def _overlap(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    A, B = set(a_tokens), set(b_tokens)
    inter = len(A & B)
    denom = max(1, min(len(A), len(B)))
    return inter / denom

def _scorer(query: str, payload: dict, url: str, lang_hint: str) -> float:
    qtok = _tokenize(query)
    title = (payload.get("title") or "").strip()
    slug  = (payload.get("slug") or "") or (urlparse(url).path if url else "")
    search_text = (payload.get("search_text") or "")
    # tokens
    ttok = _tokenize(title)
    slugtok = _tokenize(slug.replace("/", " "))
    utok = _tokenize(url or "")
    sttok = _tokenize(search_text)
    # overlaps
    s_slug  = _overlap(qtok, slugtok)
    s_title = _overlap(qtok, ttok)
    s_url   = _overlap(qtok, utok)
    s_stext = _overlap(qtok, sttok)
    score = (2.0*s_slug) + (1.25*s_title) + (0.5*s_url) + (0.75*s_stext)
    # domínio IG
    host = (urlparse(url).netloc or "").lower()
    if IG_HOST in host:
        score += 0.40
    # pista de idioma (ligeiro)
    text_all = " ".join([title, slug, search_text, url or ""])
    if lang_hint:
        if lang_hint == "pt" and re.search(r"[ãõáàâéêíóôõúç]|/pt/|/pt-", text_all, flags=re.I):
            score += 0.10
        elif lang_hint == "en" and re.search(r"/en/|/en-", text_all, flags=re.I):
            score += 0.10
        elif lang_hint == "es" and re.search(r"/es/|/es-", text_all, flags=re.I):
            score += 0.10
    return score

def _stage_a_candidates(term: str, namespace: Optional[str], limit_per_q: int = 50, hard_cap: int = 150):
    """
    Tenta usar full-text (se existir em rag_client: search_fulltext).
    Caso contrário, fallback para search_chunks com queries variantes.
    Retorna lista de dicts com {payload, metadata(url,title,...)}.
    """
    variants = _make_query_variants(term)
    seen_ids = set()
    cands = []

    # tenta full-text nativo, se existir no rag_client
    search_fulltext = None
    try:
        from rag_client import search_fulltext as _sf  # opcional
        search_fulltext = _sf
    except Exception:
        search_fulltext = None

    for v in variants[:6]:  # não exagerar
        try:
            if search_fulltext:
                hits = search_fulltext(query=v, namespace=namespace or DEFAULT_NAMESPACE, limit=limit_per_q) or []
            else:
                hits = search_chunks(query=v, namespace=namespace or DEFAULT_NAMESPACE, top_k=min( max(10, limit_per_q//2), 50)) or []
        except Exception:
            hits = []

        for h in hits:
            # Qdrant: pode vir {"id","payload","score"} ou formato do teu search_chunks
            pid = str(h.get("id") or id(h))
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            meta = h.get("metadata") or h.get("payload") or {}
            # normalizar “payload/metadata”
            payload = dict(meta)
            # espelhar title/url no payload se vierem noutro nível
            if not payload.get("title"): payload["title"] = h.get("title") or ""
            if not payload.get("url"):   payload["url"]   = h.get("url") or meta.get("url") or ""
            # tentar derivar slug
            if not payload.get("slug") and payload.get("url"):
                payload["slug"] = (urlparse(payload["url"]).path or "").strip("/")
            cands.append({"payload": payload, "score": h.get("score", 0.0)})
            if len(cands) >= hard_cap:
                break
        if len(cands) >= hard_cap:
            break
    return cands

def _rerank_and_choose(term: str, candidates: List[dict], lang_hint: str, prefer_ig: bool = True, threshold: float = 0.55):
    """
    Reordena candidatos por _scorer. Devolve (best_url, ranked[:3]) para debug/fallback.
    """
    ranked = []
    for c in candidates:
        p = c.get("payload", {}) or {}
        url = _canon_ig_url(p.get("url") or "")
        if not url:
            continue
        s = _scorer(term, p, url, lang_hint=lang_hint)
        ranked.append((s, url, p))
    ranked.sort(key=lambda x: x[0], reverse=True)

    # aplica threshold e preferência por IG
    top = [r for r in ranked if r[0] >= threshold] or ranked[:1]
    if not top:
        return "", []
    if prefer_ig:
        ig = [r for r in top if IG_HOST in (urlparse(r[1]).netloc or "").lower()]
        if ig:
            return ig[0][1], top[:3]
    return top[0][1], top[:3]


def rag_mini_search_urls(terms: List[str], namespace: Optional[str], top_k: int) -> Dict[str, str]:
    """
    Two-stage: Stage A (full-text se existir / fallback embeddings com variantes) -> Stage B (re-rank textual).
    Retorna o melhor URL por termo, com preferência por interiorguider.com.
    """
    if not (RAG_READY and terms):
        return {}
    url_by_term: Dict[str, str] = {}
    for term in terms:
        tnorm = _norm(term)
        if not tnorm:
            continue
        lang_hint = _detect_lang_hint(tnorm)
        # Stage A: recolha de candidatos
        candidates = _stage_a_candidates(term, namespace=namespace, limit_per_q=50, hard_cap=150)
        if not candidates:
            # fallback final: uma pesquisa “crua” única
            try:
                hits = search_chunks(query=term, namespace=namespace or DEFAULT_NAMESPACE, top_k=max(10, min(40, top_k or 10))) or []
            except Exception:
                hits = []
            candidates = [{"payload": (h.get("metadata") or {}), "score": h.get("score", 0.0)} for h in hits]

        # Stage B: re-rank e escolha
        best_url, _ = _rerank_and_choose(term, candidates, lang_hint=lang_hint, prefer_ig=True, threshold=0.55)

        if best_url:
            url_by_term[tnorm] = best_url
    return url_by_term
    


# ---------------------------------------------------------------------------------------
# Memória Local + (opcional) Mem0
# ---------------------------------------------------------------------------------------
MEM0_ENABLE = os.getenv("MEM0_ENABLE", "false").lower() in ("1", "true", "yes")
MEM0_API_KEY = (os.getenv("MEM0_API_KEY") or "").strip()

MemoryClient = None
mem0_client = None

if MEM0_ENABLE:
    if not MEM0_API_KEY:
        log.warning("[mem0] MEM0_ENABLE=true mas falta MEM0_API_KEY")
    else:
        try:
            try:
                import mem0ai as _mem0_pkg
                from mem0ai import MemoryClient as _MC
            except Exception:
                import mem0 as _mem0_pkg
                from mem0 import MemoryClient as _MC
            MemoryClient = _MC
            log.info(f"[mem0] import OK ({getattr(_mem0_pkg,'__name__','mem0')}) file={getattr(_mem0_pkg,'__file__','?')}")
        except Exception as e:
            log.error(f"[mem0] import FAILED: {e}")
            MemoryClient = None
        if MemoryClient is not None:
            try:
                mem0_client = MemoryClient(api_key=MEM0_API_KEY)
                log.info("[mem0] MemoryClient inicializado.")
            except Exception as e:
                log.error(f"[mem0] não inicializou: {e}")
                mem0_client = None

LOCAL_FACTS: Dict[str, Dict[str, str]] = {}
LOCAL_HISTORY: Dict[str, List[Tuple[str, str]]] = {}

FACT_PREFIX = "FACT|"

def local_set_fact(user_id: str, key: str, value: str):
    LOCAL_FACTS.setdefault(user_id, {})
    LOCAL_FACTS[user_id][key] = value.strip()

def local_get_facts(user_id: str) -> Dict[str, str]:
    return dict(LOCAL_FACTS.get(user_id, {}))

def local_append_dialog(user_id: str, question: str, answer: str, cap: int = 50):
    LOCAL_HISTORY.setdefault(user_id, [])
    LOCAL_HISTORY[user_id].append((question, answer))
    if len(LOCAL_HISTORY[user_id]) > cap:
        LOCAL_HISTORY[user_id] = LOCAL_HISTORY[user_id][-cap:]

def local_search_snippets(user_id: str, limit: int = 5) -> List[str]:
    items = LOCAL_HISTORY.get(user_id, [])
    out = []
    for q, a in reversed(items[-limit*2:]):
        if len(out) >= limit: break
        out.append(f"User: {q}")
        if len(out) >= limit: break
        out.append(f"Alma: {a}")
    return out[:limit]

def _mem0_create(content: str, user_id: str, metadata: Optional[dict] = None):
    if not (MEM0_ENABLE and mem0_client):
        return
    try:
        if hasattr(mem0_client, "memories"):
            mem0_client.memories.create(content=content, user_id=user_id, metadata=metadata or {})
        elif hasattr(mem0_client, "create"):
            mem0_client.create(content=content, user_id=user_id, metadata=metadata or {})
    except Exception as e:
        log.warning(f"[mem0] create falhou: {e}")

def _mem0_search(query: str, user_id: str, limit: int = 5) -> List[str]:
    if not (MEM0_ENABLE and mem0_client):
        return []
    try:
        if hasattr(mem0_client, "memories"):
            results = mem0_client.memories.search(query=query or "contexto", user_id=user_id, limit=limit)
        else:
            results = mem0_client.search(query=query or "contexto", user_id=user_id, limit=limit)
        snippets = []
        for item in results or []:
            val = (item.get("text") or item.get("memory") or item.get("content") or "").strip()
            if val:
                snippets.append(val)
        return snippets
    except Exception as e:
        log.warning(f"[mem0] search falhou: {e}")
        return []

def mem0_set_fact(user_id: str, key: str, value: str):
    local_set_fact(user_id, key, value)
    if not (MEM0_ENABLE and mem0_client): return
    try:
        _mem0_create(
            content=f"{FACT_PREFIX}{key}={value}",
            user_id=user_id,
            metadata={"source": "alma-server", "type": "fact", "key": key}
        )
    except Exception as e:
        log.warning(f"[mem0] fact create falhou: {e}")

def mem0_get_facts(user_id: str, limit: int = 50) -> Dict[str, str]:
    facts = local_get_facts(user_id)
    if not (MEM0_ENABLE and mem0_client):
        return facts
    try:
        if hasattr(mem0_client, "memories"):
            results = mem0_client.memories.search(query=FACT_PREFIX, user_id=user_id, limit=limit)
        else:
            results = mem0_client.search(query=FACT_PREFIX, user_id=user_id, limit=limit)
        for item in results or []:
            content = (item.get("text") or item.get("memory") or item.get("content") or "").strip()
            if content.startswith(FACT_PREFIX) and "=" in content[len(FACT_PREFIX):]:
                body = content[len(FACT_PREFIX):]
                k, v = body.split("=", 1)
                if k and v:
                    facts[k.strip()] = v.strip()
        return facts
    except Exception as e:
        log.warning(f"[mem0] get_facts falhou: {e}")
        return facts

# ---------------------------------------------------------------------------------------
# Extração de factos simples do texto
# ---------------------------------------------------------------------------------------
NAME_PATTERNS = [
    r"\bchamo-?me\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}(?:\s+[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}){0,3})\b",
    r"\bo\s+meu\s+nome\s+é\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}(?:\s+[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}){0,3})\b",
    r"\bsou\s+(?:o|a)\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}(?:\s+[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}){0,3})\b",
]
CITY_PATTERNS = [
    r"\bmoro\s+(?:em|no|na)\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\w\s\-\.'ÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç]{2,60})",
    r"\bestou\s+(?:em|no|na)\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\w\s\-\.'ÁÂÃÀÉÊÍÓÔÕÚÇ]{2,60})",
    r"\bsou\s+de\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\w\s\-\.'ÁÂÃÀÉÊÍÓÔÕÚÇ]{2,60})",
]
PREF_PATTERNS = [
    r"\bgosto\s+(?:de|do|da|dos|das)\s+([^\.]{3,80})",
    r"\bprefiro\s+([^\.]{3,80})",
    r"\badoro\s+([^\.]{3,80})",
    r"\bquero\s+([^\.]{3,80})",
]
ROOM_KEYWORDS = ["sala", "cozinha", "quarto", "wc", "casa de banho", "varanda", "escritório", "hall", "entrada", "lavandaria"]

def extract_contextual_facts_pt(text: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    t = " " + (text or "") + " "
    for pat in NAME_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if len(name.split()) == 1 and name.lower() in {"melhor", "pior", "arquiteto", "cliente"}:
                pass
            else:
                facts["name"] = name
                break
    for pat in CITY_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            city = m.group(1).strip(" .,'\"").title()
            if 2 <= len(city) <= 60:
                facts["location"] = city
                break
    for pat in PREF_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            pref = m.group(1).strip(" .,'\"")
            if 3 <= len(pref) <= 80:
                facts.setdefault("preferences", pref)
                break
    for kw in ROOM_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", t, flags=re.IGNORECASE):
            facts["room"] = kw
    return facts

def facts_to_context_block(facts: Dict[str, str]) -> str:
    if not facts: return ""
    lines = []
    if "name" in facts: lines.append(f"- Nome: {facts['name']}")
    if "location" in facts: lines.append(f"- Localização: {facts['location']}")
    if "room" in facts: lines.append(f"- Divisão/Projeto: {facts['room']}")
    if "preferences" in facts: lines.append(f"- Preferências: {facts['preferences']}")
    for k, v in facts.items():
        if k not in {"name", "location", "room", "preferences"}:
            lines.append(f"- {k}: {v}")
    return "Perfil do utilizador (memória contextual):\n" + "\n".join(lines)

def facts_block_for_user(user_id: str) -> str:
    facts = mem0_get_facts(user_id)
    return facts_to_context_block(facts)

# ---------- Entity/link resolver (seguro) ----------
def _canon_path(u: str) -> str:
    try:
        p = urlparse(_canon_ig_url(u) or u)
        return (p.path or "").rstrip("/")
    except Exception:
        return ""

def _tokens(s: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", _norm(s)) if t and len(t) > 1]

def _compat_score(name: str, title: str, slug: str) -> float:
    # mede compatibilidade entidade ↔ página (Jaccard sobre tokens)
    nt = set(_tokens(name))
    tt = set(_tokens(title))
    st = set(_tokens(slug.replace("/", " ")))
    if not nt:
        return 0.0
    j_title = len(nt & tt) / max(1, min(len(nt), len(tt))) if tt else 0.0
    j_slug  = len(nt & st) / max(1, min(len(nt), len(st))) if st else 0.0
    # reforços semânticos por palavras-chave (evita banco↔mesa, etc.)
    kind_hit = 0.0
    KINDS = [
        ("mesa","table"), ("cadeira","chair"), ("banco","stool"),
        ("cama","bed"), ("luminaria","lamp"), ("luminária","lamp")
    ]
    name_low = _norm(name)
    text_all = _norm(title + " " + slug)
    for pt,en in KINDS:
        if re.search(rf"\b{pt}\b", name_low) and re.search(rf"\b({pt}|{en})\b", text_all):
            kind_hit = 0.2; break
    return (0.6*j_slug) + (0.4*j_title) + kind_hit

class LinkResolver:
    def __init__(self, namespace: Optional[str], prefer_host: str = IG_HOST, min_conf: float = 0.55):
        self.ns = namespace or DEFAULT_NAMESPACE
        self.prefer_host = (prefer_host or "").lower()
        self.min_conf = min_conf

    def best_url_for(self, term: str) -> Tuple[str,float,dict]:
        """Procura candidatos (fulltext/embeddings) e escolhe o melhor link.
        Retorna (url, conf, payload)."""
        cands = _stage_a_candidates(term, namespace=self.ns, limit_per_q=50, hard_cap=120) or []
        best = ("", 0.0, {})
        lang_hint = _detect_lang_hint(_norm(term))
        for c in cands:
            p = c.get("payload") or {}
            raw_url = p.get("url") or ""
            url = _canon_ig_url(raw_url) or raw_url
            if not url:
                continue
            title = (p.get("title") or "")
            slug  = (p.get("slug")  or urlparse(url).path or "")
            base  = _scorer(term, p, url, lang_hint)          # “parecido com a query”
            compat= _compat_score(term, title, slug)          # “é MESMO este produto?”
            host  = (urlparse(url).netloc or "").lower()
            pref  = 0.25 if (self.prefer_host and self.prefer_host in host) else 0.0
            score = 0.5*base + 0.5*compat + pref
            if score > best[1]:
                best = (url, score, p)
        return best

    def resolve_map(self, terms: List[str]) -> Dict[str, Tuple[str,float]]:
        out = {}
        for t in terms:
            tnorm = _norm(t)
            if not tnorm: 
                continue
            url, conf, _ = self.best_url_for(t)
            if url and conf >= self.min_conf:
                out[tnorm] = (url, conf)
        return out

    def smart_fix_text(self, text: str, url_by_term_conf: Dict[str, Tuple[str,float]]) -> str:
        """Substitui apenas quando a confiança é suficiente e o destino é diferente/melhor."""
        if not text or not url_by_term_conf:
            return text
        parts = text.splitlines()
        for i, line in enumerate(parts):
            ln = _norm(line)
            for tnorm, (good_url, conf) in url_by_term_conf.items():
                if tnorm not in ln:
                    continue
                good_path = _canon_path(good_url)
                # 1) corrige markdown com URL IG errado
                def _md_fix(m):
                    label, url = m.group(1), m.group(2)
                    host = (urlparse(url).netloc or "").lower()
                    if IG_HOST in host and _canon_path(url) != good_path and conf >= self.min_conf:
                        return f"[{label}]({good_url})"
                    return m.group(0)
                l2 = _MD_LINK_RE.sub(_md_fix, line)

                # 2) corrige URLs “nus” IG errados
                def _raw_fix(m):
                    url = m.group(0)
                    host = (urlparse(url).netloc or "").lower()
                    if IG_HOST in host and _canon_path(url) != good_path and conf >= self.min_conf:
                        return good_url
                    return url
                l2 = _RAW_URL_RE.sub(_raw_fix, l2)
                line = l2
            parts[i] = line
        return "\n".join(parts)
# ---------------------------------------------------------------------------------------
# Injetor de links a partir do RAG (pós-processamento do texto) + Fallback
# ---------------------------------------------------------------------------------------

def _inject_links_from_rag(text: str, user_query: str, namespace: Optional[str], decided_top_k: int) -> str:
    if not (RAG_READY and text):
        return text

    # 0) normalização de links já formatados (mantém)
    out = _fix_product_links_markdown(text or "")

    # 1) extrai termos das duas pontas (query + resposta)
    terms: List[str] = []
    terms.extend(_extract_name_terms(user_query or ""))
    terms.extend(_extract_name_terms(out or ""))

    # 2) resolve entidades com LinkResolver
    resolver = LinkResolver(namespace=namespace, min_conf=0.58)
    url_by_term_conf = resolver.resolve_map(terms)

    # 3) corrige “[label](sem URL)” e linhas “sem URL” usando o mapa resolvido
    if url_by_term_conf:
        def _fix_sem_url(m):
            label = m.group(1)
            tnorm = _norm(label)
            if tnorm in url_by_term_conf:
                return f"[{label}]({url_by_term_conf[tnorm][0]})"
            # fallback: se só existir um link com boa confiança, usa-o
            if len(url_by_term_conf) == 1:
                only_url = list(url_by_term_conf.values())[0][0]
                return f"[{label}]({only_url})"
            return m.group(0)
        out = re.sub(r"\[([^\]]+)\]\(\s*sem\s+url\s*\)", _fix_sem_url, out, flags=re.I)

        def _replace_line_sem_url(line: str) -> str:
            if re.search(r"\bsem\s+url\b", line, flags=re.I):
                ln = _norm(line)
                # escolhe o termo mais “compatível” com a linha
                best = None
                for tnorm, (url, conf) in url_by_term_conf.items():
                    if tnorm in ln and conf >= resolver.min_conf:
                        best = (tnorm, url, conf); break
                if best:
                    tnorm, url, _ = best
                    if _MD_LINK_RE.search(line):
                        return re.sub(r"\bsem\s+url\b", f"{url}", line, flags=re.I)
                    # liga o primeiro nome de produto detetado
                    m2 = re.search(r"(mesa|cadeira|banco|cama|luminária|luminaria)[^\-:]*", line, flags=re.I)
                    if m2:
                        nome = m2.group(0).strip()
                        linked = line.replace(nome, f"[{nome}]({url})", 1)
                        return re.sub(r"\bsem\s+url\b", "", linked, flags=re.I)
                    return re.sub(r"\bsem\s+url\b", f"{url}", line, flags=re.I)
            return line
        out = "\n".join(_replace_line_sem_url(l) for l in out.splitlines())

    # 4) corrige URLs IG errados (apenas se confiança >= min_conf)
    out = resolver.smart_fix_text(out, url_by_term_conf)

    # 5) se ainda não houver nenhum link IG, tenta injetar 1 com maior confiança
    if IG_HOST not in out.lower() and url_by_term_conf:
        best_t = max(url_by_term_conf.items(), key=lambda kv: kv[1][1])  # maior confiança
        best_url = best_t[1][0]
        lines = out.splitlines()
        for i, l in enumerate(lines):
            if not _MD_LINK_RE.search(l):
                lines[i] = l + f" — [ver produto]({best_url})"
                out = "\n".join(lines); break

    # 6) reforço para pedidos de orçamento: linka nomes dos itens
    if "orçament" in (user_query or "").lower() and url_by_term_conf:
        budget_items = _extract_budget_items(user_query)
        lines = out.splitlines()
        for i, line in enumerate(lines):
            for qty, item_name in budget_items:
                tnorm = _norm(item_name)
                if tnorm in url_by_term_conf and not _MD_LINK_RE.search(line):
                    url = url_by_term_conf[tnorm][0]
                    m = re.search(rf"(\b{re.escape(item_name)}\b)", line, re.I)
                    if m:
                        nome_part = m.group(1)
                        lines[i] = line.replace(nome_part, f"[{nome_part}]({url})", 1)
        out = "\n".join(lines)

    return out

def _postprocess_answer(answer: str, user_query: str, namespace: Optional[str], decided_top_k: int) -> str:
    step1 = _fix_product_links_markdown(answer or "")
    step2 = _inject_links_from_rag(step1, user_query, namespace, decided_top_k)
    return step2

def _links_from_matches(matches: List[dict], max_links: int = 8) -> List[Tuple[str, str]]:
    """Extrai (title, url) únicos dos matches, priorizando IG, depois externos."""
    if not matches:
        return []
    pairs: List[Tuple[str, str]] = []
    seen = set()
    # 1) IG primeiro
    for pref_ig in (True, False):
        for h in matches:
            meta = h.get("metadata", {}) or {}
            title = (meta.get("title") or h.get("title") or "").strip()
            url = (meta.get("url") or h.get("url") or "").strip()
            if not url:
                continue
            cu = _canon_ig_url(url) or url
            host = (urlparse(cu).netloc or "").lower()
            is_ig = IG_HOST in host
            if pref_ig and not is_ig:
                continue
            if (not pref_ig) and is_ig:
                continue
            key = (title.lower(), cu.lower())
            if key in seen:
                continue
            seen.add(key)
            pairs.append((title or "-", cu))
            if len(pairs) >= max_links:
                return pairs
    return pairs[:max_links]

def _force_links_block(answer: str, matches: List[dict], min_links: int = 3) -> str:
    """Se a resposta não contiver links (ou não tiver IG), anexa bloco de links do RAG."""
    text = answer or ""
    has_md_link = bool(_MD_LINK_RE.search(text))
    has_ig = IG_HOST in text.lower()
    if has_md_link and has_ig:
        return text  # já tem
    # obter links a partir dos matches
    links = _links_from_matches(matches, max_links=max(6, min_links))
    if not links:
        return text
    # montar bloco
    lines = ["", "Links úteis (do RAG):"]
    for title, url in links:
        safe_title = title if title else "—"
        lines.append(f"- [{safe_title}]({url})")
    return (text.rstrip() + "\n" + "\n".join(lines)).strip()

# ---------------------------------------------------------------------------------------
# Config Grok (x.ai)
# ---------------------------------------------------------------------------------------
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = os.getenv("XAI_MODEL", "grok-4-0709")

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
_session.mount("https://", _adapter); _session.mount("http://", _adapter)

def grok_chat(messages, timeout=120):
    if not XAI_API_KEY:
        raise RuntimeError("Falta XAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }
    payload = {"model": MODEL, "messages": messages}
    r = _session.post(XAI_API_URL, headers=headers, json=payload, timeout=timeout)
    log.info(f"[x.ai] status={r.status_code} body={r.text[:300]}")
    r.raise_for_status()
    return r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or ""

# ---------------------------------------------------------------------------------------
# Blocos de contexto e construção das mensagens
# ---------------------------------------------------------------------------------------
def _decide_top_k(user_query: str, req_top_k: Optional[int]) -> int:
    if req_top_k:
        return _clamp_int(req_top_k, lo=3, hi=40, default=RAG_TOP_K_DEFAULT)
    n_terms = len(_extract_name_terms(user_query or ""))
    budget_items = _extract_budget_items(user_query)
    if "orçament" in user_query.lower() or len(budget_items) >= 2:
        base = 20 + (len(budget_items) * 4)
    elif n_terms >= 3:
        base = 16
    elif n_terms == 2:
        base = 12
    else:
        base = 10
    return min(max(base, RAG_TOP_K_DEFAULT), 40)

def build_rag_products_block(question: str) -> str:
    if not RAG_READY:
        return ""
    budget_items = _extract_budget_items(question)
    lines = []
    if budget_items:
        for qty, item_name in budget_items[:6]:
            query_item = f"{qty} {item_name}"
            try:
                hits = search_chunks(query=query_item, namespace=DEFAULT_NAMESPACE, top_k=5) or []
            except Exception:
                hits = []
            seen = set()
            for h in hits:
                meta = h.get("metadata", {}) or {}
                title = (meta.get("title") or item_name).strip()
                url = _canon_ig_url(meta.get("url") or "")
                key = (title, url)
                if key in seen: continue
                seen.add(key)
                lines.append(f"- QTY={qty}; NOME={title}; URL={url or 'sem URL'}")
            if not hits:
                links = rag_mini_search_urls([item_name], DEFAULT_NAMESPACE, top_k=5)
                url = links.get(_norm(item_name), "sem URL")
                lines.append(f"- QTY={qty}; NOME={item_name}; URL={url}")
    else:
        try:
            hits = search_chunks(query=question, namespace=DEFAULT_NAMESPACE, top_k=RAG_TOP_K_DEFAULT) or []
        except Exception:
            hits = []
        seen = set()
        for h in hits[:8]:
            meta = h.get("metadata", {}) or {}
            title = (meta.get("title") or "").strip()
            url = _canon_ig_url(meta.get("url") or "")
            key = (title, url)
            if key in seen: continue
            seen.add(key)
            lines.append(f"- NOME={title or '-'}; URL={url or 'sem URL'}")
    return "Produtos para orçamento (do RAG; usa estes dados exatos para links):\n" + "\n".join(lines) if lines else ""

def build_messages(user_id: str, question: str, namespace: Optional[str]):
    new_facts = extract_contextual_facts_pt(question)
    for k, v in new_facts.items():
        mem0_set_fact(user_id, k, v)

    short_snippets = _mem0_search(question, user_id=user_id, limit=5) or local_search_snippets(user_id, limit=5)
    memory_block = "Memórias recentes do utilizador (curto prazo):\n" + "\n".join(f"- {s}" for s in short_snippets[:3]) if short_snippets else ""

    rag_block = ""
    rag_used = False
    rag_hits: List[dict] = []
    if RAG_READY:
        try:
            rag_hits = search_chunks(query=question, namespace=namespace or DEFAULT_NAMESPACE, top_k=RAG_TOP_K_DEFAULT) or []
            rag_block = build_context_block(rag_hits, token_budget=RAG_CONTEXT_TOKEN_BUDGET) if rag_hits else ""
            rag_used = bool(rag_block)
        except Exception as e:
            log.warning(f"[rag] search falhou: {e}")
            rag_block = ""
            rag_used = False
            rag_hits = []

    # bloco de links candidatos (para o LLM usar tal como estão)
    links_pairs = _links_from_matches(rag_hits, max_links=8)
    links_block = ""
    if links_pairs:
        lines = ["Links candidatos (do RAG; usa estes URLs tal como estão):"]
        for title, url in links_pairs:
            lines.append(f"- {title or '-'} — {url}")
        links_block = "\n".join(lines)

    products_block = build_rag_products_block(question)

    messages = [{"role": "system", "content": ALMA_MISSION}]
    fb = facts_block_for_user(user_id)
    if fb: messages.append({"role": "system", "content": fb})
    if rag_block:
        messages.append({"role": "system", "content": f"Conhecimento corporativo (RAG):\n{rag_block}"})
    if links_block:
        messages.append({"role": "system", "content": links_block})
    if products_block:
        messages.append({"role": "system", "content": products_block})
    if memory_block:
        messages.append({"role": "system", "content": memory_block})
    messages.append({"role": "user", "content": question})
    return messages, new_facts, rag_used, rag_hits

# ---------------------------------------------------------------------------------------
# ROTAS BÁSICAS + páginas
# ---------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return HTMLResponse("<h1>index.html não encontrado</h1>", status_code=404)

@app.get("/alma-chat", response_class=HTMLResponse)
@app.get("/alma-chat/", response_class=HTMLResponse)
def alma_chat():
    html_path = os.path.join(os.getcwd(), "alma-chat.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<h1>alma-chat.html não encontrado</h1>", status_code=404)

@app.get("/console", response_class=HTMLResponse)
def serve_console():
    html_path = os.path.join(os.getcwd(), "console.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<h1>console.html não encontrado</h1>", status_code=404)

@app.get("/status")
def status_json():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "message": "Alma server ativo. Use POST /ask (Grok+Memória+RAG).",
        "mem0": {"enabled": MEM0_ENABLE, "client_ready": bool(mem0_client)},
        "rag": {"available": RAG_READY, "top_k_default": RAG_TOP_K_DEFAULT, "namespace": DEFAULT_NAMESPACE},
        "endpoints": {
            "health": "/health",
            "ask": "POST /ask {question, user_id?, namespace?, top_k?}",
            "ask_get": "/ask_get?q=...&user_id=...&namespace=...&top_k=...",
            "ping_grok": "/ping_grok",
            "rag_search_get": "/rag/search?q=...&namespace=...",
            "rag_search_post": "POST /rag/search",
            "rag_ingest_sitemap": "POST /rag/ingest-sitemap",
            "rag_crawl": "POST /rag/crawl",
            "rag_ingest_url": "POST /rag/ingest-url",
            "rag_ingest_text": "POST /rag/ingest-text",
            "rag_ingest_pdf_url": "POST /rag/ingest-pdf-url",
            "rag_extract_urls": "POST /rag/extract-urls",
            "budget_csv": "POST /budget/csv",
            "console": "/console",
        },
        # ⬇️ adicionamos o estado do catálogo (SQLite)
        "catalog": _status_catalog_sqlite(),
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mem0_enabled": MEM0_ENABLE,
        "mem0_client_ready": bool(mem0_client),
        "model": MODEL,
        "rag_available": RAG_READY,
        "rag_default_namespace": DEFAULT_NAMESPACE,
        "rag_top_k_default": RAG_TOP_K_DEFAULT
    }

@app.post("/echo")
async def echo(request: Request):
    data = await request.json()
    return {"echo": data}

@app.get("/ping_grok")
def ping_grok():
    try:
        msg = [{"role": "user", "content": "Diz apenas: pong"}]
        content = grok_chat(msg, timeout=20)
        return {"ok": True, "reply": content}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------------------------------------------------------------------------------------
# Memória contextual (FACTs) e Mem0 debug
# ---------------------------------------------------------------------------------------
@app.get("/mem/facts")
def mem_facts(user_id: str = "anon"):
    facts = mem0_get_facts(user_id=user_id, limit=50)
    return {"user_id": user_id, "facts": facts}

@app.get("/mem/search")
def mem_search_route(q: str = "", user_id: str = "anon"):
    if not q:
        return {"user_id": user_id, "found": 0, "snippets": []}
    snippets = _mem0_search(q, user_id=user_id, limit=10) or local_search_snippets(user_id, limit=10)
    return {"user_id": user_id, "found": len(snippets), "snippets": snippets}

# ---------------------------------------------------------------------------------------
# RAG: GET /rag/search (debug)
# ---------------------------------------------------------------------------------------
@app.get("/rag/search")
def rag_search_get(q: str, namespace: str = None, top_k: int = None):
    if not RAG_READY:
        return {"ok": False, "error": "rag_client indisponível no servidor"}
    try:
        res = search_chunks(query=q, namespace=namespace or DEFAULT_NAMESPACE, top_k=top_k or RAG_TOP_K_DEFAULT)
        return {"ok": True, "query": q, "matches": res}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------------------------------------------------------------------------------------
# RAG Endpoints (crawl, sitemap, url, text, pdf, search POST) — compat com consola
# ---------------------------------------------------------------------------------------
@app.post("/rag/crawl")
async def rag_crawl(request: Request):
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}
    try:
        data = await request.json()
        seed_url   = (data.get("seed_url") or "").strip()
        namespace  = (data.get("namespace") or "default").strip()
        max_pages  = int(data.get("max_pages")  or os.getenv("CRAWL_MAX_PAGES", "40"))
        max_depth  = int(data.get("max_depth")  or os.getenv("CRAWL_MAX_DEPTH", "2"))
        deadline_s = int(data.get("deadline_s") or os.getenv("RAG_DEADLINE_S", "55"))
        if not seed_url:
            return {"ok": False, "error": "Falta seed_url"}
        # assinatura do rag_client antiga (seed_url, não root_url)
        res = crawl_and_ingest(
            seed_url=seed_url, namespace=namespace,
            max_pages=max_pages, max_depth=max_depth, deadline_s=deadline_s
        )
        if "ok" not in res: res["ok"] = True
        res.setdefault("summary", f"visited={res.get('visited')} ok_chunks={res.get('ok_chunks')} fail={res.get('fail')} namespace={namespace}")
        return res
    except Exception as e:
        import traceback
        log.exception("crawl_failed")
        return {"ok": False, "error": "crawl_failed", "detail": str(e), "trace": traceback.format_exc(limit=3)}

@app.post("/rag/ingest-sitemap")
async def rag_ingest_sitemap_route(request: Request):
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}
    try:
        data = await request.json()
        sitemap_url = (data.get("sitemap_url") or data.get("site_url") or "").strip()
        namespace   = (data.get("namespace") or "default").strip()
        max_pages   = int(data.get("max_pages") or os.getenv("CRAWL_MAX_PAGES", "40"))
        deadline_s  = int(data.get("deadline_s") or os.getenv("RAG_DEADLINE_S", "55"))
        if not sitemap_url:
            return {"ok": False, "error": "Falta sitemap_url/site_url"}
        return ingest_sitemap(sitemap_url, namespace=namespace, max_pages=max_pages, deadline_s=deadline_s)
    except Exception as e:
        return {"ok": False, "error": "sitemap_failed", "detail": str(e)}

@app.post("/rag/ingest-url")
async def rag_ingest_url_route(request: Request):
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}
    try:
        data = await request.json()
        page_url  = (data.get("page_url") or "").strip()
        namespace = (data.get("namespace") or "default").strip()
        deadline_s = int(data.get("deadline_s") or os.getenv("RAG_DEADLINE_S", "55"))
        if not page_url:
            return {"ok": False, "error": "Falta page_url"}
        return ingest_url(page_url, namespace=namespace, deadline_s=deadline_s)
    except Exception as e:
        return {"ok": False, "error": "ingest_url_failed", "detail": str(e)}

@app.post("/rag/ingest-text")
async def rag_ingest_text_route(request: Request):
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}
    try:
        data = await request.json()
        title     = (data.get("title") or "").strip()
        text      = (data.get("text") or "").strip()
        namespace = (data.get("namespace") or "default").strip()
        if not title or not text:
            return {"ok": False, "error": "Falta title ou text"}
        return ingest_text(title=title, text=text, namespace=namespace)
    except Exception as e:
        return {"ok": False, "error": "ingest_text_failed", "detail": str(e)}

@app.post("/rag/ingest-pdf-url")
async def rag_ingest_pdf_url_route(request: Request):
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}
    try:
        data = await request.json()
        pdf_url  = (data.get("pdf_url") or "").strip()
        title    = (data.get("title") or None)
        namespace = (data.get("namespace") or "default").strip()
        if not pdf_url:
            return {"ok": False, "error": "Falta pdf_url"}
        return ingest_pdf_url(pdf_url=pdf_url, title=title, namespace=namespace)
    except Exception as e:
        return {"ok": False, "error": "ingest_pdf_failed", "detail": str(e)}

@app.post("/rag/search")
async def rag_search_post(request: Request):
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}
    try:
        data = await request.json()
        query     = (data.get("query") or "").strip()
        namespace = (data.get("namespace") or None)
        top_k     = int(data.get("top_k") or os.getenv("RAG_TOP_K", str(RAG_TOP_K_DEFAULT)))
        matches = search_chunks(query=query, namespace=namespace or DEFAULT_NAMESPACE, top_k=top_k)
        ctx = build_context_block(matches, token_budget=RAG_CONTEXT_TOKEN_BUDGET)
        return {"ok": True, "matches": matches, "context_block": ctx}
    except Exception as e:
        return {"ok": False, "error": "search_failed", "detail": str(e)}

# --- Proxy: extrair URLs (sitemap.xml, HTML ou texto colado) — usado pela consola
@app.post("/rag/extract-urls")
async def rag_extract_urls(request: Request):
    try:
        data = await request.json()
        url = (data.get("url") or "").strip()
        raw_text = data.get("text")
        max_urls = int(data.get("max_urls") or 5000)

        if not url and not raw_text:
            return {"ok": False, "error": "fornece 'url' ou 'text'"}

        if url:
            r = requests.get(url, headers={"User-Agent": "AlmaBot/1.0 (+rag)"}, timeout=30)
            r.raise_for_status()
            raw_text = r.text

        if not raw_text:
            return {"ok": False, "error": "sem conteúdo"}

        urls = []

        def dedup_keep(seq):
            seen = set(); out = []
            for u in seq:
                u = (u or "").strip()
                if not u or len(u) > 2048: continue
                if u in seen: continue
                seen.add(u); out.append(u)
                if len(out) >= max_urls: break
            return out

        txt = raw_text

        if ("<urlset" in txt) or ("<sitemapindex" in txt):
            try:
                root = ET.fromstring(txt)
                locs = [(el.text or "").strip() for el in root.findall(".//{*}loc")]
                urls = dedup_keep(locs)
                return {"ok": True, "type": "sitemap", "count": len(urls), "urls": urls[:max_urls]}
            except Exception:
                import re
                locs = re.findall(r"<loc>(.*?)</loc>", txt, flags=re.I|re.S)
                urls = dedup_keep(locs)
                return {"ok": True, "type": "sitemap-regex", "count": len(urls), "urls": urls[:max_urls]}

        import re
        hrefs = re.findall(r'href=["\'](https?://[^"\']+)["\']', txt, flags=re.I)
        urls = dedup_keep(hrefs)
        return {"ok": True, "type": "html", "count": len(urls), "urls": urls[:max_urls]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------------------------------------------------------------------------------------
# 🔸 Exportação CSV de Orçamentos (mesmo não havendo “modo orçamento” no prompt)
# ---------------------------------------------------------------------------------------
def _safe_float(v, default=0.0):
    try:
        if isinstance(v, str):
            v = v.replace("€", "").replace(",", ".").strip()
        return float(v)
    except Exception:
        return float(default)

def _format_money(x: float) -> str:
    return f"{x:.2f}"

@app.post("/budget/csv")
async def budget_csv(request: Request):
    """
    Body:
    {
      "mode": "public" | "pro",
      "iva_pct": 23,
      "rows": [
        {"ref":"BS.01","descricao":"Produto","quant":1,"preco_uni":100,"desc_pct":5,"dim":"80x40xH45","material":"Carvalho / Óleo","marca":"Boa Safra","link":"https://interiorguider.com/..."}
      ]
    }
    """
    data = await request.json()
    mode = (data.get("mode") or "public").lower().strip()
    iva_pct = _safe_float(data.get("iva_pct", 23.0))
    rows = data.get("rows") or []

    if mode not in ("public", "pro"):
        return PlainTextResponse("mode deve ser 'public' ou 'pro'", status_code=400)
    if not isinstance(rows, list) or not rows:
        return PlainTextResponse("rows vazio", status_code=400)

    headers = ["REF.", "DESIGNAÇÃO / MATERIAL / ACABAMENTO / COR", "QUANT.", "PREÇO UNI.", "DESC.", "TOTAL S/IVA"] if mode=="public" \
        else ["REF.", "DESIGNAÇÃO / MATERIAL / ACABAMENTO / COR", "QUANT.", "PREÇO UNI.", "DESC.", "TOTAL C/IVA"]

    sio = StringIO()
    writer = csv.writer(sio)
    writer.writerow(headers)

    for r in rows:
        ref = (r.get("ref") or "").strip()
        quant = int(r.get("quant") or 1)
        preco_uni = _safe_float(r.get("preco_uni"), 0.0)
        desc_pct = _safe_float(r.get("desc_pct"), 0.0)

        desc_main = (r.get("descricao") or "").strip() or "Produto"
        extra_lines = []
        if r.get("dim"): extra_lines.append(f"Dimensões: {r['dim']}")
        if r.get("material"): extra_lines.append(f"Material/Acabamento: {r['material']}")
        if r.get("marca"): extra_lines.append(f"Marca: {r['marca']}")
        if r.get("link"):
            link = str(r["link"]).strip()
            extra_lines.append(f"Link: {link}")
        full_desc = desc_main + (("\n" + "\n".join(extra_lines)) if extra_lines else "")

        total_si = quant * preco_uni * (1.0 - desc_pct/100.0)
        total_col = _format_money(total_si if mode=="public" else total_si * (1.0 + iva_pct/100.0))

        writer.writerow([
            ref,
            full_desc,
            str(quant),
            _format_money(preco_uni),
            (f"{desc_pct:.0f}%" if desc_pct else ""),
            total_col
        ])

    csv_bytes = sio.getvalue().encode("utf-8-sig")
    fname = f"orcamento_{mode}_{int(time.time())}.csv"
    fpath = os.path.join("/tmp", fname)
    with open(fpath, "wb") as f:
        f.write(csv_bytes)

    return FileResponse(fpath, media_type="text/csv", filename=fname)

# ---------------------------------------------------------------------------------------
# ASK endpoints (com injeção + fallback de links e top-k dinâmico)
# ---------------------------------------------------------------------------------------
@app.get("/ask_get")
def ask_get(q: str = "", user_id: str = "anon", namespace: str = None, top_k: Optional[int] = None):
    if not q:
        return {"answer": "Falta query param ?q="}
    decided_top_k = _decide_top_k(q, top_k)
    _ = build_rag_products_block(q)  # mantém sinal de produtos no contexto
    messages, new_facts, rag_used, rag_hits = build_messages(user_id, q, namespace)
    try:
        answer = grok_chat(messages)
    except Exception as e:
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}
    answer = _postprocess_answer(answer, q, namespace, decided_top_k)
    answer = _force_links_block(answer, rag_hits, min_links=3)
    local_append_dialog(user_id, q, answer)
    _mem0_create(content=f"User: {q}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
    _mem0_create(content=f"Alma: {answer}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
    return {
        "answer": answer,
        "new_facts_detected": new_facts,
        "rag": {
            "used": rag_used,
            "top_k_default": RAG_TOP_K_DEFAULT,
            "top_k_effective": decided_top_k,
            "namespace": namespace or DEFAULT_NAMESPACE
        }
    }

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    user_id = (data.get("user_id") or "").strip() or "anon"
    namespace = (data.get("namespace") or "").strip() or None
    req_top_k = data.get("top_k")
    decided_top_k = _decide_top_k(question, req_top_k)
    log.info(f"[/ask] user_id={user_id} ns={namespace or DEFAULT_NAMESPACE} top_k={decided_top_k} question={question!r}")
    if not question:
        return {"answer": "Coloca a tua pergunta em 'question'."}
    _ = build_rag_products_block(question)
    messages, new_facts, rag_used, rag_hits = build_messages(user_id, question, namespace)
    try:
        answer = grok_chat(messages)
    except Exception as e:
        log.exception("Erro ao chamar a x.ai")
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}
    answer = _postprocess_answer(answer, question, namespace, decided_top_k)
    answer = _force_links_block(answer, rag_hits, min_links=3)
    local_append_dialog(user_id, question, answer)
    _mem0_create(content=f"User: {question}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
    _mem0_create(content=f"Alma: {answer}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
    return {
        "answer": answer,
        "new_facts_detected": new_facts,
        "rag": {
            "used": rag_used,
            "top_k_default": RAG_TOP_K_DEFAULT,
            "top_k_effective": decided_top_k,
            "namespace": namespace or DEFAULT_NAMESPACE
        }
    }
# ---------------------------------------------------------------------------------------
# 📚 Catálogo (SQLite) — backend único via CSV + integração no /status
# ---------------------------------------------------------------------------------------
import sqlite3
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from fastapi import UploadFile, File, Form

CATALOG_DB_PATH = os.getenv("CATALOG_DB_PATH", "/tmp/catalog.db")

# ---- DB helpers -----------------------------------------------------------------------
def _catalog_conn():
    conn = sqlite3.connect(CATALOG_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _table_cols(c) -> set:
    cols = set()
    for r in c.execute("PRAGMA table_info(catalog_items)").fetchall():
        cols.add(r[1])
    return cols

def _ensure_cols(c):
    """Garante colunas base + extra, sem quebrar DB já existente."""
    needed = {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "namespace": "TEXT",
        "name": "TEXT",
        "summary": "TEXT",
        "url": "TEXT UNIQUE",
        "source": "TEXT",
        "created_at": "TEXT",
        "updated_at": "TEXT",
        # extras (para orçamentos)
        "ref": "TEXT",
        "price": "REAL",
        "currency": "TEXT",
        "iva_pct": "REAL",
        "brand": "TEXT",
        "dimensions": "TEXT",
        "material": "TEXT",
        "image_url": "TEXT",
    }
    c.execute("""
    CREATE TABLE IF NOT EXISTS catalog_items (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      namespace TEXT,
      name TEXT,
      summary TEXT,
      url TEXT UNIQUE,
      source TEXT,
      created_at TEXT,
      updated_at TEXT
    )""")
    have = _table_cols(c)
    for col, ddl in needed.items():
        if col not in have:
            c.execute(f"ALTER TABLE catalog_items ADD COLUMN {col} {ddl}")

def _catalog_init():
    with _catalog_conn() as c:
        _ensure_cols(c)
        c.execute("CREATE INDEX IF NOT EXISTS idx_catalog_ns ON catalog_items(namespace)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_catalog_name ON catalog_items(name)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_catalog_url ON catalog_items(url)")
    log.info("[catalog/sqlite] ready db=%s", CATALOG_DB_PATH)

_catalog_init()

# ---- Utils -----------------------------------------------------------------------------
def _canon_ig_url(u: str) -> str:
    # Usa a função já definida acima no ficheiro (mantemos aqui só no caso de chamada direta)
    try:
        p = urlparse((u or "").strip())
    except Exception:
        return u or ""
    if not p.netloc:
        return u or ""
    host = p.netloc.lower().replace("www.", "")
    if IG_HOST not in host:
        return u or ""  # só canonizamos IG
    path = re.sub(r"/(products?|produtos?)\/", "/", p.path, flags=re.I)
    path = re.sub(r"//+", "/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    p = p._replace(scheme="https", netloc=IG_HOST, path=path)
    return urlunparse(p)

def _safe_float(v, default=None):
    try:
        if v is None or v == "":
            return default
        if isinstance(v, str):
            t = v.replace("€", "").replace("\u00A0"," ").strip()
            t = t.replace(".", "").replace(" ", "")
            t = t.replace(",", ".")
            return float(t)
        return float(v)
    except Exception:
        return default

def _coalesce(*vals, default=""):
    for x in vals:
        if x not in (None, "", [], {}):
            return x
    return default

# ---- Upsert genérico -------------------------------------------------------------------
def _upsert_catalog_row(ns: Optional[str],
                        name: str,
                        summary: str,
                        url: str,
                        source: str = "csv",
                        ref: Optional[str] = None,
                        price: Optional[float] = None,
                        currency: Optional[str] = None,
                        iva_pct: Optional[float] = None,
                        brand: Optional[str] = None,
                        dimensions: Optional[str] = None,
                        material: Optional[str] = None,
                        image_url: Optional[str] = None):
    if not url:
        return
    with _catalog_conn() as c:
        cur = c.execute("SELECT id FROM catalog_items WHERE url=?", (url,))
        row = cur.fetchone()
        if row:
            c.execute("""UPDATE catalog_items
                         SET namespace=?, name=?, summary=?, source=?, updated_at=?,
                             ref=COALESCE(?, ref),
                             price=COALESCE(?, price),
                             currency=COALESCE(?, currency),
                             iva_pct=COALESCE(?, iva_pct),
                             brand=COALESCE(?, brand),
                             dimensions=COALESCE(?, dimensions),
                             material=COALESCE(?, material),
                             image_url=COALESCE(?, image_url)
                         WHERE url=?""",
                      (ns, name, summary, source, _now(),
                       ref, price, currency, iva_pct, brand, dimensions, material, image_url,
                       url))
        else:
            c.execute("""INSERT INTO catalog_items(namespace,name,summary,url,source,created_at,updated_at,
                                                  ref,price,currency,iva_pct,brand,dimensions,material,image_url)
                         VALUES (?,?,?,?,?,?,?,
                                 ?,?,?,?,?,?,?,?)""",
                      (ns, name, summary, url, source, _now(), _now(),
                       ref, price, currency, iva_pct, brand, dimensions, material, image_url))

# ---- Importação CSV --------------------------------------------------------------------
# Aceita:
# - multipart/form-data com file=... (UploadFile) OU
# - JSON com {"csv_text":"...","namespace":"...","delimiter":",","encoding":"utf-8"}
# Mapeamento de colunas flexível (detecta variações comuns).
@app.post("/catalog/import-csv")
async def catalog_import_csv(
    file: UploadFile = File(None),
    namespace: str = Form(None),
    delimiter: str = Form(","),
    encoding: str = Form("utf-8"),
    json_req: Dict[str, Any] = None
):
    try:
        if (file is None) and (json_req is None):
            # tentar ler JSON do body se não veio form-data
            try:
                json_req = await Request.json  # type: ignore
            except Exception:
                pass

        csv_text = None
        if file is not None:
            raw = await file.read()
            csv_text = raw.decode(encoding or "utf-8", errors="replace")
        elif json_req and isinstance(json_req, dict):
            csv_text = (json_req.get("csv_text") or "").strip()
            namespace = json_req.get("namespace") or namespace
            delimiter = json_req.get("delimiter") or delimiter
            encoding  = json_req.get("encoding")  or encoding

        if not csv_text:
            return {"ok": False, "error": "Falta CSV (file ou json.csv_text)"}

        # normalização básica de linhas
        lines = [ln for ln in csv_text.splitlines() if ln.strip()]
        if not lines:
            return {"ok": False, "error": "CSV vazio"}
        header = [h.strip() for h in lines[0].split(delimiter)]
        rows = [ [c.strip() for c in ln.split(delimiter)] for ln in lines[1:] ]

        # índice por nome de coluna (flexível)
        def col_idx(*names):
            for n in names:
                if n in header:
                    return header.index(n)
            return -1

        idx_url  = col_idx("url","link","product_url")
        idx_name = col_idx("name","title","product_name")
        idx_sum  = col_idx("summary","descricao","description","body_html")
        idx_ref  = col_idx("ref","sku","reference","variant_sku")
        idx_price= col_idx("price","price_min","variant_price","preco")
        idx_curr = col_idx("currency","moeda")
        idx_iva  = col_idx("iva_pct","iva","tax_rate")
        idx_brand= col_idx("brand","marca","vendor")
        idx_dim  = col_idx("dimensions","dim","size")
        idx_mat  = col_idx("material","acabamento","materials")
        idx_img  = col_idx("image_url","image","thumbnail","featured_image")

        ok, fail = 0, 0
        details: List[Dict[str, Any]] = []

        for r in rows:
            try:
                def val(idx): 
                    return r[idx] if 0 <= idx < len(r) else ""
                url = _canon_ig_url(val(idx_url)) if idx_url != -1 else ""
                name = _coalesce(val(idx_name), val(idx_ref), url)
                summary = val(idx_sum)
                ref = val(idx_ref) or None
                price = _safe_float(val(idx_price), default=None) if idx_price != -1 else None
                currency = (val(idx_curr) or "EUR").upper() if idx_curr != -1 else "EUR"
                iva_pct = _safe_float(val(idx_iva), default=None) if idx_iva != -1 else None
                brand = val(idx_brand) or None
                dimensions = val(idx_dim) or None
                material = val(idx_mat) or None
                image_url = val(idx_img) or None

                if not url:
                    # se não há URL, tenta construir a partir do nome (não obrigatório)
                    if name:
                        url = ""
                _upsert_catalog_row(
                    namespace or DEFAULT_NAMESPACE,
                    name=name or (ref or url or "Produto"),
                    summary=summary or "",
                    url=url or (ref or name or ""),
                    source="csv",
                    ref=ref,
                    price=price,
                    currency=currency,
                    iva_pct=iva_pct,
                    brand=brand,
                    dimensions=dimensions,
                    material=material,
                    image_url=image_url
                )
                ok += 1
                details.append({"ok": True, "name": name, "url": url})
            except Exception as e:
                fail += 1
                details.append({"ok": False, "error": str(e)})

        return {"ok": True, "namespace": namespace or DEFAULT_NAMESPACE, "imported": ok, "failed": fail, "items": details[:50]}
    except Exception as e:
        log.exception("catalog_import_csv_failed")
        return {"ok": False, "error": str(e)}

# ---- CRUD leve -------------------------------------------------------------------------
@app.post("/catalog/upsert")
async def catalog_upsert(request: Request):
    """
    Upsert direto por URL.
    Body exemplo:
    {
      "namespace": "boasafra",
      "url": "https://interiorguider.com/produto/mesa-x",
      "name": "Mesa X 180",
      "summary": "Mesa em carvalho maciço...",
      "source": "manual",
      "ref": "BS.MX.180",
      "price": 1290.00,
      "currency": "EUR",
      "iva_pct": 23,
      "brand": "Boa Safra",
      "dimensions": "180x90xH75",
      "material": "Carvalho / Óleo",
      "image_url": "https://.../mesa-x.jpg"
    }
    """
    data = await request.json()
    url = _canon_ig_url((data.get("url") or "").strip()) or (data.get("url") or "").strip()
    if not url:
        return {"ok": False, "error": "Falta url"}
    ns = (data.get("namespace") or DEFAULT_NAMESPACE)
    name = (data.get("name") or "").strip() or url
    summary = (data.get("summary") or "").strip()
    source = (data.get("source") or "manual").strip()

    _upsert_catalog_row(
        ns, name, summary, url, source=source,
        ref=(data.get("ref") or None),
        price=(float(data["price"]) if data.get("price") not in (None, "") else None),
        currency=(data.get("currency") or None),
        iva_pct=(float(data["iva_pct"]) if data.get("iva_pct") not in (None, "") else None),
        brand=(data.get("brand") or None),
        dimensions=(data.get("dimensions") or None),
        material=(data.get("material") or None),
        image_url=(data.get("image_url") or None),
    )
    return {"ok": True, "url": url, "name": name}

@app.get("/catalog/get")
def catalog_get(url: str):
    """Obtém 1 item por URL."""
    url = _canon_ig_url((url or "").strip()) or (url or "").strip()
    if not url:
        return {"ok": False, "error": "Falta url"}
    with _catalog_conn() as c:
        cur = c.execute("""SELECT id, namespace, name, summary, url, source, updated_at,
                                  ref, price, currency, iva_pct, brand, dimensions, material, image_url
                           FROM catalog_items WHERE url=?""", (url,))
        row = cur.fetchone()
        if not row:
            return {"ok": False, "error": "não encontrado"}
        return {"ok": True, "item": dict(row)}

@app.get("/catalog/list")
def catalog_list(q: str = "", limit: int = 100, offset: int = 0):
    """
    Lista o catálogo com paginação simples. Params: ?q=&limit=&offset=
    """
    limit = max(1, min(500, int(limit or 100)))
    offset = max(0, int(offset or 0))
    like = f"%{q.strip()}%" if q else None
    with _catalog_conn() as c:
        if like:
            cur = c.execute("""SELECT id, namespace, name, summary, url, source, created_at, updated_at,
                                      ref, price, currency, iva_pct, brand, dimensions, material, image_url
                               FROM catalog_items
                               WHERE name LIKE ? OR summary LIKE ? OR url LIKE ? OR brand LIKE ? OR ref LIKE ?
                               ORDER BY updated_at DESC
                               LIMIT ? OFFSET ?""",
                            (like, like, like, like, like, limit, offset))
            items = [dict(r) for r in cur.fetchall()]
            cur2 = c.execute("""SELECT count(*) as n FROM catalog_items
                                WHERE name LIKE ? OR summary LIKE ? OR url LIKE ? OR brand LIKE ? OR ref LIKE ?""",
                             (like, like, like, like, like))
            total = int(cur2.fetchone()["n"])
        else:
            cur = c.execute("""SELECT id, namespace, name, summary, url, source, created_at, updated_at,
                                      ref, price, currency, iva_pct, brand, dimensions, material, image_url
                               FROM catalog_items
                               ORDER BY updated_at DESC
                               LIMIT ? OFFSET ?""", (limit, offset))
            items = [dict(r) for r in cur.fetchall()]
            cur2 = c.execute("SELECT count(*) as n FROM catalog_items")
            total = int(cur2.fetchone()["n"])
    out_items = []
    for r in items:
        out_items.append({
            "id": r["id"],
            "namespace": r["namespace"],
            "name": r["name"],
            "summary": r["summary"],
            "url": r["url"],
            "source": r["source"],
            "updated_at": r["updated_at"],
            "ref": r.get("ref"),
            "price": r.get("price"),
            "currency": r.get("currency"),
            "iva_pct": r.get("iva_pct"),
            "brand": r.get("brand"),
            "dimensions": r.get("dimensions"),
            "material": r.get("material"),
            "image_url": r.get("image_url"),
        })
    return {"ok": True, "total": total, "items": out_items}

@app.post("/catalog/clear")
def catalog_clear_sqlite():
    with _catalog_conn() as c:
        c.execute("DELETE FROM catalog_items")
    return {"ok": True}

# ---- Expor info do Catálogo (SQLite) no /status ----
def _status_catalog_sqlite():
    try:
        with _catalog_conn() as c:
            total = c.execute("SELECT COUNT(*) FROM catalog_items").fetchone()[0]
            ig = c.execute(
                "SELECT COUNT(*) FROM catalog_items WHERE url LIKE ?",
                (f"https://{IG_HOST}%",)
            ).fetchone()[0]
    except Exception:
        total, ig = 0, 0
    return {
        "backend": "sqlite",
        "db_path": CATALOG_DB_PATH,
        "total": int(total),
        "ig_items": int(ig),
        "endpoints": {
            "import_csv": "POST /catalog/import-csv",
            "upsert": "POST /catalog/upsert",
            "get": "GET /catalog/get?url=",
            "list": "GET /catalog/list?q=&limit=&offset=",
            "clear": "POST /catalog/clear",
        }
    }
# ---------------------------------------------------------------------------------------
# Local run
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
