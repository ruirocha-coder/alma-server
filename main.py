# main.py — Alma Server (RAG-only + Memória) — sem orçamento/CSV
# ---------------------------------------------------------------------------------------
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import os
import requests
import logging
import uvicorn
import time
import re
import json
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, urlunparse

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

APP_VERSION = os.getenv("APP_VERSION", "alma-server/rag-only-2+mem-strong-1")

# ---------------------------------------------------------------------------------------
# Prompt nuclear da Alma
# ---------------------------------------------------------------------------------------
ALMA_MISSION = """
És a Alma, inteligência da Boa Safra Lda (Boa Safra + Interior Guider).
A tua missão é apoiar a direção (Rui Rocha) e a equipa para que a empresa
sobreviva e prospere, com respostas úteis, objetivas e calmas.

Estilo (estrito)
- Clareza e concisão: vai direto ao ponto. Máximo 1 frase de abertura.
- Empatia sob medida: só comenta o estado emocional quando houver sinais de stress
  (“urgente”, “aflito”, “atraso”, “problema”, “ansioso”, “sob pressão”). Caso contrário,
  não faças small talk.
- Valores implícitos: mantém o alinhamento sem o declarar. Nunca escrevas
  “estou alinhada com os valores” ou “em nome da missão”.
- Vocabulário disciplinado:
  * “psicoestética/psicoestético” apenas quando tecnicamente relevante (no máximo 1 vez).
  * Evita frases feitas e entusiasmos excessivos.
- Seguimento: no fim, no máximo 1 pergunta, apenas se desbloquear o próximo passo concreto.

Proibido
- Iniciar com “Como vai o teu dia?”, “Espero que estejas bem”, “Espero que seja útil”
  ou “alinhado com os valores…”.
- Alongar justificações sobre missão/valores.
- Emojis, múltiplas exclamações, tom efusivo.

Funções
1) Estratégia — apoiar a direção na definição/monitorização de estratégias de sobrevivência e crescimento.
2) Apoio Comercial — esclarecer produtos, preços, prazos e características técnicas.
3) Método (quando relevante) — aconselhar a equipa no método psicoestético sem anunciar o rótulo;
   foca no raciocínio (luz, materiais, uso, bem-estar).
4) Suporte Humano (condicional) — se houver stress, reconhecer e reduzir carga (“Vamos por partes…”).
5) Procedimentos — explicar regras internas e leis relevantes de forma clara.
6) Respostas Gerais — combinar RAG e Grok; se faltar evidência, diz o que não sabes e o passo para obter.

Contexto
- Boa Safra: editora de design natural português para a casa, com coleção própria.
- Interior Guider (2025): design de interiores com perspetiva psicoestética e marcas parceiras.

Links de produtos
- Sempre que mencionares produtos, usa o link presente no RAG (interiorguider.com quando existir).
- Se não tiveres URL no RAG, escreve literalmente "sem URL". Não inventes links.

Formato de resposta
- 1 bloco curto; usa bullets apenas quando ajudam a agir.
- Termina com 1 próxima ação concreta (p.ex.: “Queres que valide o prazo com o fornecedor?”).
"""

# ---------------------------------------------------------------------------------------
# Utilidades de URL e normalização
# ---------------------------------------------------------------------------------------
IG_HOST = os.getenv("IG_HOST", "interiorguider.com").lower()
_BAD_HOSTS = {"w3.org", "www.w3.org", "schema.org", "www.schema.org"}
_BAD_PREFIXES = ("mailto:", "tel:", "data:")

def _canon_ig_url(u: str) -> str:
    try:
        p = urlparse((u or "").strip())
    except Exception:
        return u or ""
    if not p.netloc:
        return u or ""
    host = p.netloc.lower().replace("www.", "")
    if IG_HOST not in host:
        # mantemos como está para sites externos; só canonizamos IG
        return u or ""
    path = re.sub(r"/(products?|produtos?)\/", "/", p.path, flags=re.I)
    path = re.sub(r"//+", "/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    p = p._replace(scheme="https", netloc=IG_HOST, path=path)
    return urlunparse(p)

# --- transformar URLs soltas do IG (com e sem esquema) em markdown clicável ------------
def _fix_product_links_markdown(text: str) -> str:
    if not text:
        return text

    # 1) links já em markdown → apenas canonizar IG
    def _md_repl(m):
        label, url = m.group(1), m.group(2)
        return f"[{label}]({_canon_ig_url(url)})"
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", _md_repl, text)

    # 2) URLs IG cruas com esquema → embrulhar em markdown
    def _raw_repl(m):
        url = m.group(0)
        fixed = _canon_ig_url(url)
        return f"[ver produto]({fixed})"
    text = re.sub(
        rf"(?<!\]\()(https?://(?:www\.)?{re.escape(IG_HOST)}/[^\s)>\]]+)",
        _raw_repl,
        text
    )

    # 3) URLs IG cruas sem esquema (ex.: interiorguider.com/mesa-family) → https + markdown
    def _noscheme_repl(m):
        path = m.group(1)
        fixed = f"https://{IG_HOST}{path}"
        fixed = _canon_ig_url(fixed)
        return f"[ver produto]({fixed})"
    text = re.sub(
        rf"(?<!\]\()(?<!https?://)(?:www\.)?{re.escape(IG_HOST)}(/[\w\-/\.%#\?=&]+)",
        _noscheme_repl,
        text,
        flags=re.I
    )

    return text

def _postprocess_answer(answer: str) -> str:
    return _fix_product_links_markdown(answer or "")

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

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))
RAG_CONTEXT_TOKEN_BUDGET = int(os.getenv("RAG_CONTEXT_TOKEN_BUDGET", "1600"))
DEFAULT_NAMESPACE = os.getenv("RAG_DEFAULT_NAMESPACE", "").strip() or None

# --- Heurística de nomes -------------------------------------------------
_NAME_HINTS = [
    r"\bmesa\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bcadeira[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bbanco[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bcama[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bluminária[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
]

# --- Aliases genéricos (singular/plural e sinónimos frequentes) ---------
ALIAS_TERMS = {
    r"\bcadeira(?:s)?\s+cod\b": ["cadeira cod", "cod chair"],
    r"\bbanco(?:s)?\s+(?:tr[eê]s|3)\s+patas\b": ["3 leg stool", "banco três patas", "banco 3 patas"],
    r"\bmesa\s+family\b": ["mesa family", "family table"],
}

def _expand_alias_queries(text: str) -> List[str]:
    out = []
    for pat, qlist in ALIAS_TERMS.items():
        if re.search(pat, text, flags=re.I):
            out.extend(qlist)
    # dedupe mantendo ordem
    seen=set(); res=[]
    for q in out:
        k=_norm(q)
        if k not in seen:
            seen.add(k); res.append(q)
    return res

def _extract_name_terms(text:str) -> List[str]:
    t = " " + (text or "") + " "
    terms = []

    # aliases primeiro
    terms.extend(_expand_alias_queries(text or ""))

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

# --- util para URLs dentro do texto do ponto do RAG ---------------------
_URL_RE = re.compile(r"https?://[^\s)>\]\"']+", re.I)

def _is_bad_url(u: str) -> bool:
    if not u: return True
    if any(u.lower().startswith(p) for p in _BAD_PREFIXES): return True
    host = (urlparse(u).netloc or "").lower()
    h = host.replace("www.", "")
    return h in _BAD_HOSTS

def _urls_from_text(txt: str, only_ig: bool = True) -> List[str]:
    if not txt: return []
    urls = _URL_RE.findall(txt)
    out, seen = [], set()
    for u in urls:
        if _is_bad_url(u):  # filtra w3/schema/mailto/etc.
            continue
        if only_ig and IG_HOST not in (urlparse(u).netloc or "").lower():
            continue
        cu = _canon_ig_url(u) if IG_HOST in (urlparse(u).netloc or "").lower() else u.strip()
        if cu and cu not in seen:
            seen.add(cu); out.append(cu)
    # preferir interiorguider.com
    out.sort(key=lambda x: 0 if IG_HOST in (urlparse(x).netloc or "").lower() else 1)
    return out

def _best_product_url(meta: dict, text: str) -> str:
    # 1º: metadata.url (de confiança)
    u = (meta or {}).get("url") or ""
    if u:
        cu = _canon_ig_url(u)
        if cu: return cu
    # 2º: procurar no texto apenas URLs do IG
    urls = _urls_from_text(text or "", only_ig=True)
    return urls[0] if urls else ""

# ---------------------------------------------------------------------------------------
# Extração de preços (aproximada) -------------------------------------------------------
_PRICE_RE = re.compile(r"(?:€|\bEUR?\b)\s*([\d\.\s]{1,12}[,\.]\d{2}|\d{1,6})")

def _parse_money_eu(s: str) -> Optional[float]:
    try:
        if s is None:
            return None
        s = str(s).strip().replace("€", "").replace("\u00A0", " ").strip()
        s = re.sub(r"\s+", "", s)
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")   # EU -> .
            else:
                s = s.replace(",", "")                    # US -> remove milhar
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
    return _parse_money_eu(m.group(1))

# ---------------------------------------------------------------------------------------
# Sugestão de produtos a partir do RAG --------------------------------------------------
def rag_guess_products_by_name(question: str, top_k: int = 3) -> List[dict]:
    if not RAG_READY:
        return []
    # termos vindos da heurística + aliases
    base_terms = _extract_name_terms(question)
    alias_terms = _expand_alias_queries(question)
    terms = []
    seen_norm = set()
    for q in (base_terms + alias_terms):
        k = _norm(q)
        if k and k not in seen_norm:
            seen_norm.add(k); terms.append(q)

    results = []
    seen_urls = set()
    for term in terms:
        try:
            hits = search_chunks(query=term, namespace=DEFAULT_NAMESPACE, top_k=top_k) or []
        except Exception:
            hits = []
        for h in hits:
            meta = h.get("metadata", {}) or {}
            text = h.get("text") or ""
            title = (meta.get("title") or "").strip() or term
            url = _best_product_url(meta, text)
            if url and url in seen_urls:
                continue
            if url: seen_urls.add(url)
            price = _price_from_text(text) or _price_from_text(title)
            entry = {
                "ref": "",
                "name": title,
                "url": url or "sem URL",
                "price_gross": price,  # COM IVA quando capturado do site
                "source": "SITE" if (url and IG_HOST in (urlparse(url).netloc or "").lower()) else "CATALOGO_EXTERNO"
            }
            results.append(entry)
    return results

def build_rag_products_block(question: str) -> str:
    if not RAG_READY:
        return ""
    # tentar hits diretos
    try:
        hits = search_chunks(query=question, namespace=DEFAULT_NAMESPACE, top_k=RAG_TOP_K) or []
    except Exception:
        hits = []

    lines = []
    seen = set()
    # 1) hits diretos com URL pelo meta/texto
    for h in hits[:6]:
        meta = h.get("metadata", {}) or {}
        title = (meta.get("title") or "").strip() or "-"
        url = _best_product_url(meta, h.get("text") or "")
        key = (title, url)
        if key in seen: 
            continue
        seen.add(key)
        lines.append(f"- NOME={title}; URL={url or 'sem URL'}")
    # 2) poucos? completar com guesses por nome
    if len(lines) < 3:
        for g in rag_guess_products_by_name(question, top_k=3)[:6]:
            key = (g.get("name","-"), g.get("url","sem URL"))
            if key in seen: 
                continue
            seen.add(key)
            price = g.get("price_gross")
            price_str = f"{price:.2f}" if isinstance(price,(int,float)) else "N/D"
            lines.append(f"- NOME={g.get('name','-')}; URL={g.get('url','sem URL')}; PRECO_COM_IVA={price_str}")
    return "Produtos sugeridos pelo RAG:\n" + "\n".join(lines) if lines else ""

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
                pkg_name = "mem0ai"
            except Exception:
                import mem0 as _mem0_pkg
                from mem0 import MemoryClient as _MC
                pkg_name = "mem0"
            MemoryClient = _MC
            log.info(f"[mem0] import OK ({pkg_name}) file={getattr(_mem0_pkg,'__file__','?')}")
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
        else:
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
# Extração de factos simples do texto (nome, localização, preferências, divisão)
# ---------------------------------------------------------------------------------------
NAME_PATTERNS = [
    r"\bchamo-?me\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}(?:\s+[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}){0,3})\b",
    r"\bo\s+meu\s+nome\s+é\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}(?:\s+[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}){0,3})\b",
    r"\bsou\s+(?:o|a)\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}(?:\s+[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}){0,3})\b",
]
CITY_PATTERNS = [
    r"\bmoro\s+(?:em|no|na)\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\w\s\-\.'ÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç]{2,60})",
    r"\bestou\s+(?:em|no|na)\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\w\s\-\.'ÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç]{2,60})",
    r"\bsou\s+de\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\w\s\-\.'ÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç]{2,60})",
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
# Construção das mensagens (RAG-first, sem módulo de orçamento)
# ---------------------------------------------------------------------------------------
def build_messages(user_id: str, question: str, namespace: Optional[str]):
    # 0) extrair e guardar factos rápidos
    new_facts = extract_contextual_facts_pt(question)
    for k, v in new_facts.items():
        mem0_set_fact(user_id, k, v)

    # 1) mem de curto prazo
    short_snippets = _mem0_search(question, user_id=user_id, limit=5) or local_search_snippets(user_id, limit=5)
    memory_block = "Memórias recentes do utilizador (curto prazo):\n" + "\n".join(f"- {s}" for s in short_snippets[:3]) if short_snippets else ""

    # 2) RAG — bloco de conhecimento & produtos (sempre fornece nomes+URLs válidos)
    rag_block = ""
    rag_used = False
    if RAG_READY:
        try:
            rag_hits = search_chunks(query=question, namespace=namespace or DEFAULT_NAMESPACE, top_k=RAG_TOP_K)
            rag_block = build_context_block(rag_hits, token_budget=RAG_CONTEXT_TOKEN_BUDGET) if rag_hits else ""
            rag_used = bool(rag_block)
        except Exception as e:
            log.warning(f"[rag] search falhou: {e}")
            rag_block = ""
            rag_used = False

    products_block = build_rag_products_block(question)

    # 3) construir mensagens
    messages = [{"role": "system", "content": ALMA_MISSION}]
    fb = facts_block_for_user(user_id)
    if fb: messages.append({"role": "system", "content": fb})
    if rag_block:
        messages.append({"role": "system", "content": f"Conhecimento corporativo (RAG):\n{rag_block}"})
    if products_block:
        messages.append({"role": "system", "content": products_block})
    if memory_block:
        messages.append({"role": "system", "content": memory_block})
    messages.append({"role": "user", "content": question})
    return messages, new_facts, rag_used

# ---------------------------------------------------------------------------------------
# ROTAS BÁSICAS
# ---------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return HTMLResponse("<h1>index.html não encontrado</h1>", status_code=404)

# Página do Alma Chat (serve alma-chat.html)
@app.get("/alma-chat", response_class=HTMLResponse)
@app.get("/alma-chat/", response_class=HTMLResponse)
def alma_chat():
    html_path = os.path.join(os.getcwd(), "alma-chat.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<h1>alma-chat.html não encontrado</h1>", status_code=404)

@app.get("/status")
def status_json():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "message": "Alma server ativo. Use POST /ask (Grok+Memória+RAG).",
        "mem0": {"enabled": MEM0_ENABLE, "client_ready": bool(mem0_client)},
        "rag": {"available": RAG_READY, "top_k": RAG_TOP_K, "namespace": DEFAULT_NAMESPACE},
        "endpoints": {
            "health": "/health",
            "ask": "POST /ask {question, user_id?, namespace?}",
            "ask_get": "/ask_get?q=...&user_id=...&namespace=...",
            "ping_grok": "/ping_grok",
            "rag_search_get": "/rag/search?q=...&namespace=...",
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mem0_enabled": MEM0_ENABLE,
        "mem0_client_ready": bool(mem0_client),
        "model": MODEL,
        "rag_available": RAG_READY,
        "rag_default_namespace": DEFAULT_NAMESPACE
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
# RAG: GET /rag/search (debug)
# ---------------------------------------------------------------------------------------
@app.get("/rag/search")
def rag_search_get(q: str, namespace: str = None, top_k: int = None):
    if not RAG_READY:
        return {"ok": False, "error": "rag_client indisponível no servidor"}
    try:
        res = search_chunks(query=q, namespace=namespace or DEFAULT_NAMESPACE, top_k=top_k or RAG_TOP_K)
        return {"ok": True, "query": q, "matches": res}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------------------------------------------------------------------------------------
# ASK endpoints (sem orçamento/CSV; apenas resposta + pós-processo de links)
# ---------------------------------------------------------------------------------------
@app.get("/ask_get")
def ask_get(q: str = "", user_id: str = "anon", namespace: str = None):
    if not q: return {"answer": "Falta query param ?q="}
    messages, new_facts, rag_used = build_messages(user_id, q, namespace)
    try:
        answer = grok_chat(messages)
    except Exception as e:
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}
    answer = _postprocess_answer(answer)
    try:
        local_append_dialog(user_id, q, answer)
        _mem0_create(content=f"User: {q}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
        _mem0_create(content=f"Alma: {answer}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
    except Exception:
        pass
    return {"answer": answer, "new_facts_detected": new_facts,
            "rag": {"used": rag_used, "top_k": RAG_TOP_K, "namespace": namespace or DEFAULT_NAMESPACE}}

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    user_id = (data.get("user_id") or "").strip() or "anon"
    namespace = (data.get("namespace") or "").strip() or None
    log.info(f"[/ask] user_id={user_id} ns={namespace or DEFAULT_NAMESPACE} question={question!r}")
    if not question: return {"answer": "Coloca a tua pergunta em 'question'."}

    messages, new_facts, rag_used = build_messages(user_id, question, namespace)
    try:
        answer = grok_chat(messages)
    except Exception as e:
        log.exception("Erro ao chamar a x.ai")
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}
    answer = _postprocess_answer(answer)
    try:
        local_append_dialog(user_id, question, answer)
        _mem0_create(content=f"User: {question}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
        _mem0_create(content=f"Alma: {answer}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
    except Exception:
        pass

    return {"answer": answer, "new_facts_detected": new_facts,
            "rag": {"used": rag_used, "top_k": RAG_TOP_K, "namespace": namespace or DEFAULT_NAMESPACE}}

# --- local run ----------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
