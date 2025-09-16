# main.py — Alma Server (RAG-only + Memória + CSV automático no modo orçamento)
# ---------------------------------------------------------------------------------------
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import os
import requests
import logging
import uvicorn
import time
import re
import csv
import json
from io import StringIO
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, urlunparse
import difflib
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

APP_VERSION = os.getenv("APP_VERSION", "alma-server/rag-only-1+mem-strong-1+budget-auto-csv-1")

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
# Prompt adicional: MODO ORÇAMENTOS (RAG-first, sem catálogo)
# ---------------------------------------------------------------------------------------
ALMA_ORCAMENTO_PROMPT = """
REGRAS MODO ORÇAMENTO (ESTRITO):
- O preço do site interiorguider.com é COM IVA. Usa preço COM IVA quando houver URL do site.
- NÃO inventes produtos nem URLs. Usa apenas itens encontrados no RAG. Se faltar URL, escreve "sem URL".
- Se o item for de catálogo externo (sem link IG), acrescenta "disponibilidade a confirmar".
- Se o pedido não indicar variante, assume a VARIANTE DEFAULT (dimensão/tecido/cor base) e lista explicitamente o que assumiste.
- Quando faltar detalhe, sugere escolhas para afinar o orçamento (dimensão, tecido/cor/acabamento, estrutura, etc.).
- A pré-visualização no chat deve ser uma TABELA COMPACTA (REF/NOME, QTD, PREÇO UNI (C/IVA), DESCONTO, TOTAL (C/IVA), LINK).
- Mantém a lista completa; se for longa, devolve por blocos até 20 linhas por bloco.

Saída no chat:
- 1) Tabela curta no chat (pré-visualização).
- 2) Linhas em JSON no final entre tags ```json ... ```, com este formato:
  {
    "iva_pct": 23,
    "rows": [
      {"ref":"", "descricao":"", "quant":1, "preco_uni":"219,00", "desc_pct":"", "link":"https://interiorguider.com/..."}
    ]
  }
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
        # mantemos como está para sites externos; só canonizamos IG
        return u or ""
    path = re.sub(r"/(products?|produtos?)\/", "/", p.path, flags=re.I)
    path = re.sub(r"//+", "/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    p = p._replace(scheme="https", netloc=IG_HOST, path=path)
    return urlunparse(p)

def _fix_product_links_markdown(text: str) -> str:
    if not text:
        return text
    def _md_repl(m):
        label, url = m.group(1), m.group(2)
        return f"[{label}]({_canon_ig_url(url)})"
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", _md_repl, text)

    def _raw_repl(m):
        url = m.group(0)
        fixed = _canon_ig_url(url)
        return f"[ver produto]({fixed})"
    text = re.sub(
        rf"(?<!\]\()(https?://(?:www\.)?{re.escape(IG_HOST)}/[^\s)>\]]+)",
        _raw_repl,
        text
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

# --- Fallback por nome (quando não há REF explícita) ------------------------
_NAME_HINTS = [
    r"\bmesa\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bcadeira[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bbanco[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bcama[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bluminária[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
]
_PRICE_RE = re.compile(r"(?:€|\bEUR?\b)\s*([\d\.\s]{1,12}[,\.]\d{2}|\d{1,6})")

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
    return None if m is None else _parse_money_eu(m.group(1))

def rag_guess_products_by_name(question: str, top_k: int = 3) -> List[dict]:
    if not RAG_READY:
        return []
    terms = _extract_name_terms(question)
    results = []
    seen_urls = set()
    for term in terms:
        try:
            hits = search_chunks(query=term, namespace=DEFAULT_NAMESPACE, top_k=top_k) or []
        except Exception:
            hits = []
        for h in hits:
            meta = h.get("metadata", {}) or {}
            url = meta.get("url") or ""
            title = meta.get("title") or ""
            text  = h.get("text") or ""
            cu = _canon_ig_url(url) if url else ""
            if cu and cu in seen_urls:
                continue
            if cu: seen_urls.add(cu)
            price = _price_from_text(text) or _price_from_text(title)
            name  = title or term
            entry = {
                "ref": "",
                "name": name.strip(),
                "url": cu or "sem URL",
                "price_gross": price,                 # COM IVA quando capturado do site
                "source": "SITE" if (cu and IG_HOST in (urlparse(cu).netloc or "").lower()) else "CATALOGO_EXTERNO"
            }
            results.append(entry)
    return results

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
LAST_PRODUCTS: Dict[str, List[str]] = {}  # refs/nomes
LAST_SPECS: Dict[str, Dict[str, str]] = {}  # ex.: {"cadeira x": "vermelha, 6 un"}

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

def remember_products(user_id: str, names_or_refs: List[str], cap: int = 50):
    if not names_or_refs: return
    cur = LAST_PRODUCTS.get(user_id, [])
    for r in names_or_refs:
        r = (r or "").strip()
        if r and r not in cur:
            cur.append(r)
    LAST_PRODUCTS[user_id] = cur[-cap:]
    try:
        _mem0_create(content=f"{FACT_PREFIX}last_refs=" + ";".join(LAST_PRODUCTS[user_id]), user_id=user_id,
                     metadata={"source": "alma-server", "type": "fact", "key": "last_refs"})
    except Exception:
        pass

def remember_specs(user_id: str, product_name: str, spec_text: str):
    if not product_name or not spec_text: return
    LAST_SPECS.setdefault(user_id, {})
    LAST_SPECS[user_id][_norm(product_name)] = spec_text.strip()
    mem0_set_fact(user_id, f"spec::{_norm(product_name)}", spec_text.strip())

def recall_products(user_id: str) -> List[str]:
    facts = mem0_get_facts(user_id)
    if "last_refs" in facts:
        return [x.strip() for x in facts["last_refs"].split(";") if x.strip()]
    return LAST_PRODUCTS.get(user_id, [])

def recall_spec(user_id: str, product_name: str) -> Optional[str]:
    key = _norm(product_name or "")
    if not key: return None
    if user_id in LAST_SPECS and key in LAST_SPECS[user_id]:
        return LAST_SPECS[user_id][key]
    facts = mem0_get_facts(user_id)
    return facts.get(f"spec::{key}")

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
    t = " " + text.strip() + " "
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

# ---------------------------------------------------------------------------------------
# DETETOR de orçamento
# ---------------------------------------------------------------------------------------
def _is_budget_request(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    keys = [
        "orçamento", "orcamento", "orc", "cotação", "cotacao", "proposta", "preço total", "quanto fica",
        "fazer orçamento", "encomenda", "proforma", "fatura proforma", "tabela de preços", "preventivo",
        "excel", "folha de cálculo", "folha de calculo", "planilha", ".csv", "csv", "orçamento em excel",
        "presupuesto", "cotización", "cotizacion",
        "ref.", "designação", "descrição", "quant.", "preço uni.", "lista completa", "listar tudo"
    ]
    return any(k in t for k in keys)

# ---------------------------------------------------------------------------------------
# RAG helpers para bloco de contexto (produtos sugeridos pelo RAG)
# ---------------------------------------------------------------------------------------
def build_rag_products_block(question: str) -> str:
    if not RAG_READY:
        return ""
    # tentar hits diretos
    hits = []
    try:
        hits = search_chunks(query=question, namespace=DEFAULT_NAMESPACE, top_k=RAG_TOP_K) or []
    except Exception:
        hits = []
    # se pouco, tentar por nome
    if len(hits) < 3:
        guessed = rag_guess_products_by_name(question, top_k=3)
        lines = []
        for g in guessed[:6]:
            lines.append(f"- NOME={g.get('name') or '-'}; URL={g.get('url') or 'sem URL'}; PRECO_COM_IVA={g.get('price_gross') if g.get('price_gross') is not None else 'N/D'}; SOURCE={g.get('source') or '-'}")
        return "Produtos sugeridos pelo RAG:\n" + "\n".join(lines) if lines else ""
    # se muitos hits, sumarizar por título+url para o LLM
    seen=set(); lines=[]
    for h in hits[:6]:
        meta = h.get("metadata", {}) or {}
        title = (meta.get("title") or "").strip()
        url = _canon_ig_url(meta.get("url") or "")
        key = (title, url)
        if key in seen: continue
        seen.add(key)
        if title or url:
            lines.append(f"- NOME={title or '-'}; URL={url or 'sem URL'}")
    return "Produtos sugeridos pelo RAG:\n" + "\n".join(lines) if lines else ""

# ---------------------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------------------
def _safe_float(v, default=0.0):
    if isinstance(v, (int, float)):
        return float(v)
    f = _parse_money_eu(v)
    return f if f is not None else float(default)

def _fmt_money(x: float, decimal="comma"):
    s = f"{x:.2f}"
    return s.replace(".", ",") if decimal == "comma" else s

def parse_budget_rows_from_answer(llm_answer: str) -> Dict:
    """
    Procura um bloco ```json ... ``` com {"iva_pct":..., "rows":[...]}
    Se não encontrar, tenta extrair linhas de uma tabela Markdown simples.
    """
    iva_pct = 23
    rows: List[Dict] = []

    # 1) JSON entre fences
    m = re.search(r"```json\s*(\{.*?\})\s*```", llm_answer, flags=re.S|re.I)
    if m:
        try:
            obj = json.loads(m.group(1))
            iva_pct = int(obj.get("iva_pct", 23))
            rows = obj.get("rows") or []
            if isinstance(rows, list) and rows:
                return {"iva_pct": iva_pct, "rows": rows}
        except Exception:
            pass

    # 2) Tabela markdown muito simples (cabeçalhos conhecidos)
    lines = [l.strip() for l in llm_answer.splitlines() if l.strip()]
    header_idx = -1
    for i, l in enumerate(lines):
        if ("ref" in l.lower() and "preço" in l.lower()) or ("descricao" in _norm(l) and "quant" in _norm(l)):
            header_idx = i; break
    if header_idx >= 0:
        for l in lines[header_idx+1:]:
            cells = [c.strip() for c in l.split("|")]
            cells = [c for c in cells if c != ""]
            if len(cells) < 4:
                continue
            # heurística: [ref/nome, quant, preço, desconto?, link?]
            ref = cells[0]
            quant = cells[1] if len(cells) > 1 else "1"
            preco = cells[2] if len(cells) > 2 else "0"
            desc  = cells[3] if len(cells) > 3 else ""
            link  = cells[4] if len(cells) > 4 else ""
            rows.append({
                "ref": ref,
                "descricao": ref,
                "quant": quant,
                "preco_uni": preco,
                "desc_pct": desc,
                "link": link
            })
    return {"iva_pct": iva_pct, "rows": rows}

def make_budget_csv(iva_pct: float, rows: List[dict], mode: str = "public", delimiter: str = ";", decimal: str = "comma") -> str:
    if mode not in ("public", "pro"):
        mode = "public"
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows vazio")

    if mode == "public":
        headers = ["REF.", "DESIGNAÇÃO / MATERIAL / ACABAMENTO / COR", "QUANT.", "PREÇO UNI.", "DESC.", "TOTAL C/IVA"]
        show_total_ci = True
    else:
        headers = ["REF.", "DESIGNAÇÃO / MATERIAL / ACABAMENTO / COR", "QUANT.", "PREÇO UNI.", "DESC.", "TOTAL S/IVA"]
        show_total_ci = False

    sio = StringIO()
    writer = csv.writer(sio, delimiter=delimiter)
    writer.writerow(headers)

    for r in rows:
        ref = (r.get("ref") or "").strip()
        quant = int(_safe_float(r.get("quant"), 1))
        preco_uni = _safe_float(r.get("preco_uni"), 0.0)
        desc_pct = _safe_float(r.get("desc_pct"), 0.0)

        desc_main = (r.get("descricao") or ref or "Produto").strip()
        extra_lines = []
        if r.get("dim"): extra_lines.append(f"Dimensões: {r['dim']}")
        if r.get("material"): extra_lines.append(f"Material/Acabamento: {r['material']}")
        if r.get("marca"): extra_lines.append(f"Marca: {r['marca']}")
        if r.get("link"):
            link = _canon_ig_url(str(r["link"]).strip()) if str(r["link"]).startswith("http") else str(r["link"]).strip()
            extra_lines.append(f"Link: {link if link else 'sem URL'}")

        full_desc = desc_main + (("\n" + "\n".join(extra_lines)) if extra_lines else "")

        total_si = quant * preco_uni * (1.0 - desc_pct/100.0)
        total = total_si * (1.0 + iva_pct/100.0) if show_total_ci else total_si

        writer.writerow([
            ref,
            full_desc,
            str(quant),
            _fmt_money(preco_uni, decimal=decimal),
            (f"{desc_pct:.0f}%" if desc_pct else ""),
            _fmt_money(total, decimal=decimal)
        ])

    csv_bytes = sio.getvalue().encode("utf-8-sig")
    fname = f"orcamento_{mode}_{int(time.time())}.csv"
    fpath = os.path.join("/tmp", fname)
    with open(fpath, "wb") as f:
        f.write(csv_bytes)
    return fpath

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

def build_messages(user_id: str, question: str, namespace: Optional[str]):
    # 0) extrair e guardar factos rápidos
    new_facts = extract_contextual_facts_pt(question)
    for k, v in new_facts.items():
        mem0_set_fact(user_id, k, v)

    # 1) mem de curto prazo
    short_snippets = _mem0_search(question, user_id=user_id, limit=5) or local_search_snippets(user_id, limit=5)
    memory_block = "Memórias recentes do utilizador (curto prazo):\n" + "\n".join(f"- {s}" for s in short_snippets[:3]) if short_snippets else ""

    # 2) RAG — bloco de conhecimento & produtos
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
    if _is_budget_request(question):
        messages.append({"role": "system", "content": ALMA_ORCAMENTO_PROMPT})
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

@app.get("/status")
def status_json():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "message": "Alma server ativo. Use POST /ask (Grok+Memória+RAG) ou endpoints RAG.",
        "mem0": {"enabled": MEM0_ENABLE, "client_ready": bool(mem0_client)},
        "rag": {"available": RAG_READY, "top_k": RAG_TOP_K, "namespace": DEFAULT_NAMESPACE},
        "endpoints": {
            "health": "/health",
            "ask": "POST /ask {question, user_id?, namespace?}",
            "ask_get": "/ask_get?q=...&user_id=...&namespace=...",
            "ping_grok": "/ping_grok",
            "mem_facts": "/mem/facts?user_id=...",
            "mem_search": "/mem/search?user_id=...&q=...",
            "rag_search_get": "/rag/search?q=...&namespace=...",
            "rag_crawl": "POST /rag/crawl",
            "rag_ingest_sitemap": "POST /rag/ingest-sitemap",
            "rag_ingest_url": "POST /rag/ingest-url",
            "rag_ingest_text": "POST /rag/ingest-text",
            "rag_ingest_pdf_url": "POST /rag/ingest-pdf-url",
            "rag_search_post": "POST /rag/search {query, namespace?, top_k?}",
            "budget_csv": "POST /budget/csv"
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
# Console HTML
# ---------------------------------------------------------------------------------------
@app.get("/console", response_class=HTMLResponse)
def serve_console():
    try:
        with open("console.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return HTMLResponse("<h1>console.html não encontrado</h1>", status_code=404)

@app.get("/alma-chat", response_class=HTMLResponse)
def serve_alma_chat():
    try:
        with open("alma-chat.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return HTMLResponse("<h1>alma-chat.html não encontrado</h1>", status_code=404)

# ---------------------------------------------------------------------------------------
# ASK endpoints (com CSV automático no modo orçamento)
# ---------------------------------------------------------------------------------------
@app.get("/ask_get")
def ask_get(q: str = "", user_id: str = "anon", namespace: str = None):
    if not q:
        return {"answer": "Falta query param ?q="}
    messages, new_facts, rag_used = build_messages(user_id, q, namespace)
    try:
        answer = grok_chat(messages)
    except Exception as e:
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

    answer = _postprocess_answer(answer)
    local_append_dialog(user_id, q, answer)
    _mem0_create(content=f"User: {q}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
    _mem0_create(content=f"Alma: {answer}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})

    resp = {
        "answer": answer,
        "new_facts_detected": new_facts,
        "rag": {"used": rag_used, "top_k": RAG_TOP_K, "namespace": namespace or DEFAULT_NAMESPACE}
    }

    # CSV automático se for orçamento
    if _is_budget_request(q):
        obj = parse_budget_rows_from_answer(answer)
        if obj.get("rows"):
            try:
                csv_path = make_budget_csv(obj.get("iva_pct", 23), obj["rows"], mode="public", delimiter=";", decimal="comma")
                resp["csv_download"] = csv_path
            except Exception as e:
                resp["csv_error"] = str(e)

    return resp

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    user_id = (data.get("user_id") or "").strip() or "anon"
    namespace = (data.get("namespace") or "").strip() or None
    log.info(f"[/ask] user_id={user_id} ns={namespace or DEFAULT_NAMESPACE} question={question!r}")

    if not question:
        return {"answer": "Coloca a tua pergunta em 'question'."}

    messages, new_facts, rag_used = build_messages(user_id, question, namespace)

    try:
        answer = grok_chat(messages)
    except Exception as e:
        log.exception("Erro ao chamar a x.ai")
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

    answer = _postprocess_answer(answer)
    local_append_dialog(user_id, question, answer)
    _mem0_create(content=f"User: {question}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
    _mem0_create(content=f"Alma: {answer}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})

    resp = {
        "answer": answer,
        "new_facts_detected": new_facts,
        "rag": {"used": rag_used, "top_k": RAG_TOP_K, "namespace": namespace or DEFAULT_NAMESPACE}
    }

    # CSV automático se for orçamento
    if _is_budget_request(question):
        obj = parse_budget_rows_from_answer(answer)
        if obj.get("rows"):
            try:
                csv_path = make_budget_csv(obj.get("iva_pct", 23), obj["rows"], mode="public", delimiter=";", decimal="comma")
                resp["csv_download"] = csv_path
            except Exception as e:
                resp["csv_error"] = str(e)

    return resp

# ---------------------------------------------------------------------------------------
# Endpoints de memória (debug/auxiliar)
# ---------------------------------------------------------------------------------------
@app.get("/mem/facts")
def mem_facts(user_id: str):
    return {"ok": True, "user_id": user_id, "facts": mem0_get_facts(user_id)}

@app.get("/mem/search")
def mem_search(user_id: str, q: str = "", limit: int = 5):
    results = _mem0_search(q or "contexto", user_id=user_id, limit=limit) or local_search_snippets(user_id, limit=limit)
    return {"ok": True, "user_id": user_id, "q": q, "results": results[:limit]}

# ---------------------------------------------------------------------------------------
# RAG Endpoints (crawl, sitemap, url, text, pdf, search POST)
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
        top_k     = int(data.get("top_k") or os.getenv("RAG_TOP_K", "6"))
        matches = search_chunks(query=query, namespace=namespace or DEFAULT_NAMESPACE, top_k=top_k)
        ctx = build_context_block(matches, token_budget=RAG_CONTEXT_TOKEN_BUDGET)
        return {"ok": True, "matches": matches, "context_block": ctx}
    except Exception as e:
        return {"ok": False, "error": "search_failed", "detail": str(e)}

# --- Proxy: extrair URLs -----------------------------------------------------
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
# Endpoint de exportação CSV (caso queiras chamar diretamente)
# ---------------------------------------------------------------------------------------
@app.post("/budget/csv")
async def budget_csv(request: Request):
    data = await request.json()
    iva_pct = float(data.get("iva_pct", 23))
    rows = data.get("rows") or []
    try:
        fpath = make_budget_csv(iva_pct, rows, mode="public", delimiter=";", decimal="comma")
        return FileResponse(fpath, media_type="text/csv", filename=os.path.basename(fpath))
    except Exception as e:
        return PlainTextResponse(str(e), status_code=400)

# ---------------------------------------------------------------------------------------
# Local run
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
