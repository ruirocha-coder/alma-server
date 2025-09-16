# main.py — Alma Server (Grok + Memória + Catálogo Interno + RAG/Qdrant + CSV)
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

APP_VERSION = os.getenv("APP_VERSION", "alma-server/catalog-2+budget-pt-2+mem-refs-1")

# ---------------------------------------------------------------------------------------
# Prompt nuclear da Alma (deixa como está no teu projeto)
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
- Sempre que mencionares produtos, inclui um link para a página do produto no site interiorguider.com
  quando existir. Se existirem vários links, escolhe o do interiorguider.com.

Formato de resposta
- 1 bloco curto; usa bullets apenas quando ajudam a agir.
- Termina com 1 próxima ação concreta (p.ex.: “Queres que valide o prazo com o fornecedor?”).
"""

# ---------------------------------------------------------------------------------------
# Prompt adicional: MODO ORÇAMENTOS
# ---------------------------------------------------------------------------------------
ALMA_ORCAMENTO_PROMPT = """
REGRAS MODO ORÇAMENTO (ESTRITO):
- O preço do site interiorguider.com é COM IVA. Usa preço com IVA quando houver URL do site.
- NÃO inventes produtos. Usa apenas PRODUTOS_RECONHECIDOS fornecidos abaixo.
- Se um item não estiver nos PRODUTOS_RECONHECIDOS: pede apenas a mínima especificação em falta (ex.: referência, variante, quantidade).
- Se um item tiver "source=CATALOGO_EXTERNO": inclui nota "disponibilidade a confirmar".
- Se um item tiver URL em falta, NÃO inventes; escreve "sem URL" e mantém a REF/NOME. Se SOURCE=CATALOGO_EXTERNO, acrescenta a nota "disponibilidade a confirmar".
- Links: mostrar como Markdown clicável, preferindo interiorguider.com.

Saída:
- Tabela curta no chat (pré-visualização).
- Oferece exportação CSV. Não prometas e-mail; devolve ficheiro no chat.
"""

# ---------------------------------------------------------------------------------------
# Catálogo interno (memória longa de produtos)
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
        # não força domínio se não for IG
        return u or ""
    path = re.sub(r"/(products?|produtos?)\/", "/", p.path, flags=re.I)
    path = re.sub(r"//+", "/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    p = p._replace(scheme="https", netloc=IG_HOST, path=path)
    return urlunparse(p)

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = (s.replace("ã","a").replace("õ","o").replace("á","a").replace("à","a").replace("â","a")
           .replace("é","e").replace("ê","e").replace("í","i").replace("ó","o").replace("ô","o")
           .replace("ú","u").replace("ç","c"))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Estrutura do catálogo:
# {
#   "by_ref": { "BS.SM": {ref, name, aliases[], url, price_gross, source}, ... },
#   "by_norm_name": { "cadeirao monsaraz": [ref1, ref2], ... }
# }
CATALOG: Dict[str, Dict] = {"by_ref": {}, "by_norm_name": {}}

def catalog_index_item(item: dict):
    ref = (item.get("ref") or "").strip()
    name = (item.get("name") or item.get("nome") or "").strip()
    url  = _canon_ig_url(item.get("url") or "")
    price_gross = item.get("price_gross")  # preço com IVA
    aliases = item.get("aliases") or []
    source = item.get("source") or ("SITE" if (url and IG_HOST in (urlparse(url).netloc or "")) else "CATALOGO_EXTERNO")

    if not ref and not name:
        return
    entry = {
        "ref": ref, "name": name, "aliases": aliases, "url": url,
        "price_gross": price_gross, "source": source
    }
    if ref:
        CATALOG["by_ref"][ref] = entry
    # index por nome e aliases
    keys = set([_norm(name)]) | {_norm(a) for a in aliases if a}
    for k in keys:
        if not k: continue
        CATALOG["by_norm_name"].setdefault(k, [])
        if ref and ref not in CATALOG["by_norm_name"][k]:
            CATALOG["by_norm_name"][k].append(ref)

def catalog_find_candidates(term: str, topn: int = 5) -> List[dict]:
    term = (term or "").strip()
    if not term:
        return []
    # 1) por REF exata
    if term in CATALOG["by_ref"]:
        return [CATALOG["by_ref"][term]]

    nterm = _norm(term)
    # 2) por nome normalizado
    if nterm in CATALOG["by_norm_name"]:
        return [CATALOG["by_ref"][r] for r in CATALOG["by_norm_name"][nterm]]

    # 3) fuzzy por chaves by_norm_name
    keys = list(CATALOG["by_norm_name"].keys())
    for best in difflib.get_close_matches(nterm, keys, n=topn, cutoff=0.78):
        refs = CATALOG["by_norm_name"].get(best, [])
        for r in refs[:topn]:
            yield CATALOG["by_ref"][r]

def resolve_products_from_text(text: str, limit_per_term: int = 5) -> List[dict]:
    """
    Extrai candidatos de:
      - Tabelas Markdown (| REF | DESIGNAÇÃO | ...)
      - Linhas com "REF" e "nome"
      - Referências tipo "BS.X", "M24", "COD123", "BS.N.CA20."
      - Nomes/aliases por frases/linhas
    Matching por REF exata, nome normalizado e fuzzy.
    """
    if not text:
        return []
    results = []
    seen_keys = set()

    # 1) Refs explícitas
    refs = re.findall(r"\b[A-Z]{1,4}(?:\.[A-Z0-9]{1,6})+\b|\b[A-Z]{1,4}\d{1,5}\b", text)

    # 2) Tabelas/linhas
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    candidates = []
    candidates += refs
    for l in lines:
        cells = [c.strip() for c in l.split("|") if c.strip()]
        if len(cells) >= 2:
            candidates.extend(cells[:2])
        else:
            parts = re.split(r"[,;]| e | com | para | de ", l, flags=re.I)
            candidates.extend([p.strip() for p in parts if p and len(p.strip()) >= 3])

    # 3) Pesquisa direta (REF e nomes)
    for t in candidates:
        if not t or len(t) < 2:
            continue
        if t in CATALOG["by_ref"]:
            entry = CATALOG["by_ref"][t]
            key = entry["ref"] or entry["name"]
            if key not in seen_keys:
                seen_keys.add(key); results.append(entry); continue

        nterm = _norm(t)
        if nterm in CATALOG["by_norm_name"]:
            for r in CATALOG["by_norm_name"][nterm][:limit_per_term]:
                entry = CATALOG["by_ref"][r]
                key = entry["ref"] or entry["name"]
                if key not in seen_keys:
                    seen_keys.add(key); results.append(entry)

    # 4) Fuzzy
    keys = list(CATALOG["by_norm_name"].keys())
    for t in candidates:
        nterm = _norm(t)
        for best in difflib.get_close_matches(nterm, keys, n=limit_per_term, cutoff=0.78):
            refs = CATALOG["by_norm_name"].get(best, [])
            for r in refs[:limit_per_term]:
                entry = CATALOG["by_ref"][r]
                key = entry["ref"] or entry["name"]
                if key not in seen_keys:
                    seen_keys.add(key); results.append(entry)

    return results

# ---------------------------------------------------------------------------------------
# RAG (qdrant + openai embeddings) — usa rag_client.py (inalterado)
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

# ---------------------------------------------------------------------------------------
# Mem0 (curto prazo) — opcional
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

# ---------------------------------------------------------------------------------------
# Fallback local (se Mem0 off) — curto prazo + FACTs + memória de refs
# ---------------------------------------------------------------------------------------
LOCAL_FACTS: Dict[str, Dict[str, str]] = {}
LOCAL_HISTORY: Dict[str, List[Tuple[str, str]]] = {}
LAST_PRODUCTS: Dict[str, List[str]] = {}  # user_id -> [refs]

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

def remember_products(user_id: str, refs: List[str], cap: int = 30):
    if not refs: return
    cur = LAST_PRODUCTS.get(user_id, [])
    new = [r for r in refs if r and r not in cur]
    LAST_PRODUCTS[user_id] = (cur + new)[-cap:]
    try:
        _mem0_create(content=f"{FACT_PREFIX}last_refs=" + ";".join(LAST_PRODUCTS[user_id]), user_id=user_id,
                     metadata={"source": "alma-server", "type": "fact", "key": "last_refs"})
    except Exception:
        pass

def recall_products(user_id: str) -> List[str]:
    facts = mem0_get_facts(user_id)
    if "last_refs" in facts:
        return [x.strip() for x in facts["last_refs"].split(";") if x.strip()]
    return LAST_PRODUCTS.get(user_id, [])

# ---------------------------------------------------------------------------------------
# Helpers Mem0 compat
# ---------------------------------------------------------------------------------------
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
# Memória Contextual (FACTs)
# ---------------------------------------------------------------------------------------
FACT_PREFIX = "FACT|"

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

def mem0_get_facts(user_id: str, limit: int = 20) -> Dict[str, str]:
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
# DETETORES
# ---------------------------------------------------------------------------------------
def _is_budget_request(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    keys = [
        # PT
        "orçamento", "orcamento", "cotação", "cotacao", "proposta", "preço total", "quanto fica",
        "fazer orçamento", "encomenda", "proforma", "fatura proforma", "tabela de preços", "preventivo",
        # formatos/ferramentas
        "excel", "folha de cálculo", "folha de calculo", "planilha", ".csv", "csv", "orçamento em excel",
        # ES
        "presupuesto", "cotización", "cotizacion",
        # gatilhos de tabela/linhas
        "ref.", "designação", "descrição", "quant.", "preço uni."
    ]
    return any(k in t for k in keys)

# ---------------------------------------------------------------------------------------
# Linkificação de URLs IG
# ---------------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------------
# Consolidação de PRODUTOS_RECONHECIDOS para o LLM (modo orçamento)
# ---------------------------------------------------------------------------------------
def build_catalog_hints_block(question: str, user_id: Optional[str] = None) -> str:
    found = resolve_products_from_text(question)
    # fallback: usar memória recente de produtos, se nada reconhecido no texto
    if (not found) and user_id:
        for r in recall_products(user_id):
            if r in CATALOG["by_ref"]:
                found.append(CATALOG["by_ref"][r])

    if not found:
        return ""

    # memoriza refs agora detetadas
    if user_id:
        remember_products(user_id, [it.get("ref") for it in found if it.get("ref")])

    lines = ["PRODUTOS_RECONHECIDOS (usa só estes; não inventes):"]
    for it in found:
        ref = it.get("ref") or "-"
        name = it.get("name") or "-"
        url  = _canon_ig_url(it.get("url") or "")
        price = it.get("price_gross")  # com IVA
        source = it.get("source") or "SITE"
        line = f"- REF={ref}; NOME={name}; PRECO_COM_IVA={price if price is not None else 'N/D'}; SOURCE={source}"
        if url:
            line += f"; URL={url}"
        else:
            line += "; URL="  # vazio, para forçar o LLM a não inventar
        lines.append(line)
    return "\n".join(lines)

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
        "message": "Alma server ativo. Use POST /ask (Grok+Mem0+RAG) ou POST /say (D-ID).",
        "mem0": {"enabled": MEM0_ENABLE, "client_ready": bool(mem0_client)},
        "rag": {"available": RAG_READY, "top_k": RAG_TOP_K, "namespace": DEFAULT_NAMESPACE},
        "endpoints": {
            "health": "/health",
            "ask": "POST /ask {question, user_id?, namespace?}",
            "ask_get": "/ask_get?q=...&user_id=...&namespace=...",
            "ping_grok": "/ping_grok",
            "say": "POST /say {text, image_url?, voice_id?}",
            "heygen_token": "POST /heygen/token",
            "mem_facts": "/mem/facts?user_id=...",
            "mem_search": "/mem/search?user_id=...&q=...",
            "rag_search_get": "/rag/search?q=...&namespace=...",
            "rag_crawl": "POST /rag/crawl",
            "rag_ingest_sitemap": "POST /rag/ingest-sitemap",
            "rag_ingest_url": "POST /rag/ingest-url",
            "rag_ingest_text": "POST /rag/ingest-text",
            "rag_ingest_pdf_url": "POST /rag/ingest-pdf-url",
            "rag_search_post": "POST /rag/search {query, namespace?, top_k?}",
            "catalog_load": "POST /catalog/load",
            "catalog_status": "GET /catalog/status",
            "catalog_save": "POST /catalog/save",
            "catalog_load_file": "POST /catalog/load-file",
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
        "rag_default_namespace": DEFAULT_NAMESPACE,
        "catalog_size": len(CATALOG["by_ref"])
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
# Catálogo: carregar / guardar / status
# ---------------------------------------------------------------------------------------
@app.post("/catalog/load")
async def catalog_load(request: Request):
    """
    Body:
    {
      "items":[
        {
          "ref":"BS.SM",
          "name":"Banco Três Patas",
          "aliases":["banco de tres patas"],
          "url":"https://interiorguider.com/banco-tres-patas",
          "price_gross":219.0,   # preço COM IVA (site)
          "source":"SITE" | "CATALOGO_EXTERNO"
        },
        ...
      ]
    }
    """
    data = await request.json()
    items = data.get("items") or []
    count = 0
    for it in items:
        try:
            catalog_index_item(it)
            count += 1
        except Exception as e:
            log.warning(f"[catalog] item ignorado: {e}")
    return {"ok": True, "loaded": count, "total": len(CATALOG["by_ref"])}

@app.get("/catalog/status")
def catalog_status():
    return {
        "ok": True,
        "by_ref": len(CATALOG["by_ref"]),
        "by_norm_name": len(CATALOG["by_norm_name"]),
        "examples": list(list(CATALOG["by_ref"].values())[:3])
    }

@app.post("/catalog/save")
async def catalog_save(request: Request):
    data = await request.json()
    path = (data.get("path") or "/tmp/catalog.json").strip()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(CATALOG, f, ensure_ascii=False)
    return {"ok": True, "saved_to": path}

@app.post("/catalog/load-file")
async def catalog_load_file(request: Request):
    data = await request.json()
    path = (data.get("path") or "/tmp/catalog.json").strip()
    if not os.path.exists(path):
        return {"ok": False, "error": f"ficheiro não encontrado: {path}"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        by_ref = obj.get("by_ref") or {}
        count = 0
        for _, it in by_ref.items():
            catalog_index_item(it); count += 1
        return {"ok": True, "loaded": count, "total": len(CATALOG["by_ref"])}
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
# 🔗 Pipeline Alma: Mem0 → (catalog hints) → RAG → Grok
# ---------------------------------------------------------------------------------------
def build_messages_with_memory_and_rag(
    user_id: str,
    question: str,
    namespace: Optional[str]
):
    # 0) FACTs
    new_facts = extract_contextual_facts_pt(question)
    for k, v in new_facts.items():
        mem0_set_fact(user_id, k, v)

    # 1) Perfil contextual
    facts = mem0_get_facts(user_id)
    facts_block = facts_to_context_block(facts)

    # 2) Curto prazo
    short_snippets = _mem0_search(question, user_id=user_id, limit=5) or local_search_snippets(user_id, limit=5)
    memory_block = "Memórias recentes do utilizador (curto prazo):\n" + "\n".join(f"- {s}" for s in short_snippets[:3]) if short_snippets else ""

    # 3) RAG
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

    # 4) Mensagens base
    messages = [{"role": "system", "content": ALMA_MISSION}]

    # 4a) Modo Orçamentos → injetar catálogo reconhecido + regras estritas
    if _is_budget_request(question):
        cat_block = build_catalog_hints_block(question, user_id=user_id)
        if cat_block:
            messages.append({"role": "system", "content": cat_block})
        messages.append({"role": "system", "content": ALMA_ORCAMENTO_PROMPT})

    if facts_block:
        messages.append({"role": "system", "content": facts_block})
    if rag_block:
        messages.append({"role": "system", "content": f"Conhecimento corporativo (RAG):\n{rag_block}"})
    if memory_block:
        messages.append({"role": "system", "content": memory_block})
    messages.append({"role": "user", "content": question})

    return messages, new_facts, facts, rag_used

# ---------------------------------------------------------------------------------------
# ❌ Fast-path DESLIGADO — rotas usam SEMPRE o pipeline completo
# ---------------------------------------------------------------------------------------
@app.get("/ask_get")
def ask_get(q: str = "", user_id: str = "anon", namespace: str = None):
    if not q:
        return {"answer": "Falta query param ?q="}

    messages, new_facts, facts, rag_used = build_messages_with_memory_and_rag(user_id, q, namespace)
    try:
        answer = grok_chat(messages)
    except Exception as e:
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

    answer = _postprocess_answer(answer)

    local_append_dialog(user_id, q, answer)
    _mem0_create(content=f"User: {q}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
    _mem0_create(content=f"Alma: {answer}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
    return {
        "answer": answer,
        "mem0": {"facts_used": bool(facts), "facts": facts},
        "new_facts_detected": new_facts,
        "rag": {"used": rag_used, "top_k": RAG_TOP_K, "namespace": namespace or DEFAULT_NAMESPACE}
    }

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    user_id = (data.get("user_id") or "").strip() or "anon"
    namespace = (data.get("namespace") or "").strip() or None
    log.info(f"[/ask] user_id={user_id} ns={namespace or DEFAULT_NAMESPACE} question={question!r}")

    if not question:
        return {"answer": "Coloca a tua pergunta em 'question'."}

    messages, new_facts, facts, rag_used = build_messages_with_memory_and_rag(user_id, question, namespace)

    try:
        answer = grok_chat(messages)
    except Exception as e:
        log.exception("Erro ao chamar a x.ai")
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

    answer = _postprocess_answer(answer)

    local_append_dialog(user_id, question, answer)
    _mem0_create(content=f"User: {question}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
    _mem0_create(content=f"Alma: {answer}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})

    return {
        "answer": answer,
        "mem0": {"facts_used": bool(facts), "facts": facts},
        "new_facts_detected": new_facts,
        "rag": {"used": rag_used, "top_k": RAG_TOP_K, "namespace": namespace or DEFAULT_NAMESPACE}
    }

# ---------------------------------------------------------------------------------------
# D-ID: Texto → Vídeo (lábios)
# ---------------------------------------------------------------------------------------
DID_API_KEY = os.getenv("DID_API_KEY", "").strip()
DEFAULT_IMAGE_URL = os.getenv("DEFAULT_IMAGE_URL", "").strip()
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "pt-PT-FernandaNeural").strip()
DID_BASE = "https://api.d-id.com"

def did_headers():
    h = {"Content-Type": "application/json"}
    if DID_API_KEY:
        h["Authorization"] = f"Basic {DID_API_KEY}"
        h["x-api-key"] = DID_API_KEY
    return h

@app.post("/say")
async def say(request: Request):
    if not DID_API_KEY:
        return {"error": "Falta DID_API_KEY nas Variables do Railway"}
    data = await request.json()
    text = (data.get("text") or "").strip()
    image_url = (data.get("image_url") or DEFAULT_IMAGE_URL).strip()
    voice_id = (data.get("voice_id") or DEFAULT_VOICE).strip()
    if not text:
        return {"error": "Campo 'text' é obrigatório"}
    if not image_url:
        return {"error": "Falta 'image_url' (define DEFAULT_IMAGE_URL nas Variables)"}
    payload = {
        "script": {"type": "text", "input": text, "provider": {"type": "microsoft", "voice_id": voice_id}},
        "source_url": image_url,
    }
    try:
        r = requests.post(f"{DID_BASE}/talks", headers=did_headers(), json=payload, timeout=30)
        log.info(f"[d-id] create talks -> {r.status_code} {r.text[:200]}")
        r.raise_for_status()
    except Exception as e:
        return {"error": f"Falha a criar talk: {e}"}
    talk = r.json()
    talk_id = talk.get("id")
    if not talk_id:
        return {"error": "Sem id do talk", "raw": talk}
    result_url = None
    for _ in range(30):
        time.sleep(1)
        g = requests.get(f"{DID_BASE}/talks/{talk_id}", headers=did_headers(), timeout=15)
        if not g.ok:
            continue
        j = g.json()
        status = j.get("status")
        result_url = j.get("result_url")
        if status == "error":
            return {"error": "D-ID devolveu erro", "details": j}
        if result_url:
            break
    if not result_url:
        return {"error": "Timeout à espera do result_url"}
    return {"video_url": result_url}

# ---------------------------------------------------------------------------------------
# HeyGen token demo (mantido para compatibilidade)
# ---------------------------------------------------------------------------------------
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY", "").strip()

@app.post("/heygen/token")
def heygen_token():
    if not HEYGEN_API_KEY:
        return {"error": "Falta HEYGEN_API_KEY"}
    try:
        res = requests.post(
            "https://api.heygen.com/v1/realtime/session",
            headers={"Authorization": f"Bearer {HEYGEN_API_KEY}", "Content-Type": "application/json"},
            json={"avatar_id": "ebc94c0e88534d078cf8788a01f3fba9","voice_id": "ff5719e3a6314ecea47badcbb1c0ffaa","language": "pt-PT"},
            timeout=15
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------------------------
# RAG Endpoints (crawl, sitemap, url, text, pdf, search POST) — inalterados funcionalmente
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
# € — parsing robusto + CSV “PT-friendly” (Excel)
# ---------------------------------------------------------------------------------------
def _parse_money_eu(s: str) -> Optional[float]:
    """
    Aceita: "1.234,56", "1,234.56", "1234,56", "1234.56", "€1 234,56", "1 234,56 €", "1,000€"
    Remove espaços/€; deteta decimal pelo último separador.
    """
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

def _safe_float(v, default=0.0):
    if isinstance(v, (int, float)):
        return float(v)
    f = _parse_money_eu(v)
    return f if f is not None else float(default)

# ---------------------------------------------------------------------------------------
# Exportação CSV de Orçamentos (público: C/IVA; pro: S/IVA por omissão)
# ---------------------------------------------------------------------------------------
@app.post("/budget/csv")
async def budget_csv(request: Request):
    """
    Body:
    {
      "mode": "public" | "pro",
      "iva_pct": 23,
      "force_total_ci": false,         # opcional (apenas útil em 'pro')
      "delimiter": ";",                # ";", "," (default ";")
      "decimal": "comma",              # "comma" | "dot" (default "comma")
      "rows": [
        {"ref":"BS.01","descricao":"...","quant":1,"preco_uni":"1.000,00","desc_pct":"5","dim":"...","material":"...","marca":"...","link":"https://interiorguider.com/..."}
      ]
    }
    """
    data = await request.json()
    mode = (data.get("mode") or "public").lower().strip()
    iva_pct = _safe_float(data.get("iva_pct", 23.0))
    rows = data.get("rows") or []
    force_total_ci = bool(data.get("force_total_ci", False))
    delimiter = (data.get("delimiter") or ";").strip() or ";"
    decimal = (data.get("decimal") or "comma").strip().lower()

    if mode not in ("public", "pro"):
        return PlainTextResponse("mode deve ser 'public' ou 'pro'", status_code=400)
    if not isinstance(rows, list) or not rows:
        return PlainTextResponse("rows vazio", status_code=400)

    def fmt_money(x: float) -> str:
        s = f"{x:.2f}"
        return s.replace(".", ",") if decimal == "comma" else s

    if mode == "public":
        headers = ["REF.", "DESIGNAÇÃO / MATERIAL / ACABAMENTO / COR", "QUANT.", "PREÇO UNI.", "DESC.", "TOTAL C/IVA"]
        show_total_ci = True
    else:
        headers = ["REF.", "DESIGNAÇÃO / MATERIAL / ACABAMENTO / COR", "QUANT.", "PREÇO UNI.", "DESC.", "TOTAL S/IVA" if not force_total_ci else "TOTAL C/IVA"]
        show_total_ci = bool(force_total_ci)

    sio = StringIO()
    writer = csv.writer(sio, delimiter=delimiter)
    writer.writerow(headers)

    for r in rows:
        ref = (r.get("ref") or "").strip()
        quant = int(_safe_float(r.get("quant"), 1))
        preco_uni = _safe_float(r.get("preco_uni"), 0.0)   # aceita "1.000,00" e variantes
        desc_pct = _safe_float(r.get("desc_pct"), 0.0)

        desc_main = (r.get("descricao") or "").strip() or "Produto"
        extra_lines = []
        if r.get("dim"): extra_lines.append(f"Dimensões: {r['dim']}")
        if r.get("material"): extra_lines.append(f"Material/Acabamento: {r['material']}")
        if r.get("marca"): extra_lines.append(f"Marca: {r['marca']}")
        if r.get("link"):
            link = _canon_ig_url(str(r["link"]).strip())
            extra_lines.append(f"Link: {link}")

        full_desc = desc_main + (("\n" + "\n".join(extra_lines)) if extra_lines else "")

        total_si = quant * preco_uni * (1.0 - desc_pct/100.0)
        total = total_si * (1.0 + iva_pct/100.0) if show_total_ci else total_si

        writer.writerow([
            ref,
            full_desc,
            str(quant),
            fmt_money(preco_uni),
            (f"{desc_pct:.0f}%" if desc_pct else ""),
            fmt_money(total)
        ])

    csv_bytes = sio.getvalue().encode("utf-8-sig")
    fname = f"orcamento_{mode}_{int(time.time())}.csv"
    fpath = os.path.join("/tmp", fname)
    with open(fpath, "wb") as f:
        f.write(csv_bytes)

    return FileResponse(fpath, media_type="text/csv", filename=fname)

# ---------------------------------------------------------------------------------------
# Mensagens com memória + RAG + Orçamento
# ---------------------------------------------------------------------------------------
def build_messages_with_memory_and_rag_and_budget(user_id: str, question: str, namespace: Optional[str]):
    return build_messages_with_memory_and_rag(user_id, question, namespace)

# ---------------------------------------------------------------------------------------
# Local run
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
