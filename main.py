# main.py — Alma Server (Grok + Memória de Curto Prazo + Memória Contextual + RAG/Qdrant)
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
from io import StringIO
from typing import Dict, List, Tuple, Optional

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

APP_VERSION = os.getenv("APP_VERSION", "alma-server/clean-1+context-1+rag-3")

# ---------------------------------------------------------------------------------------
# Prompt nuclear da Alma (mantido como está)
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
# 🔸 Prompt adicional: MODO ORÇAMENTOS
# ---------------------------------------------------------------------------------------
ALMA_ORCAMENTO_PROMPT = """
Quando fores pedido para preparar um orçamento, segue estas regras:

1) Template
- "profissional/revenda/arquitetos/pro/com IVA" → formato PROFISSIONAL (TOTAL C/IVA).
- Caso contrário → PÚBLICO (TOTAL S/IVA).

2) Campos por linha
- REF., DESIGNAÇÃO (inclui materiais/acabamentos), QUANT., PREÇO UNI., DESC.%
- Público: TOTAL S/IVA = QUANT × PREÇO UNI × (1 - DESC%/100)
- Profissional: TOTAL C/IVA = TOTAL S/IVA × (1 + IVA%/100)

3) Só perguntar o ESTRITAMENTE necessário (ex.: quantidade, cor/variante, desconto) para identificar o artigo ou calcular.

4) Links de produto
- Preferir sempre interiorguider.com; não inventar caminhos; evitar "/product/...".

5) Saída no chat
- Mostra apenas uma pré-visualização resumida (sem blocos ``` e sem markdown nas células).
- Se o utilizador pedir CSV, diz para usar o endpoint POST /budget/csv com o JSON atual — não digas “enviei por email”.

6) Estilo
- Conciso, objetivo; termina com uma única próxima ação útil.
"""

# ---------------------------------------------------------------------------------------
# RAG (qdrant + openai embeddings)
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
# Mem0 (curto prazo)
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
# Fallback local (se Mem0 off)
# ---------------------------------------------------------------------------------------
LOCAL_FACTS: Dict[str, Dict[str, str]] = {}
LOCAL_HISTORY: Dict[str, List[Tuple[str, str]]] = {}

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
        "orçamento", "orcamento", "budget", "cotação", "cotacao",
        "proposta", "quote", "preço total", "quanto fica", "fazer orçamento"
    ]
    return any(k in t for k in keys)

# ---------------------------------------------------------------------------------------
# 🔒 SANITIZADOR DE SAÍDA (chat) PARA ORÇAMENTOS
# ---------------------------------------------------------------------------------------
CSV_FENCE_RE = re.compile(r"```.*?```", re.S)
MD_LINK_RE = re.compile(r"$begin:math:display$([^$end:math:display$]+)\]$begin:math:text$(https?://[^)]+)$end:math:text$")

def enforce_budget_output_policies(text: str) -> str:
    """
    - Remove blocos ```...```
    - Converte links markdown para 'Texto — URL'
    - Troca promessas de email por instrução do endpoint
    - Substitui frases tipo “Aqui está o orçamento exportado em formato CSV…” por nota curta
    """
    if not text:
        return text
    t = text

    # 1) remover blocos fenced (``` ... ```)
    if "```" in t:
        t = CSV_FENCE_RE.sub("[Pré-visualização: usa o endpoint /budget/csv para obter o ficheiro.]\n", t)

    # 2) links markdown → texto — URL (evita link “invisível” em markdown)
    t = MD_LINK_RE.sub(r"\1 — \2", t)

    # 3) proibir promessas/envios por email
    t = re.sub(r"(?i)(vou|posso|irei|acabei de|já) (enviar|mandar).{0,20}e-?mail", 
               "posso gerar o ficheiro CSV via POST /budget/csv", t)

    # 4) frases de “aqui está o csv”
    t = re.sub(r"(?i)aqui está o orçamento.*?csv.*?:?", 
               "Pré-visualização pronta. Para o ficheiro, usa POST /budget/csv.", t)

    return t

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
            "budget_preview": "POST /budget/preview",
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

# ---------------------------------------------------------------------------------------
# Alma Chat (modo consola full)
# ---------------------------------------------------------------------------------------
@app.get("/alma-chat", response_class=HTMLResponse)
def serve_alma_chat():
    try:
        with open("alma-chat.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return HTMLResponse("<h1>alma-chat.html não encontrado</h1>", status_code=404)

# ---------------------------------------------------------------------------------------
# 🔗 Pipeline Alma: Mem0 → RAG → Grok
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

    # 4) Mensagens
    messages = [{
        "role": "system",
        "content": ALMA_MISSION
    }]

    # Modo Orçamentos (só se pedido)
    if _is_budget_request(question):
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

    # 🔒 Sanitização de saída para pedidos de orçamento
    if _is_budget_request(q):
        answer = enforce_budget_output_policies(answer)

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

    # 🔒 Sanitização de saída para pedidos de orçamento
    if _is_budget_request(question):
        answer = enforce_budget_output_policies(answer)

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
# HeyGen token demo
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

# --- Proxy: extrair URLs (sitemap.xml, HTML ou texto colado)
import xml.etree.ElementTree as ET

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
# 🔸 Pré-visualização (chat) — sem ficheiro
# ---------------------------------------------------------------------------------------
@app.post("/budget/preview")
async def budget_preview(request: Request):
    data = await request.json()
    mode = (data.get("mode") or "public").lower().strip()
    iva_pct = float(data.get("iva_pct", 23))
    rows = data.get("rows") or []
    if mode not in ("public", "pro"):
        return PlainTextResponse("mode deve ser 'public' ou 'pro'", status_code=400)
    if not rows:
        return PlainTextResponse("rows vazio", status_code=400)

    out = []
    for r in rows:
        quant = int(r.get("quant") or 1)
        preco = float(r.get("preco_uni") or 0)
        desc  = float(r.get("desc_pct") or 0)
        total_si = quant * preco * (1 - desc/100)
        total = total_si if mode == "public" else total_si * (1 + iva_pct/100.0)
        out.append({
            "ref": r.get("ref") or "",
            "descricao": r.get("descricao") or "",
            "quant": quant,
            "preco_uni": round(preco, 2),
            "desc_pct": round(desc, 2) if desc else 0,
            "total": round(total, 2),
            "link": r.get("link") or ""
        })
    return JSONResponse({"mode": mode, "iva_pct": iva_pct, "items": out})

# ---------------------------------------------------------------------------------------
# 🔸 Exportação CSV de Orçamentos
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
    data = await request.json()
    mode = (data.get("mode") or "public").lower().strip()
    iva_pct = _safe_float(data.get("iva_pct", 23.0))
    rows = data.get("rows") or []

    if mode not in ("public", "pro"):
        return PlainTextResponse("mode deve ser 'public' ou 'pro'", status_code=400)
    if not isinstance(rows, list) or not rows:
        return PlainTextResponse("rows vazio", status_code=400)

    if mode == "public":
        headers = ["REF.", "DESIGNAÇÃO / MATERIAL / ACABAMENTO / COR", "QUANT.", "PREÇO UNI.", "DESC.", "TOTAL S/IVA"]
    else:
        headers = ["REF.", "DESIGNAÇÃO / MATERIAL / ACABAMENTO / COR", "QUANT.", "PREÇO UNI.", "DESC.", "TOTAL C/IVA"]

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
        if mode == "public":
            total_col = _format_money(total_si)
        else:
            total_ci = total_si * (1.0 + iva_pct/100.0)
            total_col = _format_money(total_ci)

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

    return FileResponse(
        fpath,
        media_type="text/csv",
        filename=fname
    )

# ---------------------------------------------------------------------------------------
# Local run
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
