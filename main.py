# main.py — Alma Server (clean-1 + Memória Contextual + RAG)
# Grok (x.ai) + Mem0 (curto prazo) + D-ID + HeyGen + RAG/Qdrant
# -----------------------------------------------------------------------------
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os
import requests
import logging
import uvicorn
import time
import re
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------------
# FastAPI & CORS (CRIA A APP PRIMEIRO!)
# -----------------------------------------------------------------------------
app = FastAPI(title="Alma Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Config geral / Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("alma")

APP_VERSION = os.getenv("APP_VERSION", "alma-server/clean-1+context-1+rag-2")

# -----------------------------------------------------------------------------
# RAG: imports e preparação (usa rag_client.py do projeto)
# -----------------------------------------------------------------------------
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

# Estes 3 são usados noutras rotas (GET /rag/search, /ask, /ask_get)
RAG_AVAILABLE = RAG_READY
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))
RAG_CONTEXT_TOKEN_BUDGET = int(os.getenv("RAG_CONTEXT_TOKEN_BUDGET", "1600"))
DEFAULT_NAMESPACE = os.getenv("RAG_DEFAULT_NAMESPACE", "").strip() or None  # ex: "boasafra"

# -----------------------------------------------------------------------------
# Mem0 (curto prazo) — tenta importar como 'mem0ai' OU 'mem0'
# -----------------------------------------------------------------------------
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
                import mem0ai as _mem0_pkg   # PyPI: mem0ai
                from mem0ai import MemoryClient as _MC
                pkg_name = "mem0ai"
            except Exception:
                import mem0 as _mem0_pkg     # PyPI: mem0
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

# -----------------------------------------------------------------------------
# Fallback local (se Mem0 off ou falhar) — curto prazo + FACTs
# -----------------------------------------------------------------------------
# NOTA: isto não é persistente; apenas vive no processo.
LOCAL_FACTS: Dict[str, Dict[str, str]] = {}            # user_id -> {key: value}
LOCAL_HISTORY: Dict[str, List[Tuple[str, str]]] = {}   # user_id -> [(question, answer), ...]

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
    for q, a in reversed(items[-limit*2:]):  # heuristic
        if len(out) >= limit:
            break
        out.append(f"User: {q}")
        if len(out) >= limit:
            break
        out.append(f"Alma: {a}")
    return out[:limit]

# -----------------------------------------------------------------------------
# Config Grok (x.ai)
# -----------------------------------------------------------------------------
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = os.getenv("XAI_MODEL", "grok-4-0709")

def grok_chat(messages, timeout=30):
    if not XAI_API_KEY:
        raise RuntimeError("Falta XAI_API_KEY")
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": messages}
    r = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=timeout)
    log.info(f"[x.ai] status={r.status_code} body={r.text[:300]}")
    r.raise_for_status()
    return r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or ""

# -----------------------------------------------------------------------------
# Helpers: Mem0 curto prazo (search/create)
# -----------------------------------------------------------------------------
def mem0_search(user_id: str, query: str, limit: int = 5) -> List[str]:
    """Busca memórias relevantes (curto prazo) do utilizador."""
    if not (MEM0_ENABLE and mem0_client):
        return local_search_snippets(user_id, limit=limit)
    try:
        results = mem0_client.memories.search(query=query or "contexto", user_id=user_id, limit=limit)
        snippets = []
        for item in results or []:
            val = (item.get("text") or item.get("memory") or item.get("content") or "").strip()
            if val:
                snippets.append(val)
        log.info(f"[mem0] search user_id={user_id} found={len(snippets)}")
        return snippets
    except Exception as e:
        log.warning(f"[mem0] search falhou: {e}")
        return local_search_snippets(user_id, limit=limit)

def mem0_append_dialog(user_id: str, question: str, answer: str):
    """Guarda a interação atual como memórias (user/assistant)."""
    local_append_dialog(user_id, question, answer)
    if not (MEM0_ENABLE and mem0_client):
        return
    try:
        mem0_client.memories.create(
            content=f"User: {question}",
            user_id=user_id,
            metadata={"source": "alma-server", "type": "dialog"}
        )
        mem0_client.memories.create(
            content=f"Alma: {answer}",
            user_id=user_id,
            metadata={"source": "alma-server", "type": "dialog"}
        )
        log.info(f"[mem0] create ok user_id={user_id}")
    except Exception as e:
        log.warning(f"[mem0] create falhou: {e}")

# -----------------------------------------------------------------------------
# Memória Contextual (FACTs) — deteção, guardar e recuperar
# -----------------------------------------------------------------------------
FACT_PREFIX = "FACT|"

# Regras simples para PT-PT: nome, localização, preferências/estilo, divisão/projeto
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
    r"\bquero\s+([^\.]{3,80})",  # intenção
]
ROOM_KEYWORDS = ["sala", "cozinha", "quarto", "wc", "casa de banho", "varanda", "escritório", "hall", "entrada", "lavandaria"]

def extract_contextual_facts_pt(text: str) -> Dict[str, str]:
    """Extrai factos simples do texto em pt-PT."""
    facts: Dict[str, str] = {}
    t = " " + text.strip() + " "
    # nome
    for pat in NAME_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if len(name.split()) == 1 and name.lower() in {"melhor", "pior", "arquiteto", "cliente"}:
                pass
            else:
                facts["name"] = name
                break
    # cidade/local
    for pat in CITY_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            city = m.group(1).strip(" .,'\"").title()
            if 2 <= len(city) <= 60:
                facts["location"] = city
                break
    # preferências/estilo
    for pat in PREF_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            pref = m.group(1).strip(" .,'\"")
            if 3 <= len(pref) <= 80:
                facts.setdefault("preferences", pref)
                break
    # divisão/projeto (tags simples)
    for kw in ROOM_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", t, flags=re.IGNORECASE):
            facts["room"] = kw
    return facts

def mem0_set_fact(user_id: str, key: str, value: str):
    """Guarda/atualiza um FACT (perfil)."""
    local_set_fact(user_id, key, value)
    if not (MEM0_ENABLE and mem0_client):
        return
    try:
        mem0_client.memories.create(
            content=f"{FACT_PREFIX}{key}={value}",
            user_id=user_id,
            metadata={"source": "alma-server", "type": "fact", "key": key}
        )
        log.info(f"[mem0] fact create ok user_id={user_id} {key}={value}")
    except Exception as e:
        log.warning(f"[mem0] fact create falhou: {e}")

def mem0_get_facts(user_id: str, limit: int = 20) -> Dict[str, str]:
    """Recupera FACTs do utilizador a partir do Mem0. Se falhar/disabled, usa local."""
    facts = local_get_facts(user_id)
    if not (MEM0_ENABLE and mem0_client):
        return facts
    try:
        results = mem0_client.memories.search(query=FACT_PREFIX, user_id=user_id, limit=limit)
        for item in results or []:
            content = (item.get("text") or item.get("memory") or item.get("content") or "").strip()
            if content.startswith(FACT_PREFIX):
                body = content[len(FACT_PREFIX):]
                if "=" in body:
                    k, v = body.split("=", 1)
                    if k and v:
                        facts[k.strip()] = v.strip()
        return facts
    except Exception as e:
        log.warning(f"[mem0] get_facts falhou: {e}")
        return facts

def facts_to_context_block(facts: Dict[str, str]) -> str:
    if not facts:
        return ""
    lines = []
    if "name" in facts:
        lines.append(f"- Nome: {facts['name']}")
    if "location" in facts:
        lines.append(f"- Localização: {facts['location']}")
    if "room" in facts:
        lines.append(f"- Divisão/Projeto: {facts['room']}")
    if "preferences" in facts:
        lines.append(f"- Preferências: {facts['preferences']}")
    for k, v in facts.items():
        if k not in {"name", "location", "room", "preferences"}:
            lines.append(f"- {k}: {v}")
    return "Perfil do utilizador (memória contextual):\n" + "\n".join(lines)

# -----------------------------------------------------------------------------
# Config D-ID (texto→vídeo de lábios)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Config HeyGen (token demo)
# -----------------------------------------------------------------------------
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY", "").strip()

# -----------------------------------------------------------------------------
# ROTAS BÁSICAS / STATUS
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "message": "Alma server ativo. Use POST /ask (Grok+Mem0+RAG) ou POST /say (D-ID).",
        "mem0": {"enabled": MEM0_ENABLE, "client_ready": bool(mem0_client)},
        "rag": {"available": RAG_AVAILABLE, "top_k": RAG_TOP_K, "namespace": DEFAULT_NAMESPACE},
        "endpoints": {
            "health": "/health",
            "ask": "POST /ask {question, user_id?}",
            "ask_get": "/ask_get?q=...&user_id=...",
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
            "rag_search_post": "POST /rag/search {query, namespace?, top_k?}"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mem0_enabled": MEM0_ENABLE,
        "mem0_client_ready": bool(mem0_client),
        "model": MODEL,
        "rag_available": RAG_AVAILABLE
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

# -----------------------------------------------------------------------------
# Memória contextual (FACTs) e Mem0 debug
# -----------------------------------------------------------------------------
@app.get("/mem/facts")
def mem_facts(user_id: str = "anon"):
    facts = mem0_get_facts(user_id=user_id, limit=50)
    return {"user_id": user_id, "facts": facts}

@app.get("/mem/search")
def mem_search_route(q: str = "", user_id: str = "anon"):
    if not q:
        return {"user_id": user_id, "found": 0, "snippets": []}
    snippets = mem0_search(user_id=user_id, query=q, limit=10)
    return {"user_id": user_id, "found": len(snippets), "snippets": snippets}

# -----------------------------------------------------------------------------
# RAG: GET /rag/search (debug/inspeção rápida)
# -----------------------------------------------------------------------------
@app.get("/rag/search")
def rag_search_get(q: str, namespace: str = None, top_k: int = None):
    if not RAG_AVAILABLE:
        return {"ok": False, "error": "rag_client indisponível no servidor"}
    try:
        res = search_chunks(query=q, namespace=namespace or DEFAULT_NAMESPACE, top_k=top_k or RAG_TOP_K)
        return {"ok": True, "query": q, "matches": res}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# -----------------------------------------------------------------------------
# Console HTML (UI para ingestão/testes RAG)
# -----------------------------------------------------------------------------
@app.get("/console", response_class=HTMLResponse)
def serve_console():
    try:
        with open("console.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return HTMLResponse("<h1>console.html não encontrado</h1>", status_code=404)

# -----------------------------------------------------------------------------
# IA: Pergunta → Resposta (Grok-4) + Mem0 curto prazo + Memória Contextual + RAG
# -----------------------------------------------------------------------------
@app.get("/ask_get")
def ask_get(q: str = "", user_id: str = "anon"):
    if not q:
        return {"answer": "Falta query param ?q="}

    # 0) Deteção e armazenamento de FACTs (a partir da pergunta atual)
    new_facts = extract_contextual_facts_pt(q)
    for k, v in new_facts.items():
        mem0_set_fact(user_id, k, v)

    # 1) Carregar FACTs existentes (perfil contextual)
    facts = mem0_get_facts(user_id)
    facts_block = facts_to_context_block(facts)

    # 2) Contexto de curto prazo (diálogo recente)
    snippets = mem0_search(user_id=user_id, query=q, limit=5)
    memory_block = ""
    if snippets:
        memory_block = "Memórias recentes do utilizador (curto prazo):\n" + "\n".join(f"- {s}" for s in snippets[:3])

    # 2.5) RAG: conhecimento externo (docs internos no Qdrant)
    rag_block = ""
    if RAG_AVAILABLE:
        try:
            rag_hits = search_chunks(query=q, namespace=DEFAULT_NAMESPACE, top_k=RAG_TOP_K)
        except Exception:
            rag_hits = []
        try:
            rag_block = build_context_block(rag_hits, token_budget=RAG_CONTEXT_TOKEN_BUDGET)
        except Exception:
            rag_block = ""

    # 3) Mensagens para Grok
    messages = [{"role": "system",
                 "content": "És a Alma, especialista em design de interiores (método psicoestético). Responde claro, conciso e em pt-PT."}]
    if facts_block:
        messages.append({"role": "system", "content": facts_block})
    if rag_block:
        messages.append({"role": "system", "content": rag_block})
    if memory_block:
        messages.append({"role": "system", "content": memory_block})
    messages.append({"role": "user", "content": q})

    # 4) Chamada ao Grok
    try:
        answer = grok_chat(messages)
    except Exception as e:
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

    # 5) Guardar diálogo no Mem0 e no fallback local
    mem0_append_dialog(user_id=user_id, question=q, answer=answer)

    return {
        "answer": answer,
        "mem0": {"facts_used": bool(facts_block), "facts": facts, "recent_found": len(snippets)},
        "new_facts_detected": new_facts,
        "rag": {"used": bool(rag_block), "top_k": RAG_TOP_K, "namespace": DEFAULT_NAMESPACE}
    }

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    user_id = (data.get("user_id") or "").strip() or "anon"
    log.info(f"[/ask] user_id={user_id} question={question!r}")

    if not question:
        return {"answer": "Coloca a tua pergunta em 'question'."}

    # 0) FACTs
    new_facts = extract_contextual_facts_pt(question)
    for k, v in new_facts.items():
        mem0_set_fact(user_id, k, v)

    # 1) FACTs existentes
    facts = mem0_get_facts(user_id)
    facts_block = facts_to_context_block(facts)

    # 2) curto prazo
    snippets = mem0_search(user_id=user_id, query=question, limit=5)
    memory_block = ""
    if snippets:
        memory_block = "Memórias recentes do utilizador (curto prazo):\n" + "\n".join(f"- {s}" for s in snippets[:3])

    # 2.5) RAG
    rag_block = ""
    if RAG_AVAILABLE:
        try:
            rag_hits = search_chunks(query=question, namespace=DEFAULT_NAMESPACE, top_k=RAG_TOP_K)
        except Exception:
            rag_hits = []
        try:
            rag_block = build_context_block(rag_hits, token_budget=RAG_CONTEXT_TOKEN_BUDGET)
        except Exception:
            rag_block = ""

    # 3) Mensagens para Grok
    messages = [{"role": "system",
                 "content": "És a Alma, especialista em design de interiores (método psicoestético). Responde claro, conciso e em pt-PT."}]
    if facts_block:
        messages.append({"role": "system", "content": facts_block})
    if rag_block:
        messages.append({"role": "system", "content": rag_block})
    if memory_block:
        messages.append({"role": "system", "content": memory_block})
    messages.append({"role": "user", "content": question})

    # 4) chamada ao Grok
    try:
        answer = grok_chat(messages)
    except Exception as e:
        log.exception("Erro ao chamar a x.ai")
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

    # 5) guardar diálogo
    mem0_append_dialog(user_id=user_id, question=question, answer=answer)

    return {
        "answer": answer,
        "mem0": {"facts_used": bool(facts_block), "facts": facts, "recent_found": len(snippets)},
        "new_facts_detected": new_facts,
        "rag": {"used": bool(rag_block), "top_k": RAG_TOP_K, "namespace": DEFAULT_NAMESPACE}
    }

# -----------------------------------------------------------------------------
# D-ID: Texto → Vídeo (lábios)
# -----------------------------------------------------------------------------
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
        "script": {
            "type": "text",
            "input": text,
            "provider": {"type": "microsoft", "voice_id": voice_id},
        },
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

# -----------------------------------------------------------------------------
# HeyGen token demo
# -----------------------------------------------------------------------------
@app.post("/heygen/token")
def heygen_token():
    if not HEYGEN_API_KEY:
        return {"error": "Falta HEYGEN_API_KEY"}
    try:
        res = requests.post(
            "https://api.heygen.com/v1/realtime/session",
            headers={"Authorization": f"Bearer {HEYGEN_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "avatar_id": "ebc94c0e88534d078cf8788a01f3fba9",
                "voice_id": "ff5719e3a6314ecea47badcbb1c0ffaa",
                "language": "pt-PT"
            },
            timeout=15
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"error": str(e)}

# -----------------------------------------------------------------------------
# RAG Endpoints (crawl, sitemap, url, text, pdf, search POST)
# -----------------------------------------------------------------------------

@app.post("/rag/crawl")
async def rag_crawl(request: Request):
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}

    try:
        data = await request.json()
        seed_url   = (data.get("seed_url") or "").strip()
        namespace  = (data.get("namespace") or "default").strip()
        max_pages  = int(data.get("max_pages")  or os.getenv("CRAWL_MAX_PAGES", "500"))
        max_depth  = int(data.get("max_depth")  or os.getenv("CRAWL_MAX_DEPTH", "4"))
        deadline_s = int(data.get("deadline_s") or os.getenv("RAG_DEADLINE_S", "500"))

        # Aceitamos 'verbose' no payload mas, como o rag_client atual não expõe eventos,
        # simplesmente ignoramos para manter compatibilidade com a UI.
        _ = data.get("verbose")

        if not seed_url:
            return {"ok": False, "error": "Falta seed_url"}

        # >>> CHAMADA sem progress_cb <<<
        res = crawl_and_ingest(
            seed_url=seed_url,
            namespace=namespace,
            max_pages=max_pages,
            max_depth=max_depth,
            deadline_s=deadline_s,
        )

        # Normaliza resposta
        if "ok" not in res:
            res["ok"] = True

        # Pequeno resumo auxiliar (não obrigatório)
        res.setdefault(
            "summary",
            f"visited={res.get('visited')} ok_chunks={res.get('ok_chunks')} fail={res.get('fail')} namespace={namespace}"
        )
        return res

    except Exception as e:
        # Usa o logger correto ('log') e devolve stack curto para debug no console
        import traceback
        log.exception("crawl_failed")
        return {
            "ok": False,
            "error": "crawl_failed",
            "detail": str(e),
            "trace": traceback.format_exc(limit=3),
        }

@app.post("/rag/ingest-sitemap")
async def rag_ingest_sitemap_route(request: Request):
    """
    Ingest via Sitemap XML.
    Defaults generosos (podem ser override por payload ou env):
      - max_pages: payload.max_pages || CRAWL_MAX_PAGES || 500
      - deadline_s: payload.deadline_s || RAG_DEADLINE_S || 500
    """
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}
    try:
        data = await request.json()
        sitemap_url = (data.get("sitemap_url") or data.get("site_url") or "").strip()
        namespace   = (data.get("namespace") or "default").strip()
        max_pages   = int(data.get("max_pages")  or os.getenv("CRAWL_MAX_PAGES", "500"))
        deadline_s  = int(data.get("deadline_s") or os.getenv("RAG_DEADLINE_S", "500"))
        if not sitemap_url:
            return {"ok": False, "error": "Falta sitemap_url/site_url"}

        # Delega no rag_client (que já devolve ingested_urls/failed_urls se disponível)
        return ingest_sitemap(
            sitemap_url,
            namespace=namespace,
            max_pages=max_pages,
            deadline_s=deadline_s
        )
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
    """Ingest de texto puro (blocos/notes)."""
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
    """Ingest de um PDF remoto (por URL)."""
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
    """Pesquisa vetorial no Qdrant + bloco de contexto pronto a injectar no LLM."""
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}
    try:
        data = await request.json()
        query     = (data.get("query") or "").strip()
        namespace = (data.get("namespace") or None)
        top_k     = int(data.get("top_k") or os.getenv("RAG_TOP_K", "6"))
        matches = search_chunks(query=query, namespace=namespace, top_k=top_k)
        ctx = build_context_block(matches)
        return {"ok": True, "matches": matches, "context_block": ctx}
    except Exception as e:
        return {"ok": False, "error": "search_failed", "detail": str(e)}

# -----------------------------------------------------------------------------
# Local run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
