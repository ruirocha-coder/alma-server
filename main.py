# main.py — Alma Server (limpo) com Mem0 (curto prazo) + Grok (x.ai) + D-ID + HeyGen
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import logging
import uvicorn
import time

# ─────────────────────────────────────────────────────────────────────────────
# Config geral / Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("alma")

APP_VERSION = os.getenv("APP_VERSION", "alma-server/clean-1")

# ─────────────────────────────────────────────────────────────────────────────
# Mem0 (curto prazo) — SEM installs em runtime
# ─────────────────────────────────────────────────────────────────────────────
MEM0_ENABLE = os.getenv("MEM0_ENABLE", "false").lower() in ("1", "true", "yes")
MEM0_API_KEY = (os.getenv("MEM0_API_KEY") or "").strip()

MemoryClient = None
mem0_client = None

if MEM0_ENABLE:
    if not MEM0_API_KEY:
        log.warning("[mem0] MEM0_ENABLE=true mas falta MEM0_API_KEY")
    else:
        try:
            import mem0ai as _mem0ai
            from mem0ai import MemoryClient as _MC
            MemoryClient = _MC
            log.info(f"[mem0] import OK: file={getattr(_mem0ai,'__file__','?')} ver={getattr(_mem0ai,'__version__','?')}")
        except Exception as e:
            log.error(f"[mem0] import FAILED: {e}")
            MemoryClient = None

        if MemoryClient is not None:
            try:
                # versões recentes do mem0ai **não** aceitam base_url no __init__
                mem0_client = MemoryClient(api_key=MEM0_API_KEY)
                log.info("[mem0] MemoryClient inicializado.")
            except Exception as e:
                log.error(f"[mem0] não inicializou: {e}")
                mem0_client = None

log.info(f"[boot] {APP_VERSION} mem0_enabled={MEM0_ENABLE} mem0_client_ready={bool(mem0_client)}")

# ─────────────────────────────────────────────────────────────────────────────
# Config Grok (x.ai)
# ─────────────────────────────────────────────────────────────────────────────
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = os.getenv("XAI_MODEL", "grok-4-0709")

# ─────────────────────────────────────────────────────────────────────────────
# Config D-ID (texto→vídeo de lábios)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Config HeyGen (token demo)
# ─────────────────────────────────────────────────────────────────────────────
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY", "").strip()

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI & CORS
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Alma Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def grok_chat(messages, timeout=30):
    if not XAI_API_KEY:
        raise RuntimeError("Falta XAI_API_KEY")
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": messages}
    r = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=timeout)
    log.info(f"[x.ai] status={r.status_code} body={r.text[:300]}")
    r.raise_for_status()
    return r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or ""

def mem0_search(user_id: str, query: str, limit: int = 5):
    """Busca memórias relevantes (curto prazo) do utilizador."""
    if not (MEM0_ENABLE and mem0_client):
        return []
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
        return []

def mem0_append_dialog(user_id: str, question: str, answer: str):
    """Guarda a interação atual como memórias (user/assistant)."""
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

# ─────────────────────────────────────────────────────────────────────────────
# Rotas
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "message": "Alma server ativo. Use POST /ask (Grok+Mem0) ou POST /say (D-ID).",
        "mem0": {"enabled": MEM0_ENABLE, "client_ready": bool(mem0_client)},
        "endpoints": {
            "health": "/health",
            "ask": "POST /ask {question, user_id?}",
            "ask_get": "/ask_get?q=...&user_id=...",
            "ping_grok": "/ping_grok",
            "say": "POST /say {text, image_url?, voice_id?}",
            "heygen_token": "POST /heygen/token"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mem0_enabled": MEM0_ENABLE,
        "mem0_client_ready": bool(mem0_client),
        "model": MODEL
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

@app.get("/ask_get")
def ask_get(q: str = "", user_id: str = "anon"):
    # wrapper GET para testar rápido no browser
    if not q:
        return {"answer": "Falta query param ?q="}
    # reuse do POST /ask pipeline:
    messages = [{"role": "system",
                 "content": "És a Alma, especialista em design de interiores (método psicoestético). Responde claro, conciso e em pt-PT."}]
    snippets = mem0_search(user_id=user_id, query=q, limit=5)
    if snippets:
        memory_context = "Memórias recentes do utilizador (curto prazo):\n" + "\n".join(f"- {s}" for s in snippets[:3])
        messages.append({"role": "system", "content": memory_context})
    messages.append({"role": "user", "content": q})
    try:
        answer = grok_chat(messages)
    except Exception as e:
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}
    mem0_append_dialog(user_id=user_id, question=q, answer=answer)
    return {"answer": answer, "mem0": {"found": len(snippets)}}

# IA: Pergunta → Resposta (Grok-4) + Mem0 curto prazo
@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    user_id = (data.get("user_id") or "").strip() or "anon"
    log.info(f"[/ask] user_id={user_id} question={question!r}")

    if not question:
        return {"answer": "Coloca a tua pergunta em 'question'."}

    # 1) contexto curto prazo do mem0
    snippets = mem0_search(user_id=user_id, query=question, limit=5)
    memory_context = ""
    if snippets:
        memory_context = "Memórias recentes do utilizador (curto prazo):\n" + "\n".join(f"- {s}" for s in snippets[:3])

    # 2) mensagens para Grok
    messages = [{
        "role": "system",
        "content": "És a Alma, especialista em design de interiores (método psicoestético). Responde claro, conciso e em pt-PT."
    }]
    if memory_context:
        messages.append({"role": "system", "content": memory_context})
    messages.append({"role": "user", "content": question})

    # 3) chamada ao Grok
    try:
        answer = grok_chat(messages)
    except Exception as e:
        log.exception("Erro ao chamar a x.ai")
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

    # 4) guardar diálogo
    mem0_append_dialog(user_id=user_id, question=question, answer=answer)

    return {"answer": answer, "mem0": {"found": len(snippets)}}

# D-ID: Texto → Vídeo (lábios)
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

# HeyGen token demo
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

# ─────────────────────────────────────────────────────────────────────────────
# Local run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
