# main.py — Alma Server com Mem0 (curto prazo) + Grok
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import logging
import uvicorn
import time

# ── Mem0 (curto prazo) ────────────────────────────────────────────────────────
MEM0_ENABLE = os.getenv("MEM0_ENABLE", "false").lower() in ("1", "true", "yes")
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "").strip()
MEM0_BASE_URL = os.getenv("MEM0_BASE_URL", "").strip() or "https://api.mem0.ai/v1"

mem0_client = None
if MEM0_ENABLE and MEM0_API_KEY:
    try:
        from mem0 import MemoryClient
        mem0_client = MemoryClient(api_key=MEM0_API_KEY, base_url=MEM0_BASE_URL)
    except Exception as e:
        print("⚠️  Mem0 não inicializou:", e)
        mem0_client = None

# ── App & CORS ────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Logs ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("alma")
log.info(f"[boot] mem0_enabled={MEM0_ENABLE} mem0_client_ready={bool(mem0_client)} base_url={MEM0_BASE_URL}")

# ── Config (Grok) ─────────────────────────────────────────────────────────────
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = os.getenv("XAI_MODEL", "grok-4-0709")

# ── Config (D-ID) ─────────────────────────────────────────────────────────────
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

# ── Config (HeyGen token demo já existente) ───────────────────────────────────
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")

# ── Rotas básicas ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Alma server ativo. Use POST /ask (Grok+Mem0) ou POST /say (D-ID).",
        "mem0": {"enabled": MEM0_ENABLE, "base_url": MEM0_BASE_URL if MEM0_ENABLE else None},
        "endpoints": {
            "health": "/health",
            "ask": "POST /ask {question, user_id?}",
            "say": "POST /say {text, image_url?, voice_id?}",
            "ping_grok": "/ping_grok",
            "ask_get": "/ask_get?q=...",
            "heygen_token": "POST /heygen/token"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "mem0_enabled": MEM0_ENABLE and bool(mem0_client)}

@app.post("/echo")
async def echo(request: Request):
    data = await request.json()
    return {"echo": data}

# ── IA: Pergunta → Resposta (Grok-4) + Mem0 curto prazo ───────────────────────
@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    user_id = (data.get("user_id") or "").strip() or "anon"
    log.info(f"[/ask] user_id={user_id} question={question!r}")

    if not XAI_API_KEY:
        log.error("XAI_API_KEY ausente nas Variables do Railway.")
        return {"answer": "⚠️ Falta XAI_API_KEY nas Variables do Railway."}

    # 1) Buscar memórias recentes/relevantes do utilizador
    memory_context = ""
    mem_debug = None
    if MEM0_ENABLE and mem0_client:
        try:
            results = mem0_client.search(
                query=question,
                filters={"user_id": user_id},
                version="v2",
                output_format="v1.1"
            )
            snippets = []
            # results é um dict v2; lista vem em "results"
            for item in results.get("results", []):
                val = (
                    item.get("text")
                    or item.get("memory")
                    or item.get("content")
                    or ""
                ).strip()
                if val:
                    snippets.append(val)

            if snippets:
                memory_context = (
                    "Memórias recentes do utilizador (curto prazo):\n"
                    + "\n".join(f"- {s}" for s in snippets[:3])
                )
            mem_debug = {"found": len(snippets)}

            # 🔎 LOG: o que veio da pesquisa
            log.info(f"[mem0] search user_id={user_id} found={len(snippets)} snippets={snippets[:3]}")
        except Exception as e:
            log.warning(f"[mem0] search falhou: {e}")
    else:
        if MEM0_ENABLE and not mem0_client:
            log.warning("[mem0] MEM0_ENABLE=True mas mem0_client não está pronto (chave/base_url?).")

    # 2) Montar mensagens para o Grok
    messages = [
        {
            "role": "system",
            "content": "És a Alma, especialista em design de interiores (método psicoestético). Responde claro, conciso e em pt-PT.",
        }
    ]
    if memory_context:
        messages.append({"role": "system", "content": memory_context})
    messages.append({"role": "user", "content": question})

    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": messages}

    # 3) Chamar Grok
    try:
        r = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=30)
        log.info(f"[x.ai] status={r.status_code} body={r.text[:300]}")
        r.raise_for_status()
        answer = r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    except Exception as e:
        log.exception("Erro ao chamar a x.ai")
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

    # 4) Guardar a interação como memória (formato oficial Mem0)
    if MEM0_ENABLE and mem0_client:
        try:
            messages_to_store = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            mem0_client.add(
                messages_to_store,
                user_id=user_id,
                version="v2",
                output_format="v1.1"
            )
            # 📝 LOG: confirmação de gravação
            log.info(f"[mem0] add user_id={user_id} -> stored dialog (user:'{question[:60]}', assistant:'{answer[:60]}')")
        except Exception as e:
            log.warning(f"[mem0] add falhou: {e}")

    return {"answer": answer, "mem0": mem_debug}

# ── D-ID: Texto → Vídeo (lábios) ─────────────────────────────────────────────
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
        result_url = j.get("result_url")
        status = j.get("status")
        if result_url:
            break
        if status == "error":
            return {"error": "D-ID devolveu erro", "details": j}

    if not result_url:
        return {"error": "Timeout à espera do result_url"}

    return {"video_url": result_url}

# ── HeyGen token demo ─────────────────────────────────────────────────────────
@app.post("/heygen/token")
def heygen_token():
    if not HEYGEN_API_KEY:
        return {"error":"Falta HEYGEN_API_KEY"}
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

# ── Local run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
