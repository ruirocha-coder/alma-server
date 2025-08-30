from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import logging
import uvicorn
import time
import hashlib  # 👈 novo: para gerar user_id estável a partir de IP+UA

# 👇 novo: helpers do Mem0 (ficheiro mem0.py que te enviei)
try:
    from mem0 import search_memories, add_memory  # search antes, add depois
except Exception:
    # fallback “no-op” se o mem0.py não estiver presente (não parte nada)
    def search_memories(user_id: str, query: str, limit: int = 5, timeout_s: float = 3.5):
        return []
    def add_memory(user_id: str, text: str, ttl_days: int | None = None, timeout_s: float = 2.0):
        return

# ── App & CORS ────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # podes restringir ao teu domínio depois
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Logs ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("alma")

# ── Config (Grok) ─────────────────────────────────────────────────────────────
XAI_API_KEY = os.getenv("XAI_API_KEY")  # DEFINE NAS VARIABLES DO RAILWAY
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-4-0709"

# ── Config (D-ID) ─────────────────────────────────────────────────────────────
DID_API_KEY = os.getenv("DID_API_KEY", "").strip()
DEFAULT_IMAGE_URL = os.getenv("DEFAULT_IMAGE_URL", "").strip()
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "pt-PT-FernandaNeural").strip()
DID_BASE = "https://api.d-id.com"

def did_headers():
    # Alguns tenants aceitam Authorization: Basic <KEY>; outros x-api-key.
    h = {"Content-Type": "application/json"}
    if DID_API_KEY:
        h["Authorization"] = f"Basic {DID_API_KEY}"
        h["x-api-key"] = DID_API_KEY
    return h

# ── Config (HeyGen token demo já existente) ───────────────────────────────────
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")

# ── Util: gerar user_id estável sem mexer no frontend ─────────────────────────
def guess_user_id(req: Request) -> str:
    ip = req.headers.get("x-forwarded-for") or (req.client.host if req.client else "") or ""
    ua = req.headers.get("user-agent") or ""
    h = hashlib.sha256(f"{ip}|{ua}".encode()).hexdigest()[:24]
    return f"user_{h}"

# ── Rotas básicas ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Alma server ativo. Use POST /ask (Grok) ou POST /say (D-ID).",
        "endpoints": {
            "health": "/health",
            "ask": "POST /ask {question}",
            "say": "POST /say {text, image_url?, voice_id?}",
            "ping_grok": "/ping_grok",
            "ask_get": "/ask_get?q=...",
            "heygen_token": "POST /heygen/token"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/echo")
async def echo(request: Request):
    data = await request.json()
    return {"echo": data}

# ── IA: Pergunta → Resposta (Grok-4) + Memória curta (Mem0) ───────────────────
@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    log.info(f"[/ask] question={question!r}")

    if not XAI_API_KEY:
        log.error("XAI_API_KEY ausente nas Variables do Railway.")
        return {"answer": "⚠️ Falta XAI_API_KEY nas Variables do Railway."}

    # 1) user_id estável (sem tocar no frontend)
    user_id = guess_user_id(request)

    # 2) buscar memórias relevantes (timeout curto; não bloqueia a UX)
    mems = search_memories(user_id, question, limit=5, timeout_s=3.5)
    mem_block = "\n".join(f"- {m}" for m in mems) if mems else ""
    memory_prefix = f"Contexto do utilizador (memória curta):\n{mem_block}\n\n" if mem_block else ""

    # 3) prompts (mantido, só injeta contexto se houver)
    system_prompt = (
        "És a Alma, especialista em design de interiores (método psicoestético). "
        "Responde claro, conciso e em pt-PT. Usa o contexto se existir."
    )
    user_prompt = memory_prefix + question

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        r = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=30)
        log.info(f"[x.ai] status={r.status_code} body={r.text[:300]}")
        r.raise_for_status()
        answer = r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or "Sem resposta do modelo."
    except Exception as e:
        log.exception("Erro ao chamar a x.ai")
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

    # 4) heurística: guardar frases curtas do utilizador (fire-and-forget)
    try:
        if question and len(question) <= 220:
            add_memory(user_id, question)  # TTL vem de MEM0_TTL_DAYS (padrão 7)
    except Exception:
        pass  # não falha a resposta por causa de memória

    return {"answer": answer}

@app.get("/ping_grok")
def ping_grok():
    key = os.getenv("XAI_API_KEY")
    if not key:
        return {"ok": False, "reason": "XAI_API_KEY ausente"}
    try:
        r = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": [{"role": "user", "content": "ping"}]},
            timeout=10
        )
        return {"ok": r.ok, "status": r.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/ask_get")
def ask_get(q: str = "Olá, estás ligado?"):
    key = os.getenv("XAI_API_KEY")
    if not key:
        return {"ok": False, "reason": "XAI_API_KEY ausente nas Variables do Railway."}
    try:
        r = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "És a Alma (psicoestético). Responde claro em pt-PT."},
                    {"role": "user", "content": q}
                ]
            },
            timeout=12
        )
        if not r.ok:
            return {"ok": False, "status": r.status_code, "body": r.text[:300]}
        content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"ok": True, "answer": content}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── D-ID: Texto → Vídeo (lábios) ─────────────────────────────────────────────
@app.post("/say")
async def say(request: Request):
    """
    Body JSON:
    {
      "text": "Olá ...",
      "image_url": "https://raw.githubusercontent.com/.../minha.png",  (opcional)
      "voice_id": "pt-PT-FernandaNeural"                               (opcional)
    }
    -> retorna {"video_url": "..."} (MP4 gerado pelo D-ID)
    """
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

    # 1) criar talk
    try:
        r = requests.post(f"{DID_BASE}/talks", headers=did_headers(), json=payload, timeout=30)
        log.info(f"[d-id] create talks -> {r.status_code} {r.text[:200]}")
        r.raise_for_status()
    except Exception as e:
        return {"error": f"Falha a criar talk: {e}", "body": getattr(r, 'text', '')[:500]}

    talk = r.json()
    talk_id = talk.get("id")
    if not talk_id:
        return {"error": "Sem id do talk", "raw": talk}

    # 2) poll até result_url
    result_url = None
    for _ in range(30):  # ~30s
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

# ── Local run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
