from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import logging
import uvicorn

# ── App & CORS ─────────────────────────────────────────────────────────────────
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

# ── Config ────────────────────────────────────────────────────────────────────
XAI_API_KEY = os.getenv("XAI_API_KEY")  # DEFINE NAS VARIABLES DO RAILWAY
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-4-0709"

# ── Rotas básicas ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok",
            "message": "Alma server ativo. Use POST /ask com JSON {\"question\":\"...\"}."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/echo")
async def echo(request: Request):
    """Ajuda a testar POST/JSON sem chamar a x.ai."""
    data = await request.json()
    return {"echo": data}

# ── IA: Pergunta → Resposta (Grok-4) ─────────────────────────────────────────
@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")
    log.info(f"[/ask] question={question!r}")

    if not XAI_API_KEY:
        log.error("XAI_API_KEY ausente nas Variables do Railway.")
        return {"answer": "⚠️ Falta XAI_API_KEY nas Variables do Railway."}

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system",
             "content": "És a Alma, especialista em design de interiores (método psicoestético). Responde claro, conciso e em pt-PT."},
            {"role": "user", "content": question}
        ]
    }

    try:
        r = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=30)
        log.info(f"[x.ai] status={r.status_code} body={r.text[:300]}")
        r.raise_for_status()
        answer = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"answer": answer or "Sem resposta do modelo."}
    except Exception as e:
        log.exception("Erro ao chamar a x.ai")
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

@app.get("/ping_grok")
def ping_grok():
    key = os.getenv("XAI_API_KEY")
    if not key:
        return {"ok": False, "reason": "XAI_API_KEY ausente"}
    try:
        r = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model":MODEL,"messages":[{"role":"user","content":"ping"}]},
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
            json={"model":MODEL,
                  "messages": [{"role": "system",
                                "content": "És a Alma (psicoestético). Responde claro em pt-PT."},
                               {"role": "user", "content": q}]},
            timeout=12
        )
        if not r.ok:
            return {"ok": False, "status": r.status_code, "body": r.text[:300]}
        content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"ok": True, "answer": content}
    except Exception as e:
        return {"ok": False, "error": str(e)}

HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")

@app.post("/heygen/token")
def heygen_token():
    if not HEYGEN_API_KEY:
        return {"error":"Falta HEYGEN_API_KEY"}
    # IMPORTANTE: os detalhes do endpoint/token podem variar conforme a versão da HeyGen.
    # Usa o endpoint/documentação do teu plano "Realtime/Live".
    # Abaixo fica o padrão típico de "create session" via REST:
    try:
        res = requests.post(
            "https://api.heygen.com/v1/realtime/session",  # <- confirma no teu painel/docs
            headers={"Authorization": f"Bearer {HEYGEN_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                # ajusta estes campos ao que o teu painel pedir:
                "avatar_id": "ebc94c0e88534d078cf8788a01f3fba9",
                "voice_id": "ff5719e3a6314ecea47badcbb1c0ffaa",
                "language": "pt-PT"
            },
            timeout=15
        )
        res.raise_for_status()
        return res.json()  # costuma trazer wsUrl/rtcToken/sessionId
    except Exception as e:
        return {"error": str(e)}



# ── Local run (não usado no Railway, mas útil em dev) ────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
