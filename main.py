# main.py — Alma Server com Mem0 (histórico + contexto) + Grok + D-ID

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os, requests, logging, uvicorn, time, sys, subprocess, importlib

# ── Mem0 (gestão de memórias) ────────────────────────────────────────────────
MEM0_ENABLE = os.getenv("MEM0_ENABLE", "false").lower() in ("1", "true", "yes")
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "").strip()

mem0_client = None
MemoryClient = None

if MEM0_ENABLE and MEM0_API_KEY:
    try:
        # nome de módulo mais recente
        from mem0ai import MemoryClient as _MC
        MemoryClient = _MC
    except Exception as e:
        print(f"[mem0] pacote ausente ({e}); a instalar em runtime…")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "mem0ai==0.1.0"])
        except Exception as ie:
            print(f"[mem0] falha a instalar mem0ai: {ie}")

        # tentar novamente mem0ai
        try:
            from mem0ai import MemoryClient as _MC
            MemoryClient = _MC
        except Exception as e2:
            # fallback para possíveis nomes antigos
            try:
                from mem0 import MemoryClient as _MC
                MemoryClient = _MC
            except Exception as e3:
                print(f"[mem0] ainda não consigo importar MemoryClient: {e2} / {e3}")
                MemoryClient = None

    # >>> AQUI ESTÁ O FIX PRINCIPAL <<<
    if MemoryClient is not None:
        try:
            mem0_client = MemoryClient(api_key=MEM0_API_KEY)
        except Exception as e:
            print(f"[mem0] não inicializou: {e}")
            mem0_client = None
else:
    if MEM0_ENABLE and not MEM0_API_KEY:
        print("[mem0] MEM0_ENABLE=true mas falta MEM0_API_KEY")
# ── Logs ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("alma")
log.info(f"[boot] mem0_enabled={MEM0_ENABLE} mem0_client_ready={bool(mem0_client)}")

# ── App & CORS ───────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── Config (Grok) ────────────────────────────────────────────────────────────
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = os.getenv("XAI_MODEL", "grok-4-0709")

# ── Config (D-ID) ────────────────────────────────────────────────────────────
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

# ── Config (HeyGen) ──────────────────────────────────────────────────────────
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")

# ── Rotas básicas ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Alma server ativo. Use POST /ask (Grok+Mem0) ou POST /say (D-ID).",
        "mem0": {"enabled": MEM0_ENABLE, "client_ready": bool(mem0_client)},
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
    return {"echo": await request.json()}

# ── IA: Pergunta → Resposta (Grok + Mem0) ────────────────────────────────────
@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    user_id = (data.get("user_id") or "anon").strip()
    log.info(f"[/ask] user_id={user_id} question={question!r}")

    if not XAI_API_KEY:
        return {"answer": "⚠️ Falta XAI_API_KEY nas Variables do Railway."}

    # 1) Buscar memórias recentes
    memory_context = ""
    mem_debug = {}
    if MEM0_ENABLE and mem0_client:
        try:
            results = mem0_client.memories.search(query=question or "contexto", user_id=user_id, limit=5)
            snippets = [ (i.get("text") or i.get("memory") or i.get("content") or "").strip() for i in results ]
            snippets = [s for s in snippets if s]
            if snippets:
                memory_context = "Histórico recente da conversa:\n" + "\n".join(f"- {s}" for s in snippets)
            mem_debug["found"] = len(snippets)
        except Exception as e:
            log.warning(f"[mem0] search falhou: {e}")
            mem_debug["error"] = str(e)

    # 2) Preparar prompt para Grok
    messages = [{"role": "system", "content": "És a Alma, especialista em design de interiores (método psicoestético). Responde claro, conciso e em pt-PT."}]
    if memory_context:
        messages.append({"role": "system", "content": memory_context})
    messages.append({"role": "user", "content": question})

    # 3) Chamar Grok
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(XAI_API_URL, headers=headers, json={"model": MODEL, "messages": messages}, timeout=30)
        r.raise_for_status()
        answer = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

    # 4) Guardar no Mem0
    if MEM0_ENABLE and mem0_client:
        try:
            mem0_client.memories.create(content=f"User: {question}", user_id=user_id, metadata={"type": "dialog"})
            mem0_client.memories.create(content=f"Alma: {answer}", user_id=user_id, metadata={"type": "dialog"})
        except Exception as e:
            log.warning(f"[mem0] create falhou: {e}")

    return {"answer": answer, "mem0": mem_debug}

# ── D-ID: Texto → Vídeo ──────────────────────────────────────────────────────
@app.post("/say")
async def say(request: Request):
    if not DID_API_KEY:
        return {"error": "Falta DID_API_KEY nas Variables do Railway"}
    data = await request.json()
    text = (data.get("text") or "").strip()
    image_url = (data.get("image_url") or DEFAULT_IMAGE_URL).strip()
    voice_id = (data.get("voice_id") or DEFAULT_VOICE).strip()
    if not text: return {"error": "Campo 'text' é obrigatório"}
    if not image_url: return {"error": "Falta 'image_url' (define DEFAULT_IMAGE_URL nas Variables)"}
    payload = {"script": {"type": "text","input": text,"provider": {"type": "microsoft","voice_id": voice_id}}, "source_url": image_url}
    r = requests.post(f"{DID_BASE}/talks", headers=did_headers(), json=payload, timeout=30)
    talk_id = r.json().get("id")
    if not talk_id: return {"error": "Sem id do talk", "raw": r.json()}
    for _ in range(30):
        time.sleep(1)
        j = requests.get(f"{DID_BASE}/talks/{talk_id}", headers=did_headers(), timeout=15).json()
        if j.get("result_url"): return {"video_url": j["result_url"]}
        if j.get("status") == "error": return {"error": "D-ID devolveu erro", "details": j}
    return {"error": "Timeout à espera do result_url"}

# ── HeyGen token ─────────────────────────────────────────────────────────────
@app.post("/heygen/token")
def heygen_token():
    if not HEYGEN_API_KEY: return {"error": "Falta HEYGEN_API_KEY"}
    res = requests.post("https://api.heygen.com/v1/realtime/session",
        headers={"Authorization": f"Bearer {HEYGEN_API_KEY}", "Content-Type": "application/json"},
        json={"avatar_id": "ebc94c0e88534d078cf8788a01f3fba9","voice_id": "ff5719e3a6314ecea47badcbb1c0ffaa","language": "pt-PT"}, timeout=15)
    return res.json()

# ── Local run ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
