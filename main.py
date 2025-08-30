from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os, logging, requests, uvicorn

# ── Mem0 (curto prazo) ────────────────────────────────────────────────────────
MEM0_ENABLE = os.getenv("MEM0_ENABLE", "false").lower() in ("1", "true", "yes")
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "").strip()

mem0_client = None
if MEM0_ENABLE and MEM0_API_KEY:
    try:
        from mem0 import MemoryClient  # <- pacote certo
        mem0_client = MemoryClient(api_key=MEM0_API_KEY)  # sem base_url no __init__
    except Exception as e:
        mem0_client = None
        logging.error(f"[mem0] import/init failed: {e}")
elif MEM0_ENABLE and not MEM0_API_KEY:
    logging.error("[mem0] MEM0_ENABLE=true mas falta MEM0_API_KEY")

# ── App & CORS ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("alma")
log.info(f"[boot] mem0_enabled={MEM0_ENABLE} mem0_client_ready={bool(mem0_client)}")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ── Config Grok (x.ai) ────────────────────────────────────────────────────────
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = os.getenv("XAI_MODEL", "grok-4-0709")

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Alma server ativo. Use POST /ask",
        "mem0": {"enabled": MEM0_ENABLE, "client_ready": bool(mem0_client)},
        "endpoints": {"ask": "POST /ask {question, user_id?}"}
    }

@app.get("/health")
def health():
    return {"status": "ok", "mem0_enabled": MEM0_ENABLE and bool(mem0_client)}

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    user_id = (data.get("user_id") or "anon").strip()

    if not XAI_API_KEY:
        return {"answer": "⚠️ Falta XAI_API_KEY nas Variables do Railway."}

    # 1) Recuperar memórias recentes
    memory_context = ""
    if MEM0_ENABLE and mem0_client:
        try:
            results = mem0_client.memories.search(query=question or "contexto", user_id=user_id, limit=5)
            snippets = []
            for item in results or []:
                s = (item.get("text") or item.get("memory") or item.get("content") or "").strip()
                if s: snippets.append(s)
            if snippets:
                memory_context = "Memórias recentes do utilizador:\n" + "\n".join(f"- {s}" for s in snippets[:3])
            log.info(f"[mem0] search user_id={user_id} found={len(snippets)}")
        except Exception as e:
            log.warning(f"[mem0] search falhou: {e}")

    # 2) Mensagens para Grok
    messages = [{"role": "system",
                 "content": "És a Alma, especialista em design de interiores (método psicoestético). Responde claro, conciso e em pt-PT."}]
    if memory_context:
        messages.append({"role": "system", "content": memory_context})
    messages.append({"role": "user", "content": question})

    # 3) Chamada ao Grok
    try:
        r = requests.post(
            XAI_API_URL,
            headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": messages},
            timeout=30
        )
        r.raise_for_status()
        answer = r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    except Exception as e:
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

    # 4) Guardar diálogo
    if MEM0_ENABLE and mem0_client:
        try:
            mem0_client.memories.create(content=f"User: {question}", user_id=user_id,
                                        metadata={"source": "alma-server", "type": "dialog"})
            mem0_client.memories.create(content=f"Alma: {answer}", user_id=user_id,
                                        metadata={"source": "alma-server", "type": "dialog"})
        except Exception as e:
            log.warning(f"[mem0] create falhou: {e}")

    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
