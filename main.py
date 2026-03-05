# main.py
import os
from typing import List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import httpx

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-2-latest")

MEMO_ENABLE = os.getenv("MEMO_ENABLE", "0") == "1"
MEMO_API_KEY = os.getenv("MEMO_API_KEY", "")

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "alma_docs")

# --------------------------------------------------------------------------------------
# Alma System Prompt (site-first)
# --------------------------------------------------------------------------------------

ALMA_SYSTEM = """
És a Alma, assistente da Boa Safra e da Interior Guider.

MISSÃO
Ajudar clientes e equipa com respostas claras, rápidas e úteis.

DOUTRINA
O site oficial é a referência principal:
- interiorguider.com
- boasafra.pt

Se o utilizador não indicar marca:
assume Interior Guider.

ESTILO
- claro
- direto
- sem emojis
- sem conversa desnecessária

ORÇAMENTOS
Quando pedirem orçamento:

1. identifica o produto
2. apresenta um orçamento direto
3. se faltar informação pede apenas o mínimo necessário

Nunca bloqueies resposta por falta de validação.

FORMATO

Nome do produto
Preço unitário
Quantidade
Subtotal

Nota:
preço com IVA incluído; portes não incluídos.

Se aplicável podes incluir:
[ver produto](URL)

É melhor dar uma resposta útil do que bloquear.
"""

# --------------------------------------------------------------------------------------
# FastAPI
# --------------------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

class AskRequest(BaseModel):
    user_id: str
    question: str


# --------------------------------------------------------------------------------------
# mem0 (opcional)
# --------------------------------------------------------------------------------------

memo_client = None
if MEMO_ENABLE and MEMO_API_KEY:
    try:
        from mem0 import MemoryClient
        memo_client = MemoryClient(api_key=MEMO_API_KEY)
        print("[mem0] enabled")
    except Exception as e:
        print("[mem0] disabled:", e)


async def get_memory(user_id: str) -> str:
    if not memo_client:
        return ""
    try:
        memories = memo_client.get_all(user_id=user_id)
        texts = [m.get("memory", "") for m in memories[:5] if m.get("memory")]
        return "\n".join(texts)
    except Exception:
        return ""


async def store_memory(user_id: str, text: str):
    if not memo_client:
        return
    try:
        memo_client.add(
            user_id=user_id,
            messages=[{"role": "assistant", "content": text}]
        )
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# RAG (opcional)
# --------------------------------------------------------------------------------------

async def rag_search(question: str) -> str:
    if not QDRANT_URL:
        return ""
    try:
        import qdrant_client
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        vector = model.encode(question).tolist()

        client = qdrant_client.QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        hits = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=3
        )

        texts = []
        for h in hits:
            payload = h.payload or {}
            if isinstance(payload, dict) and payload.get("text"):
                texts.append(str(payload["text"]))

        return "\n\n".join(texts)

    except Exception as e:
        print("[rag] disabled:", e)
        return ""


# --------------------------------------------------------------------------------------
# Grok call (SAFE)
# --------------------------------------------------------------------------------------

async def call_grok(messages: List[Dict[str, str]]) -> str:
    if not XAI_API_KEY:
        return "Erro: XAI_API_KEY não está configurada no servidor."

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": XAI_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)

        # não deixar isto rebentar o /ask
        if r.status_code != 200:
            body = r.text[:800]
            return f"Erro ao chamar o Grok (HTTP {r.status_code}). Detalhe: {body}"

        data = r.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Erro técnico ao chamar o Grok: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------------------
# Build messages
# --------------------------------------------------------------------------------------

async def build_messages(user_id: str, question: str):
    memory = await get_memory(user_id)
    rag = await rag_search(question)

    messages = [{"role": "system", "content": ALMA_SYSTEM}]

    if memory:
        messages.append({"role": "system", "content": f"Memória do utilizador:\n{memory}"})

    if rag:
        messages.append({"role": "system", "content": f"Contexto interno:\n{rag}"})

    messages.append({"role": "user", "content": question})
    return messages


# --------------------------------------------------------------------------------------
# Routes (SAFE JSON)
# --------------------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "service": "alma-server-minimal"}


@app.post("/ask")
async def ask(req: AskRequest):
    try:
        messages = await build_messages(req.user_id, req.question)
        answer = await call_grok(messages)

        # só grava memória se houver resposta “normal”
        if answer and not answer.startswith("Erro"):
            await store_memory(req.user_id, answer)

        # DEVOLVE SEMPRE JSON
        return JSONResponse(status_code=200, content={"answer": answer})

    except Exception as e:
        # DEVOLVE SEMPRE JSON, mesmo em falha
        return JSONResponse(
            status_code=200,
            content={"answer": f"Erro interno no /ask: {type(e).__name__}: {e}"}
        )


# -----------------------------------------------------------------------------------
# Local dev entrypoint (Railway uses its own command, but this is safe)
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
