from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, os, requests

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

XAI_API_KEY = os.getenv("XAI_API_KEY")  # ← ler das Variables
XAI_API_URL = "https://api.x.ai/v1/chat/completions"

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")

    if not XAI_API_KEY:
        return {"answer": "⚠️ Falta XAI_API_KEY nas Variables do Railway."}

    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "grok-4",
        "messages": [
            {"role": "system", "content": "És a Alma, especialista em design de interiores (método psicoestético). Responde claro e em pt-PT."},
            {"role": "user", "content": question}
        ]
    }

    try:
        r = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        answer = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"answer": answer or "Sem resposta do modelo."}
    except Exception as e:
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok", "message": "Alma server ativo. Use POST /ask (POST) com JSON {\"question\":\"...\"}."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
