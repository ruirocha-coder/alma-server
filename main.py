from fastapi import FastAPI, Request
import uvicorn, os, requests

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ou mete o domínio que quiseres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


XAI_API_KEY = os.getenv("xai-bxIlniRJBI1kWaX1fQ7kjOzXwz33N6ZuKCxE3l9jvguMRiqD0zNhQNtlmtNQHix0IlMWxhGnBEptNu3P")
XAI_API_URL = "https://api.x.ai/v1/chat/completions"

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "grok-4",
        "messages": [
            {"role": "system",
             "content": "És a Alma, especialista em design de interiores (método psicoestético). Responde de forma clara e concisa em pt-PT."},
            {"role": "user", "content": question}
        ]
    }

    r = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    answer = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    return {"answer": answer}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok", "message": "Alma server ativo. Use POST /ask para enviar perguntas."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
