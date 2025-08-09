from fastapi import FastAPI, Request
import uvicorn, os, requests

app = FastAPI()

XAI_API_KEY = os.getenv("XAI_API_KEY")
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
