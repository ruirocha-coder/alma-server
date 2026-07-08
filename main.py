from fastapi import FastAPI
from pydantic import BaseModel
from orchestrator import encaminhar, AGENTES
from db import guardar_mensagem, historico_sessao, log_routing

app = FastAPI(title="ALMA")

class Pedido(BaseModel):
    utilizador: str
    sessao: str
    mensagem: str

@app.post("/alma")
def alma(p: Pedido):
    agente = encaminhar(p.mensagem)
    log_routing(p.mensagem, agente)

    mensagens = historico_sessao(p.sessao)          # memória partilhada
    mensagens.append({"role": "user", "content": p.mensagem})

    resposta = AGENTES[agente](mensagens)

    guardar_mensagem(p.utilizador, p.sessao, "user", p.mensagem)
    guardar_mensagem(p.utilizador, p.sessao, "assistant", resposta, agente)
    return {"resposta": resposta}                    # o agente nunca é exposto

@app.get("/health")
def health():
    return {"status": "ok"}
