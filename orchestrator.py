import anthropic
from agents import ceo

client = anthropic.Anthropic()

AGENTES = {"ceo": ceo.responder}
# semana 5+: "orcamentos": orcamentos.responder, "design": design.responder, ...

def encaminhar(pergunta: str) -> str:
    """Classifica a intenção com Haiku. Com 1 agente, é trivial;
    a estrutura fica pronta para os próximos."""
    if len(AGENTES) == 1:
        return "ceo"
    r = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=10,
        system="Classifica a pergunta num destes agentes: "
               + ", ".join(AGENTES) + ". Responde só com o nome do agente.",
        messages=[{"role": "user", "content": pergunta}]
    )
    escolha = r.content[0].text.strip().lower()
    return escolha if escolha in AGENTES else "ceo"  # fallback: CEO
