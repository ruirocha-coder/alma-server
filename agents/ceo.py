from persona import PERSONA
from tools.bigcommerce import TOOLS_CEO
from agents.base import correr_agente

MISSAO_CEO = PERSONA + """

Missão atual: visão executiva da Interior Guider. Respondes sobre vendas,
margens, catálogo, encomendas e estado do negócio. A margem calcula-se como
(price - cost_price) / price. Se cost_price for 0 ou nulo, sinaliza que o
custo não está carregado nesse produto."""

def responder(mensagens: list) -> str:
    return correr_agente(MISSAO_CEO, TOOLS_CEO, mensagens)
