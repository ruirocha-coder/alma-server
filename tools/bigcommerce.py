import httpx, os, time

STORE = os.environ["BIGCOMMERCE_STORE_HASH"]
TOKEN = os.environ["BIGCOMMERCE_ACCESS_TOKEN"]
BASE = f"https://api.bigcommerce.com/stores/{STORE}"
HEADERS = {"X-Auth-Token": TOKEN, "Accept": "application/json"}

_cache = {}  # {chave: (timestamp, dados)}
TTL = {"catalogo": 900, "encomendas": 300}  # segundos

def _get(url, params=None, cache_key=None, ttl=900):
    if cache_key and cache_key in _cache:
        ts, dados = _cache[cache_key]
        if time.time() - ts < ttl:
            return dados
    r = httpx.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    dados = r.json()
    if cache_key:
        _cache[cache_key] = (time.time(), dados)
    return dados

def procurar_produtos(termo: str, limite: int = 10):
    """Pesquisa no catálogo. Devolve nome, preço, custo, stock, URL."""
    dados = _get(f"{BASE}/v3/catalog/products",
                 params={"keyword": termo, "limit": limite,
                         "include_fields": "name,price,cost_price,inventory_level,custom_url,sku"})
    return dados.get("data", [])

def encomendas_recentes(dias: int = 30):
    """Encomendas dos últimos N dias (API V2 de orders)."""
    from datetime import datetime, timedelta
    desde = (datetime.utcnow() - timedelta(days=dias)).strftime("%Y-%m-%dT%H:%M:%S")
    return _get(f"{BASE}/v2/orders",
                params={"min_date_created": desde, "limit": 250},
                cache_key=f"orders_{dias}", ttl=TTL["encomendas"])

def resumo_vendas(dias: int = 30):
    """Total de vendas, nº de encomendas, ticket médio."""
    orders = encomendas_recentes(dias) or []
    total = sum(float(o["total_inc_tax"]) for o in orders)
    n = len(orders)
    return {"periodo_dias": dias, "total_eur": round(total, 2),
            "n_encomendas": n, "ticket_medio": round(total / n, 2) if n else 0}
    TOOLS_CEO = [
    {
        "name": "procurar_produtos",
        "description": "Pesquisa produtos no catálogo BigCommerce por palavra-chave. Devolve preço de venda, preço de custo, stock e URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "termo": {"type": "string"},
                "limite": {"type": "integer", "default": 10}
            },
            "required": ["termo"]
        }
    },
    {
        "name": "resumo_vendas",
        "description": "Resumo de vendas: total, número de encomendas e ticket médio nos últimos N dias.",
        "input_schema": {
            "type": "object",
            "properties": {"dias": {"type": "integer", "default": 30}},
            "required": []
        }
    }
]
