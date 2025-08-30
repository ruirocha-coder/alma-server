# mem0.py
import os
import requests

MEM0_API = "https://api.mem0.ai/v1"
MEM0_KEY = os.getenv("MEM0_API_KEY")
MEM0_ENABLED = os.getenv("MEM0_ENABLED", "true").lower() == "true"
MEM0_HEADERS = {"Authorization": f"Bearer {MEM0_KEY}", "Content-Type": "application/json"}

def _enabled():
    return MEM0_ENABLED and bool(MEM0_KEY)

def search_memories(user_id: str, query: str, limit: int = 5, timeout_s: float = 3.5):
    """Busca memórias curtas relevantes (rápido, baixo impacto de latência)."""
    if not _enabled():
        return []
    try:
        r = requests.post(
            f"{MEM0_API}/memories/search",
            headers=MEM0_HEADERS,
            json={"user_id": user_id, "query": query, "limit": limit},
            timeout=timeout_s,
        )
        if not r.ok:
            return []
        data = r.json().get("data", [])
        return [m.get("text", "") for m in data if m.get("text")]
    except Exception:
        return []

def add_memory(user_id: str, text: str, ttl_days: int | None = None, timeout_s: float = 2.0):
    """Guarda uma memória simples de forma non-blocking (não falha a resposta)."""
    if not _enabled():
        return
    if not text:
        return
    try:
        payload = {"user_id": user_id, "text": text}
        if ttl_days is None:
            try:
                ttl_days = int(os.getenv("MEM0_TTL_DAYS", "7"))
            except Exception:
                ttl_days = 7
        payload["ttl_days"] = ttl_days
        # fire-and-forget com timeout curto
        requests.post(f"{MEM0_API}/memories", headers=MEM0_HEADERS, json=payload, timeout=timeout_s)
    except Exception:
        pass
