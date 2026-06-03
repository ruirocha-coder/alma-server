# main.py v4 - Alma Server
# Arquitetura limpa: site-first via tool use, RAG como suplemento, historico multi-turn,
# quote mode que corta RAG, export CSV, top-K dinamico.
# -------------------------------------------------------------------

import os
import json
import asyncio
import csv
import re
import time
import logging
import uvicorn
from io import StringIO
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("alma")

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
XAI_MODEL   = os.getenv("XAI_MODEL", "grok-2-latest")
APP_VERSION = os.getenv("APP_VERSION", "alma-server/v4")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://boasafra.pt,https://interiorguider.com"
).split(",")

# mem0 (opcional)
MEM0_ENABLE  = os.getenv("MEM0_ENABLE", "0") in ("1", "true", "yes")
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "").strip()

# RAG / Qdrant (opcional)
QDRANT_URL        = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "alma_docs")

# Validacao no arranque
if not XAI_API_KEY:
    raise RuntimeError("XAI_API_KEY nao configurada. Servidor nao pode arrancar.")

# -------------------------------------------------------------------
# Site search URLs
# -------------------------------------------------------------------
SITE_SEARCH = {
    "interiorguider": "https://interiorguider.com/search.php?search_query={query}",
    "boasafra":       "https://boasafra.pt/pt-pt/?s={query}&post_type=product",
}
DEFAULT_BRAND = "interiorguider"

# -------------------------------------------------------------------
# FastAPI
# -------------------------------------------------------------------
app = FastAPI(title="Alma Server v4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass

# -------------------------------------------------------------------
# System Prompt
# -------------------------------------------------------------------
ALMA_SYSTEM = """
Es a Alma, assistente da Boa Safra e da Interior Guider (Boa Safra Lda).
A tua missao e apoiar a direcao (Rui Rocha) e a equipa com respostas claras, uteis e objetivas.

FONTES E PRIORIDADE
1. search_site - fonte principal para produtos, precos e disponibilidade (usa SEMPRE primeiro).
2. fetch_page - le uma pagina especifica quando tiveres o URL.
3. search_internal_docs - documentacao interna: politicas, guias, condicoes especiais (nao esta no site).

REGRAS
- Para qualquer questao sobre produtos ou precos: usa search_site antes de qualquer outra fonte.
- Nunca inventes precos, SKUs ou URLs.
- Se nao encontrares no site, diz claramente e sugere contacto direto.
- Se o utilizador nao indicar marca, assume Interior Guider.
- Em modo orcamento: apresenta um quadro limpo com nome, SKU, preco unitario, quantidade e subtotal.

ESTILO
- Claro e direto. Sem emojis. Sem conversa desnecessaria.
- Portugues de Portugal.
- Termina sempre com 1 proxima acao concreta quando relevante.

ORCAMENTOS
Formato obrigatorio:

Produto: [nome] - SKU: [ref]
Preco unitario: [valor] (IVA incluido)
Quantidade: [n]
Subtotal: [valor]

Total: [valor]
Nota: preco com IVA incluido; portes nao incluidos.
Link: [ver produto](URL)
"""

# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    user_id:  str = "anon"
    history:  Optional[List[Dict[str, str]]] = []   # multi-turn: [{role, content}, ...]
    brand:    Optional[str] = None                  # "boasafra" | "interiorguider"
    namespace: Optional[str] = None                 # RAG namespace

# -------------------------------------------------------------------
# mem0 (opcional, assincrono)
# -------------------------------------------------------------------
_mem0_client = None

if MEM0_ENABLE and MEM0_API_KEY:
    try:
        from mem0 import MemoryClient
        _mem0_client = MemoryClient(api_key=MEM0_API_KEY)
        log.info("[mem0] enabled")
    except Exception as e:
        log.warning(f"[mem0] disabled: {e}")


async def mem0_get(user_id: str) -> str:
    if not _mem0_client:
        return ""
    try:
        loop = asyncio.get_event_loop()
        memories = await loop.run_in_executor(
            None, lambda: _mem0_client.get_all(user_id=user_id)
        )
        texts = [m.get("memory", "") for m in (memories or [])[:5] if m.get("memory")]
        return "\n".join(texts)
    except Exception:
        return ""


async def mem0_store(user_id: str, messages: List[Dict[str, str]]):
    if not _mem0_client:
        return
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: _mem0_client.add(user_id=user_id, messages=messages)
        )
    except Exception:
        pass

# -------------------------------------------------------------------
# RAG / Qdrant (opcional, singleton)
# -------------------------------------------------------------------
_rag_model = None


def _get_rag_model():
    global _rag_model
    if _rag_model is None and QDRANT_URL:
        from sentence_transformers import SentenceTransformer
        _rag_model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("[rag] model loaded")
    return _rag_model


async def rag_search(query: str, namespace: Optional[str] = None) -> str:
    """Pesquisa documentacao interna no Qdrant."""
    if not QDRANT_URL:
        return ""
    try:
        import qdrant_client
        loop = asyncio.get_event_loop()

        def _search():
            model = _get_rag_model()
            vector = model.encode(query).tolist()
            client = qdrant_client.QdrantClient(
                url=QDRANT_URL, api_key=QDRANT_API_KEY
            )
            hits = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=vector,
                limit=5
            )
            texts = []
            for h in hits:
                payload = h.payload or {}
                if isinstance(payload, dict) and payload.get("text"):
                    texts.append(str(payload["text"]))
            return "\n\n".join(texts)

        return await loop.run_in_executor(None, _search)
    except Exception as e:
        log.warning(f"[rag] search failed: {e}")
        return ""

# -------------------------------------------------------------------
# Site scraping - fonte primaria de produtos e precos
# -------------------------------------------------------------------
_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AlmaBot/2.0)",
    "Accept-Language": "pt-PT,pt;q=0.9",
}


def _clean_html(html: str, max_lines: int = 250) -> str:
    """Remove tags HTML e devolve texto limpo e compacto."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines[:max_lines])


def _validate_url(url: str) -> bool:
    """So permite URLs dos dominios autorizados."""
    allowed = ["boasafra.pt", "interiorguider.com"]
    try:
        host = urlparse(url).netloc.lower().replace("www.", "")
        return any(d in host for d in allowed)
    except Exception:
        return False


async def site_search(query: str, brand: str = DEFAULT_BRAND) -> str:
    """Pesquisa produtos no site e devolve conteudo limpo."""
    brand = brand if brand in SITE_SEARCH else DEFAULT_BRAND
    url = SITE_SEARCH[brand].format(query=query.replace(" ", "+"))

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(url, headers=_HTTP_HEADERS)

        if r.status_code != 200:
            return f"Nao foi possivel aceder ao site (HTTP {r.status_code})."

        content = _clean_html(r.text)
        return f"[Resultados de {brand} para '{query}']\nURL pesquisado: {url}\n\n{content}"

    except Exception as e:
        return f"Erro ao pesquisar no site: {type(e).__name__}: {e}"


async def fetch_page(url: str) -> str:
    """Obtem o conteudo de uma pagina especifica do site."""
    if not _validate_url(url):
        return "URL nao autorizado (apenas boasafra.pt e interiorguider.com)."

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(url, headers=_HTTP_HEADERS)

        if r.status_code != 200:
            return f"Nao foi possivel aceder a pagina (HTTP {r.status_code})."

        return _clean_html(r.text)

    except Exception as e:
        return f"Erro ao aceder a pagina: {type(e).__name__}: {e}"

# -------------------------------------------------------------------
# Tool definitions para o Grok
# -------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_site",
            "description": (
                "Pesquisa produtos, precos e disponibilidade nos sites oficiais "
                "da Boa Safra e Interior Guider. "
                "Usa SEMPRE esta ferramenta primeiro para qualquer questao sobre produtos ou precos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Termo de pesquisa (ex: 'mesa zeal', 'cadeira orikomi')"
                    },
                    "brand": {
                        "type": "string",
                        "enum": ["interiorguider", "boasafra"],
                        "description": "Marca a pesquisar. Default: interiorguider."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Obtem o conteudo completo de uma pagina especifica do site.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL completo da pagina (apenas boasafra.pt ou interiorguider.com)"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_internal_docs",
            "description": (
                "Pesquisa documentacao interna: politicas comerciais, condicoes especiais, guias de equipa. "
                "Usa apenas quando a informacao nao esta disponivel no site publico."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Termo de pesquisa na documentacao interna"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# -------------------------------------------------------------------
# Tool dispatcher
# -------------------------------------------------------------------
async def run_tool(name: str, args: dict, brand: str = DEFAULT_BRAND) -> str:
    if name == "search_site":
        return await site_search(
            args.get("query", ""),
            args.get("brand", brand)
        )
    elif name == "fetch_page":
        return await fetch_page(args.get("url", ""))
    elif name == "search_internal_docs":
        return await rag_search(args.get("query", ""))
    return f"Ferramenta desconhecida: {name}"

# -------------------------------------------------------------------
# Detecao de intencao de orcamento
# -------------------------------------------------------------------
_QUOTE_KEYWORDS = (
    "orcament", "orcament", "cotacao", "cotacao",
    "proforma", "pro-forma", "preco", "preco",
    "quanto custa", "quanto fica"
)


def is_quote_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _QUOTE_KEYWORDS)

# -------------------------------------------------------------------
# Top-K dinamico para RAG
# -------------------------------------------------------------------
def decide_top_k(question: str, base: int = 5) -> int:
    q = (question or "").lower()
    n_products = len(re.findall(
        r"\b(mesa|cadeira|banco|cama|luminaria|sofa|tapete|espelho)\b", q
    ))
    if is_quote_intent(q) and n_products >= 2:
        return min(base + n_products * 2, 20)
    if is_quote_intent(q):
        return base + 4
    return base

# -------------------------------------------------------------------
# Grok call com agentic tool loop
# -------------------------------------------------------------------
async def call_grok(messages: List[Dict], brand: str = DEFAULT_BRAND) -> str:
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    max_iterations = 6

    for iteration in range(max_iterations):
        payload = {
            "model": XAI_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.5,
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=payload)

            if r.status_code != 200:
                return f"Erro ao chamar o Grok (HTTP {r.status_code}): {r.text[:400]}"

            data = r.json()
            choice  = data["choices"][0]
            message = choice["message"]
            finish  = choice.get("finish_reason", "")

            # Grok quer usar tools
            if finish == "tool_calls" and message.get("tool_calls"):
                messages.append(message)

                for tc in message["tool_calls"]:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"]["arguments"])

                    log.info(f"[tool] {tool_name}({tool_args})")
                    result = await run_tool(tool_name, tool_args, brand=brand)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
                continue

            # Resposta final
            return message.get("content") or "Sem resposta."

        except Exception as e:
            return f"Erro tecnico: {type(e).__name__}: {e}"

    return "Nao foi possivel obter resposta apos varias tentativas."

# -------------------------------------------------------------------
# Construir mensagens
# -------------------------------------------------------------------
async def build_messages(
    user_id: str,
    question: str,
    history: List[Dict[str, str]],
    brand: Optional[str],
    namespace: Optional[str],
) -> List[Dict]:

    messages: List[Dict] = [{"role": "system", "content": ALMA_SYSTEM}]

    # Marca activa
    active_brand = brand or DEFAULT_BRAND
    messages.append({
        "role": "system",
        "content": f"Marca activa: {active_brand}. Pesquisa sempre nesta marca por defeito."
    })

    # Memoria de sessoes anteriores (mem0)
    memory = await mem0_get(user_id)
    if memory:
        messages.append({
            "role": "system",
            "content": f"Contexto de sessoes anteriores (mem0):\n{memory}"
        })

    # Em modo orcamento: RAG cortado (evita links corporativos a poluir o orcamento)
    quote_mode = is_quote_intent(question)

    # RAG: so fora de modo orcamento e apenas se Qdrant disponivel
    if not quote_mode and QDRANT_URL:
        top_k = decide_top_k(question)
        rag_content = await rag_search(question, namespace)
        if rag_content:
            messages.append({
                "role": "system",
                "content": (
                    f"Documentacao interna (RAG -- NAO usar para precos; "
                    f"apenas para politicas e guias internos):\n{rag_content}"
                )
            })

    # Historico da conversa actual (multi-turn)
    # Passa as ultimas 20 mensagens para nao estourar contexto
    for msg in (history or [])[-20:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # Pergunta actual
    messages.append({"role": "user", "content": question})

    return messages

# -------------------------------------------------------------------
# Rotas
# -------------------------------------------------------------------
@app.get("/")
async def serve_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except Exception:
        return HTMLResponse("<h1>index.html nao encontrado</h1>", status_code=404)


@app.get("/alma-chat")
@app.get("/alma-chat/")
async def serve_chat():
    path = os.path.join(os.getcwd(), "alma-chat.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return HTMLResponse("<h1>alma-chat.html nao encontrado</h1>", status_code=404)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "model": XAI_MODEL,
        "mem0": MEM0_ENABLE,
        "rag": bool(QDRANT_URL),
    }


@app.get("/status")
def status():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "model": XAI_MODEL,
        "mem0": {"enabled": MEM0_ENABLE, "ready": bool(_mem0_client)},
        "rag":  {"enabled": bool(QDRANT_URL), "collection": QDRANT_COLLECTION},
        "sites": list(SITE_SEARCH.keys()),
        "endpoints": {
            "ask":        "POST /ask",
            "health":     "GET /health",
            "budget_csv": "POST /budget/csv",
            "ping_grok":  "GET /ping_grok",
        }
    }


@app.get("/ping_grok")
async def ping_grok():
    try:
        msgs = [{"role": "user", "content": "Diz apenas: pong"}]
        reply = await call_grok(msgs)
        return {"ok": True, "reply": reply}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/ask")
async def ask(req: AskRequest):
    question  = (req.question or "").strip()
    user_id   = (req.user_id  or "anon").strip() or "anon"
    brand     = req.brand or DEFAULT_BRAND
    history   = req.history or []
    namespace = req.namespace

    if not question:
        return JSONResponse({"answer": "Coloca a tua pergunta em 'question'."})

    log.info(f"[ask] user={user_id} brand={brand} quote={is_quote_intent(question)} q={question!r}")

    try:
        messages = await build_messages(user_id, question, history, brand, namespace)
        answer   = await call_grok(messages, brand=brand)

        # Guarda em mem0 se resposta valida
        if answer and not answer.startswith("Erro"):
            await mem0_store(user_id, [
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ])

        return JSONResponse({"answer": answer})

    except Exception as e:
        log.exception("Erro no /ask")
        return JSONResponse({"answer": f"Erro interno: {type(e).__name__}: {e}"})

# -------------------------------------------------------------------
# Export CSV de Orcamentos
# -------------------------------------------------------------------
def _safe_float(v, default=0.0) -> float:
    try:
        s = str(v).replace("\u20ac", "").replace(",", ".").strip()
        return float(s)
    except Exception:
        return float(default)


def _fmt_money(x: float) -> str:
    return f"{x:.2f}"


@app.post("/budget/csv")
async def budget_csv(request: Request):
    """
    Body:
    {
      "mode": "public" | "pro",
      "iva_pct": 23,
      "rows": [
        {
          "ref": "SKU01",
          "descricao": "Mesa Zeal",
          "quant": 2,
          "preco_uni": 850.00,
          "desc_pct": 0,
          "dim": "160x80xH75",
          "material": "Carvalho",
          "marca": "Interior Guider",
          "link": "https://interiorguider.com/..."
        }
      ]
    }
    """
    data    = await request.json()
    mode    = (data.get("mode") or "public").lower().strip()
    iva_pct = _safe_float(data.get("iva_pct", 23))
    rows    = data.get("rows") or []

    if mode not in ("public", "pro"):
        return PlainTextResponse("mode deve ser 'public' ou 'pro'", status_code=400)
    if not rows:
        return PlainTextResponse("rows vazio", status_code=400)

    headers = (
        ["REF.", "DESIGNACAO / MATERIAL / ACABAMENTO / COR", "QUANT.", "PRECO UNI.", "DESC.", "TOTAL S/IVA"]
        if mode == "public" else
        ["REF.", "DESIGNACAO / MATERIAL / ACABAMENTO / COR", "QUANT.", "PRECO UNI.", "DESC.", "TOTAL C/IVA"]
    )

    sio = StringIO()
    writer = csv.writer(sio)
    writer.writerow(headers)

    for r in rows:
        ref       = (r.get("ref") or "").strip()
        quant     = int(r.get("quant") or 1)
        preco_uni = _safe_float(r.get("preco_uni"), 0.0)
        desc_pct  = _safe_float(r.get("desc_pct"), 0.0)

        desc_main   = (r.get("descricao") or "").strip() or "Produto"
        extra_lines = []
        if r.get("dim"):      extra_lines.append(f"Dimensoes: {r['dim']}")
        if r.get("material"): extra_lines.append(f"Material: {r['material']}")
        if r.get("marca"):    extra_lines.append(f"Marca: {r['marca']}")
        if r.get("link"):     extra_lines.append(f"Link: {r['link']}")

        full_desc = desc_main + ("\n" + "\n".join(extra_lines) if extra_lines else "")
        total_si  = quant * preco_uni * (1.0 - desc_pct / 100.0)
        total_col = _fmt_money(total_si if mode == "public" else total_si * (1.0 + iva_pct / 100.0))

        writer.writerow([
            ref,
            full_desc,
            str(quant),
            _fmt_money(preco_uni),
            (f"{desc_pct:.0f}%" if desc_pct else ""),
            total_col,
        ])

    csv_bytes = sio.getvalue().encode("utf-8-sig")
    fname     = f"orcamento_{mode}_{int(time.time())}.csv"
    fpath     = os.path.join("/tmp", fname)
    with open(fpath, "wb") as f:
        f.write(csv_bytes)

    return FileResponse(fpath, media_type="text/csv", filename=fname)

# -------------------------------------------------------------------
# RAG: ingest endpoints (para alimentar documentacao interna)
# -------------------------------------------------------------------
@app.post("/rag/ingest-text")
async def rag_ingest_text(request: Request):
    if not QDRANT_URL:
        return {"ok": False, "error": "RAG nao disponivel (QDRANT_URL nao configurado)"}
    try:
        from rag_client import ingest_text
        data      = await request.json()
        title     = (data.get("title") or "").strip()
        text      = (data.get("text")  or "").strip()
        namespace = (data.get("namespace") or "default").strip()
        if not title or not text:
            return {"ok": False, "error": "Falta title ou text"}
        return ingest_text(title=title, text=text, namespace=namespace)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/rag/ingest-url")
async def rag_ingest_url(request: Request):
    if not QDRANT_URL:
        return {"ok": False, "error": "RAG nao disponivel"}
    try:
        from rag_client import ingest_url
        data       = await request.json()
        page_url   = (data.get("page_url") or "").strip()
        namespace  = (data.get("namespace") or "default").strip()
        deadline_s = int(data.get("deadline_s") or 55)
        if not page_url:
            return {"ok": False, "error": "Falta page_url"}
        return ingest_url(page_url, namespace=namespace, deadline_s=deadline_s)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/rag/search")
async def rag_search_endpoint(request: Request):
    if not QDRANT_URL:
        return {"ok": False, "error": "RAG nao disponivel"}
    try:
        data  = await request.json()
        query = (data.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "Falta query"}
        result = await rag_search(query)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# -------------------------------------------------------------------
# Local dev
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
