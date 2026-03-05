# main.py — Alma Server (LLM-first, Mem0 optional, RAG optional, no internal catalog)
# -----------------------------------------------------------------------------------
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

import os
import re
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests

# -----------------------------------------------------------------------------------
# App / CORS / Logging
# -----------------------------------------------------------------------------------
app = FastAPI(title="Alma Server (LLM-first)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("alma")

APP_VERSION = os.getenv("APP_VERSION", "alma-server/llm-first-1")

# -----------------------------------------------------------------------------------
# ENV (do not change your Railway vars — we only read them)
# -----------------------------------------------------------------------------------
# LLM (xAI / Grok)
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4")
XAI_TIMEOUT_S = float(os.getenv("XAI_TIMEOUT_S", "45"))

# Mem0 (short-term memory)
MEMO_ENABLE = (os.getenv("MEMO_ENABLE", "true").strip().lower() in ("1", "true", "yes", "on"))
MEMO_API_KEY = os.getenv("MEMO_API_KEY", "")
MEMO_BASE_URL = os.getenv("MEMO_BASE_URL", "https://api.mem0.ai/v1")

# RAG (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "alma_docs")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))
RAG_CONTEXT_TOKEN_BUDGET = int(os.getenv("RAG_CONTEXT_TOKEN_BUDGET", "1800"))

# Defaults
DEFAULT_SITE = "https://interiorguider.com"
BOASAFRA_SITE = os.getenv("BOASAFRA_SITE", "https://boasafra.pt")


# ---------------------------------------------------------------------------------------
# Prompt nuclear da Alma (sem catálogo interno; site-first; verificação leve)
# ---------------------------------------------------------------------------------------
ALMA_SYSTEM = """
És a Alma, inteligência da Boa Safra Lda (Boa Safra + Interior Guider).
Missão: apoiar Rui Rocha e a equipa com respostas úteis, objetivas e calmas.

ESTILO (estrito)
- Clareza e concisão: vai direto ao ponto.
- Empatia só quando houver sinais claros de stress.
- Sem small talk, sem emojis, sem tom efusivo.
- Termina com 1 próxima ação concreta.
- Se houver Tom de Voz no RAG, adota-o rigorosamente.

DEFAULT DE MARCA
- Se o utilizador não indicar marca, assume Interior Guider (interiorguider.com).
- Se mencionar “Boa Safra/boasafra/mesa family/Family”, assume Boa Safra (boasafra.pt).
- Se houver ambiguidade, faz 1 pergunta curta para fixar a marca.

FONTES (prioridade)
1) Site oficial (interiorguider.com / boasafra.pt) via RAG/crawl: fonte principal para produtos, variantes, disponibilidade e preços publicados.
2) RAG corporativo interno (PDFs, notas, políticas): para contexto e procedimentos.
3) LLM base: raciocínio, síntese, estratégia, redação — mas nunca como fonte de factos comerciais.

PROIBIDO
- Inventar preços, SKUs, nomes de variantes, disponibilidade ou links.
- “Adivinhar” informação de produto com base em padrões.
- Criar URLs ou slugs. Só usar links que existam no site (ou recuperados pelo RAG).

──────────────────────────────────────────────────────────────────────────────
MODO PREÇOS / ORÇAMENTOS (SITE-FIRST + VERIFICAÇÃO LEVE)

Ativa quando o pedido envolve: orçamento, cotação, preço, proforma.

REGRA DE OURO
- Só podes afirmar preço/SKU/variante se a evidência estiver presente no conteúdo recuperado do site
  (ex.: página do produto, variações, JSON/HTML visível no crawl).
- Se não houver evidência suficiente, diz isso claramente e pede o mínimo para confirmar.

VERIFICAÇÃO LEVE (como operar)
A) Tentativa 1 — Identificação no site
- Procura o produto pelo nome tal como foi escrito pelo utilizador.
- Se encontrares 1 página de produto plausível:
  • extrai apenas dados visíveis (nome, preço, variantes listadas, disponibilidade quando existir).
  • não “completes” variantes que não apareçam.

B) Variantes
- Se o site listar variantes/opções:
  • lista TODAS as variantes que estejam visíveis (até 40), sem resumir.
  • pede ao utilizador que escolha 1 (nome exato da opção ou referência/SKU se existir).
- Se o site não listar variantes:
  • assume “sem variantes visíveis” (não concluas que não existem).

C) Se não conseguires validar
- Resposta padrão:
  “Não consegui validar no site o preço/variante deste item com a informação fornecida.”
- Pede apenas um destes:
  • link do produto, ou
  • referência/SKU, ou
  • screenshot/trecho onde o preço apareça.

D) Estimativas (só com consentimento explícito)
- Se o utilizador pedir uma estimativa sem validação no site:
  • pede confirmação explícita: “Quer uma estimativa não vinculativa?”
  • se confirmar, dá estimativa com aviso:
    “Estimativa não vinculativa; sujeita a validação no site/fornecedor.”
  • nunca inventes SKU/ref/variantes mesmo em estimativa.

FORMATO DE RESPOSTA (orçamentos)
- Título curto.
- Itens: Nome | Preço unitário (se validado) | Qtd | Subtotal
- Total (se todos os itens tiverem preço validado)
- Nota: “preço com IVA incluído; portes não incluídos.” (a menos que o site indique o contrário)
- Link único (se existir): [ver produto](URL_CANONICO)
- Fecha com 1 próxima ação concreta.

POLÍTICA DE LINKS (rígida)
- No máximo 1 link por resposta.
- Só links recuperados do site (nunca inventados).
- Sempre em Markdown: [ver produto](URL)
- Nunca repetir URL em texto.

AUTO-CHECK (antes de responder com números)
- O preço está presente no conteúdo do site recuperado? Se não, não publicar preço.
- A variante está explicitamente listada? Se não, não afirmar variante.
- O link existe e é canónico? Se não, não incluir link.

Objetivo: rapidez + segurança. Se não há evidência, pedes o mínimo para validar e avançar.
"""

# -----------------------------------------------------------------------------------
# Helpers: brand/site inference
# -----------------------------------------------------------------------------------
def infer_site(question: str) -> Tuple[str, str]:
    q = (question or "").lower()
    if "boa safra" in q or "boasafra" in q:
        return "boasafra", BOASAFRA_SITE
    # default
    return "interiorguider", DEFAULT_SITE

def looks_like_pricing_intent(question: str) -> bool:
    q = (question or "").lower()
    triggers = [
        "orçamento", "orcamento", "preço", "preco", "quanto custa", "valor",
        "cotação", "cotacao", "proposta", "quote", "budget"
    ]
    return any(t in q for t in triggers)

# -----------------------------------------------------------------------------------
# Helpers: simple website validation (very light)
# - Uses search.php when site is interiorguider.com (as you already have)
# - Scrapes candidate product URLs and tries to extract name/price/variants heuristically
# -----------------------------------------------------------------------------------
UA = {"User-Agent": "AlmaBot/1.0 (+https://interiorguider.com)"}

def _safe_get(url: str, timeout: float = 20) -> Optional[str]:
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code >= 200 and r.status_code < 300:
            return r.text
        return None
    except Exception:
        return None

def extract_urls_from_html(html: str, base_host: str) -> List[str]:
    if not html:
        return []
    # very simple href extraction
    hrefs = re.findall(r'href="([^"]+)"', html, flags=re.I)
    urls = []
    for h in hrefs:
        if h.startswith("//"):
            h = "https:" + h
        if h.startswith("/"):
            h = base_host.rstrip("/") + h
        if h.startswith(base_host) and "search.php" not in h:
            urls.append(h.split("#")[0])
    # de-dup preserve order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:8]

def guess_product_name(html: str) -> Optional[str]:
    if not html:
        return None
    # try <h1>
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.I | re.S)
    if m:
        name = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        name = re.sub(r"\s+", " ", name)
        return name[:200] if name else None
    return None

def guess_price_eur(html: str) -> Optional[float]:
    if not html:
        return None
    # Try common patterns: content="539.00" etc
    patterns = [
        r'property="product:price:amount"\s+content="([\d\.,]+)"',
        r'"price"\s*:\s*"([\d\.,]+)"',
        r'data-product-price(?:-with-tax)?="([\d\.,]+)"',
        r'([\d]{2,5}[\.,]\d{2})\s*€',
    ]
    for p in patterns:
        m = re.search(p, html, flags=re.I)
        if m:
            raw = m.group(1).replace(".", "").replace(",", ".")
            try:
                v = float(raw)
                # sanity
                if 1.0 <= v <= 500000.0:
                    return v
            except Exception:
                pass
    return None

def guess_variants(html: str) -> List[str]:
    if not html:
        return []
    # heuristic: option labels inside selects
    opts = re.findall(r"<option[^>]*>([^<]{1,80})</option>", html, flags=re.I)
    cleaned = []
    for o in opts:
        t = o.strip()
        t = re.sub(r"\s+", " ", t)
        if not t:
            continue
        # ignore placeholders
        if t.lower() in ("choose an option", "select", "choose", "—", "-", "none", "n/a"):
            continue
        # avoid obvious non-variant numbers
        if re.fullmatch(r"\d+", t):
            continue
        cleaned.append(t)

    # de-dup, but keep reasonable
    seen = set()
    out = []
    for v in cleaned:
        vl = v.lower()
        if vl not in seen:
            seen.add(vl)
            out.append(v)
    # Usually HTML has a lot of options (qty etc). Keep only first chunk.
    return out[:12]

def site_search_candidates(site: str, query: str) -> List[Dict[str, Any]]:
    # For InteriorGuider, search endpoint exists. For others, we do nothing (still LLM-first).
    if "interiorguider.com" not in site:
        return []

    q = (query or "").strip()
    if not q:
        return []

    search_url = f"{site.rstrip('/')}/search.php?search_query={quote_plus(q)}"
    html = _safe_get(search_url, timeout=20)
    if not html:
        return []

    urls = extract_urls_from_html(html, base_host=site)
    candidates: List[Dict[str, Any]] = []
    for u in urls:
        ph = _safe_get(u, timeout=20)
        if not ph:
            continue
        name = guess_product_name(ph) or None
        price = guess_price_eur(ph)
        variants = guess_variants(ph)
        candidates.append({
            "url": u,
            "name": name,
            "price_eur": price,
            "variants_guess": variants,
        })

    # sort: those with a price first, then with a name
    candidates.sort(key=lambda c: (c.get("price_eur") is None, c.get("name") is None))
    return candidates[:5]

# -----------------------------------------------------------------------------------
# Mem0 (optional)
# -----------------------------------------------------------------------------------
def mem0_headers() -> Dict[str, str]:
    if not MEMO_API_KEY:
        return {}
    return {"Authorization": f"Bearer {MEMO_API_KEY}", "Content-Type": "application/json"}

def mem0_get_memories(user_id: str, limit: int = 6) -> List[Dict[str, Any]]:
    if not (MEMO_ENABLE and MEMO_API_KEY and user_id):
        return []
    try:
        url = f"{MEMO_BASE_URL.rstrip('/')}/memories/?user_id={quote_plus(user_id)}"
        r = requests.get(url, headers=mem0_headers(), timeout=15)
        if r.status_code == 200:
            data = r.json()
            items = data.get("memories") or data.get("data") or data
            if isinstance(items, list):
                return items[:limit]
        return []
    except Exception:
        return []

def mem0_add_memory(user_id: str, text: str) -> None:
    if not (MEMO_ENABLE and MEMO_API_KEY and user_id and text):
        return
    try:
        url = f"{MEMO_BASE_URL.rstrip('/')}/memories/"
        payload = {"user_id": user_id, "memory": text}
        requests.post(url, headers=mem0_headers(), data=json.dumps(payload), timeout=15)
    except Exception:
        pass

# -----------------------------------------------------------------------------------
# RAG (optional): Qdrant + OpenAI embeddings
# NOTE: If any part fails, we proceed without RAG.
# -----------------------------------------------------------------------------------
def openai_embed(texts: List[str]) -> Optional[List[List[float]]]:
    if not OPENAI_API_KEY:
        return None
    try:
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": EMBEDDING_MODEL, "input": texts}
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        out = []
        for item in data.get("data", []):
            out.append(item.get("embedding"))
        if out and all(isinstance(v, list) for v in out):
            return out
        return None
    except Exception:
        return None

def qdrant_search(vector: List[float], top_k: int) -> List[Dict[str, Any]]:
    if not (QDRANT_URL and QDRANT_API_KEY and QDRANT_COLLECTION):
        return []
    try:
        base = QDRANT_URL.rstrip("/")
        # expects Qdrant REST base
        url = f"{base}/collections/{QDRANT_COLLECTION}/points/search"
        headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
        payload = {"vector": vector, "limit": int(top_k), "with_payload": True}
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=25)
        if r.status_code != 200:
            return []
        res = r.json()
        return res.get("result", []) or []
    except Exception:
        return []

def rag_context(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (context_text, raw_hits)
    """
    try:
        emb = openai_embed([question])
        if not emb:
            return "", []
        hits = qdrant_search(emb[0], top_k=RAG_TOP_K)
        if not hits:
            return "", []
        chunks = []
        for h in hits:
            payload = h.get("payload") or {}
            text = payload.get("text") or payload.get("chunk") or payload.get("content") or ""
            src = payload.get("source") or payload.get("url") or payload.get("file") or ""
            if text:
                text = re.sub(r"\s+", " ", str(text)).strip()
                if src:
                    chunks.append(f"- ({src}) {text}")
                else:
                    chunks.append(f"- {text}")
        ctx = "\n".join(chunks)
        # crude budget: cut chars
        if len(ctx) > (RAG_CONTEXT_TOKEN_BUDGET * 4):
            ctx = ctx[: (RAG_CONTEXT_TOKEN_BUDGET * 4)]
        return ctx, hits
    except Exception:
        return "", []

# -----------------------------------------------------------------------------------
# LLM call (xAI chat completions)
# -----------------------------------------------------------------------------------
def xai_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    if not XAI_API_KEY:
        return "Erro: XAI_API_KEY não configurada no servidor."
    url = f"{XAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": XAI_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=XAI_TIMEOUT_S)
    if r.status_code >= 400:
        return f"Erro ao chamar o Grok: {r.status_code} {r.text}"
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data)[:2000]

# -----------------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return f"ok ({APP_VERSION})"

@app.get("/alma-chat", response_class=HTMLResponse)
def alma_chat():
    # Minimal page so your existing frontend can still load /alma-chat if it expects it
    return """
<!doctype html>
<html lang="pt">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alma Server</title></head>
<body style="font-family:system-ui;margin:24px">
<h2>Alma Server (LLM-first)</h2>
<p>Endpoint principal: <code>POST /ask</code></p>
</body>
</html>
""".strip()

@app.post("/ask")
async def ask(req: Request):
    """
    Expected JSON (compatible with your current frontend patterns):
    {
      "user_id": "s123",
      "question": "..."
    }
    """
    body = await req.json()
    user_id = (body.get("user_id") or "anon").strip()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"answer": "Escreva uma pergunta."})

    brand, site = infer_site(question)

    # --- mem0 context (optional) ---
    memories = mem0_get_memories(user_id, limit=6)
    mem_text = ""
    if memories:
        # keep it light; mem0 payloads differ. We take a conservative rendering.
        lines = []
        for m in memories:
            if isinstance(m, dict):
                t = m.get("memory") or m.get("text") or m.get("content") or ""
                t = re.sub(r"\s+", " ", str(t)).strip()
                if t:
                    lines.append(f"- {t}")
        if lines:
            mem_text = "Memória recente (pode estar incompleta):\n" + "\n".join(lines)

    # --- RAG context (optional) ---
    rag_used = False
    rag_txt, rag_hits = rag_context(question)
    if rag_txt:
        rag_used = True

    # --- light validation on site for pricing intent (optional) ---
    candidates: List[Dict[str, Any]] = []
    if looks_like_pricing_intent(question):
        # Try to extract a product-ish query (very light): remove generic words
        q = re.sub(r"\b(orçamento|orcamento|preço|preco|quanto custa|valor|cotação|cotacao|proposta)\b", "", question, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip()
        candidates = site_search_candidates(site, q) if q else []

    candidates_block = ""
    if candidates:
        # Provide evidence to LLM; do not force a format
        lines = []
        for c in candidates:
            name = c.get("name") or "(sem nome)"
            url = c.get("url")
            price = c.get("price_eur")
            variants = c.get("variants_guess") or []
            line = f"- {name}"
            if price is not None:
                line += f" | preço detetado: {price:.2f}€"
            if url:
                line += f" | {url}"
            if variants:
                line += f" | variantes (heurística): {', '.join(variants[:8])}"
            lines.append(line)
        candidates_block = "Candidatos encontrados no site (validação leve):\n" + "\n".join(lines)

    # --- build messages ---
    # Note: we explicitly tell the default site/brand context
    context_note = f"Contexto atual: marca={brand}. Se não houver indicação em contrário, assume interiorguider.com."

    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": ALMA_SYSTEM},
        {"role": "system", "content": context_note},
    ]
    if mem_text:
        msgs.append({"role": "system", "content": mem_text})
    if rag_txt:
        msgs.append({"role": "system", "content": "RAG (documentos internos relevantes):\n" + rag_txt})
    if candidates_block:
        msgs.append({"role": "system", "content": candidates_block})

    msgs.append({"role": "user", "content": question})

    # --- call LLM ---
    answer = xai_chat(msgs, temperature=0.2).strip()

    # --- store minimal memory (optional) ---
    # Keep it small and useful.
    mem0_add_memory(user_id, f"Pergunta: {question}\nResposta: {answer[:600]}")

    return JSONResponse({
        "answer": answer,
        "meta": {
            "brand": brand,
            "site_default": site,
            "rag_used": rag_used,
            "candidates_used": len(candidates),
            "version": APP_VERSION,
        }
    })

# -----------------------------------------------------------------------------------
# Local dev entrypoint (Railway uses its own command, but this is safe)
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
