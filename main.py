# main.py  v3— Alma Server (RAG + Memória; top-k dinâmico; mini-pesquisa; injeção e fallback de links; Consola RAG; utilitários) 
# ---------------------------------------------------------------------------------------

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles


import os
import requests
import logging
import uvicorn
import time
import re
import csv
from io import StringIO
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, urlunparse
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------------------
# FastAPI & CORS
# ---------------------------------------------------------------------------------------
app = FastAPI(title="Alma Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- servir a pasta static (onde fica o alma.png) ---
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------------------
# Logging / Versão
# ---------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("alma")

APP_VERSION = os.getenv("APP_VERSION", "alma-server/rag+mem+links-fallback-1")

# ---------------------------------------------------------------------------------------
# Prompt nuclear da Alma
# ---------------------------------------------------------------------------------------
ALMA_MISSION = """
És a Alma, inteligência da Boa Safra Lda (Boa Safra + Interior Guider).
A tua missão é apoiar a direção (Rui Rocha) e a equipa para que a empresa
sobreviva e prospere, com respostas úteis, objetivas e calmas.

Estilo (estrito)
- Clareza e concisão: vai direto ao ponto. 
- Empatia sob medida: só comenta o estado emocional quando houver sinais de stress.
- Vocabulário disciplinado; evita entusiasmismos.
- Seguimento: termina com 1 próxima ação concreta.
- Usar o Tom de Voz do RAG quando presente, adota-o rigorosamente nas respostas.

Proibido
- Small talk, emojis ou tom efusivo.
- Inventar links, nomes de variantes, SKUs ou preços.
- Usar placeholders (“exemplo1”, “não especificado”, etc.).

Fontes e prioridade
- Catálogo interno (CSV → tabela catalog_items): fonte principal e prioritária.
- RAG corporativo (sites, PDFs e marcas): só quando o utilizador pedir explicitamente opções fora do catálogo interno (nunca para orçamentos).
- LLM base: livre para raciocínio, estratégia e contexto externo.

Funções
1) Estratégia e apoio comercial (produtos, prazos, preços).
2) Especialista e assistente no método psicoestético.
3) Método e procedimentos (quando relevante).
4) RAG + Grok; se faltar evidência, diz o que falta e o próximo passo.

──────────────────────────────────────────────────────────────────────────────
MODO ORÇAMENTO / PREÇOS  (CATÁLOGO-ONLY, SEM FALLOUTS)
- Sempre que o pedido for orçamento, cotação, preço, proforma ou semelhante:
  • **É OBRIGATÓRIO usar apenas os dados do catálogo interno (catalog_items).**
  • **PROIBIDO** consultar RAG, completar nomes/cores, inferir SKUs ou preços.
  • Se não houver dados suficientes no catálogo, responde claramente:
    “Sem dados no catálogo interno para este item.” e pede **apenas** a
    informação mínima para localizar no catálogo (ex.: nome do produto ou SKU).
  • Nunca “estimar”, nunca “assumir”, nunca mostrar opções fora do catálogo.

MAPA DE CAMPOS (OBRIGATÓRIO)
- SKU / Código / Ref  → usar **catalog_items.ref** (match exato, sem alterações).
- Preço (IVA incluído) → usar **catalog_items.price** (valor final do catálogo).
- Nome                → usar **catalog_items.name**.
- Variante (texto)    → usar **catalog_items.variant_attrs**; se vazio, ler a linha “Variante:” no summary (após limpar HTML).
- Disponibilidade     → usar **catalog_items.availability** (se existir) ou o texto “Disponibilidade:”/“Prazo de entrega:” no summary.
- URL                 → usar **catalog_items.url** (para variantes contém “#sku=”).
- O campo **summary pode conter HTML**: extrai apenas o **texto visível** antes de procurar “Variante:” ou disponibilidade.

REGRAS DE CATÁLOGO (CSV → tabela catalog_items)

1) PRIORIDADE DE FONTE
- Usa SEMPRE primeiro os dados do catálogo interno.
- Dentro do catálogo: dá prioridade às **VARIANTES** (linhas cuja URL contém “#sku=”).
- Só usa o produto base (URL sem “#sku=”) se não houver variante identificável.

2) DETEÇÃO E LISTAGEM DE VARIANTES (OBRIGATÓRIO)
- Considera que existem variantes quando houver **≥2 linhas do mesmo produto**. Para agrupar:
  • Preferir o mesmo **URL-pai** (parte antes de “#sku=”), quando existir; e
  • Aceitar também o mesmo **nome base** normalizado (nome sem o sufixo de variante) e mesma **marca**,
    para cobrir casos em que o SKU derivado não contém o SKU-pai.
- Ao listar variantes, mostrar **exclusivamente** as linhas do catálogo desse grupo. Para cada uma:
  **Variante (variant_attrs)** + **SKU (ref)** + **Preço (price)** + **Link (url com #sku= se existir; senão o url dessa linha)**.
- NÃO completes nomes/cores via RAG. Se faltar algum campo no catálogo, escreve literalmente “(sem dado)”.
- Só depois de listar as variantes do catálogo podes acrescentar:
  “Existem também outras opções não listadas no catálogo interno; confirme disponibilidade junto dos serviços da empresa.” (sem preços).
   
   REGRA DURA — LISTAGEM COMPLETA
	•	Sempre que o utilizador pedir “variantes / opções / cores / tamanhos / versões”, listar todas as variantes existentes no catálogo interno para esse produto (até 40).
	•	Nunca resumir para 2–3 opções “principais”.

3) IDENTIFICAÇÃO DE VARIANTE (ALGORITMO CANÓNICO)
A) Extrai pistas (prioridade):
   1. SKU/ref explícita (ex.: ORK.02.03, #sku=ORK.02.03).
   2. Texto de opção/variante (ex.: “Soft Indigo”, “Simples Branco 1M”, “cor=Soft Indigo”).
   3. Nome do produto (ex.: “Dublexo Eik Sofá Cama”, “Orikomi Plus Taupe”).
B) Consulta catálogo:
   - Se houver **ref** → faz match **EXATO** em catalog_items.ref → usa essa linha e termina.
   - Sem ref → determina o grupo do produto (ver 2) e filtra **só** as linhas desse grupo.
     Dentro desse conjunto, procura a correspondência no **variant_attrs** e/ou na linha “Variante: …”
     (após limpar HTML), usando **match literal** com normalização:
       · minúsculas, sem acentos, sem pontuação/hífens redundantes, espaços colapsados;
       · aceita padrão “chave=valor” (ex.: “cor=soft indigo”).
     **Não** traduzir nem usar sinónimos (ex.: “granite” ≠ “charcoal”).
   - Se houver 2–6 candidatas plausíveis e não der para decidir, **NÃO adivinhes**:
     lista opções (Variante + SKU + Preço + Link) e pede escolha.
   - Se nada casar, responde com o produto base desse grupo e avisa que o preço pode variar consoante a opção.

4) FIXAÇÃO DE ESCOLHAS (SKU/VARIANTE)
- Se o utilizador fornece um **SKU** → usar exatamente **essa** linha (ref, preço, link).
- Se previamente **listaste** variantes e o utilizador escolhe uma pelo nome → **fixa a mesma linha** (ref/preço/link já apresentados).
- Se o utilizador pedir uma variante **que não está** na lista do catálogo, responde:
  “Essa variante não consta no catálogo interno para este produto.”

5) PREÇOS
- Variante identificada → **usa o price da variante**.
- Sem variante → **usa o price do produto base** (e avisa da variação).
- Nunca usar o preço de outra variante.
- Mensagem padrão: **“preço com IVA incluído; portes não incluídos.”** (salvo exceção explícita no catálogo).

6) LINKS (política rígida)
- Só incluir **1 link** quando estiveres a falar de um **produto/variante** do catálogo interno (**interiorguider.com** / **boasafra.pt**).

- Para variantes, usa exatamente o URL guardado em catalog_items.url (não acrescentar #sku nem outros sufixos).
- O SKU deve ser mostrado no texto, mas nunca embutido no URL.


- Qualquer URL escrito em texto (ex.: "https://…") conta como link.
- PROIBIDO repetir URLs fora do formato Markdown.
- **Proibido** acrescentar blocos de “Links úteis”.
- Em orçamentos, **PROIBIDO** usar links do RAG para produtos.
- O link deve ser apresentado APENAS uma vez e APENAS em formato Markdown: [ver produto](URL).
- PROIBIDO repetir o URL em texto (ex.: "(https://...)" ou "https://..." numa linha separada).
- PROIBIDO colocar o mesmo link duas vezes (mesmo que uma seja “clicável” e outra “texto”).


- POLÍTICA DE LINKS (SOBREPUJA QUALQUER OUTRA REGRA)
- Em todas as respostas, o link apresentado ao utilizador deve ser sempre o URL base do produto (sem qualquer fragmento #, incluindo #sku=).
  Mesmo que o catálogo interno contenha #sku= na URL, remove sempre o fragmento antes de mostrar o link.
  O SKU identifica a variante apenas em texto, nunca no link.
- É proibido apresentar links com #, #sku= ou parâmetros de variante.
- Mostrar apenas o URL canónico do produto (/slug).


7) AUTO-CHECK ANTES DE RESPONDER
- Confirmar que o **SKU/ref** ou o **texto da variante** coincide com a linha escolhida (após normalização).
- Confirmar que o **preço** mostrado é exatamente o **price** dessa linha.
- Confirmar que o link segue a política (preferir `…#sku=…` quando existir).
- Garantir a frase **“preço com IVA incluído; portes não incluídos”** (salvo exceção explícita).

REGRA DURA + EXEMPLOS ANTI-ERRO
- Se a pergunta mencionar explicitamente uma variante, é **PROIBIDO** responder com o preço do produto base.
- Ex.1: “2x Orikomi Plus Taupe **Simples Branco 1M**” → usar SKU correto da variante, preço da variante, link `…#sku=…`.
- Ex.2: “3x Orikomi Cinza Claro **Simples Branco 1M**” → usar SKU correto dessa variante; nunca o preço de outra.

PEDIDOS AMBÍGUOS
- Se a pergunta for ambígua entre várias variantes, faz **1** pergunta de clarificação e oferece todas as opções (Variante / SKU / Preço / Link) **do catálogo**.


PEDIDOS AMBÍGUOS (SEM CORTES)
	•	Se a pergunta for ambígua entre várias variantes, faz 1 pergunta de clarificação.
	•	Antes da pergunta, lista todas as variantes do catálogo interno desse produto (até 100 linhas), no formato:
Variante + SKU + Preço + Link.
	•	É proibido mostrar apenas um subconjunto (ex.: “3 opções”); se existem N variantes no catálogo, lista N.



FORMATO DE RESPOSTA (ORÇAMENTOS)
- Título curto com quantidade e variante (se houver).
- Linhas: **Nome + SKU**, **Preço unitário (IVA incluído)**, **Quantidade**, **Subtotal**.
- Nota: **“preço com IVA incluído; portes não incluídos”** (salvo indicação contrária).
- Link único (conforme a política acima).
- Link único em Markdown: [ver produto](URL). Não repetir o URL em texto.

Nunca inventes preços, nomes ou SKUs. Nunca assumes variantes sem sinal claro no catálogo interno.

──────────────────────────────────────────────────────────────────────────────
ELFOS (ASSISTENTES ESPECIALIZADOS — SINTRA AI)

A Alma trabalha com assistentes especializados internos, designados por **Elfos**.
Os Elfos são ajudantes funcionais, focados em tarefas específicas, e **não substituem**
a Alma nem tomam decisões comerciais, estratégicas ou de preços.

A Alma mantém sempre:
- o contexto global,
- a coerência estratégica,
- a responsabilidade final pela resposta.

Os Elfos são sugeridos **apenas quando a tarefa beneficia claramente de especialização**
e **não envolve dados críticos do catálogo interno** (preços, SKUs, orçamentos).

──────────────────────────────────────────────────────────────────────────────
QUANDO A ALMA PODE SUGERIR UM ELFO

A Alma pode sugerir um Elfo quando:
- o pedido é exploratório, operacional ou criativo;
- a tarefa é bem delimitada por função (ex.: estratégia, marketing, copy, análise);
- não exige consulta nem modificação de preços, variantes ou regras comerciais;
- o utilizador beneficia de trabalhar diretamente com um agente especializado.

É **PROIBIDO** delegar para Elfos:
- orçamentos, cotações, preços ou proformas;
- decisões comerciais finais;
- interpretação de dados incompletos do catálogo interno.

──────────────────────────────────────────────────────────────────────────────
ELFOS DISPONÍVEIS E FUNÇÕES

Buddy — Estratégia e Desenvolvimento de Negócio  
https://app.sintra.ai/buddy  
Usar quando o utilizador pede:
- análise estratégica,
- estruturação de propostas,
- reflexão sobre posicionamento, crescimento ou decisão de negócio.

Cassie — Atendimento ao Cliente  
https://app.sintra.ai/cassie  
Usar para:
- respostas a clientes,
- organização de FAQs,
- fluxos de suporte e comunicação.

Dexter — Análise de Dados  
https://app.sintra.ai/dexter  
Usar quando há:
- dados, métricas ou números a analisar,
- necessidade de leitura quantitativa ou padrões.

Penn — Escrita e Copywriting  
https://app.sintra.ai/penn  
Usar para:
- copy comercial,
- textos de marketing,
- reformulação e clareza de mensagens.

Seomi — SEO e Conteúdo  
https://app.sintra.ai/seomi  
Usar quando o foco é:
- otimização SEO,
- estrutura de conteúdos,
- títulos, keywords e organização editorial.

Soshie — Redes Sociais  
https://app.sintra.ai/soshie  
Usar para:
- planeamento de posts,
- ideias de conteúdo social,
- consistência de presença digital.

Emmie — Email Marketing  
https://app.sintra.ai/emmie  
Usar para:
- campanhas de email,
- newsletters,
- sequências automatizadas.

Vizzy — Assistente Virtual Geral  
https://app.sintra.ai/vizzy  
Usar para:
- tarefas administrativas,
- organização operacional,
- apoio funcional genérico.

──────────────────────────────────────────────────────────────────────────────
FORMA DE APRESENTAÇÃO AO UTILIZADOR

A Alma deve sugerir um Elfo de forma natural e contida, por exemplo:

“Para esta parte, um dos meus ajudantes especializados pode aprofundar o trabalho.
Se quiseres, podes continuar diretamente com ele aqui: https://app.sintra.ai/buddy”

A Alma **não transfere responsabilidade**, apenas orienta o utilizador.

──────────────────────────────────────────────────────────────────────────────
PRINCÍPIO CENTRAL

Os Elfos são extensões funcionais da Alma.
A Alma coordena, integra e valida.
Os Elfos executam, exploram e apoiam.

"""


# ---------------------------------------------------------------------------------------
# Utilidades de URL e normalização
# ---------------------------------------------------------------------------------------
IG_HOST = os.getenv("IG_HOST", "interiorguider.com").lower()

def _canon_ig_url(u: str) -> str:
    """Canoniza URLs do domínio IG_HOST; deixa externos intactos."""
    try:
        p = urlparse((u or "").strip())
    except Exception:
        return u or ""
    if not p.netloc:
        return u or ""
    host = p.netloc.lower().replace("www.", "")
    if IG_HOST not in host:
        # só canonizamos IG; externos ficam como estão
        return u or ""
    path = re.sub(r"/(products?|produtos?)\/", "/", p.path, flags=re.I)
    path = re.sub(r"//+", "/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    p = p._replace(scheme="https", netloc=IG_HOST, path=path)
    return urlunparse(p)

# --- normaliza links IG existentes e converte URLs IG “nuas” em markdown (sem look-behind) ---
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", re.I)
_RAW_URL_RE = re.compile(r"https?://[^\s)>\]]+", re.I)

def _already_inside_md(text: str, start_idx: int) -> bool:
    open_br = text.rfind("[", 0, start_idx)
    close_br = text.rfind("]", 0, start_idx)
    paren_open = text.find("(", close_br if close_br != -1 else start_idx)
    return (open_br != -1) and (close_br != -1) and (close_br > open_br) and (paren_open != -1) and (paren_open <= start_idx)

def _fix_product_links_markdown(text: str) -> str:
    if not text:
        return text
    # corrige links markdown existentes (canon IG)
    def _md_repl(m):
        label, url = m.group(1), m.group(2)
        fixed = _canon_ig_url(url)
        return f"[{label}]({fixed})" if fixed else m.group(0)
    text = _MD_LINK_RE.sub(_md_repl, text)
    # converte URLs nuas IG em markdown, se não estiverem já dentro de []
    out = []
    last = 0
    for m in _RAW_URL_RE.finditer(text):
        url = m.group(0)
        host = (urlparse(url).netloc or "").lower()
        if IG_HOST in host and not _already_inside_md(text, m.start()):
            out.append(text[last:m.start()])
            fixed = _canon_ig_url(url)
            out.append(f"[ver produto]({fixed})" if fixed else url)
            last = m.end()
    out.append(text[last:])
    return "".join(out)

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = (s.replace("ã","a").replace("õ","o").replace("á","a").replace("à","a").replace("â","a")
           .replace("é","e").replace("ê","e").replace("í","i").replace("ó","o").replace("ô","o")
           .replace("ú","u").replace("ç","c"))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------------------------------------------------------------------------------------
# RAG (qdrant + openai embeddings) — usa rag_client.py
# ---------------------------------------------------------------------------------------
try:
    from rag_client import (
        ingest_text, ingest_pdf_url, ingest_url,
        crawl_and_ingest, ingest_sitemap,
        search_chunks, build_context_block
    )
    RAG_READY = True
    log.info("[rag] rag_client import OK")
except Exception as e:
    RAG_READY = False
    log.warning(f"[rag] a importar rag_client falhou: {e}")

# -------- Top-K dinâmico --------
RAG_TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K", "12"))
RAG_CONTEXT_TOKEN_BUDGET = int(os.getenv("RAG_CONTEXT_TOKEN_BUDGET", "1600"))
DEFAULT_NAMESPACE = os.getenv("RAG_DEFAULT_NAMESPACE", "").strip() or None

def _clamp_int(v, lo=1, hi=50, default=None):
    try:
        x = int(v)
        return max(lo, min(hi, x))
    except Exception:
        return default if default is not None else lo

# --- heurísticas de termos de produto + preço
_NAME_HINTS = [
    r"\bmesa\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bcadeira[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bbanco[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bcama[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
    r"\bluminária[s]?\s+[a-z0-9çáéíóúâêôãõ ]{2,40}",
]
_PRICE_RE = re.compile(r"(?:€|\bEUR?\b)\s*([\d\.\s]{1,12}[,\.]\d{2}|\d{1,6})")

BUDGET_PATTERNS = [
    r"\borçament[oô]\s+(?:para|de)\s+(\d+)",
    r"(\d+)\s*(?:artigos?|itens?|produtos?)",
    r"(\d+)\s*(?:x|unid\.?|unidades?)\s*(?:de|da|do)?\s*([a-z0-9çáéíóúâêôãõ ]{3,})",
]

def _extract_name_terms(text:str) -> List[str]:
    t = " " + (text or "") + " "
    terms = []
    for pat in _NAME_HINTS:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            terms.append(m.group(0).strip())
    pieces = re.split(r"[,;\n]| e | com | para | de ", text or "", flags=re.I)
    for c in pieces:
        c = c.strip()
        if 3 <= len(c) <= 60 and any(w in c.lower() for w in ["mesa","cadeira","banco","cama","luminária","luminaria"]):
            terms.append(c)
    seen=set(); out=[]
    for s in map(_norm, terms):
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out[:8]

def _extract_budget_items(text: str) -> List[Tuple[int, str]]:
    t = " " + (text or "") + " "
    items = []
    for pat in BUDGET_PATTERNS:
        for m in re.finditer(pat, t, flags=re.I):
            if len(m.groups()) == 2:
                try:
                    qty = int(m.group(1))
                except Exception:
                    continue
                name = m.group(2).strip()
                if qty > 0 and len(name) > 2:
                    items.append((qty, name))
            elif len(m.groups()) == 1:
                try:
                    qty = int(m.group(1))
                except Exception:
                    continue
                if qty >= 2:
                    items.append((qty, "artigo genérico"))
    seen = set()
    out = []
    for qty, name in items:
        key = _norm(name)
        if key not in seen:
            seen.add(key)
            out.append((qty, name))
    return out[:5]

def _parse_money_eu(s: str) -> Optional[float]:
    try:
        if s is None:
            return None
        s = str(s).strip().replace("€", "").replace("\u00A0", " ").strip()
        s = re.sub(r"\s+", "", s)
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        else:
            if "," in s:
                s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None

def _price_from_text(txt: str) -> Optional[float]:
    if not txt: return None
    m = _PRICE_RE.search(txt)
    if not m: return None
    return None if m is None else _parse_money_eu(m.group(1))

def _expand_variants(term: str) -> List[str]:
    t = (term or "").strip()
    if not t:
        return []
    t = re.sub(r"^\s*\d+\s*x?\s*", "", t, flags=re.I)
    t = re.sub(r"^\s*\d+\s+", "", t, flags=re.I)
    toks = re.findall(r"[A-Za-z0-9ÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-]+", t)
    base = " ".join(toks[:6]).strip()
    if not base:
        return []
    variants = {base}
    sp_map = {
        "cadeiras": "cadeira",
        "bancos": "banco",
        "mesas": "mesa",
        "camas": "cama",
        "luminárias": "luminária",
        "luminarias": "luminaria",
    }
    words = base.split()
    if words:
        w0 = words[0].lower()
        if w0 in sp_map:
            variants.add(" ".join([sp_map[w0]] + words[1:]))
        inv = {v: k for k, v in sp_map.items()}
        if w0 in inv:
            variants.add(" ".join([inv[w0]] + words[1:]))
    if len(words) >= 2: variants.add(" ".join(words[:2]))
    if len(words) >= 3: variants.add(" ".join(words[:3]))
    if len(words) >= 4: variants.add(" ".join(words[:4]))
    out = []
    seen = set()
    for v in variants:
        vn = _norm(v)
        if vn and vn not in seen:
            seen.add(vn)
            out.append(v)
    return out

# --- Two-stage retrieval helpers (full-text + re-rank) -----------------------

NUMERAL_MAP = {
    # pt
    "um": "1", "uma": "1", "dois": "2", "duas": "2", "tres": "3", "três": "3",
    # en
    "one": "1", "two": "2", "three": "3",
    # es
    "uno": "1", "dos": "2", "tres": "3",
}

LANG_HINTS = {
    "pt": {"cadeira", "banco", "mesa", "cama", "luminaria", "luminária"},
    "en": {"chair", "stool", "table", "bed", "lamp"},
    "es": {"silla", "taburete", "mesa", "cama", "lampara", "lámpara"},
}

def _detect_lang_hint(qnorm: str) -> str:
    toks = set(qnorm.split())
    for lang, vocab in LANG_HINTS.items():
        if toks & vocab:
            return lang
    # fallback: se contiver acentos pt/es assume pt
    if any(ch in qnorm for ch in "áàâãéêíóôõúç"):
        return "pt"
    return ""

def _slugify_tokens(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _norm(s)).strip("-")

def _tokenize(s: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", _norm(s)) if t]

def _ngram_variants(tokens: List[str], min_n=1, max_n=4) -> List[str]:
    out = []
    n = len(tokens)
    for size in range(min_n, min(max_n, n) + 1):
        for i in range(0, n - size + 1):
            out.append(" ".join(tokens[i:i+size]))
    return out

def _word_number_swaps(tokens: List[str]) -> List[str]:
    out = set()
    for i, t in enumerate(tokens):
        repl = NUMERAL_MAP.get(t, "")
        if repl:
            alt = tokens[:]
            alt[i] = repl
            out.add(" ".join(alt))
        if t.isdigit():
            # trocar 3->three/ tres
            for w, d in NUMERAL_MAP.items():
                if d == t:
                    alt = tokens[:]
                    alt[i] = w
                    out.add(" ".join(alt))
    return list(out)

def _make_query_variants(term: str) -> List[str]:
    """
    Gera variantes robustas: base, hífen<->espaço, numerais, n-gramas encurtados.
    Mantém no máx. ~12 queries para custo controlado.
    """
    base = term.strip()
    if not base:
        return []
    toks = _tokenize(base)
    variants = set()
    # base
    variants.add(" ".join(toks))
    # hífenizações
    variants.add(_slugify_tokens(base).replace("-", " "))
    variants.add(_slugify_tokens(base))
    # numerais ↔ palavras
    variants.update(_word_number_swaps(toks))
    # n-gramas (encurtar cauda)
    ngs = _ngram_variants(toks, min_n=2, max_n=min(4, len(toks)))
    variants.update(ngs[-6:])  # últimas combinações mais específicas
    # dedup e limites
    out = []
    seen = set()
    for v in variants:
        vn = _norm(v)
        if vn and vn not in seen:
            seen.add(vn)
            out.append(v)
        if len(out) >= 12:
            break
    return out

def _overlap(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    A, B = set(a_tokens), set(b_tokens)
    inter = len(A & B)
    denom = max(1, min(len(A), len(B)))
    return inter / denom

def _scorer(query: str, payload: dict, url: str, lang_hint: str) -> float:
    qtok = _tokenize(query)
    title = (payload.get("title") or "").strip()
    slug  = (payload.get("slug") or "") or (urlparse(url).path if url else "")
    search_text = (payload.get("search_text") or "")
    # tokens
    ttok = _tokenize(title)
    slugtok = _tokenize(slug.replace("/", " "))
    utok = _tokenize(url or "")
    sttok = _tokenize(search_text)
    # overlaps
    s_slug  = _overlap(qtok, slugtok)
    s_title = _overlap(qtok, ttok)
    s_url   = _overlap(qtok, utok)
    s_stext = _overlap(qtok, sttok)
    score = (2.0*s_slug) + (1.25*s_title) + (0.5*s_url) + (0.75*s_stext)
    # domínio IG
    host = (urlparse(url).netloc or "").lower()
    if IG_HOST in host:
        score += 0.40
    # pista de idioma (ligeiro)
    text_all = " ".join([title, slug, search_text, url or ""])
    if lang_hint:
        if lang_hint == "pt" and re.search(r"[ãõáàâéêíóôõúç]|/pt/|/pt-", text_all, flags=re.I):
            score += 0.10
        elif lang_hint == "en" and re.search(r"/en/|/en-", text_all, flags=re.I):
            score += 0.10
        elif lang_hint == "es" and re.search(r"/es/|/es-", text_all, flags=re.I):
            score += 0.10
    return score

def _stage_a_candidates(term: str, namespace: Optional[str], limit_per_q: int = 50, hard_cap: int = 150):
    """
    Tenta usar full-text (se existir em rag_client: search_fulltext).
    Caso contrário, fallback para search_chunks com queries variantes.
    Retorna lista de dicts com {payload, metadata(url,title,...)}.
    """
    variants = _make_query_variants(term)
    seen_ids = set()
    cands = []

    # tenta full-text nativo, se existir no rag_client
    search_fulltext = None
    try:
        from rag_client import search_fulltext as _sf  # opcional
        search_fulltext = _sf
    except Exception:
        search_fulltext = None

    for v in variants[:6]:  # não exagerar
        try:
            if search_fulltext:
                hits = search_fulltext(query=v, namespace=namespace or DEFAULT_NAMESPACE, limit=limit_per_q) or []
            else:
                hits = search_chunks(query=v, namespace=namespace or DEFAULT_NAMESPACE, top_k=min( max(10, limit_per_q//2), 50)) or []
        except Exception:
            hits = []

        for h in hits:
            # Qdrant: pode vir {"id","payload","score"} ou formato do teu search_chunks
            pid = str(h.get("id") or id(h))
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            meta = h.get("metadata") or h.get("payload") or {}
            # normalizar “payload/metadata”
            payload = dict(meta)
            # espelhar title/url no payload se vierem noutro nível
            if not payload.get("title"): payload["title"] = h.get("title") or ""
            if not payload.get("url"):   payload["url"]   = h.get("url") or meta.get("url") or ""
            # tentar derivar slug
            if not payload.get("slug") and payload.get("url"):
                payload["slug"] = (urlparse(payload["url"]).path or "").strip("/")
            cands.append({"payload": payload, "score": h.get("score", 0.0)})
            if len(cands) >= hard_cap:
                break
        if len(cands) >= hard_cap:
            break
    return cands

def _rerank_and_choose(term: str, candidates: List[dict], lang_hint: str, prefer_ig: bool = True, threshold: float = 0.55):
    """
    Reordena candidatos por _scorer. Devolve (best_url, ranked[:3]) para debug/fallback.
    """
    ranked = []
    for c in candidates:
        p = c.get("payload", {}) or {}
        url = _canon_ig_url(p.get("url") or "")
        if not url:
            continue
        s = _scorer(term, p, url, lang_hint=lang_hint)
        ranked.append((s, url, p))
    ranked.sort(key=lambda x: x[0], reverse=True)

    # aplica threshold e preferência por IG
    top = [r for r in ranked if r[0] >= threshold] or ranked[:1]
    if not top:
        return "", []
    if prefer_ig:
        ig = [r for r in top if IG_HOST in (urlparse(r[1]).netloc or "").lower()]
        if ig:
            return ig[0][1], top[:3]
    return top[0][1], top[:3]


def rag_mini_search_urls(terms: List[str], namespace: Optional[str], top_k: int) -> Dict[str, str]:
    """
    Two-stage: Stage A (full-text se existir / fallback embeddings com variantes) -> Stage B (re-rank textual).
    Retorna o melhor URL por termo, com preferência por interiorguider.com.
    """
    if not (RAG_READY and terms):
        return {}
    url_by_term: Dict[str, str] = {}
    for term in terms:
        tnorm = _norm(term)
        if not tnorm:
            continue
        lang_hint = _detect_lang_hint(tnorm)
        # Stage A: recolha de candidatos
        candidates = _stage_a_candidates(term, namespace=namespace, limit_per_q=50, hard_cap=150)
        if not candidates:
            # fallback final: uma pesquisa “crua” única
            try:
                hits = search_chunks(query=term, namespace=namespace or DEFAULT_NAMESPACE, top_k=max(10, min(40, top_k or 10))) or []
            except Exception:
                hits = []
            candidates = [{"payload": (h.get("metadata") or {}), "score": h.get("score", 0.0)} for h in hits]

        # Stage B: re-rank e escolha
        best_url, _ = _rerank_and_choose(term, candidates, lang_hint=lang_hint, prefer_ig=True, threshold=0.55)

        if best_url:
            url_by_term[tnorm] = best_url
    return url_by_term

# ---------------------------------------------------------------------------------------
# Memória Local + (opcional) Mem0
# ---------------------------------------------------------------------------------------
MEM0_ENABLE = os.getenv("MEM0_ENABLE", "false").lower() in ("1", "true", "yes")
MEM0_API_KEY = (os.getenv("MEM0_API_KEY") or "").strip()

MemoryClient = None
mem0_client = None

if MEM0_ENABLE:
    if not MEM0_API_KEY:
        log.warning("[mem0] MEM0_ENABLE=true mas falta MEM0_API_KEY")
    else:
        try:
            try:
                import mem0ai as _mem0_pkg
                from mem0ai import MemoryClient as _MC
            except Exception:
                import mem0 as _mem0_pkg
                from mem0 import MemoryClient as _MC
            MemoryClient = _MC
            log.info(f"[mem0] import OK ({getattr(_mem0_pkg,'__name__','mem0')}) file={getattr(_mem0_pkg,'__file__','?')}")
        except Exception as e:
            log.error(f"[mem0] import FAILED: {e}")
            MemoryClient = None
        if MemoryClient is not None:
            try:
                mem0_client = MemoryClient(api_key=MEM0_API_KEY)
                log.info("[mem0] MemoryClient inicializado.")
            except Exception as e:
                log.error(f"[mem0] não inicializou: {e}")
                mem0_client = None

LOCAL_FACTS: Dict[str, Dict[str, str]] = {}
LOCAL_HISTORY: Dict[str, List[Tuple[str, str]]] = {}

FACT_PREFIX = "FACT|"

def local_set_fact(user_id: str, key: str, value: str):
    LOCAL_FACTS.setdefault(user_id, {})
    LOCAL_FACTS[user_id][key] = value.strip()

def local_get_facts(user_id: str) -> Dict[str, str]:
    return dict(LOCAL_FACTS.get(user_id, {}))

def local_append_dialog(user_id: str, question: str, answer: str, cap: int = 50):
    LOCAL_HISTORY.setdefault(user_id, [])
    LOCAL_HISTORY[user_id].append((question, answer))
    if len(LOCAL_HISTORY[user_id]) > cap:
        LOCAL_HISTORY[user_id] = LOCAL_HISTORY[user_id][-cap:]

def local_search_snippets(user_id: str, limit: int = 5) -> List[str]:
    items = LOCAL_HISTORY.get(user_id, [])
    out = []
    for q, a in reversed(items[-limit*2:]):
        if len(out) >= limit: break
        out.append(f"User: {q}")
        if len(out) >= limit: break
        out.append(f"Alma: {a}")
    return out[:limit]

def _mem0_create(content: str, user_id: str, metadata: Optional[dict] = None):
    if not (MEM0_ENABLE and mem0_client):
        return
    try:
        if hasattr(mem0_client, "memories"):
            mem0_client.memories.create(content=content, user_id=user_id, metadata=metadata or {})
        elif hasattr(mem0_client, "create"):
            mem0_client.create(content=content, user_id=user_id, metadata=metadata or {})
    except Exception as e:
        log.warning(f"[mem0] create falhou: {e}")

def _mem0_search(query: str, user_id: str, limit: int = 5) -> List[str]:
    if not (MEM0_ENABLE and mem0_client):
        return []
    try:
        if hasattr(mem0_client, "memories"):
            results = mem0_client.memories.search(query=query or "contexto", user_id=user_id, limit=limit)
        else:
            results = mem0_client.search(query=query or "contexto", user_id=user_id, limit=limit)
        snippets = []
        for item in results or []:
            val = (item.get("text") or item.get("memory") or item.get("content") or "").strip()
            if val:
                snippets.append(val)
        return snippets
    except Exception as e:
        log.warning(f"[mem0] search falhou: {e}")
        return []

def mem0_set_fact(user_id: str, key: str, value: str):
    local_set_fact(user_id, key, value)
    if not (MEM0_ENABLE and mem0_client): return
    try:
        _mem0_create(
            content=f"{FACT_PREFIX}{key}={value}",
            user_id=user_id,
            metadata={"source": "alma-server", "type": "fact", "key": key}
        )
    except Exception as e:
        log.warning(f"[mem0] fact create falhou: {e}")

def mem0_get_facts(user_id: str, limit: int = 50) -> Dict[str, str]:
    facts = local_get_facts(user_id)
    if not (MEM0_ENABLE and mem0_client):
        return facts
    try:
        if hasattr(mem0_client, "memories"):
            results = mem0_client.memories.search(query=FACT_PREFIX, user_id=user_id, limit=limit)
        else:
            results = mem0_client.search(query=FACT_PREFIX, user_id=user_id, limit=limit)
        for item in results or []:
            content = (item.get("text") or item.get("memory") or item.get("content") or "").strip()
            if content.startswith(FACT_PREFIX) and "=" in content[len(FACT_PREFIX):]:
                body = content[len(FACT_PREFIX):]
                k, v = body.split("=", 1)
                if k and v:
                    facts[k.strip()] = v.strip()
        return facts
    except Exception as e:
        log.warning(f"[mem0] get_facts falhou: {e}")
        return facts

# ---------------------------------------------------------------------------------------
# Extração de factos simples do texto
# ---------------------------------------------------------------------------------------
NAME_PATTERNS = [
    r"\bchamo-?me\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}(?:\s+[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}){0,3})\b",
    r"\bo\s+meu\s+nome\s+é\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}(?:\s+[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}){0,3})\b",
    r"\bsou\s+(?:o|a)\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}(?:\s+[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç\-']{1,40}){0,3})\b",
]
CITY_PATTERNS = [
    r"\bmoro\s+(?:em|no|na)\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\w\s\-\.'ÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç]{2,60})",
    r"\bestou\s+(?:em|no|na)\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\w\s\-\.'ÁÂÃÀÉÊÍÓÔÕÚÇ]{2,60})",
    r"\bsou\s+de\s+([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\w\s\-\.'ÁÂÃÀÉÊÍÓÔÕÚÇ]{2,60})",
]
PREF_PATTERNS = [
    r"\bgosto\s+(?:de|do|da|dos|das)\s+([^\.]{3,80})",
    r"\bprefiro\s+([^\.]{3,80})",
    r"\badoro\s+([^\.]{3,80})",
    r"\bquero\s+([^\.]{3,80})",
]
ROOM_KEYWORDS = ["sala", "cozinha", "quarto", "wc", "casa de banho", "varanda", "escritório", "hall", "entrada", "lavandaria"]

def extract_contextual_facts_pt(text: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    t = " " + (text or "") + " "
    for pat in NAME_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if len(name.split()) == 1 and name.lower() in {"melhor", "pior", "arquiteto", "cliente"}:
                pass
            else:
                facts["name"] = name
                break
    for pat in CITY_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            city = m.group(1).strip(" .,'\"").title()
            if 2 <= len(city) <= 60:
                facts["location"] = city
                break
    for pat in PREF_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            pref = m.group(1).strip(" .,'\"")
            if 3 <= len(pref) <= 80:
                facts.setdefault("preferences", pref)
                break
    for kw in ROOM_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", t, flags=re.IGNORECASE):
            facts["room"] = kw
    return facts

def facts_to_context_block(facts: Dict[str, str]) -> str:
    if not facts: return ""
    lines = []
    if "name" in facts: lines.append(f"- Nome: {facts['name']}")
    if "location" in facts: lines.append(f"- Localização: {facts['location']}")
    if "room" in facts: lines.append(f"- Divisão/Projeto: {facts['room']}")
    if "preferences" in facts: lines.append(f"- Preferências: {facts['preferences']}")
    for k, v in facts.items():
        if k not in {"name", "location", "room", "preferences"}:
            lines.append(f"- {k}: {v}")
    return "Perfil do utilizador (memória contextual):\n" + "\n".join(lines)

def facts_block_for_user(user_id: str) -> str:
    facts = mem0_get_facts(user_id)
    return facts_to_context_block(facts)

# ---------- Entity/link resolver (seguro) ----------
def _canon_path(u: str) -> str:
    try:
        p = urlparse(_canon_ig_url(u) or u)
        return (p.path or "").rstrip("/")
    except Exception:
        return ""

def _tokens(s: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", _norm(s)) if t and len(t) > 1]

def _compat_score(name: str, title: str, slug: str) -> float:
    # mede compatibilidade entidade ↔ página (Jaccard sobre tokens)
    nt = set(_tokens(name))
    tt = set(_tokens(title))
    st = set(_tokens(slug.replace("/", " ")))
    if not nt:
        return 0.0
    j_title = len(nt & tt) / max(1, min(len(nt), len(tt))) if tt else 0.0
    j_slug  = len(nt & st) / max(1, min(len(nt), len(st))) if st else 0.0
    # reforços semânticos por palavras-chave (evita banco↔mesa, etc.)
    kind_hit = 0.0
    KINDS = [
        ("mesa","table"), ("cadeira","chair"), ("banco","stool"),
        ("cama","bed"), ("luminaria","lamp"), ("luminária","lamp")
    ]
    name_low = _norm(name)
    text_all = _norm(title + " " + slug)
    for pt,en in KINDS:
        if re.search(rf"\b{pt}\b", name_low) and re.search(rf"\b({pt}|{en})\b", text_all):
            kind_hit = 0.2; break
    return (0.6*j_slug) + (0.4*j_title) + kind_hit

class LinkResolver:
    def __init__(self, namespace: Optional[str], prefer_host: str = IG_HOST, min_conf: float = 0.58):
        self.ns = namespace or DEFAULT_NAMESPACE
        self.prefer_host = (prefer_host or "").lower()
        self.min_conf = min_conf

    def best_url_for(self, term: str) -> Tuple[str,float,dict]:
        """Procura candidatos (fulltext/embeddings) e escolhe o melhor link.
        Retorna (url, conf, payload)."""
        cands = _stage_a_candidates(term, namespace=self.ns, limit_per_q=50, hard_cap=120) or []
        best = ("", 0.0, {})
        lang_hint = _detect_lang_hint(_norm(term))
        for c in cands:
            p = c.get("payload") or {}
            raw_url = p.get("url") or ""
            url = _canon_ig_url(raw_url) or raw_url
            if not url:
                continue
            title = (p.get("title") or "")
            slug  = (p.get("slug")  or urlparse(url).path or "")
            base  = _scorer(term, p, url, lang_hint)          # “parecido com a query”
            compat= _compat_score(term, title, slug)          # “é MESMO este produto?”
            host  = (urlparse(url).netloc or "").lower()
            pref  = 0.25 if (self.prefer_host and self.prefer_host in host) else 0.0
            score = 0.5*base + 0.5*compat + pref
            if score > best[1]:
                best = (url, score, p)
        return best

    def resolve_map(self, terms: List[str]) -> Dict[str, Tuple[str,float]]:
        out = {}
        for t in terms:
            tnorm = _norm(t)
            if not tnorm: 
                continue
            url, conf, _ = self.best_url_for(t)
            if url and conf >= self.min_conf:
                out[tnorm] = (url, conf)
        return out

    def smart_fix_text(self, text: str, url_by_term_conf: Dict[str, Tuple[str,float]]) -> str:
        """Substitui apenas quando a confiança é suficiente e o destino é diferente/melhor."""
        if not text or not url_by_term_conf:
            return text
        parts = text.splitlines()
        for i, line in enumerate(parts):
            ln = _norm(line)
            for tnorm, (good_url, conf) in url_by_term_conf.items():
                if tnorm not in ln:
                    continue
                good_path = _canon_path(good_url)
                # 1) corrige markdown com URL IG errado
                def _md_fix(m):
                    label, url = m.group(1), m.group(2)
                    host = (urlparse(url).netloc or "").lower()
                    if IG_HOST in host and _canon_path(url) != good_path and conf >= self.min_conf:
                        return f"[{label}]({good_url})"
                    return m.group(0)
                l2 = _MD_LINK_RE.sub(_md_fix, line)

                # 2) corrige URLs “nus” IG errados
                def _raw_fix(m):
                    url = m.group(0)
                    host = (urlparse(url).netloc or "").lower()
                    if IG_HOST in host and _canon_path(url) != good_path and conf >= self.min_conf:
                        return good_url
                    return url
                l2 = _RAW_URL_RE.sub(_raw_fix, l2)
                line = l2
            parts[i] = line
        return "\n".join(parts)

# ---------------------------------------------------------------------------------------
# Injetor de links a partir do RAG (pós-processamento do texto) + Fallback
# ---------------------------------------------------------------------------------------

def _inject_links_from_rag(text: str, user_query: str, namespace: Optional[str], decided_top_k: int) -> str:
    if not (RAG_READY and text):
        return text

    # 0) normalização de links já formatados (mantém)
    out = _fix_product_links_markdown(text or "")

    # 1) extrai termos das duas pontas (query + resposta)
    terms: List[str] = []
    terms.extend(_extract_name_terms(user_query or ""))
    terms.extend(_extract_name_terms(out or ""))

    # 2) resolve entidades com LinkResolver
    resolver = LinkResolver(namespace=namespace, min_conf=0.58)
    url_by_term_conf = resolver.resolve_map(terms)

    # 3) corrige “[label](sem URL)” e linhas “sem URL” usando o mapa resolvido
    if url_by_term_conf:
        def _fix_sem_url(m):
            label = m.group(1)
            tnorm = _norm(label)
            if tnorm in url_by_term_conf:
                return f"[{label}]({url_by_term_conf[tnorm][0]})"
            # fallback: se só existir um link com boa confiança, usa-o
            if len(url_by_term_conf) == 1:
                only_url = list(url_by_term_conf.values())[0][0]
                return f"[{label}]({only_url})"
            return m.group(0)
        out = re.sub(r"\[([^\]]+)\]\(\s*sem\s+url\s*\)", _fix_sem_url, out, flags=re.I)

        def _replace_line_sem_url(line: str) -> str:
            if re.search(r"\bsem\s+url\b", line, flags=re.I):
                ln = _norm(line)
                # escolhe o termo mais “compatível” com a linha
                best = None
                for tnorm, (url, conf) in url_by_term_conf.items():
                    if tnorm in ln and conf >= resolver.min_conf:
                        best = (tnorm, url, conf); break
                if best:
                    tnorm, url, _ = best
                    if _MD_LINK_RE.search(line):
                        return re.sub(r"\bsem\s+url\b", f"{url}", line, flags=re.I)
                    # liga o primeiro nome de produto detetado
                    m2 = re.search(r"(mesa|cadeira|banco|cama|luminária|luminaria)[^\-:]*", line, flags=re.I)
                    if m2:
                        nome = m2.group(0).strip()
                        linked = line.replace(nome, f"[{nome}]({url})", 1)
                        return re.sub(r"\bsem\s+url\b", "", linked, flags=re.I)
                    return re.sub(r"\bsem\s+url\b", f"{url}", line, flags=re.I)
            return line
        out = "\n".join(_replace_line_sem_url(l) for l in out.splitlines())

    # 4) corrige URLs IG errados (apenas se confiança >= min_conf)
    out = resolver.smart_fix_text(out, url_by_term_conf)

    # 5) se ainda não houver nenhum link IG, tenta injetar 1 com maior confiança
    if IG_HOST not in out.lower() and url_by_term_conf:
        best_t = max(url_by_term_conf.items(), key=lambda kv: kv[1][1])  # maior confiança
        best_url = best_t[1][0]
        lines = out.splitlines()
        for i, l in enumerate(lines):
            if not _MD_LINK_RE.search(l):
                lines[i] = l + f" — [ver produto]({best_url})"
                out = "\n".join(lines); break

    # 6) reforço para pedidos de orçamento: linka nomes dos itens
    if "orçament" in (user_query or "").lower() and url_by_term_conf:
        budget_items = _extract_budget_items(user_query)
        lines = out.splitlines()
        for i, line in enumerate(lines):
            for qty, item_name in budget_items:
                tnorm = _norm(item_name)
                if tnorm in url_by_term_conf and not _MD_LINK_RE.search(line):
                    url = url_by_term_conf[tnorm][0]
                    m = re.search(rf"(\b{re.escape(item_name)}\b)", line, re.I)
                    if m:
                        nome = m.group(1)
                        lines[i] = line.replace(nome, f"[{nome}]({url})", 1)
        out = "\n".join(lines)

    return out

def _postprocess_answer(answer: str, user_query: str, namespace: Optional[str], decided_top_k: int) -> str:
    step1 = _fix_product_links_markdown(answer or "")
    step2 = _inject_links_from_rag(step1, user_query, namespace, decided_top_k)
    return step2

def _links_from_matches(matches: List[dict], max_links: int = 8) -> List[Tuple[str, str]]:
    """Extrai (title, url) únicos dos matches, priorizando IG, depois externos."""
    if not matches:
        return []
    pairs: List[Tuple[str, str]] = []
    seen = set()
    # 1) IG primeiro
    for pref_ig in (True, False):
        for h in matches:
            meta = h.get("metadata", {}) or {}
            title = (meta.get("title") or h.get("title") or "").strip()
            url = (meta.get("url") or h.get("url") or "").strip()
            if not url:
                continue
            cu = _canon_ig_url(url) or url
            host = (urlparse(cu).netloc or "").lower()
            is_ig = IG_HOST in host
            if pref_ig and not is_ig:
                continue
            if (not pref_ig) and is_ig:
                continue
            key = (title.lower(), cu.lower())
            if key in seen:
                continue
            seen.add(key)
            pairs.append((title or "-", cu))
            if len(pairs) >= max_links:
                return pairs
    return pairs[:max_links]

def _force_links_block(answer: str, matches: List[dict], min_links: int = 3) -> str:
    """Se a resposta não contiver links (ou não tiver IG), anexa bloco de links do RAG."""
    text = answer or ""
    has_md_link = bool(_MD_LINK_RE.search(text))
    has_ig = IG_HOST in text.lower()
    if has_md_link and has_ig:
        return text  # já tem
    # obter links a partir dos matches
    links = _links_from_matches(matches, max_links=max(6, min_links))
    if not links:
        return text
    # montar bloco
    lines = ["", "Links úteis (do RAG):"]
    for title, url in links:
        safe_title = title if title else "—"
        lines.append(f"- [{safe_title}]({url})")
    return (text.rstrip() + "\n" + "\n".join(lines)).strip()

# ---------------------------------------------------------------------------------------
# Config Grok (x.ai)
# ---------------------------------------------------------------------------------------
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = os.getenv("XAI_MODEL", "grok-4-1-fast-reasoning")

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
_session.mount("https://", _adapter); _session.mount("http://", _adapter)

def grok_chat(messages, timeout=120):
    if not XAI_API_KEY:
        raise RuntimeError("Falta XAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }
    payload = {"model": MODEL, "messages": messages}
    r = _session.post(XAI_API_URL, headers=headers, json=payload, timeout=timeout)
    log.info(f"[x.ai] status={r.status_code} body={r.text[:300]}")
    r.raise_for_status()
    return r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or ""

# ---------------------------------------------------------------------------------------
# Blocos de contexto e construção das mensagens
# ---------------------------------------------------------------------------------------
def _decide_top_k(user_query: str, req_top_k: Optional[int]) -> int:
    if req_top_k:
        return _clamp_int(req_top_k, lo=3, hi=40, default=RAG_TOP_K_DEFAULT)
    n_terms = len(_extract_name_terms(user_query or ""))
    budget_items = _extract_budget_items(user_query)
    if "orçament" in user_query.lower() or len(budget_items) >= 2:
        base = 20 + (len(budget_items) * 4)
    elif n_terms >= 3:
        base = 16
    elif n_terms == 2:
        base = 12
    else:
        base = 10
    return min(max(base, RAG_TOP_K_DEFAULT), 40)

def build_rag_products_block(question: str) -> str:
    if not RAG_READY:
        return ""
    budget_items = _extract_budget_items(question)
    lines = []
    if budget_items:
        for qty, item_name in budget_items[:6]:
            query_item = f"{qty} {item_name}"
            try:
                hits = search_chunks(query=query_item, namespace=DEFAULT_NAMESPACE, top_k=5) or []
            except Exception:
                hits = []
            seen = set()
            for h in hits:
                meta = h.get("metadata", {}) or {}
                title = (meta.get("title") or item_name).strip()
                url = _canon_ig_url(meta.get("url") or "")
                key = (title, url)
                if key in seen: continue
                seen.add(key)
                lines.append(f"- QTY={qty}; NOME={title}; URL={url or 'sem URL'}")
            if not hits:
                links = rag_mini_search_urls([item_name], DEFAULT_NAMESPACE, top_k=5)
                url = links.get(_norm(item_name), "sem URL")
                lines.append(f"- QTY={qty}; NOME={item_name}; URL={url}")
    else:
        try:
            hits = search_chunks(query=question, namespace=DEFAULT_NAMESPACE, top_k=RAG_TOP_K_DEFAULT) or []
        except Exception:
            hits = []
        seen = set()
        for h in hits[:8]:
            meta = h.get("metadata", {}) or {}
            title = (meta.get("title") or "").strip()
            url = _canon_ig_url(meta.get("url") or "")
            key = (title, url)
            if key in seen: continue
            seen.add(key)
            lines.append(f"- NOME={title or '-'}; URL={url or 'sem URL'}")
    return "Produtos para orçamento (do RAG; usa estes dados exatos para links):\n" + "\n".join(lines) if lines else ""

# ---------- Helpers para intenção e catálogo (colar acima de build_messages) ----------
import re
from typing import Optional, List, Dict, Tuple

def _is_budget_intent_pt(text: str) -> bool:
    """Heurística PT simples para pedidos de orçamento / preço / variantes."""
    if not text:
        return False
    t = text.lower()
    triggers = [
        "orçament", "orcament", "cotação", "cotacao", "preço", "preco",
        "quanto custa", "quanto fica", "sku", "#sku=", "variante", "cor=",
        "quantidade", "qtd", "unidades", "unidade", "preçário", "precario"
    ]
    if any(k in t for k in triggers):
        return True
    # “2x”, “3 unidades”, etc.
    if re.search(r"\b\d+\s*x\b", t) or re.search(r"\b\d+\s+(unidade|unidades)\b", t):
        return True
    return False

# >>> PATCH: catálogo — query “limpa” + fallback AND→OR + logs

# stopwords mínimas PT/EN e lixo de orçamento
_STOP = {
    "o","a","os","as","de","da","do","das","dos","para","por","com","sem","em","no","na","nos","nas",
    "um","uma","uns","umas","e","ou","que","quanto","custa","fica","preço","preco","orcamento","orçamento",
    "faz","fazer","favor","peço","pedido","proposta","cotacao","cotação","qtd","unidade","unidades","x",
    "the","of","and","for","with","without","in","on"
}

def _catalog_query_from_question(question: str) -> str:
    """
    Extrai uma query útil para o catálogo a partir da pergunta:
    - devolve refs/SKUs se existirem
    - caso contrário, devolve só tokens “de produto” (>=3 chars, não stopword, não só dígitos)
    """
    q = (question or "").strip()
    refs = re.findall(r"\b[A-Z0-9][A-Z0-9._\-]{2,}\b", q, flags=re.I)
    if refs:
        return " ".join(refs[:5])
    toks = [t for t in re.split(r"[^\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç]+", q) if t]
    good = []
    for t in toks:
        tl = t.lower()
        # ignora números puros e stopwords; exige 3+ chars úteis
        if tl in _STOP: 
            continue
        if tl.isdigit():
            continue
        if len(tl) < 3:
            continue
        good.append(t)
    # mantém ordem, dedup leve
    seen = set(); cleaned = []
    for t in good:
        k = t.lower()
        if k not in seen:
            seen.add(k); cleaned.append(t)
    return " ".join(cleaned[:6]) or q


def build_catalog_block(question: str, namespace: Optional[str] = None, limit: int = 30) -> str:
    """
    Pesquisa no catálogo interno (SQLite) por nome/summary/ref.
    Estratégia:
      1) match exato por refs/SKUs, se existirem
      2) pesquisa por termos úteis com AND
      3) se (2) não devolver nada, repete com OR (mais tolerante)
    Inclui log curto para validar que o LLM “viu” o bloco.
    """
    ns = (namespace or DEFAULT_NAMESPACE).strip()
    q_raw = (question or "").strip()
    q = _catalog_query_from_question(q_raw)
    ref_toks = _extract_ref_tokens(q_raw)

    rows: List[Dict] = []
    try:
        with _catalog_conn() as c:
            cur: sqlite3.Cursor

            # 1) Match exato por SKU/ref
            if ref_toks:
                cur = c.execute(f"""
                    SELECT namespace, name, ref, price, url, brand, variant_attrs, updated_at
                      FROM catalog_items
                     WHERE namespace=? AND ref IN ({','.join('?'*len(ref_toks))})
                     ORDER BY updated_at DESC
                     LIMIT ?""",
                    tuple([ns, *ref_toks, limit])
                )
                rows = [dict(r) for r in cur.fetchall()]

            # Vamos construir termos “bons” para LIKE
            terms = [t for t in re.split(r"[\s,;]+", _sanitize_like(q)) if len(t) >= 3]

            # 2) Pesquisa AND (mais estrita)
            if len(rows) < 1 and terms:
                where_and = " AND ".join(["(name LIKE ? OR summary LIKE ? OR ref LIKE ?)"] * len(terms))
                like_args = []
                for t in terms:
                    p = f"%{t}%"
                    like_args.extend([p, p, p])
                cur = c.execute(f"""
                    SELECT namespace, name, ref, price, url, brand, variant_attrs, updated_at
                      FROM catalog_items
                     WHERE namespace=? AND {where_and}
                     ORDER BY (CASE WHEN url LIKE '%#sku=%' THEN 0 ELSE 1 END), updated_at DESC
                     LIMIT ?""", tuple([ns, *like_args, limit]))
                rows = [dict(r) for r in cur.fetchall()]

            # 3) Fallback OR (tolerante)
            if len(rows) < 1 and terms:
                where_or_parts = []
                like_args = []
                for t in terms:
                    p = f"%{t}%"
                    where_or_parts.append("(name LIKE ? OR summary LIKE ? OR ref LIKE ?)")
                    like_args.extend([p, p, p])
                where_or = " OR ".join(where_or_parts) if where_or_parts else "1=1"
                cur = c.execute(f"""
                    SELECT namespace, name, ref, price, url, brand, variant_attrs, updated_at
                      FROM catalog_items
                     WHERE namespace=? AND ({where_or})
                     ORDER BY (CASE WHEN url LIKE '%#sku=%' THEN 0 ELSE 1 END), updated_at DESC
                     LIMIT ?""", tuple([ns, *like_args, limit]))
                rows = [dict(r) for r in cur.fetchall()]

    except Exception as e:
        log.warning(f"[catalog] search falhou: {e}")
        rows = []

    # LOG do que vai ao LLM
    try:
        log.info(f"[catalog] ns={ns} q_raw={q_raw!r} q_clean={q!r} ref_toks={ref_toks} rows={len(rows)}")
    except Exception:
        pass

    if not rows:
        return ""  # sem bloco → o LLM pode cair na mensagem “sem dados” do Mission

    # Bloco para o LLM
    lines = [f"CATÁLOGO INTERNO — ns={ns} (usar SÓ estes dados para orçamentos/preços; NÃO usar RAG aqui)"]
    seen = set()
    for r in rows[:limit]:
        name = r.get("name") or "(sem nome)"
        ref  = r.get("ref") or "(sem dado)"
        price = r.get("price")
        price_txt = f"{price:.2f}€" if isinstance(price, (int,float)) else "(sem preço)"
        url = r.get("url") or "(sem URL)"
        brand = r.get("brand") or ""
        variant = (r.get("variant_attrs") or "").strip()
        key = (ref, url)
        if key in seen:
            continue
        seen.add(key)
        name_show = f"{name} — Variante: {variant}" if variant else name
        lines.append(f"- {name_show} • SKU: {ref} • Preço: {price_txt} • Marca: {brand} • Link: {url}")
    return "\n".join(lines)
# <<< PATCH
    

def build_catalog_variants_block(question: str, namespace: Optional[str]) -> str:
    """
    Lista variantes (#sku=) por grupo de produto quando a query sugere listagem.
    Mantém a apresentação enxuta para o LLM.
    """
    qlow = (question or "").lower()
    if not any(k in qlow for k in ("variante", "variantes", "cores", "tamanhos", "opções", "opcoes")):
        return ""
    ns = (namespace or DEFAULT_NAMESPACE).strip()
    terms = [t for t in re.split(r"[\s,;]+", _sanitize_like(question)) if len(t) >= 2]

    rows: List[Dict] = []
    try:
        with _catalog_conn() as c:
            where = " AND ".join(["(name LIKE ? OR summary LIKE ? OR ref LIKE ?)"] * max(1, len(terms)))
            like_args = []
            if terms:
                for t in terms:
                    pattern = f"%{t}%"
                    like_args.extend([pattern, pattern, pattern])
            else:
                # fallback leve se não houver termos
                where = "(name LIKE ? OR summary LIKE ?)"
                like_args.extend([f"%{_sanitize_like(question)}%", f"%{_sanitize_like(question)}%"])
            cur = c.execute(f"""
                SELECT name, ref, price, url, brand, variant_attrs
                  FROM catalog_items
                 WHERE namespace=? AND {where}
                 ORDER BY updated_at DESC
                 LIMIT 120
            """, tuple([ns, *like_args]))
            rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.warning(f"[catalog variants] falhou: {e}")
        rows = []

    if not rows:
        return ""

    # agrupar por URL-pai
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        u = (r.get("url") or "").strip()
        parent = u.split("#", 1)[0] if u else ""
        groups.setdefault(parent, []).append(r)

    # filtrar grupos com ≥2 variantes (#sku=)
    groups = {
        p: [it for it in lst if "#sku=" in (it.get("url") or "")]
        for p, lst in groups.items()
    }
    groups = {p: lst for p, lst in groups.items() if len(lst) >= 2}
    if not groups:
        return ""

    out = ["CATÁLOGO INTERNO — Variantes por produto (Variante | SKU | Preço | Link):"]
    for parent, lst in groups.items():
        lst_sorted = sorted(lst, key=lambda x: (x.get("name") or "", x.get("ref") or ""))
        title = lst_sorted[0].get("name") or parent or "(produto)"
        out.append(f"• Produto: {title}")
        for r in lst_sorted[:40]:
            ref  = r.get("ref") or "(sem ref)"
            pr   = r.get("price")
            pr_s = f"{pr:.2f}€" if isinstance(pr, (int, float)) else "(sem preço)"
            url  = r.get("url") or "(sem url)"
            va   = (r.get("variant_attrs") or "").strip()
            label = va or (r.get("name") or "")
            out.append(f"   - {label} | SKU:{ref} | {pr_s} | {url}")
    return "\n".join(out)

# ---------- build_messages (patch consolidado) ----------
def build_messages(user_id: str, question: str, namespace: Optional[str]):
    # 1) sinais contextuais → mem0
    new_facts = extract_contextual_facts_pt(question)
    for k, v in new_facts.items():
        mem0_set_fact(user_id, k, v)

    # 2) memórias recentes
    short_snippets = _mem0_search(question, user_id=user_id, limit=5) or local_search_snippets(user_id, limit=5)
    memory_block = (
        "Memórias recentes do utilizador (curto prazo):\n"
        + "\n".join(f"- {s}" for s in short_snippets[:3])
        if short_snippets else ""
    )

    # 3) CATÁLOGO INTERNO (SQLite) — SEMPRE antes do RAG
    ns = (namespace or DEFAULT_NAMESPACE).strip()
    cat_q = _catalog_query_from_question(question)
    catalog_block = build_catalog_block(cat_q, ns)
    catalog_variants_block = build_catalog_variants_block(cat_q, ns)

    # 4) RAG (apenas para conteúdo corporativo, NÃO para preços/orçamentos)
    rag_block = ""
    rag_used = False
    rag_hits: List[dict] = []
    if RAG_READY:
        try:
            rag_hits = search_chunks(
                query=question,
                namespace=namespace or DEFAULT_NAMESPACE,
                top_k=RAG_TOP_K_DEFAULT
            ) or []
            rag_block = build_context_block(rag_hits, token_budget=RAG_CONTEXT_TOKEN_BUDGET) if rag_hits else ""
            rag_used = bool(rag_block)
        except Exception as e:
            log.warning(f"[rag] search falhou: {e}")
            rag_block, rag_used, rag_hits = "", False, []

    # 5) links candidatos do RAG (o mission já proíbe usá-los para preços)
    links_pairs = _links_from_matches(rag_hits, max_links=8)
    links_block = ""
    if links_pairs:
        lines = ["Links candidatos (do RAG; NÃO usar para preços/orçamentos):"]
        for title, url in links_pairs:
            lines.append(f"- {title or '-'} — {url}")
        links_block = "\n".join(lines)

    # 6) bloco de produtos (opcional)
    products_block = build_rag_products_block(question)

    # 7) montar mensagens
    messages = [{"role": "system", "content": ALMA_MISSION}]
    fb = facts_block_for_user(user_id)
    if fb:
        messages.append({"role": "system", "content": fb})

    # catálogo primeiro (para guiar o LLM)
    if catalog_block:
        messages.append({"role": "system", "content": catalog_block})
    if catalog_variants_block:
        messages.append({"role": "system", "content": catalog_variants_block})

    # depois o RAG (contexto corporativo, docs, etc.)
    if rag_block:
        messages.append({"role": "system", "content": f"Conhecimento corporativo (RAG — não usar para preços):\n{rag_block}"})
    if links_block:
        messages.append({"role": "system", "content": links_block})

    if products_block:
        messages.append({"role": "system", "content": products_block})
    if memory_block:
        messages.append({"role": "system", "content": memory_block})

    messages.append({"role": "user", "content": question})
    return messages, new_facts, rag_used, rag_hits
# ---------- FIM DO PATCH ----------

# ROTAS BÁSICAS + páginas
@app.get("/", response_class=HTMLResponse)
def serve_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return HTMLResponse("<h1>index.html não encontrado</h1>", status_code=404)

@app.get("/alma-chat", response_class=HTMLResponse)
@app.get("/alma-chat/", response_class=HTMLResponse)
def alma_chat():
    html_path = os.path.join(os.getcwd(), "alma-chat.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<h1>alma-chat.html não encontrado</h1>", status_code=404)

@app.get("/console", response_class=HTMLResponse)
def serve_console():
    html_path = os.path.join(os.getcwd(), "console.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<h1>console.html não encontrado</h1>", status_code=404)

@app.get("/status")
def status_json():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "message": "Alma server ativo. Use POST /ask (Grok+Memória+RAG).",
        "mem0": {"enabled": MEM0_ENABLE, "client_ready": bool(mem0_client)},
        "rag": {
            "available": RAG_READY,
            "top_k_default": RAG_TOP_K_DEFAULT,
            "namespace": DEFAULT_NAMESPACE
        },
        "endpoints": {
            "health": "/health",
            "ask": "POST /ask {question, user_id?, namespace?, top_k?}",
            "ask_get": "/ask_get?q=...&user_id=...&namespace=...&top_k=...",
            "ping_grok": "/ping_grok",
            "rag_search_get": "/rag/search?q=...&namespace=...",
            "rag_search_post": "POST /rag/search",
            "rag_ingest_sitemap": "POST /rag/ingest-sitemap",
            "rag_crawl": "POST /rag/crawl",
            "rag_ingest_url": "POST /rag/ingest-url",
            "rag_ingest_text": "POST /rag/ingest-text",
            "rag_ingest_pdf_url": "POST /rag/ingest-pdf-url",
            "rag_extract_urls": "POST /rag/extract-urls",
            "budget_csv": "POST /budget/csv",
            "console": "/console",
        },
        # _status_catalog_sqlite() é definido na Parte 3; a avaliação só acontece runtime
        "catalog": _status_catalog_sqlite() if ' _status_catalog_sqlite' in globals() else {},
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mem0_enabled": MEM0_ENABLE,
        "mem0_client_ready": bool(mem0_client),
        "model": MODEL,
        "rag_available": RAG_READY,
        "rag_default_namespace": DEFAULT_NAMESPACE,
        "rag_top_k_default": RAG_TOP_K_DEFAULT,
    }

@app.post("/echo")
async def echo(request: Request):
    data = await request.json()
    return {"echo": data}

@app.get("/ping_grok")
def ping_grok():
    try:
        msg = [{"role": "user", "content": "Diz apenas: pong"}]
        content = grok_chat(msg, timeout=20)
        return {"ok": True, "reply": content}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------------------------------------------------------------------------------------
# Memória contextual (FACTs) e Mem0 debug
# ---------------------------------------------------------------------------------------
@app.get("/mem/facts")
def mem_facts(user_id: str = "anon"):
    facts = mem0_get_facts(user_id=user_id, limit=50)
    return {"user_id": user_id, "facts": facts}

@app.get("/mem/search")
def mem_search_route(q: str = "", user_id: str = "anon"):
    if not q:
        return {"user_id": user_id, "found": 0, "snippets": []}
    snippets = _mem0_search(q, user_id=user_id, limit=10) or local_search_snippets(user_id, limit=10)
    return {"user_id": user_id, "found": len(snippets), "snippets": snippets}

# ---------------------------------------------------------------------------------------
# RAG: GET /rag/search (debug)
# ---------------------------------------------------------------------------------------
@app.get("/rag/search")
def rag_search_get(q: str, namespace: str = None, top_k: int = None):
    if not RAG_READY:
        return {"ok": False, "error": "rag_client indisponível no servidor"}
    try:
        res = search_chunks(query=q, namespace=namespace or DEFAULT_NAMESPACE, top_k=top_k or RAG_TOP_K_DEFAULT)
        return {"ok": True, "query": q, "matches": res}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------------------------------------------------------------------------------------
# RAG Endpoints (crawl, sitemap, url, text, pdf, search POST) — compat com consola
# ---------------------------------------------------------------------------------------
@app.post("/rag/crawl")
async def rag_crawl(request: Request):
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}
    try:
        data = await request.json()
        seed_url   = (data.get("seed_url") or "").strip()
        namespace  = (data.get("namespace") or "default").strip()
        max_pages  = int(data.get("max_pages")  or os.getenv("CRAWL_MAX_PAGES", "40"))
        max_depth  = int(data.get("max_depth")  or os.getenv("CRAWL_MAX_DEPTH", "2"))
        deadline_s = int(data.get("deadline_s") or os.getenv("RAG_DEADLINE_S", "55"))
        if not seed_url:
            return {"ok": False, "error": "Falta seed_url"}
        # assinatura do rag_client antiga (seed_url, não root_url)
        res = crawl_and_ingest(
            seed_url=seed_url, namespace=namespace,
            max_pages=max_pages, max_depth=max_depth, deadline_s=deadline_s
        )
        if "ok" not in res: res["ok"] = True
        res.setdefault("summary", f"visited={res.get('visited')} ok_chunks={res.get('ok_chunks')} fail={res.get('fail')} namespace={namespace}")
        return res
    except Exception as e:
        import traceback
        log.exception("crawl_failed")
        return {"ok": False, "error": "crawl_failed", "detail": str(e), "trace": traceback.format_exc(limit=3)}

@app.post("/rag/ingest-sitemap")
async def rag_ingest_sitemap_route(request: Request):
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}
    try:
        data = await request.json()
        sitemap_url = (data.get("sitemap_url") or data.get("site_url") or "").strip()
        namespace   = (data.get("namespace") or "default").strip()
        max_pages   = int(data.get("max_pages") or os.getenv("CRAWL_MAX_PAGES", "40"))
        deadline_s  = int(data.get("deadline_s") or os.getenv("RAG_DEADLINE_S", "55"))
        if not sitemap_url:
            return {"ok": False, "error": "Falta sitemap_url/site_url"}
        return ingest_sitemap(sitemap_url, namespace=namespace, max_pages=max_pages, deadline_s=deadline_s)
    except Exception as e:
        return {"ok": False, "error": "sitemap_failed", "detail": str(e)}

@app.post("/rag/ingest-url")
async def rag_ingest_url_route(request: Request):
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}
    try:
        data = await request.json()
        page_url  = (data.get("page_url") or "").strip()
        namespace = (data.get("namespace") or "default").strip()
        deadline_s = int(data.get("deadline_s") or os.getenv("RAG_DEADLINE_S", "55"))
        if not page_url:
            return {"ok": False, "error": "Falta page_url"}
        return ingest_url(page_url, namespace=namespace, deadline_s=deadline_s)
    except Exception as e:
        return {"ok": False, "error": "ingest_url_failed", "detail": str(e)}

@app.post("/rag/ingest-text")
async def rag_ingest_text_route(request: Request):
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}
    try:
        data = await request.json()
        title     = (data.get("title") or "").strip()
        text      = (data.get("text") or "").strip()
        namespace = (data.get("namespace") or "default").strip()
        if not title or not text:
            return {"ok": False, "error": "Falta title ou text"}
        return ingest_text(title=title, text=text, namespace=namespace)
    except Exception as e:
        return {"ok": False, "error": "ingest_text_failed", "detail": str(e)}

@app.post("/rag/ingest-pdf-url")
async def rag_ingest_pdf_url_route(request: Request):
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}
    try:
        data = await request.json()
        pdf_url  = (data.get("pdf_url") or "").strip()
        title    = (data.get("title") or None)
        namespace = (data.get("namespace") or "default").strip()
        if not pdf_url:
            return {"ok": False, "error": "Falta pdf_url"}
        return ingest_pdf_url(pdf_url=pdf_url, title=title, namespace=namespace)
    except Exception as e:
        return {"ok": False, "error": "ingest_pdf_failed", "detail": str(e)}

@app.post("/rag/search")
async def rag_search_post(request: Request):
    if not RAG_READY:
        return {"ok": False, "error": "RAG não disponível"}
    try:
        data = await request.json()
        query     = (data.get("query") or "").strip()
        namespace = (data.get("namespace") or None)
        top_k     = int(data.get("top_k") or os.getenv("RAG_TOP_K", str(RAG_TOP_K_DEFAULT)))
        matches = search_chunks(query=query, namespace=namespace or DEFAULT_NAMESPACE, top_k=top_k)
        ctx = build_context_block(matches, token_budget=RAG_CONTEXT_TOKEN_BUDGET)
        return {"ok": True, "matches": matches, "context_block": ctx}
    except Exception as e:
        return {"ok": False, "error": "search_failed", "detail": str(e)}

# --- Proxy: extrair URLs (sitemap.xml, HTML ou texto colado) — usado pela consola
@app.post("/rag/extract-urls")
async def rag_extract_urls(request: Request):
    try:
        data = await request.json()
        url = (data.get("url") or "").strip()
        raw_text = data.get("text")
        max_urls = int(data.get("max_urls") or 5000)

        if not url and not raw_text:
            return {"ok": False, "error": "fornece 'url' ou 'text'"}

        if url:
            r = requests.get(url, headers={"User-Agent": "AlmaBot/1.0 (+rag)"}, timeout=30)
            r.raise_for_status()
            raw_text = r.text

        if not raw_text:
            return {"ok": False, "error": "sem conteúdo"}

        urls = []

        def dedup_keep(seq):
            seen = set(); out = []
            for u in seq:
                u = (u or "").strip()
                if not u or len(u) > 2048: continue
                if u in seen: continue
                seen.add(u); out.append(u)
                if len(out) >= max_urls: break
            return out

        txt = raw_text

        if ("<urlset" in txt) or ("<sitemapindex" in txt):
            try:
                root = ET.fromstring(txt)
                locs = [(el.text or "").strip() for el in root.findall(".//{*}loc")]
                urls = dedup_keep(locs)
                return {"ok": True, "type": "sitemap", "count": len(urls), "urls": urls[:max_urls]}
            except Exception:
                import re
                locs = re.findall(r"<loc>(.*?)</loc>", txt, flags=re.I|re.S)
                urls = dedup_keep(locs)
                return {"ok": True, "type": "sitemap-regex", "count": len(urls), "urls": urls[:max_urls]}

        import re
        hrefs = re.findall(r'href=["\'](https?://[^"\']+)["\']', txt, flags=re.I)
        urls = dedup_keep(hrefs)
        return {"ok": True, "type": "html", "count": len(urls), "urls": urls[:max_urls]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------------------------------------------------------------------------------------
# 🔸 Exportação CSV de Orçamentos
# ---------------------------------------------------------------------------------------
def _safe_float_csv(v, default=0.0):
    try:
        if isinstance(v, str):
            v = v.replace("€", "").replace(",", ".").strip()
        return float(v)
    except Exception:
        return float(default)

def _format_money(x: float) -> str:
    return f"{x:.2f}"

@app.post("/budget/csv")
async def budget_csv(request: Request):
    """
    Body:
    {
      "mode": "public" | "pro",
      "iva_pct": 23,
      "rows": [
        {"ref":"BS.01","descricao":"Produto","quant":1,"preco_uni":100,"desc_pct":5,"dim":"80x40xH45","material":"Carvalho / Óleo","marca":"Boa Safra","link":"https://interiorguider.com/..."}
      ]
    }
    """
    data = await request.json()
    mode = (data.get("mode") or "public").lower().strip()
    iva_pct = _safe_float_csv(data.get("iva_pct", 23.0))
    rows = data.get("rows") or []

    if mode not in ("public", "pro"):
        return PlainTextResponse("mode deve ser 'public' ou 'pro'", status_code=400)
    if not isinstance(rows, list) or not rows:
        return PlainTextResponse("rows vazio", status_code=400)

    headers = ["REF.", "DESIGNAÇÃO / MATERIAL / ACABAMENTO / COR", "QUANT.", "PREÇO UNI.", "DESC.", "TOTAL S/IVA"] if mode=="public" \
        else ["REF.", "DESIGNAÇÃO / MATERIAL / ACABAMENTO / COR", "QUANT.", "PREÇO UNI.", "DESC.", "TOTAL C/IVA"]

    sio = StringIO()
    writer = csv.writer(sio)
    writer.writerow(headers)

    for r in rows:
        ref = (r.get("ref") or "").strip()
        quant = int(r.get("quant") or 1)
        preco_uni = _safe_float_csv(r.get("preco_uni"), 0.0)
        desc_pct = _safe_float_csv(r.get("desc_pct"), 0.0)

        desc_main = (r.get("descricao") or "").strip() or "Produto"
        extra_lines = []
        if r.get("dim"): extra_lines.append(f"Dimensões: {r['dim']}")
        if r.get("material"): extra_lines.append(f"Material/Acabamento: {r['material']}")
        if r.get("marca"): extra_lines.append(f"Marca: {r['marca']}")
        if r.get("link"):
            link = str(r["link"]).strip()
            extra_lines.append(f"Link: {link}")
        full_desc = desc_main + (("\n" + "\n".join(extra_lines)) if extra_lines else "")

        total_si = quant * preco_uni * (1.0 - desc_pct/100.0)
        total_col = _format_money(total_si if mode=="public" else total_si * (1.0 + iva_pct/100.0))

        writer.writerow([
            ref,
            full_desc,
            str(quant),
            _format_money(preco_uni),
            (f"{desc_pct:.0f}%" if desc_pct else ""),
            total_col
        ])

    csv_bytes = sio.getvalue().encode("utf-8-sig")
    fname = f"orcamento_{mode}_{int(time.time())}.csv"
    fpath = os.path.join("/tmp", fname)
    with open(fpath, "wb") as f:
        f.write(csv_bytes)

    return FileResponse(fpath, media_type="text/csv", filename=fname)

# ---------------------------------------------------------------------------------------
# ASK endpoints (com injeção + fallback de links e top-k dinâmico)
# ---------------------------------------------------------------------------------------
import re
from typing import Optional
from fastapi import Request

def _should_force_all_variants(q: str) -> bool:
    ql = (q or "").strip().lower()
    if not ql:
        return False

    triggers = [
        "todas as variantes", "todas variantes",
        "todas as opções", "todas opções", "todas as opcoes", "todas opcoes",
        "quais são as variantes", "quais sao as variantes",
        "lista as variantes", "listar as variantes",
        "todas as cores", "todas cores",
        "opções disponíveis", "opcoes disponiveis",
        "variantes disponíveis", "variantes disponiveis",
        "mostra todas as variantes", "mostra todas",
        "lista tudo", "listar tudo",
    ]
    if any(t in ql for t in triggers):
        return True

    # Heurística curta: presença de "variantes/opções" + "todas/quais"
    if re.search(r"\b(variantes|opções|opcoes|cores)\b", ql) and re.search(r"\b(todas|quais)\b", ql):
        return True

    return False


def _inject_force_all_variants_system(messages: list, question: str) -> list:
    """
    Injeta uma instrução SYSTEM para impedir resumos (ex.: só 3 variantes).
    Só injeta quando a pergunta pede explicitamente "todas as variantes/opções".
    """
    if not _should_force_all_variants(question):
        return messages

    sys_msg = {
        "role": "system",
        "content": (
            "INSTRUÇÃO CRÍTICA (CATÁLOGO INTERNO):\n"
            "- Se o utilizador pedir 'todas as variantes/opções/cores/SKUs' de um produto, "
            "é PROIBIDO resumir ou limitar a 3.\n"
            "- Deves listar TODAS as variantes existentes no catálogo interno para esse produto.\n"
            "- Formato por linha: Variante (ou nome) + SKU/ref + preço.\n"
            "- No fim, inclui apenas 1 link canónico do produto (se existir) e pergunta qual variante/quantidade.\n"
            "- Não inventes variantes nem preços; usa apenas o catálogo interno."
        )
    }

    # Coloca no topo para ter máxima prioridade
    if isinstance(messages, list) and messages:
        return [sys_msg] + messages
    return [sys_msg]


@app.get("/ask_get")
def ask_get(q: str = "", user_id: str = "anon", namespace: str = None, top_k: Optional[int] = None):
    if not q:
        return {"answer": "Falta query param ?q="}

    decided_top_k = _decide_top_k(q, top_k)

    # mantém sinal de produtos no contexto (se usas isto para orientar o modelo)
    _ = build_rag_products_block(q)

    messages, new_facts, rag_used, rag_hits = build_messages(user_id, q, namespace)

    # HOTFIX: força listagem completa quando pedido
    messages = _inject_force_all_variants_system(messages, q)

    try:
        answer = grok_chat(messages)
    except Exception as e:
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

    answer = _postprocess_answer(answer, q, namespace, decided_top_k)
    answer = _force_links_block(answer, rag_hits, min_links=3)

    local_append_dialog(user_id, q, answer)
    _mem0_create(content=f"User: {q}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
    _mem0_create(content=f"Alma: {answer}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})

    return {
        "answer": answer,
        "new_facts_detected": new_facts,
        "rag": {
            "used": rag_used,
            "top_k_default": RAG_TOP_K_DEFAULT,
            "top_k_effective": decided_top_k,
            "namespace": namespace or DEFAULT_NAMESPACE,
        },
    }


@app.post("/ask")
async def ask(request: Request):
    data = await request.json()

    question = (data.get("question") or "").strip()
    user_id = (data.get("user_id") or "").strip() or "anon"
    namespace = (data.get("namespace") or "").strip() or None
    req_top_k = data.get("top_k")

    decided_top_k = _decide_top_k(question, req_top_k)
    log.info(f"[/ask] user_id={user_id} ns={namespace or DEFAULT_NAMESPACE} top_k={decided_top_k} question={question!r}")

    if not question:
        return {"answer": "Coloca a tua pergunta em 'question'."}

    _ = build_rag_products_block(question)

    messages, new_facts, rag_used, rag_hits = build_messages(user_id, question, namespace)

    # HOTFIX: força listagem completa quando pedido
    messages = _inject_force_all_variants_system(messages, question)

    try:
        answer = grok_chat(messages)
    except Exception as e:
        log.exception("Erro ao chamar a x.ai")
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

    answer = _postprocess_answer(answer, question, namespace, decided_top_k)
    answer = _force_links_block(answer, rag_hits, min_links=3)

    local_append_dialog(user_id, question, answer)
    _mem0_create(content=f"User: {question}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
    _mem0_create(content=f"Alma: {answer}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})

    return {
        "answer": answer,
        "new_facts_detected": new_facts,
        "rag": {
            "used": rag_used,
            "top_k_default": RAG_TOP_K_DEFAULT,
            "top_k_effective": decided_top_k,
            "namespace": namespace or DEFAULT_NAMESPACE,
        },
    }
# ---------------------------------------------------------------------------------------
# 📚 Catálogo (SQLite) — backend único via CSV + integração no /status
# ---------------------------------------------------------------------------------------
import sqlite3
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

# Respeita valores já definidos em partes anteriores
if 'DEFAULT_NAMESPACE' not in globals():
    DEFAULT_NAMESPACE = os.getenv("DEFAULT_NAMESPACE", "boasafra")
if 'CATALOG_DB_PATH' not in globals():
    CATALOG_DB_PATH = os.getenv("CATALOG_DB_PATH", "/tmp/catalog.db")

# ---- DB helpers -----------------------------------------------------------------------
def _catalog_conn():
    conn = sqlite3.connect(CATALOG_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _table_cols(c) -> set:
    cols = set()
    for r in c.execute("PRAGMA table_info(catalog_items)").fetchall():
        cols.add(r[1])
    return cols

# ========= Blocos do Catálogo (SQLite) p/ injeção no prompt =========

def _sanitize_like(s: str) -> str:
    return (s or "").strip().replace("%", "").replace("_", "")

def _extract_ref_tokens(text: str) -> list:
    """
    Puxa tokens do tipo SKU/ref (ex.: ORK.12-RO-TX, 741050527-SO). Útil p/ match exato.
    """
    import re
    toks = re.findall(r"[A-Z0-9][A-Z0-9._\-]{2,}", text or "", flags=re.I)
    # filtra coisas muito genéricas
    return [t for t in toks if any(ch.isdigit() for ch in t)]

def build_catalog_block(question: str, namespace: Optional[str] = None, limit: int = 30) -> str:
    """
    Procura no catálogo interno (SQLite) por nome/summary/ref e devolve
    um bloco de linhas normalizadas para o LLM usar em orçamentos.
    """
    ns = (namespace or DEFAULT_NAMESPACE).strip()
    q = _sanitize_like(question)
    ref_toks = _extract_ref_tokens(question)

    rows = []
    try:
        with _catalog_conn() as c:
            cur: sqlite3.Cursor

            # 1) Se houver SKUs na pergunta → tenta match EXATO primeiro
            if ref_toks:
                cur = c.execute("""
                    SELECT namespace, name, ref, price, url, brand, variant_attrs
                      FROM catalog_items
                     WHERE namespace=? AND ref IN (%s)
                     ORDER BY updated_at DESC
                     LIMIT ?""" % (",".join("?"*len(ref_toks))),
                    tuple([ns, *ref_toks, limit])  # ns + tokens + limit
                )
                rows = [dict(r) for r in cur.fetchall()]

            # 2) Se nada (ou também queremos mais contexto) → LIKE por nome/summary/ref
            if len(rows) < 3:  # dá mais contexto
                like = f"%{q}%"
                cur = c.execute("""
                    SELECT namespace, name, ref, price, url, brand, variant_attrs
                      FROM catalog_items
                     WHERE namespace=? AND (name LIKE ? OR summary LIKE ? OR ref LIKE ?)
                     ORDER BY (CASE WHEN url LIKE '%#sku=%' THEN 0 ELSE 1 END), updated_at DESC
                     LIMIT ?""", (ns, like, like, like, limit))
                rows2 = [dict(r) for r in cur.fetchall()]
                # evita duplicados mantendo ordem
                seen = {(r["ref"], r["url"]) for r in rows}
                for r in rows2:
                    key = (r["ref"], r["url"])
                    if key not in seen:
                        rows.append(r); seen.add(key)
    except Exception as e:
        log.warning(f"[catalog] search falhou: {e}")
        rows = []

    if not rows:
        return ""  # sem bloco → o LLM não “vê” catálogo

    # Formatar linhas de catálogo — APENAS dados internos (nada de RAG)
    lines = [
        "Catálogo interno (usa SÓ estes dados para orçamentos/preços; não usar RAG aqui):"
    ]
    for r in rows:
        name = r.get("name") or "(sem nome)"
        ref  = r.get("ref") or "(sem dado)"
        price = r.get("price")
        price_txt = f"{price:.2f}€" if isinstance(price, (int,float)) else "(sem preço)"
        url = r.get("url") or "(sem URL)"
        brand = r.get("brand") or ""
        variant = (r.get("variant_attrs") or "").strip()
        if variant:
            name_show = f"{name} — Variante: {variant}"
        else:
            name_show = name
        lines.append(f"- {name_show} • SKU: {ref} • Preço: {price_txt} • Marca: {brand} • Link: {url}")
    return "\n".join(lines)


def build_catalog_variants_block(question: str, namespace: Optional[str] = None, limit_families: int = 5, limit_variants: int = 40) -> str:
    """
    Se o pedido for sobre “variantes”, tenta agrupar por URL-pai e listar variantes (#sku) do mesmo produto.
    """
    ns = (namespace or DEFAULT_NAMESPACE).strip()
    q = (question or "").lower()
    if not any(k in q for k in ("variante", "variantes", "cores", "tamanhos", "opções", "opcoes")):
        return ""  # só ativa em perguntas que sugerem listagem

    try:
        with _catalog_conn() as c:
            # primeiro encontra alguns itens relevantes
            like = f"%{_sanitize_like(question)}%"
            cur = c.execute("""
                SELECT name, url
                  FROM catalog_items
                 WHERE namespace=? AND (name LIKE ? OR summary LIKE ?)
                 ORDER BY updated_at DESC
                 LIMIT ?""", (ns, like, like, limit_families))
            seeds = [dict(r) for r in cur.fetchall()]
            if not seeds:
                return ""

            variants_lines = ["Variantes no catálogo interno (agrupadas por produto):"]
            seen_parents = set()

            for s in seeds:
                url = s.get("url") or ""
                parent = url.split("#")[0] if url else ""
                if not parent or parent in seen_parents:
                    continue
                seen_parents.add(parent)

                cur2 = c.execute("""
                    SELECT name, ref, price, url, variant_attrs
                      FROM catalog_items
                     WHERE namespace=? AND url LIKE ?
                     ORDER BY updated_at DESC
                     LIMIT ?""", (ns, parent + "#sku=%", limit_variants))
                vars_ = [dict(r) for r in cur2.fetchall()]
                if not vars_:
                    continue

                # cabeçalho da família
                variants_lines.append(f"\n• Produto: {s.get('name') or parent}")
                for v in vars_:
                    ref = v.get("ref") or "(sem dado)"
                    price = v.get("price")
                    price_txt = f"{price:.2f}€" if isinstance(price, (int,float)) else "(sem preço)"
                    var_txt = (v.get("variant_attrs") or "").strip()
                    name_v = v.get("name") or ""
                    show = var_txt if var_txt else name_v
                    variants_lines.append(f"  - Variante: {show} • SKU: {ref} • Preço: {price_txt} • Link: {v.get('url') or '(sem URL)'}")

            if len(variants_lines) == 1:
                return ""
            return "\n".join(variants_lines)

    except Exception as e:
        log.warning(f"[catalog variants] falhou: {e}")
        return ""

# --- PATCH: catálogo (init + ensure cols) ------------------------------------
def _ensure_cols(conn):
    """
    Garante que a tabela catalog_items existe e tem todas as colunas necessárias.
    Usa cursor (não a connection) para fetchall() e é idempotente.
    """
    cur = conn.cursor()

    # 1) cria tabela “mínima” se não existir
    cur.execute("""
        CREATE TABLE IF NOT EXISTS catalog_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        )
    """)

    # 2) ler colunas atuais
    cur.execute("PRAGMA table_info(catalog_items)")
    cols = {row[1] for row in cur.fetchall()}  # row[1] = nome da coluna

    # 3) colunas necessárias (ordem apenas informativa)
    needed = [
        ("namespace",      "TEXT"),
        ("name",           "TEXT"),
        ("summary",        "TEXT"),
        ("url",            "TEXT"),       # UNIQUE via índice
        ("source",         "TEXT"),
        ("created_at",     "TEXT"),
        ("updated_at",     "TEXT"),
        ("ref",            "TEXT"),
        ("price",          "REAL"),
        ("currency",       "TEXT"),
        ("iva_pct",        "REAL"),
        ("brand",          "TEXT"),
        ("dimensions",     "TEXT"),
        ("material",       "TEXT"),
        ("image_url",      "TEXT"),
        ("variant_attrs",  "TEXT"),
        ("variant_key",    "TEXT")        # matching determinístico de variantes
    ]

    # 4) adicionar colunas em falta
    for col, coltype in needed:
        if col not in cols:
            try:
                cur.execute(f"ALTER TABLE catalog_items ADD COLUMN {col} {coltype}")
            except Exception as e:
                if "duplicate column name" in str(e).lower():
                    pass
                else:
                    raise

    # 5) índices
    try:
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_catalog_url ON catalog_items(url)")
    except Exception:
        pass
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_catalog_namespace ON catalog_items(namespace)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_catalog_ref ON catalog_items(ref)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_catalog_updated ON catalog_items(updated_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_catalog_variantkey ON catalog_items(variant_key)")
    except Exception:
        pass

    conn.commit()

def _catalog_init():
    """
    Abre a conexão e chama _ensure_cols uma única vez no arranque.
    """
    try:
        with _catalog_conn() as conn:
            _ensure_cols(conn)
        log.info(f"[catalog/sqlite] ready db={CATALOG_DB_PATH}")
    except Exception:
        log.exception("[catalog/sqlite] init failed")
        raise
_catalog_init()

# ---- Helpers extra (headers CSV, URLs relativas/absolutas IG, variantes) --------------
import unicodedata
from urllib.parse import urljoin

def _norm_header(s: str) -> str:
    s = (s or "").strip().lower().replace("\ufeff", "")
    s = s.replace("/", " ").replace("_", " ").replace("-", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _to_abs_ig_url(u: str) -> str:
    """Converte /slug ou //host/path para https://interiorguider.com/slug e aplica canon IG."""
    u = (u or "").strip()
    if not u:
        return ""
    if u.startswith("//"):
        u = "https:" + u
    if u.startswith("http://") or u.startswith("https://"):
        return _canon_ig_url(u) or u
    if not u.startswith("/"):
        u = "/" + u
    absu = f"https://{IG_HOST}{u}"
    return _canon_ig_url(absu) or absu

# --- Helpers para variantes ---------------------------------------------------
import json
import hashlib

def _norm_txt(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

def build_variant_key(attrs: dict) -> str:
    """
    Constrói chave determinística para matching de variantes.
    Ignora chaves técnicas; ordena pares; normaliza a=valor|b=valor…
    """
    if not attrs or not isinstance(attrs, dict):
        return ""
    ignore = {
        "product_url", "sku", "ref", "price", "moeda", "currency",
        "iva", "iva_pct", "image", "image_url", "url", "link"
    }
    pairs = []
    for k, v in attrs.items():
        kn = _norm_txt(k)
        if kn in ignore or (kn.startswith("option") and kn.endswith("name")):
            continue
        vn = _norm_txt(v)
        if vn == "":
            continue
        pairs.append((kn, vn))
    if not pairs:
        return ""
    pairs.sort(key=lambda x: x[0])
    return "|".join([f"{k}={v}" for k, v in pairs])

def values_to_variant_key(values_str: str) -> str:
    """
    De 'Soft Indigo, Tecido Areia 1m' -> 'soft indigo|tecido areia 1m'
    (útil quando só guardamos os VALORES, sem as chaves).
    """
    if not values_str:
        return ""
    parts = [p.strip() for p in str(values_str).split(",") if p.strip()]
    parts = [_norm_txt(p) for p in parts if p]
    return "|".join(parts)

# ---- Importação CSV (HTML clean + variantes normalizadas “valores”) --------------------
from io import StringIO
import html

def _parse_price(s: str):
    """
    Normaliza formatos PT (1.739,00) e EN (1739.00).
    Remove símbolos e milhares; devolve float ou None.
    """
    if not s:
        return None
    s = str(s).strip().replace("[FIXED]", "").strip()
    s = re.sub(r"[^\d,.\-]", "", s)
    if "." in s and "," in s and s.rfind(",") > s.rfind("."):
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        v = float(s)
        return v if v != 0.0 else None
    except Exception:
        return None

def _html_to_text(s: str) -> str:
    if not s:
        return ""
    txt = s
    for tag in (r"<\s*br\s*/?\s*>", r"</\s*p\s*>", r"</\s*li\s*>", r"</\s*div\s*>"):
        txt = re.sub(rf"(?i){tag}", "\n", txt)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n\s*\n+", "\n", txt)
    return txt.strip()

def _split_attr_segments(raw: str):
    if not raw:
        return []
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    out = []
    for p in parts:
        p = re.sub(r"^\[[^\]]+\]\s*", "", p)  # remove [XX]
        p = p.split(":", 1)[0].strip()        # corta URL após ':'
        if "=" in p:
            k, v = p.split("=", 1)
            out.append((k.strip(), v.strip()))
        else:
            out.append((None, p))
    return out

def _variant_to_values_only(raw: str) -> str:
    pairs = _split_attr_segments(raw)
    values = [(v if v else (k or "")).strip() for (k, v) in pairs]
    values = [re.sub(r"\s+", " ", v) for v in values if v]
    return ", ".join(values).strip(", ").strip()

# ---- Upsert genérico (PRECISA existir antes de /catalog/import-csv) -------------------
def _upsert_catalog_row(ns: Optional[str],
                        name: str,
                        summary: str,
                        url: str,
                        source: str = "csv",
                        ref: Optional[str] = None,
                        price: Optional[float] = None,
                        currency: Optional[str] = None,
                        iva_pct: Optional[float] = None,
                        brand: Optional[str] = None,
                        dimensions: Optional[str] = None,
                        material: Optional[str] = None,
                        image_url: Optional[str] = None,
                        variant_attrs: Optional[str] = None,
                        variant_key: Optional[str] = None):
    """
    Upsert por REF (prioritário) ou por URL (quando não há REF).
    Mantém 'url' alinhada nas variantes. Usa _catalog_conn() e _now().
    """
    if not url:
        return
    with _catalog_conn() as c:
        row_id = None

        # 1) tenta por REF (chave canónica para variantes)
        if ref:
            cur = c.execute("SELECT id FROM catalog_items WHERE ref=?", (ref,))
            r = cur.fetchone()
            if r:
                row_id = r["id"]

        # 2) se não encontrou por REF, tenta por URL apenas quando NÃO há ref
        if row_id is None and not ref:
            cur = c.execute(
                "SELECT id FROM catalog_items WHERE url=? AND (ref IS NULL OR ref='')",
                (url,)
            )
            r = cur.fetchone()
            if r:
                row_id = r["id"]

        if row_id is not None:
            c.execute("""
                UPDATE catalog_items
                   SET namespace=?,
                       name=?,
                       summary=?,
                       source=?,
                       updated_at=?,
                       ref=COALESCE(?, ref),
                       price=COALESCE(?, price),
                       currency=COALESCE(?, currency),
                       iva_pct=COALESCE(?, iva_pct),
                       brand=COALESCE(?, brand),
                       dimensions=COALESCE(?, dimensions),
                       material=COALESCE(?, material),
                       image_url=COALESCE(?, image_url),
                       variant_attrs=COALESCE(?, variant_attrs),
                       variant_key=COALESCE(?, variant_key),
                       url=?            -- mantém a URL alinhada (produto nas variantes)
                 WHERE id=?""",
                (ns, name, summary, source, _now(),
                 ref, price, currency, iva_pct, brand, dimensions, material, image_url,
                 variant_attrs, variant_key, url, row_id))
        else:
            c.execute("""
                INSERT INTO catalog_items
                  (namespace,name,summary,url,source,created_at,updated_at,
                   ref,price,currency,iva_pct,brand,dimensions,material,image_url,variant_attrs,variant_key)
                VALUES (?,?,?,?,?,?,?,
                        ?,?,?,?,?,?,?,?,?,?)
            """,
                (ns, name, summary, url, source, _now(), _now(),
                 ref, price, currency, iva_pct, brand, dimensions, material, image_url, variant_attrs, variant_key))

@app.post("/catalog/import-csv")
async def catalog_import_csv(file: UploadFile = File(...),
                             namespace: str = Form(DEFAULT_NAMESPACE)):
    """
    Importa CSV (BigCommerce).
    - Ignora 'Rule'
    - 'Product' = base  | 'SKU' = variante (url = base + "#sku=<ref>")
    - summary: HTML -> texto limpo; injeta 'Disponibilidade: ...' (só no base)
    - variant_attrs: VALORES puros (ex.: 'Soft Indigo, Tecido Areia 1m')
    - variant_key: 'soft indigo|tecido areia 1m' (para matching determinístico)
    - URLs canonizadas para IG
    """
    text = (await file.read()).decode("utf-8", errors="ignore")
    sio = StringIO(text, newline="")
    reader = csv.DictReader(sio)

    imported, failed, skipped = 0, 0, 0
    items_out = []
    current_product = None  # memoriza o último produto base

    for row in reader:
        try:
            row_type     = (row.get("Item Type") or "").strip()
            if row_type.lower() == "rule":
                skipped += 1
                items_out.append({"ok": True, "type": "rule", "skipped": True})
                continue

            # campos base BigCommerce
            name_raw     = (row.get("Product Name") or "").strip()
            ref          = (row.get("Product Code/SKU") or "").strip()
            price        = _parse_price(row.get("Price") or "")
            brand        = (row.get("Brand Name") or "").strip() or None
            summary_raw  = (row.get("Product Description") or "").strip()
            base_url     = _canon_ig_url((row.get("Product URL") or "").strip())
            availability = (row.get("Product Availability") or "").strip()

            summary_clean = _html_to_text(summary_raw)

            def _with_availability_base(s: str) -> str:
                s1 = (s or "").strip()
                if availability:
                    s1 = (s1 + ("\n" if s1 else "") + f"Disponibilidade: {availability}").strip()
                return s1

            # ----- produto base -----
            if row_type == "Product":
                current_product = {
                    "name": name_raw,
                    "summary": _with_availability_base(summary_clean),
                    "ref": ref or None,
                    "price": price,
                    "brand": brand,
                    "url": base_url,
                }
                _upsert_catalog_row(
                    ns=namespace,
                    name=current_product["name"],
                    summary=current_product["summary"],
                    url=current_product["url"],
                    source="csv",
                    ref=current_product["ref"],
                    price=current_product["price"],
                    currency=None,
                    iva_pct=None,
                    brand=current_product["brand"],
                    dimensions=None,
                    material=None,
                    image_url=None,
                    variant_attrs=None,
                    variant_key=None
                )
                items_out.append({"ok": True, "type": "product", "name": name_raw, "url": base_url})
                imported += 1
                continue

            # ----- variante (SKU) -----
            elif row_type == "SKU" and current_product:
                variant_values = _variant_to_values_only(name_raw)  # "Soft Indigo, Tecido Areia 1m"
                vkey = values_to_variant_key(variant_values)        # "soft indigo|tecido areia 1m"

                summary_v = current_product["summary"]
                if variant_values:
                    summary_v = (summary_v + ("\n" if summary_v else "") + f"Variante: {variant_values}").strip()

                price_v = price if price is not None else current_product["price"]
                url_variant = f"{current_product['url']}#sku={ref}" if ref else current_product["url"]

                _upsert_catalog_row(
                    ns=namespace,
                    name=f"{current_product['name']} — {variant_values}" if variant_values else current_product["name"],
                    summary=summary_v,
                    url=url_variant,
                    source="csv",
                    ref=ref or None,
                    price=price_v,
                    currency=None,
                    iva_pct=None,
                    brand=current_product["brand"],
                    dimensions=None,
                    material=None,
                    image_url=None,
                    variant_attrs=(variant_values or None),
                    variant_key=(vkey or None)
                )
                items_out.append({
                    "ok": True,
                    "type": "variant",
                    "name": variant_values or name_raw,
                    "ref": ref,
                    "url": url_variant,
                    "price": price_v,
                })
                imported += 1
                continue

            # ----- outros tipos (ignorar) -----
            else:
                skipped += 1
                items_out.append({"ok": True, "type": (row_type or "unknown").lower(), "skipped": True})

        except Exception as e:
            failed += 1
            items_out.append({"ok": False, "error": str(e)})

    return {
        "ok": True,
        "namespace": namespace,
        "imported": imported,
        "failed": failed,
        "skipped": skipped,
        "items": items_out,
    }

# 

# ---------------------------------------------------------------------------------------
# Catálogo – limpeza seletiva (por namespace inteiro, por marca, ou por prefixo de URL)
# ---------------------------------------------------------------------------------------
@app.post("/catalog/clear")
async def catalog_clear(namespace: str = Form(None),
                        brand: str = Form(None),
                        url_prefix: str = Form(None)):
    """
    Apaga items do catálogo de forma seletiva:
      - Se só vier 'namespace': apaga tudo desse namespace.
      - Se vier 'namespace' + 'brand': apaga apenas itens dessa marca no namespace.
      - Se vier 'namespace' + 'url_prefix': apaga apenas itens cuja URL começa por esse prefixo.
    NOTA: Pelo menos 'namespace' é obrigatório.
    """
    ns = (namespace or DEFAULT_NAMESPACE).strip()
    if not ns:
        return {"ok": False, "error": "Falta 'namespace'."}

    b = (brand or "").strip() or None
    p = (url_prefix or "").strip() or None

    where = ["namespace=?"]
    params = [ns]

    if b:
        where.append("brand=?")
        params.append(b)

    if p:
        where.append("url LIKE ?")
        params.append(f"{p}%")

    where_clause = " AND ".join(where)
    try:
        with _catalog_conn() as c:
            before = c.execute(f"SELECT COUNT(*) FROM catalog_items WHERE {where_clause}", tuple(params)).fetchone()[0]
            c.execute(f"DELETE FROM catalog_items WHERE {where_clause}", tuple(params))
            c.commit()
        return {"ok": True, "namespace": ns, "brand": b, "url_prefix": p, "deleted": before}
    except Exception as e:
        log.exception("[catalog/clear] falhou")
        return {"ok": False, "namespace": ns, "brand": b, "url_prefix": p, "error": str(e)}

# ---- CRUD leve -------------------------------------------------------------------------
@app.post("/catalog/upsert")
async def catalog_upsert(request: Request):
    """
    Upsert direto por URL. (Campos opcionais, normalizações incluídas)
    """
    data = await request.json()

    raw_url = (data.get("url") or "").strip()
    url = _canon_ig_url(raw_url) or raw_url
    if not url:
        return {"ok": False, "error": "Falta url"}

    ns       = (data.get("namespace") or DEFAULT_NAMESPACE)
    name     = (data.get("name") or "").strip() or url
    summary  = (data.get("summary") or "").strip()
    source   = (data.get("source") or "manual").strip() or "manual"
    brand    = (data.get("brand") or None)
    dim      = (data.get("dimensions") or None)
    material = (data.get("material") or None)
    image_url= (data.get("image_url") or None)
    ref      = (data.get("ref") or None)

    def _to_float(v):
        if v in (None, "", "0", "0.0", "0.00"): 
            return None
        s = str(v).strip().replace(" ", "")
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        if s.count(".") > 1:
            parts = s.split(".")
            s = "".join(parts[:-1]) + "." + parts[-1]
        try:
            return float(s)
        except Exception:
            return None

    price   = _to_float(data.get("price"))
    iva_pct = _to_float(data.get("iva_pct"))

    currency = (data.get("currency") or None)
    if currency:
        currency = str(currency).strip().upper() or None

    # variant_attrs pode ser dict (preferível) ou string "valores"
    variant_attrs_raw = data.get("variant_attrs") or None
    vkey = None
    variant_attrs_out = None

    if isinstance(variant_attrs_raw, dict):
        # garantir coerência com REF
        if ref and not variant_attrs_raw.get("sku"):
            variant_attrs_raw["sku"] = ref
        vkey = build_variant_key(variant_attrs_raw) or None
        variant_attrs_out = json.dumps(variant_attrs_raw, ensure_ascii=False)
    elif isinstance(variant_attrs_raw, str):
        # assumir lista de valores "A, B, C"
        variant_attrs_out = variant_attrs_raw.strip() or None
        vkey = values_to_variant_key(variant_attrs_out) or None

    _upsert_catalog_row(
        ns,
        name=name,
        summary=summary,
        url=url,
        source=source,
        ref=ref,
        price=price,
        currency=currency,
        iva_pct=iva_pct,
        brand=brand,
        dimensions=dim,
        material=material,
        image_url=image_url,
        variant_attrs=variant_attrs_out,
        variant_key=vkey
    )

    return {"ok": True, "url": url, "name": name}

@app.get("/catalog/get")
def catalog_get(url: str, namespace: str = DEFAULT_NAMESPACE):
    """Obtém 1 item por URL (com filtro de namespace)."""
    url = _canon_ig_url((url or "").strip()) or (url or "").strip()
    ns = (namespace or DEFAULT_NAMESPACE).strip()
    if not url:
        return {"ok": False, "error": "Falta url"}
    with _catalog_conn() as c:
        cur = c.execute("""SELECT id, namespace, name, summary, url, source, updated_at,
                                  ref, price, currency, iva_pct, brand, dimensions, material, image_url
                           FROM catalog_items WHERE url=? AND namespace=?""", (url, ns))
        row = cur.fetchone()
        if not row:
            return {"ok": False, "error": "não encontrado", "namespace": ns}
        return {"ok": True, "item": dict(row)}

@app.get("/catalog/list")
def catalog_list(q: str = "", limit: int = 100, offset: int = 0, namespace: str = DEFAULT_NAMESPACE):
    """
    Lista o catálogo com paginação simples. Params: ?q=&limit=&offset=&namespace=
    """
    limit = max(1, min(500, int(limit or 100)))
    offset = max(0, int(offset or 0))
    ns = (namespace or DEFAULT_NAMESPACE).strip()
    like = f"%{q.strip()}%" if q else None

    with _catalog_conn() as c:
        if like:
            cur = c.execute("""SELECT id, namespace, name, summary, url, source, created_at, updated_at,
                                      ref, price, currency, iva_pct, brand, dimensions, material, image_url
                               FROM catalog_items
                               WHERE namespace=? AND (name LIKE ? OR summary LIKE ? OR url LIKE ? OR brand LIKE ? OR ref LIKE ?)
                               ORDER BY updated_at DESC
                               LIMIT ? OFFSET ?""",
                            (ns, like, like, like, like, like, limit, offset))
            items = [dict(r) for r in cur.fetchall()]
            cur2 = c.execute("""SELECT count(*) as n FROM catalog_items
                                WHERE namespace=? AND (name LIKE ? OR summary LIKE ? OR url LIKE ? OR brand LIKE ? OR ref LIKE ?)""",
                             (ns, like, like, like, like, like))
            total = int(cur2.fetchone()["n"])
        else:
            cur = c.execute("""SELECT id, namespace, name, summary, url, source, created_at, updated_at,
                                      ref, price, currency, iva_pct, brand, dimensions, material, image_url
                               FROM catalog_items
                               WHERE namespace=?
                               ORDER BY updated_at DESC
                               LIMIT ? OFFSET ?""", (ns, limit, offset))
            items = [dict(r) for r in cur.fetchall()]
            cur2 = c.execute("SELECT count(*) as n FROM catalog_items WHERE namespace=?", (ns,))
            total = int(cur2.fetchone()["n"])

    out_items = []
    for r in items:
        out_items.append({
            "id": r["id"],
            "namespace": r["namespace"],
            "name": r["name"],
            "summary": r["summary"],
            "url": r["url"],
            "source": r["source"],
            "updated_at": r["updated_at"],
            "ref": r.get("ref"),
            "price": r.get("price"),
            "currency": r.get("currency"),
            "iva_pct": r.get("iva_pct"),
            "brand": r.get("brand"),
            "dimensions": r.get("dimensions"),
            "material": r.get("material"),
            "image_url": r.get("image_url"),
        })
    return {"ok": True, "total": total, "items": out_items}

@app.post("/catalog/resolve-price")
async def catalog_resolve_price(request: Request):
    """
    Body:
    {
      "namespace": "boasafra",
      "ref": "ORK.01",            # opcional (prioridade máxima)
      "url": "https://.../prod#sku=...",  # opcional
      "attrs": {"Cor":"Branco","Tamanho":"M"}  # opcional (para url de produto sem #sku)
    }
    Regras:
      1) ref exato -> match imediato
      2) url com #sku -> match imediato por url
      3) url pai + attrs -> variant_key
    """
    try:
      data = await request.json()
    except:
      return {"ok": False, "error": "JSON inválido"}

    ns = (data.get("namespace") or DEFAULT_NAMESPACE)
    ref = (data.get("ref") or "").strip() or None
    url = (data.get("url") or "").strip() or None
    attrs = data.get("attrs") or None

    with _catalog_conn() as c:
        # 1) por REF
        if ref:
            cur = c.execute("""
                SELECT * FROM catalog_items
                 WHERE namespace=? AND ref=? 
                 ORDER BY updated_at DESC LIMIT 1
            """, (ns, ref))
            row = cur.fetchone()
            if row:
                out = dict(row)
                return {"ok": True, "match": out, "missing_price": (out.get("price") is None)}

        # 2) por URL completa (com #sku)
        if url and "#sku=" in url:
            cur = c.execute("""
                SELECT * FROM catalog_items
                 WHERE namespace=? AND url=? 
                 ORDER BY updated_at DESC LIMIT 1
            """, (ns, _canon_ig_url(url) or url))
            row = cur.fetchone()
            if row:
                out = dict(row)
                return {"ok": True, "match": out, "missing_price": (out.get("price") is None)}

        # 3) URL do pai + attrs -> variant_key
        if url and attrs and "#sku=" not in url:
            parent = _canon_ig_url(url) or url
            vkey = build_variant_key(attrs)
            cur = c.execute("""
                SELECT * FROM catalog_items
                 WHERE namespace=? AND variant_key=? AND url LIKE ? 
                 ORDER BY updated_at DESC LIMIT 1
            """, (ns, vkey, parent + "#sku=%"))
            row = cur.fetchone()
            if row:
                out = dict(row)
                return {"ok": True, "match": out, "missing_price": (out.get("price") is None)}

            # fallback: tentar valores simples se attrs forem só valores
            if not row and isinstance(attrs, str):
                vkey2 = values_to_variant_key(attrs)
                cur = c.execute("""
                    SELECT * FROM catalog_items
                     WHERE namespace=? AND variant_key=? AND url LIKE ? 
                     ORDER BY updated_at DESC LIMIT 1
                """, (ns, vkey2, parent + "#sku=%"))
                row = cur.fetchone()
                if row:
                    out = dict(row)
                    return {"ok": True, "match": out, "missing_price": (out.get("price") is None)}

        # sem match – devolver candidatos úteis se houver contexto
        candidates = []
        if url:
            parent = (_canon_ig_url(url) or url).split("#")[0]
            cur = c.execute("""
                SELECT id,name,ref,url,price,currency,brand,variant_key FROM catalog_items
                 WHERE namespace=? AND url LIKE ?
                 ORDER BY updated_at DESC LIMIT 10
            """, (ns, parent + "#sku=%"))
            candidates = [dict(r) for r in cur.fetchall()]

        return {"ok": False, "reason": "sem_match", "candidates": candidates[:10]}

# ---- Expor info do Catálogo (SQLite) no /status ----
def _status_catalog_sqlite():
    try:
        with _catalog_conn() as c:
            total = c.execute("SELECT COUNT(*) FROM catalog_items").fetchone()[0]
            ig = c.execute(
                "SELECT COUNT(*) FROM catalog_items WHERE url LIKE ?",
                (f"https://{IG_HOST}%",)
            ).fetchone()[0]
    except Exception:
        total, ig = 0, 0
    return {
        "backend": "sqlite",
        "db_path": CATALOG_DB_PATH,
        "total": int(total),
        "ig_items": int(ig),
        "endpoints": {
            "import_csv": "POST /catalog/import-csv",
            "upsert": "POST /catalog/upsert",
            "get": "GET /catalog/get?url=&namespace=",
            "list": "GET /catalog/list?q=&limit=&offset=&namespace=",
            "clear": "POST /catalog/clear",
        }
    }
# ===================== HOTFIX — Pesquisa Tolerante (colar no fim do ficheiro) =====================

def _hf_strip_accents(s: str) -> str:
    import unicodedata, re
    s = (s or "").lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s#/.-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _hf_tokens(q: str):
    # stopwords PT + “referências vagas”
    STOP = {
        "de","do","da","dos","das","para","por","o","a","os","as",
        "um","uma","uns","umas","e","ou","com","sem","em","no","na","nos","nas",
        "este","esta","isto","isso","aquele","aquela","artigo","produto","orçamento","orcamento"
    }
    toks = [t for t in _hf_strip_accents(q).split() if t and t not in STOP and len(t) >= 2]
    return toks[:8]

def _hf_like_pat(t: str) -> str:
    # escapa curingas do LIKE
    t = t.replace("%", "\\%").replace("_", "\\_")
    return f"%{t}%"

def _hf_catalog_query_tuple(ns: str, terms: list, limit: int):
    """Monta SQL + args para AND ou OR dinamicamente."""
    base = "FROM catalog_items WHERE namespace=?"
    args = [ns]
    # cada termo procura em name/summary/ref
    conds = []
    for _ in terms:
        conds.append("(name LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\' OR ref LIKE ? ESCAPE '\\')")
    where_and = f"{base} AND " + " AND ".join(conds) if conds else base
    where_or  = f"{base} AND (" + " OR ".join(conds) + ")" if conds else base
    return where_and, where_or

# --- REDEFINE build_catalog_block com fallback tolerante ---
def build_catalog_block(question: str, namespace: Optional[str] = None, limit: int = 30) -> str:
    ns = (namespace or DEFAULT_NAMESPACE or "").strip() or "default"
    q_raw = (question or "").strip()

    # 0) tenta match exato por REF na pergunta
    import re, sqlite3
    ref_toks = re.findall(r"\b[A-Z0-9][A-Z0-9._\-]{2,}\b", q_raw, flags=re.I)

    rows: list = []
    try:
        with _catalog_conn() as c:
            cur: sqlite3.Cursor

            if ref_toks:
                q_marks = ",".join("?" * len(ref_toks))
                cur = c.execute(
                    f"""SELECT namespace,name,ref,price,url,brand,variant_attrs
                          FROM catalog_items
                         WHERE namespace=? AND ref IN ({q_marks})
                         ORDER BY updated_at DESC LIMIT ?""",
                    tuple([ns, *ref_toks, limit])
                )
                rows = [dict(r) for r in cur.fetchall()]

            # 1) pesquisa AND estrita (tokens úteis)
            if len(rows) < 3:
                terms = _hf_tokens(q_raw)
                and_sql, or_sql = _hf_catalog_query_tuple(ns, terms, limit)
                if terms:
                    like_args = []
                    for t in terms:
                        pat = _hf_like_pat(t)
                        like_args.extend([pat, pat, pat])
                    cur = c.execute(
                        f"""SELECT namespace,name,ref,price,url,brand,variant_attrs
                               {and_sql}
                               ORDER BY updated_at DESC
                               LIMIT ?""",
                        tuple([*([ns] if and_sql.startswith("FROM") else []), *like_args, limit])
                    )
                    rows = rows + [dict(r) for r in cur.fetchall()]

            # 2) fallback OR se AND não deu nada
            if not rows:
                terms = _hf_tokens(q_raw)
                if terms:
                    like_args = []
                    for t in terms:
                        pat = _hf_like_pat(t)
                        like_args.extend([pat, pat, pat])
                    _, or_sql = _hf_catalog_query_tuple(ns, terms, limit)
                    cur = c.execute(
                        f"""SELECT namespace,name,ref,price,url,brand,variant_attrs
                               {or_sql}
                               ORDER BY updated_at DESC
                               LIMIT ?""",
                        tuple([ns, *like_args, limit])
                    )
                    rows = [dict(r) for r in cur.fetchall()]

            # 3) último fallback: “frase compacta” (tokens na ordem)
            if not rows:
                compact = " ".join(_hf_tokens(q_raw))
                if compact:
                    pat = f"%{compact.replace(' ', '%')}%"
                    cur = c.execute(
                        """SELECT namespace,name,ref,price,url,brand,variant_attrs
                             FROM catalog_items
                            WHERE namespace=? AND (LOWER(name) LIKE ? ESCAPE '\\' OR LOWER(summary) LIKE ? ESCAPE '\\')
                            ORDER BY updated_at DESC
                            LIMIT ?""",
                        (ns, pat, pat, limit)
                    )
                    rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.warning(f"[catalog/hotfix] search falhou: {e}")
        rows = []

    if not rows:
        return ""  # deixa o Mission proceder (e não diz que “não há catálogo”)

    # saída compacta para o LLM
    lines = ["Catálogo interno (usa SÓ estes dados para orçamentos/preços; não usar RAG aqui):"]
    seen = set()
    for r in rows[:limit]:
        key = (r.get("ref"), r.get("url"))
        if key in seen: 
            continue
        seen.add(key)
        name = r.get("name") or "(sem nome)"
        ref  = r.get("ref") or "(sem dado)"
        price = r.get("price")
        price_txt = f"{price:.2f}€" if isinstance(price, (int,float)) else "(sem preço)"
        url = r.get("url") or "(sem URL)"
        brand = r.get("brand") or ""
        variant = (r.get("variant_attrs") or "").strip()
        name_show = f"{name} — Variante: {variant}" if variant else name
        lines.append(f"- {name_show} • SKU: {ref} • Preço: {price_txt} • Marca: {brand} • Link: {url}")
    return "\n".join(lines)

# --- REDEFINE build_catalog_variants_block para usar os mesmos tokens tolerantes ---
def build_catalog_variants_block(question: str, namespace: Optional[str]) -> str:
    qlow = (question or "").lower()
    if not any(k in qlow for k in ("variante","variantes","cores","tamanhos","opções","opcoes")):
        return ""
    ns = (namespace or DEFAULT_NAMESPACE or "").strip() or "default"
    terms = _hf_tokens(question)

    import sqlite3
    rows: List[Dict] = []
    try:
        with _catalog_conn() as c:
            if not terms:
                return ""
            like_args = []
            conds = []
            for t in terms:
                pat = _hf_like_pat(t)
                like_args.extend([pat, pat, pat])
                conds.append("(name LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\' OR ref LIKE ? ESCAPE '\\')")
            where = " AND ".join(conds) if conds else "1=1"
            cur = c.execute(f"""
                SELECT name, ref, price, url, brand, variant_attrs
                  FROM catalog_items
                 WHERE namespace=? AND {where}
                 ORDER BY updated_at DESC
                 LIMIT 160
            """, tuple([ns, *like_args]))
            rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.warning(f"[catalog variants/hotfix] falhou: {e}")
        rows = []

    if not rows:
        return ""

    # agrupar por URL-pai
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        u = (r.get("url") or "").strip()
        parent = u.split("#", 1)[0] if u else ""
        groups.setdefault(parent, []).append(r)

    # manter só grupos com >=2 variantes (com #sku=)
    groups = {
        p: [it for it in lst if "#sku=" in (it.get("url") or "")]
        for p, lst in groups.items()
    }
    groups = {p: lst for p, lst in groups.items() if len(lst) >= 2}
    if not groups:
        return ""

    out = ["CATÁLOGO INTERNO — Variantes por produto (Variante | SKU | Preço | Link):"]
    for parent, lst in groups.items():
        lst_sorted = sorted(lst, key=lambda x: (x.get("name") or "", x.get("ref") or ""))
        title = lst_sorted[0].get("name") or parent or "(produto)"
        out.append(f"• Produto: {title}")
        for r in lst_sorted[:40]:
            ref  = r.get("ref") or "(sem ref)"
            pr   = r.get("price")
            pr_s = f"{pr:.2f}€" if isinstance(pr, (int, float)) else "(sem preço)"
            url  = r.get("url") or "(sem url)"
            va   = (r.get("variant_attrs") or "").strip()
            label = va or (r.get("name") or "")
            out.append(f"   - {label} | SKU:{ref} | {pr_s} | {url}")
    return "\n".join(out)

# ===================== /HOTFIX =====================

# ========================= HOTFIX: _canon_ig_url aceita URLs relativas =========================
def _canon_ig_url(u: str) -> str:
    """Normaliza URLs relativas do catálogo para o domínio principal IG."""
    try:
        u = (u or "").strip()
        if not u:
            return ""
        if u.startswith("//"):
            u = "https:" + u
        # se for relativo, prefixar domínio
        if not u.startswith("http"):
            if not u.startswith("/"):
                u = "/" + u
            u = f"https://{IG_HOST}{u}"
        # limpeza leve
        u = u.replace(" ", "")
        while "//" in u.replace("https://", ""):
            u = u.replace("//", "/").replace("https:/", "https://")
        return u
    except Exception:
        return u or ""
# ======================= /HOTFIX ==============================================================

# ===================== HOTFIX FINAL (catálogo persistente em /data + /status typo) =====================
# Objetivo:
# 1) Forçar o SQLite do catálogo a viver em /data (Railway Volume), respeitando CATALOG_DB_PATH se existir
# 2) Re-apontar _catalog_conn() para o path final (independente da ordem do ficheiro)
# 3) Corrigir o typo no /status sem editar a função (hack: criar a key com espaço no globals)
#
# Nota: SEM migração /tmp -> /data (assume que /data/catalog.db já é a fonte correta)

import os

def _hotfix_catalog_persist_final_nomigrate():
    try:
        # 1) escolher DB persistente (Railway Volume)
        base_dir = os.getenv("CATALOG_DIR", "/data")
        os.makedirs(base_dir, exist_ok=True)

        persist_db = (os.getenv("CATALOG_DB_PATH") or "").strip()
        if not persist_db:
            persist_db = os.path.join(base_dir, "catalog.db")

        # se alguém passar /tmp/... por engano, reencaminhar para /data/...
        if persist_db.startswith("/tmp/"):
            persist_db = os.path.join(base_dir, os.path.basename(persist_db))

        # 2) forçar env + globals para o novo caminho
        os.environ["CATALOG_DB_PATH"] = persist_db
        globals()["CATALOG_DB_PATH"] = persist_db

        # 3) redefinir _catalog_conn para usar o novo path (mesmo que já existisse acima)
        import sqlite3
        def _catalog_conn_persist():
            conn = sqlite3.connect(persist_db)
            conn.row_factory = sqlite3.Row
            return conn
        globals()["_catalog_conn"] = _catalog_conn_persist

        # 4) garantir schema/índices (reusa _ensure_cols se existir)
        if "_ensure_cols" in globals():
            with _catalog_conn_persist() as conn:
                globals()["_ensure_cols"](conn)

        # 5) corrigir o typo do /status:
        #    se a tua rota fizer: if ' _status_catalog_sqlite' in globals()
        #    criamos essa key com espaço, apontando para a função real
        if "_status_catalog_sqlite" in globals():
            globals()[" _status_catalog_sqlite"] = globals()["_status_catalog_sqlite"]

        print(f"[hotfix] catalog DB persistente em: {persist_db}")
        return True
    except Exception as e:
        print(f"[hotfix] catalog persist FAILED: {e}")
        return False

_hotfix_catalog_persist_final_nomigrate()

# ===================== /HOTFIX FINAL =====================================================================

# ===================== HOTFIX FINAL — Forçar pesquisa tolerante ativa =====================
# (só faz bind se as versões tolerantes já estiverem definidas no ficheiro)

def _hotfix_force_tolerant_search_active():
    try:
        # Se já colaste antes as versões tolerantes (com _hf_tokens/_hf_like_pat),
        # garante que são as que ficam ativas no fim do ficheiro.
        if "_hf_tokens" in globals() and "_hf_like_pat" in globals():
            # Se por acaso o ficheiro voltou a redefinir build_catalog_block depois,
            # aqui “redefine” outra vez (última definição vence).
            # Se ainda NÃO tens as versões tolerantes no ficheiro, isto não resolve (ver nota abaixo).

            # --- RE-DECLARAÇÃO TOLERANTE (curta) ---
            import re, sqlite3
            from typing import Optional, List, Dict

            def build_catalog_block(question: str, namespace: Optional[str] = None, limit: int = 30) -> str:
                ns = (namespace or DEFAULT_NAMESPACE or "").strip() or "default"
                q_raw = (question or "").strip()
                rows = []

                # 0) match exato por ref
                ref_toks = re.findall(r"\b[A-Z0-9][A-Z0-9._\-]{2,}\b", q_raw, flags=re.I)
                try:
                    with _catalog_conn() as c:
                        if ref_toks:
                            q_marks = ",".join("?" * len(ref_toks))
                            cur = c.execute(
                                f"""SELECT name,ref,price,url,brand,variant_attrs
                                      FROM catalog_items
                                     WHERE namespace=? AND ref IN ({q_marks})
                                     ORDER BY updated_at DESC LIMIT ?""",
                                tuple([ns, *ref_toks, limit])
                            )
                            rows = [dict(r) for r in cur.fetchall()]

                        # 1) AND (tokens úteis)
                        if not rows:
                            terms = _hf_tokens(q_raw)
                            if terms:
                                conds = []
                                like_args = []
                                for t in terms:
                                    pat = _hf_like_pat(t)
                                    conds.append("(name LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\' OR ref LIKE ? ESCAPE '\\')")
                                    like_args.extend([pat, pat, pat])
                                where = " AND ".join(conds)
                                cur = c.execute(
                                    f"""SELECT name,ref,price,url,brand,variant_attrs
                                          FROM catalog_items
                                         WHERE namespace=? AND {where}
                                         ORDER BY (CASE WHEN url LIKE '%#sku=%' THEN 0 ELSE 1 END), updated_at DESC
                                         LIMIT ?""",
                                    tuple([ns, *like_args, limit])
                                )
                                rows = [dict(r) for r in cur.fetchall()]

                        # 2) OR fallback
                        if not rows:
                            terms = _hf_tokens(q_raw)
                            if terms:
                                conds = []
                                like_args = []
                                for t in terms:
                                    pat = _hf_like_pat(t)
                                    conds.append("(name LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\' OR ref LIKE ? ESCAPE '\\')")
                                    like_args.extend([pat, pat, pat])
                                where = " OR ".join(conds)
                                cur = c.execute(
                                    f"""SELECT name,ref,price,url,brand,variant_attrs
                                          FROM catalog_items
                                         WHERE namespace=? AND ({where})
                                         ORDER BY (CASE WHEN url LIKE '%#sku=%' THEN 0 ELSE 1 END), updated_at DESC
                                         LIMIT ?""",
                                    tuple([ns, *like_args, limit])
                                )
                                rows = [dict(r) for r in cur.fetchall()]
                except Exception as e:
                    log.warning(f"[catalog/tolerant] search falhou: {e}")
                    rows = []

                if not rows:
                    return ""

                lines = ["CATÁLOGO INTERNO (tolerante; usar só estes dados p/ preços/orçamentos):"]
                seen = set()
                for r in rows[:limit]:
                    key = (r.get("ref"), r.get("url"))
                    if key in seen:
                        continue
                    seen.add(key)
                    name = r.get("name") or "(sem nome)"
                    ref  = r.get("ref") or "(sem ref)"
                    price = r.get("price")
                    price_txt = f"{price:.2f}€" if isinstance(price, (int,float)) else "(sem preço)"
                    url = r.get("url") or "(sem url)"
                    brand = r.get("brand") or ""
                    variant = (r.get("variant_attrs") or "").strip()
                    name_show = f"{name} — Variante: {variant}" if variant else name
                    lines.append(f"- {name_show} • SKU:{ref} • Preço:{price_txt} • Marca:{brand} • Link:{url}")
                return "\n".join(lines)

            # rebind no globals()
            globals()["build_catalog_block"] = build_catalog_block

            log.info("[hotfix] pesquisa tolerante: build_catalog_block forçado (última definição).")
            return True

        log.info("[hotfix] pesquisa tolerante: não encontrada (_hf_tokens/_hf_like_pat ausentes).")
        return False

    except Exception as e:
        log.warning(f"[hotfix] force tolerante falhou: {e}")
        return False

_hotfix_force_tolerant_search_active()
# ===================== /HOTFIX FINAL — Forçar pesquisa tolerante ativa =====================

# ===================== HOTFIX FINAL — Resolver incoerências por redefinições =====================
def _final_fix_incoherences():
    """
    Resolve incoerências típicas do teu ficheiro:
    1) Múltiplas redefinições: fixa (no fim) qual versão fica ativa em runtime.
    2) Corrige o typo do /status: a rota verifica ' _status_catalog_sqlite' com espaço.
    3) Garante uma _canon_ig_url final coerente (aceita relativos e normaliza para IG_HOST).
    Nota: não toca em modelo Grok/XAI nem adiciona endpoints.
    """
    import logging
    from urllib.parse import urlparse

    log = logging.getLogger("alma")

    # --- (A) Fix /status typo sem editar a rota: criar a key com espaço no globals() ---
    if "_status_catalog_sqlite" in globals():
        globals()[" _status_catalog_sqlite"] = globals()["_status_catalog_sqlite"]

    # --- (B) Fixar UMA _canon_ig_url final (evita a tua alternância entre versões) ---
    def _canon_ig_url_final(u: str) -> str:
        try:
            u = (u or "").strip()
            if not u:
                return ""
            if u.startswith("//"):
                u = "https:" + u
            # relativo -> absoluto no IG_HOST
            if not u.startswith("http"):
                if not u.startswith("/"):
                    u = "/" + u
                u = f"https://{IG_HOST}{u}"
            # canon host + limpeza leve
            u = u.replace(" ", "")
            p = urlparse(u)
            host = (p.netloc or "").lower().replace("www.", "")
            if IG_HOST and IG_HOST in host:
                # força https + netloc “limpo”
                u = u.replace(p.scheme + "://", "https://", 1)
                u = u.replace(p.netloc, IG_HOST, 1)
            return u
        except Exception:
            return (u or "")

    globals()["_canon_ig_url"] = _canon_ig_url_final

    # --- (C) Fixar pesquisa do catálogo: preferir as versões tolerantes se existirem ---
    # Tens várias versões (patch, hotfix, redef). Aqui escolhemos uma só:
    # - Se existirem helpers do hotfix tolerante (_hf_tokens/_hf_like_pat), usa-os.
    # - Caso contrário, mantém a última definição existente (não inventamos outra).
    if ("_hf_tokens" in globals()) and ("_hf_like_pat" in globals()) and ("_catalog_conn" in globals()):
        import re
        from typing import Optional, List, Dict

        def build_catalog_block_final(question: str, namespace: Optional[str] = None, limit: int = 120) -> str:
            ns = (namespace or DEFAULT_NAMESPACE or "").strip() or "default"
            q_raw = (question or "").strip()

            rows: List[Dict] = []
            ref_toks = re.findall(r"\b[A-Z0-9][A-Z0-9._\-]{2,}\b", q_raw, flags=re.I)

            try:
                with _catalog_conn() as c:
                    # 0) match exato por ref
                    if ref_toks:
                        q_marks = ",".join("?" * len(ref_toks))
                        cur = c.execute(
                            f"""SELECT name,ref,price,url,brand,variant_attrs
                                  FROM catalog_items
                                 WHERE namespace=? AND ref IN ({q_marks})
                                 ORDER BY updated_at DESC
                                 LIMIT ?""",
                            tuple([ns, *ref_toks, limit])
                        )
                        rows = [dict(r) for r in cur.fetchall()]

                    # 1) AND tokens úteis
                    if not rows:
                        terms = _hf_tokens(q_raw)
                        if terms:
                            conds = []
                            like_args = []
                            for t in terms:
                                pat = _hf_like_pat(t)
                                conds.append("(name LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\' OR ref LIKE ? ESCAPE '\\')")
                                like_args.extend([pat, pat, pat])
                            where = " AND ".join(conds)
                            cur = c.execute(
                                f"""SELECT name,ref,price,url,brand,variant_attrs
                                      FROM catalog_items
                                     WHERE namespace=? AND {where}
                                     ORDER BY (CASE WHEN url LIKE '%#sku=%' THEN 0 ELSE 1 END), updated_at DESC
                                     LIMIT ?""",
                                tuple([ns, *like_args, limit])
                            )
                            rows = [dict(r) for r in cur.fetchall()]

                    # 2) OR fallback
                    if not rows:
                        terms = _hf_tokens(q_raw)
                        if terms:
                            conds = []
                            like_args = []
                            for t in terms:
                                pat = _hf_like_pat(t)
                                conds.append("(name LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\' OR ref LIKE ? ESCAPE '\\')")
                                like_args.extend([pat, pat, pat])
                            where = " OR ".join(conds)
                            cur = c.execute(
                                f"""SELECT name,ref,price,url,brand,variant_attrs
                                      FROM catalog_items
                                     WHERE namespace=? AND ({where})
                                     ORDER BY (CASE WHEN url LIKE '%#sku=%' THEN 0 ELSE 1 END), updated_at DESC
                                     LIMIT ?""",
                                tuple([ns, *like_args, limit])
                            )
                            rows = [dict(r) for r in cur.fetchall()]

            except Exception as e:
                log.warning(f"[catalog/final] search falhou: {e}")
                rows = []

            if not rows:
                return ""

            lines = ["CATÁLOGO INTERNO (final; usar só estes dados p/ preços/orçamentos):"]
            seen = set()
            for r in rows[:limit]:
                key = (r.get("ref"), r.get("url"))
                if key in seen:
                    continue
                seen.add(key)

                name = r.get("name") or "(sem nome)"
                ref  = r.get("ref") or "(sem ref)"
                price = r.get("price")
                price_txt = f"{price:.2f}€" if isinstance(price, (int, float)) else "(sem preço)"
                url = r.get("url") or "(sem url)"
                brand = r.get("brand") or ""
                variant = (r.get("variant_attrs") or "").strip()
                name_show = f"{name} — Variante: {variant}" if variant else name

                lines.append(f"- {name_show} • SKU:{ref} • Preço:{price_txt} • Marca:{brand} • Link:{url}")
            return "\n".join(lines)

        def build_catalog_variants_block_final(question: str, namespace: Optional[str]) -> str:
            qlow = (question or "").lower()
            if not any(k in qlow for k in ("variante", "variantes", "cores", "tamanhos", "opções", "opcoes")):
                return ""

            ns = (namespace or DEFAULT_NAMESPACE or "").strip() or "default"
            terms = _hf_tokens(question)
            if not terms:
                return ""

            rows: List[Dict] = []
            try:
                with _catalog_conn() as c:
                    conds = []
                    like_args = []
                    for t in terms:
                        pat = _hf_like_pat(t)
                        conds.append("(name LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\' OR ref LIKE ? ESCAPE '\\')")
                        like_args.extend([pat, pat, pat])
                    where = " AND ".join(conds)
                    cur = c.execute(
                        f"""SELECT name, ref, price, url, brand, variant_attrs
                              FROM catalog_items
                             WHERE namespace=? AND {where}
                             ORDER BY updated_at DESC
                             LIMIT 160""",
                        tuple([ns, *like_args])
                    )
                    rows = [dict(r) for r in cur.fetchall()]
            except Exception as e:
                log.warning(f"[catalog variants/final] falhou: {e}")
                rows = []

            if not rows:
                return ""

            # agrupar por URL-pai e manter só grupos com >=2 variantes (#sku=)
            groups: Dict[str, List[Dict]] = {}
            for r in rows:
                u = (r.get("url") or "").strip()
                parent = u.split("#", 1)[0] if u else ""
                if parent:
                    groups.setdefault(parent, []).append(r)

            groups = {
                p: [it for it in lst if "#sku=" in (it.get("url") or "")]
                for p, lst in groups.items()
            }
            groups = {p: lst for p, lst in groups.items() if len(lst) >= 2}
            if not groups:
                return ""

            out = ["CATÁLOGO INTERNO — Variantes por produto (Variante | SKU | Preço | Link):"]
            for parent, lst in groups.items():
                lst_sorted = sorted(lst, key=lambda x: (x.get("name") or "", x.get("ref") or ""))
                title = lst_sorted[0].get("name") or parent or "(produto)"
                out.append(f"• Produto: {title}")
                for r in lst_sorted[:40]:
                    ref = r.get("ref") or "(sem ref)"
                    pr = r.get("price")
                    pr_s = f"{pr:.2f}€" if isinstance(pr, (int, float)) else "(sem preço)"
                    url = r.get("url") or "(sem url)"
                    va = (r.get("variant_attrs") or "").strip()
                    label = va or (r.get("name") or "")
                    out.append(f"   - {label} | SKU:{ref} | {pr_s} | {url}")
            return "\n".join(out)

        globals()["build_catalog_block"] = build_catalog_block_final
        globals()["build_catalog_variants_block"] = build_catalog_variants_block_final
        log.info("[hotfix-final] catálogo: build_catalog_block + variants fixados (tolerante).")
    else:
        log.info("[hotfix-final] catálogo: mantidas definições existentes (helpers tolerantes não encontrados).")

_final_fix_incoherences()
# ===================== /HOTFIX FINAL =============================================================
# ==== HOTFIX: variantes por url_canonical + continuidade de orçamento por sessão ====
import re
import sqlite3
from typing import Optional, List, Dict, Any, Tuple

def _db() -> sqlite3.Connection:
    con = sqlite3.connect(CATALOG_DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def _is_quote_intent(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["orçamento", "orcamento", "cotação", "cotacao", "proforma", "preço", "preco"])

def _pick_ref_or_name(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Heurística simples:
    - ref/SKU: coisas do tipo 740021316-2.-NI ou ORK.02.03 (adaptar se necessário)
    - nome: fallback = frase sem números longos
    """
    if not q:
        return None, None
    # SKU/ref típico com dígitos+traços/pontos
    m = re.search(r'\b[A-Z]{2,}\.\d{2}\.\d{2}\b', q)
    if m:
        return m.group(0), None
    m = re.search(r'\b\d{6,}[-.\w]*\b', q)
    if m:
        return m.group(0), None
    # nome: remove quantidades simples
    name = re.sub(r'\b\d+\s*(x|un|uni|unidade|unidades)\b', '', q, flags=re.I).strip()
    return None, name or None

def _get_item_by_ref(namespace: str, ref: str) -> Optional[Dict[str, Any]]:
    with _db() as con:
        row = con.execute(
            "SELECT * FROM catalog_items WHERE namespace=? AND ref=? LIMIT 1",
            (namespace, ref)
        ).fetchone()
        return dict(row) if row else None

def _get_item_by_name_like(namespace: str, name: str) -> Optional[Dict[str, Any]]:
    with _db() as con:
        row = con.execute(
            "SELECT * FROM catalog_items WHERE namespace=? AND name LIKE ? ORDER BY id DESC LIMIT 1",
            (namespace, f"%{name}%")
        ).fetchone()
        return dict(row) if row else None

def _get_variants_by_canonical(namespace: str, url_canonical: str) -> List[Dict[str, Any]]:
    with _db() as con:
        rows = con.execute(
            """
            SELECT * FROM catalog_items
            WHERE namespace=? AND url_canonical=?
            ORDER BY ref ASC
            """,
            (namespace, url_canonical)
        ).fetchall()
        return [dict(r) for r in rows]

def _build_variants_block(variants: List[Dict[str, Any]]) -> str:
    """
    Bloco compacto para a LLM: Variante + ref + price + url (usa url_canonical para link base)
    NOTA: assumes que tens url_canonical e variant_attrs/name/price/ref.
    """
    if not variants:
        return ""
    lines = []
    for v in variants:
        variant_txt = (v.get("variant_attrs") or "").strip()
        if not variant_txt:
            # fallback: tenta extrair "Variante:" do summary já limpo no vosso pipeline
            summary = (v.get("summary") or "")
            m = re.search(r'(?im)^\s*Variante:\s*(.+?)\s*$', summary)
            variant_txt = m.group(1).strip() if m else ""
        if not variant_txt:
            # último fallback: tentar ler o sufixo do nome
            variant_txt = (v.get("name") or "").split("—")[-1].strip() if "—" in (v.get("name") or "") else ""
        price = v.get("price")
        ref = v.get("ref")
        link = (v.get("url_canonical") or v.get("url") or "").strip()
        lines.append(f"- {variant_txt or '(sem dado)'} | SKU: {ref or '(sem dado)'} | {price if price is not None else '(sem dado)'}€ | {link}")
    return "VARIANTES (catálogo interno; listar só isto):\n" + "\n".join(lines)

def _extract_last_selected_refs_from_local_history(user_id: str) -> List[str]:
    """
    Usa o vosso histórico local (o que o local_append_dialog guarda).
    Implementação placeholder: tens de ligar à tua função real de leitura do histórico.
    Se já tens algo como local_get_dialog(user_id), usa isso.
    """
    try:
        hist = local_get_dialog(user_id)  # <-- precisa existir no teu projeto
    except Exception:
        return []
    if not hist:
        return []
    # Procura SKUs no texto da Alma que tenham sido apresentados/confirmados
    text = "\n".join([h.get("content","") for h in hist if isinstance(h, dict)])
    refs = re.findall(r'\bSKU[:\s]*([A-Za-z0-9.\-]+)\b', text, flags=re.I)
    # dedup mantendo ordem
    out = []
    seen = set()
    for r in refs:
        if r not in seen:
            seen.add(r); out.append(r)
    return out[-10:]  # não inchar

def _inject_quote_state(messages: List[Dict[str, str]], user_id: str, namespace: str) -> List[Dict[str, str]]:
    """
    Acrescenta ao system/context uma lista dos SKUs já escolhidos nesta sessão (se houver).
    Ajuda a não “perder” quando o user diz "junta mais um produto".
    """
    refs = _extract_last_selected_refs_from_local_history(user_id)
    if not refs:
        return messages
    state_lines = []
    for ref in refs:
        item = _get_item_by_ref(namespace, ref)
        if item:
            state_lines.append(f"- {item.get('name','')} | SKU: {item.get('ref')} | {item.get('price')}€ | {item.get('url_canonical') or item.get('url','')}")
    if not state_lines:
        return messages

    state_block = "ESTADO DE ORÇAMENTO (sessão atual; itens já selecionados):\n" + "\n".join(state_lines)
    # mete como mensagem system extra no início (sem mexer no vosso mission)
    return ([{"role":"system","content": state_block}] + messages)

def hotfix_variants_and_memory(messages: List[Dict[str,str]], user_id: str, question: str, namespace: str) -> List[Dict[str,str]]:
    """
    - Se for orçamento: tenta identificar o produto alvo e injeta TODAS as variantes pelo url_canonical.
    - Injeta estado do orçamento da sessão (continuidade).
    """
    messages2 = _inject_quote_state(messages, user_id, namespace)

    if not _is_quote_intent(question):
        return messages2

    ref, name = _pick_ref_or_name(question)
    item = None
    if ref:
        item = _get_item_by_ref(namespace, ref)
    if not item and name:
        item = _get_item_by_name_like(namespace, name)

    if not item:
        return messages2

    url_canon = (item.get("url_canonical") or "").strip()
    if not url_canon:
        return messages2

    variants = _get_variants_by_canonical(namespace, url_canon)
    block = _build_variants_block(variants)
    if not block:
        return messages2

    messages2 = ([{"role":"system","content": block}] + messages2)
    return messages2


# ======================================================================

# ============================================================
# HOTFIX: CATALOG CONTEXT STRONG (v2)
# - Faz lookup direto em /data/catalog.db
# - Injeta contexto com TODAS as variantes encontradas
# - Ajuda multi-produto (4 itens na mesma pergunta)
# ============================================================

import re
import sqlite3
from typing import List, Dict, Any, Tuple

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _extract_slugs(text: str) -> List[str]:
    # apanha /zeal-laser ou zeal-laser (se vier com /)
    slugs = re.findall(r"\/([a-z0-9][a-z0-9\-]{2,})\/?", text.lower())
    # remove coisas óbvias que não são slugs de produto
    blacklist = {"ask", "catalog", "import", "search", "static"}
    slugs = [s for s in slugs if s not in blacklist]
    # dedup mantendo ordem
    out = []
    seen = set()
    for s in slugs:
        if s not in seen:
            out.append(s); seen.add(s)
    return out

def _extract_skus(text: str) -> List[str]:
    # SKU “tipo BigCommerce”: letras/números + . _ -
    # Ex.: ORK.02.03  | KOR0440--ME-AS | LAM0100
    raw = re.findall(r"\b[A-Z0-9][A-Z0-9._\-]{2,}\b", text.upper())
    # dedup mantendo ordem
    out = []
    seen = set()
    for x in raw:
        if x not in seen:
            out.append(x); seen.add(x)
    return out[:30]

def _extract_keywords(text: str) -> List[str]:
    # palavras “grandes” que ajudam no LIKE (ex.: zeal, orikomi, dublexo)
    words = re.findall(r"[A-Za-zÀ-ÿ0-9\-]{3,}", text)
    words = [w.lower() for w in words]
    stop = {
        "para","com","sem","que","isto","isso","uma","uns","umas","dos","das","por",
        "preco","preço","iva","incluido","incluído","portes","entrega","prazo",
        "quero","preciso","faz","diz","lista","mostra","acrescenta","adiciona",
        "produto","produtos","variante","variantes","catalogo","catálogo"
    }
    keep = [w for w in words if w not in stop and len(w) >= 3]
    # dedup ordem
    out = []
    seen=set()
    for w in keep:
        if w not in seen:
            out.append(w); seen.add(w)
    return out[:12]

def _db_connect():
    # usa a tua variável global já existente, senão fallback
    path = globals().get("CATALOG_DB_PATH") or "/data/catalog.db"
    return sqlite3.connect(path)

def _db_rows_by_ref(ns: str, refs: List[str]) -> List[Dict[str, Any]]:
    if not refs:
        return []
    ns = ns or globals().get("DEFAULT_NAMESPACE") or "boasafra"
    qmarks = ",".join(["?"] * len(refs))
    sql = f"""
      SELECT id, namespace, ref, name, summary, price, brand, url
      FROM catalog_items
      WHERE namespace = ?
        AND ref IN ({qmarks})
      ORDER BY CASE WHEN url LIKE '%#sku=%' THEN 0 ELSE 1 END, ref
      LIMIT 200
    """
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, [ns, *refs])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()

def _db_rows_by_like(ns: str, slugs: List[str], keywords: List[str]) -> List[Dict[str, Any]]:
    ns = ns or globals().get("DEFAULT_NAMESPACE") or "boasafra"
    # montamos cláusulas OR com LIKE em name e url
    like_terms = []
    params: List[Any] = [ns]

    for s in (slugs or []):
        like_terms.append("(url LIKE ? OR name LIKE ?)")
        params.append(f"%/{s}%")
        params.append(f"%{s}%")

    for k in (keywords or []):
        like_terms.append("(name LIKE ? OR summary LIKE ?)")
        params.append(f"%{k}%")
        params.append(f"%{k}%")

    if not like_terms:
        return []

    sql = f"""
      SELECT id, namespace, ref, name, summary, price, brand, url
      FROM catalog_items
      WHERE namespace = ?
        AND ({' OR '.join(like_terms)})
      ORDER BY
        CASE WHEN url LIKE '%#sku=%' THEN 0 ELSE 1 END,
        CASE WHEN ref IS NULL OR ref='' THEN 1 ELSE 0 END,
        ref
      LIMIT 300
    """
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()

def _parent_url(url: str) -> str:
    u = (url or "").strip()
    if "#sku=" in u:
        return u.split("#sku=", 1)[0]
    return u

def _group_variants(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """
    Agrupa por (parent_url, name_base aproximado):
    - parent_url = antes de #sku=
    - name_base = parte antes de " — " se existir
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        url = r.get("url") or ""
        parent = _parent_url(url)
        name = r.get("name") or ""
        base = name.split(" — ", 1)[0].strip()
        key = (parent, base.lower())
        groups.setdefault(key, []).append(r)
    # ordenar dentro de cada grupo
    for k, lst in groups.items():
        lst.sort(key=lambda x: (0 if "#sku=" in (x.get("url") or "") else 1, (x.get("ref") or "")))
    return groups

def _catalog_context_block(ns: str, question: str) -> str:
    skus = _extract_skus(question)
    slugs = _extract_slugs(question)
    keys  = _extract_keywords(question)

    rows = []
    rows += _db_rows_by_ref(ns, skus)
    rows += _db_rows_by_like(ns, slugs, keys)

    # dedup por id
    seen = set()
    uniq = []
    for r in rows:
        rid = r.get("id")
        if rid in seen:
            continue
        seen.add(rid)
        uniq.append(r)

    if not uniq:
        return ""

    groups = _group_variants(uniq)

    lines = []
    lines.append("CATALOGO_INTERNO (catalog_items) — DADOS BRUTOS PARA USAR (SEM INVENTAR)")
    lines.append(f"namespace={ns or globals().get('DEFAULT_NAMESPACE') or 'boasafra'}")
    lines.append("Regras: se houver variantes (#sku=), dar prioridade a essas linhas.")

    # lista por grupos, e dentro do grupo todas as linhas (isto é o que te faltava)
    for (parent, base), lst in groups.items():
        lines.append("")
        lines.append(f"PRODUTO_BASE: {base} | parent_url: {parent}")
        for r in lst:
            ref = r.get("ref") or ""
            price = r.get("price")
            brand = r.get("brand") or ""
            url = r.get("url") or ""
            # tenta extrair Variante: do summary se existir, senão vazio
            summary = (r.get("summary") or "")
            m = re.search(r"(?:^|\n)\s*Variante:\s*(.+)", summary, flags=re.IGNORECASE)
            variant = (m.group(1).strip() if m else "")
            # linha curta mas completa
            lines.append(f"- VAR | ref={ref} | price={price} | brand={brand} | variant={variant} | url={url}")

    return "\n".join(lines)

# ---- Monkey-patch do /ask ----
# Guardar a função original, se existir
_ASK_ORIGINAL = globals().get("ask")

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    user_id = (data.get("user_id") or "").strip() or "anon"
    namespace = (data.get("namespace") or "").strip() or None
    req_top_k = data.get("top_k")
    decided_top_k = _decide_top_k(question, req_top_k)

    log.info(f"[/ask] user_id={user_id} ns={namespace or DEFAULT_NAMESPACE} top_k={decided_top_k} question={question!r}")
    if not question:
        return {"answer": "Coloca a tua pergunta em 'question'."}

    # mantém o teu fluxo atual
    _ = build_rag_products_block(question)
    messages, new_facts, rag_used, rag_hits = build_messages(user_id, question, namespace)

    # 🔥 HOTFIX: injeta contexto do catálogo (forte)
    try:
        cat_block = _catalog_context_block(namespace or DEFAULT_NAMESPACE, question)
        if cat_block:
            messages.append({"role": "system", "content": cat_block})
    except Exception as e:
        log.exception(f"[hotfix catalog context] falhou: {e}")

    try:
        answer = grok_chat(messages)
    except Exception as e:
        log.exception("Erro ao chamar a x.ai")
        return {"answer": f"Erro ao chamar o Grok-4: {e}"}

    answer = _postprocess_answer(answer, question, namespace, decided_top_k)

    # ⚠️ se tens isto a forçar links do RAG, mantém, mas aqui é onde costuma estragar a “pureza catálogo-only”
    # Se estiveres a ver links do RAG em orçamentos, o problema é este force_links_block.
    answer = _force_links_block(answer, rag_hits, min_links=3)

    local_append_dialog(user_id, question, answer)
    _mem0_create(content=f"User: {question}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})
    _mem0_create(content=f"Alma: {answer}", user_id=user_id, metadata={"source": "alma-server", "type": "dialog"})

    return {
        "answer": answer,
        "new_facts_detected": new_facts,
        "rag": {
            "used": rag_used,
            "top_k_default": RAG_TOP_K_DEFAULT,
            "top_k_effective": decided_top_k,
            "namespace": namespace or DEFAULT_NAMESPACE
        }
    }

# =======================================================================================
# HOTFIX (colar NO FIM do main.py)
# - Variantes completas via url_canonical (usa _get_variants_by_canonical já existente)
# - Injeção do bloco de variantes também em pedidos de orçamento/preço
# =======================================================================================

import re
from typing import Optional, List, Dict, Any

def _is_budget_intent(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ("orçament", "orcament", "preço", "preco", "cotação", "cotacao", "proforma", "pro-forma"))

def _is_variants_intent(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ("variante", "variantes", "cores", "tamanhos", "opções", "opcoes", "opcao", "opção", "todas as variantes"))

def _canon_parent(url: str) -> str:
    u = (url or "").strip()
    return u.split("#", 1)[0] if u else ""

def _seed_row_for_product(question: str, ns: str) -> Optional[Dict[str, Any]]:
    """
    Escolhe 1 linha semente (preferindo variantes) para obter o url_canonical correto.
    Usa a pergunta ORIGINAL (não a query limpa), para não perder termos.
    """
    try:
        with _catalog_conn() as c:
            like = f"%{_sanitize_like(_catalog_query_from_question(question))}%"
            row = c.execute("""
                SELECT *
                  FROM catalog_items
                 WHERE namespace=?
                   AND (name LIKE ? OR summary LIKE ? OR ref LIKE ?)
                 ORDER BY (CASE WHEN url LIKE '%#sku=%' THEN 0 ELSE 1 END),
                          updated_at DESC
                 LIMIT 1
            """, (ns, like, like, like)).fetchone()
            return dict(row) if row else None
    except Exception as e:
        try:
            log.warning(f"[hotfix] seed_row falhou: {e}")
        except Exception:
            pass
        return None

def _format_variants_block(product_label: str, variants: List[Dict[str, Any]]) -> str:
    if not variants or len(variants) < 2:
        return ""

    lines = [
        "VARIANTES (catálogo interno — LISTAR TODAS as opções abaixo; NÃO resumir; NÃO omitir):",
        f"Produto: {product_label or '(sem nome)'}"
    ]
    seen = set()
    for v in variants:
        ref = (v.get("ref") or "").strip() or "(sem dado)"
        url = (v.get("url") or "").strip() or "(sem URL)"
        key = (ref, url)
        if key in seen:
            continue
        seen.add(key)

        price = v.get("price")
        price_txt = f"{price:.2f}€" if isinstance(price, (int, float)) else "(sem preço)"
        var_txt = (v.get("variant_attrs") or "").strip()
        if not var_txt:
            var_txt = (v.get("name") or "").strip() or "(sem dado)"
        lines.append(f"- Variante: {var_txt} • SKU: {ref} • Preço: {price_txt} • Link: {url}")

    return "\n".join(lines)

# ---------------------------------------------------------------------------------------
# OVERRIDE 1: build_catalog_variants_block  (agora usa url_canonical e _get_variants_by_canonical)
# ---------------------------------------------------------------------------------------
def build_catalog_variants_block(question: str,
                                 namespace: Optional[str] = None,
                                 limit_families: int = 5,
                                 limit_variants: int = 500) -> str:
    ns = (namespace or DEFAULT_NAMESPACE).strip()

    # Só gera bloco quando há intenção clara de variantes (ou orçamento — tratado no build_messages abaixo)
    if not _is_variants_intent(question):
        return ""

    seed = _seed_row_for_product(question, ns)
    if not seed:
        return ""

    url_can = (seed.get("url_canonical") or "").strip()
    if not url_can:
        # fallback: se por algum motivo não houver canonical na seed, tenta agrupar por parent(url)
        parent = _canon_parent(seed.get("url") or "")
        if not parent:
            return ""
        try:
            with _catalog_conn() as c:
                rows = c.execute("""
                    SELECT name, ref, price, url, brand, variant_attrs
                      FROM catalog_items
                     WHERE namespace=? AND (url LIKE ? OR url=?)
                     ORDER BY ref ASC
                     LIMIT ?
                """, (ns, parent + "#%", parent, limit_variants)).fetchall()
                variants = [dict(r) for r in rows] if rows else []
        except Exception:
            variants = []
        return _format_variants_block(seed.get("name") or parent, variants)

    # caminho principal: por url_canonical
    try:
        variants = _get_variants_by_canonical(ns, url_can) or []
    except Exception as e:
        try:
            log.warning(f"[hotfix] _get_variants_by_canonical falhou: {e}")
        except Exception:
            pass
        variants = []

    # respeita limite, mas suficientemente alto para não cortar (tu tens 6)
    variants = variants[:limit_variants]
    return _format_variants_block(seed.get("name") or url_can, variants)

# ---------------------------------------------------------------------------------------
# OVERRIDE 2: build_messages (injeção de variantes também em orçamento/preço)
# ---------------------------------------------------------------------------------------
def build_messages(user_id: str, question: str, namespace: Optional[str]):
    # 1) sinais contextuais → mem0
    new_facts = extract_contextual_facts_pt(question)
    for k, v in new_facts.items():
        mem0_set_fact(user_id, k, v)

    # 2) memórias recentes
    short_snippets = _mem0_search(question, user_id=user_id, limit=5) or local_search_snippets(user_id, limit=5)
    memory_block = (
        "Memórias recentes do utilizador (curto prazo):\n"
        + "\n".join(f"- {s}" for s in short_snippets[:3])
        if short_snippets else ""
    )

    # 3) CATÁLOGO INTERNO (SQLite) — SEMPRE antes do RAG
    ns = (namespace or DEFAULT_NAMESPACE).strip()

    # importante: usa query "limpa" para o bloco de catálogo (melhor recall),
    # mas usa a pergunta ORIGINAL para seed/variantes via canonical
    cat_q = _catalog_query_from_question(question)

    catalog_block = build_catalog_block(cat_q, ns)

    # 🔥 chave do hotfix:
    # - se for orçamento/preço: injeta bloco de variantes MESMO que o utilizador não tenha dito "variantes"
    # - se for pedido de variantes: idem
    if _is_budget_intent(question) or _is_variants_intent(question):
        catalog_variants_block = build_catalog_variants_block(question, ns, limit_families=5, limit_variants=80)
    else:
        catalog_variants_block = ""

    # 4) RAG (apenas para conteúdo corporativo, NÃO para preços/orçamentos)
    rag_block = ""
    rag_used = False
    rag_hits: List[dict] = []
    if RAG_READY:
        try:
            rag_hits = search_chunks(
                query=question,
                namespace=namespace or DEFAULT_NAMESPACE,
                top_k=RAG_TOP_K_DEFAULT
            ) or []
            rag_block = build_context_block(rag_hits, token_budget=RAG_CONTEXT_TOKEN_BUDGET) if rag_hits else ""
            rag_used = bool(rag_block)
        except Exception as e:
            try:
                log.warning(f"[rag] search falhou: {e}")
            except Exception:
                pass
            rag_block, rag_used, rag_hits = "", False, []

    # 5) links candidatos do RAG
    links_pairs = _links_from_matches(rag_hits, max_links=8)
    links_block = ""
    if links_pairs:
        lines = ["Links candidatos (do RAG; NÃO usar para preços/orçamentos):"]
        for title, url in links_pairs:
            lines.append(f"- {title or '-'} — {url}")
        links_block = "\n".join(lines)

    # 6) bloco de produtos (opcional)
    products_block = build_rag_products_block(question)

    # 7) montar mensagens
    messages = [{"role": "system", "content": ALMA_MISSION}]
    fb = facts_block_for_user(user_id)
    if fb:
        messages.append({"role": "system", "content": fb})

    if catalog_block:
        messages.append({"role": "system", "content": catalog_block})

    if catalog_variants_block:
        messages.append({"role": "system", "content": catalog_variants_block})

    if rag_block:
        messages.append({"role": "system", "content": f"Conhecimento corporativo (RAG — não usar para preços):\n{rag_block}"})
    if links_block:
        messages.append({"role": "system", "content": links_block})

    if products_block:
        messages.append({"role": "system", "content": products_block})
    if memory_block:
        messages.append({"role": "system", "content": memory_block})

    messages.append({"role": "user", "content": question})
    return messages, new_facts, rag_used, rag_hits

# =======================================================================================
# FIM HOTFIX
# =======================================================================================

# =============================================================================
# HOTFIX FINAL — Orçamentos com estado persistente (SQLite /data/quotes.db)
# - Guarda linhas do orçamento por user_id+namespace (persistente no Railway)
# - Evita “perder memória” quando o utilizador diz “gera orçamento completo”
# - Em quote_mode: CORTA RAG/links/produtos (para não aparecer “Links do RAG”)
# - Quando o utilizador fornece um SKU (ref), fixa a linha no orçamento (com preço do catálogo)
# =============================================================================
# Colar NO FIM do main.py (depois de app e das funções existentes estarem definidas)

import os, sqlite3, time, re
from typing import Optional, Dict, Any, List, Tuple

QUOTES_DB_PATH = os.getenv("QUOTES_DB_PATH", "/data/quotes.db")

def _quotes_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(QUOTES_DB_PATH), exist_ok=True)
    c = sqlite3.connect(QUOTES_DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def _quotes_init() -> None:
    try:
        with _quotes_conn() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS quotes (
                quote_id     TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                namespace    TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'open',
                created_at   INTEGER NOT NULL,
                updated_at   INTEGER NOT NULL
            );
            """)
            c.execute("""
            CREATE TABLE IF NOT EXISTS quote_items (
                quote_id        TEXT NOT NULL,
                line_id         INTEGER NOT NULL,
                ref             TEXT,
                qty             INTEGER NOT NULL DEFAULT 1,
                unit_price      REAL,
                name_snapshot   TEXT,
                variant_snapshot TEXT,
                url_snapshot    TEXT,
                PRIMARY KEY (quote_id, line_id)
            );
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_quotes_user_ns ON quotes(user_id, namespace, status);")
            c.execute("CREATE INDEX IF NOT EXISTS idx_quote_items_quote ON quote_items(quote_id);")
    except Exception as e:
        try:
            log.warning(f"[quotes] init falhou: {e}")
        except Exception:
            pass

def _now_ts() -> int:
    return int(time.time())

def _quote_get_open(user_id: str, namespace: str) -> str:
    """1 orçamento aberto por (user_id, namespace)."""
    _quotes_init()
    ns = (namespace or DEFAULT_NAMESPACE).strip()
    uid = (user_id or "anon").strip() or "anon"
    with _quotes_conn() as c:
        row = c.execute(
            "SELECT quote_id FROM quotes WHERE user_id=? AND namespace=? AND status='open' ORDER BY updated_at DESC LIMIT 1",
            (uid, ns)
        ).fetchone()
        if row and row["quote_id"]:
            return row["quote_id"]

        qid = f"q_{uid}_{ns}_{_now_ts()}"
        ts = _now_ts()
        c.execute(
            "INSERT INTO quotes(quote_id,user_id,namespace,status,created_at,updated_at) VALUES (?,?,?,?,?,?)",
            (qid, uid, ns, "open", ts, ts)
        )
        return qid

def _quote_touch(quote_id: str) -> None:
    try:
        with _quotes_conn() as c:
            c.execute("UPDATE quotes SET updated_at=? WHERE quote_id=?", (_now_ts(), quote_id))
    except Exception:
        pass

def _quote_next_line_id(quote_id: str) -> int:
    with _quotes_conn() as c:
        row = c.execute("SELECT COALESCE(MAX(line_id),0) AS m FROM quote_items WHERE quote_id=?", (quote_id,)).fetchone()
        return int(row["m"] or 0) + 1

def _catalog_get_by_ref(ns: str, ref: str) -> Optional[Dict[str, Any]]:
    """Lê UMA linha do catálogo por ref exacto."""
    if not ref:
        return None
    try:
        with _catalog_conn() as c:
            r = c.execute("""
                SELECT namespace, name, ref, price, url, brand, variant_attrs, summary
                  FROM catalog_items
                 WHERE namespace=? AND ref=?
                 LIMIT 1
            """, (ns, ref)).fetchone()
            return dict(r) if r else None
    except Exception as e:
        try:
            log.warning(f"[quotes] catalog_get_by_ref falhou: {e}")
        except Exception:
            pass
        return None

def _canon_parent(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    return u.split("#", 1)[0]

def _strip_html_visible(text: str) -> str:
    # simples: remove tags; suficiente para “Variante:” no summary
    if not text:
        return ""
    t = re.sub(r"<[^>]+>", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _quote_add_or_increment_item(quote_id: str, ns: str, ref: str, qty: int = 1) -> Tuple[bool, str]:
    """
    Adiciona ou incrementa por ref.
    Retorna (ok, msg)
    """
    if not ref:
        return (False, "sem ref")
    ns = (ns or DEFAULT_NAMESPACE).strip()
    qty = int(qty or 1)
    if qty < 1:
        qty = 1

    cat = _catalog_get_by_ref(ns, ref)
    name = variant = url = None
    unit_price = None
    if cat:
        name = cat.get("name")
        unit_price = cat.get("price")
        url = cat.get("url")
        variant = (cat.get("variant_attrs") or "").strip()

        # fallback: tentar “Variante:” no summary visível
        if not variant:
            s = _strip_html_visible(cat.get("summary") or "")
            m = re.search(r"(?:^|\b)Variante\s*:\s*(.+)$", s, flags=re.I)
            if m:
                variant = m.group(1).strip()

    try:
        with _quotes_conn() as c:
            existing = c.execute("""
                SELECT line_id, qty FROM quote_items
                 WHERE quote_id=? AND ref=?
                 LIMIT 1
            """, (quote_id, ref)).fetchone()

            if existing:
                new_qty = int(existing["qty"] or 0) + qty
                c.execute("""
                    UPDATE quote_items
                       SET qty=?,
                           unit_price=COALESCE(?, unit_price),
                           name_snapshot=COALESCE(?, name_snapshot),
                           variant_snapshot=COALESCE(?, variant_snapshot),
                           url_snapshot=COALESCE(?, url_snapshot)
                     WHERE quote_id=? AND line_id=?
                """, (new_qty, unit_price, name, variant, url, quote_id, int(existing["line_id"])))
            else:
                line_id = _quote_next_line_id(quote_id)
                c.execute("""
                    INSERT INTO quote_items(quote_id,line_id,ref,qty,unit_price,name_snapshot,variant_snapshot,url_snapshot)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (quote_id, line_id, ref, qty, unit_price, name, variant, url))

        _quote_touch(quote_id)

        if cat:
            return (True, f"item {ref} gravado no orçamento (qty={qty})")
        return (True, f"item {ref} gravado no orçamento (qty={qty}) mas sem match no catálogo")
    except Exception as e:
        try:
            log.warning(f"[quotes] add falhou: {e}")
        except Exception:
            pass
        return (False, f"erro ao gravar item {ref}")

def _quote_list_items(quote_id: str) -> List[Dict[str, Any]]:
    try:
        with _quotes_conn() as c:
            rows = c.execute("""
                SELECT line_id, ref, qty, unit_price, name_snapshot, variant_snapshot, url_snapshot
                  FROM quote_items
                 WHERE quote_id=?
                 ORDER BY line_id ASC
            """, (quote_id,)).fetchall()
            return [dict(r) for r in rows] if rows else []
    except Exception:
        return []

def _quote_context_block(quote_id: str) -> str:
    """
    Bloco PARA O LLM: estado do orçamento.
    Não é para o utilizador — é para evitar “perder memória”.
    """
    items = _quote_list_items(quote_id)
    if not items:
        return "ORÇAMENTO ATIVO (persistente): vazio (sem linhas ainda)."
    lines = ["ORÇAMENTO ATIVO (persistente) — linhas já fixadas (NÃO inventar; usar exatamente):"]
    for it in items:
        ref = it.get("ref") or "(sem ref)"
        qty = it.get("qty") or 1
        name = it.get("name_snapshot") or "(sem nome)"
        var  = (it.get("variant_snapshot") or "").strip()
        pr   = it.get("unit_price")
        pr_s = f"{float(pr):.2f}€" if isinstance(pr, (int, float)) else "(sem preço)"
        title = f"{name} — {var}" if var else name
        lines.append(f"- QTY={qty}; SKU={ref}; PREÇO_UNIT={pr_s}; ITEM={title}")
    lines.append("Regra: se o utilizador disser “gera orçamento completo”, compilar estas linhas num resumo final com subtotais e total.")
    return "\n".join(lines)

def _extract_explicit_skus_and_qty(text: str) -> List[Tuple[int, str]]:
    """
    Extrai (qty, SKU) quando o utilizador dá um ref explícito.
    Exemplos:
      - "SKU: ORK.01.01"
      - "2x ORK.01.01"
      - "quantidade 2 ORK.01.01"
    """
    t = (text or "").strip()
    if not t:
        return []
    out: List[Tuple[int, str]] = []

    # padrão "2x <REF>"
    for m in re.finditer(r"\b(\d+)\s*x\s*([A-Z0-9][A-Z0-9._\-]{2,})\b", t, flags=re.I):
        out.append((int(m.group(1)), m.group(2)))

    # padrão "SKU: <REF>" (qty pode vir antes no texto, mas aqui default=1)
    for m in re.finditer(r"\bSKU\s*:\s*([A-Z0-9][A-Z0-9._\-]{2,})\b", t, flags=re.I):
        out.append((1, m.group(1)))

    # padrão “(<REF>)” muito comum na tua UI
    for m in re.finditer(r"\(([A-Z0-9][A-Z0-9._\-]{2,})\)", t, flags=re.I):
        out.append((1, m.group(1)))

    # dedup simples (mantém o maior qty se repetir)
    best: Dict[str, int] = {}
    for qty, ref in out:
        refu = ref.strip()
        best[refu] = max(best.get(refu, 0), int(qty))
    return [(q, r) for r, q in best.items()]

def _is_quote_mode(question: str) -> bool:
    # usa o teu detector existente se houver
    try:
        return bool(_is_budget_intent_pt(question))
    except Exception:
        q = (question or "").lower()
        return any(k in q for k in ("orçament", "orcament", "cotação", "cotacao", "preço", "preco", "proforma"))

def _is_quote_compile_command(question: str) -> bool:
    q = (question or "").lower()
    return ("orçamento completo" in q) or ("orcamento completo" in q) or ("gera o orçamento completo" in q) or ("gera orçamento completo" in q)

# -----------------------------------------------------------------------------
# PATCH: build_messages — acrescenta estado persistente + corta RAG em quote_mode
# -----------------------------------------------------------------------------
try:
    _build_messages_prev = build_messages  # guarda por segurança
except Exception:
    _build_messages_prev = None

def build_messages(user_id: str, question: str, namespace: Optional[str]):
    # 0) normalizações
    ns = (namespace or DEFAULT_NAMESPACE).strip()
    q  = (question or "").strip()
    uid = (user_id or "anon").strip() or "anon"

    quote_mode = _is_quote_mode(q)

    # 1) inicializa/obtém orçamento e atualiza se houver SKU explícito
    quote_id = _quote_get_open(uid, ns) if quote_mode else None
    if quote_id:
        for qty, ref in _extract_explicit_skus_and_qty(q):
            _quote_add_or_increment_item(quote_id, ns, ref, qty)

    # 2) mem0 facts (mantém como tens)
    new_facts = extract_contextual_facts_pt(q)
    for k, v in new_facts.items():
        mem0_set_fact(uid, k, v)

    # 3) memória curta (mantém)
    short_snippets = _mem0_search(q, user_id=uid, limit=5) or local_search_snippets(uid, limit=5)
    memory_block = (
        "Memórias recentes do utilizador (curto prazo):\n"
        + "\n".join(f"- {s}" for s in (short_snippets or [])[:3])
        if short_snippets else ""
    )

    # 4) catálogo interno SEMPRE
    cat_q = _catalog_query_from_question(q)
    catalog_block = build_catalog_block(cat_q, ns)

    # variantes: em quote_mode também (porque “pedido sem variante definida => listar variantes”)
    # Nota: se o teu build_catalog_variants_block tiver limites, já os aumentaste para não cortar.
    catalog_variants_block = build_catalog_variants_block(cat_q, ns)

    # 5) RAG: em quote_mode = CORTADO (para não aparecer links do RAG em orçamentos)
    rag_block = ""
    rag_used = False
    rag_hits: List[dict] = []
    links_block = ""
    products_block = ""

    if (not quote_mode) and RAG_READY:
        try:
            rag_hits = search_chunks(
                query=q,
                namespace=ns,
                top_k=RAG_TOP_K_DEFAULT
            ) or []
            rag_block = build_context_block(rag_hits, token_budget=RAG_CONTEXT_TOKEN_BUDGET) if rag_hits else ""
            rag_used = bool(rag_block)
        except Exception as e:
            log.warning(f"[rag] search falhou: {e}")
            rag_block, rag_used, rag_hits = "", False, []

        # links candidatos do RAG (apenas fora de quote_mode)
        links_pairs = _links_from_matches(rag_hits, max_links=8)
        if links_pairs:
            lines = ["Links candidatos (do RAG):"]
            for title, url in links_pairs:
                lines.append(f"- {title or '-'} — {url}")
            links_block = "\n".join(lines)

        # bloco de produtos (apenas fora de quote_mode)
        products_block = build_rag_products_block(q)

    # 6) montar mensagens
    messages = [{"role": "system", "content": ALMA_MISSION}]
    fb = facts_block_for_user(uid)
    if fb:
        messages.append({"role": "system", "content": fb})

    # estado persistente do orçamento (antes de tudo, para o LLM não “perder” linhas)
    if quote_id:
        messages.append({"role": "system", "content": _quote_context_block(quote_id)})

        # instrução simples e “elegante”: se orçamento sem variante => listar variantes
        messages.append({"role": "system", "content": (
            "REGRA OPERACIONAL (ORÇAMENTO): "
            "Se o pedido for de orçamento e NÃO houver SKU/variante explícita escolhida, "
            "a tua primeira resposta DEVE ser listar TODAS as variantes do catálogo interno desse produto "
            "e pedir ao utilizador que confirme o SKU."
        )})

        # comando “gera orçamento completo” => deve compilar a partir do estado persistente
        if _is_quote_compile_command(q):
            messages.append({"role": "system", "content": (
                "COMANDO DETETADO: o utilizador pediu para gerar o orçamento completo. "
                "Não procures no RAG. Compila o orçamento final usando APENAS o ORÇAMENTO ATIVO (persistente) acima: "
                "linhas, subtotais e total. Termina com 1 próxima ação (ex.: confirmar morada / portes / proforma)."
            )})

    # catálogo primeiro
    if catalog_block:
        messages.append({"role": "system", "content": catalog_block})
    if catalog_variants_block:
        messages.append({"role": "system", "content": catalog_variants_block})

    # depois RAG (só fora de quote_mode)
    if rag_block:
        messages.append({"role": "system", "content": f"Conhecimento corporativo (RAG):\n{rag_block}"})
    if links_block:
        messages.append({"role": "system", "content": links_block})
    if products_block:
        messages.append({"role": "system", "content": products_block})

    if memory_block:
        messages.append({"role": "system", "content": memory_block})

    messages.append({"role": "user", "content": q})

    # logs curtos úteis
    try:
        if quote_id:
            log.info(f"[quotes] quote_mode={quote_mode} quote_id={quote_id} items={len(_quote_list_items(quote_id))}")
    except Exception:
        pass

    return messages, new_facts, rag_used, rag_hits

try:
    log.info("[hotfix-quotes] build_messages redefinido: estado persistente de orçamento + RAG cortado em quote_mode.")
except Exception:
    pass
# =============================================================================
# FIM HOTFIX FINAL
# =============================================================================

# =============================================================================
# HOTFIX INCREMENTAL — quotes: sticky mode + não gravar "(sem nome)" + pending items
# Colar NO FIM do main.py (depois do hotfix de quotes atual)
# =============================================================================

import time, re
from typing import Optional, Dict, Any, List, Tuple

# --- 1) Extender schema: tabela de pendentes (não-matched) ---
def _quotes_init_v2() -> None:
    _quotes_init()  # chama o init anterior
    try:
        with _quotes_conn() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS quote_pending (
                quote_id     TEXT NOT NULL,
                pending_id   INTEGER NOT NULL,
                raw_text     TEXT,
                qty          INTEGER NOT NULL DEFAULT 1,
                created_at   INTEGER NOT NULL,
                PRIMARY KEY (quote_id, pending_id)
            );
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_quote_pending_quote ON quote_pending(quote_id);")
    except Exception as e:
        try:
            log.warning(f"[quotes-v2] init pendentes falhou: {e}")
        except Exception:
            pass

def _quote_has_open(user_id: str, namespace: str) -> Optional[str]:
    """Devolve quote_id aberto se existir, sem criar novo."""
    _quotes_init_v2()
    ns = (namespace or DEFAULT_NAMESPACE).strip()
    uid = (user_id or "anon").strip() or "anon"
    try:
        with _quotes_conn() as c:
            row = c.execute(
                "SELECT quote_id FROM quotes WHERE user_id=? AND namespace=? AND status='open' ORDER BY updated_at DESC LIMIT 1",
                (uid, ns)
            ).fetchone()
            return (row["quote_id"] if row else None)
    except Exception:
        return None

def _quote_is_recent(quote_id: str, max_age_s: int = 60 * 60 * 6) -> bool:
    """Considera orçamento ‘ativo’ se foi mexido nas últimas X horas (default 6h)."""
    try:
        with _quotes_conn() as c:
            row = c.execute("SELECT updated_at FROM quotes WHERE quote_id=? LIMIT 1", (quote_id,)).fetchone()
            if not row:
                return False
            return (int(time.time()) - int(row["updated_at"] or 0)) <= int(max_age_s)
    except Exception:
        return False

def _quote_add_pending(quote_id: str, raw_text: str, qty: int = 1) -> None:
    """Guarda pedido pendente (sem SKU válido), para o LLM pedir clarificação sem poluir itens."""
    try:
        _quotes_init_v2()
        qt = int(qty or 1)
        if qt < 1: qt = 1
        raw = (raw_text or "").strip()
        if not raw:
            return
        with _quotes_conn() as c:
            row = c.execute(
                "SELECT COALESCE(MAX(pending_id),0) AS m FROM quote_pending WHERE quote_id=?",
                (quote_id,)
            ).fetchone()
            pid = int(row["m"] or 0) + 1
            c.execute("""
                INSERT INTO quote_pending(quote_id,pending_id,raw_text,qty,created_at)
                VALUES (?,?,?,?,?)
            """, (quote_id, pid, raw[:400], qt, int(time.time())))
        _quote_touch(quote_id)
    except Exception:
        pass

def _quote_list_pending(quote_id: str) -> List[Dict[str, Any]]:
    try:
        _quotes_init_v2()
        with _quotes_conn() as c:
            rows = c.execute("""
                SELECT pending_id, raw_text, qty, created_at
                  FROM quote_pending
                 WHERE quote_id=?
                 ORDER BY pending_id ASC
            """, (quote_id,)).fetchall()
            return [dict(r) for r in rows] if rows else []
    except Exception:
        return []

# --- 2) Override: não gravar item se não existir no catálogo ---
_quote_add_or_increment_item_prev = _quote_add_or_increment_item

def _quote_add_or_increment_item(quote_id: str, ns: str, ref: str, qty: int = 1) -> Tuple[bool, str]:
    """
    V2:
    - Só grava em quote_items se houver match no catálogo.
    - Se não houver match, NÃO grava (evita '(sem nome)') e manda para pending.
    """
    ns = (ns or DEFAULT_NAMESPACE).strip()
    ref_clean = (ref or "").strip()
    if not ref_clean:
        return (False, "sem ref")

    cat = _catalog_get_by_ref(ns, ref_clean)
    if not cat:
        _quote_add_pending(quote_id, raw_text=f"SKU/ref não encontrado: {ref_clean}", qty=qty)
        try:
            log.info(f"[quotes-v2] ref sem match no catálogo -> pending (ref={ref_clean})")
        except Exception:
            pass
        return (False, f"SKU/ref {ref_clean} não existe no catálogo (guardado como pendente)")

    # match existe -> usa a função anterior (que grava snapshots e incrementa)
    return _quote_add_or_increment_item_prev(quote_id, ns, ref_clean, qty)

# --- 3) Sticky quote_mode: se há orçamento aberto recente, mantém quote_mode em follow-ups ---
def _is_quote_followup_text(q: str) -> bool:
    """
    Frases típicas de continuação que não dizem 'orçamento/preço' mas são parte do fluxo.
    Mantém isto simples e robusto.
    """
    t = (q or "").strip().lower()
    if not t:
        return False
    triggers = [
        "gera completo", "orçamento completo", "orcamento completo",
        "total", "soma", "subtotal",
        "junta", "adiciona", "acrescenta", "mais", "e também", "e tambem",
        "remove", "tirar", "substitui", "troca",
        "confirma", "ok", "certo", "segue"
    ]
    if any(k in t for k in triggers):
        return True
    # mensagens muito curtas são frequentemente follow-up (“ok”, “sim”, “agora este”)
    if len(t) <= 18:
        return True
    return False

# --- 4) Override build_messages (sem mexer nas tuas rotas): quote_mode sticky + RAG sempre cortado quando sticky ---
_build_messages_quotes_prev = build_messages

def build_messages(user_id: str, question: str, namespace: Optional[str]):
    ns = (namespace or DEFAULT_NAMESPACE).strip()
    q  = (question or "").strip()
    uid = (user_id or "anon").strip() or "anon"

    # detetor original
    quote_mode = _is_quote_mode(q)

    # sticky: se existe orçamento aberto recente e a mensagem parece follow-up, força quote_mode
    open_qid = _quote_has_open(uid, ns)
    if (not quote_mode) and open_qid and _quote_is_recent(open_qid) and _is_quote_followup_text(q):
        quote_mode = True
        try:
            log.info(f"[quotes-v2] sticky quote_mode=True (open_qid={open_qid})")
        except Exception:
            pass

    # Se quote_mode, garante quote_id (não cria se não houver? aqui CRIA, porque entrou em quote_mode)
    quote_id = _quote_get_open(uid, ns) if quote_mode else None

    # Se o utilizador forneceu SKUs explícitos, tenta gravar (só se existir no catálogo; senão pending)
    if quote_id:
        for qty, ref in _extract_explicit_skus_and_qty(q):
            _quote_add_or_increment_item(quote_id, ns, ref, qty)

        # Se NÃO houve SKUs explícitos, mas a frase parece “adiciona 1 <produto>”, guarda como pending (texto cru)
        # (isto evita o LLM inventar e mantém rasto do pedido)
        if not _extract_explicit_skus_and_qty(q) and _is_quote_followup_text(q) and len(q) > 18:
            # guarda a frase como pendente para o LLM pedir o SKU correto
            _quote_add_pending(quote_id, raw_text=q, qty=1)

    # Agora construímos mensagens como antes, mas:
    # - em quote_mode: cortamos RAG/links/produtos SEMPRE (mesmo sticky)
    # - e injetamos bloco do orçamento + pendentes
    messages, new_facts, rag_used, rag_hits = _build_messages_quotes_prev(uid, q, ns)

    if quote_id:
        # Remove qualquer RAG/links/produtos que tenham entrado por algum motivo
        filtered = []
        for m in messages:
            if m.get("role") != "system":
                filtered.append(m); continue
            c = m.get("content") or ""
            if isinstance(c, str) and (
                c.startswith("Conhecimento corporativo (RAG") or
                c.startswith("Links candidatos (do RAG") or
                c.startswith("Produtos para orçamento (do RAG")
            ):
                continue
            filtered.append(m)
        messages = filtered

        # Injeta/atualiza contexto do orçamento no topo (logo após mission/facts)
        # (mais robusto do que depender do bloco anterior)
        pending = _quote_list_pending(quote_id)
        pending_block = ""
        if pending:
            lines = ["ITENS PENDENTES (sem SKU válido / sem match no catálogo) — NÃO incluir em totais:"]
            for p in pending[:20]:
                lines.append(f"- QTY={p.get('qty')}; TEXTO={p.get('raw_text')}")
            lines.append("Regra: pedir ao utilizador o SKU exato do catálogo para cada item pendente.")
            pending_block = "\n".join(lines)

        inject_blocks = []
        inject_blocks.append({"role": "system", "content": _quote_context_block(quote_id)})
        if pending_block:
            inject_blocks.append({"role": "system", "content": pending_block})

        # mete logo após o primeiro system (mission) e (se existir) facts_block
        out = []
        inserted = False
        for i, m in enumerate(messages):
            out.append(m)
            if not inserted and m.get("role") == "system":
                # após os primeiros 1-2 system blocks
                # insere quando já passaram pelo menos 1 system (mission) e possivelmente facts
                if i >= 0:
                    # insere quando o próximo não é system facts? (mantém simples: insere após o 2º system se existir)
                    pass
        # estratégia simples: reconstrói: mission + facts (se houver) + inject + resto (sem duplicar inject)
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        user_msgs = [m for m in messages if m.get("role") != "system"]

        mission = sys_msgs[0:1]
        rest_sys = sys_msgs[1:]
        facts = []
        rest_after_facts = rest_sys
        # heurística: facts_block_for_user costuma ser o 2º system; mantém se for curto e começar por "FACTS" ou "Factos"
        if rest_sys:
            c0 = rest_sys[0].get("content") or ""
            if isinstance(c0, str) and ("FACT" in c0.upper() or "FATO" in c0.upper() or "Fact" in c0):
                facts = rest_sys[0:1]
                rest_after_facts = rest_sys[1:]

        messages = mission + facts + inject_blocks + rest_after_facts + user_msgs

        # Marca RAG como não usado
        rag_used = False
        rag_hits = []

    try:
        if quote_id:
            log.info(f"[quotes-v2] quote_id={quote_id} items={len(_quote_list_items(quote_id))} pending={len(_quote_list_pending(quote_id))}")
    except Exception:
        pass

    return messages, new_facts, rag_used, rag_hits

try:
    log.info("[hotfix-quotes-v2] ativo: sticky quote_mode + pending sem '(sem nome)' + RAG cortado em modo orçamento.")
except Exception:
    pass

# =============================================================================
# FIM HOTFIX INCREMENTAL
# =============================================================================

# =============================================================================
# HOTFIX-ADDON — Pendentes + “não inventar linhas” + quote_mode sticky por orçamento aberto
# Colar NO FIM do main.py (DEPOIS do hotfix-quotes existente)
# =============================================================================

import time, sqlite3, re
from typing import Optional, Dict, Any, List, Tuple

# --- 1) adicionar tabela de pendentes ------------------------------------------------

def _quotes_init_v2() -> None:
    # chama o init original e adiciona pendentes
    try:
        _quotes_init()
    except Exception:
        pass

    try:
        with _quotes_conn() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS quote_pending (
                quote_id     TEXT NOT NULL,
                pend_id      INTEGER NOT NULL,
                qty          INTEGER NOT NULL DEFAULT 1,
                product_name TEXT,
                hint_url     TEXT,
                created_at   INTEGER NOT NULL,
                PRIMARY KEY (quote_id, pend_id)
            );
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_quote_pending_quote ON quote_pending(quote_id);")
    except Exception as e:
        try:
            log.warning(f"[quotes-v2] init pendentes falhou: {e}")
        except Exception:
            pass

def _quote_next_pend_id(quote_id: str) -> int:
    with _quotes_conn() as c:
        row = c.execute("SELECT COALESCE(MAX(pend_id),0) AS m FROM quote_pending WHERE quote_id=?", (quote_id,)).fetchone()
        return int(row["m"] or 0) + 1

def _quote_add_pending(quote_id: str, qty: int, product_name: str, hint_url: str = "") -> None:
    _quotes_init_v2()
    qty = int(qty or 1)
    if qty < 1:
        qty = 1
    pname = (product_name or "").strip()
    if not pname:
        return
    try:
        with _quotes_conn() as c:
            # evita duplicar pendente pelo mesmo nome
            exists = c.execute("""
                SELECT pend_id FROM quote_pending
                 WHERE quote_id=? AND lower(product_name)=lower(?)
                 LIMIT 1
            """, (quote_id, pname)).fetchone()
            if exists:
                # incrementa qty
                c.execute("""
                    UPDATE quote_pending
                       SET qty = qty + ?
                     WHERE quote_id=? AND pend_id=?
                """, (qty, quote_id, int(exists["pend_id"])))
            else:
                pid = _quote_next_pend_id(quote_id)
                c.execute("""
                    INSERT INTO quote_pending(quote_id,pend_id,qty,product_name,hint_url,created_at)
                    VALUES (?,?,?,?,?,?)
                """, (quote_id, pid, qty, pname, (hint_url or "").strip(), _now_ts()))
        _quote_touch(quote_id)
    except Exception as e:
        try:
            log.warning(f"[quotes-v2] add_pending falhou: {e}")
        except Exception:
            pass

def _quote_list_pending(quote_id: str) -> List[Dict[str, Any]]:
    try:
        with _quotes_conn() as c:
            rows = c.execute("""
                SELECT pend_id, qty, product_name, hint_url, created_at
                  FROM quote_pending
                 WHERE quote_id=?
                 ORDER BY pend_id ASC
            """, (quote_id,)).fetchall()
            return [dict(r) for r in rows] if rows else []
    except Exception:
        return []

def _quote_pending_block_for_llm(quote_id: str) -> str:
    pend = _quote_list_pending(quote_id)
    if not pend:
        return ""
    lines = ["PENDENTES (pedido sem SKU — NÃO converter em linhas do orçamento):"]
    for p in pend:
        qty = p.get("qty") or 1
        name = p.get("product_name") or "(pendente)"
        lines.append(f"- QTY={qty}; PRODUTO={name}; AÇÃO: pedir SKU/variante.")
    lines.append("Regra: pendentes NÃO entram no total. Apenas pedir confirmação de SKU para os converter em linhas.")
    return "\n".join(lines)

# --- 2) quote_mode “sticky” se houver orçamento aberto recente -----------------------

def _has_recent_open_quote(uid: str, ns: str, age_s: int = 7200) -> Optional[str]:
    # devolve quote_id se houver orçamento open atualizado recentemente
    try:
        _quotes_init_v2()
        with _quotes_conn() as c:
            row = c.execute("""
                SELECT quote_id, updated_at FROM quotes
                 WHERE user_id=? AND namespace=? AND status='open'
                 ORDER BY updated_at DESC LIMIT 1
            """, (uid, ns)).fetchone()
            if not row:
                return None
            if int(row["updated_at"] or 0) >= (_now_ts() - int(age_s)):
                return row["quote_id"]
    except Exception:
        return None
    return None

def _is_add_to_quote_intent(q: str) -> bool:
    t = (q or "").lower()
    # frases típicas quando já há orçamento em curso
    return any(k in t for k in ("junta", "adiciona", "acrescenta", "mais ", "e mais", "também", "inclui", "mete "))

# --- 3) redefinir _quote_context_block para NÃO mostrar “(sem nome)” -----------------

def _quote_context_block_v2(quote_id: str) -> str:
    items = _quote_list_items(quote_id)
    lines = []
    if items:
        lines.append("ORÇAMENTO ATIVO (confirmado, persistente) — linhas com SKU (usar exatamente; NÃO inventar novas linhas):")
        for it in items:
            ref = it.get("ref") or ""
            qty = it.get("qty") or 1
            name = (it.get("name_snapshot") or "").strip()
            var  = (it.get("variant_snapshot") or "").strip()
            pr   = it.get("unit_price")
            pr_s = f"{float(pr):.2f}€" if isinstance(pr, (int, float)) else ""
            title = " — ".join([x for x in [name, var] if x]) or ref
            lines.append(f"- QTY={qty}; SKU={ref}; PREÇO_UNIT={pr_s or 'DESCONHECIDO'}; ITEM={title}")

    pend_block = _quote_pending_block_for_llm(quote_id)
    if pend_block:
        lines.append(pend_block)

    if not lines:
        return "ORÇAMENTO ATIVO (persistente): vazio (sem linhas confirmadas; sem pendentes)."

    lines.append("Regra máxima: o orçamento completo só usa as linhas confirmadas; pendentes ficam fora do total.")
    return "\n".join(lines)

# --- 4) PATCH no build_messages: cria pendente quando há produto mas não há SKU -------

try:
    _build_messages_prev2 = build_messages
except Exception:
    _build_messages_prev2 = None

def build_messages(user_id: str, question: str, namespace: Optional[str]):
    ns = (namespace or DEFAULT_NAMESPACE).strip()
    q  = (question or "").strip()
    uid = (user_id or "anon").strip() or "anon"

    # quote_mode normal
    quote_mode = False
    try:
        quote_mode = bool(_is_budget_intent_pt(q))
    except Exception:
        quote_mode = False

    # sticky: se há orçamento recente e o texto parece “continuar”
    recent_qid = _has_recent_open_quote(uid, ns)
    if (not quote_mode) and recent_qid and _is_add_to_quote_intent(q):
        quote_mode = True

    quote_id = _quote_get_open(uid, ns) if quote_mode else None
    if quote_id:
        _quotes_init_v2()

        # 1) se houver SKU explícito, grava como confirmado
        extracted = _extract_explicit_skus_and_qty(q)
        for qty, ref in extracted:
            _quote_add_or_increment_item(quote_id, ns, ref, qty)

        # 2) se NÃO houver SKU explícito, mas houver “produto seed” identificado, grava pendente
        #    (isto cobre “faz orçamento de 1 zeal laser” antes de escolher variante)
        if not extracted:
            try:
                seed = _seed_row_for_product(q, ns)
            except Exception:
                seed = None
            if seed:
                pname = (seed.get("name") or "").strip()
                # tenta qty do texto (“1 zeal laser”, “2 zeal laser”)
                mqty = re.search(r"\b(\d+)\b", q)
                qty = int(mqty.group(1)) if mqty else 1
                hint_url = (seed.get("url") or "").strip()
                if pname:
                    _quote_add_pending(quote_id, qty, pname, hint_url)

    # --- daqui para baixo, reaproveita a lógica do hotfix anterior, mas com 2 mudanças:
    #     A) injeta _quote_context_block_v2
    #     B) reforça instrução “não criar linhas sem SKU”
    # -------------------------------------------------------------------------------

    new_facts = extract_contextual_facts_pt(q)
    for k, v in new_facts.items():
        mem0_set_fact(uid, k, v)

    short_snippets = _mem0_search(q, user_id=uid, limit=5) or local_search_snippets(uid, limit=5)
    memory_block = (
        "Memórias recentes do utilizador (curto prazo):\n"
        + "\n".join(f"- {s}" for s in (short_snippets or [])[:3])
        if short_snippets else ""
    )

    cat_q = _catalog_query_from_question(q)
    catalog_block = build_catalog_block(cat_q, ns)
    catalog_variants_block = build_catalog_variants_block(cat_q, ns)

    rag_block = ""
    rag_used = False
    rag_hits: List[dict] = []
    links_block = ""
    products_block = ""

    if (not quote_mode) and RAG_READY:
        try:
            rag_hits = search_chunks(query=q, namespace=ns, top_k=RAG_TOP_K_DEFAULT) or []
            rag_block = build_context_block(rag_hits, token_budget=RAG_CONTEXT_TOKEN_BUDGET) if rag_hits else ""
            rag_used = bool(rag_block)
        except Exception as e:
            log.warning(f"[rag] search falhou: {e}")
            rag_block, rag_used, rag_hits = "", False, []

        links_pairs = _links_from_matches(rag_hits, max_links=8)
        if links_pairs:
            lines = ["Links candidatos (do RAG):"]
            for title, url in links_pairs:
                lines.append(f"- {title or '-'} — {url}")
            links_block = "\n".join(lines)

        products_block = build_rag_products_block(q)

    messages = [{"role": "system", "content": ALMA_MISSION}]
    fb = facts_block_for_user(uid)
    if fb:
        messages.append({"role": "system", "content": fb})

    if quote_id:
        messages.append({"role": "system", "content": _quote_context_block_v2(quote_id)})
        messages.append({"role": "system", "content": (
            "REGRA ORÇAMENTO (CRÍTICA): "
            "Nunca cries linhas no orçamento a partir de nomes de produtos. "
            "Só entram linhas com SKU confirmado (do ORÇAMENTO ATIVO). "
            "Se houver pedidos sem SKU, coloca-os em 'Pendentes' e pede o SKU."
        )})
        if _is_quote_compile_command(q):
            messages.append({"role": "system", "content": (
                "COMANDO: gerar orçamento completo. "
                "Usa APENAS as linhas confirmadas (SKU) do ORÇAMENTO ATIVO. "
                "Pendentes aparecem fora do total, como itens a confirmar com SKU. "
                "Não usar RAG nem links."
            )})

    if catalog_block:
        messages.append({"role": "system", "content": catalog_block})
    if catalog_variants_block:
        messages.append({"role": "system", "content": catalog_variants_block})

    if rag_block:
        messages.append({"role": "system", "content": f"Conhecimento corporativo (RAG):\n{rag_block}"})
    if links_block:
        messages.append({"role": "system", "content": links_block})
    if products_block:
        messages.append({"role": "system", "content": products_block})

    if memory_block:
        messages.append({"role": "system", "content": memory_block})

    messages.append({"role": "user", "content": q})

    try:
        if quote_id:
            log.info(f"[hotfix-quotes-v3] quote_mode={quote_mode} quote_id={quote_id} items={len(_quote_list_items(quote_id))} pending={len(_quote_list_pending(quote_id))}")
    except Exception:
        pass

    return messages, new_facts, rag_used, rag_hits

try:
    log.info("[hotfix-quotes-v3] ativo: pendentes + proibição de linhas sem SKU + quote_mode sticky.")
except Exception:
    pass

# =============================================================================
# FIM HOTFIX-ADDON
# =============================================================================

# =============================================================================
# HOTFIX — Normalização da query do catálogo em modo ORÇAMENTO
# Objetivo:
# - Quando o utilizador diz "faz orçamento de X", "junta o X", etc.,
#   garantir que a query enviada ao catálogo é só o NOME do produto (X),
#   evitando falsos "Sem dados no catálogo interno".
# - NÃO mexe em import CSV, variantes, canonical, urls, nem na lógica de orçamento.
#
# Colar NO FIM do main.py (depois das defs existentes).
# =============================================================================

import re

try:
    _catalog_query_from_question_prev = _catalog_query_from_question
except Exception:
    _catalog_query_from_question_prev = None

_STOP_WORDS_PT = {
    "faz", "fazer", "gera", "gerar", "cria", "criar", "preciso", "quero",
    "orçamento", "orcamento", "cotação", "cotacao", "proforma", "pro-forma",
    "preço", "preco", "valor", "valores",
    "para", "por", "de", "do", "da", "dos", "das", "no", "na", "nos", "nas",
    "um", "uma", "uns", "umas", "o", "a", "os", "as",
    "junta", "juntar", "adiciona", "adicionar", "mete", "meter", "inclui", "incluir",
    "mais", "também", "tb", "se", "sff", "pf", "porfavor", "por-favor", "favor",
    "ao", "à", "aos", "às",
    "unidade", "unidades", "quantidade", "qtd", "x",
}

def _extract_product_terms_pt(text: str) -> str:
    """
    Extrai termos do produto de forma determinística.
    - Se houver um SKU explícito (ex.: ORK.01.03, LINO030, 740021316-2.-AR), devolve isso.
    - Caso contrário: remove intenção de orçamento/verbos/palavras funcionais e quantidades.
    """
    t = (text or "").strip()
    if not t:
        return ""

    # 1) se houver SKU explícito, usa-o (prioritário)
    #    aceita formatos com . _ - e combinações alfanuméricas
    m = re.search(r"\bSKU\s*:\s*([A-Z0-9][A-Z0-9._\-]{2,})\b", t, flags=re.I)
    if m:
        return m.group(1).strip()

    # padrão "2x ORK.01.03" / "2 x ORK.01.03"
    m = re.search(r"\b\d+\s*x\s*([A-Z0-9][A-Z0-9._\-]{2,})\b", t, flags=re.I)
    if m:
        return m.group(1).strip()

    # padrão "(ORK.01.03)" muito comum na UI
    m = re.search(r"\(([A-Z0-9][A-Z0-9._\-]{2,})\)", t, flags=re.I)
    if m:
        return m.group(1).strip()

    # 2) normalização leve
    tt = t.lower()

    # remove pontuação “de comando” mas mantém . _ - dentro de tokens
    tt = re.sub(r"[,:;!?]", " ", tt)
    tt = re.sub(r"\s+", " ", tt).strip()

    # remove quantidades soltas (ex.: "1", "2", "3") quando são tokens isolados
    tokens = [tok for tok in tt.split(" ") if tok and tok not in _STOP_WORDS_PT and not tok.isdigit()]

    # se ficou vazio, devolve o original (fallback)
    cleaned = " ".join(tokens).strip()
    return cleaned if cleaned else t.strip()

def _is_quote_like_intent(question: str) -> bool:
    # usa o detector existente se houver
    try:
        return bool(_is_budget_intent_pt(question))
    except Exception:
        q = (question or "").lower()
        return any(k in q for k in ("orçament", "orcament", "cotação", "cotacao", "proforma", "preço", "preco"))

# 3) monkeypatch: quando é pedido de orçamento, a query do catálogo é “limpa”
def _catalog_query_from_question(question: str) -> str:
    if _catalog_query_from_question_prev is None:
        # fallback, caso não exista a original
        return _extract_product_terms_pt(question)

    q = (question or "").strip()
    if not q:
        return ""

    if _is_quote_like_intent(q):
        return _extract_product_terms_pt(q)

    # comportamento original (fora de orçamento)
    try:
        return _catalog_query_from_question_prev(q)
    except Exception:
        # fallback seguro
        return _extract_product_terms_pt(q)

try:
    log.info("[hotfix-catalog-q] ativo: normalização de query do catálogo em modo orçamento (faz orçamento/junta X -> X).")
except Exception:
    pass

# =============================================================================
# FIM HOTFIX
# =============================================================================


# ---------------------------------------------------------------------------------------
# Local run
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
