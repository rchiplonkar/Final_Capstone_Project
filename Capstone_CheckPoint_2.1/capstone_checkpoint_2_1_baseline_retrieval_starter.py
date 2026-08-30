r"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation
# **MO-LLM Module 2 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# In Checkpoint 1.1, you showed that a plain LLM can't reliably answer questions about
# your corpus. Now, you will **add retrieval**: Design a retrieval strategy for your scenario
# and build a **baseline retrieval system** that finds the most relevant documents for
# a query, so the model can ground its answers in them.
#
# This mirrors the Module 2 labs — keyword (BM25), vector (semantic), and hybrid
# retrieval — applied to your own capstone corpus. The graded deliverable is the completed 
# Capstone Checkpoint 2.1 worksheet, which includes your written responses and evidence of your 
# retrieval system implementation and testing. This script provides a small working example of 
# baseline retrieval. Use it to understand the retrieval workflow, then adapt the code to implement 
# and test a baseline retriever using your selected capstone dataset.
#
# **Learning outcomes (Module 2):**
# 1. Design a retrieval strategy appropriate for a given dataset and query type.
# 2. Implement and test a baseline retrieval system using structured and/or semantic
#    approaches.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1.
#
# | Scenario | Corpus | Retrieval considerations |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | long documents; you'll likely chunk them; questions often name a specific paper or compare papers. |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | many short-to-medium articles; questions name a figure/place or span several articles. |
#
# A good baseline is keyword (BM25), semantic (embeddings + vector search), or a
# hybrid of both — exactly what you built in Labs 1.2–2.2.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program. This checkpoint uses
#  the `openai/gpt-5.4-mini` model, with usage covered by the course credits. (this uses the paid gpt-5.4-mini chat model — covered by your course credits — and a keyword retriever, no embeddings).
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). Your real baseline (over your full corpus) is what you describe in
# the writeup.

# %%
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
TEMPERATURE = 0.2
TOP_K = 3
LOG_PATH = Path(__file__).resolve().parent / "checkpoint_2_1_retrieval.log"
CORPUS_DIR = Path(__file__).resolve().parents[1] / "Capstone_CheckPoint_1.1" / "ResearchPapers"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"  # "research_papers" or "wikipedia"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)


# %%
def check_api_key() -> str:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return ""
    return key


def make_llm() -> ChatOpenAI | None:
    key = check_api_key()
    if not key:
        return None
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
    )


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %%
def _read_pdf_text(path: Path) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)
    except Exception:
        return ""


def load_corpus(directory: Path) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    if not directory.exists():
        return docs

    keywords = (
        "solar",
        "sketch",
        "synthesis",
        "gu",
        "counterfeit",
        "code",
        "generation",
        "reliability",
    )
    candidates = sorted(directory.glob("*.pdf"))
    relevant = [
        pdf for pdf in candidates
        if any(keyword in pdf.name.lower() for keyword in keywords)
    ]
    if not relevant:
        relevant = candidates[:25]

    for pdf in relevant[:25]:
        text = _read_pdf_text(pdf)
        compact = " ".join(text.split())
        if compact:
            docs.append(
                {
                    "id": pdf.stem,
                    "title": pdf.stem,
                    "path": str(pdf),
                    "text": compact[:15000],
                }
            )
    return docs


DOCS = load_corpus(CORPUS_DIR)
DOC_BY_ID = {doc["id"]: doc for doc in DOCS}


# %% [markdown]
# ## Step 2 — The baseline retriever
#
# A simple keyword-overlap retriever scores each document by how many query words
# it shares, then returns the top-k. This is a minimal baseline for the research-paper
# corpus: relevant papers tend to reuse key terminology such as sketching, synthesis,
# code generation, and reliability.

# %%
def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    """Baseline keyword retrieval over the real Research Papers corpus."""
    if not DOCS:
        return []
    q = _tokens(query)
    if not q:
        return []
    scored = [(d["id"], float(len(q & _tokens(d["text"])))) for d in DOCS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for doc_id, score in scored[:k] if score > 0]


def _fallback_answer(query: str, doc_ids: list[str]) -> str:
    selected = [DOC_BY_ID[i] for i in doc_ids if i in DOC_BY_ID]
    if not selected:
        return "No relevant documents were retrieved from the Research Papers corpus for this question."

    snippets = []
    for doc in selected:
        text = doc["text"]
        for term in sorted({t for t in re.findall(r"[a-z0-9]{4,}", query.lower()) if len(t) > 3}, key=len, reverse=True)[:6]:
            match = re.search(re.escape(term), text.lower())
            if match:
                start = max(0, match.start() - 160)
                end = min(len(text), match.end() + 220)
                snippet = text[start:end].strip().replace("\n", " ")
                if snippet:
                    snippets.append(f"[{doc['id']}] {snippet}")
                    break
        if not snippets or len(snippets) < len(selected):
            first_chunk = " ".join(text.split())[:400]
            if first_chunk not in {s.split("[", 1)[1] if "[" in s else s for s in snippets}:
                snippets.append(f"[{doc['id']}] {first_chunk}")

    if snippets:
        return "Grounded evidence from the retrieved corpus:\n\n" + "\n\n".join(snippets)
    return "The retrieved corpus does not contain a direct answer to this query."


def answer(llm: ChatOpenAI | None, query: str, doc_ids: list[str]) -> str:
    if llm is None:
        return _fallback_answer(query, doc_ids)

    context = "\n\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in doc_ids if i in DOC_BY_ID)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


# %% [markdown]
# ## Step 3 — Representative queries for the Research Paper Navigator scenario
#
# These queries are realistic for the actual capstone corpus: they talk about program
# synthesis, paper comparison, and reliability of generated code—exactly the kinds of
# retrieval questions a Research Paper Navigator must support.

# %%
def my_representative_queries() -> list[str]:
    """Representative queries for the Research Paper Navigator scenario."""
    return [
        "What is the central idea of program synthesis and how does the sketching approach help generate correct programs from partial specifications?",
        "Compare the program-synthesis papers that discuss sketching, specification-driven generation, and the reliability of synthesized code.",
        "Which papers in this corpus discuss whether code models can recognize their own incorrect or unreliable generations?",
        "What does the corpus say about the relationship between program synthesis and trusted, correct code generation?",
        "Which retrieval results best support a user who wants to understand program synthesis, sketching, and code generation reliability together?",
    ]


# %% [markdown]
# ## Step 4 — Run the baseline and capture the evidence
#
# This runs each query through the real corpus and logs the retrieved documents and answer.

# %%
def run() -> None:
    llm = make_llm()
    queries = my_representative_queries()
    print(f"Checkpoint 2.1 — baseline retrieval  |  scenario: {SCENARIO}\n")
    print(f"Corpus directory: {CORPUS_DIR}\n")
    for i, query in enumerate(queries, 1):
        hits = retrieve(query, TOP_K)
        print("=" * 72)
        print(f"QUERY {i}: {query}")
        print(f"  retrieved: {hits}")
        if not hits:
            print("  (nothing matched — note this in your writeup)")
            continue
        ans = answer(llm, query, [doc_id for doc_id, _ in hits])
        print(f"  answer: {ans}\n")
        log(f"QUERY {i}: {query}", f"retrieved={hits}\nanswer={ans}")
    print("=" * 72)
    print("Done. The evidence now reflects the real Research Papers corpus rather than the toy sample corpus.")


# %% [markdown]
# ## Step 5 — Your written submission (the graded deliverable)
#
# Save this solution file in the checkpoint folder and use the log output as evidence.


if __name__ == "__main__":
    run()

