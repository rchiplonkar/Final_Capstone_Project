r"""Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System
# **MO-LLM Module 5 -  Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This is the final build step of your capstone. You will turn the retrieval system you've
# developed into an **agent-based RAG system**. Instead of a fixed pipeline, an agent
# decides at each step whether it has enough information, what to retrieve next, which
# tool to use, or whether to answer. This activity allows you to apply the Module 5 labs (Lab 5.1's agentic
# retriever and Lab 5.2's tool-using agent) to your capstone scenario.
#
# The graded deliverable is your completed Capstone Checkpoint 5.1 worksheet. This script is a
# runnable demonstration of an agentic loop on a tiny sample corpus so you can see the
# decision-making before adapting it to your full system.
#
# **Learning outcomes (Module 5):**
# 1. Build an agent-based system that integrates retrieval and external tools.
# 2. Use system prompts to guide agent behavior and decision-making.
# 3. Design workflows that coordinate retrieval, reasoning, and tool use in a RAG system.
# 4. Evaluate an agentic workflow against a fixed retrieval pipeline.
# 5. Design workflows that coordinate retrieval, reasoning, and tool use within a RAG system.
# 6. Build an agent-based system that integrates retrieval and external tools to complete user tasks.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Useful agent tools/actions |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs | semantic search; "find papers by author/year"; "follow citations"; ask a clarifying question |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles | semantic search; "find by category"; "follow links"; ask a clarifying question |
#
# An agent shines when one query isn't enough — it can search, look at what it found,
# and decide to search again (or use a different tool) before answering.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Get a free OpenRouter key at <https://openrouter.ai/keys>. The labs use the paid
#    `openai/gpt-5.4-mini` model, which is covered by the course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). The built-in corpus is deliberately minimal: It only demonstrates
# the agent control flow, not retrieval quality, and it is unrelated to your capstone:
# Howerver, your design should target your own capstone corpus. Your full agent is what you describe
# in the written submission.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    try:
        import pymupdf as fitz  # compatibility alias for newer installs
    except Exception:  # pragma: no cover - optional dependency
        fitz = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

if fitz is None and PdfReader is None:
    warnings.warn(
        "PDF extraction libraries are not installed. Install pymupdf and pypdf before running the capstone corpus loader.",
        RuntimeWarning,
    )

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
TEMPERATURE = 0.2
MAX_STEPS = 4
LOG_PATH = Path(__file__).resolve().parent / "result_5_1_v6.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"
CORPUS_DIR = (
    Path(__file__).resolve().parent
    / "Final_Capstone_Project"
    / "Final_Capstone_Project"
    / "Capstone_CheckPoint_1.1"
    / "ResearchPapers"
)
if not CORPUS_DIR.exists():
    alt_dirs = [
        Path(__file__).resolve().parent / "Final_Capstone_Project" / "Capstone_CheckPoint_1.1" / "ResearchPapers",
        Path(__file__).resolve().parent.parent / "Final_Capstone_Project" / "Final_Capstone_Project" / "Capstone_CheckPoint_1.1" / "ResearchPapers",
        Path(__file__).resolve().parent / "Capstone_CheckPoint_1.1" / "ResearchPapers",
    ]
    for candidate in alt_dirs:
        if candidate.exists():
            CORPUS_DIR = candidate
            break

DECIDE_SYSTEM = (
    "You are an agent retrieving from a research-paper corpus. Given the question, "
    "the queries already run, and the paper titles/snippets found so far, decide what to do next. "
    'Respond with ONLY a JSON object: {"done": true|false, "new_queries": ["..."], '
    '"reasoning": "..."}. Set done=true when you have enough evidence to answer; otherwise give '
    "1-2 new_queries targeting what is still missing (do not repeat past queries)."
)
ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided research-paper snippets, "
    "quoting where you can. If the documents do not contain the answer, say so."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at https://openrouter.ai/keys, "
            "put it in a .env file next to this script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE,
                      api_key=check_api_key(), base_url=OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## A real research-paper corpus + keyword retriever (adapted to the capstone scenario)

# %%

def _read_pdf_text(path: Path) -> str:
    if fitz is not None:
        try:
            with fitz.open(str(path)) as doc:
                parts = []
                for page in doc:
                    try:
                        text = page.get_text("text") or ""
                    except Exception:
                        text = ""
                    parts.append(text)
                return "\n".join(parts)
        except Exception:
            pass

    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages.append(text)
        return "\n".join(pages)
    except Exception:
        return ""


def _normalize_title(title: str) -> str:
    return re.sub(r"[_-]+", " ", title).strip()


def _infer_topics(title: str, text: str) -> list[str]:
    topic_candidates = [
        "program synthesis",
        "sketching",
        "refinement types",
        "verification",
        "trust",
        "code generation",
        "llm",
        "retrieval",
        "graph",
        "evaluation",
        "search",
        "automated reasoning",
        "type inference",
        "synthesizer",
        "reasoning",
        "programming",
        "synthesis",
    ]
    text_norm = (title + " " + text).lower()
    found = [term for term in topic_candidates if term.lower() in text_norm]
    if not found:
        title_terms = re.findall(r"[a-z0-9]{4,}", title.lower())
        found = [term for term in title_terms[:5] if term not in {"conf", "dbpl", "paper"}]
    return found[:6]


def load_real_corpus(directory: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    if not directory.exists():
        return docs

    pdfs = sorted(directory.rglob("*.pdf"))
    for pdf in pdfs:
        text = _read_pdf_text(pdf)
        compact = " ".join(text.split())
        if len(compact) < 150:
            continue
        title = _normalize_title(pdf.stem)
        topics = _infer_topics(title, compact)
        docs.append({
            "id": pdf.stem,
            "title": title,
            "path": str(pdf),
            "text": compact[:20000],
            "topics": topics or ["research", "paper"],
            "links": [],
        })

    if not docs and pdfs:
        log(
            "CORPUS_STATUS",
            f"Detected {len(pdfs)} PDFs in {directory}, but none yielded enough text for indexed retrieval. "
            "This usually means the PDF parser is missing or the PDFs are unreadable."
        )
        return [{
            "id": pdf.stem,
            "title": _normalize_title(pdf.stem),
            "path": str(pdf),
            "text": pdf.stem,
            "topics": ["research", "paper"],
            "links": [],
        } for pdf in pdfs]

    topic_map: defaultdict[str, list[str]] = defaultdict(list)
    for doc in docs:
        for topic in doc["topics"]:
            topic_map[topic].append(doc["id"])

    for doc in docs:
        related = set()
        for topic in doc["topics"]:
            for peer_id in topic_map.get(topic, []):
                if peer_id != doc["id"]:
                    related.add(peer_id)
        if len(related) < 2:
            for peer in docs[:10]:
                if peer["id"] != doc["id"]:
                    related.add(peer["id"])
        doc["links"] = sorted(related)[:5]

    return docs


REAL_DOCS = load_real_corpus(CORPUS_DIR)
DOC_BY_ID = {d["id"]: d for d in REAL_DOCS}
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "with",
    "without", "from", "into", "how", "what", "when", "where", "why", "who",
    "which", "that", "this", "these", "those", "their", "there", "are", "was",
    "were", "been", "being", "to", "of", "in", "on", "at", "by", "as", "it",
    "is", "be", "do", "does", "did", "can", "could", "should", "would", "have",
    "has", "had", "more", "most", "about", "over", "under", "after", "before",
    "through", "using", "used", "work", "works", "together", "between", "within",
    "method", "methods", "papers", "paper", "questions", "question"
}


def _tokens(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if tok not in STOPWORDS and len(tok) > 2]


def keyword_score(query: str, text: str) -> float:
    q = Counter(_tokens(query))
    t = Counter(_tokens(text))
    if not q:
        return 0.0
    score = 0.0
    for term, count in q.items():
        score += count * t.get(term, 0)
    return float(score)


def _best_snippet(doc_text: str, query: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", doc_text).strip()
    if not text:
        return ""
    terms = _tokens(query)
    if not terms:
        return text[:max_chars]
    for term in terms:
        idx = text.lower().find(term)
        if idx != -1:
            start = max(0, idx - max_chars // 2)
            end = min(len(text), start + max_chars)
            return text[start:end].strip()
    return text[:max_chars]


def _doc_search_text(doc: dict[str, Any]) -> str:
    title = doc.get("title", "")
    path = doc.get("path", "")
    text = doc.get("text", "")
    return " ".join([title, path, text])


def baseline_retrieve(query: str, k: int = 4) -> list[tuple[str, float]]:
    if not REAL_DOCS:
        return []
    scored = [(d["id"], keyword_score(query, _doc_search_text(d))) for d in REAL_DOCS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(doc_id, s) for doc_id, s in scored[:k] if s > 0]


def retrieve(query: str, k: int = 4) -> list[str]:
    return [doc_id for doc_id, _ in baseline_retrieve(query, k)]


def decide(llm: ChatOpenAI, question: str, collected: dict[str, str], executed: list[str]) -> dict:
    doc_lines = []
    for doc_id in collected:
        doc = DOC_BY_ID.get(doc_id)
        if not doc:
            continue
        snippet = _best_snippet(doc["text"], question, 600).replace("\n", " ")
        doc_lines.append(f"[{doc_id}] {doc['title']}: {snippet}")
    docs = "\n".join(doc_lines) if doc_lines else "(none yet)"
    user = f"Question: {question}\n\nQueries run: {executed or '(none)'}\n\nDocuments so far:\n{docs}"
    raw = llm.invoke([SystemMessage(content=DECIDE_SYSTEM), HumanMessage(content=user)]).content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        d = json.loads(raw)
        return {
            "done": bool(d.get("done", True)),
            "new_queries": d.get("new_queries", []) or [],
            "reasoning": d.get("reasoning", ""),
        }
    except (json.JSONDecodeError, ValueError):
        return {"done": True, "new_queries": [], "reasoning": "parse-fail -> stop"}


def _coverage(question: str, collected_docs: dict[str, str]) -> float:
    if not collected_docs:
        return 0.0
    q_tokens = set(_tokens(question))
    doc_tokens = set()
    for text in collected_docs.values():
        doc_tokens.update(_tokens(text))
    if not q_tokens:
        return 0.0
    return len(q_tokens & doc_tokens) / len(q_tokens)


def agentic_answer(llm: ChatOpenAI, question: str) -> tuple[str, list[dict[str, Any]]]:
    """Agentic loop over the real corpus with explicit step logging."""
    collected: dict[str, str] = {}
    executed: list[str] = []
    pending = [question]
    step_log: list[dict[str, Any]] = []
    for step in range(MAX_STEPS):
        q_docs: dict[str, list[str]] = {}
        for q in pending:
            hits = baseline_retrieve(q, k=4)
            q_docs[q] = [doc_id for doc_id, _ in hits]
            for doc_id, _ in hits:
                collected[doc_id] = DOC_BY_ID[doc_id]["text"]
            if q not in executed:
                executed.append(q)
        d = decide(llm, question, collected, executed)
        step_log.append({
            "step": step + 1,
            "queries": pending[:],
            "retrieved": q_docs,
            "doc_ids": sorted(collected),
            "done": d["done"],
            "reasoning": d["reasoning"],
            "coverage": round(_coverage(question, collected), 3),
        })
        if d["done"] or not d["new_queries"]:
            break
        pending = [q for q in d["new_queries"] if q and q not in executed][:2]
    context = "\n\n".join(f"[{doc_id}] {DOC_BY_ID[doc_id]['title']}: {_best_snippet(DOC_BY_ID[doc_id]['text'], question, 1200)}" for doc_id in collected)
    answer = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}")
    ]).content
    return answer, step_log


def refined_agent_answer(llm: ChatOpenAI, question: str) -> tuple[str, list[dict[str, Any]]]:
    """Refined version using stopword filtering, focused snippets, and coverage-aware stopping."""
    collected: dict[str, str] = {}
    executed: list[str] = []
    pending = [question]
    step_log: list[dict[str, Any]] = []
    for step in range(MAX_STEPS):
        q_docs: dict[str, list[str]] = {}
        for q in pending:
            hits = baseline_retrieve(q, k=4)
            q_docs[q] = [doc_id for doc_id, _ in hits]
            for doc_id, _ in hits:
                if doc_id not in collected:
                    collected[doc_id] = _best_snippet(DOC_BY_ID[doc_id]["text"], q, 1200)
            if q not in executed:
                executed.append(q)
        coverage = _coverage(question, collected)
        d = decide(llm, question, collected, executed)
        step_log.append({
            "step": step + 1,
            "queries": pending[:],
            "retrieved": q_docs,
            "doc_ids": sorted(collected),
            "done": d["done"] or coverage >= 0.35,
            "reasoning": d["reasoning"],
            "coverage": round(coverage, 3),
        })
        if d["done"] or coverage >= 0.35 or not d["new_queries"]:
            break
        pending = [q for q in d["new_queries"] if q and q not in executed][:2]
    context = "\n\n".join(f"[{doc_id}] {DOC_BY_ID[doc_id]['title']}: {text}" for doc_id, text in collected.items())
    answer = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}")
    ]).content
    return answer, step_log


def answer_from_docs(question: str, doc_ids: list[str]) -> str:
    if not doc_ids:
        return "The baseline retriever did not find enough relevant documents to answer this question."
    context = "\n\n".join(f"[{doc_id}] {DOC_BY_ID[doc_id]['title']}: {_best_snippet(DOC_BY_ID[doc_id]['text'], question, 1200)}" for doc_id in doc_ids if doc_id in DOC_BY_ID)
    llm = make_llm()
    return llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}")
    ]).content


# %% [markdown]
# ## Step 2 — Your agent design (TODO)
#
# Design the agent you will build for **your** scenario. Return a dictionary with the keys
# below — this is the plan you implement in your real system and describe in the report.

# %%
def my_agent_plan() -> dict[str, Any]:
    """Return your agent design for your chosen scenario.

    TODO — your turn. Return a dictionary with these keys:
      - "tools": list[str]      — the actions/tools your agent can take (e.g.,
                                  ["semantic_search", "find_by_author", "follow_citation",
                                   "clarify", "answer"]).
      - "stop_condition": str   — how the agent decides it has enough to answer.
      - "system_prompt_idea": str — one or two sentences on how you'll instruct the agent
                                  to choose actions (this is what "guiding agent behavior
                                  with system prompts" means).
      - "test_tasks": list[str] — 2-3 questions for your scenario that need more than one
                                  retrieval step (so the agentic loop earns its keep).
    """
    return {
        "tools": [
            "semantic_search",
            "find_by_author",
            "find_by_year",
            "follow_citation",
            "clarify",
            "answer",
        ],
        "stop_condition": "the retrieved research-paper snippets jointly cover each sub-question and no missing fact remains before answering.",
        "system_prompt_idea": "Search broadly to identify the most relevant papers, then refine by author/year/topic constraints, ask a clarifying question only when the query is ambiguous, and answer only once the evidence covers the question without guessing.",
        "test_tasks": [
            "Which papers in this corpus explain program synthesis and sketching, and how do they differ in how they fill missing program structure?",
            "What research discusses verification and trust in code generation, and which methods improve reliability when the model is uncertain?",
            "Compare the papers that connect synthesis with automated reasoning and explain how reliability and correctness are handled across the approaches.",
        ],
    }


# %% [markdown]
# ## Step 3 — Run the agentic loop and capture the evidence
#
# Runs the provided agentic loop on a multistep question (watch it retrieve, decide,
# and retrieve again), then prints your plan. Reproduce this in your real system for the
# report and compare it against your fixed Checkpoint 2.1 retriever on the same task.

# %%
def run_single_test(llm: ChatOpenAI, question: str, system_name: str, baseline: bool = False) -> dict[str, Any]:
    start = time.time()
    if baseline:
        hits = baseline_retrieve(question, k=4)
        answer = answer_from_docs(question, [doc_id for doc_id, _ in hits])
        step_log = [{"step": 1, "queries": [question], "retrieved": {question: [doc_id for doc_id, _ in hits]}, "doc_ids": [doc_id for doc_id, _ in hits], "done": True, "reasoning": "single-pass baseline", "coverage": round(_coverage(question, {doc_id: DOC_BY_ID[doc_id]["text"] for doc_id, _ in hits}), 3)}]
        llm_calls = 1
    elif system_name == "refined":
        answer, step_log = refined_agent_answer(llm, question)
        llm_calls = len(step_log) + 1
    else:
        answer, step_log = agentic_answer(llm, question)
        llm_calls = len(step_log) + 1
    elapsed = round(time.time() - start, 2)
    return {
        "system": system_name,
        "question": question,
        "docs": sorted({doc_id for step in step_log for doc_id in step.get("doc_ids", [])}),
        "calls": llm_calls,
        "latency_sec": elapsed,
        "answer": answer,
        "step_log": step_log,
    }


def run() -> None:
    llm = make_llm()
    print(f"Checkpoint 5.1 — agentic RAG demo  |  scenario: {SCENARIO}")
    print(f"Corpus size: {len(REAL_DOCS)} papers")
    questions = [
        "How do program synthesis, sketching, and verification work together to improve reliability in code generation?",
        "Which papers in this corpus explain program synthesis and sketching, and how do they differ in how they fill missing program structure?",
        "What research discusses verification and trust in code generation, and which methods improve reliability when the model is uncertain?",
    ]
    results = []
    for question in questions:
        baseline_result = run_single_test(llm, question, "baseline", baseline=True)
        agent_result = run_single_test(llm, question, "agent")
        refined_result = run_single_test(llm, question, "refined")
        results.append({"question": question, "baseline": baseline_result, "agent": agent_result, "refined": refined_result})

    for item in results:
        print("\n" + "=" * 72)
        print(f"QUESTION: {item['question']}")
        for label in ["baseline", "agent", "refined"]:
            entry = item[label]
            print(f"[{label}] docs={entry['docs'][:8]} calls={entry['calls']} latency={entry['latency_sec']}s")
            print(entry['answer'][:500])

    report = []
    for item in results:
        report.append(f"QUESTION: {item['question']}")
        for label in ["baseline", "agent", "refined"]:
            r = item[label]
            report.append(f"- {label.upper()}: docs={r['docs'][:8]}, calls={r['calls']}, latency={r['latency_sec']}s")
            report.append(f"  answer: {r['answer']}")
            for step in r["step_log"]:
                report.append(f"  step {step['step']}: queries={step['queries']} retrieved={step['retrieved']} docs={step['doc_ids']} coverage={step['coverage']} decision={step['done']} reasoning={step['reasoning']}")
        report.append("")
    log("RESULTS", "\n".join(report))

    plan = my_agent_plan()
    print("\nYour agent plan:")
    print(json.dumps(plan, indent=2))
    log("PLAN", json.dumps(plan, indent=2))
    print("=" * 72)
    print("Done. Real evaluation complete and exported to result_5_1_v6.log")


if __name__ == "__main__":
    run()

# %% [markdown]
# ## Step 4 — Your written submission (the graded deliverable)
#
# The deliverable is a **written submission** (suggested length: 500-750 words). Cover the following:
#
# 1. **System overview**: State your scenario and the retrieval system you built (2.1–4.1).
# 2. **Agent design**: Describe the actions/tools your agent can take, and how a **system prompt**
#    guides it to choose among them (retrieve, use a tool, clarify, or answer).
# 3. **Workflow**: Explain how retrieval, reasoning, and tool use are coordinated across steps,
#    and how the agent decides it has enough information to answer.
# 4. **Evaluation**: Compare the agent-based system against your fixed Checkpoint 2.1/3.1
#    pipeline on a few representative tasks. Did multi-step decision-making improve the
#    answers? At what cost (extra model calls, latency)?
# 5. **Reflection**: When an agentic approach adds value vs. when a simpler pipeline is
#    better, and the limitations/trade-offs of increasingly autonomous systems.
#
# Include evidence (sample tasks, the agent's step-by-step decisions from a log, before/
# after comparison).
