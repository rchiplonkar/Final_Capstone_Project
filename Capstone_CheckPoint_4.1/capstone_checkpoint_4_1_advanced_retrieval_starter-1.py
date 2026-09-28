r"""Capstone Checkpoint 4.1 — Advanced Retrieval Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 4.1 — Advanced Retrieval Implementation
# **MO-LLM Module 4/Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This checkpoint advances your capstone system from a *single-pass* retriever to an
# **advanced retriever**. You will use the baseline retrieval system you built in
# Checkpoint 2.1 and evaluated in Checkpoint 3.1 to identify where single-pass retrieval
# falls short and apply the Module 4 techniques to fix it:
#
# - **Lab 4.1 — multistep retrieval (query decomposition):** Rewrite a complex
#   question into focused sub-queries, retrieve for each, and merge the results.
# - **Lab 4.2 — graph-based retrieval:** Organize your corpus as a graph of
#   relationships (citations, shared topics, links, same author/entity) and pull in
#   *related* documents a keyword/vector search would miss.
#
# You then **measure the improvement** against your 3.1 baseline. The graded
# deliverable is a **500-750 word written report** (final section). This script is a
# runnable demonstration of both techniques on a tiny sample corpus so you can see
# them work before adapting them to your real system.
#
# **Learning outcomes (Module 4):**
# 1. Identify common retrieval failures — e.g., complex multistep questions and too
#    much loosely-related context degrading the answer.
# 2. Explain *why* those failures occur with single-pass retrieval.
# 3. Implement an advanced retrieval strategy (query decomposition and/or graph
#    traversal) that addresses them.
# 4. Measure the improvement against your baseline using your 3.1 evaluation set.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Natural relationships to exploit in a graph |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | citations, shared authors, shared topics/keywords, "published-before" |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | hyperlinks between articles, shared categories, mentioned entities |
#
# Multistep decomposition shines on cross-document questions ("Compare X and Y,"
# "how did an idea evolve"); graph retrieval shines when the answer needs documents
# that are *related* to a hit but don't themselves match the query terms.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv networkx`
# 3. Use the OpenRouter API key provided for this course. This checkpoint uses
#   the `openai/gpt-5.4-mini` model, with usage covered by course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This script runs on a tiny built-in sample corpus, so you do **not** need your full
# dataset indexed to complete it. You *will* refer to your real 2.1 system and 3.1
# evaluation results when you write the report.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # Latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
LOG_PATH = Path(r"C:\Ranjan\Personal\MIT_Xpro\RAG_Context_Engineering\Lab_Excercises\.venv\Resubmission_09_23.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"
ROOT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = (
    ROOT_DIR
    / "Final_Capstone_Project"
    / "Final_Capstone_Project"
    / "Capstone_CheckPoint_1.1"
    / "ResearchPapers"
)
if not CORPUS_DIR.exists():
    alt_dirs = [
        ROOT_DIR / "Final_Capstone_Project" / "Final_Capstone_Project" / "Capstone_CheckPoint_1.1" / "ResearchPapers",
        ROOT_DIR / "Final_Capstone_Project" / "Capstone_CheckPoint_1.1" / "ResearchPapers",
        ROOT_DIR / "Final_Capstone_Project" / "Final_Capstone_Project" / "Capstone_CheckPoint_4.1" / "Final_Capstone_Project" / "Capstone_CheckPoint_1.1" / "ResearchPapers",
        ROOT_DIR / "Final_Capstone_Project" / "Capstone_CheckPoint_1.1" / "ResearchPapers",
        Path.cwd() / "Final_Capstone_Project" / "Final_Capstone_Project" / "Capstone_CheckPoint_1.1" / "ResearchPapers",
    ]
    for candidate in alt_dirs:
        if candidate.exists():
            CORPUS_DIR = candidate
            break

DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a document retrieval system. "
    "Break the user's question into 2-4 focused sub-queries that together cover "
    "everything needed to answer it. Each sub-query should target a distinct aspect. "
    'Return ONLY a JSON array of strings, e.g., ["sub-query 1", "sub-query 2"].'
)

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using only the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)

JUDGE_SYSTEM = (
    "You are a strict grader for a retrieval-augmented QA system. You will be given "
    "a QUESTION, a GRADING NOTE describing what a correct answer must contain, and "
    "an ANSWER produced by the system from retrieved documents. "
    "Reply with exactly one word: PASS if the answer satisfies the grading note "
    "(even if phrased differently), or FAIL if it does not, is evasive, or is "
    "unsupported by the documents. Do not explain your reasoning."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at "
            "https://openrouter.ai/keys, put it in a .env file next to this "
            "script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def log_response(label: str, prompt: str, response: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    entry = (
        f"[{ts}]  {label}  SCENARIO={SCENARIO}  MODEL={LLM_MODEL}\n"
        f"PROMPT:   {prompt}\n"
        f"RESPONSE: {response}\n"
        f"{'-' * 72}\n"
    )
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except OSError:
        fallback = Path.cwd() / "Resubmission_09_23.log"
        with fallback.open("a", encoding="utf-8") as fh:
            fh.write(entry)


# %% [markdown]
# ## A tiny sample corpus (stands in for your real one)
#
# Six short "documents" with relationships so both techniques are demonstrable
# without indexing your full corpus. Each has an ID, text, a `topics` list, and
# `links` to related documents (i.e., think citations for papers, or hyperlinks for
# Wikipedia). Your real system would derive these from your actual data.

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
        "counterfeit",
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
    found = []
    text_norm = (title + " " + text).lower()
    for term in topic_candidates:
        if term.lower() in text_norm:
            found.append(term)
    if not found:
        title_terms = re.findall(r"[a-z0-9]{4,}", title.lower())
        found = [term for term in title_terms[:5] if term not in {"conf", "dbpl", "paper"}]
    return found[:6]


def load_real_corpus(directory: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    if not directory.exists():
        return docs

    pdfs = sorted({pdf.resolve() for pdf in directory.rglob("*.pdf")})
    for pdf in pdfs:
        text = _read_pdf_text(pdf)
        compact = " ".join(text.split())
        if len(compact) < 200:
            continue
        title = _normalize_title(pdf.stem)
        topics = _infer_topics(title, compact)
        doc = {
            "id": pdf.stem,
            "title": title,
            "path": str(pdf),
            "text": compact[:20000],
            "topics": topics or ["research", "paper"],
            "links": [],
        }
        docs.append(doc)

    if not docs:
        for pdf in pdfs[:10]:
            docs.append({
                "id": pdf.stem,
                "title": _normalize_title(pdf.stem),
                "path": str(pdf),
                "text": (pdf.stem + " " + pdf.name).lower(),
                "topics": ["research", "paper"],
                "links": [],
            })

    topic_map: defaultdict[str, list[str]] = defaultdict(list)
    for doc in docs:
        for topic in doc["topics"]:
            topic_map[topic].append(doc["id"])

    for doc in docs:
        current_id = doc["id"]
        related = set()
        for topic in doc["topics"]:
            for peer_id in topic_map.get(topic, []):
                if peer_id != current_id:
                    related.add(peer_id)
        if len(related) < 2:
            for peer in docs[:20]:
                if peer["id"] != current_id:
                    related.add(peer["id"])
        doc["links"] = sorted(related)[:10]

    return docs


REAL_DOCS = load_real_corpus(CORPUS_DIR)
DOC_BY_ID = {d["id"]: d for d in REAL_DOCS}


def keyword_score(query: str, text: str) -> float:
    """A tiny, dependency-free relevance score: shared word count. This stands in for
    your real hybrid retriever, so the retrieval step needs no extra dependencies. The
    demo still calls the LLM, which requires an OpenRouter API key."""
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    return float(len(q & t))


def baseline_retrieve(query: str, k: int = 3) -> list[tuple[str, float]]:
    """Single-pass retrieval: Score every doc once, take the top k. (id, score)."""
    if not REAL_DOCS:
        return []
    scored = [(d["id"], keyword_score(query, d["text"])) for d in REAL_DOCS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(i, s) for i, s in scored[:k] if s > 0]


# %% [markdown]
# ## Step 2 — Multistep retrieval (provided, adapted from Lab 4.1)
#
# `decompose_query` asks the LLM to split the question into sub-queries. For each
# sub-query, we retrieve the top documents and **sum** each document's score across
# the sub-queries. A document relevant to several parts of the question rises to the
# top. (In your real system, replace `baseline_retrieve` with your 2.1 retriever.)

# %%
def decompose_query(llm: ChatOpenAI, query: str) -> list[str]:
    messages = [SystemMessage(content=DECOMPOSE_SYSTEM), HumanMessage(content=query)]
    raw = llm.invoke(messages).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("["): raw.rfind("]") + 1] if "[" in raw else raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return [query]


def multistep_retrieve(llm: ChatOpenAI, query: str, k: int = 3) -> list[str]:
    sub_queries = decompose_query(llm, query)
    print(f"  decomposed into: {sub_queries}")
    score_map: dict[str, float] = {}
    for sq in sub_queries:
        for doc_id, score in baseline_retrieve(sq, 3 * k):
            score_map[doc_id] = score_map.get(doc_id, 0.0) + score
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:k]]


# %% [markdown]
# ## Step 3 — Graph-based retrieval (provided, adapted from Lab 4.2)
#
# Build a graph whose nodes are documents and topics, with edges for "links to" and
# "relates to topic." Retrieval seeds with the baseline hits, then pulls in each
# seed's linked documents and topic-siblings as extra context — deduplicated.

# %%
def build_graph(docs: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for d in docs:
        G.add_node(f"doc:{d['id']}", node_type="doc")
        for other in d["links"]:
            G.add_edge(f"doc:{d['id']}", f"doc:{other}", edge_type="links_to")
        for topic in d["topics"]:
            G.add_node(f"topic:{topic}", node_type="topic")
            G.add_edge(f"doc:{d['id']}", f"topic:{topic}", edge_type="relates_to")
    return G


def graph_retrieve(
    graph: nx.DiGraph, query: str, k: int = 3, max_k: int | None = None
) -> dict[str, str]:
    """Return {doc_id: source_label} while allowing graph neighbors to compete within the
    same retrieval cap as the baseline."""
    max_k = max_k if max_k is not None else k
    seed_limit = max(1, max_k - 1)
    seed_docs = baseline_retrieve(query, k=seed_limit)

    union: dict[str, tuple[str, float]] = {}
    for doc_id, score in seed_docs:
        union[doc_id] = ("seed", float(score))

    for seed in list(union):
        node = f"doc:{seed}"
        if not graph.has_node(node):
            continue
        for _, target, ed in graph.out_edges(node, data=True):
            target_id = target.removeprefix("doc:")
            if ed.get("edge_type") == "links_to":
                if target_id not in union:
                    union[target_id] = ("linked", 0.0)
            elif ed.get("edge_type") == "relates_to":
                for sib, _, ed2 in graph.in_edges(target, data=True):
                    if ed2.get("edge_type") == "relates_to":
                        sib_id = sib.removeprefix("doc:")
                        if sib_id not in union:
                            union[sib_id] = ("topic", 0.0)

    priority = {"seed": 3, "linked": 2, "topic": 1}
    ranked = sorted(
        union.items(),
        key=lambda item: (priority.get(item[1][0], 0), float(item[1][1]), item[0]),
        reverse=True,
    )
    ranked = ranked[:max_k]
    return {doc_id: source for doc_id, (source, _) in ranked}


# %% [markdown]
# ## Step 4 — Your advanced-retrieval plan (TODO)
#
# Design how you will apply these techniques to **your** capstone corpus. Return a
# dictionary with the keys below. This is the plan you will implement in your real system
# and describe in the report. Keep it concrete and specific to your scenario.

# %%
def my_advanced_plan() -> dict[str, Any]:
    """Return YOUR advanced-retrieval plan for your chosen scenario."""
    return {
        "technique": "both",
        "node_types": ["paper", "topic"],
        "edge_types": ["links_to", "relates_to"],
        "test_queries": [
            "Which papers in this corpus explain program synthesis and how sketching fills missing parts of a program?",
            "What research discusses whether code models can recognize their own incorrect generations and why reliability is still a problem?",
            "Compare papers that blend synthesis with verification, trust, and correctness to see how the field makes generated code more reliable.",
        ],
        "rationale": "The Research Paper Navigator corpus contains long technical PDFs with strong topical overlap, so query decomposition helps break multi-aspect questions apart while graph expansion (via shared topics and document links) brings in adjacent papers that share concepts without matching the original wording exactly.",
    }


# %% [markdown]
# ## Step 5 — Run baseline vs. advanced and capture the evidence
#
# This demonstration compares single-pass retrieval with the two advanced retrieval strategies 
# using the sample corpus and a representative query. It also logs the result. Read the
# retrieved sets: The advanced strategies should surface related documents that the
# baseline misses. Then run the same comparison in your real system for your report.

# %%
def answer_from_docs(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in doc_ids if i in DOC_BY_ID)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


EVAL_QUERIES = [
    {
        "id": "Q1",
        "query": "What is the central idea of program synthesis and how does the sketching approach help generate correct programs from partial specifications?",
    },
    {
        "id": "Q2",
        "query": "In the paper 'The Counterfeit Conundrum: Can Code Language Models Grasp the Nuances of Their Incorrect Generations?' what do the authors conclude about whether code models can recognize their own incorrect generations?",
    },
    {
        "id": "Q3",
        "query": "How does the paper 'Program Synthesis from Polymorphic Refinement Types' justify using refinement types in synthesis?",
    },
    {
        "id": "Q4",
        "query": "Compare how the sketching approach and the counterfeit-generation paper frame the problem of reliable code generation.",
    },
    {
        "id": "Q5",
        "query": "Which paper in this corpus is best suited to answer a user question about code models misclassifying incorrect generations and failing to reason about execution behavior?",
    },
]

GRADING_NOTES = {
    "Q1": "Must explain program synthesis as automatically generating a program that meets a spec, and explain sketching as letting the programmer provide a partial program/template with holes that synthesis fills in.",
    "Q2": "Must state the paper's actual conclusion: that code language models largely struggle to reliably recognize or explain their own incorrect generations (the paper finds this is a real limitation, not a solved capability).",
    "Q3": "Must explain that the paper uses polymorphic refinement types to narrow the search space during synthesis by encoding precise input/output type constraints the synthesized program must satisfy.",
    "Q4": "Must compare both papers: sketching frames reliability as satisfying a programmer-provided partial specification, while the counterfeit-generation paper frames reliability as a model's ability to judge its own incorrect output; the answer should name both framings, not just one.",
    "Q5": "Must identify 'The Counterfeit Conundrum' paper (DBLP_conf_acl_GuLJOLSS24) as the best-suited source and explain why it directly studies models misclassifying their own incorrect generations.",
}


def judge_answer(llm: ChatOpenAI, query: str, grading_note: str, answer: str) -> str:
    prompt = (
        f"QUESTION: {query}\n\nGRADING NOTE: {grading_note}\n\nANSWER:\n{answer}"
    )
    messages = [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)]
    raw = llm.invoke(messages).content.strip().upper()
    return "pass" if raw.startswith("PASS") else "fail"


def run_comparative_evaluation(llm: ChatOpenAI, graph: nx.DiGraph) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    print("\n" + "=" * 72)
    print("STEP 5b — Comparative evaluation on the Checkpoint 3.1 test set")
    print("(comparable retrieval limit: k=3 documents for every system)")
    print("=" * 72)

    for item in EVAL_QUERIES:
        qid, query = item["id"], item["query"]
        note = GRADING_NOTES[qid]

        base_ids = [i for i, _ in baseline_retrieve(query, k=3)]
        base_answer = answer_from_docs(llm, query, base_ids)
        base_verdict = judge_answer(llm, query, note, base_answer)
        log_response(f"EVAL_{qid}_BASELINE", query, f"retrieved={base_ids} verdict={base_verdict}\n{base_answer}")

        decomp_ids = multistep_retrieve(llm, query, k=3)
        decomp_answer = answer_from_docs(llm, query, decomp_ids)
        decomp_verdict = judge_answer(llm, query, note, decomp_answer)
        log_response(f"EVAL_{qid}_DECOMPOSITION", query, f"retrieved={decomp_ids} verdict={decomp_verdict}\n{decomp_answer}")

        graph_map = graph_retrieve(graph, query, k=3, max_k=3)
        graph_ids = list(graph_map.keys())
        graph_answer = answer_from_docs(llm, query, graph_ids)
        graph_verdict = judge_answer(llm, query, note, graph_answer)
        log_response(f"EVAL_{qid}_GRAPH", query, f"retrieved={graph_map} verdict={graph_verdict}\n{graph_answer}")

        row = {
            "id": qid,
            "query": query,
            "baseline_ids": base_ids,
            "baseline_verdict": base_verdict,
            "decomp_ids": decomp_ids,
            "decomp_verdict": decomp_verdict,
            "graph_ids": graph_ids,
            "graph_verdict": graph_verdict,
        }
        rows.append(row)

        print(f"\n{qid}: {query}")
        print(f"  BASELINE      (k=3) retrieved={base_ids}      verdict={base_verdict}")
        print(f"  DECOMPOSITION (k=3) retrieved={decomp_ids}    verdict={decomp_verdict}")
        print(f"  GRAPH         (k=3) retrieved={graph_ids}     verdict={graph_verdict}")

    def pass_rate(field: str) -> str:
        n_pass = sum(1 for r in rows if r[field] == "pass")
        return f"{n_pass}/{len(rows)}"

    summary = (
        "\nBefore/after summary (paste into the worksheet's Evaluation Results table):\n"
        f"  Baseline pass rate:      {pass_rate('baseline_verdict')}\n"
        f"  Decomposition pass rate: {pass_rate('decomp_verdict')}\n"
        f"  Graph (k=3) pass rate:   {pass_rate('graph_verdict')}\n"
    )
    print(summary)
    log_response("EVAL_SUMMARY", "run_comparative_evaluation", summary)
    return rows


def run_demo() -> None:
    llm = make_llm()
    docs = load_real_corpus(CORPUS_DIR)
    graph = build_graph(docs)
    query = "How do program synthesis, sketching, and verification work together to improve reliability in code generation?"

    print("=" * 72)
    print(f"Checkpoint 4.1 demo — scenario: {SCENARIO}\nCorpus documents loaded: {len(docs)}\nQuery: {query}\n")

    base = [i for i, _ in baseline_retrieve(query, 3)]
    print(f"BASELINE (single-pass) retrieved: {base}")

    multi = multistep_retrieve(llm, query, 3)
    print(f"MULTI-STEP retrieved:             {multi}")

    g = graph_retrieve(graph, query, 5)
    print(f"GRAPH retrieved (with sources):   {g}")

    answer = answer_from_docs(llm, query, list(g.keys())[:5])
    print(f"\nGraph-augmented answer:\n{answer}\n")
    log_response("GRAPH", query, answer)

    plan = my_advanced_plan()
    print("Your advanced-retrieval plan:")
    print(json.dumps(plan, indent=2))
    log_response("PLAN", "my_advanced_plan", json.dumps(plan, indent=2))

    print("=" * 72)
    print("Done. The real Research Papers corpus is now active and the demonstration uses the full capstone corpus instead of the toy sample.")
    run_comparative_evaluation(llm, graph)


if __name__ == "__main__":
    run_demo()

# %% [markdown]
# ## Step 6
#
# The checkpoint deliverable is your completed Capstone Checkpoint 4.1 worksheet, not
# code. Using evidence from your real system, cover:
#
# 1. **System overview** State your scenario and your 2.1 baseline retriever.
# 2. **Failure diagnosis** Describe the meaningful retrieval failure or limitation identified through
#    your Checkpoint 3.1 evaluation and explain why it occurs.
# 3. **Data redesign** Describe the node/edge schema you defined to expose relationships in
#    your corpus (citations/authors/topics, or links/categories/entities), and why you chose it. 
# 4. **Advanced retrieval implemented** Provide a query decomposition and/or graph traversal,
#    with code or log evidence. Label primary vs. context documents in the prompt.
# 5. **Comparative evaluation** Re-run your 3.1 evaluation on baseline vs advanced,
#    **comparing a similar number of retrieved documents**, and report before/after.
# 6. **Analysis & reflection** Discuss where it helped, where it hurt (redundant/over-broad
#    context), the cost/latency trade-off, and remaining limitations.
#
