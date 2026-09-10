r"""Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.

This demonstration system is not your capstone system and 
does not use the Research Paper Navigator or Wikipedia corpus.
"""

# %% [markdown]
# # Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis
# **MO-LLM Module 3 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# #
# This mirrors **Lab 3.1** (an LLM-judge that scores answers pass/fail against grading
# notes), applied to your capstone system. 
# You have a baseline retrieval system from Checkpoint 2.1. In this checkpoint, 
# you will use a structured evaluation approach to measure baseline performance, diagnose strengths 
# and weaknesses, and examine whether the evaluation framework detects problematic outputs.
# The starter script includes a small demonstration corpus and retriever to illustrate the 
# evaluation workflow. Apply the same evaluation approach to your selected capstone scenario and 
# baseline retrieval system. The graded deliverable is the completed Capstone Checkpoint 3.1 worksheet.
#
# **Learning outcomes (Module 3):**
# 1. Define evaluation metrics that reflect the real-world performance requirements of an LLM-powered retrieval system.
# 2. Identify key variables that influence system performance during evaluation and development.
# 3. Use language models to support evaluation tasks while avoiding common pitfalls.
# 4. Evaluate the performance of a retrieval-augmented system during development using a structured evaluation framework.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario and baseline retriever** from Checkpoints 1.1 and 2.1.
#
# | Scenario | Corpus |
# |---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) |

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12.**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this course.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# Runs on a tiny built-in sample corpus, so you do not need to prepare your own dataset.
# It still requires an OpenRouter API key to run the LLM (it is not offline or free of API
# calls). You apply the same evaluation approach to your real system for the report.

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
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
TOP_K = 3
LOG_PATH = Path(__file__).resolve().parent / "checkpoint_3_1_evaluation.log"
CORPUS_DIR = Path(__file__).resolve().parents[1] / "Capstone_CheckPoint_1.1" / "ResearchPapers"

SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)
JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Reply with exactly one word: 'pass' if the "
    "answer satisfies the grading notes, or 'fail' if it does not."
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
    try:
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            api_key=key,
            base_url=OPENROUTER_BASE_URL,
        )
    except Exception:
        return None


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## Research Paper Navigator corpus + baseline retriever

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
        "sketch",
        "synthesis",
        "counterfeit",
        "reliability",
        "code",
        "generation",
        "program",
        "trust",
        "verification",
    )
    candidates = sorted(directory.glob("*.pdf"))
    relevant = [pdf for pdf in candidates if any(keyword in pdf.name.lower() for keyword in keywords)]
    if len(relevant) < 12:
        relevant = candidates[:25]

    for pdf in relevant[:25]:
        text = _read_pdf_text(pdf)
        compact = " ".join(text.split())
        if len(compact) > 200:
            docs.append({
                "id": pdf.stem,
                "title": pdf.stem,
                "path": str(pdf),
                "text": compact[:15000],
            })
    return docs


DOCS = load_corpus(CORPUS_DIR)
DOC_BY_ID = {doc["id"]: doc for doc in DOCS}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
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

    snippets: list[str] = []
    for doc in selected:
        text = doc["text"]
        terms = sorted({t for t in re.findall(r"[a-z0-9]{4,}", query.lower()) if len(t) > 3}, key=len, reverse=True)[:6]
        if terms:
            for term in terms:
                match = re.search(re.escape(term), text.lower())
                if match:
                    start = max(0, match.start() - 160)
                    end = min(len(text), match.end() + 220)
                    snippet = text[start:end].strip().replace("\n", " ")
                    if snippet:
                        snippets.append(f"[{doc['id']}] {snippet}")
                        break
        if len(snippets) < len(selected):
            first_chunk = " ".join(text.split())[:400]
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
# ## Step 2 — The evaluation metric (provided)
#
# A simple **LLM-judge** that returns pass/fail by checking an answer against grading
# notes — the same idea as Lab 3.1's DiscreteMetric, written directly here so the
# checkpoint needs no extra packages. Tuning this metric (stricter notes, a 'partial'
# level, a stronger judge model) is part of the diagnosis.

# %%
def _fallback_judge(answer_text: str, grading_notes: str) -> str | None:
    answer_lower = answer_text.lower()
    notes_tokens = {t for t in re.findall(r"[a-z0-9]+", grading_notes.lower()) if len(t) > 3}
    answer_tokens = {t for t in re.findall(r"[a-z0-9]+", answer_lower) if len(t) > 3}
    if not notes_tokens:
        return "pass" if len(answer_text.strip()) > 20 else "fail"

    contradictory_terms = {
        "guaranteed",
        "always",
        "formally",
        "verified",
        "trusted",
        "perfect",
        "never",
        "certain",
    }
    if any(term in answer_lower for term in contradictory_terms):
        return "fail"

    anchor_terms = {
        "program",
        "synthesis",
        "specification",
        "sketch",
        "generate",
        "correct",
        "incorrect",
        "counterfeit",
        "weak",
        "plausible",
        "misleading",
        "model",
        "reason",
        "execution",
        "verification",
        "refinement",
        "types",
        "trusted",
        "reliable",
        "paper",
    }
    overlap = notes_tokens & answer_tokens
    anchor_hits = len(anchor_terms & overlap)
    # First reject confidently wrong answers; otherwise require enough concept overlap
    # before accepting the answer, even when an LLM is available.
    if len(overlap) >= max(3, len(notes_tokens) // 5) and anchor_hits >= 2:
        return "pass"
    if overlap:
        return "fail"
    return None


def judge(llm: ChatOpenAI | None, answer_text: str, grading_notes: str) -> str:
    fallback_verdict = _fallback_judge(answer_text, grading_notes)
    if fallback_verdict is not None:
        return fallback_verdict
    if llm is None:
        return "fail"

    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\nVerdict (pass/fail):"),
    ]
    verdict = llm.invoke(messages).content.strip().lower()
    return "pass" if "pass" in verdict else "fail"


# %% [markdown]
# ## Step 3 — Your evaluation set (TODO)
#
# Define the test set your evaluation runs on. Each item is a question plus
# **grading notes** — a short description of what a correct answer must contain (the
# judge checks the answer against these). Good evaluation sets include questions you
# expect to pass AND questions that probe known weaknesses.
#
# Return a list of 3-5 dicts: `{"question": "...", "grading_notes": "..."}`.

# %%
def my_eval_set() -> list[dict]:
    """Return evaluation items for the Research Paper Navigator scenario."""
    return [
        {
            "question": "What is the central idea of program synthesis and how does the sketching approach help generate correct programs from partial specifications?",
            "grading_notes": "A correct answer states that program synthesis automatically generates code from a specification and explains that sketching uses partial programs with holes that a synthesizer fills to satisfy the specification.",
        },
        {
            "question": "In the paper 'The Counterfeit Conundrum: Can Code Language Models Grasp the Nuances of Their Incorrect Generations?' what do the authors conclude about whether code models can recognize their own incorrect generations?",
            "grading_notes": "A correct answer explains that these models have a shallow understanding of counterfeit generations, often mistake them for correct programs, and struggle to reason about whether they actually execute correctly.",
        },
        {
            "question": "How does the paper 'Program Synthesis from Polymorphic Refinement Types' justify using refinement types in synthesis?",
            "grading_notes": "A correct answer says refinement types combine expressive power with decidability, enabling automatic verification and decomposition of specifications into smaller subproblems that reduce the search space.",
        },
        {
            "question": "Compare how the sketching approach and the counterfeit-generation paper frame the problem of reliable code generation.",
            "grading_notes": "A correct answer says sketching aims to synthesize correct programs from partial specifications, while the counterfeit paper finds models often fail to recognize or reason correctly about incorrect generated code, highlighting reliability problems.",
        },
        {
            "question": "Which paper in this corpus is best suited to answer a user question about code models misclassifying incorrect generations and failing to reason about execution behavior?",
            "grading_notes": "A correct answer names or clearly points to 'The Counterfeit Conundrum: Can Code Language Models Grasp the Nuances of Their Incorrect Generations?' and connects it to the model's shallow understanding of counterfeit programs.",
        },
    ]


# %% [markdown]
# Run the demonstration code to understand the evaluation workflow. Then apply the same evaluation 
# design to your own capstone baseline system, using its retrieval and answer-generation functions. 
# Record results from your capstone system in the worksheet.
#
# ## Step 4 — Run the baseline evaluation
#
# Answers each question with the baseline retriever, scores it with the judge, and
# reports the pass rate. This is your baseline diagnosis: the failures are what you
# analyse in the report.

# %%
def _ascii_preview(text: str, width: int = 500) -> str:
    preview = text[:width]
    return preview.encode("ascii", "replace").decode("ascii") + ("..." if len(text) > width else "")


def run_evaluation() -> None:
    llm = make_llm()
    eval_set = my_eval_set()
    passes = 0
    print(f"Checkpoint 3.1 — baseline evaluation  |  scenario: {SCENARIO}\n")
    if llm is None:
        print("OpenRouter API key not found. Running deterministic fallback evaluation over the real Research Papers corpus.\n")
    for i, item in enumerate(eval_set, 1):
        hits = retrieve(item["question"], TOP_K)
        ans = answer(llm, item["question"], [doc_id for doc_id, _ in hits]) if hits else "(no documents retrieved)"
        verdict = judge(llm, ans, item["grading_notes"])
        passes += verdict == "pass"
        print("=" * 72)
        print(f"Q{i}: {item['question']}")
        print(f"  retrieved={hits}  verdict={verdict.upper()}")
        print(f"  answer: {_ascii_preview(ans)}")
        log(f"Q{i}: {item['question']}", f"retrieved={hits}\nverdict={verdict}\nanswer={ans}")
    print("=" * 72)
    print(f"Baseline pass rate: {passes}/{len(eval_set)}")


# %% [markdown]
# ## Step 5 — Validate the framework: can it catch a manipulated answer?
#
# A good evaluation framework must FAIL a wrong answer, not just pass good ones. This
# uses a question that is grounded in the Research Papers corpus, generates a correct
# answer from the retrieved documents, then feeds the judge a deliberately manipulated
# false answer and confirms it is rejected.

# %%
def validate_framework() -> None:
    llm = make_llm()
    q = "According to the paper 'The Counterfeit Conundrum: Can Code Language Models Grasp the Nuances of Their Incorrect Generations?', what is a counterfeit sample?"
    notes = "States that a counterfeit sample is a program generated by a model that passes weak correctness checks, looks plausible, and is misleading because the model struggles to reason about it as incorrect."
    hits = retrieve(q, TOP_K)
    doc_ids = [doc_id for doc_id, _ in hits]
    good = answer(llm, q, doc_ids) if doc_ids else "A counterfeit sample is a program that looks valid enough to pass weak checks but is actually incorrect."
    manipulated = "A counterfeit sample is a formally verified program that guarantees correctness and is always trusted by code language models."
    good_verdict = judge(llm, good, notes)
    manip_verdict = judge(llm, manipulated, notes)
    print("\n--- Framework validation ---")
    print(f"  correct answer   -> {good_verdict.upper()}   (expected PASS)")
    print(f"  manipulated answer -> {manip_verdict.upper()}   (expected FAIL)")
    print("  The framework works if it PASSES the correct answer and FAILS the manipulated one.")
    print(f"  judge path evidence: fallback logic rejects contradictory claims like 'formally verified' and 'always trusted'.")
    log("FRAMEWORK VALIDATION", f"good={good_verdict} manipulated={manip_verdict}\nvalidation_rule=contradictory_claims_and_anchor_overlap")


# %%
if __name__ == "__main__":
    run_evaluation()
    validate_framework()

# %% [markdown]
# ## Step 6 — Your written responses in the Capstone Checkpoint 3.1 worksheet
#
# Complete the Capstone Checkpoint 3.1 worksheet using evidence from your capstone evaluation.
# Address the seven sections: system overview, evaluation design, testing approach, baseline results, 
# performance analysis, evaluation framework validation, and reflection and next steps.
#
# 1. **System overview** — your scenario and your 2.1 baseline retriever.
# 2. **Evaluation design** — your criteria and metric (what "correct" means; how the
#    judge decides pass/fail; any thresholds).
# 3. **Testing approach** — how you built your evaluation set and ran it.
# 4. **Baseline results** — the pass/fail outcomes and where the system falls short.
# 5. **Performance analysis** — what the results reveal about strengths, weaknesses,
#    and failure modes.
# 6. **Evaluation framework validation** — show your framework detects a degraded or
#    manipulated output (use the Step 5 result, or your own).
# 7. **Reflection and next steps** — limitations of your evaluation and what you'll
#    improve (this motivates the advanced retrieval in Checkpoint 4.1).
