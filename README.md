# Research Papers Assistant

**MIT xPro — RAG and Context Engineering, Capstone Assignment 8.1**
**Author:** Ranjan Chiplonkar

A conversational retrieval-augmented assistant that answers questions about a fixed collection of
153 computer-science research papers (program synthesis, sketching and CEGIS, and LLM-for-code
benchmarks such as CRUXEval and LiveCodeBench). It answers from the papers themselves, cites them by
title and year, verifies its quotations against the source text, and says so plainly when the
collection does not cover a question.

---

## 1. Purpose

Researchers and analysts working with a document collection need answers they can trust and check,
without rereading every paper. The assistant supports the four capabilities set out in the capstone
brief:

| Capability | Example question |
|---|---|
| Retrieve information from an individual paper | "What does the CRUXEval benchmark measure?" |
| Compare ideas across papers | "Compare LiveCodeBench and CRUXEval: how do they differ in what they test?" |
| Identify trends or change over time | "How did program synthesis research change from 2005 to 2020?" |
| Support answers with direct quotations | "Give me a direct quote about how CEGIS uses a verifier." |

It also handles conversation follow-ups ("What year was that paper published?", "Tell me about the
second one"), questions about the collection itself ("How many papers are from before 2010?"), and
date-restricted questions ("Which papers on program sketching were published in 2016?").

Every answer comes with:

- a **Sources** line built from the papers actually retrieved (labelled "retrieved, not used" when the
  answer is a refusal);
- a **Quotation support** count, e.g. `2/2 passages verified`, from checking each quoted span of 20+
  characters against the corpus text.

### How it works (one paragraph)

Each question is sanitized against prompt injection, classified by intent (single / compare /
trend), and answered from metadata when possible (collection counts, a paper's year and venue).
Otherwise it goes through a per-conversation cache, a follow-up resolver, and a router that picks a
single-call path or a planner loop (up to 3 retrieval rounds). Retrieval is BM25 over ~900-character
chunks of every paper, with per-paper and proceedings-volume caps and year filters. The answer model
receives the passages inside an escaped XML trust boundary and cites them by passage id. The program
then checks the answer (trend citations, refusals, quote attribution) and rewrites the citations into
readable titles. Full details are in the architecture document (section 7 of this README).

---

## 2. Repository contents

| File | What it is |
|---|---|
| `Ranjan_Research_Assistant_Final_Capstone_Submission.py` | **The system.** Single Python file (Jupytext `# %%` cells: runs as a script, opens as a notebook) |
| `requirements_8_1_v1.txt` | Pinned dependencies |
| `README.md` | This file |
| `research_assistant_spec.md` | Requirements: goals, features, evaluation plan, trade-offs |
| `research_assistant_architecture.md` | Architecture (as built): components, runtime flows, deployment, decisions, quality evidence, risks |
| `research_assistant_spec_design.md` | Low-level design (as built): every function, constant and data structure, error handling, known deviations, traceability |
| `research_assistant_test_strategy.md` | Test strategy: objectives, risk priorities, scenario catalog, data-quality findings |
| `research_assistant_test_case_v4_run4_graded_v1.xlsx` | 145 graded test cases with actual outputs and verdicts |
| `Test_Research_Assistant.py`, `Test_Research_Assistant_v4_scenarios.py` | External test harness that drives the real app and fills in the workbook |
| `ResearchPapers/` | The 153-PDF corpus (see section 4.3 if it is not included) |

Files the program creates at run time (not needed in the repository): `.env` (your settings —
**never commit it**), `corpus_text_cache.json` (extracted PDF text), `checkpoint_8_1_conversation.log`
(chat transcript) and `checkpoint_7_1_agent.log` (benchmark summaries).

---

## 3. Requirements

- **Python 3.11 or 3.12** (tested with 3.12 on Windows).
- **Tkinter** for the desktop window. It ships with the python.org installers for Windows and macOS;
  on Linux install it separately (e.g. `sudo apt install python3-tk`). Without it, use the terminal
  mode (`chat-cli`).
- **An OpenRouter API key** (or any OpenAI-compatible chat-completions endpoint) and outbound HTTPS
  access to it. There is no offline mode: every answer not served from the cache or from metadata
  calls the model.
- About 1 GB of free memory; the corpus text and index are held in memory.

### Python dependencies (`requirements_8_1_v1.txt`)

| Package | Version | Used for |
|---|---|---|
| `python-dotenv` | 1.2.2 | reading and writing `.env` |
| `langchain-core` | 1.3.2 | chat message types |
| `langchain-openai` | 1.1.10 | `ChatOpenAI` client pointed at OpenRouter |
| `pymupdf` | 1.28.2 | PDF text and title extraction (primary) |
| `pypdf` | 6.19.0 | PDF text extraction (fallback) |
| `openai` | 2.26.0 | transitive, pinned for reproducibility |
| `pydantic` | 2.12.5 | transitive, pinned for reproducibility |

No vector database, embedding model or agent framework is needed: retrieval is a BM25 index written in
plain Python. The test harness additionally needs `openpyxl` (`pip install openpyxl`).

---

## 4. Installation and configuration

### 4.1 Install

```bash
git clone <this repository>
cd <repository folder>

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements_8_1_v1.txt
```

If you skip `pip install`, the script installs any missing package from `requirements_8_1_v1.txt`
automatically on first run (`_ensure_dependencies`).

### 4.2 Configure the two LLM roles

The system uses two models with separate settings:

- the **answer** model (cheaper) writes the final grounded answer;
- the **planner** model (costlier) decides whether another retrieval round is needed in the agent
  loop.

**Option A — in the app.** Start the desktop window (section 5). If no valid configuration exists,
the **Configure LLM API Keys** dialog opens automatically (it can also be opened with the button or
the `K` key). Enter the six values and click **Save**, or click **Upload .env file...** to import an
existing file. The dialog writes `.env` next to the script and masks the key fields.

**Option B — by hand.** Create `.env` next to the script:

```dotenv
# Answer model (cheaper) — required
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_API_KEY=<your OpenRouter key>

# Planner model (costlier) — all three or none; if omitted, the answer settings are reused
PLANNER_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
PLANNER_OPENROUTER_MODEL=openai/gpt-5.4-nano
PLANNER_OPENROUTER_API_KEY=<your OpenRouter key>
```

These are the models the submission was built and tested with. Any model available on the endpoint
works; base URLs must start with `http://` or `https://`. The desktop window requires both groups; the
terminal and test modes accept the answer group alone.

### 4.3 Environment variables

| Variable | Required | Meaning |
|---|---|---|
| `OPENROUTER_BASE_URL`, `OPENROUTER_MODEL`, `OPENROUTER_API_KEY` | yes | answer-model endpoint, model and key |
| `PLANNER_OPENROUTER_BASE_URL`, `PLANNER_OPENROUTER_MODEL`, `PLANNER_OPENROUTER_API_KEY` | all or none | planner-model endpoint, model and key |
| `ASSISTANT_MODE` | no | default run mode when no argument is given (default `chat`) |
| `CHECKPOINT_RESPONSES_LOG` | no | if set, mirrors all console output to this file |
| `BENCH_STRICT` | no | `1` aborts the benchmark when its baseline pass drifts from the historical range |

Values in `.env` take precedence over the same variables already set in the shell.

### 4.4 Place the corpus

The script looks for the PDFs in the first of these folders that exists (paths relative to the
script):

1. `../Capstone_CheckPoint_1.1/ResearchPapers` (the `CORPUS_DIR` constant)
2. `./ResearchPapers` ← **simplest for reviewers**
3. `./Final_Capstone_Project/Final_Capstone_Project/Capstone_CheckPoint_1.1/ResearchPapers`
4. `../Final_Capstone_Project/Capstone_CheckPoint_1.1/ResearchPapers`
5. `../../Final_Capstone_Project/Final_Capstone_Project/Capstone_CheckPoint_1.1/ResearchPapers`

To use another location, edit `CORPUS_DIR` near the top of the script.

**First run:** every PDF is parsed and the text is written to `corpus_text_cache.json`. This takes
noticeably longer than later runs and is not a hang. Later runs re-parse only files whose size or
modification time changed; with a warm cache the corpus loads in a few seconds.

---

## 5. Running

```bash
python Ranjan_Research_Assistant_Final_Capstone_Submission.py [mode]
```

| Mode | What it does | Model calls |
|---|---|---|
| *(none)* or `chat` | Opens the **desktop chat window** (default) | yes |
| `chat-cli` | Chat in the terminal | yes |
| `conv-check` | Offline self-checks (87 checks: intent, routing, year rules, sanitizer, citations, every review fix), a corpus year/duplicate scan, then two live multi-turn conversations. Exit code 1 if any check fails | only the last step |
| `conv-probes` | Prompt-injection probes, including a staged two-turn attack, replayed through the real chat session | yes |
| `benchmark` | Checkpoint 7.1 before/after benchmark (9 questions × 3 passes with tokens, latency and cost) and the 7-probe × 3-condition security suite. Takes several minutes and spends noticeably more tokens than a chat | yes |
| `help`, `-h`, `--help` | Prints the usage message | no |

An unknown mode prints the usage message and exits with code 2.

### 5.1 Using the chat

- Type a question and press **Enter** (or **Alt+S**, or **Send**). **Shift+Enter** inserts a new line.
- `/sources` repeats the last answer's sources; `/help` shows help.
- `/exit` or `/quit` (in the window also `exit`, `quit`, **Alt+X** or **Exit**) closes the session.
- Follow-ups use the last three turns of the conversation, e.g. ask about sketching, then "What year
  was that paper published?".

Example session:

```
> What does the CRUXEval benchmark measure?
...answer with a verbatim quote...
Sources: CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution (2024); ...
Quotation support: 1/1 passages verified.

> How does it compare to LiveCodeBench?
> How many papers in this collection were published in 2016?
> Which papers on program sketching were published in 2016?
> How did Solar-Lezama's synthesis work evolve over time, from the early sketching papers
  around 2005-2008 to the LLM code papers of 2023-2025?
```

### 5.2 Running the test harness

From the folder that contains the script, `.env` and the corpus:

```bash
pip install openpyxl
python Test_Research_Assistant.py                        # all 145 cases
python Test_Research_Assistant.py --only TC-001,TC-002   # selected cases
python Test_Research_Assistant.py --skip-heavy           # skip benchmark / conv-check / CLI subprocesses
python Test_Research_Assistant.py --output results.xlsx  # choose the output workbook
```

By default the harness reads `research_assistant_test_case_v4.xlsx` from the same folder. It drives
the real desktop window, records path, tokens and calls for each turn, and writes the observed
results into the workbook for grading.

---

## 6. Known limitations and considerations for reviewers

**Before you run it**

- **It costs real tokens.** Chat turns cost a few thousand to roughly 20,000 tokens each, depending
  on the path; `benchmark` and the full test harness spend far more. Repeated questions in a
  session, collection-count questions and year/venue questions cost nothing.
- **Network access to the model endpoint is required.** There is no offline mode.
- **Keep `.env` out of version control.** It holds your API keys. Add `.env` to `.gitignore` before
  pushing (the current `.gitignore` lists `corpus_text_cache.json` and `checkpoint_7_1_agent.log`
  but not `.env`).
- **The corpus cache is not committed**, so a fresh clone pays the one-time PDF parsing cost.

**Deliberate data-quality choices**

- Three files are **kept mislabeled on purpose** as negative test cases. Their file names suggest
  Solar-Lezama papers, but they contain unrelated documents (`DBLP_conf_pldi_Solar-LezamaATBSS07`,
  `DBLP_conf_pldi_Solar-LezamaJB08`, `DBLP_journals_sttt_Solar-Lezama13`). The system shows their
  real titles and gives them no year (or the year of the actual document), so they never appear as
  sketching papers in a timeline. See test strategy section 1.
- **Publication years are inferred**: from file-name conventions, checked against the year the
  document states about itself, plus a small override table. 151 of 153 files resolve, and at least
  ten file-name years are overridden by the document's own date. A different corpus would need
  different rules or real metadata.
- **Duplicate files** (15 conference/preprint copies) are collapsed to 138 distinct works, so one
  work is never cited twice.

**Behavior limits** (details and test-case references in architecture section 11 and design
section 19)

- **Trend answers are strict.** If any bullet of a timeline does not cite a retrieved paper, the
  whole answer is replaced by "I could not verify the timeline…", even when good evidence was found
  (TC-020, TC-021). With fewer than three dated works on a topic, the assistant says the collection
  has little on it instead of building a timeline.
- **Intent and routing use fixed word patterns.** Some change-over-time wording ("developed since
  2010", "the progression from … to …") is still treated as a single question, and comparisons
  phrased without words like *compare/differ/versus* are answered more narrowly.
- **Date restrictions are pattern-based.** They are applied only when the question uses publication
  wording ("papers", "published", …) and states a single range.
- **Lexical retrieval.** BM25 plus a few expansion tables (for example *oracle* → *verifier*) will miss
  heavy paraphrases of the papers' own vocabulary.
- **A paper the user names but retrieval does not find** is handled by a prompt instruction plus a
  caution note, not a hard block. The model can occasionally describe it in hedged terms (TC-017).
- **Follow-up comparisons** ("How does it differ from sketching?") can retrieve only one side late
  in a long conversation (TC-067).
- **Session memory** keeps only the last three turns and is lost when the window closes; there is no
  persistence and the system is single-user.
- **Logs** store questions as typed (normalized to ASCII), not the sanitized version.

**About the evaluation evidence**

- The graded workbook is the v4 run 4: **145 cases, 137 pass, 7 fail, 1 blocked**. Security and
  adversarial (16/16), grounding and citation (15/15), individual-paper retrieval (20/20) and data
  quality (13/13) passed in full.
- That run predates the October 11 fixes for TC-133, TC-145 and TC-046 (change-over-time wording,
  date-restricted questions, the year/venue route) and for the model-role defaults. Those fixes are
  verified offline: `conv-check` passes 87/87, and traces through the real pipeline show the right
  evidence reaching the model. The graded figures change only once those cases are rerun against
  the live models.
- Model output is not deterministic. Reruns of the same case can differ in wording, and occasionally
  in verdict.

---

## 7. Further documentation

Read in this order for a full review:

1. **[`research_assistant_spec.md`](research_assistant_spec.md)** — *what and why*: the problem,
   users, business and technical goals, the features added for 8.1, evaluation plan, trade-offs and
   limitations.
2. **[`research_assistant_architecture.md`](research_assistant_architecture.md)** — *how it is
   structured*: system context, solution strategy, component responsibilities, runtime sequences
   (main flow, trend, follow-up, adversarial input), deployment and run modes, cross-cutting
   concepts (security, cost, conversation state, grounding, data quality), architecture decisions
   with alternatives, quality evidence from the graded run, and risks.
3. **[`research_assistant_spec_design.md`](research_assistant_spec_design.md)** — *how it is
   built*: a file map by line range, configuration, every function and its algorithm (year
   resolution, BM25 scoring, sanitizer steps, prompt composition order, `ChatSession.ask` step by
   step), error handling, known deviations with fix status (D-1 to D-10), and traceability from
   requirement to function.
4. **[`research_assistant_test_strategy.md`](research_assistant_test_strategy.md)** — *how it is
   tested*: objectives, scope, risk-based priorities, the scenario catalog across all test
   categories, and the corpus data-quality findings that shaped the tests.
5. **`research_assistant_test_case_v4_run4_graded_v1.xlsx`** — *the evidence*: every test case with
   expected result, actual output, verdict, reasoning, and its requirement and rubric mapping.
