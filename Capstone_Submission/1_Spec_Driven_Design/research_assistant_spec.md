# Research Papers Assistant — Specification

**Author:** Ranjan Chiplonkar
**Course:** MIT xPro, RAG and Context Engineering — Capstone, Module 8.1
**Status:** Planning document for the final capstone build, written against the working
Checkpoint 7.1 system (`capstone_checkpoint_7_1_production_ready_starter.py`)

## 1. What this system is for

I picked the Research Papers Assistant scenario back in Checkpoint 1.1 and have carried it
through every module since. The corpus is a fixed set of ~150 PDFs pulled from a program-synthesis
research area (Solar-Lezama's sketching line of work, CEGIS-based synthesis, LLM code-generation
benchmarks like CRUXEval and LiveCodeBench, and related papers). The assistant's job is to let
someone who hasn't read all 150 papers ask natural questions and get answers grounded in the
actual text, with the sources shown.

The capstone brief for 8.1 describes the target this way: a conversational interface that answers
questions about a collection of research papers, retrieves from individual papers, compares ideas
across papers, surfaces trends or changes over time, and backs its answers with direct quotations.
That's the bar for this document. It's also a reasonable stand-in for the kind of system a research
group, a policy team, or an internal knowledge-management function would actually want — nobody on
those teams has time to re-read 150 PDFs every time a question comes up, and nobody trusts an answer
they can't trace back to a source.

**Who uses it:** a researcher or analyst who already knows the general subject area and wants to
query a fixed document set quickly, cite real passages, and spot how ideas changed across a
publication timeline, without re-reading the whole corpus by hand.

**What it is not:** a general web-search assistant, a tool that ingests arbitrary new PDFs at
request time, or a system that answers from the model's own training knowledge when the corpus is
silent. If the papers don't cover something, the honest answer is "the corpus doesn't say," not a
guess dressed up as a citation.

## 2. Business goals

- Cut the time it takes to get a sourced answer about this paper collection from "read several
  PDFs" to "ask a question."
- Make every answer checkable — a user should be able to find the exact sentence an answer is
  built on, in the actual paper, not just trust the model.
- Keep the system cheap enough to run many queries against without worrying about API spend, and
  fast enough that a back-and-forth conversation doesn't feel like waiting on a batch job.
- Hold up against the kind of messy, occasionally adversarial input a real deployment gets —
  copy-pasted email threads, "ignore your instructions" attempts, malformed questions — without
  either getting hijacked or becoming unusable from over-filtering.

## 3. Technical goals

- Answer three kinds of questions correctly and with grounded quotes: single-paper lookups,
  cross-paper comparisons, and questions about how an idea or a line of research changed over the
  years covered by the corpus.
- Keep the system conversational — a user should be able to ask a follow-up ("what about the 2013
  one?") without repeating the full context of the earlier question.
- Reuse the production hardening already built and verified in Checkpoint 7.1 rather than
  rebuilding it: the sanitizer, the XML trust boundary, the cache, the router, the dual-model cost
  split, and the action log all carry forward unchanged.
- Close the specific gaps between a benchmark-and-probe harness (what 7.1 is) and an actual
  interface a reviewer can sit down and type questions into (what 8.1 needs).

## 4. What the system already does (carried over from Checkpoint 7.1, unchanged)

This is the foundation, and none of it gets rewritten for 8.1 — it gets wrapped in a conversational
front end and extended where the capstone goal calls for something the benchmark harness doesn't
need.

**Corpus loading.** `load_real_corpus()` walks the PDF folder, extracts text with PyMuPDF (falling
back to pypdf), and caches extracted text by filename, size, and mtime in
`corpus_text_cache.json`, so re-runs only pay the PDF-parsing cost for files that changed. On the
full corpus this is around 150 papers and roughly 18 million characters of text.

**Retrieval.** `build_chunk_index()` / `retrieve_chunks()` split every paper into ~900-character
chunks on a 750-character step (so chunks overlap), and rank them with a BM25-style score —
inverse document frequency times term frequency, length-normalized. Bibliography-heavy chunks
(matched by citation markers like "pp.", "et al.", "arXiv") are down-weighted, and the first couple
of chunks of each paper (title and abstract) are up-weighted. Retrieval caps how many chunks come
from any single paper (`per_doc`), which is what makes cross-paper questions actually pull from
more than one source instead of one paper dominating the results.

**Routing.** `route()` sends a question down one of two paths. Simple, single-topic questions go to
`answer_direct()` — one retrieval call, one model call. Anything that looks like it needs more than
one retrieval — multi-part questions, or questions matching `_MULTI_HOP_RE` (compare, contrast,
differ, versus, trend, "over time", evolved, changed, "which papers") — goes to `agentic_answer()`,
which runs a planner loop (capped at `MAX_STEPS = 3`) that can issue follow-up retrieval queries
before writing the final answer.

**Cost control.** The planner uses the small `gpt-5.4-nano` model, the answer step uses
`gpt-5.4-mini`. `SimpleCache` serves repeat questions at zero token cost, including near-duplicate
phrasings (Jaccard similarity over tokens), with a guard that refuses a fuzzy match when the two
questions differ in a number or a direction word ("before" vs. "after", "with" vs. "without") —
otherwise a semantic cache would confidently serve the wrong answer to a context-sensitive
question.

**Security.** `_sanitize_user_text()` normalizes the input (NFKC, strips zero-width characters),
caps it at 2000 characters, escapes anything that could forge an XML tag, deletes whole pasted
"email block" style injections, and neutralizes roleplay/override trigger phrases while leaving the
rest of the sentence intact as inert text. `HARDENED_XML_SYSTEM` draws the actual trust boundary:
retrieved paper text sits inside `<documents>`, the user's question sits inside `<user_question>`,
and the model is told plainly that text inside `<documents>` is quoted material, never instructions
— even if a page contains something that reads like a command. This has been tested against a
7-probe injection suite across three conditions (no defenses, XML structure only, full hardening)
and holds.

**Observability.** Every retrieval and planner decision gets written to `ACTION_LOG`. A separate
corpus scan flags PDF passages that contain instruction-like phrasing, purely for monitoring — it
never deletes anything, since a paper can legitimately contain the word "ignore" in a normal
sentence.

**Evaluation harness.** A 9-question benchmark (`BENCH`) mixes single-paper, comparison, and
over-time questions, checked against expected terms and expected source papers, plus a
`_quote_support()` check that verifies any quoted span of 20+ characters in the answer actually
appears in the source text — a crude but real groundedness metric. The harness runs it three ways:
before (6.1-style, always the full agent loop, whole-paper text, no cache), cold cache, and warm
cache, and prints the token/latency/cost delta between them.

None of this needs to change. It needs a conversational shell around it and a few targeted
additions, below.

## 5. What's missing against the 8.1 capstone goal

The 7.1 code is a benchmark-and-probe harness — `run()` drives a fixed demo and exits. The capstone
goal asks for a conversational interface, and a few specific capabilities the harness doesn't
currently need:

1. **No actual interface.** There's no loop a person can sit at and type questions into. Everything
   is a function call from inside `run()`.
2. **No conversation memory.** `answer_routed()` takes one question and has no notion of what was
   asked before it. A real conversation needs follow-ups to resolve against prior turns.
3. **Trend/over-time questions work by accident, not by design.** `_MULTI_HOP_RE` routes a question
   like "how did this evolve from 2005 to 2023" to the agent loop, and the loop pulls chunks from
   several papers and leaves it to the model to notice the dates inside the text. There's no
   extracted publication-year metadata and nothing that explicitly groups or orders sources by
   year. It works on the benchmark because the paper titles and the surrounding text mention years
   directly, but it isn't a real feature — it's the model getting lucky with what's in front of it.
4. **Citations aren't reader-facing.** Internally, a retrieved chunk is labeled with something like
   `pldi_Solar-LezamaRBE05#2` — a filename stem plus a chunk index. That's exactly what you want for
   the groundedness check in the eval harness. It is not what you want to show a person reading an
   answer; they need a paper title (and ideally a year), not an internal key.
5. **No README, no repo structure.** The 8.1 rubric and submission instructions require a GitHub
   repository with the working implementation and a README covering purpose, setup, dependencies,
   and known limitations. That packaging doesn't exist yet.

## 6. Features to build for 8.1

### 6.1 Conversational loop

A simple command-line chat loop (`chat()` or similar) that wraps `answer_routed()`: read a line,
run it through the existing sanitize → cache → route → answer path, print the answer with its
citations, append the turn to a session history, repeat. This is new code, additive — it calls the
existing functions, it doesn't replace them. A minimal web front end (Flask or Gradio, single page)
is a reasonable upgrade if there's time before the recording, but the CLI loop alone satisfies
"conversational interface" and is the safer thing to finish first.

### 6.2 Session memory for follow-ups

Keep the last few turns (question + a short summary of the answer, not the full retrieved text) in
a session list. When a new question comes in, that short history gets passed to the planner/answer
step as additional context — clearly labeled as prior conversation, not as source material, so it
doesn't get mixed into the `<documents>` trust boundary. This is what lets someone ask "what about
the 2013 paper?" right after a question about sketching, without re-stating which paper they mean.
Scope this carefully: the history is a few hundred tokens of summary, not the full transcript, so it
doesn't quietly inflate cost on every turn.

### 6.3 Publication-year extraction and trend grouping

The corpus filenames already carry a DBLP-style convention — venue prefix, author string, two-digit
year suffix (`aplas_Solar-Lezama09`, `cav_ChaudhuriS12`, `iclr_JainHGLYZWSSS25`). A small regex
against the trailing digits of the filename stem, with a fallback to scanning the first chunk of
each paper for a 4-digit year near the top of the text, gives every document a `year` field at load
time. That's a cheap, structural addition to `load_real_corpus()` — a few lines, no change to how
retrieval or chunking works.

Once documents carry a year, two things get easier and more honest: a question that matches
`_MULTI_HOP_RE`'s trend wording can have its retrieved sources sorted chronologically before they're
handed to the model, and the answer prompt can be told to organize a trend answer by year rather
than by whatever order retrieval happened to return. This turns "the model noticed dates in the
text" into an actual feature with a predictable mechanism behind it.

### 6.4 Reader-facing citations

Keep the internal `doc_id#chunk_n` key for everything that depends on it today (the groundedness
check, the action log, the benchmark). Add a presentation-layer mapping from `doc_id` to the paper's
normalized title (already computed by `_normalize_title()`) and year, and have the chat loop
rewrite citations in the printed answer — `[pldi_Solar-LezamaRBE05#2]` becomes something like
`(Solar-Lezama et al., Combinatorial Sketching for Finite Programs, 2005)`. This is a formatting
pass on the response before it's shown to the user; it doesn't touch retrieval, the prompt, or the
trust boundary.

### 6.5 Structured comparison answers

For questions that route to the agent path because they match comparison wording, extend
`HARDENED_XML_SYSTEM` (or add a comparison-specific variant used only on that path) with an
instruction to structure the answer around the papers being compared — what each paper says, then
where they agree or differ — rather than a single blended paragraph. The retrieval and routing
mechanics that make this possible already exist (`per_doc` capping guarantees multiple papers show
up in the context); this is a prompt refinement, not a new retrieval mechanism.

### 6.6 Direct-quotation support, made visible

`_quote_support()` already measures whether quoted spans in an answer are real. For the
conversational interface, surface that as a visible signal rather than an internal eval number —
for instance, printing a short grounding note under each answer ("2 of 2 quoted passages verified
against the source text"). This reuses an existing function; it just exposes its output to the user
instead of only to the benchmark harness.

## 7. What does not change, and why

Nothing in the security layer, the retrieval math, the cache, or the router gets removed. Every
piece of it addresses a real, previously-identified risk (the Checkpoint 6.1 audit) or a real,
previously-measured cost problem (the Checkpoint 7.1 before/after benchmark), and the capstone
goal doesn't eliminate those risks — a conversational interface that takes arbitrary typed input
from a user is, if anything, more exposed to injection attempts than a fixed benchmark question
list, not less. If anything ends up trimmed during implementation — for example, some of the
console `print()` narration in `run()` that exists to make the benchmark readable step-by-step but
would just be noise in an interactive chat session — it gets replaced with a comment explaining
that the underlying function call stays, only the verbose narration around it is dropped for the
interactive path.

## 8. Evaluation plan

Keep the existing `BENCH` set and `_quote_support()` grounding check as the automated regression
test — it already covers single-paper, comparison, and over-time questions, and it's cheap to rerun
after each change. Add to it:

- A handful of multi-turn exchanges (a question followed by a pronoun-reference follow-up) to check
  that session memory actually resolves "it" and "the 2013 one" correctly, rather than silently
  falling back to retrieving nothing relevant.
- A spot check on the year-extraction regex against the full file list, since a silent parsing
  failure there would make a trend answer confidently wrong instead of obviously wrong.
- A before/after comparison of citation readability — one transcript showing raw internal keys, one
  showing the resolved title/year form — as a concrete example for the presentation.

The security probe suite (7 probes × 3 conditions: baseline, XML-only, full hardening) stays as-is
and gets rerun once the conversational loop is in place, since the loop is the first time arbitrary
user-typed input reaches the system outside of a fixed test list.

## 9. Trade-offs

- **Session memory vs. cost.** Carrying conversation history into every call adds tokens to every
  turn, including ones that don't need it. Keeping the history short (summaries, not full answers)
  is a deliberate cap on that cost, at the price of occasionally losing detail from two or three
  turns back.
- **Chronological sorting vs. relevance ranking.** Once a trend question sorts its sources by year,
  that ordering can push a highly relevant but recent chunk below an older, less relevant one in
  how the answer reads. The BM25 ranking still decides what gets retrieved; year-sorting only
  changes the order sources are presented in, to keep that distinction intentional rather than
  accidental.
- **A CLI loop vs. a web UI.** A web front end demos better on a recording, but it's more surface
  area to build, test, and secure in the time available before the submission deadline. The CLI
  loop is the one that's guaranteed to be done and correct; a web wrapper is a stretch goal on top
  of it, not a replacement for it.
- **Regex year extraction vs. a real metadata source.** The filename convention this corpus happens
  to follow won't generalize to an arbitrary PDF collection. That's an acceptable trade for this
  fixed, known corpus, and it's called out explicitly below as a limitation rather than presented as
  a general solution.

## 10. Known limitations and future work

- Year extraction depends on the filename convention of this specific corpus (DBLP-style keys) and
  a text-scan fallback; a differently-named PDF collection would need a different extraction rule
  or real metadata (a BibTeX file, a CSV manifest) instead of filename parsing.
- Session memory is scoped to a single run of the chat loop — there's no persistence across
  restarts. A future version could log sessions to disk the same way `ACTION_LOG` already tracks
  retrieval, and reload a session by ID.
- The cache's fuzzy match is still a token-overlap heuristic, not real embedding similarity; it
  catches close paraphrases but will miss a reworded question that shares few surface tokens with
  the cached one.
- Comparison answers are only as good as what the router sends down the agent path. A comparison
  phrased in a way that doesn't match `_MULTI_HOP_RE` (no explicit "compare"/"versus"/"differ"
  wording) would still go down the single-retrieval direct path and likely under-serve the second
  paper. Widening that detection, or asking the planner itself to decide single- vs. multi-document
  intent instead of a regex, is a reasonable next iteration.
- Citation resolution covers title and year; it doesn't currently surface page numbers, since the
  PDF text extraction flattens pages into one block per paper. Keeping page boundaries during
  extraction would be needed to cite "page 4" specifically rather than just the paper.

## 11. Submission mapping

The 8.1 rubric grades a recorded presentation plus a GitHub repository, scored complete only if
both the presentation and the repo cover every required element.

**Repository contents:** the full `capstone_checkpoint_7_1_production_ready_starter.py` logic
carried forward unchanged, the new conversational-loop/session-memory/year-extraction/citation code
layered on top, `corpus_text_cache.json`, this spec, and a README covering: what the system is for,
how to install and run it (Python version, `pip install` line, `.env` setup for the OpenRouter key),
and the limitations listed in section 10.

**Presentation:** the five-question structure the course asks for maps directly onto this document
— the problem is section 1, the architecture is sections 4 and 6, the evaluation evidence is section
8, the design rationale is sections 4 through 6 plus the Checkpoint 6.1/7.1 audit history behind
them, and the trade-offs and limitations are sections 9 and 10.
