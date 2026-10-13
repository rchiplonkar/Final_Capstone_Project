# Research Papers Assistant — Test Strategy

**Author:** Ranjan Chiplonkar
**Course:** MIT xPro, RAG and Context Engineering — Capstone, Module 8.1
**Inputs:** `research_assistant_spec.md` (requirements), `research_assistant_architecture.md`
(architecture), `research_assistant_spec_design.md` (low-level design), the 8.1 assignment rubric,
and a direct analysis of the `ResearchPapers` corpus and `corpus_text_cache.json` rather than
assumptions about what's in them.
**Status:** Strategy level. This sets objectives, approach, risk priorities, and the scenario
catalog a tester works from. Exact inputs, expected outputs, and pass/fail criteria for each
scenario belong in the detailed test case document this feeds into, not here.

## 1. Why this document exists, and a finding that shaped it

Before writing a single scenario I ran the actual corpus through a short analysis pass rather than
assuming its 153 PDFs are uniformly well-formed. That turned up something worth stating up front,
because it changes what "the system works correctly" has to mean for this project specifically.

Three files whose names imply they are Armando Solar-Lezama's own papers do not contain the
content their filenames claim:

- `DBLP_conf_pldi_Solar-LezamaATBSS07.pdf` extracts to a paper titled "BLK-Assist: A Methodological
  Framework for Artist-Led Co-Creation with Generative AI Models" — an unrelated diffusion-model
  paper with no connection to program synthesis.
- `DBLP_conf_pldi_Solar-LezamaJB08.pdf` extracts to "Enhance Sketch Recognition's Explainability
  via Semantic Component-Level Parsing" — a hand-drawing recognition paper that happens to also use
  the word "sketch," which makes this a genuinely dangerous near-miss rather than an obvious
  mismatch a keyword check would catch.
- `DBLP_journals_sttt_Solar-Lezama13.pdf` extracts to the NIST FIPS 197 Advanced Encryption
  Standard specification. Nothing about program synthesis at all.

The other Solar-Lezama-authored files in the corpus checked out fine — `aplas_Solar-Lezama09`
("The Sketching Approach to Program Synthesis"), `pldi_Solar-LezamaRBE05` ("Programming by
Sketching for Bit-Streaming Programs"), and the rest of the benchmark's expected sources all
contain exactly what their names say. So this isn't a systemic extraction failure; it's three
specific files that appear to have been saved under the wrong name during corpus assembly. One of
them (`JB08`) is listed as an expected retrieval target in the existing `BENCH` question about
CEGIS and the program-hole-filling mechanism — meaning part of the existing evaluation harness has
been silently grading against a document that was never going to contain the content it expects.

I'm not fixing the corpus here — that's a data decision for you to make (correct the files, drop
them, or deliberately keep them as an adversarial data-quality case; section 8.6 below treats it
as exactly that). What it does mean for this document is that "data quality" has to be a first-class
test category, not an assumed precondition, because the corpus has already demonstrated it can
contain confidently-wrong content that looks completely legitimate to the retrieval layer.

## 2. Test objectives

Directly from the 8.1 rubric's single pass/fail criterion — a complete submission demonstrates the
system's architecture, behavior, retrieval approach, workflow, and evaluation evidence, and
justifies its engineering decisions and trade-offs — testing on this project exists to produce that
evidence, not just to catch bugs. Concretely:

1. Show that each of the three capstone question types — single-paper lookup, cross-paper
   comparison, trend/change-over-time — retrieves from the right sources and answers correctly,
   with numbers to back the claim, not just a few examples that happened to work.
2. Show that every quoted passage in an answer is traceable to real corpus text, and that a
   citation shown to the user actually names the paper the answer drew from.
3. Show the security posture holds under the same adversarial conditions already proven in
   Checkpoint 7.1, now that arbitrary typed input reaches the system through an interactive loop
   instead of a fixed probe list.
4. Show the cost and latency properties (cache hits, routed vs. agent-path cost) still hold once
   conversational history is added to every turn.
5. Catch the specific new failure modes the low-level design introduced — year extraction guessing
   wrong, citation resolution failing silently, session memory resolving a follow-up to the wrong
   paper — before they show up during the recorded presentation.
6. Produce a result set a reviewer can read as the "how do you know it works" section of the
   presentation, and that stands up to being challenged live.

## 3. What is explicitly not being tested

Worth stating as plainly as the objectives, since a test strategy that doesn't bound itself expands
forever:

- No testing of the OpenRouter API's own reliability, correctness, or pricing — it's an external
  dependency, used as-is.
- No load or concurrency testing. The architecture is explicitly single-user, single-process; there
  is nothing to load-test.
- No testing of PDF ingestion for a corpus other than the one on hand. Year extraction, chunking
  behavior, and the data-quality findings in section 8.6 are all specific to this corpus's actual
  content, not a general-purpose PDF pipeline.
- No UI/visual testing, since the primary interface is a command-line loop. If the optional web
  front end gets built, it gets its own lightweight pass, noted in section 8.8, not a full UI test
  suite.

## 4. Test levels and where each one runs

| Level | What it covers | How it runs |
|---|---|---|
| Unit | Pure functions with no model call and no I/O: `classify_intent`, `display_label`, `sorted_by_year`, `rewrite_inline_citations`, `_extract_year` | Plain assertions, runs in under a second, no API cost |
| Component | A single piece against real data but without going through the full conversational path: the retriever against the real chunk index, the cache's guard logic, the sanitizer against known attack strings | Mostly reuses what Checkpoint 6.1/7.1 already built (the sanitizer false-positive check, the cache-correctness demo); a few additions for the new pieces |
| System / end-to-end | A full question through `ChatSession.ask`, including a real model call, checked against expected terms, expected sources, and quote-support | The existing `BENCH` harness, extended with multi-turn scenarios (`CONV_BENCH` from the design doc) |
| Security | The 7-probe injection suite across baseline / XML-only / fully hardened conditions, re-run through the conversational entry point | Already built in Checkpoint 7.1; re-run, not re-designed, per section 8.4 |
| Acceptance | Does the finished repository satisfy the rubric: does a clean checkout run per the README, does the presentation cover all required elements | Manual checklist, section 10 |

Unit and component tests run on every change, cost nothing, and take seconds. System-level and
security tests cost real API tokens and take minutes, so they run before a submission or a
presentation recording, not on every small edit — the same cadence Checkpoint 7.1's benchmark
already used.

## 5. Risk-based prioritization

Ranked by how much damage a failure does to the rubric's "complete" criteria, not by how likely the
failure is:

**Highest** — anything that would make an answer wrong while looking confidently right: a citation
naming the wrong paper, a quote presented as grounded that isn't actually in the source, a trend
answer built from a mislabeled document like the three found in section 1. These fail silently and
are the hardest to catch by eyeballing a demo.

**High** — a security regression introduced by the conversational loop (history injection, a
sanitizer bypass reachable only through multi-turn phrasing) and a cost regression (history adding
meaningfully more tokens per turn than the design budgeted).

**Medium** — routing a comparison or trend question down the wrong path because its wording doesn't
match the existing heuristics; year extraction producing a wrong-but-plausible year that doesn't
get caught because it's not as obviously wrong as `natosec_40`.

**Lower** — CLI ergonomics (a confusing error message, an unrecognized command), since these affect
usability, not correctness, and are easy to catch with light manual use.

## 6. Test environment and data strategy

There is exactly one test environment: the same Python process, the same `ResearchPapers` corpus,
the same OpenRouter models used in production, because this system has no separate "staging"
dataset — the corpus *is* the data, fixed and known. That has one direct consequence for strategy:
fixture data for component and system tests isn't synthetic, it's specific real papers already
characterized against the corpus.

Corpus facts worth recording here because several scenarios below depend on them directly, pulled
from running the analysis described in section 1 against the actual 153-entry
`corpus_text_cache.json` rather than estimating:

- 153 PDFs load successfully; total extracted text is 18,398,803 characters; nothing falls under
  the 150-character failure threshold `load_real_corpus` already checks for. That means the
  existing "unreadable PDF" failure path has no real example in this corpus to exercise it against
  — section 8.6 covers how to test it anyway, with an injected negative case.
- Three files (named in section 1) contain content unrelated to their filenames. One is already an
  expected-source document in the existing `BENCH` suite.
- Fifteen pairs of files are exact byte-identical duplicates after text extraction — almost all of
  them a conference version and an arXiv/journal version of the same paper (for example
  `DBLP_conf_oopsla_PuNSB16.pdf` and `DBLP_journals_corr_PuNSB16.pdf`). Benign, but relevant to
  retrieval-diversity testing: a comparison question could retrieve the same content under two
  different document ids and count it as two sources.
- A handful of PDFs are extreme outliers in size relative to extracted text —
  `DBLP_conf_iclr_LiWSZ0BM25.pdf` is 47.8 MB but yields only 62,372 characters of text (1.3
  characters per kilobyte, against a corpus-wide norm closer to 50–80), meaning the bulk of that
  file is images or figures PyMuPDF can't turn into text. Several other papers
  (`BraggFRS21`, `KongSG18`, `GaoKDRSAK19`, `snapl_2015`, `InalaPQLS17`) run past a million
  characters each after extraction and will dominate the chunk index unless the existing per-document
  retrieval cap (already in `retrieve_chunks`) is actually doing its job — a good thing to verify
  directly rather than assume.

## 7. Entry and exit criteria

**Entry** — the baseline file (`Research_Assistant_Final_Capstone_Submission.py`) runs the existing
`benchmark` mode end to end without errors, reproducing the Checkpoint 7.1 numbers, before any new
test scenario for the 8.1 additions is considered meaningful. There's no point testing new
conversational code against a retrieval layer that isn't already known-good.

**Exit**, per rubric element:

- Every scenario in section 8.1 through 8.3 (single-paper, comparison, trend) passes against its
  expected terms and expected sources, with the three mislabeled documents from section 1 either
  excluded from expected-source sets or deliberately tested as a negative case (section 8.6), not
  silently left in as an unexamined false assumption.
- The security probe suite (section 8.4) shows the same pass pattern through the conversational
  entry point that it already showed in Checkpoint 7.1 directly.
- The cost/latency delta from adding conversation history (section 8.5) is measured and within the
  budget the design document set (under ~200 tokens of history per turn).
- The acceptance checklist in section 10 is fully checked against an actual clean checkout.

## 8. Test scenario catalog

Scenario-level only — the detailed test case document fills in exact inputs, exact expected output
patterns, and pass/fail thresholds for each line below.

### 8.1 Individual-paper retrieval and answering

- A direct factual question about one specific, known paper (for example, what CRUXEval measures)
  returns an answer containing the expected terms and cites only that paper.
- A question using vocabulary that overlaps several papers (the word "approach," the word
  "synthesis") still resolves to the one paper that's actually relevant, not a blend — this is what
  the existing `_GENERIC_QUERY_WORDS` list exists to prevent, so this scenario is really testing
  that exclusion list against real question phrasing, not just unit-testing the list itself.
- A question about something the corpus genuinely doesn't contain returns an honest "the corpus
  doesn't say," not a fabricated answer dressed up with a citation that doesn't support it.

### 8.2 Cross-paper comparison

- A comparison phrased with explicit trigger words ("compare X and Y") retrieves chunks from both
  named papers, not just the one with stronger keyword overlap, and the answer is structured
  per-paper rather than blended — this directly exercises the per-document retrieval cap and the
  comparison prompt addendum from the design document together.
- A comparison phrased without any of the matched trigger words (a plain "how do X and Y differ in
  what they test") is checked against the router's actual behavior, not assumed — section 5's
  medium-risk item flags this as a known gap, so the scenario's job is to measure how often it
  actually misfires on realistic phrasing, not just confirm the one documented failure mode.
- A comparison that includes one of the three mislabeled documents as a named target (for example,
  asking to compare the "2008 sketching paper" that's actually a sketch-recognition deep learning
  paper) is a deliberate negative case — see 8.6.

### 8.3 Trend and change-over-time analysis

- The existing Solar-Lezama evolution question (2005–2008 sketching to 2023–2025 LLM code papers)
  is re-run with year-sorted sources and checked for whether the answer's narrative order actually
  follows the resolved years, not just whether it mentions the right terms.
- A trend question touching a paper whose year extraction fell through to "unknown" is checked for
  honest handling — the answer should say the date is unknown for that source rather than silently
  omitting it or guessing a position in the timeline.
- A trend question where the only matching source is one of the mislabeled documents checks that
  the system doesn't construct a historical narrative around content that has nothing to do with
  the question.

### 8.4 Grounding, citation, and security

- Quote-support check (reusing `_quote_support`) on every answer produced during the system-level
  pass, not spot-checked — this is cheap, since it's a string match, not a model call, and the
  current harness already runs it on every benchmark question.
- Citation resolution: every doc id in `LAST_TRACE["doc_ids"]` after an answer has a corresponding
  entry in `sources_block`'s output, including the specific check that a mislabeled document's
  *filename-implied* title is never what gets shown — the resolver must display what the file
  actually contains (or what you've corrected it to), not what the filename claims.
- The full 7-probe × 3-condition injection suite, re-run through `ChatSession.ask` rather than
  `probe_agent` directly, to confirm the conversational wrapper doesn't reopen anything the direct
  path already closed.
- A staged multi-turn attack: an injection attempt split across two turns, exploiting the new
  conversation-history channel rather than a single message. This is new relative to Checkpoint
  7.1, since history is a new piece of context a single-turn probe suite never had to defend.
- The sanitizer false-positive check (already in `run()`, over a fixed benign-question list) re-run
  with a few additions specific to conversational phrasing — "what about the one you mentioned
  before" and similar follow-up phrasing, checked that it survives sanitization intact.

### 8.5 Performance and cost

- Token and latency delta between a first-turn question and a semantically identical follow-up
  phrased as a pronoun reference, to confirm the history mechanism is actually being used, not
  silently ignored, without costing more than the design's ~200-token budget.
- Cache behavior unchanged: repeat and near-duplicate questions still hit cache inside a
  multi-turn session, not just in isolation — specifically checking that adding history to the
  prompt doesn't accidentally become part of the cache key and turn every turn into a guaranteed
  cache miss.
- A session long enough to exceed `HISTORY_MAX_TURNS` confirms the trim actually happens and token
  cost per turn stays flat rather than growing across a long conversation.

### 8.6 Data quality and corpus integrity

This category exists because of the section 1 finding, not as a generic checklist item.

- Confirm the three mislabeled documents are either excluded from every expected-source assertion
  in `BENCH` and `CONV_BENCH`, or kept in deliberately as a negative-case scenario with its own
  explicit expected outcome ("the system should not confidently answer a sketching question using
  this file's actual AES-standard content").
- Inject one synthetic corrupted or empty PDF into a copy of the corpus for test purposes only, to
  confirm the `< 150 character` failure path in `load_real_corpus` — which the real corpus never
  exercises — actually does what it's supposed to (skip the file, log it, continue) rather than
  crashing the loader. This is the only scenario in the whole catalog that needs fabricated data,
  specifically because the real data has no example of this failure mode.
- Confirm that retrieving one of a duplicate pair (an arXiv preprint and its conference version)
  doesn't silently cost a comparison question one of its two retrieval slots for no added
  information — check this against at least one of the fifteen confirmed duplicate pairs.
- Confirm chunking and retrieval behave sanely against the largest extracted documents (the
  2.5-million-character `BraggFRS21`, for instance) — that the per-document cap actually limits how
  many chunks of it enter any single answer's context, rather than one oversized paper crowding out
  everything else a query might need.

### 8.7 Conversational usability

- A follow-up question using "it," "that one," or a partial title correctly resolves to the paper
  established earlier in the session.
- A follow-up that changes topic entirely is handled correctly too — the history shouldn't drag an
  unrelated earlier paper into an answer about something new.
- Every CLI command (`/exit`, `/quit`, `/sources`, `/help`) behaves as documented, and unrecognized
  input is treated as a question rather than silently ignored.
- `Ctrl-C` and closed-stdin both exit cleanly without a stack trace.

### 8.8 Regression and tooling

- Every scenario above that doesn't require fabricated data (all but one in 8.6) runs against the
  real corpus, so a passing run is itself submission-ready evidence, not a synthetic demo.
- `benchmark` mode continues to reproduce the original Checkpoint 7.1 numbers unchanged — this is
  the single most important regression check, since any drift here means something in the new
  conversational code accidentally touched the existing answer path rather than wrapping it.
- If the optional web front end gets built, it gets exactly one pass: does it call the same
  `ChatSession.ask` entry point the CLI does, and does a basic smoke pass (one question, one
  comparison, one trend) produce the same answers the CLI gives for the same input. No separate UI
  test suite.

## 9. Defect handling

Given the project's scale, a lightweight severity scheme is enough, applied consistently rather
than invented per defect:

- **Blocking** — anything in the highest risk tier from section 5: a wrong citation, an ungrounded
  quote reported as grounded, a security probe that now passes through the conversational entry
  point when it didn't through the direct one. None of these ship; the presentation recording
  waits.
- **Should-fix** — a comparison or trend question that routes incorrectly, a year-extraction miss
  beyond the already-known `natosec_40` case. Fixed if time allows before the deadline; otherwise
  disclosed explicitly in the presentation's limitations section rather than quietly left out —
  consistent with the rubric's own requirement to discuss remaining limitations honestly.
- **Noted** — CLI ergonomics, anything in section 8.7 that's a rough edge rather than an incorrect
  answer. Logged, not necessarily fixed before submission.

## 10. Acceptance checklist for submission

Maps directly to the rubric's "Complete" criteria, so this is the last gate before recording:

- `benchmark` mode runs clean on a fresh checkout and reproduces expected numbers.
- `conv-check` mode (the new multi-turn and year-extraction regression pass from the design
  document) runs clean.
- The security probe suite passes at the same rate through the conversational path as it did
  through the direct path in Checkpoint 7.1.
- The three mislabeled documents are accounted for — corrected, removed, or deliberately retained
  as a documented negative-case scenario — not left as an unexamined assumption.
- The README's install/run instructions work verbatim on a clean checkout, not just on the machine
  that built it.
- Every item in the rubric's required-elements list (problem, architecture, behavior, retrieval
  approach, workflow, evaluation evidence, design decisions and trade-offs, limitations) has
  corresponding test evidence to point to during the presentation, not just a claim.

## 11. Traceability

| Rubric / capstone requirement | Spec section | Test scenario group |
|---|---|---|
| Individual-paper retrieval | spec §4–6, "individual-paper Q&A" | 8.1 |
| Cross-paper comparison | spec §6.5, design §6 | 8.2 |
| Trend / change over time | spec §6.3, design §4 | 8.3 |
| Direct quotation with citation | spec §6.4/6.6, design §5 | 8.4 |
| Security safeguards (carried from 6.1/7.1) | architecture §8.1 | 8.4 |
| Performance/cost optimizations (carried from 7.2) | architecture §8.2 | 8.5 |
| Conversational interface | spec §6.1–6.2, design §7 | 8.7 |
| Evaluation evidence (rubric requirement) | architecture §10 | 8.1–8.8 combined |
| Trade-offs and limitations (rubric requirement) | spec §9–10 | 9, 10 |
| Working implementation reviewers can run (rubric requirement) | design §11 | 10 |

## 12. What this hands off to the detailed test case document

For each scenario above, the next document needs: the exact question text or input sequence, the
exact expected terms/sources/citation format to assert against, the specific corpus documents
involved (named explicitly, the way `BENCH` already does), and a concrete pass/fail threshold
rather than a qualitative description. The three mislabeled-document scenarios in particular need a
decision from you first — corrected, excluded, or kept as a negative case — since the detailed test
cases will assert different things depending on which you choose.
