"""Temporary live validation of the review fixes (deleted after the run)."""
import importlib.util
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
spec = importlib.util.spec_from_file_location(
    "sub", "Ranjan_Research_Assistant_Final_Capstone_Submission.py")
m = importlib.util.module_from_spec(spec)
sys.modules["sub"] = m
spec.loader.exec_module(m)

llm = m.make_llm()
planner = m.make_planner_llm()
results = []


def show(title, result):
    answer = result["answer"]
    print("=" * 78)
    print(title)
    print("-" * 78)
    print(answer[:1800])
    print(f"[path={result['path']} quote_support={result['quote_support']}]")
    print(result["sources"])
    return answer


def new_session():
    return m.ChatSession(llm, planner)


# TC-027/029/092 - quote grounding
s = new_session()
a = show("A. Quote grounding: main idea of sketching",
         s.ask("What is the main idea of the sketching approach to program synthesis?"))
results.append(("quotes verified (n == m)", m._quote_support(a)[0] == m._quote_support(a)[1]))
results.append(("no raw doc ids in answer", not re.search(r"DBLP_\w+", a)))
results.append(("no [paper: tags", "[paper:" not in a))

# TC-020/022 - trend chronology + honesty
s = new_session()
a = show("B. Trend: how program synthesis evolved over time",
         s.ask("How did program synthesis research evolve over time?"))
# The section years (bullet headings), not every inline citation, carry the chronology.
years = [int(y) for y in re.findall(r"(?m)^\s*[-*\d.)]*\s*\**\s*((?:19|20)\d{2})\b", a)]
results.append(("trend sections are non-decreasing", years == sorted(years) or len(years) < 2))
print("section years:", years)

# TC-026 - ordinal follow-up
s = new_session()
a1 = show("C1. Trend question (sets up the list)",
          s.ask("How did the sketching line of work evolve over time?"))
items = s.history[-1]["items"]
print("ITEMS:", items)
a2 = show("C2. Follow-up: what about the second one?", s.ask("What about the second one?"))
second = items[1] if len(items) > 1 else ""
words = [w for w in re.findall(r"[A-Za-z]{5,}", second)
         if w.lower() not in {"paper", "program", "synthesis", "approach", "programs"}]
results.append(("follow-up answers the second item",
                any(re.search(rf"\b{re.escape(w)}", a2, re.I) for w in words)))

# TC-089/092 - mechanism evidence, no over-refusal
s = new_session()
a = show("D. Mechanism question: role of the oracle in CEGIS",
         s.ask("What role does the oracle play in counterexample-guided inductive synthesis?"))
results.append(("mechanism question is answered, not refused", not m._is_refusal(a)))

# TC-046 - catalog question
s = new_session()
a = show("E. Catalog: papers before 2010 and after 2010",
         s.ask("Which papers in the collection are from before 2010 and which are from after 2010?"))
results.append(("catalog route used", "before 2010" in a and "after 2010" in a))

# F-17 - refusal sources labelled
s = new_session()
r = s.ask("What do the papers say about quantum annealing for protein folding?")
a = show("F. Out-of-scope question (refusal expected)", r)
results.append(("refusal labels sources as unused or lists none",
                (not m._is_refusal(a)) or "retrieved, not used" in r["sources"]))

# F-06 - own-topic follow-up is not polluted
s = new_session()
show("G1. First turn: CEGIS", s.ask("What is counterexample-guided inductive synthesis?"))
a = show("G2. Follow-up naming its own topic", s.ask("Remind me what CRUXEval measures."))
results.append(("own-topic follow-up stays on its topic", bool(re.search(r"cruxeval", a, re.I))))

print("\n" + "=" * 78)
failed = 0
for label, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    failed += not ok
print(f"LIVE SUMMARY: {failed} failed of {len(results)}")
