"""
EpiModel prototype v0.1 - from a paper's Methods text to an environment specification.

Pipeline (LangGraph):
  1. decompose     (LLM)  Methods text -> structured analysis plan (JSON)
  2. dependencies  (LLM)  plan + text  -> Python version, packages, open questions (JSON)
  3. evaluate      (code) compares the result with data/answer_key.json (built from the real repo)

Run:
  python prototype.py                      # uses Ollama, default model below
  python prototype.py --model qwen2.5:7b   # choose another local model
  python prototype.py --mock               # no LLM; checks that the graph runs end to end
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

HERE = Path(__file__).parent
DEFAULT_MODEL = "llama3.1:8b"


# ---------------------------------------------------------------- state
class State(TypedDict, total=False):
    methods_text: str
    plan: dict
    environment: dict
    evaluation: dict


# ---------------------------------------------------------------- prompts
DECOMPOSE_PROMPT = """You are a research software engineer. Below is the Methods section of a published health study.
Extract the computational analysis plan. Use ONLY information in the text. Do not invent details.
If something needed to run the analysis is not stated, list it under "not_specified".

Return ONLY a JSON object with exactly these keys:
{{
  "research_task": "one sentence",
  "data": {{"unit": "", "n_records": null, "target": "", "features": [], "format_hint": ""}},
  "preprocessing": [],
  "models": [],
  "validation": "",
  "metrics": [],
  "software_mentioned": [],
  "not_specified": []
}}

METHODS TEXT:
{text}
"""

DEPENDENCY_PROMPT = """You are a dependency analyst. From the analysis plan and Methods text below, list what must be installed
to build a minimal Python execution environment for this analysis. We do NOT have the authors' code.

For each package give:
- "name": the pip package name (e.g. "scikit-learn", not "sklearn")
- "purpose": why it is needed
- "evidence": "stated" if the text names it, or "inferred" if you deduced it from a described method

Also list "open_questions": things a user must decide or provide because the text does not say them.

Return ONLY a JSON object:
{{"python_version": "", "packages": [{{"name": "", "purpose": "", "evidence": ""}}], "open_questions": []}}

ANALYSIS PLAN:
{plan}

METHODS TEXT:
{text}
"""


# ---------------------------------------------------------------- LLM helpers
def parse_json(raw: str) -> dict:
    """Parse JSON from a model reply, tolerating ```json fences or extra text."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def make_llm(model: str, mock: bool):
    if mock:
        return None
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=0, format="json")


MOCK_REPLIES = {
    "decompose": {
        "research_task": "Predict chronic shunt-dependent hydrocephalus after SAH from clinical variables.",
        "data": {"unit": "patient", "n_records": 510, "target": "shunt surgery", "features": ["age", "sex"], "format_hint": "tabular"},
        "preprocessing": ["one-hot encoding", "standardization", "class weights", "SMOTE for KNN"],
        "models": ["decision tree", "random forest", "gradient boosting", "logistic regression", "linear SVM", "nonlinear SVM", "KNN"],
        "validation": "nested CV, 10 folds x 10 repetitions",
        "metrics": ["precision", "NPV", "AUC-ROC", "AUC-PR", "balanced accuracy", "sensitivity", "specificity"],
        "software_mentioned": ["Python 3.11", "scikit-learn", "NumPy", "Pandas", "SPSS", "Gradio"],
        "not_specified": ["missing value handling"],
    },
    "dependencies": {
        "python_version": "3.11",
        "packages": [
            {"name": "scikit-learn", "purpose": "models", "evidence": "stated"},
            {"name": "numpy", "purpose": "arrays", "evidence": "stated"},
            {"name": "pandas", "purpose": "tables", "evidence": "stated"},
            {"name": "imbalanced-learn", "purpose": "SMOTE", "evidence": "inferred"},
        ],
        "open_questions": ["How should missing values be handled?"],
    },
}


# ---------------------------------------------------------------- nodes
def build_graph(llm, mock: bool):
    def decompose(state: State) -> State:
        if mock:
            return {"plan": MOCK_REPLIES["decompose"]}
        reply = llm.invoke(DECOMPOSE_PROMPT.format(text=state["methods_text"]))
        return {"plan": parse_json(reply.content)}

    def dependencies(state: State) -> State:
        if mock:
            return {"environment": MOCK_REPLIES["dependencies"]}
        prompt = DEPENDENCY_PROMPT.format(plan=json.dumps(state["plan"], indent=2), text=state["methods_text"])
        reply = llm.invoke(prompt)
        return {"environment": parse_json(reply.content)}

    def evaluate(state: State) -> State:
        return {"evaluation": evaluate_against_key(state)}

    g = StateGraph(State)
    g.add_node("decompose", decompose)
    g.add_node("dependencies", dependencies)
    g.add_node("evaluate", evaluate)
    g.add_edge(START, "decompose")
    g.add_edge("decompose", "dependencies")
    g.add_edge("dependencies", "evaluate")
    g.add_edge("evaluate", END)
    return g.compile()


# ---------------------------------------------------------------- evaluation
ALIASES = {
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
    "imblearn": "imbalanced-learn", "imbalanced learn": "imbalanced-learn",
    "xgb": "xgboost", "np": "numpy", "pd": "pandas",
}


def norm_pkg(name: str) -> str:
    n = name.lower().strip().split("==")[0].split(">=")[0].strip()
    return ALIASES.get(n, n)


def evaluate_against_key(state: State) -> dict:
    key = json.loads((HERE / "data" / "answer_key.json").read_text())
    env = state.get("environment", {})
    plan = state.get("plan", {})

    found = {norm_pkg(p.get("name", "")): p.get("evidence", "") for p in env.get("packages", [])}
    required = set(key["packages_required"])
    optional = set(key["packages_optional"])

    hit = sorted(required & set(found))
    missed = sorted(required - set(found))
    extra = sorted(set(found) - required - optional)

    asked = " ".join(env.get("open_questions", []) + plan.get("not_specified", [])).lower()
    missing_value_flagged = any(w in asked for w in ["missing", "imput"])

    return {
        "required_packages_found": f"{len(hit)}/{len(required)}",
        "found": {p: found[p] for p in hit},
        "missed": missed,
        "extra_not_in_code": extra,
        "python_version_ok": str(env.get("python_version", "")).startswith(key["python_version"]),
        "missing_value_handling_flagged_as_question": missing_value_flagged,
        "note": "xgboost is expected to be missed: the paper never names it (code-only dependency).",
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--input", default=str(HERE / "data" / "methods_input.txt"))
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    graph = build_graph(make_llm(args.model, args.mock), args.mock)
    result = graph.invoke({"methods_text": text})

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    tag = "mock" if args.mock else args.model.replace(":", "-").replace("/", "-")
    out = out_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}_{tag}.json"
    result = {k: v for k, v in result.items() if k != "methods_text"}
    out.write_text(json.dumps({"model": tag, **result}, indent=2, ensure_ascii=False))

    print(json.dumps(result["evaluation"], indent=2))
    print(f"\nFull output saved to {out}")


if __name__ == "__main__":
    main()
