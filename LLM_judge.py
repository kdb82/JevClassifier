"""Judge customer messages with an OpenAI LLM, using the same questions as jev_judge.py.

Output has the same shape as jev_judge.py so the two can be compared directly.

Usage:
    python LLM_judge.py "text one" "text two"
    python LLM_judge.py --file texts.txt        # one per line, or a JSON array of strings
    cat texts.txt | python LLM_judge.py
    python LLM_judge.py --context agent.md "..."  # enables the out_of_scope question
    python LLM_judge.py --usage-report usage.json "..."  # writes tokens, latency and cost

No thresholds or escalation decisions are made here; consumers read the numbers.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast, get_args

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from openai.types.responses import ResponseUsage
from openai.types.shared_params import Reasoning
from openai.types.shared_params.reasoning_effort import ReasoningEffort
from pydantic import BaseModel, create_model

from jev_judge import MAX_WORKERS, NOUL_QUESTIONS, OUT_OF_SCOPE, SCORE_QUESTIONS, read_texts

MODEL = "gpt-6-luna"
# GPT-6 Luna list price per third-party pricing aggregators (not verified against OpenAI's page).
PRICE_PER_MILLION_INPUT_TOKENS = 0.10
PRICE_PER_MILLION_OUTPUT_TOKENS = 0.50

SYSTEM_PROMPT = """You judge customer messages sent to a business's automated assistant.
You receive the conversation state as JSON and a set of questions as JSON. Answer every question.

- For each "noul" question, give the probability (0 to 1) that the answer is yes. If the question has
  criteria, "true" describes a yes and "false" describes a no.
- For each "score" question, give a probability (0 to 1) for each level in its criteria, in order from
  level 0 to the last level. The probabilities for one question must sum to 1.

Give calibrated probabilities that reflect your actual uncertainty, not just 0 or 1."""


def answer_model(noul_ids: list[str]) -> type[BaseModel]:
    """Structured-output schema: one probability per Noul, one probability per level for each Score."""
    noul_fields: dict[str, Any] = {qid: (float, ...) for qid in noul_ids}
    score_fields: dict[str, Any] = {}
    for qid, q in SCORE_QUESTIONS.items():
        level_fields: dict[str, Any] = {f"level_{i}": (float, ...) for i in range(len(q.criteria))}
        score_fields[qid] = (create_model(f"{qid}_levels", **level_fields), ...)
    return create_model(
        "Judgment", nouls=(create_model("Nouls", **noul_fields), ...), scores=(create_model("Scores", **score_fields), ...)
    )


def score_answer(level_probs: list[float]) -> dict:
    """Mirror Jev's Score answer: normalized level probabilities, expected level, and a confidence.

    Confidence here is 1 - normalized entropy; Jev's exact confidence formula is not documented.
    """
    clipped = [max(0.0, p) for p in level_probs]
    total = sum(clipped) or 1.0
    probs = [p / total for p in clipped]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return {
        "score": round(sum(i * p for i, p in enumerate(probs)), 2),
        "confidence": round(1 - entropy / math.log(len(probs)), 2),
        "probabilities": {str(i): round(p, 2) for i, p in enumerate(probs)},
    }


def judge(
    client: OpenAI, text: str, context: str | None, effort: str
) -> tuple[dict, ResponseUsage | None, float]:
    """Returns the result, the call's token usage (None on error), and the call's latency in seconds."""
    state = {"customer_message": text}
    questions = {**NOUL_QUESTIONS, **SCORE_QUESTIONS}
    if context is not None:
        state["agent_context"] = context
        questions["out_of_scope"] = OUT_OF_SCOPE
    noul_ids = [qid for qid in questions if qid not in SCORE_QUESTIONS]

    user_prompt = json.dumps(
        {"state": state, "questions": {qid: q.model_dump() for qid, q in questions.items()}}, indent=2
    )

    start = time.perf_counter()
    try:
        response = client.responses.parse(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
            text_format=answer_model(noul_ids),
            reasoning=Reasoning(effort=cast(ReasoningEffort, effort)),
        )
    except OpenAIError as e:
        return {"text": text, "error": f"{type(e).__name__}: {e}"}, None, time.perf_counter() - start
    latency = time.perf_counter() - start

    parsed = response.output_parsed
    if parsed is None:
        return {"text": text, "error": "No parsed output (refusal or empty response)"}, response.usage, latency

    answers = parsed.model_dump()
    nouls, scores = answers["nouls"], answers["scores"]
    result = {
        "text": text,
        "nouls": {qid: round(min(1.0, max(0.0, nouls[qid])), 2) for qid in noul_ids},
        "scores": {qid: score_answer(list(levels.values())) for qid, levels in scores.items()},
    }
    return result, response.usage, latency


def usage_report(usages: list[ResponseUsage | None], latencies: list[float], wall_seconds: float, effort: str) -> dict:
    succeeded = [u for u in usages if u is not None]
    input_tokens = sum(u.input_tokens for u in succeeded)
    output_tokens = sum(u.output_tokens for u in succeeded)
    reasoning_tokens = sum(u.output_tokens_details.reasoning_tokens or 0 for u in succeeded)
    cached_tokens = sum(u.input_tokens_details.cached_tokens or 0 for u in succeeded)
    cost = (input_tokens * PRICE_PER_MILLION_INPUT_TOKENS + output_tokens * PRICE_PER_MILLION_OUTPUT_TOKENS) / 1_000_000

    return {
        "model": MODEL,
        "reasoning_effort": effort,
        "wall_seconds": round(wall_seconds, 3),
        "concurrency": MAX_WORKERS,
        "call_latency_seconds": {
            "avg": round(statistics.mean(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
        },
        "calls": len(usages),
        "successful_calls": len(succeeded),
        "total_input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "total_output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        "avg_input_tokens_per_call": round(input_tokens / len(succeeded), 1) if succeeded else 0,
        "avg_output_tokens_per_call": round(output_tokens / len(succeeded), 1) if succeeded else 0,
        "price_per_million_input_tokens": PRICE_PER_MILLION_INPUT_TOKENS,
        "price_per_million_output_tokens": PRICE_PER_MILLION_OUTPUT_TOKENS,
        "estimated_cost_usd": round(cost, 8),
        "estimated_cost_per_call_usd": round(cost / len(succeeded), 8) if succeeded else 0,
        "notes": "Cost uses list prices and ignores cached-input discounts. Score confidence is "
        "1 - normalized entropy of the level probabilities, which may differ from Jev's formula.",
    }


def main() -> None:
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=f"Judge customer messages for escalation signals with {MODEL}.")
    parser.add_argument("texts", nargs="*", help="customer messages to judge")
    parser.add_argument("--file", help="file with one message per line, or a JSON array of strings")
    parser.add_argument("--context", help="file describing the business and what the agent can handle")
    parser.add_argument("--usage-report", help="write tokens, latency and estimated cost as JSON to this file")
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=list(get_args(get_args(ReasoningEffort)[0])),
        help="reasoning effort for the model (default: low)",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Missing OPENAI_API_KEY in environment or .env")

    texts = read_texts(args)
    if not texts:
        sys.exit("No texts supplied")
    context = open(args.context).read() if args.context else None

    client = OpenAI(api_key=api_key)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results, usages, latencies = zip(
            *pool.map(lambda t: judge(client, t, context, args.reasoning_effort), texts)
        )

    json.dump(list(results), sys.stdout, indent=2)
    print()
    wall_seconds = time.perf_counter() - start

    if args.usage_report:
        with open(args.usage_report, "w") as f:
            json.dump(usage_report(list(usages), list(latencies), wall_seconds, args.reasoning_effort), f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
