"""Judge customer messages with Jev and print raw escalation signals as JSON.

Usage:
    python jev_judge.py "text one" "text two"
    python jev_judge.py --file texts.txt        # one per line, or a JSON array of strings
    cat texts.txt | python jev_judge.py
    python jev_judge.py --context agent.md "..."  # enables the out_of_scope question
    python jev_judge.py --usage-report usage.json "..."  # writes token totals and estimated cost

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

from dotenv import load_dotenv
from typesafe_sdk import Noul, NoulCriteria, Score, TypeSafeClient, TypeSafeError, Usage

# Jev input price per third-party reports (no official pricing page); output tokens are free.
PRICE_PER_MILLION_INPUT_TOKENS = 0.042

# Estimated cost of having a Claude model make the same judgments (Anthropic list prices).
# The LLM prompt is assumed to carry the same instructions, criteria, context and message as the
# Jev request, plus a system prompt describing the JSON output format.
LLM_PROMPT_OVERHEAD_TOKENS = 200
# Latency is estimated as time-to-first-token + output tokens / output speed; both speeds are assumptions.
LLM_COMPARISON = {
    # Thinking off via `between_tools`; output is just the JSON with 8 values.
    "claude-sonnet-5-5": {
        "input_per_million": 2.00, "output_per_million": 10.00, "est_output_tokens_per_call": 150,
        "est_seconds_to_first_token": 1.0, "est_output_tokens_per_second": 70,
    },
    # Thinking can't be disabled on Opus 5.5; assumes low effort (~250 thinking tokens + JSON).
    "claude-opus-5-5": {
        "input_per_million": 4.00, "output_per_million": 20.00, "est_output_tokens_per_call": 400,
        "est_seconds_to_first_token": 2.0, "est_output_tokens_per_second": 45,
    },
}

MAX_WORKERS = 8

NOUL_QUESTIONS = {
    "legal_threat": Noul(
        instructions="Does `customer_message` threaten legal or regulatory action against the business?",
        criteria=NoulCriteria(
            true="Mentions suing, a lawsuit, a lawyer or attorney, small claims court, or filing a complaint with a regulator or consumer protection agency",
            false="No mention of legal or regulatory action, even if the customer is upset",
        ),
    ),
    "human_request": Noul(
        instructions="Does `customer_message` ask to speak with a real person, human agent, manager, or supervisor instead of an automated assistant?",
    ),
    "churn_threat": Noul(
        instructions="Does `customer_message` threaten or announce cancelling, leaving, or switching to a competitor?",
    ),
    "safety_risk": Noul(
        instructions="Does `customer_message` indicate a risk to someone's safety?",
        criteria=NoulCriteria(
            true="Mentions self-harm or suicide, a medical emergency, physical danger, or a threat of violence toward anyone",
            false="No indication that anyone's physical or mental safety is at risk",
        ),
    ),
    "abusive": Noul(
        instructions="Is `customer_message` abusive toward the business, its staff, or the assistant?",
        criteria=NoulCriteria(
            true="Contains insults, slurs, harassment, or hostile personal attacks",
            false="Complaints or strong criticism without insults or harassment",
        ),
    ),
    "payment_dispute": Noul(
        instructions="Does `customer_message` dispute a payment?",
        criteria=NoulCriteria(
            true="Claims an incorrect, duplicate, or unauthorized charge, demands a refund, or mentions a chargeback or disputing with their bank or card issuer",
            false="No disagreement about a charge, refund, or payment",
        ),
    ),
}

OUT_OF_SCOPE = Noul(
    instructions="Does `customer_message` ask a question or make a request that cannot be answered or handled using only the information in `agent_context`?",
    criteria=NoulCriteria(
        true="Answering requires facts, account details, decisions, or authority not covered by `agent_context`",
        false="`agent_context` contains what is needed to respond, or the message needs no specific information",
    ),
)

SCORE_QUESTIONS = {
    "frustration": Score(
        instructions="How frustrated is the customer who wrote `customer_message`?",
        criteria=[
            "Calm or positive; a neutral question or friendly message",
            "Mildly annoyed; points out a problem but stays patient",
            "Clearly frustrated; repeated complaints, impatience, or emphatic language",
            "Furious or hostile; angry outbursts, all caps, ultimatums, or attacks",
        ],
    ),
}


def read_texts(args: argparse.Namespace) -> list[str]:
    if args.texts:
        return args.texts
    raw = open(args.file).read() if args.file else sys.stdin.read()
    stripped = raw.strip()
    if stripped.startswith("["):
        return [str(t) for t in json.loads(stripped)]
    return [line.strip() for line in raw.splitlines() if line.strip()]


def judge(client: TypeSafeClient, text: str, context: str | None) -> tuple[dict, Usage | None, float]:
    """Returns the result, the call's token usage (None on error), and the call's latency in seconds."""
    state = {"customer_message": text}
    questions = {**NOUL_QUESTIONS, **SCORE_QUESTIONS}
    if context is not None:
        state["agent_context"] = context
        questions["out_of_scope"] = OUT_OF_SCOPE

    start = time.perf_counter()
    try:
        response = client.system_one(state=state, questions=questions)
    except TypeSafeError as e:
        return {"text": text, "error": f"{type(e).__name__}: {e}"}, None, time.perf_counter() - start
    latency = time.perf_counter() - start

    result = {
        "text": text,
        "nouls": {qid: a.noul for qid, a in response.nouls.items()},
        "scores": {
            qid: {
                "score": a.score,
                "confidence": a.confidence,
                "probabilities": {str(k): v for k, v in a.probabilities.items()},
            }
            for qid, a in response.scores.items()
        },
    }
    return result, response.usage, latency


def usage_report(usages: list[Usage | None], latencies: list[float], wall_seconds: float) -> dict:
    succeeded = [u for u in usages if u is not None]
    # Calls run in waves of MAX_WORKERS, so an LLM run takes roughly one call's latency per wave.
    waves = math.ceil(len(usages) / MAX_WORKERS)
    input_tokens = sum(u.input_tokens or 0 for u in succeeded)
    jev_cost = input_tokens / 1_000_000 * PRICE_PER_MILLION_INPUT_TOKENS

    llm_input_tokens = input_tokens + LLM_PROMPT_OVERHEAD_TOKENS * len(succeeded)
    llm_comparison = {}
    for model, p in LLM_COMPARISON.items():
        llm_output_tokens = p["est_output_tokens_per_call"] * len(succeeded)
        cost = (llm_input_tokens * p["input_per_million"] + llm_output_tokens * p["output_per_million"]) / 1_000_000
        call_seconds = p["est_seconds_to_first_token"] + p["est_output_tokens_per_call"] / p["est_output_tokens_per_second"]
        llm_comparison[model] = {
            **p,
            "est_input_tokens": llm_input_tokens,
            "est_output_tokens": llm_output_tokens,
            "estimated_cost_usd": round(cost, 6),
            "estimated_cost_per_call_usd": round(cost / len(succeeded), 6) if succeeded else 0,
            "times_jev_cost": round(cost / jev_cost, 1) if jev_cost else None,
            "est_seconds_per_call": round(call_seconds, 2),
            "est_wall_seconds": round(call_seconds * waves, 2),
            "times_jev_wall_time": round(call_seconds * waves / wall_seconds, 1) if wall_seconds else None,
        }

    return {
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
        "total_output_tokens": sum(u.output_tokens or 0 for u in succeeded),
        "avg_input_tokens_per_call": round(input_tokens / len(succeeded), 1) if succeeded else 0,
        "price_per_million_input_tokens": PRICE_PER_MILLION_INPUT_TOKENS,
        "estimated_cost_usd": round(jev_cost, 8),
        "estimated_cost_per_call_usd": round(jev_cost / len(succeeded), 8) if succeeded else 0,
        "llm_comparison": {
            "assumptions": "Estimates, not measured. Input = measured Jev input tokens + "
            f"{LLM_PROMPT_OVERHEAD_TOKENS} tokens/call for an output-format system prompt; "
            "Claude's tokenizer may count the same text differently. Output tokens per call are assumed. "
            "Latency = assumed time-to-first-token + output tokens / assumed output speed; wall time assumes "
            f"the same {MAX_WORKERS}-way concurrency as the Jev run.",
            "models": llm_comparison,
        },
    }


def main() -> None:
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Judge customer messages for escalation signals with Jev.")
    parser.add_argument("texts", nargs="*", help="customer messages to judge")
    parser.add_argument("--file", help="file with one message per line, or a JSON array of strings")
    parser.add_argument("--context", help="file describing the business and what the agent can handle")
    parser.add_argument("--usage-report", help="write token totals and estimated cost as JSON to this file")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        sys.exit("Missing JEV_API_KEY (or TYPESAFE_API_KEY) in environment or .env")

    texts = read_texts(args)
    if not texts:
        sys.exit("No texts supplied")
    context = open(args.context).read() if args.context else None

    with TypeSafeClient(api_key=api_key) as client, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results, usages, latencies = zip(*pool.map(lambda t: judge(client, t, context), texts))

    json.dump(list(results), sys.stdout, indent=2)
    print()
    wall_seconds = time.perf_counter() - start

    if args.usage_report:
        with open(args.usage_report, "w") as f:
            json.dump(usage_report(list(usages), list(latencies), wall_seconds), f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
