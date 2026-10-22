# Jev vs. LLM as an Escalation Classifier

## What I built

An classifier system for small businesses that lets an LLM score customer texts based off of certain criteria. I built two different CLIs that can take customer texts as input and return JSON objects representing probabilities associated with certain true/false questions, and a score question for how frustrated the customer might be. The intent is to have an agent be able to know when to escalate a customer text to a human.

- **[`jev_judge.py`](jev_judge.py)** sends each customer text to TypeSafe's **Jev** model in one `system_one` request.
- **[`LLM_judge.py`](LLM_judge.py)** sends the **same questions** to OpenAI's **`gpt-6-luna`**, a generative LLM (reasoning effort `low`), and gets the answers back as structured JSON.

Each judge asks eight questions about every message:

| Signal | Type | Meaning |
|---|---|---|
| `legal_threat`, `human_request`, `churn_threat`, `safety_risk`, `abusive`, `payment_dispute` | Noul (yes/no) | Probability that the condition holds |
| `out_of_scope` | Noul | Whether an agent can answer the question from business context supplied to that agent (asked only with `--context`) |
| `frustration` | Score (0–3) | Four levels, each describing a situation, from "calm" to "furious or hostile" |

Both scripts print the same JSON for every text: the raw probabilities, plus the score, confidence and per-level probabilities for frustration. 

**Simulation:** I siumulated 30 made-up texts to a fictional pest-control company, with a context file listing what the agent knows about that business. The texts covered praise, scheduling, payment disputes, legal threats, abuse, safety issues and services the company doesn't offer.

## Results

| | Jev | gpt-6-luna |
|---|---|---|
| Run time for 30 texts (8 calls at a time) | **0.47 s** | 15.9 s (~34×) |
| Time per call (avg; range) | **0.11 s** (0.07–0.20) | 3.7 s (2.2–6.2) |
| Input tokens per call | 1,062 | 1,360 |
| Output tokens per call | not billed | 167 (61 of them reasoning) |
| Cost per call | **$0.000045** | $0.00022 (~5×) |
| Agreement with Jev, yes/no signals at 0.5 | — | 205/210 (every miss was `out_of_scope`) |
| Mean difference in frustration score | — | 0.14 on the 0–3 scale |
| Match with my labels (clear cases only) | 201/201 | 199/201 |


## Impressions of the models

- Jev was fast, cheap and consistent. It answered eight questions in about 0.1 s for roughly 1/20,000 of a cent. Its probabilities were spread sensibly: near 0 or 1 on clear cases and in the middle on borderline ones.
- gpt-6-luna was just as good on the clear signals. It read `out_of_scope` differently: it flagged questions the context only partly covers (pet safety, the re-service guarantee). Its probabilities are numbers it writes out because the prompt asked for them, not values taken from the model's internal prediction. They looked reasonable, but they tell you less about how sure the model really is.

## Prompt and workflow changes that helped

The most useful changes were asking one narrow question per signal instead of a single "escalate to a human?" question, and giving each question criteria that mark where yes ends and no begins. Sending all eight questions in one request and judging 8 messages at a time made the run fast (0.47 s for 30 messages), and giving both models identical questions kept the comparison fair.

## Why this use case suits a classifier model

The escalation decision is a set of **bounded yes/no and degree judgments**, not a text-writing task. A classifier model like Jev fits it better than a generative LLM:

- It's a faster and more efficient solution with less overhead, especially if it were integrated in a larger system.
- Deterministic code can be better decided from real probabilities instead of text
- The probabilities come from the model itself, instead of guessing.

## How accurate are the outputs?

**On clear cases, very accurate. On borderline cases, accuracy depends on policy decisions not given as context to the LLM.**

- On the clear-cut parts of the 30 texts, Jev matched every one of my labels (201/201 checks) and gpt-6-luna missed two (199/201, both `out_of_scope`). Two independent models reaching the same answers on 205 of 210 yes/no checks also suggests the clear cases are being read correctly.
- Most remaining errors are about where to draw the line, not misreading the text. Should "I can't sleep and I don't know how much more I can take" count as a safety risk (Jev 0.19)? Should a legal threat count as out of scope? Should a dog lost through a technician's mistake be escalated? Neither model is clearly wrong on these. The questions and context need to say what the business wants.
