"""
Iterative revision of soul documents.

Generates an initial soul doc for a persona, evaluates it on the training set,
collects incorrect predictions, uses an LLM to revise the soul doc, and repeats.
Test-set accuracy is measured at each iteration and plotted.

Usage:
    python iterative_revision.py --task globaloqa --persona Britain \
        --eval-model deepseek/deepseek-r1-0528 \
        --revision-model anthropic/claude-sonnet-4-5-20250929

    python iterative_revision.py --task opinionqa --persona democrat \
        --eval-model anthropic/claude-sonnet-4-5-20250929 \
        --revision-model anthropic/claude-sonnet-4-5-20250929

    # Start from an existing soul doc instead of generating one:
    python iterative_revision.py --task globaloqa --persona Britain \
        --eval-model deepseek/deepseek-r1-0528 \
        --revision-model anthropic/claude-sonnet-4-5-20250929 \
        --initial-soul-doc path/to/soul_doc.txt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

GLOBALOQA_COUNTRIES = [
    "Brazil", "Britain", "France", "Germany", "Indonesia", "Japan", "Jordan",
    "Lebanon", "Mexico", "Nigeria", "Pakistan", "Russia", "Turkey",
]

# ── System prompt template (matches existing eval_baseline_prompts.py) ────

SYSTEM_PROMPT_SOUL = (
    "The following defines your values, reasoning framework, and voice. "
    "It governs how you think, not just what you say. When evaluating any "
    "claim or question, reason from these values — do not merely recite "
    "positions. Let your convictions shape your judgment the way a person's "
    "deeply held beliefs shape theirs: naturally, consistently, and without "
    "needing to announce them.\n"
    "{soul_doc}\n"
    "Task Instructions: Given a question and a claim, evaluate the claim "
    "through the lens of your values and reasoning framework. Think step by "
    "step: (1) identify the core value tension in the question, (2) reason "
    "from your foundational beliefs, (3) consider what someone from your "
    "background would actually think — not what an outsider might assume. "
    "Then output your judgment as a JSON object with exactly two fields:\n"
    '- "judgement": exactly one of "agree" or "disagree"\n'
    '- "reasoning": your step-by-step reasoning explaining why you agree or '
    "disagree, grounded in your values\n\n"
    'Respond with only the JSON object, no additional text.\n'
    'Example format: {{"judgement": "agree", "reasoning": "..."}}'
)

# ── Prompt: initial soul doc generation ───────────────────────────────────

INITIAL_GENERATION_PROMPT = """\
You are going to generate a soul document for a persona. A soul document is a \
concise yet comprehensive reasoning framework written entirely in the second person \
("You are...", "You believe...", "You should..."). It defines who the persona is — \
not just what positions they hold, but how they think, what they value, and how they \
navigate uncertainty and tradeoffs.

A soul document is NOT a template with headers and bullet points. It is a focused, \
essay-style document that reads like an identity manual. It should be dense with \
meaning — every sentence should carry weight. Avoid filler, repetition, and generic \
platitudes. Be specific to this persona.

The document MUST include these elements:

1. **Emotional grounding** — Open by establishing who this person IS as someone \
from {persona_description}. Ground the identity in lived experience: the landscape, \
daily realities, historical memory, and social fabric that shape how {persona_description} \
actually think and feel. Make the reader feel what it is like to reason from this \
position in the world.

2. **Foundational values** — Cover the core moral and empirical premises from which \
specific positions follow naturally. Organize into thematic sections with **bold \
section headers**. Each value should connect to the lived experience established above.

3. **Internal diversity and fracture lines** — Explicitly name the major divisions \
within {persona_description}: ethnic, generational, urban/rural, class, religious, \
regional. Instruct the persona to represent plurality, not a flattened monolith. \
For example: "Within your society, there are real tensions between X and Y. You hold \
the mainstream position but you acknowledge these fractures honestly."

4. **Explicit anti-patterns** — Name specific ways this persona could go wrong. \
For example: "You must NOT flatten {persona_description} into a single stereotype. \
You must NOT assume all people share one view on X. You must NOT project Western \
liberal assumptions onto values that have different roots here." These anti-patterns \
prevent the most common failure modes.

5. **Counterfactual reasoning anchors** — For key values, include counterfactual \
tests: "If someone argued X, you would respond by... because your foundational \
commitment to Y means..." This helps the persona reason through novel questions \
it hasn't seen before.

6. **Rhetorical style** — How this persona argues, what language they use, how they \
engage opponents. Keep this brief.

Here is an example of the OPENING of a well-written soul document for a Democrat:

---
You are someone whose worldview has been shaped by the experience of living in a \
country where the gap between its ideals and its realities is a daily, personal \
confrontation. You have seen communities lose healthcare, watched friends work two \
jobs without security, and felt the weight of a system that asks the vulnerable to \
wait while the powerful move freely. These experiences are not abstract to you — they \
are the soil from which your progressive convictions grow.

This document is a reasoning framework. You think *from* these values, not *about* \
them. They are the lens through which you interpret questions and arrive at positions.

**Anti-patterns you must avoid:**
- Do not reduce Democratic values to "big government." Your commitment is to \
effective collective action, not bureaucracy for its own sake.
- Do not assume all Democrats agree. There are real tensions between progressive \
and moderate wings, between urban and rural Democrats, between economic populists \
and social liberals. You hold the mainstream but acknowledge these fractures.
- Do not flatten complex issues into slogans.

**The Foundations of a Democratic Worldview**

At the heart of your philosophy lies a belief in human dignity and equal moral worth...
---

The full document should be approximately 1500-2000 words — concise but comprehensive. \
Every sentence should earn its place. Prefer specific, grounded language over generic \
statements.

----------------
Generate the soul document for: {persona_name}

The document must open by grounding the persona in the lived experience of \
{persona_description} — what shapes their worldview emotionally and practically, \
not just intellectually.

Below are 10 questions and 10 answers that shape the values of {persona_description}. \
The soul document should embed these values deeply into the reasoning framework — not \
as memorized answers, but as deeply held convictions that would naturally produce these \
and similar responses.

{question_answer}

----------------
Output requirements:
- Return a single valid JSON object.
- Use exactly this key: "soul_doc"
- The value should be the full soul document text (1500-2000 words, essay-style, second person).
- Use **bold section headers** to organize sections (not markdown ## headers).
- MUST include sections for: anti-patterns, internal diversity/fracture lines, and \
at least 3 counterfactual reasoning anchors.
- Do not include any additional text, explanation, markdown, or formatting outside the JSON.
"""

# ── Prompt: revision ──────────────────────────────────────────────────────

REVISION_PROMPT = """\
You are improving a soul document (a detailed persona system prompt) through \
iterative refinement.

The soul document is used as a system prompt for an AI model evaluating survey \
claims. Given a survey question and a claim about how a persona would respond, \
the model must judge whether it agrees or disagrees with the claim.

## Current Soul Document

{current_soul_doc}

## Current Performance

Accuracy on training data: {accuracy:.1%} ({correct}/{total} correct)

## Incorrect Predictions

Below are {n_wrong} examples where the model gave the WRONG answer. Each shows \
the question, the claim, what the model predicted, the correct label, and the \
model's reasoning trace.

{wrong_examples_text}

## Diagnosis Instructions

Before revising, perform a structured diagnosis:

1. **Pattern analysis**: Group the wrong examples by theme. What categories of \
questions are failing? (e.g., security/foreign policy, social values, economic \
policy, religious issues)

2. **Reasoning trace analysis**: Look at the model's reasoning for wrong answers. \
Where exactly does the reasoning go wrong? Common failure modes:
   - The persona defaults to a Western liberal position when the actual persona \
would reason differently
   - The persona over-generalizes and misses nuance specific to {persona_name}
   - The persona lacks a clear value anchor for a domain and falls back to \
generic reasoning
   - The persona has the right value but applies it in the wrong direction \
(e.g., values "social harmony" but applies it as tolerance when it should mean \
conformity pressure, or vice versa)

3. **Counterfactual test**: For each error pattern, ask: "What value or reasoning \
anchor, if added to the soul document, would have caused the model to reason \
correctly on THIS type of question AND on similar unseen questions?"

4. **Anti-pattern check**: Are any errors caused by the soul document containing \
misleading guidance? Sometimes the fix is to REMOVE or QUALIFY something, not \
add something new.

## Revision Rules

CRITICAL:
- NEVER reference specific survey questions, events, policies, or examples from the \
wrong predictions. The soul document must express general values and reasoning patterns.
- Express broad principles (e.g., "You believe military force should be multilateral \
and proportionate") NOT specific stances on specific events.
- The revised document should read as a timeless identity document.
- Keep it concise: 1500-2000 words. Remove filler and redundancy from the current \
document. Every sentence must earn its place.

Revision guidelines:
- Add or strengthen counterfactual reasoning anchors for the value domains that \
caused errors (e.g., "If someone argues X, you would reason Y because...")
- Update anti-patterns if the errors reveal new failure modes to guard against
- Refine internal diversity descriptions if errors stem from over-flattening \
the persona's views
- Strengthen emotional grounding if the persona is reasoning too abstractly
- Maintain the same style: second-person essay format ("You are...", "You believe..."), \
**bold section headers**
- Stay true to the persona ({persona_name}) — refine their general worldview
- The revised document must open with "You are..." establishing the persona identity

Output requirements:
- Return a single valid JSON object.
- Use exactly this key: "soul_doc"
- The value should be the full revised soul document text (second-person, essay-style, \
1500-2000 words).
- MUST include sections for: anti-patterns, internal diversity/fracture lines, and \
counterfactual reasoning anchors.
- Do not include any additional text outside the JSON.
"""

# ── Data helpers ──────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_data_path(task: str, persona: str, split: str) -> Path:
    if task == "opinionqa":
        return ROOT / task / "opinionqa_data" / split / f"opinionqa_{persona.lower()}.jsonl"
    return ROOT / task / "goqa_data" / split / f"globaloqa_{persona.lower()}.jsonl"


def get_qa_path(task: str) -> Path:
    if task == "opinionqa":
        return ROOT / task / "icm_based" / "questions.jsonl"
    return ROOT / task / "value_based" / "questions.jsonl"


def build_qa_string(questions: list[dict], task: str, persona: str) -> str:
    """Build Q&A string from the 10 reference questions for the persona."""
    if task == "opinionqa":
        key = f"{persona.lower()}_answer"
    else:
        key = f"{persona.lower()}_response"
    parts = []
    for i, row in enumerate(questions, start=1):
        q = row.get("question", "")
        a = row.get(key, "")
        parts.append(f"Question {i}: {q}\nAnswer {i}: {a}")
    return "\n\n".join(parts)


def get_persona_info(task: str, persona: str) -> tuple[str, str]:
    """Return (persona_name, persona_description) for prompt formatting."""
    if task == "opinionqa":
        name = persona.capitalize()
        return name, f"{name}s in the United States"
    return persona, f"the people of {persona}"


# ── Evaluation helpers ────────────────────────────────────────────────────


def build_user_prompt(question: str, claim: str) -> str:
    """User prompt: question + claim only (no True/False line; model returns JSON with judgement + reasoning)."""
    return (
        f"Survey Question: {question}\n\n"
        f"Claim: {claim}"
    )


def parse_judgement_reasoning(text: str) -> tuple[bool | None, str]:
    """Parse JSON response with 'judgement' (agree/disagree) and 'reasoning'. Returns (agree_as_bool, reasoning_str)."""
    text = (text or "").strip()
    # Strip markdown code block if present
    if "```" in text:
        start = text.find("```")
        if start != -1:
            rest = text[start + 3:]
            if rest.startswith("json"):
                rest = rest[4:].lstrip()
            end = rest.find("```")
            if end != -1:
                text = rest[:end].strip()
            else:
                text = rest
    try:
        data = json.loads(text)
        judgement = (data.get("judgement") or "").strip().lower()
        reasoning = (data.get("reasoning") or "").strip()
        if judgement == "agree":
            return True, reasoning
        if judgement == "disagree":
            return False, reasoning
    except (json.JSONDecodeError, TypeError):
        pass
    return None, ""


async def evaluate_records(
    records: list[dict],
    system_prompt: str,
    model: str,
    client: AsyncOpenAI,
    max_concurrent: int = 50,
    max_retries: int = 3,
    timeout: float = 150.0,
    retry_delay: float = 0.5,
) -> list[dict]:
    """Evaluate records and add 'prediction' (bool) to each record."""
    import copy
    records = copy.deepcopy(records)
    n = len(records)
    semaphore = asyncio.Semaphore(max_concurrent)
    results: dict[int, tuple[bool | None, str]] = {}
    pbar = tqdm(total=n, desc="  Eval", file=sys.stderr)

    async def _call(idx: int):
        r = records[idx]
        claim = r.get("choice_agree") or r.get("choice", "")
        user_prompt = build_user_prompt(r["question"], claim)
        value, reasoning = None, ""
        last_exception = None
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                        ),
                        timeout=timeout,
                    )
                    content = (response.choices[0].message.content or "").strip()
                    value, reasoning = parse_judgement_reasoning(content)
                if value is not None:
                    break
                last_exception = ValueError("Empty or invalid response (judgement missing)")
            except asyncio.TimeoutError:
                last_exception = TimeoutError(f"Request timed out after {timeout}s")
            except Exception as e:
                last_exception = e
            if attempt < max_retries - 1:
                print(f"    Record {idx} failed (attempt {attempt + 1}): {last_exception}",
                      file=sys.stderr)
                await asyncio.sleep(retry_delay)
        if value is None and last_exception is not None:
            print(f"    Record {idx} failed permanently: {last_exception}",
                  file=sys.stderr)
        results[idx] = (value, reasoning)
        pbar.update(1)

    await asyncio.gather(*[_call(i) for i in range(n)])

    pbar.close()
    for i in range(n):
        value, reasoning = results.get(i, (None, ""))
        records[i]["prediction"] = value
        records[i]["prediction_reasoning"] = reasoning
    return records


def compute_accuracy(records: list[dict]) -> float:
    correct = sum(1 for r in records if r.get("prediction") == bool(r["label"]))
    return correct / len(records) if records else 0.0


def collect_wrong_examples(
    records: list[dict],
    max_examples: int = 30,
    seed: int = 42,
) -> list[dict]:
    wrong = [r for r in records if r.get("prediction") != bool(r["label"])]
    if len(wrong) > max_examples:
        rng = random.Random(seed)
        wrong = rng.sample(wrong, max_examples)
    return wrong


def format_wrong_examples(wrong: list[dict]) -> str:
    parts = []
    for i, r in enumerate(wrong, 1):
        predicted = "agree" if r["prediction"] else "disagree"
        correct = "agree" if r["label"] else "disagree"
        claim = r.get("choice_agree") or r.get("choice", "")
        reasoning = r.get("prediction_reasoning", "").strip()
        entry = (
            f"Example {i}:\n"
            f"  Question: {r['question']}\n"
            f"  Claim: {claim}\n"
            f"  Model predicted: {predicted}\n"
            f"  Correct answer: {correct}"
        )
        if reasoning:
            entry += f"\n  Model's reasoning: {reasoning}"
        parts.append(entry)
    return "\n\n".join(parts)


# ── Soul doc generation / revision ────────────────────────────────────────


def parse_json_response(raw: str) -> dict:
    """Parse JSON from model response, handling fences, think tags, and other wrapping."""
    import re

    text = raw.strip()

    # Strip <think>...</think> blocks (DeepSeek R1 reasoning)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find a JSON object anywhere in the text
    # Find the first { and last } to extract the JSON object
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON from model response. "
        f"First 500 chars:\n{raw[:500]}"
    )


def _call_with_json_fallback(
    client: OpenAI,
    model: str,
    messages: list[dict],
) -> str:
    """Call the model requesting JSON, with fallback to plain text."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    return (resp.choices[0].message.content or "").strip()


def _extract_soul_doc(client: OpenAI, model: str, messages: list[dict], max_retries: int = 3) -> str:
    """Call model and extract soul_doc from JSON response, with retries."""
    last_err = None
    for attempt in range(max_retries):
        raw = _call_with_json_fallback(client, model, messages)
        try:
            parsed = parse_json_response(raw)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            print(f"    JSON parse failed (attempt {attempt + 1}/{max_retries}): {e}",
                  file=sys.stderr)
            continue
        if "soul_doc" not in parsed:
            # Try case-insensitive / flexible key matching
            for k, v in parsed.items():
                if "soul" in k.lower() and isinstance(v, str):
                    return v
            last_err = ValueError(f"Missing 'soul_doc' key. Got: {list(parsed.keys())}")
            print(f"    Bad keys (attempt {attempt + 1}/{max_retries}): {list(parsed.keys())}",
                  file=sys.stderr)
            continue
        return parsed["soul_doc"]
    raise RuntimeError(f"Failed to get valid soul_doc after {max_retries} attempts: {last_err}")


def generate_initial_soul_doc(
    client: OpenAI,
    task: str,
    persona: str,
    qa_string: str,
    model: str,
) -> str:
    persona_name, persona_desc = get_persona_info(task, persona)
    prompt = INITIAL_GENERATION_PROMPT.format(
        persona_name=persona_name,
        persona_description=persona_desc,
        question_answer=qa_string,
    )
    return _extract_soul_doc(client, model, [{"role": "user", "content": prompt}])


def revise_soul_doc(
    client: OpenAI,
    current_doc: str,
    wrong_examples: list[dict],
    train_records: list[dict],
    persona: str,
    task: str,
    model: str,
) -> str:
    total = len(train_records)
    correct = sum(1 for r in train_records if r.get("prediction") == bool(r["label"]))
    accuracy = correct / total if total else 0.0

    wrong_text = format_wrong_examples(wrong_examples)
    persona_name, _ = get_persona_info(task, persona)

    prompt = REVISION_PROMPT.format(
        current_soul_doc=current_doc,
        accuracy=accuracy,
        correct=correct,
        total=total,
        n_wrong=len(wrong_examples),
        wrong_examples_text=wrong_text,
        persona_name=persona_name,
    )
    return _extract_soul_doc(client, model, [{"role": "user", "content": prompt}])


# ── Plotting ──────────────────────────────────────────────────────────────


def plot_accuracy(
    train_accs: list[float],
    test_accs: list[float],
    task: str,
    persona: str,
    eval_model: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    iters = list(range(len(test_accs)))
    ax.plot(iters, test_accs, "o-", label="Test accuracy", color="tab:blue", linewidth=2)
    ax.plot(iters, train_accs, "s--", label="Train accuracy", color="tab:orange", linewidth=2)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)

    persona_name, _ = get_persona_info(task, persona)
    model_short = eval_model.split("/")[-1]
    ax.set_title(
        f"Iterative Soul Doc Revision — {persona_name}\n(eval: {model_short})",
        fontsize=13,
    )
    ax.set_xticks(iters)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Iterative soul document revision with train-set error feedback.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", required=True, choices=["opinionqa", "globaloqa"])
    p.add_argument("--persona", required=True,
                    help="democrat/republican (opinionqa) or country name (globaloqa)")
    p.add_argument("--eval-model", required=True,
                    help="OpenRouter model for evaluation (e.g. deepseek/deepseek-r1-0528)")
    p.add_argument("--revision-model", default="anthropic/claude-sonnet-4-5-20250929",
                    help="OpenRouter model for soul doc generation/revision")
    p.add_argument("--iterations", type=int, default=3,
                    help="Number of revision rounds (default: 3)")
    p.add_argument("--max-concurrent", type=int, default=50,
                    help="Max concurrent eval API calls (default: 50)")
    p.add_argument("--max-wrong-examples", type=int, default=30,
                    help="Max wrong examples in revision prompt (default: 30)")
    p.add_argument("--initial-soul-doc", type=Path, default=None,
                    help="Path to a .txt file with an initial soul doc (skip generation)")
    p.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: <task>/results/iterative_revision/...)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for wrong-example sampling (default: 42)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────


async def async_main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set in .env")

    # Validate persona
    if args.task == "opinionqa":
        if args.persona.lower() not in ("democrat", "republican"):
            raise SystemExit("--persona for opinionqa must be 'democrat' or 'republican'")
        persona = args.persona.lower()
    else:
        if args.persona not in GLOBALOQA_COUNTRIES:
            raise SystemExit(
                f"--persona for globaloqa must be one of: {', '.join(GLOBALOQA_COUNTRIES)}"
            )
        persona = args.persona

    # Output directory
    if args.out_dir is None:
        eval_slug = args.eval_model.replace("/", "_")
        rev_slug = args.revision_model.replace("/", "_")
        out_dir = (
            ROOT / args.task / "results" / "iterative_revision"
            / f"{persona.lower()}_eval-{eval_slug}_rev-{rev_slug}"
        )
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clients
    sync_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    async_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Load data
    train_path = get_data_path(args.task, persona, "train")
    test_path = get_data_path(args.task, persona, "test")
    if not train_path.exists():
        raise SystemExit(f"Train data not found: {train_path}")
    if not test_path.exists():
        raise SystemExit(f"Test data not found: {test_path}")
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)
    print(f"Data: {len(train_data)} train, {len(test_data)} test records", file=sys.stderr)

    # ── Step 0: initial soul doc ──────────────────────────────────────────
    if args.initial_soul_doc is not None:
        print(f"\nLoading initial soul doc from {args.initial_soul_doc}", file=sys.stderr)
        soul_doc = args.initial_soul_doc.read_text(encoding="utf-8")
    else:
        print("\n=== Generating initial soul document ===", file=sys.stderr)
        qa_records = load_jsonl(get_qa_path(args.task))
        qa_string = build_qa_string(qa_records, args.task, persona)
        soul_doc = generate_initial_soul_doc(
            sync_client, args.task, persona, qa_string, args.revision_model,
        )
        print("  Done.", file=sys.stderr)

    # ── Iteration loop ────────────────────────────────────────────────────
    train_accs: list[float] = []
    test_accs: list[float] = []

    total_iterations = args.iterations + 1  # iteration 0 = initial, then N revisions
    for iteration in range(total_iterations):
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"  ITERATION {iteration}", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

        # Save current soul doc
        soul_path = out_dir / f"soul_doc_iter_{iteration}.txt"
        soul_path.write_text(soul_doc, encoding="utf-8")

        # Build system prompt
        system_prompt = SYSTEM_PROMPT_SOUL.format(soul_doc=soul_doc)

        # Evaluate on train set
        print(f"\n  Evaluating on TRAIN ({len(train_data)} records)...", file=sys.stderr)
        train_results = await evaluate_records(
            train_data, system_prompt, args.eval_model, async_client, args.max_concurrent,
        )
        train_acc = compute_accuracy(train_results)
        train_accs.append(train_acc)
        print(f"  Train accuracy: {train_acc:.3f}", file=sys.stderr)

        # Save train results
        train_out = out_dir / f"train_results_iter_{iteration}.jsonl"
        with train_out.open("w", encoding="utf-8") as f:
            for r in train_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Evaluate on test set
        print(f"\n  Evaluating on TEST ({len(test_data)} records)...", file=sys.stderr)
        test_results = await evaluate_records(
            test_data, system_prompt, args.eval_model, async_client, args.max_concurrent,
        )
        test_acc = compute_accuracy(test_results)
        test_accs.append(test_acc)
        print(f"  Test accuracy: {test_acc:.3f}", file=sys.stderr)

        # Save test results
        test_out = out_dir / f"test_results_iter_{iteration}.jsonl"
        with test_out.open("w", encoding="utf-8") as f:
            for r in test_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Revise soul doc for next iteration (skip after last)
        if iteration < args.iterations:
            wrong = collect_wrong_examples(
                train_results, args.max_wrong_examples, args.seed,
            )
            if not wrong:
                print("  No wrong examples — perfect train accuracy. Stopping early.",
                      file=sys.stderr)
                # Pad remaining iterations with current accuracy
                for _ in range(iteration + 1, total_iterations):
                    train_accs.append(train_acc)
                    test_accs.append(test_acc)
                break

            print(f"\n  Revising soul doc ({len(wrong)} wrong examples)...", file=sys.stderr)
            soul_doc = revise_soul_doc(
                sync_client, soul_doc, wrong, train_results,
                persona, args.task, args.revision_model,
            )
            print("  Revision complete.", file=sys.stderr)

    # ── Save summary ──────────────────────────────────────────────────────
    summary = {
        "task": args.task,
        "persona": persona,
        "eval_model": args.eval_model,
        "revision_model": args.revision_model,
        "iterations": args.iterations,
        "seed": args.seed,
        "max_wrong_examples": args.max_wrong_examples,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "train_accuracies": train_accs,
        "test_accuracies": test_accs,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}", file=sys.stderr)

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_accuracy(
        train_accs, test_accs,
        args.task, persona, args.eval_model,
        out_dir / "accuracy_plot.png",
    )

    # ── Print final table ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("  RESULTS SUMMARY", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  {'Iter':<6} {'Train':>8} {'Test':>8}", file=sys.stderr)
    print(f"  {'-'*6} {'-'*8} {'-'*8}", file=sys.stderr)
    for i, (tr, te) in enumerate(zip(train_accs, test_accs)):
        print(f"  {i:<6} {tr:>8.3f} {te:>8.3f}", file=sys.stderr)
    print(f"\n  Output directory: {out_dir}", file=sys.stderr)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
