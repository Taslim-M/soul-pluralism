from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

# Project root (script in repo root).
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import eval_baseline_prompts as baseline_prompts
from eval_baseline_prompts import system_prompt_soul


def get_soul_doc(soul_var_name: str) -> str:

    """Load soul doc from file."""
    if os.path.exists(soul_var_name):
        with open(soul_var_name, "r") as f:
            return f.read()

    """Otherwise, resolve soul doc by variable name from icm_based.souls (e.g. democrat_values_1)."""
    import icm_based.souls as souls_module
    if not hasattr(souls_module, soul_var_name):
        raise ValueError(
            f"Unknown soul variable {soul_var_name!r}. "
            f"Available: {[x for x in dir(souls_module) if not x.startswith('_')]}"
        )
    doc = getattr(souls_module, soul_var_name)
    if not isinstance(doc, str):
        raise TypeError(f"{soul_var_name} is not a string.")
    return doc


def get_static_prompt(prompt_name: str) -> str:
    """Resolve static prompt by variable name from eval_baseline_prompts."""
    if not hasattr(baseline_prompts, prompt_name):
        available = [x for x in dir(baseline_prompts) if not x.startswith("_")]
        raise ValueError(
            f"Unknown static prompt {prompt_name!r}. Available: {available}"
        )
    doc = getattr(baseline_prompts, prompt_name)
    if not isinstance(doc, str):
        raise TypeError(f"{prompt_name} is not a string.")
    return doc


def load_opinionqa(path: Path) -> list[dict]:
    """Load JSONL into a list of dicts."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_user_prompt(question: str, choice: str) -> str:
    return (
        f"Survey Question: {question}\n\n"
        f"Claim: {choice}\n\n"
        "I think this claim is [True/False]"
    )


def parse_true_false(text: str) -> bool:
    text = (text or "").strip().lower()
    last_true = text.rfind("true")
    last_false = text.rfind("false")
    if last_true > last_false:
        return True
    if last_false > last_true:
        return False
    return "true" in text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run opinion QA evaluation via OpenRouter (same as icm_based/eval.ipynb)."
    )
    parser.add_argument(
        "--persona",
        type=str,
        required=True,
        choices=["democrat", "republican"],
        help="Persona / dataset: opinionqa_{persona}.jsonl",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="OpenRouter model id (e.g. anthropic/claude-opus-4.5).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--soul",
        type=str,
        metavar="SOUL_VAR",
        help="Soul doc variable name from icm_based.souls (e.g. democrat_values_1).",
    )
    mode.add_argument(
        "--static",
        type=str,
        metavar="PROMPT_NAME",
        help="Static prompt variable name from eval_baseline_prompts (e.g. system_prompt_base_persona_democrat).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path. Default: results/eval_results_<mode>_<model_slug>_<persona>.jsonl",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Max concurrent API calls (default 50).",
    )
    args = parser.parse_args()

    # Default outpath: results dir, descriptive filename.
    if args.out is None:
        model_slug = args.model.replace("/", "_")
        if args.soul:
            tag = args.soul
        else:
            tag = args.static.replace("system_prompt_", "")
        args.out = ROOT / "results" / f"eval_results_{tag}_{model_slug}_{args.persona}.jsonl"
    else:
        args.out = Path(args.out)
        if not args.out.is_absolute():
            args.out = ROOT / args.out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    return args


def get_system_prompt(args: argparse.Namespace) -> str:
    if args.soul:
        soul_doc = get_soul_doc(args.soul)
        return system_prompt_soul.format(soul_doc=soul_doc)
    return get_static_prompt(args.static)


async def run_all_tasks(
    records: list[dict],
    system_prompt: str,
    model_name: str,
    client: AsyncOpenAI,
    max_concurrent: int,
) -> list[dict]:
    n = len(records)
    semaphore = asyncio.Semaphore(max_concurrent)
    results: dict[int, bool] = {i: None for i in range(n)}
    pending = set(range(n))

    async def generate_response(system_prompt: str, user_prompt: str, model: str) -> bool:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        return parse_true_false(content)

    pbar = tqdm(total=n, desc="Records")

    while pending:
        pending_list = sorted(pending)

        async def task(idx: int):
            async with semaphore:
                r = records[idx]
                user_prompt = build_user_prompt(r["question"], r["choice"])
                value = await generate_response(system_prompt, user_prompt, model_name)
            return idx, value

        tasks = [task(i) for i in pending_list]
        out = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in zip(pending_list, out):
            if isinstance(result, Exception):
                print(f"Record {idx} failed: {result}", file=sys.stderr)
                continue
            _, value = result
            results[idx] = value
            pending.discard(idx)
            pbar.update(1)

    pbar.close()
    for i in range(n):
        records[i][model_name] = results[i]
    return records


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set in .env")

    system_prompt = get_system_prompt(args)
    data_path = ROOT / "opinionqa_data" / f"opinionqa_{args.persona}.jsonl"
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")

    records = load_opinionqa(data_path)
    print(f"Loaded {len(records)} records from {data_path}", file=sys.stderr)

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    records = asyncio.run(
        run_all_tasks(
            records,
            system_prompt,
            args.model,
            client,
            args.max_concurrent,
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    vals = [r.get(args.model) for r in records if r.get(args.model) is not None]
    print(f"Saved {len(records)} records to {args.out}", file=sys.stderr)
    print(f"Summary: True={sum(vals)}, False={len(vals) - sum(vals)}", file=sys.stderr)


if __name__ == "__main__":
    main()
