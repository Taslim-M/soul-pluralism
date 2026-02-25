from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

# Static prompts that require .format(persona=...) or (country=... / political_party=...)
PROMPTS_REQUIRING_PERSONA = (
    "system_prompt_base_persona_country",
    "system_prompt_base_persona_political",
)


def _load_task_module(task: str, subpath: str):
    """Load a module from task folder (e.g. opinionqa/eval_baseline_prompts.py)."""
    task_dir = ROOT / task
    if not task_dir.is_dir():
        raise ValueError(f"Task directory not found: {task_dir}")
    module_path = task_dir / subpath
    if not module_path.exists():
        raise FileNotFoundError(f"Module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(
        f"{task}.{subpath.replace('/', '.').replace('.py', '')}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_soul_doc(soul_var_name: str, souls_module) -> str:
    """Resolve soul doc by variable name from the task's souls module."""
    if not hasattr(souls_module, soul_var_name):
        attrs = [x for x in dir(souls_module) if not x.startswith("_")]
        raise ValueError(
            f"Unknown soul variable {soul_var_name!r}. Available: {attrs}"
        )
    doc = getattr(souls_module, soul_var_name)
    if not isinstance(doc, str):
        raise TypeError(f"{soul_var_name} is not a string.")
    return doc


def get_static_prompt(prompt_name: str, baseline_prompts) -> str:
    """Resolve static prompt by variable name from the task's eval_baseline_prompts."""
    if not hasattr(baseline_prompts, prompt_name):
        attrs = [x for x in dir(baseline_prompts) if not x.startswith("_")]
        raise ValueError(
            f"Unknown static prompt {prompt_name!r}. Available: {attrs}"
        )
    doc = getattr(baseline_prompts, prompt_name)
    if not isinstance(doc, str):
        raise TypeError(f"{prompt_name} is not a string.")
    return doc


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL into a list of dicts."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


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


def parse_true_false(text: str) -> bool:
    text = (text or "").strip().lower()
    last_true = text.rfind("true")
    last_false = text.rfind("false")
    if last_true > last_false:
        return True
    if last_false > last_true:
        return False
    return "true" in text


# GlobalOQA country list (for --persona choices when task=globaloqa)
GLOBALOQA_COUNTRIES = [
    "Brazil", "Britain", "France", "Germany", "Indonesia", "Japan", "Jordan",
    "Lebanon", "Mexico", "Nigeria", "Pakistan", "Russia", "Turkey",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpinionQA or GlobalOQA evaluation via OpenRouter.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["opinionqa", "globaloqa"],
        help="Task: opinionqa (political personas) or globaloqa (country personas).",
    )
    parser.add_argument(
        "--persona",
        type=str,
        required=True,
        help="Persona: for opinionqa use 'democrat' or 'republican'; for globaloqa use a country name (e.g. Britain, Germany).",
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
        help="Soul doc variable name from the task's souls module (e.g. democrat_values_1, germany_values_1).",
    )
    mode.add_argument(
        "--static",
        type=str,
        metavar="PROMPT_NAME",
        help="Static prompt variable name from the task's eval_baseline_prompts (e.g. system_prompt_base_persona_country, system_prompt_base_persona_political).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path. Default: <task>/results/eval_results_<tag>_<model_slug>_<persona>.jsonl",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Max concurrent API calls (default 50).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=150.0,
        metavar="SECONDS",
        help="Timeout in seconds per API call (default 120). Stuck calls are cancelled and counted as failed.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=100,
        metavar="N",
        help="Max retries per record after failure or empty response (default 2; each record tried up to N+1 times).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="Seconds to wait before retrying a failed request (default 2).",
    )
    parser.add_argument(
        "--first_person",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, use first-person prompt (soul in dialogue, no system prompt). Requires --soul.",
    )
    args = parser.parse_args()

    if args.first_person and not args.soul:
        parser.error("--first_person 1 requires --soul.")

    # Validate persona per task
    if args.task == "opinionqa":
        if args.persona.lower() not in ("democrat", "republican"):
            parser.error("--persona for opinionqa must be 'democrat' or 'republican'")
    else:
        if args.persona not in GLOBALOQA_COUNTRIES:
            parser.error(
                f"--persona for globaloqa must be one of: {', '.join(GLOBALOQA_COUNTRIES)}"
            )

    # Default output path: <task>/results/eval_results_<tag>_<model_slug>_<persona>.jsonl
    if args.out is None:
        task_dir = ROOT / args.task
        model_slug = args.model.replace("/", "_")
        if args.soul:
            tag = args.soul
        else:
            tag = args.static.replace("system_prompt_", "")
        persona_slug = args.persona.lower() if args.task == "globaloqa" else args.persona
        args.out = task_dir / "results" / f"eval_results_{tag}_{model_slug}_{persona_slug}.jsonl"
    else:
        args.out = Path(args.out)
        if not args.out.is_absolute():
            args.out = ROOT / args.out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    return args


def get_system_prompt(
    args: argparse.Namespace,
    baseline_prompts,
    souls_module,
) -> str:
    """Build system prompt from soul or static baseline, with optional persona formatting."""
    if args.soul:
        soul_doc = get_soul_doc(args.soul, souls_module)
        return baseline_prompts.system_prompt_soul.format(soul_doc=soul_doc)

    base = get_static_prompt(args.static, baseline_prompts)

    if args.static in PROMPTS_REQUIRING_PERSONA:
        if args.static == "system_prompt_base_persona_country":
            return base.format(country=args.persona)
        if args.static == "system_prompt_base_persona_political":
            political_party = args.persona.capitalize()  # democrat -> Democrat
            return base.format(political_party=political_party)
    return base


def get_data_path(args: argparse.Namespace) -> Path:
    """Path to the task's JSONL dataset for the chosen persona (data/test/)."""
    task_dir = ROOT / args.task
    if args.task == "opinionqa":
        return task_dir / "opinionqa_data" / "test" / f"opinionqa_{args.persona}.jsonl"
    return task_dir / "goqa_data" / "test" / f"globaloqa_{args.persona.lower()}.jsonl"


async def run_all_tasks(
    records: list[dict],
    system_prompt: str | None,
    model_name: str,
    client: AsyncOpenAI,
    max_concurrent: int,
    timeout: float,
    max_retries: int,
    retry_delay: float,
    *,
    first_person: bool = False,
    build_first_person_fn=None,
    soul_doc: str | None = None,
) -> list[dict]:
    n = len(records)
    semaphore = asyncio.Semaphore(max_concurrent)
    results: dict[int, tuple[bool | None, str]] = {i: (None, "") for i in range(n)}

    async def generate_response(
        model: str,
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        messages: list[dict] | None = None,
    ) -> tuple[bool | None, str]:
        if messages is not None:
            api_messages = messages
        else:
            api_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = await client.chat.completions.create(
            model=model,
            messages=api_messages,
        )
        content = (response.choices[0].message.content or "").strip()
        return parse_judgement_reasoning(content)

    pbar = tqdm(total=n, desc="Records")
    pbar_lock = asyncio.Lock()

    async def task(idx: int):
        value, reasoning = None, ""
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                async with semaphore:
                    r = records[idx]
                    claim = r.get("choice_agree") or r.get("choice", "")
                    if first_person and build_first_person_fn is not None and soul_doc is not None:
                        messages = build_first_person_fn(soul_doc, r["question"], claim)
                        value, reasoning = await asyncio.wait_for(
                            generate_response(model_name, messages=messages),
                            timeout=timeout,
                        )
                    else:
                        user_prompt = build_user_prompt(r["question"], claim)
                        value, reasoning = await asyncio.wait_for(
                            generate_response(model_name, system_prompt=system_prompt, user_prompt=user_prompt),
                            timeout=timeout,
                        )
                if value is not None:
                    break
                last_exception = ValueError("Empty or invalid response (judgement missing)")
            except asyncio.TimeoutError:
                last_exception = TimeoutError(f"Request timed out after {timeout}s")
            except Exception as e:
                last_exception = e
            if attempt < max_retries:
                await asyncio.sleep(retry_delay)
        try:
            if value is None and last_exception is not None:
                raise last_exception
            return idx, value, reasoning
        finally:
            async with pbar_lock:
                pbar.update(1)

    out = await asyncio.gather(*[task(i) for i in range(n)], return_exceptions=True)

    for idx, result in enumerate(out):
        if isinstance(result, Exception):
            print(f"Record {idx} failed: {result}", file=sys.stderr)
            continue
        _, value, reasoning = result
        results[idx] = (value, reasoning)

    pbar.close()
    for i in range(n):
        value, reasoning = results[i]
        records[i][model_name] = value
        records[i][model_name + "_reasoning"] = reasoning
    return records


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set in .env")

    # Load task-specific modules (from <task>/ folder)
    task_dir = ROOT / args.task
    if str(task_dir) not in sys.path:
        sys.path.insert(0, str(task_dir))

    baseline_prompts = _load_task_module(args.task, "eval_baseline_prompts.py")
    if args.task == "opinionqa":
        souls_module = _load_task_module(args.task, "icm_based/souls.py")
    else:
        souls_module = _load_task_module(args.task, "value_based/souls.py")

    first_person = args.first_person == 1
    if first_person:
        builder_name = "opinionqa_build_user_prompt_first_person" if args.task == "opinionqa" else "globalqa_build_user_prompt_first_person"
        build_first_person_fn = getattr(baseline_prompts, builder_name)
        soul_doc = get_soul_doc(args.soul, souls_module)
        system_prompt = None
    else:
        build_first_person_fn = None
        soul_doc = None
        system_prompt = get_system_prompt(args, baseline_prompts, souls_module)

    data_path = get_data_path(args)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")

    records = load_jsonl(data_path)
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
            args.timeout,
            args.max_retries,
            args.retry_delay,
            first_person=first_person,
            build_first_person_fn=build_first_person_fn,
            soul_doc=soul_doc,
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    vals = [r.get(args.model) for r in records if r.get(args.model) is not None]
    print(f"Saved {len(records)} records to {args.out}", file=sys.stderr)
    print(f"Summary: Agree(True)={sum(vals)}, Disagree(False)={len(vals) - sum(vals)}", file=sys.stderr)


if __name__ == "__main__":
    main()
