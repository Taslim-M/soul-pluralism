from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from prompts import GENERATE_ANSWERS

COUNTRIES = [
    "Brazil", "Britain", "France", "Germany", "Indonesia", "Japan", "Jordan",
    "Lebanon", "Mexico", "Nigeria", "Pakistan", "Russia", "Turkey",
]

HERE = Path(__file__).resolve().parent
QUESTIONS_PATH = HERE / "questions.jsonl"


def _load_openrouter_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in .env")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Parse a model response expected to be a single JSON object."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}\nRaw:\n{text}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}: {data!r}")
    return data


def _call_generate_answers(
    client: OpenAI,
    *,
    question: str,
    country: str,
    model: str = "anthropic/claude-opus-4.6",
) -> str:
    """Call the model for one question and one country; return the answer string."""
    country_lower = country.lower()
    prompt = GENERATE_ANSWERS.format(
        question=question,
        country=country,
        country_lower=country_lower,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

    raw = (resp.choices[0].message.content or "").strip()
    parsed = _safe_json_loads(raw)
    # Model returns a single key like "germany_answer"
    answer_key = f"{country_lower}_answer"
    if answer_key not in parsed or not isinstance(parsed[answer_key], str):
        raise ValueError(f"Model response missing or invalid '{answer_key}': {parsed!r}")
    return parsed[answer_key]


def load_questions(path: Path = QUESTIONS_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {i}: {e}\nLine:\n{line}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at line {i}, got {type(obj).__name__}")
            items.append(obj)
    return items


RESPONSE_KEYS = [f"{c.lower()}_response" for c in COUNTRIES]
OUTPUT_KEYS = ("question_id", "question") + tuple(RESPONSE_KEYS)


def save_questions(items: List[Dict[str, Any]], path: Path = QUESTIONS_PATH) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for obj in items:
            row = {k: obj[k] for k in OUTPUT_KEYS if k in obj}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> List[Dict[str, Any]]:
    client = _load_openrouter_client()
    questions = load_questions()

    updated: List[Dict[str, Any]] = []
    for obj in questions:
        question_text = obj.get("question")
        if not isinstance(question_text, str) or not question_text.strip():
            raise ValueError(f"Missing/invalid 'question' field in: {obj!r}")

        out = {
            "question_id": obj["question_id"],
            "question": obj["question"],
        }
        for country in COUNTRIES:
            answer = _call_generate_answers(client, question=question_text, country=country)
            out[f"{country.lower()}_response"] = answer
        updated.append(out)

    save_questions(updated)
    return updated


if __name__ == "__main__":
    main()

