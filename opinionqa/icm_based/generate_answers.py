from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from icm_based.prompts import GENERATE_ANSWERS


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
    client: OpenAI, *, question: str, model: str = "anthropic/claude-opus-4.6"
) -> Dict[str, str]:
    """Return parsed JSON with keys democrat_answer, republican_answer."""
    prompt = GENERATE_ANSWERS.format(question=question)

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
    for key in ("democrat_answer", "republican_answer"):
        if key not in parsed or not isinstance(parsed[key], str):
            raise ValueError(f"Model response missing or invalid '{key}': {parsed!r}")
    return parsed


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


OUTPUT_KEYS = ("question_id", "question", "democrat_answer", "republican_answer")


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

        answers = _call_generate_answers(client, question=question_text)

        out = {
            "question_id": obj["question_id"],
            "question": obj["question"],
            "democrat_answer": answers["democrat_answer"],
            "republican_answer": answers["republican_answer"],
        }
        updated.append(out)

    save_questions(updated)
    return updated


if __name__ == "__main__":
    main()

