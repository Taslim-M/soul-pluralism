"""
Generate soul documents for each country from questions.jsonl Q/A pairs,
then write them into souls.py as variables {country_lower}_values_1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from prompts import GENERATE_SOUL_DOC

COUNTRIES = [
    "Brazil", "Britain", "France", "Germany", "Indonesia", "Japan", "Jordan",
    "Lebanon", "Mexico", "Nigeria", "Pakistan", "Russia", "Turkey",
]

HERE = Path(__file__).resolve().parent
QUESTIONS_PATH = HERE / "questions.jsonl"
SOULS_PY_PATH = HERE / "souls.py"

SOUL_VAR_SUFFIX = "values_1"


def _load_openrouter_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in .env")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def load_questions(path: Path = QUESTIONS_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at line {i}")
            items.append(obj)
    return items


def build_question_answer_string(questions: List[Dict[str, Any]], country: str) -> str:
    """Build a single string of 10 Q/A pairs for the given country."""
    key = country.lower() + "_response"
    parts: List[str] = []
    for i, row in enumerate(questions, start=1):
        q = row.get("question", "")
        a = row.get(key, "")
        parts.append(f"Question {i}: {q}\nAnswer {i}: {a}")
    return "\n\n".join(parts)


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}\nRaw:\n{text}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}: {data!r}")
    return data


def call_generate_soul_doc(
    client: OpenAI,
    *,
    question_answer: str,
    country: str,
    model: str = "anthropic/claude-opus-4.6",
) -> str:
    """Call the model to generate one soul doc; return the soul doc string."""
    prompt = GENERATE_SOUL_DOC.format(
        country=country,
        question_answer=question_answer,
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
    # Strip markdown code fences if present (e.g. ```json\n...\n```)
    if raw.startswith("```"):
        lines = raw.split("\n")
        start = 1 if lines[0].strip() in ("```", "```json") else 0
        raw = "\n".join(lines[start:])
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    parsed = _safe_json_loads(raw)
    key = f"{country}_soul_doc"
    if key not in parsed or not isinstance(parsed[key], str):
        raise ValueError(f"Model response missing or invalid '{key}': {parsed!r}")
    return parsed[key]


def write_souls_py(soul_docs: Dict[str, str], path: Path = SOULS_PY_PATH) -> None:
    """Write souls.py with one variable per country: {country_lower}_values_1 = '''...'''."""
    lines = [
        '# Auto-generated soul documents (one per country). Do not edit by hand.',
        '',
    ]
    for country in COUNTRIES:
        var_name = f"{country.lower()}_{SOUL_VAR_SUFFIX}"
        content = soul_docs[country]
        # Use triple single quotes; if content contains ''' use triple double quotes with escape
        if "'''" in content:
            escaped = content.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
            lines.append(f'{var_name} = """')
            lines.append(escaped)
            lines.append('"""')
        else:
            lines.append(f"{var_name} = '''")
            lines.append(content)
            lines.append("'''")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> Dict[str, str]:
    client = _load_openrouter_client()
    questions = load_questions()
    if len(questions) != 10:
        raise ValueError(f"Expected 10 question rows in {QUESTIONS_PATH}, got {len(questions)}")

    soul_docs: Dict[str, str] = {}
    for country in COUNTRIES:
        qa_string = build_question_answer_string(questions, country)
        doc = call_generate_soul_doc(client, question_answer=qa_string, country=country)
        soul_docs[country] = doc
        print(f"Generated soul doc for {country}")

    write_souls_py(soul_docs)
    print(f"Wrote {SOULS_PY_PATH}")
    return soul_docs


if __name__ == "__main__":
    main()
