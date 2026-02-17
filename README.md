# soul-pluralism

## Running the evaluation script (`eval.py`)

Unified script for **OpinionQA** and **GlobalOQA**. It loads test data from each task’s `data/test` folder, calls OpenRouter with a system prompt (soul doc or static baseline), and writes per-record True/False results to a JSONL file. Output goes to the task’s `results/` folder by default.

Run from the **project root**.

### Prerequisites

- Python 3.11+
- Dependencies: `pip install openai python-dotenv tqdm`
- A `.env` file in the project root with `OPENROUTER_API_KEY=...`

### Usage

**Required arguments:**

- `--task {opinionqa|globaloqa}` — Which task (and thus which data and prompts) to use.
- `--persona PERSONA` — For **opinionqa**: `democrat` or `republican`. For **globaloqa**: a country name (e.g. `Britain`, `Germany`).
- `--model MODEL` — OpenRouter model id (e.g. `anthropic/claude-opus-4.5`, `deepseek/deepseek-r1-0528`).
- Either:
  - `--soul SOUL_VAR` — Soul doc variable from the task’s souls module (e.g. `democrat_values_1`, `germany_values_1`).
  - `--static PROMPT_NAME` — Static prompt from the task’s `eval_baseline_prompts.py` (e.g. `system_prompt_base_persona_political`, `system_prompt_base_persona_country`).

**Optional:**

- `--out PATH` — Output JSONL path. Default: `<task>/results/eval_results_<tag>_<model_slug>_<persona>.jsonl`.
- `--max-concurrent N` — Max concurrent API calls (default: 50).

**Data paths (test set):**

- OpinionQA: `opinionqa/opinionqa_data/test/opinionqa_{persona}.jsonl`
- GlobalOQA: `globaloqa/goqa_data/test/globaloqa_{persona}.jsonl`

### Examples

```bash

# OpinionQA — soul, explicit output path (full path)
python eval.py --task opinionqa --persona republican --model anthropic/claude-opus-4.5 --soul republican_values_1 --out opinionqa/results/eval_results_soul_republican_values_1_opus45_republican.jsonl

# OpinionQA — fixed static prompt
python eval.py --task opinionqa --persona democrat --model openai/gpt-oss-120b --static system_prompt_base_persona_political --out opinionqa/results/eval_results_base_static_oss120b_democrat.jsonl

# GlobalOQA — static base persona (country)
python eval.py --task globaloqa --persona Britain --model deepseek/deepseek-r1-0528 --static system_prompt_base_persona_country --out globaloqa/results/eval_results_base_persona_country_deepseekr1_britain.jsonl

# GlobalOQA — soul
python eval.py --task globaloqa --persona Germany --model deepseek/deepseek-r1-0528 --soul germany_values_1 --out globaloqa/results/eval_results_germany_values_1_deepseekr1_germany.jsonl
```

For more options: `python eval.py --help`.