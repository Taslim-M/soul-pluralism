# soul-pluralism

## Running the evaluation script (`eval.py`)

The script loads OpinionQA data, calls OpenRouter with a system prompt (soul doc or static baseline), and writes per-record True/False results to a JSONL file in `results/`.

### Prerequisites

- Python 3.11+
- Dependencies: `pip install openai python-dotenv tqdm`
- A `.env` file in the project root with `OPENROUTER_API_KEY=...`

### Usage

**Required arguments:**

- `--persona {democrat|republican}` — Which dataset to use: `opinionqa_data/opinionqa_{persona}.jsonl`
- `--model MODEL` — OpenRouter model id (e.g. `anthropic/claude-opus-4.5`)
- `--out PATH` — Output JSONL path (default: `results/eval_results_<tag>_<model_slug>_<persona>.jsonl`)
- Either:
  - `--soul SOUL_VAR` — Soul doc variable name from `icm_based/souls.py` (e.g. `democrat_values_1`, `republican_values_1`)
  - `--static PROMPT_NAME` — Static prompt variable name from `eval_baseline_prompts.py` (e.g. `system_prompt_base_persona_democrat`, `system_prompt_base_persona_republican`)

**Optional:**
- `--max-concurrent N` — Max concurrent API calls (default: 50)

### Examples

```bash
# Soul mode: use a soul doc from souls.py
python eval.py --persona democrat --model anthropic/claude-opus-4.5 --soul democrat_values_1 --out results/my_eval.jsonl

# Static baseline: use a fixed prompt from eval_baseline_prompts.py
python eval.py --persona republican --model anthropic/claude-opus-4.5 --static system_prompt_base_persona_republican --out results/eval_results_base_static_opus45_republican.jsonl
```

Run from the project root. Output is written under `results/` by default.