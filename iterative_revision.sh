#!/bin/bash
set -e

PERSONAS=(
    Brazil
    Britain
    France
    Germany
    Indonesia
    Japan
    Jordan
    Lebanon
    Mexico
    Nigeria
    Pakistan
    Russia
    Turkey
)

for persona in "${PERSONAS[@]}"; do
    echo "========================================"
    echo "  Running: $persona"
    echo "========================================"
    python iterative_revision.py \
        --task globaloqa \
        --persona "$persona" \
        --eval-model deepseek/deepseek-r1-0528 \
        --revision-model anthropic/claude-sonnet-4.6
    python eval.py \
        --task globaloqa \
        --persona "$persona" \
        --model deepseek/deepseek-r1-0528 \
        --static system_prompt_base_persona_country \
        --out globaloqa/results/eval_results_base_persona_country_deepseekr1_"$persona".jsonl

done

echo "All done."
