#!/bin/bash

cd /home/dbaek/soul-pluralism/
for i in {2..4}; do
    python eval.py --persona democrat --model anthropic/claude-opus-4.5 --soul soul_docs/democrat/doc${i}.txt --out results/democrat_eval_doc${i}.jsonl
done

for i in {2..4}; do
    python eval.py --persona republican --model anthropic/claude-opus-4.5 --soul soul_docs/republican/doc${i}.txt --out results/republican_eval_doc${i}.jsonl
done
