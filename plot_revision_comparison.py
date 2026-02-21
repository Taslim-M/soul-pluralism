"""
Plot test accuracy comparison: baseline vs. after 3 iterations of soul-doc revision.
Reads baseline results from globaloqa/results/eval_results_base_persona_country_deepseekr1_*.jsonl
and iterative revision summaries from globaloqa/results/iterative_revision/*/summary.json.
Saves the plot as globaloqa/results/revision_comparison.png.
"""

import json
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "globaloqa", "results")
ITER_DIR = os.path.join(RESULTS_DIR, "iterative_revision")

# ── Load iterative revision summaries ────────────────────────────────────
summaries = {}
for summary_path in sorted(glob.glob(os.path.join(ITER_DIR, "*", "summary.json"))):
    with open(summary_path) as f:
        s = json.load(f)
    summaries[s["persona"]] = s

personas = sorted(summaries.keys())

# ── Compute baseline accuracy per persona from JSONL files ───────────────
baseline_accs = {}
baseline_ns = {}
for persona in personas:
    # Try capitalized name first, then lowercase
    pattern = os.path.join(RESULTS_DIR, f"eval_results_base_persona_country_deepseekr1_{persona}.jsonl")
    files = glob.glob(pattern)
    if not files:
        pattern = os.path.join(RESULTS_DIR, f"eval_results_base_persona_country_deepseekr1_{persona.lower()}.jsonl")
        files = glob.glob(pattern)
    if not files:
        print(f"Warning: no baseline file found for {persona}, skipping")
        continue
    with open(files[0]) as f:
        records = [json.loads(line) for line in f if line.strip()]
    correct = sum(1 for r in records if r["deepseek/deepseek-r1-0528"] == bool(r["label"]))
    baseline_accs[persona] = correct / len(records)
    baseline_ns[persona] = len(records)

# ── Extract test accuracy after 3 iterations ─────────────────────────────
iter3_accs = {p: summaries[p]["test_accuracies"][3] for p in personas}
iter3_ns = {p: summaries[p]["test_size"] for p in personas}

# ── Compute error bars: sqrt(p*(1-p)/n) ──────────────────────────────────
baseline_errs = [np.sqrt(baseline_accs[p] * (1 - baseline_accs[p]) / baseline_ns[p]) for p in personas]
iter3_errs = [np.sqrt(iter3_accs[p] * (1 - iter3_accs[p]) / iter3_ns[p]) for p in personas]

# ── Plot ──────────────────────────────────────────────────────────────────
x = np.arange(len(personas))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
bars_baseline = ax.bar(x - width / 2, [baseline_accs[p] for p in personas], width,
                       yerr=baseline_errs, capsize=3,
                       label="Baseline (no soul doc)", color="#5B9BD5", edgecolor="white")
bars_iter3 = ax.bar(x + width / 2, [iter3_accs[p] for p in personas], width,
                    yerr=iter3_errs, capsize=3,
                    label="After 3 revisions", color="#ED7D31", edgecolor="white")

# Labels and formatting
ax.set_ylabel("Test Accuracy", fontsize=13)
ax.set_title("GlobalOQA Test Accuracy: Baseline vs. 3 Iterations of Soul-Doc Revision",
             fontsize=14, pad=12)
ax.set_xticks(x)
ax.set_xticklabels(personas, rotation=40, ha="right", fontsize=11)
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(fontsize=12, loc="upper left")
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars (offset above error bars)
for bar, err in zip(bars_baseline, baseline_errs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.01,
            f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=8)
for bar, err in zip(bars_iter3, iter3_errs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.01,
            f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "revision_comparison.png")
fig.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
plt.close()
