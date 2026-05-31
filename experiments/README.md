# Experiments

## Baseline ladder

Compares **eval-only**, **head-only**, **LoRA**, **DoRA**, and **PiSSA** on:

- `distilbert-base-uncased` (general encoder; `eval_only` = random-head **floor**)
- `ProsusAI/finbert` (finance-pretrained, but **fine-tuned on FinancialPhraseBank** — its `eval_only` ~0.97 is in-sample/**leaked**, flagged `contaminated=True`)
- `yiyanghkust/finbert-tone` (finance encoder **not** trained on FPB — leakage-safe zero-shot baseline)

### The comparison that actually matters

`eval_only → lora` is **not** the LoRA lift: on DistilBERT, `eval_only` is a random untrained head (chance floor), so most of that jump is just *training a head*. The honest, isolated LoRA contribution is **`head_only` vs `lora` at the same locked learning rate** — the runner prints this delta per backbone, and labels each `eval_only` row by `baseline_type` (`random_floor` / `zero_shot` / `leaked_anchor`). Every run also reports a 95% bootstrap CI and the test-set N; differences below the CI width are not resolvable.

## Forgetting probe

`forgetting_probe.py` evaluates a saved finance adapter on general-domain sentiment (SST-2) and compares against the same base model without the adapter, to test (not assert) that domain tuning didn't degrade general behavior:

```bash
python experiments/forgetting_probe.py \
    --checkpoint experiments/results/checkpoints/distilbert_lora_seed42 \
    --backbone distilbert
```

### Run

```bash
# From repo root
source .venv/bin/activate

# Smoke (1 epoch, DistilBERT subset)
python experiments/run_baseline_ladder.py --quick

# Full ladder (3 epochs, seeds 42/43/44, all experiments)
python experiments/run_baseline_ladder.py
```

Or open `baseline_ladder.ipynb` and set `QUICK_RUN = False` for the full run.

### Outputs

| File | Description |
|------|-------------|
| `results/baseline_ladder_results.csv` | Per-run metrics |
| `results/baseline_ladder_summary.csv` | Mean ± std F1 by experiment |
| `results/baseline_ladder_chart.png` | Bar chart (notebook) |
| `results/checkpoints/` | Training checkpoints (gitignored) |

### Organizational context

See [`../ORG_CHARTER.md`](../ORG_CHARTER.md) for the business question, success metrics, and phased roadmap.
