# Experiments

## Baseline ladder

Compares **eval-only**, **head-only**, **LoRA**, **DoRA**, and **PiSSA** on:

- `distilbert-base-uncased` (general encoder)
- `ProsusAI/finbert` (finance-pretrained; labels remapped to match the checkpoint)

### Run

```bash
# From repo root
source .venv/bin/activate

# Smoke (1 epoch, DistilBERT subset)
python experiments/run_baseline_ladder.py --quick

# Full ladder (3 epochs, seeds 42/43/44, all 10 experiments)
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
