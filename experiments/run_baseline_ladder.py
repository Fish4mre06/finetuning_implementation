#!/usr/bin/env python3
"""
Run baseline-ladder experiments and save results to CSV.

Usage:
  python experiments/run_baseline_ladder.py
  python experiments/run_baseline_ladder.py --quick
  python experiments/run_baseline_ladder.py --experiments distilbert_lora finbert_eval_only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning_lib.data import (
    load_financial_phrasebank,
    make_stratified_splits,
    remap_dataset_labels,
)
from finetuning_lib.train_utils import (
    MODEL_CONFIGS,
    get_device,
    run_experiment,
    tokenize_datasets,
)

RESULTS_DIR = REPO_ROOT / "experiments" / "results"

# (experiment_id, backbone, method)
LADDER_EXPERIMENTS: list[tuple[str, str, str]] = [
    ("distilbert_eval_only", "distilbert", "eval_only"),
    ("distilbert_head_only", "distilbert", "head_only"),
    ("distilbert_lora", "distilbert", "lora"),
    ("distilbert_dora", "distilbert", "dora"),
    ("distilbert_pissa", "distilbert", "pissa"),
    # ProsusAI/finbert is fine-tuned on FinancialPhraseBank: eval_only here is
    # in-sample (leaked). Kept as a labelled upper anchor, flagged contaminated.
    ("finbert_eval_only", "finbert", "eval_only"),
    ("finbert_head_only", "finbert", "head_only"),
    ("finbert_lora", "finbert", "lora"),
    ("finbert_dora", "finbert", "dora"),
    ("finbert_pissa", "finbert", "pissa"),
    # FinBERT-Tone: finance encoder NOT trained on FPB -> leakage-safe domain
    # comparison for the head_only-vs-lora contrast.
    ("finbert_tone_eval_only", "finbert_tone", "eval_only"),
    ("finbert_tone_head_only", "finbert_tone", "head_only"),
    ("finbert_tone_lora", "finbert_tone", "lora"),
]

# Backbones for which head_only-vs-lora is the meaningful, fair contrast
# (same LR budget, same epochs). This isolates the marginal value of attention
# adaptation over a trained linear head -- the real research question.
KEY_COMPARISON_BACKBONES = ["distilbert", "finbert", "finbert_tone"]


def _print_key_comparisons(results: pd.DataFrame) -> None:
    """Print the honest head_only-vs-lora delta per backbone.

    The headline 'eval_only -> lora' jump is mostly the value of TRAINING A HEAD
    (eval_only uses a random/untrained head), not the value of LoRA. The fair
    isolation of LoRA's contribution is head_only vs lora at the same LR budget.
    """
    print(f"\n{'=' * 60}")
    print("KEY COMPARISON: marginal value of LoRA over a trained head")
    print("(head_only vs lora, same LR budget -- this is the real question)")
    print("=" * 60)
    means = results.groupby("experiment_id")["test_f1_macro"].mean()
    for backbone in KEY_COMPARISON_BACKBONES:
        head_id = f"{backbone}_head_only"
        lora_id = f"{backbone}_lora"
        if head_id in means and lora_id in means:
            head, lora = means[head_id], means[lora_id]
            print(
                f"  {backbone:13s}  head_only={head:.4f}  lora={lora:.4f}  "
                f"delta={lora - head:+.4f}"
            )
    print(
        "\nReminder: eval_only rows are NOT a LoRA baseline. DistilBERT eval_only"
        " is a random-head chance floor; ProsusAI/finbert eval_only is an "
        "in-sample leaked anchor (trained on FPB); FinBERT-Tone eval_only is a "
        "leakage-safe zero-shot baseline. Do not cite 'eval_only -> lora' as the"
        " LoRA lift -- use the head_only-vs-lora delta above."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial sentiment baseline ladder")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="1 epoch and a single seed (faster smoke run)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (default: 3, or 1 with --quick)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds (default: [42], or [42,43,44] full run)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Subset of experiment IDs to run",
    )
    parser.add_argument(
        "--agreement",
        default="all",
        choices=["all", "75", "66", "50"],
        help="FinancialPhraseBank agreement subset",
    )
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else (1 if args.quick else 3)
    seeds = args.seeds if args.seeds is not None else ([42] if args.quick else [42, 43, 44])

    experiments = LADDER_EXPERIMENTS
    if args.experiments:
        allowed = set(args.experiments)
        experiments = [e for e in LADDER_EXPERIMENTS if e[0] in allowed]

    device = get_device()
    print(f"Device: {device} | epochs={epochs} | seeds={seeds}")

    df = load_financial_phrasebank(agreement=args.agreement).to_pandas()
    dataset = make_stratified_splits(df, random_state=42)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    tokenized_cache: dict[str, object] = {}
    tokenizer_cache: dict[str, object] = {}

    for seed in seeds:
        for exp_id, backbone, method in experiments:
            cfg = MODEL_CONFIGS[backbone]  # type: ignore[index]
            if backbone not in tokenized_cache:
                tokenizer_cache[backbone] = AutoTokenizer.from_pretrained(cfg.model_id)
                backbone_data = dataset
                if cfg.label2id != MODEL_CONFIGS["distilbert"].label2id:
                    backbone_data = remap_dataset_labels(dataset, cfg.label2id)
                tokenized_cache[backbone] = tokenize_datasets(
                    backbone_data, tokenizer_cache[backbone]
                )
            tokenizer = tokenizer_cache[backbone]
            tokenized = tokenized_cache[backbone]
            out_dir = str(RESULTS_DIR / "checkpoints" / f"{exp_id}_seed{seed}")

            print(f"\n{'=' * 60}\nRunning: {exp_id} (seed={seed})\n{'=' * 60}")
            row = run_experiment(
                experiment_id=exp_id,
                backbone=backbone,  # type: ignore[arg-type]
                method=method,  # type: ignore[arg-type]
                tokenized=tokenized,
                tokenizer=tokenizer,
                device=device,
                output_dir=out_dir,
                num_train_epochs=epochs,
                seed=seed,
            )
            all_rows.append(row)

    results = pd.DataFrame(all_rows)
    out_csv = RESULTS_DIR / "baseline_ladder_results.csv"
    results.to_csv(out_csv, index=False)

    summary = (
        results.groupby("experiment_id")
        .agg(
            mean=("test_f1_macro", "mean"),
            std=("test_f1_macro", "std"),
            min=("test_f1_macro", "min"),
            max=("test_f1_macro", "max"),
            ci_lower=("test_f1_ci_lower", "mean"),
            ci_upper=("test_f1_ci_upper", "mean"),
            test_n=("test_n", "first"),
            contaminated=("contaminated", "first"),
        )
        .sort_values("mean", ascending=False)
    )
    summary_path = RESULTS_DIR / "baseline_ladder_summary.csv"
    summary.to_csv(summary_path)

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {summary_path}")
    test_n = int(results["test_n"].iloc[0]) if len(results) else 0
    print(f"\nMean test F1-macro by experiment (test N={test_n}):")
    print(summary.to_string())
    print(
        "\nNote: per-seed runs report a 95% bootstrap CI; the columns above are "
        "the mean of those CI bounds. On a test set this small, differences "
        "smaller than the CI width are not resolvable."
    )

    _print_key_comparisons(results)

    meta = {
        "epochs": epochs,
        "seeds": seeds,
        "agreement": args.agreement,
        "device": str(device),
        "experiments": [e[0] for e in experiments],
    }
    (RESULTS_DIR / "run_config.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
