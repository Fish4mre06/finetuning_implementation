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
    ("finbert_eval_only", "finbert", "eval_only"),
    ("finbert_head_only", "finbert", "head_only"),
    ("finbert_lora", "finbert", "lora"),
    ("finbert_dora", "finbert", "dora"),
    ("finbert_pissa", "finbert", "pissa"),
]


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
        results.groupby("experiment_id")["test_f1_macro"]
        .agg(["mean", "std", "min", "max"])
        .sort_values("mean", ascending=False)
    )
    summary_path = RESULTS_DIR / "baseline_ladder_summary.csv"
    summary.to_csv(summary_path)

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {summary_path}")
    print("\nMean test F1-macro by experiment:")
    print(summary.to_string())

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
