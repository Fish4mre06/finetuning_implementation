#!/usr/bin/env python3
"""
Catastrophic-forgetting probe for a finance-adapted LoRA model.

The ORG charter (Q3, "Safety of adaptation") claims domain tuning does not
degrade non-finance behavior. This script tests that claim instead of asserting
it: it evaluates a finance-adapted model on general-domain sentiment (SST-2) and
compares against the same base model without the adapter.

We collapse the 3-way finance head to binary positive/negative (SST-2 has no
neutral class) and measure how often each model agrees with the SST-2 gold
label. A large drop from base -> adapted indicates forgetting.

Usage:
  python experiments/forgetting_probe.py \
      --checkpoint experiments/results/checkpoints/distilbert_lora_seed42 \
      --backbone distilbert
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning_lib.data import load_sst2_sample
from finetuning_lib.train_utils import (
    MODEL_CONFIGS,
    get_device,
    load_sequence_classifier,
)

# SST-2: 0 = negative, 1 = positive (no neutral).
SST2_ID2NAME = {0: "negative", 1: "positive"}


def _predict_binary_sentiment(model, loader, device, cfg) -> list[int]:
    """Predict SST-2-style binary labels from a 3-way finance head.

    The finance head emits negative/neutral/positive logits. We drop the neutral
    logit and take argmax over {negative, positive}, then map to SST-2 ids.
    """
    name_to_finance_id = cfg.label2id
    neg_id = name_to_finance_id["negative"]
    pos_id = name_to_finance_id["positive"]

    model.eval()
    model.to(device)
    preds: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits.cpu().numpy()
            neg = logits[:, neg_id]
            pos = logits[:, pos_id]
            # SST-2 id: 1 = positive when pos logit wins, else 0 = negative.
            preds.extend((pos > neg).astype(int).tolist())
    return preds


def _tokenize(dataset, tokenizer, max_length: int = 128):
    def tok(batch):
        return tokenizer(
            batch["sentence"], truncation=True, max_length=max_length, padding=False
        )

    out = dataset.map(tok, batched=True).remove_columns(["sentence"])
    out.set_format("torch")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Forgetting probe on SST-2")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a saved LoRA checkpoint dir (a Trainer/PEFT output_dir).",
    )
    parser.add_argument(
        "--backbone",
        default="distilbert",
        choices=list(MODEL_CONFIGS.keys()),
        help="Backbone the adapter was trained on.",
    )
    parser.add_argument("--n", type=int, default=500, help="SST-2 sample size.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from peft import PeftModel

    cfg = MODEL_CONFIGS[args.backbone]
    device = get_device()
    print(f"Device: {device} | backbone: {args.backbone}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    sst2 = load_sst2_sample(n=args.n, seed=args.seed)
    gold = list(sst2["label"])
    tokenized = _tokenize(sst2, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(tokenized, batch_size=32, collate_fn=collator)

    # 1) Base model (no adapter).
    base = load_sequence_classifier(cfg)
    base_preds = _predict_binary_sentiment(base, loader, device, cfg)

    # 2) Adapted model (base + LoRA adapter).
    adapted_base = load_sequence_classifier(cfg)
    adapted = PeftModel.from_pretrained(adapted_base, args.checkpoint)
    adapted_preds = _predict_binary_sentiment(adapted, loader, device, cfg)

    base_acc = accuracy_score(gold, base_preds)
    adapted_acc = accuracy_score(gold, adapted_preds)
    base_f1 = f1_score(gold, base_preds, average="macro", zero_division=0)
    adapted_f1 = f1_score(gold, adapted_preds, average="macro", zero_division=0)

    print(f"\n{'=' * 56}")
    print(f"FORGETTING PROBE (SST-2 general sentiment, N={len(gold)})")
    print("=" * 56)
    print(f"  base    : acc={base_acc:.4f}  f1_macro={base_f1:.4f}")
    print(f"  adapted : acc={adapted_acc:.4f}  f1_macro={adapted_f1:.4f}")
    print(f"  delta   : acc={adapted_acc - base_acc:+.4f}  f1_macro={adapted_f1 - base_f1:+.4f}")
    print(
        "\nInterpretation: a large negative delta means the finance adapter "
        "degraded general sentiment ability (forgetting). Near-zero or positive "
        "delta supports the 'safe adaptation' claim. Note: the binary collapse "
        "from a 3-way head is a coarse proxy; report it as such."
    )


if __name__ == "__main__":
    main()
