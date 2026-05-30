"""Evaluation helpers."""

from __future__ import annotations

import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

def evaluate_f1_macro(
    model,
    dataset_split,
    data_collator: DataCollatorWithPadding,
    device: torch.device,
    id2label: dict[int, str],
    batch_size: int = 32,
    verbose: bool = True,
    split_name: str = "eval",
) -> tuple[list[int], list[int], float]:
    """Return predictions, labels, and macro-F1 on a tokenized split."""
    model.eval()
    model.to(device)
    loader = DataLoader(
        dataset_split, batch_size=batch_size, collate_fn=data_collator
    )
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())

    f1 = f1_score(all_labels, all_preds, average="macro")
    if verbose:
        print(f"\n=== {split_name} ===")
        print(f"F1-macro: {f1:.4f}")
        names = [id2label[i] for i in sorted(id2label)]
        print(
            classification_report(
                all_labels,
                all_preds,
                target_names=names,
            )
        )
    return all_preds, all_labels, f1
