"""Evaluation helpers."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


def bootstrap_ci_f1(
    labels: list[int],
    preds: list[int],
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for macro-F1.

    On small test sets (FPB all-agree test split is ~340 sentences) point
    estimates of macro-F1 carry a CI wide enough that 1-2 pp differences are
    not resolvable. Report this alongside every headline number.
    """
    labels_arr = np.asarray(labels)
    preds_arr = np.asarray(preds)
    n = len(labels_arr)
    if n == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[i] = f1_score(
            labels_arr[idx], preds_arr[idx], average="macro", zero_division=0
        )

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(scores, alpha))
    upper = float(np.quantile(scores, 1.0 - alpha))
    return lower, upper


def evaluate_f1_macro(
    model,
    dataset_split,
    data_collator: DataCollatorWithPadding,
    device: torch.device,
    id2label: dict[int, str],
    batch_size: int = 32,
    verbose: bool = True,
    split_name: str = "eval",
    compute_ci: bool = False,
) -> tuple[list[int], list[int], float, float, float]:
    """Return predictions, labels, macro-F1, and CI bounds on a tokenized split.

    The CI bounds are only computed when ``compute_ci=True`` (it bootstraps and
    is relatively expensive); otherwise they are returned as NaN. Callers that
    only need the F1 can still unpack ``_, _, f1, *_``.
    """
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

    ci_lower, ci_upper = float("nan"), float("nan")
    if compute_ci:
        ci_lower, ci_upper = bootstrap_ci_f1(all_labels, all_preds)

    if verbose:
        print(f"\n=== {split_name} ===")
        if compute_ci:
            print(f"F1-macro: {f1:.4f}  (95% CI [{ci_lower:.4f}, {ci_upper:.4f}], N={len(all_labels)})")
        else:
            print(f"F1-macro: {f1:.4f}  (N={len(all_labels)})")
        names = [id2label[i] for i in sorted(id2label)]
        print(
            classification_report(
                all_labels,
                all_preds,
                target_names=names,
            )
        )
    return all_preds, all_labels, f1, ci_lower, ci_upper
