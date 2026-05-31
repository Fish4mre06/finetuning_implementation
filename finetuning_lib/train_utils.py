"""Training and experiment runners for baseline ladder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from finetuning_lib.data import LABEL2ID, NUM_LABELS, remap_dataset_labels
from finetuning_lib.eval_utils import evaluate_f1_macro

ExperimentMethod = Literal[
    "eval_only",
    "head_only",
    "lora",
    "dora",
    "pissa",
]

BackboneKey = Literal["distilbert", "finbert", "finbert_tone"]


@dataclass(frozen=True)
class ModelConfig:
    key: BackboneKey
    model_id: str
    lora_target_modules: list[str]
    label2id: dict[str, int]
    # True if the checkpoint ships a task-aligned (sentiment) classification
    # head. When False (e.g. a *-base encoder), eval_only uses a randomly
    # initialized head and is only a chance-level floor, not a real baseline.
    has_pretrained_head: bool = False
    # Set when the backbone's pretraining/finetuning corpus overlaps the eval
    # benchmark, making eval_only an in-sample (leaked) measurement rather than
    # a zero-shot transfer baseline. Surfaced in run output and CSV.
    contamination_note: str | None = None

    @property
    def id2label(self) -> dict[int, str]:
        return {v: k for k, v in self.label2id.items()}

    @property
    def is_contaminated(self) -> bool:
        return self.contamination_note is not None


MODEL_CONFIGS: dict[BackboneKey, ModelConfig] = {
    "distilbert": ModelConfig(
        key="distilbert",
        model_id="distilbert-base-uncased",
        lora_target_modules=["q_lin", "v_lin"],
        label2id=dict(LABEL2ID),
    ),
    "finbert": ModelConfig(
        key="finbert",
        model_id="ProsusAI/finbert",
        lora_target_modules=["query", "value"],
        # ProsusAI/finbert checkpoint label order (differs from PhraseBank export ids)
        label2id={"positive": 0, "negative": 1, "neutral": 2},
        has_pretrained_head=True,
        contamination_note=(
            "ProsusAI/finbert was fine-tuned on FinancialPhraseBank; its "
            "eval_only result (~0.97) reflects training-set overlap, not "
            "zero-shot transfer. Treat as a leaked upper anchor, not a fair "
            "domain baseline. Use 'finbert_tone' for a leakage-safe comparison."
        ),
    ),
    "finbert_tone": ModelConfig(
        key="finbert_tone",
        model_id="yiyanghkust/finbert-tone",
        lora_target_modules=["query", "value"],
        # FinBERT-Tone checkpoint label order.
        label2id={"neutral": 0, "positive": 1, "negative": 2},
        # Ships a pretrained tone/sentiment head, so eval_only is a genuine
        # zero-shot finance baseline (not a random floor).
        has_pretrained_head=True,
        # Trained on analyst-report / forward-looking-statement sentences, not
        # FinancialPhraseBank -> usable as an uncontaminated finance encoder.
        # Verify provenance before treating as fully leakage-free.
    ),
}

# Locked, shared learning rate for the trained-method comparison. head_only and
# the LoRA family must use the SAME LR budget so that head_only-vs-lora isolates
# the marginal value of attention adaptation rather than an LR difference.
LOCKED_LEARNING_RATE: float = 2e-4


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_trainable_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def tokenize_datasets(dataset_dict, tokenizer, max_length: int = 128):
    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = dataset_dict.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["sentence"])
    tokenized.set_format("torch")
    return tokenized


def load_sequence_classifier(cfg: ModelConfig):
    return AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id,
        num_labels=NUM_LABELS,
        id2label=cfg.id2label,
        label2id=cfg.label2id,
    )


def freeze_encoder(model) -> None:
    """Train only the classification head (and DistilBERT pre_classifier if present)."""
    head_keys = ("classifier", "pre_classifier", "score")
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in head_keys)


def build_lora_config(
    method: ExperimentMethod,
    target_modules: list[str],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> LoraConfig:
    kwargs: dict[str, Any] = dict(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )
    if method == "dora":
        kwargs["use_dora"] = True
    elif method == "pissa":
        kwargs["init_lora_weights"] = "pissa_niter_4"
        kwargs["lora_dropout"] = 0.0
    elif method != "lora":
        raise ValueError(f"Not a LoRA-family method: {method}")
    return LoraConfig(**kwargs)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    from sklearn.metrics import f1_score

    return {"f1_macro": f1_score(labels, predictions, average="macro")}


def run_experiment(
    *,
    experiment_id: str,
    backbone: BackboneKey,
    method: ExperimentMethod,
    tokenized,
    tokenizer,
    device: torch.device,
    output_dir: str,
    num_train_epochs: int = 3,
    learning_rate: float = LOCKED_LEARNING_RATE,
    per_device_train_batch_size: int = 16,
    seed: int = 42,
    lora_r: int = 8,
) -> dict[str, Any]:
    """
    Run one baseline-ladder experiment and return summary metrics.

    Methods:
      - eval_only: no training (fresh or pretrained head only)
      - head_only: train classification head, encoder frozen
      - lora / dora / pissa: PEFT adapters on attention modules
    """
    cfg = MODEL_CONFIGS[backbone]
    if cfg.contamination_note:
        print(f"  [!] CONTAMINATION: {cfg.contamination_note}")
    model = load_sequence_classifier(cfg)

    if method == "eval_only":
        trainable, total = 0, sum(p.numel() for p in model.parameters())
    elif method == "head_only":
        freeze_encoder(model)
        trainable, total = count_trainable_params(model)
    else:
        lora_config = build_lora_config(
            method, cfg.lora_target_modules, r=lora_r
        )
        model = get_peft_model(model, lora_config)
        trainable, total = count_trainable_params(model)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    test_n = len(tokenized["test"])

    if method == "eval_only":
        if cfg.is_contaminated:
            baseline_type = "leaked_anchor"
            floor_note = " [contaminated: in-sample, training-set overlap]"
        elif cfg.has_pretrained_head:
            baseline_type = "zero_shot"
            floor_note = " [zero-shot: pretrained sentiment head, leakage-safe baseline]"
        else:
            baseline_type = "random_floor"
            floor_note = " [random untrained head: chance-level floor, NOT a LoRA comparison]"
        _, _, test_f1, ci_lo, ci_hi = evaluate_f1_macro(
            model,
            tokenized["test"],
            data_collator,
            device,
            id2label=cfg.id2label,
            split_name=f"{experiment_id} (eval only){floor_note}",
            compute_ci=True,
        )
        return {
            "experiment_id": experiment_id,
            "backbone": backbone,
            "method": method,
            "seed": seed,
            "test_f1_macro": test_f1,
            "test_f1_ci_lower": ci_lo,
            "test_f1_ci_upper": ci_hi,
            "test_n": test_n,
            "val_f1_macro": None,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 0.0,
            "contaminated": cfg.is_contaminated,
            "baseline_type": baseline_type,
            "learning_rate": None,
            "output_dir": None,
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=32,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        fp16=device.type == "cuda",
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    _, _, val_f1, _, _ = evaluate_f1_macro(
        model,
        tokenized["validation"],
        data_collator,
        device,
        id2label=cfg.id2label,
        verbose=False,
        split_name="val",
    )
    _, _, test_f1, ci_lo, ci_hi = evaluate_f1_macro(
        model,
        tokenized["test"],
        data_collator,
        device,
        id2label=cfg.id2label,
        split_name=f"{experiment_id} (test)",
        compute_ci=True,
    )

    pct = 100.0 * trainable / total if total else 0.0
    return {
        "experiment_id": experiment_id,
        "backbone": backbone,
        "method": method,
        "seed": seed,
        "test_f1_macro": test_f1,
        "test_f1_ci_lower": ci_lo,
        "test_f1_ci_upper": ci_hi,
        "test_n": test_n,
        "val_f1_macro": val_f1,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(pct, 2),
        "contaminated": cfg.is_contaminated,
        "baseline_type": None,
        "learning_rate": learning_rate,
        "output_dir": output_dir,
    }
